"""
持仓管理模块 - 手动录入持仓，实时监控分析

功能:
1. 手动录入/修改/删除持仓
2. 实时盈亏计算
3. 个股分析触发（调用三层决策）
4. 持仓风险提示
5. 组合分析
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.data.database import get_db, PortfolioPosition, AnalysisTask, AnalysisResult
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionStatus(Enum):
    HOLDING = "持有中"
    CLOSED = "已平仓"
    WATCHING = "观察中"


class RiskLevel(Enum):
    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "极高风险"


@dataclass
class PositionMetrics:
    """持仓指标"""
    symbol: str
    name: str = ""
    shares: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    cost_basis: float = 0.0          # 总成本
    market_value: float = 0.0         # 市值
    unrealized_pnl: float = 0.0       # 浮动盈亏
    unrealized_pnl_pct: float = 0.0   # 浮动盈亏率
    day_pnl: float = 0.0              # 当日盈亏
    day_pnl_pct: float = 0.0          # 当日盈亏率
    holding_days: int = 0             # 持仓天数
    highest_price: float = 0.0        # 持仓期最高价
    lowest_price: float = 0.0         # 持仓期最低价
    max_profit_pct: float = 0.0       # 最大盈利比例
    max_loss_pct: float = 0.0         # 最大亏损比例
    distance_to_high: float = 0.0     # 距高点回撤
    distance_to_low: float = 0.0      # 距低点反弹


@dataclass
class PortfolioSummary:
    """组合汇总"""
    total_cost: float = 0.0           # 总成本
    total_market_value: float = 0.0   # 总市值
    total_unrealized_pnl: float = 0.0 # 总浮动盈亏
    total_unrealized_pnl_pct: float = 0.0
    total_day_pnl: float = 0.0        # 当日总盈亏
    cash_ratio: float = 0.0           # 现金比例
    position_count: int = 0           # 持仓数量
    risk_level: RiskLevel = RiskLevel.LOW
    concentration_risk: float = 0.0   # 集中度风险 (最大单票占比)
    sector_exposure: Dict[str, float] = field(default_factory=dict)  # 行业暴露


@dataclass
class PositionAlert:
    """持仓预警"""
    symbol: str
    alert_type: str                   # stop_loss / take_profit / max_drawdown / holding_timeout
    level: str                        # warning / danger
    message: str
    suggested_action: str
    triggered_at: datetime = field(default_factory=datetime.now)


class PositionManager:
    """持仓管理器"""

    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or next(get_db())
        self.positions: Dict[str, PortfolioPosition] = {}
        self._load_positions()

    def _load_positions(self):
        """从数据库加载持仓"""
        positions = self.db.query(PortfolioPosition).filter(
            PortfolioPosition.status == PositionStatus.HOLDING.value
        ).all()
        for pos in positions:
            self.positions[pos.symbol] = pos
        logger.info(f"加载持仓: {len(self.positions)}只股票")

    def add_position(self, symbol: str, name: str, shares: int,
                     entry_price: float, entry_date: Optional[datetime] = None,
                     sector: str = "", notes: str = "") -> PortfolioPosition:
        """添加持仓"""
        if symbol in self.positions:
            logger.warning(f"{symbol} 已存在持仓，更新数量")
            return self.update_position(symbol, shares_delta=shares, price=entry_price)

        position = PortfolioPosition(
            symbol=symbol,
            name=name,
            shares=shares,
            entry_price=entry_price,
            entry_date=entry_date or datetime.now(),
            sector=sector,
            status=PositionStatus.HOLDING.value,
            notes=notes,
            created_at=datetime.now()
        )

        self.db.add(position)
        self.db.commit()
        self.positions[symbol] = position

        logger.info(f"添加持仓: {symbol} {shares}股 @ {entry_price:.2f}")
        return position

    def update_position(self, symbol: str, shares_delta: int = 0,
                        price: Optional[float] = None,
                        notes: Optional[str] = None) -> Optional[PortfolioPosition]:
        """更新持仓"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        old_shares = pos.shares
        old_cost = old_shares * pos.entry_price

        # 更新股数
        new_shares = old_shares + shares_delta
        if new_shares <= 0:
            return self.close_position(symbol, price or pos.current_price or pos.entry_price)

        # 更新成本价 (加权平均)
        if shares_delta > 0 and price:
            new_cost = shares_delta * price
            pos.entry_price = (old_cost + new_cost) / new_shares

        pos.shares = new_shares

        if notes:
            pos.notes = notes

        pos.updated_at = datetime.now()
        self.db.commit()

        logger.info(f"更新持仓: {symbol} {old_shares} -> {new_shares}股")
        return pos

    def close_position(self, symbol: str, exit_price: float,
                       exit_date: Optional[datetime] = None) -> Optional[PortfolioPosition]:
        """平仓"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.exit_price = exit_price
        pos.exit_date = exit_date or datetime.now()
        pos.realized_pnl = pos.shares * (exit_price - pos.entry_price)
        pos.realized_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        pos.status = PositionStatus.CLOSED.value
        pos.updated_at = datetime.now()

        self.db.commit()
        del self.positions[symbol]

        logger.info(f"平仓: {symbol} @ {exit_price:.2f}, 实现盈亏{pos.realized_pnl:.2f}")
        return pos

    def delete_position(self, symbol: str) -> bool:
        """删除持仓记录"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        self.db.delete(pos)
        self.db.commit()
        del self.positions[symbol]

        logger.info(f"删除持仓记录: {symbol}")
        return True

    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """获取持仓"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[PortfolioPosition]:
        """获取所有持仓"""
        return list(self.positions.values())

    def update_prices(self, price_data: Dict[str, Dict]):
        """更新持仓价格"""
        for symbol, pos in self.positions.items():
            if symbol in price_data:
                data = price_data[symbol]
                pos.current_price = data.get('close', pos.current_price)
                pos.prev_close = data.get('prev_close', pos.prev_close)
                pos.day_high = data.get('high', pos.day_high)
                pos.day_low = data.get('low', pos.day_low)

                # 更新持仓期最高最低价
                if pos.highest_price is None or pos.day_high > pos.highest_price:
                    pos.highest_price = pos.day_high
                if pos.lowest_price is None or pos.day_low < pos.lowest_price:
                    pos.lowest_price = pos.day_low

        self.db.commit()

    def calculate_metrics(self, symbol: str) -> Optional[PositionMetrics]:
        """计算持仓指标"""
        pos = self.positions.get(symbol)
        if not pos or pos.current_price is None:
            return None

        metrics = PositionMetrics(
            symbol=symbol,
            name=pos.name or symbol,
            shares=pos.shares,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            cost_basis=pos.shares * pos.entry_price,
            market_value=pos.shares * pos.current_price,
        )

        # 盈亏计算
        metrics.unrealized_pnl = metrics.market_value - metrics.cost_basis
        metrics.unrealized_pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price

        # 当日盈亏
        if pos.prev_close:
            metrics.day_pnl = pos.shares * (pos.current_price - pos.prev_close)
            metrics.day_pnl_pct = (pos.current_price - pos.prev_close) / pos.prev_close

        # 持仓天数
        metrics.holding_days = (datetime.now() - pos.entry_date).days

        # 极值
        metrics.highest_price = pos.highest_price or pos.entry_price
        metrics.lowest_price = pos.lowest_price or pos.entry_price
        metrics.max_profit_pct = (metrics.highest_price - pos.entry_price) / pos.entry_price
        metrics.max_loss_pct = (metrics.lowest_price - pos.entry_price) / pos.entry_price

        # 距极值距离
        if metrics.highest_price > 0:
            metrics.distance_to_high = (metrics.highest_price - pos.current_price) / metrics.highest_price
        if metrics.lowest_price > 0:
            metrics.distance_to_low = (pos.current_price - metrics.lowest_price) / metrics.lowest_price

        return metrics

    def get_all_metrics(self) -> List[PositionMetrics]:
        """获取所有持仓指标"""
        return [m for m in [self.calculate_metrics(s) for s in self.positions] if m]

    def get_portfolio_summary(self) -> PortfolioSummary:
        """获取组合汇总"""
        summary = PortfolioSummary()
        metrics_list = self.get_all_metrics()

        if not metrics_list:
            return summary

        summary.position_count = len(metrics_list)
        summary.total_cost = sum(m.cost_basis for m in metrics_list)
        summary.total_market_value = sum(m.market_value for m in metrics_list)
        summary.total_unrealized_pnl = sum(m.unrealized_pnl for m in metrics_list)
        summary.total_day_pnl = sum(m.day_pnl for m in metrics_list)

        if summary.total_cost > 0:
            summary.total_unrealized_pnl_pct = summary.total_unrealized_pnl / summary.total_cost

        # 集中度风险
        if summary.total_market_value > 0:
            max_weight = max(m.market_value for m in metrics_list) / summary.total_market_value
            summary.concentration_risk = max_weight

        # 风险等级
        if summary.concentration_risk > 0.5:
            summary.risk_level = RiskLevel.CRITICAL
        elif summary.concentration_risk > 0.3:
            summary.risk_level = RiskLevel.HIGH
        elif summary.concentration_risk > 0.2:
            summary.risk_level = RiskLevel.MEDIUM
        else:
            summary.risk_level = RiskLevel.LOW

        return summary

    def check_alerts(self, config: Optional[Dict] = None) -> List[PositionAlert]:
        """检查持仓预警"""
        alerts = []
        cfg = config or {
            'stop_loss_pct': 0.07,
            'take_profit_pct': 0.15,
            'max_drawdown_pct': 0.10,
            'max_holding_days': 20
        }

        for symbol, pos in self.positions.items():
            metrics = self.calculate_metrics(symbol)
            if not metrics:
                continue

            # 止损检查
            if metrics.unrealized_pnl_pct <= -cfg['stop_loss_pct']:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type='stop_loss',
                    level='danger',
                    message=f"{symbol} 亏损达{abs(metrics.unrealized_pnl_pct)*100:.1f}%，超过止损线",
                    suggested_action="建议止损平仓"
                ))

            # 止盈检查
            if metrics.unrealized_pnl_pct >= cfg['take_profit_pct']:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type='take_profit',
                    level='warning',
                    message=f"{symbol} 盈利达{metrics.unrealized_pnl_pct*100:.1f}%，达到止盈目标",
                    suggested_action="建议考虑部分止盈"
                ))

            # 最大回撤检查
            if metrics.distance_to_high >= cfg['max_drawdown_pct']:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type='max_drawdown',
                    level='warning',
                    message=f"{symbol} 从高点回撤{metrics.distance_to_high*100:.1f}%",
                    suggested_action="关注趋势是否反转"
                ))

            # 持仓超时检查
            if metrics.holding_days >= cfg['max_holding_days']:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type='holding_timeout',
                    level='warning',
                    message=f"{symbol} 持仓已达{metrics.holding_days}天，超过最大持仓期限",
                    suggested_action="建议重新评估持仓理由"
                ))

        return alerts


class PositionAnalyzer:
    """
    持仓分析器 - 对持仓股票触发三层决策分析
    """

    def __init__(self, data_source, hmm_engine, technical_engine,
                 debate_engine, forecast_generator):
        self.data_source = data_source
        self.hmm_engine = hmm_engine
        self.technical_engine = technical_engine
        self.debate_engine = debate_engine
        self.forecast_generator = forecast_generator
        self.logger = get_logger(self.__class__.__name__)

    def analyze_position(self, symbol: str, position: PortfolioPosition) -> Dict:
        """
        对持仓股票进行完整分析

        Returns:
            包含三层分析结果和预测报告的字典
        """
        self.logger.info(f"开始分析持仓: {symbol}")

        # 获取数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        df = self.data_source.get_daily_data(symbol, start_date, end_date)

        if df is None or len(df) < 60:
            return {"error": "数据不足"}

        current_price = df['close'].iloc[-1]
        name = position.name or symbol

        # Layer 1: HMM趋势定性
        trend_result = self.hmm_engine.predict_trend(df)

        # Layer 2: 技术面信号
        signal_report = self.technical_engine.generate_signals(df)

        # Layer 3: AI辩论
        debate_result = self.debate_engine.debate(symbol, trend_result, signal_report)

        # 生成预测报告
        forecast = self.forecast_generator.generate(
            symbol=symbol,
            name=name,
            current_price=current_price,
            trend_qualification=trend_result,
            signal_report=signal_report,
            debate_result=debate_result
        )

        # 持仓特定分析
        position_analysis = self._analyze_position_specific(position, df, current_price)

        return {
            'symbol': symbol,
            'name': name,
            'current_price': current_price,
            'position': {
                'shares': position.shares,
                'entry_price': position.entry_price,
                'unrealized_pnl': position.shares * (current_price - position.entry_price),
                'unrealized_pnl_pct': (current_price - position.entry_price) / position.entry_price,
                'holding_days': (datetime.now() - position.entry_date).days
            },
            'layer1_trend': trend_result,
            'layer2_signal': signal_report,
            'layer3_debate': debate_result,
            'forecast': forecast.to_dict(),
            'position_analysis': position_analysis
        }

    def _analyze_position_specific(self, position: PortfolioPosition,
                                    df: pd.DataFrame, current_price: float) -> Dict:
        """持仓特定分析"""
        entry_price = position.entry_price

        # 成本分析
        cost_analysis = {
            'entry_price': entry_price,
            'current_price': current_price,
            'breakeven_price': entry_price * 1.003,  # 考虑手续费
            'distance_to_breakeven': (current_price - entry_price * 1.003) / (entry_price * 1.003)
        }

        # 支撑压力分析
        recent_low = df['low'].tail(20).min()
        recent_high = df['high'].tail(20).max()
        support_resistance = {
            'nearest_support': recent_low,
            'nearest_resistance': recent_high,
            'distance_to_support': (current_price - recent_low) / current_price,
            'distance_to_resistance': (recent_high - current_price) / current_price
        }

        # 建议操作
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct > 0.15:
            suggestion = "盈利丰厚，建议分批止盈"
        elif pnl_pct > 0.05:
            suggestion = "小幅盈利，可持有观察"
        elif pnl_pct > -0.05:
            suggestion = "盈亏不大，按信号操作"
        elif pnl_pct > -0.07:
            suggestion = "亏损扩大，关注止损"
        else:
            suggestion = "触发止损，建议平仓"

        return {
            'cost_analysis': cost_analysis,
            'support_resistance': support_resistance,
            'suggestion': suggestion
        }

    def analyze_all_positions(self, position_manager: PositionManager) -> List[Dict]:
        """分析所有持仓"""
        results = []
        for symbol in position_manager.positions:
            pos = position_manager.get_position(symbol)
            if pos:
                result = self.analyze_position(symbol, pos)
                results.append(result)
        return results


class ReviewScheduler:
    """
    Review调度器 - 管理次日复核任务
    """

    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or next(get_db())

    def schedule_review(self, symbol: str, prediction_date: datetime,
                        prediction_data: Dict) -> AnalysisTask:
        """调度次日复核任务"""
        review_date = prediction_date + timedelta(days=1)

        # 跳过周末
        while review_date.weekday() >= 5:
            review_date += timedelta(days=1)

        task = AnalysisTask(
            symbol=symbol,
            task_type='review',
            status='pending',
            scheduled_at=review_date,
            parameters={
                'prediction_date': prediction_date.isoformat(),
                'prediction_data': prediction_data
            }
        )

        self.db.add(task)
        self.db.commit()
        return task

    def execute_pending_reviews(self, analyzer: PositionAnalyzer,
                                 position_manager: PositionManager):
        """执行待处理的复核任务"""
        now = datetime.now()
        pending_tasks = self.db.query(AnalysisTask).filter(
            AnalysisTask.task_type == 'review',
            AnalysisTask.status == 'pending',
            AnalysisTask.scheduled_at <= now
        ).all()

        for task in pending_tasks:
            self._execute_review(task, analyzer, position_manager)

    def _execute_review(self, task: AnalysisTask, analyzer: PositionAnalyzer,
                        position_manager: PositionManager):
        """执行单个复核"""
        symbol = task.symbol
        params = task.parameters or {}
        prediction_data = params.get('prediction_data', {})

        # 获取实际数据
        pos = position_manager.get_position(symbol)
        if not pos:
            task.status = 'cancelled'
            self.db.commit()
            return

        # 重新分析
        current_analysis = analyzer.analyze_position(symbol, pos)

        # 对比预测与实际
        review_result = self._compare_prediction(prediction_data, current_analysis)

        # 保存结果
        result = AnalysisResult(
            task_id=task.id,
            symbol=symbol,
            result_type='review',
            content=review_result
        )
        self.db.add(result)

        task.status = 'completed'
        self.db.commit()

        logger.info(f"复核完成: {symbol}, 准确率评估: {review_result.get('accuracy', 'N/A')}")

    def _compare_prediction(self, prediction: Dict, actual: Dict) -> Dict:
        """对比预测与实际结果"""
        review = {
            'prediction': prediction,
            'actual': actual,
            'accuracy': {},
            'optimization_suggestions': []
        }

        # 方向预测准确率
        pred_direction = prediction.get('direction')
        actual_price_change = actual.get('position', {}).get('unrealized_pnl_pct', 0)
        actual_direction = 'up' if actual_price_change > 0 else 'down'

        review['accuracy']['direction'] = pred_direction == actual_direction

        # 目标价偏差
        pred_target = prediction.get('target_price')
        actual_price = actual.get('current_price')
        if pred_target and actual_price:
            review['accuracy']['target_deviation'] = abs(pred_target - actual_price) / actual_price

        # 生成优化建议
        if not review['accuracy']['direction']:
            review['optimization_suggestions'].append(
                "方向预测错误，建议检查Layer 1趋势判断或Layer 2信号强度阈值"
            )

        if review['accuracy'].get('target_deviation', 0) > 0.05:
            review['optimization_suggestions'].append(
                "目标价偏差较大，建议调整预测模型参数或增加波动率考量"
            )

        return review
