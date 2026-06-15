"""
回测引擎 - 无未来函数回测系统
核心原则:
1. T日收盘后生成信号，T+1日开盘价执行
2. 严格避免使用未来数据
3. 排除ST个股
4. 模拟真实交易环境（滑点、手续费）
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.data.database import get_db, BacktestResult, TradeRecord
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"
    WAIT = "观望"


class OrderType(Enum):
    MARKET_OPEN = "开盘价"  # T+1开盘价执行
    MARKET_CLOSE = "收盘价"  # 仅用于回测基准对比


@dataclass
class TradeSignal:
    """交易信号"""
    date: datetime
    symbol: str
    signal_type: SignalType
    price_target: Optional[float] = None  # 目标价
    stop_loss: Optional[float] = None     # 止损价
    confidence: float = 0.0               # 信号置信度 (0-1)
    layer1_score: float = 0.0             # Layer 1 HMM评分
    layer2_score: float = 0.0             # Layer 2 技术面评分
    layer3_score: float = 0.0             # Layer 3 AI辩论评分
    reason: str = ""                      # 信号理由


@dataclass
class TradeExecution:
    """交易执行记录"""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    symbol: str = ""
    direction: str = "long"  # long/short
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""  # stop_loss / target / signal_reverse / timeout
    transaction_cost: float = 0.0


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000.0
    max_positions: int = 10               # 最大持仓数
    position_size_pct: float = 0.1        # 单票仓位比例
    commission_rate: float = 0.0003       # 手续费率 (万3)
    stamp_tax_rate: float = 0.001         # 印花税 (千1，卖出收)
    slippage: float = 0.001               # 滑点 (千1)
    stop_loss_pct: float = 0.07           # 止损比例 7%
    take_profit_pct: float = 0.15         # 止盈比例 15%
    max_holding_days: int = 20            # 最大持仓天数
    exclude_st: bool = True               # 排除ST股
    exclude_gem: bool = True              # 排除创业板
    min_price: float = 5.0                # 最低股价
    benchmark: str = "000001.SH"          # 基准指数


@dataclass
class DailyPortfolio:
    """每日投资组合状态"""
    date: datetime
    cash: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    positions: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class BacktestReport:
    """回测报告"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_days: float = 0.0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    daily_records: List[DailyPortfolio] = field(default_factory=list)
    trades: List[TradeExecution] = field(default_factory=list)
    signals: List[TradeSignal] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class SignalGenerator:
    """
    信号生成器 - 纯程序化，无AI参与
    整合Layer 1 (HMM) + Layer 2 (技术面因子) 生成交易信号
    """

    def __init__(self, hmm_engine, technical_engine):
        self.hmm_engine = hmm_engine
        self.technical_engine = technical_engine
        self.logger = get_logger(self.__class__.__name__)

    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        current_date: datetime) -> Optional[TradeSignal]:
        """
        生成交易信号 - T日收盘后调用

        Args:
            symbol: 股票代码
            df: 历史数据到T日收盘
            current_date: T日日期

        Returns:
            TradeSignal or None
        """
        if len(df) < 60:
            return None

        # Layer 1: HMM趋势定性分析
        try:
            trend_result = self.hmm_engine.analyze_trend(symbol, df)
            if 'error' in trend_result:
                layer1_score = 0
                trend_state = 'unknown'
            else:
                # 从analyze_trend结果提取trend_score
                direction = trend_result.get('trend_direction', '震荡')
                up_prob = trend_result.get('up_probability', 0.5)
                if direction == '上升':
                    layer1_score = up_prob * 0.8
                elif direction == '下降':
                    layer1_score = -(1 - up_prob) * 0.8
                else:
                    layer1_score = (up_prob - 0.5) * 0.4
                trend_state = trend_result.get('trend_direction', 'unknown')
        except Exception as e:
            self.logger.warning(f"HMM分析失败 {symbol}: {e}")
            layer1_score = 0
            trend_state = 'unknown'

        # Layer 2: 技术面因子信号
        try:
            trend_qual = {'trend_direction': trend_state, 'trend_strength': '中'}
            signal_report = self.technical_engine.generate_signal(symbol, df, trend_qual)
            direction = signal_report.get('direction', '震荡')
            probability = signal_report.get('probability', 50)
            if direction == '涨':
                layer2_score = probability / 100
            elif direction == '跌':
                layer2_score = -probability / 100
            else:
                layer2_score = 0
            if layer2_score > 0.5:
                primary_signal = 'strong_buy'
            elif layer2_score > 0.2:
                primary_signal = 'buy'
            elif layer2_score < -0.5:
                primary_signal = 'strong_sell'
            elif layer2_score < -0.2:
                primary_signal = 'sell'
            else:
                primary_signal = 'neutral'
        except Exception as e:
            self.logger.warning(f"技术面分析失败 {symbol}: {e}")
            layer2_score = 0
            primary_signal = 'neutral'

        # 信号合成规则 (纯程序化)
        signal_type, confidence, reason = self._combine_layers(
            layer1_score, layer2_score, trend_state, primary_signal
        )

        if signal_type == SignalType.WAIT:
            return None

        # 计算目标价和止损价 (基于T日收盘价)
        current_price = df['close'].iloc[-1]
        if signal_type == SignalType.BUY:
            target = current_price * 1.15
            stop = current_price * 0.93
        elif signal_type == SignalType.SELL:
            target = current_price * 0.85
            stop = current_price * 1.07
        else:
            target = None
            stop = None

        return TradeSignal(
            date=current_date,
            symbol=symbol,
            signal_type=signal_type,
            price_target=target,
            stop_loss=stop,
            confidence=confidence,
            layer1_score=layer1_score,
            layer2_score=layer2_score,
            layer3_score=0,  # Layer 3在回测中不参与
            reason=reason
        )

    def _combine_layers(self, layer1_score: float, layer2_score: float,
                        trend_state: str, primary_signal: str) -> Tuple[SignalType, float, str]:
        """
        纯程序化信号合成规则

        买入条件:
        - Layer 1: 趋势评分 > 0.3 (偏多状态)
        - Layer 2: 综合信号为 buy 或 strong_buy
        - 两者共振

        卖出条件:
        - Layer 1: 趋势评分 < -0.3 (偏空状态)
        - Layer 2: 综合信号为 sell 或 strong_sell
        - 两者共振
        """
        # 趋势状态映射
        bullish_states = ['强烈上涨', '上涨', '偏多震荡']
        bearish_states = ['强烈下跌', '下跌', '偏空震荡']

        layer1_bullish = trend_state in bullish_states or layer1_score > 0.3
        layer1_bearish = trend_state in bearish_states or layer1_score < -0.3

        layer2_bullish = primary_signal in ['strong_buy', 'buy']
        layer2_bearish = primary_signal in ['strong_sell', 'sell']

        # 共振判断
        if layer1_bullish and layer2_bullish:
            confidence = min(0.95, (abs(layer1_score) + abs(layer2_score)) / 2)
            reason = f"Layer1[{trend_state}, {layer1_score:.2f}] + Layer2[{primary_signal}, {layer2_score:.2f}] 共振看多"
            return SignalType.BUY, confidence, reason

        if layer1_bearish and layer2_bearish:
            confidence = min(0.95, (abs(layer1_score) + abs(layer2_score)) / 2)
            reason = f"Layer1[{trend_state}, {layer1_score:.2f}] + Layer2[{primary_signal}, {layer2_score:.2f}] 共振看空"
            return SignalType.SELL, confidence, reason

        # 单一信号但较强
        if layer2_bullish and layer2_score > 0.6:
            return SignalType.BUY, layer2_score * 0.7, f"Layer2强势买入信号[{primary_signal}]"

        if layer2_bearish and layer2_score < -0.6:
            return SignalType.SELL, abs(layer2_score) * 0.7, f"Layer2强势卖出信号[{primary_signal}]"

        return SignalType.WAIT, 0.0, "无明确信号"


class PortfolioManager:
    """组合管理器 - 处理仓位、风控"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.daily_records: List[DailyPortfolio] = []

    def can_open_position(self, symbol: str, price: float) -> bool:
        """检查是否可以开仓"""
        if len(self.positions) >= self.config.max_positions:
            return False
        if symbol in self.positions:
            return False

        position_value = self.config.initial_capital * self.config.position_size_pct
        shares = int(position_value / price / 100) * 100  # 100股整数
        cost = shares * price * (1 + self.config.commission_rate)

        return cost <= self.cash and shares > 0

    def open_position(self, symbol: str, price: float,
                      signal: TradeSignal) -> Optional[TradeExecution]:
        """开仓 - T+1开盘价执行"""
        position_value = self.config.initial_capital * self.config.position_size_pct
        shares = int(position_value / price / 100) * 100

        if shares <= 0:
            return None

        # 应用滑点
        executed_price = price * (1 + self.config.slippage)
        cost = shares * executed_price
        commission = cost * self.config.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            # 调整股数
            shares = int(self.cash / executed_price / 100) * 100
            if shares <= 0:
                return None
            cost = shares * executed_price
            commission = cost * self.config.commission_rate
            total_cost = cost + commission

        self.cash -= total_cost
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': executed_price,
            'entry_date': signal.date,
            'signal': signal,
            'highest_price': executed_price,
            'lowest_price': executed_price
        }

        execution = TradeExecution(
            entry_date=signal.date,
            symbol=symbol,
            direction='long',
            entry_price=executed_price,
            shares=shares,
            transaction_cost=commission
        )

        logger.info(f"开仓 {symbol}: {shares}股 @ {executed_price:.2f}, 成本{total_cost:.2f}")
        return execution

    def close_position(self, symbol: str, price: float,
                       date: datetime, reason: str) -> Optional[TradeExecution]:
        """平仓 - T+1开盘价执行"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        shares = pos['shares']

        # 应用滑点
        executed_price = price * (1 - self.config.slippage)
        revenue = shares * executed_price
        commission = revenue * self.config.commission_rate
        stamp_tax = revenue * self.config.stamp_tax_rate
        total_cost = commission + stamp_tax

        pnl = shares * (executed_price - pos['entry_price']) - total_cost
        pnl_pct = (executed_price - pos['entry_price']) / pos['entry_price']
        holding_days = (date - pos['entry_date']).days

        self.cash += revenue - total_cost

        execution = TradeExecution(
            entry_date=pos['entry_date'],
            exit_date=date,
            symbol=symbol,
            direction='long',
            entry_price=pos['entry_price'],
            exit_price=executed_price,
            shares=shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=reason,
            transaction_cost=total_cost
        )

        del self.positions[symbol]

        logger.info(f"平仓 {symbol}: {shares}股 @ {executed_price:.2f}, "
                   f"盈亏{pnl:.2f}({pnl_pct*100:.1f}%), 原因:{reason}")
        return execution

    def update_positions(self, date: datetime, market_data: Dict[str, pd.DataFrame]) -> List[TradeExecution]:
        """更新持仓状态，检查止盈止损"""
        closed_trades = []

        for symbol in list(self.positions.keys()):
            if symbol not in market_data:
                continue

            df = market_data[symbol]
            if len(df) == 0:
                continue

            current_price = df['close'].iloc[-1]
            high_price = df['high'].iloc[-1]
            low_price = df['low'].iloc[-1]
            pos = self.positions[symbol]

            # 更新最高/最低价
            pos['highest_price'] = max(pos['highest_price'], high_price)
            pos['lowest_price'] = min(pos['lowest_price'], low_price)

            # 检查止损
            stop_price = pos['entry_price'] * (1 - self.config.stop_loss_pct)
            if low_price <= stop_price:
                trade = self.close_position(symbol, stop_price, date, 'stop_loss')
                if trade:
                    closed_trades.append(trade)
                continue

            # 检查止盈
            target_price = pos['entry_price'] * (1 + self.config.take_profit_pct)
            if high_price >= target_price:
                trade = self.close_position(symbol, target_price, date, 'take_profit')
                if trade:
                    closed_trades.append(trade)
                continue

            # 检查最大持仓天数
            holding_days = (date - pos['entry_date']).days
            if holding_days >= self.config.max_holding_days:
                trade = self.close_position(symbol, current_price, date, 'timeout')
                if trade:
                    closed_trades.append(trade)
                continue

        return closed_trades

    def get_portfolio_value(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """计算组合总市值"""
        positions_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in market_data and len(market_data[symbol]) > 0:
                price = market_data[symbol]['close'].iloc[-1]
                positions_value += pos['shares'] * price
        return self.cash + positions_value

    def record_daily(self, date: datetime, market_data: Dict[str, pd.DataFrame]):
        """记录每日状态"""
        positions_value = 0.0
        positions_info = {}

        for symbol, pos in self.positions.items():
            if symbol in market_data and len(market_data[symbol]) > 0:
                price = market_data[symbol]['close'].iloc[-1]
                value = pos['shares'] * price
                positions_value += value
                positions_info[symbol] = {
                    'shares': pos['shares'],
                    'entry_price': pos['entry_price'],
                    'current_price': price,
                    'value': value,
                    'pnl': value - pos['shares'] * pos['entry_price'],
                    'pnl_pct': (price - pos['entry_price']) / pos['entry_price']
                }

        total = self.cash + positions_value

        daily = DailyPortfolio(
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total,
            positions=positions_info
        )

        if len(self.daily_records) > 0:
            prev_total = self.daily_records[-1].total_value
            daily.daily_pnl = total - prev_total
            daily.daily_return = daily.daily_pnl / prev_total if prev_total > 0 else 0

        self.daily_records.append(daily)
        return daily


class BacktestEngine:
    """
    回测引擎主类

    执行流程:
    1. 加载历史数据
    2. 逐日遍历 (T日)
    3. T日收盘后 -> 生成信号 (Layer 1 + Layer 2)
    4. T+1日开盘 -> 执行交易
    5. 记录结果
    """

    def __init__(self, data_source, hmm_engine, technical_engine,
                 backtest_config: Optional[BacktestConfig] = None):
        self.data_source = data_source
        self.signal_generator = SignalGenerator(hmm_engine, technical_engine)
        self.config = backtest_config or BacktestConfig()
        self.portfolio = PortfolioManager(self.config)
        self.trades: List[TradeExecution] = []
        self.signals: List[TradeSignal] = []

    def run(self, symbols: List[str], start_date: datetime,
            end_date: datetime) -> BacktestReport:
        """
        运行回测

        Args:
            symbols: 股票池
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        logger.info(f"开始回测: {start_date.date()} ~ {end_date.date()}, 股票池{len(symbols)}只")

        # 加载数据
        all_data = self._load_data(symbols, start_date, end_date)
        valid_symbols = list(all_data.keys())
        logger.info(f"有效数据: {len(valid_symbols)}只股票")

        # 获取交易日历
        trading_days = self._get_trading_days(all_data, start_date, end_date)
        logger.info(f"交易日: {len(trading_days)}天")

        # 逐日回测
        for i, current_date in enumerate(trading_days):
            logger.debug(f"回测日期: {current_date.date()}")

            # 获取T日数据 (收盘后可用)
            day_data = self._get_day_data(all_data, current_date)

            # 更新持仓 (检查止盈止损)
            closed_trades = self.portfolio.update_positions(current_date, day_data)
            self.trades.extend(closed_trades)

            # T日收盘后生成信号
            signals_today = []
            for symbol in valid_symbols:
                if symbol in self.portfolio.positions:
                    continue  # 已有持仓，不生成新信号

                df = self._get_data_upto_date(all_data, symbol, current_date)
                if df is None or len(df) < 60:
                    continue

                signal = self.signal_generator.generate_signal(symbol, df, current_date)
                if signal and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    signals_today.append(signal)

            # 按置信度排序，选择前N个
            signals_today.sort(key=lambda x: x.confidence, reverse=True)
            selected_signals = signals_today[:self.config.max_positions - len(self.portfolio.positions)]
            self.signals.extend(selected_signals)

            # T+1日开盘执行 (如果存在T+1数据)
            if i + 1 < len(trading_days):
                next_date = trading_days[i + 1]
                next_day_data = self._get_day_data(all_data, next_date)

                for signal in selected_signals:
                    if signal.signal_type == SignalType.BUY:
                        # 买入 - T+1开盘价
                        if signal.symbol in next_day_data:
                            open_price = next_day_data[signal.symbol]['open'].iloc[0]
                            if self.portfolio.can_open_position(signal.symbol, open_price):
                                trade = self.portfolio.open_position(signal.symbol, open_price, signal)
                                if trade:
                                    self.trades.append(trade)

                    elif signal.signal_type == SignalType.SELL:
                        # 卖出信号 - 如果持有则平仓
                        if signal.symbol in self.portfolio.positions:
                            open_price = next_day_data[signal.symbol]['open'].iloc[0]
                            trade = self.portfolio.close_position(
                                signal.symbol, open_price, next_date, 'signal_reverse'
                            )
                            if trade:
                                self.trades.append(trade)

            # 记录每日组合状态
            self.portfolio.record_daily(current_date, day_data)

        # 回测结束，平仓所有持仓
        if trading_days:
            last_date = trading_days[-1]
            last_data = self._get_day_data(all_data, last_date)
            for symbol in list(self.portfolio.positions.keys()):
                if symbol in last_data:
                    close_price = last_data[symbol]['close'].iloc[0]
                    trade = self.portfolio.close_position(symbol, close_price, last_date, 'backtest_end')
                    if trade:
                        self.trades.append(trade)

        # 生成报告
        report = self._generate_report(start_date, end_date, trading_days)
        logger.info(f"回测完成: 总收益{report.total_return*100:.2f}%, 夏普{report.sharpe_ratio:.2f}")

        return report

    def _load_data(self, symbols: List[str], start_date: datetime,
                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """加载历史数据"""
        all_data = {}
        # 预留更多历史用于计算指标
        load_start = start_date - timedelta(days=120)

        for symbol in symbols:
            try:
                df = self.data_source.get_kline(symbol, load_start.strftime('%Y%m%d') if hasattr(load_start, 'strftime') else str(load_start).replace('-',''), end_date.strftime('%Y%m%d') if hasattr(end_date, 'strftime') else str(end_date).replace('-',''))
                if df is not None and len(df) >= 60:
                    # 排除ST股
                    if self.config.exclude_st and self._is_st_stock(df):
                        continue
                    # 排除创业板
                    if self.config.exclude_gem and symbol.startswith('300'):
                        continue
                    # 最低股价检查
                    if df['close'].iloc[-1] < self.config.min_price:
                        continue

                    all_data[symbol] = df
            except Exception as e:
                logger.warning(f"加载{symbol}数据失败: {e}")

        return all_data

    def _is_st_stock(self, df: pd.DataFrame) -> bool:
        """检查是否为ST股 (基于名称或特殊标记)"""
        # 简化判断：如果数据中有name列且包含ST
        if 'name' in df.columns:
            return 'ST' in str(df['name'].iloc[-1])
        return False

    def _get_trading_days(self, all_data: Dict[str, pd.DataFrame],
                          start_date: datetime, end_date: datetime) -> List[datetime]:
        """获取交易日历"""
        if not all_data:
            return []

        # 使用第一只股票的数据确定交易日
        first_df = list(all_data.values())[0]
        mask = (first_df.index >= start_date) & (first_df.index <= end_date)
        return list(first_df.index[mask])

    def _get_day_data(self, all_data: Dict[str, pd.DataFrame],
                      date: datetime) -> Dict[str, pd.DataFrame]:
        """获取某日的数据切片"""
        result = {}
        for symbol, df in all_data.items():
            if date in df.index:
                result[symbol] = df.loc[[date]]
        return result

    def _get_data_upto_date(self, all_data: Dict[str, pd.DataFrame],
                            symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """获取截止到某日的历史数据 (无未来函数)"""
        if symbol not in all_data:
            return None
        df = all_data[symbol]
        mask = df.index <= date
        return df[mask]

    def _generate_report(self, start_date: datetime, end_date: datetime,
                         trading_days: List[datetime]) -> BacktestReport:
        """生成回测报告"""
        report = BacktestReport(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            daily_records=self.portfolio.daily_records,
            trades=self.trades,
            signals=self.signals
        )

        if not self.portfolio.daily_records:
            return report

        # 基础统计
        initial_value = self.config.initial_capital
        final_value = self.portfolio.daily_records[-1].total_value
        report.total_return = (final_value - initial_value) / initial_value

        # 年化收益
        days = max(1, (end_date - start_date).days)
        report.annual_return = (1 + report.total_return) ** (365 / days) - 1

        # 最大回撤
        values = [d.total_value for d in self.portfolio.daily_records]
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        report.max_drawdown = max_dd

        # 夏普比率
        returns = [d.daily_return for d in self.portfolio.daily_records[1:]]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                report.sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))

        # 交易统计
        report.total_trades = len(self.trades)
        report.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        report.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        report.win_rate = report.winning_trades / report.total_trades if report.total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        if self.trades:
            report.avg_holding_days = np.mean([t.holding_days for t in self.trades])

        # 权益曲线
        report.equity_curve = pd.DataFrame({
            'date': [d.date for d in self.portfolio.daily_records],
            'total_value': [d.total_value for d in self.portfolio.daily_records],
            'cash': [d.cash for d in self.portfolio.daily_records],
            'positions_value': [d.positions_value for d in self.portfolio.daily_records]
        })

        return report

    def save_results(self, report: BacktestReport, db: Session):
        """保存回测结果到数据库"""
        result = BacktestResult(
            start_date=report.start_date,
            end_date=report.end_date,
            initial_capital=self.config.initial_capital,
            final_value=report.daily_records[-1].total_value if report.daily_records else self.config.initial_capital,
            total_return=report.total_return,
            annual_return=report.annual_return,
            max_drawdown=report.max_drawdown,
            sharpe_ratio=report.sharpe_ratio,
            win_rate=report.win_rate,
            total_trades=report.total_trades,
            parameters=self.config.__dict__
        )
        db.add(result)
        db.commit()

        # 保存交易记录
        for trade in report.trades:
            record = TradeRecord(
                backtest_id=result.id,
                symbol=trade.symbol,
                entry_date=trade.entry_date,
                exit_date=trade.exit_date,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                shares=trade.shares,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                holding_days=trade.holding_days,
                exit_reason=trade.exit_reason
            )
            db.add(record)

        db.commit()
        logger.info(f"回测结果已保存: ID={result.id}")
        return result.id


class WalkForwardOptimizer:
    """
    滚动前向优化器 - 避免未来函数的参数优化

    方法:
    1. 将数据分为训练集和测试集
    2. 在训练集上优化参数
    3. 在测试集上验证
    4. 滚动窗口前进
    """

    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine

    def optimize(self, symbols: List[str], start_date: datetime, end_date: datetime,
                 train_window: int = 252, test_window: int = 63) -> Dict:
        """
        滚动前向优化

        Args:
            train_window: 训练窗口天数 (默认1年)
            test_window: 测试窗口天数 (默认1季度)
        """
        logger.info("开始滚动前向优化...")

        current_date = start_date
        results = []

        while current_date + timedelta(days=train_window + test_window) <= end_date:
            train_end = current_date + timedelta(days=train_window)
            test_end = min(train_end + timedelta(days=test_window), end_date)

            # 训练期：参数优化
            best_params = self._optimize_params(symbols, current_date, train_end)

            # 测试期：验证
            self.engine.config = BacktestConfig(**best_params)
            report = self.engine.run(symbols, train_end, test_end)

            results.append({
                'train_start': current_date,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'params': best_params,
                'test_return': report.total_return,
                'test_sharpe': report.sharpe_ratio
            })

            current_date += timedelta(days=test_window)

        # 汇总结果
        avg_return = np.mean([r['test_return'] for r in results])
        avg_sharpe = np.mean([r['test_sharpe'] for r in results])

        logger.info(f"优化完成: 平均收益{avg_return*100:.2f}%, 平均夏普{avg_sharpe:.2f}")

        return {
            'results': results,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe
        }

    def _optimize_params(self, symbols: List[str], start_date: datetime,
                         end_date: datetime) -> Dict:
        """简单网格搜索优化参数"""
        best_sharpe = -float('inf')
        best_params = {}

        # 参数网格
        param_grid = {
            'stop_loss_pct': [0.05, 0.07, 0.10],
            'take_profit_pct': [0.10, 0.15, 0.20],
            'position_size_pct': [0.05, 0.10, 0.15]
        }

        for sl in param_grid['stop_loss_pct']:
            for tp in param_grid['take_profit_pct']:
                for ps in param_grid['position_size_pct']:
                    config = BacktestConfig(
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        position_size_pct=ps
                    )
                    self.engine.config = config
                    report = self.engine.run(symbols, start_date, end_date)

                    if report.sharpe_ratio > best_sharpe:
                        best_sharpe = report.sharpe_ratio
                        best_params = {
                            'stop_loss_pct': sl,
                            'take_profit_pct': tp,
                            'position_size_pct': ps
                        }

        return best_params
