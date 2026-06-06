"""
A股智能分析选股系统 - 主入口

三层决策架构:
Layer 1: HMM趋势定性分析 (程序化)
Layer 2: 技术面因子策略信号 (程序化)
Layer 3: AI辩论决策分析 (AI)

Review阶段: 次日对决策分析结果进行复核，输出优化建议
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.config import config
from src.utils.logger import get_logger, setup_logging
from src.data.database import init_db, get_db
from src.data.data_source import DataSourceManager
from src.models.feature_engineering import FeatureEngineer
from src.models.hmm_engine import SupervisedHMM, HMMTrendAnalyzer, HMMTrainingPipeline
from src.models.technical_factors import TechnicalFactorStrategy
from src.ai_debate.debate_engine import DebateEngine, ReviewAnalyzer
from src.ai_debate.llm_client import LLMClient
from src.analysis.forecast import ForecastReportGenerator
from src.analysis.screener import StockScreener
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig, WalkForwardOptimizer
from src.portfolio.portfolio_manager import PositionManager, PositionAnalyzer, ReviewScheduler
from src.sentiment.collector import SentimentPipeline

logger = get_logger(__name__)


class AStockAnalyzer:
    """
    A股智能分析选股系统主类

    整合所有模块，提供统一接口
    """

    def __init__(self):
        setup_logging()
        logger.info("=" * 60)
        logger.info("A股智能分析选股系统启动")
        logger.info("=" * 60)

        # 初始化数据库
        init_db()
        logger.info("数据库初始化完成")

        # 初始化数据源
        self.data_source = DataSourceManager()
        logger.info("数据源管理器初始化完成")

        # 初始化特征工程
        self.feature_engineer = FeatureEngineer()

        # 初始化HMM引擎
        self.hmm_engine = HMMTrendAnalyzer(
            market_hmm=SupervisedHMM("market", n_components=5),
            sector_hmms={},
            stock_hmms={}
        )

        # 初始化技术面因子
        self.technical_engine = TechnicalFactorStrategy()

        # 初始化AI辩论
        llm_client = LLMClient()
        self.debate_engine = DebateEngine(llm_client)
        self.review_analyzer = ReviewAnalyzer(llm_client)

        # 初始化预测报告生成器
        self.forecast_generator = ForecastReportGenerator()

        # 初始化股票筛选器
        self.screener = StockScreener()

        # 初始化持仓管理
        self.position_manager = PositionManager()
        self.position_analyzer = PositionAnalyzer(
            self.data_source,
            self.hmm_engine,
            self.technical_engine,
            self.debate_engine,
            self.forecast_generator
        )
        self.review_scheduler = ReviewScheduler()

        # 初始化舆情采集
        self.sentiment_pipeline = SentimentPipeline()

        logger.info("所有模块初始化完成")

    def analyze_stock(self, symbol: str) -> dict:
        """
        对单只股票执行完整的三层决策分析

        Args:
            symbol: 股票代码 (如 '600519.SH')

        Returns:
            包含三层分析结果的字典
        """
        logger.info(f"开始分析股票: {symbol}")

        # 获取数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        df = self.data_source.get_daily_data(symbol, start_date, end_date)

        if df is None or len(df) < 60:
            return {"error": "数据不足，无法分析"}

        current_price = df['close'].iloc[-1]
        name = symbol  # 实际应从数据源获取名称

        # Layer 1: HMM趋势定性分析 (程序化)
        logger.info("Layer 1: HMM趋势定性分析")
        trend_result = self.hmm_engine.predict_trend(df)

        # Layer 2: 技术面因子策略信号 (程序化)
        logger.info("Layer 2: 技术面因子策略信号")
        signal_report = self.technical_engine.generate_signals(df)

        # Layer 3: AI辩论决策分析
        logger.info("Layer 3: AI辩论决策分析")
        debate_result = self.debate_engine.debate(symbol, trend_result, signal_report)

        # 生成天气预报式预测报告
        forecast = self.forecast_generator.generate(
            symbol=symbol,
            name=name,
            current_price=current_price,
            trend_qualification=trend_result,
            signal_report=signal_report,
            debate_result=debate_result
        )

        # 调度Review任务
        self.review_scheduler.schedule_review(
            symbol=symbol,
            prediction_date=datetime.now(),
            prediction_data=forecast.to_dict()
        )

        result = {
            'symbol': symbol,
            'name': name,
            'analysis_time': datetime.now().isoformat(),
            'current_price': current_price,
            'layer1_trend': trend_result,
            'layer2_signal': signal_report,
            'layer3_debate': debate_result,
            'forecast': forecast.to_dict()
        }

        logger.info(f"股票分析完成: {symbol}")
        return result

    def screen_stocks(self, stock_pool: Optional[list] = None) -> list:
        """
        股票筛选

        Args:
            stock_pool: 股票池，None则使用默认沪深300成分股

        Returns:
            筛选后的股票列表及评分
        """
        logger.info("开始股票筛选")

        if stock_pool is None:
            # 获取沪深300成分股
            stock_pool = self.data_source.get_index_components('000300.SH')

        results = []
        for symbol in stock_pool:
            try:
                score = self.screener.screen(symbol)
                if score['passed']:
                    results.append(score)
            except Exception as e:
                logger.warning(f"筛选 {symbol} 失败: {e}")

        # 按综合评分排序
        results.sort(key=lambda x: x['composite_score'], reverse=True)

        logger.info(f"筛选完成: {len(results)}只股票通过")
        return results

    def run_backtest(self, symbols: list, start_date: datetime,
                     end_date: datetime, config: Optional[BacktestConfig] = None) -> dict:
        """
        运行回测

        Args:
            symbols: 股票池
            start_date: 开始日期
            end_date: 结束日期
            config: 回测配置

        Returns:
            回测报告
        """
        logger.info(f"开始回测: {start_date.date()} ~ {end_date.date()}")

        backtest_engine = BacktestEngine(
            self.data_source,
            self.hmm_engine,
            self.technical_engine,
            config
        )

        report = backtest_engine.run(symbols, start_date, end_date)

        # 保存结果
        db = next(get_db())
        backtest_engine.save_results(report, db)

        logger.info("回测完成")
        return report

    def optimize_parameters(self, symbols: list, start_date: datetime,
                           end_date: datetime) -> dict:
        """
        滚动前向优化参数

        Args:
            symbols: 股票池
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            优化结果
        """
        logger.info("开始参数优化")

        backtest_engine = BacktestEngine(
            self.data_source,
            self.hmm_engine,
            self.technical_engine
        )

        optimizer = WalkForwardOptimizer(backtest_engine)
        result = optimizer.optimize(symbols, start_date, end_date)

        logger.info("参数优化完成")
        return result

    def update_portfolio_prices(self):
        """更新持仓价格"""
        positions = self.position_manager.get_all_positions()
        if not positions:
            return

        symbols = [p.symbol for p in positions]
        price_data = {}

        for symbol in symbols:
            try:
                df = self.data_source.get_daily_data(symbol, datetime.now() - timedelta(days=5), datetime.now())
                if df is not None and len(df) > 0:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    price_data[symbol] = {
                        'close': latest['close'],
                        'prev_close': prev['close'],
                        'high': latest['high'],
                        'low': latest['low']
                    }
            except Exception as e:
                logger.warning(f"获取 {symbol} 价格失败: {e}")

        self.position_manager.update_prices(price_data)
        logger.info("持仓价格更新完成")

    def check_portfolio_alerts(self) -> list:
        """检查持仓预警"""
        alerts = self.position_manager.check_alerts()
        if alerts:
            logger.info(f"发现 {len(alerts)} 条持仓预警")
        return alerts

    def collect_sentiment(self, category: str = "stock", num: int = 50) -> dict:
        """
        采集舆情信息

        Args:
            category: 新闻类别
            num: 采集数量

        Returns:
            采集结果统计
        """
        logger.info("开始舆情采集")
        result = self.sentiment_pipeline.run(category, num)
        logger.info("舆情采集完成")
        return result

    def execute_reviews(self):
        """执行待处理的Review任务"""
        logger.info("执行Review任务")
        self.review_scheduler.execute_pending_reviews(
            self.position_analyzer,
            self.position_manager
        )
        logger.info("Review任务执行完成")

    def get_sentiment_context(self, symbol: str, industry: str = None) -> str:
        """
        获取舆情上下文供AI辩论使用

        Args:
            symbol: 股票代码
            industry: 行业名称

        Returns:
            格式化的舆情摘要
        """
        return self.sentiment_pipeline.get_context_for_debate(symbol, industry)


def run_streamlit():
    """运行Streamlit UI"""
    import subprocess
    app_path = os.path.join(project_root, "app.py")
    subprocess.run(["streamlit", "run", app_path, "--server.port=8501"])


def run_cli():
    """命令行模式"""
    analyzer = AStockAnalyzer()

    print("=" * 60)
    print("A股智能分析选股系统 - CLI模式")
    print("=" * 60)

    while True:
        print("\n选项:")
        print("1. 分析单只股票")
        print("2. 股票筛选")
        print("3. 运行回测")
        print("4. 查看持仓")
        print("5. 采集舆情")
        print("6. 退出")

        choice = input("\n选择: ").strip()

        if choice == "1":
            symbol = input("输入股票代码: ").strip()
            result = analyzer.analyze_stock(symbol)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

        elif choice == "2":
            results = analyzer.screen_stocks()
            print(f"筛选结果: {len(results)}只股票")
            for r in results[:10]:
                print(f"{r['symbol']}: 评分 {r['composite_score']:.2f}")

        elif choice == "3":
            print("回测功能需要更多参数配置")

        elif choice == "4":
            positions = analyzer.position_manager.get_all_positions()
            print(f"当前持仓: {len(positions)}只股票")
            for p in positions:
                print(f"{p.symbol}: {p.shares}股 @ {p.entry_price}")

        elif choice == "5":
            result = analyzer.collect_sentiment()
            print(f"采集完成: {result}")

        elif choice == "6":
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A股智能分析选股系统")
    parser.add_argument("--mode", choices=["ui", "cli"], default="ui",
                        help="运行模式: ui=Streamlit界面, cli=命令行")

    args = parser.parse_args()

    if args.mode == "ui":
        run_streamlit()
    else:
        run_cli()
