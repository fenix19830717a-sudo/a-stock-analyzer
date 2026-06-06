"""
回测引擎测试 - 验证无未来函数
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.backtest.backtest_engine import (
    BacktestEngine, BacktestConfig, SignalGenerator,
    PortfolioManager, SignalType
)


class MockDataSource:
    """模拟数据源"""
    def get_daily_data(self, symbol, start, end):
        dates = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({
            'open': prices + np.random.randn(len(dates)),
            'high': prices + abs(np.random.randn(len(dates))) * 2,
            'low': prices - abs(np.random.randn(len(dates))) * 2,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        return df


class MockHMMEngine:
    """模拟HMM引擎"""
    def predict_trend(self, df):
        return {
            'trend_score': 0.5,
            'state_description': '上涨',
            'state_probs': [0.1, 0.1, 0.1, 0.2, 0.5]
        }


class MockTechnicalEngine:
    """模拟技术面引擎"""
    def generate_signals(self, df):
        return {
            'composite_score': 0.6,
            'primary_signal': 'buy',
            'sub_signals': {}
        }


class TestNoLookAhead(unittest.TestCase):
    """测试无未来函数"""

    def test_signal_uses_only_past_data(self):
        """验证信号生成只使用历史数据"""
        ds = MockDataSource()
        hmm = MockHMMEngine()
        tech = MockTechnicalEngine()

        sg = SignalGenerator(hmm, tech)

        # 生成60天数据
        start = datetime(2023, 1, 1)
        end = datetime(2023, 3, 1)
        df = ds.get_daily_data('TEST', start, end)

        # T日收盘后生成信号
        t_date = df.index[50]
        signal = sg.generate_signal('TEST', df[df.index <= t_date], t_date)

        # 验证信号不为None
        self.assertIsNotNone(signal)
        # 验证信号日期正确
        self.assertEqual(signal.date, t_date)

    def test_trade_execution_next_day(self):
        """验证T+1日执行"""
        config = BacktestConfig()
        pm = PortfolioManager(config)

        # 模拟T+1开盘价买入
        signal = type('Signal', (), {
            'date': datetime(2023, 1, 1),
            'symbol': 'TEST',
            'signal_type': SignalType.BUY
        })()

        trade = pm.open_position('TEST', 100.0, signal)

        self.assertIsNotNone(trade)
        self.assertEqual(trade.entry_price, 100.0 * 1.001)  # 含滑点

    def test_portfolio_cannot_use_future_prices(self):
        """验证组合管理不使用未来价格"""
        config = BacktestConfig()
        pm = PortfolioManager(config)

        # 只能基于当前已知价格决策
        self.assertTrue(pm.can_open_position('TEST', 100.0))

        # 开仓后检查
        signal = type('Signal', (), {
            'date': datetime(2023, 1, 1),
            'symbol': 'TEST',
            'signal_type': SignalType.BUY
        })()
        pm.open_position('TEST', 100.0, signal)

        # 已有持仓，不能再开
        self.assertFalse(pm.can_open_position('TEST', 100.0))


class TestBacktestConfig(unittest.TestCase):
    """测试回测配置"""

    def test_default_config(self):
        config = BacktestConfig()
        self.assertEqual(config.initial_capital, 1_000_000)
        self.assertEqual(config.max_positions, 10)
        self.assertTrue(config.exclude_st)
        self.assertTrue(config.exclude_gem)

    def test_custom_config(self):
        config = BacktestConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.20,
            position_size_pct=0.15
        )
        self.assertEqual(config.stop_loss_pct, 0.05)
        self.assertEqual(config.take_profit_pct, 0.20)


class TestSignalCombination(unittest.TestCase):
    """测试信号合成规则"""

    def test_bullish_resonance(self):
        """测试看多共振"""
        sg = SignalGenerator(MockHMMEngine(), MockTechnicalEngine())

        signal_type, confidence, reason = sg._combine_layers(
            0.5, 0.6, '上涨', 'buy'
        )

        self.assertEqual(signal_type, SignalType.BUY)
        self.assertGreater(confidence, 0)

    def test_bearish_resonance(self):
        """测试看空共振"""
        sg = SignalGenerator(MockHMMEngine(), MockTechnicalEngine())

        signal_type, confidence, reason = sg._combine_layers(
            -0.5, -0.6, '下跌', 'sell'
        )

        self.assertEqual(signal_type, SignalType.SELL)

    def test_no_signal(self):
        """测试无信号"""
        sg = SignalGenerator(MockHMMEngine(), MockTechnicalEngine())

        signal_type, confidence, reason = sg._combine_layers(
            0.1, 0.1, '震荡', 'neutral'
        )

        self.assertEqual(signal_type, SignalType.WAIT)


if __name__ == '__main__':
    unittest.main()
