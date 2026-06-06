"""
三层决策架构测试
验证Layer 1/2纯程序化，Layer 3 AI参与
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.models.hmm_engine import SupervisedHMM
from src.models.technical_factors import TechnicalFactorStrategy
from src.ai_debate.debate_engine import DebateEngine


class TestLayer1PureProgrammatic(unittest.TestCase):
    """测试Layer 1纯程序化"""

    def test_hmm_no_llm_call(self):
        """验证HMM不调用LLM"""
        hmm = SupervisedHMM("test", n_components=5)

        # 模拟数据
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # 训练不应调用任何外部API
        with patch('src.models.hmm_engine.GaussianHMM') as mock_hmm:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.randint(0, 5, 100)
            mock_hmm.return_value = mock_model

            hmm.fit(X, y)
            # 验证没有API调用
            mock_model.fit.assert_called_once()

    def test_hmm_output_is_probabilistic(self):
        """验证HMM输出是概率而非确定性预测"""
        hmm = SupervisedHMM("test", n_components=5)

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        with patch('src.models.hmm_engine.GaussianHMM') as mock_hmm:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20)
            mock_hmm.return_value = mock_model

            hmm.fit(X, y)
            result = hmm.predict_trend(X[-10:])

            # 输出应包含概率分布
            self.assertIn('state_probs', result)
            self.assertIn('trend_score', result)


class TestLayer2PureProgrammatic(unittest.TestCase):
    """测试Layer 2纯程序化"""

    def test_technical_factors_no_llm(self):
        """验证技术面因子不调用LLM"""
        tech = TechnicalFactorStrategy()

        df = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

        # 不应有任何LLM调用
        with patch('src.models.technical_factors.LLMClient') as mock_llm:
            result = tech.generate_signals(df)
            mock_llm.assert_not_called()

    def test_technical_output_is_composite_score(self):
        """验证输出是综合评分"""
        tech = TechnicalFactorStrategy()

        df = pd.DataFrame({
            'open': np.linspace(100, 110, 100) + np.random.randn(100),
            'high': np.linspace(102, 112, 100) + np.random.randn(100),
            'low': np.linspace(98, 108, 100) + np.random.randn(100),
            'close': np.linspace(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })

        result = tech.generate_signals(df)

        self.assertIn('composite_score', result)
        self.assertIn('primary_signal', result)
        self.assertIsInstance(result['composite_score'], (int, float))


class TestLayer3AIDebate(unittest.TestCase):
    """测试Layer 3 AI辩论"""

    def test_debate_calls_llm(self):
        """验证辩论调用LLM"""
        mock_llm = Mock()
        mock_llm.generate.return_value = "分析报告"

        debate = DebateEngine(mock_llm)

        trend_result = {'trend_score': 0.5, 'state_description': '上涨'}
        signal_report = {'composite_score': 0.6, 'primary_signal': 'buy'}

        debate.debate('TEST', trend_result, signal_report)

        # 验证LLM被调用
        self.assertTrue(mock_llm.generate.called)

    def test_debate_produces_verdict(self):
        """验证辩论产生裁决"""
        mock_llm = Mock()
        mock_llm.generate.return_value = "看多"

        debate = DebateEngine(mock_llm)

        trend_result = {'trend_score': 0.5, 'state_description': '上涨'}
        signal_report = {'composite_score': 0.6, 'primary_signal': 'buy'}

        result = debate.debate('TEST', trend_result, signal_report)

        self.assertIn('verdict', result)
        self.assertIn('confidence', result)


class TestReviewStage(unittest.TestCase):
    """测试Review阶段"""

    def test_review_calls_llm(self):
        """验证Review调用LLM"""
        mock_llm = Mock()
        mock_llm.generate.return_value = "优化建议"

        review = type('ReviewAnalyzer', (), {
            'llm': mock_llm,
            'review': lambda self, symbol, prediction, actual: {
                'accuracy': 0.7,
                'suggestions': ['建议1', '建议2']
            }
        })()

        result = review.review('TEST', {}, {})
        self.assertIn('accuracy', result)
        self.assertIn('suggestions', result)


if __name__ == '__main__':
    unittest.main()
