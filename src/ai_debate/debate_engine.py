"""AI辩论引擎 - Layer 3 决策分析"""
from typing import Dict, List
from datetime import datetime
from src.ai_debate.llm_client import LLMClient
from src.ai_debate import prompts
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalystAgent:
    """分析师Agent基类"""

    def __init__(self, name: str, system_prompt: str, llm: LLMClient):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm

    def analyze(self, data: Dict) -> str:
        """生成分析报告"""
        user_prompt = self._format_data(data)
        return self.llm.chat(self.system_prompt, user_prompt)

    def _format_data(self, data: Dict) -> str:
        """格式化数据为提示词"""
        return f"""
股票代码: {data.get('symbol', 'Unknown')}
股票名称: {data.get('name', 'Unknown')}
当前价格: {data.get('price', 'Unknown')}

HMM趋势定性:
{data.get('trend_qualification', 'N/A')}

技术面信号:
{data.get('signal_report', 'N/A')}

近期数据:
{data.get('recent_data', 'N/A')}
"""


class TechnicalAnalyst(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("技术面分析师", prompts.TECHNICAL_ANALYST_PROMPT, llm)


class FundamentalAnalyst(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("基本面分析师", prompts.FUNDAMENTAL_ANALYST_PROMPT, llm)


class MacroAnalyst(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("宏观分析师", prompts.MACRO_ANALYST_PROMPT, llm)


class BullResearcher(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("多头研究员", prompts.BULL_RESEARCHER_PROMPT, llm)

    def argue(self, analyst_reports: Dict, hmm_prediction: Dict) -> str:
        """生成看涨论证"""
        data = {
            'analyst_reports': analyst_reports,
            'hmm_prediction': hmm_prediction
        }
        return self.analyze(data)

    def rebut(self, bear_report: str) -> str:
        """反驳空头观点"""
        user_prompt = f"空头观点:\n{bear_report}\n\n请针对上述空头观点进行反驳，强化看涨论证。"
        return self.llm.chat(self.system_prompt, user_prompt)


class BearResearcher(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("空头研究员", prompts.BEAR_RESEARCHER_PROMPT, llm)

    def argue(self, analyst_reports: Dict, hmm_prediction: Dict, bull_report: str = None) -> str:
        """生成看跌论证"""
        data = {
            'analyst_reports': analyst_reports,
            'hmm_prediction': hmm_prediction,
            'bull_report': bull_report or 'N/A'
        }
        return self.analyze(data)

    def rebut(self, bull_report: str) -> str:
        """反驳多头观点"""
        user_prompt = f"多头观点:\n{bull_report}\n\n请针对上述多头观点进行反驳，强化看跌论证。"
        return self.llm.chat(self.system_prompt, user_prompt)


class NeutralJudge(AnalystAgent):
    def __init__(self, llm: LLMClient):
        super().__init__("中立观察员", prompts.JUDGE_PROMPT, llm)

    def vote(self, analyst_reports: Dict, bull_report: str, bear_report: str,
             hmm_prediction: Dict, sentiment_data: Dict = None) -> Dict:
        """综合投票决策"""
        data = {
            'analyst_reports': analyst_reports,
            'bull_report': bull_report,
            'bear_report': bear_report,
            'hmm_prediction': hmm_prediction,
            'sentiment_data': sentiment_data or 'N/A'
        }

        response = self.analyze(data)

        return {
            'vote': self._extract_vote(response),
            'position': self._extract_position(response),
            'reasoning': response,
            'bull_score': self._extract_score(response, '多头'),
            'bear_score': self._extract_score(response, '空头')
        }

    def _extract_vote(self, text: str) -> str:
        """提取投票结果"""
        if '买入' in text:
            return '买入'
        elif '卖出' in text:
            return '卖出'
        return '持有'

    def _extract_position(self, text: str) -> str:
        """提取建议仓位"""
        for pos in ['满仓', '半仓', '轻仓', '空仓']:
            if pos in text:
                return pos
        return '观望'

    def _extract_score(self, text: str, side: str) -> int:
        """提取得分"""
        import re
        patterns = [
            rf'{side}得分[:：]\s*(\d+)',
            rf'{side}[^\d]*(\d+)\s*分',
            rf'{side}[^\d]*(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                score = int(m.group(1))
                return max(1, min(10, score))
        return 5


class DebateEngine:
    """
    AI辩论引擎 - Layer 3 决策分析

    流程:
    Phase 1: 分析师并行生成报告
    Phase 2: 研究员辩论
    Phase 3: 裁判投票
    """

    def __init__(self, llm_client: LLMClient = None, max_debate_rounds: int = 2):
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient()
        self.max_debate_rounds = max_debate_rounds

        # 初始化角色
        self.analysts = {
            'technical': TechnicalAnalyst(self.llm),
            'fundamental': FundamentalAnalyst(self.llm),
            'macro': MacroAnalyst(self.llm)
        }
        self.researchers = {
            'bull': BullResearcher(self.llm),
            'bear': BearResearcher(self.llm)
        }
        self.judge = NeutralJudge(self.llm)

    def debate(self, symbol: str, trend_qualification: Dict,
               signal_report: Dict) -> Dict:
        """
        执行完整辩论流程

        Args:
            symbol: 股票代码
            trend_qualification: Layer 1趋势定性报告
            signal_report: Layer 2信号报告

        Returns:
            dict: 辩论结果
        """
        logger.info(f"开始对 {symbol} 进行AI辩论")

        # 准备数据
        stock_data = {
            'symbol': symbol,
            'trend_qualification': trend_qualification,
            'signal_report': signal_report
        }

        # Phase 1: 分析师并行生成报告
        logger.info("Phase 1: 分析师报告生成")
        analyst_reports = {}
        for name, analyst in self.analysts.items():
            logger.info(f"  {analyst.name} 分析中...")
            analyst_reports[name] = analyst.analyze(stock_data)

        # Phase 2: 研究员辩论
        logger.info("Phase 2: 研究员辩论")
        bull_report = self.researchers['bull'].argue(analyst_reports, trend_qualification)
        bear_report = self.researchers['bear'].argue(analyst_reports, trend_qualification, bull_report)

        # 多轮辩论
        for round_num in range(self.max_debate_rounds):
            logger.info(f"  辩论轮次 {round_num + 1}")
            bull_report = self.researchers['bull'].rebut(bear_report)
            bear_report = self.researchers['bear'].rebut(bull_report)

        # Phase 3: 中立观察员投票
        logger.info("Phase 3: 最终投票")
        final_decision = self.judge.vote(
            analyst_reports=analyst_reports,
            bull_report=bull_report,
            bear_report=bear_report,
            hmm_prediction=trend_qualification
        )

        logger.info(f"辩论完成，最终投票: {final_decision['vote']}")

        return {
            'analyst_reports': analyst_reports,
            'bull_report': bull_report,
            'bear_report': bear_report,
            'final_decision': final_decision
        }


class ReviewAnalyzer:
    """
    Review阶段 - 次日复核分析

    对昨日的预测进行复核，输出优化建议
    """

    def __init__(self, llm_client: LLMClient = None):
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient()

    def review(self, symbol: str, prediction: Dict, actual: Dict) -> Dict:
        """
        执行复核分析

        Args:
            symbol: 股票代码
            prediction: 昨日预测报告
            actual: 实际走势数据

        Returns:
            dict: 复核报告
        """
        logger.info(f"对 {symbol} 进行Review复核")

        # 构建复核数据
        review_data = {
            'symbol': symbol,
            'prediction': prediction,
            'actual': actual
        }

        # 调用LLM进行复核分析
        system_prompt = prompts.REVIEW_ANALYZER_PROMPT
        user_prompt = self._format_review_data(review_data)

        analysis = self.llm.chat(system_prompt, user_prompt)

        # 解析复核结果
        return {
            'symbol': symbol,
            'review_date': datetime.now(),
            'prediction_accuracy': self._calculate_accuracy(prediction, actual),
            'deviation_analysis': self._analyze_deviation(prediction, actual),
            'layer_performance': self._evaluate_layers(prediction, actual),
            'optimization_suggestions': analysis,
            'confidence_adjustment': self._suggest_adjustment(prediction, actual)
        }

    def _format_review_data(self, data: Dict) -> str:
        """格式化复核数据"""
        pred = data['prediction']
        actual = data['actual']

        return f"""
股票: {data['symbol']}

昨日预测:
- 方向: {pred.get('direction', 'N/A')}
- 概率: {pred.get('probability', 'N/A')}%
- 目标价: {pred.get('target_price', 'N/A')}
- HMM趋势: {pred.get('trend_qualification', {}).get('trend_direction', 'N/A')}
- 因子信号: {pred.get('factor_signals', 'N/A')}
- AI投票: {pred.get('ai_vote', 'N/A')}

实际走势:
- 实际方向: {actual.get('direction', 'N/A')}
- 实际收盘价: {actual.get('close_price', 'N/A')}
- 实际涨跌幅: {actual.get('change_pct', 'N/A')}%

请进行复核分析。
"""

    def _calculate_accuracy(self, prediction: Dict, actual: Dict) -> Dict:
        """计算预测准确率"""
        direction_correct = prediction.get('direction') == actual.get('direction')

        # 计算幅度偏差
        pred_change = prediction.get('target_price', 0) - prediction.get('current_price', 0)
        actual_change = actual.get('close_price', 0) - prediction.get('current_price', 0)

        if pred_change != 0:
            magnitude_error = abs(actual_change - pred_change) / abs(pred_change)
        else:
            magnitude_error = 0

        return {
            'direction_correct': direction_correct,
            'magnitude_error': magnitude_error,
            'overall_score': 1.0 if direction_correct else 0.0
        }

    def _analyze_deviation(self, prediction: Dict, actual: Dict) -> str:
        """分析偏差原因"""
        direction_correct = prediction.get('direction') == actual.get('direction')
        magnitude_error = self._calculate_accuracy(prediction, actual).get('magnitude_error', 0)
        pred_dir = prediction.get('direction', '未知')
        act_dir = actual.get('direction', '未知')
        
        if direction_correct and magnitude_error < 0.05:
            return f"方向判断正确（预测{pred_dir}，实际{act_dir}），幅度预测较为准确，偏差{magnitude_error:.1%}在可接受范围内"
        elif direction_correct and magnitude_error >= 0.05:
            return f"方向判断正确（预测{pred_dir}，实际{act_dir}），但幅度预测偏差较大（偏差{magnitude_error:.1%}），可能原因：波动率估计不足或市场突发事件"
        elif pred_dir == '涨' and act_dir == '跌':
            return f"方向判断失误（预测涨但实际跌），可能原因：突发利空消息/趋势反转信号未及时捕捉/支撑位破位"
        else:
            return f"方向判断失误（预测跌但实际涨），可能原因：突发利好消息/超跌反弹/资金面突然改善"

    def _evaluate_layers(self, prediction: Dict, actual: Dict) -> Dict:
        """评估各Layer表现"""
        return {
            'layer1_hmm': 'accurate' if prediction.get('trend_qualification', {}).get('trend_direction') == actual.get('direction') else 'inaccurate',
            'layer2_factor': 'accurate' if prediction.get('direction') == actual.get('direction') else 'inaccurate',
            'layer3_ai': 'accurate' if prediction.get('ai_vote') == actual.get('direction') else 'inaccurate'
        }

    def _suggest_adjustment(self, prediction: Dict, actual: Dict) -> str:
        """建议置信度调整"""
        direction_correct = prediction.get('direction') == actual.get('direction')
        magnitude_error = self._calculate_accuracy(prediction, actual).get('magnitude_error', 0)
        
        if direction_correct and magnitude_error < 0.10:
            return "维持当前置信度，模型表现良好，方向和幅度预测均在合理范围内"
        elif direction_correct and magnitude_error >= 0.10:
            return f"建议将置信度下调5-10%，方向判断正确但幅度偏差较大（{magnitude_error:.1%}），需校准波动率参数"
        else:
            return "建议将置信度下调15-20%，方向判断失误，需重新评估趋势判断逻辑和因子权重配置"
