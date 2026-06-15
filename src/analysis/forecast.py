"""天气预报式预测报告生成器"""
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DayForecast:
    """单日预测"""
    date: datetime
    weather_icon: str
    trend: str
    probability: float
    price_range: tuple
    key_factors: List[str]


@dataclass
class WeatherForecastReport:
    """天气预报式预测报告"""
    symbol: str
    name: str
    current_price: float
    forecast_date: datetime
    
    day1_forecast: DayForecast
    day2_forecast: DayForecast
    day3_forecast: DayForecast
    
    overall_trend: str
    overall_confidence: float
    
    recommendation: str
    position_suggestion: str
    
    risk_factors: List[str]
    
    layer1_summary: str
    layer2_summary: str
    layer3_summary: str


class ForecastReportGenerator:
    """预测报告生成器"""
    
    def generate(self, symbol: str, name: str, current_price: float,
                 trend_qualification: Dict, signal_report: Dict,
                 debate_result: Dict) -> WeatherForecastReport:
        """
        生成天气预报式预测报告
        
        Args:
            symbol: 股票代码
            name: 股票名称
            current_price: 当前价格
            trend_qualification: Layer 1趋势定性
            signal_report: Layer 2信号报告
            debate_result: Layer 3辩论结果
        
        Returns:
            WeatherForecastReport: 预测报告
        """
        # 生成未来3天预测
        day1 = self._generate_day_forecast(
            symbol, current_price, trend_qualification, signal_report, 1
        )
        day2 = self._generate_day_forecast(
            symbol, current_price, trend_qualification, signal_report, 2
        )
        day3 = self._generate_day_forecast(
            symbol, current_price, trend_qualification, signal_report, 3
        )
        
        # 综合评估
        overall_trend = self._determine_overall_trend(trend_qualification, signal_report)
        overall_confidence = self._calculate_overall_confidence(
            trend_qualification, signal_report, debate_result
        )
        
        # 决策建议
        recommendation = self._generate_recommendation(debate_result)
        position = self._suggest_position(debate_result, overall_confidence)
        
        # 风险因素
        risks = self._extract_risks(trend_qualification, signal_report)
        
        return WeatherForecastReport(
            symbol=symbol,
            name=name,
            current_price=current_price,
            forecast_date=datetime.now(),
            day1_forecast=day1,
            day2_forecast=day2,
            day3_forecast=day3,
            overall_trend=overall_trend,
            overall_confidence=overall_confidence,
            recommendation=recommendation,
            position_suggestion=position,
            risk_factors=risks,
            layer1_summary=self._summarize_layer1(trend_qualification),
            layer2_summary=self._summarize_layer2(signal_report),
            layer3_summary=self._summarize_layer3(debate_result)
        )
    
    def _generate_day_forecast(self, symbol: str, current_price: float,
                               trend_qualification: Dict, signal_report: Dict,
                               day: int) -> DayForecast:
        """生成单日预测"""
        date = datetime.now() + timedelta(days=day)
        
        # 根据趋势和概率确定天气图标
        direction = signal_report.get('direction', '震荡')
        probability = signal_report.get('probability', 50)
        
        if direction == '涨':
            if probability > 70:
                icon = '☀️'
                trend = '明显上涨'
            else:
                icon = '⛅'
                trend = '小幅上涨'
        elif direction == '跌':
            if probability > 70:
                icon = '🌧️'
                trend = '明显下跌'
            else:
                icon = '☁️'
                trend = '小幅下跌'
        else:
            icon = '🌤️'
            trend = '震荡整理'
        
        # 计算价格区间 (简化实现)
        volatility = 0.02  # 假设2%日波动
        expected_change = (probability / 100 - 0.5) * 2 * volatility * day
        
        center_price = current_price * (1 + expected_change)
        price_low = center_price * (1 - volatility)
        price_high = center_price * (1 + volatility)
        
        # 准确率随天数递减
        day_accuracy = max(probability - (day - 1) * 7, 40)
        
        return DayForecast(
            date=date,
            weather_icon=icon,
            trend=trend,
            probability=day_accuracy,
            price_range=(round(price_low, 2), round(price_high, 2)),
            key_factors=self._get_key_factors(trend_qualification, day)
        )
    
    def _determine_overall_trend(self, trend_qualification: Dict, signal_report: Dict) -> str:
        """确定总体趋势"""
        trend = trend_qualification.get('trend_direction', '震荡')
        direction = signal_report.get('direction', '震荡')
        
        if trend == '上升' and direction == '涨':
            return '☀️ 晴 (上升趋势)'
        elif trend == '下降' and direction == '跌':
            return '🌧️ 雨 (下降趋势)'
        else:
            return '⛅ 多云 (震荡)'
    
    def _calculate_overall_confidence(self, trend_qualification: Dict,
                                     signal_report: Dict, debate_result: Dict) -> float:
        """计算综合准确率 - 从辩论结果中提取真实置信度"""
        hmm_conf = trend_qualification.get('confidence', 50)
        factor_prob = signal_report.get('probability', 50)

        # 从辩论结果中提取Layer 3置信度
        final_decision = debate_result.get('final_decision', {})
        bull_score = final_decision.get('bull_score', 0)
        bear_score = final_decision.get('bear_score', 0)
        if bull_score > 0 or bear_score > 0:
            # 根据多头/空头得分计算置信度
            total_score = bull_score + bear_score
            if total_score > 0:
                debate_conf = (bull_score / total_score) * 100
            else:
                debate_conf = 60
        else:
            # 如果没有得分数据，使用默认值60
            debate_conf = 60

        # 综合三层置信度
        overall = (hmm_conf * 0.3 + factor_prob * 0.4 + debate_conf * 0.3)

        return round(overall, 1)
    
    def _generate_recommendation(self, debate_result: Dict) -> str:
        """生成操作建议"""
        vote = debate_result.get('final_decision', {}).get('vote', '持有')
        return vote
    
    def _suggest_position(self, debate_result: Dict, confidence: float) -> str:
        """建议仓位"""
        vote = debate_result.get('final_decision', {}).get('vote', '持有')
        
        if vote == '买入':
            if confidence > 70:
                return '半仓'
            else:
                return '轻仓'
        elif vote == '卖出':
            return '空仓'
        else:
            return '观望'
    
    def _extract_risks(self, trend_qualification: Dict, signal_report: Dict) -> List[str]:
        """提取风险因素"""
        risks = []
        
        if trend_qualification.get('trend_strength') == '弱':
            risks.append('趋势强度较弱，可能反转')
        
        if signal_report.get('factor_breakdown', {}).get('trend_tracking', {}).get('details', {}).get('bollinger') == 'bearish':
            risks.append('布林带突破上轨，短期可能回调')
        
        return risks
    
    def _get_key_factors(self, trend_qualification: Dict, day: int) -> List[str]:
        """获取关键因子 - 从trend_qualification中提取真实因子信息"""
        factors = []

        # 从趋势定性中提取趋势方向和强度
        trend_direction = trend_qualification.get('trend_direction', '')
        trend_strength = trend_qualification.get('trend_strength', '')

        if trend_direction:
            factors.append(f"趋势方向: {trend_direction}")
        if trend_strength:
            factors.append(f"趋势强度: {trend_strength}")

        # 从趋势定性的regime信息中提取
        regime = trend_qualification.get('regime', '')
        if regime:
            factors.append(f"市场状态: {regime}")

        # 从置信度提取
        confidence = trend_qualification.get('confidence', 0)
        if confidence:
            factors.append(f"HMM置信度: {confidence}%")

        # 根据天数添加时间相关因子
        if day == 1:
            factors.append("短期: 均线支撑/压力位")
        elif day == 2:
            factors.append("中期: 成交量变化/行业轮动")
        else:
            factors.append("长期: 趋势延续/反转信号")

        # 如果没有提取到任何因子，返回默认值
        if not factors:
            factors.append('趋势延续')

        return factors
    
    def _summarize_layer1(self, trend_qualification: Dict) -> str:
        """总结Layer 1"""
        return f"趋势: {trend_qualification.get('trend_direction')} | 强度: {trend_qualification.get('trend_strength')} | 置信度: {trend_qualification.get('confidence')}%"
    
    def _summarize_layer2(self, signal_report: Dict) -> str:
        """总结Layer 2"""
        return f"方向: {signal_report.get('direction')} | 概率: {signal_report.get('probability')}% | 目标价: {signal_report.get('target_price')}"
    
    def _summarize_layer3(self, debate_result: Dict) -> str:
        """总结Layer 3"""
        vote = debate_result.get('final_decision', {}).get('vote', 'N/A')
        return f"投票: {vote}"
    
    def to_string(self, report: WeatherForecastReport) -> str:
        """转换为字符串格式"""
        return f"""
═══════════════════════════════════════════════════════════════════
🌤️ A股天气预报 - {report.name} ({report.symbol})
═══════════════════════════════════════════════════════════════════

📍 当前状况
   股价: {report.current_price}元
   日期: {report.forecast_date.strftime('%Y-%m-%d')}
   天气: {report.overall_trend}

┌─────────────────────────────────────────────────────────────┐
│ 📅 未来3天走势预测                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  {report.day1_forecast.date.strftime('%m-%d')}  {report.day1_forecast.weather_icon}  {report.day1_forecast.trend}
│  概率: {report.day1_forecast.probability}%  区间: {report.day1_forecast.price_range[0]}~{report.day1_forecast.price_range[1]}元
│                                                             │
│  {report.day2_forecast.date.strftime('%m-%d')}  {report.day2_forecast.weather_icon}  {report.day2_forecast.trend}
│  概率: {report.day2_forecast.probability}%  区间: {report.day2_forecast.price_range[0]}~{report.day2_forecast.price_range[1]}元
│                                                             │
│  {report.day3_forecast.date.strftime('%m-%d')}  {report.day3_forecast.weather_icon}  {report.day3_forecast.trend}
│  概率: {report.day3_forecast.probability}%  区间: {report.day3_forecast.price_range[0]}~{report.day3_forecast.price_range[1]}元
│                                                             │
└─────────────────────────────────────────────────────────────┘

🎯 综合评估
   总体趋势: {report.overall_trend}
   综合准确率: {report.overall_confidence}%

💡 决策建议
   操作建议: {report.recommendation}
   建议仓位: {report.position_suggestion}

⚠️ 风险提示
   {chr(10).join('   - ' + r for r in report.risk_factors)}

📋 三层分析摘要
   ┌────────────────────────────────────────┐
   │ Layer 1 (HMM定性)                      │
   │   {report.layer1_summary}              │
   ├────────────────────────────────────────┤
   │ Layer 2 (因子信号)                     │
   │   {report.layer2_summary}              │
   ├────────────────────────────────────────┤
   │ Layer 3 (AI辩论)                       │
   │   {report.layer3_summary}              │
   └────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
"""
