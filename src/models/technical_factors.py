"""技术面因子策略模块 - Layer 2 纯程序化"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from collections import Counter


class TrendTrackingFactor:
    """经典趋势跟踪因子 (权重30%)"""
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        分析经典技术指标
        
        Args:
            data: DataFrame
        
        Returns:
            dict: 信号结果
        """
        signals = {}
        
        # 均线系统
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma10 = data['close'].rolling(10).mean().iloc[-1]
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        current = data['close'].iloc[-1]
        
        signals['ma_alignment'] = self._check_ma_alignment(current, ma5, ma10, ma20)
        
        # MACD
        signals['macd'] = self._calculate_macd(data)
        
        # 布林带
        signals['bollinger'] = self._check_bollinger(data)
        
        # 综合判断
        bullish_count = sum(1 for s in signals.values() if s == 'bullish')
        bearish_count = sum(1 for s in signals.values() if s == 'bearish')
        
        if bullish_count > bearish_count:
            return {
                'signal': 'buy',
                'strength': bullish_count / len(signals),
                'details': signals
            }
        elif bearish_count > bullish_count:
            return {
                'signal': 'sell',
                'strength': bearish_count / len(signals),
                'details': signals
            }
        else:
            return {
                'signal': 'neutral',
                'strength': 0.5,
                'details': signals
            }
    
    def _check_ma_alignment(self, current, ma5, ma10, ma20) -> str:
        """检查均线排列"""
        if current > ma5 > ma10 > ma20:
            return 'bullish'
        elif current < ma5 < ma10 < ma20:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_macd(self, data: pd.DataFrame) -> str:
        """计算MACD信号"""
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return 'bullish'
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return 'bearish'
        elif macd.iloc[-1] > 0:
            return 'bullish'
        else:
            return 'bearish'
    
    def _check_bollinger(self, data: pd.DataFrame) -> str:
        """检查布林带"""
        ma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        current = data['close'].iloc[-1]
        
        if current > upper.iloc[-1]:
            return 'bearish'  # 突破上轨，可能回调
        elif current < lower.iloc[-1]:
            return 'bullish'  # 突破下轨，可能反弹
        else:
            return 'neutral'


class MultiFactorScorer:
    """多因子打分器 (权重40%)"""
    
    def __init__(self):
        self.factor_weights = {
            'momentum': 0.25,
            'volatility': 0.20,
            'volume': 0.20,
            'valuation': 0.15,
            'reversal': 0.20
        }
    
    def score(self, data: pd.DataFrame) -> Dict:
        """
        多因子综合打分
        
        Args:
            data: DataFrame
        
        Returns:
            dict: 打分结果
        """
        scores = {}
        
        # 动量因子 (过去20日收益率)
        returns_20d = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)
        scores['momentum'] = np.clip(returns_20d / 0.2, -1, 1)
        
        # 波动率因子 (波动率越低越好)
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        scores['volatility'] = 1 - np.clip(volatility / 0.05, 0, 1)
        
        # 成交量因子 (放量上涨好)
        volume_ma = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        price_change = data['close'].pct_change().iloc[-1]
        
        if price_change > 0 and current_volume > volume_ma:
            scores['volume'] = min((current_volume / volume_ma - 1) * 2, 1)
        else:
            scores['volume'] = -0.3
        
        # 估值因子 (价格相对位置)
        price_position = (
            (data['close'].iloc[-1] - data['low'].rolling(60).min().iloc[-1]) /
            (data['high'].rolling(60).max().iloc[-1] - data['low'].rolling(60).min().iloc[-1])
        )
        scores['valuation'] = 1 - 2 * price_position
        
        # 反转因子 (RSI)
        rsi = self._calculate_rsi(data['close'], 14)
        if rsi > 70:
            scores['reversal'] = -1
        elif rsi < 30:
            scores['reversal'] = 1
        else:
            scores['reversal'] = 0
        
        # 加权综合
        total_score = sum(scores[f] * self.factor_weights[f] for f in scores)
        
        if total_score > 0.2:
            signal = 'buy'
        elif total_score < -0.2:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        return {
            'total_score': total_score,
            'factor_scores': scores,
            'signal': signal,
            'strength': abs(total_score)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]


class VolumePricePatternFactor:
    """量价关系模式因子 (权重30%)"""
    
    def detect_pattern(self, data: pd.DataFrame) -> Dict:
        """
        检测7日量价关系模式
        
        Args:
            data: DataFrame
        
        Returns:
            dict: 模式检测结果
        """
        recent = data.tail(7)
        
        patterns = []
        for i in range(1, len(recent)):
            price_up = recent['close'].iloc[i] > recent['close'].iloc[i-1]
            volume_up = recent['volume'].iloc[i] > recent['volume'].iloc[i-1]
            
            if price_up and volume_up:
                patterns.append('价涨量增')
            elif price_up and not volume_up:
                patterns.append('价涨量缩')
            elif not price_up and volume_up:
                patterns.append('价跌量增')
            else:
                patterns.append('价跌量缩')
        
        # 统计模式频率
        pattern_counts = Counter(patterns)
        
        # 判断主导模式
        if pattern_counts:
            dominant_pattern = pattern_counts.most_common(1)[0][0]
            dominant_ratio = pattern_counts[dominant_pattern] / len(patterns)
        else:
            dominant_pattern = '无数据'
            dominant_ratio = 0
        
        # 模式信号映射
        signal_map = {
            '价涨量增': ('buy', 0.8),
            '价涨量缩': ('neutral', 0.4),
            '价跌量增': ('sell', 0.7),
            '价跌量缩': ('neutral', 0.3)
        }
        
        signal, strength = signal_map.get(dominant_pattern, ('neutral', 0.5))
        
        return {
            'dominant_pattern': dominant_pattern,
            'pattern_ratio': dominant_ratio,
            'pattern_counts': dict(pattern_counts),
            'signal': signal,
            'strength': strength * dominant_ratio
        }


class TechnicalFactorStrategy:
    """
    技术面因子策略引擎 - Layer 2
    
    纯程序化，无需AI参与
    """
    
    def __init__(self):
        self.factors = {
            'trend_tracking': TrendTrackingFactor(),
            'multi_factor': MultiFactorScorer(),
            'volume_price': VolumePricePatternFactor()
        }
        self.factor_weights = {
            'trend_tracking': 0.3,
            'multi_factor': 0.4,
            'volume_price': 0.3
        }
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, trend_qualification: Dict) -> Dict:
        """
        生成趋势信号
        
        Args:
            symbol: 股票代码
            data: DataFrame
            trend_qualification: Layer 1趋势定性报告
        
        Returns:
            dict: 信号报告
        """
        signals = {}
        
        # 1. 经典趋势跟踪
        signals['trend_tracking'] = self.factors['trend_tracking'].analyze(data)
        
        # 2. 多因子打分
        signals['multi_factor'] = self.factors['multi_factor'].score(data)
        
        # 3. 量价关系模式
        signals['volume_price'] = self.factors['volume_price'].detect_pattern(data)
        
        # 加权综合
        weighted_signal = self._combine_signals(signals, trend_qualification)
        
        # 计算目标价和止损价
        target_price, stop_loss = self._calculate_prices(data, weighted_signal, trend_qualification)
        
        return {
            'symbol': symbol,
            'direction': weighted_signal['direction'],
            'probability': weighted_signal['probability'],
            'target_price': target_price,
            'stop_loss': stop_loss,
            'risk_reward_ratio': abs(target_price - data['close'].iloc[-1]) / abs(stop_loss - data['close'].iloc[-1]) if stop_loss != data['close'].iloc[-1] else 0,
            'factor_breakdown': signals,
            'trend_qualification': trend_qualification
        }
    
    def _combine_signals(self, signals: Dict, trend_qualification: Dict) -> Dict:
        """综合三个子策略的信号"""
        
        trend_signal = signals['trend_tracking']
        factor_signal = signals['multi_factor']
        vp_signal = signals['volume_price']
        
        # 加权计算方向概率
        weights = self.factor_weights
        
        buy_prob = (
            weights['trend_tracking'] * (1 if trend_signal['signal'] == 'buy' else 0) +
            weights['multi_factor'] * (1 if factor_signal['signal'] == 'buy' else 0) +
            weights['volume_price'] * (1 if vp_signal['signal'] == 'buy' else 0)
        )
        
        sell_prob = (
            weights['trend_tracking'] * (1 if trend_signal['signal'] == 'sell' else 0) +
            weights['multi_factor'] * (1 if factor_signal['signal'] == 'sell' else 0) +
            weights['volume_price'] * (1 if vp_signal['signal'] == 'sell' else 0)
        )
        
        # 结合HMM趋势定性调整概率
        if trend_qualification.get('trend_direction') == "上升":
            buy_prob += 0.1
        elif trend_qualification.get('trend_direction') == "下降":
            sell_prob += 0.1
        
        # 归一化
        total = buy_prob + sell_prob
        if total > 0:
            buy_prob /= total
            sell_prob /= total
        
        if buy_prob > sell_prob:
            direction = "涨"
            probability = buy_prob * 100
        else:
            direction = "跌"
            probability = sell_prob * 100
        
        return {
            'direction': direction,
            'probability': probability
        }
    
    def _calculate_prices(self, data: pd.DataFrame, signal: Dict, trend_qualification: Dict) -> Tuple[float, float]:
        """计算目标价和止损价"""
        
        current_price = data['close'].iloc[-1]
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        
        if signal['direction'] == "涨":
            if trend_qualification.get('trend_strength') == "强":
                target = current_price * (1 + 3 * volatility)
            elif trend_qualification.get('trend_strength') == "中":
                target = current_price * (1 + 2 * volatility)
            else:
                target = current_price * (1 + 1.5 * volatility)
            
            stop_loss = max(data['low'].tail(5).min(), current_price * 0.95)
        else:
            target = current_price * (1 - 2 * volatility)
            stop_loss = current_price * 1.05
        
        return round(target, 2), round(stop_loss, 2)
