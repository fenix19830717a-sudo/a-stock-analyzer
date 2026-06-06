"""HMM模型引擎 - 监督式聚类"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Dict, List, Tuple, Optional
from src.utils.logger import get_logger
from src.models.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class SupervisedHMM:
    """
    监督式HMM - 通过有标签数据校准隐藏状态语义
    
    核心思想:
    1. HMM进行无监督聚类（n_components >= 3）
    2. 使用历史标签数据校准每个隐藏状态的涨跌倾向
    3. 输出聚类权重（各状态的后验概率）
    4. 聚类权重映射为涨跌概率
    """
    
    def __init__(self, name: str, n_components: int = 5):
        self.name = name
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type='diag',
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False
        self.state_up_prob = {}  # 每个状态的上涨概率
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        训练HMM
        
        Args:
            X: 特征矩阵
            y: 标签（0=跌, 1=涨），用于监督式校准
        """
        logger.info(f"训练{self.name}, 样本数: {len(X)}")
        self.model.fit(X)
        
        if y is not None:
            # 监督式校准：计算每个隐藏状态的上涨概率
            hidden_states = self.model.predict(X)
            for state in range(self.n_components):
                mask = hidden_states == state
                if mask.sum() > 0:
                    self.state_up_prob[state] = y[mask].mean()
                else:
                    self.state_up_prob[state] = 0.5
            
            logger.info(f"{self.name}状态校准完成: {self.state_up_prob}")
        
        self.is_fitted = True
        return self
    
    def get_cluster_weights(self, X: np.ndarray) -> np.ndarray:
        """
        获取聚类权重（各隐藏状态的后验概率）
        
        Args:
            X: 特征矩阵
        
        Returns:
            ndarray: shape=(n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("模型未训练")
        _, posteriors = self.model.predict_proba(X)
        return posteriors
    
    def predict_up_probability(self, X: np.ndarray) -> np.ndarray:
        """
        预测上涨概率
        
        通过聚类权重加权各状态的上涨概率
        
        Args:
            X: 特征矩阵
        
        Returns:
            ndarray: 上涨概率
        """
        posteriors = self.get_cluster_weights(X)
        up_prob = np.zeros(len(X))
        for state, prob in self.state_up_prob.items():
            up_prob += posteriors[:, state] * prob
        return up_prob
    
    def predict_direction(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        预测涨跌方向
        
        Args:
            X: 特征矩阵
            threshold: 阈值
        
        Returns:
            ndarray: 0=跌, 1=涨
        """
        up_prob = self.predict_up_probability(X)
        return (up_prob >= threshold).astype(int)


class MarketHMM(SupervisedHMM):
    """大盘HMM - 第一层"""
    
    def __init__(self, n_components: int = 5):
        super().__init__("MarketHMM", n_components)


class SectorHMM(SupervisedHMM):
    """行业HMM - 第二层"""
    
    def __init__(self, sector_code: str, n_components: int = 5):
        super().__init__(f"SectorHMM_{sector_code}", n_components)
        self.sector_code = sector_code
    
    def prepare_features(self, sector_data: np.ndarray, market_weights: np.ndarray) -> np.ndarray:
        """
        准备特征，融入大盘聚类权重
        
        Args:
            sector_data: 行业特征矩阵
            market_weights: 大盘HMM的聚类权重
        
        Returns:
            ndarray: 融合后的特征
        """
        # 对齐长度
        min_len = min(len(sector_data), len(market_weights))
        sector_data = sector_data[-min_len:]
        market_weights = market_weights[-min_len:]
        
        features = np.hstack([sector_data, market_weights])
        return features


class StockHMM(SupervisedHMM):
    """个股HMM - 第三层"""
    
    def __init__(self, symbol: str, n_components: int = 5):
        super().__init__(f"StockHMM_{symbol}", n_components)
        self.symbol = symbol
    
    def prepare_features(self, stock_data: np.ndarray,
                        market_weights: np.ndarray,
                        sector_weights: np.ndarray) -> np.ndarray:
        """
        准备个股特征，融入大盘和行业聚类权重
        
        Args:
            stock_data: 个股特征矩阵
            market_weights: 大盘聚类权重
            sector_weights: 行业聚类权重
        
        Returns:
            ndarray: 融合后的特征
        """
        # 对齐长度
        min_len = min(len(stock_data), len(market_weights), len(sector_weights))
        stock_data = stock_data[-min_len:]
        market_weights = market_weights[-min_len:]
        sector_weights = sector_weights[-min_len:]
        
        features = np.hstack([stock_data, market_weights, sector_weights])
        return features


class HMMTrendAnalyzer:
    """
    HMM趋势定性分析器
    
    输出趋势定性报告，用于Layer 1分析
    """
    
    def __init__(self, market_hmm: MarketHMM, sector_hmms: Dict, stock_hmms: Dict):
        self.market_hmm = market_hmm
        self.sector_hmms = sector_hmms
        self.stock_hmms = stock_hmms
        self.feature_engineer = FeatureEngineer()
    
    def analyze_trend(self, symbol: str, data: pd.DataFrame) -> dict:
        """
        对个股进行趋势定性分析
        
        Args:
            symbol: 股票代码
            data: DataFrame
        
        Returns:
            dict: 趋势定性报告
        """
        # 1. 获取个股HMM聚类权重
        features, labels = self.feature_engineer.prepare_features(data)
        
        if symbol not in self.stock_hmms:
            return {'error': f'{symbol}模型未训练'}
        
        stock_weights = self.stock_hmms[symbol].get_cluster_weights(features.values)
        
        # 2. 分析主导状态
        latest_weights = stock_weights[-1]
        dominant_state = np.argmax(latest_weights)
        
        # 3. 判断趋势方向
        up_prob = self.stock_hmms[symbol].predict_up_probability(features.values)[-1]
        
        if up_prob > 0.65:
            trend_direction = "上升"
        elif up_prob < 0.35:
            trend_direction = "下降"
        else:
            trend_direction = "震荡"
        
        # 4. 判断趋势强度
        max_prob = np.max(latest_weights)
        if max_prob > 0.5:
            trend_strength = "强"
        elif max_prob > 0.35:
            trend_strength = "中"
        else:
            trend_strength = "弱"
        
        # 5. 判断趋势阶段
        trend_stage = self._judge_stage(data, trend_direction)
        
        # 6. 计算置信度
        confidence = min(max_prob * 100 + 10, 100)
        
        return {
            'symbol': symbol,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'trend_stage': trend_stage,
            'confidence': confidence,
            'cluster_weights': latest_weights.tolist(),
            'dominant_state': f"状态{dominant_state}",
            'up_probability': up_prob
        }
    
    def _judge_stage(self, data: pd.DataFrame, trend_direction: str) -> str:
        """判断趋势阶段"""
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        ma60 = data['close'].rolling(60).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if trend_direction == "上升":
            if current_price < ma5 < ma20:
                return "初期"
            elif ma5 < current_price < ma20:
                return "中期"
            else:
                return "末期"
        elif trend_direction == "下降":
            if current_price > ma5 > ma20:
                return "初期"
            elif ma5 > current_price > ma20:
                return "中期"
            else:
                return "末期"
        else:
            return "不明"


class HMMTrainingPipeline:
    """
    HMM三级监督式训练流水线
    
    训练流程:
    1. 训练大盘HMM（使用大盘标签校准）
    2. 使用大盘聚类权重训练各行业HMM
    3. 使用大盘+行业聚类权重训练个股HMM
    """
    
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.market_hmm = MarketHMM(n_components)
        self.sector_hmms = {}
        self.stock_hmms = {}
        self.feature_engineer = FeatureEngineer()
    
    def train(self, market_data: pd.DataFrame,
              sector_data_dict: Dict[str, pd.DataFrame],
              stock_data_dict: Dict[str, pd.DataFrame]) -> 'HMMTrainingPipeline':
        """
        训练三级HMM
        
        Args:
            market_data: 大盘数据
            sector_data_dict: {sector_code: 行业数据}
            stock_data_dict: {symbol: 个股数据}
        
        Returns:
            self
        """
        # Step 1: 训练大盘HMM
        logger.info("Step 1: 训练大盘HMM")
        market_features, market_labels = self.feature_engineer.prepare_features(market_data)
        self.market_hmm.fit(market_features.values, market_labels.values)
        market_weights = self.market_hmm.get_cluster_weights(market_features.values)
        
        # Step 2: 训练行业HMM
        logger.info("Step 2: 训练行业HMM")
        for sector_code, sector_data in sector_data_dict.items():
            sector_features, sector_labels = self.feature_engineer.prepare_features(sector_data)
            
            # 融入大盘聚类权重
            aligned_weights = market_weights[-len(sector_features):]
            combined_features = SectorHMM(sector_code).prepare_features(
                sector_features.values, aligned_weights
            )
            
            hmm = SectorHMM(sector_code, self.n_components)
            hmm.fit(combined_features, sector_labels.values)
            self.sector_hmms[sector_code] = hmm
        
        # Step 3: 训练个股HMM
        logger.info("Step 3: 训练个股HMM")
        for symbol, stock_data in stock_data_dict.items():
            # 获取行业代码 (简化实现)
            sector_code = self._get_sector_code(symbol)
            if sector_code not in self.sector_hmms:
                continue
            
            stock_features, stock_labels = self.feature_engineer.prepare_features(stock_data)
            
            # 获取对应的大盘和行业聚类权重
            aligned_market_weights = market_weights[-len(stock_features):]
            sector_weights = self.sector_hmms[sector_code].get_cluster_weights(stock_features.values)
            
            # 融入聚类权重
            combined_features = StockHMM(symbol).prepare_features(
                stock_features.values, aligned_market_weights, sector_weights
            )
            
            hmm = StockHMM(symbol, self.n_components)
            hmm.fit(combined_features, stock_labels.values)
            self.stock_hmms[symbol] = hmm
        
        logger.info("HMM训练完成")
        return self
    
    def _get_sector_code(self, symbol: str) -> str:
        """获取股票所属行业代码 (简化实现)"""
        # 实际应从数据库查询
        return "default_sector"
