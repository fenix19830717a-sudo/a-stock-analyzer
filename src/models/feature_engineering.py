"""特征工程模块 - 无未来函数"""
import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """
    特征工程 - 计算HMM所需特征
    
    核心原则:
    - 仅使用当日及历史数据
    - 不使用任何未来数据
    - 包含OHLC基础价格数据
    """
    
    @staticmethod
    def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每日基础特征 - 仅使用当日及历史数据
        
        Args:
            df: DataFrame，包含open/high/low/close/volume
        
        Returns:
            DataFrame: 特征矩阵
        """
        features = pd.DataFrame(index=df.index)
        
        # === 基础价格特征（直接使用OHLC数据）===
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # === 价格特征 ===
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['range'] = (df['high'] - df['low']) / df['close']
        features['body'] = (df['close'] - df['open']) / df['open']
        features['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # === 市场认知特征（无未来函数）===
        features['market_perception'] = df['close'] - df['open']
        features['overnight_gap'] = df['open'] - df['close'].shift(1)
        
        # === K线形态特征 ===
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # === 波动特征 ===
        features['price_variance'] = df[['high', 'low']].var(axis=1)
        
        # === 成交量特征 ===
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma5'] = df['volume'].rolling(5).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma5']
        features['amount'] = df['volume'] * df['close']
        
        return features.dropna()
    
    @staticmethod
    def compute_window_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        计算窗口特征
        
        Args:
            df: DataFrame
            window: 窗口大小，默认7日
        
        Returns:
            DataFrame: 窗口特征
        """
        features = pd.DataFrame(index=df.index)
        
        # 量价关系特征
        features['price_up_volume_shrink'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['volume'] < df['volume'].shift(1))
        ).astype(int).rolling(window).sum()
        
        features['price_up_volume_expand'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        ).astype(int).rolling(window).sum()
        
        features['price_down_volume_shrink'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['volume'] < df['volume'].shift(1))
        ).astype(int).rolling(window).sum()
        
        features['price_down_volume_expand'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        ).astype(int).rolling(window).sum()
        
        # 均线系统
        features['ma5'] = df['close'].rolling(5).mean()
        features['ma10'] = df['close'].rolling(10).mean()
        features['ma20'] = df['close'].rolling(20).mean()
        features['ma_bias'] = (df['close'] - features['ma5']) / features['ma5']
        
        # 波动率
        features['volatility_7d'] = df['close'].pct_change().rolling(window).std()
        
        # 价格位置
        features['price_position'] = (
            (df['close'] - df['low'].rolling(60).min()) /
            (df['high'].rolling(60).max() - df['low'].rolling(60).min())
        )
        
        return features.dropna()
    
    @staticmethod
    def compute_labels(df: pd.DataFrame) -> pd.Series:
        """
        计算涨跌标签
        
        涨: 今日均价 > 昨日均价
        跌: 今日均价 <= 昨日均价
        
        Args:
            df: DataFrame
        
        Returns:
            Series: 标签 (1=涨, 0=跌)
        """
        avg_price = (df['high'] + df['low'] + df['close']) / 3
        labels = (avg_price > avg_price.shift(1)).astype(int)
        return labels
    
    @classmethod
    def prepare_features(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备完整特征和标签
        
        Args:
            df: DataFrame
        
        Returns:
            Tuple[DataFrame, Series]: (特征矩阵, 标签)
        """
        daily = cls.compute_daily_features(df)
        window = cls.compute_window_features(df)
        labels = cls.compute_labels(df)
        
        features = pd.concat([daily, window], axis=1).dropna()
        labels = labels.loc[features.index]
        
        return features, labels
