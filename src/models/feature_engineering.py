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
    def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强技术指标特征（无未来函数）

        包含: MACD, RSI, 布林带位置, ATR, 换手率,
              均线斜率, 价格动量, 量比偏离
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # MACD特征
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        features['macd_hist'] = macd_hist
        features['macd_hist_change'] = macd_hist.diff()
        features['macd_above_signal'] = (macd > macd_signal).astype(int)

        # RSI (多周期)
        for period in [6, 14, 24]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 布林带位置 (%B)
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            features[f'boll_pctb_{period}'] = (close - lower) / (upper - lower + 1e-10)
            features[f'boll_width_{period}'] = (upper - lower) / ma

        # ATR (平均真实波幅)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_ratio'] = tr / features['atr_14']

        # 均线斜率 (趋势强度)
        for period in [5, 10, 20]:
            ma = close.rolling(period).mean()
            features[f'ma{period}_slope'] = (ma - ma.shift(3)) / (ma.shift(3) + 1e-10)

        # 价格动量 (多周期)
        for period in [5, 10, 20, 60]:
            features[f'momentum_{period}'] = close.pct_change(period)

        # 成交量特征增强
        features['volume_ratio_5'] = volume / volume.rolling(5).mean()
        features['volume_ratio_20'] = volume / volume.rolling(20).mean()
        features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # 换手率（如果有turn列）
        if 'turn' in df.columns:
            features['turnover'] = df['turn']
            features['turnover_ma5'] = df['turn'].rolling(5).mean()
            features['turnover_change'] = df['turn'].pct_change()

        # 波动率特征
        features['realized_vol_5'] = close.pct_change().rolling(5).std()
        features['realized_vol_20'] = close.pct_change().rolling(20).std()
        features['vol_ratio'] = features['realized_vol_5'] / (features['realized_vol_20'] + 1e-10)

        # 价格位置特征
        high_60 = high.rolling(60).max()
        low_60 = low.rolling(60).min()
        features['price_position_60'] = (close - low_60) / (high_60 - low_60 + 1e-10)

        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        features['price_position_20'] = (close - low_20) / (high_20 - low_20 + 1e-10)

        return features

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

    @staticmethod
    def compute_enhanced_labels(df: pd.DataFrame, forward_days: int = 3) -> pd.Series:
        """
        计算增强标签 - 使用N日未来收益率

        涨: N日后均价 > 今日均价
        跌: N日后均价 <= 今日均价

        Args:
            df: DataFrame
            forward_days: 前瞻天数

        Returns:
            Series: 标签 (1=涨, 0=跌)
        """
        avg_price = (df['high'] + df['low'] + df['close']) / 3
        future_avg = avg_price.shift(-forward_days).rolling(forward_days).mean().shift(-(forward_days - 1))
        labels = (future_avg > avg_price).astype(int)
        return labels

    @classmethod
    def prepare_features_basic(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """原始特征（保持向后兼容）"""
        daily = cls.compute_daily_features(df)
        window = cls.compute_window_features(df)
        labels = cls.compute_labels(df)
        features = pd.concat([daily, window], axis=1).dropna()
        labels = labels.loc[features.index]
        return features, labels

    @classmethod
    def prepare_features(cls, df: pd.DataFrame, enhanced: bool = False, forward_days: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备完整特征和标签

        Args:
            df: DataFrame
            enhanced: 是否使用增强特征（默认False，避免未来函数风险）
            forward_days: 标签前瞻天数（默认3日）

        Returns:
            Tuple[DataFrame, Series]: (特征矩阵, 标签)
        """
        daily = cls.compute_daily_features(df)
        window = cls.compute_window_features(df)

        if enhanced:
            enh = cls.compute_enhanced_features(df)
            features = pd.concat([daily, window, enh], axis=1).dropna()
            labels = cls.compute_enhanced_labels(df, forward_days)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("使用增强标签（含未来信息），仅用于训练，不可用于回测信号生成")
        else:
            features = pd.concat([daily, window], axis=1).dropna()
            labels = cls.compute_labels(df)

        labels = labels.loc[features.index]
        # 去除NaN标签
        valid_mask = labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]

        return features, labels
