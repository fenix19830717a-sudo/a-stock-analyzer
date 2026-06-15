"""
大盘预测模型 - 上证指数(sh.000001)多因子预测模型
预测未来1日、2日、3日的涨跌方向和幅度

技术栈: BaoStock + XGBoost + 多因子特征工程
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 依赖检查
# ============================================================
try:
    import baostock as bs
except ImportError:
    print("错误: baostock 未安装，请执行: pip install baostock")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("错误: xgboost 未安装，请执行: pip install xgboost")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report,
        mean_squared_error, mean_absolute_error, r2_score
    )
except ImportError:
    print("错误: scikit-learn 未安装，请执行: pip install scikit-learn")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("错误: joblib 未安装，请执行: pip install joblib")
    sys.exit(1)

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR = r"E:\a-stock-analyzer-main"
MODEL_DIR = os.path.join(PROJECT_DIR, "index_models")
RESULT_FILE = os.path.join(PROJECT_DIR, "index_predict_result.json")

# 确保模型目录存在
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Step 1: 数据获取
# ============================================================
def fetch_index_data():
    """通过BaoStock获取上证指数日K线数据"""
    print("=" * 60)
    print("Step 1: 获取上证指数(sh.000001)日K线数据")
    print("=" * 60)

    # 登录BaoStock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"BaoStock登录失败: {lg.error_msg}")
        sys.exit(1)
    print("BaoStock登录成功")

    start_date = "2018-01-01"
    end_date = "2026-06-05"

    print(f"查询范围: {start_date} ~ {end_date}")

    rs = bs.query_history_k_data_plus(
        "sh.000001",
        "date,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )

    if rs.error_code != '0':
        print(f"查询失败: {rs.error_msg}")
        bs.logout()
        sys.exit(1)

    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)

    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 去除空值
    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.reset_index(drop=True)

    print(f"获取数据量: {len(df)} 条")
    print(f"日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"最新收盘价: {df['close'].iloc[-1]:.2f}")

    bs.logout()
    print("BaoStock已登出")

    return df


# ============================================================
# Step 2: 基础指标计算
# ============================================================
def compute_basic_indicators(df):
    """计算每日基础技术指标"""
    print("\n" + "=" * 60)
    print("Step 2: 计算基础技术指标")
    print("=" * 60)

    # --- 均线系统 ---
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()

    # --- MACD ---
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9).mean()
    df['macd_hist'] = 2 * (df['dif'] - df['dea'])

    # --- RSI (多周期) ---
    for period in [6, 14, 24]:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # --- 布林带 ---
    df['boll_mid'] = df['close'].rolling(20).mean()
    df['boll_std'] = df['close'].rolling(20).std()
    df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_down'] = df['boll_mid'] - 2 * df['boll_std']
    df['boll_pos'] = (df['close'] - df['boll_down']) / (df['boll_up'] - df['boll_down'] + 1e-10)
    df['boll_width'] = df['boll_std'] / (df['boll_mid'] + 1e-10) * 200

    # --- KDJ ---
    low9 = df['low'].rolling(9).min()
    high9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low9) / (high9 - low9 + 1e-10) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # --- 成交量 ---
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma10'] = df['volume'].rolling(10).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1)

    # --- 波动率 ---
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

    # --- 动量 ---
    df['momentum_1d'] = df['close'].pct_change(1)
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)

    # --- ATR ---
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()

    # --- 涨跌幅 ---
    df['pct_change'] = df['close'].pct_change()

    # --- 价格位置 ---
    df['price_pos_20d'] = (df['close'] - df['low'].rolling(20).min()) / (
        df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    df['price_pos_60d'] = (df['close'] - df['low'].rolling(60).min()) / (
        df['high'].rolling(60).max() - df['low'].rolling(60).min() + 1e-10)

    # --- 均线斜率 ---
    df['ma5_slope'] = df['ma5'].pct_change(5)
    df['ma20_slope'] = df['ma20'].pct_change(5)

    # --- OBV ---
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_ma10'] = df['obv'].rolling(10).mean()

    # --- 大盘宽度指标 ---
    df['width_5d'] = (df['close'] - df['low'].rolling(5).min()) / (
        df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-10)
    df['width_20d'] = df['price_pos_20d']

    # --- 连涨连跌天数 ---
    df['up_days'] = 0
    df['down_days'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['pct_change'] > 0:
            df.iloc[i, df.columns.get_loc('up_days')] = df.iloc[i - 1]['up_days'] + 1
            df.iloc[i, df.columns.get_loc('down_days')] = 0
        elif df.iloc[i]['pct_change'] < 0:
            df.iloc[i, df.columns.get_loc('down_days')] = df.iloc[i - 1]['down_days'] + 1
            df.iloc[i, df.columns.get_loc('up_days')] = 0
        else:
            df.iloc[i, df.columns.get_loc('up_days')] = 0
            df.iloc[i, df.columns.get_loc('down_days')] = 0

    print(f"基础指标计算完成，共 {len(df.columns)} 列")
    return df


# ============================================================
# Step 3: 关系特征构建
# ============================================================
def build_relation_features(df):
    """构建因子关系特征"""
    print("\n" + "=" * 60)
    print("Step 3: 构建因子关系特征")
    print("=" * 60)

    feats = pd.DataFrame(index=df.index)

    # --- 均线排列关系 ---
    feats['ma_spread_5_10'] = (df['ma5'] - df['ma10']) / (df['ma10'] + 1e-10)
    feats['ma_spread_5_20'] = (df['ma5'] - df['ma20']) / (df['ma20'] + 1e-10)
    feats['ma_spread_5_60'] = (df['ma5'] - df['ma60']) / (df['ma60'] + 1e-10)
    feats['ma_spread_20_60'] = (df['ma20'] - df['ma60']) / (df['ma60'] + 1e-10)
    feats['ma_alignment'] = (df['ma5'] - df['ma10']) / (df['ma10'] - df['ma20'] + 1e-10)

    # --- 斜率关系 ---
    feats['slope_ratio_5_20'] = df['ma5_slope'] / (df['ma20_slope'] + 1e-10)
    feats['slope_accel'] = df['ma5_slope'] - df['ma20_slope']

    # --- MACD关系 ---
    feats['macd_hist_ratio'] = df['macd_hist'] / (df['boll_std'] + 1e-10)
    feats['dif_dea_ratio'] = df['dif'] / (df['dea'] + 1e-10)
    feats['dif_position'] = df['dif'] / (df['boll_std'] + 1e-10)

    # --- RSI关系 ---
    feats['rsi_ratio_6_14'] = df['rsi_6'] / (df['rsi_14'] + 1e-10)
    feats['rsi_ratio_14_24'] = df['rsi_14'] / (df['rsi_24'] + 1e-10)
    feats['rsi_deviation'] = (df['rsi_14'] - 50) / 50

    # --- 布林带关系 ---
    feats['boll_squeeze'] = df['boll_width'] / 4.0
    feats['boll_breakout_up'] = (df['boll_up'] - df['close']) / (df['atr_14'] + 1e-10)
    feats['boll_breakout_down'] = (df['close'] - df['boll_down']) / (df['atr_14'] + 1e-10)

    # --- 量价关系 ---
    feats['vol_price_divergence'] = df['vol_ratio'] * np.sign(df['pct_change'])
    feats['vol_acceleration'] = df['volume'] / (df['vol_ma5'].shift(1) + 1)

    # --- 波动率关系 ---
    feats['vol_ratio_short_long'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)
    feats['vol_expansion'] = df['volatility_5d'] - df['volatility_20d']

    # --- 动量关系 ---
    feats['momentum_gradient'] = (df['momentum_5d'] - df['momentum_20d']) / 15
    feats['momentum_accel'] = df['momentum_5d'] - df['momentum_5d'].shift(1)

    # --- ATR关系 ---
    feats['atr_price_ratio'] = df['atr_14'] / df['close']
    feats['atr_expansion'] = df['atr_14'] / (df['atr_14'].rolling(20).mean() + 1e-10)

    # --- KDJ关系 ---
    feats['kdj_stoch'] = df['j'] / (df['k'] + df['d'] + 1e-10)
    feats['kd_divergence'] = (df['k'] - df['d']) / (df['d'] + 1e-10)

    # --- OBV趋势 ---
    feats['obv_trend'] = df['obv'] / (df['obv_ma10'] + 1e-10)

    # --- 价格位置梯度 ---
    feats['price_pos_gradient'] = df['price_pos_20d'] - df['price_pos_60d']

    # --- 大盘宽度 ---
    feats['width_gradient'] = df['width_5d'] - df['width_20d']

    # --- 连涨连跌 ---
    feats['streak'] = df['up_days'] - df['down_days']

    print(f"关系特征构建完成，共 {len(feats.columns)} 个特征")
    return feats


# ============================================================
# Step 4: 标签构建
# ============================================================
def build_labels(df):
    """构建预测标签"""
    print("\n" + "=" * 60)
    print("Step 4: 构建预测标签")
    print("=" * 60)

    # 未来N日涨跌幅
    df['future_1d'] = df['close'].shift(-1) / df['close'] - 1
    df['future_2d'] = df['close'].shift(-2) / df['close'] - 1
    df['future_3d'] = df['close'].shift(-3) / df['close'] - 1

    # 分类标签
    df['label_1d_up'] = (df['future_1d'] > 0.01).astype(int)    # 明日涨超1%
    df['label_1d_down'] = (df['future_1d'] < -0.01).astype(int)   # 明日跌超1%
    df['label_2d_up'] = (df['future_2d'] > 0.02).astype(int)    # 2日涨超2%
    df['label_2d_down'] = (df['future_2d'] < -0.02).astype(int)
    df['label_3d_up'] = (df['future_3d'] > 0.03).astype(int)    # 3日涨超3%
    df['label_3d_down'] = (df['future_3d'] < -0.03).astype(int)

    # 三分类标签: 1=涨, 0=平, -1=跌
    df['label_1d_3class'] = 0
    df.loc[df['label_1d_up'] == 1, 'label_1d_3class'] = 1
    df.loc[df['label_1d_down'] == 1, 'label_1d_3class'] = -1

    df['label_2d_3class'] = 0
    df.loc[df['label_2d_up'] == 1, 'label_2d_3class'] = 1
    df.loc[df['label_2d_down'] == 1, 'label_2d_3class'] = -1

    df['label_3d_3class'] = 0
    df.loc[df['label_3d_up'] == 1, 'label_3d_3class'] = 1
    df.loc[df['label_3d_down'] == 1, 'label_3d_3class'] = -1

    print("标签构建完成:")
    print(f"  1日涨(>1%): {df['label_1d_up'].sum()} / 平: {((df['label_1d_3class'] == 0)).sum()} / 跌(<-1%): {df['label_1d_down'].sum()}")
    print(f"  2日涨(>2%): {df['label_2d_up'].sum()} / 平: {((df['label_2d_3class'] == 0)).sum()} / 跌(<-2%): {df['label_2d_down'].sum()}")
    print(f"  3日涨(>3%): {df['label_3d_up'].sum()} / 平: {((df['label_3d_3class'] == 0)).sum()} / 跌(<-3%): {df['label_3d_down'].sum()}")

    return df


# ============================================================
# Step 5: 模型训练
# ============================================================
def get_feature_columns():
    """获取所有特征列名"""
    # 基础指标特征
    basic_features = [
        'ma5', 'ma10', 'ma20', 'ma60', 'ma120',
        'dif', 'dea', 'macd_hist',
        'rsi_6', 'rsi_14', 'rsi_24',
        'boll_pos', 'boll_width',
        'k', 'd', 'j',
        'vol_ratio',
        'volatility_5d', 'volatility_20d',
        'momentum_1d', 'momentum_5d', 'momentum_10d', 'momentum_20d',
        'atr_14',
        'pct_change',
        'price_pos_20d', 'price_pos_60d',
        'ma5_slope', 'ma20_slope',
        'obv_trend',
        'width_5d', 'width_20d',
        'up_days', 'down_days',
    ]

    # 关系特征
    relation_features = [
        'ma_spread_5_10', 'ma_spread_5_20', 'ma_spread_5_60', 'ma_spread_20_60', 'ma_alignment',
        'slope_ratio_5_20', 'slope_accel',
        'macd_hist_ratio', 'dif_dea_ratio', 'dif_position',
        'rsi_ratio_6_14', 'rsi_ratio_14_24', 'rsi_deviation',
        'boll_squeeze', 'boll_breakout_up', 'boll_breakout_down',
        'vol_price_divergence', 'vol_acceleration',
        'vol_ratio_short_long', 'vol_expansion',
        'momentum_gradient', 'momentum_accel',
        'atr_price_ratio', 'atr_expansion',
        'kdj_stoch', 'kd_divergence',
        'obv_trend',
        'price_pos_gradient',
        'width_gradient',
        'streak',
    ]

    return basic_features + list(set(relation_features) - set(basic_features))


def train_classification_model(X_train, y_train, X_test, y_test, model_name, params=None):
    """
    训练XGBoost分类模型（三分类: 涨/平/跌）
    """
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 整体指标
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # 各类别指标
    report = classification_report(y_test, y_pred, target_names=['跌', '平', '涨'], output_dict=True, zero_division=0)

    # AUC (one-vs-rest)
    try:
        y_test_bin = label_binarize(y_test, classes=[-1, 0, 1])
        auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc_ovr = 0.0

    # 特征重要性
    importance = model.feature_importances_
    feat_names = X_train.columns.tolist()
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)

    metrics = {
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_macro, 4),
        'auc_ovr': round(auc_ovr, 4),
        'class_report': {
            '涨': {
                'precision': round(report['涨']['precision'], 4),
                'recall': round(report['涨']['recall'], 4),
                'f1': round(report['涨']['f1-score'], 4),
                'support': int(report['涨']['support']),
            },
            '平': {
                'precision': round(report['平']['precision'], 4),
                'recall': round(report['平']['recall'], 4),
                'f1': round(report['平']['f1-score'], 4),
                'support': int(report['平']['support']),
            },
            '跌': {
                'precision': round(report['跌']['precision'], 4),
                'recall': round(report['跌']['recall'], 4),
                'f1': round(report['跌']['f1-score'], 4),
                'support': int(report['跌']['support']),
            },
        },
        'feature_importance_top15': [
            {'feature': name, 'importance': round(imp, 4)}
            for name, imp in feat_imp[:15]
        ],
    }

    print(f"\n--- {model_name} ---")
    print(f"准确率: {acc:.4f}")
    print(f"F1(macro): {f1_macro:.4f}")
    print(f"AUC(OVR): {auc_ovr:.4f}")
    print(f"特征重要性 Top5: {[(n, round(i, 4)) for n, i in feat_imp[:5]]}")

    return model, metrics


def train_regression_model(X_train, y_train, X_test, y_test, model_name, params=None):
    """
    训练XGBoost回归模型（预测未来1日涨跌幅）
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 方向准确率
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test))

    # 特征重要性
    importance = model.feature_importances_
    feat_names = X_train.columns.tolist()
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)

    metrics = {
        'mse': round(mse, 6),
        'mae': round(mae, 6),
        'r2': round(r2, 4),
        'direction_accuracy': round(direction_acc, 4),
        'feature_importance_top15': [
            {'feature': name, 'importance': round(imp, 4)}
            for name, imp in feat_imp[:15]
        ],
    }

    print(f"\n--- {model_name} ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.4f}")
    print(f"方向准确率: {direction_acc:.4f}")
    print(f"特征重要性 Top5: {[(n, round(i, 4)) for n, i in feat_imp[:5]]}")

    return model, metrics


def train_all_models(df, feature_columns):
    """训练所有模型"""
    print("\n" + "=" * 60)
    print("Step 5: 训练XGBoost预测模型")
    print("=" * 60)

    # 准备特征和标签
    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 三分类标签映射: -1->0, 0->1, 1->2 (XGBoost要求从0开始)
    y_1d = df['label_1d_3class'].copy().map({-1: 0, 0: 1, 1: 2})
    y_2d = df['label_2d_3class'].copy().map({-1: 0, 0: 1, 1: 2})
    y_3d = df['label_3d_3class'].copy().map({-1: 0, 0: 1, 1: 2})
    y_reg = df['future_1d'].copy()

    # 去除NaN
    valid_mask = X.notna().all(axis=1) & y_1d.notna() & y_2d.notna() & y_3d.notna() & y_reg.notna()
    X = X[valid_mask]
    y_1d = y_1d[valid_mask]
    y_2d = y_2d[valid_mask]
    y_3d = y_3d[valid_mask]
    y_reg = y_reg[valid_mask]

    print(f"有效样本数: {len(X)}")

    # 时间序列分割: 前7年训练，后1年测试
    split_date = "2025-06-01"
    split_idx = df.loc[valid_mask].index[df.loc[valid_mask]['date'] >= split_date].min()
    if pd.isna(split_idx):
        split_idx = int(len(X) * 0.85)

    # 转换为位置索引
    train_mask = df.loc[valid_mask]['date'] < split_date
    test_mask = df.loc[valid_mask]['date'] >= split_date

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_1d_train = y_1d[train_mask]
    y_1d_test = y_1d[test_mask]
    y_2d_train = y_2d[train_mask]
    y_2d_test = y_2d[test_mask]
    y_3d_train = y_3d[train_mask]
    y_3d_test = y_3d[test_mask]
    y_reg_train = y_reg[train_mask]
    y_reg_test = y_reg[test_mask]

    print(f"训练集: {len(X_train)} 条 (截至 {df.loc[valid_mask][train_mask]['date'].max()})")
    print(f"测试集: {len(X_test)} 条 (从 {df.loc[valid_mask][test_mask]['date'].min()} 起)")

    models = {}
    all_metrics = {}

    # 模型1: 1日涨跌方向（三分类）
    print("\n>>> 训练模型1: 1日涨跌方向预测（涨/平/跌）")
    m1, met1 = train_classification_model(
        X_train, y_1d_train, X_test, y_1d_test,
        "1日涨跌方向模型"
    )
    models['model_1d_direction'] = m1
    all_metrics['model_1d_direction'] = met1
    joblib.dump(m1, os.path.join(MODEL_DIR, "model_1d_direction.pkl"))

    # 模型2: 2日涨跌方向（三分类）
    print("\n>>> 训练模型2: 2日涨跌方向预测（涨/平/跌）")
    m2, met2 = train_classification_model(
        X_train, y_2d_train, X_test, y_2d_test,
        "2日涨跌方向模型"
    )
    models['model_2d_direction'] = m2
    all_metrics['model_2d_direction'] = met2
    joblib.dump(m2, os.path.join(MODEL_DIR, "model_2d_direction.pkl"))

    # 模型3: 3日涨跌方向（三分类）
    print("\n>>> 训练模型3: 3日涨跌方向预测（涨/平/跌）")
    m3, met3 = train_classification_model(
        X_train, y_3d_train, X_test, y_3d_test,
        "3日涨跌方向模型"
    )
    models['model_3d_direction'] = m3
    all_metrics['model_3d_direction'] = met3
    joblib.dump(m3, os.path.join(MODEL_DIR, "model_3d_direction.pkl"))

    # 模型4: 回归预测未来1日涨跌幅
    print("\n>>> 训练模型4: 未来1日涨跌幅回归预测")
    m4, met4 = train_regression_model(
        X_train, y_reg_train, X_test, y_reg_test,
        "1日涨跌幅回归模型"
    )
    models['model_1d_regression'] = m4
    all_metrics['model_1d_regression'] = met4
    joblib.dump(m4, os.path.join(MODEL_DIR, "model_1d_regression.pkl"))

    return models, all_metrics, X_test, y_1d_test, y_2d_test, y_3d_test, y_reg_test, test_mask


# ============================================================
# Step 6: 大盘状态判断
# ============================================================
def judge_market_status(pred_1d, pred_2d, pred_3d):
    """
    基于模型输出判断大盘状态

    pred格式: -1=跌, 0=平, 1=涨

    状态定义:
    - "强势上涨": 模型1看涨 + 模型2看涨 + 模型3看涨
    - "偏多震荡": 模型1看涨 + (模型2或3看平)
    - "中性震荡": 所有模型看平
    - "偏空震荡": 模型1看跌 + (模型2或3看平)
    - "强势下跌": 模型1看跌 + 模型2看跌 + 模型3看跌
    """
    if pred_1d == 1 and pred_2d == 1 and pred_3d == 1:
        return "强势上涨"
    elif pred_1d == 1 and pred_2d <= 0 and pred_3d <= 0:
        return "偏多震荡"
    elif pred_1d == -1 and pred_2d >= 0 and pred_3d >= 0:
        return "偏空震荡"
    elif pred_1d == -1 and pred_2d == -1 and pred_3d == -1:
        return "强势下跌"
    elif pred_1d == 1:
        return "偏多震荡"
    elif pred_1d == -1:
        return "偏空震荡"
    else:
        return "中性震荡"


def generate_market_status_report(models, X_test, df_test):
    """生成大盘状态判断报告"""
    print("\n" + "=" * 60)
    print("Step 6: 大盘状态判断")
    print("=" * 60)

    # 预测
    pred_1d_prob = models['model_1d_direction'].predict_proba(X_test)
    pred_2d_prob = models['model_2d_direction'].predict_proba(X_test)
    pred_3d_prob = models['model_3d_direction'].predict_proba(X_test)

    # 概率转标签: 0->-1(跌), 1->0(平), 2->1(涨)
    class_map = {0: -1, 1: 0, 2: 1}
    pred_1d = np.array([class_map[p] for p in models['model_1d_direction'].predict(X_test)])
    pred_2d = np.array([class_map[p] for p in models['model_2d_direction'].predict(X_test)])
    pred_3d = np.array([class_map[p] for p in models['model_3d_direction'].predict(X_test)])

    # 最近一个交易日的状态
    latest_idx = -1
    latest_status = judge_market_status(pred_1d[latest_idx], pred_2d[latest_idx], pred_3d[latest_idx])
    latest_date = df_test['date'].iloc[latest_idx]

    print(f"\n最新交易日({latest_date})大盘状态: {latest_status}")
    print(f"  1日预测: {'涨' if pred_1d[latest_idx] == 1 else '平' if pred_1d[latest_idx] == 0 else '跌'} "
          f"(涨概率: {pred_1d_prob[latest_idx][2]:.2%}, 跌概率: {pred_1d_prob[latest_idx][0]:.2%})")
    print(f"  2日预测: {'涨' if pred_2d[latest_idx] == 1 else '平' if pred_2d[latest_idx] == 0 else '跌'} "
          f"(涨概率: {pred_2d_prob[latest_idx][2]:.2%}, 跌概率: {pred_2d_prob[latest_idx][0]:.2%})")
    print(f"  3日预测: {'涨' if pred_3d[latest_idx] == 1 else '平' if pred_3d[latest_idx] == 0 else '跌'} "
          f"(涨概率: {pred_3d_prob[latest_idx][2]:.2%}, 跌概率: {pred_3d_prob[latest_idx][0]:.2%})")

    # 回测期状态分布
    status_list = [judge_market_status(p1, p2, p3) for p1, p2, p3 in zip(pred_1d, pred_2d, pred_3d)]
    status_counts = pd.Series(status_list).value_counts()
    print(f"\n回测期大盘状态分布:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}天 ({count / len(status_list):.1%})")

    # 大盘状态判断规则
    rules = {
        "强势上涨": "模型1看涨 + 模型2看涨 + 模型3看涨",
        "偏多震荡": "模型1看涨 + (模型2或3非涨)",
        "中性震荡": "模型1看平 + (模型2和3非一致看涨/看跌)",
        "偏空震荡": "模型1看跌 + (模型2或3非跌)",
        "强势下跌": "模型1看跌 + 模型2看跌 + 模型3看跌",
    }
    print(f"\n大盘状态判断规则:")
    for status, rule in rules.items():
        print(f"  {status}: {rule}")

    return {
        'latest_date': latest_date,
        'latest_status': latest_status,
        'latest_predictions': {
            '1d': {'direction': int(pred_1d[latest_idx]), 'up_prob': round(float(pred_1d_prob[latest_idx][2]), 4), 'down_prob': round(float(pred_1d_prob[latest_idx][0]), 4)},
            '2d': {'direction': int(pred_2d[latest_idx]), 'up_prob': round(float(pred_2d_prob[latest_idx][2]), 4), 'down_prob': round(float(pred_2d_prob[latest_idx][0]), 4)},
            '3d': {'direction': int(pred_3d[latest_idx]), 'up_prob': round(float(pred_3d_prob[latest_idx][2]), 4), 'down_prob': round(float(pred_3d_prob[latest_idx][0]), 4)},
        },
        'status_distribution': {status: int(count) for status, count in status_counts.items()},
        'rules': rules,
    }


# ============================================================
# Step 7: 回测验证
# ============================================================
def run_backtest(models, X_test, df_test, y_1d_test, y_2d_test, y_3d_test, y_reg_test):
    """样本外回测"""
    print("\n" + "=" * 60)
    print("Step 7: 回测验证 (2025-06-01 ~ 2026-06-05)")
    print("=" * 60)

    class_map = {0: -1, 1: 0, 2: 1}
    pred_1d = np.array([class_map[p] for p in models['model_1d_direction'].predict(X_test)])
    pred_2d = np.array([class_map[p] for p in models['model_2d_direction'].predict(X_test)])
    pred_3d = np.array([class_map[p] for p in models['model_3d_direction'].predict(X_test)])
    pred_reg = models['model_1d_regression'].predict(X_test)

    dates = df_test['date'].values
    actual_1d = y_1d_test.values
    actual_2d = y_2d_test.values
    actual_3d = y_3d_test.values
    actual_reg = y_reg_test.values

    # --- 1日方向胜率 ---
    correct_1d = (pred_1d == actual_1d)
    win_rate_1d = correct_1d.mean()
    print(f"\n1日方向预测胜率: {win_rate_1d:.2%} ({correct_1d.sum()}/{len(correct_1d)})")

    # --- 2日方向胜率 ---
    correct_2d = (pred_2d == actual_2d)
    win_rate_2d = correct_2d.mean()
    print(f"2日方向预测胜率: {win_rate_2d:.2%} ({correct_2d.sum()}/{len(correct_2d)})")

    # --- 3日方向胜率 ---
    correct_3d = (pred_3d == actual_3d)
    win_rate_3d = correct_3d.mean()
    print(f"3日方向预测胜率: {win_rate_3d:.2%} ({correct_3d.sum()}/{len(correct_3d)})")

    # --- 回归方向准确率 ---
    reg_direction_acc = np.mean(np.sign(pred_reg) == np.sign(actual_reg))
    print(f"回归模型方向准确率: {reg_direction_acc:.2%}")

    # --- 模拟交易: 看涨买入持有1日，看跌空仓 ---
    print("\n--- 模拟交易回测 ---")
    cumulative_return = 1.0
    returns_list = [1.0]
    trade_count = 0
    win_count = 0
    loss_count = 0
    total_profit = 0
    total_loss = 0

    for i in range(len(dates)):
        if pred_1d[i] == 1:  # 看涨 -> 买入持有1日
            daily_return = actual_reg[i]
            cumulative_return *= (1 + daily_return)
            trade_count += 1
            if daily_return > 0:
                win_count += 1
                total_profit += daily_return
            else:
                loss_count += 1
                total_loss += abs(daily_return)
        # 看跌或看平 -> 空仓（收益不变）
        returns_list.append(cumulative_return)

    print(f"交易次数: {trade_count}")
    print(f"交易胜率: {win_count / max(trade_count, 1):.2%}")
    print(f"盈亏比: {total_profit / max(total_loss, 1e-10):.2f}")
    print(f"累计收益: {cumulative_return:.4f} ({(cumulative_return - 1) * 100:.2f}%)")

    # --- 持有2日策略 ---
    cumulative_return_2d = 1.0
    trade_count_2d = 0
    win_count_2d = 0

    for i in range(len(dates) - 1):
        if pred_2d[i] == 1:  # 看涨 -> 买入持有2日
            ret = actual_reg[i] + actual_reg[min(i + 1, len(actual_reg) - 1)]
            cumulative_return_2d *= (1 + ret)
            trade_count_2d += 1
            if ret > 0:
                win_count_2d += 1

    print(f"\n持有2日策略:")
    print(f"交易次数: {trade_count_2d}")
    print(f"交易胜率: {win_count_2d / max(trade_count_2d, 1):.2%}")
    print(f"累计收益: {cumulative_return_2d:.4f} ({(cumulative_return_2d - 1) * 100:.2f}%)")

    # --- 持有3日策略 ---
    cumulative_return_3d = 1.0
    trade_count_3d = 0
    win_count_3d = 0

    for i in range(len(dates) - 2):
        if pred_3d[i] == 1:
            ret = actual_reg[i] + actual_reg[min(i + 1, len(actual_reg) - 1)] + actual_reg[min(i + 2, len(actual_reg) - 1)]
            cumulative_return_3d *= (1 + ret)
            trade_count_3d += 1
            if ret > 0:
                win_count_3d += 1

    print(f"\n持有3日策略:")
    print(f"交易次数: {trade_count_3d}")
    print(f"交易胜率: {win_count_3d / max(trade_count_3d, 1):.2%}")
    print(f"累计收益: {cumulative_return_3d:.4f} ({(cumulative_return_3d - 1) * 100:.2f}%)")

    # --- 基准: 买入持有 ---
    benchmark_return = 1.0
    for i in range(len(actual_reg)):
        benchmark_return *= (1 + actual_reg[i])
    print(f"\n基准(买入持有): {benchmark_return:.4f} ({(benchmark_return - 1) * 100:.2f}%)")

    backtest_result = {
        'backtest_period': f"{dates[0]} ~ {dates[-1]}",
        'trading_days': len(dates),
        'direction_accuracy': {
            '1d': round(float(win_rate_1d), 4),
            '2d': round(float(win_rate_2d), 4),
            '3d': round(float(win_rate_3d), 4),
            'regression': round(float(reg_direction_acc), 4),
        },
        'strategy_1d': {
            'trade_count': int(trade_count),
            'win_rate': round(float(win_count / max(trade_count, 1)), 4),
            'profit_loss_ratio': round(float(total_profit / max(total_loss, 1e-10)), 4),
            'cumulative_return': round(float(cumulative_return), 4),
        },
        'strategy_2d': {
            'trade_count': int(trade_count_2d),
            'win_rate': round(float(win_count_2d / max(trade_count_2d, 1)), 4),
            'cumulative_return': round(float(cumulative_return_2d), 4),
        },
        'strategy_3d': {
            'trade_count': int(trade_count_3d),
            'win_rate': round(float(win_count_3d / max(trade_count_3d, 1)), 4),
            'cumulative_return': round(float(cumulative_return_3d), 4),
        },
        'benchmark': {
            'cumulative_return': round(float(benchmark_return), 4),
        },
    }

    return backtest_result


# ============================================================
# Step 8: 输出结果
# ============================================================
def save_results(all_metrics, market_status, backtest_result):
    """保存结果到JSON文件"""
    print("\n" + "=" * 60)
    print("Step 8: 保存结果")
    print("=" * 60)

    result = {
        'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_metrics': all_metrics,
        'market_status': market_status,
        'backtest': backtest_result,
    }

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {RESULT_FILE}")
    print(f"模型已保存到: {MODEL_DIR}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)

    print("\n【模型性能】")
    for name, metrics in all_metrics.items():
        if 'accuracy' in metrics:
            print(f"  {name}: 准确率={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}, AUC={metrics['auc_ovr']:.4f}")
        else:
            print(f"  {name}: MAE={metrics['mae']:.6f}, R2={metrics['r2']:.4f}, 方向准确率={metrics['direction_accuracy']:.4f}")

    print("\n【大盘状态】")
    print(f"  最新日期: {market_status['latest_date']}")
    print(f"  当前状态: {market_status['latest_status']}")

    print("\n【回测结果】")
    bt = backtest_result
    print(f"  回测区间: {bt['backtest_period']}")
    print(f"  1日方向胜率: {bt['direction_accuracy']['1d']:.2%}")
    print(f"  1日策略累计收益: {bt['strategy_1d']['cumulative_return']:.4f}")
    print(f"  基准收益: {bt['benchmark']['cumulative_return']:.4f}")

    return result


# ============================================================
# 主函数
# ============================================================
def main():
    """主流程"""
    start_time = time.time()
    print("=" * 60)
    print("上证指数(sh.000001)多因子预测模型")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: 数据获取
    df = fetch_index_data()

    # Step 2: 基础指标计算
    df = compute_basic_indicators(df)

    # Step 3: 关系特征
    relation_feats = build_relation_features(df)
    df = pd.concat([df, relation_feats], axis=1)

    # Step 4: 标签构建
    df = build_labels(df)

    # 获取特征列
    feature_columns = get_feature_columns()
    # 确保特征列存在
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"\n最终特征数: {len(feature_columns)}")

    # Step 5: 训练模型
    models, all_metrics, X_test, y_1d_test, y_2d_test, y_3d_test, y_reg_test, test_mask = train_all_models(df, feature_columns)

    # 获取测试集对应的df
    df_test = df.loc[test_mask].reset_index(drop=True)

    # Step 6: 大盘状态判断
    market_status = generate_market_status_report(models, X_test, df_test)

    # Step 7: 回测
    backtest_result = run_backtest(models, X_test, df_test, y_1d_test, y_2d_test, y_3d_test, y_reg_test)

    # Step 8: 保存结果
    result = save_results(all_metrics, market_status, backtest_result)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}秒")
    print("=" * 60)
    print("全部完成!")
    print("=" * 60)

    return result


if __name__ == '__main__':
    main()
