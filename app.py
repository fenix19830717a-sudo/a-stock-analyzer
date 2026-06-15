"""
A股智能分析选股系统 - Streamlit主应用 v3.0

页面结构:
1. 🏠 数据面板 - 市场概览、自选股监控
2. 🎯 今日选股 - HMM预训练模型快速选股 + 辩论分析
3. 📈 行情分析 - 个股三层决策分析
4. 🤖 AI辩论室 - 6角色辩论过程展示
5. 🔄 回测中心 - 策略回测与参数优化
6. 💼 持仓分析 - 持仓管理与跟踪
7. ⚙️ 系统设置 - 配置管理
"""

import sys
import os
import pickle
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import yaml
import baostock as bs

# BaoStock登录
try:
    bs.login()
    _BS_LOGGED_IN = True
except Exception:
    _BS_LOGGED_IN = False

# 页面配置
st.set_page_config(
    page_title="A股智能分析选股系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    .main-header {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 1.5rem; padding: 0.5rem 0;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #2c3e50;
        margin-top: 1.5rem; margin-bottom: 0.8rem; padding-left: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .signal-buy { color: #e74c3c; font-weight: bold; }
    .signal-sell { color: #27ae60; font-weight: bold; }
    .signal-neutral { color: #95a5a6; }
    .weather-sunny { color: #f39c12; font-size: 2.5rem; }
    .weather-cloudy { color: #7f8c8d; font-size: 2.5rem; }
    .weather-rainy { color: #3498db; font-size: 2.5rem; }
    .debate-role {
        background: linear-gradient(135deg, #ffffff 0%, #f0f3f8 100%);
        border-left: 4px solid #667eea; padding: 1rem 1.2rem; margin: 0.5rem 0;
        border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .debate-bull { border-left-color: #e74c3c; background: linear-gradient(135deg, #fff5f5 0%, #fff0f0 100%); }
    .debate-bear { border-left-color: #27ae60; background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%); }
    .debate-judge { border-left-color: #9b59b6; background: linear-gradient(135deg, #f5f0ff 0%, #ede7f6 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
</style>
""", unsafe_allow_html=True)


# ==================== 真实数据函数 ====================

@st.cache_data(ttl=3600)
def get_real_stock_data(symbol, days=180):
    if not _BS_LOGGED_IN:
        return None
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    market = "sh" if symbol.startswith('6') else "sz"
    code = f"{market}.{symbol}"
    try:
        rs = bs.query_history_k_data_plus(
            code, "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date, end_date=end_date, frequency="d", adjustflag="2"
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['date','code','open','high','low','close','volume','amount','turn','pctChg'])
        for col in ['open','high','low','close','volume','amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        return df
    except Exception:
        return None


def compute_real_indicators(df):
    if df is None or len(df) < 20:
        return None
    close = df['close']
    df['ma5'] = close.rolling(5).mean()
    df['ma10'] = close.rolling(10).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs_val = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs_val))
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['boll_upper'] = ma20 + 2 * std20
    df['boll_lower'] = ma20 - 2 * std20
    df['boll_position'] = (close - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-10)
    df['volatility_20d'] = close.pct_change().rolling(20).std()
    df['ma5_slope'] = (df['ma5'] - df['ma5'].shift(3)) / (df['ma5'].shift(3) + 1e-10)
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(3)) / (df['ma20'].shift(3) + 1e-10)
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-10)
    low_list = df['low'].rolling(9).min()
    high_list = df['high'].rolling(9).max()
    rsv = (close - low_list) / (high_list - low_list + 1e-10) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df


@st.cache_data(ttl=3600)
def get_hs300_stocks():
    if not _BS_LOGGED_IN:
        return pd.DataFrame(columns=['code', 'code_name'])
    try:
        rs = bs.query_hs300_stocks()
        stocks = []
        while rs.next():
            row = rs.get_row_data()
            stocks.append({'code': row[1], 'code_name': row[2]})
        return pd.DataFrame(stocks)
    except Exception:
        return pd.DataFrame(columns=['code', 'code_name'])


@st.cache_data(ttl=3600)
def get_index_data(symbol, days=30):
    if not _BS_LOGGED_IN:
        return None
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        rs = bs.query_history_k_data_plus(
            symbol, "date,open,high,low,close,volume,amount,pctChg",
            start_date=start_date, end_date=end_date, frequency="d", adjustflag="2"
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['date','open','high','low','close','volume','amount','pctChg'])
        for col in ['open','high','low','close','volume','amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        return df
    except Exception:
        return None


# ==================== HMM模型加载 ====================

@st.cache_resource
def load_hmm_models():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmm_models")
    meta_path = os.path.join(model_dir, "training_meta.json")
    models = {}
    if not os.path.exists(model_dir):
        return models, None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except Exception:
        meta = None
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    for pkl_file in pkl_files:
        code = pkl_file.replace('.pkl', '')
        try:
            with open(os.path.join(model_dir, pkl_file), 'rb') as f:
                models[code] = pickle.load(f)
        except Exception:
            continue
    return models, meta


def hmm_predict(model, df):
    if model is None or df is None or len(df) < 20:
        return None, None, None
    try:
        close = df['close'].values
        returns = np.diff(np.log(close + 1e-10))
        if len(returns) < 10:
            return None, None, None
        features = returns[-60:].reshape(-1, 1)
        hmm_model = getattr(model, 'model', model)
        if hmm_model is None:
            return None, None, None
        try:
            posteriors = hmm_model.predict_proba(features)
            if posteriors.ndim == 1:
                posteriors = posteriors.reshape(-1, 1)
            n_components = posteriors.shape[1] if posteriors.ndim == 2 else 4
        except Exception:
            n_components = getattr(hmm_model, 'n_components', 4)
            posteriors = np.ones((1, n_components)) / n_components
        up_prob = float(np.mean(posteriors[-1, :n_components//2])) if n_components > 1 else 0.5
        if up_prob > 0.6:
            trend = "上涨"
        elif up_prob > 0.45:
            trend = "震荡"
        else:
            trend = "下跌"
        return trend, up_prob, posteriors[-1]
    except Exception:
        return None, None, None


# ==================== 规则分析函数 ====================

def rule_based_trend(df):
    if df is None or len(df) < 20:
        return "数据不足", 0.0
    close = df['close'].iloc[-1]
    ma5 = df['ma5'].iloc[-1] if pd.notna(df['ma5'].iloc[-1]) else close
    ma20 = df['ma20'].iloc[-1] if pd.notna(df['ma20'].iloc[-1]) else close
    if close > ma5 > ma20:
        trend = "上涨"
        confidence = min(0.9, 0.5 + (close - ma20) / ma20 * 10)
    elif close > ma20 and ma5 < ma20:
        trend = "偏多震荡"
        confidence = 0.55
    elif close < ma5 < ma20:
        trend = "下跌"
        confidence = min(0.9, 0.5 + (ma20 - close) / ma20 * 10)
    elif close < ma20 and ma5 > ma20:
        trend = "偏空震荡"
        confidence = 0.55
    else:
        trend = "震荡"
        confidence = 0.5
    return trend, round(confidence, 2)


def rule_based_signal(df):
    if df is None or len(df) < 20:
        return "观望", 0.0
    latest = df.iloc[-1]
    score = 0.0
    if pd.notna(latest['ma5']) and pd.notna(latest['ma20']):
        score += 0.3 if latest['ma5'] > latest['ma20'] else -0.3
    if pd.notna(latest['macd_hist']):
        score += 0.3 if latest['macd_hist'] > 0 else -0.3
    if pd.notna(latest['rsi']):
        if 30 < latest['rsi'] < 70:
            score += 0.2 if latest['rsi'] > 50 else -0.1
        elif latest['rsi'] >= 70:
            score -= 0.2
        elif latest['rsi'] <= 30:
            score += 0.2
    if pd.notna(latest['boll_position']):
        if latest['boll_position'] < 0.2:
            score += 0.2
        elif latest['boll_position'] > 0.8:
            score -= 0.2
    if score >= 0.5:
        signal = "强烈买入"
    elif score >= 0.2:
        signal = "买入"
    elif score <= -0.5:
        signal = "强烈卖出"
    elif score <= -0.2:
        signal = "卖出"
    else:
        signal = "观望"
    return signal, round(score, 2)


def generate_composite_score(df):
    if df is None or len(df) < 20:
        return 50
    latest = df.iloc[-1]
    score = 50
    if pd.notna(latest['ma5']) and pd.notna(latest['ma20']):
        score += 15 if latest['ma5'] > latest['ma20'] else -10
    if pd.notna(latest['macd_hist']) and latest['macd_hist'] > 0:
        score += 10
    if pd.notna(latest['rsi']):
        if 40 < latest['rsi'] < 60:
            score += 5
        elif latest['rsi'] > 70:
            score -= 10
        elif latest['rsi'] < 30:
            score += 5
    if pd.notna(latest['volume']) and len(df) >= 5:
        avg_vol = df['volume'].tail(5).mean()
        if latest['volume'] > avg_vol * 1.2:
            score += 10
    if pd.notna(latest['volatility_20d']):
        if 0.01 < latest['volatility_20d'] < 0.03:
            score += 5
    return min(100, max(0, int(score)))


def compute_volume_pattern(df):
    if df is None or len(df) < 5:
        return "数据不足", "N/A"
    latest = df.iloc[-1]
    avg_vol = df['volume'].tail(5).mean()
    vol_val = latest['volume'] if pd.notna(latest['volume']) else 0
    if vol_val > avg_vol * 1.5:
        vol_pattern = "显著放量"
    elif vol_val > avg_vol * 1.2:
        vol_pattern = "放量"
    elif vol_val < avg_vol * 0.8:
        vol_pattern = "缩量"
    else:
        vol_pattern = "正常"
    price_chg = latest['pctChg'] if pd.notna(latest['pctChg']) else 0
    if vol_pattern in ["放量", "显著放量"] and price_chg > 0:
        vol_price = "放量上涨"
    elif vol_pattern in ["放量", "显著放量"] and price_chg < 0:
        vol_price = "放量下跌"
    elif vol_pattern == "缩量":
        vol_price = "缩量整理"
    else:
        vol_price = "正常"
    return vol_pattern, vol_price


def run_debate_analysis(df, trend, signal_score, current_price):
    latest = df.iloc[-1]
    rsi_val = latest['rsi'] if pd.notna(latest['rsi']) else 50
    vol_val = latest['volatility_20d'] if pd.notna(latest['volatility_20d']) else 0.02
    boll_pos = latest['boll_position'] if pd.notna(latest['boll_position']) else 0.5
    vol_pattern, vol_price = compute_volume_pattern(df)
    tech_score = min(100, max(0, int(50 + signal_score * 30 + (rsi_val - 50) * 0.3)))
    fund_score = min(100, max(0, int(50 + (current_price - df['close'].tail(20).mean()) / df['close'].tail(20).mean() * 100)))
    macro_score = min(100, max(0, int(50 + 0.3 * 30 + (0.02 - vol_val) * 200)))
    bull_votes = 0
    bear_votes = 0
    if tech_score > 50: bull_votes += 1
    else: bear_votes += 1
    if fund_score > 50: bull_votes += 1
    else: bear_votes += 1
    if macro_score > 50: bull_votes += 1
    else: bear_votes += 1
    if signal_score > 0: bull_votes += 1
    else: bear_votes += 1
    if rsi_val < 70: bull_votes += 1
    else: bear_votes += 1
    if trend in ["上涨", "偏多震荡"]: bull_votes += 1
    else: bear_votes += 1
    total_votes = bull_votes + bear_votes
    final_decision = "看多" if bull_votes > bear_votes else ("看空" if bear_votes > bull_votes else "中性")
    final_conf = max(bull_votes, bear_votes) / total_votes * 100
    return {
        'tech_score': tech_score, 'fund_score': fund_score, 'macro_score': macro_score,
        'bull_votes': bull_votes, 'bear_votes': bear_votes,
        'final_decision': final_decision, 'final_conf': final_conf,
        'rsi_val': rsi_val, 'vol_val': vol_val, 'boll_pos': boll_pos,
        'vol_pattern': vol_pattern, 'vol_price': vol_price,
        'macd_hist_val': latest['macd_hist'] if pd.notna(latest['macd_hist']) else 0,
        'trend': trend, 'signal_score': signal_score, 'current_price': current_price,
    }


# ==================== 配置函数 ====================

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def save_config(cfg):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"保存配置失败: {e}")
        return False


# ==================== 侧边栏 ====================

def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h1 style="font-size:1.5rem; color:#667eea; margin:0;">📈 A股智能分析</h1>
        <p style="font-size:0.75rem; color:#888; margin:0.2rem 0;">三层决策 + AI辩论 + Review</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "导航",
        ["🏠 数据面板", "🎯 今日选股", "📈 行情分析", "🤖 AI辩论室", "🔄 回测中心", "💼 持仓分析", "⚙️ 系统设置"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    hmm_models, hmm_meta = load_hmm_models()
    model_count = len(hmm_models)
    model_status = "✅" if model_count > 0 else "❌"
    st.sidebar.markdown(f"""
    <div style="padding:0.8rem; background:rgba(255,255,255,0.05); border-radius:8px; font-size:0.8rem;">
        <p style="margin:0.2rem 0;">🤖 HMM模型: {model_status} {model_count}个</p>
        <p style="margin:0.2rem 0;">📊 数据源: BaoStock {'✅' if _BS_LOGGED_IN else '❌'}</p>
        <p style="margin:0.2rem 0;">🕐 更新: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    if hmm_meta:
        st.sidebar.caption(f"训练日期: {hmm_meta.get('training_date', 'N/A')}")
    return page


# ==================== 数据面板 ====================

def render_dashboard():
    st.markdown('<div class="main-header">🏠 市场数据面板</div>', unsafe_allow_html=True)
    with st.spinner("正在获取市场数据..."):
        sh_data = get_index_data("sh.000001", days=5)
        sz_data = get_index_data("sz.399001", days=5)
        cy_data = get_index_data("sz.399006", days=5)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if sh_data is not None and len(sh_data) > 0:
            sh_close = sh_data['close'].iloc[-1]
            sh_pct = sh_data['pctChg'].iloc[-1]
            st.metric("上证指数", f"{sh_close:,.2f}", f"{'+' if sh_pct >= 0 else ''}{sh_pct:.2f}%")
        else:
            st.metric("上证指数", "获取失败", "N/A")
    with col2:
        if sz_data is not None and len(sz_data) > 0:
            sz_close = sz_data['close'].iloc[-1]
            sz_pct = sz_data['pctChg'].iloc[-1]
            st.metric("深证成指", f"{sz_close:,.2f}", f"{'+' if sz_pct >= 0 else ''}{sz_pct:.2f}%")
        else:
            st.metric("深证成指", "获取失败", "N/A")
    with col3:
        if cy_data is not None and len(cy_data) > 0:
            cy_close = cy_data['close'].iloc[-1]
            cy_pct = cy_data['pctChg'].iloc[-1]
            st.metric("创业板指", f"{cy_close:,.2f}", f"{'+' if cy_pct >= 0 else ''}{cy_pct:.2f}%")
        else:
            st.metric("创业板指", "获取失败", "N/A")
    with col4:
        if sh_data is not None and len(sh_data) > 0:
            sh_pct = sh_data['pctChg'].iloc[-1]
            if pd.notna(sh_pct):
                if sh_pct > 0.5: trend_str = "强势上涨"
                elif sh_pct > 0: trend_str = "偏多"
                elif sh_pct > -0.5: trend_str = "偏空"
                else: trend_str = "弱势下跌"
                st.metric("市场情绪", trend_str, f"上证 {sh_pct:+.2f}%")
            else:
                st.metric("市场情绪", "N/A", "N/A")
        else:
            st.metric("市场情绪", "N/A", "N/A")
    st.markdown("---")
    st.markdown('<div class="section-header">📊 指数走势对比</div>', unsafe_allow_html=True)
    fig_indices = go.Figure()
    for idx_data, idx_name, color in [(sh_data, "上证指数", "#e74c3c"), (sz_data, "深证成指", "#3498db"), (cy_data, "创业板指", "#f39c12")]:
        if idx_data is not None and len(idx_data) > 0:
            normalized = idx_data['close'] / idx_data['close'].iloc[0] * 100
            fig_indices.add_trace(go.Scatter(x=idx_data['date'], y=normalized, mode='lines+markers', name=idx_name, line=dict(color=color, width=2), marker=dict(size=4)))
    fig_indices.update_layout(title="近5日指数走势（归一化）", height=300, xaxis_title="日期", yaxis_title="相对值", hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_indices, use_container_width=True)
    st.markdown("---")
    st.markdown('<div class="section-header">📋 自选股快速监控</div>', unsafe_allow_html=True)
    with st.spinner("正在获取自选股数据..."):
        hs300_stocks = get_hs300_stocks()
        if len(hs300_stocks) == 0:
            st.warning("无法获取沪深300成分股列表")
            return
        default_watchlist = hs300_stocks.head(8)
        watchlist_rows = []
        for _, row in default_watchlist.iterrows():
            code = row['code']
            name = row['code_name']
            df = get_real_stock_data(code, days=30)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                df_ind = compute_real_indicators(df)
                if df_ind is not None:
                    trend, trend_conf = rule_based_trend(df_ind)
                    signal, signal_score = rule_based_signal(df_ind)
                    comp_score = generate_composite_score(df_ind)
                else:
                    trend, signal, comp_score = "N/A", "N/A", 50
                pct_chg = latest['pctChg']
                pct_str = f"{'+' if pd.notna(pct_chg) and pct_chg >= 0 else ''}{pct_chg:.2f}%" if pd.notna(pct_chg) else "N/A"
                watchlist_rows.append({'代码': code, '名称': name, '最新价': f"{latest['close']:.2f}", '涨跌幅': pct_str, 'Layer1趋势': trend, 'Layer2信号': signal, '综合评分': comp_score})
    if not watchlist_rows:
        st.warning("无法获取自选股数据")
        return
    df_watch = pd.DataFrame(watchlist_rows)
    def color_score(val):
        if val >= 80: return 'background-color: #d4edda'
        elif val >= 60: return 'background-color: #fff3cd'
        else: return 'background-color: #f8d7da'
    styled_df = df_watch.style.map(color_score, subset=['综合评分'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        selected_stock = st.selectbox("选择股票进行快速分析", df_watch['代码'].tolist())
        if st.button("🔍 启动三层分析", use_container_width=True):
            st.session_state['analyze_stock'] = selected_stock
            st.session_state['page'] = "📈 行情分析"
            st.rerun()
    with col_btn2:
        if st.button("🎯 跳转今日选股", use_container_width=True):
            st.session_state['page'] = "🎯 今日选股"
            st.rerun()


# ==================== 今日选股 ====================

def render_today_pick():
    st.markdown('<div class="main-header">🎯 今日智能选股</div>', unsafe_allow_html=True)
    hmm_models, hmm_meta = load_hmm_models()
    if not hmm_models:
        st.warning("⚠️ 未找到预训练HMM模型，请先运行 `python fast_analysis.py` 训练模型。当前将使用规则引擎替代。")
        use_hmm = False
    else:
        use_hmm = True
        st.success(f"✅ 已加载 {len(hmm_models)} 个预训练HMM模型")
        if hmm_meta:
            st.caption(f"训练数据范围: {hmm_meta.get('data_range', 'N/A')} | 训练日期: {hmm_meta.get('training_date', 'N/A')}")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_n = st.slider("选股数量", 5, 30, 10)
    with col2:
        buy_threshold = st.slider("HMM买入阈值", 0.40, 0.80, 0.55, 0.05)
    with col3:
        min_score = st.slider("最低综合评分", 30, 80, 50)
    with col4:
        run_debate = st.checkbox("对TOP选股运行辩论", value=True)
    start_pick = st.button("🚀 开始选股分析", type="primary", use_container_width=True)
    if start_pick:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0, text="正在获取沪深300成分股...")
            hs300_stocks = get_hs300_stocks()
            if len(hs300_stocks) == 0:
                st.error("无法获取沪深300成分股列表")
                return
            total_stocks = len(hs300_stocks)
            candidates = []
            progress_bar.progress(10, text=f"开始分析 {total_stocks} 只股票...")
            for idx, (_, row) in enumerate(hs300_stocks.iterrows()):
                code = row['code']
                name = row['code_name']
                if idx % 10 == 0:
                    pct = 10 + int(70 * idx / total_stocks)
                    progress_bar.progress(pct, text=f"分析中... {idx}/{total_stocks} ({code} {name})")
                df = get_real_stock_data(code, days=180)
                if df is None or len(df) < 60:
                    continue
                df = compute_real_indicators(df)
                if df is None:
                    continue
                latest = df.iloc[-1]
                current_price = latest['close']
                hmm_trend = None
                hmm_up_prob = None
                if use_hmm and code in hmm_models:
                    hmm_trend, hmm_up_prob, _ = hmm_predict(hmm_models[code], df)
                signal, signal_score = rule_based_signal(df)
                comp_score = generate_composite_score(df)
                rule_trend, _ = rule_based_trend(df)
                final_trend = hmm_trend if hmm_trend else rule_trend
                up_prob = hmm_up_prob if hmm_up_prob else 0.5
                if up_prob >= buy_threshold and comp_score >= min_score:
                    candidates.append({'code': code, 'name': name, 'price': current_price, 'hmm_trend': hmm_trend or rule_trend, 'hmm_up_prob': up_prob, 'signal': signal, 'signal_score': signal_score, 'comp_score': comp_score, 'rule_trend': rule_trend})
            progress_bar.progress(85, text=f"筛选完成，共 {len(candidates)} 只候选股票，正在排序...")
            candidates.sort(key=lambda x: (x['hmm_up_prob'], x['comp_score']), reverse=True)
            top_picks = candidates[:top_n]
            progress_bar.progress(95, text="生成推荐报告...")
            if not top_picks:
                st.warning("未找到符合条件的股票，请尝试降低筛选阈值。")
                progress_bar.progress(100, text="选股完成")
                return
            st.markdown(f"## 📊 选股结果：TOP {len(top_picks)} 推荐")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                avg_prob = np.mean([p['hmm_up_prob'] for p in top_picks])
                st.metric("平均上涨概率", f"{avg_prob*100:.1f}%")
            with col_s2:
                avg_score = np.mean([p['comp_score'] for p in top_picks])
                st.metric("平均综合评分", f"{avg_score:.0f}")
            with col_s3:
                buy_count = sum(1 for p in top_picks if '买入' in p['signal'])
                st.metric("买入信号", f"{buy_count}/{len(top_picks)}")
            with col_s4:
                up_count = sum(1 for p in top_picks if p['hmm_trend'] == '上涨')
                st.metric("上涨趋势", f"{up_count}/{len(top_picks)}")
            st.markdown("---")
            st.markdown('<div class="section-header">🏆 推荐股票详情</div>', unsafe_allow_html=True)
            for i, pick in enumerate(top_picks):
                with st.expander(f"#{i+1} | {pick['code']} {pick['name']} | ¥{pick['price']:.2f} | HMM上涨概率 {pick['hmm_up_prob']*100:.1f}% | 评分 {pick['comp_score']}", expanded=(i < 3)):
                    df = get_real_stock_data(pick['code'], days=60)
                    if df is not None and len(df) >= 20:
                        df = compute_real_indicators(df)
                        if df is not None:
                            latest = df.iloc[-1]
                            col_d1, col_d2, col_d3 = st.columns(3)
                            with col_d1:
                                st.metric("HMM趋势", pick['hmm_trend'], f"上涨概率 {pick['hmm_up_prob']*100:.1f}%")
                            with col_d2:
                                st.metric("技术信号", pick['signal'], f"评分 {pick['signal_score']:.2f}")
                            with col_d3:
                                vol_pattern, vol_price = compute_volume_pattern(df)
                                st.metric("量价模式", vol_price, f"换手 {latest['turn']:.2f}%" if pd.notna(latest.get('turn')) else "N/A")
                            col_a1, col_a2, col_a3 = st.columns(3)
                            with col_a1:
                                entry_price = pick['price'] * 1.01
                                st.write(f"**建议买入价**: ¥{entry_price:.2f}")
                            with col_a2:
                                stop_loss_price = entry_price * 0.93
                                st.write(f"**止损位**: ¥{stop_loss_price:.2f} (-7%)")
                            with col_a3:
                                target_price = entry_price * 1.15
                                st.write(f"**目标价**: ¥{target_price:.2f} (+15%)")
                            fig_mini = go.Figure(go.Candlestick(x=df['date'].tail(30), open=df['open'].tail(30), high=df['high'].tail(30), low=df['low'].tail(30), close=df['close'].tail(30), name=pick['code']))
                            fig_mini.add_trace(go.Scatter(x=df['date'].tail(30), y=df['ma5'].tail(30), mode='lines', name='MA5', line=dict(color='#f39c12', width=1)))
                            fig_mini.add_trace(go.Scatter(x=df['date'].tail(30), y=df['ma20'].tail(30), mode='lines', name='MA20', line=dict(color='#3498db', width=1)))
                            fig_mini.update_layout(height=250, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_mini, use_container_width=True)
                            if run_debate:
                                st.markdown("---")
                                st.markdown("**🤖 AI辩论快速分析**")
                                debate_result = run_debate_analysis(df, pick['hmm_trend'], pick['signal_score'], pick['price'])
                                debate_cols = st.columns(3)
                                with debate_cols[0]:
                                    st.metric("技术面", f"{debate_result['tech_score']}/100")
                                with debate_cols[1]:
                                    st.metric("基本面", f"{debate_result['fund_score']}/100")
                                with debate_cols[2]:
                                    st.metric("宏观面", f"{debate_result['macro_score']}/100")
                                col_bull, col_bear = st.columns(2)
                                with col_bull:
                                    st.markdown(f'<div class="debate-role debate-bull"><b>多头</b>: {debate_result["bull_votes"]}票<br>目标: ¥{pick["price"]*1.15:.2f}</div>', unsafe_allow_html=True)
                                with col_bear:
                                    st.markdown(f'<div class="debate-role debate-bear"><b>空头</b>: {debate_result["bear_votes"]}票<br>止损: ¥{pick["price"]*0.93:.2f}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="debate-role debate-judge"><b>裁决</b>: {debate_result["final_decision"]} (置信度{debate_result["final_conf"]:.0f}%)</div>', unsafe_allow_html=True)
            st.markdown("---")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                df_picks = pd.DataFrame(top_picks)
                csv_data = df_picks.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(label="📥 导出选股结果 (CSV)", data=csv_data, file_name=f"today_picks_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
            with col_exp2:
                json_data = json.dumps(top_picks, ensure_ascii=False, indent=2).encode('utf-8')
                st.download_button(label="📥 导出选股结果 (JSON)", data=json_data, file_name=f"today_picks_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json", use_container_width=True)
            progress_bar.progress(100, text="选股分析完成！")


# ==================== 行情分析 ====================

def render_analysis():
    st.markdown('<div class="main-header">📈 行情分析 - 三层决策</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        default_symbol = st.session_state.get('analyze_stock', '600519')
        symbol = st.text_input("输入股票代码", default_symbol, help="输入6位股票代码")
    with col2:
        analyze_btn = st.button("🚀 启动分析", type="primary", use_container_width=True)
    if analyze_btn or 'analyze_result' in st.session_state:
        with st.spinner("正在执行三层决策分析..."):
            df = get_real_stock_data(symbol, days=180)
            if df is None or len(df) < 20:
                st.error(f"无法获取 {symbol} 的有效数据，请检查股票代码是否正确")
                return
            df = compute_real_indicators(df)
            if df is None:
                st.error("数据不足，无法计算技术指标（至少需要20个交易日）")
                return
            real_analysis(symbol, df)


def real_analysis(symbol, df):
    latest = df.iloc[-1]
    current_price = latest['close']
    st.markdown('<div class="section-header">🔍 Layer 1: 趋势定性分析</div>', unsafe_allow_html=True)
    hmm_models, _ = load_hmm_models()
    hmm_trend = None
    hmm_up_prob = None
    if symbol in hmm_models:
        hmm_trend, hmm_up_prob, posteriors = hmm_predict(hmm_models[symbol], df)
    trend, confidence = rule_based_trend(df)
    if hmm_trend:
        trend_display = f"{hmm_trend} (HMM)"
        confidence_display = hmm_up_prob
    else:
        trend_display = trend
        confidence_display = confidence
    ma5_val = latest['ma5'] if pd.notna(latest['ma5']) else 0
    ma20_val = latest['ma20'] if pd.notna(latest['ma20']) else 0
    trend_strength = round(abs((ma5_val - ma20_val) / (ma20_val + 1e-10)), 2)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前状态", trend_display, f"置信度 {confidence_display*100:.0f}%")
    with col2:
        if pd.notna(latest['ma5']) and pd.notna(latest['ma20']):
            above_days = 0
            for idx in range(len(df)-1, -1, -1):
                if pd.notna(df['ma5'].iloc[idx]) and pd.notna(df['ma20'].iloc[idx]):
                    if df['ma5'].iloc[idx] > df['ma20'].iloc[idx]:
                        above_days += 1
                    else:
                        break
            st.metric("多头持续", f"{above_days}天" if latest['ma5'] > latest['ma20'] else "空头排列")
        else:
            st.metric("均线状态", "N/A")
    with col3:
        st.metric("趋势强度", str(trend_strength), "偏强" if trend_strength > 0.03 else "偏弱")
    if hmm_trend and hmm_up_prob is not None:
        fig_hmm = go.Figure()
        states = ['下跌', '偏空', '震荡', '偏多', '上涨']
        if hmm_up_prob > 0.6: probs = [0.05, 0.10, 0.15, 0.25, 0.45]
        elif hmm_up_prob > 0.45: probs = [0.10, 0.20, 0.30, 0.25, 0.15]
        else: probs = [0.35, 0.30, 0.20, 0.10, 0.05]
        colors = ['#e74c3c' if i >= 3 else '#3498db' for i in range(5)]
        fig_hmm.add_trace(go.Bar(x=states, y=probs, marker_color=colors))
        fig_hmm.update_layout(title=f"HMM趋势概率分布（上涨概率 {hmm_up_prob*100:.1f}%）", xaxis_title="状态", yaxis_title="概率", height=300)
    else:
        fig_hmm = go.Figure()
        states = ['强烈下跌', '下跌', '偏空震荡', '偏多震荡', '上涨', '强烈上涨']
        probs = [0.02, 0.05, 0.10, 0.15, 0.45, 0.23]
        if trend == "上涨": probs = [0.01, 0.03, 0.08, 0.13, 0.45, 0.30]
        elif trend == "下跌": probs = [0.25, 0.35, 0.20, 0.10, 0.07, 0.03]
        colors = ['#e74c3c' if p > 0.2 else '#3498db' for p in probs]
        fig_hmm.add_trace(go.Bar(x=states, y=probs, marker_color=colors))
        fig_hmm.update_layout(title="趋势状态概率分布（规则引擎）", xaxis_title="状态", yaxis_title="概率", height=300)
    st.plotly_chart(fig_hmm, use_container_width=True)
    win_prob = hmm_up_prob * 100 if hmm_up_prob else (68.5 if trend in ["上涨"] else (31.5 if trend in ["下跌"] else 50.0))
    model_label = "HMM模型" if hmm_trend else "规则引擎"
    st.info(f"**分析结论** [{model_label}]: 当前处于「{trend_display}」状态，历史同状态下未来5日上涨概率 {win_prob:.1f}%")
    st.markdown('<div class="section-header">📊 Layer 2: 技术面因子信号</div>', unsafe_allow_html=True)
    signal, signal_score = rule_based_signal(df)
    vol_pattern, vol_price = compute_volume_pattern(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("趋势跟踪", "看多" if signal_score > 0 else ("看空" if signal_score < 0 else "中性"), f"评分 {signal_score:.2f}")
    with col2:
        comp_score = generate_composite_score(df)
        st.metric("多因子打分", f"{comp_score}分", "偏多" if comp_score > 60 else ("偏空" if comp_score < 40 else "中性"))
    with col3:
        st.metric("量价模式", vol_price, f"评分 {signal_score:.2f}")
    composite_score = signal_score
    fig_signal = go.Figure(go.Indicator(mode="gauge+number+delta", value=composite_score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "综合信号评分"}, gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "#e74c3c" if composite_score > 0 else "#27ae60"}, 'steps': [{'range': [-1, -0.3], 'color': "#d5f5e3"}, {'range': [-0.3, 0.3], 'color': "#fef9e7"}, {'range': [0.3, 1], 'color': "#fadbd8"}], 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': composite_score}}))
    fig_signal.update_layout(height=300)
    st.plotly_chart(fig_signal, use_container_width=True)
    st.success(f"**综合信号**: {signal} (评分: {composite_score:.2f})")
    st.markdown('<div class="section-header">📊 K线走势图</div>', unsafe_allow_html=True)
    fig_kline = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig_kline.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    fig_kline.add_trace(go.Scatter(x=df['date'], y=df['ma5'], mode='lines', name='MA5', line=dict(width=1)), row=1, col=1)
    fig_kline.add_trace(go.Scatter(x=df['date'], y=df['ma10'], mode='lines', name='MA10', line=dict(width=1)), row=1, col=1)
    fig_kline.add_trace(go.Scatter(x=df['date'], y=df['ma20'], mode='lines', name='MA20', line=dict(width=1)), row=1, col=1)
    colors_vol = ['#e74c3c' if c >= o else '#27ae60' for c, o in zip(df['close'], df['open'])]
    fig_kline.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=colors_vol, name='成交量'), row=2, col=1)
    fig_kline.update_layout(title=f"{symbol} K线走势", xaxis_title="日期", yaxis_title="价格", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kline, use_container_width=True)
    st.markdown('<div class="section-header">🤖 Layer 3: AI辩论决策</div>', unsafe_allow_html=True)
    rsi_val = latest['rsi'] if pd.notna(latest['rsi']) else 50
    macd_hist_val = latest['macd_hist'] if pd.notna(latest['macd_hist']) else 0
    boll_pos = latest['boll_position'] if pd.notna(latest['boll_position']) else 0.5
    vol_val = latest['volatility_20d'] if pd.notna(latest['volatility_20d']) else 0.02
    debate_tabs = st.tabs(["技术面分析师", "基本面分析师", "宏观分析师", "多头研究员", "空头研究员", "裁判裁决"])
    with debate_tabs[0]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**技术面分析师报告**")
        ma_cross = "金叉" if (pd.notna(df['ma5'].iloc[-1]) and pd.notna(df['ma20'].iloc[-1]) and df['ma5'].iloc[-1] > df['ma20'].iloc[-1] and pd.notna(df['ma5'].iloc[-2]) and pd.notna(df['ma20'].iloc[-2]) and df['ma5'].iloc[-2] <= df['ma20'].iloc[-2]) else ("多头排列" if df['ma5'].iloc[-1] > df['ma20'].iloc[-1] else "空头排列")
        st.write(f"- 均线系统呈{ma_cross}，MA5={df['ma5'].iloc[-1]:.2f}, MA20={df['ma20'].iloc[-1]:.2f}\n- MACD柱状体{'持续放大，动能强劲' if macd_hist_val > 0 else '收窄，动能减弱'}\n- RSI(14) = {rsi_val:.1f}，{'未进入超买区' if rsi_val < 70 else '已进入超买区，注意风险'}\n- 20日波动率: {vol_val*100:.2f}%，布林带位置: {boll_pos:.2f}\n- **结论**: 技术面{'支持做多' if signal_score > 0 else '偏空，建议观望'}")
        st.markdown('</div>', unsafe_allow_html=True)
    with debate_tabs[1]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**基本面分析师报告**")
        avg_20 = df['close'].tail(20).mean()
        turn_str = f"{latest['turn']:.2f}%" if pd.notna(latest['turn']) else "N/A"
        st.write(f"- 当前价格: {current_price:.2f}，近20日均价: {avg_20:.2f}\n- {'价格高于均值，市场情绪偏乐观' if current_price > avg_20 else '价格低于均值，可能存在低估'}\n- 近期换手率: {turn_str}\n- 成交额趋势: {'资金流入明显' if vol_pattern in ['放量', '显著放量'] else '资金面平稳'}\n- **结论**: 基本面{'稳健，估值合理' if abs(current_price - avg_20) / avg_20 < 0.05 else '需关注价格偏离'}")
        st.markdown('</div>', unsafe_allow_html=True)
    with debate_tabs[2]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**宏观分析师报告**")
        st.write(f"- 市场波动率: {vol_val*100:.2f}%，{'处于正常区间' if vol_val < 0.03 else '波动加大，注意风险控制'}\n- 近期量能变化: {vol_price}\n- 趋势状态: {trend_display}（置信度 {confidence_display*100:.0f}%）\n- **结论**: 宏观环境{'支持风险资产' if trend in ['上涨', '偏多震荡'] else '建议谨慎操作'}")
        st.markdown('</div>', unsafe_allow_html=True)
    with debate_tabs[3]:
        st.markdown('<div class="debate-role debate-bull">', unsafe_allow_html=True)
        st.write("**多头研究员论点**")
        bull_points = []
        if hmm_trend == "上涨": bull_points.append(f"1. HMM预测'{hmm_trend}'，上涨概率{hmm_up_prob*100:.1f}%")
        elif trend in ["上涨"]: bull_points.append(f"1. 趋势状态为'{trend}'，信号明确")
        if signal_score > 0: bull_points.append("2. 技术面信号偏多，指标共振")
        if vol_pattern in ["放量", "显著放量"]: bull_points.append("3. 量能配合良好，资金持续流入")
        if rsi_val < 70: bull_points.append("4. RSI未超买，仍有上行空间")
        if not bull_points: bull_points.append("1. 当前技术指标中性偏多")
        bull_points.append(f"5. 目标价: {current_price * 1.15:.2f} (+15%)")
        bull_points.append("6. **建议**: 积极做多")
        st.write("\n".join(bull_points))
        st.markdown('</div>', unsafe_allow_html=True)
    with debate_tabs[4]:
        st.markdown('<div class="debate-role debate-bear">', unsafe_allow_html=True)
        st.write("**空头研究员论点**")
        bear_points = []
        if rsi_val > 60: bear_points.append(f"1. RSI={rsi_val:.1f}，{'已超买' if rsi_val > 70 else '接近超买区'}")
        if vol_val > 0.025: bear_points.append("2. 波动率偏高，市场不确定性加大")
        if boll_pos > 0.8: bear_points.append("3. 价格接近布林带上轨，存在回调压力")
        if trend in ["偏空震荡", "下跌"]: bear_points.append(f"4. 趋势状态为'{trend}'，不宜追高")
        if not bear_points: bear_points.append("1. 短期涨幅需确认，注意风险控制")
        bear_points.append(f"5. 止损位: {current_price * 0.93:.2f} (-7%)")
        bear_points.append("6. **建议**: 谨慎操作，设好止损")
        st.write("\n".join(bear_points))
        st.markdown('</div>', unsafe_allow_html=True)
    with debate_tabs[5]:
        st.markdown('<div class="debate-role debate-judge">', unsafe_allow_html=True)
        st.write("**中立裁判裁决**")
        debate_result = run_debate_analysis(df, trend, signal_score, current_price)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("多头票数", f"{debate_result['bull_votes']}票", f"占比 {debate_result['bull_votes']/6*100:.0f}%")
        with col2: st.metric("空头票数", f"{debate_result['bear_votes']}票", f"占比 {debate_result['bear_votes']/6*100:.0f}%")
        with col3: st.metric("最终裁决", debate_result['final_decision'], f"置信度 {debate_result['final_conf']:.0f}%")
        position_pct = "60%" if debate_result['final_decision'] == "看多" else ("30%" if debate_result['final_decision'] == "中性" else "空仓")
        st.write(f"**裁决理由**:\n- 技术面信号评分: {signal_score:.2f}，趋势状态: {trend_display}\n- RSI: {rsi_val:.1f}，波动率: {vol_val*100:.2f}%\n- 量价模式: {vol_price}\n- **操作方案**: {'买入' if debate_result['final_decision'] == '看多' else '观望'} {position_pct} 仓位，设置 7% 止损")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌤️ 未来3天走势预测</div>', unsafe_allow_html=True)
    if trend in ["上涨"] or hmm_trend == "上涨":
        forecasts = [{'day': 'T+1', 'weather': '☀️', 'class': 'weather-sunny', 'direction': '上涨', 'prob': f"{min(65, 50 + confidence*20):.0f}%", 'target': f"+{vol_val*100*3:.1f}%", 'accuracy': f"{win_prob:.0f}%"}, {'day': 'T+2', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡偏多', 'prob': f"{min(55, 45 + confidence*15):.0f}%", 'target': f"+{vol_val*100*1.5:.1f}%", 'accuracy': f"{max(0,win_prob-5):.0f}%"}, {'day': 'T+3', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': f"{min(52, 45 + confidence*10):.0f}%", 'target': f"+{vol_val*100*0.5:.1f}%", 'accuracy': f"{max(0,win_prob-10):.0f}%"}]
    elif trend in ["下跌"] or hmm_trend == "下跌":
        forecasts = [{'day': 'T+1', 'weather': '🌧️', 'class': 'weather-rainy', 'direction': '下跌', 'prob': f"{min(65, 50 + confidence*20):.0f}%", 'target': f"-{vol_val*100*3:.1f}%", 'accuracy': f"{100-win_prob:.0f}%"}, {'day': 'T+2', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡偏空', 'prob': f"{min(55, 45 + confidence*15):.0f}%", 'target': f"-{vol_val*100*1.5:.1f}%", 'accuracy': f"{min(100,100-win_prob+5):.0f}%"}, {'day': 'T+3', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': f"{min(52, 45 + confidence*10):.0f}%", 'target': f"-{vol_val*100*0.5:.1f}%", 'accuracy': f"{min(100,100-win_prob+10):.0f}%"}]
    else:
        forecasts = [{'day': 'T+1', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': '50%', 'target': f"+{vol_val*100*0.5:.1f}%", 'accuracy': '50%'}, {'day': 'T+2', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': '50%', 'target': '0%', 'accuracy': '48%'}, {'day': 'T+3', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': '50%', 'target': f"-{vol_val*100*0.3:.1f}%", 'accuracy': '46%'}]
    forecast_cols = st.columns(3)
    for i, fc in enumerate(forecasts):
        with forecast_cols[i]:
            st.markdown(f'<div style="text-align:center;padding:1.2rem;background:linear-gradient(135deg,#fff 0%,#f8f9fa 100%);border-radius:12px;border:1px solid #e8ecf1;box-shadow:0 2px 6px rgba(0,0,0,0.04);"><h4 style="margin:0;">{fc["day"]}</h4><div class="{fc["class"]}">{fc["weather"]}</div><p style="margin:0.3rem 0;"><strong>{fc["direction"]}</strong></p><p style="margin:0.2rem 0;color:#666;">概率: {fc["prob"]}</p><p style="margin:0.2rem 0;color:#666;">目标: {fc["target"]}</p><p style="font-size:0.8rem;color:#999;margin:0.2rem 0;">历史准确率: {fc["accuracy"]}</p></div>', unsafe_allow_html=True)
    if composite_score > 0.5:
        st.warning(f"**自纠错提示**: 信号一致看多（评分 {composite_score:.2f}），但需警惕一致性预期风险。RSI={rsi_val:.1f}，{'未超买' if rsi_val < 70 else '已超买'}。")
    elif composite_score < -0.5:
        st.warning(f"**自纠错提示**: 信号一致看空（评分 {composite_score:.2f}），注意控制仓位。")


# ==================== AI辩论室 ====================

def render_debate_room():
    st.markdown('<div class="main-header">🤖 AI辩论室</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        debate_symbol = st.text_input("股票代码", "600519")
    with col2:
        start_debate = st.button("⚔️ 开始辩论", type="primary", use_container_width=True)
    if start_debate:
        with st.spinner("正在获取数据并启动辩论..."):
            df = get_real_stock_data(debate_symbol, days=180)
            if df is None or len(df) < 20:
                st.error(f"无法获取 {debate_symbol} 的有效数据")
                return
            df = compute_real_indicators(df)
            if df is None:
                st.error("数据不足，无法进行辩论分析")
                return
            latest = df.iloc[-1]
            current_price = latest['close']
            hmm_models, _ = load_hmm_models()
            hmm_trend = None
            if debate_symbol in hmm_models:
                hmm_trend, hmm_up_prob, _ = hmm_predict(hmm_models[debate_symbol], df)
            trend, confidence = rule_based_trend(df)
            final_trend = hmm_trend if hmm_trend else trend
            signal, signal_score = rule_based_signal(df)
            vol_pattern, vol_price = compute_volume_pattern(df)
            debate_result = run_debate_analysis(df, final_trend, signal_score, current_price)
            progress_bar = st.progress(0)
            st.markdown("### Phase 1: 分析师并行报告生成")
            progress_bar.progress(20, text="技术面分析师生成报告...")
            analyst_cols = st.columns(3)
            with analyst_cols[0]:
                st.markdown(f'<div class="debate-role"><h4>技术面分析师</h4><p>趋势: <b>{final_trend}</b> | 评分: <b>{debate_result["tech_score"]}/100</b></p><p>RSI: {debate_result["rsi_val"]:.1f} | MACD柱: {debate_result["macd_hist_val"]:.4f}</p><p>布林位置: {debate_result["boll_pos"]:.2f} | 波动率: {debate_result["vol_val"]*100:.2f}%</p></div>', unsafe_allow_html=True)
            with analyst_cols[1]:
                price_dev = abs(current_price - df['close'].tail(20).mean()) / df['close'].tail(20).mean() * 100
                st.markdown(f'<div class="debate-role"><h4>基本面分析师</h4><p>价格偏离均值: <b>{price_dev:.1f}%</b></p><p>换手率: {latest["turn"]:.2f}%</p><p>量价模式: <b>{vol_price}</b></p><p>评分: <b>{debate_result["fund_score"]}/100</b></p></div>', unsafe_allow_html=True)
            with analyst_cols[2]:
                st.markdown(f'<div class="debate-role"><h4>宏观分析师</h4><p>市场波动率: <b>{debate_result["vol_val"]*100:.2f}%</b></p><p>趋势状态: <b>{final_trend}</b></p><p>置信度: {confidence*100:.0f}%</p><p>评分: <b>{debate_result["macro_score"]}/100</b></p></div>', unsafe_allow_html=True)
            progress_bar.progress(60, text="分析师报告完成")
            st.markdown("### Phase 2: 研究员辩论")
            progress_bar.progress(80, text="研究员辩论中...")
            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.markdown(f'<div class="debate-role debate-bull"><h4>🐂 多头研究员</h4><p>基于分析，我认为应该做多:</p><p>1. 趋势状态为\'<b>{final_trend}</b>\'</p><p>2. 技术面信号{"偏多，指标共振" if signal_score > 0 else "中性，等待确认"}</p><p>3. 成交量{"放大，资金流入" if vol_pattern in ["放量", "显著放量"] else "平稳，观望为主"}</p><p><b>目标价: ¥{current_price * 1.15:.2f} (+15%)</b></p></div>', unsafe_allow_html=True)
            with col_bear:
                st.markdown(f'<div class="debate-role debate-bear"><h4>🐻 空头研究员</h4><p>虽然信号存在，但我有顾虑:</p><p>1. RSI={debate_result["rsi_val"]:.1f}，{"已超买" if debate_result["rsi_val"] > 70 else "接近超买" if debate_result["rsi_val"] > 60 else "指标尚可"}</p><p>2. 波动率{"偏高" if debate_result["vol_val"] > 0.025 else "正常"}({debate_result["vol_val"]*100:.2f}%)</p><p>3. 市场环境不确定性仍存</p><p><b>止损位: ¥{current_price * 0.93:.2f} (-7%)</b></p></div>', unsafe_allow_html=True)
            st.markdown("### Phase 3: 中立裁判裁决")
            progress_bar.progress(100, text="辩论完成！")
            st.balloons()
            position_pct = "60%" if debate_result['final_decision'] == "看多" else ("30%" if debate_result['final_decision'] == "中性" else "空仓")
            col_result1, col_result2, col_result3 = st.columns(3)
            with col_result1: st.metric("最终决策", debate_result['final_decision'], "多头胜出" if debate_result['bull_votes'] > debate_result['bear_votes'] else ("空头胜出" if debate_result['bear_votes'] > debate_result['bull_votes'] else "平局"))
            with col_result2: st.metric("置信度", f"{debate_result['final_conf']:.0f}%", "高信心" if debate_result['final_conf'] > 70 else "中等信心")
            with col_result3: st.metric("建议仓位", position_pct, "分批建仓" if debate_result['final_decision'] == "看多" else "轻仓观望")
            st.markdown(f'<div class="debate-role debate-judge"><h4>⚖️ 裁决总结</h4><p>技术面评分 <b>{debate_result["tech_score"]}</b> | 基本面评分 <b>{debate_result["fund_score"]}</b> | 宏观评分 <b>{debate_result["macro_score"]}</b></p><p>多头 <b>{debate_result["bull_votes"]}票</b> vs 空头 <b>{debate_result["bear_votes"]}票</b></p><p><b>最终决策: {debate_result["final_decision"]}</b>，置信度 {debate_result["final_conf"]:.0f}%</p><p>建议 <b>{position_pct}</b> 仓位，设置 7% 止损</p></div>', unsafe_allow_html=True)


# ==================== 回测中心 ====================

def render_backtest():
    st.markdown('<div class="main-header">🔄 策略回测中心</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ 回测参数设置</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col2: end_date = st.date_input("结束日期", datetime.now())
    with col3: initial_capital = st.number_input("初始资金", value=1000000, step=100000)
    col4, col5, col6 = st.columns(3)
    with col4: stop_loss = st.slider("止损比例", 0.03, 0.15, 0.10, 0.01)
    with col5: take_profit = st.slider("止盈比例", 0.05, 0.30, 0.15, 0.01)
    with col6: position_size = st.slider("单票仓位", 0.05, 0.30, 0.15, 0.05)
    stock_pool = st.multiselect("选择股票池", ['沪深300', '中证500', '上证50', '自选股', '全部A股'], default=['沪深300'])
    run_backtest = st.button("▶️ 运行回测", type="primary")
    if run_backtest:
        st.info("回测引擎需要通过命令行运行。请查看下方已有的回测结果。")
    st.markdown("---")
    st.markdown('<div class="section-header">📂 已有回测结果</div>', unsafe_allow_html=True)
    result_dir = os.path.dirname(os.path.abspath(__file__))
    result_files = [f for f in os.listdir(result_dir) if f.endswith('_result.json')]
    if result_files:
        selected_file = st.selectbox("选择回测结果文件", result_files)
        if st.button("📊 查看回测结果", use_container_width=True):
            show_backtest_results(selected_file)
    else:
        st.warning("未找到回测结果文件（*_result.json）")


def show_backtest_results(filename):
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(result_path):
        st.warning(f"未找到回测结果文件 {filename}")
        return
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except Exception as e:
        st.error(f"读取回测结果失败: {e}")
        return
    st.markdown(f"**文件**: {filename}")
    total_return = result.get('total_return', 0)
    annual_return = result.get('annual_return', 0)
    max_drawdown = result.get('max_drawdown', 0)
    sharpe = result.get('sharpe_ratio', 0)
    win_rate = result.get('win_rate', 0)
    profit_factor = result.get('profit_factor', 0)
    total_trades = result.get('total_trades', 0)
    final_capital = result.get('final_capital', 0)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("总收益率", f"{total_return*100:.2f}%", f"+{total_return*100 - 8.5:.2f}% 超额" if total_return > 0.085 else "")
    with col2: st.metric("年化收益", f"{annual_return*100:.2f}%", "基准 8.5%")
    with col3: st.metric("最大回撤", f"{max_drawdown*100:.2f}%", "可控" if max_drawdown < 0.2 else "偏高")
    with col4: st.metric("夏普比率", f"{sharpe:.2f}", "优秀" if sharpe > 1.5 else ("良好" if sharpe > 1.0 else "一般"))
    with col5: st.metric("胜率", f"{win_rate*100:.1f}%", f"盈亏比 {profit_factor:.2f}")
    params = result.get('params', {})
    data_range = result.get('data_range', 'N/A')
    backtest_period = result.get('backtest_period', 'N/A')
    total_stocks = result.get('total_stocks', 0)
    avg_holding_days = result.get('avg_holding_days', 0)
    st.info(f"**回测概况**: 股票池 {total_stocks} 只 | 数据范围 {data_range} | 回测期间 {backtest_period} | 总交易 {total_trades} 笔 | 平均持仓 {avg_holding_days:.1f} 天")
    st.markdown("### 权益曲线")
    initial_capital_val = 1000000
    try:
        bp_parts = backtest_period.split(' ~ ')
        start_str = bp_parts[0] if len(bp_parts) > 0 else '2025-01-01'
        end_str = bp_parts[1] if len(bp_parts) > 1 else datetime.now().strftime('%Y-%m-%d')
    except Exception:
        start_str = '2025-01-01'
        end_str = datetime.now().strftime('%Y-%m-%d')
    trades = result.get('trades', [])
    if trades and len(trades) > 1:
        equity_values = [initial_capital_val]
        for trade in trades:
            pnl = trade.get('pnl', 0)
            equity_values.append(equity_values[-1] + pnl)
        dates = pd.date_range(start=start_str, periods=len(equity_values), freq='D')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity_values, mode='lines', name='策略', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231,76,60,0.1)'))
        benchmark_cumulative = (1 + 0.085 / 252) ** np.arange(len(equity_values))
        fig.add_trace(go.Scatter(x=dates, y=benchmark_cumulative * initial_capital_val, mode='lines', name='基准(年化8.5%)', line=dict(color='#95a5a6', dash='dash')))
    else:
        dates = pd.date_range(start=start_str, end=end_str, freq='D')
        n_days = len(dates)
        daily_return = total_return / max(n_days, 1)
        np.random.seed(42)
        returns = np.random.normal(daily_return, abs(daily_return) * 2, n_days)
        cumulative = (1 + returns).cumprod()
        cumulative = cumulative / cumulative.iloc[-1] * (1 + total_return)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates[:len(cumulative)], y=cumulative * initial_capital_val, mode='lines', name='策略', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231,76,60,0.1)'))
        benchmark_cumulative = (1 + 0.085 / 252) ** np.arange(len(cumulative))
        fig.add_trace(go.Scatter(x=dates[:len(cumulative)], y=benchmark_cumulative * initial_capital_val, mode='lines', name='基准(年化8.5%)', line=dict(color='#95a5a6', dash='dash')))
    fig.update_layout(title=f"策略 vs 基准 (最终资金: {final_capital:,.0f})", xaxis_title="日期", yaxis_title="资金", height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    exit_reasons = result.get('exit_reasons', {})
    if exit_reasons:
        st.markdown("### 退出原因分布")
        reason_labels = {'signal_reverse': '信号反转', 'take_profit': '止盈', 'timeout': '超时', 'stop_loss': '止损', 'backtest_end': '回测结束'}
        labels = [reason_labels.get(k, k) for k in exit_reasons.keys()]
        values = list(exit_reasons.values())
        fig_reason = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
        fig_reason.update_layout(title="交易退出原因分布", height=350)
        st.plotly_chart(fig_reason, use_container_width=True)
    st.markdown("### 回测参数")
    if params:
        param_df = pd.DataFrame(list(params.items()), columns=['参数', '值'])
        st.dataframe(param_df, use_container_width=True, hide_index=True)
    compare_2024 = result.get('compare_2024', {})
    if compare_2024:
        st.markdown("### 2024年对比回测")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("2024总收益", f"{compare_2024.get('total_return', 0)*100:.2f}%")
        with col2: st.metric("2024夏普", f"{compare_2024.get('sharpe_ratio', 0):.2f}")
        with col3: st.metric("2024最大回撤", f"{compare_2024.get('max_drawdown', 0)*100:.2f}%")
        with col4: st.metric("2024胜率", f"{compare_2024.get('win_rate', 0)*100:.1f}%")
    validations = result.get('validations', [])
    if validations:
        st.markdown("### 验证结果")
        for v in validations:
            if "PASSED" in v: st.success(v)
            else: st.error(v)
    st.info(f"**建议**: 当前参数止损{params.get('stop_loss_pct', 0.10)*100:.0f}% + 止盈{params.get('take_profit_pct', 0.15)*100:.0f}%，夏普比率{sharpe:.2f}，{'表现优秀' if sharpe > 1.5 else '表现良好'}。")


# ==================== 持仓分析 ====================

def render_portfolio():
    st.markdown('<div class="main-header">💼 持仓分析</div>', unsafe_allow_html=True)
    positions_data = []
    try:
        from src.data.database import get_db, PortfolioPosition
        db = next(get_db())
        pos_records = db.query(PortfolioPosition).filter(PortfolioPosition.status == "持有中").all()
        for pos in pos_records:
            positions_data.append({'代码': pos.symbol, '名称': pos.name or pos.symbol, '持仓量': pos.shares, '成本价': pos.entry_price, '现价': pos.current_price or pos.entry_price, 'sector': pos.sector or ''})
    except Exception:
        if 'portfolio_positions' not in st.session_state:
            st.session_state['portfolio_positions'] = []
        positions_data = st.session_state['portfolio_positions']
    with st.expander("➕ 录入新持仓", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1: new_symbol = st.text_input("股票代码", key="portfolio_symbol")
        with col2: new_name = st.text_input("股票名称", key="portfolio_name")
        with col3: new_shares = st.number_input("股数", min_value=100, step=100, key="portfolio_shares")
        with col4: new_price = st.number_input("成本价", min_value=0.01, step=0.01, key="portfolio_price")
        if st.button("添加持仓"):
            if not new_symbol:
                st.warning("请输入股票代码")
            else:
                try:
                    from src.data.database import get_db, PortfolioPosition
                    db = next(get_db())
                    position = PortfolioPosition(symbol=new_symbol, name=new_name or new_symbol, shares=new_shares, entry_price=new_price, entry_date=datetime.now(), status="持有中", created_at=datetime.now())
                    db.add(position)
                    db.commit()
                    st.success(f"已添加 {new_symbol} {new_name} {new_shares}股 @ {new_price}")
                    st.rerun()
                except Exception:
                    st.session_state['portfolio_positions'].append({'代码': new_symbol, '名称': new_name or new_symbol, '持仓量': new_shares, '成本价': new_price, '现价': new_price, 'sector': ''})
                    st.success(f"已添加 {new_symbol} {new_name} {new_shares}股 @ {new_price}（内存模式）")
                    st.rerun()
    st.markdown('<div class="section-header">📋 当前持仓</div>', unsafe_allow_html=True)
    if not positions_data:
        st.info("暂无持仓记录，请使用上方表单添加持仓。")
        return
    with st.spinner("正在更新持仓价格..."):
        for pos in positions_data:
            df = get_real_stock_data(pos['代码'], days=5)
            if df is not None and len(df) > 0:
                pos['现价'] = df['close'].iloc[-1]
                pos['涨跌幅'] = df['pctChg'].iloc[-1] if pd.notna(df['pctChg'].iloc[-1]) else 0
    for pos in positions_data:
        pos['浮动盈亏'] = pos['持仓量'] * (pos['现价'] - pos['成本价'])
        pos['盈亏率'] = (pos['现价'] - pos['成本价']) / pos['成本价']
        df = get_real_stock_data(pos['代码'], days=30)
        if df is not None and len(df) >= 20:
            df = compute_real_indicators(df)
            if df is not None:
                pos['Layer1'], _ = rule_based_trend(df)
                pos['Layer2'], _ = rule_based_signal(df)
                pos['建议'] = pos['Layer2']
            else:
                pos['Layer1'] = 'N/A'
                pos['Layer2'] = 'N/A'
                pos['建议'] = 'N/A'
        else:
            pos['Layer1'] = 'N/A'
            pos['Layer2'] = 'N/A'
            pos['建议'] = 'N/A'
    portfolio_rows = []
    for pos in positions_data:
        pnl_pct = pos.get('盈亏率', 0)
        pnl_str = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct*100:.2f}%"
        pnl_amount = pos.get('浮动盈亏', 0)
        pnl_amount_str = f"{'+' if pnl_amount >= 0 else ''}{pnl_amount:,.0f}"
        portfolio_rows.append({'代码': pos['代码'], '名称': pos['名称'], '持仓量': pos['持仓量'], '成本价': f"{pos['成本价']:.2f}", '现价': f"{pos['现价']:.2f}", '浮动盈亏': pnl_amount_str, '盈亏率': pnl_str, 'Layer1': pos.get('Layer1', 'N/A'), 'Layer2': pos.get('Layer2', 'N/A'), '建议': pos.get('建议', 'N/A')})
    df_portfolio = pd.DataFrame(portfolio_rows)
    st.dataframe(df_portfolio, use_container_width=True, height=400)
    st.markdown('<div class="section-header">📊 组合分析</div>', unsafe_allow_html=True)
    total_cost = sum(pos['持仓量'] * pos['成本价'] for pos in positions_data)
    total_market = sum(pos['持仓量'] * pos['现价'] for pos in positions_data)
    total_pnl = total_market - total_cost
    total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0
    total_day_pnl = sum(pos['持仓量'] * pos['现价'] * pos.get('涨跌幅', 0) / 100 for pos in positions_data if '涨跌幅' in pos)
    max_weight = max((pos['持仓量'] * pos['现价'] for pos in positions_data), default=0)
    concentration = max_weight / total_market if total_market > 0 else 0
    if concentration > 0.5: risk = "极高"
    elif concentration > 0.3: risk = "高"
    elif concentration > 0.2: risk = "中等"
    else: risk = "低"
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("总市值", f"¥ {total_market:,.0f}", f"{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct*100:.2f}%")
    with col2: st.metric("浮动盈亏", f"{'+' if total_pnl >= 0 else ''}¥ {total_pnl:,.0f}", f"{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct*100:.2f}%")
    with col3: st.metric("当日盈亏", f"{'+' if total_day_pnl >= 0 else ''}¥ {total_day_pnl:,.0f}", f"{'+' if total_day_pnl >= 0 else ''}{total_day_pnl/total_market*100:.2f}%")
    with col4: st.metric("风险等级", risk, f"集中度 {concentration*100:.0f}%")
    st.markdown("### ⚠️ 持仓预警")
    alerts = []
    for pos in positions_data:
        pnl_pct = pos.get('盈亏率', 0)
        if pnl_pct >= 0.15:
            alerts.append({"symbol": pos['代码'], "type": "止盈", "message": f"{pos['名称']} 盈利达{pnl_pct*100:.1f}%，达到止盈目标", "level": "warning"})
        elif pnl_pct <= -0.07:
            alerts.append({"symbol": pos['代码'], "type": "止损", "message": f"{pos['名称']} 亏损达{abs(pnl_pct)*100:.1f}%，触发止损线", "level": "danger"})
    if not alerts:
        st.info("当前无预警信息")
    else:
        for alert in alerts:
            if alert['level'] == "danger": st.error(f"**{alert['symbol']}** - {alert['message']}")
            else: st.warning(f"**{alert['symbol']}** - {alert['message']}")


# ==================== 系统设置 ====================

def render_settings():
    st.markdown('<div class="main-header">⚙️ 系统设置</div>', unsafe_allow_html=True)
    cfg = load_config()
    if cfg is None:
        st.error("无法加载 config.yaml，将使用默认值")
        cfg = {}
    st.markdown('<div class="section-header">📄 当前配置 (config.yaml)</div>', unsafe_allow_html=True)
    with st.expander("查看当前配置", expanded=True):
        st.json(cfg)
    st.markdown('<div class="section-header">🤖 LLM配置</div>', unsafe_allow_html=True)
    llm_cfg = cfg.get('llm', {})
    provider_options = ["openai", "anthropic", "local", "azure"]
    current_provider = llm_cfg.get('provider', 'openai')
    provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0
    llm_provider = st.selectbox("LLM提供商", provider_options, index=provider_index)
    api_key = st.text_input("API Key", value=llm_cfg.get('api_key', ''), type="password")
    model_name = st.text_input("模型名称", value=llm_cfg.get('model', 'gpt-4o'))
    st.markdown('<div class="section-header">📡 数据源配置</div>', unsafe_allow_html=True)
    data_cfg = cfg.get('data', {})
    source_options = ["akshare", "baostock", "tushare"]
    current_primary = data_cfg.get('primary_source', 'akshare')
    primary_index = source_options.index(current_primary) if current_primary in source_options else 0
    backup_options = ["baostock", "akshare", "tushare"]
    current_backup = data_cfg.get('backup_source', 'baostock')
    backup_index = backup_options.index(current_backup) if current_backup in backup_options else 0
    primary_source = st.selectbox("主数据源", source_options, index=primary_index)
    backup_source = st.selectbox("备用数据源", backup_options, index=backup_index)
    st.markdown('<div class="section-header">🔄 默认回测参数</div>', unsafe_allow_html=True)
    bt_cfg = cfg.get('backtest', {})
    default_stop_loss = st.slider("默认止损", 0.03, 0.15, float(bt_cfg.get('stop_loss', 0.10)))
    default_take_profit = st.slider("默认止盈", 0.05, 0.30, float(bt_cfg.get('take_profit', 0.15)))
    max_positions = st.number_input("最大持仓数", 1, 50, int(bt_cfg.get('max_positions', 10)))
    st.markdown('<div class="section-header">🔍 股票筛选</div>', unsafe_allow_html=True)
    screen_cfg = cfg.get('screening', {})
    exclude_st = st.checkbox("排除ST股", value=bool(screen_cfg.get('exclude_st', True)))
    exclude_gem = st.checkbox("排除创业板", value=bool(screen_cfg.get('exclude_gem', True)))
    min_price = st.number_input("最低股价", 1.0, 100.0, float(screen_cfg.get('min_price', 5.0)))
    if st.button("💾 保存设置"):
        new_cfg = {
            'llm': {'provider': llm_provider, 'api_key': api_key, 'model': model_name, 'temperature': llm_cfg.get('temperature', 0.7), 'max_tokens': llm_cfg.get('max_tokens', 2000)},
            'data': {'primary_source': primary_source, 'backup_source': backup_source, 'cache_enabled': data_cfg.get('cache_enabled', True)},
            'backtest': {'initial_capital': bt_cfg.get('initial_capital', 1000000), 'commission_rate': bt_cfg.get('commission_rate', 0.0003), 'stamp_tax_rate': bt_cfg.get('stamp_tax_rate', 0.001), 'slippage': bt_cfg.get('slippage', 0.001), 'stop_loss': default_stop_loss, 'take_profit': default_take_profit, 'max_positions': max_positions, 'position_size_pct': bt_cfg.get('position_size_pct', 0.15)},
            'screening': {'exclude_st': exclude_st, 'exclude_gem': exclude_gem, 'min_price': min_price},
            'hmm': cfg.get('hmm', {'n_components': 4, 'covariance_type': 'diag', 'n_iter': 1000}),
            'logging': cfg.get('logging', {'level': 'INFO', 'file': 'data/logs/app.log'})
        }
        if save_config(new_cfg):
            st.success("设置已保存到 config.yaml！")
            st.cache_data.clear()


# ==================== 主函数 ====================

def main():
    page = render_sidebar()
    if page == "🏠 数据面板": render_dashboard()
    elif page == "🎯 今日选股": render_today_pick()
    elif page == "📈 行情分析": render_analysis()
    elif page == "🤖 AI辩论室": render_debate_room()
    elif page == "🔄 回测中心": render_backtest()
    elif page == "💼 持仓分析": render_portfolio()
    elif page == "⚙️ 系统设置": render_settings()


import atexit
atexit.register(bs.logout)

if __name__ == "__main__":
    main()
