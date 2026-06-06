"""
A股智能分析选股系统 - Streamlit主应用

页面结构:
1. 数据面板 - 市场概览、自选股监控
2. 行情分析 - 个股三层决策分析
3. AI辩论室 - 6角色辩论过程展示
4. 回测中心 - 策略回测与参数优化
5. 持仓分析 - 持仓管理与跟踪
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# 页面配置
st.set_page_config(
    page_title="A股智能分析选股系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .signal-buy {
        color: #e74c3c;
        font-weight: bold;
    }
    .signal-sell {
        color: #27ae60;
        font-weight: bold;
    }
    .signal-neutral {
        color: #95a5a6;
    }
    .weather-sunny { color: #f39c12; font-size: 2rem; }
    .weather-cloudy { color: #7f8c8d; font-size: 2rem; }
    .weather-rainy { color: #3498db; font-size: 2rem; }
    .debate-role {
        background-color: #ecf0f1;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .debate-bull { border-left-color: #e74c3c; }
    .debate-bear { border-left-color: #27ae60; }
    .debate-judge { border-left-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)


# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏导航"""
    st.sidebar.title("📊 导航菜单")

    page = st.sidebar.radio(
        "选择页面",
        ["🏠 数据面板", "📈 行情分析", "🤖 AI辩论室", "🔄 回测中心", "💼 持仓分析", "⚙️ 系统设置"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **系统版本**: v2.0
    **架构**: 三层决策 + Review
    **Layer 1**: HMM趋势定性 (程序化)
    **Layer 2**: 技术面因子 (程序化)
    **Layer 3**: AI辩论决策
    """)

    return page


# ==================== 数据面板 ====================
def render_dashboard():
    """渲染数据面板"""
    st.markdown('<div class="main-header">🏠 市场数据面板</div>', unsafe_allow_html=True)

    # 市场概览指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("上证指数", "3,052.34", "+0.45%")
    with col2:
        st.metric("深证成指", "9,876.54", "+0.82%")
    with col3:
        st.metric("创业板指", "1,987.65", "+1.23%")
    with col4:
        st.metric("涨跌比", "2856:2145", "偏多")

    st.markdown("---")

    # 自选股监控
    st.markdown('<div class="section-header">📋 自选股监控</div>', unsafe_allow_html=True)

    # 模拟自选股数据
    watchlist_data = {
        '代码': ['600519', '000858', '002594', '300750', '601318'],
        '名称': ['贵州茅台', '五粮液', '比亚迪', '宁德时代', '中国平安'],
        '最新价': [1680.50, 145.30, 245.60, 198.50, 42.35],
        '涨跌': ['+1.2%', '+0.8%', '-0.5%', '+2.1%', '-0.3%'],
        'Layer1趋势': ['上涨', '偏多震荡', '上涨', '强烈上涨', '偏空震荡'],
        'Layer2信号': ['买入', '持有', '买入', '强烈买入', '观望'],
        '综合评分': [85, 72, 78, 92, 45]
    }

    df_watch = pd.DataFrame(watchlist_data)

    # 根据评分着色
    def color_score(val):
        if val >= 80:
            return 'background-color: #d4edda'
        elif val >= 60:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'

    styled_df = df_watch.style.applymap(color_score, subset=['综合评分'])
    st.dataframe(styled_df, use_container_width=True)

    # 快速分析按钮
    selected_stock = st.selectbox("选择股票进行快速分析", df_watch['代码'].tolist())
    if st.button("🔍 启动三层分析"):
        st.session_state['analyze_stock'] = selected_stock
        st.session_state['page'] = "📈 行情分析"
        st.rerun()


# ==================== 行情分析 ====================
def render_analysis():
    """渲染行情分析页面"""
    st.markdown('<div class="main-header">📈 行情分析 - 三层决策</div>', unsafe_allow_html=True)

    # 股票输入
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("输入股票代码", "600519", help="输入6位股票代码")
    with col2:
        analyze_btn = st.button("🚀 启动分析", type="primary", use_container_width=True)

    if analyze_btn or 'analyze_result' in st.session_state:
        # 模拟分析结果
        with st.spinner("正在执行三层决策分析..."):
            # 这里实际应调用后端分析引擎
            simulate_analysis(symbol)


def simulate_analysis(symbol: str):
    """模拟分析结果展示"""

    # Layer 1: HMM趋势定性
    st.markdown('<div class="section-header">🔍 Layer 1: HMM趋势定性分析</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前状态", "上涨", "置信度 78%")
    with col2:
        st.metric("状态持续时间", "5天", "+2天")
    with col3:
        st.metric("趋势强度", "0.65", "偏强")

    # HMM状态概率图
    fig_hmm = go.Figure()
    states = ['强烈下跌', '下跌', '偏空震荡', '偏多震荡', '上涨', '强烈上涨']
    probs = [0.02, 0.05, 0.10, 0.15, 0.45, 0.23]

    colors = ['#e74c3c' if p > 0.2 else '#3498db' for p in probs]
    fig_hmm.add_trace(go.Bar(x=states, y=probs, marker_color=colors))
    fig_hmm.update_layout(
        title="HMM隐藏状态概率分布",
        xaxis_title="状态",
        yaxis_title="概率",
        height=300
    )
    st.plotly_chart(fig_hmm, use_container_width=True)

    st.info("**分析结论**: 当前处于「上涨」状态，历史同状态下未来5日上涨概率 68.5%")

    # Layer 2: 技术面因子
    st.markdown('<div class="section-header">📊 Layer 2: 技术面因子信号</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("趋势跟踪", "看多", "评分 0.72")
    with col2:
        st.metric("多因子打分", "75分", "偏多")
    with col3:
        st.metric("量价模式", "放量上涨", "评分 0.68")

    # 综合信号
    composite_score = 0.72 * 0.3 + 0.75 * 0.4 + 0.68 * 0.3

    fig_signal = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=composite_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "综合信号评分"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "#e74c3c" if composite_score > 0 else "#27ae60"},
            'steps': [
                {'range': [-1, -0.3], 'color': "#d5f5e3"},
                {'range': [-0.3, 0.3], 'color': "#fef9e7"},
                {'range': [0.3, 1], 'color': "#fadbd8"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': composite_score
            }
        }
    ))
    fig_signal.update_layout(height=300)
    st.plotly_chart(fig_signal, use_container_width=True)

    st.success(f"**综合信号**: 买入 (评分: {composite_score:.2f})")

    # Layer 3: AI辩论
    st.markdown('<div class="section-header">🤖 Layer 3: AI辩论决策</div>', unsafe_allow_html=True)

    # 辩论过程展示
    debate_tabs = st.tabs(["技术面分析师", "基本面分析师", "宏观分析师", "多头研究员", "空头研究员", "裁判裁决"])

    with debate_tabs[0]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**技术面分析师报告**")
        st.write("""
        - 均线系统呈多头排列，5日线上穿20日线形成金叉
        - MACD柱状体持续放大，动能强劲
        - RSI(14) = 62，未进入超买区，仍有上行空间
        - 成交量连续3日放大，资金流入明显
        - **结论**: 技术面支持做多
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with debate_tabs[1]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**基本面分析师报告**")
        st.write("""
        - Q3营收同比增长23%，超预期
        - 毛利率提升至42%，盈利能力改善
        - 机构持仓比例上升至65%，筹码集中
        - 估值PE(TTM) = 28x，处于历史中位数
        - **结论**: 基本面稳健，估值合理
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with debate_tabs[2]:
        st.markdown('<div class="debate-role">', unsafe_allow_html=True)
        st.write("**宏观分析师报告**")
        st.write("""
        - 央行近期释放流动性，市场资金面宽松
        - 行业政策利好频出，监管环境友好
        - 北向资金连续净流入，外资看好
        - **结论**: 宏观环境支持风险资产
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with debate_tabs[3]:
        st.markdown('<div class="debate-role debate-bull">', unsafe_allow_html=True)
        st.write("**🔴 多头研究员论点**")
        st.write("""
        1. 三层信号共振，趋势明确向上
        2. 量价配合良好，资金持续流入
        3. 基本面改善，业绩有支撑
        4. 目标价: 当前价 +15%
        5. **建议**: 积极做多
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with debate_tabs[4]:
        st.markdown('<div class="debate-role debate-bear">', unsafe_allow_html=True)
        st.write("**🟢 空头研究员论点**")
        st.write("""
        1. 短期涨幅过大，存在回调压力
        2. RSI接近超买区，技术指标有背离迹象
        3. 宏观不确定性仍存，需警惕黑天鹅
        4. 止损位: 当前价 -7%
        5. **建议**: 谨慎追高，设好止损
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with debate_tabs[5]:
        st.markdown('<div class="debate-role debate-judge">', unsafe_allow_html=True)
        st.write("**⚖️ 中立裁判裁决**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("多头票数", "4票", "占比 67%")
        with col2:
            st.metric("空头票数", "2票", "占比 33%")
        with col3:
            st.metric("最终裁决", "看多", "置信度 72%")

        st.write("""
        **裁决理由**:
        - 技术面和基本面均给出积极信号
        - 宏观环境有利，资金面宽松
        - 风险点在于短期涨幅较大，建议分批建仓
        - **操作方案**: 买入 60% 仓位，设置 7% 止损
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # 天气预报式预测
    st.markdown('<div class="section-header">🌤️ 未来3天走势预测</div>', unsafe_allow_html=True)

    forecast_cols = st.columns(3)
    forecasts = [
        {'day': 'T+1', 'weather': '☀️', 'class': 'weather-sunny', 'direction': '上涨', 'prob': '65%', 'target': '+2.5%', 'accuracy': '78%'},
        {'day': 'T+2', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡偏多', 'prob': '55%', 'target': '+1.2%', 'accuracy': '72%'},
        {'day': 'T+3', 'weather': '⛅', 'class': 'weather-cloudy', 'direction': '震荡', 'prob': '50%', 'target': '+0.5%', 'accuracy': '65%'},
    ]

    for i, fc in enumerate(forecasts):
        with forecast_cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
                <h4>{fc['day']}</h4>
                <div class="{fc['class']}">{fc['weather']}</div>
                <p><strong>{fc['direction']}</strong></p>
                <p>概率: {fc['prob']}</p>
                <p>目标: {fc['target']}</p>
                <p style="font-size: 0.8rem; color: gray;">历史准确率: {fc['accuracy']}</p>
            </div>
            """, unsafe_allow_html=True)

    # 自纠错提示
    if composite_score > 0.5:
        st.warning("⚠️ **自纠错提示**: Layer 1/2/3信号一致看多，但需警惕一致性预期风险。建议关注成交量变化。")


# ==================== AI辩论室 ====================
def render_debate_room():
    """渲染AI辩论室"""
    st.markdown('<div class="main-header">🤖 AI辩论室</div>', unsafe_allow_html=True)

    # 辩论配置
    col1, col2 = st.columns([2, 1])
    with col1:
        debate_symbol = st.text_input("股票代码", "600519")
    with col2:
        start_debate = st.button("⚔️ 开始辩论", type="primary", use_container_width=True)

    if start_debate:
        with st.spinner("AI分析师正在准备报告..."):
            # 模拟辩论流程
            progress_bar = st.progress(0)

            # Phase 1: 分析师报告
            st.markdown("### Phase 1: 分析师并行报告生成")
            analyst_cols = st.columns(3)

            with analyst_cols[0]:
                with st.container():
                    st.write("**📈 技术面分析师**")
                    st.write("生成报告中...")
                    progress_bar.progress(20)
                    time.sleep(0.5)
                    st.success("✅ 报告完成")
                    st.write("趋势: 看多 | 评分: 72/100")

            with analyst_cols[1]:
                with st.container():
                    st.write("**📊 基本面分析师**")
                    st.write("生成报告中...")
                    progress_bar.progress(40)
                    time.sleep(0.5)
                    st.success("✅ 报告完成")
                    st.write("估值: 合理 | 评分: 68/100")

            with analyst_cols[2]:
                with st.container():
                    st.write("**🌍 宏观分析师**")
                    st.write("生成报告中...")
                    progress_bar.progress(60)
                    time.sleep(0.5)
                    st.success("✅ 报告完成")
                    st.write("环境: 友好 | 评分: 75/100")

            # Phase 2: 研究员辩论
            st.markdown("### Phase 2: 研究员辩论")
            progress_bar.progress(80)

            debate_container = st.container()
            with debate_container:
                col_bull, col_bear = st.columns(2)

                with col_bull:
                    st.markdown('<div class="debate-role debate-bull">', unsafe_allow_html=True)
                    st.write("**🔴 多头研究员**")
                    st.write("基于三层分析，我认为应该做多:")
                    st.write("1. HMM状态为'上涨'，历史胜率68%")
                    st.write("2. 技术面金叉确认，量能配合")
                    st.write("3. 基本面Q3超预期，估值合理")
                    st.write("**目标价: +15%**")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_bear:
                    st.markdown('<div class="debate-role debate-bear">', unsafe_allow_html=True)
                    st.write("**🟢 空头研究员**")
                    st.write("虽然信号偏多，但我有顾虑:")
                    st.write("1. 短期涨幅已大，RSI接近超买")
                    st.write("2. 大盘面临压力位，系统性风险")
                    st.write("3. 建议等待回调后再入场")
                    st.write("**止损位: -7%**")
                    st.markdown('</div>', unsafe_allow_html=True)

            # Phase 3: 裁判裁决
            st.markdown("### Phase 3: 中立裁判裁决")
            progress_bar.progress(100)

            st.balloons()

            col_result1, col_result2, col_result3 = st.columns(3)
            with col_result1:
                st.metric("最终决策", "看多", "多头胜出")
            with col_result2:
                st.metric("置信度", "72%", "中等信心")
            with col_result3:
                st.metric("建议仓位", "60%", "分批建仓")

            st.write("**裁决总结**: 综合考虑技术面、基本面和宏观环境，多头论点更具说服力。但需注意短期回调风险，建议分批建仓并设置止损。")


# ==================== 回测中心 ====================
def render_backtest():
    """渲染回测中心"""
    st.markdown('<div class="main-header">🔄 策略回测中心</div>', unsafe_allow_html=True)

    # 回测配置
    st.markdown('<div class="section-header">⚙️ 回测参数设置</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("结束日期", datetime.now())
    with col3:
        initial_capital = st.number_input("初始资金", value=1000000, step=100000)

    col4, col5, col6 = st.columns(3)
    with col4:
        stop_loss = st.slider("止损比例", 0.03, 0.15, 0.07, 0.01)
    with col5:
        take_profit = st.slider("止盈比例", 0.05, 0.30, 0.15, 0.01)
    with col6:
        position_size = st.slider("单票仓位", 0.05, 0.30, 0.10, 0.05)

    # 股票池选择
    stock_pool = st.multiselect(
        "选择股票池",
        ['沪深300', '中证500', '上证50', '自选股', '全部A股'],
        default=['沪深300']
    )

    run_backtest = st.button("▶️ 运行回测", type="primary")

    if run_backtest:
        with st.spinner("回测运行中..."):
            # 模拟回测结果
            simulate_backtest_results()


def simulate_backtest_results():
    """模拟回测结果"""

    # 回测结果指标
    st.markdown('<div class="section-header">📊 回测结果</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("总收益率", "+23.5%", "+15.2% 超额")
    with col2:
        st.metric("年化收益", "25.3%", "基准 8.5%")
    with col3:
        st.metric("最大回撤", "-12.8%", "可控")
    with col4:
        st.metric("夏普比率", "1.45", "优秀")
    with col5:
        st.metric("胜率", "58.3%", "盈亏比 1.8")

    # 权益曲线
    st.markdown("### 权益曲线")

    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    cumulative = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative * 1000000,
        mode='lines', name='策略',
        line=dict(color='#e74c3c')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative * 0.85 * 1000000,
        mode='lines', name='基准',
        line=dict(color='#95a5a6', dash='dash')
    ))
    fig.update_layout(
        title="策略 vs 基准",
        xaxis_title="日期",
        yaxis_title="资金",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # 交易记录
    st.markdown("### 交易记录")

    trades_data = {
        '日期': ['2023-02-15', '2023-03-20', '2023-04-10', '2023-05-25', '2023-06-30'],
        '代码': ['600519', '000858', '002594', '300750', '601318'],
        '操作': ['买入', '买入', '卖出', '买入', '卖出'],
        '价格': [1680.0, 145.0, 250.0, 195.0, 43.0],
        '盈亏': ['-', '-', '+5.2%', '-', '+3.8%'],
        '原因': ['Layer1/2共振', '趋势确认', '止盈', '突破买入', '信号反转']
    }

    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)

    # 参数优化
    st.markdown("### 参数敏感性分析")

    param_data = pd.DataFrame({
        '止损比例': [0.05, 0.07, 0.10, 0.05, 0.07, 0.10],
        '止盈比例': [0.10, 0.10, 0.10, 0.15, 0.15, 0.15],
        '总收益': [18.5, 23.5, 20.2, 22.1, 25.8, 21.5],
        '夏普比率': [1.25, 1.45, 1.32, 1.38, 1.52, 1.40]
    })

    st.dataframe(param_data, use_container_width=True)
    st.info("💡 **建议**: 当前最优参数组合为止损7% + 止盈15%，夏普比率最高。")


# ==================== 持仓分析 ====================
def render_portfolio():
    """渲染持仓分析页面"""
    st.markdown('<div class="main-header">💼 持仓分析</div>', unsafe_allow_html=True)

    # 持仓录入
    with st.expander("➕ 录入新持仓", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_symbol = st.text_input("股票代码")
        with col2:
            new_name = st.text_input("股票名称")
        with col3:
            new_shares = st.number_input("股数", min_value=100, step=100)
        with col4:
            new_price = st.number_input("成本价", min_value=0.01, step=0.01)

        if st.button("添加持仓"):
            st.success(f"已添加 {new_symbol} {new_shares}股 @ {new_price}")

    # 持仓列表
    st.markdown('<div class="section-header">📋 当前持仓</div>', unsafe_allow_html=True)

    portfolio_data = {
        '代码': ['600519', '000858', '002594'],
        '名称': ['贵州茅台', '五粮液', '比亚迪'],
        '持仓量': [100, 200, 500],
        '成本价': [1650.0, 142.0, 240.0],
        '现价': [1680.5, 145.3, 245.6],
        '浮动盈亏': ['+3050', '+660', '+2800'],
        '盈亏率': ['+1.85%', '+2.32%', '+2.33%'],
        'Layer1': ['上涨', '偏多震荡', '上涨'],
        'Layer2': ['买入', '持有', '买入'],
        '建议': ['持有', '持有', '持有']
    }

    df_portfolio = pd.DataFrame(portfolio_data)
    st.dataframe(df_portfolio, use_container_width=True)

    # 组合分析
    st.markdown('<div class="section-header">📊 组合分析</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总市值", "¥ 425,860", "+2.1%")
    with col2:
        st.metric("浮动盈亏", "+¥ 6,510", "+1.57%")
    with col3:
        st.metric("当日盈亏", "+¥ 2,340", "+0.55%")
    with col4:
        st.metric("风险等级", "中等", "集中度 32%")

    # 持仓分析
    st.markdown("### 个股深度分析")

    selected_position = st.selectbox("选择持仓进行分析", df_portfolio['代码'].tolist())

    if st.button("🔍 启动持仓分析"):
        with st.spinner("分析中..."):
            # 模拟持仓分析结果
            st.write(f"**{selected_position} 持仓分析报告**")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**成本分析**")
                st.write("- 成本价: 1650.0")
                st.write("- 现价: 1680.5")
                st.write("- 盈亏平衡点: 1654.95")
                st.write("- 距平衡点: +1.54%")

            with col2:
                st.write("**支撑压力**")
                st.write("- 最近支撑: 1620.0")
                st.write("- 最近压力: 1720.0")
                st.write("- 距支撑: +3.7%")
                st.write("- 距压力: -2.3%")

            st.write("**操作建议**: 盈利状态，趋势向好，建议继续持有，关注1720压力位。")

    # 预警信息
    st.markdown("### ⚠️ 持仓预警")

    alerts = [
        {"symbol": "300750", "type": "止盈", "message": "盈利达15.2%，达到止盈目标", "level": "warning"},
        {"symbol": "601318", "type": "止损", "message": "亏损达7.5%，触发止损线", "level": "danger"},
    ]

    for alert in alerts:
        if alert['level'] == 'danger':
            st.error(f"🚨 **{alert['symbol']}** - {alert['message']}")
        else:
            st.warning(f"⚠️ **{alert['symbol']}** - {alert['message']}")


# ==================== 系统设置 ====================
def render_settings():
    """渲染系统设置页面"""
    st.markdown('<div class="main-header">⚙️ 系统设置</div>', unsafe_allow_html=True)

    # LLM配置
    st.markdown('<div class="section-header">🤖 LLM配置</div>', unsafe_allow_html=True)

    llm_provider = st.selectbox(
        "LLM提供商",
        ["OpenAI", "Anthropic", "本地模型", "Azure OpenAI"]
    )

    api_key = st.text_input("API Key", type="password")
    model_name = st.text_input("模型名称", "gpt-4")

    # 数据源配置
    st.markdown('<div class="section-header">📡 数据源配置</div>', unsafe_allow_html=True)

    primary_source = st.selectbox("主数据源", ["AkShare", "BaoStock", "Tushare"])
    backup_source = st.selectbox("备用数据源", ["BaoStock", "AkShare", "Tushare"])

    # 回测配置
    st.markdown('<div class="section-header">🔄 默认回测参数</div>', unsafe_allow_html=True)

    default_stop_loss = st.slider("默认止损", 0.03, 0.15, 0.07)
    default_take_profit = st.slider("默认止盈", 0.05, 0.30, 0.15)
    max_positions = st.number_input("最大持仓数", 1, 50, 10)

    # 筛选配置
    st.markdown('<div class="section-header">🔍 股票筛选</div>', unsafe_allow_html=True)

    exclude_st = st.checkbox("排除ST股", value=True)
    exclude_gem = st.checkbox("排除创业板", value=True)
    min_price = st.number_input("最低股价", 1.0, 100.0, 5.0)

    if st.button("💾 保存设置"):
        st.success("设置已保存！")


# ==================== 主函数 ====================
def main():
    """主函数"""
    page = render_sidebar()

    if page == "🏠 数据面板":
        render_dashboard()
    elif page == "📈 行情分析":
        render_analysis()
    elif page == "🤖 AI辩论室":
        render_debate_room()
    elif page == "🔄 回测中心":
        render_backtest()
    elif page == "💼 持仓分析":
        render_portfolio()
    elif page == "⚙️ 系统设置":
        render_settings()


if __name__ == "__main__":
    main()
