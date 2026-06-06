# A股智能分析选股系统 v2.0

## 系统架构

### 三层决策机制

```
┌─────────────────────────────────────────────────────────────┐
│                    三层决策架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: HMM趋势定性分析 (纯程序化)                          │
│  ├── 三级HMM: 大盘 → 行业 → 个股                              │
│  ├── 监督式聚类: n_components=5                              │
│  └── 输出: 趋势评分 + 状态概率分布                             │
│                                                             │
│  Layer 2: 技术面因子策略信号 (纯程序化)                        │
│  ├── 趋势跟踪因子 (30%)                                      │
│  ├── 多因子打分模型 (40%)                                     │
│  └── 量价模式识别 (30%)                                      │
│                                                             │
│  Layer 3: AI辩论决策分析 (AI参与)                             │
│  ├── 6角色Agent: 技术面/基本面/宏观分析师 + 多头/空头研究员      │
│  ├── 裁判投票机制                                            │
│  └── 输出: 最终决策 + 置信度                                  │
│                                                             │
│  Review阶段: 次日复核 (AI参与)                               │
│  ├── 对比预测 vs 实际                                        │
│  └── 输出: 准确率评估 + 优化建议                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心特性

- **无未来函数**: T日收盘生成信号，T+1开盘执行
- **监督式HMM**: 通过历史标签校准隐藏状态语义
- **天气预报式预测**: ☀️晴/⛅多云/🌧️雨，未来3天走势预测
- **向量化舆情关联**: 政策-行业-企业-个股四层关联
- **自纠错机制**: 三层不一致时触发深度分析

## 快速开始

### 安装依赖

```bash
pip install akshare baostock hmmlearn openai tushare yfinance
pip install sqlalchemy pandas numpy plotly streamlit requests beautifulsoup4
```

### 配置LLM

编辑 `config.yaml`:

```yaml
llm:
  provider: "openai"
  api_key: "sk-your-key-here"
  model: "gpt-4o"
  temperature: 0.7
```

### 启动系统

```bash
# Streamlit UI
streamlit run app.py

# 命令行模式
python main.py --mode cli
```

## 项目结构

```
a_stock_analyzer/
├── app.py              # Streamlit主应用
├── main.py             # 主入口/CLI
├── config.yaml         # 配置文件
├── requirements.txt    # 依赖清单
├── src/
│   ├── data/           # 数据源(AkShare/BaoStock)
│   ├── models/         # HMM + 技术面因子
│   ├── ai_debate/      # AI辩论引擎 + Review
│   ├── backtest/       # 回测引擎
│   ├── portfolio/      # 持仓管理
│   ├── sentiment/      # 舆情采集
│   └── analysis/       # 预测报告 + 筛选器
└── tests/              # 单元测试
```

## 使用示例

### 分析单只股票

```python
from main import AStockAnalyzer

analyzer = AStockAnalyzer()
result = analyzer.analyze_stock("600519")

print(result['layer1_trend'])   # HMM趋势
print(result['layer2_signal'])  # 技术面信号
print(result['layer3_debate'])  # AI辩论
print(result['forecast'])       # 天气预报预测
```

## License

MIT
