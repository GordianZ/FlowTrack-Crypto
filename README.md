这是docker容器版，让大家免于配置python环境的烦恼。
安装docker之后可以使用以下`compose.yaml`启动容器：

```yaml
services:
  web:
    image: gordianz/flowtrack
    environment:
      "DEEPSEEK_API_KEY": "你的DS API KEY"
    ports:
      - 8501:8501
```
将API KEY替换为你自己的KEY之后保存为`compose.yaml`，在该目录下运行下面的命令即可访问 http://localhost:8501 使用该工具。
```bash
docker compose up
```
以下是原项目的介绍：



# Binance资金流向分析工具

## 项目简介

这是一个专业的加密货币资金流向分析工具，主要通过分析Binance交易所的现货和期货市场数据，提供深度的市场洞察和交易策略建议。该工具结合K线数据、订单簿深度和DeepSeek AI模型，为交易者提供全面的市场分析。

## 主要功能

1. **资金流向分析**
   - 分析现货和期货市场的资金流入/流出趋势
   - 识别主力资金的建仓、出货行为
   - 对比不同交易对的资金流向差异

2. **市场阶段判断**
   - 自动判断市场所处阶段（顶部、底部、上涨中、下跌中、整理中）
   - 提供判断的置信度和具体依据
   - 分析不同交易对之间可能存在的轮动关系

3. **订单簿深度分析**
   - 计算买卖盘不平衡度
   - 分析关键价格区间内的买卖盘压力
   - 结合资金流向评估市场压力方向和强度

4. **异常交易检测**
   - 识别成交量异常但价格变化不大的情况
   - 检测价格异常波动但成交量不高的情况
   - 发现极端资金净流入/流出的异常交易

5. **AI驱动的专业分析**
   - 通过DeepSeek API提供专业交易员视角的市场解读
   - 生成短期趋势预判和交易策略建议
   - 输出结构化的markdown格式分析报告

## 技术特点

- **多维度数据采集**：同时分析现货和期货市场数据，提供更全面的市场视角
- **高级统计分析**：使用相关性分析、线性回归等方法挖掘市场趋势
- **领先/滞后关系分析**：研究资金流向与价格变化之间的时间关系
- **异常值检测**：使用统计方法识别可能的市场操纵行为
- **API速率限制处理**：内置请求限流机制，确保稳定运行

## 使用方法

1. **配置API密钥**
   - 设置Binance API密钥和密钥（`BINANCE_API_KEY`和`BINANCE_API_SECRET`）
   - 设置DeepSeek API密钥（`DEEPSEEK_API_KEY`）

2. **设置监控交易对**
   - 在`SYMBOLS`列表中添加或修改需要监控的交易对

3. **运行分析**
   ```
   python binance_funding_flow_analyzer.py
   ```

4. **查看结果**
   - 分析结果将输出到控制台
   - 同时保存为markdown文件（`binance_analysis.md`）

## 系统要求

- Python 3.7+
- 依赖库：requests, pandas, numpy, scipy, binance-python, telegram, ratelimit

## 安装依赖

```bash
pip install requests pandas numpy scipy python-binance python-telegram-bot ratelimit
```

## 注意事项

1. 确保Binance API密钥具有读取权限（无需交易权限）
2. 分析结果仅供参考，不构成投资建议
3. 请遵守Binance API使用条款，避免过于频繁的请求
4. DeepSeek API调用会产生费用，请合理使用

## 未来计划

- 添加更多交易所数据源
- 实现自动交易策略执行
- 开发Web界面展示分析结果
- 增加更多技术指标和分析维度

## 免责声明

本工具仅供学习和研究目的使用，不构成任何投资建议。加密货币市场风险高，请谨慎投资。作者不对使用本工具产生的任何投资损失负责。
