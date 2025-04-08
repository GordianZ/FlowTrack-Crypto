import json
import logging
import os
from datetime import datetime

import numpy as np
import requests
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route('/analysis/<interval>/<symbol>')
def analysis(symbol,interval):
    # show the post with the given id, the id is an integer
    return run_analysis(symbol.split(),interval)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置API密钥和URL
BINANCE_API_URL = "https://api.binance.com"
BINANCE_FUTURES_API_URL = "https://fapi.binance.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "api信息")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 数据获取和分析函数
def format_number(num):
    """格式化数字，保留适当的小数位数"""
    if isinstance(num, (int, float)):
        if abs(num) >= 1000:
            return f"{num:.2f}"
        elif abs(num) >= 1:
            return f"{num:.4f}"
        else:
            return f"{num:.8f}"
    return num


def get_klines_data(symbol, interval="5m", limit=50, is_futures=False):
    """获取K线数据"""
    try:
        base_url = BINANCE_FUTURES_API_URL if is_futures else BINANCE_API_URL
        endpoint = "/fapi/v1/klines" if is_futures else "/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit + 1  # 多获取一根，用于计算最后一根的变化
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        response.raise_for_status()

        klines = response.json()

        # 移除最后一根未完成的K线
        klines = klines[:-1]

        # 处理K线数据
        processed_klines = []
        for i, kline in enumerate(klines):
            open_time = datetime.fromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            close_time = datetime.fromtimestamp(kline[6] / 1000).strftime('%Y-%m-%d %H:%M:%S')

            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            quote_asset_volume = float(kline[7])

            # 计算买入和卖出量（简化估算）
            if close_price >= open_price:
                # 上涨K线，假设60%的成交量是买入
                buy_volume = volume * 0.6
                sell_volume = volume * 0.4
            else:
                # 下跌K线，假设40%的成交量是买入
                buy_volume = volume * 0.4
                sell_volume = volume * 0.6

            # 计算净流入资金
            net_inflow = (buy_volume - sell_volume) * close_price

            # 计算价格变化百分比
            price_change_pct = ((close_price - open_price) / open_price) * 100

            processed_kline = {
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "quote_volume": quote_asset_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "net_inflow": net_inflow,
                "price_change_pct": price_change_pct
            }

            processed_klines.append(processed_kline)

        return processed_klines

    except Exception as e:
        logger.error(f"获取K线数据出错: {e}")
        raise Exception(f"获取{symbol} {interval}K线数据失败: {str(e)}")


def get_orderbook_stats(symbol, is_futures=False, limit=1000):
    """获取订单簿数据并计算统计信息"""
    try:
        base_url = BINANCE_FUTURES_API_URL if is_futures else BINANCE_API_URL
        endpoint = "/fapi/v1/depth" if is_futures else "/api/v3/depth"

        params = {
            "symbol": symbol,
            "limit": limit
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        response.raise_for_status()

        orderbook = response.json()

        # 处理订单簿数据
        bids = [[float(price), float(qty)] for price, qty in orderbook["bids"]]
        asks = [[float(price), float(qty)] for price, qty in orderbook["asks"]]

        # 计算买卖盘总量
        total_bid_qty = sum(bid[1] for bid in bids)
        total_ask_qty = sum(ask[1] for ask in asks)

        # 计算买卖盘不平衡度
        imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) if (
                                                                                                     total_bid_qty + total_ask_qty) > 0 else 0

        # 计算买卖盘压力
        bid_pressure = sum(bid[0] * bid[1] for bid in bids)
        ask_pressure = sum(ask[0] * ask[1] for ask in asks)

        # 计算买卖盘压力比
        pressure_ratio = bid_pressure / ask_pressure if ask_pressure > 0 else float('inf')

        # 计算价格范围
        bid_prices = [bid[0] for bid in bids]
        ask_prices = [ask[0] for ask in asks]

        price_range = {
            "highest_bid": max(bid_prices) if bid_prices else 0,
            "lowest_ask": min(ask_prices) if ask_prices else 0,
            "spread": min(ask_prices) - max(bid_prices) if bid_prices and ask_prices else 0,
            "spread_pct": (
                        (min(ask_prices) - max(bid_prices)) / max(bid_prices) * 100) if bid_prices and ask_prices else 0
        }

        return {
            "total_bid_qty": total_bid_qty,
            "total_ask_qty": total_ask_qty,
            "imbalance": imbalance,
            "bid_pressure": bid_pressure,
            "ask_pressure": ask_pressure,
            "pressure_ratio": pressure_ratio,
            "price_range": price_range
        }

    except Exception as e:
        logger.error(f"获取订单簿数据出错: {e}")
        raise Exception(f"获取{symbol}订单簿数据失败: {str(e)}")


def analyze_funding_flow_trend(klines_data, window_size=10):
    """分析资金流向趋势"""
    if not klines_data or len(klines_data) < window_size:
        return {
            "trend": "unknown",
            "confidence": 0,
            "net_inflow_total": 0,
            "net_inflow_recent": 0,
            "price_stage": "unknown"
        }

    # 计算总净流入
    net_inflow_total = sum(k["net_inflow"] for k in klines_data)

    # 计算最近窗口的净流入
    net_inflow_recent = sum(k["net_inflow"] for k in klines_data[-window_size:])

    # 计算净流入的移动平均
    window_inflows = []
    for i in range(len(klines_data) - window_size + 1):
        window_inflow = sum(k["net_inflow"] for k in klines_data[i:i + window_size])
        window_inflows.append(window_inflow)

    # 确定趋势
    trend = "neutral"
    if len(window_inflows) >= 3:
        recent_inflows = window_inflows[-3:]
        if all(x > 0 for x in recent_inflows) and recent_inflows[-1] > recent_inflows[-2]:
            trend = "increasing"
        elif all(x < 0 for x in recent_inflows) and recent_inflows[-1] < recent_inflows[-2]:
            trend = "decreasing"
        elif sum(1 for x in recent_inflows if x > 0) >= 2:
            trend = "slightly_increasing"
        elif sum(1 for x in recent_inflows if x < 0) >= 2:
            trend = "slightly_decreasing"

    # 计算趋势置信度
    if trend in ["increasing", "decreasing"]:
        confidence = 0.8
    elif trend in ["slightly_increasing", "slightly_decreasing"]:
        confidence = 0.6
    else:
        confidence = 0.4

    # 判断价格所处阶段
    price_stage = "unknown"
    if len(klines_data) >= 20:
        recent_prices = [k["close"] for k in klines_data[-20:]]
        price_changes = [recent_prices[i] - recent_prices[i - 1] for i in range(1, len(recent_prices))]

        # 计算价格变化的移动平均
        price_ma = sum(recent_prices) / len(recent_prices)
        latest_price = recent_prices[-1]

        # 计算价格波动率
        price_volatility = np.std(price_changes) / price_ma if price_ma > 0 else 0

        # 判断价格阶段
        if latest_price > price_ma * 1.05 and trend in ["increasing", "slightly_increasing"]:
            price_stage = "上涨中"
        elif latest_price < price_ma * 0.95 and trend in ["decreasing", "slightly_decreasing"]:
            price_stage = "下跌中"
        elif price_volatility < 0.01 and abs(latest_price - price_ma) / price_ma < 0.02:
            price_stage = "整理中"
        elif latest_price > price_ma * 1.08 and trend in ["decreasing", "slightly_decreasing"]:
            price_stage = "可能顶部"
        elif latest_price < price_ma * 0.92 and trend in ["increasing", "slightly_increasing"]:
            price_stage = "可能底部"
        else:
            price_stage = "波动中"

    return {
        "trend": trend,
        "confidence": confidence,
        "net_inflow_total": net_inflow_total,
        "net_inflow_recent": net_inflow_recent,
        "price_stage": price_stage
    }


def detect_anomalies(klines_data, window_size=10, threshold=2.0):
    """检测异常交易"""
    if not klines_data or len(klines_data) < window_size * 2:
        return {
            "has_anomalies": False,
            "anomalies": []
        }

    anomalies = []

    # 计算成交量和净流入的均值和标准差
    volumes = [k["volume"] for k in klines_data]
    inflows = [k["net_inflow"] for k in klines_data]

    volume_mean = np.mean(volumes)
    volume_std = np.std(volumes)
    inflow_mean = np.mean(inflows)
    inflow_std = np.std(inflows)

    # 检测异常成交量和净流入
    for i, kline in enumerate(klines_data):
        anomaly = {}

        # 检测异常成交量
        volume_z_score = (kline["volume"] - volume_mean) / volume_std if volume_std > 0 else 0
        if abs(volume_z_score) > threshold:
            anomaly["volume"] = {
                "value": kline["volume"],
                "z_score": volume_z_score,
                "direction": "high" if volume_z_score > 0 else "low"
            }

        # 检测异常净流入
        inflow_z_score = (kline["net_inflow"] - inflow_mean) / inflow_std if inflow_std > 0 else 0
        if abs(inflow_z_score) > threshold:
            anomaly["net_inflow"] = {
                "value": kline["net_inflow"],
                "z_score": inflow_z_score,
                "direction": "high" if inflow_z_score > 0 else "low"
            }

        # 检测价格和成交量不匹配的情况
        price_change = kline["price_change_pct"]
        if abs(price_change) > 1.0 and volume_z_score < 0:
            anomaly["price_volume_mismatch"] = {
                "price_change": price_change,
                "volume_z_score": volume_z_score
            }

        # 如果存在异常，添加到列表
        if anomaly:
            anomaly["time"] = kline["close_time"]
            anomalies.append(anomaly)

    return {
        "has_anomalies": len(anomalies) > 0,
        "anomalies": anomalies[-5:] if anomalies else []  # 只返回最近的5个异常
    }


def analyze_funding_pressure(klines_data, orderbook_stats):
    """分析资金压力"""
    if not klines_data or not orderbook_stats:
        return {
            "pressure_direction": "unknown",
            "confidence": 0,
            "imbalance": 0
        }

    # 获取订单簿不平衡度
    imbalance = orderbook_stats["imbalance"]

    # 获取最近的价格变化
    recent_klines = klines_data[-5:] if len(klines_data) >= 5 else klines_data
    recent_price_changes = [k["price_change_pct"] for k in recent_klines]
    avg_price_change = sum(recent_price_changes) / len(recent_price_changes) if recent_price_changes else 0

    # 判断资金压力方向
    pressure_direction = "neutral"
    if imbalance > 0.2 and avg_price_change > 0:
        pressure_direction = "upward_strong"
    elif imbalance > 0.1 and avg_price_change > 0:
        pressure_direction = "upward"
    elif imbalance < -0.2 and avg_price_change < 0:
        pressure_direction = "downward_strong"
    elif imbalance < -0.1 and avg_price_change < 0:
        pressure_direction = "downward"
    elif imbalance > 0.1 and avg_price_change < 0:
        pressure_direction = "potential_reversal_up"
    elif imbalance < -0.1 and avg_price_change > 0:
        pressure_direction = "potential_reversal_down"

    # 计算置信度
    confidence = abs(imbalance) * 2 if abs(imbalance) < 0.5 else 1.0

    return {
        "pressure_direction": pressure_direction,
        "confidence": confidence,
        "imbalance": imbalance,
        "bid_ask_ratio": orderbook_stats["pressure_ratio"]
    }


def send_to_deepseek(data, interval):
    """将数据发送给DeepSeek API并获取解读"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # 根据不同时间间隔设置相应的分析参数
    interval_settings = {
        "5m": {
            "forecast_period": "未来2-6小时",
            "trade_horizon": "短线（数小时内）",
            "stop_loss_range": "较小（0.5%-1.5%）",
            "analysis_depth": "微观市场结构和短期波动",
            "position_sizing": "建议小仓位（5%-15%）"
        },
        "15m": {
            "forecast_period": "未来6-12小时",
            "trade_horizon": "短线至中短线（半天至1天）",
            "stop_loss_range": "中小（1%-2%）",
            "analysis_depth": "短期趋势和支撑阻力位",
            "position_sizing": "建议小至中等仓位（10%-20%）"
        },
        "30m": {
            "forecast_period": "未来12-24小时",
            "trade_horizon": "中短线（1-2天）",
            "stop_loss_range": "中等（1.5%-3%）",
            "analysis_depth": "日内趋势和关键价格区间",
            "position_sizing": "建议中等仓位（15%-25%）"
        },
        "1h": {
            "forecast_period": "未来1-3天",
            "trade_horizon": "中线（2-5天）",
            "stop_loss_range": "中等（2%-4%）",
            "analysis_depth": "中期趋势和市场结构转换",
            "position_sizing": "建议中等仓位（20%-30%）"
        },
        "4h": {
            "forecast_period": "未来3-7天",
            "trade_horizon": "中长线（1-2周）",
            "stop_loss_range": "中大（3%-6%）",
            "analysis_depth": "中长期趋势和市场周期",
            "position_sizing": "建议中至大仓位（25%-40%）"
        },
        "1d": {
            "forecast_period": "未来1-4周",
            "trade_horizon": "长线（2周-1个月）",
            "stop_loss_range": "较大（5%-10%）",
            "analysis_depth": "长期趋势、市场周期和宏观因素影响",
            "position_sizing": "建议大仓位或分批建仓（30%-50%）"
        }
    }

    # 获取当前时间间隔的设置
    interval_key = interval.lower()
    if interval_key not in interval_settings:
        interval_key = "1h"  # 默认使用1小时设置

    settings = interval_settings[interval_key]

    prompt = (
            f"## Binance资金流向专业分析任务 (K线周期: {interval})\n\n"
            f"我已收集了Binance现货和期货市场过去50根{interval}K线的资金流向数据（已剔除最新未完成的一根），包括：\n"
            "- 各交易对的资金流向趋势分析\n"
            "- 价格所处阶段预测（顶部、底部、上涨中、下跌中、整理中）\n"
            "- 订单簿数据（买卖盘不平衡度）\n"
            "- 资金压力分析\n"
            "- 异常交易检测\n\n"

            f"请从专业交易员和机构投资者角度，针对{interval}周期特点进行深度分析：\n\n"

            "1. **主力资金行为解读**：\n"
            "   - 通过资金流向趋势变化，识别主力资金的建仓、出货行为\n"
            "   - 结合订单簿数据，分析主力资金的意图（吸筹、出货、洗盘等）\n"
            "   - 特别关注资金流向与价格变化不匹配的异常情况\n"
            f"   - 重点分析{settings['analysis_depth']}\n\n"

            "2. **价格阶段判断**：\n"
            "   - 根据资金流向趋势和价格关系，判断各交易对处于什么阶段（顶部、底部、上涨中、下跌中、整理中）\n"
            "   - 提供判断的置信度和依据\n"
            "   - 对比不同交易对的阶段差异，分析可能的轮动关系\n"
            f"   - 结合{interval}周期特有的市场结构特征\n\n"

            "3. **趋势预判**：\n"
            f"   - 基于资金流向和资金压力分析，预判{settings['forecast_period']}可能的价格走势\n"
            "   - 识别可能的反转信号或趋势延续信号\n"
            "   - 关注异常交易数据可能暗示的行情变化\n"
            f"   - 给出具体的价格目标区间和时间预期\n\n"

            "4. **交易策略建议**：\n"
            "   - 针对每个交易对，给出具体的交易建议（观望、做多、做空、减仓等）\n"
            f"   - 提供适合{settings['trade_horizon']}的入场点位和止损位\n"
            f"   - 建议止损范围：{settings['stop_loss_range']}\n"
            f"   - {settings['position_sizing']}\n"
            "   - 评估风险和回报比\n\n"

            "请使用专业术语，保持分析简洁但深入，避免泛泛而谈。数据如下：\n\n" +
            json.dumps(data, indent=2, ensure_ascii=False) +
            "\n\n回复格式要求：中文，使用markdown格式，重点突出，适当使用表格对比分析。"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        raise Exception(f"AI分析失败: {str(e)}")


def run_analysis(symbols, interval):
    """运行完整的分析流程并返回结果"""
    logger.info(f"开始分析，当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"目标交易对: {symbols}")
    logger.info(f"选择的时间间隔: {interval}")

    # 创建进度条
    # progress_bar = st.progress(0)
    # status_text = st.empty()

    try:
        # 获取现货和期货的K线数据
        logger.info(f"正在获取{interval}周期K线数据...")
        spot_klines_data = {}
        futures_klines_data = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"正在获取 {symbol} 现货{interval}K线数据...")
            spot_klines_data[symbol] = get_klines_data(symbol, interval=interval, limit=50,
                                                       is_futures=False)

            logger.info(f"正在获取 {symbol} 期货{interval}K线数据...")
            futures_klines_data[symbol] = get_klines_data(symbol, interval=interval, limit=50,
                                                          is_futures=True)

            # 更新进度条
            #progress_bar.progress((i + 1) / (len(symbols) * 4))

        # 获取订单簿数据
        logger.info("正在获取订单簿数据...")
        spot_order_books = {}
        futures_order_books = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"正在获取 {symbol} 订单簿数据...")
            spot_order_books[symbol] = get_orderbook_stats(symbol, is_futures=False)
            futures_order_books[symbol] = get_orderbook_stats(symbol, is_futures=True)

            # 更新进度条
            # progress_bar.progress(0.25 + (i + 1) / (len(symbols) * 4))

        # 分析资金流向趋势
        logger.info("正在分析资金流向趋势...")
        spot_trend_analysis = {}
        futures_trend_analysis = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"正在分析 {symbol} 资金流向趋势...")
            spot_trend_analysis[symbol] = analyze_funding_flow_trend(spot_klines_data[symbol])
            futures_trend_analysis[symbol] = analyze_funding_flow_trend(futures_klines_data[symbol])

            # 更新进度条
            # progress_bar.progress(0.5 + (i + 1) / (len(symbols) * 4))

        # 检测异常交易
        logger.info("正在检测异常交易...")
        spot_anomalies = {}
        futures_anomalies = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"正在检测 {symbol} 异常交易...")
            spot_anomalies[symbol] = detect_anomalies(spot_klines_data[symbol])
            futures_anomalies[symbol] = detect_anomalies(futures_klines_data[symbol])

            # 更新进度条
            # progress_bar.progress(0.75 + (i + 1) / (len(symbols) * 4))

        # 分析资金压力
        logger.info("正在分析资金压力...")
        spot_pressure_analysis = {}
        futures_pressure_analysis = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"正在分析 {symbol} 资金压力...")
            spot_pressure_analysis[symbol] = analyze_funding_pressure(spot_klines_data[symbol],
                                                                      spot_order_books[symbol])
            futures_pressure_analysis[symbol] = analyze_funding_pressure(futures_klines_data[symbol],
                                                                         futures_order_books[symbol])

        # 整合数据
        logger.info("正在整合分析数据...")
        analysis_data = {}

        for symbol in symbols:
            analysis_data[symbol] = {
                "spot": {
                    "klines_summary": {
                        "first_time": spot_klines_data[symbol][0]["open_time"] if spot_klines_data[symbol] else None,
                        "last_time": spot_klines_data[symbol][-1]["close_time"] if spot_klines_data[symbol] else None,
                        "price_change": (spot_klines_data[symbol][-1]["close"] - spot_klines_data[symbol][0]["open"]) /
                                        spot_klines_data[symbol][0]["open"] * 100 if spot_klines_data[symbol] else 0,
                        "current_price": spot_klines_data[symbol][-1]["close"] if spot_klines_data[symbol] else 0,
                        "total_volume": sum(k["volume"] for k in spot_klines_data[symbol]) if spot_klines_data[
                            symbol] else 0,
                        "total_quote_volume": sum(k["quote_volume"] for k in spot_klines_data[symbol]) if
                        spot_klines_data[symbol] else 0
                    },
                    "funding_trend": spot_trend_analysis[symbol],
                    "anomalies": spot_anomalies[symbol],
                    "order_book": spot_order_books[symbol],
                    "funding_pressure": spot_pressure_analysis[symbol]
                },
                "futures": {
                    "klines_summary": {
                        "first_time": futures_klines_data[symbol][0]["open_time"] if futures_klines_data[
                            symbol] else None,
                        "last_time": futures_klines_data[symbol][-1]["close_time"] if futures_klines_data[
                            symbol] else None,
                        "price_change": (futures_klines_data[symbol][-1]["close"] - futures_klines_data[symbol][0][
                            "open"]) / futures_klines_data[symbol][0]["open"] * 100 if futures_klines_data[
                            symbol] else 0,
                        "current_price": futures_klines_data[symbol][-1]["close"] if futures_klines_data[symbol] else 0,
                        "total_volume": sum(k["volume"] for k in futures_klines_data[symbol]) if futures_klines_data[
                            symbol] else 0,
                        "total_quote_volume": sum(k["quote_volume"] for k in futures_klines_data[symbol]) if
                        futures_klines_data[symbol] else 0
                    },
                    "funding_trend": futures_trend_analysis[symbol],
                    "anomalies": futures_anomalies[symbol],
                    "order_book": futures_order_books[symbol],
                    "funding_pressure": futures_pressure_analysis[symbol]
                },
                "comparison": {
                    "spot_vs_futures_price_diff": (spot_klines_data[symbol][-1]["close"] -
                                                   futures_klines_data[symbol][-1]["close"]) /
                                                  spot_klines_data[symbol][-1]["close"] * 100 if spot_klines_data[
                                                                                                     symbol] and
                                                                                                 futures_klines_data[
                                                                                                     symbol] else 0,
                    "spot_vs_futures_volume_ratio": sum(k["volume"] for k in spot_klines_data[symbol]) / sum(
                        k["volume"] for k in futures_klines_data[symbol]) if spot_klines_data[symbol] and
                                                                             futures_klines_data[symbol] and sum(
                        k["volume"] for k in futures_klines_data[symbol]) > 0 else 0,
                    "spot_vs_futures_net_inflow_diff": spot_trend_analysis[symbol]["net_inflow_total"] -
                                                       futures_trend_analysis[symbol]["net_inflow_total"] if
                    spot_trend_analysis[symbol] and futures_trend_analysis[symbol] else 0
                }
            }

        # 添加分析时间和参数信息
        analysis_metadata = {
            "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "interval": interval,
            "symbols_analyzed": symbols,
            "klines_count": 50
        }

        # 整合所有数据
        deepseek_data = {
            "metadata": analysis_metadata,
            "analysis": analysis_data
        }

        # 发送到DeepSeek进行解读
        logger.info("正在通过AI解读分析结果...")
        deepseek_result = send_to_deepseek(deepseek_data, interval)

        # 清除进度条和状态文本
        # progress_bar.empty()
        # status_text.empty()

        return deepseek_result

    except Exception as e:
        # 清除进度条和状态文本
        # progress_bar.empty()
        # status_text.empty()
        # 重新抛出异常，让上层处理
        raise e


# 主程序入口
if __name__ == "__main__":
    # 所有逻辑都在上面实现了
    pass

