#!/usr/bin/env python3
"""
Campaign奖励计算模块

功能：
1. 计算市场总交易量和手续费
2. 计算用户手续费占比
3. 估算用户应得奖励
"""

import asyncio
import aiohttp
import json
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

ASTER_BASE = "https://sapi.asterdex.com"
CACHE_DIR = Path("logs")
CACHE_DIR.mkdir(exist_ok=True)

# Aster手续费率
MAKER_FEE_RATE = Decimal("0.00005")  # 0.005%
TAKER_FEE_RATE = Decimal("0.0004")   # 0.04%


async def fetch_aster_price() -> Decimal:
    """
    获取ASTER/USDT实时价格

    Returns:
        ASTER的USDT价格
    """
    url = f"{ASTER_BASE}/api/v1/ticker/24hr"
    params = {"symbol": "ASTERUSDT"}

    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                print(f"  ⚠️  获取ASTER价格失败，使用默认值0.714")
                return Decimal("0.714")

            data = await resp.json()
            price = Decimal(str(data["lastPrice"]))
            return price


@dataclass
class CampaignConfig:
    """Campaign配置"""
    symbol: str
    start_time: datetime  # UTC
    end_time: datetime    # UTC
    reward_pool_usdt: Decimal
    fee_type: str = "buy_only"  # "buy_only" 或 "total"
    reward_cap: Decimal = Decimal(0)  # 每人最多拿奖池的百分比，0 表示无上限


@dataclass
class MarketStats:
    """市场统计数据"""
    total_trades: int
    total_volume_usdt: Decimal
    maker_volume_usdt: Decimal
    taker_volume_usdt: Decimal
    maker_fee_usdt: Decimal
    taker_fee_usdt: Decimal
    total_fee_usdt: Decimal
    # Buy order 统计（用于奖励分配计算）
    buy_fee_usdt: Decimal  # 买方手续费总和
    # 市场总手续费（买方+卖方，用于 fee_type=total 的规则）
    market_total_fee_usdt: Decimal = Decimal(0)


@dataclass
class UserStats:
    """用户统计数据"""
    total_trades: int
    total_volume_usdt: Decimal
    maker_volume_usdt: Decimal
    taker_volume_usdt: Decimal
    maker_fee_aster: Decimal
    taker_fee_aster: Decimal
    total_fee_aster: Decimal
    # Buy order 统计（用于奖励分配计算）
    buy_fee_aster: Decimal  # 买单手续费（ASTER）


@dataclass
class UserTradeStats:
    """用户交易统计数据（含 PNL 计算）"""
    total_trades: int
    total_volume_usdt: Decimal      # 总交易量
    maker_volume_usdt: Decimal      # Maker 交易量
    taker_volume_usdt: Decimal      # Taker 交易量
    fee_usdt: Decimal               # 手续费（按 fee rate 计算的 USDT）
    net_usdt_flow: Decimal          # 净 USDT 流动（卖出 - 买入）
    net_base_qty: Decimal           # 净持仓变化（买入 - 卖出）
    last_trade_id: int              # 最后一笔交易 ID（用于断点续传）

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """计算 PNL（不含手续费）= 净 USDT 流动 + 净持仓 × 当前价格"""
        return self.net_usdt_flow + self.net_base_qty * current_price


@dataclass
class RewardEstimate:
    """奖励估算"""
    user_fee_aster: Decimal
    market_total_fee_aster: Decimal
    user_share_pct: Decimal
    expected_reward_usdt: Decimal
    expected_reward_aster: Decimal
    reward_pool_usdt: Decimal


async def fetch_and_calculate_market_stats(
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
    use_cache: bool = True
) -> MarketStats:
    """
    获取市场公开交易数据并计算统计（边获取边计算，不保存原始数据，支持断点续传）

    Args:
        symbol: 交易对
        start_time_ms: 开始时间（毫秒时间戳）
        end_time_ms: 结束时间（毫秒时间戳）
        use_cache: 是否使用缓存的统计结果
    """
    # 缓存统计结果而非原始数据
    stats_cache_file = CACHE_DIR / f"{symbol.lower()}_stats_{start_time_ms}_{end_time_ms}.json"
    progress_file = CACHE_DIR / f"{symbol.lower()}_progress_{start_time_ms}_{end_time_ms}.json"

    # 尝试加载缓存的统计结果
    import time
    now_ms = int(time.time() * 1000)
    campaign_ended = end_time_ms < now_ms

    # 初始化统计变量
    total_trades = 0
    total_volume = Decimal(0)
    maker_volume = Decimal(0)
    taker_volume = Decimal(0)
    from_id = 0

    if use_cache and stats_cache_file.exists():
        with open(stats_cache_file, 'r') as f:
            data = json.load(f)

            # 如果 campaign 已结束且标记为 completed，直接返回缓存
            if data.get("completed", False) and campaign_ended:
                print(f"  📂 使用完整缓存: {stats_cache_file.name}")
                total_fee = Decimal(str(data["total_fee_usdt"]))
                buy_fee = Decimal(str(data.get("buy_fee_usdt", total_fee)))
                total_volume = Decimal(str(data["total_volume_usdt"]))
                # 兼容旧缓存：如果没有 market_total_fee_usdt，重新计算
                if "market_total_fee_usdt" in data:
                    market_total_fee = Decimal(str(data["market_total_fee_usdt"]))
                else:
                    market_total_fee = total_volume * (MAKER_FEE_RATE + TAKER_FEE_RATE)
                return MarketStats(
                    total_trades=data["total_trades"],
                    total_volume_usdt=total_volume,
                    maker_volume_usdt=Decimal(str(data["maker_volume_usdt"])),
                    taker_volume_usdt=Decimal(str(data["taker_volume_usdt"])),
                    maker_fee_usdt=Decimal(str(data["maker_fee_usdt"])),
                    taker_fee_usdt=Decimal(str(data["taker_fee_usdt"])),
                    total_fee_usdt=total_fee,
                    buy_fee_usdt=buy_fee,
                    market_total_fee_usdt=market_total_fee
                )

            # Campaign 进行中，从缓存断点续传
            if not campaign_ended and "last_trade_id" in data:
                total_trades = data["total_trades"]
                total_volume = Decimal(str(data["total_volume_usdt"]))
                maker_volume = Decimal(str(data["maker_volume_usdt"]))
                taker_volume = Decimal(str(data["taker_volume_usdt"]))
                from_id = data["last_trade_id"] + 1
                print(f"  🔄 Campaign 进行中，从 ID {from_id} 断点续传（已有 {total_trades:,} 条）")

    async with aiohttp.ClientSession(trust_env=True) as session:
        # 检查 progress_file（优先级高于 stats_cache）
        if use_cache and progress_file.exists():
            print(f"  🔄 发现未完成任务，从断点继续...")
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                total_trades = progress["total_trades"]
                total_volume = Decimal(str(progress["total_volume"]))
                maker_volume = Decimal(str(progress["maker_volume"]))
                taker_volume = Decimal(str(progress["taker_volume"]))
                from_id = progress["last_trade_id"] + 1
                print(f"  ↪️  从 trade ID {from_id} 继续，已处理 {total_trades:,} 条")
        elif from_id > 0:
            # 已从 stats_cache 加载了断点，继续
            print(f"  🔄 从缓存断点继续获取数据...")
        else:
            print(f"  🔄 开始获取并计算市场数据...")

        last_trade_id = from_id - 1 if from_id > 0 else 0
        batch_count = 0
        reached_end_time = False

        # 每日统计
        current_day = None
        daily_trades = 0
        daily_volume = Decimal(0)
        daily_buy_maker = 0   # 买方是 maker
        daily_buy_taker = 0   # 买方是 taker
        daily_stats = []  # [(date_str, trades, volume), ...]

        while True:
            # 使用 aggTrades 端点（支持 startTime/endTime 过滤，trades 端点不支持）
            url = f"{ASTER_BASE}/api/v1/aggTrades"
            params = {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000
            }

            # 分页：使用 fromId
            if from_id > 0:
                params["fromId"] = from_id

            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    print(f"  ⚠️  限流，等待 {retry_after}秒...")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status != 200:
                    print(f"  ❌ API错误: {resp.status}")
                    break

                # 读取rate limit信息
                used_weight = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", "0"))

                trades = await resp.json()
                if not trades:
                    break

                # aggTrades 字段映射: a=id, p=price, q=qty, T=time, m=isBuyerMaker
                for trade in trades:
                    trade_time = trade["T"]

                    # 检查是否超出时间范围
                    if trade_time < start_time_ms:
                        continue
                    if trade_time > end_time_ms:
                        reached_end_time = True
                        break

                    qty = Decimal(str(trade["q"]))
                    price = Decimal(str(trade["p"]))
                    quote_qty = qty * price

                    total_trades += 1
                    total_volume += quote_qty

                    if trade.get("m", False):
                        maker_volume += quote_qty
                    else:
                        taker_volume += quote_qty

                    last_trade_id = trade["a"]

                    # 每日统计
                    trade_date = datetime.fromtimestamp(trade_time/1000, tz=timezone.utc).strftime('%Y-%m-%d')
                    is_buyer_maker = trade.get("m", False)

                    if current_day is None:
                        current_day = trade_date
                    elif trade_date != current_day:
                        # 打印前一天的统计
                        # buy_maker/sell_taker 数量相同，buy_taker/sell_maker 数量相同
                        daily_stats.append((current_day, daily_trades, daily_volume))
                        print(f"  📅 {current_day}: {daily_trades:,} 条, ${daily_volume:,.2f} | "
                              f"BuyMaker:{daily_buy_maker:,} BuyTaker:{daily_buy_taker:,} "
                              f"SellMaker:{daily_buy_taker:,} SellTaker:{daily_buy_maker:,}")
                        current_day = trade_date
                        daily_trades = 0
                        daily_volume = Decimal(0)
                        daily_buy_maker = 0
                        daily_buy_taker = 0

                    daily_trades += 1
                    daily_volume += quote_qty
                    if is_buyer_maker:
                        daily_buy_maker += 1
                    else:
                        daily_buy_taker += 1

                # 如果已经超出结束时间，停止获取更多数据
                if reached_end_time:
                    break

                batch_count += 1
                # 每50批保存一次进度（静默）
                if batch_count % 50 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump({
                            "last_trade_id": last_trade_id,
                            "total_trades": total_trades,
                            "total_volume": str(total_volume),
                            "maker_volume": str(maker_volume),
                            "taker_volume": str(taker_volume)
                        }, f)

                # 如果返回的数据不足1000条，说明已经获取完该时间范围内的所有数据
                if len(trades) < 1000:
                    break

                from_id = trades[-1]["a"] + 1  # aggTrades 使用 'a' 字段作为 ID

                # 动态限流：接近限制时增加延迟
                if used_weight > 5500:  # 超过91%
                    await asyncio.sleep(1.0)
                elif used_weight > 5000:  # 超过83%
                    await asyncio.sleep(0.5)
                elif used_weight > 4500:  # 超过75%
                    await asyncio.sleep(0.2)
                # else: 不延迟，全速运行

    # 打印最后一天的统计
    if current_day is not None and daily_trades > 0:
        daily_stats.append((current_day, daily_trades, daily_volume))
        print(f"  📅 {current_day}: {daily_trades:,} 条, ${daily_volume:,.2f} | "
              f"BuyMaker:{daily_buy_maker:,} BuyTaker:{daily_buy_taker:,} "
              f"SellMaker:{daily_buy_taker:,} SellTaker:{daily_buy_maker:,}")

    # 计算手续费
    # 注意：maker/taker 是根据买方是否是 maker 来分的
    # maker_fee = 买方作为 maker 时的手续费
    # taker_fee = 买方作为 taker 时的手续费
    # 所以 total_fee = buy_fee（买方总手续费）
    maker_fee = maker_volume * MAKER_FEE_RATE
    taker_fee = taker_volume * TAKER_FEE_RATE
    total_fee = maker_fee + taker_fee
    buy_fee = total_fee  # buy order 总手续费

    # 市场总手续费（买方+卖方）
    # 买方手续费 = maker_volume * maker_rate + taker_volume * taker_rate
    # 卖方手续费 = maker_volume * taker_rate + taker_volume * maker_rate
    # 市场总手续费 = total_volume * (maker_rate + taker_rate)
    market_total_fee = total_volume * (MAKER_FEE_RATE + TAKER_FEE_RATE)

    stats = MarketStats(
        total_trades=total_trades,
        total_volume_usdt=total_volume,
        maker_volume_usdt=maker_volume,
        taker_volume_usdt=taker_volume,
        maker_fee_usdt=maker_fee,
        taker_fee_usdt=taker_fee,
        total_fee_usdt=total_fee,
        buy_fee_usdt=buy_fee,
        market_total_fee_usdt=market_total_fee
    )

    # 保存统计结果（支持断点续传）
    with open(stats_cache_file, 'w') as f:
        json.dump({
            "completed": campaign_ended,  # 只有 campaign 结束时才标记为完成
            "last_trade_id": last_trade_id,  # 保存最后的 trade ID，用于断点续传
            "total_trades": stats.total_trades,
            "total_volume_usdt": str(stats.total_volume_usdt),
            "maker_volume_usdt": str(stats.maker_volume_usdt),
            "taker_volume_usdt": str(stats.taker_volume_usdt),
            "maker_fee_usdt": str(stats.maker_fee_usdt),
            "taker_fee_usdt": str(stats.taker_fee_usdt),
            "total_fee_usdt": str(stats.total_fee_usdt),
            "buy_fee_usdt": str(stats.buy_fee_usdt),
            "market_total_fee_usdt": str(stats.market_total_fee_usdt)
        }, f, indent=2)

    # 删除进度文件（已完成）
    if progress_file.exists():
        progress_file.unlink()

    print(f"  ✅ 共 {total_trades:,} 条交易，统计完成")
    return stats


def calculate_market_stats(trades: list) -> MarketStats:
    """计算市场统计数据"""
    total_volume = Decimal(0)
    maker_volume = Decimal(0)
    taker_volume = Decimal(0)

    for trade in trades:
        qty = Decimal(str(trade["qty"]))
        price = Decimal(str(trade["price"]))
        quote_qty = qty * price
        total_volume += quote_qty

        # isBuyerMaker: true = 买方maker，卖方taker
        is_buyer_maker = trade.get("isBuyerMaker", False)
        if is_buyer_maker:
            maker_volume += quote_qty
        else:
            taker_volume += quote_qty

    maker_fee = maker_volume * MAKER_FEE_RATE
    taker_fee = taker_volume * TAKER_FEE_RATE
    total_fee = maker_fee + taker_fee
    buy_fee = total_fee  # buy order 总手续费
    market_total_fee = total_volume * (MAKER_FEE_RATE + TAKER_FEE_RATE)  # 买+卖

    return MarketStats(
        total_trades=len(trades),
        total_volume_usdt=total_volume,
        maker_volume_usdt=maker_volume,
        taker_volume_usdt=taker_volume,
        maker_fee_usdt=maker_fee,
        taker_fee_usdt=taker_fee,
        total_fee_usdt=total_fee,
        buy_fee_usdt=buy_fee,
        market_total_fee_usdt=market_total_fee
    )


def calculate_user_stats(user_trades: list) -> UserStats:
    """
    计算用户统计数据

    Args:
        user_trades: 用户交易记录（来自 /api/v1/myTrades）
    """
    total_volume = Decimal(0)
    maker_volume = Decimal(0)
    taker_volume = Decimal(0)
    maker_fee = Decimal(0)
    taker_fee = Decimal(0)
    buy_fee = Decimal(0)  # 只统计 buy order 的手续费

    for trade in user_trades:
        qty = Decimal(str(trade["qty"]))
        price = Decimal(str(trade["price"]))
        commission = Decimal(str(trade["commission"]))  # ASTER
        quote_qty = qty * price
        total_volume += quote_qty

        is_maker = trade.get("maker", False)
        if is_maker:
            maker_volume += quote_qty
            maker_fee += commission
        else:
            taker_volume += quote_qty
            taker_fee += commission

        # 统计 buy order 的手续费（用于奖励分配计算）
        is_buyer = trade.get("buyer", False)
        if is_buyer:
            buy_fee += commission

    return UserStats(
        total_trades=len(user_trades),
        total_volume_usdt=total_volume,
        maker_volume_usdt=maker_volume,
        taker_volume_usdt=taker_volume,
        maker_fee_aster=maker_fee,
        taker_fee_aster=taker_fee,
        total_fee_aster=maker_fee + taker_fee,
        buy_fee_aster=buy_fee
    )


def calculate_user_trade_stats(
    user_trades: list,
    existing_stats: Optional[UserTradeStats] = None
) -> UserTradeStats:
    """
    计算用户交易统计（含 PNL），支持增量更新

    Args:
        user_trades: 用户交易记录（来自 /api/v1/userTrades）
        existing_stats: 已有的统计数据（用于增量更新）

    Returns:
        UserTradeStats: 包含交易量、手续费、PNL 相关数据
    """
    # 初始化或使用已有数据
    if existing_stats:
        total_trades = existing_stats.total_trades
        total_volume = existing_stats.total_volume_usdt
        maker_volume = existing_stats.maker_volume_usdt
        taker_volume = existing_stats.taker_volume_usdt
        fee_usdt = existing_stats.fee_usdt
        net_usdt_flow = existing_stats.net_usdt_flow
        net_base_qty = existing_stats.net_base_qty
        last_trade_id = existing_stats.last_trade_id
    else:
        total_trades = 0
        total_volume = Decimal(0)
        maker_volume = Decimal(0)
        taker_volume = Decimal(0)
        fee_usdt = Decimal(0)
        net_usdt_flow = Decimal(0)
        net_base_qty = Decimal(0)
        last_trade_id = 0

    for trade in user_trades:
        trade_id = trade["id"]

        # 跳过已处理的交易（断点续传）
        if trade_id <= last_trade_id:
            continue

        qty = Decimal(str(trade["qty"]))           # base 数量
        quote_qty = Decimal(str(trade["quoteQty"]))  # quote 数量 (USDT)
        is_buyer = trade.get("isBuyer", False)
        is_maker = trade.get("maker", False)

        total_trades += 1
        total_volume += quote_qty

        # Maker/Taker 统计
        if is_maker:
            maker_volume += quote_qty
            fee_usdt += quote_qty * MAKER_FEE_RATE
        else:
            taker_volume += quote_qty
            fee_usdt += quote_qty * TAKER_FEE_RATE

        # PNL 计算：买入花费 USDT，卖出获得 USDT
        if is_buyer:
            net_usdt_flow -= quote_qty   # 买入花费 USDT
            net_base_qty += qty          # 买入获得 base
        else:
            net_usdt_flow += quote_qty   # 卖出获得 USDT
            net_base_qty -= qty          # 卖出失去 base

        last_trade_id = max(last_trade_id, trade_id)

    return UserTradeStats(
        total_trades=total_trades,
        total_volume_usdt=total_volume,
        maker_volume_usdt=maker_volume,
        taker_volume_usdt=taker_volume,
        fee_usdt=fee_usdt,
        net_usdt_flow=net_usdt_flow,
        net_base_qty=net_base_qty,
        last_trade_id=last_trade_id
    )


def estimate_reward(
    user_stats: UserStats,
    market_stats: MarketStats,
    reward_pool_usdt: Decimal,
    aster_price_usdt: Optional[Decimal] = None,
    fee_type: str = "buy_only",
    reward_cap: Decimal = Decimal(0)
) -> RewardEstimate:
    """
    估算用户奖励

    Args:
        user_stats: 用户统计
        market_stats: 市场统计
        reward_pool_usdt: 奖池大小（USDT）
        aster_price_usdt: ASTER价格（可选，用于换算）
        fee_type: "buy_only" 只算买方手续费，"total" 算总手续费（买+卖）
        reward_cap: 每人最多拿奖池的百分比，0 表示无上限
    """
    # 如果没有提供ASTER价格，尝试从奖池反推
    if aster_price_usdt is None:
        # 默认使用0.714 (BIO campaign的参考价格)
        aster_price_usdt = Decimal("0.714")

    # 根据 fee_type 选择计算方式
    if fee_type == "total":
        # 用户总手续费（ASTER → USDT）
        user_fee_usdt = user_stats.total_fee_aster * aster_price_usdt
        # 市场总手续费（买+卖）
        market_fee_usdt = market_stats.market_total_fee_usdt
        user_fee_aster = user_stats.total_fee_aster
    else:  # buy_only
        # 用户的 buy order 手续费（ASTER → USDT）
        user_fee_usdt = user_stats.buy_fee_aster * aster_price_usdt
        # 市场的 buy order 总手续费
        market_fee_usdt = market_stats.buy_fee_usdt
        user_fee_aster = user_stats.buy_fee_aster

    # 计算用户占比
    if market_fee_usdt > 0:
        user_share = user_fee_usdt / market_fee_usdt
    else:
        user_share = Decimal(0)

    # 应用 reward_cap（每人最多拿奖池的百分比）
    if reward_cap > 0 and user_share > reward_cap:
        user_share = reward_cap

    # 计算预期奖励（USDT）
    expected_reward_usdt = user_share * reward_pool_usdt

    # 换算为ASTER（按发放时价格）
    expected_reward_aster = expected_reward_usdt / aster_price_usdt

    return RewardEstimate(
        user_fee_aster=user_fee_aster,
        market_total_fee_aster=market_fee_usdt / aster_price_usdt,
        user_share_pct=user_share * 100,
        expected_reward_usdt=expected_reward_usdt,
        expected_reward_aster=expected_reward_aster,
        reward_pool_usdt=reward_pool_usdt
    )


async def get_campaign_report(
    campaign: CampaignConfig,
    user_trades: list,
    aster_price_usdt: Optional[Decimal] = None,
    use_cache: bool = True
) -> Tuple[MarketStats, UserStats, RewardEstimate]:
    """
    生成完整的campaign报告

    Args:
        campaign: Campaign配置
        user_trades: 用户交易记录
        aster_price_usdt: ASTER价格（可选，不提供则自动获取实时价格）
        use_cache: 是否使用市场数据缓存

    Returns:
        (market_stats, user_stats, reward_estimate)
    """
    # 如果没有提供ASTER价格，获取实时价格
    if aster_price_usdt is None:
        aster_price_usdt = await fetch_aster_price()
        print(f"  💱 ASTER实时价格: ${aster_price_usdt}")

    # 转换时间为毫秒
    start_ms = int(campaign.start_time.timestamp() * 1000)
    end_ms = int(campaign.end_time.timestamp() * 1000)

    # 获取并计算市场统计（边获取边计算，不保存原始数据）
    market_stats = await fetch_and_calculate_market_stats(
        campaign.symbol,
        start_ms,
        end_ms,
        use_cache=use_cache
    )

    # 计算用户统计
    user_stats = calculate_user_stats(user_trades)

    # 估算奖励
    reward = estimate_reward(
        user_stats,
        market_stats,
        campaign.reward_pool_usdt,
        aster_price_usdt,
        fee_type=campaign.fee_type,
        reward_cap=campaign.reward_cap
    )

    return market_stats, user_stats, reward


# 用于测试的main函数
async def main():
    """测试函数"""
    # 示例：BIO campaign
    campaign = CampaignConfig(
        symbol="BIOUSDT",
        start_time=datetime(2025, 12, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 12, 15, 23, 59, 59, tzinfo=timezone.utc),
        reward_pool_usdt=Decimal("200000")
    )

    print(f"Campaign: {campaign.symbol}")
    print(f"Period: {campaign.start_time} - {campaign.end_time}")
    print()

    # 这里需要实际的用户交易数据
    # 示例中使用空列表
    user_trades = []

    market_stats, user_stats, reward = await get_campaign_report(
        campaign,
        user_trades,
        aster_price_usdt=Decimal("0.714")
    )

    print("Market Stats:")
    print(f"  Total Trades: {market_stats.total_trades:,}")
    print(f"  Total Volume: ${market_stats.total_volume_usdt:,.2f}")
    print(f"  Total Fee: ${market_stats.total_fee_usdt:,.2f}")
    print()

    print("User Stats:")
    print(f"  Total Trades: {user_stats.total_trades:,}")
    print(f"  Total Volume: ${user_stats.total_volume_usdt:,.2f}")
    print(f"  Total Fee: {user_stats.total_fee_aster:.2f} ASTER")
    print()

    print("Reward Estimate:")
    print(f"  Your Share: {reward.user_share_pct:.4f}%")
    print(f"  Expected Reward: ${reward.expected_reward_usdt:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
