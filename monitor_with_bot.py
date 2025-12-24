#!/usr/bin/env python3
"""
Aster spot monitor with Telegram Bot:
- Decrypt per-account env (gpg) to load API/Telegram credentials.
- Fetch balances and quote price to report PnL and fee burn.
- Check if trading process is running (by env-file in cmdline).
- Send Telegram summary every hour automatically.
- Support Telegram Bot commands:
  /status - Get current status summary
  /stats  - Get detailed statistics
  /help   - Show help message
"""

import asyncio
import hmac
import hashlib
import os
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import aiohttp
import csv
import argparse
import yaml
import resource

# Import campaign calculator module
from campaign_calculator import (
    CampaignConfig,
    MarketStats,
    UserStats,
    RewardEstimate,
    get_campaign_report,
    calculate_user_stats,
)


TELEGRAM_TIMEOUT = 10
ASTER_BASE = "https://sapi.asterdex.com"
DEFAULT_CAMPAIGNS_FILE = "monitor_campaigns.csv"
SYMBOLS: List[str] = []
CAMPAIGNS: Dict[str, CampaignConfig] = {}  # symbol -> campaign config
ENV_CACHE: Dict[str, Dict[str, str]] = {}
VOLUME_STATE_DIR = Path("logs")
VOLUME_STATE_DIR.mkdir(exist_ok=True)
VOLUME_START_MS = int(datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


@dataclass
class AccountConfig:
    name: str
    config_file: str  # YAML config file (e.g., "config/account.yaml" or "config/account.yaml.gpg")


# Global accounts list, loaded from config file
ACCOUNTS: List[AccountConfig] = []


def load_accounts_config(config_file: str) -> List[AccountConfig]:
    """
    Load accounts from YAML config file.

    Example config (accounts.yaml):
    ```
    accounts:
      - name: account1
        config_file: config/account1.yaml.gpg
      - name: account2
        config_file: config/account2.yaml
    ```
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Accounts config file not found: {config_file}")

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    accounts = []
    for acc in data.get('accounts', []):
        accounts.append(AccountConfig(
            name=acc['name'],
            config_file=acc['config_file'],
        ))

    return accounts


class AsterAPI:
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.session: Optional[aiohttp.ClientSession] = None

    def __del__(self):
        """Clear sensitive data when object is destroyed."""
        if hasattr(self, 'api_key'):
            self.api_key = None
        if hasattr(self, 'secret'):
            self.secret = None

    def _sign(self, params: Dict[str, str]) -> str:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method: str, path: str, params: Optional[Dict[str, str]] = None, sign: bool = False):
        params = params or {}
        headers = {"X-MBX-APIKEY": self.api_key}
        if sign:
            params["timestamp"] = str(int(time.time() * 1000))
            params["signature"] = self._sign(params)

        url = f"{ASTER_BASE}{path}"
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(trust_env=True)
        if method == "GET":
            async with self.session.get(url, params=params, headers=headers, timeout=15) as resp:
                text = await resp.text()
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "5")
                    await asyncio.sleep(float(retry_after))
                    raise RuntimeError(f"{resp.status} {text[:200]}")
                if resp.status != 200:
                    raise RuntimeError(f"{resp.status} {text[:200]}")
                return await resp.json()
        else:
            raise RuntimeError(f"Unsupported method: {method}")

    async def account(self):
        return await self._request("GET", "/api/v1/account", sign=True)

    async def book_ticker(self, symbol: str):
        return await self._request("GET", "/api/v1/ticker/bookTicker", params={"symbol": symbol}, sign=False)

    async def user_trades(self, **params):
        params.setdefault("limit", 1000)
        return await self._request("GET", "/api/v1/userTrades", params=params, sign=True)


def load_campaigns(file_path: str) -> Dict[str, CampaignConfig]:
    """
    Load campaign configurations from CSV file.

    Format: symbol,start_time_utc,end_time_utc,reward_pool_usdt,fee_type,reward_cap
    Example: BIOUSDT,2025-12-01 12:00:00,2025-12-15 23:59:59,200000,buy_only,0
    """
    campaigns = {}

    if not os.path.exists(file_path):
        print(f"Info: {file_path} not found, skipping campaign tracking")
        return campaigns

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            # Skip empty lines and comments
            if not row or (row[0] and row[0].strip().startswith('#')):
                continue

            try:
                if len(row) < 4:
                    print(f"Warning: Line {line_num} has insufficient columns, skipping")
                    continue

                symbol = row[0].strip().upper()
                start_str = row[1].strip()
                end_str = row[2].strip()
                reward_usdt = Decimal(row[3].strip())

                # Parse optional fields (fee_type, reward_cap)
                fee_type = "buy_only"  # default
                reward_cap = Decimal(0)  # default: no cap
                if len(row) >= 5 and row[4].strip():
                    fee_type = row[4].strip().lower()
                if len(row) >= 6 and row[5].strip():
                    reward_cap = Decimal(row[5].strip())

                # Parse datetime (support both "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD")
                def parse_dt(s: str) -> datetime:
                    s = s.strip()
                    if ' ' in s:
                        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    else:
                        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

                start_time = parse_dt(start_str)
                end_time = parse_dt(end_str)

                campaigns[symbol] = CampaignConfig(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    reward_pool_usdt=reward_usdt,
                    fee_type=fee_type,
                    reward_cap=reward_cap
                )

            except Exception as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    if campaigns:
        print(f"Loaded {len(campaigns)} campaign(s): {', '.join(campaigns.keys())}")

    return campaigns


def parse_base_asset(symbol: str) -> str:
    """Extract base asset from symbol like BIOUSDT."""
    upper = symbol.upper()
    # 注意：长的后缀要先匹配，避免 "USD" 匹配到 "USDT"
    for quote in ("USDT", "USDC", "USD1", "BUSD"):
        if upper.endswith(quote):
            return upper[: -len(quote)]
    return upper


def parse_quote_asset(symbol: str) -> str:
    """Extract quote asset from symbol like RAVEUSD1 -> USD1."""
    upper = symbol.upper()
    # 注意：长的后缀要先匹配
    for quote in ("USDT", "USDC", "USD1", "BUSD"):
        if upper.endswith(quote):
            return quote
    return "USDT"  # default


def load_yaml_config(config_file: str, cache: bool = False) -> Dict[str, Any]:
    """Load and decrypt YAML config file (supports .gpg encrypted files).

    Args:
        config_file: Path to YAML config file
        cache: If True, cache in memory. If False (default), reload each time for better security.

    Security: By default, configs are not cached to minimize exposure time in memory.
    """
    if cache and config_file in ENV_CACHE:
        return ENV_CACHE[config_file]

    try:
        if config_file.endswith('.gpg'):
            # Decrypt GPG file
            output = subprocess.check_output(["gpg", "-dq", config_file], text=True)
            config = yaml.safe_load(output)
            # Clear the decrypted output from memory
            del output
        else:
            # Plain YAML file
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

        if cache:
            ENV_CACHE[config_file] = config

        return config
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to decrypt {config_file}: {e}")
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {config_file}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML {config_file}: {e}")


def process_running(env_file: str) -> bool:
    """Check if a trading process using the env file is running."""
    try:
        subprocess.check_output(["pgrep", "-f", env_file])
        return True
    except subprocess.CalledProcessError:
        return False


async def send_telegram(token: str, chat_id: str, text: str, parse_mode: str = "HTML"):
    """Send a message to Telegram."""
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(api_url, json=payload, timeout=TELEGRAM_TIMEOUT) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Telegram send failed: {resp.status} {body[:200]}")


async def set_bot_commands(token: str):
    """Set bot command menu."""
    api_url = f"https://api.telegram.org/bot{token}/setMyCommands"
    commands = [
        {"command": "status", "description": "📊 Get current status summary"},
        {"command": "stats", "description": "📈 Get detailed statistics"},
        {"command": "help", "description": "❓ Show help message"},
    ]
    payload = {"commands": commands}
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(api_url, json=payload, timeout=TELEGRAM_TIMEOUT) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"Warning: Failed to set bot commands: {resp.status} {body[:200]}")


async def get_telegram_updates(token: str, offset: int = 0) -> List[Dict]:
    """Get updates from Telegram using Long Polling."""
    api_url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset, "timeout": 30}
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(api_url, params=params, timeout=35) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("result", [])
    except asyncio.TimeoutError:
        return []
    except Exception as e:
        print(f"Error getting updates: {e}")
        return []


def volume_state_path(name: str, symbol: str) -> Path:
    return VOLUME_STATE_DIR / f"{symbol.lower()}_volume_{name}.json"


def load_volume_state(name: str, symbol: str) -> Optional[Dict[str, Any]]:
    path = volume_state_path(name, symbol)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["cum_quote_qty"] = Decimal(data.get("cum_quote_qty", "0"))
    return data


def save_volume_state(name: str, symbol: str, state: Dict[str, Any]) -> None:
    path = volume_state_path(name, symbol)
    to_save = state.copy()
    to_save["cum_quote_qty"] = str(to_save.get("cum_quote_qty", "0"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


async def init_volume(api: AsterAPI, name: str, symbol: str) -> Dict[str, Any]:
    """Initial scan of all historical trades using fromId pagination."""
    cum = Decimal("0")
    last_id: Optional[int] = None
    from_id = 0  # 从最早的交易开始

    # 使用 fromId 分页（不使用时间范围，从头拉取所有历史）
    while True:
        trades = await api.user_trades(symbol=symbol, fromId=from_id, limit=1000)

        if not trades:
            break

        for t in trades:
            cum += Decimal(t["quoteQty"])
            tid = t["id"]
            if last_id is None or tid > last_id:
                last_id = tid

        # 如果返回少于1000条，说明已经拉完
        if len(trades) < 1000:
            break

        # 继续拉取下一页
        from_id = trades[-1]["id"] + 1

    if last_id is None:
        last_id = -1

    state = {
        "symbol": symbol,
        "start_time_ms": VOLUME_START_MS,
        "last_trade_id": int(last_id),
        "cum_quote_qty": cum,
    }
    save_volume_state(name, symbol, state)
    return state


async def update_volume(api: AsterAPI, name: str, symbol: str, state: Dict[str, Any]) -> Decimal:
    """Incremental update using fromId."""
    last_id = state.get("last_trade_id", -1)
    cum: Decimal = state.get("cum_quote_qty", Decimal("0"))
    from_id = last_id + 1 if last_id is not None and last_id >= 0 else None
    max_id = last_id

    if from_id is None:
        trades = []
    else:
        while True:
            trades = await api.user_trades(symbol=symbol, fromId=from_id, limit=1000)
            if not trades:
                break
            for t in trades:
                cum += Decimal(t["quoteQty"])
                tid = t["id"]
                if max_id is None or tid > max_id:
                    max_id = tid
            if len(trades) < 1000:
                break
            from_id = trades[-1]["id"] + 1

    if max_id is not None:
        state["last_trade_id"] = int(max_id)
    state["cum_quote_qty"] = cum
    save_volume_state(name, symbol, state)
    return cum


async def calculate_volume(api: AsterAPI, name: str, symbol: str) -> Decimal:
    state = load_volume_state(name, symbol)

    if state is None or state.get("last_trade_id", -1) == -1:
        try:
            state = await init_volume(api, name, symbol)
        except Exception as e:
            print(f"[ERROR] Failed to init volume for {name}/{symbol}: {e}")
            return Decimal("0")

    try:
        return await update_volume(api, name, symbol, state)
    except Exception as e:
        print(f"[WARNING] Failed to update volume for {name}/{symbol}: {e}, using cached value")
        return state.get("cum_quote_qty", Decimal("0"))


async def fetch_user_trades_in_range(
    api: AsterAPI,
    symbol: str,
    start_time_ms: int,
    end_time_ms: int
) -> list:
    """
    获取用户在特定时间范围内的交易记录
    """
    all_trades = []
    from_id = 0

    while True:
        trades = await api.user_trades(symbol=symbol, fromId=from_id, limit=1000)
        if not trades:
            break

        # 过滤时间范围
        for t in trades:
            t_time = t["time"]
            if start_time_ms <= t_time <= end_time_ms:
                all_trades.append(t)
            elif t_time > end_time_ms:
                # 已经超过结束时间，不需要继续
                return all_trades

        if len(trades) < 1000:
            break

        from_id = trades[-1]["id"] + 1

    return all_trades


async def gather_account_state(cfg: AccountConfig, api_cache: Dict[str, AsterAPI], symbol: str) -> Tuple[str, Dict[str, Decimal], str, Dict[str, str]]:
    # Load from YAML config file (no cache for security)
    config = load_yaml_config(cfg.config_file, cache=True)  # Cache config to avoid re-decrypting every hour
    api_key = config.get("exchange", {}).get("api_key")
    secret = config.get("exchange", {}).get("secret_key")
    tg_token = config.get("risk", {}).get("telegram_token")
    tg_chat = config.get("risk", {}).get("telegram_chat_id")

    if not api_key or not secret:
        raise RuntimeError(f"{cfg.name}: missing ASTER keys")

    # Reuse existing API instance to avoid storing keys multiple times
    api = api_cache.get(cfg.name)
    if api is None:
        api = AsterAPI(api_key, secret)
        api_cache[cfg.name] = api

    # Clear sensitive data from memory immediately after use
    del api_key, secret, config
    account = await api.account()
    ticker = await api.book_ticker(symbol)
    bid = Decimal(ticker.get("bidPrice", "0"))
    ask = Decimal(ticker.get("askPrice", "0"))
    mid = (bid + ask) / 2 if bid and ask else Decimal("0")

    base_asset = parse_base_asset(symbol)
    quote_asset = parse_quote_asset(symbol)

    # balances
    bals = {b["asset"]: Decimal(b.get("free", "0")) + Decimal(b.get("locked", "0")) for b in account.get("balances", [])}
    usdt = bals.get("USDT", Decimal("0"))
    aster = bals.get("ASTER", Decimal("0"))
    base_asset_bal = bals.get(base_asset, Decimal("0"))

    # 计算所有稳定币的总 USDT 价值
    total_stable_in_usdt = usdt  # 从 USDT 开始

    # 添加 USD1 余额（如果有）
    usd1_bal = bals.get("USD1", Decimal("0"))
    if usd1_bal > 0:
        try:
            usd1_ticker = await api.book_ticker("USD1USDT")
            usd1_bid = Decimal(usd1_ticker.get("bidPrice", "1"))
            usd1_ask = Decimal(usd1_ticker.get("askPrice", "1"))
            usd1_rate = (usd1_bid + usd1_ask) / 2 if usd1_bid and usd1_ask else Decimal("1")
            total_stable_in_usdt += usd1_bal * usd1_rate
        except Exception as e:
            print(f"[WARNING] Failed to get USD1USDT rate: {e}, assuming 1:1")
            total_stable_in_usdt += usd1_bal

    # 可以继续添加其他稳定币（USDC 等）
    usdc_bal = bals.get("USDC", Decimal("0"))
    if usdc_bal > 0:
        total_stable_in_usdt += usdc_bal  # USDC ≈ 1:1 USDT

    volume = await calculate_volume(api, cfg.name, symbol)

    # Check if trading process is running
    status = "UP" if process_running(cfg.config_file) else "DOWN"
    stats = {
        "usdt": total_stable_in_usdt,  # Total balance in USDT (including converted USD1, USDC, etc.)
        "aster": aster,
        "base": base_asset_bal,
        "price": mid,
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
    }
    tg = {"token": tg_token, "chat": tg_chat}
    return status, stats, cfg.name, tg


async def collect_all_stats(api_cache: Dict[str, AsterAPI]) -> Dict[str, List[Tuple]]:
    """Collect stats for all symbols and accounts."""
    all_states = {}

    for symbol in SYMBOLS:
        states = []
        for cfg in ACCOUNTS:
            try:
                state = await gather_account_state(cfg, api_cache, symbol)
                states.append(state)
            except Exception as exc:
                states.append(("ERROR", {
                    "base": Decimal("0"), "usdt": Decimal("0"), "price": Decimal("0"),
                    "bid": Decimal("0"), "ask": Decimal("0"), "volume": Decimal("0"),
                    "aster": Decimal("0"), "base_asset": parse_base_asset(symbol),
                    "quote_asset": parse_quote_asset(symbol),
                }, cfg.name, {"token": None, "chat": None}))
                print(f"[{cfg.name}][{symbol}] monitor failed: {exc}")

        all_states[symbol] = states

    return all_states


async def get_campaign_reports(api_cache: Dict[str, AsterAPI]) -> Dict[str, Dict[str, Any]]:
    """
    为每个有campaign的symbol生成报告

    Returns:
        {symbol: {account_name: {user_stats, market_stats, reward}}}
    """
    campaign_reports = {}

    for symbol, campaign in CAMPAIGNS.items():
        print(f"\n📊 Processing campaign: {symbol}")
        campaign_reports[symbol] = {}

        start_ms = int(campaign.start_time.timestamp() * 1000)
        end_ms = int(campaign.end_time.timestamp() * 1000)

        # 为每个账户计算
        for cfg in ACCOUNTS:
            api = api_cache.get(cfg.name)
            if not api:
                continue

            try:
                # 获取用户交易
                print(f"  Fetching trades for {cfg.name}...")
                user_trades = await fetch_user_trades_in_range(api, symbol, start_ms, end_ms)

                if not user_trades:
                    print(f"  No trades found for {cfg.name}")
                    continue

                # 计算用户统计
                user_stats = calculate_user_stats(user_trades)

                # 计算 cost（刷单损耗）
                # buy_usdt: 买入花费的 USDT
                # sell_usdt: 卖出获得的 USDT
                # net_qty: 净持仓变化（买入数量 - 卖出数量）
                buy_usdt = Decimal(0)
                sell_usdt = Decimal(0)
                buy_qty = Decimal(0)
                sell_qty = Decimal(0)

                for trade in user_trades:
                    quote_qty = Decimal(str(trade["quoteQty"]))
                    qty = Decimal(str(trade["qty"]))
                    is_buyer = trade.get("buyer", False)  # API 返回的字段名是 "buyer"

                    if is_buyer:
                        buy_usdt += quote_qty
                        buy_qty += qty
                    else:
                        sell_usdt += quote_qty
                        sell_qty += qty

                net_usdt = sell_usdt - buy_usdt  # 正值表示卖出多于买入
                net_qty = buy_qty - sell_qty      # 正值表示买多于卖（有持仓）

                # 获取市场数据和计算奖励（自动获取ASTER实时价格）
                print(f"  Calculating market stats...")
                market_stats, _, reward = await get_campaign_report(
                    campaign,
                    user_trades,
                    use_cache=True
                )

                campaign_reports[symbol][cfg.name] = {
                    'user_stats': user_stats,
                    'market_stats': market_stats,
                    'reward': reward,
                    'net_usdt': net_usdt,
                    'net_qty': net_qty,
                }

                print(f"  ✅ {cfg.name}: {user_stats.total_trades} trades, {user_stats.total_fee_aster:.2f} ASTER fee")

            except Exception as e:
                print(f"  ❌ Error processing {cfg.name}: {e}")
                continue

    return campaign_reports


def format_message(all_states, title: str = "📊 Aster Multi-Asset Monitor", campaign_reports: Optional[Dict] = None):
    """Format HTML message for multiple symbols."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"<b>{title}</b>",
        f"⏰ {ts}",
        ""
    ]

    # Collect account data
    account_data = {}
    for symbol, states in all_states.items():
        for status, stats, name, _tg in states:
            if name not in account_data:
                account_data[name] = {
                    'symbols': {},
                    'usdt': stats['usdt'],
                    'aster': stats['aster'],
                    'status': status
                }
            account_data[name]['symbols'][symbol] = stats

    # 获取 ASTER 价格（用于换算）
    aster_price = Decimal("0.70")  # 默认值
    for symbol, states in all_states.items():
        for _status, stats, _name, _tg in states:
            if stats.get('aster', Decimal("0")) > 0:
                # 尝试从 campaign_reports 获取 ASTER 价格
                break

    # 按账户分组，显示每个 ticker 的详细统计
    lines.append("<b>📈 Trading Stats</b>")

    account_totals = {}  # {account_name: total_net_profit}

    for name in sorted(account_data.keys()):
        data = account_data[name]
        icon = '🟢' if data['status'] == 'UP' else '🔴'
        lines.append(f"\n{icon} <b>{name.upper()}</b>")

        account_total_net = Decimal("0")

        for symbol in sorted(data['symbols'].keys()):
            stats = data['symbols'][symbol]
            base_asset = stats['base_asset']
            base_qty = stats['base']
            price = stats['price']

            # 从 campaign_reports 获取数据
            campaign = CAMPAIGNS.get(symbol)
            account_report = None
            if campaign_reports and symbol in campaign_reports:
                account_report = campaign_reports[symbol].get(name)

            if account_report:
                user = account_report['user_stats']
                market = account_report['market_stats']
                reward = account_report['reward']
                net_usdt = account_report.get('net_usdt', Decimal(0))
                net_qty = account_report.get('net_qty', Decimal(0))

                # 交易量
                volume = user.total_volume_usdt
                # 手续费（ASTER）
                fee_aster = user.total_fee_aster  # 总手续费（用于计算 Net）

                # 获取 ASTER 价格
                if reward.expected_reward_aster > 0:
                    aster_price_calc = reward.expected_reward_usdt / reward.expected_reward_aster
                else:
                    aster_price_calc = aster_price

                fee_usdt_value = fee_aster * aster_price_calc

                # 预期奖励
                reward_usdt = reward.expected_reward_usdt

                # Cost = 刷单损耗
                # = 买入花费 - 卖出获得 - 持仓价值
                # = -net_usdt - net_qty × price
                # cost > 0 表示亏损，cost < 0 表示盈利
                cost = -net_usdt - net_qty * price

                # Net = Reward - Fee - Cost
                net_profit = reward_usdt - fee_usdt_value - cost
                account_total_net += net_profit

                # 计算 R/M 比值
                reward_pool = reward.reward_pool_usdt
                # 根据 fee_type 选择使用哪个市场手续费
                if campaign and campaign.fee_type == "total":
                    market_fee = market.market_total_fee_usdt  # 买+卖
                else:
                    market_fee = market.buy_fee_usdt  # 只算买方
                r_m_ratio = reward_pool / market_fee if market_fee > 0 else Decimal(0)

                # R/M > 1.125 显示绿色，否则红色
                ticker_color = "🟢" if r_m_ratio > Decimal("1.125") else "🔴"

                lines.append(f"  ─────────────────────")
                lines.append(f"  {ticker_color} <b>{base_asset}</b> (R/M: {r_m_ratio:.2f})")
                lines.append(f"  <code>Vol:   {volume:>11,.2f} USDT</code>")
                lines.append(f"  <code>Fee:   {fee_aster:>11.2f} ASTER</code>")
                lines.append(f"  <code>Cost:  {cost:>+11.2f} USDT</code>")
                lines.append(f"  <code>Share: {reward.user_share_pct:>6.4f}%</code> | Reward: ${reward_usdt:.2f}")
                lines.append(f"  <b><code>Net:   {net_profit:>+11.2f} USDT</code></b>")
                if base_qty > 0:
                    lines.append(f"  <code>Pos:   {base_qty:>11.2f} {base_asset}</code> @ {price:.5f}")

            else:
                # 没有 campaign 数据，只显示基本信息
                volume = stats.get('volume', Decimal('0'))
                lines.append(f"  <code>{base_asset}</code>")
                lines.append(f"    Vol: {volume:>12,.2f} USDT")
                if base_qty > 0:
                    lines.append(f"    Position: {base_qty:.2f} {base_asset} @ {price:.5f}")

        account_totals[name] = account_total_net

    # 账户汇总
    lines.append("")
    lines.append("<b>📊 Account Summary</b>")

    total_net_profit = Decimal("0")
    for name in sorted(account_data.keys()):
        data = account_data[name]
        net_profit = account_totals.get(name, Decimal("0"))
        total_net_profit += net_profit

        # 持仓汇总
        positions = []
        for symbol in sorted(data['symbols'].keys()):
            stats = data['symbols'][symbol]
            base_qty = stats['base']
            if base_qty > 0:
                positions.append(f"{stats['base_asset']}:{base_qty:.1f}")

        position_str = " | ".join(positions) if positions else "-"
        net_emoji = "💚" if net_profit >= 0 else "❤️"

        lines.append(
            f"<b>{name.upper()}</b>: "
            f"USDT {data['usdt']:.2f} | "
            f"Net {net_emoji}{net_profit:+.2f} | "
            f"Pos [{position_str}]"
        )

    # 总计
    lines.append("")
    total_emoji = "💚" if total_net_profit >= 0 else "❤️"
    lines.append(f"{total_emoji} <b>Total Net Profit: {total_net_profit:+.2f} USDT</b>")

    return "\n".join(lines)


async def handle_command(command: str, chat_id: str, token: str, api_cache: Dict[str, AsterAPI]):
    """Handle bot commands."""
    if command == "/start" or command == "/help":
        help_text = (
            "<b>📊 Aster Monitor Bot</b>\n\n"
            "Available commands:\n"
            "/status - Get current status summary\n"
            "/stats - Get detailed statistics\n"
            "/help - Show this help message\n\n"
            "The bot also sends automatic hourly reports."
        )
        await send_telegram(token, chat_id, help_text)

    elif command == "/status":
        try:
            all_states = await collect_all_stats(api_cache)
            campaign_reports = None
            if CAMPAIGNS:
                try:
                    campaign_reports = await get_campaign_reports(api_cache)
                except:
                    pass
            message = format_message(all_states, title="📊 Current Status", campaign_reports=campaign_reports)
            await send_telegram(token, chat_id, message)
        except Exception as e:
            await send_telegram(token, chat_id, f"❌ Error collecting stats: {str(e)}")

    elif command == "/stats":
        try:
            all_states = await collect_all_stats(api_cache)
            campaign_reports = None
            if CAMPAIGNS:
                try:
                    campaign_reports = await get_campaign_reports(api_cache)
                except:
                    pass
            message = format_message(all_states, title="📈 Detailed Statistics", campaign_reports=campaign_reports)
            await send_telegram(token, chat_id, message)
        except Exception as e:
            await send_telegram(token, chat_id, f"❌ Error collecting stats: {str(e)}")

    else:
        await send_telegram(token, chat_id, f"Unknown command: {command}\nUse /help to see available commands.")


async def telegram_bot_listener(token: str, api_cache: Dict[str, AsterAPI]):
    """Listen for Telegram bot commands."""
    offset = 0
    print("🤖 Telegram bot listener started")

    # Set bot commands menu
    await set_bot_commands(token)

    while True:
        try:
            updates = await get_telegram_updates(token, offset)

            for update in updates:
                offset = update["update_id"] + 1

                if "message" in update:
                    message = update["message"]
                    chat_id = str(message["chat"]["id"])
                    text = message.get("text", "")

                    if text.startswith("/"):
                        command = text.split()[0].lower()
                        print(f"📥 Received command: {command} from {chat_id}")
                        await handle_command(command, chat_id, token, api_cache)

            await asyncio.sleep(1)

        except Exception as e:
            print(f"❌ Bot listener error: {e}")
            await asyncio.sleep(5)


async def main(loop_forever: bool = True, enable_bot: bool = True):
    # Security: Disable core dumps to prevent API keys from being written to disk
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    apis = {}

    # Get Telegram token from first account
    bot_token = None
    for cfg in ACCOUNTS:
        try:
            config = load_yaml_config(cfg.config_file)
            bot_token = config.get("risk", {}).get("telegram_token")
            if bot_token:
                break
        except Exception:
            continue

    # Start bot listener in background if enabled
    bot_task = None
    if enable_bot and bot_token:
        bot_task = asyncio.create_task(telegram_bot_listener(bot_token, apis))
        print("✅ Telegram bot enabled - you can now use commands!")
    else:
        print("⚠️  Telegram bot disabled")

    try:
        while True:
            # Collect and send hourly report
            all_states = await collect_all_stats(apis)

            # Get campaign reports if any campaigns configured
            campaign_reports = None
            if CAMPAIGNS:
                try:
                    campaign_reports = await get_campaign_reports(apis)
                except Exception as e:
                    print(f"⚠️  Failed to get campaign reports: {e}")

            # Send summary
            summary = format_message(all_states, campaign_reports=campaign_reports)
            target = None

            for symbol_states in all_states.values():
                for _status, _stats, _name, tg in symbol_states:
                    if tg.get("token") and tg.get("chat"):
                        target = tg
                        break
                if target:
                    break

            if target:
                try:
                    await send_telegram(target["token"], target["chat"], summary)
                    print(f"✅ Hourly report sent at {datetime.now()}")
                except Exception as exc:
                    print(f"❌ Failed to send summary: {exc}")
            else:
                print("⚠️  No Telegram config found; summary not sent")

            if not loop_forever:
                break

            # Wait 1 hour
            await asyncio.sleep(3600)

    finally:
        # Clean up
        if bot_task:
            bot_task.cancel()
        for api in apis.values():
            if getattr(api, "session", None) and not api.session.closed:
                await api.session.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Aster spot monitor with Telegram bot")
    parser.add_argument("--accounts", default="accounts.yaml", help="Accounts config file (default: accounts.yaml)")
    parser.add_argument("--symbol", default=None, help="Single trading symbol (overrides config file)")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--no-bot", action="store_true", help="Disable Telegram bot commands")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load accounts configuration
    try:
        ACCOUNTS.clear()
        ACCOUNTS.extend(load_accounts_config(args.accounts))
        print(f"Loaded {len(ACCOUNTS)} account(s): {', '.join(a.name for a in ACCOUNTS)}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"\nPlease create {args.accounts} with format:")
        print("accounts:")
        print("  - name: myaccount")
        print("    config_file: config/myaccount.yaml")
        exit(1)

    if not ACCOUNTS:
        print(f"❌ No accounts configured in {args.accounts}")
        exit(1)

    # Load campaigns (symbols are derived from campaigns)
    CAMPAIGNS.clear()
    CAMPAIGNS.update(load_campaigns(DEFAULT_CAMPAIGNS_FILE))

    # Set symbols from campaigns or command line
    if args.symbol:
        SYMBOLS = [args.symbol.upper()]
        print(f"Monitoring single symbol: {args.symbol}")
    else:
        SYMBOLS = list(CAMPAIGNS.keys())
        if not SYMBOLS:
            print(f"❌ No campaigns found in {DEFAULT_CAMPAIGNS_FILE}")
            exit(1)
        print(f"Monitoring {len(SYMBOLS)} symbols from {DEFAULT_CAMPAIGNS_FILE}: {', '.join(SYMBOLS)}")

    print()

    try:
        asyncio.run(main(loop_forever=not args.once, enable_bot=not args.no_bot))
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
