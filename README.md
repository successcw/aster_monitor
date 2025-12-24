# Aster DEX Campaign Monitor

监控 Aster DEX 交易活动挖矿的收益计算工具。

## 功能

- 实时计算 R/M 比值（奖池/市场手续费）
- 估算预期奖励和净收益
- 支持多账户、多 Campaign 监控
- Telegram Bot 推送

## 安装

```bash
pip install aiohttp pyyaml
```

## 配置

### 1. 账户配置

复制示例文件并修改：

```bash
cp accounts.example.yaml accounts.yaml
cp config.example.yaml config/myaccount.yaml
```

**accounts.yaml** - 账户列表：
```yaml
accounts:
  - name: main
    config_file: config/main.yaml
```

**config/myaccount.yaml** - API 密钥：
```yaml
api_key: "your_api_key"
secret: "your_secret"

risk:
  telegram_token: "bot_token"  # 可选
  telegram_chat: "chat_id"     # 可选
```

### 2. Campaign 配置 (monitor_campaigns.csv)

```csv
# symbol,start_time,end_time,reward_pool,fee_type,reward_cap
GUAUSDT,2025-12-03 13:00:00,2025-12-10 23:59:00,50000,total,0.03
```

参数说明：
- `fee_type`: `buy_only`（只按买方手续费分奖励）或 `total`（按总手续费分奖励）
- `reward_cap`: 每人最多拿奖池的百分比，如 `0.03` = 3%

## 使用

```bash
# 默认使用 accounts.yaml
python monitor_with_bot.py

# 指定配置文件
python monitor_with_bot.py --accounts myconfig.yaml

# 运行一次后退出
python monitor_with_bot.py --once

# 禁用 Telegram bot
python monitor_with_bot.py --no-bot
```

---

## 什么时候应该刷交易量？

**核心指标：R/M 比值**

```
R/M = 奖池 / 市场手续费
```

### 判断标准

| 条件 | 是否刷 | 原因 |
|------|--------|------|
| R/M > 盈亏线 | ✅ 刷 | 有利可图 |
| R/M < 盈亏线 | ❌ 不刷 | 会亏损 |

### 不同 fee_type 的盈亏线和最优策略

| fee_type | 最优策略 | 盈亏线 |
|----------|---------|--------|
| `total` | 全 taker | R/M > 1.0 |
| `buy_only` | 买 taker + 卖 maker | R/M > 1.125 |

### 为什么？

**fee_type = total**（按总手续费分奖励）：
- 手续费越高 → 奖励越多
- 全 taker 手续费最高，绝对利润最大

**fee_type = buy_only**（只按买方手续费分奖励）：
- 只有买方手续费能获得奖励
- 卖方手续费是纯成本，应最小化
- 买单用 taker（最大化奖励），卖单用 maker（最小化成本）

### 收益计算公式

```
Reward = 我的手续费 × R/M
Net = Fee × (R/M - 1) - Cost
```

### 实战建议

1. **Campaign 开始时 R/M 最高**，越早参与越好
2. **持续监控 R/M**，低于盈亏线立即停止
3. **自己对敲可以降低 Cost**，提高净收益
4. **注意 reward_cap**，份额过大会被截断

---

## License

MIT
