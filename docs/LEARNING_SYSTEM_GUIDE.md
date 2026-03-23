# Continuous Learning System Guide

The TradingAgents learning system enables the trading system to improve from experience. This guide covers all 5 phases of the learning pipeline.

---

## Overview

The learning system transforms trade outcomes into lessons that improve future decisions:

```
Trade Closes → Outcome Analysis → Reflection → Memory Storage → Agent Retrieval
```

### The 5 Phases

| Phase | Component | Purpose |
|-------|-----------|---------|
| 1 | Reward Signals | Calculate multi-factor trade quality scores |
| 2 | Regime Detection | Detect market conditions (trend, volatility) |
| 3 | RAG Decision Support | Query similar historical trades |
| 4 | Online Learning | Update agent weights based on performance |
| 5 | Risk Guardrails | Circuit breakers and position limits |

---

## Quick Start

### Check System Status

```bash
# View complete learning system status
python -m cli.main learning-status

# Check risk guardrails
python -m cli.main risk-status

# Detect market regime
python -m cli.main regime --symbol XAUUSD

# Find similar historical trades
python -m cli.main similar-trades --symbol XAUUSD --direction BUY

# Update patterns and weights (every 30 trades)
python -m cli.main update-patterns
```

### Python API

```python
# Phase 1: Reward Signals
from tradingagents.learning import RewardCalculator, PortfolioStateTracker

# Phase 2: Regime Detection
from tradingagents.indicators import RegimeDetector

# Phase 3: RAG
from tradingagents.learning import TradeSimilaritySearch

# Phase 4: Pattern Analysis & Weights
from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

# Phase 5: Risk Guardrails
from tradingagents.risk import RiskGuardrails
```

---

## Phase 1: Reward Signals

Calculates a composite reward score for each closed trade.

### Components

| Factor | Weight | Measures |
|--------|--------|----------|
| Realized R:R | 40% | Actual risk-reward achieved |
| Sharpe Contribution | 30% | Impact on portfolio Sharpe ratio |
| Drawdown Impact | 30% | Effect on portfolio drawdown |

### Automatic Calculation

Rewards are calculated automatically when you close a trade:

```python
from tradingagents.trade_decisions import close_decision

close_decision(
    decision_id="XAUUSD_20260111_140000",
    exit_price=2690.0,
    exit_reason="tp-hit"
)

# Automatically calculates and stores:
# - pnl_percent: +1.51%
# - rr_realized: +2.0R
# - reward_signal: +1.85
```

### Reward Interpretation

- `> +1.5`: Excellent trade
- `+0.5 to +1.5`: Good trade
- `-0.5 to +0.5`: Neutral
- `< -0.5`: Poor trade

---

## Phase 2: Regime Detection

Detects current market conditions for context-aware decisions.

### Regime Types

| Regime | Values | Trading Implication |
|--------|--------|---------------------|
| Market | trending-up, trending-down, ranging | Trade direction |
| Volatility | low, normal, high, extreme | Position sizing |
| Expansion | expansion, contraction, neutral | Breakout potential |

### Usage

```python
from tradingagents.indicators import RegimeDetector
import numpy as np

detector = RegimeDetector()
regime = detector.get_full_regime(high, low, close)

print(regime)
# {'market_regime': 'trending-up', 'volatility_regime': 'normal', ...}

# Check if favorable for trend trading
if detector.is_favorable_for_trend_trading(regime):
    print("Good for trend following")

# Get risk adjustment factor
adjustment = detector.get_risk_adjustment_factor(regime)
position_size = base_size * adjustment
```

### Store with Trades

```python
from tradingagents.trade_decisions import store_decision, set_decision_regime

decision_id = store_decision(symbol="XAUUSD", action="BUY", ...)
set_decision_regime(decision_id, regime)
```

---

## Phase 3: RAG Decision Support

Queries similar historical trades before making decisions.

### Similarity Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Symbol | 0.2 | Same trading instrument |
| Direction | 0.2 | Same BUY/SELL direction |
| Regime | 0.5 | Similar market conditions |
| Setup | varies | Setup type, confluence |

### Usage

```python
from tradingagents.learning import TradeSimilaritySearch

searcher = TradeSimilaritySearch()

result = searcher.find_similar_trades({
    "symbol": "XAUUSD",
    "direction": "BUY",
    "setup_type": "ob_bounce",
    "market_regime": "trending-up"
}, n_results=5)

stats = result['statistics']
print(f"Win rate: {stats['win_rate']*100:.1f}%")
print(f"Confidence adjustment: {stats['confidence_adjustment']:+.2f}")
```

### Confidence Adjustment

Based on historical performance:
- Strong history (>60% WR): +0.1 to +0.3
- Neutral history: 0
- Poor history (<40% WR): -0.1 to -0.3

---

## Phase 4: Online Learning

Updates agent weights based on performance every 30 trades.

### Pattern Analysis

```python
from tradingagents.learning import PatternAnalyzer

analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(lookback_days=30)

# Groups trades by:
# - setup_type: Which setups work best?
# - regime: Which market conditions are profitable?
# - session: Which trading sessions work best?
```

### Agent Weight Updates

```python
from tradingagents.learning import OnlineRLUpdater

updater = OnlineRLUpdater()

# Check if update needed
should_update, trades_since = updater.should_update()

if should_update:
    performances = updater.calculate_agent_performances()
    result = updater.update_weights(performances)
    print(updater.format_report(result))

# Get current weights for consensus
weights = updater.get_current_weights()
# {'bull': 0.45, 'bear': 0.25, 'market': 0.30}
```

### Weight Mechanics

- Default: bull=0.33, bear=0.33, market=0.34
- Updated every 30 closed trades
- Based on: win_rate × normalized_reward × sqrt(sample_size)
- Learning rate: 0.1 with momentum 0.9

---

## Phase 5: Risk Guardrails

Hard limits to prevent catastrophic losses.

### Limits

| Limit | Default | Purpose |
|-------|---------|---------|
| Daily Loss | 3.0% | Max loss per day |
| Consecutive Losses | 2 | Max losses in a row |
| Max Position Size | 2.0% | Per-trade risk cap |
| Cooldown | 24 hours | Timeout after breach |

### Usage

```python
from tradingagents.risk import RiskGuardrails

guardrails = RiskGuardrails(
    daily_loss_limit_pct=3.0,
    max_consecutive_losses=2,
    max_position_size_pct=2.0,
    cooldown_hours=24
)

# Before trading
can_trade, reason = guardrails.check_can_trade(account_balance)
if not can_trade:
    print(f"Trading blocked: {reason}")
    return

# Validate position size
is_valid, reason, adjusted = guardrails.validate_position_size(
    requested_size, account_balance
)

# After trade closes
result = guardrails.record_trade_result(was_win, pnl_pct, balance)
if result['breach_triggered']:
    print(f"Circuit breaker: {result['breach_type']}")
```

---

## Trade Assessment Pipeline

When a trade closes, automatic assessment occurs:

```
1. Calculate P&L
   └─ Entry/exit prices → pnl_percent

2. Calculate Risk-Reward
   └─ (exit - entry) / (entry - stop) → rr_realized

3. Calculate Reward Signal
   ├─ RR Component (40%)
   ├─ Sharpe Contribution (30%)
   └─ Drawdown Impact (30%)

4. Update Portfolio State
   ├─ Equity curve
   ├─ Returns history
   └─ Sharpe ratio

5. Update Risk Guardrails
   ├─ Consecutive losses
   └─ Daily loss tracking

6. Store for Learning
   └─ RAG database for future queries
```

---

## Memory System

### Tier System

| Tier | Criteria | Weight | Decay |
|------|----------|--------|-------|
| SHORT | Default, <2% returns | 0.5 | 30 days |
| MID | 3+ refs or 2-5% returns | 0.3 | Slower |
| LONG | 5%+ returns or promoted | 0.2 | Persistent |

### Per-Agent Retrieval

| Agent | Matches | Min Confidence | Half-life |
|-------|---------|----------------|-----------|
| bull_researcher | 3 | 0.4 | 45 days |
| bear_researcher | 3 | 0.4 | 45 days |
| trader | 4 | 0.5 | 30 days |
| invest_judge | 3 | 0.5 | 60 days |
| risk_manager | 2 | 0.6 | 90 days |

### SMC Pattern Memory

Stores per-pattern outcomes with metadata:
- setup_type, direction, was_win, returns_pct
- entry_zone, zone_strength, with_trend
- sl_placement, tp_placement, exit_type
- Auto-generated lesson text

---

## Daily Workflow

### Morning

```bash
# Check if trading allowed
python -m cli.main risk-status

# Check system status
python -m cli.main learning-status

# Detect regime
python -m cli.main regime --symbol XAUUSD
```

### Before Trading

```bash
# Find similar trades
python -m cli.main similar-trades --symbol XAUUSD --direction BUY --setup ob_bounce
```

### After 30 Trades

```bash
# Update patterns and weights
python -m cli.main update-patterns
```

---

## Key Files

| File | Purpose |
|------|---------|
| `tradingagents/learning/reward.py` | Reward calculation |
| `tradingagents/indicators/regime.py` | Regime detection |
| `tradingagents/learning/trade_similarity.py` | RAG search |
| `tradingagents/learning/pattern_analyzer.py` | Pattern clustering |
| `tradingagents/learning/online_rl.py` | Agent weight updates |
| `tradingagents/risk/guardrails.py` | Risk limits |
| `tradingagents/agents/utils/memory.py` | Memory system |
| `tradingagents/graph/reflection.py` | Reflection pipeline |
| `tradingagents/trade_decisions.py` | Trade storage/closing |

---

## Troubleshooting

### "No similar trades found"
Need at least 5-10 closed trades with metadata.

### "Pattern analysis returns empty"
Reduce `min_cluster_size` or wait for more trades.

### "Weights not updating"
Need 30 closed trades since last update.

### "Circuit breaker won't clear"
Wait for cooldown to expire (default 24h).

---

*For technical implementation details, see `.claude/rules/learning-reflection-architecture.md`*
