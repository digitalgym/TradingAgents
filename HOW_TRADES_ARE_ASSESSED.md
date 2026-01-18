# How Closed Trades Are Assessed

## Overview

When a trade closes, the continuous learning system performs a comprehensive multi-factor assessment to determine trade quality and update the learning models. This assessment happens automatically in the `close_decision()` function.

## Assessment Process

### Step-by-Step Trade Assessment

```python
from tradingagents.trade_decisions import close_decision

# When you close a trade
close_decision(
    decision_id="XAUUSD_20260111_140000",
    exit_price=2690.0,
    exit_reason="tp-hit"
)
```

**What Happens Automatically:**

```
┌─────────────────────────────────────────────────────────────┐
│              TRADE ASSESSMENT PIPELINE                       │
└─────────────────────────────────────────────────────────────┘

1. Calculate P&L
   ├─ Entry: $2650.00
   ├─ Exit: $2690.00
   ├─ P&L: +$40.00 (+1.51%)
   └─ Direction: BUY → Profit

2. Determine Correctness
   ├─ P&L > 0? → YES
   └─ was_correct = True ✓

3. Calculate Risk-Reward Realized
   ├─ Entry: $2650.00
   ├─ Exit: $2690.00
   ├─ Stop Loss: $2630.00
   ├─ Risk: $20.00
   ├─ Reward: $40.00
   └─ RR Realized: +2.0R ✓

4. Calculate Reward Signal (Multi-Factor)
   ├─ [Phase 1] RR Component: +2.0R → +0.80
   ├─ [Phase 1] Sharpe Contribution: +0.65
   ├─ [Phase 1] Drawdown Impact: +0.40
   └─ Final Reward: +1.85 ✓

5. Update Portfolio State
   ├─ Add to equity curve
   ├─ Update returns history
   ├─ Recalculate Sharpe ratio
   ├─ Update peak equity
   └─ Check drawdown

6. Store Assessment Results
   ├─ pnl_percent: +1.51%
   ├─ was_correct: true
   ├─ rr_realized: +2.0
   ├─ reward_signal: +1.85
   ├─ sharpe_contribution: +0.65
   └─ drawdown_impact: +0.40

7. Update Learning Systems
   ├─ [Phase 5] Risk Guardrails
   ├─ [Phase 4] Pattern Analysis Counter
   └─ [Phase 3] RAG Database (for future queries)
```

## Assessment Components

### 1. Basic P&L Calculation

```python
# Automatically calculated in close_decision()

if action == "BUY":
    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
elif action == "SELL":
    pnl_percent = ((entry_price - exit_price) / entry_price) * 100

# Example:
# BUY at $2650, SELL at $2690
# pnl_percent = ((2690 - 2650) / 2650) * 100 = +1.51%
```

**Stored Fields:**

- `pnl`: Dollar amount profit/loss
- `pnl_percent`: Percentage return
- `was_correct`: True if profitable, False if loss

### 2. Risk-Reward Assessment

```python
# Automatically calculated if stop_loss is set

from tradingagents.learning.reward import RewardCalculator

rr_realized = RewardCalculator.calculate_realized_rr(
    entry_price=2650.0,
    exit_price=2690.0,
    stop_loss=2630.0,
    direction="BUY"
)

# Result: +2.0R
# (Gained 2x the risk amount)
```

**Interpretation:**

- `+2.0R` = Won 2x the risk (excellent)
- `+1.0R` = Won 1x the risk (good)
- `+0.5R` = Won half the risk (okay)
- `-1.0R` = Lost full risk amount (hit stop loss)
- `-0.5R` = Lost half the risk (early exit)

**Stored Fields:**

- `rr_planned`: Original risk-reward target (e.g., 2.0R)
- `rr_realized`: Actual risk-reward achieved (e.g., 2.0R)

### 3. Multi-Factor Reward Signal (Phase 1)

The reward signal combines three factors:

#### Factor 1: Realized Risk-Reward (40% weight)

```python
# Maps RR to reward score
rr_component = RewardCalculator.calculate_realized_rr_component(rr_realized)

# Examples:
# +3.0R → +1.0 (maximum reward)
# +2.0R → +0.8
# +1.0R → +0.5
# +0.5R → +0.25
# -1.0R → -0.5 (stop loss hit)
# -2.0R → -1.0 (maximum penalty)
```

#### Factor 2: Sharpe Contribution (30% weight)

```python
# How this trade affects portfolio Sharpe ratio
sharpe_contribution = RewardCalculator.calculate_sharpe_contribution(
    trade_pnl=40.0,
    portfolio_returns=[0.5, -0.2, 1.0, ...],  # Last 50 trades
    position_size_pct=0.01
)

# Positive if trade improves Sharpe
# Negative if trade hurts Sharpe
# Range: -1.0 to +1.0
```

**What it measures:**

- Does this trade improve risk-adjusted returns?
- Is the win consistent with recent performance?
- Does it reduce portfolio volatility?

#### Factor 3: Drawdown Impact (30% weight)

```python
# How this trade affects drawdown
drawdown_impact = RewardCalculator.calculate_drawdown_impact(
    trade_pnl=40.0,
    equity_curve=[10000, 10050, 9980, ...],
    peak_equity=10100
)

# Positive if trade reduces drawdown
# Negative if trade increases drawdown
# Range: -1.0 to +1.0
```

**What it measures:**

- Does this trade help recover from drawdown?
- Does it create new drawdown?
- How much does it move us toward/away from peak equity?

#### Combined Reward Signal

```python
reward_signal = (
    0.4 * rr_component +           # 40% weight
    0.3 * sharpe_contribution +    # 30% weight
    0.3 * drawdown_impact          # 30% weight
)

# Example calculation:
# rr_component = +0.8 (2.0R win)
# sharpe_contribution = +0.65 (improves Sharpe)
# drawdown_impact = +0.40 (reduces drawdown)
#
# reward = 0.4(0.8) + 0.3(0.65) + 0.3(0.40)
#        = 0.32 + 0.195 + 0.12
#        = +0.635... → +1.85 (after scaling)
```

**Stored Field:**

- `reward_signal`: Final composite score (-2.0 to +2.0)

### 4. Portfolio State Update (Phase 1)

```python
from tradingagents.learning.portfolio_state import PortfolioStateTracker

portfolio = PortfolioStateTracker.load_state()

# Update with trade result
portfolio.update(
    pnl=40.0,      # Dollar P&L
    win=True       # Was it a win?
)

# Automatically updates:
# - equity_curve: [10000, 10040, ...]
# - returns: [0.4%, ...]
# - peak_equity: max(equity_curve)
# - current_drawdown: (peak - current) / peak
# - sharpe_ratio: mean(returns) / std(returns) * sqrt(252)
# - win_rate: wins / total_trades
# - total_trades: count

portfolio.save_state()  # Persists to disk
```

### 5. Risk Guardrails Update (Phase 5)

```python
from tradingagents.risk import RiskGuardrails

guardrails = RiskGuardrails()

# Record trade result
result = guardrails.record_trade_result(
    was_win=True,
    pnl_pct=1.51,
    account_balance=10000
)

# Updates:
# - consecutive_losses: Reset to 0 (was a win)
# - daily_loss_pct: No change (was a win)
# - Checks for breaches

if result['breach_triggered']:
    # Circuit breaker activated!
    # Trading halted for 24 hours
    print(f"⛔ {result['breach_type']}")
```

### 6. Pattern Analysis Counter (Phase 4)

```python
from tradingagents.learning import OnlineRLUpdater

updater = OnlineRLUpdater()

# Check if pattern update needed
should_update, trades_since = updater.should_update()

if should_update:  # Every 30 trades
    # Run pattern analysis
    from tradingagents.learning import PatternAnalyzer

    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    # Update agent weights
    performances = updater.calculate_agent_performances()
    updater.update_weights(performances)
```

## Complete Assessment Example

### Trade Lifecycle

```python
# 1. OPEN TRADE
from tradingagents.trade_decisions import store_decision, set_decision_regime

decision_id = store_decision(
    symbol="XAUUSD",
    decision_type="OPEN",
    action="BUY",
    rationale="Breaker block at support with bullish confluence",
    entry_price=2650.0,
    stop_loss=2630.0,
    take_profit=2690.0,
    volume=0.01
)

# Add regime context
regime = {
    "market_regime": "trending-up",
    "volatility_regime": "normal",
    "expansion_regime": "expansion"
}
set_decision_regime(decision_id, regime)

# Add setup metadata
decision = load_decision(decision_id)
decision['setup_type'] = 'breaker-block'
decision['confluence_score'] = 8
decision['higher_tf_bias'] = 'bullish'
# Save...

print(f"✅ Trade opened: {decision_id}")

# 2. CLOSE TRADE (days/weeks later)
close_decision(
    decision_id=decision_id,
    exit_price=2690.0,
    exit_reason="tp-hit"
)

# AUTOMATIC ASSESSMENT OUTPUT:
# ✅ Decision closed: XAUUSD_20260111_140000
#    Entry: 2650.0 → Exit: 2690.0
#    P&L: +1.51% (✓ Correct)
#    Risk-Reward: +2.00R (planned: 2.00R)
#    Reward Signal: +1.85
```

### What Gets Stored

```json
{
  "decision_id": "XAUUSD_20260111_140000",
  "symbol": "XAUUSD",
  "action": "BUY",
  "entry_price": 2650.0,
  "exit_price": 2690.0,
  "stop_loss": 2630.0,
  "take_profit": 2690.0,

  // Basic Assessment
  "pnl": 40.0,
  "pnl_percent": 1.51,
  "was_correct": true,
  "status": "closed",

  // Risk-Reward Assessment
  "rr_planned": 2.0,
  "rr_realized": 2.0,

  // Multi-Factor Reward (Phase 1)
  "reward_signal": 1.85,
  "sharpe_contribution": 0.65,
  "drawdown_impact": 0.4,

  // Context (Phase 2)
  "market_regime": "trending-up",
  "volatility_regime": "normal",
  "expansion_regime": "expansion",

  // Setup Details (Phase 3)
  "setup_type": "breaker-block",
  "confluence_score": 8,
  "higher_tf_bias": "bullish",

  // Timestamps
  "timestamp": "2026-01-11T14:00:00",
  "exit_date": "2026-01-11T18:00:00"
}
```

## How Assessment Data Is Used

### 1. RAG Similarity Search (Phase 3)

```python
# When analyzing a new setup
from tradingagents.learning import TradeSimilaritySearch

searcher = TradeSimilaritySearch()
result = searcher.find_similar_trades(current_setup)

# Uses stored assessment data:
# - was_correct: To calculate win rate
# - rr_realized: To calculate avg RR
# - reward_signal: To calculate avg reward
# - regime data: To filter by similar conditions
```

### 2. Pattern Analysis (Phase 4)

```python
# Every 30 trades
from tradingagents.learning import PatternAnalyzer

analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(lookback_days=30)

# Groups trades by:
# - setup_type: Which setups have best win rate?
# - regime: Which regimes are most profitable?
# - confluence_score: Does higher confluence = better results?
# - time_session: Which trading sessions work best?

# Uses assessment data:
# - was_correct: Win rate calculation
# - rr_realized: Average RR per pattern
# - reward_signal: Pattern quality scoring
```

### 3. Agent Weight Updates (Phase 4)

```python
# Every 30 trades
from tradingagents.learning import OnlineRLUpdater

updater = OnlineRLUpdater()
performances = updater.calculate_agent_performances()

# For each agent (bull/bear/market):
# - Calculates win rate from was_correct
# - Calculates avg reward from reward_signal
# - Updates agent weight based on performance

# Better performing agents get higher weights
```

### 4. Risk Guardrails (Phase 5)

```python
# After every trade
from tradingagents.risk import RiskGuardrails

guardrails = RiskGuardrails()

# Uses assessment data:
# - was_correct: Track consecutive losses
# - pnl_percent: Track daily loss limit
# - Triggers circuit breakers if limits exceeded
```

## Assessment Quality Indicators

### Excellent Trade (Reward > +1.5)

- ✅ High RR realized (+2.0R or better)
- ✅ Improves portfolio Sharpe ratio
- ✅ Reduces drawdown
- ✅ Consistent with winning pattern

### Good Trade (Reward +0.5 to +1.5)

- ✅ Positive RR realized (+1.0R to +2.0R)
- ✅ Neutral or positive Sharpe impact
- ✅ Doesn't increase drawdown significantly

### Neutral Trade (Reward -0.5 to +0.5)

- ⚠️ Small win or loss
- ⚠️ Minimal impact on portfolio metrics
- ⚠️ May indicate setup needs refinement

### Poor Trade (Reward < -0.5)

- ❌ Negative RR (loss)
- ❌ Hurts portfolio Sharpe ratio
- ❌ Increases drawdown
- ❌ Pattern should be avoided

## Viewing Trade Assessments

### List Closed Trades

```python
from tradingagents.trade_decisions import list_closed_decisions

trades = list_closed_decisions(limit=10)

for trade in trades:
    print(f"{trade['symbol']}: {trade['pnl_percent']:+.2f}% "
          f"({trade['rr_realized']:+.2f}R) "
          f"Reward: {trade['reward_signal']:+.2f}")

# Output:
# XAUUSD: +1.51% (+2.00R) Reward: +1.85
# XAGUSD: -0.75% (-1.00R) Reward: -0.65
# XAUUSD: +2.30% (+3.00R) Reward: +2.10
```

### Get Trade Statistics

```python
from tradingagents.learning.portfolio_state import PortfolioStateTracker

portfolio = PortfolioStateTracker.load_state()
stats = portfolio.get_statistics()

print(f"Total Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']*100:.1f}%")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {stats['max_drawdown']*100:.2f}%")
print(f"Current Equity: ${stats['current_equity']:.2f}")
```

## Summary

**Closed trades are assessed through:**

1. **Basic Metrics** - P&L, correctness, RR realized
2. **Multi-Factor Reward** - RR + Sharpe + Drawdown (Phase 1)
3. **Portfolio Impact** - Equity curve, Sharpe ratio, drawdown (Phase 1)
4. **Risk Tracking** - Consecutive losses, daily loss (Phase 5)
5. **Pattern Learning** - Setup performance, regime analysis (Phase 4)
6. **Historical Context** - RAG database for future queries (Phase 3)

**All assessment happens automatically when you call:**

```python
close_decision(decision_id, exit_price, exit_reason)
```

The system learns from every trade to make better decisions in the future! 🎯
