# How to Use the Continuous Learning System

## Quick Start Guide

This guide shows you how to use all 5 phases of the continuous learning system in your trading workflow.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Phase 1: Reward Signals](#phase-1-reward-signals)
3. [Phase 2: Regime Detection](#phase-2-regime-detection)
4. [Phase 3: RAG Decision Support](#phase-3-rag-decision-support)
5. [Phase 4: Online Learning](#phase-4-online-learning)
6. [Phase 5: Risk Guardrails](#phase-5-risk-guardrails)
7. [Complete Workflow](#complete-workflow)
8. [Daily Operations](#daily-operations)

---

## Basic Setup

### Installation

All components are already installed. Just import what you need:

```python
# Phase 1: Reward Signals
from tradingagents.learning import RewardCalculator, PortfolioStateTracker

# Phase 2: Regime Detection
from tradingagents.indicators import RegimeDetector
from tradingagents.dataflows.regime_utils import get_current_regime_from_prices

# Phase 3: RAG
from tradingagents.learning import TradeSimilaritySearch
from tradingagents.learning.rag_prompts import enhance_prompt_with_rag

# Phase 4: Online Learning
from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

# Phase 5: Risk Guardrails
from tradingagents.risk import RiskGuardrails

# Trade Decisions
from tradingagents.trade_decisions import (
    store_decision, close_decision, set_decision_regime
)
```

---

## Phase 1: Reward Signals

### When to Use

After every trade closes - automatically calculates how good/bad the trade was.

### Basic Usage

```python
from tradingagents.learning import RewardCalculator, PortfolioStateTracker

# Initialize (once at startup)
calculator = RewardCalculator()
portfolio = PortfolioStateTracker()

# After closing a trade
reward = calculator.calculate_reward(
    rr_realized=2.5,           # Your R:R (2.5R profit)
    portfolio_tracker=portfolio
)

print(f"Trade Reward: {reward:+.2f}")
# Output: Trade Reward: +1.85

# Update portfolio
portfolio.update(pnl_pct=1.5)  # 1.5% gain
```

### What It Does

- Calculates multi-factor reward: RR + Sharpe contribution + drawdown impact
- Tracks portfolio performance over time
- Automatically integrated into `close_decision()`

### Example: Close Trade with Reward

```python
from tradingagents.trade_decisions import close_decision

# Close trade - reward calculated automatically
close_decision(
    decision_id="XAUUSD_20260111_140000",
    exit_price=2690.0,
    exit_reason="tp-hit"
)

# Reward signal automatically stored in decision
```

---

## Phase 2: Regime Detection

### When to Use

Before analyzing any setup - tells you current market conditions.

### Basic Usage

```python
from tradingagents.indicators import RegimeDetector
import numpy as np

# Get your price data (from MT5, CSV, etc.)
high = np.array([2650, 2655, 2660, ...])   # Last 100+ bars
low = np.array([2640, 2645, 2650, ...])
close = np.array([2648, 2653, 2658, ...])

# Detect regime
detector = RegimeDetector()
regime = detector.get_full_regime(high, low, close)

print(regime)
# Output: {
#     "market_regime": "trending-up",
#     "volatility_regime": "normal",
#     "expansion_regime": "expansion",
#     "timestamp": "2026-01-11T19:30:00"
# }
```

### Use in Decisions

```python
# Check if favorable for your strategy
if detector.is_favorable_for_trend_trading(regime):
    print("✓ Good for trend following")
else:
    print("✗ Avoid trend trades")

# Adjust position size based on volatility
risk_adjustment = detector.get_risk_adjustment_factor(regime)
position_size = base_size * risk_adjustment

print(f"Position size: {position_size:.2f}% (adjusted by {risk_adjustment}x)")
```

### Store Regime with Trade

```python
from tradingagents.trade_decisions import store_decision, set_decision_regime

# Store trade
decision_id = store_decision(
    symbol="XAUUSD",
    decision_type="OPEN",
    action="BUY",
    entry_price=2650.0,
    stop_loss=2630.0,
    take_profit=2690.0
)

# Add regime context
set_decision_regime(decision_id, regime)
```

---

## Phase 3: RAG Decision Support

### When to Use

Before making any trade decision - shows historical performance in similar conditions.

### Basic Usage

```python
from tradingagents.learning import TradeSimilaritySearch

# Define current setup
current_setup = {
    "symbol": "XAUUSD",
    "direction": "BUY",
    "setup_type": "breaker-block",
    "market_regime": "trending-up",
    "volatility_regime": "normal",
    "confluence_score": 8
}

# Find similar historical trades
searcher = TradeSimilaritySearch()
result = searcher.find_similar_trades(current_setup, n_results=5)

# Check performance
stats = result['statistics']
print(f"Similar trades: {stats['sample_size']}")
print(f"Win rate: {stats['win_rate']*100:.1f}%")
print(f"Avg RR: {stats['avg_rr']:.2f}")
print(f"Confidence adjustment: {stats['confidence_adjustment']:+.2f}")
```

### Apply to Decision

```python
from tradingagents.learning.rag_prompts import apply_confidence_adjustment

# Your agent's base confidence
base_confidence = 0.75

# Apply historical adjustment
final_confidence = apply_confidence_adjustment(
    base_confidence,
    stats['confidence_adjustment']
)

print(f"Base: {base_confidence:.2f} → Final: {final_confidence:.2f}")

# Make decision
if final_confidence > 0.7:
    print("✓ ENTER TRADE")
elif final_confidence > 0.5:
    print("⚠️  CONSIDER (lower size)")
else:
    print("✗ SKIP")
```

### Enhance Agent Prompts

```python
from tradingagents.learning.rag_prompts import enhance_prompt_with_rag

base_prompt = """Analyze this XAUUSD setup and provide recommendation."""

enhanced_prompt, adjustment = enhance_prompt_with_rag(
    base_prompt,
    current_setup,
    n_similar=5
)

# Now enhanced_prompt includes:
# - Historical win rate
# - Top 3 similar trades
# - Recommended confidence adjustment
```

---

## Phase 4: Online Learning

### When to Use

Automatically every 30 trades - analyzes patterns and updates agent weights.

### Check If Update Needed

```python
from tradingagents.learning import OnlineRLUpdater

updater = OnlineRLUpdater()

should_update, trades_since = updater.should_update()

if should_update:
    print(f"Time to update! ({trades_since} new trades)")
else:
    print(f"Not yet ({trades_since}/30 trades)")
```

### Run Pattern Analysis

```python
from tradingagents.learning import PatternAnalyzer

analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(
    lookback_days=30,
    min_cluster_size=3
)

# Print report
report = analyzer.format_report(analysis)
print(report)

# Check recommendations
for rec in analysis['recommendations']:
    print(rec)
# Output:
# ✓ INCREASE focus on breaker-block (75% win rate, 2.3R avg)
# ✗ AVOID resistance-rejection (30% win rate, 0.5R avg)
```

### Update Agent Weights

```python
# Calculate agent performances
performances = updater.calculate_agent_performances(lookback_days=30)

# Update weights
result = updater.update_weights(performances)

# Print report
print(updater.format_report(result))

# Get current weights for decisions
weights = updater.get_current_weights()
print(f"Bull: {weights['bull']:.2f}")
print(f"Bear: {weights['bear']:.2f}")
print(f"Market: {weights['market']:.2f}")
```

### Use Weights in Consensus

```python
def make_decision(bull_conf, bear_conf, market_conf):
    # Get adaptive weights
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    # Weighted consensus
    consensus = (
        bull_conf * weights['bull'] +
        bear_conf * weights['bear'] +
        market_conf * weights['market']
    )

    return consensus
```

---

## Phase 5: Risk Guardrails

### When to Use

**Before EVERY trade** - enforces hard risk limits.

### Initialize (Once)

```python
from tradingagents.risk import RiskGuardrails

guardrails = RiskGuardrails(
    daily_loss_limit_pct=3.0,        # Max 3% loss per day
    max_consecutive_losses=2,         # Max 2 losses in a row
    max_position_size_pct=2.0,       # Max 2% per trade
    cooldown_hours=24                 # 24h timeout after breach
)
```

### Check Before Trading

```python
account_balance = 10000

# Check if allowed to trade
can_trade, reason = guardrails.check_can_trade(account_balance)

if not can_trade:
    print(f"⛔ Trading blocked: {reason}")
    # DO NOT TRADE - respect the guardrail
    exit()

print(f"✅ Trading allowed: {reason}")
```

### Validate Position Size

```python
requested_size = 2.5  # Want 2.5% risk

is_valid, reason, adjusted_size = guardrails.validate_position_size(
    requested_size,
    account_balance
)

if not is_valid:
    print(f"⚠️  {reason}")
    print(f"Using {adjusted_size}% instead of {requested_size}%")

# Use adjusted_size for trade
```

### Record Trade Results

```python
# After trade closes
was_win = True
pnl_pct = 1.5  # 1.5% profit

result = guardrails.record_trade_result(was_win, pnl_pct, account_balance)

if result['breach_triggered']:
    print(f"⛔ CIRCUIT BREAKER ACTIVATED!")
    print(f"Type: {result['breach_type']}")
    print(f"Cooldown until: {result['cooldown_until']}")

    # Send alert, log, etc.
    send_alert("Trading halted - circuit breaker triggered")

print(f"Status: {result['status']}")
```

### Monitor Status

```python
# Check current status anytime
status = guardrails.get_status()

print(f"Can trade: {status['can_trade']}")
print(f"Consecutive losses: {status['consecutive_losses']}/2")
print(f"Daily loss: {status['daily_loss_pct']:.2f}%/3.0%")

# Or get full report
report = guardrails.format_report()
print(report)
```

---

## Complete Workflow

### Full Trading Decision Process

```python
def complete_trading_decision(
    symbol,
    setup_type,
    high, low, close,
    bull_conf, bear_conf, market_conf,
    account_balance
):
    """
    Complete adaptive trading decision with all 5 phases
    """

    print("🎯 ADAPTIVE TRADING SYSTEM\n")

    # ============================================================
    # PHASE 5: Risk Guardrails - CHECK FIRST
    # ============================================================
    print("1️⃣  Risk Guardrails Check...")

    from tradingagents.risk import RiskGuardrails
    guardrails = RiskGuardrails()

    can_trade, reason = guardrails.check_can_trade(account_balance)
    if not can_trade:
        print(f"   ⛔ BLOCKED: {reason}")
        return {"decision": "BLOCKED", "reason": reason}

    print(f"   ✅ {reason}")

    # ============================================================
    # PHASE 2: Regime Detection
    # ============================================================
    print("\n2️⃣  Regime Detection...")

    from tradingagents.indicators import RegimeDetector
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)

    print(f"   Market: {regime['market_regime']}")
    print(f"   Volatility: {regime['volatility_regime']}")

    # Check if favorable
    if setup_type in ["breaker-block", "FVG"]:
        if not detector.is_favorable_for_trend_trading(regime):
            print(f"   ⚠️  Regime not favorable for trend trading")

    # ============================================================
    # PHASE 4: Pattern Analysis
    # ============================================================
    print("\n3️⃣  Pattern Analysis...")

    from tradingagents.learning import PatternAnalyzer
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    # Find this setup's pattern
    pattern = next((p for p in analysis['patterns']
                   if p['pattern_value'] == setup_type), None)

    if pattern:
        print(f"   Setup: {setup_type}")
        print(f"   Quality: {pattern['quality']}")
        print(f"   Win Rate: {pattern['win_rate']*100:.0f}%")

        if pattern['quality'] == 'poor':
            print(f"   ✗ SKIP - Poor historical performance")
            return {"decision": "SKIP", "reason": "Poor pattern"}

    # ============================================================
    # PHASE 3: RAG Historical Context
    # ============================================================
    print("\n4️⃣  RAG Historical Context...")

    from tradingagents.learning import TradeSimilaritySearch
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment

    current_setup = {
        "symbol": symbol,
        "direction": "BUY",
        "setup_type": setup_type,
        **regime
    }

    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup, n_results=5)

    stats = rag_result['statistics']
    print(f"   Similar trades: {stats['sample_size']}")
    print(f"   Win rate: {stats['win_rate']*100:.1f}%")
    print(f"   Adjustment: {stats['confidence_adjustment']:+.2f}")

    # ============================================================
    # PHASE 4: Agent Weights
    # ============================================================
    print("\n5️⃣  Agent Weights (Adaptive)...")

    from tradingagents.learning import OnlineRLUpdater
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    print(f"   Bull: {weights['bull']:.2f}")
    print(f"   Bear: {weights['bear']:.2f}")
    print(f"   Market: {weights['market']:.2f}")

    # ============================================================
    # Calculate Final Decision
    # ============================================================
    print("\n6️⃣  Final Decision Calculation...")

    # Weighted consensus
    base_consensus = (
        bull_conf * weights['bull'] +
        bear_conf * weights['bear'] +
        market_conf * weights['market']
    )

    # Apply RAG adjustment
    final_confidence = apply_confidence_adjustment(
        base_consensus,
        stats['confidence_adjustment']
    )

    # Apply regime risk adjustment
    risk_adj = detector.get_risk_adjustment_factor(regime)

    print(f"   Base consensus: {base_consensus:.2f}")
    print(f"   RAG adjusted: {final_confidence:.2f}")
    print(f"   Risk adjustment: {risk_adj:.2f}x")

    # ============================================================
    # Position Sizing
    # ============================================================
    print("\n7️⃣  Position Sizing...")

    base_size = 1.5  # 1.5% base risk
    regime_adjusted = base_size * risk_adj

    is_valid, reason, final_size = guardrails.validate_position_size(
        regime_adjusted,
        account_balance
    )

    print(f"   Base: {base_size}%")
    print(f"   Regime adjusted: {regime_adjusted:.2f}%")
    print(f"   Final: {final_size:.2f}%")

    # ============================================================
    # Final Decision
    # ============================================================
    print("\n8️⃣  Final Decision...")

    if final_confidence > 0.7:
        decision = "ENTER"
        print(f"   ✅ ENTER TRADE (confidence: {final_confidence:.2f})")
    elif final_confidence > 0.5:
        decision = "CONSIDER"
        print(f"   ⚠️  CONSIDER (confidence: {final_confidence:.2f})")
        print(f"   → Use reduced size or wait for better setup")
    else:
        decision = "SKIP"
        print(f"   ✗ SKIP (confidence: {final_confidence:.2f})")

    return {
        "decision": decision,
        "confidence": final_confidence,
        "position_size_pct": final_size,
        "regime": regime,
        "pattern_quality": pattern['quality'] if pattern else "unknown",
        "rag_adjustment": stats['confidence_adjustment'],
        "agent_weights": weights,
        "risk_adjustment": risk_adj
    }


# ============================================================
# USAGE EXAMPLE
# ============================================================

# Your market data
import numpy as np
high = np.array([...])   # Last 100+ bars
low = np.array([...])
close = np.array([...])

# Your agent confidences
bull_conf = 0.80
bear_conf = 0.30
market_conf = 0.60

# Run complete decision
result = complete_trading_decision(
    symbol="XAUUSD",
    setup_type="breaker-block",
    high=high,
    low=low,
    close=close,
    bull_conf=bull_conf,
    bear_conf=bear_conf,
    market_conf=market_conf,
    account_balance=10000
)

print(f"\n{'='*60}")
print(f"FINAL RESULT: {result['decision']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Position Size: {result['position_size_pct']:.2f}%")
print(f"{'='*60}")
```

---

## Daily Operations

### Morning Routine

```python
def morning_check():
    """Run before trading day starts"""

    from tradingagents.risk import RiskGuardrails
    from tradingagents.learning import OnlineRLUpdater

    print("🌅 MORNING CHECK\n")

    # 1. Check risk status
    guardrails = RiskGuardrails()
    status = guardrails.get_status()

    print("Risk Status:")
    print(f"  Can trade: {status['can_trade']}")
    print(f"  Consecutive losses: {status['consecutive_losses']}")
    print(f"  Daily loss: {status['daily_loss_pct']:.2f}%")

    # 2. Check agent weights
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    print("\nAgent Weights:")
    for agent, weight in weights.items():
        print(f"  {agent.capitalize()}: {weight:.2f}")

    # 3. Check if update needed
    should_update, trades_since = updater.should_update()
    if should_update:
        print(f"\n⚠️  Pattern analysis needed ({trades_since} trades)")

    print("\n✅ Ready to trade\n")
```

### After Each Trade

```python
def after_trade(decision_id, was_win, pnl_pct, account_balance):
    """Run after closing each trade"""

    from tradingagents.risk import RiskGuardrails
    from tradingagents.learning import OnlineRLUpdater

    # 1. Record in guardrails
    guardrails = RiskGuardrails()
    result = guardrails.record_trade_result(was_win, pnl_pct, account_balance)

    if result['breach_triggered']:
        print(f"⛔ CIRCUIT BREAKER: {result['breach_type']}")
        send_alert("Trading halted")

    # 2. Check if pattern update needed
    updater = OnlineRLUpdater()
    should_update, trades_since = updater.should_update()

    if should_update:
        print(f"\n📊 Running pattern analysis ({trades_since} trades)...")
        run_pattern_update()
```

### Every 30 Trades

```python
def run_pattern_update():
    """Run pattern analysis and weight update"""

    from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

    print("="*60)
    print("PATTERN ANALYSIS & WEIGHT UPDATE")
    print("="*60)

    # 1. Pattern analysis
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    report = analyzer.format_report(analysis)
    print(report)

    # 2. Update weights
    updater = OnlineRLUpdater()
    performances = updater.calculate_agent_performances()
    result = updater.update_weights(performances)

    print("\n" + updater.format_report(result))

    # 3. Save reports
    with open("pattern_analysis.txt", "a") as f:
        f.write(f"\n{report}\n")
```

### End of Day

```python
def end_of_day():
    """Run at end of trading day"""

    from tradingagents.risk import RiskGuardrails

    guardrails = RiskGuardrails()

    # Print daily summary
    report = guardrails.format_report()
    print(report)

    # Reset daily counters
    guardrails.reset_daily_loss()

    print("\n✅ Daily counters reset for tomorrow")
```

---

## Testing the System

### Run Demo Scripts

```bash
# Test reward system
python examples/test_reward_system.py

# Test regime detection
python examples/test_regime_detection.py

# Test RAG decision support
python examples/test_rag_decision.py

# Test online learning & guardrails
python examples/test_online_learning.py
```

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific phases
pytest tests/test_reward_calculator.py -v
pytest tests/test_regime_detector.py -v
pytest tests/test_trade_similarity.py -v
pytest tests/test_pattern_analyzer.py -v
pytest tests/test_risk_guardrails.py -v
```

---

## Troubleshooting

### "No similar trades found"

- **Cause**: Not enough historical data
- **Fix**: Need at least 5-10 closed trades with metadata

### "Pattern analysis returns empty"

- **Cause**: `min_cluster_size` too high or insufficient data
- **Fix**: Reduce to `min_cluster_size=2` or wait for more trades

### "Weights not updating"

- **Cause**: Less than 30 trades since last update
- **Fix**: Check with `should_update()` - need 30 closed trades

### "Circuit breaker won't clear"

- **Cause**: Still in cooldown period
- **Fix**: Wait for cooldown to expire or manually reset (after review)

### "Regime always 'ranging'"

- **Cause**: Insufficient price data (<100 bars)
- **Fix**: Provide at least 100 bars of OHLC data

---

## Best Practices

1. **Always check guardrails first** - Never bypass risk limits
2. **Store complete metadata** - Include setup_type, confluence_score, regime
3. **Review patterns weekly** - Check what's working/not working
4. **Respect circuit breakers** - They prevent catastrophic losses
5. **Monitor weight changes** - Understand why system is adapting
6. **Test with demo account first** - Verify everything works before live

---

## Quick Reference

### Import Cheat Sheet

```python
# Reward & Portfolio
from tradingagents.learning import RewardCalculator, PortfolioStateTracker

# Regime
from tradingagents.indicators import RegimeDetector

# RAG
from tradingagents.learning import TradeSimilaritySearch
from tradingagents.learning.rag_prompts import enhance_prompt_with_rag

# Pattern & Weights
from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

# Risk
from tradingagents.risk import RiskGuardrails

# Decisions
from tradingagents.trade_decisions import store_decision, close_decision
```

### Key Functions

```python
# Regime
regime = detector.get_full_regime(high, low, close)

# RAG
result = searcher.find_similar_trades(current_setup)

# Pattern
analysis = analyzer.analyze_patterns(lookback_days=30)

# Weights
result = updater.update_weights(performances)

# Risk
can_trade, reason = guardrails.check_can_trade(balance)
```

---

## Support

For detailed integration guides, see:

- `PHASE2_INTEGRATION_GUIDE.md` - Regime Detection
- `PHASE3_INTEGRATION_GUIDE.md` - RAG Decision Support
- `PHASE4_PHASE5_INTEGRATION_GUIDE.md` - Online Learning & Risk Guardrails

For implementation details, see:

- `CONTINUOUS_LEARNING_PLAN.md` - Full technical specification

---

**You now have a fully adaptive, self-improving trading system!** 🚀

The system learns from every trade, adapts to market conditions, and protects you from catastrophic losses.
