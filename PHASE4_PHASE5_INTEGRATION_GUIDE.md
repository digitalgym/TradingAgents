# Phase 4 & 5: Online Learning + Risk Guardrails - Integration Guide

## Overview

**Phase 4** adds online learning through pattern analysis and agent weight updates. The system automatically adapts to changing market conditions by analyzing what's working and adjusting agent influence accordingly.

**Phase 5** adds hard risk guardrails to prevent catastrophic losses through circuit breakers, daily loss limits, and automatic cooldown periods.

## Phase 4: Online Learning

### Components Added

#### 1. Pattern Analyzer (`tradingagents/learning/pattern_analyzer.py`)

Analyzes historical trades to identify winning and losing patterns:

```python
from tradingagents.learning.pattern_analyzer import PatternAnalyzer

analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=3)

# Returns:
# {
#     "patterns": [
#         {
#             "pattern_type": "setup_type",
#             "pattern_value": "breaker-block",
#             "sample_size": 10,
#             "win_rate": 0.75,
#             "avg_rr": 2.3,
#             "quality": "excellent"
#         },
#         ...
#     ],
#     "recommendations": [
#         "✓ INCREASE focus on breaker-block (75% win rate, 2.3R avg)",
#         "✗ AVOID resistance-rejection (30% win rate, 0.5R avg)"
#     ],
#     "statistics": {...}
# }
```

**Pattern Types:**

- **setup_type**: Breaker blocks, FVG, support bounces, etc.
- **regime**: Market conditions (trending-up/normal, ranging/low, etc.)
- **time_session**: Asian, London, New York sessions
- **confluence**: High (8-10), Medium (6-7), Low (0-5)

#### 2. Online RL Updater (`tradingagents/learning/online_rl.py`)

Updates agent weights based on performance:

```python
from tradingagents.learning.online_rl import OnlineRLUpdater

updater = OnlineRLUpdater(learning_rate=0.1, momentum=0.9)

# Calculate agent performances
performances = updater.calculate_agent_performances(lookback_days=30)

# Update weights
result = updater.update_weights(performances)

# Returns:
# {
#     "old_weights": {"bull": 0.33, "bear": 0.33, "market": 0.34},
#     "new_weights": {"bull": 0.45, "bear": 0.25, "market": 0.30},
#     "changes": {"bull": +0.12, "bear": -0.08, "market": -0.04},
#     "reasoning": "BULL: INCREASED from 0.33 to 0.45 (win rate: 70%, avg reward: +1.5)"
# }
```

**Weight Update Logic:**

- Score = win_rate × normalized_reward × √sample_size
- Momentum-based updates for smooth transitions
- Automatic normalization to sum to 1.0

### Integration Steps

#### Step 1: Periodic Pattern Analysis

Run pattern analysis every 30 trades:

```python
from tradingagents.learning.pattern_analyzer import PatternAnalyzer
from tradingagents.learning.online_rl import OnlineRLUpdater

def check_and_update_if_needed():
    updater = OnlineRLUpdater()

    # Check if update needed (every 30 trades)
    should_update, trades_since = updater.should_update()

    if should_update:
        print(f"📊 Running pattern analysis ({trades_since} new trades)...")

        # Analyze patterns
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30)

        # Print report
        report = analyzer.format_report(analysis)
        print(report)

        # Update agent weights
        performances = updater.calculate_agent_performances()
        result = updater.update_weights(performances)

        print("\n🔄 Agent weights updated:")
        print(updater.format_report(result))

        return analysis, result

    return None, None
```

#### Step 2: Use Updated Weights in Decisions

Apply current weights to agent consensus:

```python
from tradingagents.learning.online_rl import OnlineRLUpdater

def make_consensus_decision(bull_conf, bear_conf, market_conf):
    # Get current weights
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    # Weighted consensus
    consensus = (
        bull_conf * weights["bull"] +
        bear_conf * weights["bear"] +
        market_conf * weights["market"]
    )

    print(f"Agent Weights: Bull={weights['bull']:.2f}, "
          f"Bear={weights['bear']:.2f}, Market={weights['market']:.2f}")
    print(f"Consensus: {consensus:.2f}")

    return consensus
```

#### Step 3: Apply Pattern Insights

Use pattern analysis to filter setups:

```python
def should_take_setup(setup_type, regime):
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    # Find pattern for this setup
    pattern = next((p for p in analysis["patterns"]
                   if p["pattern_value"] == setup_type), None)

    if pattern:
        if pattern["quality"] == "excellent":
            print(f"✓ {setup_type}: Excellent pattern ({pattern['win_rate']*100:.0f}% WR)")
            return True
        elif pattern["quality"] == "poor":
            print(f"✗ {setup_type}: Poor pattern ({pattern['win_rate']*100:.0f}% WR)")
            return False

    return True  # No data, proceed with caution
```

## Phase 5: Risk Guardrails

### Components Added

#### Risk Guardrails (`tradingagents/risk/guardrails.py`)

Hard risk limits with circuit breakers:

```python
from tradingagents.risk.guardrails import RiskGuardrails

guardrails = RiskGuardrails(
    daily_loss_limit_pct=3.0,        # Max 3% loss per day
    max_consecutive_losses=2,         # Max 2 losses in a row
    max_position_size_pct=2.0,       # Max 2% per trade
    cooldown_hours=24                 # 24h cooldown after breach
)
```

**Features:**

- **Daily Loss Limit**: Stops trading after 3% daily loss
- **Consecutive Loss Limit**: Stops after 2 losses in a row
- **Position Size Caps**: Enforces maximum position size
- **Circuit Breakers**: Automatic cooldown after breaches
- **Breach History**: Tracks all violations

### Integration Steps

#### Step 1: Check Before Every Trade

```python
from tradingagents.risk.guardrails import RiskGuardrails

def can_enter_trade(account_balance):
    guardrails = RiskGuardrails()

    # Check if trading allowed
    can_trade, reason = guardrails.check_can_trade(account_balance)

    if not can_trade:
        print(f"⛔ Trading blocked: {reason}")
        return False

    print(f"✅ Trading allowed: {reason}")
    return True
```

#### Step 2: Validate Position Size

```python
def validate_and_adjust_position(requested_size_pct, account_balance):
    guardrails = RiskGuardrails()

    is_valid, reason, adjusted_size = guardrails.validate_position_size(
        requested_size_pct,
        account_balance
    )

    if not is_valid:
        print(f"⚠️  Position size adjusted: {reason}")
        print(f"   Requested: {requested_size_pct}% → Using: {adjusted_size}%")

    return adjusted_size
```

#### Step 3: Record Trade Results

```python
def record_trade_outcome(was_win, pnl_pct, account_balance):
    guardrails = RiskGuardrails()

    result = guardrails.record_trade_result(was_win, pnl_pct, account_balance)

    if result["breach_triggered"]:
        print(f"⛔ CIRCUIT BREAKER TRIGGERED!")
        print(f"   Type: {result['breach_type']}")
        print(f"   Cooldown until: {result['cooldown_until']}")

        # Send alert, log to file, etc.
        send_alert(f"Trading halted: {result['breach_type']}")

    print(f"Status: {result['status']}")
```

#### Step 4: Monitor Status

```python
def show_risk_status():
    guardrails = RiskGuardrails()

    report = guardrails.format_report()
    print(report)

    # Or get structured data
    status = guardrails.get_status()

    if not status["can_trade"]:
        print(f"\n⚠️  WARNING: Trading currently disabled")
        print(f"   Reason: {status['reason']}")
```

## Complete Integration Example

```python
from tradingagents.learning.pattern_analyzer import PatternAnalyzer
from tradingagents.learning.online_rl import OnlineRLUpdater
from tradingagents.risk.guardrails import RiskGuardrails
from tradingagents.learning.trade_similarity import TradeSimilaritySearch

def adaptive_trading_system(
    symbol,
    setup_type,
    regime,
    bull_conf,
    bear_conf,
    market_conf,
    account_balance
):
    """Complete adaptive trading system with all features"""

    # 1. Check risk guardrails
    print("1️⃣  Risk Check...")
    guardrails = RiskGuardrails()
    can_trade, reason = guardrails.check_can_trade(account_balance)

    if not can_trade:
        return {"decision": "BLOCKED", "reason": reason}

    # 2. Check pattern analysis
    print("2️⃣  Pattern Analysis...")
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    # Find this setup's pattern
    pattern = next((p for p in analysis["patterns"]
                   if p["pattern_value"] == setup_type), None)

    if pattern and pattern["quality"] == "poor":
        return {"decision": "SKIP", "reason": f"Poor pattern: {pattern['win_rate']*100:.0f}% WR"}

    # 3. Get agent weights
    print("3️⃣  Agent Weights...")
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    # 4. RAG historical context
    print("4️⃣  Historical Context...")
    searcher = TradeSimilaritySearch()
    current_setup = {
        "symbol": symbol,
        "direction": "BUY",
        "setup_type": setup_type,
        **regime
    }
    rag_result = searcher.find_similar_trades(current_setup)

    # 5. Calculate weighted consensus
    print("5️⃣  Consensus Decision...")
    base_consensus = (
        bull_conf * weights["bull"] +
        bear_conf * weights["bear"] +
        market_conf * weights["market"]
    )

    # Apply RAG adjustment
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment
    final_confidence = apply_confidence_adjustment(
        base_consensus,
        rag_result['statistics']['confidence_adjustment']
    )

    # 6. Validate position size
    print("6️⃣  Position Sizing...")
    requested_size = 1.5  # 1.5% risk
    _, _, adjusted_size = guardrails.validate_position_size(
        requested_size,
        account_balance
    )

    # 7. Make decision
    if final_confidence > 0.7:
        decision = "ENTER"
    elif final_confidence > 0.5:
        decision = "CONSIDER"
    else:
        decision = "SKIP"

    return {
        "decision": decision,
        "confidence": final_confidence,
        "position_size_pct": adjusted_size,
        "agent_weights": weights,
        "rag_adjustment": rag_result['statistics']['confidence_adjustment'],
        "pattern_quality": pattern["quality"] if pattern else "unknown",
        "risk_status": guardrails.get_status()["status_summary"]
    }
```

## Periodic Maintenance

### Every 30 Trades

```python
def periodic_update():
    """Run after every 30 trades"""

    # 1. Pattern analysis
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30)

    print("📊 PATTERN ANALYSIS")
    print(analyzer.format_report(analysis))

    # 2. Update agent weights
    updater = OnlineRLUpdater()
    performances = updater.calculate_agent_performances()
    result = updater.update_weights(performances)

    print("\n🔄 AGENT WEIGHT UPDATE")
    print(updater.format_report(result))

    # 3. Review risk status
    guardrails = RiskGuardrails()
    print("\n🛡️  RISK STATUS")
    print(guardrails.format_report())
```

### Daily Review

```python
def daily_review():
    """Run at end of each trading day"""

    guardrails = RiskGuardrails()

    # Reset daily counters
    guardrails.reset_daily_loss()

    # Check for breaches
    breaches = guardrails.get_breach_history(5)
    if breaches:
        print("⚠️  Recent breaches:")
        for breach in breaches:
            print(f"   {breach['timestamp']}: {breach['type']}")
```

## Configuration

### Pattern Analyzer

```python
analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(
    lookback_days=30,        # Days to analyze
    min_cluster_size=3       # Min trades for pattern
)
```

### Online RL Updater

```python
updater = OnlineRLUpdater(
    learning_rate=0.1,       # How fast to adapt (0.0-1.0)
    momentum=0.9             # Smoothing factor (0.0-1.0)
)
```

### Risk Guardrails

```python
guardrails = RiskGuardrails(
    daily_loss_limit_pct=3.0,        # Daily loss limit
    max_consecutive_losses=2,         # Consecutive loss limit
    max_position_size_pct=2.0,       # Max position size
    cooldown_hours=24                 # Cooldown duration
)
```

## Testing

```bash
# Unit tests
pytest tests/test_pattern_analyzer.py -v
pytest tests/test_risk_guardrails.py -v

# Integration demo
python examples/test_online_learning.py
```

## Best Practices

### 1. Start Conservative

```python
# Use stricter limits initially
guardrails = RiskGuardrails(
    daily_loss_limit_pct=2.0,    # Stricter than default 3%
    max_consecutive_losses=1,     # Stricter than default 2
    cooldown_hours=48             # Longer than default 24h
)
```

### 2. Monitor Weight Changes

```python
# Log weight changes for review
updater = OnlineRLUpdater()
history = updater.get_weight_history(10)

for entry in history:
    print(f"{entry['timestamp']}: {entry['weights']}")
```

### 3. Review Patterns Weekly

```python
# Weekly pattern review
analyzer = PatternAnalyzer()
analysis = analyzer.analyze_patterns(lookback_days=7)

# Focus on recent patterns
for pattern in analysis["patterns"][:5]:
    print(f"{pattern['pattern_value']}: {pattern['quality']}")
```

### 4. Respect Circuit Breakers

```python
# Never override circuit breakers programmatically
# Only manual reset after review

guardrails = RiskGuardrails()
can_trade, reason = guardrails.check_can_trade(balance)

if not can_trade:
    # DO NOT trade - respect the guardrail
    log_blocked_trade(reason)
    return
```

## Troubleshooting

**Issue**: Weights not updating

- **Check**: Run `should_update()` to see trades count
- **Fix**: Need 30 closed trades since last update

**Issue**: Circuit breaker too sensitive

- **Adjust**: Increase `daily_loss_limit_pct` or `max_consecutive_losses`
- **Note**: Don't disable - adjust thresholds instead

**Issue**: Pattern analysis finds no patterns

- **Cause**: Insufficient data or min_cluster_size too high
- **Fix**: Reduce `min_cluster_size` or wait for more trades

**Issue**: Cooldown won't clear

- **Check**: `get_status()` to see cooldown end time
- **Manual**: Use `reset_cooldown()` only after review

## Performance Metrics

Track system effectiveness:

```python
def measure_adaptive_system():
    """Compare performance before/after adaptive features"""

    # Get recent trades
    from tradingagents.trade_decisions import list_closed_decisions
    trades = list_closed_decisions(limit=100)

    # Split by date (before/after implementation)
    cutoff = datetime(2026, 1, 1)
    before = [t for t in trades if datetime.fromisoformat(t['timestamp']) < cutoff]
    after = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= cutoff]

    # Calculate metrics
    def calc_metrics(trades):
        wins = sum(1 for t in trades if t['was_correct'])
        win_rate = wins / len(trades) if trades else 0
        avg_rr = np.mean([t['rr_realized'] for t in trades if t.get('rr_realized')])
        return win_rate, avg_rr

    wr_before, rr_before = calc_metrics(before)
    wr_after, rr_after = calc_metrics(after)

    print(f"Before Adaptive System:")
    print(f"  Win Rate: {wr_before*100:.1f}%")
    print(f"  Avg RR: {rr_before:.2f}")

    print(f"\nAfter Adaptive System:")
    print(f"  Win Rate: {wr_after*100:.1f}%")
    print(f"  Avg RR: {rr_after:.2f}")

    print(f"\nImprovement:")
    print(f"  Win Rate: {(wr_after - wr_before)*100:+.1f}%")
    print(f"  Avg RR: {(rr_after - rr_before):+.2f}")
```

---

**Phase 4 & 5 Complete** ✅

The system now:

- ✅ Automatically analyzes patterns every 30 trades
- ✅ Adapts agent weights based on performance
- ✅ Enforces hard risk limits with circuit breakers
- ✅ Prevents catastrophic losses through cooldowns
- ✅ Continuously learns and improves

**Result**: A fully adaptive, self-improving trading system with robust risk management.
