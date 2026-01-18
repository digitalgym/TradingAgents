# Phase 2: Regime Detection - Integration Guide

## Overview

Phase 2 adds market regime detection to enable context-aware trading decisions. The system now detects:

- **Trend regime**: trending-up, trending-down, ranging
- **Volatility regime**: low, normal, high, extreme
- **Expansion regime**: expansion, contraction, neutral

## Components Added

### 1. Core Module: `tradingagents/indicators/regime.py`

```python
from tradingagents.indicators.regime import RegimeDetector

detector = RegimeDetector()
regime = detector.get_full_regime(high, low, close)

# Returns:
# {
#     "market_regime": "trending-up",
#     "volatility_regime": "high",
#     "expansion_regime": "expansion",
#     "timestamp": "2026-01-11T14:30:00"
# }
```

### 2. Memory System Enhancement

Memories can now be filtered by regime:

```python
from tradingagents.agents.utils.memory import FinancialSituationMemory

memory = FinancialSituationMemory("trading", config)

# Add memory with regime context
memory.add_situations(
    [("Market situation", "Recommendation")],
    regime={"market_regime": "trending-up", "volatility_regime": "high"}
)

# Retrieve memories from similar regime
regime_filter = {"market_regime": "trending-up", "volatility_regime": "high"}
similar_memories = memory.get_memories(
    "Current situation",
    regime_filter=regime_filter
)
```

### 3. Trade Decision Integration

Trade decisions now capture regime context:

```python
from tradingagents.trade_decisions import store_decision, populate_regime_from_prices

# Store decision
decision_id = store_decision(
    symbol="XAUUSD",
    decision_type="OPEN",
    action="BUY",
    entry_price=2650.0,
    stop_loss=2630.0,
    take_profit=2690.0
)

# Populate regime from price data
populate_regime_from_prices(decision_id, high, low, close)

# Decision now includes:
# - market_regime
# - volatility_regime
# - expansion_regime
```

### 4. Utility Functions

```python
from tradingagents.dataflows.regime_utils import (
    get_current_regime_from_prices,
    format_regime_for_prompt,
    get_regime_summary
)

# Get regime
regime = get_current_regime_from_prices(high, low, close)

# Format for LLM prompt
prompt_text = format_regime_for_prompt(regime)

# Get one-line summary
summary = get_regime_summary(regime)  # "trending-up / high volatility"
```

## Integration Steps

### Step 1: Add Regime Detection to Analysis Workflow

In your analysis agents (Bull/Bear/Market analysts):

```python
# In tradingagents/agents/researchers/bull_researcher.py

def bull_researcher_node(state) -> dict:
    # ... existing code ...

    # Add regime detection
    from tradingagents.dataflows.regime_utils import (
        get_current_regime_from_prices,
        format_regime_for_prompt
    )

    # Get price data
    high = state.get("high_prices")
    low = state.get("low_prices")
    close = state.get("close_prices")

    # Detect regime
    regime = get_current_regime_from_prices(high, low, close)
    state["regime"] = regime

    # Add to prompt
    regime_context = format_regime_for_prompt(regime)

    prompt = f"""
    {existing_prompt}

    {regime_context}

    Consider the current market regime when making your recommendation.
    """

    # ... rest of analysis ...
```

### Step 2: Store Regime with Decisions

When storing trade decisions:

```python
# In your trading execution code

from tradingagents.trade_decisions import store_decision, set_decision_regime

# Store decision
decision_id = store_decision(
    symbol="XAUUSD",
    decision_type="OPEN",
    action="BUY",
    entry_price=2650.0,
    stop_loss=2630.0,
    take_profit=2690.0
)

# Add regime context
regime = state.get("regime")  # From analysis
set_decision_regime(decision_id, regime)
```

### Step 3: Use Regime for Memory Retrieval

When querying historical trades:

```python
# In your decision-making code

from tradingagents.agents.utils.memory import FinancialSituationMemory

memory = FinancialSituationMemory("trading", config)

# Get current regime
current_regime = state.get("regime")

# Filter memories by regime
regime_filter = {
    "market_regime": current_regime["market_regime"],
    "volatility_regime": current_regime["volatility_regime"]
}

similar_trades = memory.get_memories(
    current_situation="Looking for BUY setup on XAUUSD",
    regime_filter=regime_filter,
    n_matches=5
)

# Use similar_trades to inform decision
```

### Step 4: Adjust Risk Based on Regime

```python
from tradingagents.indicators.regime import RegimeDetector

detector = RegimeDetector()
regime = state.get("regime")

# Get risk adjustment factor
risk_adjustment = detector.get_risk_adjustment_factor(regime)

# Apply to position sizing
base_position_size = 0.01  # 1% risk
adjusted_size = base_position_size * risk_adjustment

print(f"Position size: {adjusted_size:.4f} ({risk_adjustment*100:.0f}% of base)")
```

## Usage Examples

### Example 1: Complete Analysis with Regime

```python
def analyze_market_with_regime(symbol, high, low, close):
    from tradingagents.indicators.regime import RegimeDetector
    from tradingagents.dataflows.regime_utils import format_regime_for_prompt

    # Detect regime
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)

    # Get trading implications
    trend_favorable = detector.is_favorable_for_trend_trading(regime)
    range_favorable = detector.is_favorable_for_range_trading(regime)
    risk_adj = detector.get_risk_adjustment_factor(regime)

    # Build context
    context = {
        "regime": regime,
        "trend_favorable": trend_favorable,
        "range_favorable": range_favorable,
        "risk_adjustment": risk_adj,
        "regime_description": detector.get_regime_description(regime)
    }

    return context
```

### Example 2: Regime-Aware Trade Entry

```python
def should_enter_trade(setup_type, regime):
    from tradingagents.indicators.regime import RegimeDetector

    detector = RegimeDetector()

    # Trend following setups
    if setup_type in ["breaker-block", "FVG", "trend-continuation"]:
        if not detector.is_favorable_for_trend_trading(regime):
            print("⚠️ Trend setup in non-trending regime - SKIP")
            return False

    # Range trading setups
    elif setup_type in ["support-bounce", "resistance-rejection"]:
        if not detector.is_favorable_for_range_trading(regime):
            print("⚠️ Range setup in trending regime - SKIP")
            return False

    # Check volatility
    if regime["volatility_regime"] == "extreme":
        print("⚠️ Extreme volatility - REDUCE SIZE or SKIP")
        return False

    return True
```

### Example 3: Pattern Analysis by Regime

```python
def analyze_patterns_by_regime():
    from tradingagents.trade_decisions import list_closed_decisions

    trades = list_closed_decisions(limit=100)

    # Group by regime
    by_regime = {}
    for trade in trades:
        regime_key = f"{trade.get('market_regime')} / {trade.get('volatility_regime')}"
        if regime_key not in by_regime:
            by_regime[regime_key] = []
        by_regime[regime_key].append(trade)

    # Calculate win rate by regime
    for regime_key, regime_trades in by_regime.items():
        wins = sum(1 for t in regime_trades if t.get("was_correct"))
        win_rate = wins / len(regime_trades) if regime_trades else 0

        print(f"{regime_key}: {win_rate*100:.1f}% win rate ({len(regime_trades)} trades)")
```

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_regime_detector.py -v

# Integration demo
python examples/test_regime_detection.py
```

## Configuration

Customize regime detection thresholds:

```python
from tradingagents.indicators.regime import RegimeDetector

detector = RegimeDetector(
    adx_threshold_trending=30.0,      # Higher = stricter trend detection
    atr_percentile_high=80.0,         # Higher = less sensitive to volatility
    atr_percentile_extreme=95.0,
    bb_width_percentile_expansion=75.0,
    lookback_period=100               # Historical data for percentiles
)
```

## Next Steps (Phase 3)

With regime detection in place, Phase 3 will add:

1. **Trade similarity search** - Find similar historical setups
2. **RAG-enhanced prompts** - Include historical performance in agent prompts
3. **Confidence adjustment** - Boost/reduce confidence based on regime-specific win rates

## Troubleshooting

**Issue**: Regime always returns "ranging"

- **Cause**: Insufficient price data (need 100+ bars)
- **Fix**: Ensure you're passing enough historical data

**Issue**: Volatility always "normal"

- **Cause**: Not enough lookback data for percentile calculation
- **Fix**: Increase lookback_period or provide more data

**Issue**: Regime not saved in decisions

- **Cause**: Forgot to call `set_decision_regime()` or `populate_regime_from_prices()`
- **Fix**: Add regime population after storing decision

## API Reference

### RegimeDetector

**Methods:**

- `detect_trend_regime(high, low, close, period=14)` → str
- `detect_volatility_regime(high, low, close, period=14)` → str
- `detect_expansion_regime(close, period=20, std_dev=2.0)` → str
- `get_full_regime(high, low, close, timestamp=None)` → dict
- `get_regime_description(regime)` → str
- `is_favorable_for_trend_trading(regime)` → bool
- `is_favorable_for_range_trading(regime)` → bool
- `get_risk_adjustment_factor(regime)` → float

### Memory System

**Enhanced Methods:**

- `add_situations(..., regime=dict)` - Add memory with regime context
- `get_memories(..., regime_filter=dict)` - Filter by regime

### Trade Decisions

**New Functions:**

- `set_decision_regime(decision_id, regime)` - Set regime for decision
- `populate_regime_from_prices(decision_id, high, low, close)` - Auto-detect and set

---

**Phase 2 Complete** ✅

All regime detection components are now integrated and ready for use in Phase 3 (RAG decision support).
