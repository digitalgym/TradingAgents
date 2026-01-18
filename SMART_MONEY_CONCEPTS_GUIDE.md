# Smart Money Concepts (SMC) Integration Guide

## Problem Solved

**Before:** TP and SL levels didn't align with institutional order blocks and unmitigated support/resistance zones, causing premature exits and missed opportunities.

**After:** System now analyzes order blocks, FVGs, BOS, CHOC on 1H+ timeframes to suggest TP/SL levels that align with smart money zones.

## What is Smart Money Concepts?

SMC tracks institutional trading behavior through:

### 1. **Order Blocks (OB)**

Last opposing candle before a strong move. Represents institutional entry zones.

- **Bullish OB:** Last down candle before strong up move → Support
- **Bearish OB:** Last up candle before strong down move → Resistance

**Why it matters:** Institutions leave "footprints" - these zones often hold on retests.

### 2. **Fair Value Gaps (FVG)**

3-candle imbalance where price moves so fast it leaves a gap.

- **Bullish FVG:** Gap between candle 1 high and candle 3 low
- **Bearish FVG:** Gap between candle 1 low and candle 3 high

**Why it matters:** Price often returns to "fill" these gaps before continuing.

### 3. **Break of Structure (BOS)**

Breaking a swing high/low in the trend direction → Continuation signal.

- **Bullish BOS:** Breaking swing high in uptrend
- **Bearish BOS:** Breaking swing low in downtrend

### 4. **Change of Character (CHOC)**

Breaking a swing high/low against the trend → Reversal signal.

- **Bullish CHOC:** Breaking swing high in downtrend (potential reversal up)
- **Bearish CHOC:** Breaking swing low in uptrend (potential reversal down)

### 5. **Mitigation**

When price returns to an OB or FVG zone, it's "mitigated" (used up).

**Unmitigated zones** = Fresh institutional levels that haven't been retested yet.

## CLI Commands

### 1. Analyze SMC

```bash
# Analyze order blocks, FVGs, BOS, CHOC
python -m cli.main smc XAUUSD

# Specify timeframes
python -m cli.main smc XAUUSD --timeframes "1H,4H,D1"
```

**Output:**

```
═══ SMC ANALYSIS: XAUUSD ═══

1H TIMEFRAME
┌─────────────────────────────┬──────────────────┐
│ Metric                      │ Value            │
├─────────────────────────────┼──────────────────┤
│ Current Price               │ $2650.00         │
│ Market Bias                 │ BULLISH          │
│ Order Blocks (Unmitigated)  │ 3                │
│ Fair Value Gaps (Unmitigated)│ 2              │
│ Recent BOS                  │ ✓ 2              │
└─────────────────────────────┴──────────────────┘

  Support: $2630.00-$2635.00 (order_block) | -0.57%
  Resistance: $2680.00-$2690.00 (fvg) | +1.13%

4H TIMEFRAME
...

D1 TIMEFRAME
...

═══ HIGHER TIMEFRAME ALIGNMENT ═══

All timeframes aligned BULLISH
Bias: BULLISH
Strength: STRONG
```

### 2. Get SMC-Based Levels

```bash
# Get stop loss and take profit suggestions
python -m cli.main smc-levels XAUUSD --direction BUY

# With specific entry
python -m cli.main smc-levels XAUUSD --direction BUY --entry 2650
```

**Output:**

```
═══ SMC LEVELS: XAUUSD BUY ═══

Entry Price: $2650.00
Direction: BUY

STOP LOSS SUGGESTION:

  Price: $2628.50
  Zone: $2630.00-$2635.00
  Source: order_block
  Strength: 85%
  Distance: 0.81%
  Reason: Below order_block zone at $2630.00

TAKE PROFIT SUGGESTIONS:

┌────────┬──────────┬─────────────────────┬──────────┬─────────────┐
│ Target │ Price    │ Zone                │ Distance │ Source      │
├────────┼──────────┼─────────────────────┼──────────┼─────────────┤
│ TP1    │ $2680.00 │ $2680.00-$2690.00   │ +1.1%    │ fvg         │
│ TP2    │ $2710.00 │ $2710.00-$2720.00   │ +2.3%    │ order_block │
│ TP3    │ $2750.00 │ $2750.00-$2760.00   │ +3.8%    │ order_block │
└────────┴──────────┴─────────────────────┴──────────┴─────────────┘
```

### 3. Validate Trade Plan

```bash
# Check if your plan aligns with SMC levels
python -m cli.main smc-validate XAUUSD \
  --direction BUY \
  --entry 2650 \
  --stop 2640 \
  --target 2700
```

**Output:**

```
═══ SMC VALIDATION: XAUUSD ═══

Trade Plan:
  Direction: BUY
  Entry: $2650.00
  Stop Loss: $2640.00
  Take Profit: $2700.00

Validation Score: 60/100
Valid: ✗ NO

Issues:
  ⚠️  Stop loss ($2640.00) above support zone - may get stopped out prematurely
  ⚠️  TP ($2700.00) before resistance - leaving profit on table

Suggestions:
  Consider moving stop to $2628.50 (below support)
  Consider TP at $2710.00 (resistance zone)
```

## Python API

### Analyze Multi-Timeframe SMC

```python
from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc

# Analyze 1H, 4H, D1
mtf_analysis = analyze_multi_timeframe_smc(
    symbol="XAUUSD",
    timeframes=['1H', '4H', 'D1']
)

# Access each timeframe
d1_analysis = mtf_analysis['D1']

print(f"Bias: {d1_analysis['bias']}")
print(f"Unmitigated OBs: {d1_analysis['order_blocks']['unmitigated']}")
print(f"Unmitigated FVGs: {d1_analysis['fair_value_gaps']['unmitigated']}")

# Nearest support/resistance
if d1_analysis['nearest_support']:
    s = d1_analysis['nearest_support']
    print(f"Support: ${s['bottom']:.2f}-${s['top']:.2f} ({s['type']})")
```

### Get SMC Stop Loss

```python
from tradingagents.dataflows.smc_utils import suggest_smc_stop_loss

stop_suggestion = suggest_smc_stop_loss(
    smc_analysis=d1_analysis,
    direction="BUY",
    entry_price=2650.00,
    max_distance_pct=3.0  # Max 3% away
)

if stop_suggestion:
    print(f"Stop: ${stop_suggestion['price']:.2f}")
    print(f"Zone: ${stop_suggestion['zone_bottom']:.2f}-${stop_suggestion['zone_top']:.2f}")
    print(f"Source: {stop_suggestion['source']}")
    print(f"Reason: {stop_suggestion['reason']}")
```

### Get SMC Take Profits

```python
from tradingagents.dataflows.smc_utils import suggest_smc_take_profits

tp_suggestions = suggest_smc_take_profits(
    smc_analysis=d1_analysis,
    direction="BUY",
    entry_price=2650.00,
    num_targets=3
)

for tp in tp_suggestions:
    print(f"TP{tp['number']}: ${tp['price']:.2f} (+{tp['distance_pct']:.1f}%)")
    print(f"  Zone: ${tp['zone_bottom']:.2f}-${tp['zone_top']:.2f}")
    print(f"  Source: {tp['source']}")
```

### Validate Trade Plan

```python
from tradingagents.dataflows.smc_utils import validate_trade_against_smc

validation = validate_trade_against_smc(
    direction="BUY",
    entry_price=2650.00,
    stop_loss=2640.00,
    take_profit=2700.00,
    smc_analysis=d1_analysis
)

print(f"Score: {validation['score']}/100")
print(f"Valid: {validation['valid']}")

for issue in validation['issues']:
    print(f"Issue: {issue}")

for suggestion in validation['suggestions']:
    print(f"Suggestion: {suggestion}")
```

### Check HTF Alignment

```python
from tradingagents.dataflows.smc_utils import get_htf_bias_alignment

alignment = get_htf_bias_alignment(mtf_analysis)

print(f"Aligned: {alignment['aligned']}")
print(f"Bias: {alignment['bias']}")
print(f"Strength: {alignment['strength']}")
print(f"Message: {alignment['message']}")
```

## How It Improves TP/SL Placement

### Before SMC

```
Entry: $2650
Stop: $2640 (arbitrary 10 points)
TP: $2700 (arbitrary 50 points)

Problem: Stop at $2640 is in the middle of a support zone ($2630-$2645)
→ Gets stopped out when price retests support
→ Then price bounces and hits target without you
```

### After SMC

```
Entry: $2650
Stop: $2628 (below order block at $2630-$2635)
TP1: $2680 (at FVG resistance)
TP2: $2710 (at order block resistance)
TP3: $2750 (at next order block)

Benefit: Stop below institutional zone, won't get prematurely stopped
→ TPs at actual resistance zones where price likely to react
→ Better risk-reward and higher win rate
```

## Integration with Analysis

### In Your Analysis Workflow

```python
from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    format_smc_for_prompt,
    suggest_smc_stop_loss,
    suggest_smc_take_profits
)

# 1. Run SMC analysis
mtf_smc = analyze_multi_timeframe_smc("XAUUSD", ['1H', '4H', 'D1'])

# 2. Add to LLM prompt
smc_context = format_smc_for_prompt(mtf_smc, "XAUUSD")
# Include this in your analyst prompts

# 3. Get SMC-based levels
d1 = mtf_smc['D1']
stop = suggest_smc_stop_loss(d1, "BUY", entry_price=2650)
tps = suggest_smc_take_profits(d1, "BUY", entry_price=2650)

# 4. Use in trade plan
print(f"Entry: $2650")
print(f"Stop: ${stop['price']:.2f} (below {stop['source']})")
for tp in tps:
    print(f"TP{tp['number']}: ${tp['price']:.2f} (at {tp['source']})")
```

## Key Concepts

### Unmitigated vs Mitigated

**Unmitigated:** Fresh zone that hasn't been retested

- **Strong:** Price likely to react when it reaches this zone
- **Use for:** TP/SL placement

**Mitigated:** Zone that's been retested

- **Weak:** Already "used up" by institutions
- **Ignore:** Don't use for TP/SL

### Higher Timeframe = Stronger

Order blocks and FVGs on higher timeframes are stronger:

- **1H OB:** Weak, may break easily
- **4H OB:** Moderate strength
- **D1 OB:** Strong, likely to hold
- **W1 OB:** Very strong, major institutional level

**Recommendation:** Use D1 or higher for stop loss placement.

### BOS vs CHOC

**BOS (Continuation):**

- Trend is intact
- Safe to trade in trend direction
- Look for entries on pullbacks to OBs

**CHOC (Reversal Warning):**

- Trend may be ending
- Be cautious with trend trades
- Consider counter-trend setups

## Example: Complete Workflow

```bash
# 1. Analyze SMC
python -m cli.main smc XAUUSD

# Output shows:
# - D1 bias: BULLISH
# - Recent BOS: 2 (continuation confirmed)
# - Support OB: $2630-$2635
# - Resistance FVG: $2680-$2690

# 2. Get suggested levels
python -m cli.main smc-levels XAUUSD --direction BUY --entry 2650

# Output suggests:
# - Stop: $2628 (below OB)
# - TP1: $2680 (at FVG)
# - TP2: $2710 (at next OB)

# 3. Create your plan
# Entry: $2650
# Stop: $2628 (0.83% risk)
# TP1: $2680 (1.13% gain, 1.36R)
# TP2: $2710 (2.26% gain, 2.72R)

# 4. Validate it
python -m cli.main smc-validate XAUUSD \
  --direction BUY --entry 2650 --stop 2628 --target 2680

# Output: Score 100/100 ✓ Aligned with SMC levels

# 5. Execute with confidence
python -m cli.main execute-plan my_plan.txt --live
```

## Benefits

### 1. Better Stop Placement

- ✅ Below institutional zones (won't get prematurely stopped)
- ✅ Accounts for volatility and structure
- ✅ Higher win rate

### 2. Better Target Placement

- ✅ At actual resistance/support zones
- ✅ Where price likely to react
- ✅ Multiple targets for scaling out

### 3. Higher Timeframe Confirmation

- ✅ D1 bias alignment
- ✅ Stronger levels
- ✅ More reliable signals

### 4. Avoid Common Mistakes

- ❌ Stop in middle of support zone
- ❌ TP before resistance
- ❌ Ignoring institutional levels
- ❌ Trading against HTF bias

## Summary

The SMC system gives you:

1. **Order Block Detection** - Find institutional entry zones on 1H+
2. **FVG Detection** - Identify imbalance zones
3. **BOS/CHOC Detection** - Confirm trend or reversal
4. **Unmitigated Zones** - Fresh levels that haven't been retested
5. **Smart TP/SL Suggestions** - Aligned with institutional levels
6. **Multi-Timeframe Analysis** - D1 for stops, 4H/1H for entries
7. **Trade Validation** - Check if your plan aligns with SMC

**Result:** Your TP and SL levels now align with where institutions are actually trading, not arbitrary percentages! 🎯
