# SMC Integration with `analyze` Command

## ✅ Completed

Smart Money Concepts (SMC) analysis is now **automatically integrated** into the `analyze` command for commodity trading with MT5.

## How It Works

### 1. Automatic SMC Analysis

When you run `python -m cli.main analyze` with:

- Asset type: **Commodity**
- Data vendor: **MT5**

The system automatically:

1. ✅ Runs multi-timeframe SMC analysis (1H, 4H, D1)
2. ✅ Detects order blocks, FVGs, BOS, CHOC
3. ✅ Adds SMC context to analyst prompts
4. ✅ Shows SMC validation after analysis
5. ✅ Suggests SMC-based TP/SL levels

### 2. What You'll See

#### During Analysis

```
System: Running Smart Money Concepts analysis...
System: SMC Analysis: 2/3 timeframes bullish - 2 TFs with unmitigated OBs
```

#### After Analysis (Before Execution Prompt)

```
╭─────────────────────────────────────────────────╮
│ Smart Money Concepts Validation                 │
╰─────────────────────────────────────────────────╯

SMC-Suggested Stop Loss:
  $89.50 (below fvg at $90.20)
  Distance: 3.17% | Strength: 75%

SMC-Suggested Take Profits:
  TP1: $95.80 (+3.6%) at order_block zone
  TP2: $98.50 (+6.5%) at fvg zone
  TP3: $102.00 (+10.3%) at order_block zone
```

### 3. SMC Context in Analyst Prompts

The analysts now receive SMC information directly in their prompts. This happens automatically:

**Trader Agent:**

- Receives full SMC analysis showing order blocks, FVGs, BOS, CHOC on 1H/4H/D1
- Instructed to align entry, stop loss, and take profit with institutional zones
- Places stops BELOW support (buys) or ABOVE resistance (sells)
- Targets TPs AT resistance/support where institutions trade

**Risk Analysts:**

- **Risky Analyst:** Uses SMC to argue for aggressive targets at strong resistance zones
- **Safe Analyst:** Uses SMC to ensure stops are beyond institutional zones (not in the middle)
- **Neutral Analyst:** Uses SMC to balance risk with realistic institutional levels

Here's what they see:

````
======================================================================
SMART MONEY CONCEPTS ANALYSIS - XAGUSD
======================================================================

[1H TIMEFRAME]
  Bias: NEUTRAL
  Order Blocks: 0 unmitigated
  Fair Value Gaps: 0 unmitigated

[4H TIMEFRAME]
  Bias: BULLISH
  Order Blocks: 0 unmitigated
  Fair Value Gaps: 2 unmitigated
  ✓ BOS detected: 1 recent
  Nearest Support: $80.16-$82.33 (fvg) | -10.93%

[D1 TIMEFRAME]
  Bias: BULLISH
  Order Blocks: 0 unmitigated
  Fair Value Gaps: 3 unmitigated
  Nearest Support: $80.51-$83.42 (fvg) | -9.75%

======================================================================

This helps analysts:
- **Align TP/SL with institutional levels** - Not arbitrary percentages
- **Consider higher timeframe bias** - D1 for stops, 4H/1H for entries
- **Identify key support/resistance zones** - Order blocks and FVGs
- **Avoid premature stop-outs** - Stops placed BEYOND zones, not in the middle
- **Target realistic exits** - TPs at resistance/support where institutions fill orders
- **Debate with SMC data** - Risk analysts reference specific zones in their arguments

## Example Workflow

### Before SMC Integration

```bash
python -m cli.main analyze
# Select: XAGUSD, Commodity, MT5

# Analysis output:
# Entry: $92.43
# Stop: $90.00 (arbitrary 2.6%)
# TP: $97.00 (arbitrary 5%)

# Problem: Stop might be in middle of support zone
````

### After SMC Integration

```bash
python -m cli.main analyze
# Select: XAGUSD, Commodity, MT5

# System runs SMC automatically:
# "SMC Analysis: 2/3 timeframes bullish - 2 TFs with unmitigated OBs"

# Analysis output includes SMC context
# Analysts see order blocks and FVG zones

# After analysis, SMC validation shows:
# SMC-Suggested Stop: $89.50 (below FVG at $90.20)
# SMC-Suggested TP1: $95.80 (at order_block)

# Better alignment with institutional levels!
```

## Benefits

### 1. Better Stop Placement

- ✅ Stops below unmitigated support zones
- ✅ Won't get stopped out on institutional retests
- ✅ Based on D1 timeframe (stronger levels)

### 2. Better Target Placement

- ✅ TPs at resistance zones where price likely to react
- ✅ Multiple targets for scaling out
- ✅ Aligned with order blocks and FVGs

### 3. Higher Timeframe Confirmation

- ✅ Analysts see if HTF bias is aligned
- ✅ Can adjust plan if timeframes conflict
- ✅ More confident entries

### 4. Automatic Integration

- ✅ No extra commands needed
- ✅ Works seamlessly with existing workflow
- ✅ Only runs for commodity/MT5 mode

## When SMC Runs

SMC analysis runs automatically when **ALL** of these conditions are met:

1. ✅ Asset type is **Commodity**
2. ✅ Data vendor is **MT5**
3. ✅ MT5 is running and logged in

If MT5 is not available, analysis continues without SMC (graceful fallback).

## Manual SMC Commands Still Available

You can still use standalone SMC commands:

```bash
# Analyze SMC separately
python -m cli.main smc XAGUSD

# Get level suggestions
python -m cli.main smc-levels XAGUSD --direction BUY

# Validate a plan
python -m cli.main smc-validate XAGUSD \
  --direction BUY --entry 92.43 --stop 89.50 --target 95.80
```

## What Gets Added to State

The SMC data is added to the graph state:

```python
init_agent_state["smc_context"] = """
SMART MONEY CONCEPTS ANALYSIS - XAGUSD
[1H TIMEFRAME] Bias: NEUTRAL ...
[4H TIMEFRAME] Bias: BULLISH ...
[D1 TIMEFRAME] Bias: BULLISH ...
"""

init_agent_state["smc_analysis"] = {
    '1H': {...},
    '4H': {...},
    'D1': {...}
}
```

Analysts can reference this in their reasoning to align TP/SL with SMC levels.

## Summary

**Before:** Analysts suggested arbitrary TP/SL percentages without considering institutional levels.

**After:** Analysts receive SMC context showing order blocks, FVGs, and key zones. System validates final plan against SMC and suggests better levels if needed.

**Result:** Your trades now align with where institutions are actually trading! 🎯

---

## Technical Details

### Files Modified

- `cli/main.py` - Added SMC analysis to `run_analysis()` function
  - Runs SMC before creating initial state
  - Adds SMC context to state
  - Displays SMC validation after analysis
  - Passes SMC to execution prompt

### Functions Added

- `display_smc_validation()` - Shows SMC-suggested levels after analysis
- Updated `prompt_trade_execution()` - Now accepts SMC parameter

### Integration Points

1. **Before Analysis:** Fetch SMC data from MT5
2. **During Analysis:** Include SMC context in prompts
3. **After Analysis:** Display SMC validation
4. **At Execution:** Use SMC for intelligent order placement

### Error Handling

If SMC analysis fails (MT5 not connected, etc.):

- System logs: "SMC analysis skipped: [error]"
- Analysis continues normally without SMC
- No disruption to workflow
