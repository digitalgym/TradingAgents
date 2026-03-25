# MTF Analysis Data Collection (Pending Review)

**Status:** Collecting data (started 2026-03-25)

## What's Happening

We're collecting data to answer: "Do H1 levels reliably cause pullbacks on D1 trades?"

Before implementing automated partial closes at H1 resistance, we need evidence.

## Data Being Captured

Every new trade now records:
- Entry timeframe (`decision.timeframe`)
- H1/H4 levels at entry (`decision.mtf_context`)
- Price extremes during trade (`max_favorable_price`, `max_adverse_price`)
- Events: `opposing_position_opened`, `mtf_conflict`, `partial_close`

## When to Review

**Trigger analysis when:**
1. ~50 trades have closed with new data
2. User mentions MTF conflicts, H1 headwinds, or partial close strategy
3. Pattern of losses where partials might have helped

## Quick Check Query

```python
# Count trades with MTF data
from tradingagents.trade_decisions import list_closed_decisions
closed = [d for d in list_closed_decisions() if d.get("timeframe")]
print(f"Trades with MTF data: {len(closed)}")
```

## Full Analysis

See CLEANUP.md section "Data Capture Enhancements for MTF Analysis" for:
- SQL queries to run
- What to look for in results
- Implementation plan if data validates hypothesis

## Do NOT Forget

If user asks about:
- Why D1 trades are getting stopped out
- H1 resistance causing problems
- Whether to take partials
- Opposing positions on same symbol

**Check the MTF conflict data first** before recommending changes.
