# Testing SMC State Flow

## Issue

User reports SMC not appearing in analyst reasoning even after:

1. Adding SMC fields to AgentState
2. Initializing SMC in create_initial_state
3. Adding SMC context injection in trader/risk analysts

## Hypothesis

The Python process was already running when we made the state changes. The modules were cached, so the new AgentState definition wasn't loaded.

## Next Steps

1. **Kill the existing Python process** (PID 15708)
2. **Start fresh analysis** with new process
3. **Watch for debug output:**
   - `[DEBUG] Added SMC to state: X chars`
   - `[DEBUG] Trader received SMC context (X chars)`
   - `[DEBUG] SMC preview: ...`

## What to Look For

If SMC is working, you'll see in the trader's plan:

- "The D1 order block at $X provides support"
- "Entry at $Y near the 4H FVG zone"
- "Stop below the institutional zone at $Z"
- References to "order blocks", "FVGs", "support zones"

If still not working, debug output will show:

- Whether SMC is added to state (CLI debug)
- Whether trader receives it (trader debug)
- This will tell us where the data is being lost

## Commands to Test

```bash
# Kill existing process
# Then run fresh:
python -m cli.main analyze

# Select:
# - Symbol: silver or XAGUSD
# - Asset type: Commodity
# - Data vendor: MT5
```

Look for the debug messages in the output!
