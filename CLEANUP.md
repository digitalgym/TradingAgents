# TradingAgents Cleanup & Simplification Tasks

Created: 2026-03-18

---

## CRITICAL — Broken Imports (blocks functionality if called)

### 1. Deleted dataflow modules still imported

These files were deleted from `tradingagents/dataflows/` but are still imported:

| Deleted File | Imported In | Import |
|---|---|---|
| `google.py` | `interface.py:6` | `get_google_news`, `get_google_global_news` |
| `reddit_utils.py` | `local.py:8` | `fetch_top_from_category` |
| `stockstats_utils.py` | `y_finance.py:6` | `StockstatsUtils` |

Also deleted (no remaining imports found):
- `alpha_vantage.py`, `alpha_vantage_common.py`, `alpha_vantage_fundamentals.py`
- `alpha_vantage_indicator.py`, `alpha_vantage_news.py`, `alpha_vantage_stock.py`
- `googlenews_utils.py`

**Fix:** Remove broken imports from `interface.py`, `local.py`, `y_finance.py`. Remove corresponding dead code paths that use these imports. Clean up `VENDOR_METHODS` mapping in `interface.py`.

---

## HIGH — Code Quality

### 2. Dead CLI options for removed data vendors

`cli/utils.py` (lines 332, 342, 350) still shows `alpha_vantage` and `google` as selectable UI options even though the underlying dataflow modules no longer exist.

**Fix:** Remove dead options from CLI menus.

### 3. Quant analyst code duplication (~1500+ lines)

Five analyst files share significant boilerplate:

| File | Lines |
|---|---|
| `smc_mtf_quant.py` | 1,653 |
| `smc_quant.py` | 698 |
| `range_quant.py` | 791 |
| `breakout_quant.py` | 759 |
| `volume_profile_quant.py` | 578 |

Shared patterns:
- Logger setup (~40 lines each, identical)
- `_get_*_logger()` functions
- Trade decision structure
- SMC analysis loops with minor variations

**Fix:** Extract a `BaseQuantAnalyst` class with shared logger/decision/analysis infrastructure. Each quant only implements its specific strategy logic.

### 4. Stale active decisions (from orphaned btc_smc_lmm)

~60 active decisions in `examples/trade_decisions/` whose MT5 positions no longer exist. Previous cleanup attempt (`POST /api/decisions/cleanup-stale`) reported 58 cleaned but re-check showed 60 still active.

**Fix:** Debug why `close_decision()` didn't persist. Likely the function expects parameters that `cleanup_stale_decisions()` doesn't provide correctly. Verify JSON files were actually updated on disk.

---

## MODERATE — Disk Space & File Hygiene

### 5. Orphaned pending predictions (13MB)

`examples/pending_predictions/` has 50+ pickle files from Jan-Feb 2026 for symbols no longer traded (XAGUSD, XPTUSD, COPPER-C).

**Fix:** Archive to `archive/` or delete. These are from before the current BTCUSD/XAUUSD focus.

### 6. Log sprawl (122MB total)

| Directory | Size | Age |
|---|---|---|
| `logs/quant_prompts/` | 51MB | Feb-Mar 17 |
| `logs/smc_quant_prompts/` | 22MB | Mar 10-17 |
| `logs/range_quant_prompts/` | 20MB | Mar 16-17 |
| `logs/quant_automation/` | 20MB | Mar 1-15 |
| `logs/vp_quant_prompts/` | 6.7MB | Mar 5-17 |
| `logs/daily_cycle/` | 1.7MB | Jan 29-Mar 11 |
| `logs/scheduler/` | 1.5MB | Jan 31-Feb 2 |
| `logs/portfolio/` | 1.2MB | Jan 22-Mar 9 |

**Fix:** Archive logs older than 7 days. Delete `logs/scheduler/` (all Jan-Feb, obsolete). Consider log rotation config.

### 7. .gitignore gaps

These generated files should be gitignored:
- `examples/trade_decisions/*.json` (or archive them)
- `examples/pending_predictions/*.pkl`
- `examples/*_context.pkl`
- `web/backend/quant_automation_state*.json`
- `logs/` (all subdirectories)
- `memory_db/chroma.sqlite3`
- `data/state.db`

**Fix:** Update `.gitignore` to exclude generated/runtime files.

---

## LOW — Maintenance

### 8. Orphaned state files (already deleted from git)

13 `quant_automation_state_*.json` files in git status marked as `D` (deleted). These are from removed automation configs (btc_smc_basic, btc_smc_lmm, generic_quant, etc.).

**Status:** Already deleted on disk. Just need to stage the deletions in next commit.

### 9. `tests/test_news.py` deleted

Marked as `D` in git status. No remaining references. Stage the deletion.

### 10. Trade decisions directory size (16MB, 124 files)

Historical decision JSON files from Mar 14-18. Not broken but growing.

**Fix:** Consider a retention policy — archive decisions older than 30 days, or compress to a single archive per month.

---

## NOT Needed (leave as-is)

- **`web/backend/main.py`** (9,435 lines) — Well-structured, no dead code found
- **`automation_configs.json`** — Clean, 10 configs all matching active state files
- **Pickle files** (`portfolio_state.pkl`, `risk_state.pkl`, `agent_weights.pkl`) — Active, in use
- **ChromaDB** (`memory_db/chroma.sqlite3`) — Active memory store

---

## Suggested Order of Execution

1. Fix broken imports (#1) — prevents runtime errors
2. Clean dead CLI options (#2) — quick win
3. Update .gitignore (#7) — prevent future sprawl
4. Stage git deletions (#8, #9) — clean git status
5. Fix stale decisions (#4) — data integrity
6. Archive old files (#5, #6) — disk space
7. Refactor quant analysts (#3) — biggest effort, save for last

---

# Enhanced Trade Review: H1 Conflict Detection & Partial Close System

Created: 2026-03-25

## Problem Statement

1. **D1 trades face H1 headwinds**: Entries on D1 setups often encounter significant pullbacks at H1 resistance/support levels before continuing in the intended direction.

2. **Opposing entries as signals**: When an automation opens a SELL on H1 while BUY positions exist on D1, this is actually a signal that H1 resistance is being respected — a natural trigger for partial profit-taking on the D1 longs.

## Current Architecture

### Trade Management Agent (TMA)
Location: `tradingagents/automation/trade_management_agent.py`

**Already has partial close capability:**
```python
enable_partial_tp: bool = False        # Currently disabled by default
partial_tp_percent: float = 50.0       # Close 50% of position
partial_tp_rr_ratio: float = 1.0       # Trigger at 1:1 R:R
```

**Limitation**: Partial TP is only triggered by R:R ratio — no awareness of H1 levels.

### Position Assumption Review
Location: `tradingagents/automation/position_assumption_review.py`

**Currently:**
- Reviews positions on single timeframe (H1 hardcoded)
- Detects bias shifts, structure breaks, zone mitigation
- Outputs: "hold", "adjust_sl", "adjust_tp", "close"

**Missing:**
- No multi-timeframe conflict detection
- No awareness of opposing positions
- No partial close recommendations

## Proposed Enhancement

### 1. Multi-Timeframe Conflict Detector

New module: `tradingagents/automation/mtf_conflict_detector.py`

```python
@dataclass
class TimeframeConflict:
    """Detected conflict between timeframes."""
    higher_tf: str               # "D1"
    lower_tf: str                # "H1"
    higher_tf_direction: str     # "BUY"
    lower_tf_direction: str      # "SELL"
    conflict_type: str           # "opposing_signal", "resistance_hit", "support_hit"
    lower_tf_zone: Dict          # The H1 zone causing conflict
    severity: str                # "minor", "moderate", "significant"
    suggested_action: str        # "partial_close", "trail_tight", "hold"
    partial_close_pct: float     # Suggested % to close (0-100)

def detect_mtf_conflicts(
    symbol: str,
    position_direction: str,
    position_tf: str,           # Timeframe the position was entered on
    entry_price: float,
    current_price: float,
) -> List[TimeframeConflict]:
    """
    Detect conflicts between position's timeframe and lower timeframes.

    For a D1 BUY position, checks:
    1. H1 resistance zones between entry and TP
    2. H1 bearish bias/structure
    3. H4 intermediate resistance
    """
```

### 2. Opposing Position Detector

Enhance position assumption review to detect when automation opens opposing trades:

```python
def detect_opposing_positions(symbol: str) -> List[Dict]:
    """
    Find positions on same symbol with opposite directions.

    Returns:
        List of conflicts: [
            {
                "long_ticket": 123, "long_tf": "D1", "long_entry": 1950.00,
                "short_ticket": 456, "short_tf": "H1", "short_entry": 1980.00,
                "conflict_zone": {"type": "resistance", "level": 1978.50},
                "suggested_action": "partial_close_long",
                "reason": "H1 resistance respected, take partials on D1 long"
            }
        ]
    """
```

### 3. Enhanced Position Review Report

Extend `PositionAssumptionReport`:

```python
@dataclass
class PositionAssumptionReport:
    # ... existing fields ...

    # NEW: Multi-timeframe awareness
    entry_timeframe: str = "D1"                    # TF the trade was entered on
    conflicting_levels: List[Dict] = field(default_factory=list)  # H1/H4 levels in the way
    opposing_positions: List[Dict] = field(default_factory=list)  # Same symbol, opposite direction

    # NEW: Partial close recommendation
    partial_close_recommended: bool = False
    partial_close_pct: float = 0.0
    partial_close_reason: str = ""
    trail_to_level: Optional[float] = None         # After partial, trail SL here
```

### 4. Integration with Trade Management Agent

Modify TMA to accept "H1 conflict" as a partial TP trigger:

```python
# In TradeManagementConfig
class TradeManagementConfig:
    # ... existing ...

    # NEW: H1 Conflict-based partial close
    enable_conflict_partial: bool = True
    conflict_partial_pct: float = 50.0          # Close 50% when H1 conflicts
    move_sl_to_breakeven_after_partial: bool = True
```

**Trigger Logic:**
```python
# In _manage_positions()
if enable_conflict_partial:
    conflicts = detect_opposing_positions(symbol)
    for conflict in conflicts:
        if conflict["position_ticket"] == ticket:
            if conflict["suggested_action"] == "partial_close":
                # Execute partial close
                result = self._partial_close(ticket, symbol, direction, partial_volume)

                # Move SL to breakeven on remaining position
                if move_sl_to_breakeven_after_partial:
                    modify_position(ticket, sl=entry_price)
```

### 5. Opposing Entry as Trigger (Key Insight)

When quant automation places an opposing trade, it should notify TMA:

```python
# In quant_automation.py, after opening a new position
async def _on_position_opened(self, ticket: int, symbol: str, direction: str):
    """Called after successfully opening a position."""

    # Check for opposing positions on same symbol
    opposing = self._find_opposing_positions(symbol, direction)

    for opp_ticket in opposing:
        # Trigger partial close review
        await self._trigger_partial_close_review(
            ticket=opp_ticket,
            reason=f"Opposing {direction} opened at {symbol}",
            suggested_pct=50.0
        )
```

## Decision Flow

```
D1 BUY position opened
        ↓
Price approaches H1 resistance
        ↓
┌───────────────────────────────────────┐
│ H1 automation detects SHORT setup     │
│ Opens SELL position (separate trade)  │
└───────────────────────────────────────┘
        ↓
Opposing position detector triggers
        ↓
┌───────────────────────────────────────┐
│ For D1 BUY position:                  │
│ • Close 50% (lock in profit)          │
│ • Move SL to breakeven on remaining   │
│ • Trail remaining with H1 zones       │
└───────────────────────────────────────┘
        ↓
If H1 SELL wins → D1 remainder at BE, no loss
If H1 SELL loses → D1 remainder continues to TP
```

## Configuration Options

Add to `QuantAutomationConfig`:

```python
# MTF Conflict Management
mtf_conflict_detection: bool = True
conflict_partial_close_pct: float = 50.0
conflict_trail_to_breakeven: bool = True
opposing_position_partial_trigger: bool = True  # Partial close when opposing entry made
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/positions/conflicts` | GET | All current MTF conflicts for open positions |
| `/api/positions/{ticket}/partial-close` | POST | Manual partial close with reason logging |
| `/api/positions/{ticket}/mtf-analysis` | GET | Full MTF analysis (H1, H4, D1 levels vs price) |

## Implementation Phases

### Phase 1: Detection (read-only)
- `mtf_conflict_detector.py` - detect H1 levels vs D1 positions
- `detect_opposing_positions()` - find conflicting positions
- Add to position review reports (no auto-action yet)

### Phase 2: Manual Triggers
- API endpoint for manual partial close
- UI shows MTF conflicts with "Take Partial" button
- User-initiated, logged

### Phase 3: Automated Partials
- TMA integration with conflict detector
- Opposing position → partial close trigger
- Auto-trail to breakeven after partial

### Phase 4: Learning
- Track: "Did partial + hold work better than full exit?"
- Store outcomes in reflection system
- Tune partial_close_pct based on historical performance

## Files to Modify

| File | Changes |
|------|---------|
| `tradingagents/automation/mtf_conflict_detector.py` | NEW - core detection logic |
| `tradingagents/automation/position_assumption_review.py` | Add MTF conflict fields to report |
| `tradingagents/automation/trade_management_agent.py` | Add conflict-based partial TP trigger |
| `tradingagents/automation/quant_automation.py` | Notify on opposing position open |
| `tradingagents/schemas/position_review.py` | Extend schemas |
| `web/backend/main.py` | New API endpoints |
| `web/frontend/src/components/positions/` | UI for conflicts + partial close |

## Key Insight

> "Opposing entries almost like a trigger for taking partials"

This is exactly right. The system is already detecting the H1 level (that's why it's opening the SELL). We just need to:
1. Recognize this as a partial close signal for the D1 position
2. Execute the partial
3. Protect the remainder with breakeven

The H1 SELL isn't "wrong" — it's the system correctly identifying short-term resistance. We use that signal constructively rather than having two conflicting positions fight each other.

---

# Data Capture Enhancements for MTF Analysis

Created: 2026-03-25

## Goal

Capture data now so we can answer these questions later:
1. Do H1 levels reliably cause pullbacks on D1 trades?
2. Would partial closes at H1 resistance improve outcomes?
3. Are opposing entries (H1 vs D1) predictive of pullbacks?

## Current State

### What We Capture

**At Entry:**
- Symbol, direction, entry/SL/TP, volume
- Setup type, higher_tf_bias, volatility/market regime
- SMC context (entry_zone, zone_strength, with_trend)
- Confidence, pipeline, rationale

**During Trade:**
- breakeven_set, trailing_stop events
- reversal_signal detections

**At Close:**
- Exit price, PnL, exit_reason
- Structured outcome (direction_correct, sl/tp placement)
- max_favorable/max_adverse (often NULL - not being passed)

### What's Missing

| Gap | Impact |
|-----|--------|
| Entry timeframe not stored | Can't filter D1 vs H1 trades |
| No H1 level snapshot at entry | Don't know if we entered near resistance |
| No price path during trade | Don't know if H1 levels were tested |
| No opposing position tracking | Can't correlate H1 signals with D1 pullbacks |
| max_favorable/adverse often NULL | Excursion analysis incomplete |

## Enhancements

### 1. Add Entry Timeframe to Decisions

**File:** `tradingagents/trade_decisions.py`

```python
def store_decision(
    # ... existing params ...
    timeframe: Optional[str] = None,  # NEW: "M15", "H1", "H4", "D1"
)
```

**File:** `tradingagents/automation/quant_automation.py`

Pass `timeframe=self.config.timeframe` to `store_decision()`.

### 2. Snapshot H1/H4 Levels at Entry

**New field in decision:**

```python
"mtf_context": {
    "entry_timeframe": "D1",
    "h1_resistance_levels": [1985.50, 1992.00],  # Nearest H1 resistance at entry time
    "h1_support_levels": [1970.25, 1965.00],
    "h4_resistance_levels": [1998.00],
    "h4_support_levels": [1958.50],
    "distance_to_h1_resistance_pct": 0.35,  # How far entry is from nearest H1 resistance
    "distance_to_h1_support_pct": 0.82,
}
```

**Implementation:** Call `get_smc_zones(symbol, "H1")` at entry time, store nearest levels.

### 3. Track Price Path During Trade (Excursion Log)

**New event type:** `price_snapshot`

```python
add_trade_event(decision_id, "price_snapshot", {
    "price": 1982.50,
    "pnl_pct": 0.45,
    "h1_level_tested": 1985.50,  # If price reached within 0.1% of H1 level
    "h1_level_type": "resistance",
    "minutes_since_entry": 240,
})
```

**Trigger:** Every N minutes while position open, or when price approaches known H1 level.

**Lightweight option:** Just log high/low watermarks per hour.

### 4. Track Opposing Positions

**New event type:** `opposing_position_opened`

```python
add_trade_event(decision_id, "opposing_position_opened", {
    "opposing_ticket": 456,
    "opposing_direction": "SELL",
    "opposing_timeframe": "H1",
    "opposing_entry": 1985.00,
    "this_position_pnl_pct": 0.52,  # P&L at moment opposing opened
    "h1_zone_type": "resistance",
    "h1_zone_level": 1984.50,
})
```

**Trigger:** In `quant_automation._execute_trade()`, after opening, check for opposing positions and log.

### 5. Fix max_favorable/max_adverse Capture

**Problem:** These are params in `close_decision()` but rarely passed.

**Solution:** Track in `_manage_positions()` loop:

```python
# In TMA or quant_automation position management
if ticket not in self._price_extremes:
    self._price_extremes[ticket] = {"high": current_price, "low": current_price}
else:
    self._price_extremes[ticket]["high"] = max(self._price_extremes[ticket]["high"], current_price)
    self._price_extremes[ticket]["low"] = min(self._price_extremes[ticket]["low"], current_price)
```

Then pass to `close_decision()`:
```python
direction = decision.get("action")
if direction == "BUY":
    max_favorable = extremes["high"]
    max_adverse = extremes["low"]
else:
    max_favorable = extremes["low"]
    max_adverse = extremes["high"]
```

### 6. New Event: MTF Conflict Detected

**Event type:** `mtf_conflict`

```python
add_trade_event(decision_id, "mtf_conflict", {
    "conflict_type": "h1_resistance_hit",
    "level": 1985.50,
    "price_at_detection": 1984.80,
    "pnl_pct_at_detection": 0.48,
    "suggested_action": "partial_close",  # For future analysis
    "action_taken": "none",  # Initially just log, don't act
})
```

### 7. Partial Close Events

**Event type:** `partial_close`

```python
add_trade_event(decision_id, "partial_close", {
    "volume_closed": 0.05,
    "volume_remaining": 0.05,
    "close_price": 1985.20,
    "pnl_on_partial": 52.30,
    "reason": "h1_resistance",  # or "rr_target", "manual"
    "sl_moved_to": 1975.00,  # If BE set after partial
})
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)

1. **Add timeframe to store_decision** - simple param
2. **Fix max_favorable/adverse tracking** - add dict in TMA, pass to close_decision
3. **Add opposing_position_opened event** - check after execution

### Phase 2: MTF Context (2-3 hours)

4. **Snapshot H1/H4 levels at entry** - call SMC analysis, store subset
5. **Add mtf_conflict event type** - log when price hits H1 level

### Phase 3: Price Path (optional, more complex)

6. **Periodic price snapshots** - adds data but also storage/complexity
7. **H1 level proximity alerts** - detect when price within X% of level

## Analysis Queries (After Data Collected)

Once we have 50+ closed trades with this data:

```sql
-- Q1: How often do D1 trades hit H1 resistance before TP?
SELECT
    COUNT(*) as total_d1_trades,
    SUM(CASE WHEN events LIKE '%h1_resistance_hit%' THEN 1 ELSE 0 END) as hit_h1_resistance
FROM decisions
WHERE mtf_context->>'entry_timeframe' = 'D1';

-- Q2: Do opposing entries predict pullbacks?
SELECT
    d.decision_id,
    e.data->>'opposing_direction' as opposing_dir,
    d.structured_outcome->>'result' as outcome,
    d.pnl_percent
FROM decisions d
JOIN events e ON d.decision_id = e.decision_id
WHERE e.type = 'opposing_position_opened';

-- Q3: What % of trades would have benefited from partial at H1?
SELECT
    d.decision_id,
    e.data->>'pnl_pct_at_detection' as pnl_at_h1,
    d.pnl_percent as final_pnl,
    CASE WHEN d.pnl_percent < (e.data->>'pnl_pct_at_detection')::float THEN 'partial_better' ELSE 'hold_better' END
FROM decisions d
JOIN events e ON d.decision_id = e.decision_id
WHERE e.type = 'mtf_conflict';
```

## Summary

| Enhancement | Effort | Value | Priority | Status |
|-------------|--------|-------|----------|--------|
| Timeframe in decision | Low | High | P1 | DONE |
| Fix max_favorable/adverse | Low | High | P1 | DONE |
| Opposing position events | Low | High | P1 | DONE |
| H1/H4 level snapshot | Medium | High | P2 | DONE |
| MTF conflict events | Medium | High | P2 | DONE |
| Partial close events | Low | High | P2 | DONE |
| Price path snapshots | High | Medium | P3 | - |

---

## Implementation Complete (2026-03-25)

### Files Modified

1. **`tradingagents/trade_decisions.py`**
   - Added `timeframe` and `mtf_context` params to `store_decision()`
   - Added `capture_mtf_context()` function to snapshot H1/H4 levels at entry
   - Added new event types to `add_trade_event()` docstring

2. **`tradingagents/automation/quant_automation.py`**
   - Import `capture_mtf_context`
   - Call `capture_mtf_context()` before `store_decision()`
   - Pass `timeframe` and `mtf_context` to `store_decision()`
   - Added `_price_extremes` dict for tracking high/low during trade
   - Track price extremes in position management loop
   - Pass `max_favorable_price`/`max_adverse_price` to `close_decision()`
   - Added `_log_opposing_positions()` method - logs event when opposing position opened
   - Added MTF conflict detection in position loop - logs when D1 trade hits H1 level

3. **`tradingagents/automation/trade_management_agent.py`**
   - Added `_price_extremes` dict for tracking
   - Track price extremes in position management loop
   - Pass `max_favorable_price`/`max_adverse_price` to `close_decision()`
   - Log `partial_close` event when TMA takes partial profits

### New Event Types

| Event Type | When Logged | Data Captured |
|------------|-------------|---------------|
| `opposing_position_opened` | New position opens against existing position | opposing ticket, direction, timeframe, entry, current P&L |
| `mtf_conflict` | D1/H4 trade price hits H1 level | conflict type, level, price, P&L at detection |
| `partial_close` | TMA takes partial profits | volume closed/remaining, price, P&L, reason |

### New Decision Fields

```json
{
  "timeframe": "D1",
  "mtf_context": {
    "entry_timeframe": "D1",
    "h1_resistance_levels": [{"price": 1985.50, "strength": 0.8, "source": "ob"}],
    "h1_support_levels": [...],
    "h4_resistance_levels": [...],
    "h4_support_levels": [...],
    "nearest_h1_resistance": 1985.50,
    "nearest_h1_support": 1970.25,
    "distance_to_h1_resistance_pct": 0.35,
    "distance_to_h1_support_pct": 0.82,
    "captured_at": "2026-03-25T10:30:00"
  }
}
```

### What Happens Now

1. **New trades** will have `timeframe` and `mtf_context` populated
2. **During trade**: price extremes tracked, H1 level hits logged
3. **When opposing position opens**: event logged on existing position
4. **At close**: `max_favorable_price` and `max_adverse_price` passed for excursion analysis
5. **After 50+ trades**: run analysis queries to validate MTF partial close hypothesis
