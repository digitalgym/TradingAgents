# Migration: Trade Decisions from Files to Database

## Problem Statement

Trade decisions are currently stored as JSON files in `examples/trade_decisions/`. This causes issues when the quant automation runs on a remote worker:

1. **Silent failures**: If `store_decision()` fails (disk error, permissions), the trade executes in MT5 but no decision record is created
2. **No tracking**: Positions can't be linked back to decisions for reflection/learning
3. **Lost trades**: Recent example - 3 trades executed (tickets 1666788, 1671695, 1692240) showed "Executed" in UI but had no decision files

## Current Architecture

### File-Based Storage

```
Location: examples/trade_decisions/
Format: {SYMBOL}_{YYYYMMDD_HHMMSS}.json
Index files: _ticket_index.json, _active_index.json
Context: {decision_id}_context.pkl (large analysis context)
```

### Key File: `tradingagents/trade_decisions.py`

**27 functions** that interact with decisions:

| Function | Purpose | DB Migration Notes |
|----------|---------|-------------------|
| `store_decision()` | Create new decision | INSERT |
| `load_decision()` | Load by ID | SELECT by decision_id |
| `close_decision()` | Update with exit data | UPDATE |
| `add_trade_event()` | Append to events array | UPDATE (JSONB append) |
| `list_decisions()` | List all/filtered | SELECT with filters |
| `list_active_decisions()` | List status='active' | SELECT WHERE status='active' |
| `list_failed_decisions()` | List status='failed' | SELECT WHERE status='failed' |
| `list_closed_decisions()` | List status='closed' | SELECT WHERE status='closed' |
| `find_decision_by_ticket()` | Find by MT5 ticket | SELECT WHERE mt5_ticket=? |
| `link_decision_to_ticket()` | Update ticket link | UPDATE mt5_ticket |
| `mark_decision_reviewed()` | Set reviewed flag | UPDATE reviewed_at |
| `cancel_decision()` | Set status='cancelled' | UPDATE status |
| `set_smc_context()` | Update SMC context | UPDATE smc_context |
| `set_decision_regime()` | Update regime info | UPDATE regime fields |
| `get_decision_stats()` | Aggregate stats | SELECT with aggregation |
| `get_smc_pattern_stats()` | Pattern statistics | SELECT with GROUP BY |
| `get_structured_outcome_stats()` | Outcome analysis | SELECT with aggregation |
| `cleanup_stale_decisions()` | Find orphaned decisions | SELECT + DELETE |
| `reconcile_decisions()` | Sync with MT5 history | Complex: SELECT + UPDATE |
| `get_trade_memories()` | Generate lesson text | SELECT recent + format |
| `_update_ticket_index()` | Index maintenance | N/A (replaced by DB index) |
| `_load_ticket_index()` | Index maintenance | N/A |
| `_add_to_active_index()` | Index maintenance | N/A |
| `_remove_from_active_index()` | Index maintenance | N/A |

## Decision Schema

```python
{
    # Core identification
    "decision_id": str,           # PK: "XAUUSD_20260320_064435"
    "symbol": str,                # Index
    "mt5_ticket": int,            # Unique index (nullable)

    # Decision details
    "decision_type": str,         # OPEN, ADJUST, CLOSE, HOLD
    "action": str,                # BUY, SELL, MODIFY_SL, etc.
    "rationale": str,             # Text (up to 500 chars typically)
    "source": str,                # "btcusd_smc_quant", "manual", etc.

    # Trade parameters
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "volume": float,

    # Timestamps
    "created_at": datetime,
    "exit_date": datetime,        # Nullable

    # Status
    "status": str,                # active, failed, closed, cancelled, retried
    "execution_error": str,       # Nullable

    # Setup classification
    "setup_type": str,            # fvg_bounce, ob_bounce, etc.
    "higher_tf_bias": str,        # bullish, bearish, neutral
    "confluence_score": float,    # 0-10
    "confluence_factors": list,   # JSONB array

    # SMC context (nested object)
    "smc_context": {
        "setup_type": str,
        "entry_zone": str,
        "entry_zone_strength": float,
        "with_trend": bool,
        "higher_tf_aligned": bool,
        "confluences": list,
        "zone_tested_before": bool
    },

    # Market context
    "volatility_regime": str,
    "market_regime": str,
    "session": str,

    # Outcome (filled when closed)
    "exit_price": float,
    "pnl": float,
    "pnl_percent": float,
    "outcome_notes": str,
    "was_correct": bool,
    "exit_reason": str,

    # Risk/reward
    "confidence": float,          # 0.0-1.0
    "pipeline": str,
    "trailing_stop_atr_multiplier": float,
    "rr_planned": float,
    "rr_realized": float,

    # Learning signals
    "reward_signal": float,
    "sharpe_contribution": float,
    "drawdown_impact": float,
    "pattern_tags": list,         # JSONB array

    # Structured outcome (nested)
    "structured_outcome": {
        "result": str,            # win, loss, breakeven
        "returns_pct": float,
        "pnl_pips": float,
        "direction_correct": bool,
        "sl_placement": str,
        "tp_placement": str,
        "entry_quality": str,
        "entry_timing": str,
        "exit_type": str,
        "lessons": list
    },

    # Event log
    "events": list,               # JSONB array of event objects

    # Context reference
    "has_context": bool,
    "reviewed_at": datetime       # Nullable
}
```

## Proposed Database Schema

```sql
CREATE TABLE trade_decisions (
    -- Primary key
    decision_id VARCHAR(50) PRIMARY KEY,

    -- Core fields
    symbol VARCHAR(20) NOT NULL,
    mt5_ticket BIGINT UNIQUE,
    decision_type VARCHAR(20) NOT NULL,
    action VARCHAR(20) NOT NULL,
    rationale TEXT,
    source VARCHAR(100),

    -- Trade parameters
    entry_price DECIMAL(18,8),
    stop_loss DECIMAL(18,8),
    take_profit DECIMAL(18,8),
    volume DECIMAL(18,8),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exit_date TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    execution_error TEXT,

    -- Classification
    setup_type VARCHAR(50),
    higher_tf_bias VARCHAR(20),
    confluence_score DECIMAL(4,2),
    confluence_factors JSONB DEFAULT '[]',

    -- SMC context (JSONB for flexibility)
    smc_context JSONB DEFAULT '{}',

    -- Market context
    volatility_regime VARCHAR(20),
    market_regime VARCHAR(30),
    session VARCHAR(20),

    -- Outcome
    exit_price DECIMAL(18,8),
    pnl DECIMAL(18,8),
    pnl_percent DECIMAL(8,4),
    outcome_notes TEXT,
    was_correct BOOLEAN,
    exit_reason VARCHAR(30),

    -- Confidence & pipeline
    confidence DECIMAL(4,3),
    pipeline VARCHAR(50),
    trailing_stop_atr_multiplier DECIMAL(4,2),
    rr_planned DECIMAL(6,3),
    rr_realized DECIMAL(6,3),

    -- Learning signals
    reward_signal DECIMAL(8,4),
    sharpe_contribution DECIMAL(8,4),
    drawdown_impact DECIMAL(8,4),
    pattern_tags JSONB DEFAULT '[]',

    -- Structured outcome (JSONB)
    structured_outcome JSONB DEFAULT '{}',

    -- Event log (JSONB array)
    events JSONB DEFAULT '[]',

    -- Context flag
    has_context BOOLEAN DEFAULT FALSE
);

-- Indexes for common queries
CREATE INDEX idx_decisions_symbol ON trade_decisions(symbol);
CREATE INDEX idx_decisions_status ON trade_decisions(status);
CREATE INDEX idx_decisions_created ON trade_decisions(created_at DESC);
CREATE INDEX idx_decisions_source ON trade_decisions(source);
CREATE INDEX idx_decisions_symbol_status ON trade_decisions(symbol, status);

-- Context storage (separate table for large pickle data)
CREATE TABLE decision_contexts (
    decision_id VARCHAR(50) PRIMARY KEY REFERENCES trade_decisions(decision_id),
    context_data BYTEA NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Migration Strategy

### Phase 1: Add DB Support (Parallel Write)

1. Add database connection to `trade_decisions.py`
2. Modify `store_decision()` to write to BOTH file and DB
3. Add retry logic for DB writes
4. Log any DB write failures (don't fail the trade)

### Phase 2: Read from DB

1. Modify read functions to prefer DB over files
2. Fallback to files if DB record not found (backward compat)
3. Add migration script to import existing JSON files

### Phase 3: Remove File Dependency

1. Remove file writes from `store_decision()`
2. Remove index file maintenance
3. Archive old JSON files
4. Remove file-based helper functions

## Key Considerations

### Connection Management

The quant automation runs in a subprocess on the home machine. It needs:
- Async connection pool (use `asyncpg` like mt5_worker)
- Connection string from environment variable
- Graceful handling of connection failures

### Large Context Data

The `_context.pkl` files can be large. Options:
1. Store as BYTEA in separate `decision_contexts` table
2. Keep pickle files but store path in DB
3. Compress before storing (gzip)

### Backward Compatibility

During migration:
- Read from DB first, fallback to files
- Write to both DB and files
- Migration script to bulk import existing decisions

### Error Handling

Critical fix needed regardless of migration:
```python
try:
    decision_id = store_decision(...)
except Exception as e:
    logger.error(f"Failed to store decision: {e}")
    # Consider: should we still mark result.executed = True?
```

## Files to Modify

1. **`tradingagents/trade_decisions.py`** - All 27 functions
2. **`tradingagents/automation/quant_automation.py`** - Error handling around store_decision
3. **`web/backend/main.py`** - Add decision API endpoints
4. **New: `tradingagents/db/decisions.py`** - Database layer

## Existing Infrastructure

- **Postgres**: Already used by mt5_worker for commands/status
- **asyncpg**: Already a dependency
- **state_store.py**: Uses SQLite for local state (not Postgres)

## Next Steps

1. Add error handling around `store_decision()` calls (immediate fix)
2. Create database schema and migration script
3. Implement DB-backed `store_decision()` with file fallback
4. Test with quant automation
5. Migrate read functions
6. Remove file dependency
