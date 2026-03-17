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
