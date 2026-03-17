# Learning & Reflection System Architecture

This document describes the TradingAgents learning pipeline — how trade outcomes become lessons that improve future decisions.

---

## Overview

The system learns from closed trades through a multi-layer pipeline:

```
Trade Closes → Decision Closed → Outcome Analysis → Reflection → Memory Storage → Agent Retrieval
```

Two automation types trigger this pipeline differently:
- **Portfolio Automation**: Full reflection (LLM-based agent reflections + SMC patterns)
- **Quant Automation**: SMC pattern reflection only (rule-based, no LLM)

---

## Decision Lifecycle

### 1. Decision Created (`store_decision()`)

File: `tradingagents/trade_decisions.py`
Storage: `examples/trade_decisions/{symbol}_{YYYYMMDD_HHMMSS}.json`

Fields stored:
- Core: symbol, direction, entry_price, stop_loss, take_profit, volume
- Classification: setup_type, higher_tf_bias, confluence_score, confluence_factors
- SMC context: entry_zone, entry_zone_strength, with_trend, higher_tf_aligned, confluences, zone_tested_before
- Market context: volatility_regime, market_regime, session
- Analysis context: saved as `{id}_context.pkl` if available

Setup types: `fvg_bounce`, `ob_bounce`, `liquidity_sweep`, `choch`, `bos`, `trend_continuation`

### 2. Decision Closed (`close_decision()`)

Triggered when MT5 reports a position closed. Adds:
- exit_price, pnl, pnl_percent, was_correct, exit_reason
- **Structured outcome** (`analyze_trade_outcome()`):
  - `result`: win/loss/breakeven
  - `direction_correct`: did price move our way?
  - `sl_placement`: too_tight/appropriate/too_wide
  - `tp_placement`: too_ambitious/appropriate/too_conservative
  - `exit_type`: tp_hit/sl_hit/manual_close/trailing_stop/time_exit
  - `entry_quality`: good/poor/neutral
  - `max_favorable_pips`, `max_adverse_pips` (excursion analysis)
  - `lessons`: auto-generated strings from outcome analysis
- **Reward signal** (from `RewardCalculator`): realized_rr, sharpe_contribution, drawdown_impact

### 3. Reflection Triggered

After closing, the automation calls reflection to generate lessons.

---

## Reflection Pipelines

### Quant Automation (current XAUUSD/BTCUSD instances)

File: `tradingagents/automation/quant_automation.py` (lines 1383-1506)

**Flow:**
1. `_close_decision_for_ticket(ticket)` — finds decision linked to MT5 ticket
2. Calls `close_decision()` with exit data from MT5
3. Updates guardrails (circuit breaker trade count)
4. Calls `_reflect_on_closed_trade(decision_id)`

**`_reflect_on_closed_trade()` does:**
1. Loads closed decision from JSON
2. Extracts setup_type from `smc_context` or infers from rationale text:
   - "fvg" → `fvg_entry`
   - "order block" → `ob_entry`
   - "liquidity" + "sweep" → `liquidity_sweep`
   - "breakout"/"bos" → `bos`
   - "choch" → `choch`
   - "volume"+"poc/val/vah" → `volume_profile`
   - "mean reversion" → `mean_reversion`
3. Creates `Reflector` instance **without LLM** (uses `__new__`)
4. Calls `reflector.reflect_smc_pattern(decision, smc_memory)`
5. Stores pattern in ChromaDB `smc_patterns` collection

**Limitation**: Only generates SMC pattern lessons. Does NOT create general agent memories (bull, bear, trader, judge, risk_manager).

### Portfolio Automation

File: `tradingagents/automation/portfolio_automation.py` (lines 964-1094)

**Flow (`run_evening_reflect()`):**
1. Gets all active decisions
2. Checks MT5 history for closed positions
3. For each closed trade: calls `close_decision()` + updates guardrails
4. **Full reflection** (if context pickle available):
   - Loads full multi-agent state from `{id}_context.pkl`
   - Calls `graph.reflect_and_remember(pnl_percent, decision=closed_decision)`
   - This reflects on ALL agents: bull, bear, trader, judge, risk_manager
   - Also stores SMC pattern
5. Generates reflection report

**`reflect_and_remember()`** calls `reflector.reflect_with_smc()` which:
- Runs per-agent reflections (LLM generates lessons from each agent's perspective)
- Stores in agent-specific ChromaDB collections
- Runs SMC pattern reflection (rule-based)

---

## Memory System

File: `tradingagents/agents/utils/memory.py`
Storage: `memory_db/chroma.sqlite3` (ChromaDB vector database)

### Tier System

| Tier | Criteria | Weight | Decay |
|------|----------|--------|-------|
| SHORT | Default for new memories, <2% returns | 0.5 | 30 days |
| MID | 3+ references or 2-5% returns | 0.3 | Slower |
| LONG | 5%+ returns or promoted from MID | 0.2 | Persistent |

### FinancialSituationMemory

Per-agent collections in ChromaDB. Stores situation + recommendation pairs.

**Metadata per memory:**
- confidence (0.0-1.0), tier, timestamp
- prediction_correct, market_regime, volatility_regime
- outcome_quality, reference_count, validated_count, invalidated_count

**Agent-specific retrieval configs (`AGENT_MEMORY_CONFIG`):**
- bull_researcher: 3 matches, 0.4 min confidence, 45-day half-life
- bear_researcher: 3 matches, 0.4 min confidence, 45-day half-life
- trader: 4 matches, 0.5 min confidence, 30-day half-life, includes SMC
- invest_judge: 3 matches, 0.5 min confidence, 60-day half-life
- risk_manager: 2 matches (fewer, higher quality), 0.6 min confidence, 90-day half-life

**Confidence scoring:**
- Correct prediction: `min(0.5 + abs(returns) / 10, 1.0)`
- Incorrect prediction: `max(0.2, 0.5 - abs(returns) / 10)`

**Tier promotion (`_check_tier_promotion()`):**
- SHORT → MID: 3+ references or good outcome
- MID → LONG: 5%+ returns
- LONG → MID: Multiple invalidations

**Memory validation (on use):**
- Trade won → validated_count += 1, confidence += 0.05
- Trade lost → invalidated_count += 1, may reduce confidence

### SMCPatternMemory

Collection: `smc_patterns` in ChromaDB

Stores per-pattern outcomes with rich metadata:
- decision_id, symbol, setup_type, direction
- entry_zone, entry_zone_strength, with_trend, higher_tf_aligned
- confluences, zone_tested_before
- was_win, returns_pct, direction_correct
- sl_placement, tp_placement, exit_type
- lesson (generated rule-based text, max 500 chars)

**Query:** `get_patterns_by_setup(setup_type, symbol?, direction?, limit=20)`

### Embedding Models

Configurable in config. Options:
- FastEmbed: `BAAI/bge-small-en-v1.5` (lightweight, default)
- Sentence-transformers: `all-MiniLM-L6-v2`
- OpenAI: `text-embedding-3-small`
- Ollama: `nomic-embed-text`

---

## Reflection Logic

File: `tradingagents/graph/reflection.py`

### Per-Agent Reflection (LLM-based, portfolio automation only)

For each agent type (bull, bear, trader, judge, risk_manager):
1. Extracts relevant state (market report, sentiment, debate history)
2. Sends to LLM with system prompt for lesson analysis
3. LLM generates: reasoning, improvement suggestions, summary lesson
4. Stores in agent's memory collection with tier/confidence

### SMC Pattern Reflection (rule-based, both automations)

`reflect_smc_pattern(decision, smc_memory)`:
1. Extracts setup_type and structured_outcome from closed decision
2. Generates lesson based on outcome analysis:
   - Win + with_trend → "Setup worked well WITH trend"
   - Win + high zone strength → "High strength zone provided good entry"
   - Loss + counter-trend → "Failed counter-trend. Consider only with-trend"
   - Direction correct + SL hit → "Direction correct but SL too tight"
   - Confluence present → "Confluences (X, Y) supported the trade"
3. Stores via `smc_memory.store_pattern()`

---

## Learning Modules

Directory: `tradingagents/learning/`

### Online RL — Agent Weight Updates (`online_rl.py`)

**OnlineRLUpdater**: Adjusts bull/bear/market agent weights based on performance.
- Default weights: bull=0.33, bear=0.33, market=0.34
- Update trigger: every 30 closed trades
- Score: `win_rate * normalized_reward * sqrt(sample_size)`
- Momentum-based update: learning_rate=0.1, momentum=0.9
- Weights saved to `examples/agent_weights.pkl`

### Reward Calculation (`reward.py`)

**RewardCalculator**: Multi-factor reward signal.
- Components: realized_rr (40%), sharpe_contribution (30%), drawdown_impact (30%)
- Range: [-5.0, 5.0]
- Used for agent weight updates and pattern quality scoring

### Pattern Discovery (`pattern_analyzer.py`)

**PatternAnalyzer**: Clusters trades to find winning/losing patterns.
- Groups by: setup_type, regime, session, confluence level
- Quality: excellent (65%+ WR, 1.5+ RR), good, neutral, poor
- Generates recommendations: "INCREASE focus on X", "AVOID Y"

### Trade Similarity RAG (`trade_similarity.py`)

**TradeSimilaritySearch**: Finds similar historical trades for decision support.
- Similarity factors: symbol (0.2), direction (0.2), regime (0.5), setup characteristics
- Returns confidence adjustment: -0.3 to +0.3 based on historical performance
- Used during analysis to adjust confidence before execution

---

## Why You May Have No Lessons

Lessons require **closed trades**. The chain that must complete:

1. Auto Execute must be ON → trades are placed in MT5
2. Positions must close (SL/TP hit or manual) → MT5 reports closure
3. `_close_decision_for_ticket()` fires → decision record closed
4. `_reflect_on_closed_trade()` fires → lesson stored in ChromaDB

**Common blockers:**
- Auto Execute disabled (most common)
- No SL/TP on signals → trades not placed even with auto_execute on
- Positions still open → no reflection yet
- Pipeline doesn't generate setup_type → SMC reflection skipped

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/stats` | GET | Collection counts |
| `/api/memory/query` | POST | Semantic search in collection |
| `/api/memory/lessons` | GET | Filtered lesson list with summary |
| `/api/memory/reflect` | POST | Trigger reflection (async) |
| `/api/memory/reflect/status/{id}` | GET | Poll reflection status |
| `/api/memory/{collection}/{id}` | DELETE | Remove specific memory |

---

## Key Files

| File | Purpose |
|------|---------|
| `tradingagents/agents/utils/memory.py` | FinancialSituationMemory, SMCPatternMemory |
| `tradingagents/graph/reflection.py` | Reflector class, all reflection methods |
| `tradingagents/trade_decisions.py` | Decision storage, closing, outcome analysis |
| `tradingagents/automation/quant_automation.py:1383-1506` | Quant reflection (SMC only) |
| `tradingagents/automation/portfolio_automation.py:964-1094` | Portfolio reflection (full) |
| `tradingagents/learning/online_rl.py` | Agent weight updates |
| `tradingagents/learning/reward.py` | Reward signal calculation |
| `tradingagents/learning/pattern_analyzer.py` | Pattern clustering |
| `tradingagents/learning/trade_similarity.py` | Similar trade RAG |
| `memory_db/chroma.sqlite3` | ChromaDB persistent storage |
| `examples/trade_decisions/` | Decision JSON + context pickle files |
| `examples/agent_weights.pkl` | Serialized agent weights |
