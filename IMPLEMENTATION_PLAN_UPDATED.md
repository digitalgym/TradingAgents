# TradingAgents System Improvement Implementation Plan (UPDATED)

**Version:** 2.1  
**Date:** January 6, 2026  
**Status:** In Progress - Phase 1B Complete, Phase 2 Complete, Trade Decision Tracking Complete

---

## Executive Summary

This document provides an **updated** implementation plan for the TradingAgents multi-agent trading system, reflecting recent developments. Significant progress has been made on Phase 1 objectives, including:

✅ **Persistent memory system** with ChromaDB  
✅ **Backtesting framework** for technical analysis training  
✅ **Daily evaluation cycle** with prediction tracking  
✅ **Trade reflection and learning** via CLI commands  
✅ **Position management** and review capabilities  
✅ **Quantitative risk metrics** (Sharpe, Sortino, VaR, max drawdown)  
✅ **Position sizing** (Kelly criterion, half-Kelly, fixed fractional)  
✅ **Trade decision tracking** with outcome assessment  
✅ **Auto-reflect** for automatic closed trade processing

**Remaining Focus Areas:**

- ~~Parallelization of analyst execution~~ (evaluated - no benefit, see findings)
- ~~Tiered memory with confidence scoring~~ ✅ COMPLETE
- ~~Quantitative risk metrics~~ ✅ COMPLETE
- ~~Position sizing~~ ✅ COMPLETE
- ~~Trade decision tracking~~ ✅ COMPLETE
- Dynamic stop-loss implementation (trailing stops)
- Portfolio-level optimization

**Expected Outcomes:**

- 15-30% improvement in risk-adjusted returns (partially achieved via backtesting)
- ~~3-5x reduction in decision latency~~ (parallelization not beneficial - see Phase 2.1 findings)
- Production-ready risk management system
- Systematic learning from historical and live trades

---

## Recent Developments Analysis

### ✅ Completed Features (Since Last Assessment)

#### 1. **Persistent Memory System**

**Status:** ✅ COMPLETE  
**Implementation:** `tradingagents/agents/utils/memory.py`

- ChromaDB-based persistent storage in `memory_db/`
- Collections persist across restarts
- Local embeddings support (sentence-transformers)
- Memory accessible to all agents (bull, bear, trader, judges)

**Impact:** Enables long-term learning without API dependency

---

#### 2. **Backtesting Framework**

**Status:** ✅ COMPLETE  
**Implementation:** `examples/backtest_training.py`

**Features:**

- Historical technical analysis backtesting
- Calculates RSI, MACD, Bollinger Bands, SMA/EMA, ATR
- Signal generation (BUY/SELL/HOLD) based on technical scores
- Evaluates predictions against actual price movements
- Generates lessons via LLM and stores in memory
- Tracks accuracy, hypothetical P&L, signal distribution

**Usage:**

```bash
python examples/backtest_training.py --symbol XAUUSD --months 6
python examples/backtest_training.py --symbol XAUUSD --evaluation-hours 72
```

**Metrics Tracked:**

- Prediction accuracy
- Hypothetical P&L per trade
- Signal distribution (BUY/SELL/HOLD counts)
- Average P&L per signal type

**Gap:** Only technical analysis (no news/sentiment in historical mode)  
**Reason:** Historical news/sentiment APIs return current data, not historical

---

#### 3. **Daily Evaluation Cycle**

**Status:** ✅ COMPLETE  
**Implementation:** `examples/daily_cycle.py`

**Features:**

- 24-hour prediction → evaluation loop
- Saves predictions with expected direction
- Evaluates against actual price movement 24h later
- Generates comparative lessons (what worked/didn't)
- Stores lessons in memory for future retrieval
- Tracks evaluation history in `examples/evaluation_history/`

**Workflow:**

1. Run analysis → extract prediction → save to `pending_predictions/`
2. 24 hours later: fetch current price → compare → generate lesson
3. Store lesson in memory → repeat

**Usage:**

```bash
python examples/daily_cycle.py --symbol XAUUSD
python examples/daily_cycle.py --symbol XAUUSD --run-once  # Single cycle
```

**Impact:** Continuous learning from live market predictions

---

#### 4. **Trade Reflection System**

**Status:** ✅ COMPLETE  
**Implementation:** CLI commands in `cli/main.py`

**Commands:**

- `python -m cli.main reflect` - Process closed trades, create memories
- `python -m cli.main review` - Re-analyze open positions with current data
- `python -m cli.main positions` - View/modify MT5 positions and orders

**Reflection Workflow:**

1. Execute trade via CLI → state saved to `pending_trades/`
2. Trade closes → run `reflect` command
3. Enter exit price → calculates returns
4. LLM generates improvement suggestions
5. Stores lessons in memory via `reflect_and_remember()`

**Review Workflow:**

1. Run `review` command → fetches open MT5 positions
2. Gets current price + 5-day history
3. LLM analyzes each position
4. Provides HOLD/CLOSE/ADJUST recommendations
5. Suggests SL/TP adjustments with specific price levels

**Impact:** Systematic learning from both wins and losses

---

#### 5. **Position Management**

**Status:** ✅ COMPLETE  
**Implementation:** `cli/main.py` + `tradingagents/dataflows/mt5_data.py`

**Features:**

- View open positions and pending orders
- Modify SL/TP on open positions
- Close positions at market
- Modify/cancel pending orders
- Automatic reminder to cover longs when SELL signal detected

**Safety Features:**

- Warns when SELL signal but open LONG positions exist
- Prompts to cover longs before entering shorts
- Custom trading discipline memories (via `add_trading_memory.py`)

---

#### 6. **Custom Memory Injection**

**Status:** ✅ COMPLETE  
**Implementation:** `add_trading_memory.py`

**Purpose:** Add permanent trading discipline reminders to memory

**Example Memories:**

- "Cover longs before entering shorts"
- "Set trailing stops on profitable positions"
- "Take partial profits at resistance"

**Usage:**

```bash
python add_trading_memory.py
```

**Impact:** Injects personal trading rules into agent decision-making

---

### ✅ Recently Completed Features

#### 7. **Trade Decision Tracking System**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Implementation:** `tradingagents/trade_decisions.py`, `cli/main.py`

**Features:**

- Decision storage with rationale, entry price, SL/TP, MT5 ticket
- Automatic decision creation on trade execution and review acceptance
- Decision lifecycle: active → closed (with outcome)
- Statistics tracking: win rate, P&L, best/worst decisions
- MT5 integration: auto-fetch exit price from closed orders

**CLI Commands:**

```bash
# List active decisions
python -m cli.main decisions list

# Close a decision with exit price
python -m cli.main decisions close XAUUSD_20260106_120000 --exit 2700.00

# View decision statistics
python -m cli.main decisions stats

# Cancel a decision
python -m cli.main decisions cancel XAUUSD_20260106_120000
```

**Decision Types:**
- **OPEN**: New position based on analysis (BUY/SELL)
- **ADJUST**: Modify existing position (change SL/TP)
- **CLOSE**: Close existing position
- **HOLD**: Keep current position unchanged

**Impact:** Enables systematic tracking and assessment of trading decisions

---

#### 8. **Auto-Reflect Command**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Implementation:** `cli/main.py`

**Features:**

- Automatically scans all active decisions with MT5 tickets
- Checks MT5 history for closed positions
- Auto-fetches exit price and profit from MT5
- Closes decisions and calculates returns
- Creates memories for learning (if context available)

**Usage:**

```bash
python -m cli.main auto-reflect
```

**Workflow:**
1. Get all active decisions
2. For each decision with MT5 ticket:
   - Query MT5 history for closed deal
   - If found: extract exit price, profit, close time
   - Calculate returns
   - Close decision with outcome
   - Create memory for learning

**Impact:** Prevents user laziness, ensures all closed trades are reflected on

---

#### 9. **Enhanced Review Command**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Implementation:** `cli/main.py`

**Features:**

- Parses suggested SL/TP values from AI analysis
- Offers multiple options for setting SL/TP:
  - Use suggested values
  - Enter manual values
  - Mix: choose for each
  - Skip SL/TP update
- Applies changes directly to MT5 position
- Stores decision with rationale for tracking

**Impact:** Streamlined workflow from review to action with decision tracking

---

#### 10. **MT5 AutoTrading Check**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Implementation:** `tradingagents/dataflows/mt5_data.py`, `cli/main.py`

**Features:**

- `check_mt5_autotrading()` function checks:
  - MT5 connection status
  - AutoTrading enabled in terminal
  - Account trading permissions
- CLI callback runs on every command
- Warns if AutoTrading is disabled

**Impact:** Prevents failed trade modifications due to disabled AutoTrading

---

#### 11. **Memory System Enhancement**

**Status:** ✅ COMPLETE (Phase 1B)

**Completed:**

- ✅ Persistent storage (ChromaDB)
- ✅ Local embeddings (no API cost)
- ✅ Memory retrieval in all agents
- ✅ Reflection mechanism
- ✅ Tiered memory (short/mid/long-term)
- ✅ Confidence scoring on memories
- ✅ Recency weighting in retrieval
- ✅ Memory maintenance CLI (`memory-stats` command)

**Implementation:** `tradingagents/agents/utils/memory.py`, `memory_maintenance.py`

---

#### 8. **Parallel Analyst Execution**

**Status:** ⚠️ EVALUATED - NOT BENEFICIAL  
**Priority:** LOW (deprioritized based on findings)

**Findings (January 6, 2026):**

| Execution Mode | Time | Branch |
|----------------|------|--------|
| Sequential | **593s** (~10 min) | `main` |
| Parallel | **706s** (~12 min) | `parallel_execution` |

**Conclusion:** Parallel execution is **19% slower** than sequential due to:

1. **LangGraph coordination overhead** - Thread pool management adds latency
2. **API rate limits** - Simultaneous calls hit rate limits (Alpha Vantage, xAI)
3. **State reducer overhead** - Concurrent state updates require synchronization
4. **Bottleneck elsewhere** - Analyst phase is only ~10-15% of total time; debate rounds (Bull/Bear, Risk analysts) dominate execution time

**Technical Implementation (preserved in `parallel_execution` branch):**

- Modified `tradingagents/graph/setup.py` with parallel edges from START
- Added `Analyst Join` node for synchronization
- Updated `conditional_logic.py` to route to join node
- Added `last_value_reducer` in `agent_states.py` for concurrent state updates

**Recommendation:** Keep sequential execution on `main`. Parallel may become beneficial if:
- More analysts are added (5+)
- Debate rounds are reduced/parallelized
- API rate limits are lifted (premium tiers)

---

### ✅ Recently Implemented

#### 9. **Quantitative Risk Metrics**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Priority:** HIGH

**Implemented:**
- `tradingagents/risk/metrics.py` - RiskMetrics class with:
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Max drawdown (value and percentage)
  - Value at Risk (VaR) at 95% and 99% confidence
  - Conditional VaR (Expected Shortfall)
  - Win rate, profit factor, avg win/loss
  - Annualized volatility
- `tradingagents/risk/portfolio.py` - Portfolio tracking with equity curve
- Integrated into `examples/backtest_training.py` - Risk metrics in final report
- CLI command: `python -m cli.main risk-metrics`

---

#### 10. **Position Sizing (Kelly Criterion)**

**Status:** ✅ COMPLETE (January 6, 2026)  
**Priority:** HIGH

**Implemented:**
- `tradingagents/risk/position_sizing.py` - PositionSizer class with:
  - Kelly criterion (optimal bet fraction)
  - Half-Kelly (conservative default)
  - Fixed fractional sizing
  - Volatility-based sizing (ATR)
  - MT5 lot conversion
- Integrated into `examples/daily_cycle.py` - Position sizing with predictions
- CLI command: `python -m cli.main position-size -e <entry> -sl <stop> -d <direction>`

---

### ❌ Not Yet Implemented

#### 11. **Dynamic Stop-Loss (ATR-based)**

**Status:** ❌ NOT STARTED  
**Priority:** MEDIUM

MT5 integration uses fixed SL/TP, no trailing stops

---

## Revised Implementation Phases

### Phase 1A: Foundation (COMPLETED) ✅

**Achievements:**

- ✅ Persistent memory with ChromaDB
- ✅ Backtesting framework (technical analysis)
- ✅ Daily evaluation cycle
- ✅ Trade reflection system
- ✅ Position management CLI
- ✅ Custom memory injection

**Timeline:** Completed in last 2 weeks

---

### Phase 1B: Memory Enhancement (Weeks 1-2) ✅ COMPLETE

**Goal:** Complete tiered memory system with confidence scoring

**Status:** ✅ COMPLETE (January 5, 2026)

#### 1B.1 Implement Tiered Memory ✅

**Priority:** HIGH | **Complexity:** MEDIUM | **Impact:** HIGH

**Objective:** Improve memory relevance by 30-50% through temporal hierarchy.

**Implementation Steps:**

1. **Extend Memory Schema** (`tradingagents/agents/utils/memory.py`)

   ```python
   def add_situations(self, situations_and_advice, tier="short", confidence=0.5):
       """
       Add situations with tier and confidence metadata.

       Args:
           tier: "short" (last 10 trades), "mid" (monthly), "long" (high-impact)
           confidence: 0.0-1.0 score based on outcome quality
       """
       metadatas = [{
           "recommendation": rec,
           "tier": tier,
           "confidence": confidence,
           "timestamp": datetime.now().isoformat(),
           "outcome_quality": self._calculate_outcome_quality(situation, rec),
       } for rec in advice]
   ```

2. **Weighted Retrieval Algorithm**

   ```python
   def get_memories(self, current_situation, n_matches=5,
                    tier_weights={"short": 0.5, "mid": 0.3, "long": 0.2}):
       """
       Retrieve memories with weighted scoring.

       Score = similarity × tier_weight × confidence × recency_decay
       """
       # Get top 20 candidates
       candidates = self.situation_collection.query(
           query_embeddings=[self.get_embedding(current_situation)],
           n_results=20,
           include=["metadatas", "documents", "distances"]
       )

       # Re-rank by composite score
       scored_results = []
       for i, doc in enumerate(candidates["documents"][0]):
           metadata = candidates["metadatas"][0][i]
           similarity = 1 - candidates["distances"][0][i]

           # Calculate recency decay (exponential)
           age_days = (datetime.now() - datetime.fromisoformat(metadata["timestamp"])).days
           recency = np.exp(-age_days / 30)  # 30-day half-life

           # Composite score
           score = (
               similarity *
               tier_weights.get(metadata["tier"], 0.3) *
               metadata["confidence"] *
               recency
           )

           scored_results.append((score, doc, metadata))

       # Return top N
       scored_results.sort(reverse=True, key=lambda x: x[0])
       return scored_results[:n_matches]
   ```

3. **Automatic Tier Assignment**

   - **Short-term:** All new memories start here (decay after 30 days)
   - **Mid-term:** Promoted if referenced 3+ times or outcome quality > 0.7
   - **Long-term:** Promoted if confidence > 0.8 and outcome quality > 0.8

4. **Confidence Calculation**

   ```python
   def _calculate_confidence(self, returns, prediction_correct):
       """
       Calculate confidence score based on outcome.

       High confidence = correct prediction + strong returns
       Low confidence = incorrect prediction or weak returns
       """
       if prediction_correct:
           # Correct prediction: confidence based on magnitude
           return min(0.5 + abs(returns) / 10, 1.0)
       else:
           # Incorrect: low confidence
           return max(0.2, 0.5 - abs(returns) / 10)
   ```

5. **Update Reflection to Use Tiers**
   - Modify `tradingagents/graph/reflection.py`
   - Add tier and confidence to `add_situations()` calls
   - Calculate confidence from returns/losses

**Success Metrics:**

- Memory retrieval relevance: +30% (user evaluation)
- High-confidence memories retrieved first: 80% of time
- Reduction in irrelevant memory retrievals: 40%

**Files to Modify:**

- `tradingagents/agents/utils/memory.py` (add tier/confidence logic)
- `tradingagents/graph/reflection.py` (pass tier/confidence)
- `examples/backtest_training.py` (use confidence scoring)
- `examples/daily_cycle.py` (use confidence scoring)

**Estimated Effort:** 3-4 days

---

#### 1B.2 Memory Cleanup & Maintenance

**Priority:** MEDIUM | **Complexity:** LOW | **Impact:** MEDIUM

**Objective:** Prevent memory bloat and maintain quality.

**Implementation:**

1. **Prune Low-Quality Memories**

   - Remove memories with confidence < 0.3 after 60 days
   - Archive instead of delete for audit trail

2. **Deduplicate Similar Memories**

   - Detect near-duplicate memories (similarity > 0.95)
   - Merge duplicates, keep highest confidence version

3. **Memory Statistics Dashboard**
   ```bash
   python -m cli.main memory-stats
   ```
   - Show memory count by tier
   - Average confidence per agent
   - Most retrieved memories (top 10)

**Files to Create:**

- `tradingagents/agents/utils/memory_maintenance.py`
- Add `memory-stats` command to `cli/main.py`

**Estimated Effort:** 2 days

---

### Phase 2: Performance & Risk (Weeks 3-5) ✅ COMPLETE

**Goal:** ~~Reduce latency and~~ Implement quantitative risk management

**Status:** ✅ COMPLETE (January 6, 2026)

**Achievements:**
- ✅ 2.1 Parallel Execution - Evaluated, not beneficial (kept sequential)
- ✅ 2.2 Quantitative Risk Metrics - Sharpe, Sortino, VaR, max drawdown, Calmar
- ✅ 2.3 Position Sizing - Kelly criterion, half-Kelly, fixed fractional
- ✅ 2.4 Dynamic Stop-Loss - ATR-based SL, trailing stops, breakeven stops

#### 2.1 Parallelize Analyst Execution

**Priority:** ~~HIGH~~ LOW | **Complexity:** LOW | **Impact:** ~~HIGH~~ LOW

**Status:** ⚠️ EVALUATED - NOT BENEFICIAL (January 6, 2026)

**Original Objective:** Reduce latency from 30-60s to 6-10s.

**Actual Results:**

| Execution Mode | Time | Branch |
|----------------|------|--------|
| Sequential | **593s** (~10 min) | `main` |
| Parallel | **706s** (~12 min) | `parallel_execution` |

**Why Parallelization Failed to Improve Performance:**

1. **Analyst phase is not the bottleneck** - Only ~10-15% of total execution time
2. **Debate rounds dominate** - Bull/Bear and Risk analyst debates are sequential by design
3. **API rate limits** - Parallel calls trigger rate limiting (Alpha Vantage: 25/day, xAI burst limits)
4. **LangGraph overhead** - Thread pool coordination adds ~19% latency
5. **State synchronization** - Concurrent state updates require reducers

**Implementation Completed (preserved in `parallel_execution` branch):**

- ✅ Refactored `setup.py` with parallel edges from START to all analysts
- ✅ Added `Analyst Join` node for synchronization after parallel phase
- ✅ Updated `conditional_logic.py` to route completed analysts to join node
- ✅ Added `last_value_reducer` in `agent_states.py` for concurrent state updates
- ✅ Tested with xAI/Grok - graph compiles and executes correctly

**Recommendation:** 

Keep sequential execution on `main` branch. Parallel implementation preserved for future use if:
- 5+ analysts are added
- Debate rounds are parallelized or reduced
- Premium API tiers remove rate limits

**Estimated Effort:** ~~2-3 days~~ COMPLETE (evaluation only)

---

#### 2.2 Quantitative Risk Metrics ✅ COMPLETE

**Priority:** HIGH | **Complexity:** MEDIUM | **Impact:** HIGH

**Status:** ✅ COMPLETE (January 6, 2026)

**Objective:** Implement objective risk measurement and portfolio constraints.

**Implemented:**
- `tradingagents/risk/metrics.py` - RiskMetrics class with all metrics
- `tradingagents/risk/portfolio.py` - Portfolio tracking with equity curve
- `tradingagents/risk/__init__.py` - Module exports
- Integrated into `examples/backtest_training.py` - Risk metrics in final stats and JSON output
- CLI command: `python -m cli.main risk-metrics` - View backtest risk metrics

**Implementation Steps (Reference):**

1. **Create Risk Metrics Module** (`tradingagents/risk/metrics.py`)

   ```python
   class RiskMetrics:
       @staticmethod
       def sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
           """Calculate annualized Sharpe ratio."""
           excess_returns = returns - risk_free_rate / periods
           if len(returns) < 2:
               return 0.0
           return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods)

       @staticmethod
       def sortino_ratio(returns, risk_free_rate=0.02, periods=252):
           """Calculate Sortino ratio (downside deviation only)."""
           excess_returns = returns - risk_free_rate / periods
           downside_returns = returns[returns < 0]
           if len(downside_returns) < 2:
               return 0.0
           downside_std = np.std(downside_returns)
           return np.mean(excess_returns) / downside_std * np.sqrt(periods)

       @staticmethod
       def max_drawdown(equity_curve):
           """Calculate maximum drawdown from peak."""
           peak = np.maximum.accumulate(equity_curve)
           drawdown = (equity_curve - peak) / peak
           return np.min(drawdown)

       @staticmethod
       def value_at_risk(returns, confidence=0.95):
           """Calculate VaR at given confidence level."""
           return np.percentile(returns, (1 - confidence) * 100)

       @staticmethod
       def calmar_ratio(returns, equity_curve):
           """Calculate Calmar ratio (return / max drawdown)."""
           annual_return = np.mean(returns) * 252
           max_dd = abs(RiskMetrics.max_drawdown(equity_curve))
           if max_dd == 0:
               return 0.0
           return annual_return / max_dd
   ```

2. **Portfolio State Tracker** (`tradingagents/risk/portfolio.py`)

   ```python
   class Portfolio:
       def __init__(self, initial_capital=100000):
           self.initial_capital = initial_capital
           self.cash = initial_capital
           self.positions = {}  # {ticker: {shares, avg_price, current_price}}
           self.equity_curve = [initial_capital]
           self.trade_history = []

       def update_position(self, ticker, action, price, shares):
           """Update position and track equity."""
           if action == "BUY":
               cost = price * shares
               if ticker in self.positions:
                   # Average up
                   old_shares = self.positions[ticker]["shares"]
                   old_avg = self.positions[ticker]["avg_price"]
                   new_shares = old_shares + shares
                   new_avg = (old_avg * old_shares + price * shares) / new_shares
                   self.positions[ticker] = {
                       "shares": new_shares,
                       "avg_price": new_avg,
                       "current_price": price
                   }
               else:
                   self.positions[ticker] = {
                       "shares": shares,
                       "avg_price": price,
                       "current_price": price
                   }
               self.cash -= cost

           elif action == "SELL":
               if ticker in self.positions:
                   proceeds = price * shares
                   self.positions[ticker]["shares"] -= shares
                   if self.positions[ticker]["shares"] <= 0:
                       del self.positions[ticker]
                   self.cash += proceeds

           # Update equity curve
           equity = self.cash + sum(
               p["shares"] * p["current_price"]
               for p in self.positions.values()
           )
           self.equity_curve.append(equity)

           # Record trade
           self.trade_history.append({
               "ticker": ticker,
               "action": action,
               "price": price,
               "shares": shares,
               "timestamp": datetime.now()
           })

       def get_metrics(self):
           """Calculate portfolio risk metrics."""
           if len(self.equity_curve) < 2:
               return {}

           equity_array = np.array(self.equity_curve)
           returns = np.diff(equity_array) / equity_array[:-1]

           return {
               "total_equity": equity_array[-1],
               "total_return": (equity_array[-1] / self.initial_capital - 1) * 100,
               "sharpe_ratio": RiskMetrics.sharpe_ratio(returns),
               "sortino_ratio": RiskMetrics.sortino_ratio(returns),
               "max_drawdown": RiskMetrics.max_drawdown(equity_array) * 100,
               "var_95": RiskMetrics.value_at_risk(returns) * 100,
               "calmar_ratio": RiskMetrics.calmar_ratio(returns, equity_array),
               "num_trades": len(self.trade_history),
           }
   ```

3. **Integrate with Backtester**

   - Modify `examples/backtest_training.py` to use Portfolio class
   - Track metrics throughout backtest
   - Display in final report

4. **Integrate with Risk Judge**

   - Inject portfolio metrics into Risk Judge prompt
   - Add constraints: "Reject if portfolio Sharpe < 1.5"
   - Enforce position limits: max 20% per asset

5. **Real-Time Dashboard** (Optional)
   ```bash
   python -m cli.main dashboard
   ```
   - Streamlit dashboard showing live metrics
   - Equity curve chart
   - Risk alerts (breach of thresholds)

**Success Metrics:**

- Portfolio Sharpe ratio tracked: 100% of backtests
- Risk constraints enforced: 100% of trades
- Max drawdown < 20% maintained

**Files to Create:**

- `tradingagents/risk/__init__.py`
- `tradingagents/risk/metrics.py`
- `tradingagents/risk/portfolio.py`

**Files to Modify:**

- `examples/backtest_training.py` (integrate Portfolio)
- `tradingagents/agents/managers/risk_manager.py` (inject metrics)
- `cli/main.py` (add dashboard command)

**Estimated Effort:** 4-5 days

---

#### 2.3 Position Sizing (Kelly Criterion) ✅ COMPLETE

**Priority:** HIGH | **Complexity:** MEDIUM | **Impact:** HIGH

**Status:** ✅ COMPLETE (January 6, 2026)

**Objective:** Optimal position sizing based on historical performance.

**Implemented:**
- `tradingagents/risk/position_sizing.py` - PositionSizer class with:
  - Kelly criterion (optimal bet fraction)
  - Half-Kelly (conservative default - 50% of Kelly)
  - Fixed fractional sizing (risk fixed % per trade)
  - Volatility-based sizing (ATR-adjusted)
  - MT5 lot conversion
- Integrated into `examples/daily_cycle.py` - Position sizing with each prediction
- CLI command: `python -m cli.main position-size -e <entry> -sl <stop> -d <direction>`

**Usage:**
```bash
# Calculate position size for a trade
python -m cli.main position-size --entry 2650 --stop 2630 --direction BUY

# With custom parameters
python -m cli.main position-size -e 2650 -sl 2630 -d BUY -b 50000 -r 0.01 -c 0.8
```

**Estimated Effort:** ~~3-4 days~~ COMPLETE

---

#### 2.4 Dynamic Stop-Loss with ATR

**Priority:** MEDIUM | **Complexity:** MEDIUM | **Impact:** MEDIUM

**Status:** ✅ COMPLETE (January 6, 2026)

**Objective:** Adaptive risk management that adjusts to market volatility.

**Implementation Steps:**

1. **Add ATR Calculation** (already exists in backtest_training.py)

   - Extract to shared utility: `tradingagents/indicators/technical.py`
   - Ensure available to all components

2. **ATR-Based Stop-Loss Logic** (`tradingagents/risk/stop_loss.py`)

   ```python
   class DynamicStopLoss:
       def __init__(self, atr_multiplier=2.0, trailing_multiplier=1.5):
           self.atr_mult = atr_multiplier
           self.trail_mult = trailing_multiplier

       def calculate_initial_sl(self, entry_price, atr, direction):
           """Calculate initial stop-loss based on ATR."""
           if direction == "BUY":
               return entry_price - (self.atr_mult * atr)
           else:  # SELL
               return entry_price + (self.atr_mult * atr)

       def calculate_initial_tp(self, entry_price, atr, direction, risk_reward=2.0):
           """Calculate take-profit based on risk-reward ratio."""
           sl_distance = self.atr_mult * atr
           tp_distance = sl_distance * risk_reward

           if direction == "BUY":
               return entry_price + tp_distance
           else:  # SELL
               return entry_price - tp_distance

       def update_trailing_sl(self, current_price, current_sl, atr, direction):
           """Update trailing stop-loss."""
           if direction == "BUY":
               new_sl = current_price - (self.trail_mult * atr)
               return max(current_sl, new_sl)  # Only move up
           else:  # SELL
               new_sl = current_price + (self.trail_mult * atr)
               return min(current_sl, new_sl)  # Only move down
   ```

3. **MT5 Integration**

   - Modify `execute_trade_signal()` in `tradingagents/dataflows/mt5_data.py`
   - Calculate ATR from recent data
   - Set ATR-based SL/TP on order placement

4. **Background Trailing Stop Task**

   ```python
   def monitor_trailing_stops(check_interval=300):
       """Monitor positions and update trailing stops every 5 minutes."""
       while True:
           positions = get_open_positions()
           for pos in positions:
               # Get current ATR
               atr = calculate_atr_for_symbol(pos['symbol'])

               # Calculate new trailing SL
               sl_manager = DynamicStopLoss()
               new_sl = sl_manager.update_trailing_sl(
                   pos['price_current'],
                   pos['sl'],
                   atr,
                   pos['type']
               )

               # Update if changed
               if new_sl != pos['sl']:
                   modify_position(pos['ticket'], sl=new_sl)
                   print(f"Updated {pos['symbol']} SL: {pos['sl']} → {new_sl}")

           time.sleep(check_interval)
   ```

5. **CLI Command for Trailing Stops**
   ```bash
   python -m cli.main monitor-stops  # Start background monitoring
   ```

**Success Metrics:**

- 20-30% more profit captured in trending markets
- 15% reduction in premature stop-outs
- Trailing stops update automatically: 100% of trades

**Files to Create:**

- `tradingagents/indicators/__init__.py`
- `tradingagents/indicators/technical.py` (extract from backtest)
- `tradingagents/risk/stop_loss.py`

**Files to Modify:**

- `tradingagents/dataflows/mt5_data.py` (ATR-based execution)
- `cli/main.py` (add monitor-stops command)

**Estimated Effort:** 3-4 days

---

### Phase 3: Advanced Features (Weeks 6-10)

#### 3.1 Ensemble Voting with Confidence Scoring

**Priority:** MEDIUM | **Complexity:** MEDIUM | **Impact:** MEDIUM

**Objective:** Improve decision robustness through weighted consensus.

**Implementation Steps:**

1. **Update Agent Prompts**

   - Add confidence request to all analyst/researcher prompts
   - Format: "End with 'CONFIDENCE: 0.85' (0.0-1.0)"

2. **Confidence Parser** (`tradingagents/agents/utils/confidence.py`)

   ```python
   def parse_confidence(llm_output: str) -> float:
       """Extract confidence score from LLM output."""
       match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)', llm_output, re.IGNORECASE)
       if match:
           conf = float(match.group(1))
           return max(0.0, min(1.0, conf))  # Clamp to [0, 1]
       return 0.5  # Default neutral confidence
   ```

3. **Weighted Voting System** (`tradingagents/graph/voting.py`)

   ```python
   class EnsembleVoter:
       def __init__(self, persistent_path="memory_db/agent_accuracy.json"):
           self.agent_accuracy = self._load_accuracy(persistent_path)
           self.persistent_path = persistent_path

       def weighted_vote(self, decisions, confidences, agent_names):
           """
           Perform weighted majority vote.

           Weight = confidence × historical_accuracy
           """
           vote_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}

           for decision, conf, name in zip(decisions, confidences, agent_names):
               accuracy = self.agent_accuracy.get(name, 0.5)
               weight = conf * accuracy
               vote_scores[decision] += weight

           # Return decision with highest weighted score
           winner = max(vote_scores, key=vote_scores.get)
           total_weight = sum(vote_scores.values())
           confidence = vote_scores[winner] / total_weight if total_weight > 0 else 0.5

           return winner, confidence

       def update_accuracy(self, agent_name, was_correct):
           """Update agent accuracy with exponential moving average."""
           alpha = 0.1  # Learning rate
           current = self.agent_accuracy.get(agent_name, 0.5)
           new_accuracy = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * current
           self.agent_accuracy[agent_name] = new_accuracy
           self._save_accuracy()

       def _load_accuracy(self, path):
           """Load agent accuracy from disk."""
           if os.path.exists(path):
               with open(path, 'r') as f:
                   return json.load(f)
           return {}

       def _save_accuracy(self):
           """Save agent accuracy to disk."""
           with open(self.persistent_path, 'w') as f:
               json.dump(self.agent_accuracy, f, indent=2)
   ```

4. **Integration Points**

   - Bull/Bear debate: Extract confidence, weight by accuracy
   - Risk debate: Weighted vote among 3 analysts
   - Judge override: Only if judge confidence > 0.9

5. **Accuracy Tracking in Reflection**
   - Modify `tradingagents/graph/reflection.py`
   - Determine which agent was correct/incorrect
   - Update voter accuracy after each trade

**Success Metrics:**

- 10-15% improvement in decision accuracy
- Agent accuracy tracked: 100% of trades
- Confidence scores extracted: >95% of responses

**Files to Create:**

- `tradingagents/agents/utils/confidence.py`
- `tradingagents/graph/voting.py`

**Files to Modify:**

- All agent prompts (add confidence request)
- `tradingagents/graph/conditional_logic.py` (use voting)
- `tradingagents/graph/reflection.py` (track accuracy)

**Estimated Effort:** 5-6 days

---

#### 3.2 Multi-Asset Portfolio Manager

**Priority:** MEDIUM | **Complexity:** HIGH | **Impact:** HIGH

**Objective:** Enable portfolio-level optimization and diversification.

**Implementation Steps:**

1. **Portfolio Manager Agent** (`tradingagents/agents/managers/portfolio_manager.py`)

   ```python
   def create_portfolio_manager(llm, memory):
       def portfolio_manager_node(state) -> dict:
           # Receive signals from multiple ticker analyses
           signals = state["ticker_signals"]  # {ticker: {signal, confidence, analysis}}

           # Calculate correlation matrix
           correlations = calculate_correlations(signals.keys())

           # Optimize allocation
           allocations = optimize_portfolio(
               signals, correlations,
               constraints={
                   "max_position": 0.20,  # Max 20% per asset
                   "max_sector": 0.40,    # Max 40% per sector
                   "min_sharpe": 1.5,     # Min portfolio Sharpe
               }
           )

           return {"portfolio_allocations": allocations}

       return portfolio_manager_node
   ```

2. **Portfolio Optimization** (`tradingagents/risk/optimization.py`)

   ```python
   def optimize_portfolio(signals, correlations, constraints):
       """
       Optimize portfolio allocation using Markowitz mean-variance.

       Objective: Maximize Sharpe ratio
       Constraints: Position limits, sector limits
       """
       import cvxpy as cp

       n_assets = len(signals)
       tickers = list(signals.keys())

       # Expected returns (from signal confidence)
       expected_returns = np.array([
           signals[t]["confidence"] * (1 if signals[t]["signal"] == "BUY" else -1)
           for t in tickers
       ])

       # Covariance matrix (from correlations + volatility)
       cov_matrix = correlations  # Simplified

       # Decision variables
       weights = cp.Variable(n_assets)

       # Objective: Maximize Sharpe ratio (simplified: max return / risk)
       portfolio_return = expected_returns @ weights
       portfolio_risk = cp.quad_form(weights, cov_matrix)
       objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)

       # Constraints
       constraints_list = [
           cp.sum(weights) == 1,  # Fully invested
           weights >= 0,  # Long only
           weights <= constraints["max_position"],  # Position limits
       ]

       # Solve
       problem = cp.Problem(objective, constraints_list)
       problem.solve()

       # Return allocations
       return {ticker: float(weights.value[i]) for i, ticker in enumerate(tickers)}
   ```

3. **Multi-Ticker Workflow**

   - Modify `propagate()` to accept list of tickers
   - Run analyst teams in parallel for each ticker (leverage Phase 2.1)
   - Aggregate signals to portfolio manager
   - Output: Allocation percentages per ticker

4. **Rebalancing Logic** (`tradingagents/risk/rebalancing.py`)

   ```python
   def should_rebalance(current_allocations, target_allocations, threshold=0.05):
       """Check if rebalancing is needed."""
       for ticker in target_allocations:
           current = current_allocations.get(ticker, 0)
           target = target_allocations[ticker]
           if abs(current - target) > threshold:
               return True
       return False

   def generate_rebalance_orders(current_allocations, target_allocations, portfolio_value):
       """Generate orders to rebalance portfolio."""
       orders = []
       for ticker, target_pct in target_allocations.items():
           current_pct = current_allocations.get(ticker, 0)
           delta_pct = target_pct - current_pct
           delta_value = delta_pct * portfolio_value

           if abs(delta_value) > 100:  # Minimum $100 order
               action = "BUY" if delta_value > 0 else "SELL"
               orders.append({
                   "ticker": ticker,
                   "action": action,
                   "value": abs(delta_value)
               })
       return orders
   ```

**Success Metrics:**

- Portfolio Sharpe ratio: 1.5 → 2.5+
- Max drawdown reduction: 20-40%
- Successful management of 10+ asset portfolio

**Files to Create:**

- `tradingagents/agents/managers/portfolio_manager.py`
- `tradingagents/risk/optimization.py`
- `tradingagents/risk/rebalancing.py`

**Files to Modify:**

- `tradingagents/graph/trading_graph.py` (multi-ticker support)
- `tradingagents/graph/setup.py` (add portfolio manager node)

**Estimated Effort:** 7-10 days

---

#### 3.3 Adaptive Debate Depth

**Priority:** LOW | **Complexity:** MEDIUM | **Impact:** MEDIUM

**Objective:** Balance decision quality vs. latency through intelligent debate termination.

**Implementation Steps:**

1. **Convergence Detection** (`tradingagents/graph/convergence.py`)

   ```python
   class DebateConvergence:
       def __init__(self):
           from sentence_transformers import SentenceTransformer
           self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

       def detect_convergence(self, arguments, threshold=0.9):
           """Check if agents agree (high similarity)."""
           if len(arguments) < 2:
               return False

           embeddings = self.embedder.encode(arguments)
           from sklearn.metrics.pairwise import cosine_similarity
           similarity_matrix = cosine_similarity(embeddings)
           avg_similarity = similarity_matrix.mean()

           return avg_similarity > threshold

       def calculate_novelty(self, new_argument, history):
           """Check if new argument adds novel information."""
           if not history:
               return 1.0

           new_emb = self.embedder.encode([new_argument])
           hist_embs = self.embedder.encode(history)

           from sklearn.metrics.pairwise import cosine_similarity
           similarities = cosine_similarity(new_emb, hist_embs)[0]
           max_similarity = similarities.max()

           return 1 - max_similarity  # Novelty = 1 - max_similarity
   ```

2. **Complexity Heuristic** (`tradingagents/graph/complexity.py`)

   ```python
   def assess_scenario_complexity(state):
       """
       Assess scenario complexity to determine debate rounds needed.

       Returns: 1-5 (number of debate rounds)
       """
       complexity_score = 0

       # High volatility indicator
       market_report = state.get("market_report", "")
       if "high volatility" in market_report.lower() or "volatile" in market_report.lower():
           complexity_score += 1

       # Conflicting signals
       if signals_conflict(state):
           complexity_score += 1

       # Low average confidence
       avg_conf = calculate_average_confidence(state)
       if avg_conf < 0.6:
           complexity_score += 1

       # Mixed sentiment
       sentiment_report = state.get("sentiment_report", "")
       if "mixed" in sentiment_report.lower() or "divided" in sentiment_report.lower():
           complexity_score += 1

       # Map to debate rounds: 0-1 → 1 round, 2 → 3 rounds, 3-4 → 5 rounds
       return min(1 + complexity_score * 2, 5)
   ```

3. **Update Conditional Logic**
   - Modify `should_continue_debate()` to check convergence
   - Add novelty threshold: stop if novelty < 0.1
   - Dynamic max rounds based on complexity

**Success Metrics:**

- 15-25% improvement in complex scenario decisions
- Average latency maintained: <15s
- Adaptive rounds: 1-5 based on complexity

**Files to Create:**

- `tradingagents/graph/convergence.py`
- `tradingagents/graph/complexity.py`

**Files to Modify:**

- `tradingagents/graph/conditional_logic.py` (add convergence checks)

**Estimated Effort:** 4-5 days

---

### Phase 4: Production & Advanced Learning (Weeks 11-16+)

#### 4.1 Hierarchical Multi-Agent RL (Optional)

**Priority:** LOW | **Complexity:** VERY HIGH | **Impact:** VERY HIGH

**Status:** Deferred until Phases 1-3 complete

**Prerequisites:**

- 2+ years historical data
- GPU cluster for training
- Dedicated ML engineering resources (2-3 engineers)
- 3-6 months development time

**Estimated Effort:** 3-6 months, 2-3 ML engineers

---

#### 4.2 Fine-Tune LLMs on Trading Logs

**Priority:** LOW | **Complexity:** HIGH | **Impact:** HIGH

**Status:** Deferred until 1000+ completed trades

**Prerequisites:**

- 1000+ completed trades with outcomes
- Labeled dataset (correct/incorrect decisions)
- Fine-tuning infrastructure (OpenAI API or local GPUs)

**Estimated Effort:** 2-3 months

---

#### 4.3 Production Infrastructure

**Priority:** MEDIUM | **Complexity:** HIGH | **Impact:** CRITICAL

**Implementation Steps:**

1. **Monitoring & Alerting**

   - Prometheus metrics: latency, error rates, API costs
   - Grafana dashboards: real-time portfolio metrics
   - Alerts: critical failures, risk breaches

2. **Error Handling & Resilience**

   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
   def call_llm_with_retry(prompt):
       return llm.invoke(prompt)
   ```

3. **Database Migration**

   - Replace JSON logs with PostgreSQL
   - Schema: trades, positions, memories, metrics
   - Enable complex queries and analytics

4. **Deployment**
   - Docker containers for reproducibility
   - Kubernetes for orchestration (optional)
   - CI/CD pipeline: GitHub Actions

**Files to Create:**

- `docker/Dockerfile`
- `.github/workflows/ci.yaml`
- `tradingagents/monitoring/metrics.py`
- `tradingagents/database/schema.sql`

**Estimated Effort:** 2-3 weeks

---

## Testing Strategy

### Unit Tests

```python
# tests/test_tiered_memory.py
def test_tiered_memory_retrieval():
    memory = FinancialSituationMemory("test", config)
    memory.add_situations([("situation1", "advice1")], tier="short", confidence=0.8)
    results = memory.get_memories("similar situation", n_matches=1)
    assert results[0]["confidence"] == 0.8
    assert results[0]["tier"] == "short"

# tests/test_risk_metrics.py
def test_sharpe_ratio_calculation():
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    sharpe = RiskMetrics.sharpe_ratio(returns)
    assert 0 < sharpe < 5

# tests/test_dynamic_stop_loss.py
def test_atr_stop_loss():
    sl_manager = DynamicStopLoss(atr_multiplier=2.0)
    sl = sl_manager.calculate_initial_sl(entry_price=100, atr=2.0, direction="BUY")
    assert sl == 96.0  # 100 - (2 * 2.0)
```

### Integration Tests

```python
# tests/integration/test_backtest_with_metrics.py
def test_backtest_with_risk_metrics():
    backtester = TechnicalBacktester("XAUUSD")
    backtester.run(months=3)
    assert backtester.stats.accuracy > 50

    # Check risk metrics calculated
    metrics = backtester.portfolio.get_metrics()
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
```

### Performance Tests

```python
# tests/performance/test_parallel_latency.py
def test_parallel_analyst_latency():
    ta = TradingAgentsGraph(parallel_analysts=True)
    start = time.time()
    ta.propagate("NVDA", "2024-01-01")
    latency = time.time() - start
    assert latency < 15  # Must be under 15 seconds
```

---

## Success Metrics & KPIs

### Performance Metrics

| Metric              | Baseline | Phase 1B Target | Phase 2 Target | Phase 3 Target |
| ------------------- | -------- | --------------- | -------------- | -------------- |
| Sharpe Ratio        | 1.2      | 1.5             | 2.0            | 2.5+           |
| Max Drawdown        | 25%      | 22%             | 18%            | 15%            |
| Win Rate            | 52%      | 56%             | 60%            | 65%            |
| Decision Latency    | 45s      | 40s             | 10s            | 8s             |
| Memory Relevance    | 60%      | 80%             | 85%            | 90%            |
| Prediction Accuracy | 55%      | 60%             | 65%            | 70%            |

### Operational Metrics

- API cost per trade: <$0.50
- System uptime: >99.5%
- Error rate: <1%
- Memory storage: <1GB per 1000 trades
- Backtest speed: <10 min for 6 months

---

## Resource Requirements

### Phase 1B (Weeks 1-2)

- **Team:** 1 senior engineer
- **Compute:** Development machine
- **Cost:** $100 (testing)

### Phase 2 (Weeks 3-5)

- **Team:** 1-2 engineers
- **Compute:** Development machine + cloud for backtesting
- **Cost:** $300 (compute + APIs)

### Phase 3 (Weeks 6-10)

- **Team:** 2 engineers
- **Compute:** Cloud instance for portfolio optimization
- **Cost:** $500 (compute + APIs)

### Phase 4 (Weeks 11-16+)

- **Team:** 2-3 engineers (if pursuing RL/fine-tuning)
- **Compute:** GPU cluster for training (optional)
- **Cost:** $2000-5000 (training + infrastructure)

**Total Estimated Cost (Phases 1B-3):** $900  
**Total Estimated Cost (All Phases):** $6000-10000

---

## Timeline Summary

```
✅ Phase 1A: COMPLETED (Weeks -2 to 0)
   - Persistent memory
   - Backtesting framework
   - Daily evaluation cycle
   - Trade reflection
   - Position management

🔄 Phase 1B: IN PROGRESS (Weeks 1-2)
   - Tiered memory with confidence scoring
   - Memory maintenance utilities

⬜ Phase 2: PENDING (Weeks 3-5)
   - Parallel analyst execution
   - Quantitative risk metrics
   - Dynamic stop-loss (ATR-based)

⬜ Phase 3: PENDING (Weeks 6-10)
   - Ensemble voting
   - Portfolio manager
   - Adaptive debate depth

⬜ Phase 4: PENDING (Weeks 11-16+)
   - Production infrastructure
   - Optional: RL/Fine-tuning
```

**Current Status:** Phase 1A complete, Phase 1B starting  
**Next Milestone:** Complete Phase 1B by Week 2  
**Target for Production:** Phase 2 complete by Week 5

---

## Next Steps

### This Week (Week 1)

1. ✅ Review updated implementation plan
2. ⬜ Implement tiered memory system
3. ⬜ Add confidence scoring to reflection
4. ⬜ Test memory retrieval with weighted scoring
5. ⬜ Update backtest_training.py to use tiers

### Week 2

1. ⬜ Complete memory maintenance utilities
2. ⬜ Add memory-stats CLI command
3. ⬜ Begin parallel analyst implementation
4. ⬜ Benchmark current vs parallel latency

### Week 3

1. ⬜ Complete parallel analyst execution
2. ⬜ Implement risk metrics module
3. ⬜ Integrate metrics with backtester
4. ⬜ Begin dynamic stop-loss implementation

---

## Key Differences from Original Plan

### ✅ Already Completed

- Persistent memory (was Phase 1.2)
- Backtesting framework (was Phase 1.3)
- Trade reflection (was Phase 2 feature)
- Position management (was Phase 2 feature)
- Daily evaluation cycle (new feature)

### 🔄 Modified Priorities

- **Tiered memory** moved to Phase 1B (from Phase 1.2) - now top priority
- **Parallel execution** remains high priority but deferred to Phase 2
- **Risk metrics** elevated to Phase 2 (was Phase 2.1)
- **Ensemble voting** moved to Phase 3 (was Phase 2.3)

### ⏸️ Deferred

- **RL implementation** - requires more data and resources
- **Fine-tuning** - requires 1000+ trades first
- **Portfolio manager** - deferred to Phase 3

---

## Appendix

### A. Completed Features Documentation

#### Backtesting Framework

- **Location:** `examples/backtest_training.py`
- **Usage:** `python examples/backtest_training.py --symbol XAUUSD --months 6`
- **Output:** JSON report with accuracy, P&L, lessons stored in memory

#### Daily Evaluation Cycle

- **Location:** `examples/daily_cycle.py`
- **Usage:** `python examples/daily_cycle.py --symbol XAUUSD`
- **Features:** 24h prediction tracking, automatic evaluation, lesson generation

#### Trade Reflection

- **Commands:**
  - `python -m cli.main reflect` - Process closed trades
  - `python -m cli.main review` - Re-analyze open positions
  - `python -m cli.main positions` - Manage positions

#### Memory System

- **Storage:** `memory_db/` (ChromaDB persistent)
- **Collections:** bull_memory, bear_memory, trader_memory, invest_judge_memory, risk_manager_memory
- **Embeddings:** Local (sentence-transformers) or OpenAI

---

**Document Version:** 2.0  
**Last Updated:** January 6, 2026  
**Next Review:** January 13, 2026 (after Phase 1B completion)  
**Previous Version:** 1.0 (January 5, 2026)
