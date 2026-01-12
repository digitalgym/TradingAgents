# Continuous Learning & Regime-Aware Trading System Implementation Plan

**Version:** 1.0  
**Date:** January 11, 2026  
**Status:** Planning Phase  
**Target Completion:** 6-8 weeks

---

## Executive Summary

This plan transforms the current TradingAgents system from a **memory-storing** architecture to a **continuously learning** system with:

1. **RAG-based decision making** - Query similar past trades before recommendations
2. **Reward signal integration** - Learn from realized RR, Sharpe contribution, drawdown impact
3. **Regime awareness** - Detect and adapt to market conditions (trending/range/volatile)
4. **Online learning** - Update decision weights every 10-30 trades
5. **Automated pattern analysis** - Cluster wins/losses, generate improvement recommendations
6. **Hard risk guardrails** - Circuit breakers during learning phase
7. **Enriched trade memory** - Full context including setup types, confluence, exit reasons

**Expected Impact:**

- 25-40% improvement in risk-adjusted returns through regime-specific strategies
- 30-50% reduction in drawdowns via online learning from mistakes
- 15-25% increase in win rate through RAG-based decision filtering

---

## Current State Assessment

### ✅ Existing Infrastructure (Leverage These)

- ChromaDB vector memory with tiered storage (short/mid/long-term)
- Trade decision tracking (`tradingagents/trade_decisions.py`)
- Risk metrics calculation (Sharpe, Sortino, VaR, Calmar)
- Position sizing (Kelly criterion, ATR-based)
- Backtesting framework with technical indicators
- MT5 integration for live trading

### ❌ Missing Components (Build These)

- Reward signal calculation and storage
- Regime detection and classification
- RAG integration in agent decision pipeline
- Online learning/weight update mechanism
- Automated pattern analysis
- Risk guardrails layer
- Enhanced trade memory schema

---

## Implementation Phases

### **Phase 1: Foundation (Week 1-2)** - Reward Signals & Enhanced Memory

**Goal:** Enable outcome-based learning by calculating and storing reward signals

#### 1.1 Reward Signal Module

**File:** `tradingagents/learning/reward.py`

**Components:**

```python
class RewardCalculator:
    """Calculate multi-factor reward signals for trade outcomes"""

    @staticmethod
    def calculate_reward(
        realized_rr: float,
        sharpe_contribution: float,
        drawdown_impact: float,
        win: bool,
        weights: dict = {"rr": 0.4, "sharpe": 0.3, "drawdown": 0.3}
    ) -> float:
        """
        Composite reward signal:
        reward = (realized_RR × 0.4) + (Sharpe_contribution × 0.3) - (drawdown_impact × 0.3)

        Args:
            realized_rr: Actual risk-reward achieved (e.g., 2.5 for 2.5R win)
            sharpe_contribution: Impact on portfolio Sharpe ratio
            drawdown_impact: Contribution to drawdown (0 if none, negative if caused DD)
            win: True if profitable trade
            weights: Factor weights (must sum to 1.0)

        Returns:
            reward: Float in range [-5.0, 5.0] (normalized)
        """

    @staticmethod
    def calculate_sharpe_contribution(
        trade_return: float,
        portfolio_returns: list,
        position_size_pct: float
    ) -> float:
        """Calculate how this trade affected portfolio Sharpe ratio"""

    @staticmethod
    def calculate_drawdown_impact(
        trade_pnl: float,
        equity_curve: list,
        peak_equity: float
    ) -> float:
        """Calculate if trade contributed to drawdown"""
```

**Integration Points:**

- Modify `tradingagents/trade_decisions.py::close_decision()` to calculate reward
- Store reward in decision outcome
- Add portfolio state tracking for Sharpe/DD calculations

**Success Metrics:**

- Reward calculated for 100% of closed trades
- Reward correlates with trade quality (manual validation on 20 trades)

**Estimated Effort:** 2-3 days

---

#### 1.2 Enhanced Trade Memory Schema

**File:** `tradingagents/trade_decisions.py` (modify existing)

**New Fields to Add:**

```python
decision = {
    # ... existing fields ...

    # NEW: Setup classification
    "setup_type": None,  # "breaker-block", "FVG", "liquidity-sweep", "trend-continuation"
    "higher_tf_bias": None,  # "bullish", "bearish", "neutral" from H4/D1
    "confluence_score": None,  # 0-10 based on number of confirming factors
    "confluence_factors": [],  # ["support-zone", "fib-618", "ema-bounce", "rsi-divergence"]

    # NEW: Market context
    "volatility_regime": None,  # "low", "normal", "high", "extreme"
    "market_regime": None,  # "trending-up", "trending-down", "ranging", "expansion"
    "session": None,  # "asian", "london", "ny", "overlap"

    # NEW: Exit analysis
    "exit_reason": None,  # "tp-hit", "sl-hit", "manual", "trailing-stop", "time-exit"
    "rr_planned": None,  # Planned risk-reward ratio
    "rr_realized": None,  # Actual risk-reward achieved

    # NEW: Learning signals
    "reward_signal": None,  # Calculated reward for RL
    "pattern_tags": [],  # Auto-generated tags for pattern clustering
}
```

**Migration Script:**

```python
# scripts/migrate_trade_decisions.py
def backfill_trade_decisions():
    """Add new fields to existing trade decisions"""
    # Load all existing decisions
    # Add None values for new fields
    # Attempt to infer setup_type from rationale using LLM
    # Save updated decisions
```

**Success Metrics:**

- All new trades capture full context
- 80%+ of existing trades backfilled with inferred data

**Estimated Effort:** 2 days

---

#### 1.3 Portfolio State Tracker

**File:** `tradingagents/learning/portfolio_state.py`

**Purpose:** Track equity curve and metrics for reward calculation

```python
class PortfolioStateTracker:
    """Maintain running portfolio state for reward calculations"""

    def __init__(self, initial_capital: float = 100000):
        self.equity_curve = [initial_capital]
        self.returns = []
        self.peak_equity = initial_capital
        self.current_equity = initial_capital

    def update(self, trade_pnl: float):
        """Update portfolio state after trade"""
        self.current_equity += trade_pnl
        self.equity_curve.append(self.current_equity)

        if len(self.equity_curve) > 1:
            ret = trade_pnl / self.equity_curve[-2]
            self.returns.append(ret)

        self.peak_equity = max(self.peak_equity, self.current_equity)

    def get_sharpe_ratio(self) -> float:
        """Calculate current portfolio Sharpe"""

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""

    def save_state(self, path: str):
        """Persist portfolio state"""

    @classmethod
    def load_state(cls, path: str):
        """Load portfolio state from disk"""
```

**Integration:**

- Initialize on system startup
- Update after each trade closes
- Persist to `examples/portfolio_state.pkl`

**Estimated Effort:** 1 day

---

### **Phase 2: Regime Detection (Week 2-3)** - Market State Classification

**Goal:** Detect and classify market regimes for context-aware decision making

#### 2.1 Regime Detection Module

**File:** `tradingagents/indicators/regime.py`

**Components:**

```python
class RegimeDetector:
    """Detect market regime using multiple indicators"""

    def __init__(self):
        self.adx_threshold_trending = 25
        self.atr_percentile_high = 75
        self.bb_width_percentile_expansion = 70

    def detect_trend_regime(self, prices: np.ndarray, period: int = 14) -> str:
        """
        Detect trend regime using ADX

        Returns: "trending-up", "trending-down", "ranging"
        """
        adx = self._calculate_adx(prices, period)

        if adx > self.adx_threshold_trending:
            # Strong trend - check direction
            if prices[-1] > prices[-period]:
                return "trending-up"
            else:
                return "trending-down"
        else:
            return "ranging"

    def detect_volatility_regime(self, prices: np.ndarray, lookback: int = 100) -> str:
        """
        Detect volatility regime using ATR percentile

        Returns: "low", "normal", "high", "extreme"
        """
        atr = self._calculate_atr(prices, period=14)
        atr_history = [self._calculate_atr(prices[i-14:i+1], 14)
                       for i in range(14, len(prices))]

        percentile = np.percentile(atr_history, [25, 50, 75, 90])

        if atr < percentile[0]:
            return "low"
        elif atr < percentile[1]:
            return "normal"
        elif atr < percentile[2]:
            return "high"
        else:
            return "extreme"

    def detect_expansion_regime(self, prices: np.ndarray) -> str:
        """
        Detect expansion/contraction using Bollinger Band width

        Returns: "expansion", "contraction", "neutral"
        """

    def get_full_regime(self, prices: np.ndarray) -> dict:
        """
        Get complete regime classification

        Returns:
            {
                "market_regime": "trending-up",
                "volatility_regime": "high",
                "expansion": "expansion",
                "timestamp": "2026-01-11T13:00:00"
            }
        """
```

**Integration Points:**

- Add to `tradingagents/dataflows/mt5_data.py::get_market_data()`
- Include regime in all analyst reports
- Tag memories with regime metadata
- Store regime in trade decisions

**Success Metrics:**

- Regime detected for 100% of analyses
- Regime changes logged and trackable
- Regime-specific win rates measurable

**Estimated Effort:** 3-4 days

---

#### 2.2 Regime-Aware Memory Retrieval

**File:** `tradingagents/agents/utils/memory.py` (modify existing)

**Enhancement:**

```python
class FinancialSituationMemory:
    # ... existing code ...

    def get_memories(
        self,
        current_situation: str,
        n_matches: int = 5,
        regime_filter: dict = None,  # NEW
        tier_weights: dict = DEFAULT_TIER_WEIGHTS
    ):
        """
        Retrieve memories with optional regime filtering

        Args:
            regime_filter: {"market_regime": "trending-up", "volatility_regime": "high"}
                          Only return memories from similar regimes
        """
        # Get top 20 candidates
        candidates = self.situation_collection.query(...)

        # Filter by regime if specified
        if regime_filter:
            filtered = []
            for i, metadata in enumerate(candidates["metadatas"][0]):
                regime_match = all(
                    metadata.get(k) == v
                    for k, v in regime_filter.items()
                    if k in metadata
                )
                if regime_match or not metadata.get("market_regime"):
                    # Include if regime matches OR no regime data (old memories)
                    filtered.append(i)

            # Re-rank filtered candidates
            candidates = self._filter_candidates(candidates, filtered)

        # ... existing scoring logic ...
```

**Success Metrics:**

- Regime filtering reduces irrelevant memories by 40%+
- Win rate improves when using regime-filtered memories

**Estimated Effort:** 1-2 days

---

### **Phase 3: RAG Decision Integration (Week 3-4)** - Query Past Trades

**Goal:** Integrate memory retrieval into agent decision pipeline

#### 3.1 Trade Similarity Search

**File:** `tradingagents/learning/trade_similarity.py`

**Components:**

```python
class TradeSimilaritySearch:
    """Find similar historical trades for decision support"""

    def __init__(self, memory: FinancialSituationMemory):
        self.memory = memory

    def find_similar_trades(
        self,
        current_setup: dict,
        n_results: int = 5,
        min_confidence: float = 0.6
    ) -> list:
        """
        Find similar past trades based on setup characteristics

        Args:
            current_setup: {
                "symbol": "XAUUSD",
                "direction": "BUY",
                "setup_type": "breaker-block",
                "market_regime": "trending-up",
                "volatility_regime": "normal",
                "confluence_score": 7,
                "higher_tf_bias": "bullish"
            }

        Returns:
            List of similar trades with outcomes and statistics
        """
        # Build search query
        query = self._build_search_query(current_setup)

        # Search with regime filter
        regime_filter = {
            "market_regime": current_setup.get("market_regime"),
            "volatility_regime": current_setup.get("volatility_regime")
        }

        similar = self.memory.get_memories(
            query,
            n_matches=n_results * 2,  # Get extras for filtering
            regime_filter=regime_filter
        )

        # Filter by setup similarity
        filtered = self._filter_by_setup_similarity(similar, current_setup)

        # Calculate statistics
        stats = self._calculate_similarity_stats(filtered[:n_results])

        return {
            "similar_trades": filtered[:n_results],
            "statistics": stats,
            "recommendation": self._generate_recommendation(stats)
        }

    def _calculate_similarity_stats(self, trades: list) -> dict:
        """
        Calculate aggregate statistics from similar trades

        Returns:
            {
                "win_rate": 0.65,
                "avg_rr": 2.3,
                "avg_reward": 1.8,
                "sample_size": 5,
                "confidence_adjustment": +0.15  # Boost confidence if strong history
            }
        """
```

**Integration Points:**

- Call before Bull/Bear analysts make recommendations
- Include similarity stats in analyst prompts
- Adjust confidence scores based on historical performance

**Estimated Effort:** 3 days

---

#### 3.2 Agent Prompt Enhancement

**Files:**

- `tradingagents/agents/researchers/bull_researcher.py`
- `tradingagents/agents/researchers/bear_researcher.py`
- `tradingagents/agents/trader/trader.py`

**Enhancement:**

```python
def create_bull_researcher(llm, memory):
    def bull_researcher_node(state) -> dict:
        # ... existing code ...

        # NEW: Query similar trades
        from tradingagents.learning.trade_similarity import TradeSimilaritySearch

        current_setup = {
            "symbol": state["ticker"],
            "direction": "BUY",
            "market_regime": state.get("market_regime"),
            "volatility_regime": state.get("volatility_regime"),
            # ... extract from analysis ...
        }

        similarity_search = TradeSimilaritySearch(memory)
        similar_trades = similarity_search.find_similar_trades(current_setup)

        # Add to prompt
        prompt = f"""
        ... existing prompt ...

        HISTORICAL CONTEXT:
        Found {similar_trades['statistics']['sample_size']} similar BUY setups:
        - Win Rate: {similar_trades['statistics']['win_rate']:.1%}
        - Avg Risk-Reward: {similar_trades['statistics']['avg_rr']:.2f}
        - Avg Reward Signal: {similar_trades['statistics']['avg_reward']:.2f}

        Top Similar Trades:
        {self._format_similar_trades(similar_trades['similar_trades'][:3])}

        RECOMMENDATION FROM HISTORY:
        {similar_trades['recommendation']}

        Consider this historical performance when making your bullish case.
        If historical win rate is low (<50%), you MUST provide strong justification
        for why THIS setup is different.

        Adjust your confidence based on historical performance:
        - Strong history (>65% win rate): Increase confidence by +0.1 to +0.2
        - Weak history (<45% win rate): Decrease confidence by -0.1 to -0.2
        """

        # ... rest of existing code ...
```

**Success Metrics:**

- Agents query similar trades in 100% of analyses
- Confidence adjustments correlate with actual outcomes
- Win rate improves by 10-15% through historical filtering

**Estimated Effort:** 2-3 days

---

### **Phase 4: Online Learning (Week 4-5)** - Weight Updates & Pattern Analysis

**Goal:** Implement continuous learning from trade outcomes

#### 4.1 Simple Policy Gradient (Rule-Based First)

**File:** `tradingagents/learning/online_rl.py`

**Components:**

```python
class OnlineLearner:
    """Simple online learning using policy gradients on decision confidence"""

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.agent_weights = {
            "bull_researcher": 1.0,
            "bear_researcher": 1.0,
            "market_analyst": 1.0,
            "news_analyst": 1.0,
            "risk_manager": 1.0
        }
        self.update_history = []

    def update_weights(self, decision_outcome: dict):
        """
        Update agent weights based on decision outcome

        Args:
            decision_outcome: {
                "decision_id": "XAUUSD_20260111_130000",
                "agents_involved": ["bull_researcher", "market_analyst"],
                "agent_confidences": {"bull_researcher": 0.8, "market_analyst": 0.7},
                "reward_signal": 2.3,  # Positive for good trade
                "was_correct": True
            }
        """
        reward = decision_outcome["reward_signal"]

        # Update weights for involved agents
        for agent, confidence in decision_outcome["agent_confidences"].items():
            if agent in self.agent_weights:
                # Policy gradient: weight += lr * reward * confidence
                gradient = self.lr * reward * confidence
                self.agent_weights[agent] += gradient

                # Clip weights to reasonable range
                self.agent_weights[agent] = np.clip(
                    self.agent_weights[agent],
                    0.5,  # Min weight
                    2.0   # Max weight
                )

        self.update_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision_id": decision_outcome["decision_id"],
            "reward": reward,
            "weights_after": self.agent_weights.copy()
        })

        self.save_weights()

    def get_weighted_confidence(self, agent: str, base_confidence: float) -> float:
        """Apply learned weight to agent confidence"""
        weight = self.agent_weights.get(agent, 1.0)
        return base_confidence * weight

    def save_weights(self):
        """Persist weights to disk"""

    @classmethod
    def load_weights(cls):
        """Load weights from disk"""
```

**Integration:**

- Call `update_weights()` in `cli/main.py auto-reflect` after closing trades
- Apply weights in agent nodes before returning confidence
- Track weight evolution over time

**Success Metrics:**

- Weights update every 10-30 trades
- Agent weights correlate with actual performance
- System learns to trust better-performing agents

**Estimated Effort:** 3-4 days

---

#### 4.2 Automated Pattern Analysis

**File:** `tradingagents/learning/pattern_analyzer.py`

**Components:**

```python
class PatternAnalyzer:
    """Cluster and analyze trade patterns for improvement insights"""

    def analyze_recent_trades(self, n_trades: int = 30) -> dict:
        """
        Analyze last N trades for patterns

        Returns:
            {
                "win_patterns": [...],
                "loss_patterns": [...],
                "recommendations": [...],
                "high_confidence_patterns": [...]  # For long-term memory
            }
        """
        # Load recent closed trades
        trades = self._load_recent_trades(n_trades)

        # Separate wins and losses
        wins = [t for t in trades if t["was_correct"]]
        losses = [t for t in trades if not t["was_correct"]]

        # Cluster by features
        win_clusters = self._cluster_trades(wins, features=[
            "setup_type", "market_regime", "volatility_regime",
            "confluence_score", "session"
        ])

        loss_clusters = self._cluster_trades(losses, features=[
            "setup_type", "market_regime", "volatility_regime",
            "confluence_score", "session"
        ])

        # Generate insights
        insights = {
            "win_patterns": self._describe_clusters(win_clusters),
            "loss_patterns": self._describe_clusters(loss_clusters),
            "recommendations": self._generate_recommendations(
                win_clusters, loss_clusters
            ),
            "high_confidence_patterns": self._identify_high_confidence(
                win_clusters, min_win_rate=0.75, min_sample=5
            )
        }

        return insights

    def _cluster_trades(self, trades: list, features: list) -> list:
        """Cluster trades by similar features"""
        # Simple rule-based clustering first
        # Can upgrade to K-means later

    def _generate_recommendations(self, win_clusters, loss_clusters) -> list:
        """
        Generate actionable recommendations

        Examples:
        - "Avoid SELL setups during London session in ranging markets (0% win rate, 5 trades)"
        - "Prioritize breaker-block BUYs in trending-up + high volatility (80% win rate, 8 trades)"
        - "Reduce position size in extreme volatility (avg loss 2.5R vs 1.2R in normal)"
        """
```

**CLI Command:**

```bash
python -m cli.main analyze-patterns --trades 30
```

**Auto-Trigger:** Run every 30 closed trades

**Success Metrics:**

- Patterns identified with >70% win rate promoted to long-term memory
- Recommendations actionable and specific
- Pattern-based filtering improves win rate by 10%+

**Estimated Effort:** 4-5 days

---

### **Phase 5: Risk Guardrails (Week 5-6)** - Circuit Breakers

**Goal:** Protect capital during learning phase with hard limits

#### 5.1 Guardrails Module

**File:** `tradingagents/risk/guardrails.py`

**Components:**

```python
class RiskGuardrails:
    """Hard risk limits to prevent catastrophic losses during learning"""

    def __init__(self, config: dict = None):
        self.config = config or {
            "max_daily_loss_pct": 3.0,  # Max 3% daily loss
            "max_consecutive_losses": 2,  # Stop after 2 losses in a row
            "min_sharpe_ratio": 1.5,  # Require 1.5+ Sharpe
            "max_position_size_pct": 2.0,  # Max 2% risk per trade
            "max_drawdown_pct": 15.0,  # Stop if 15% drawdown
            "cooldown_after_breach_hours": 24,  # 24h cooldown after breach
        }
        self.breach_log = []
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

    def check_trade_allowed(
        self,
        portfolio_state: dict,
        proposed_trade: dict
    ) -> tuple[bool, str]:
        """
        Check if trade is allowed under guardrails

        Returns:
            (allowed: bool, reason: str)
        """
        # Check cooldown
        if self._in_cooldown():
            return False, "System in cooldown after guardrail breach"

        # Reset daily counters
        if datetime.now().date() > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = datetime.now().date()

        # Check daily loss limit
        if self.daily_pnl < -self.config["max_daily_loss_pct"]:
            self._log_breach("daily_loss_limit")
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}%"

        # Check consecutive losses
        if self.consecutive_losses >= self.config["max_consecutive_losses"]:
            self._log_breach("consecutive_losses")
            return False, f"Max consecutive losses: {self.consecutive_losses}"

        # Check Sharpe ratio
        if portfolio_state.get("sharpe_ratio", 0) < self.config["min_sharpe_ratio"]:
            if portfolio_state.get("num_trades", 0) > 20:  # Only enforce after 20 trades
                return False, f"Portfolio Sharpe too low: {portfolio_state['sharpe_ratio']:.2f}"

        # Check drawdown
        current_dd = portfolio_state.get("current_drawdown_pct", 0)
        if abs(current_dd) > self.config["max_drawdown_pct"]:
            self._log_breach("max_drawdown")
            return False, f"Max drawdown exceeded: {current_dd:.2f}%"

        # Check position size
        if proposed_trade.get("risk_pct", 0) > self.config["max_position_size_pct"]:
            return False, f"Position size too large: {proposed_trade['risk_pct']:.2f}%"

        return True, "All guardrails passed"

    def update_after_trade(self, trade_outcome: dict):
        """Update guardrail state after trade closes"""
        if trade_outcome["was_correct"]:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        self.daily_pnl += trade_outcome.get("pnl_pct", 0)

    def _in_cooldown(self) -> bool:
        """Check if system is in cooldown after breach"""
        if not self.breach_log:
            return False

        last_breach = self.breach_log[-1]
        hours_since = (datetime.now() - last_breach["timestamp"]).total_seconds() / 3600
        return hours_since < self.config["cooldown_after_breach_hours"]

    def _log_breach(self, breach_type: str):
        """Log guardrail breach"""
        self.breach_log.append({
            "timestamp": datetime.now(),
            "type": breach_type,
            "state": {
                "daily_pnl": self.daily_pnl,
                "consecutive_losses": self.consecutive_losses
            }
        })
```

**Integration:**

- Check before executing any trade in `cli/main.py`
- Display breach warnings prominently
- Log all breaches for analysis

**Success Metrics:**

- Zero catastrophic losses (>10% drawdown)
- Guardrails trigger appropriately (not too strict)
- System recovers after cooldowns

**Estimated Effort:** 2-3 days

---

### **Phase 6: Integration & Testing (Week 6-7)** - Wire Everything Together

**Goal:** Integrate all components into cohesive workflow

#### 6.1 End-to-End Workflow

**Modified Files:**

- `tradingagents/graph/trading_graph.py` - Add regime detection
- `tradingagents/agents/researchers/*.py` - Add RAG queries
- `cli/main.py` - Add guardrails checks, pattern analysis commands
- `examples/daily_cycle.py` - Include all new features

**New Workflow:**

```
1. Market Analysis
   ├─ Fetch prices + calculate regime (trending/ranging/volatile)
   ├─ Store regime in state
   └─ Pass to analysts

2. Agent Analysis (Bull/Bear/Market/News)
   ├─ Query similar trades with regime filter
   ├─ Adjust confidence based on historical performance
   ├─ Apply learned agent weights
   └─ Return weighted recommendation

3. Risk Management
   ├─ Check guardrails (daily loss, consecutive losses, drawdown)
   ├─ Calculate position size with Kelly criterion
   └─ Approve/reject trade

4. Trade Execution
   ├─ Store decision with full context (setup, regime, confluence)
   ├─ Execute via MT5
   └─ Track in decision system

5. Trade Close
   ├─ Calculate reward signal (RR + Sharpe + DD impact)
   ├─ Update portfolio state
   ├─ Update agent weights (online learning)
   ├─ Store outcome in memory with regime tags
   └─ Update guardrails state

6. Periodic Analysis (every 30 trades)
   ├─ Run pattern analyzer
   ├─ Generate recommendations
   ├─ Promote high-confidence patterns to long-term memory
   └─ Adjust strategy based on insights
```

**Estimated Effort:** 3-4 days

---

#### 6.2 CLI Commands

**New Commands:**

```bash
# Pattern analysis
python -m cli.main analyze-patterns --trades 30

# View learning state
python -m cli.main learning-stats

# Check guardrails status
python -m cli.main guardrails-status

# View agent weights
python -m cli.main agent-weights

# Regime analysis
python -m cli.main regime --symbol XAUUSD
```

**Estimated Effort:** 1-2 days

---

#### 6.3 Testing Strategy

**Unit Tests:**

```python
# tests/test_reward_calculator.py
def test_reward_calculation():
    reward = RewardCalculator.calculate_reward(
        realized_rr=2.5, sharpe_contribution=0.3,
        drawdown_impact=0.0, win=True
    )
    assert 0 < reward < 5

# tests/test_regime_detector.py
def test_regime_detection():
    detector = RegimeDetector()
    regime = detector.detect_trend_regime(trending_prices)
    assert regime in ["trending-up", "trending-down", "ranging"]

# tests/test_guardrails.py
def test_daily_loss_limit():
    guardrails = RiskGuardrails()
    guardrails.daily_pnl = -3.5
    allowed, reason = guardrails.check_trade_allowed({}, {})
    assert not allowed
    assert "Daily loss limit" in reason
```

**Integration Tests:**

```python
# tests/integration/test_rag_decision_flow.py
def test_rag_enhances_decision():
    # Create similar winning trades in memory
    # Run analysis
    # Verify confidence adjusted based on history

# tests/integration/test_online_learning.py
def test_weights_update_after_trades():
    # Execute 10 trades
    # Verify agent weights changed
    # Verify better agents have higher weights
```

**Estimated Effort:** 3-4 days

---

### **Phase 7: Production Deployment (Week 7-8)** - Monitoring & Optimization

**Goal:** Deploy to production with monitoring

#### 7.1 Monitoring Dashboard

**File:** `tradingagents/monitoring/dashboard.py` (Streamlit)

**Metrics to Display:**

- Real-time equity curve
- Current regime and regime history
- Agent weights over time
- Guardrails status (daily P&L, consecutive losses)
- Recent pattern analysis insights
- Win rate by regime
- Reward signal distribution

**Estimated Effort:** 3-4 days

---

#### 7.2 Performance Optimization

**Optimizations:**

- Cache regime calculations (update every 15 min)
- Batch memory queries
- Async pattern analysis (don't block trades)
- Optimize vector search with HNSW index

**Estimated Effort:** 2-3 days

---

## Success Metrics & KPIs

### Learning Effectiveness

| Metric                   | Baseline | Target (Week 4) | Target (Week 8)  |
| ------------------------ | -------- | --------------- | ---------------- |
| Win Rate                 | 52%      | 58%             | 65%              |
| Avg Risk-Reward          | 1.8      | 2.2             | 2.5              |
| Sharpe Ratio             | 1.2      | 1.8             | 2.3              |
| Max Drawdown             | 18%      | 14%             | 10%              |
| Regime-Specific Win Rate | N/A      | Measurable      | >60% all regimes |

### System Performance

| Metric                   | Target             |
| ------------------------ | ------------------ |
| Reward signal calculated | 100% of trades     |
| Regime detected          | 100% of analyses   |
| RAG queries executed     | 100% of decisions  |
| Agent weights updated    | Every 10-30 trades |
| Pattern analysis runs    | Every 30 trades    |
| Guardrail breaches       | <5% of trades      |

---

## Resource Requirements

### Development

- **Team:** 1 senior engineer (full-time)
- **Timeline:** 6-8 weeks
- **Compute:** Development machine + cloud for backtesting

### Infrastructure

- **Storage:** +500MB for enhanced trade memory
- **Compute:** Minimal (regime detection is fast)
- **APIs:** Existing (no new costs)

### Costs

- **Development:** $0 (internal)
- **Testing:** $100 (cloud compute for backtests)
- **Production:** $50/month (monitoring)

**Total:** ~$200 for 8 weeks

---

## Risk Mitigation

### Technical Risks

| Risk                                | Mitigation                                         |
| ----------------------------------- | -------------------------------------------------- |
| Online learning destabilizes system | Guardrails + small learning rate + manual override |
| Regime detection inaccurate         | Multi-indicator approach + manual validation       |
| RAG retrieval too slow              | Cache + optimize vector search + async queries     |
| Memory bloat from enhanced schema   | Compression + archival of old trades               |

### Trading Risks

| Risk                           | Mitigation                                     |
| ------------------------------ | ---------------------------------------------- |
| Learning phase causes losses   | Hard guardrails (3% daily loss, 15% max DD)    |
| Overfitting to recent patterns | Regime-aware retrieval + diverse training data |
| Agent weights diverge          | Clip weights to [0.5, 2.0] + periodic resets   |

---

## Next Steps

### Week 1 (Jan 11-17)

- [ ] Create reward calculator module
- [ ] Enhance trade decision schema
- [ ] Implement portfolio state tracker
- [ ] Write unit tests for reward calculation

### Week 2 (Jan 18-24)

- [ ] Build regime detection module
- [ ] Add regime-aware memory retrieval
- [ ] Integrate regime into analysis workflow
- [ ] Test regime detection accuracy

### Week 3 (Jan 25-31)

- [ ] Implement trade similarity search
- [ ] Enhance agent prompts with RAG
- [ ] Test RAG decision improvements
- [ ] Measure confidence adjustment impact

### Week 4 (Feb 1-7)

- [ ] Build online learner module
- [ ] Implement pattern analyzer
- [ ] Create CLI commands
- [ ] Test weight updates

### Week 5 (Feb 8-14)

- [ ] Implement risk guardrails
- [ ] Integrate all components
- [ ] End-to-end testing
- [ ] Fix integration issues

### Week 6 (Feb 15-21)

- [ ] Build monitoring dashboard
- [ ] Performance optimization
- [ ] Documentation
- [ ] User acceptance testing

### Week 7-8 (Feb 22-Mar 7)

- [ ] Production deployment
- [ ] Live testing with small positions
- [ ] Monitor and tune
- [ ] Final adjustments

---

## Appendix

### A. File Structure

```
tradingagents/
├── learning/
│   ├── __init__.py
│   ├── reward.py              # Reward signal calculation
│   ├── online_rl.py           # Agent weight updates
│   ├── pattern_analyzer.py   # Trade pattern clustering
│   ├── trade_similarity.py   # Similar trade search
│   └── portfolio_state.py    # Portfolio tracking
├── indicators/
│   ├── __init__.py
│   └── regime.py             # Regime detection
├── risk/
│   ├── __init__.py
│   ├── guardrails.py         # Risk circuit breakers
│   ├── metrics.py            # Existing
│   ├── position_sizing.py    # Existing
│   └── stop_loss.py          # Existing
└── monitoring/
    ├── __init__.py
    └── dashboard.py          # Streamlit dashboard

cli/
└── main.py                   # Enhanced with new commands

examples/
├── portfolio_state.pkl       # Persistent portfolio state
└── trade_decisions/          # Enhanced decision storage
```

### B. Configuration

**File:** `tradingagents/default_config.py`

```python
# Learning configuration
LEARNING_CONFIG = {
    "reward_weights": {
        "rr": 0.4,
        "sharpe": 0.3,
        "drawdown": 0.3
    },
    "online_learning": {
        "enabled": True,
        "learning_rate": 0.01,
        "update_frequency": 10  # trades
    },
    "pattern_analysis": {
        "enabled": True,
        "analysis_frequency": 30,  # trades
        "min_pattern_confidence": 0.75
    },
    "guardrails": {
        "max_daily_loss_pct": 3.0,
        "max_consecutive_losses": 2,
        "min_sharpe_ratio": 1.5,
        "max_position_size_pct": 2.0,
        "max_drawdown_pct": 15.0,
        "cooldown_hours": 24
    },
    "regime_detection": {
        "adx_threshold": 25,
        "atr_percentile_high": 75,
        "update_interval_minutes": 15
    }
}
```

---

**Document Version:** 1.0  
**Last Updated:** January 11, 2026  
**Next Review:** January 18, 2026 (after Week 1 completion)  
**Owner:** TradingAgents Development Team
