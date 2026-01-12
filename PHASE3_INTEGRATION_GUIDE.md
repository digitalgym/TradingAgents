# Phase 3: RAG Decision Integration - Integration Guide

## Overview

Phase 3 adds Retrieval-Augmented Generation (RAG) to the decision pipeline. Agents now query similar historical trades before making recommendations, adjusting confidence based on past performance in similar market conditions.

## Components Added

### 1. Trade Similarity Search (`tradingagents/learning/trade_similarity.py`)

Finds similar historical trades based on:

- Symbol match
- Direction (BUY/SELL)
- Market regime (trending/ranging)
- Volatility regime
- Setup characteristics (type, confluence, higher TF bias)

```python
from tradingagents.learning.trade_similarity import TradeSimilaritySearch

searcher = TradeSimilaritySearch()

current_setup = {
    "symbol": "XAUUSD",
    "direction": "BUY",
    "setup_type": "breaker-block",
    "market_regime": "trending-up",
    "volatility_regime": "normal",
    "confluence_score": 8
}

result = searcher.find_similar_trades(current_setup, n_results=5)

# Returns:
# {
#     "similar_trades": [...],
#     "similarity_scores": [0.92, 0.87, ...],
#     "statistics": {
#         "sample_size": 5,
#         "win_rate": 0.75,
#         "avg_rr": 2.3,
#         "confidence_adjustment": +0.15
#     },
#     "recommendation": "STRONG historical performance..."
# }
```

### 2. RAG Prompt Enhancement (`tradingagents/learning/rag_prompts.py`)

Utilities to enhance agent prompts with historical context:

```python
from tradingagents.learning.rag_prompts import enhance_prompt_with_rag

base_prompt = """Analyze this XAUUSD setup..."""

enhanced_prompt, adjustment = enhance_prompt_with_rag(
    base_prompt,
    current_setup,
    n_similar=5
)

# Enhanced prompt now includes:
# - Historical win rate
# - Average RR from similar trades
# - Top 3 similar trade examples
# - Recommended confidence adjustment
```

### 3. Confidence Adjustment

Automatic confidence adjustment based on historical performance:

```python
from tradingagents.learning.rag_prompts import apply_confidence_adjustment

base_confidence = 0.75  # Agent's initial confidence
adjustment = 0.15       # From historical data

final_confidence = apply_confidence_adjustment(base_confidence, adjustment)
# Returns: 0.90 (clipped to 0.1-0.95 range)
```

## Integration Steps

### Step 1: Add RAG to Agent Nodes

Enhance Bull/Bear researcher nodes with historical context:

```python
# In tradingagents/agents/researchers/bull_researcher.py

def bull_researcher_node(state) -> dict:
    # ... existing code to get market data ...

    # NEW: Build current setup
    from tradingagents.learning.trade_similarity import TradeSimilaritySearch

    current_setup = {
        "symbol": state["ticker"],
        "direction": "BUY",
        "setup_type": state.get("setup_type"),  # Extract from analysis
        "market_regime": state.get("regime", {}).get("market_regime"),
        "volatility_regime": state.get("regime", {}).get("volatility_regime"),
        "confluence_score": state.get("confluence_score"),
        "higher_tf_bias": state.get("higher_tf_bias")
    }

    # Find similar trades
    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup, n_results=5)

    # Add to state for other agents
    state["rag_context"] = rag_result

    # Format for prompt
    historical_context = searcher.format_for_prompt(
        rag_result["similar_trades"],
        rag_result["statistics"]
    )

    # Enhance prompt
    prompt = f"""
    {existing_prompt}

    ---

    {historical_context}

    IMPORTANT: Adjust your confidence based on historical performance.
    Recommended adjustment: {rag_result['statistics']['confidence_adjustment']:+.2f}

    If historical win rate is <45%, you MUST provide strong justification
    for why THIS setup is different.
    """

    # ... rest of analysis ...

    # Apply confidence adjustment
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment

    base_confidence = agent_response.confidence  # From LLM
    adjusted_confidence = apply_confidence_adjustment(
        base_confidence,
        rag_result['statistics']['confidence_adjustment']
    )

    return {
        "bull_confidence": adjusted_confidence,
        "bull_reasoning": agent_response.reasoning,
        "rag_adjustment": rag_result['statistics']['confidence_adjustment']
    }
```

### Step 2: Store Setup Metadata with Decisions

When storing decisions, include setup details for future RAG queries:

```python
from tradingagents.trade_decisions import store_decision, set_decision_regime

# Store decision with full context
decision_id = store_decision(
    symbol="XAUUSD",
    decision_type="OPEN",
    action="BUY",
    rationale="Breaker block with strong confluence",
    entry_price=2650.0,
    stop_loss=2630.0,
    take_profit=2690.0
)

# Load decision and add metadata
decision = load_decision(decision_id)
decision["setup_type"] = "breaker-block"
decision["higher_tf_bias"] = "bullish"
decision["confluence_score"] = 8
decision["confluence_factors"] = ["support-zone", "fib-618", "ema-bounce"]

# Set regime
set_decision_regime(decision_id, regime)

# Save updated decision
decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
with open(decision_file, "w") as f:
    json.dump(decision, f, indent=2, default=str)
```

### Step 3: Display RAG Context in Reports

Show historical context in analysis reports:

```python
def generate_analysis_report(state):
    rag_context = state.get("rag_context", {})
    stats = rag_context.get("statistics", {})

    report = f"""
    MARKET ANALYSIS REPORT

    Current Setup: {state['ticker']} {state['direction']}
    Regime: {state['regime']['market_regime']} / {state['regime']['volatility_regime']}

    HISTORICAL PERFORMANCE:
    - Similar Trades: {stats.get('sample_size', 0)}
    - Win Rate: {stats.get('win_rate', 0)*100:.1f}%
    - Avg RR: {stats.get('avg_rr', 0):.2f}
    - Confidence Adjustment: {stats.get('confidence_adjustment', 0):+.2f}

    RECOMMENDATION:
    {rag_context.get('recommendation', 'No historical data')}

    AGENT ANALYSIS:
    Bull Confidence: {state['bull_confidence']:.2f} (adjusted from {state['base_bull_confidence']:.2f})
    Bear Confidence: {state['bear_confidence']:.2f}
    ...
    """

    return report
```

## Usage Examples

### Example 1: Complete RAG-Enhanced Analysis

```python
def analyze_with_rag(symbol, direction, regime, setup_details):
    from tradingagents.learning.trade_similarity import TradeSimilaritySearch
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment

    # Build setup
    current_setup = {
        "symbol": symbol,
        "direction": direction,
        **regime,
        **setup_details
    }

    # Find similar trades
    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup)

    # Get agent's base analysis
    base_confidence = run_agent_analysis(current_setup)

    # Apply RAG adjustment
    final_confidence = apply_confidence_adjustment(
        base_confidence,
        rag_result['statistics']['confidence_adjustment']
    )

    # Decision logic
    if final_confidence > 0.7:
        decision = "ENTER TRADE"
    elif final_confidence > 0.5:
        decision = "CONSIDER (with caution)"
    else:
        decision = "SKIP"

    return {
        "decision": decision,
        "confidence": final_confidence,
        "historical_context": rag_result,
        "reasoning": f"Base: {base_confidence:.2f}, Adjusted: {final_confidence:.2f}"
    }
```

### Example 2: Regime-Specific Filtering

```python
def get_regime_specific_performance(setup_type, regime):
    searcher = TradeSimilaritySearch()

    setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "setup_type": setup_type,
        **regime
    }

    result = searcher.find_similar_trades(setup, n_results=20)

    print(f"{setup_type} in {regime['market_regime']}:")
    print(f"  Win Rate: {result['statistics']['win_rate']*100:.1f}%")
    print(f"  Sample: {result['statistics']['sample_size']} trades")
    print(f"  Recommendation: {result['recommendation']}")
```

### Example 3: Multi-Agent Consensus with RAG

```python
def multi_agent_decision_with_rag(current_setup):
    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup)

    # Get each agent's opinion
    agents = {
        "bull": analyze_bullish(current_setup, rag_result),
        "bear": analyze_bearish(current_setup, rag_result),
        "market": analyze_market(current_setup, rag_result)
    }

    # Apply RAG adjustment to each
    adjustment = rag_result['statistics']['confidence_adjustment']
    for agent_name, analysis in agents.items():
        analysis['adjusted_confidence'] = apply_confidence_adjustment(
            analysis['base_confidence'],
            adjustment
        )

    # Weighted consensus
    consensus = sum(
        a['adjusted_confidence'] * a['weight']
        for a in agents.values()
    )

    return {
        "consensus_confidence": consensus,
        "agents": agents,
        "historical_support": rag_result['statistics']
    }
```

## Configuration

### Similarity Scoring Weights

Adjust how similarity is calculated:

```python
searcher = TradeSimilaritySearch()

result = searcher.find_similar_trades(
    current_setup,
    n_results=5,
    min_confidence=0.6,      # Minimum similarity score
    regime_weight=0.7        # Higher = prioritize regime match
)
```

### Confidence Adjustment Bounds

```python
from tradingagents.learning.rag_prompts import apply_confidence_adjustment

adjusted = apply_confidence_adjustment(
    base_confidence=0.75,
    adjustment=0.20,
    min_confidence=0.2,      # Floor
    max_confidence=0.9       # Ceiling
)
```

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_trade_similarity.py -v

# Integration demo
python examples/test_rag_decision.py
```

## Performance Metrics

Track RAG effectiveness:

```python
def measure_rag_impact():
    """Compare decisions with and without RAG"""

    # Get recent decisions
    decisions = list_closed_decisions(limit=50)

    # Separate RAG-adjusted vs non-adjusted
    with_rag = [d for d in decisions if d.get('rag_adjustment')]
    without_rag = [d for d in decisions if not d.get('rag_adjustment')]

    # Calculate metrics
    rag_win_rate = sum(1 for d in with_rag if d['was_correct']) / len(with_rag)
    base_win_rate = sum(1 for d in without_rag if d['was_correct']) / len(without_rag)

    print(f"Win Rate with RAG: {rag_win_rate*100:.1f}%")
    print(f"Win Rate without RAG: {base_win_rate*100:.1f}%")
    print(f"Improvement: {(rag_win_rate - base_win_rate)*100:+.1f}%")
```

## Best Practices

### 1. Always Include Setup Metadata

```python
# BAD: Missing setup details
decision_id = store_decision(symbol="XAUUSD", action="BUY", ...)

# GOOD: Full context for future RAG queries
decision_id = store_decision(...)
decision["setup_type"] = "breaker-block"
decision["confluence_score"] = 8
decision["higher_tf_bias"] = "bullish"
set_decision_regime(decision_id, regime)
```

### 2. Use RAG as Guidance, Not Absolute Rule

```python
# Agent can override RAG if justified
if base_confidence > 0.8 and rag_adjustment < -0.2:
    # Strong conviction despite poor history
    if agent_has_strong_justification():
        final_confidence = 0.7  # Reduce but don't eliminate
    else:
        final_confidence = apply_confidence_adjustment(base_confidence, rag_adjustment)
```

### 3. Require Minimum Sample Size

```python
if rag_result['statistics']['sample_size'] < 5:
    # Not enough data - use base confidence
    final_confidence = base_confidence
else:
    # Sufficient data - apply adjustment
    final_confidence = apply_confidence_adjustment(
        base_confidence,
        rag_result['statistics']['confidence_adjustment']
    )
```

### 4. Log RAG Decisions for Analysis

```python
decision["rag_metadata"] = {
    "similar_trades_count": rag_result['statistics']['sample_size'],
    "historical_win_rate": rag_result['statistics']['win_rate'],
    "confidence_adjustment": rag_result['statistics']['confidence_adjustment'],
    "base_confidence": base_confidence,
    "final_confidence": final_confidence
}
```

## Troubleshooting

**Issue**: No similar trades found

- **Cause**: Very specific setup or new market conditions
- **Fix**: Reduce `min_confidence` threshold or broaden search criteria

**Issue**: RAG always suggests negative adjustment

- **Cause**: Poor historical performance in this regime
- **Fix**: This is working correctly - avoid this setup or wait for regime change

**Issue**: Similarity scores too low

- **Cause**: Missing metadata in historical trades
- **Fix**: Backfill old trades with setup_type, confluence_score, etc.

**Issue**: Confidence adjustment too aggressive

- **Cause**: Small sample size amplifying adjustment
- **Fix**: Already handled - adjustments reduced for sample_size < 10

## API Reference

### TradeSimilaritySearch

**Methods:**

- `find_similar_trades(current_setup, n_results=5, min_confidence=0.0, regime_weight=0.5)` → dict
- `format_for_prompt(similar_trades, statistics, max_trades=3)` → str

### RAG Prompts

**Functions:**

- `enhance_prompt_with_rag(base_prompt, current_setup, n_similar=5)` → (str, float)
- `apply_confidence_adjustment(base, adjustment, min=0.1, max=0.95)` → float
- `format_confidence_explanation(base, adjustment, statistics)` → str

---

**Phase 3 Complete** ✅

Agents now query historical trades before making recommendations, dramatically improving decision quality through experience-based learning.

**Next**: Phase 4 will add online learning to automatically update agent weights based on performance.
