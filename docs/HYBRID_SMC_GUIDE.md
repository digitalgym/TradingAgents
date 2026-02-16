# Hybrid SMC Trade Plan System Guide

This guide explains how the hybrid Smart Money Concepts (SMC) trade planning system works, combining systematic rules with AI-powered refinement.

## Overview

The system has two layers that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID TRADE DECISION FLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   LAYER 1: RULES (SMCTradePlanGenerator)                        │
│   ═══════════════════════════════════════                       │
│   • Enforces systematic SMC strategy                            │
│   • Calculates entry/SL/TP from zones                          │
│   • Scores zone quality (0-100)                                 │
│   • Validates entry checklist                                   │
│   • Provides CONSISTENT, DISCIPLINED foundation                 │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│   LAYER 2: AI (LLMTradeRefiner)                                 │
│   ═════════════════════════════                                 │
│   • Evaluates context rules can't capture                       │
│   • Learns from historical trade outcomes                       │
│   • Adjusts levels for micro-structure                          │
│   • Decides position sizing                                     │
│   • Provides ADAPTIVE, LEARNING refinement                      │
│                                                                  │
│                          ↓                                       │
│                                                                  │
│   FINAL DECISION                                                 │
│   ══════════════                                                │
│   • Systematic levels with AI adjustments                       │
│   • Clear reasoning for every decision                          │
│   • Confidence score                                            │
│   • Historical context considered                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Hybrid?

| Pure Rules | Pure AI | Hybrid (Best of Both) |
|------------|---------|----------------------|
| Consistent but rigid | Adaptive but inconsistent | Consistent AND adaptive |
| No learning | Learns but may overfit | Learns within boundaries |
| Clear logic | Black box decisions | Clear logic + reasoning |
| Misses context | May hallucinate | Context-aware with validation |

---

## Layer 1: SMCTradePlanGenerator

**Location**: `tradingagents/dataflows/smc_trade_plan.py`

### What It Does

The rule-based generator follows the SMC trading strategy exactly:

#### Step 1: Determine Trend
```python
# Only trade WITH the trend
if market_regime == "trending-up":
    trade_direction = "BUY"
elif market_regime == "trending-down":
    trade_direction = "SELL"
```

#### Step 2: Find Best Entry Zone
Evaluates all detected zones (Order Blocks, FVGs, Breakers) and scores them:
- **Zone strength** (0-40 points)
- **Proximity to price** (0-20 points)
- **Unmitigated status** (10 points)
- **FVG confluence** (15 points)
- **Structure confirmation** (15 points)

#### Step 3: Calculate Entry Price
```python
# For BUY: Enter at upper portion of bullish zone
entry = zone_top - (zone_size * 0.5)  # 50% of zone

# For SELL: Enter at lower portion of bearish zone
entry = zone_bottom + (zone_size * 0.5)
```

#### Step 4: Calculate Stop Loss
```python
# For BUY: SL below zone with ATR buffer
sl = zone_bottom - (atr * 0.5)

# Check for liquidity - extend if needed
if liquidity_just_below_sl:
    sl = liquidity_price - (atr * 0.3)
```

#### Step 5: Calculate Take Profit
Targets are selected in order of priority:
1. Next liquidity zone (buy-side for longs, sell-side for shorts)
2. Opposing order block
3. Calculated 3:1 R:R fallback

#### Step 6: Validate Checklist
```
Entry Checklist (7 items):
[ ] HTF Trend Aligned
[ ] Zone Unmitigated
[ ] Has Confluence (OB+FVG)
[ ] Liquidity Target Exists
[ ] Structure Confirmed (BOS/CHoCH)
[ ] In Premium/Discount Zone
[ ] Session Favorable
```

#### Step 7: Make Recommendation
```python
if zone_quality >= 60 and rr_ratio >= 1.5 and checklist.pass_rate >= 0.5:
    recommendation = "TAKE"
else:
    recommendation = "SKIP"
    skip_reason = "Quality/RR/Checklist failed"
```

### Output: SMCTradePlan

```python
{
    "signal": "BUY",
    "entry_price": 2842.50,
    "stop_loss": 2835.00,
    "take_profit": 2870.00,
    "zone_quality_score": 75,
    "setup_type": "ob_fvg_confluence",
    "risk_reward_ratio": 3.67,
    "checklist": {
        "htf_trend_aligned": True,
        "zone_unmitigated": True,
        "has_confluence": True,
        "passed": 5,
        "total": 7
    },
    "confluence_factors": ["Bullish OB", "FVG Confluence", "Recent BOS"],
    "recommendation": "TAKE"
}
```

---

## Layer 2: LLMTradeRefiner

**Location**: `tradingagents/dataflows/llm_trade_refiner.py`

### What It Does

The AI refiner receives the rule-based plan and adds intelligence:

#### 1. Historical Learning
```python
# Query past trades with similar setup
historical_context = {
    "setup_type": "ob_fvg_confluence",
    "total_trades": 23,
    "win_rate": 0.65,
    "avg_rr_achieved": 2.1,
    "similar_trades": [
        {"outcome": "WIN", "lesson": "London session entries performed best"},
        {"outcome": "LOSS", "lesson": "Avoided during high-impact news"}
    ],
    "key_lessons": [
        "This setup works 70% in trending markets, only 45% in ranging",
        "Wait for LTF confirmation improves win rate by 15%"
    ]
}
```

#### 2. Context Evaluation
```python
# Current market context
market_context = {
    "session": "london_ny_overlap",  # Best session
    "volatility": "normal",
    "market_regime": "trending-up",
    "upcoming_news": "FOMC in 4 hours",  # Risk factor
    "daily_pnl": -1.2%,  # Already down today
    "existing_positions": ["XAGUSD LONG"]  # Correlation
}
```

#### 3. Refinement Decision

The LLM considers all factors and outputs:

```python
{
    "action": "MODIFY",  # TAKE, SKIP, or MODIFY
    "confidence": 0.75,

    # Adjusted levels (null = use base plan)
    "adjusted_entry": 2843.00,  # Slightly higher for micro FVG
    "adjusted_sl": 2834.00,     # Extended past liquidity
    "adjusted_tp": null,        # Keep base plan TP

    # Partial TPs for scaling out
    "partial_tp_levels": [2855.00, 2870.00],

    # Position sizing
    "size_multiplier": 0.8,  # Reduced due to FOMC

    # Reasoning
    "reasoning": "Setup is valid but FOMC risk suggests caution.
                  Historical win rate for this setup is 65%.
                  Adjusting entry up slightly to catch unfilled FVG at 2843.
                  Extending SL past visible liquidity at 2835.",

    "key_factors": [
        "Strong OB+FVG confluence",
        "HTF trend aligned",
        "FOMC risk (reduced size)"
    ],

    "warnings": [
        "FOMC announcement in 4 hours",
        "Correlated XAGUSD position open"
    ]
}
```

### What the LLM CAN and CANNOT Do

| LLM CAN | LLM CANNOT |
|---------|------------|
| Adjust entry within zone | Place SL on wrong side |
| Widen/tighten SL for liquidity | Violate min R:R ratio |
| Add partial TP levels | Skip high-quality setups without reason |
| Reduce size for uncertainty | Increase size beyond 1.5x |
| Skip based on news/correlation | Override all rules |

---

## Integration with Trader Agent

**Location**: `tradingagents/agents/trader/trader.py`

The trader agent now uses the hybrid system:

```python
def create_trader(llm, memory, smc_memory=None, use_hybrid_smc=True):
    def trader_node(state, name):
        # ... existing code ...

        if use_hybrid_smc and smc_analysis and current_price:
            # Step 1: Generate rule-based plan
            generator = SMCTradePlanGenerator()
            base_plan = generator.generate_plan(
                smc_analysis=flat_smc,
                current_price=current_price,
                atr=atr,
                market_regime=market_regime,
            )

            if base_plan:
                # Step 2: Refine with LLM
                refiner = LLMTradeRefiner(llm=llm)
                refined_plan = refiner.refine_plan(
                    base_plan=base_plan,
                    historical_context=historical_context,
                    market_context=market_context,
                )

                # Include in trader's context
                hybrid_plan_context = format_plan_for_prompt(refined_plan)
```

The trader sees both the systematic plan AND the AI refinement, then makes the final decision with full context.

---

## Usage Examples

### Basic Usage

```python
from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator
from tradingagents.dataflows.llm_trade_refiner import LLMTradeRefiner

# Your SMC analysis (from SmartMoneyAnalyzer)
smc_analysis = analyzer.analyze_full_smc(df)

# Step 1: Generate rule-based plan
generator = SMCTradePlanGenerator()
base_plan = generator.generate_plan(
    smc_analysis=smc_analysis,
    current_price=2850.0,
    atr=10.0,
    market_regime="trending-up"
)

print(f"Signal: {base_plan.signal}")
print(f"Entry: {base_plan.entry_price}")
print(f"SL: {base_plan.stop_loss}")
print(f"TP: {base_plan.take_profit}")
print(f"Quality: {base_plan.zone_quality_score}/100")
print(f"Recommendation: {base_plan.recommendation}")

# Step 2: Refine with LLM
refiner = LLMTradeRefiner()
refined = refiner.refine_plan(
    base_plan=base_plan,
    historical_context={"win_rate": 0.65, "total_trades": 20},
    market_context={"session": "london", "volatility": "normal"}
)

print(f"\nAI Action: {refined.action}")
print(f"Confidence: {refined.confidence:.0%}")
print(f"Final Entry: {refined.final_entry}")
print(f"Final SL: {refined.final_sl}")
print(f"Final TP: {refined.final_tp}")
print(f"Size: {refined.size_multiplier:.1f}x")
print(f"Reasoning: {refined.reasoning}")
```

### Convenience Function

```python
from tradingagents.dataflows.llm_trade_refiner import create_hybrid_trade_decision

# One-liner for full hybrid decision
refined = create_hybrid_trade_decision(
    smc_analysis=smc_analysis,
    current_price=2850.0,
    atr=10.0,
    market_regime="trending-up",
    smc_memory=smc_pattern_memory,  # For historical patterns
    trade_similarity=similarity_search,  # For RAG context
)

if refined:
    print(refined.to_dict())
```

---

## How Learning Works

### 1. Trade Execution
When a trade is taken, the system stores:
- Setup type (e.g., "ob_fvg_confluence")
- Entry conditions (regime, session, confluence factors)
- Final levels used

### 2. Trade Outcome
When trade closes, the system records:
- Win/Loss result
- Actual R:R achieved
- Why it won/lost (structured analysis)

### 3. Pattern Storage
```python
smc_memory.store_pattern(
    symbol="XAUUSD",
    setup_type="ob_fvg_confluence",
    direction="BUY",
    outcome={"result": "win", "returns_pct": 2.5},
    lesson="OB+FVG in London session during uptrend = high probability"
)
```

### 4. Future Retrieval
Next time a similar setup appears:
```python
similar = smc_memory.get_similar_patterns(
    current_situation,
    symbol="XAUUSD",
    n_matches=5
)
# Returns: "Last 5 similar trades: 4 wins, 1 loss, avg +1.8%"
```

### 5. LLM Uses History
The refiner receives this context and adjusts:
- High historical win rate → Higher confidence, potentially larger size
- Low historical win rate → Lower confidence, smaller size, or SKIP
- Specific lessons → Adjusted entry/SL/TP based on what worked

---

## Configuration Options

### SMCTradePlanGenerator

```python
generator = SMCTradePlanGenerator(
    min_quality_score=60.0,   # Skip setups below this quality
    min_rr_ratio=1.5,         # Minimum risk:reward ratio
    sl_buffer_atr=0.5,        # ATR multiplier for SL buffer
    entry_zone_percent=0.5,   # Where in zone to enter (0.5 = middle)
)
```

### LLMTradeRefiner

```python
refiner = LLMTradeRefiner(
    llm=your_llm,             # Or will create GPT-4o-mini
    temperature=0.3,          # Lower = more consistent
)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `tradingagents/dataflows/smc_trade_plan.py` | Rule-based plan generator |
| `tradingagents/dataflows/llm_trade_refiner.py` | AI refinement layer |
| `tradingagents/agents/trader/trader.py` | Integration with trader agent |
| `tradingagents/indicators/smart_money.py` | SMC zone detection |
| `docs/SMC_REFERENCE.md` | SMC strategy reference |
| `tests/test_smc_trade_plan.py` | Unit tests |

---

## Summary

The hybrid system gives you:

1. **Discipline**: Rules ensure you don't chase bad setups
2. **Consistency**: Same criteria applied every time
3. **Learning**: AI improves based on past outcomes
4. **Adaptability**: Context-aware adjustments
5. **Transparency**: Clear reasoning for every decision
6. **Safety**: Invalid adjustments are rejected

The rules set the boundaries; the AI works within them to optimize.
