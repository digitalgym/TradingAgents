# SMC Trade Plan System - Implementation Summary

## What Was Built

A hybrid trade planning system that combines **systematic SMC rules** with **AI-powered refinement**.

```
┌────────────────────────────────────────────────────────────┐
│                  HYBRID ARCHITECTURE                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   SMC Analysis (zones, structure, liquidity)               │
│                        ↓                                   │
│   ┌────────────────────────────────────┐                  │
│   │  LAYER 1: SMCTradePlanGenerator    │ ← Rules          │
│   │  • Zone quality scoring            │                  │
│   │  • Entry/SL/TP calculation         │                  │
│   │  • Checklist validation            │                  │
│   │  • TAKE/SKIP recommendation        │                  │
│   └────────────────┬───────────────────┘                  │
│                    ↓                                       │
│   ┌────────────────────────────────────┐                  │
│   │  LAYER 2: LLMTradeRefiner          │ ← Intelligence   │
│   │  • Historical learning             │                  │
│   │  • Context evaluation              │                  │
│   │  • Level adjustments               │                  │
│   │  • Position sizing                 │                  │
│   └────────────────┬───────────────────┘                  │
│                    ↓                                       │
│   Final Decision with reasoning                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Files Created

| File | Description |
|------|-------------|
| `tradingagents/dataflows/smc_trade_plan.py` | Rule-based plan generator |
| `tradingagents/dataflows/llm_trade_refiner.py` | AI refinement layer |
| `tests/test_smc_trade_plan.py` | 31 unit tests |
| `docs/HYBRID_SMC_GUIDE.md` | Detailed usage guide |
| `docs/SMC_REFERENCE.md` | SMC strategy reference |

## Files Modified

| File | Changes |
|------|---------|
| `tradingagents/agents/trader/trader.py` | Integrated hybrid system |
| `tradingagents/indicators/smart_money.py` | Added regime/time-decay methods |

---

## Layer 1: Rule-Based Generator

### What It Enforces

1. **Trend Alignment**: Only BUY in uptrend, SELL in downtrend
2. **Zone Quality**: Scores 0-100, requires ≥60 to trade
3. **Entry Placement**: At zone level (configurable %)
4. **Stop Loss**: Below zone (BUY) or above zone (SELL) with ATR buffer
5. **Take Profit**: Next liquidity zone or opposing OB
6. **Risk:Reward**: Minimum 1.5:1 required
7. **Checklist**: 7-point validation before recommending

### Entry Checklist

```
[ ] HTF Trend Aligned
[ ] Zone Unmitigated
[ ] Has Confluence (OB+FVG)
[ ] Liquidity Target Exists
[ ] Structure Confirmed (BOS/CHoCH)
[ ] In Premium/Discount Zone
[ ] Session Favorable
```

### Output

```python
SMCTradePlan(
    signal="BUY",
    entry_price=2842.50,
    stop_loss=2835.00,
    take_profit=2870.00,
    zone_quality_score=75,
    setup_type="ob_fvg_confluence",
    risk_reward_ratio=3.67,
    recommendation="TAKE"
)
```

---

## Layer 2: AI Refiner

### What It Considers

| Input | How It's Used |
|-------|---------------|
| Historical trades | Win rate, lessons learned |
| Market context | Session, volatility, news |
| Existing positions | Correlation warnings |
| Daily P/L | Risk adjustment |

### What It Can Adjust

| Adjustment | Constraints |
|------------|-------------|
| Entry price | Within zone bounds |
| Stop loss | Must stay on correct side |
| Take profit | Must stay on correct side |
| Position size | 0.5x to 1.5x multiplier |
| Partial TPs | Additional exit levels |

### Output

```python
RefinedTradePlan(
    action="MODIFY",
    confidence=0.75,
    adjusted_entry=2843.00,
    adjusted_sl=2834.00,
    size_multiplier=0.8,
    reasoning="Strong setup but FOMC risk...",
    key_factors=["OB+FVG confluence", "HTF aligned"],
    warnings=["FOMC in 4 hours"]
)
```

---

## How Learning Works

```
Trade Taken
    ↓
Outcome Recorded (win/loss, R:R achieved, lessons)
    ↓
Stored in SMC Pattern Memory
    ↓
Future Similar Setup
    ↓
Historical Context Retrieved
    ↓
LLM Considers: "Last 5 similar: 4 wins, 1 loss"
    ↓
Adjusts confidence/size accordingly
```

---

## Quick Start

### Basic Usage

```python
from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator
from tradingagents.dataflows.llm_trade_refiner import LLMTradeRefiner

# Generate rule-based plan
generator = SMCTradePlanGenerator()
plan = generator.generate_plan(
    smc_analysis=your_analysis,
    current_price=2850.0,
    atr=10.0,
    market_regime="trending-up"
)

# Refine with AI
refiner = LLMTradeRefiner()
refined = refiner.refine_plan(plan)

print(f"Action: {refined.action}")
print(f"Entry: {refined.final_entry}")
print(f"Reasoning: {refined.reasoning}")
```

### One-Liner

```python
from tradingagents.dataflows.llm_trade_refiner import create_hybrid_trade_decision

refined = create_hybrid_trade_decision(
    smc_analysis=analysis,
    current_price=2850.0,
    atr=10.0,
    market_regime="trending-up"
)
```

---

## Configuration

### Generator Options

```python
SMCTradePlanGenerator(
    min_quality_score=60.0,   # Skip below this
    min_rr_ratio=1.5,         # Minimum R:R
    sl_buffer_atr=0.5,        # SL buffer in ATR
    entry_zone_percent=0.5    # Entry at 50% of zone
)
```

### Refiner Options

```python
LLMTradeRefiner(
    model_name="gpt-4o-mini",
    temperature=0.3           # Lower = more consistent
)
```

---

## Validation Rules

### Always Enforced

| Rule | BUY | SELL |
|------|-----|------|
| Stop Loss | Below entry | Above entry |
| Take Profit | Above entry | Below entry |
| Min R:R | ≥ 1.5 | ≥ 1.5 |
| Min Quality | ≥ 60 | ≥ 60 |

### LLM Cannot Override

- SL direction (would be rejected)
- TP direction (would be rejected)
- Size > 1.5x (capped)
- Size < 0.5x (capped)

---

## Integration Points

### Trader Agent

The trader now receives a `hybrid_plan_context` in its prompt:

```
=== SYSTEMATIC SMC TRADE PLAN ===

RULE-BASED PLAN:
Signal: BUY
Entry: 2842.50
Stop Loss: 2835.00
Take Profit: 2870.00
Zone Quality: 75/100
R:R Ratio: 3.67
Recommendation: TAKE

AI REFINEMENT:
Action: MODIFY
Confidence: 75%
Final Entry: 2843.00
Reasoning: ...
```

### State Output

The trader returns `smc_trade_plan` in its output for downstream use:

```python
{
    "trader_investment_plan": "...",
    "smc_trade_plan": {
        "action": "TAKE",
        "entry": 2843.00,
        "stop_loss": 2834.00,
        "take_profit": 2870.00,
        ...
    }
}
```

---

## Tests

31 tests in `tests/test_smc_trade_plan.py`:

- Generator initialization
- BUY/SELL signal generation
- SL/TP directional validation
- Quality scoring
- R:R calculation
- Checklist population
- LLM response parsing
- Error handling
- Integration tests

Run with:
```bash
pytest tests/test_smc_trade_plan.py -v
```

---

## Benefits

| Aspect | Rules Provide | AI Adds |
|--------|---------------|---------|
| Consistency | Same criteria every time | - |
| Discipline | No chasing bad setups | - |
| Learning | - | Improves from outcomes |
| Context | - | News, session, correlation |
| Transparency | Clear calculations | Reasoning for adjustments |
| Safety | Directional validation | Cannot override |
