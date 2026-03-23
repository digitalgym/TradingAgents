# Smart Money Concepts (SMC) System Guide

This guide covers the TradingAgents SMC implementation - from core concepts to the hybrid trade planning system.

---

## Table of Contents

1. [Overview](#overview)
2. [Core SMC Concepts](#core-smc-concepts)
3. [Integration with Analysis](#integration-with-analysis)
4. [CLI Commands](#cli-commands)
5. [Hybrid Trade Planning System](#hybrid-trade-planning-system)
6. [Python API](#python-api)
7. [Implementation Details](#implementation-details)

---

## Overview

Smart Money Concepts (SMC) tracks institutional trading behavior through price structure analysis. The TradingAgents system uses SMC to:

1. **Identify institutional zones** (Order Blocks, FVGs)
2. **Detect market structure** (BOS, CHoCH)
3. **Locate liquidity** (stop-loss clusters)
4. **Generate trade plans** with SMC-aligned entries, stops, and targets

### Why SMC Matters

**Before SMC:** Arbitrary TP/SL levels based on percentages
```
Entry: $2650 | Stop: $2640 (1% risk) | TP: $2700 (2% target)
Problem: Stop may be in middle of support zone
```

**After SMC:** Levels aligned with institutional zones
```
Entry: $2650 | Stop: $2628 (below OB) | TP: $2710 (at resistance OB)
Result: Stops avoid premature exits, TPs at actual reaction zones
```

---

## Core SMC Concepts

### 1. Order Blocks (OB)

**Definition:** Zones where institutions accumulate/distribute positions. Identified as the last opposing candle before a strong move.

| Type | Location | Entry Action | Stop Placement |
|------|----------|--------------|----------------|
| Bullish OB | Below price | BUY when price returns | Below OB low |
| Bearish OB | Above price | SELL when price returns | Above OB high |

**Quality Factors:**
- Caused BOS or CHoCH
- Has FVG confluence
- Fresh (untested)
- Aligned with HTF bias

### 2. Fair Value Gaps (FVG)

**Definition:** Price imbalances where the market moved so fast it left a gap between candle wicks. Price tends to return to fill these gaps.

**3-Candle Pattern:**
```
Bullish FVG: Gap between Candle 1 HIGH and Candle 3 LOW
Bearish FVG: Gap between Candle 1 LOW and Candle 3 HIGH
```

### 3. Break of Structure (BOS)

**Definition:** Trend continuation signal - price breaks a previous swing high (uptrend) or swing low (downtrend).

- Multiple BOS in same direction = strong trend
- Look for entries on pullbacks after BOS

### 4. Change of Character (CHoCH)

**Definition:** First sign of trend reversal - price breaks structure against the current trend.

- CHoCH = Stop looking for trend continuation trades
- High probability setup: CHoCH + return to order block

### 5. Liquidity

**Definition:** Price levels where stop-loss clusters exist.

| Type | Location | Description |
|------|----------|-------------|
| Buy-Side Liquidity (BSL) | Above swing highs | Short sellers' stops |
| Sell-Side Liquidity (SSL) | Below swing lows | Long buyers' stops |

**Liquidity Sweep:** Quick spike through liquidity then reversal - often precedes strong moves.

### 6. Premium/Discount Zones

- **Premium:** Above 50% of range - ideal for SELL entries
- **Discount:** Below 50% of range - ideal for BUY entries

---

## Integration with Analysis

SMC analysis runs **automatically** when using `analyze` command with:
- Asset type: **Commodity**
- Data vendor: **MT5**

### Data Flow

```
1. User runs: python -m cli.main analyze
   └─> Selects: Commodity + MT5

2. CLI fetches SMC data (1H, 4H, D1)
   └─> Detects: Order Blocks, FVGs, BOS, CHoCH
   └─> Formats: Multi-timeframe analysis

3. SMC context added to graph state
   └─> state["smc_context"] = formatted SMC text
   └─> state["smc_analysis"] = raw SMC data

4. Trader Agent receives SMC
   └─> Instructed to align TP/SL with institutional levels

5. Risk Analysts receive SMC
   ├─> Risky: Uses SMC for aggressive targets
   ├─> Safe: Ensures stops beyond zones
   └─> Neutral: Balanced recommendations

6. Final decision includes SMC-aligned levels
```

### What You'll See

**During Analysis:**
```
System: Running Smart Money Concepts analysis...
System: SMC Analysis: 2/3 timeframes bullish - 2 TFs with unmitigated OBs
```

**After Analysis:**
```
╭─────────────────────────────────────────────────╮
│ Smart Money Concepts Validation                 │
╰─────────────────────────────────────────────────╯

SMC-Suggested Stop Loss:
  $89.50 (below fvg at $90.20)
  Distance: 3.17% | Strength: 75%

SMC-Suggested Take Profits:
  TP1: $95.80 (+3.6%) at order_block zone
  TP2: $98.50 (+6.5%) at fvg zone
```

---

## CLI Commands

### Analyze SMC

```bash
python -m cli.main smc XAUUSD
python -m cli.main smc XAUUSD --timeframes "1H,4H,D1"
```

### Get SMC-Based Levels

```bash
python -m cli.main smc-levels XAUUSD --direction BUY
python -m cli.main smc-levels XAUUSD --direction BUY --entry 2650
```

### Validate Trade Plan

```bash
python -m cli.main smc-validate XAUUSD \
  --direction BUY \
  --entry 2650 \
  --stop 2628 \
  --target 2680
```

---

## Hybrid Trade Planning System

The system combines **systematic rules** with **AI refinement**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID TRADE DECISION FLOW                    │
├─────────────────────────────────────────────────────────────────┤
│   LAYER 1: RULES (SMCTradePlanGenerator)                        │
│   • Enforces systematic SMC strategy                            │
│   • Calculates entry/SL/TP from zones                           │
│   • Scores zone quality (0-100)                                 │
│   • Validates entry checklist                                   │
│                          ↓                                       │
│   LAYER 2: AI (LLMTradeRefiner)                                 │
│   • Evaluates context rules can't capture                       │
│   • Learns from historical trade outcomes                       │
│   • Adjusts levels for micro-structure                          │
│   • Decides position sizing                                     │
│                          ↓                                       │
│   FINAL DECISION                                                 │
│   • Systematic levels with AI adjustments                       │
│   • Clear reasoning for every decision                          │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Rule-Based Generator

**Location:** `tradingagents/dataflows/smc_trade_plan.py`

**Entry Checklist (7 items):**
- [ ] HTF Trend Aligned
- [ ] Zone Unmitigated
- [ ] Has Confluence (OB+FVG)
- [ ] Liquidity Target Exists
- [ ] Structure Confirmed (BOS/CHoCH)
- [ ] In Premium/Discount Zone
- [ ] Session Favorable

**Validation Rules:**
- Zone quality ≥ 60 to trade
- Risk:Reward ≥ 1.5:1
- SL below zone (BUY) / above zone (SELL)
- TP at liquidity or opposing zone

### Layer 2: AI Refiner

**Location:** `tradingagents/dataflows/llm_trade_refiner.py`

**What It Considers:**
- Historical trades with similar setup (win rate, lessons)
- Market context (session, volatility, news)
- Existing positions (correlation)
- Daily P/L (risk adjustment)

**What It Can Adjust:**
| Adjustment | Constraints |
|------------|-------------|
| Entry price | Within zone bounds |
| Stop loss | Must stay on correct side |
| Position size | 0.5x to 1.5x multiplier |
| Partial TPs | Additional exit levels |

**What It Cannot Override:**
- SL direction
- Minimum R:R ratio
- Zone quality threshold

---

## Python API

### Multi-Timeframe Analysis

```python
from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc

mtf = analyze_multi_timeframe_smc("XAUUSD", ['1H', '4H', 'D1'])
d1 = mtf['D1']

print(f"Bias: {d1['bias']}")
print(f"Unmitigated OBs: {d1['order_blocks']['unmitigated']}")
```

### Get SMC Stop Loss

```python
from tradingagents.dataflows.smc_utils import suggest_smc_stop_loss

stop = suggest_smc_stop_loss(
    smc_analysis=d1,
    direction="BUY",
    entry_price=2650.00,
    max_distance_pct=3.0
)
print(f"Stop: ${stop['price']:.2f} - {stop['reason']}")
```

### Get SMC Take Profits

```python
from tradingagents.dataflows.smc_utils import suggest_smc_take_profits

tps = suggest_smc_take_profits(d1, "BUY", entry_price=2650.00, num_targets=3)
for tp in tps:
    print(f"TP{tp['number']}: ${tp['price']:.2f} ({tp['source']})")
```

### Hybrid Trade Decision

```python
from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator
from tradingagents.dataflows.llm_trade_refiner import LLMTradeRefiner

# Step 1: Rule-based plan
generator = SMCTradePlanGenerator()
plan = generator.generate_plan(
    smc_analysis=analysis,
    current_price=2850.0,
    atr=10.0,
    market_regime="trending-up"
)

# Step 2: AI refinement
refiner = LLMTradeRefiner()
refined = refiner.refine_plan(plan, historical_context, market_context)

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

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| `tradingagents/indicators/smart_money.py` | SMC zone detection (2600+ lines) |
| `tradingagents/dataflows/smc_utils.py` | SMC utility functions |
| `tradingagents/dataflows/smc_trade_plan.py` | Rule-based plan generator |
| `tradingagents/dataflows/llm_trade_refiner.py` | AI refinement layer |
| `tradingagents/agents/trader/trader.py` | Trader agent integration |

### Detection Functions

The `SmartMoneyAnalyzer` class provides:

1. Order Blocks (candle-based + structural)
2. Fair Value Gaps with fill tracking
3. BOS/CHoCH with trend context
4. Liquidity Zones with sweep detection
5. Equal Highs/Lows
6. Breaker Blocks
7. Premium/Discount Zones
8. OTE (Optimal Trade Entry) Zones
9. Session Analysis (Asian, London, NY)
10. Inducements (false breakouts)
11. Rejection Blocks
12. Turtle Soup patterns

### Zone Quality Scoring

```python
score = 0
if caused_structure_break: score += 25
if has_nearby_fvg: score += 20
if has_liquidity_nearby: score += 20
if fresh_unmitigated: score += 20
if htf_aligned: score += 15

# 80+ = Excellent | 60-79 = Good | 40-59 = Moderate | <40 = Avoid
```

### Regime Adjustments

Zone strength is adjusted by market regime:
- Trending-up: Bullish zones +20%, bearish zones -30%
- Trending-down: Bearish zones +20%, bullish zones -30%
- Ranging: FVGs +15%, Equal levels +20%

### Time Decay

Older zones receive strength penalties:
- Fresh (< 20 candles): 100%
- Aging (20-50 candles): 75%
- Old (50-100 candles): 50%
- Very old (> 100 candles): 25%

---

## Glossary

| Term | Definition |
|------|------------|
| **BOS** | Break of Structure - trend continuation signal |
| **CHoCH** | Change of Character - potential reversal signal |
| **FVG** | Fair Value Gap - price imbalance zone |
| **OB** | Order Block - institutional entry zone |
| **BSL** | Buy-Side Liquidity - stops above swing highs |
| **SSL** | Sell-Side Liquidity - stops below swing lows |
| **OTE** | Optimal Trade Entry - 0.62-0.79 Fib zone |
| **Mitigation** | When price returns to and fills a zone |
| **Sweep** | Quick move through liquidity then reversal |

---

*Last Updated: 2026-03-23*
