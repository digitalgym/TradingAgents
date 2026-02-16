# Smart Money Concept (SMC) Trading Strategy Reference

This document serves as the authoritative reference for implementing and reviewing SMC-based trading strategies in the TradingAgents system.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Key Concepts](#key-concepts)
   - [Order Blocks (OB)](#1-order-blocks-ob)
   - [Breaker Blocks](#2-breaker-blocks)
   - [Fair Value Gaps (FVG)](#3-fair-value-gaps-fvg)
   - [Break of Structure (BOS)](#4-break-of-structure-bos)
   - [Change of Character (CHoCH)](#5-change-of-character-choch)
   - [Liquidity](#6-liquidity)
3. [Trading Strategy](#trading-strategy)
4. [Implementation Checklist](#implementation-checklist)
5. [Zone Quality Scoring](#zone-quality-scoring)
6. [Entry/Exit Rules](#entryexit-rules)

---

## Core Philosophy

Smart Money Concept (SMC) is based on the theory that:

1. **Market Makers Control Price**: Banks, hedge funds, and institutions ("smart money") leave footprints of their trading decisions on charts
2. **Follow the Money**: Retail traders should identify and follow these institutional footprints
3. **Supply & Demand**: Markets are driven by institutional accumulation (buying) and distribution (selling)
4. **Liquidity Hunting**: Smart money often triggers retail stop losses to accumulate positions at better prices

### Key Principle
> "Instead of just looking at the chart, SMC traders try to identify where the smart money goes."

---

## Key Concepts

### 1. Order Blocks (OB)

**Definition**: Zones where institutions accumulate or distribute large positions through multiple orders, without causing market panic.

**Identification**:
- Appears as a **ranging/consolidation zone** before a strong move
- The **last opposite candle** before an impulsive move
- For bullish OB: Last bearish candle before strong bullish move
- For bearish OB: Last bullish candle before strong bearish move

**Characteristics**:
| Property | Bullish OB | Bearish OB |
|----------|------------|------------|
| Location | Below current price | Above current price |
| Candle | Last bearish candle before up-move | Last bullish candle before down-move |
| Expected Action | Price bounces UP from zone | Price rejects DOWN from zone |
| Entry | Buy when price returns to zone | Sell when price returns to zone |
| Stop Loss | Below the OB low | Above the OB high |

**Quality Factors**:
- [ ] Caused a Break of Structure (BOS)
- [ ] Has unmitigated FVG nearby
- [ ] Fresh (not previously tested)
- [ ] Aligned with higher timeframe bias
- [ ] Strong impulse move away from zone

**Implementation Requirements**:
```
- detect_order_blocks(ohlc_data, lookback=50)
- Returns: list of {type, high, low, timestamp, strength, mitigated}
- Strength: 0-100% based on impulse move size, volume, freshness
```

---

### 2. Breaker Blocks

**Definition**: Order blocks that **failed** to hold - price broke through them, converting support to resistance (or vice versa).

**Formation**:
1. Order block forms (accumulation/distribution zone)
2. Price breaks through the order block
3. The broken order block becomes a breaker block
4. Price often returns to the breaker block before continuing

**Trading Logic**:
- Bullish Breaker: Former resistance becomes support
- Bearish Breaker: Former support becomes resistance

**Use Case**: Identifies failed institutional levels that now act as reversal zones.

**Implementation Requirements**:
```
- detect_breaker_blocks(ohlc_data, order_blocks)
- Input: existing order blocks
- Returns: list of broken OBs with new polarity
```

---

### 3. Fair Value Gaps (FVG)

**Definition**: Price imbalances created when the market moves so quickly that a gap forms between candle wicks - areas where price didn't trade.

**Identification (3-Candle Pattern)**:
```
Bullish FVG:
  Candle 1: Any candle
  Candle 2: Strong bullish candle (impulse)
  Candle 3: Gap exists between Candle 1's HIGH and Candle 3's LOW

  FVG Zone = [Candle 1 High, Candle 3 Low]

Bearish FVG:
  Candle 1: Any candle
  Candle 2: Strong bearish candle (impulse)
  Candle 3: Gap exists between Candle 1's LOW and Candle 3's HIGH

  FVG Zone = [Candle 3 High, Candle 1 Low]
```

**Trading Logic**:
- Price tends to **return to fill** FVGs before continuing
- Unfilled FVGs act as magnets for price
- FVGs near order blocks increase zone strength

**Quality Factors**:
- Size of gap (larger = more significant)
- Unfilled (fresh) vs partially/fully mitigated
- Alignment with trend direction
- Confluence with OB

**Implementation Requirements**:
```
- detect_fair_value_gaps(ohlc_data)
- Returns: list of {type, high, low, timestamp, filled_pct, size}
- Track mitigation: update filled_pct when price enters zone
```

---

### 4. Break of Structure (BOS)

**Definition**: Confirmation of trend continuation when price breaks a previous swing high (uptrend) or swing low (downtrend).

**Identification**:
```
Bullish BOS:
  - Price makes Higher High (HH)
  - Breaks above previous swing high
  - Confirms uptrend continuation

Bearish BOS:
  - Price makes Lower Low (LL)
  - Breaks below previous swing low
  - Confirms downtrend continuation
```

**Key Points**:
- BOS confirms the current trend is intact
- Multiple BOS in same direction = strong trend
- Look for entries on pullbacks after BOS

**Implementation Requirements**:
```
- detect_structure_breaks(ohlc_data, swing_lookback=5)
- Returns: list of {type: "BOS_UP"|"BOS_DOWN", price, timestamp, swing_broken}
- Track swing highs/lows using zigzag or pivot detection
```

---

### 5. Change of Character (CHoCH)

**Definition**: The **first sign** of a potential trend reversal - when price breaks structure in the opposite direction of the current trend.

**Identification**:
```
Bullish CHoCH (Reversal from downtrend to uptrend):
  - Market was making Lower Lows (LL) and Lower Highs (LH)
  - Price breaks above a recent Lower High
  - First Higher High (HH) signals CHoCH

Bearish CHoCH (Reversal from uptrend to downtrend):
  - Market was making Higher Highs (HH) and Higher Lows (HL)
  - Price breaks below a recent Higher Low
  - First Lower Low (LL) signals CHoCH
```

**Trading Significance**:
- CHoCH = Stop looking for trend continuation trades
- Wait for confirmation before trading the new direction
- High probability setup: CHoCH + return to order block

**Key Difference from BOS**:
| BOS | CHoCH |
|-----|-------|
| Continues current trend | Reverses current trend |
| Breaks with trend | Breaks against trend |
| Confirmation signal | Warning signal |

**Implementation Requirements**:
```
- detect_change_of_character(ohlc_data, trend_state)
- Returns: {type: "CHOCH_BULLISH"|"CHOCH_BEARISH", price, timestamp, strength}
- Requires tracking current trend state
```

---

### 6. Liquidity

**Definition**: Price levels where clusters of orders (especially stop losses) are waiting to be triggered.

**Types of Liquidity**:

| Type | Location | Description |
|------|----------|-------------|
| **Buy-Side Liquidity (BSL)** | Above swing highs | Stop losses from short sellers |
| **Sell-Side Liquidity (SSL)** | Below swing lows | Stop losses from long buyers |
| **Trendline Liquidity** | Along trendlines | Stops placed behind trendlines |
| **Equal Highs/Lows** | Double tops/bottoms | Obvious stop placement levels |

**Liquidity Concepts**:
- **Liquidity Sweep/Grab**: Price quickly spikes through liquidity zone then reverses
- **Liquidity Run**: Price targets liquidity before continuing trend
- **Internal Liquidity**: Minor swing points within a larger structure
- **External Liquidity**: Major swing highs/lows

**Trading Logic**:
1. Identify where liquidity pools exist
2. Wait for price to sweep (take) the liquidity
3. Enter after liquidity grab with reversal confirmation

**Implementation Requirements**:
```
- detect_liquidity_levels(ohlc_data, swing_lookback=20)
- Returns: list of {type: "BSL"|"SSL", price, strength, swept: bool}
- Track equal highs/lows as high-probability liquidity
- Detect liquidity sweeps (wick through level then close back)
```

---

## Trading Strategy

### Step 1: Determine the Trend (Market Structure)

**Process**:
1. Identify swing highs and swing lows
2. Determine if making HH/HL (uptrend) or LH/LL (downtrend)
3. Mark any CHoCH that signals potential reversal

**Rules**:
- **Uptrend**: Series of Higher Highs (HH) and Higher Lows (HL)
- **Downtrend**: Series of Lower Highs (LH) and Lower Lows (LL)
- **Ranging**: No clear HH/HL or LH/LL pattern

```
if trend == UPTREND:
    only_look_for = BUY setups
elif trend == DOWNTREND:
    only_look_for = SELL setups
```

### Step 2: Identify High-Probability Order Block

**Criteria for High-Probability OB**:
1. ✅ Caused a CHoCH or BOS
2. ✅ Has liquidity above/below (will be swept)
3. ✅ Has FVG in confluence
4. ✅ Fresh (not yet tested/mitigated)
5. ✅ Aligned with higher timeframe bias

**Confluence Scoring**:
```
score = 0
if caused_structure_break: score += 25
if has_nearby_fvg: score += 20
if has_liquidity_nearby: score += 20
if fresh_unmitigated: score += 20
if htf_aligned: score += 15

# 80+ = High probability
# 60-79 = Medium probability
# <60 = Low probability
```

### Step 3: Entry and Exit

**Entry Rules**:
- **Buy**: Place limit order at top of bullish OB (or 50% of zone)
- **Sell**: Place limit order at bottom of bearish OB (or 50% of zone)
- Wait for price to return to the zone

**Stop Loss**:
- **Buy**: Below the OB low (with small buffer)
- **Sell**: Above the OB high (with small buffer)

**Take Profit**:
- TP1: Previous structure point (swing high/low)
- TP2: Next liquidity target
- TP3: Measured move (1:2 or 1:3 RR minimum)

---

## Implementation Checklist

### Core Detection Functions

| Function | Status | Description |
|----------|--------|-------------|
| `detect_swing_points()` | ✅ | Find swing highs/lows |
| `detect_trend_structure()` | ✅ | Via BOS/CHoCH detection |
| `detect_bos()` | ✅ | Break of Structure (in `detect_structure_breaks()`) |
| `detect_choch()` | ✅ | Change of Character (in `detect_structure_breaks()`) |
| `detect_order_blocks()` | ✅ | Institutional zones + structural OB detection |
| `detect_breaker_blocks()` | ✅ | Failed OBs with polarity flip |
| `detect_fvg()` | ✅ | Fair Value Gaps (`detect_fair_value_gaps()`) |
| `detect_liquidity()` | ✅ | Liquidity pools (`detect_liquidity_zones()`) |
| `calculate_zone_strength()` | ✅ | Inline in OB detection with ATR/volume factors |

### Advanced Detection Functions (Bonus)

| Function | Status | Description |
|----------|--------|-------------|
| `detect_structural_order_blocks()` | ✅ | Multi-candle consolidation OBs |
| `detect_liquidity_sweeps()` | ✅ | Identify when liquidity was taken |
| `detect_inducements()` | ✅ | False breakout traps |
| `detect_rejection_blocks()` | ✅ | Failed OB tests with reversal |
| `detect_turtle_soup()` | ✅ | Linda Raschke reversal pattern |
| `detect_equal_levels()` | ✅ | Equal highs/lows (EQH/EQL) |
| `find_ote_zones()` | ✅ | Optimal Trade Entry (0.62-0.79 fib) |
| `calculate_premium_discount()` | ✅ | Position in range |
| `calculate_confluence_score()` | ✅ | Multi-factor zone quality |

### Multi-Timeframe Analysis

| Timeframe | Purpose |
|-----------|---------|
| HTF (D1/H4) | Trend direction, major zones |
| MTF (H1) | Entry zones, structure |
| LTF (M15/M5) | Precise entry timing |

### Confluence Requirements

For a valid trade setup:
- [ ] HTF trend alignment
- [ ] MTF structure break (BOS/CHoCH)
- [ ] Valid order block identified
- [ ] FVG in confluence (optional but increases probability)
- [ ] Liquidity target identified
- [ ] Risk:Reward minimum 1:2

---

## Zone Quality Scoring

### Strength Calculation Formula

```python
def calculate_zone_strength(zone, context):
    score = 0

    # Base score from impulse move
    impulse_strength = measure_impulse(zone.impulse_candles)
    score += impulse_strength * 0.25  # Max 25 points

    # Structure significance
    if zone.caused_bos:
        score += 20
    if zone.caused_choch:
        score += 25

    # Freshness
    if zone.test_count == 0:
        score += 20  # Untested
    elif zone.test_count == 1:
        score += 10  # Tested once
    else:
        score += 0   # Multiple tests (weak)

    # Confluence
    if zone.has_fvg_confluence:
        score += 15
    if zone.has_liquidity_nearby:
        score += 15

    # Higher timeframe alignment
    if zone.aligned_with_htf_bias:
        score += 15

    return min(score, 100)
```

### Strength Thresholds

| Score | Rating | Action |
|-------|--------|--------|
| 80-100 | Excellent | High confidence entry |
| 60-79 | Good | Standard entry |
| 40-59 | Moderate | Reduced size or skip |
| <40 | Weak | Avoid |

---

## Entry/Exit Rules

### Entry Criteria Checklist

Before taking a trade:

1. **Trend Confirmed**
   - [ ] HTF shows clear trend (HH/HL or LH/LL)
   - [ ] No recent CHoCH against trade direction

2. **Zone Validated**
   - [ ] Order block identified with score ≥60
   - [ ] Zone is fresh (untested or tested only once)
   - [ ] Zone caused a structure break

3. **Confluence Present**
   - [ ] FVG overlapping or nearby
   - [ ] Liquidity swept or about to be swept
   - [ ] HTF POI (Point of Interest) alignment

4. **Risk Management**
   - [ ] Stop loss below/above zone with buffer
   - [ ] Risk:Reward ≥ 1:2
   - [ ] Position size within risk limits

### Exit Rules

**Stop Loss Movement**:
- Move to breakeven after 1R profit
- Trail behind structure (swing points)
- Never widen stop loss

**Take Profit Levels**:
```
TP1 = Previous swing point (partial close: 50%)
TP2 = Next liquidity level (partial close: 30%)
TP3 = Final target or trailing (remaining: 20%)
```

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
| **HH** | Higher High |
| **HL** | Higher Low |
| **LH** | Lower High |
| **LL** | Lower Low |
| **POI** | Point of Interest - key zone for entries |
| **Mitigation** | When price returns to and fills a zone |
| **Sweep** | Quick move through liquidity then reversal |
| **Impulse** | Strong directional move |

---

## Implementation Status

### Current Implementation (`tradingagents/indicators/smart_money.py`)

The TradingAgents system has a **comprehensive SMC implementation** with 2,600+ lines covering:

**16 Detection Functions:**
1. Order Blocks (candle-based + structural multi-candle)
2. Fair Value Gaps with fill percentage tracking
3. BOS/CHoCH with proper trend context
4. Liquidity Zones with sweep detection
5. Equal Highs/Lows
6. Breaker Blocks
7. Premium/Discount Zones
8. OTE (Optimal Trade Entry) Zones
9. Session Analysis (Asian, London, NY kill zones)
10. Inducements (false breakouts)
11. Rejection Blocks
12. Turtle Soup patterns

**Quality Features:**
- Session-based strength multipliers
- Volume-adjusted strength calculation
- Mitigation and invalidation tracking
- Confluence scoring across multiple factors
- Multi-timeframe alignment support

### Gap Analysis

| Gap | Priority | Status | Description |
|-----|----------|--------|-------------|
| Regime Integration | HIGH | ✅ DONE | Zone strength adjusted by market regime |
| Zone Recency Weighting | MEDIUM | ✅ DONE | Time decay applied to older zones |
| Confluence Regime | MEDIUM | ✅ DONE | Confluence scoring includes regime alignment |
| Breaker Reactivation | LOW | ⬜ TODO | Breaker blocks aren't tracked for re-tests after sweep |
| Time-of-Day Filtering | LOW | ✅ EXISTS | Session multipliers already in place |

### Implemented Improvements (2026-02-02)

1. **Regime filtering in zone strength calculation (`get_regime_adjusted_strength()`):**
   - Trending-up: Bullish zones +20%, bearish zones -30%
   - Trending-down: Bearish zones +20%, bullish zones -30%
   - Ranging: FVGs +15%, Equal levels +20%

2. **Time-decay for zone strength (`get_time_decay_factor()`):**
   - Fresh zones (< 20 candles): 100% strength
   - Aging zones (20-50 candles): 75% strength
   - Old zones (50-100 candles): 50% strength
   - Very old zones (> 100 candles): 25% strength

3. **Enhanced confluence scoring (`calculate_confluence_score()`):**
   - Added `market_regime` and `trade_direction` parameters
   - +15 points when trade direction aligns with regime
   - Factors displayed in confluence breakdown

### New Methods Added

```python
# Adjust zone strength by market regime
analyzer.get_regime_adjusted_strength(
    base_strength=0.8,
    zone_type="bullish",
    market_regime="trending-up",
    zone_category="ob"
)  # Returns 0.96 (20% boost)

# Get time decay factor
analyzer.get_time_decay_factor(candles_since_formation=35)  # Returns 0.75

# Apply adjustments to zone list
analyzer.apply_zone_adjustments(
    zones=order_blocks,
    current_candle_index=150,
    market_regime="trending-up",
    zone_category="ob"
)

# Confluence with regime
analyzer.calculate_confluence_score(
    price=2800.0,
    analysis=full_analysis,
    market_regime="trending-up",
    trade_direction="bullish"
)
```

---

## References

- Source: HowToTrade SMC Trading Strategy PDF
- ICT (Inner Circle Trader) concepts
- Price action fundamentals

---

*Last Updated: 2026-02-02*
