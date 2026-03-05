"""
Test script to preview the Volume Profile Quant prompt.

Run: python scripts/test_vp_quant_prompt.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer, VolumeProfile


def build_vp_prompt(
    ticker: str,
    current_price: float,
    volume_profile: VolumeProfile,
    volume_profile_context: str,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    trading_session: str,
    current_date: str,
) -> str:
    """Build the complete VP quant prompt (standalone version)."""

    # Build data context
    data_context = f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price**: {current_price:.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}

## VOLUME PROFILE ANALYSIS
{volume_profile_context}

## TECHNICAL INDICATORS
{market_report}
"""

    # Full prompt template
    return f"""You are a systematic Volume Profile trader with strict risk discipline. You trade based on volume-at-price analysis to identify high-probability entries.

## VOLUME PROFILE TRADING RULES

### Core Concepts
1. **POC (Point of Control)**: The price level where most volume traded. Acts as a MAGNET - price tends to return here. Use as profit target in ranging markets.

2. **Value Area (VA)**: The price range containing 70% of volume. This is "fair value."
   - Price ABOVE VA = Premium, look for SHORTS (mean reversion down)
   - Price BELOW VA = Discount, look for LONGS (mean reversion up)
   - Price INSIDE VA = Fair value, wait for extremes or breakouts

3. **Value Area High (VAH)**: Upper boundary of value area. Acts as RESISTANCE.
4. **Value Area Low (VAL)**: Lower boundary of value area. Acts as SUPPORT.

5. **High Volume Nodes (HVN)**: Price levels with high volume.
   - Act as strong support/resistance
   - Good for stop loss placement (behind HVN)
   - Price tends to consolidate at HVN

6. **Low Volume Nodes (LVN)**: Price levels with low volume.
   - Price moves FAST through these zones
   - Avoid placing entries in LVN (will get run through)
   - Can be used to identify breakout acceleration zones

### Trading Setups

**Setup 1: Mean Reversion to POC**
- Price is significantly above/below POC
- Enter in direction of POC
- Target: POC level
- Stop: Beyond the extreme

**Setup 2: Value Area Edge Trade**
- Price touches VAH/VAL from inside
- Fade the move (short at VAH, long at VAL)
- Target: POC
- Stop: Beyond VAH/VAL by 1 ATR

**Setup 3: Value Area Breakout**
- Price closes outside VA with increasing volume
- Trade the breakout direction
- Target: Next HVN or previous day's POC
- Stop: Back inside VA

**Setup 4: HVN Bounce**
- Price approaches significant HVN
- Enter on rejection at HVN
- Target: Next HVN or POC
- Stop: Beyond the HVN

## RISK MANAGEMENT RULES (NEVER BREAK)

1. **Risk no more than 1-2% of account value per trade**
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and invalidation before entry**
5. **Place stops behind HVN levels when possible** (avoid LVN for stops)
6. **Fees are 0.025% maker / 0.05% taker + funding - size accordingly**
7. **Leverage is a tool, not a goal - default 5-20x**

## STOP LOSS VALIDATION (CRITICAL)
- For BUY orders: Stop loss MUST be BELOW entry price
- For SELL orders: Stop loss MUST be ABOVE entry price
- Prefer placing stops behind HVN (high volume = strong support/resistance)
- If you cannot identify a valid stop loss placement, output "hold"

{data_context}

## YOUR TASK

Analyze the Volume Profile data and make a systematic trading decision.

Think step-by-step:
1. Where is price relative to POC? (above/below/at)
2. Where is price relative to Value Area? (inside/above/below)
3. Are there nearby HVN levels for support/resistance?
4. Are there LVN zones that could accelerate price movement?
5. What Volume Profile setup applies? (mean reversion, VA edge, breakout, HVN bounce)
6. Where should stop loss be placed? (behind HVN preferred)
7. What's the risk:reward ratio? (must be >1.5:1)

## SIGNAL OPTIONS (you MUST pick one)
- **buy_to_enter** - Open a long position. Use when price is at VAL, below POC in discount, or bouncing off bullish HVN. MUST provide entry_price, stop_loss, and profit_target.
- **sell_to_enter** - Open a short position. Use when price is at VAH, above POC in premium, or rejecting at bearish HVN. MUST provide entry_price, stop_loss, and profit_target.
- **hold** - No action. Use when price is at fair value (inside VA near POC), no clear setup, or in LVN territory.
- **close** - Close existing position. Use when original thesis is invalidated.

Remember:
- Volume Profile shows WHERE institutional volume traded
- POC is a magnet - price tends to return
- Value Area edges are decision points
- HVN = consolidation/reversal zones
- LVN = acceleration zones
- Trade mean reversion inside VA, breakouts outside VA"""


def main():
    # Create realistic sample data
    print("Generating sample market data...")
    np.random.seed(123)
    n = 150

    # Simulate a market with uptrend -> consolidation -> continuation
    price = [5200]
    for i in range(1, n):
        if i < 50:
            price.append(price[-1] + np.random.randn() * 5 + 3)  # Uptrend
        elif i < 100:
            price.append(price[-1] + np.random.randn() * 8 - 0.5)  # Consolidation
        else:
            price.append(price[-1] + np.random.randn() * 6 + 2)  # Another push

    price = np.array(price)
    df = pd.DataFrame({
        'open': price - np.random.rand(n) * 5,
        'high': price + np.random.rand(n) * 15,
        'low': price - np.random.rand(n) * 15,
        'close': price,
        'volume': np.random.randint(5000, 25000, n),
        'tick_volume': np.random.randint(5000, 25000, n),
    })

    current_price = float(df['close'].iloc[-1])
    print(f"Current Price: {current_price:.2f}")

    # Calculate Volume Profile
    print("\nCalculating Volume Profile...")
    vp = VolumeProfileAnalyzer()
    profile = vp.calculate_volume_profile(df, num_bins=50, lookback=100)
    vp_context = vp.format_for_prompt(profile, current_price)

    print(f"POC: {profile.poc:.2f}")
    print(f"Value Area: {profile.value_area_low:.2f} - {profile.value_area_high:.2f}")
    print(f"HVN count: {len(profile.high_volume_nodes)}")
    print(f"LVN count: {len(profile.low_volume_nodes)}")

    # Build indicator context (simulated)
    indicators = f"""### Momentum
- **RSI(14)**: 58.2 (Neutral)
- **MACD**: 12.50 | Signal: 10.20
  - Bullish (MACD > Signal)

### Trend
- **EMA20**: {current_price - 15:.5f} (Price above)
- **EMA50**: {current_price - 40:.5f} (Price above)
- **ADX**: 32.5 (Strong trend)
- **Regime**: trending-up

### Volatility
- **ATR(14)**: 18.50000
- **Volatility**: normal

### Volume
- **Current Volume**: 18,500
- **Avg Volume (20)**: 15,200
- **Volume Ratio**: 1.22x average
"""

    # Build the full prompt
    full_prompt = build_vp_prompt(
        ticker='XAUUSD',
        current_price=current_price,
        volume_profile=profile,
        volume_profile_context=vp_context,
        market_report=indicators,
        market_regime='trending-up',
        volatility_regime='normal',
        trading_session='london_ny_overlap',
        current_date='2026-03-05',
    )

    print("\n" + "=" * 80)
    print("FULL VP QUANT PROMPT (what gets sent to LLM)")
    print("=" * 80)
    print(full_prompt)
    print("=" * 80)
    print(f"\nPrompt length: {len(full_prompt)} characters")
    print(f"Estimated tokens: ~{len(full_prompt) // 4}")

    # Save to file for easier review
    output_path = os.path.join(os.path.dirname(__file__), "vp_quant_prompt_preview.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_prompt)
    print(f"\nPrompt saved to: {output_path}")


if __name__ == "__main__":
    main()
