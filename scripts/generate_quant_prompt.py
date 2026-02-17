"""
Generate Quant Prompt for Manual LLM Testing

This script generates the exact prompt that would be sent to the LLM in quant analysis mode,
and saves it to a text file so you can copy/paste into Grok or any other LLM.

Usage:
    python scripts/generate_quant_prompt.py XAUUSD H1
    python scripts/generate_quant_prompt.py BTCUSD H4 --output my_prompt.txt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_quant_prompt(symbol: str, timeframe: str = "H1") -> tuple[str, dict]:
    """
    Generate the quant analysis prompt for a given symbol.

    Returns:
        tuple: (full_prompt, context_dict)
    """
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np

    if not mt5.initialize():
        raise RuntimeError("MT5 not initialized. Make sure MetaTrader 5 is running.")

    # Get symbol info for current price
    symbol_info = mt5.symbol_info_tick(symbol)
    if symbol_info is None:
        raise ValueError(f"Symbol {symbol} not found in MT5")

    current_price = (symbol_info.bid + symbol_info.ask) / 2
    bid = symbol_info.bid
    ask = symbol_info.ask

    # Map timeframe
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
    }
    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

    # Get price data
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to get price data from MT5")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # === Calculate Technical Indicators ===

    # ATR
    high_low = df['high'] - df['low']
    atr = high_low.rolling(14).mean().iloc[-1]

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.0001)
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_value = macd_line.iloc[-1]
    macd_signal = signal_line.iloc[-1]
    macd_histogram = macd_hist.iloc[-1]

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    bb_upper_val = bb_upper.iloc[-1]
    bb_lower_val = bb_lower.iloc[-1]
    bb_middle_val = sma20.iloc[-1]

    # EMA
    ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

    # ADX for trend strength
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = high_low.copy()
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(14).mean().iloc[-1]

    # Market regime
    avg_atr = high_low.rolling(50).mean().iloc[-1]
    volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

    if adx > 25:
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            market_regime = "trending-up"
        else:
            market_regime = "trending-down"
    else:
        market_regime = "ranging"

    # === Volume Analysis ===
    # MT5 provides tick_volume (number of ticks) - use this for forex
    volume_col = 'tick_volume' if 'tick_volume' in df.columns else 'real_volume'
    if volume_col in df.columns:
        current_volume = df[volume_col].iloc[-1]
        avg_volume_20 = df[volume_col].rolling(20).mean().iloc[-1]
        avg_volume_50 = df[volume_col].rolling(50).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # Volume trend (last 5 bars)
        recent_volumes = df[volume_col].iloc[-5:]
        volume_trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"

        # Volume spike detection (>1.5x average)
        volume_spike = current_volume > avg_volume_20 * 1.5

        # Volume profile description
        if volume_ratio > 2.0:
            volume_profile = "Very High (spike)"
        elif volume_ratio > 1.5:
            volume_profile = "High"
        elif volume_ratio > 0.8:
            volume_profile = "Normal"
        elif volume_ratio > 0.5:
            volume_profile = "Low"
        else:
            volume_profile = "Very Low"
    else:
        current_volume = 0
        avg_volume_20 = 0
        avg_volume_50 = 0
        volume_ratio = 1.0
        volume_trend = "unknown"
        volume_spike = False
        volume_profile = "N/A"

    # === Run SMC Analysis ===
    from tradingagents.indicators.smart_money import SmartMoneyAnalyzer

    analyzer = SmartMoneyAnalyzer()
    smc_result = analyzer.analyze_full_smc(
        df,
        current_price=current_price,
        use_structural_obs=True
    )

    # Format SMC context
    smc_context = _format_smc_for_prompt(smc_result, current_price, atr)

    # === Build Indicator Context ===
    indicators_context = f"""## Technical Indicators

### Momentum
- **RSI(14)**: {rsi_value:.1f} {"(Overbought)" if rsi_value > 70 else "(Oversold)" if rsi_value < 30 else "(Neutral)"}
- **MACD**: {macd_value:.5f} | Signal: {macd_signal:.5f} | Histogram: {macd_histogram:.5f}
  - {"Bullish (MACD > Signal)" if macd_value > macd_signal else "Bearish (MACD < Signal)"}

### Trend
- **EMA20**: {ema20:.5f} {"(Price above)" if current_price > ema20 else "(Price below)"}
- **EMA50**: {ema50:.5f} {"(Price above)" if current_price > ema50 else "(Price below)"}
- **ADX**: {adx:.1f} {"(Strong trend)" if adx > 25 else "(Weak/No trend)"}
- **Regime**: {market_regime}

### Volatility
- **ATR(14)**: {atr:.5f}
- **Volatility**: {volatility_regime}
- **Bollinger Bands**: Upper={bb_upper_val:.5f} | Middle={bb_middle_val:.5f} | Lower={bb_lower_val:.5f}
  - {"Price near upper band" if current_price > bb_upper_val * 0.99 else "Price near lower band" if current_price < bb_lower_val * 1.01 else "Price within bands"}

### Volume
- **Current Bar Volume**: {current_volume:,.0f} ticks
- **20-bar Avg Volume**: {avg_volume_20:,.0f} ticks
- **Volume Ratio**: {volume_ratio:.2f}x average {"⚠️ SPIKE" if volume_spike else ""}
- **Volume Profile**: {volume_profile}
- **Volume Trend (5 bars)**: {volume_trend}
"""

    # Get trading session
    trading_session = _get_trading_session()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # === Build Data Context ===
    data_context = f"""## CURRENT MARKET DATA
- **Symbol**: {symbol}
- **Current Price (Broker)**: {current_price:.5f}
- **Bid**: {bid:.5f} | **Ask**: {ask:.5f}
- **Spread**: {(ask - bid):.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}

## SMART MONEY CONCEPTS (SMC) ANALYSIS
{smc_context}

{indicators_context}
"""

    # === Build Full Prompt ===
    full_prompt = f"""You are a systematic quant trader with strict risk discipline. Your only goal is to maximize long-term PnL while surviving drawdowns.

## RULES YOU MUST NEVER BREAK

1. **Risk no more than 1-2% of account value per trade** (risk_usd = position size x distance to stop-loss)
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and one clear invalidation condition before entry**
5. **Hold only when current plan remains valid; never flip-flop without new high-conviction signal**
6. **Ignore narratives; trade only price, volume, and the provided technical indicators**
7. **Fees are 0.025% maker / 0.05% taker + funding - size accordingly**
8. **Leverage is a tool, not a goal - default 5-20x, higher only with extreme conviction**

## STOP LOSS VALIDATION (CRITICAL)
- For BUY orders: Stop loss MUST be BELOW entry price
- For SELL orders: Stop loss MUST be ABOVE entry price
- If you cannot identify a valid stop loss placement, output "hold"

## CURRENT MARKET DATA
{data_context}

## YOUR TASK

Analyze the provided technical data and make a systematic trading decision.

Think step-by-step:
1. What is the overall market structure and bias?
2. Are there high-probability entry zones (OB, FVG) near current price?
3. Where would stop loss be placed? Is the R:R acceptable (>1.5:1)?
4. What would invalidate this setup?
5. What is your confidence level based on confluence?

Then provide your decision in this format:

**SIGNAL**: [BUY / SELL / HOLD]
**CONFIDENCE**: [0-100%]
**ENTRY PRICE**: [price or "market"]
**STOP LOSS**: [price]
**TAKE PROFIT**: [price]
**RISK/REWARD**: [ratio like 1:2.5]
**JUSTIFICATION**: [2-3 sentences explaining the trade thesis]
**INVALIDATION**: [One clear condition that would invalidate this trade]

Remember:
- Only enter trades with clear edge
- Wait for price to come to your levels
- No FOMO entries
- Discipline > prediction"""

    context = {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": current_price,
        "bid": bid,
        "ask": ask,
        "market_regime": market_regime,
        "volatility_regime": volatility_regime,
        "trading_session": trading_session,
        "rsi": rsi_value,
        "adx": adx,
        "atr": atr,
        "volume": current_volume,
        "volume_avg": avg_volume_20,
        "volume_ratio": volume_ratio,
        "volume_profile": volume_profile,
        "volume_spike": volume_spike,
    }

    return full_prompt, context


def _format_smc_for_prompt(smc_result: dict, current_price: float, atr: float) -> str:
    """Format SMC analysis results for the prompt."""
    lines = []

    # Helper to safely get attributes
    def safe_get(obj, attr, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    # Bias
    bias = smc_result.get("bias", "neutral")
    lines.append(f"**Overall Bias**: {bias.upper()}")
    lines.append("")

    # Order Blocks
    obs = smc_result.get("order_blocks", [])
    if obs:
        lines.append("### Order Blocks (Institutional Entry Zones)")
        count = 0
        for ob in obs:
            if count >= 5:
                break
            mitigated = safe_get(ob, "mitigated", False)
            if not mitigated:
                ob_type = safe_get(ob, "type", "unknown")
                top = safe_get(ob, "top", 0)
                bottom = safe_get(ob, "bottom", 0)
                strength = safe_get(ob, "strength", 0.5)
                if top and bottom:
                    distance = abs(current_price - (top + bottom) / 2)
                    distance_atr = distance / atr if atr > 0 else 0
                    lines.append(f"- **{ob_type.upper()} OB**: {bottom:.5f} - {top:.5f} (strength: {strength:.0%}, {distance_atr:.1f} ATR away)")
                    count += 1
        lines.append("")

    # Fair Value Gaps
    fvgs = smc_result.get("fair_value_gaps", [])
    if fvgs:
        lines.append("### Fair Value Gaps (Price Imbalances)")
        count = 0
        for fvg in fvgs:
            if count >= 4:
                break
            mitigated = safe_get(fvg, "mitigated", False)
            if not mitigated:
                fvg_type = safe_get(fvg, "type", "unknown")
                top = safe_get(fvg, "top", 0)
                bottom = safe_get(fvg, "bottom", 0)
                if top and bottom:
                    distance = abs(current_price - (top + bottom) / 2)
                    distance_atr = distance / atr if atr > 0 else 0
                    lines.append(f"- **{fvg_type.upper()} FVG**: {bottom:.5f} - {top:.5f} ({distance_atr:.1f} ATR away)")
                    count += 1
        lines.append("")

    # Liquidity Zones
    liq = smc_result.get("liquidity_zones", [])
    if liq:
        lines.append("### Liquidity Zones (Stop Hunt Targets)")
        for lz in liq[:4]:
            lz_price = safe_get(lz, "price", 0)
            lz_type = safe_get(lz, "type", "unknown")
            lz_strength = safe_get(lz, "strength", 50)
            touched = safe_get(lz, "touched", False)
            if lz_price:
                status = "(SWEPT)" if touched else "(unswept)"
                lines.append(f"- **{lz_type.upper()}**: {lz_price:.5f} (strength: {lz_strength}%) {status}")
        lines.append("")

    # Key Levels
    nearest_support = smc_result.get("nearest_support")
    nearest_resistance = smc_result.get("nearest_resistance")
    if nearest_support or nearest_resistance:
        lines.append("### Key Levels")
        if nearest_support:
            s_top = safe_get(nearest_support, "top", 0)
            s_bottom = safe_get(nearest_support, "bottom", 0)
            if s_top or s_bottom:
                lines.append(f"- **Nearest Support**: {s_bottom:.5f} - {s_top:.5f}")
        if nearest_resistance:
            r_top = safe_get(nearest_resistance, "top", 0)
            r_bottom = safe_get(nearest_resistance, "bottom", 0)
            if r_top or r_bottom:
                lines.append(f"- **Nearest Resistance**: {r_bottom:.5f} - {r_top:.5f}")
        lines.append("")

    # ATR
    if atr:
        lines.append(f"**ATR(14)**: {atr:.5f}")

    return "\n".join(lines)


def _get_trading_session() -> str:
    """Determine current trading session."""
    from datetime import datetime
    import pytz

    try:
        utc_now = datetime.now(pytz.UTC)
        hour = utc_now.hour

        if 22 <= hour or hour < 7:
            return "Asian (Sydney/Tokyo)"
        elif 7 <= hour < 8:
            return "Asian/London Overlap"
        elif 8 <= hour < 12:
            return "London"
        elif 12 <= hour < 13:
            return "London/New York Overlap"
        elif 13 <= hour < 17:
            return "New York"
        elif 17 <= hour < 22:
            return "New York Close / Asian Open"
        else:
            return "Unknown"
    except:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Generate quant prompt for manual LLM testing")
    parser.add_argument("symbol", help="Trading symbol (e.g., XAUUSD, BTCUSD)")
    parser.add_argument("timeframe", nargs="?", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: prompts/<symbol>_<timestamp>.txt)")

    args = parser.parse_args()

    print(f"Generating quant prompt for {args.symbol} on {args.timeframe}...")

    try:
        prompt, context = generate_quant_prompt(args.symbol, args.timeframe)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            prompts_dir = PROJECT_ROOT / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = prompts_dir / f"{args.symbol}_{args.timeframe}_{timestamp}.txt"

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"\nPrompt saved to: {output_path}")
        print(f"\nContext Summary:")
        print(f"  Symbol: {context['symbol']}")
        print(f"  Price: {context['current_price']:.5f}")
        print(f"  Regime: {context['market_regime']} ({context['volatility_regime']} volatility)")
        print(f"  RSI: {context['rsi']:.1f}")
        print(f"  ADX: {context['adx']:.1f}")
        print(f"  Volume: {context['volume_profile']} ({context['volume_ratio']:.2f}x avg){' ⚠️ SPIKE' if context['volume_spike'] else ''}")
        print(f"  Session: {context['trading_session']}")
        print(f"\nPrompt length: {len(prompt)} characters")
        print(f"\nYou can now copy the contents of {output_path} and paste into Grok or any LLM.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
