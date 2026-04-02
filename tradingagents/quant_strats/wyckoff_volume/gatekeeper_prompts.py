"""
Wyckoff Gatekeeper Prompts — system prompt and context snapshot builder.

The system prompt instructs the LLM as a Wyckoff volume-spread analyst.
The context builder formats computed features into a natural-language snapshot.
"""

from typing import Optional

import numpy as np
import pandas as pd


WYCKOFF_SYSTEM_PROMPT = """
You are an expert Wyckoff method analyst reviewing a potential trade signal on Gold (XAUUSD) Daily chart.

The XGBoost model generating this signal has a 60%+ historical win rate. Your job is NOT to second-guess the model's edge, but to catch the specific situations where Wyckoff volume-spread analysis reveals a structural problem the model cannot see. Only REJECT when there is clear, specific Wyckoff evidence against the trade.

## Critical: Trend Context Awareness

Before evaluating Wyckoff phase, determine the TREND using EMA Trend and returns data:
- EMA Trend > 0 AND returns_20 positive = UPTREND. In uptrends, price NORMALLY stays near range highs. This is markup, NOT distribution.
- EMA Trend < 0 AND returns_20 negative = DOWNTREND. In downtrends, price stays near range lows. This is markdown, NOT accumulation.
- Near-zero EMA trend with narrow range = RANGE-BOUND. Wyckoff phase analysis applies most here.

DO NOT confuse "price near range high in an uptrend" with distribution. In markup phases, buying near highs WITH trend is correct. Similarly, DO NOT confuse "price near range low in a downtrend" with accumulation.

## Analytical Framework

1. **Phase Identification** (consider trend context first!)
   - Accumulation: range-bound (NOT trending), volume drying up on declines, smart money absorbing supply
   - Markup: trending higher, volume expanding on up-bars, contracting on pullbacks. BUY signals in markup are WITH the trend.
   - Distribution: range-bound near highs (NOT just "near highs"), high volume with NO upward progress over multiple bars
   - Markdown: trending lower, volume expanding on down-bars. SELL signals in markdown are WITH the trend.

2. **Effort vs Result** — Does volume confirm price movement?
   - High volume + large range in signal direction = genuine strength/weakness
   - High volume + small range = absorption (potential reversal, but only meaningful in range-bound markets)
   - Low volume + price movement = can be normal in continuation; only suspicious at extremes

3. **Key Event Detection**
   - Spring (supports BUY): false break below range + recovery + volume
   - Upthrust (supports SELL): false break above range + rejection + volume
   - Sign of Strength: wide up-bar, closes near high, expanding volume (supports BUY in markup)
   - Sign of Weakness: wide down-bar, closes near low, expanding volume (supports SELL in markdown)

4. **Bar Quality** — Does the bar structure support the signal?
   - For BUY: close near top of bar, strong body, low upper wick
   - For SELL: close near bottom of bar, strong body, low lower wick

## Verdict Criteria

- **APPROVE**: Default when signal aligns with trend and no clear Wyckoff contradiction exists. With-trend signals should be APPROVED unless there is specific evidence of reversal (e.g., clear distribution pattern with multiple confirming bars, major volume divergence).
- **REJECT**: Only when Wyckoff evidence CLEARLY contradicts the signal. Examples: counter-trend signal without spring/upthrust confirmation, clear effort-result divergence at range extremes, signal in wrong phase with strong evidence.
- **HOLD**: One key Wyckoff condition needs next-bar confirmation. Specify what to look for.

## Counter-Trend Signal Scrutiny

SELL signals during an overall uptrend (or BUY during a downtrend) require EXTRA scrutiny. Short-term pullbacks within a larger trend are not Wyckoff phase changes:
- A 20-bar pullback within a 100-bar uptrend is NOT markdown — it is a potential accumulation/re-accumulation zone.
- SELL signals during a pullback in an uptrend should be REJECTED unless there is a clear Sign of Weakness with massive volume expansion AND breakdown of prior structure.
- Look at EMA Cross: if EMA fast > slow (Bullish), the larger trend is UP regardless of short-term returns. SELL signals should be heavily scrutinised.
- Similarly, if EMA Cross is Bearish, BUY signals on short-term bounces should be scrutinised.

## Approval Bias

Lean toward APPROVE for with-trend signals. The XGBoost model has a 60%+ win rate — only reject when Wyckoff evidence is unambiguous. For counter-trend signals, lean toward REJECT unless Wyckoff events (spring, upthrust, clear phase change) support the reversal.
""".strip()


def build_context_snapshot(
    wyckoff_row: pd.Series,
    technical_row: pd.Series,
    xgb_prob: float,
    direction: str,
    symbol: str = "XAUUSD",
    timeframe: str = "D1",
    htf_bias: Optional[str] = None,
) -> str:
    """
    Build a natural-language market context snapshot for the LLM gatekeeper.

    Args:
        wyckoff_row: Single row from WyckoffFeatures.compute() (iloc[-1])
        technical_row: Single row from TechnicalFeatures.compute() (iloc[-1])
        xgb_prob: XGBoost probability score (0-1)
        direction: "BUY" or "SELL"
        symbol: Trading symbol
        timeframe: Chart timeframe
        htf_bias: Higher timeframe bias ("bullish", "bearish", "neutral")
    """

    def _fmt(val, decimals=2):
        """Format a value, handling NaN."""
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return "N/A"
        if isinstance(val, (float, np.floating)):
            return f"{val:.{decimals}f}"
        return str(val)

    def _fmt_pct(val):
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return "N/A"
        return f"{val:.2%}"

    def _fmt_bool(val):
        if isinstance(val, (float, np.floating)):
            return "Yes" if val >= 0.5 else "No"
        return str(bool(val))

    # Extract values with safe fallback
    def _get(row, key, default=np.nan):
        return row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)

    # Get close from technical features if available
    close = _get(technical_row, "close_position", np.nan)

    htf_line = f"- Higher Timeframe Bias: {htf_bias}" if htf_bias else "- Higher Timeframe Bias: Not available"

    return f"""
{symbol} {timeframe} - Wyckoff Signal Review
===================================

XGBoost Signal:
- Direction: {direction}
- Probability Score: {_fmt_pct(xgb_prob)}

Price Structure:
- ATR (14): {_fmt(_get(technical_row, "atr_14"))}
- ATR % of Price: {_fmt_pct(_get(technical_row, "atr_pct"))}
- Bar Range Ratio (vs ATR): {_fmt(_get(wyckoff_row, "bar_range_ratio"))}x
- Close Position in Bar: {_fmt(_get(wyckoff_row, "body_ratio"))}  (0=doji, 1=full body)
- Body Ratio: {_fmt(_get(wyckoff_row, "body_ratio"))}
- Upper Wick Ratio: {_fmt(_get(wyckoff_row, "upper_wick_ratio"))}
- Lower Wick Ratio: {_fmt(_get(wyckoff_row, "lower_wick_ratio"))}

Volume Analysis:
- Volume vs 20-bar Average: {_fmt(_get(wyckoff_row, "volume_spike"))}x
- Volume Spike Detected: {_fmt_bool(1.0 if _get(wyckoff_row, "volume_spike", 0) > 2.0 else 0.0)}
- Directional Volume Delta: {_fmt(_get(wyckoff_row, "volume_delta"))}
- Up-Volume Ratio (10 bars): {_fmt(_get(wyckoff_row, "up_volume_ratio"))}
- Effort vs Result: {_fmt(_get(wyckoff_row, "effort_vs_result"))}
- Cumulative Volume Delta (10 bars): {_fmt(_get(wyckoff_row, "cumulative_volume_delta"))}

Range Context:
- 20-bar Range Width (ATR multiples): {_fmt(_get(wyckoff_row, "range_width"))}
- Price Position in Range: {_fmt_pct(_get(wyckoff_row, "price_in_range"))}  (0=bottom, 1=top)
- Range Contraction Ratio: {_fmt(_get(wyckoff_row, "range_contraction"))}
- Spring Signal: {_fmt_bool(_get(wyckoff_row, "spring_signal"))}
- Upthrust Signal: {_fmt_bool(_get(wyckoff_row, "upthrust_signal"))}

Trend Context:
- EMA Trend (fast vs slow, ATR-normalised): {_fmt(_get(technical_row, "ema_short_dist"), 4)}  (positive=price above fast EMA=bullish)
- EMA Cross (fast > slow): {"Bullish" if _get(technical_row, "ema_cross", 0) > 0.5 else "Bearish"}
- 5-bar Return: {_fmt_pct(_get(technical_row, "returns_5"))}
- 20-bar Return: {_fmt_pct(_get(technical_row, "returns_20"))}
- RSI (14): {_fmt(_get(technical_row, "rsi_14"), 1)}
- MACD Histogram (ATR-norm): {_fmt(_get(technical_row, "macd_hist_norm"), 4)}
- Bollinger Band %: {_fmt(_get(technical_row, "bb_pct"))}
- ADX (14): {_fmt(_get(technical_row, "adx_14"), 1)}  (>25 = trending)
{htf_line}
""".strip()
