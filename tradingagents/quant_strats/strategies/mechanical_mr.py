"""
Mechanical Mean Reversion Strategies

Simple, rule-based mean reversion strategies that avoid ML overfitting entirely.
These use fixed indicator thresholds — no XGBoost model needed.

Strategies:
1. BB Bounce: Enter when price touches/pierces Bollinger Band and reverses
2. Ratio Z-Score: Enter when cross-pair z-score hits extremes (e.g. gold/silver ratio)

Both strategies require the market to be confirmed ranging (same regime gate
as the XGBoost MR strategy) and only trade at range extremes.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MechanicalSignal:
    """Signal from a mechanical strategy — no ML confidence, just triggered or not."""
    direction: str          # "BUY", "SELL", or "HOLD"
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    rationale: str = ""
    strategy: str = ""


# ---------------------------------------------------------------------------
# Strategy 1: Bollinger Band Bounce
# ---------------------------------------------------------------------------

def bb_bounce_signal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    atr_period: int = 14,
    require_reversal: bool = True,
    structural_bias: str = "neutral",
    rsi_filter: bool = True,
    rsi_period: int = 14,
) -> MechanicalSignal:
    """
    Bollinger Band Bounce — enter when price touches a band and reverses.

    BUY: low pierces lower band AND current bar closes above it (wick rejection)
         AND RSI < 35 (oversold confirmation, prevents catching falling knives).
    SELL: high pierces upper band AND current bar closes below it
         AND RSI > 65 (overbought confirmation).

    Structural bias filter:
    - Bullish bias: only BUY signals allowed (don't fade the uptrend)
    - Bearish bias: only SELL signals allowed (don't fade the downtrend)
    - Neutral: both directions valid

    Args:
        high, low, close: Price arrays (need at least bb_period + 10 bars)
        bb_period: BB SMA lookback
        bb_std: Number of standard deviations for bands
        sl_atr_mult: Stop loss distance in ATR multiples
        tp_atr_mult: Take profit distance in ATR multiples
        atr_period: ATR lookback
        require_reversal: If True, require the candle to close back inside the band
        structural_bias: "bullish", "bearish", or "neutral" from higher TF
        rsi_filter: If True, require RSI confirmation (oversold for BUY, overbought for SELL)
        rsi_period: RSI lookback period
    """
    n = len(close)
    if n < max(bb_period, atr_period, rsi_period) + 5:
        return MechanicalSignal(direction="HOLD", rationale="Not enough data", strategy="bb_bounce")

    cs = pd.Series(close)
    sma = cs.rolling(bb_period).mean().values
    std = cs.rolling(bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std

    # ATR for SL/TP
    atr = _compute_atr_simple(high, low, close, atr_period)
    current_atr = atr[-1]
    if np.isnan(current_atr) or current_atr < 1e-10:
        return MechanicalSignal(direction="HOLD", rationale="ATR unavailable", strategy="bb_bounce")

    # RSI for confirmation
    current_rsi = np.nan
    if rsi_filter:
        delta = cs.diff()
        gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi_values = (100 - (100 / (1 + rs))).values
        current_rsi = rsi_values[-1]

    curr_close = close[-1]
    curr_low = low[-1]
    curr_high = high[-1]
    curr_lower = lower[-1]
    curr_upper = upper[-1]

    if np.isnan(curr_lower) or np.isnan(curr_upper):
        return MechanicalSignal(direction="HOLD", rationale="BB not ready", strategy="bb_bounce")

    # BUY: low pierced lower band, close back inside (rejection wick)
    if curr_low <= curr_lower and structural_bias != "bearish":
        reversal_ok = not require_reversal or curr_close > curr_lower
        rsi_ok = not rsi_filter or (not np.isnan(current_rsi) and current_rsi < 35)
        if reversal_ok and rsi_ok:
            entry = curr_close
            sl = entry - current_atr * sl_atr_mult
            tp = entry + current_atr * tp_atr_mult
            rsi_str = f", RSI={current_rsi:.1f}" if not np.isnan(current_rsi) else ""
            return MechanicalSignal(
                direction="BUY", entry=entry, stop_loss=sl, take_profit=tp,
                rationale=(
                    f"BB Bounce BUY: low={curr_low:.5f} pierced lower band={curr_lower:.5f}, "
                    f"close={curr_close:.5f} back inside{rsi_str}. "
                    f"Bias={structural_bias}. SL={sl:.5f}, TP={tp:.5f}"
                ),
                strategy="bb_bounce",
            )
        elif reversal_ok and not rsi_ok:
            return MechanicalSignal(
                direction="HOLD",
                rationale=f"BB lower touch but RSI={current_rsi:.1f} not oversold (<35) — no confirmation",
                strategy="bb_bounce",
            )

    # SELL: high pierced upper band, close back inside
    if curr_high >= curr_upper and structural_bias != "bullish":
        reversal_ok = not require_reversal or curr_close < curr_upper
        rsi_ok = not rsi_filter or (not np.isnan(current_rsi) and current_rsi > 65)
        if reversal_ok and rsi_ok:
            entry = curr_close
            sl = entry + current_atr * sl_atr_mult
            tp = entry - current_atr * tp_atr_mult
            rsi_str = f", RSI={current_rsi:.1f}" if not np.isnan(current_rsi) else ""
            return MechanicalSignal(
                direction="SELL", entry=entry, stop_loss=sl, take_profit=tp,
                rationale=(
                    f"BB Bounce SELL: high={curr_high:.5f} pierced upper band={curr_upper:.5f}, "
                    f"close={curr_close:.5f} back inside{rsi_str}. "
                    f"Bias={structural_bias}. SL={sl:.5f}, TP={tp:.5f}"
                ),
                strategy="bb_bounce",
            )
        elif reversal_ok and not rsi_ok:
            return MechanicalSignal(
                direction="HOLD",
                rationale=f"BB upper touch but RSI={current_rsi:.1f} not overbought (>65) — no confirmation",
                strategy="bb_bounce",
            )

    return MechanicalSignal(
        direction="HOLD",
        rationale=f"No BB touch: close={curr_close:.5f}, lower={curr_lower:.5f}, upper={curr_upper:.5f}",
        strategy="bb_bounce",
    )


# ---------------------------------------------------------------------------
# Strategy 2: Ratio Z-Score Mean Reversion
# ---------------------------------------------------------------------------

def ratio_zscore_signal(
    close_a: np.ndarray,
    close_b: np.ndarray,
    zscore_period: int = 50,
    entry_threshold: float = 2.0,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 2.5,
    high_a: Optional[np.ndarray] = None,
    low_a: Optional[np.ndarray] = None,
    atr_period: int = 14,
) -> MechanicalSignal:
    """
    Ratio Z-Score — trade when the ratio between two correlated assets
    hits z-score extremes (e.g. gold/silver ratio, AUDUSD/NZDUSD).

    BUY asset A: when ratio z-score < -entry_threshold (A is cheap vs B)
    SELL asset A: when ratio z-score > +entry_threshold (A is expensive vs B)

    Args:
        close_a: Close prices of the asset to trade
        close_b: Close prices of the reference asset (same length)
        zscore_period: Lookback for mean/std of ratio
        entry_threshold: Z-score magnitude to trigger entry (default 2.0)
        sl_atr_mult: Stop loss in ATR multiples of asset A
        tp_atr_mult: Take profit in ATR multiples of asset A
        high_a, low_a: For ATR calculation on asset A
        atr_period: ATR lookback
    """
    n = min(len(close_a), len(close_b))
    if n < zscore_period + 5:
        return MechanicalSignal(direction="HOLD", rationale="Not enough data", strategy="ratio_zscore")

    # Trim to same length
    close_a = close_a[-n:]
    close_b = close_b[-n:]

    # Compute ratio and z-score
    ratio = close_a / (close_b + 1e-10)
    rs = pd.Series(ratio)
    ratio_mean = rs.rolling(zscore_period).mean().values
    ratio_std = rs.rolling(zscore_period).std().values
    zscore = (ratio - ratio_mean) / (ratio_std + 1e-10)

    current_z = zscore[-1]
    if np.isnan(current_z):
        return MechanicalSignal(direction="HOLD", rationale="Z-score NaN", strategy="ratio_zscore")

    # ATR for SL/TP on asset A
    if high_a is not None and low_a is not None:
        atr = _compute_atr_simple(high_a[-n:], low_a[-n:], close_a, atr_period)
    else:
        # Fallback: use close-to-close volatility
        returns = np.abs(np.diff(close_a))
        atr = np.full(n, np.nan)
        if len(returns) >= atr_period:
            atr[atr_period:] = pd.Series(returns).rolling(atr_period).mean().values[atr_period - 1:]

    current_atr = atr[-1]
    if np.isnan(current_atr) or current_atr < 1e-10:
        return MechanicalSignal(direction="HOLD", rationale="ATR unavailable", strategy="ratio_zscore")

    entry = float(close_a[-1])

    # BUY: ratio z-score deeply negative → A is cheap relative to B
    if current_z <= -entry_threshold:
        sl = entry - current_atr * sl_atr_mult
        tp = entry + current_atr * tp_atr_mult
        return MechanicalSignal(
            direction="BUY", entry=entry, stop_loss=sl, take_profit=tp,
            rationale=(
                f"Ratio Z-Score BUY: z={current_z:.2f} <= -{entry_threshold} "
                f"(A cheap vs B). Ratio={ratio[-1]:.4f}, "
                f"mean={ratio_mean[-1]:.4f}. SL={sl:.5f}, TP={tp:.5f}"
            ),
            strategy="ratio_zscore",
        )

    # SELL: ratio z-score deeply positive → A is expensive relative to B
    if current_z >= entry_threshold:
        sl = entry + current_atr * sl_atr_mult
        tp = entry - current_atr * tp_atr_mult
        return MechanicalSignal(
            direction="SELL", entry=entry, stop_loss=sl, take_profit=tp,
            rationale=(
                f"Ratio Z-Score SELL: z={current_z:.2f} >= +{entry_threshold} "
                f"(A expensive vs B). Ratio={ratio[-1]:.4f}, "
                f"mean={ratio_mean[-1]:.4f}. SL={sl:.5f}, TP={tp:.5f}"
            ),
            strategy="ratio_zscore",
        )

    return MechanicalSignal(
        direction="HOLD",
        rationale=f"Z-score={current_z:.2f} within ±{entry_threshold} band",
        strategy="ratio_zscore",
    )


# ---------------------------------------------------------------------------
# Predefined ratio pairs for ratio_zscore_signal
# ---------------------------------------------------------------------------

RATIO_PAIRS: Dict[str, str] = {
    # Asset A → reference asset B
    "XAUUSD": "XAGUSD",      # Gold / Silver ratio
    "AUDUSD": "NZDUSD",      # AUD / NZD (highly correlated)
    "EURUSD": "GBPUSD",      # EUR / GBP
    "USDCAD": "USDCHF",      # CAD / CHF (commodity dollars)
}


# ---------------------------------------------------------------------------
# Combined mechanical MR entry — checks both strategies with regime gate
# ---------------------------------------------------------------------------

def mechanical_mr_signal(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    symbol: str = "",
    ref_close: Optional[np.ndarray] = None,
    bb_params: Optional[Dict[str, Any]] = None,
    ratio_params: Optional[Dict[str, Any]] = None,
) -> MechanicalSignal:
    """
    Combined mechanical MR — runs regime gate, then checks BB bounce and
    optionally ratio z-score.

    Returns the first triggered signal, or HOLD if neither fires.
    """
    from tradingagents.agents.analysts.range_quant import analyze_range
    from tradingagents.quant_strats.strategies.mean_reversion import MR_EXCLUDED_PAIRS

    if symbol in MR_EXCLUDED_PAIRS:
        return MechanicalSignal(
            direction="HOLD",
            rationale=f"{symbol} excluded from mean reversion",
            strategy="mechanical_mr",
        )

    # Regime gate
    gate = analyze_range(high, low, close)
    if not gate.get("is_ranging", False):
        return MechanicalSignal(
            direction="HOLD",
            rationale=f"Regime gate: not ranging (mr_score={gate.get('mean_reversion_score', 0):.0f})",
            strategy="mechanical_mr",
        )

    # 1. BB Bounce (pass structural bias from range analysis)
    bb_kw = bb_params or {}
    bb_kw.setdefault("structural_bias", gate.get("structural_bias", "neutral"))
    bb = bb_bounce_signal(high, low, close, **bb_kw)
    if bb.direction != "HOLD":
        bb.rationale = f"[Range confirmed, MR={gate['mean_reversion_score']:.0f}] {bb.rationale}"
        return bb

    # 2. Ratio Z-Score (if reference asset provided)
    if ref_close is not None and len(ref_close) >= 50:
        ratio_kw = ratio_params or {}
        rz = ratio_zscore_signal(close, ref_close, high_a=high, low_a=low, **ratio_kw)
        if rz.direction != "HOLD":
            rz.rationale = f"[Range confirmed, MR={gate['mean_reversion_score']:.0f}] {rz.rationale}"
            return rz

    return MechanicalSignal(
        direction="HOLD",
        rationale=f"Range confirmed (MR={gate['mean_reversion_score']:.0f}) but no BB touch or z-score extreme",
        strategy="mechanical_mr",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_atr_simple(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Simple ATR calculation."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr
