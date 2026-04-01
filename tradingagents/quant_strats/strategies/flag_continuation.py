"""
Flag / Pennant Continuation Strategy

Detects strong impulse move (the "pole") followed by consolidation
(the "flag"), then enters on breakout from consolidation in the
original trend direction.

Pattern:  Strong move  →  Tight consolidation  →  Continuation breakout

REGIME GATE: Only fires in trending markets (efficiency-ratio ADX > 25).
DIRECTION LOCK: Only enters in the direction of the prior impulse.

Best suited for: trending instruments (XAUUSD, XAGUSD, GBPJPY, BTCUSD).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.xgb_quant.strategies.base import BaseStrategy, Signal
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.config import FeatureWindows, RiskDefaults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (imported from config once added, but defined here as fallback)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "impulse_lookback": 20,
    "impulse_min_atr_mult": 3.0,
    "consolidation_min_bars": 5,
    "consolidation_max_bars": 20,
    "consolidation_retrace_max": 0.50,
    "range_contraction_pct": 0.60,
    "vol_decline_threshold": 0.80,
    "breakout_atr_mult": 0.5,
    "adx_threshold": 25,
    "sl_atr_mult": 1.5,
    "rr_target": 2.5,
    "atr_period": 14,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _efficiency_ratio(close: np.ndarray, lookback: int) -> np.ndarray:
    """Efficiency ratio (|displacement| / path) as ADX proxy, scaled 0-50."""
    n = len(close)
    er = np.full(n, np.nan)
    for i in range(lookback, n):
        displacement = abs(close[i] - close[i - lookback])
        path = np.sum(np.abs(np.diff(close[i - lookback: i + 1])))
        er[i] = (displacement / (path + 1e-10)) * 50.0
    return er


def _impulse_detection(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    lookback: int,
) -> dict:
    """
    For each bar, detect the prior impulse move over `lookback` bars.

    Returns dict of arrays:
      - impulse_size_atr: |displacement| / ATR
      - impulse_direction: +1 bullish, -1 bearish, 0 none
      - impulse_linearity: efficiency ratio of the impulse (0-1)
      - impulse_body_ratio: avg |body|/|range| over impulse window
    """
    n = len(close)
    size_atr = np.full(n, np.nan)
    direction = np.zeros(n)
    linearity = np.full(n, np.nan)
    body_ratio = np.full(n, np.nan)

    for i in range(lookback, n):
        seg_close = close[i - lookback: i + 1]
        displacement = seg_close[-1] - seg_close[0]
        path = np.sum(np.abs(np.diff(seg_close)))

        a = atr[i] if not np.isnan(atr[i]) else 1e-10
        size_atr[i] = abs(displacement) / max(a, 1e-10)
        linearity[i] = abs(displacement) / max(path, 1e-10)

        if size_atr[i] > 1.0:  # Only assign direction for meaningful moves
            direction[i] = 1.0 if displacement > 0 else -1.0

        # Average body/range ratio over the impulse window
        seg_body = np.abs(close[i - lookback: i + 1] - open_[i - lookback: i + 1])
        seg_range = high[i - lookback: i + 1] - low[i - lookback: i + 1]
        valid = seg_range > 1e-10
        if valid.sum() > 0:
            body_ratio[i] = np.mean(seg_body[valid] / seg_range[valid])

    return {
        "impulse_size_atr": size_atr,
        "impulse_direction": direction,
        "impulse_linearity": linearity,
        "impulse_body_ratio": body_ratio,
    }


def _consolidation_metrics(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    impulse_direction: np.ndarray,
    lookback: int,
    consol_min: int,
    consol_max: int,
) -> dict:
    """
    Measure consolidation quality after an impulse.

    For each bar, looks back to find the impulse peak (highest/lowest
    point in the impulse window), then measures the consolidation from
    peak to current bar.
    """
    n = len(close)
    range_vs_impulse = np.full(n, np.nan)
    retrace_pct = np.full(n, np.nan)
    bar_count = np.full(n, np.nan)
    atr_ratio = np.full(n, np.nan)
    slope_norm = np.full(n, np.nan)
    narrowing = np.zeros(n)

    for i in range(lookback + consol_min, n):
        d = impulse_direction[i]
        if d == 0:
            continue

        # Find impulse peak in [i - lookback - consol_max, i - consol_min]
        search_start = max(0, i - lookback - consol_max)
        search_end = max(0, i - consol_min)
        if search_end <= search_start:
            continue

        if d > 0:
            peak_idx = search_start + int(np.argmax(high[search_start:search_end]))
        else:
            peak_idx = search_start + int(np.argmin(low[search_start:search_end]))

        bars_since = i - peak_idx
        if bars_since < consol_min or bars_since > lookback + consol_max:
            continue

        # Impulse range (from lookback before peak to peak)
        imp_start = max(0, peak_idx - lookback)
        if d > 0:
            impulse_range = high[peak_idx] - low[imp_start:peak_idx + 1].min()
        else:
            impulse_range = high[imp_start:peak_idx + 1].max() - low[peak_idx]

        if impulse_range < 1e-10:
            continue

        # Consolidation window: peak to current bar
        c_high = high[peak_idx:i + 1]
        c_low = low[peak_idx:i + 1]
        c_close = close[peak_idx:i + 1]
        consol_range = c_high.max() - c_low.min()

        range_vs_impulse[i] = consol_range / impulse_range
        bar_count[i] = bars_since / consol_max  # Normalized

        # Retracement
        if d > 0:
            retrace_pct[i] = max(0, (high[peak_idx] - close[i]) / impulse_range)
        else:
            retrace_pct[i] = max(0, (close[i] - low[peak_idx]) / impulse_range)

        # ATR during consolidation vs impulse
        consol_atr = np.nanmean(atr[peak_idx:i + 1])
        imp_atr = np.nanmean(atr[imp_start:peak_idx + 1])
        if imp_atr > 1e-10:
            atr_ratio[i] = consol_atr / imp_atr

        # Slope of close during consolidation (normalized by ATR)
        if len(c_close) >= 3:
            x = np.arange(len(c_close))
            slope = np.polyfit(x, c_close, 1)[0]
            a = atr[i] if not np.isnan(atr[i]) else 1.0
            slope_norm[i] = slope / max(a, 1e-10)

        # Narrowing: second-half range < first-half range
        if len(c_high) >= 4:
            mid = len(c_high) // 2
            first_range = c_high[:mid].max() - c_low[:mid].min()
            second_range = c_high[mid:].max() - c_low[mid:].min()
            narrowing[i] = 1.0 if second_range < first_range else 0.0

    return {
        "consol_range_vs_impulse": range_vs_impulse,
        "consol_retrace_pct": retrace_pct,
        "consol_bar_count": bar_count,
        "consol_atr_ratio": atr_ratio,
        "consol_slope": slope_norm,
        "consol_narrowing": narrowing,
    }


def generate_signals(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Generate mechanical flag-continuation signals (no ML).

    Conditions for LONG:
    1. Regime is trending (ADX proxy > threshold)
    2. Bullish impulse detected (size > min_atr_mult)
    3. Consolidation quality: retrace < max, range contracting, vol declining
    4. Breakout: close > consolidation high by breakout_atr_mult * ATR

    For SHORT: mirror all conditions.
    """
    from tradingagents.xgb_quant.config import FLAG_CONTINUATION_DEFAULTS
    p = {**FLAG_CONTINUATION_DEFAULTS, **(params or {})}

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    n = len(close)

    # ATR
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(p["atr_period"]).mean().values

    # ADX proxy
    adx = _efficiency_ratio(close, lookback=25)

    # Impulse detection
    imp = _impulse_detection(close, high, low, open_, atr, p["impulse_lookback"])

    # Consolidation metrics
    consol = _consolidation_metrics(
        close, high, low, atr, imp["impulse_direction"],
        p["impulse_lookback"], p["consolidation_min_bars"], p["consolidation_max_bars"],
    )

    # Output arrays
    signals = np.zeros(n, dtype=int)
    entries = np.full(n, np.nan)
    sls = np.full(n, np.nan)
    tps = np.full(n, np.nan)

    warmup = p["impulse_lookback"] + p["consolidation_max_bars"] + 5

    for i in range(warmup, n):
        if np.isnan(atr[i]) or atr[i] < 1e-10:
            continue

        # Regime gate
        if np.isnan(adx[i]) or adx[i] < p["adx_threshold"]:
            continue

        d = imp["impulse_direction"][i]
        if d == 0:
            continue

        # Impulse strength check
        if np.isnan(imp["impulse_size_atr"][i]) or imp["impulse_size_atr"][i] < p["impulse_min_atr_mult"]:
            continue

        # Consolidation quality checks
        rv = consol["consol_range_vs_impulse"][i]
        rp = consol["consol_retrace_pct"][i]
        ar = consol["consol_atr_ratio"][i]
        if np.isnan(rv) or np.isnan(rp) or np.isnan(ar):
            continue

        if rv > p["range_contraction_pct"]:
            continue
        if rp > p["consolidation_retrace_max"]:
            continue
        if ar > p["vol_decline_threshold"]:
            continue

        # Find consolidation boundaries
        search_start = max(0, i - p["consolidation_max_bars"])
        c_high = high[search_start:i + 1].max()
        c_low = low[search_start:i + 1].min()

        # Breakout check
        breakout_dist = atr[i] * p["breakout_atr_mult"]

        if d > 0 and close[i] > c_high - breakout_dist:
            # Bullish continuation
            entry = close[i]
            sl = entry - atr[i] * p["sl_atr_mult"]
            tp = entry + (entry - sl) * p["rr_target"]
            signals[i] = 1
            entries[i] = entry
            sls[i] = sl
            tps[i] = tp

        elif d < 0 and close[i] < c_low + breakout_dist:
            # Bearish continuation
            entry = close[i]
            sl = entry + atr[i] * p["sl_atr_mult"]
            tp = entry - (sl - entry) * p["rr_target"]
            signals[i] = -1
            entries[i] = entry
            sls[i] = sl
            tps[i] = tp

    return pd.DataFrame({
        "signal": signals,
        "entry_price": entries,
        "sl": sls,
        "tp": tps,
    }, index=df.index)


# ---------------------------------------------------------------------------
# XGBoost-integrated strategy
# ---------------------------------------------------------------------------

class FlagContinuationFeatures(TechnicalFeatures):
    """Technical features + flag continuation-specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            # Impulse detection
            "impulse_size_atr",       # Impulse displacement in ATR multiples
            "impulse_direction",      # +1 bullish, -1 bearish, 0 none
            "impulse_linearity",      # Efficiency ratio of impulse (0-1)
            "impulse_body_ratio",     # Avg candle body/range during impulse
            # Consolidation quality
            "consol_range_vs_impulse",  # Flag range / impulse range
            "consol_retrace_pct",     # Retracement of impulse (0-1)
            "consol_bar_count",       # Bars in consolidation (normalized)
            "consol_atr_ratio",       # Consolidation ATR / impulse ATR
            "consol_slope",           # Slope of close in consolidation / ATR
            "consol_narrowing",       # 1 if range narrowing, 0 otherwise
            # Breakout confirmation
            "breakout_distance_atr",  # Distance beyond flag boundary / ATR
            "breakout_volume_spike",  # Current vol / consolidation avg vol
            "breakout_bar_strength",  # Current bar body / range
            "bars_since_peak_norm",   # Bars since impulse peak (normalized)
            "trend_alignment",        # EMA alignment with impulse direction
            "adx_proxy",              # Efficiency-ratio ADX for regime gate
        ]

    @property
    def warmup_bars(self) -> int:
        return max(super().warmup_bars, 80)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)
        volume = df["volume"].values.astype(float)

        atr = features["atr_14"].values
        w = self.windows
        lookback = 20
        consol_min = 5
        consol_max = 20

        # Impulse detection
        imp = _impulse_detection(close, high, low, open_, atr, lookback)
        features["impulse_size_atr"] = imp["impulse_size_atr"]
        features["impulse_direction"] = imp["impulse_direction"]
        features["impulse_linearity"] = imp["impulse_linearity"]
        features["impulse_body_ratio"] = imp["impulse_body_ratio"]

        # Consolidation metrics
        consol = _consolidation_metrics(
            close, high, low, atr, imp["impulse_direction"],
            lookback, consol_min, consol_max,
        )
        features["consol_range_vs_impulse"] = consol["consol_range_vs_impulse"]
        features["consol_retrace_pct"] = consol["consol_retrace_pct"]
        features["consol_bar_count"] = consol["consol_bar_count"]
        features["consol_atr_ratio"] = consol["consol_atr_ratio"]
        features["consol_slope"] = consol["consol_slope"]
        features["consol_narrowing"] = consol["consol_narrowing"]

        n = len(close)

        # Breakout distance: close beyond recent consolidation boundary / ATR
        breakout_dist = np.full(n, np.nan)
        for i in range(lookback + consol_min, n):
            d = imp["impulse_direction"][i]
            if d == 0 or np.isnan(atr[i]) or atr[i] < 1e-10:
                continue
            search_start = max(0, i - consol_max)
            if d > 0:
                boundary = high[search_start:i].max()
                breakout_dist[i] = (close[i] - boundary) / atr[i]
            else:
                boundary = low[search_start:i].min()
                breakout_dist[i] = (boundary - close[i]) / atr[i]
        features["breakout_distance_atr"] = breakout_dist

        # Breakout volume spike: current volume / consolidation avg
        vol_spike = np.full(n, np.nan)
        vol_sma = pd.Series(volume).rolling(consol_max).mean().values
        valid = vol_sma > 1e-10
        vol_spike[valid] = volume[valid] / vol_sma[valid]
        features["breakout_volume_spike"] = vol_spike

        # Breakout bar strength: |body| / range
        bar_body = np.abs(close - open_)
        bar_range = high - low
        features["breakout_bar_strength"] = self._safe_divide(bar_body, bar_range)

        # Bars since impulse peak (normalized)
        bars_since = np.full(n, np.nan)
        for i in range(lookback + consol_min, n):
            d = imp["impulse_direction"][i]
            if d == 0:
                continue
            search_start = max(0, i - lookback - consol_max)
            search_end = max(0, i - consol_min)
            if search_end <= search_start:
                continue
            if d > 0:
                peak_idx = search_start + int(np.argmax(high[search_start:search_end]))
            else:
                peak_idx = search_start + int(np.argmin(low[search_start:search_end]))
            bars_since[i] = (i - peak_idx) / consol_max
        features["bars_since_peak_norm"] = bars_since

        # Trend alignment: EMA short vs long, aligned with impulse direction
        ema_s = pd.Series(close).ewm(span=w.ema_short).mean().values
        ema_l = pd.Series(close).ewm(span=w.ema_long).mean().values
        ema_dir = np.where(ema_s > ema_l, 1.0, np.where(ema_s < ema_l, -1.0, 0.0))
        features["trend_alignment"] = ema_dir * imp["impulse_direction"]

        # ADX proxy
        features["adx_proxy"] = _efficiency_ratio(close, lookback=25)

        return features


class FlagContinuationStrategy(BaseStrategy):
    """
    Flag / Pennant Continuation strategy.

    XGBoost learns which flag patterns lead to successful continuations,
    with a hard regime gate that blocks signals in range-bound markets
    and a direction lock ensuring entries only in the impulse direction.
    """

    @property
    def name(self) -> str:
        return "flag_continuation"

    def get_feature_set(self) -> BaseFeatureSet:
        return FlagContinuationFeatures(windows=self.windows)

    def check_regime_gate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> dict:
        """Only allow signals in trending markets."""
        n = len(close)
        if n < 30:
            return {"regime": "unknown", "is_valid": False, "blocked_reason": "Not enough data"}

        lookback = min(25, n - 1)
        displacement = abs(close[-1] - close[-1 - lookback])
        path = np.sum(np.abs(np.diff(close[-lookback - 1:])))
        adx_proxy = (displacement / max(path, 1e-10)) * 50.0

        from tradingagents.xgb_quant.config import FLAG_CONTINUATION_DEFAULTS
        threshold = FLAG_CONTINUATION_DEFAULTS.get("adx_threshold", 25)

        if adx_proxy >= threshold:
            return {"regime": "trend", "adx_proxy": adx_proxy, "is_valid": True}
        return {
            "regime": "range",
            "adx_proxy": adx_proxy,
            "is_valid": False,
            "blocked_reason": f"ADX proxy {adx_proxy:.1f} < {threshold} — not trending",
        }

    def predict_signal(
        self,
        features: pd.DataFrame,
        atr: float,
        current_price: float,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        symbol: str = "",
    ) -> Signal:
        """
        Generate signal with regime gate and impulse direction lock.
        """
        # Regime gate
        if high is not None and low is not None and close is not None:
            gate = self.check_regime_gate(high, low, close)
            if not gate.get("is_valid", False):
                reason = gate.get("blocked_reason", "Market not trending")
                logger.info(f"Flag continuation regime gate BLOCKED for {symbol}: {reason}")
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    rationale=f"flag_continuation: regime gate blocked — {reason}",
                )

        # XGBoost prediction
        signal = super().predict_signal(features, atr, current_price)

        if signal.direction == "HOLD":
            return signal

        # Direction lock: only allow entry in the impulse direction
        if "impulse_direction" in features.columns:
            imp_dir = features["impulse_direction"].iloc[-1]
            if signal.direction == "BUY" and imp_dir < 0:
                logger.info(f"Flag continuation direction lock: BUY blocked (impulse is bearish)")
                return Signal(
                    direction="HOLD", confidence=0.0,
                    rationale="flag_continuation: BUY blocked — impulse is bearish",
                )
            if signal.direction == "SELL" and imp_dir > 0:
                logger.info(f"Flag continuation direction lock: SELL blocked (impulse is bullish)")
                return Signal(
                    direction="HOLD", confidence=0.0,
                    rationale="flag_continuation: SELL blocked — impulse is bullish",
                )

        # Adjust SL/TP using RR-based target
        from tradingagents.xgb_quant.config import FLAG_CONTINUATION_DEFAULTS
        rr = FLAG_CONTINUATION_DEFAULTS.get("rr_target", 2.5)
        sl_distance = abs(signal.entry - signal.stop_loss)
        if signal.direction == "BUY":
            signal.take_profit = signal.entry + sl_distance * rr
        else:
            signal.take_profit = signal.entry - sl_distance * rr

        return signal
