"""
Donchian Channel Breakout Strategy

Enters on Donchian channel breakouts in trending or squeeze regimes,
with optional silver-lead confirmation for gold trades and SMA trend bias.

REGIME GATE: Only fires in "trend" (ADX > 25) or "squeeze" (BB Width compressed
and expanding) regimes. Blocked in range-bound markets.

SILVER LEAD: For XAUUSD/XAGUSD, requires silver to have broken its own Donchian
channel within the last N bars as confirmation.

Best suited for: XAUUSD, GBPJPY, EURJPY, BTCUSD — trending instruments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import (
    FeatureWindows, RiskDefaults, DONCHIAN_BREAKOUT_DEFAULTS,
)

logger = logging.getLogger(__name__)

# Pairs that benefit from silver-lead confirmation
SILVER_LEAD_PAIRS = frozenset({"XAUUSD", "XAGUSD"})


# ---------------------------------------------------------------------------
# Donchian helpers (reusable outside the strategy)
# ---------------------------------------------------------------------------

def calculate_donchian(
    df: pd.DataFrame, length: int = 20,
) -> tuple:
    """
    Calculate Donchian Channel (upper, lower, middle).

    Returns:
        (upper, lower, middle) as numpy arrays
    """
    high = pd.Series(df["high"].values.astype(float))
    low = pd.Series(df["low"].values.astype(float))
    upper = high.rolling(length).max().values
    lower = low.rolling(length).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle


def is_trend_or_squeeze_regime(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    bb_squeeze_threshold: float = 0.018,
) -> dict:
    """
    Classify whether market is in a trend or squeeze regime.

    Uses ADX proxy (efficiency ratio) and Bollinger Band width.

    Returns:
        dict with keys: regime ("trend", "squeeze", "range"),
        adx_proxy, bb_width, is_valid (True if trend or squeeze)
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    if n < 30:
        return {"regime": "unknown", "adx_proxy": 0, "bb_width": 0, "is_valid": False}

    # ADX proxy via efficiency ratio (same method as scanner.py)
    lookback = min(25, n - 1)
    displacement = abs(close[-1] - close[-1 - lookback])
    total_path = sum(abs(close[i] - close[i - 1]) for i in range(-lookback, 0))
    efficiency = displacement / (total_path + 1e-10)
    adx_proxy = efficiency * 50.0

    # Bollinger Band width
    cs = pd.Series(close)
    sma20 = cs.rolling(20).mean()
    std20 = cs.rolling(20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = ((bb_upper - bb_lower) / (sma20 + 1e-10)).values

    current_bb_width = bb_width[-1] if not np.isnan(bb_width[-1]) else 0
    prev_bb_width = bb_width[-2] if len(bb_width) > 1 and not np.isnan(bb_width[-2]) else current_bb_width

    # Classify
    if adx_proxy >= adx_threshold:
        regime = "trend"
        is_valid = True
    elif current_bb_width < bb_squeeze_threshold and current_bb_width > prev_bb_width:
        # Squeeze that's starting to expand — this IS the breakout moment
        regime = "squeeze"
        is_valid = True
    elif current_bb_width < bb_squeeze_threshold:
        # Still compressed, not expanding yet — wait for expansion
        regime = "squeeze_building"
        is_valid = False
    else:
        regime = "range"
        is_valid = False

    return {
        "regime": regime,
        "adx_proxy": adx_proxy,
        "bb_width": current_bb_width,
        "is_valid": is_valid,
    }


def has_silver_lead(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    donchian_length: int = 20,
    lookback_bars: int = 3,
) -> bool:
    """
    Check if silver broke its Donchian upper within last `lookback_bars`.

    Silver often leads gold — a silver breakout confirms gold momentum.

    Args:
        gold_df: Gold OHLCV (used only for alignment check)
        silver_df: Silver OHLCV
        donchian_length: Donchian channel period
        lookback_bars: How many bars back to check for silver breakout

    Returns:
        True if silver confirmed the breakout
    """
    if silver_df is None or len(silver_df) < donchian_length + lookback_bars:
        # No silver data — skip confirmation (don't block)
        return True

    silver_close = silver_df["close"].values.astype(float)
    silver_high = pd.Series(silver_df["high"].values.astype(float))
    silver_low = pd.Series(silver_df["low"].values.astype(float))

    silver_don_upper = silver_high.rolling(donchian_length).max().values
    silver_don_lower = silver_low.rolling(donchian_length).min().values

    # Check if silver closed above its Donchian upper in any of the last N bars
    for offset in range(1, lookback_bars + 1):
        idx = -(offset + 1)  # -2, -3, -4 (previous bars' Donchian upper)
        bar_idx = -offset     # -1, -2, -3 (the bar that broke out)

        if abs(idx) > len(silver_close) or abs(bar_idx) > len(silver_close):
            continue

        prev_upper = silver_don_upper[idx]
        if not np.isnan(prev_upper) and silver_close[bar_idx] > prev_upper:
            return True

    return False


def generate_signals(
    df_gold: pd.DataFrame,
    df_silver: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Generate mechanical Donchian breakout signals (no ML).

    All conditions must be true on bar close for a LONG entry:
    1. Regime is "trend" or "squeeze"
    2. Trend bias: SMA(fast) > SMA(slow)
    3. Gold close > previous bar's Donchian Upper
    4. Silver confirmation (if available): silver broke its Donchian within N bars

    For SHORT: mirror all conditions.

    Args:
        df_gold: Gold OHLCV DataFrame
        df_silver: Silver OHLCV DataFrame (optional, for confirmation)
        params: Override defaults from DONCHIAN_BREAKOUT_DEFAULTS

    Returns:
        DataFrame with columns: signal (1=long, -1=short, 0=none),
        entry_price, sl, tp, regime
    """
    p = {**DONCHIAN_BREAKOUT_DEFAULTS, **(params or {})}

    close = df_gold["close"].values.astype(float)
    high = df_gold["high"].values.astype(float)
    low = df_gold["low"].values.astype(float)
    n = len(close)

    # Donchian channels
    don_upper, don_lower, don_middle = calculate_donchian(df_gold, p["donchian_length"])

    # Trend bias: fast MA > slow MA
    cs = pd.Series(close)
    sma_fast = cs.rolling(p["trend_bias_fast_ma"]).mean().values
    sma_slow = cs.rolling(p["trend_bias_slow_ma"]).mean().values

    # ATR for SL/TP
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(p["atr_period"]).mean().values

    # Output arrays
    signals = np.zeros(n, dtype=int)
    entries = np.full(n, np.nan)
    sls = np.full(n, np.nan)
    tps = np.full(n, np.nan)
    regimes = [""] * n

    # Pre-compute regime for each bar (rolling window)
    warmup = max(p["donchian_length"], p["trend_bias_slow_ma"], p["atr_period"]) + 5

    for i in range(warmup, n):
        # Regime check (use last 30 bars of data up to this bar)
        window_start = max(0, i - 30)
        window_df = df_gold.iloc[window_start:i + 1]
        regime_info = is_trend_or_squeeze_regime(
            window_df,
            adx_threshold=p["adx_threshold"],
            bb_squeeze_threshold=p["bb_width_squeeze_threshold"],
        )
        regimes[i] = regime_info["regime"]

        if not regime_info["is_valid"]:
            continue

        if np.isnan(atr[i]) or atr[i] < 1e-10:
            continue

        # Previous bar's Donchian levels (avoid look-ahead)
        prev_don_upper = don_upper[i - 1] if i > 0 and not np.isnan(don_upper[i - 1]) else np.nan
        prev_don_lower = don_lower[i - 1] if i > 0 and not np.isnan(don_lower[i - 1]) else np.nan

        if np.isnan(prev_don_upper) or np.isnan(prev_don_lower):
            continue

        # LONG conditions
        if (
            close[i] > prev_don_upper
            and not np.isnan(sma_fast[i])
            and not np.isnan(sma_slow[i])
            and sma_fast[i] > sma_slow[i]
        ):
            # Silver confirmation (only check if silver data available)
            if df_silver is not None and len(df_silver) > p["donchian_length"]:
                # Align silver data to current bar index
                silver_slice = df_silver.iloc[:min(i + 1, len(df_silver))]
                if not has_silver_lead(
                    df_gold.iloc[:i + 1], silver_slice,
                    donchian_length=p["donchian_length"],
                    lookback_bars=p["silver_lead_bars"],
                ):
                    continue

            entry = close[i]
            sl = entry - atr[i] * p["sl_atr_mult"]
            tp = entry + (entry - sl) * p["rr_target"]

            signals[i] = 1
            entries[i] = entry
            sls[i] = sl
            tps[i] = tp

        # SHORT conditions (mirror)
        elif (
            close[i] < prev_don_lower
            and not np.isnan(sma_fast[i])
            and not np.isnan(sma_slow[i])
            and sma_fast[i] < sma_slow[i]
        ):
            if df_silver is not None and len(df_silver) > p["donchian_length"]:
                silver_slice = df_silver.iloc[:min(i + 1, len(df_silver))]
                # For short: check if silver broke Donchian lower
                if silver_slice is not None and len(silver_slice) >= p["donchian_length"]:
                    silver_close = silver_slice["close"].values.astype(float)
                    silver_don_lower = pd.Series(
                        silver_slice["low"].values.astype(float)
                    ).rolling(p["donchian_length"]).min().values
                    confirmed = False
                    for offset in range(1, p["silver_lead_bars"] + 1):
                        idx = -(offset + 1)
                        bar_idx = -offset
                        if abs(idx) > len(silver_close):
                            continue
                        prev_lower = silver_don_lower[idx]
                        if not np.isnan(prev_lower) and silver_close[bar_idx] < prev_lower:
                            confirmed = True
                            break
                    if not confirmed:
                        continue

            entry = close[i]
            sl = entry + atr[i] * p["sl_atr_mult"]
            tp = entry - (sl - entry) * p["rr_target"]

            signals[i] = -1
            entries[i] = entry
            sls[i] = sl
            tps[i] = tp

    result = pd.DataFrame({
        "signal": signals,
        "entry_price": entries,
        "sl": sls,
        "tp": tps,
        "regime": regimes,
    }, index=df_gold.index)

    return result


# ---------------------------------------------------------------------------
# XGBoost-integrated strategy (plugs into trainer / predictor pipeline)
# ---------------------------------------------------------------------------

class DonchianBreakoutFeatures(TechnicalFeatures):
    """Technical features + Donchian breakout-specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "donchian_upper_break",     # Close > prev Donchian upper (binary)
            "donchian_lower_break",     # Close < prev Donchian lower (binary)
            "donchian_position",        # Price position in channel (0-1)
            "donchian_width_pct",       # Channel width / price
            "donchian_middle_dist",     # Distance from middle line / ATR
            "trend_bias",               # SMA50 > SMA200 (+1) or < (-1) or 0
            "trend_bias_strength",      # |SMA50 - SMA200| / ATR
            "adx_proxy",               # Efficiency ratio * 50
            "bb_squeeze_pctile",       # BB width percentile (low = squeeze)
            "bb_expanding",            # BB width increasing (binary)
            "atr_expansion",           # Current ATR / 20-bar avg ATR
            "bars_since_upper_break",  # Bars since last Donchian upper break
            "bars_since_lower_break",  # Bars since last Donchian lower break
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        w = self.windows
        n = len(close)

        don_length = w.donchian_period  # Default 20

        # Donchian channels
        hs = pd.Series(high)
        ls = pd.Series(low)
        don_upper = hs.rolling(don_length).max().values
        don_lower = ls.rolling(don_length).min().values
        don_middle = (don_upper + don_lower) / 2.0
        don_range = don_upper - don_lower

        # Previous bar Donchian (avoid look-ahead)
        prev_don_upper = np.roll(don_upper, 1)
        prev_don_lower = np.roll(don_lower, 1)
        prev_don_upper[0] = np.nan
        prev_don_lower[0] = np.nan

        # Break detection
        features["donchian_upper_break"] = (close > prev_don_upper).astype(float)
        features["donchian_lower_break"] = (close < prev_don_lower).astype(float)

        # Position in channel
        features["donchian_position"] = self._safe_divide(close - don_lower, don_range)

        # Width as % of price
        features["donchian_width_pct"] = self._safe_divide(don_range, close)

        # Distance from middle line normalized by ATR
        atr = features["atr_14"].values
        features["donchian_middle_dist"] = self._safe_divide(close - don_middle, atr)

        # Trend bias: SMA50 vs SMA200
        cs = pd.Series(close)
        sma_fast = cs.rolling(50).mean().values
        sma_slow = cs.rolling(200).mean().values
        bias = np.where(sma_fast > sma_slow, 1.0, np.where(sma_fast < sma_slow, -1.0, 0.0))
        # Handle NaN
        nan_mask = np.isnan(sma_fast) | np.isnan(sma_slow)
        bias[nan_mask] = np.nan
        features["trend_bias"] = bias

        features["trend_bias_strength"] = self._safe_divide(
            np.abs(sma_fast - sma_slow), atr
        )

        # ADX proxy (efficiency ratio) — vectorised
        lookback = 25
        abs_diffs = np.abs(np.diff(close, prepend=close[0]))
        rolling_path = pd.Series(abs_diffs).rolling(lookback).sum().values
        displacement = np.abs(close - np.roll(close, lookback))
        displacement[:lookback] = np.nan
        adx_proxy = (displacement / (rolling_path + 1e-10)) * 50.0
        adx_proxy[:lookback] = np.nan
        features["adx_proxy"] = adx_proxy

        # BB squeeze percentile
        bb_width = features["bb_width"].values
        bb_series = pd.Series(bb_width)
        bb_rank = bb_series.rolling(100, min_periods=20).rank(pct=True)
        features["bb_squeeze_pctile"] = bb_rank.values

        # BB expanding (current > previous)
        bb_prev = pd.Series(bb_width).shift(1).values
        features["bb_expanding"] = (bb_width > bb_prev).astype(float)
        features.loc[pd.Series(np.isnan(bb_prev)), "bb_expanding"] = np.nan

        # ATR expansion
        atr_sma = pd.Series(atr).rolling(20).mean().values
        features["atr_expansion"] = self._safe_divide(atr, atr_sma)

        # Bars since last break
        upper_breaks = (close > prev_don_upper)
        lower_breaks = (close < prev_don_lower)
        bars_since_upper = np.full(n, np.nan)
        bars_since_lower = np.full(n, np.nan)
        last_upper = np.nan
        last_lower = np.nan
        for i in range(n):
            if upper_breaks[i]:
                last_upper = 0
            elif not np.isnan(last_upper):
                last_upper += 1
            bars_since_upper[i] = last_upper

            if lower_breaks[i]:
                last_lower = 0
            elif not np.isnan(last_lower):
                last_lower += 1
            bars_since_lower[i] = last_lower

        features["bars_since_upper_break"] = bars_since_upper
        features["bars_since_lower_break"] = bars_since_lower

        return features


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout strategy.

    XGBoost learns when breakouts succeed, with a hard regime gate that
    blocks signals in range-bound markets. Optionally uses silver-lead
    confirmation for metals.
    """

    @property
    def name(self) -> str:
        return "donchian_breakout"

    def get_feature_set(self) -> BaseFeatureSet:
        return DonchianBreakoutFeatures(windows=self.windows)

    def check_regime_gate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> dict:
        """
        Regime gate — only allow signals in trend or squeeze markets.

        Returns dict with regime info and whether signal is allowed.
        """
        n = len(close)
        if n < 30:
            return {"regime": "unknown", "is_valid": False, "blocked_reason": "Not enough data"}

        # Build a minimal DataFrame for the helper
        df_tmp = pd.DataFrame({"high": high, "low": low, "close": close})
        info = is_trend_or_squeeze_regime(df_tmp)

        if not info["is_valid"]:
            info["blocked_reason"] = (
                f"Market is {info['regime']} (ADX proxy={info['adx_proxy']:.1f}, "
                f"BB width={info['bb_width']:.4f}) — need trend or squeeze"
            )
        return info

    def check_silver_lead(
        self,
        silver_df: Optional[pd.DataFrame],
        donchian_length: int = 20,
        lookback_bars: int = 3,
        direction: str = "BUY",
    ) -> bool:
        """
        Check silver-lead confirmation.

        Returns True if silver confirmed (or if no silver data provided).
        """
        if silver_df is None or len(silver_df) < donchian_length + lookback_bars:
            return True  # No silver data — don't block

        if direction == "BUY":
            return has_silver_lead(
                pd.DataFrame(),  # gold_df not used in has_silver_lead
                silver_df,
                donchian_length=donchian_length,
                lookback_bars=lookback_bars,
            )
        else:
            # Short: check if silver broke Donchian lower
            silver_close = silver_df["close"].values.astype(float)
            silver_don_lower = pd.Series(
                silver_df["low"].values.astype(float)
            ).rolling(donchian_length).min().values

            for offset in range(1, lookback_bars + 1):
                idx = -(offset + 1)
                bar_idx = -offset
                if abs(idx) > len(silver_close):
                    continue
                prev_lower = silver_don_lower[idx]
                if not np.isnan(prev_lower) and silver_close[bar_idx] < prev_lower:
                    return True
            return False

    def predict_signal(
        self,
        features: pd.DataFrame,
        atr: float,
        current_price: float,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        symbol: str = "",
        silver_df: Optional[pd.DataFrame] = None,
    ) -> Signal:
        """
        Generate signal with regime gate and optional silver confirmation.

        Args:
            features: Computed feature DataFrame
            atr: Current ATR value
            current_price: Current close price
            high/low/close: Price arrays for regime gate check
            symbol: Trading pair name
            silver_df: Silver OHLCV for lead confirmation (metals only)
        """
        # Regime gate
        if high is not None and low is not None and close is not None:
            gate = self.check_regime_gate(high, low, close)
            if not gate.get("is_valid", False):
                reason = gate.get("blocked_reason", "Market not in trend/squeeze")
                logger.info(f"Donchian regime gate BLOCKED for {symbol}: {reason}")
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    rationale=f"donchian_breakout: regime gate blocked — {reason}",
                )

        # Get XGBoost prediction
        signal = super().predict_signal(features, atr, current_price)

        if signal.direction == "HOLD":
            return signal

        # Silver-lead confirmation for metals
        if symbol in SILVER_LEAD_PAIRS and silver_df is not None:
            donchian_length = DONCHIAN_BREAKOUT_DEFAULTS.get("donchian_length", 20)
            silver_bars = DONCHIAN_BREAKOUT_DEFAULTS.get("silver_lead_bars", 3)

            if not self.check_silver_lead(silver_df, donchian_length, silver_bars, signal.direction):
                logger.info(
                    f"Donchian silver-lead BLOCKED for {symbol} {signal.direction}: "
                    "Silver did not confirm breakout"
                )
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    rationale=f"donchian_breakout: silver-lead not confirmed for {signal.direction}",
                )

        # Adjust SL/TP to use RR-based TP instead of flat ATR mult
        rr_target = DONCHIAN_BREAKOUT_DEFAULTS.get("rr_target", 3.0)
        sl_distance = abs(signal.entry - signal.stop_loss)
        if signal.direction == "BUY":
            signal.take_profit = signal.entry + sl_distance * rr_target
        else:
            signal.take_profit = signal.entry - sl_distance * rr_target

        signal.rationale = (
            f"donchian_breakout: {signal.direction} "
            f"conf={signal.confidence:.3f}, "
            f"RR=1:{rr_target:.1f}"
        )

        return signal
