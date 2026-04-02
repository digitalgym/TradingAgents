"""
Wyckoff Volume-Spread Analysis Features.

Computes Wyckoff-specific indicators: candlestick structure, volume dynamics,
effort vs result, spring/upthrust signals, and range context. Designed for
the LLM Wyckoff gatekeeper but reusable by any strategy.
"""

import numpy as np
import pandas as pd
from typing import Optional

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows


class WyckoffFeatures(BaseFeatureSet):
    """Wyckoff volume-spread analysis features."""

    @property
    def feature_names(self) -> list:
        return [
            # Candlestick structure
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "bar_range_ratio",
            # Volume dynamics
            "volume_spike", "volume_delta", "up_volume_ratio",
            "cumulative_volume_delta",
            # Effort vs result
            "effort_vs_result",
            # Range context
            "range_width", "price_in_range", "range_contraction",
            # Wyckoff events
            "spring_signal", "upthrust_signal",
            # Trend
            "ema_trend",
        ]

    @property
    def warmup_bars(self) -> int:
        return 60

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)
        volume = df["volume"].values.astype(float)
        n = len(close)

        features = pd.DataFrame(index=df.index)

        # --- ATR (needed for normalisation) ---
        atr = self._compute_atr(high, low, close, 14)

        # --- Candlestick structure ---
        bar_range = high - low
        body = np.abs(close - open_)
        features["body_ratio"] = self._safe_divide(body, bar_range)
        features["upper_wick_ratio"] = self._safe_divide(
            high - np.maximum(close, open_), bar_range
        )
        features["lower_wick_ratio"] = self._safe_divide(
            np.minimum(close, open_) - low, bar_range
        )
        features["bar_range_ratio"] = self._safe_divide(bar_range, atr)

        # --- Volume dynamics ---
        vol_sma20 = pd.Series(volume).rolling(20).mean().values
        features["volume_spike"] = self._safe_divide(volume, vol_sma20)

        # Signed volume delta proxy: direction of bar * volume
        bar_direction = self._safe_divide(close - open_, bar_range)
        vol_delta = bar_direction * volume
        features["volume_delta"] = vol_delta

        # Up-volume ratio over 10 bars
        is_up_bar = (close > open_).astype(float)
        up_vol = is_up_bar * volume
        features["up_volume_ratio"] = self._safe_divide(
            pd.Series(up_vol).rolling(10).sum().values,
            pd.Series(volume).rolling(10).sum().values,
        )

        # Cumulative volume delta over 10 bars
        features["cumulative_volume_delta"] = (
            pd.Series(vol_delta).rolling(10).sum().values
        )

        # --- Effort vs result ---
        # High volume + small range = absorption (ratio > 1)
        # Low volume + big range = lack of conviction (ratio < 1)
        vol_ratio = self._safe_divide(volume, vol_sma20)
        range_ratio = self._safe_divide(bar_range, atr)
        features["effort_vs_result"] = self._safe_divide(vol_ratio, range_ratio)

        # --- Range context ---
        rolling_high_20 = pd.Series(high).rolling(20).max().values
        rolling_low_20 = pd.Series(low).rolling(20).min().values
        range_20 = rolling_high_20 - rolling_low_20
        features["range_width"] = self._safe_divide(range_20, atr)
        features["price_in_range"] = self._safe_divide(
            close - rolling_low_20, range_20
        )

        # Range contraction: 10-bar range / 20-bar range
        rolling_high_10 = pd.Series(high).rolling(10).max().values
        rolling_low_10 = pd.Series(low).rolling(10).min().values
        range_10 = rolling_high_10 - rolling_low_10
        features["range_contraction"] = self._safe_divide(range_10, range_20)

        # --- Wyckoff events ---
        # Spring: price dips below 20-bar low then closes back above it
        # (low penetrates range floor but close recovers)
        spring = np.zeros(n)
        for i in range(1, n):
            if not np.isnan(rolling_low_20[i - 1]):
                floor = rolling_low_20[i - 1]
                if low[i] < floor and close[i] > floor:
                    spring[i] = 1.0
        features["spring_signal"] = spring

        # Upthrust: price spikes above 20-bar high then closes back below it
        upthrust = np.zeros(n)
        for i in range(1, n):
            if not np.isnan(rolling_high_20[i - 1]):
                ceiling = rolling_high_20[i - 1]
                if high[i] > ceiling and close[i] < ceiling:
                    upthrust[i] = 1.0
        features["upthrust_signal"] = upthrust

        # --- Trend: EMA20 vs EMA50, ATR-normalised ---
        ema_short = self._ema(close, 20)
        ema_long = self._ema(close, 50)
        features["ema_trend"] = self._safe_divide(ema_short - ema_long, atr)

        return features

    # ------------------------------------------------------------------
    # Helpers (reuse from TechnicalFeatures pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _compute_atr(high, low, close, period: int = 14) -> np.ndarray:
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr
