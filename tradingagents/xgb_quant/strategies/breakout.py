"""
Breakout Strategy

Detects volatility compression and enters on expansion using
Donchian channels, ATR squeeze detection, and volume spikes.
Best suited for pairs exiting consolidation or pre-news setups.
"""

import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.features.regime import RegimeFeatures
from tradingagents.xgb_quant.features.composite import CompositeFeatures
from tradingagents.xgb_quant.config import FeatureWindows


class BreakoutFeatures(TechnicalFeatures):
    """Technical features + breakout-specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "donchian_position",   # Price position in Donchian channel (0-1)
            "donchian_width_pct",  # Channel width / price
            "atr_ratio",           # Current ATR / 20-bar avg ATR (squeeze detection)
            "range_vs_atr",        # Current bar range / ATR
            "volume_spike",        # Volume > 1.5x average (binary)
            "consecutive_narrow",  # Count of narrow-range bars before current
            "high_break",          # Price broke above recent Donchian high (binary)
            "low_break",           # Price broke below recent Donchian low (binary)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        w = self.windows

        period = w.donchian_period

        # Donchian Channel
        hs = pd.Series(high)
        ls = pd.Series(low)
        don_high = hs.rolling(period).max().values
        don_low = ls.rolling(period).min().values
        don_range = don_high - don_low

        features["donchian_position"] = self._safe_divide(close - don_low, don_range)
        features["donchian_width_pct"] = self._safe_divide(don_range, close)

        # ATR ratio (current / 20-bar average = squeeze detection)
        atr = features["atr_14"].values
        atr_sma = pd.Series(atr).rolling(20).mean().values
        features["atr_ratio"] = self._safe_divide(atr, atr_sma)

        # Range vs ATR
        bar_range = high - low
        features["range_vs_atr"] = self._safe_divide(bar_range, atr)

        # Volume spike
        vol_sma = pd.Series(volume).rolling(w.volume_avg_period).mean().values
        features["volume_spike"] = (volume > vol_sma * 1.5).astype(float)

        # Consecutive narrow range bars
        narrow = bar_range < (atr * 0.5)
        consec = np.zeros(len(close))
        for i in range(1, len(close)):
            if narrow[i - 1]:
                consec[i] = consec[i - 1] + 1
        features["consecutive_narrow"] = consec

        # Break detection
        prev_don_high = pd.Series(don_high).shift(1).values
        prev_don_low = pd.Series(don_low).shift(1).values
        features["high_break"] = (close > prev_don_high).astype(float)
        features["low_break"] = (close < prev_don_low).astype(float)

        return features


class BreakoutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "breakout"

    def get_feature_set(self) -> BaseFeatureSet:
        return BreakoutFeatures(windows=self.windows)
