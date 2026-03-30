"""
Breakout Strategy (Rule-Based)

Detects volatility compression (squeeze) and enters on expansion using
Donchian channels, ATR squeeze detection, and volume spikes.
Best suited for pairs exiting consolidation or pre-news setups.

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Price breaks above Donchian high with volume spike
  +0.25  ATR ratio expanding (>1.2x = squeeze releasing)
  +0.20  Consecutive narrow bars > 3 then range expansion
  +0.15  EMA alignment confirms direction
  +0.10  ADX rising (trend starting)
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows


class BreakoutFeatures(TechnicalFeatures):
    """Technical features + breakout-specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "donchian_position",
            "donchian_width_pct",
            "atr_ratio",
            "range_vs_atr",
            "volume_spike",
            "consecutive_narrow",
            "high_break",
            "low_break",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        w = self.windows

        period = w.donchian_period

        hs = pd.Series(high)
        ls = pd.Series(low)
        don_high = hs.rolling(period).max().values
        don_low = ls.rolling(period).min().values
        don_range = don_high - don_low

        features["donchian_position"] = self._safe_divide(close - don_low, don_range)
        features["donchian_width_pct"] = self._safe_divide(don_range, close)

        atr = features["atr_14"].values
        atr_sma = pd.Series(atr).rolling(20).mean().values
        features["atr_ratio"] = self._safe_divide(atr, atr_sma)

        bar_range = high - low
        features["range_vs_atr"] = self._safe_divide(bar_range, atr)

        vol_sma = pd.Series(volume).rolling(w.volume_avg_period).mean().values
        features["volume_spike"] = (volume > vol_sma * 1.5).astype(float)

        narrow = bar_range < (atr * 0.5)
        consec = np.zeros(len(close))
        for i in range(1, len(close)):
            if narrow[i - 1]:
                consec[i] = consec[i - 1] + 1
        features["consecutive_narrow"] = consec

        prev_don_high = pd.Series(don_high).shift(1).values
        prev_don_low = pd.Series(don_low).shift(1).values
        features["high_break"] = (close > prev_don_high).astype(float)
        features["low_break"] = (close < prev_don_low).astype(float)

        return features


class BreakoutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "donchian_breakout"

    def get_feature_set(self) -> BaseFeatureSet:
        return BreakoutFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        high_break = row.get("high_break", 0.0)
        low_break = row.get("low_break", 0.0)

        # No break = no signal (core requirement)
        if (np.isnan(high_break) or high_break < 0.5) and \
           (np.isnan(low_break) or low_break < 0.5):
            return 0.0

        # Determine direction from break
        if not np.isnan(high_break) and high_break > 0.5:
            direction = 1.0   # Bullish breakout
        else:
            direction = -1.0  # Bearish breakout

        # 1. Breakout detected (+0.30)
        score += 0.30 * direction

        # 2. Volume confirmation
        volume_spike = row.get("volume_spike", np.nan)
        if not np.isnan(volume_spike) and volume_spike > 0.5:
            score += 0.20 * direction  # Breakout with volume = stronger
        else:
            score += 0.05 * direction  # No volume = weaker signal

        # 3. ATR expansion (squeeze releasing)
        atr_ratio = row.get("atr_ratio", np.nan)
        if not np.isnan(atr_ratio):
            if atr_ratio > 1.3:
                score += 0.20 * direction  # Strong expansion
            elif atr_ratio > 1.1:
                score += 0.10 * direction  # Moderate expansion

        # 4. Prior compression (consecutive narrow bars)
        consec_narrow = row.get("consecutive_narrow", np.nan)
        if not np.isnan(consec_narrow) and consec_narrow >= 3:
            score += 0.15 * direction  # Breakout after squeeze

        # 5. EMA alignment confirms direction
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            if (direction > 0 and ema_cross > 0.5) or \
               (direction < 0 and ema_cross < 0.5):
                score += 0.10 * direction  # Aligned

        return float(np.clip(score, -1.0, 1.0))
