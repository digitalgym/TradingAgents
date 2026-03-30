"""
Volume Profile Strategy (Rule-Based)

Uses Volume Profile levels (POC, VAH, VAL) combined with
technical context to generate signals at high-volume nodes.

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Price at/near VAL with bullish structure (bounce expected)
  +0.25  Volume confirmation at level
  +0.20  POC acting as support/resistance
  +0.15  Trend alignment (EMA)
  +0.10  RSI not extreme
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.volume_profile import VolumeProfileFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class VolumeProfileStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "volume_profile_strat"

    def get_feature_set(self) -> BaseFeatureSet:
        return CompositeFeatures(
            providers=[
                TechnicalFeatures(windows=self.windows),
                VolumeProfileFeatures(windows=self.windows),
            ],
            windows=self.windows,
        )

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # Volume profile position features
        vp_position = row.get("vp_position", np.nan)  # 0=VAL, 0.5=POC, 1=VAH
        poc_dist = row.get("poc_dist_atr", np.nan)     # Distance from POC in ATR
        vah_dist = row.get("vah_dist_atr", np.nan)     # Distance from VAH
        val_dist = row.get("val_dist_atr", np.nan)     # Distance from VAL

        # Need at least one VP feature
        has_vp = not (np.isnan(vp_position) and np.isnan(poc_dist))
        if not has_vp:
            # Fall back to pure technical
            return self._score_technical_only(row)

        # 1. Price at value area edges (mean reversion to POC)
        if not np.isnan(vp_position):
            if vp_position < 0.15:
                score += 0.30  # Near VAL — expect bounce UP
            elif vp_position < 0.30:
                score += 0.15  # Below POC — mild bullish
            elif vp_position > 0.85:
                score -= 0.30  # Near VAH — expect rejection DOWN
            elif vp_position > 0.70:
                score -= 0.15  # Above POC — mild bearish

        # 2. Distance from POC (closer = stronger)
        if not np.isnan(poc_dist):
            if poc_dist < 0.5:
                pass  # At POC — neutral (price accepted here)
            elif poc_dist > 2.0 and score != 0:
                score *= 1.2  # Far from POC with directional bias = stronger

        # 3. Volume confirmation
        volume_ratio = row.get("volume_ratio", np.nan)
        if not np.isnan(volume_ratio) and volume_ratio > 1.3:
            # High volume at level = institutional activity
            if score > 0:
                score += 0.15
            elif score < 0:
                score -= 0.15

        # 4. EMA trend alignment
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            if score > 0 and ema_cross > 0.5:
                score += 0.15  # Bullish bias + bullish trend
            elif score < 0 and ema_cross < 0.5:
                score -= 0.15  # Bearish bias + bearish trend

        # 5. RSI filter
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if score > 0 and rsi > 75:
                score *= 0.5  # Overbought, reduce bullish
            elif score < 0 and rsi < 25:
                score *= 0.5  # Oversold, reduce bearish

        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def _score_technical_only(row) -> float:
        """Fallback when no volume profile data is available."""
        score = 0.0

        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            score += 0.15 if ema_cross > 0.5 else -0.15

        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if rsi < 30:
                score += 0.15
            elif rsi > 70:
                score -= 0.15

        return float(np.clip(score, -1.0, 1.0))
