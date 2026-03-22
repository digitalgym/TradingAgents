"""
Mean Reversion Strategy

Fades overextended price moves back to the mean using
Bollinger Band position, Z-score, RSI extremes, and Stochastic.
Best suited for range-bound, low-volatility pairs (EURGBP, AUDNZD, USDCHF).
"""

import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.config import FeatureWindows, RiskDefaults


class MeanReversionFeatures(TechnicalFeatures):
    """Technical features + mean reversion specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "zscore_20", "zscore_50",
            "stoch_k", "stoch_d",
            "close_vs_sma20", "close_vs_sma50",
            "bb_squeeze",  # BB width percentile (low = squeeze)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        w = self.windows

        # Z-scores
        cs = pd.Series(close)
        sma20 = cs.rolling(w.mid).mean()
        std20 = cs.rolling(w.mid).std()
        sma50 = cs.rolling(w.long).mean()
        std50 = cs.rolling(w.long).std()

        features["zscore_20"] = ((cs - sma20) / (std20 + 1e-10)).values
        features["zscore_50"] = ((cs - sma50) / (std50 + 1e-10)).values

        # Stochastic %K / %D
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        period = w.stoch_period

        lowest_low = pd.Series(low).rolling(period).min()
        highest_high = pd.Series(high).rolling(period).max()
        stoch_k = 100 * (cs - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()

        features["stoch_k"] = stoch_k.values
        features["stoch_d"] = stoch_d.values

        # Close vs SMA (normalised)
        features["close_vs_sma20"] = ((cs - sma20) / (sma20 + 1e-10)).values
        features["close_vs_sma50"] = ((cs - sma50) / (sma50 + 1e-10)).values

        # BB squeeze (rolling percentile of BB width)
        bb_width = features["bb_width"].values
        bb_series = pd.Series(bb_width)
        bb_rank = bb_series.rolling(100, min_periods=20).rank(pct=True)
        features["bb_squeeze"] = bb_rank.values

        return features


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    def get_feature_set(self) -> BaseFeatureSet:
        return MeanReversionFeatures(windows=self.windows)
