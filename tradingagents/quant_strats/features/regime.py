"""
Market Regime Features for XGBoost.

Wraps RegimeDetector to provide regime classification as numerical features.
"""

import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows


class RegimeFeatures(BaseFeatureSet):
    """Market regime features using existing RegimeDetector."""

    @property
    def feature_names(self) -> list:
        return [
            "regime_trending_up",
            "regime_trending_down",
            "regime_ranging",
            "vol_regime_low",
            "vol_regime_normal",
            "vol_regime_high",
            "vol_regime_extreme",
            "squeeze_strength",
            "breakout_ready",
        ]

    @property
    def warmup_bars(self) -> int:
        return 100

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime features for each bar."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()
        n = len(df)
        features = pd.DataFrame(index=df.index, columns=self.feature_names, dtype=float)
        features[:] = 0.0

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        lookback = 100

        for i in range(lookback, n):
            h = high[i - lookback:i + 1]
            l = low[i - lookback:i + 1]
            c = close[i - lookback:i + 1]

            try:
                regime = detector.get_full_regime(h, l, c)

                # One-hot encode market regime
                mr = regime.get("market_regime", "ranging")
                features.iloc[i, features.columns.get_loc("regime_trending_up")] = 1.0 if mr == "trending-up" else 0.0
                features.iloc[i, features.columns.get_loc("regime_trending_down")] = 1.0 if mr == "trending-down" else 0.0
                features.iloc[i, features.columns.get_loc("regime_ranging")] = 1.0 if mr == "ranging" else 0.0

                # One-hot encode volatility regime
                vr = regime.get("volatility_regime", "normal")
                features.iloc[i, features.columns.get_loc("vol_regime_low")] = 1.0 if vr == "low" else 0.0
                features.iloc[i, features.columns.get_loc("vol_regime_normal")] = 1.0 if vr == "normal" else 0.0
                features.iloc[i, features.columns.get_loc("vol_regime_high")] = 1.0 if vr == "high" else 0.0
                features.iloc[i, features.columns.get_loc("vol_regime_extreme")] = 1.0 if vr == "extreme" else 0.0

                # Consolidation / squeeze
                consol = detector.detect_consolidation(h, l, c)
                features.iloc[i, features.columns.get_loc("squeeze_strength")] = consol.get("squeeze_strength", 0) / 100.0
                features.iloc[i, features.columns.get_loc("breakout_ready")] = 1.0 if consol.get("breakout_ready", False) else 0.0

            except Exception:
                features.iloc[i, features.columns.get_loc("regime_ranging")] = 1.0
                features.iloc[i, features.columns.get_loc("vol_regime_normal")] = 1.0

        return features
