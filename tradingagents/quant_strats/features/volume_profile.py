"""
Volume Profile Features for XGBoost.

Wraps VolumeProfileAnalyzer to extract numerical features:
distance to POC/VAH/VAL, value area position, volume concentration.
"""

import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows


class VolumeProfileFeatures(BaseFeatureSet):
    """Volume Profile features for XGBoost."""

    def __init__(self, windows: Optional[FeatureWindows] = None, vp_lookback: int = 100):
        super().__init__(windows)
        self.vp_lookback = vp_lookback

    @property
    def feature_names(self) -> list:
        return [
            "poc_dist_atr",          # Distance to POC in ATR units
            "vah_dist_atr",          # Distance to VAH in ATR units
            "val_dist_atr",          # Distance to VAL in ATR units
            "value_area_position",   # 0=below VA, 0.5=inside, 1=above
            "poc_volume_pct",        # Volume concentration at POC
            "value_area_width_pct",  # Width of value area / price
            "hvn_count",             # Number of high volume nodes
            "lvn_count",             # Number of low volume nodes
            "nearest_hvn_dist_atr",  # Distance to nearest HVN
            "nearest_lvn_dist_atr",  # Distance to nearest LVN
        ]

    @property
    def warmup_bars(self) -> int:
        return self.vp_lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VP features for each bar using trailing window."""
        from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer

        analyzer = VolumeProfileAnalyzer()
        n = len(df)
        features = pd.DataFrame(index=df.index, columns=self.feature_names, dtype=float)
        features[:] = np.nan

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        atr = self._compute_atr_simple(high, low, close)

        for i in range(self.vp_lookback, n):
            window_df = df.iloc[i - self.vp_lookback:i + 1].copy()
            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else 1.0

            try:
                profile = analyzer.calculate_volume_profile(window_df, num_bins=50)
                self._extract_features(features, i, profile, current_price, current_atr)
            except Exception:
                pass

        return features

    def compute_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute VP features for latest bar only."""
        from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer

        analyzer = VolumeProfileAnalyzer()
        features = pd.DataFrame(index=[df.index[-1]], columns=self.feature_names, dtype=float)
        features[:] = np.nan

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        atr = self._compute_atr_simple(high, low, close)
        current_price = close[-1]
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 1.0

        start = max(0, len(df) - self.vp_lookback)
        window_df = df.iloc[start:].copy()

        try:
            profile = analyzer.calculate_volume_profile(window_df, num_bins=50)
            self._extract_features(features, 0, profile, current_price, current_atr)
        except Exception:
            pass

        return features

    def _extract_features(self, features: pd.DataFrame, idx: int,
                          profile, price: float, atr: float):
        """Extract numerical features from VolumeProfile dataclass."""
        safe_atr = max(atr, 1e-10)

        features.iloc[idx, features.columns.get_loc("poc_dist_atr")] = (price - profile.poc) / safe_atr
        features.iloc[idx, features.columns.get_loc("vah_dist_atr")] = (price - profile.value_area_high) / safe_atr
        features.iloc[idx, features.columns.get_loc("val_dist_atr")] = (price - profile.value_area_low) / safe_atr

        # Value area position: below=0, inside=0.5, above=1
        if price < profile.value_area_low:
            va_pos = 0.0
        elif price > profile.value_area_high:
            va_pos = 1.0
        else:
            va_range = profile.value_area_high - profile.value_area_low
            if va_range > 0:
                va_pos = 0.25 + 0.5 * (price - profile.value_area_low) / va_range
            else:
                va_pos = 0.5
        features.iloc[idx, features.columns.get_loc("value_area_position")] = va_pos

        features.iloc[idx, features.columns.get_loc("poc_volume_pct")] = profile.poc_volume_pct / 100.0 if profile.poc_volume_pct else 0.0

        price_range = profile.profile_high - profile.profile_low
        va_width = profile.value_area_high - profile.value_area_low
        features.iloc[idx, features.columns.get_loc("value_area_width_pct")] = va_width / max(price, 1e-10)

        features.iloc[idx, features.columns.get_loc("hvn_count")] = len(profile.high_volume_nodes)
        features.iloc[idx, features.columns.get_loc("lvn_count")] = len(profile.low_volume_nodes)

        # Nearest HVN/LVN distance
        if profile.high_volume_nodes:
            nearest_hvn = min(profile.high_volume_nodes, key=lambda n: abs(n.price - price))
            features.iloc[idx, features.columns.get_loc("nearest_hvn_dist_atr")] = (price - nearest_hvn.price) / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("nearest_hvn_dist_atr")] = 0.0

        if profile.low_volume_nodes:
            nearest_lvn = min(profile.low_volume_nodes, key=lambda n: abs(n.price - price))
            features.iloc[idx, features.columns.get_loc("nearest_lvn_dist_atr")] = (price - nearest_lvn.price) / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("nearest_lvn_dist_atr")] = 0.0

    @staticmethod
    def _compute_atr_simple(high, low, close, period: int = 14) -> np.ndarray:
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
