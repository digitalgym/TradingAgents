"""
Session / Killzone Features for XGBoost.

Encodes the trading session (Asian, London, NY, overlaps) as numerical
features so the model can learn session-dependent behavior.

Session definitions mirror dataflows/smc_utils.py:
  Asian:              00:00 - 07:00 UTC
  Asian/London:       07:00 - 09:00 UTC
  London:             09:00 - 12:00 UTC
  London/NY overlap:  12:00 - 16:00 UTC  (highest volatility)
  New York:           16:00 - 21:00 UTC
  Late NY:            21:00 - 00:00 UTC
"""

import math
import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows


# Session boundaries (UTC hours) — start inclusive, end exclusive
_SESSIONS = {
    "asian":              (0, 7),
    "asian_london":       (7, 9),
    "london":             (9, 12),
    "london_ny_overlap":  (12, 16),
    "new_york":           (16, 21),
    "late_ny":            (21, 24),
}

# Which sessions count as "killzone" (high-probability SMC windows)
_KILLZONE_SESSIONS = {"london", "london_ny_overlap", "new_york"}

# Session open hours (for hours_since_session_open)
_SESSION_OPEN = {
    "asian": 0,
    "asian_london": 7,
    "london": 9,
    "london_ny_overlap": 12,
    "new_york": 16,
    "late_ny": 21,
}


def _hour_to_session(hour: int) -> str:
    """Map a UTC hour (0-23) to session name."""
    for name, (start, end) in _SESSIONS.items():
        if start <= hour < end:
            return name
    return "late_ny"


class SessionFeatures(BaseFeatureSet):
    """Trading session and killzone features."""

    @property
    def feature_names(self) -> list:
        return [
            "session_asian",
            "session_london",
            "session_ny",
            "session_london_ny_overlap",
            "is_killzone",
            "hours_since_session_open",
            "day_of_week_sin",
            "day_of_week_cos",
        ]

    @property
    def warmup_bars(self) -> int:
        return 0  # No lookback needed — pure timestamp features

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute session features for every bar."""
        features = pd.DataFrame(index=df.index, columns=self.feature_names, dtype=float)
        features[:] = 0.0

        hours = self._extract_utc_hours(df)
        days = self._extract_day_of_week(df)

        for i in range(len(df)):
            h = hours[i]
            d = days[i]

            if np.isnan(h):
                continue

            hour = int(h)
            session = _hour_to_session(hour)

            # Binary session flags
            features.iloc[i, features.columns.get_loc("session_asian")] = 1.0 if session in ("asian", "asian_london") else 0.0
            features.iloc[i, features.columns.get_loc("session_london")] = 1.0 if session in ("london", "asian_london") else 0.0
            features.iloc[i, features.columns.get_loc("session_ny")] = 1.0 if session in ("new_york", "london_ny_overlap") else 0.0
            features.iloc[i, features.columns.get_loc("session_london_ny_overlap")] = 1.0 if session == "london_ny_overlap" else 0.0
            features.iloc[i, features.columns.get_loc("is_killzone")] = 1.0 if session in _KILLZONE_SESSIONS else 0.0

            # Hours since current session opened
            session_open = _SESSION_OPEN.get(session, 0)
            hours_in = (hour - session_open) % 24
            features.iloc[i, features.columns.get_loc("hours_since_session_open")] = hours_in

            # Cyclical day-of-week encoding (Mon=0 .. Fri=4)
            if not np.isnan(d):
                day_frac = d / 5.0  # Normalize to [0, 1) for trading week
                features.iloc[i, features.columns.get_loc("day_of_week_sin")] = math.sin(2 * math.pi * day_frac)
                features.iloc[i, features.columns.get_loc("day_of_week_cos")] = math.cos(2 * math.pi * day_frac)

        return features

    def compute_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute session features for the latest bar only."""
        features = pd.DataFrame(index=[df.index[-1]], columns=self.feature_names, dtype=float)
        features[:] = 0.0

        hours = self._extract_utc_hours(df)
        days = self._extract_day_of_week(df)

        h = hours[-1]
        d = days[-1]

        if not np.isnan(h):
            hour = int(h)
            session = _hour_to_session(hour)

            features.iloc[0, features.columns.get_loc("session_asian")] = 1.0 if session in ("asian", "asian_london") else 0.0
            features.iloc[0, features.columns.get_loc("session_london")] = 1.0 if session in ("london", "asian_london") else 0.0
            features.iloc[0, features.columns.get_loc("session_ny")] = 1.0 if session in ("new_york", "london_ny_overlap") else 0.0
            features.iloc[0, features.columns.get_loc("session_london_ny_overlap")] = 1.0 if session == "london_ny_overlap" else 0.0
            features.iloc[0, features.columns.get_loc("is_killzone")] = 1.0 if session in _KILLZONE_SESSIONS else 0.0

            session_open = _SESSION_OPEN.get(session, 0)
            features.iloc[0, features.columns.get_loc("hours_since_session_open")] = (hour - session_open) % 24

            if not np.isnan(d):
                day_frac = d / 5.0
                features.iloc[0, features.columns.get_loc("day_of_week_sin")] = math.sin(2 * math.pi * day_frac)
                features.iloc[0, features.columns.get_loc("day_of_week_cos")] = math.cos(2 * math.pi * day_frac)

        return features

    @staticmethod
    def _extract_utc_hours(df: pd.DataFrame) -> np.ndarray:
        """Extract UTC hour from the DataFrame index or a time column."""
        idx = df.index
        if hasattr(idx, "hour"):
            return idx.hour.values.astype(float)
        # Try to parse as datetime
        try:
            dt_index = pd.to_datetime(idx, utc=True)
            return dt_index.hour.values.astype(float)
        except Exception:
            pass
        # Check for a 'time' column
        if "time" in df.columns:
            try:
                times = pd.to_datetime(df["time"], utc=True)
                return times.dt.hour.values.astype(float)
            except Exception:
                pass
        return np.full(len(df), np.nan)

    @staticmethod
    def _extract_day_of_week(df: pd.DataFrame) -> np.ndarray:
        """Extract day of week (Monday=0) from DataFrame index."""
        idx = df.index
        if hasattr(idx, "dayofweek"):
            return idx.dayofweek.values.astype(float)
        try:
            dt_index = pd.to_datetime(idx, utc=True)
            return dt_index.dayofweek.values.astype(float)
        except Exception:
            pass
        if "time" in df.columns:
            try:
                times = pd.to_datetime(df["time"], utc=True)
                return times.dt.dayofweek.values.astype(float)
            except Exception:
                pass
        return np.full(len(df), np.nan)
