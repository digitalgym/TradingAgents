"""
Base Feature Set — abstract class for all feature providers.

All feature sets take a DataFrame with OHLCV columns and return
a DataFrame with computed feature columns. Features are normalised
so they're comparable across symbols with different price scales.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from tradingagents.quant_strats.config import FeatureWindows


class BaseFeatureSet(ABC):
    """Abstract base class for feature sets."""

    def __init__(self, windows: Optional[FeatureWindows] = None):
        self.windows = windows or FeatureWindows()

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Must have these column names (lowercase).

        Returns:
            DataFrame with feature columns added (same index as input).
            Original OHLCV columns are NOT included — only feature columns.
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list:
        """Return list of feature column names this set produces."""
        ...

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before features are valid (non-NaN)."""
        return max(self.windows.ema_long, self.windows.bb_period,
                   self.windows.adx_period + 14, 60)

    @staticmethod
    def _safe_divide(a, b, fill: float = 0.0):
        """Division that handles zero/NaN denominators."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(b) > 1e-10, a / b, fill)
        return result
