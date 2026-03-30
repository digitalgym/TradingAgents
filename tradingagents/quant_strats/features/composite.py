"""
Composite Feature Set — combines multiple feature providers into one.

Used to build strategy-specific feature sets by selecting which
providers to include.
"""

import pandas as pd
from typing import List, Optional

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows


class CompositeFeatures(BaseFeatureSet):
    """Combines multiple feature sets into a single DataFrame."""

    def __init__(self, providers: List[BaseFeatureSet],
                 windows: Optional[FeatureWindows] = None):
        super().__init__(windows)
        self.providers = providers

    @property
    def feature_names(self) -> list:
        names = []
        for provider in self.providers:
            names.extend(provider.feature_names)
        return names

    @property
    def warmup_bars(self) -> int:
        return max(p.warmup_bars for p in self.providers)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all provider features and concatenate."""
        result = pd.DataFrame(index=df.index)
        for provider in self.providers:
            provider_features = provider.compute(df)
            result = pd.concat([result, provider_features], axis=1)
        return result
