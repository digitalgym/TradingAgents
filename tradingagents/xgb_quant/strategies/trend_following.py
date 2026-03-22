"""
Trend Following Strategy

Captures sustained directional moves using EMA crossovers,
ADX trend strength, RSI momentum, and rate of change.
Best suited for trending, volatile pairs (XAUUSD, GBPJPY, EURJPY).
"""

from typing import Optional, Dict, Any

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.config import FeatureWindows, RiskDefaults


class TrendFollowingStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "trend_following"

    def get_feature_set(self) -> BaseFeatureSet:
        return TechnicalFeatures(windows=self.windows)
