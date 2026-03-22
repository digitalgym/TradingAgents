"""
Volume Profile Strategy

Uses XGBoost to learn price reactions at Volume Profile levels
(POC, VAH, VAL, HVN, LVN) combined with technical context.

Best for volume-heavy instruments (BTCUSD, indices).
"""

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.features.volume_profile import VolumeProfileFeatures
from tradingagents.xgb_quant.features.composite import CompositeFeatures


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
