"""
SMC Zones Strategy

Uses XGBoost to learn which Smart Money Concept zones actually hold
and which fail, based on numerical SMC features + technical context.

Better than LLM interpretation because it learns from hundreds of
historical zone interactions, not single-shot analysis.
"""

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.features.smc import SMCFeatures
from tradingagents.xgb_quant.features.composite import CompositeFeatures


class SMCZonesStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "smc_zones"

    def get_feature_set(self) -> BaseFeatureSet:
        return CompositeFeatures(
            providers=[
                TechnicalFeatures(windows=self.windows),
                SMCFeatures(windows=self.windows),
            ],
            windows=self.windows,
        )
