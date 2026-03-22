"""Feature engineering modules for XGBoost strategies."""

from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.features.smc import SMCFeatures
from tradingagents.xgb_quant.features.volume_profile import VolumeProfileFeatures
from tradingagents.xgb_quant.features.regime import RegimeFeatures
from tradingagents.xgb_quant.features.composite import CompositeFeatures

__all__ = [
    "BaseFeatureSet",
    "TechnicalFeatures",
    "SMCFeatures",
    "VolumeProfileFeatures",
    "RegimeFeatures",
    "CompositeFeatures",
]
