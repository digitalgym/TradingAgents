"""Feature engineering modules for XGBoost strategies."""

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.volume_profile import VolumeProfileFeatures
from tradingagents.quant_strats.features.regime import RegimeFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures

__all__ = [
    "BaseFeatureSet",
    "TechnicalFeatures",
    "SMCFeatures",
    "VolumeProfileFeatures",
    "RegimeFeatures",
    "CompositeFeatures",
]
