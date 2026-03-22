"""XGBoost trading strategies."""

from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.strategies.trend_following import TrendFollowingStrategy
from tradingagents.xgb_quant.strategies.mean_reversion import MeanReversionStrategy
from tradingagents.xgb_quant.strategies.breakout import BreakoutStrategy
from tradingagents.xgb_quant.strategies.smc_zones import SMCZonesStrategy
from tradingagents.xgb_quant.strategies.volume_profile_strat import VolumeProfileStrategy

__all__ = [
    "BaseStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "SMCZonesStrategy",
    "VolumeProfileStrategy",
]
