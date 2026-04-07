"""Rule-based trading strategies."""

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.strategies.trend_following import TrendFollowingStrategy
from tradingagents.quant_strats.strategies.mean_reversion import MeanReversionStrategy
from tradingagents.quant_strats.strategies.breakout import BreakoutStrategy
from tradingagents.quant_strats.strategies.smc_zones import SMCZonesStrategy
from tradingagents.quant_strats.strategies.volume_profile_strat import VolumeProfileStrategy
from tradingagents.quant_strats.strategies.keltner_mean_reversion import KeltnerMeanReversionStrategy
from tradingagents.quant_strats.strategies.copper_ema_pullback import CopperEMAPullbackStrategy
from tradingagents.quant_strats.strategies.gold_platinum_ratio import GoldPlatinumRatioStrategy
from tradingagents.quant_strats.strategies.smc_confluence_ml import SMCConfluenceMLStrategy
from tradingagents.quant_strats.strategies.smc_sweep_ml import SMCSweepMLStrategy
from tradingagents.quant_strats.strategies.gold_silver_ratio import GoldSilverRatioStrategy

__all__ = [
    "BaseStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "SMCZonesStrategy",
    "VolumeProfileStrategy",
    "KeltnerMeanReversionStrategy",
    "CopperEMAPullbackStrategy",
    "GoldPlatinumRatioStrategy",
    "SMCConfluenceMLStrategy",
    "SMCSweepMLStrategy",
    "GoldSilverRatioStrategy",
]
