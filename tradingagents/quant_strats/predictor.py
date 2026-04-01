"""
Live Predictor — loads trained models and generates signals.

Used by the API endpoint and automation loop for real-time predictions.
Sub-100ms inference time (no LLM calls).
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import asdict

import numpy as np
import pandas as pd

from tradingagents.quant_strats.config import MODELS_DIR, RiskDefaults, FeatureWindows
from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal
from tradingagents.quant_strats.ensemble import StrategyEnsemble

logger = logging.getLogger(__name__)

# Strategy registry
STRATEGY_CLASSES = {}


def _ensure_registry():
    """Lazy-load strategy classes to avoid circular imports."""
    if STRATEGY_CLASSES:
        return
    from tradingagents.quant_strats.strategies.trend_following import TrendFollowingStrategy
    from tradingagents.quant_strats.strategies.mean_reversion import MeanReversionStrategy
    from tradingagents.quant_strats.strategies.breakout import BreakoutStrategy
    from tradingagents.quant_strats.strategies.smc_zones import SMCZonesStrategy
    from tradingagents.quant_strats.strategies.volume_profile_strat import VolumeProfileStrategy
    from tradingagents.quant_strats.strategies.keltner_mean_reversion import KeltnerMeanReversionStrategy
    from tradingagents.quant_strats.strategies.copper_ema_pullback import CopperEMAPullbackStrategy
    from tradingagents.quant_strats.strategies.gold_platinum_ratio import GoldPlatinumRatioStrategy
    from tradingagents.quant_strats.strategies.donchian_breakout import DonchianBreakoutStrategy
    from tradingagents.quant_strats.strategies.flag_continuation import FlagContinuationStrategy

    STRATEGY_CLASSES.update({
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
        "smc_zones": SMCZonesStrategy,
        "volume_profile_strat": VolumeProfileStrategy,
        "keltner_mean_reversion": KeltnerMeanReversionStrategy,
        "copper_ema_pullback": CopperEMAPullbackStrategy,
        "gold_platinum_ratio": GoldPlatinumRatioStrategy,
        "donchian_breakout": DonchianBreakoutStrategy,
        "flag_continuation": FlagContinuationStrategy,
    })


class LivePredictor:
    """Loads trained models and generates live trading signals."""

    def __init__(self):
        self._loaded_strategies: Dict[str, BaseStrategy] = {}

    def load_strategy(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        risk: Optional[RiskDefaults] = None,
    ) -> bool:
        """
        Load a trained strategy model from disk.

        Returns True if loaded successfully, False if no model found.
        """
        _ensure_registry()

        cls = STRATEGY_CLASSES.get(strategy_name)
        if cls is None:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False

        strategy = cls(risk=risk)
        key = f"{strategy_name}_{symbol}_{timeframe}"

        if strategy.load_model(symbol, timeframe):
            self._loaded_strategies[key] = strategy
            return True
        return False

    def predict_single(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
    ) -> Signal:
        """
        Get signal from a single strategy.

        Args:
            strategy_name: Name of strategy
            symbol: Symbol being analysed
            timeframe: Timeframe
            df: OHLCV DataFrame (needs at least 200 bars for feature warmup)
            current_price: Current market price
            atr: Current ATR value
        """
        key = f"{strategy_name}_{symbol}_{timeframe}"
        strategy = self._loaded_strategies.get(key)

        if strategy is None:
            if not self.load_strategy(strategy_name, symbol, timeframe):
                return Signal(direction="HOLD", confidence=0.0,
                              rationale=f"No trained model for {strategy_name} on {symbol}")
            strategy = self._loaded_strategies[key]

        feature_set = strategy.get_feature_set()
        features = feature_set.compute(df)

        # Pass price arrays for strategies with regime gates (e.g. mean_reversion)
        if hasattr(strategy, "check_regime_gate"):
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            close = df["close"].values.astype(float)
            return strategy.predict_signal(
                features, atr, current_price,
                high=high, low=low, close=close, symbol=symbol,
            )

        return strategy.predict_signal(features, atr, current_price)

    def predict_ensemble(
        self,
        strategy_names: List[str],
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        min_agree: int = 2,
        min_prob: float = 0.60,
    ) -> Signal:
        """
        Get ensemble signal from multiple strategies.

        Loads all requested strategies and combines their predictions.
        """
        strategies = []

        for name in strategy_names:
            key = f"{name}_{symbol}_{timeframe}"
            if key not in self._loaded_strategies:
                self.load_strategy(name, symbol, timeframe)

            strategy = self._loaded_strategies.get(key)
            if strategy is not None:
                strategies.append(strategy)

        if len(strategies) < min_agree:
            return Signal(
                direction="HOLD", confidence=0.0,
                rationale=f"Only {len(strategies)} strategies loaded, need {min_agree}",
            )

        ensemble = StrategyEnsemble(strategies, min_agree=min_agree, min_prob=min_prob)
        return ensemble.predict(df, current_price, atr)

    def get_available_models(self, symbol: str, timeframe: str) -> List[str]:
        """Check which trained models exist for a symbol/timeframe."""
        _ensure_registry()
        available = []
        for name in STRATEGY_CLASSES:
            path = MODELS_DIR / name / f"{symbol}_{timeframe}.json"
            if path.exists():
                available.append(name)
        return available

    def signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convert Signal to dict for API response."""
        return asdict(signal)
