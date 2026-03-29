"""
XGBoost Quant Strategy System

A library of XGBoost-based trading strategies that learn from historical data.
Runs alongside existing LLM-based quant pipelines as new pipeline types.
"""

from tradingagents.xgb_quant.config import (
    ScannerConfig,
    TrainingConfig,
    RiskDefaults,
    FeatureWindows,
    MODELS_DIR,
    RESULTS_DIR,
    DEFAULT_WATCHLIST,
    PAIR_STRATEGY_DEFAULTS,
    REGIME_SUITABILITY,
)
from tradingagents.xgb_quant.scanner import PairScanner, PairScore, ScanResult
from tradingagents.xgb_quant.predictor import LivePredictor
from tradingagents.xgb_quant.trainer import WalkForwardTrainer
from tradingagents.xgb_quant.ensemble import StrategyEnsemble
from tradingagents.xgb_quant.strategy_selector import StrategySelector
from tradingagents.xgb_quant.parameter_tuner import ParameterTuner

__all__ = [
    # Config
    "ScannerConfig",
    "TrainingConfig",
    "RiskDefaults",
    "FeatureWindows",
    "MODELS_DIR",
    "RESULTS_DIR",
    "DEFAULT_WATCHLIST",
    "PAIR_STRATEGY_DEFAULTS",
    "REGIME_SUITABILITY",
    # Scanner
    "PairScanner",
    "PairScore",
    "ScanResult",
    # Inference
    "LivePredictor",
    "StrategyEnsemble",
    # Training
    "WalkForwardTrainer",
    "ParameterTuner",
    # Selection
    "StrategySelector",
]
