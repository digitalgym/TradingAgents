"""
Configuration for XGBoost Quant Strategy System.

Central config for feature windows, model parameters, risk settings,
and strategy defaults. Override per-pair at deployment via parameter tuner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# Base paths
XGB_QUANT_DIR = Path(__file__).parent
MODELS_DIR = XGB_QUANT_DIR / "models"
FEATURE_CACHE_DIR = XGB_QUANT_DIR / "feature_cache"
RESULTS_DIR = XGB_QUANT_DIR / "results"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
FEATURE_CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Feature window defaults
# ---------------------------------------------------------------------------

@dataclass
class FeatureWindows:
    """Configurable window sizes for technical indicators."""
    short: int = 10
    mid: int = 20
    long: int = 50
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_short: int = 20
    ema_long: int = 50
    volume_avg_period: int = 20
    stoch_period: int = 14
    donchian_period: int = 20


# ---------------------------------------------------------------------------
# XGBoost default hyperparameters per strategy
# ---------------------------------------------------------------------------

STRATEGY_XGB_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "trend_following": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "mean_reversion": {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.7,
        "min_child_weight": 10,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "breakout": {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 3,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "smc_zones": {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 1.5,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "volume_profile_strat": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "min_child_weight": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
}


# ---------------------------------------------------------------------------
# Risk / execution defaults
# ---------------------------------------------------------------------------

@dataclass
class RiskDefaults:
    """Default risk parameters — tuned per pair by parameter tuner."""
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    signal_threshold: float = 0.60
    max_hold_bars: int = 20


# ---------------------------------------------------------------------------
# Walk-forward training defaults
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Walk-forward training configuration."""
    train_window: int = 500
    test_window: int = 100
    min_train_bars: int = 200     # Minimum bars after indicator warmup
    warmup_bars: int = 60         # Bars needed for indicator warmup (longest EMA)
    total_bars: int = 2000        # How many bars to fetch from MT5


# ---------------------------------------------------------------------------
# Scanner defaults
# ---------------------------------------------------------------------------

@dataclass
class ScannerConfig:
    """Pair scanner configuration."""
    scan_timeframe: str = "H4"
    min_momentum_score: int = 40
    min_adx: float = 20.0
    max_spread_atr_ratio: float = 0.15
    atr_expansion_threshold: float = 1.2
    lookback_bars: int = 200


# ---------------------------------------------------------------------------
# Strategy selector regime suitability (human priors)
# ---------------------------------------------------------------------------

REGIME_SUITABILITY: Dict[str, Dict[str, float]] = {
    "trend_following": {
        "trending-up": 0.9, "trending-down": 0.9, "ranging": 0.2,
    },
    "mean_reversion": {
        "trending-up": 0.2, "trending-down": 0.2, "ranging": 0.9,
    },
    "breakout": {
        "trending-up": 0.6, "trending-down": 0.6, "ranging": 0.7,
    },
    "smc_zones": {
        "trending-up": 0.8, "trending-down": 0.8, "ranging": 0.5,
    },
    "volume_profile_strat": {
        "trending-up": 0.5, "trending-down": 0.5, "ranging": 0.8,
    },
}


# ---------------------------------------------------------------------------
# Default watchlist for scanner
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: List[str] = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # Crosses
    "EURJPY", "GBPJPY", "EURGBP", "AUDNZD", "CADJPY", "EURAUD",
    # Metals
    "XAUUSD", "XAGUSD",
    # Crypto
    "BTCUSD", "ETHUSD",
]


# ---------------------------------------------------------------------------
# Cold start pair-strategy defaults (before backtest data exists)
# ---------------------------------------------------------------------------

PAIR_STRATEGY_DEFAULTS: Dict[str, str] = {
    "XAUUSD": "trend_following",
    "GBPJPY": "trend_following",
    "EURJPY": "trend_following",
    "BTCUSD": "volume_profile_strat",
    "ETHUSD": "volume_profile_strat",
    "EURGBP": "mean_reversion",
    "AUDNZD": "mean_reversion",
    "USDCHF": "mean_reversion",
    "GBPUSD": "breakout",
}
