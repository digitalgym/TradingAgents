"""
Configuration for Rule-Based Quant Strategy System.

Central config for feature windows, risk settings, and strategy defaults.
Override per-pair at deployment via parameter tuner.
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
# Risk / execution defaults
# ---------------------------------------------------------------------------

@dataclass
class RiskDefaults:
    """Default risk parameters — optimised via Optuna on XAUUSD H4 breakout."""
    sl_atr_mult: float = 1.39
    tp_atr_mult: float = 2.77
    signal_threshold: float = 0.625
    max_hold_bars: int = 25


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
    # Minimum trades required for a backtest result to be considered valid
    min_trades: int = 30
    # Overfitting detection: max allowed gap between train and test win rates
    max_train_test_gap: float = 0.20  # 20 percentage points
    # Overfitting detection: min fold-to-fold consistency (std of fold win rates)
    max_fold_wr_std: float = 0.25     # Reject if fold win rates vary wildly


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
    "donchian_breakout": {
        "trending-up": 0.6, "trending-down": 0.6, "ranging": 0.7,
    },
    "smc_zones": {
        "trending-up": 0.8, "trending-down": 0.8, "ranging": 0.5,
    },
    "volume_profile_strat": {
        "trending-up": 0.5, "trending-down": 0.5, "ranging": 0.8,
    },
    "keltner_mean_reversion": {
        "trending-up": 0.1, "trending-down": 0.1, "ranging": 0.95,
    },
    "copper_ema_pullback": {
        "trending-up": 0.85, "trending-down": 0.85, "ranging": 0.3,
    },
    "gold_platinum_ratio": {
        "trending-up": 0.5, "trending-down": 0.5, "ranging": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Keltner Mean-Reversion default risk params (copper/platinum)
# ---------------------------------------------------------------------------

KELTNER_MR_DEFAULTS = RiskDefaults(
    sl_atr_mult=1.6,
    tp_atr_mult=2.0,       # Fallback if midline target not available
    signal_threshold=0.60,
    max_hold_bars=18,
)


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
    "GBPUSD": "donchian_breakout",
    "COPPER-C": "copper_ema_pullback",
    "XPTUSD": "gold_platinum_ratio",
}


# ---------------------------------------------------------------------------
# Strategy-specific risk defaults
# ---------------------------------------------------------------------------

COPPER_EMA_DEFAULTS = RiskDefaults(
    sl_atr_mult=1.5,       # Tighter for copper trend-following
    tp_atr_mult=3.0,       # 2:1 R:R minimum
    signal_threshold=0.60,
    max_hold_bars=20,
)

GOLD_PLAT_RATIO_DEFAULTS = RiskDefaults(
    sl_atr_mult=2.0,       # Wider for platinum (volatile)
    tp_atr_mult=4.0,       # Fallback if ratio target not available
    signal_threshold=0.60,
    max_hold_bars=25,       # Ratio reversion takes time
)


# ---------------------------------------------------------------------------
# FOMC/NFP news blackout dates (no new trades within 24hrs)
# Update this list periodically. Format: YYYY-MM-DD.
# ---------------------------------------------------------------------------

NEWS_BLACKOUT_DATES: List[str] = [
    # 2026 FOMC meeting dates (announcement day)
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    # 2026 NFP release dates (first Friday of month)
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]
