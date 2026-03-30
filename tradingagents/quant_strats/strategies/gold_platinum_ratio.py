"""
Gold/Platinum Ratio Mean-Reversion Strategy (Rule-Based)

Trades platinum based on the XAUUSD/XPTUSD ratio. When the ratio is
historically high (gold expensive vs platinum), platinum is undervalued
and we go long platinum expecting mean reversion.

Economic logic: Gold and platinum share industrial/precious metal demand.
When the ratio stretches beyond historical norms, it tends to revert.
Historical mean is ~1.0-1.2, sustained >1.8 is extreme.

Entry rules (long platinum):
  1. Gold/Platinum ratio > 1.8 (platinum cheap vs gold)
  2. Ratio is falling (starting to revert — don't catch falling knife)
  3. RSI on platinum < 70 (room to move up)
  4. Platinum price at or below 20 EMA (good entry point)

Entry rules (short platinum):
  1. Gold/Platinum ratio < 1.0 (platinum expensive vs gold)
  2. Ratio is rising (starting to revert)
  3. RSI on platinum > 30 (room to move down)

Exit:
  SL: 2x ATR (wider for platinum)
  TP: ratio-based or 2:1 R:R
  Max hold: 25 bars (ratio reversion takes time)

Note: This strategy computes features from XPTUSD data but loads XAUUSD
data internally to compute the ratio. During backtesting, the gold data
is loaded once and cached.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows

logger = logging.getLogger(__name__)

# Cache gold data at module level to avoid re-fetching on every bar
_gold_data_cache: Optional[pd.DataFrame] = None
_gold_timeframe_cache: Optional[str] = None


def _load_gold_data(timeframe: str, bars: int = 5000) -> Optional[pd.DataFrame]:
    """Load XAUUSD data from MT5, cached."""
    global _gold_data_cache, _gold_timeframe_cache
    if _gold_data_cache is not None and _gold_timeframe_cache == timeframe:
        return _gold_data_cache
    try:
        from tradingagents.automation.auto_tuner import load_mt5_data
        _gold_data_cache = load_mt5_data("XAUUSD", timeframe, bars)
        _gold_timeframe_cache = timeframe
        logger.info(f"Loaded {len(_gold_data_cache)} bars of XAUUSD {timeframe} for ratio computation")
        return _gold_data_cache
    except Exception as e:
        logger.warning(f"Failed to load XAUUSD data for ratio: {e}")
        return None


class GoldPlatinumRatioFeatures(TechnicalFeatures):
    """Technical features + gold/platinum ratio features."""

    def __init__(self, windows: Optional[FeatureWindows] = None, timeframe: str = "H4"):
        super().__init__(windows=windows)
        self._timeframe = timeframe

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "gp_ratio",             # Gold/Platinum price ratio
            "gp_ratio_sma20",       # 20-period SMA of ratio
            "gp_ratio_zscore",      # Z-score of ratio (20-period)
            "gp_ratio_above_1_8",   # Binary: ratio > 1.8 (platinum cheap)
            "gp_ratio_below_1_0",   # Binary: ratio < 1.0 (platinum expensive)
            "gp_ratio_falling",     # Binary: ratio SMA trending down (reverting from high)
            "gp_ratio_rising",      # Binary: ratio SMA trending up (reverting from low)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        plat_close = df["close"].values.astype(float)

        # Load gold data and align by timestamp
        gold_df = _load_gold_data(self._timeframe, len(df) + 500)

        if gold_df is not None and len(gold_df) > 0:
            # Align gold closes to platinum timestamps
            gold_close = self._align_gold_to_platinum(df, gold_df)
        else:
            # No gold data — fill with NaN
            gold_close = np.full(len(plat_close), np.nan)

        # Gold/Platinum ratio
        ratio = self._safe_divide(gold_close, plat_close, fill=np.nan)
        ratio_series = pd.Series(ratio)

        features["gp_ratio"] = ratio

        # Ratio moving average and z-score
        sma20 = ratio_series.rolling(20).mean()
        std20 = ratio_series.rolling(20).std()
        features["gp_ratio_sma20"] = sma20.values
        features["gp_ratio_zscore"] = ((ratio_series - sma20) / (std20 + 1e-10)).values

        # Threshold signals
        features["gp_ratio_above_1_8"] = (ratio > 1.8).astype(float)
        features["gp_ratio_below_1_0"] = (ratio < 1.0).astype(float)

        # Ratio direction (is it reverting?)
        ratio_sma5 = ratio_series.rolling(5).mean()
        ratio_sma5_prev = ratio_sma5.shift(1)
        features["gp_ratio_falling"] = (ratio_sma5 < ratio_sma5_prev).astype(float).values
        features["gp_ratio_rising"] = (ratio_sma5 > ratio_sma5_prev).astype(float).values

        return features

    @staticmethod
    def _align_gold_to_platinum(plat_df: pd.DataFrame, gold_df: pd.DataFrame) -> np.ndarray:
        """Align gold close prices to platinum timestamps using nearest match."""
        n = len(plat_df)
        gold_close = np.full(n, np.nan)

        # If both have datetime indices, merge on nearest timestamp
        if hasattr(plat_df.index, 'hour') and hasattr(gold_df.index, 'hour'):
            gold_series = gold_df["close"]
            for i in range(n):
                ts = plat_df.index[i]
                idx = gold_series.index.get_indexer([ts], method="nearest")[0]
                if 0 <= idx < len(gold_series):
                    gold_close[i] = gold_series.iloc[idx]
        elif "time" in plat_df.columns and "time" in gold_df.columns:
            # Fallback: merge on time column
            gold_series = gold_df.set_index("time")["close"]
            for i in range(n):
                ts = plat_df["time"].iloc[i] if "time" in plat_df.columns else plat_df.index[i]
                idx = gold_series.index.get_indexer([ts], method="nearest")[0]
                if 0 <= idx < len(gold_series):
                    gold_close[i] = gold_series.iloc[idx]
        else:
            # Last resort: assume same ordering, trim to shorter
            min_len = min(n, len(gold_df))
            gold_close[-min_len:] = gold_df["close"].values[-min_len:].astype(float)

        return gold_close


class GoldPlatinumRatioStrategy(BaseStrategy):

    def __init__(self, risk=None, windows=None, timeframe: str = "H4"):
        super().__init__(risk=risk, windows=windows)
        self._timeframe = timeframe

    @property
    def name(self) -> str:
        return "gold_platinum_ratio"

    def get_feature_set(self) -> BaseFeatureSet:
        return GoldPlatinumRatioFeatures(windows=self.windows, timeframe=self._timeframe)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        ratio = row.get("gp_ratio", np.nan)
        if np.isnan(ratio):
            return 0.0

        # --- Long platinum: ratio > 1.8 (platinum cheap vs gold) ---
        above_1_8 = row.get("gp_ratio_above_1_8", 0.0)
        below_1_0 = row.get("gp_ratio_below_1_0", 0.0)

        if not np.isnan(above_1_8) and above_1_8 > 0.5:
            direction = 1.0  # Long platinum (expect ratio to fall)

            # Core signal: ratio is extreme
            score += 0.35 * direction

            # Ratio must be reverting (falling) — don't catch falling knife
            falling = row.get("gp_ratio_falling", np.nan)
            if not np.isnan(falling) and falling > 0.5:
                score += 0.25 * direction  # Ratio starting to drop — good
            else:
                score *= 0.3  # Ratio still rising — very weak signal

        elif not np.isnan(below_1_0) and below_1_0 > 0.5:
            direction = -1.0  # Short platinum (expect ratio to rise)

            score += 0.35 * direction

            rising = row.get("gp_ratio_rising", np.nan)
            if not np.isnan(rising) and rising > 0.5:
                score += 0.25 * direction
            else:
                score *= 0.3

        else:
            return 0.0  # Ratio in normal range — no signal

        # --- RSI filter ---
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if direction > 0 and rsi > 70:
                return 0.0  # Overbought — don't long
            elif direction < 0 and rsi < 30:
                return 0.0  # Oversold — don't short

            if direction > 0 and rsi < 50:
                score += 0.20 * direction  # Good entry zone for long
            elif direction < 0 and rsi > 50:
                score += 0.20 * direction

        # --- Price near EMA20 (good entry, not chasing) ---
        dist_ema = row.get("ema_short_dist", np.nan)
        if not np.isnan(dist_ema):
            if abs(dist_ema) < 1.0:
                score += 0.15 * direction  # Near EMA — good entry
            elif direction > 0 and dist_ema < -0.5:
                score += 0.10 * direction  # Below EMA in long setup

        # --- Z-score intensity (how extreme is the ratio?) ---
        zscore = row.get("gp_ratio_zscore", np.nan)
        if not np.isnan(zscore):
            if direction > 0 and zscore > 1.5:
                score += 0.05 * direction  # Very extreme — extra conviction
            elif direction < 0 and zscore < -1.5:
                score += 0.05 * direction

        return float(np.clip(score, -1.0, 1.0))
