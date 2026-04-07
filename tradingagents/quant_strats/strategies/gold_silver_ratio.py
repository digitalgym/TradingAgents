"""
Gold-Silver Ratio (GSR) Mean-Reversion Strategy (Rule-Based)

Trades based on the Gold/Silver ratio — one of the oldest relative-value
tools in financial markets. When the ratio reaches historical extremes,
it reverts toward its long-run average of 50-65.

Formula: GSR = XAUUSD price / XAGUSD price

Historical reference levels:
  < 35     Silver extremely expensive vs gold (rare)
  35-50    Silver relatively expensive
  50-65    Fair value / long-run average
  65-80    Silver becoming cheap (mild opportunity)
  80-100   Silver historically undervalued (strong signal)
  > 100    Extreme — every time it crossed 100, silver bounced hard

This strategy trades XAGUSD by default (loads XAUUSD internally for
the ratio). Can also trade XAUUSD with inverted signals.

When trading XAGUSD:
  GSR > 80 and reverting down  -> BUY silver  (silver undervalued)
  GSR < 50 and reverting up    -> SELL silver (silver overvalued)

When trading XAUUSD:
  GSR > 80 and reverting down  -> SELL gold   (gold overvalued vs silver)
  GSR < 50 and reverting up    -> BUY gold    (gold undervalued vs silver)

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  GSR in extreme zone (>80 or <50)
  +0.25  Ratio reverting (SMA turning from extreme)
  +0.15  Z-score intensity (how far from mean)
  +0.10  RSI filter (room to move in signal direction)
  +0.10  Price near EMA (good entry, not chasing)
  +0.10  Rate of change confirming reversion momentum
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows, RiskDefaults

logger = logging.getLogger(__name__)

# Cache the "other" symbol's data at module level
_other_data_cache: Optional[pd.DataFrame] = None
_other_symbol_cache: Optional[str] = None
_other_timeframe_cache: Optional[str] = None


def _load_other_metal_data(
    symbol: str, timeframe: str, bars: int = 5000,
) -> Optional[pd.DataFrame]:
    """Load the counterpart metal data from MT5, cached."""
    global _other_data_cache, _other_symbol_cache, _other_timeframe_cache
    if (
        _other_data_cache is not None
        and _other_symbol_cache == symbol
        and _other_timeframe_cache == timeframe
    ):
        return _other_data_cache
    try:
        from tradingagents.automation.auto_tuner import load_mt5_data
        _other_data_cache = load_mt5_data(symbol, timeframe, bars)
        _other_symbol_cache = symbol
        _other_timeframe_cache = timeframe
        logger.info(
            f"Loaded {len(_other_data_cache)} bars of {symbol} {timeframe} "
            "for GSR computation"
        )
        return _other_data_cache
    except Exception as e:
        logger.warning(f"Failed to load {symbol} data for GSR: {e}")
        return None


class GoldSilverRatioFeatures(TechnicalFeatures):
    """Technical features + Gold/Silver ratio features."""

    def __init__(
        self,
        windows: Optional[FeatureWindows] = None,
        timeframe: str = "D1",
        trade_symbol: str = "XAGUSD",
    ):
        super().__init__(windows=windows)
        self._timeframe = timeframe
        self._trade_symbol = trade_symbol
        # We load whichever symbol we're NOT trading
        self._other_symbol = (
            "XAUUSD" if trade_symbol == "XAGUSD" else "XAGUSD"
        )

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "gsr",                  # Gold/Silver price ratio
            "gsr_sma20",            # 20-period SMA of ratio
            "gsr_sma50",            # 50-period SMA of ratio
            "gsr_zscore",           # Z-score of ratio (50-period)
            "gsr_percentile",       # Percentile rank over 200 bars
            "gsr_roc_10",           # Rate of change of ratio (10 bars)
            "gsr_above_80",         # Binary: ratio > 80 (silver cheap)
            "gsr_above_100",        # Binary: ratio > 100 (extreme)
            "gsr_below_50",         # Binary: ratio < 50 (silver expensive)
            "gsr_below_35",         # Binary: ratio < 35 (extreme)
            "gsr_in_fair_value",    # Binary: ratio 50-65 (neutral)
            "gsr_reverting_down",   # Ratio SMA trending down from high
            "gsr_reverting_up",     # Ratio SMA trending up from low
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        trade_close = df["close"].values.astype(float)

        # Load the other metal's data and align
        other_df = _load_other_metal_data(
            self._other_symbol, self._timeframe, len(df) + 500,
        )

        if other_df is not None and len(other_df) > 0:
            other_close = self._align_prices(df, other_df)
        else:
            other_close = np.full(len(trade_close), np.nan)

        # Compute GSR = gold / silver
        if self._trade_symbol == "XAGUSD":
            # We have silver (trade_close), loaded gold (other_close)
            gold = other_close
            silver = trade_close
        else:
            # We have gold (trade_close), loaded silver (other_close)
            gold = trade_close
            silver = other_close

        ratio = np.where(silver > 0, gold / silver, np.nan)
        ratio_series = pd.Series(ratio)

        features["gsr"] = ratio

        # Moving averages
        sma20 = ratio_series.rolling(20, min_periods=10).mean()
        sma50 = ratio_series.rolling(50, min_periods=20).mean()
        features["gsr_sma20"] = sma20.values
        features["gsr_sma50"] = sma50.values

        # Z-score (50-period)
        std50 = ratio_series.rolling(50, min_periods=20).std()
        features["gsr_zscore"] = ((ratio_series - sma50) / (std50 + 1e-10)).values

        # Percentile rank over 200 bars
        features["gsr_percentile"] = (
            ratio_series.rolling(200, min_periods=50).rank(pct=True).values
        )

        # Rate of change (10-bar)
        features["gsr_roc_10"] = ratio_series.pct_change(10).values

        # Threshold signals
        features["gsr_above_80"] = (ratio > 80).astype(float)
        features["gsr_above_100"] = (ratio > 100).astype(float)
        features["gsr_below_50"] = (ratio < 50).astype(float)
        features["gsr_below_35"] = (ratio < 35).astype(float)
        features["gsr_in_fair_value"] = ((ratio >= 50) & (ratio <= 65)).astype(float)

        # Reversion detection: SMA5 of ratio turning
        ratio_sma5 = ratio_series.rolling(5, min_periods=3).mean()
        ratio_sma5_prev = ratio_sma5.shift(1)
        ratio_sma10 = ratio_series.rolling(10, min_periods=5).mean()
        ratio_sma10_prev = ratio_sma10.shift(1)

        # Reverting down = both short and mid-term SMAs declining
        features["gsr_reverting_down"] = (
            (ratio_sma5 < ratio_sma5_prev) & (ratio_sma10 < ratio_sma10_prev)
        ).astype(float).values
        features["gsr_reverting_up"] = (
            (ratio_sma5 > ratio_sma5_prev) & (ratio_sma10 > ratio_sma10_prev)
        ).astype(float).values

        return features

    @staticmethod
    def _align_prices(
        target_df: pd.DataFrame, source_df: pd.DataFrame,
    ) -> np.ndarray:
        """Align source close prices to target timestamps using nearest match."""
        n = len(target_df)
        aligned = np.full(n, np.nan)

        if hasattr(target_df.index, "hour") and hasattr(source_df.index, "hour"):
            source_series = source_df["close"]
            for i in range(n):
                ts = target_df.index[i]
                idx = source_series.index.get_indexer([ts], method="nearest")[0]
                if 0 <= idx < len(source_series):
                    aligned[i] = source_series.iloc[idx]
        elif "time" in target_df.columns and "time" in source_df.columns:
            source_series = source_df.set_index("time")["close"]
            for i in range(n):
                ts = target_df["time"].iloc[i]
                idx = source_series.index.get_indexer([ts], method="nearest")[0]
                if 0 <= idx < len(source_series):
                    aligned[i] = source_series.iloc[idx]
        else:
            # Fallback: assume same ordering, trim to shorter
            min_len = min(n, len(source_df))
            aligned[-min_len:] = source_df["close"].values[-min_len:].astype(float)

        return aligned


class GoldSilverRatioStrategy(BaseStrategy):
    """
    Gold-Silver Ratio mean-reversion strategy.

    Trades silver (default) or gold based on GSR extremes.
    When GSR > 80, silver is undervalued -> buy silver / sell gold.
    When GSR < 50, silver is overvalued -> sell silver / buy gold.
    """

    # Default risk params — wider stops and longer hold for ratio reversion
    _DEFAULT_RISK = RiskDefaults(
        sl_atr_mult=2.0,       # Ratio reversion needs room
        tp_atr_mult=6.0,       # 3:1 R:R
        signal_threshold=0.55,  # Need decent conviction
        max_hold_bars=40,      # Ratio reversion is slow — be patient
    )

    def __init__(
        self,
        risk: Optional[RiskDefaults] = None,
        windows: Optional[FeatureWindows] = None,
        timeframe: str = "D1",
        trade_symbol: str = "XAGUSD",
    ):
        super().__init__(risk=risk or self._DEFAULT_RISK, windows=windows)
        self._timeframe = timeframe
        self._trade_symbol = trade_symbol

    @property
    def name(self) -> str:
        return "gold_silver_ratio"

    def get_feature_set(self) -> BaseFeatureSet:
        return GoldSilverRatioFeatures(
            windows=self.windows,
            timeframe=self._timeframe,
            trade_symbol=self._trade_symbol,
        )

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        gsr = row.get("gsr", np.nan)
        if np.isnan(gsr):
            return 0.0

        # Determine raw direction from ratio extremes
        # Positive = bullish for the traded symbol
        above_80 = row.get("gsr_above_80", 0.0)
        above_100 = row.get("gsr_above_100", 0.0)
        below_50 = row.get("gsr_below_50", 0.0)
        below_35 = row.get("gsr_below_35", 0.0)
        in_fair = row.get("gsr_in_fair_value", 0.0)

        # No signal in fair value zone (50-65)
        if not np.isnan(in_fair) and in_fair > 0.5:
            return 0.0

        # --- HIGH RATIO: Silver undervalued ---
        if not np.isnan(above_80) and above_80 > 0.5:
            # For XAGUSD: BUY silver (positive)
            # For XAUUSD: SELL gold (negative)
            if self._trade_symbol == "XAGUSD":
                direction = 1.0
            else:
                direction = -1.0

            # Core signal: ratio in extreme zone (30%)
            score += 0.30

            # Extra conviction for extreme >100 (+5%)
            if not np.isnan(above_100) and above_100 > 0.5:
                score += 0.05

            # GATE: Ratio must be reverting (falling from high)
            reverting = row.get("gsr_reverting_down", np.nan)
            if not np.isnan(reverting) and reverting > 0.5:
                score += 0.25  # Reversion confirmed
            else:
                score *= 0.25  # Not reverting yet — very weak, don't catch knife

        # --- LOW RATIO: Silver overvalued ---
        elif not np.isnan(below_50) and below_50 > 0.5:
            if self._trade_symbol == "XAGUSD":
                direction = -1.0  # SELL silver
            else:
                direction = 1.0   # BUY gold

            score += 0.30

            if not np.isnan(below_35) and below_35 > 0.5:
                score += 0.05

            reverting = row.get("gsr_reverting_up", np.nan)
            if not np.isnan(reverting) and reverting > 0.5:
                score += 0.25
            else:
                score *= 0.25

        # --- MILD ZONE: 65-80, weak opportunity ---
        elif gsr > 65:
            if self._trade_symbol == "XAGUSD":
                direction = 1.0
            else:
                direction = -1.0

            # Weaker core signal (15%)
            score += 0.15

            reverting = row.get("gsr_reverting_down", np.nan)
            if not np.isnan(reverting) and reverting > 0.5:
                score += 0.15
            else:
                return 0.0  # Mild zone + not reverting = no signal

        else:
            return 0.0  # Ratio 50-65 already caught above

        # --- Z-score intensity (15%) ---
        zscore = row.get("gsr_zscore", np.nan)
        if not np.isnan(zscore):
            abs_z = abs(zscore)
            if abs_z > 2.0:
                score += 0.15  # Very extreme
            elif abs_z > 1.5:
                score += 0.10
            elif abs_z > 1.0:
                score += 0.05

        # --- Rate of change confirming reversion (10%) ---
        roc = row.get("gsr_roc_10", np.nan)
        if not np.isnan(roc):
            if direction > 0 and self._trade_symbol == "XAGUSD":
                # Bullish silver = ratio falling = negative roc
                if roc < -0.02:
                    score += 0.10
            elif direction < 0 and self._trade_symbol == "XAGUSD":
                # Bearish silver = ratio rising = positive roc
                if roc > 0.02:
                    score += 0.10
            elif direction < 0 and self._trade_symbol == "XAUUSD":
                # Bearish gold = ratio falling
                if roc < -0.02:
                    score += 0.10
            elif direction > 0 and self._trade_symbol == "XAUUSD":
                # Bullish gold = ratio rising
                if roc > 0.02:
                    score += 0.10

        # --- RSI filter (10%) ---
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if direction > 0 and rsi > 75:
                return 0.0  # Overbought — don't long
            elif direction < 0 and rsi < 25:
                return 0.0  # Oversold — don't short

            if direction > 0 and rsi < 55:
                score += 0.10  # Room to move up
            elif direction < 0 and rsi > 45:
                score += 0.10  # Room to move down

        # --- Price near EMA (good entry, not chasing) (10%) ---
        dist_ema = row.get("ema_short_dist", np.nan)
        if not np.isnan(dist_ema):
            if abs(dist_ema) < 1.5:
                score += 0.10  # Near EMA — good entry
            elif direction > 0 and dist_ema > 3.0:
                score *= 0.7  # Already extended — weaker entry
            elif direction < 0 and dist_ema < -3.0:
                score *= 0.7

        return float(np.clip(score * direction, -1.0, 1.0))

    @property
    def default_risk(self):
        return RiskDefaults(
            sl_atr_mult=2.0,       # Wider stops — ratio reversion takes time
            tp_atr_mult=6.0,       # 3:1 R:R
            signal_threshold=0.55,  # Need decent conviction
            max_hold_bars=40,      # Ratio reversion is slow — be patient
        )
