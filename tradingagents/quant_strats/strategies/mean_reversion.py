"""
Mean Reversion Strategy (Rule-Based)

Fades overextended price moves back to the mean using
Bollinger Band position, Z-score, RSI extremes, and Stochastic.
Best suited for range-bound, low-volatility pairs (EURGBP, AUDNZD, USDCHF).

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Z-score < -2.0 (very oversold, expect reversion up)
  +0.25  RSI < 30 (oversold)
  +0.25  Stochastic %K < 20 and crossing up
  +0.20  BB %B < 0.0 (below lower band)
  Reversed for bearish signals.

ADX filter: Only fire when ADX < 30 (ranging market).
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows

# Pairs excluded from mean-reversion: too volatile / trending for fade setups
MR_EXCLUDED_PAIRS = frozenset({"BTCUSD", "GBPJPY", "ETHUSD", "XAUUSD", "XAGUSD"})


class MeanReversionFeatures(TechnicalFeatures):
    """Technical features + mean reversion specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "zscore_20", "zscore_50",
            "stoch_k", "stoch_d",
            "close_vs_sma20", "close_vs_sma50",
            "bb_squeeze",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        w = self.windows

        cs = pd.Series(close)
        sma20 = cs.rolling(w.mid).mean()
        std20 = cs.rolling(w.mid).std()
        sma50 = cs.rolling(w.long).mean()
        std50 = cs.rolling(w.long).std()

        features["zscore_20"] = ((cs - sma20) / (std20 + 1e-10)).values
        features["zscore_50"] = ((cs - sma50) / (std50 + 1e-10)).values

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        period = w.stoch_period

        lowest_low = pd.Series(low).rolling(period).min()
        highest_high = pd.Series(high).rolling(period).max()
        stoch_k = 100 * (cs - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()

        features["stoch_k"] = stoch_k.values
        features["stoch_d"] = stoch_d.values

        features["close_vs_sma20"] = ((cs - sma20) / (sma20 + 1e-10)).values
        features["close_vs_sma50"] = ((cs - sma50) / (sma50 + 1e-10)).values

        bb_width = features["bb_width"].values
        bb_series = pd.Series(bb_width)
        bb_rank = bb_series.rolling(100, min_periods=20).rank(pct=True)
        features["bb_squeeze"] = bb_rank.values

        return features


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    def get_feature_set(self) -> BaseFeatureSet:
        return MeanReversionFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # ADX filter: only trade in ranging markets
        adx = row.get("adx_14", np.nan)
        if not np.isnan(adx) and adx > 30:
            return 0.0  # Too trendy for mean reversion

        # 1. Z-score (20-period): core mean reversion signal
        zscore = row.get("zscore_20", np.nan)
        if not np.isnan(zscore):
            if zscore < -2.0:
                score += 0.30   # Very oversold
            elif zscore < -1.0:
                score += 0.15   # Moderately oversold
            elif zscore > 2.0:
                score -= 0.30   # Very overbought
            elif zscore > 1.0:
                score -= 0.15   # Moderately overbought

        # 2. RSI extremes
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if rsi < 25:
                score += 0.25
            elif rsi < 30:
                score += 0.15
            elif rsi > 75:
                score -= 0.25
            elif rsi > 70:
                score -= 0.15

        # 3. Stochastic %K
        stoch_k = row.get("stoch_k", np.nan)
        stoch_d = row.get("stoch_d", np.nan)
        if not np.isnan(stoch_k):
            if stoch_k < 20:
                # Oversold — extra boost if %K crossing above %D
                score += 0.20
                if not np.isnan(stoch_d) and stoch_k > stoch_d:
                    score += 0.05  # Bullish crossover
            elif stoch_k > 80:
                score -= 0.20
                if not np.isnan(stoch_d) and stoch_k < stoch_d:
                    score -= 0.05  # Bearish crossover

        # 4. Bollinger Band position
        bb_pct = row.get("bb_pct", np.nan)
        if not np.isnan(bb_pct):
            if bb_pct < 0.0:
                score += 0.20  # Below lower band
            elif bb_pct < 0.1:
                score += 0.10  # Near lower band
            elif bb_pct > 1.0:
                score -= 0.20  # Above upper band
            elif bb_pct > 0.9:
                score -= 0.10  # Near upper band

        return float(np.clip(score, -1.0, 1.0))
