"""
Trend Following Strategy (Rule-Based)

Captures sustained directional moves using EMA crossovers,
ADX trend strength, RSI momentum, and rate of change.
Best suited for trending, volatile pairs (XAUUSD, GBPJPY, EURJPY).

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.25  EMA20 > EMA50 (bullish structure)
  +0.25  ADX > 25 and +DI > -DI (strong bullish trend)
  +0.20  RSI 40-70 (healthy momentum, not overbought)
  +0.15  MACD histogram positive and rising
  +0.15  Price above EMA20 (pullback has ended)
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures


class TrendFollowingStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "trend_following"

    def get_feature_set(self) -> BaseFeatureSet:
        return TechnicalFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # 1. EMA crossover: EMA20 > EMA50 = bullish structure
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            score += 0.25 if ema_cross > 0.5 else -0.25

        # 2. ADX trend strength + direction
        adx = row.get("adx_14", np.nan)
        di_diff = row.get("di_diff", np.nan)
        if not np.isnan(adx) and not np.isnan(di_diff):
            if adx > 25:
                # Strong trend — use DI direction
                score += 0.25 if di_diff > 0 else -0.25
            # Weak trend (ADX < 20) = no contribution

        # 3. RSI momentum (favour 40-60 zone for trend continuation)
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if 40 <= rsi <= 60:
                # Neutral zone — slight bullish if EMA bullish
                score += 0.10 if score > 0 else -0.10
            elif 60 < rsi <= 70:
                score += 0.20  # Bullish momentum
            elif 30 <= rsi < 40:
                score -= 0.20  # Bearish momentum
            elif rsi > 70:
                score += 0.05  # Overbought — reduce bullish conviction
            elif rsi < 30:
                score -= 0.05  # Oversold — reduce bearish conviction

        # 4. MACD histogram direction
        macd_hist = row.get("macd_hist_norm", np.nan)
        if not np.isnan(macd_hist):
            if macd_hist > 0.5:
                score += 0.15
            elif macd_hist > 0:
                score += 0.08
            elif macd_hist < -0.5:
                score -= 0.15
            elif macd_hist < 0:
                score -= 0.08

        # 5. Price vs EMA20 (is price still above/below short-term trend?)
        ema_short_dist = row.get("ema_short_dist", np.nan)
        if not np.isnan(ema_short_dist):
            if ema_short_dist > 0:
                score += 0.15
            elif ema_short_dist < 0:
                score -= 0.15

        return float(np.clip(score, -1.0, 1.0))
