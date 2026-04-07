"""
Crypto Trend Following Strategy (Rule-Based)

Multi-timeframe trend following designed for crypto:
- HTF bias: EMA 50/200 alignment on higher timeframe
- Entry: Pullback to 21/50 EMA with rejection candle
- Confluence: FVG fill at EMA level
- High R:R (3-6:1) with trailing via 21 EMA

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  EMA alignment (50 > 200 = bull, 50 < 200 = bear)
  +0.25  Pullback to 21 EMA (price near EMA after trending away)
  +0.15  ADX confirms strong trend (> 25)
  +0.10  RSI supports direction (> 50 for bull, < 50 for bear)
  +0.10  Volume supports trend (above average on trend candles)
  +0.10  FVG confluence at EMA level
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class CryptoTrendFeatures(CompositeFeatures):
    """Technical + SMC features plus trend-specific derived features."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[TechnicalFeatures(windows=windows), SMCFeatures(windows=windows)],
            windows=windows,
        )
        self._extra = [
            "ema_200_dist",         # Distance from 200 EMA in ATR units
            "ema_50_200_cross",     # 1 if golden cross, -1 if death cross
            "pullback_to_21ema",    # How close price is to 21 EMA (0-1, 1=at EMA)
            "pullback_to_50ema",    # How close price is to 50 EMA
            "higher_highs",         # Recent higher highs count
            "lower_lows",           # Recent lower lows count
            "trend_strength_combo", # Combined trend indicators
            "fvg_at_ema",           # FVG coincides with EMA zone
        ]

    @property
    def feature_names(self):
        return super().feature_names + self._extra

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        n = len(features)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        for col in self._extra:
            features[col] = np.nan

        # Compute EMAs
        ema_21 = pd.Series(close).ewm(span=21).mean().values
        ema_50 = pd.Series(close).ewm(span=50).mean().values
        ema_200 = pd.Series(close).ewm(span=200).mean().values

        for i in range(max(200, self.warmup_bars), n):
            atr = features.iloc[i].get("atr_14", np.nan)
            if np.isnan(atr) or atr <= 0:
                continue

            # EMA 200 distance
            features.iloc[i, features.columns.get_loc("ema_200_dist")] = (close[i] - ema_200[i]) / atr

            # Golden/Death cross
            features.iloc[i, features.columns.get_loc("ema_50_200_cross")] = 1.0 if ema_50[i] > ema_200[i] else -1.0

            # Pullback to 21 EMA (1.0 = at EMA, 0.0 = far)
            dist_21 = abs(close[i] - ema_21[i]) / atr
            features.iloc[i, features.columns.get_loc("pullback_to_21ema")] = max(0, 1.0 - dist_21 / 2.0)

            # Pullback to 50 EMA
            dist_50 = abs(close[i] - ema_50[i]) / atr
            features.iloc[i, features.columns.get_loc("pullback_to_50ema")] = max(0, 1.0 - dist_50 / 3.0)

            # Higher highs / lower lows (last 10 bars)
            lookback = min(10, i)
            hh = sum(1 for j in range(i - lookback + 1, i + 1)
                     if high[j] > high[j - 1]) if lookback > 1 else 0
            ll = sum(1 for j in range(i - lookback + 1, i + 1)
                     if low[j] < low[j - 1]) if lookback > 1 else 0
            features.iloc[i, features.columns.get_loc("higher_highs")] = hh / max(lookback, 1)
            features.iloc[i, features.columns.get_loc("lower_lows")] = ll / max(lookback, 1)

            # Combined trend strength
            adx = features.iloc[i].get("adx_14", 0) or 0
            di_diff = features.iloc[i].get("di_diff", 0) or 0
            ema_cross = 1.0 if ema_50[i] > ema_200[i] else -1.0
            features.iloc[i, features.columns.get_loc("trend_strength_combo")] = (
                (adx / 50) * np.sign(di_diff) + ema_cross * 0.5
            )

            # FVG at EMA level
            fvg_dist = features.iloc[i].get("nearest_fvg_dist_atr", 5)
            if not np.isnan(fvg_dist) and abs(fvg_dist) < 1.5 and dist_21 < 1.0:
                features.iloc[i, features.columns.get_loc("fvg_at_ema")] = 1.0
            else:
                features.iloc[i, features.columns.get_loc("fvg_at_ema")] = 0.0

        return features


class CryptoTrendFollowStrategy(BaseStrategy):

    @property
    def name(self):
        return "crypto_trend_follow"

    def get_feature_set(self):
        return CryptoTrendFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # GATE: EMA alignment required
        ema_cross = row.get("ema_50_200_cross", 0)
        if np.isnan(ema_cross) or ema_cross == 0:
            return 0.0

        direction = 1.0 if ema_cross > 0 else -1.0

        # Component 1: EMA alignment (30%)
        score += 0.30

        # Component 2: Pullback to 21 EMA (25%)
        pb_21 = row.get("pullback_to_21ema", 0)
        pb_50 = row.get("pullback_to_50ema", 0)
        pullback = max(pb_21 if not np.isnan(pb_21) else 0,
                       pb_50 if not np.isnan(pb_50) else 0)
        if pullback > 0.5:
            score += 0.25 * pullback

        # Component 3: ADX trend strength (15%)
        adx = row.get("adx_14", 0)
        if not np.isnan(adx) and adx > 25:
            score += 0.15 * min(adx / 50, 1.0)

        # Component 4: RSI direction (10%)
        rsi = row.get("rsi_14", 50)
        if not np.isnan(rsi):
            if direction > 0 and rsi > 50:
                score += 0.10
            elif direction < 0 and rsi < 50:
                score += 0.10

        # Component 5: Volume (10%)
        vol_ratio = row.get("volume_ratio", 1.0)
        if not np.isnan(vol_ratio) and vol_ratio > 1.0:
            score += 0.10 * min(vol_ratio / 2.0, 1.0)

        # Component 6: FVG at EMA (10%)
        fvg_ema = row.get("fvg_at_ema", 0)
        if not np.isnan(fvg_ema) and fvg_ema > 0:
            score += 0.10

        # PENALTY: No pullback = chasing
        if pullback < 0.3:
            score *= 0.5

        return score * direction

    @property
    def default_risk(self):
        from tradingagents.quant_strats.config import RiskDefaults
        return RiskDefaults(
            sl_atr_mult=1.5,
            tp_atr_mult=5.0,    # 3.3:1 R:R
            signal_threshold=0.55,
            max_hold_bars=40,
        )
