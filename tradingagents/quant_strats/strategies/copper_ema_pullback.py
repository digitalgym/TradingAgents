"""
Copper EMA Pullback Strategy (Rule-Based)

Trend-following on copper using 20/50 EMA crossover with pullback entry.
Instead of entering on the breakout candle (which gets chopped), waits for
price to pull back to the 20 EMA after a crossover, then enters with trend.

Designed specifically for COPPER-C H4.

Entry rules (long example - all must be true on bar close):
  1. EMA20 > EMA50 (bullish trend established)
  2. Price pulled back to within 0.5 ATR of EMA20 (not a breakout candle)
  3. Price closed above EMA20 (bounce confirmed)
  4. RSI(14) < 70 (not overbought)
  5. ADX > 20 (some trend strength present)

Short: mirror (EMA20 < EMA50, price pulled up to EMA20, RSI > 30)

Exit:
  SL: 1.5x ATR beyond EMA20 (gives room for noise)
  TP: 2:1 R:R minimum (SL distance * 2)
  Max hold: 20 bars

Scoring (max +1.0 bullish, -1.0 bearish):
  +0.30  EMA crossover direction (trend bias)
  +0.30  Price at EMA20 pullback zone (within 0.5 ATR)
  +0.20  RSI confirms room to move
  +0.10  ADX shows trend strength
  +0.10  Price closed on correct side of EMA20 (bounce confirmed)
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows


class CopperEMAPullbackFeatures(TechnicalFeatures):
    """Technical features + EMA pullback detection features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "ema20",                # Raw EMA20 price
            "ema50",                # Raw EMA50 price
            "pullback_to_ema20",    # Binary: price within 0.5 ATR of EMA20
            "bounced_off_ema20",    # Binary: close on correct side of EMA20 for trend
            "dist_to_ema20_atr",    # Distance from close to EMA20 in ATR units
            "ema_spread_atr",       # EMA20-EMA50 distance in ATR (trend strength)
            "bars_since_cross",     # How many bars since last EMA crossover
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        w = self.windows

        cs = pd.Series(close)
        ema20 = cs.ewm(span=w.ema_short, adjust=False).mean().values  # 20
        ema50 = cs.ewm(span=w.ema_long, adjust=False).mean().values   # 50
        atr = features["atr_14"].values

        features["ema20"] = ema20
        features["ema50"] = ema50

        # Distance from close to EMA20 in ATR units
        dist = self._safe_divide(np.abs(close - ema20), atr)
        features["dist_to_ema20_atr"] = dist

        # Pullback: price came within 0.5 ATR of EMA20
        features["pullback_to_ema20"] = (dist < 0.5).astype(float)

        # Bounce confirmation: close is on the "right" side of EMA20
        # In uptrend (EMA20>EMA50): close should be >= EMA20
        # In downtrend (EMA20<EMA50): close should be <= EMA20
        bullish_trend = ema20 > ema50
        bullish_bounce = bullish_trend & (close >= ema20)
        bearish_bounce = (~bullish_trend) & (close <= ema20)
        features["bounced_off_ema20"] = (bullish_bounce | bearish_bounce).astype(float)

        # EMA spread in ATR units (how strong is the trend separation)
        features["ema_spread_atr"] = self._safe_divide(ema20 - ema50, atr)

        # Bars since last EMA crossover
        cross_signal = pd.Series(bullish_trend.astype(int)).diff().abs()
        bars_since = np.zeros(len(close))
        for i in range(1, len(close)):
            if cross_signal.iloc[i] > 0:
                bars_since[i] = 0
            else:
                bars_since[i] = bars_since[i - 1] + 1
        features["bars_since_cross"] = bars_since

        return features


class CopperEMAPullbackStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "copper_ema_pullback"

    def get_feature_set(self) -> BaseFeatureSet:
        return CopperEMAPullbackFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # --- Determine trend direction from EMA crossover ---
        ema_cross = row.get("ema_cross", np.nan)  # 1.0 if EMA20 > EMA50
        if np.isnan(ema_cross):
            return 0.0

        direction = 1.0 if ema_cross > 0.5 else -1.0

        # 1. EMA crossover bias (+0.30)
        score += 0.30 * direction

        # --- Require pullback — reject breakout candles ---
        pullback = row.get("pullback_to_ema20", np.nan)
        if np.isnan(pullback) or pullback < 0.5:
            return 0.0  # Not at EMA20 — no entry

        # 2. Pullback to EMA20 confirmed (+0.30)
        score += 0.30 * direction

        # 3. Bounce confirmation (close on correct side)
        bounced = row.get("bounced_off_ema20", np.nan)
        if not np.isnan(bounced) and bounced > 0.5:
            score += 0.10 * direction
        else:
            # Price at EMA20 but hasn't bounced yet — weaker signal
            pass

        # 4. RSI filter — no longs above 70, no shorts below 30
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if direction > 0 and rsi > 70:
                return 0.0  # Overbought — skip long
            elif direction < 0 and rsi < 30:
                return 0.0  # Oversold — skip short
            # Bonus: RSI in healthy range
            if direction > 0 and rsi < 60:
                score += 0.20 * direction  # Plenty of room to run
            elif direction < 0 and rsi > 40:
                score += 0.20 * direction
            else:
                score += 0.10 * direction  # Some room

        # 5. ADX trend strength
        adx = row.get("adx_14", np.nan)
        if not np.isnan(adx):
            if adx > 25:
                score += 0.10 * direction  # Good trend
            elif adx < 15:
                score *= 0.5  # Very weak trend — halve conviction

        # --- Reject very fresh crossovers (first 3 bars are noisy) ---
        bars_since = row.get("bars_since_cross", np.nan)
        if not np.isnan(bars_since) and bars_since < 3:
            return 0.0  # Too close to crossover — wait for confirmation

        return float(np.clip(score, -1.0, 1.0))
