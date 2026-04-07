"""
D1 Range Trading Strategy (Rule-Based)

Trades validated Daily range boundaries with H4 confirmation:
- Identify D1 range (support/resistance tested 2-3x over 3-4 weeks)
- Enter at range boundary with CHOCH/FVG confirmation
- Target: opposite side of range
- Bonus: breakout trade when range eventually breaks

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Price at range extreme (support for buy, resistance for sell)
  +0.25  Range is validated (ADX low = ranging confirmed)
  +0.15  CHOCH detected at range boundary
  +0.10  FVG at boundary for precise entry
  +0.10  RSI divergence at boundary
  +0.10  Volume declining at boundary (exhaustion)
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class D1RangeFeatures(CompositeFeatures):
    """Technical + SMC features plus range-specific derived features."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[TechnicalFeatures(windows=windows), SMCFeatures(windows=windows)],
            windows=windows,
        )
        self._extra = [
            "range_position",       # Where price is in recent range (0=bottom, 1=top)
            "range_width_atr",      # Range width in ATR units
            "at_range_support",     # Price near range bottom
            "at_range_resistance",  # Price near range top
            "range_confirmed",      # ADX < 25 = ranging market
            "range_age_bars",       # How many bars since range started
            "bb_squeeze",           # Bollinger Band squeeze (tight = ranging)
            "rsi_divergence",       # RSI diverging from price at extremes
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

        for i in range(max(50, self.warmup_bars), n):
            atr = features.iloc[i].get("atr_14", np.nan)
            if np.isnan(atr) or atr <= 0:
                continue

            # Calculate range from last 50 bars (approximately 2-3 weeks on H4)
            lookback = min(50, i)
            period_high = np.max(high[i - lookback:i + 1])
            period_low = np.min(low[i - lookback:i + 1])
            range_width = period_high - period_low

            if range_width <= 0:
                continue

            # Range position (0 = at bottom, 1 = at top)
            rng_pos = (close[i] - period_low) / range_width
            features.iloc[i, features.columns.get_loc("range_position")] = rng_pos

            # Range width in ATR
            features.iloc[i, features.columns.get_loc("range_width_atr")] = range_width / atr

            # At range support (bottom 15%)
            features.iloc[i, features.columns.get_loc("at_range_support")] = max(0, 1.0 - rng_pos / 0.15) if rng_pos < 0.15 else 0.0

            # At range resistance (top 15%)
            features.iloc[i, features.columns.get_loc("at_range_resistance")] = max(0, (rng_pos - 0.85) / 0.15) if rng_pos > 0.85 else 0.0

            # Range confirmed (low ADX)
            adx = features.iloc[i].get("adx_14", 50)
            features.iloc[i, features.columns.get_loc("range_confirmed")] = max(0, 1.0 - adx / 30) if not np.isnan(adx) else 0.0

            # Range age (how many bars range has existed)
            # Count bars where price stayed within the range
            age = 0
            for j in range(i, max(i - lookback, 0), -1):
                if high[j] <= period_high * 1.01 and low[j] >= period_low * 0.99:
                    age += 1
                else:
                    break
            features.iloc[i, features.columns.get_loc("range_age_bars")] = age

            # BB squeeze
            bb_width = features.iloc[i].get("bb_width", np.nan)
            if not np.isnan(bb_width):
                # Lower bb_width = tighter squeeze = more ranging
                features.iloc[i, features.columns.get_loc("bb_squeeze")] = max(0, 1.0 - bb_width * 5)

            # RSI divergence at range extremes
            rsi = features.iloc[i].get("rsi_14", 50)
            if not np.isnan(rsi) and lookback > 5:
                # At support: price at new low but RSI higher = bullish divergence
                if rng_pos < 0.2 and rsi > 35:
                    features.iloc[i, features.columns.get_loc("rsi_divergence")] = (rsi - 30) / 40  # positive = bullish
                # At resistance: price at new high but RSI lower = bearish divergence
                elif rng_pos > 0.8 and rsi < 65:
                    features.iloc[i, features.columns.get_loc("rsi_divergence")] = -(70 - rsi) / 40  # negative = bearish
                else:
                    features.iloc[i, features.columns.get_loc("rsi_divergence")] = 0.0

        return features


class D1RangeTradeStrategy(BaseStrategy):

    @property
    def name(self):
        return "d1_range_trade"

    def get_feature_set(self):
        return D1RangeFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # GATE: Must be in a confirmed range
        range_confirmed = row.get("range_confirmed", 0)
        if np.isnan(range_confirmed) or range_confirmed < 0.3:
            return 0.0  # Not ranging

        # Check range width (must be meaningful)
        range_width = row.get("range_width_atr", 0)
        if np.isnan(range_width) or range_width < 3:
            return 0.0  # Range too narrow

        # Determine direction
        at_support = row.get("at_range_support", 0) or 0
        at_resistance = row.get("at_range_resistance", 0) or 0

        if at_support > 0.3:
            direction = 1.0  # Buy at support
            boundary_strength = at_support
        elif at_resistance > 0.3:
            direction = -1.0  # Sell at resistance
            boundary_strength = at_resistance
        else:
            return 0.0  # Not at a range boundary

        # Component 1: At range extreme (30%)
        score += 0.30 * boundary_strength

        # Component 2: Range confirmed by ADX (25%)
        score += 0.25 * range_confirmed

        # Component 3: CHOCH at boundary (15%)
        choch = row.get("choch_detected", 0)
        if choch:
            score += 0.15

        # Component 4: FVG at boundary (10%)
        fvg_proximity = row.get("nearest_fvg_dist_atr", 5)
        if not np.isnan(fvg_proximity) and abs(fvg_proximity) < 1.0:
            score += 0.10

        # Component 5: RSI divergence (10%)
        rsi_div = row.get("rsi_divergence", 0)
        if not np.isnan(rsi_div):
            if direction > 0 and rsi_div > 0:
                score += 0.10 * rsi_div
            elif direction < 0 and rsi_div < 0:
                score += 0.10 * abs(rsi_div)

        # Component 6: Volume exhaustion (10%)
        vol = row.get("volume_ratio", 1.0)
        if not np.isnan(vol) and vol < 0.8:
            score += 0.10  # Low volume at boundary = exhaustion

        # BONUS: Range age (older = more validated)
        age = row.get("range_age_bars", 0)
        if not np.isnan(age) and age > 30:
            score *= 1.15  # 15% bonus for validated range

        return min(1.0, score) * direction

    @property
    def default_risk(self):
        from tradingagents.quant_strats.config import RiskDefaults
        return RiskDefaults(
            sl_atr_mult=1.5,       # Stop just beyond range boundary
            tp_atr_mult=5.0,       # Target opposite side of range
            signal_threshold=0.50,
            max_hold_bars=40,       # Ranges resolve over days
        )
