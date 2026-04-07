"""
Crypto Breakout Strategy (Rule-Based)

Volume-confirmed breakout from consolidation:
- Find tight consolidation (BB squeeze, low ATR ratio)
- Breakout on volume > 120% average
- Conservative: enter on retest of broken level
- Target: measured move (consolidation height projected)

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Breakout above/below consolidation with volume
  +0.25  Prior squeeze/compression (BB width < 20th percentile)
  +0.15  Volume confirmation (> 1.2x average on breakout)
  +0.10  Donchian channel break
  +0.10  ADX rising (momentum building)
  +0.10  FVG left behind by breakout impulse
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class CryptoBreakoutFeatures(CompositeFeatures):
    """Technical + SMC features plus breakout-specific derived features."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[TechnicalFeatures(windows=windows), SMCFeatures(windows=windows)],
            windows=windows,
        )
        self._extra = [
            "squeeze_strength",      # How tight the prior compression was (0-1)
            "breakout_detected",     # +1 bullish breakout, -1 bearish, 0 none
            "volume_on_break",       # Volume ratio on breakout candle
            "consolidation_bars",    # How many bars of compression before break
            "atr_expansion",         # ATR expanding vs recent average
            "donchian_break_dist",   # Distance past Donchian channel in ATR
            "retest_zone",           # Price returning to retest broken level
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
        volume = df["tick_volume"].values.astype(float) if "tick_volume" in df.columns else np.ones(n)

        for col in self._extra:
            features[col] = np.nan

        # Donchian channels
        don_period = 20
        don_high = pd.Series(high).rolling(don_period).max().values
        don_low = pd.Series(low).rolling(don_period).min().values

        # BB width for squeeze detection
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_widths = std20 / (sma20 + 1e-10)

        # ATR for expansion
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr_arr = np.concatenate([[np.nan], pd.Series(tr).rolling(14).mean().values])

        # Volume average
        vol_sma = pd.Series(volume).rolling(20).mean().values

        for i in range(max(30, self.warmup_bars), n):
            atr = atr_arr[i] if i < len(atr_arr) else features.iloc[i].get("atr_14", np.nan)
            if np.isnan(atr) or atr <= 0:
                continue

            # Squeeze strength (BB width percentile over last 100 bars)
            lookback = min(100, i)
            recent_widths = bb_widths[i - lookback:i + 1]
            recent_widths = recent_widths[~np.isnan(recent_widths)]
            if len(recent_widths) > 10:
                current_pctile = np.sum(recent_widths < bb_widths[i]) / len(recent_widths)
                features.iloc[i, features.columns.get_loc("squeeze_strength")] = max(0, 1.0 - current_pctile)
            else:
                features.iloc[i, features.columns.get_loc("squeeze_strength")] = 0.0

            # Breakout detection
            if i >= don_period and not np.isnan(don_high[i - 1]) and not np.isnan(don_low[i - 1]):
                if close[i] > don_high[i - 1]:
                    features.iloc[i, features.columns.get_loc("breakout_detected")] = 1.0
                elif close[i] < don_low[i - 1]:
                    features.iloc[i, features.columns.get_loc("breakout_detected")] = -1.0
                else:
                    features.iloc[i, features.columns.get_loc("breakout_detected")] = 0.0

            # Volume on breakout candle
            if vol_sma[i] > 0:
                features.iloc[i, features.columns.get_loc("volume_on_break")] = volume[i] / vol_sma[i]
            else:
                features.iloc[i, features.columns.get_loc("volume_on_break")] = 1.0

            # Consolidation bars (count narrow-range bars before this one)
            narrow_count = 0
            avg_range = atr * 0.8  # Bars with range < 80% of ATR are "narrow"
            for j in range(i - 1, max(i - 20, 0), -1):
                if (high[j] - low[j]) < avg_range:
                    narrow_count += 1
                else:
                    break
            features.iloc[i, features.columns.get_loc("consolidation_bars")] = narrow_count

            # ATR expansion
            if i >= 20:
                atr_sma = np.mean(atr_arr[max(0, i - 20):i])
                if atr_sma > 0:
                    features.iloc[i, features.columns.get_loc("atr_expansion")] = atr / atr_sma
                else:
                    features.iloc[i, features.columns.get_loc("atr_expansion")] = 1.0

            # Donchian break distance
            if i >= don_period and not np.isnan(don_high[i - 1]):
                if close[i] > don_high[i - 1]:
                    features.iloc[i, features.columns.get_loc("donchian_break_dist")] = (close[i] - don_high[i - 1]) / atr
                elif close[i] < don_low[i - 1]:
                    features.iloc[i, features.columns.get_loc("donchian_break_dist")] = (don_low[i - 1] - close[i]) / atr
                else:
                    features.iloc[i, features.columns.get_loc("donchian_break_dist")] = 0.0

            # Retest zone (price returning to broken level after breakout)
            # Check if prior bar was a breakout and current bar is pulling back
            if i > 1:
                prev_breakout = features.iloc[i - 1].get("breakout_detected", 0) if i > 0 else 0
                if prev_breakout == 1.0 and close[i] < close[i - 1] and close[i] > don_high[i - 2] if i >= don_period + 1 else False:
                    features.iloc[i, features.columns.get_loc("retest_zone")] = 1.0
                elif prev_breakout == -1.0 and close[i] > close[i - 1] and close[i] < don_low[i - 2] if i >= don_period + 1 else False:
                    features.iloc[i, features.columns.get_loc("retest_zone")] = -1.0
                else:
                    features.iloc[i, features.columns.get_loc("retest_zone")] = 0.0

        return features


class CryptoBreakoutStrategy(BaseStrategy):

    @property
    def name(self):
        return "crypto_breakout"

    def get_feature_set(self):
        return CryptoBreakoutFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # GATE: Must have breakout
        breakout = row.get("breakout_detected", 0)
        if np.isnan(breakout) or breakout == 0:
            return 0.0

        direction = 1.0 if breakout > 0 else -1.0

        # Component 1: Breakout detected (30%)
        score += 0.30

        # Component 2: Prior squeeze (25%)
        squeeze = row.get("squeeze_strength", 0)
        if not np.isnan(squeeze) and squeeze > 0.5:
            score += 0.25 * squeeze

        # Component 3: Volume confirmation (15%)
        vol = row.get("volume_on_break", 1.0)
        if not np.isnan(vol) and vol > 1.2:
            score += 0.15 * min(vol / 2.0, 1.0)
        elif not np.isnan(vol) and vol < 0.8:
            score *= 0.5  # Low volume breakout = likely false

        # Component 4: Donchian break distance (10%)
        don_dist = row.get("donchian_break_dist", 0)
        if not np.isnan(don_dist) and don_dist > 0:
            score += 0.10 * min(don_dist, 1.0)

        # Component 5: ADX rising (10%)
        adx = row.get("adx_14", 0)
        if not np.isnan(adx) and adx > 20:
            score += 0.10

        # Component 6: FVG from breakout impulse (10%)
        unfilled_fvg = row.get("unfilled_fvg_count", 0)
        if not np.isnan(unfilled_fvg) and unfilled_fvg > 0:
            score += 0.10

        # PENALTY: No consolidation before break (chasing a move)
        consol = row.get("consolidation_bars", 0)
        if np.isnan(consol) or consol < 3:
            score *= 0.6

        return score * direction

    @property
    def default_risk(self):
        from tradingagents.quant_strats.config import RiskDefaults
        return RiskDefaults(
            sl_atr_mult=1.5,
            tp_atr_mult=4.5,    # 3:1 R:R
            signal_threshold=0.50,
            max_hold_bars=30,
        )
