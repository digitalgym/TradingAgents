"""
Crypto Momentum Strategy (Rule-Based)

Momentum continuation after brief consolidation:
- Identify strong impulse move (RSI + MACD + Volume)
- Wait for brief consolidation (flag/tight range)
- Enter on break of consolidation in impulse direction
- Best used as confirmation filter for other strategies

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  RSI momentum (> 60 bull, < 40 bear) with MACD histogram expanding
  +0.25  Volume confirmation (rising price + rising volume)
  +0.15  Brief consolidation after impulse (flag pattern)
  +0.10  Rate of change confirms direction
  +0.10  DI spread (strong directional movement)
  +0.10  No momentum divergence (price and RSI agree)
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class CryptoMomentumFeatures(CompositeFeatures):
    """Technical + SMC features plus momentum-specific derived features."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[TechnicalFeatures(windows=windows), SMCFeatures(windows=windows)],
            windows=windows,
        )
        self._extra = [
            "momentum_score",        # Combined RSI + MACD momentum (-1 to 1)
            "volume_price_confirm",  # Rising price + rising volume alignment
            "impulse_strength",      # Strength of recent impulse move
            "consolidation_after_impulse",  # Brief pause after strong move
            "rsi_macd_agree",        # RSI and MACD both pointing same direction
            "momentum_divergence",   # Price making new high but RSI isn't (bearish div) or vice versa
            "roc_acceleration",      # Rate of change accelerating or decelerating
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

        # Pre-compute RSI series for divergence detection
        rsi_series = features["rsi_14"].values if "rsi_14" in features.columns else np.full(n, 50)

        for i in range(max(30, self.warmup_bars), n):
            atr = features.iloc[i].get("atr_14", np.nan)
            if np.isnan(atr) or atr <= 0:
                continue

            # Combined momentum score
            rsi = features.iloc[i].get("rsi_14", 50)
            macd_hist = features.iloc[i].get("macd_hist_norm", 0)
            roc = features.iloc[i].get("roc_10", 0)

            mom_score = 0.0
            if not np.isnan(rsi):
                mom_score += (rsi - 50) / 50  # -1 to +1
            if not np.isnan(macd_hist):
                mom_score += np.clip(macd_hist * 2, -1, 1)
            if not np.isnan(roc):
                mom_score += np.clip(roc * 10, -1, 1)
            features.iloc[i, features.columns.get_loc("momentum_score")] = np.clip(mom_score / 3, -1, 1)

            # Volume-price confirmation
            if i >= 5:
                price_up = close[i] > close[i - 5]
                vol_up = volume[i] > np.mean(volume[max(0, i - 5):i])
                if price_up and vol_up:
                    features.iloc[i, features.columns.get_loc("volume_price_confirm")] = 1.0
                elif not price_up and vol_up:
                    features.iloc[i, features.columns.get_loc("volume_price_confirm")] = -1.0  # Bearish volume
                elif price_up and not vol_up:
                    features.iloc[i, features.columns.get_loc("volume_price_confirm")] = -0.5  # Weak rally
                else:
                    features.iloc[i, features.columns.get_loc("volume_price_confirm")] = 0.5  # Bearish with low vol = less conviction

            # Impulse strength (how far did price move in last 5 bars vs ATR)
            if i >= 5:
                move = abs(close[i] - close[i - 5])
                impulse = move / (atr * 5)  # Normalized by expected move
                features.iloc[i, features.columns.get_loc("impulse_strength")] = min(impulse * 2, 1.0)

            # Consolidation after impulse
            # Strong move in bars 10-5 ago, then narrow range in last 5 bars
            if i >= 10:
                prior_move = abs(close[i - 5] - close[i - 10]) / atr
                recent_range = (np.max(high[i - 4:i + 1]) - np.min(low[i - 4:i + 1])) / atr
                if prior_move > 2.0 and recent_range < 2.0:
                    features.iloc[i, features.columns.get_loc("consolidation_after_impulse")] = 1.0
                else:
                    features.iloc[i, features.columns.get_loc("consolidation_after_impulse")] = 0.0

            # RSI + MACD agreement
            if not np.isnan(rsi) and not np.isnan(macd_hist):
                rsi_bull = rsi > 55
                macd_bull = macd_hist > 0
                if rsi_bull == macd_bull:
                    features.iloc[i, features.columns.get_loc("rsi_macd_agree")] = 1.0 if rsi_bull else -1.0
                else:
                    features.iloc[i, features.columns.get_loc("rsi_macd_agree")] = 0.0

            # Momentum divergence
            if i >= 20:
                # Check if price made new 20-bar high but RSI didn't
                price_at_high = close[i] >= np.max(close[i - 20:i])
                rsi_at_high = rsi_series[i] >= np.nanmax(rsi_series[i - 20:i]) if not np.isnan(rsi_series[i]) else True
                price_at_low = close[i] <= np.min(close[i - 20:i])
                rsi_at_low = rsi_series[i] <= np.nanmin(rsi_series[i - 20:i]) if not np.isnan(rsi_series[i]) else True

                if price_at_high and not rsi_at_high:
                    features.iloc[i, features.columns.get_loc("momentum_divergence")] = -1.0  # Bearish divergence
                elif price_at_low and not rsi_at_low:
                    features.iloc[i, features.columns.get_loc("momentum_divergence")] = 1.0  # Bullish divergence
                else:
                    features.iloc[i, features.columns.get_loc("momentum_divergence")] = 0.0

            # ROC acceleration
            if not np.isnan(roc) and i >= 5:
                prev_roc = features.iloc[i - 5].get("roc_10", 0)
                if not np.isnan(prev_roc):
                    features.iloc[i, features.columns.get_loc("roc_acceleration")] = roc - prev_roc

        return features


class CryptoMomentumStrategy(BaseStrategy):

    @property
    def name(self):
        return "crypto_momentum"

    def get_feature_set(self):
        return CryptoMomentumFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # GATE: Need clear momentum direction
        mom_score = row.get("momentum_score", 0)
        if np.isnan(mom_score) or abs(mom_score) < 0.2:
            return 0.0

        direction = 1.0 if mom_score > 0 else -1.0

        # Component 1: Momentum strength (30%)
        score += 0.30 * abs(mom_score)

        # Component 2: Volume-price confirmation (25%)
        vol_confirm = row.get("volume_price_confirm", 0)
        if not np.isnan(vol_confirm):
            if direction > 0 and vol_confirm > 0:
                score += 0.25 * vol_confirm
            elif direction < 0 and vol_confirm < 0:
                score += 0.25 * abs(vol_confirm)

        # Component 3: Consolidation after impulse = continuation pattern (15%)
        consol = row.get("consolidation_after_impulse", 0)
        if not np.isnan(consol) and consol > 0:
            score += 0.15

        # Component 4: ROC acceleration (10%)
        roc_accel = row.get("roc_acceleration", 0)
        if not np.isnan(roc_accel):
            if direction > 0 and roc_accel > 0:
                score += 0.10 * min(roc_accel * 5, 1.0)
            elif direction < 0 and roc_accel < 0:
                score += 0.10 * min(abs(roc_accel) * 5, 1.0)

        # Component 5: DI spread (10%)
        di_diff = row.get("di_diff", 0)
        if not np.isnan(di_diff):
            if direction > 0 and di_diff > 5:
                score += 0.10
            elif direction < 0 and di_diff < -5:
                score += 0.10

        # Component 6: RSI + MACD agreement (10%)
        agree = row.get("rsi_macd_agree", 0)
        if not np.isnan(agree):
            if (direction > 0 and agree > 0) or (direction < 0 and agree < 0):
                score += 0.10

        # PENALTY: Momentum divergence = exit signal, not entry
        div = row.get("momentum_divergence", 0)
        if not np.isnan(div):
            if (direction > 0 and div < 0) or (direction < 0 and div > 0):
                score *= 0.3  # Heavy penalty for divergence

        # PENALTY: Extended move (> 3 bars without pullback)
        impulse = row.get("impulse_strength", 0)
        consol_after = row.get("consolidation_after_impulse", 0)
        if not np.isnan(impulse) and impulse > 0.8 and (np.isnan(consol_after) or consol_after == 0):
            score *= 0.5  # Chasing an extended move without consolidation

        return score * direction

    @property
    def default_risk(self):
        from tradingagents.quant_strats.config import RiskDefaults
        return RiskDefaults(
            sl_atr_mult=1.25,     # Tighter stop for momentum trades
            tp_atr_mult=3.5,      # 2.8:1 R:R
            signal_threshold=0.50,
            max_hold_bars=20,      # Momentum trades should work quickly
        )
