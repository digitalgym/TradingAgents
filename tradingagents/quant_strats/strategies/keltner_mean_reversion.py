"""
Keltner Channel Mean-Reversion Strategy (Rule-Based)

Fades channel extremes in ranging markets using Keltner Channels
with stochastic confirmation. Designed for copper and platinum
which mean-revert better than they trend.

Only activates in ranging regimes (ADX < threshold, narrow BB width).

Entry rules (long example — all must be true on bar close):
  1. Regime is "range" (ADX < adx_threshold AND bb_width < bb_width_threshold)
  2. Price touches or closes below lower Keltner band
  3. Stochastic %K crosses above %D, both < stoch_oversold
  4. Symbol is in allowed pair list

Exit rules:
  - TP at Keltner basis (EMA midline) or fixed R:R
  - SL at entry - ATR * sl_atr_mult (beyond channel extreme)
  - Max hold timeout

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.35  Price at/below lower Keltner band (or above upper for short)
  +0.30  Stochastic crossover in oversold/overbought zone
  +0.20  Regime confirmed as ranging (ADX + BB width)
  +0.15  Price position within recent range (lower 30% for long)
"""

import numpy as np
import pandas as pd
from typing import Optional

from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.config import FeatureWindows, RiskDefaults


class KeltnerFeatures(TechnicalFeatures):
    """Technical features + Keltner channel mean-reversion features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "keltner_upper",        # Upper Keltner band (raw price)
            "keltner_lower",        # Lower Keltner band (raw price)
            "keltner_basis",        # EMA midline (raw price)
            "keltner_position",     # Price position in channel (0=lower, 1=upper)
            "keltner_width_pct",    # Channel width / price
            "below_lower_keltner",  # Binary: close <= lower band
            "above_upper_keltner",  # Binary: close >= upper band
            "stoch_k",              # Stochastic %K
            "stoch_d",              # Stochastic %D
            "stoch_cross_up",       # Binary: %K crosses above %D
            "stoch_cross_down",     # Binary: %K crosses below %D
            "regime_ranging",       # Binary: ADX < threshold AND bb_width narrow
            "price_range_pct",      # Where price is in recent 50-bar range (0-1)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        w = self.windows

        cs = pd.Series(close)

        # --- Keltner Channel: EMA(20) +/- ATR(14) * mult ---
        keltner_length = w.mid  # 20
        keltner_mult = 2.0

        basis = cs.ewm(span=keltner_length, adjust=False).mean().values
        atr = features["atr_14"].values

        upper = basis + keltner_mult * atr
        lower = basis - keltner_mult * atr
        channel_width = upper - lower

        features["keltner_upper"] = upper
        features["keltner_lower"] = lower
        features["keltner_basis"] = basis
        features["keltner_position"] = self._safe_divide(close - lower, channel_width)
        features["keltner_width_pct"] = self._safe_divide(channel_width, close)
        features["below_lower_keltner"] = (close <= lower).astype(float)
        features["above_upper_keltner"] = (close >= upper).astype(float)

        # --- Stochastic %K / %D ---
        period = w.stoch_period  # 14
        lowest_low = pd.Series(low).rolling(period).min()
        highest_high = pd.Series(high).rolling(period).max()
        stoch_k = 100 * (cs - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()

        features["stoch_k"] = stoch_k.values
        features["stoch_d"] = stoch_d.values

        # Crossover detection
        prev_k = stoch_k.shift(1)
        prev_d = stoch_d.shift(1)
        features["stoch_cross_up"] = (
            (stoch_k > stoch_d) & (prev_k <= prev_d)
        ).astype(float).values
        features["stoch_cross_down"] = (
            (stoch_k < stoch_d) & (prev_k >= prev_d)
        ).astype(float).values

        # --- Regime: ranging = ADX low AND BB width not extreme ---
        # Thresholds calibrated for metals (copper/platinum have wider BB than FX)
        adx = features["adx_14"].values
        bb_width = features["bb_width"].values
        features["regime_ranging"] = (
            (adx < 30) & (bb_width < 0.05)
        ).astype(float)

        # --- Price position in recent range (0 = at low, 1 = at high) ---
        range_period = 50
        rolling_high = pd.Series(high).rolling(range_period).max().values
        rolling_low = pd.Series(low).rolling(range_period).min().values
        features["price_range_pct"] = self._safe_divide(
            close - rolling_low, rolling_high - rolling_low
        )

        return features


class KeltnerMeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "keltner_mean_reversion"

    def get_feature_set(self) -> BaseFeatureSet:
        return KeltnerFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # --- Gate: must be in ranging regime ---
        regime_ranging = row.get("regime_ranging", np.nan)
        if np.isnan(regime_ranging) or regime_ranging < 0.5:
            return 0.0  # Not ranging — no signal

        # Regime confirmed: +0.20 base (direction determined below)
        regime_bonus = 0.20

        # --- Keltner band touch ---
        below_lower = row.get("below_lower_keltner", 0.0)
        above_upper = row.get("above_upper_keltner", 0.0)

        if not np.isnan(below_lower) and below_lower > 0.5:
            direction = 1.0   # Bullish mean reversion (fade lower band)
        elif not np.isnan(above_upper) and above_upper > 0.5:
            direction = -1.0  # Bearish mean reversion (fade upper band)
        else:
            # Not at a band extreme — check if close to band
            keltner_pos = row.get("keltner_position", np.nan)
            if np.isnan(keltner_pos):
                return 0.0
            if keltner_pos < 0.10:
                direction = 1.0   # Very near lower band
            elif keltner_pos > 0.90:
                direction = -1.0  # Very near upper band
            else:
                return 0.0  # Not at extremes — no signal

        # Band touch score
        score += 0.35 * direction

        # --- Stochastic confirmation ---
        stoch_k = row.get("stoch_k", np.nan)
        stoch_d = row.get("stoch_d", np.nan)
        stoch_cross_up = row.get("stoch_cross_up", 0.0)
        stoch_cross_down = row.get("stoch_cross_down", 0.0)

        if direction > 0:
            # Long: need oversold stochastic with bullish cross
            if not np.isnan(stoch_k) and stoch_k < 30:
                if not np.isnan(stoch_cross_up) and stoch_cross_up > 0.5:
                    score += 0.30 * direction  # Full confirmation
                elif not np.isnan(stoch_d) and stoch_k > stoch_d:
                    score += 0.15 * direction  # %K above %D but no fresh cross
                else:
                    score += 0.08 * direction  # Oversold, no cross yet
        else:
            # Short: need overbought stochastic with bearish cross
            if not np.isnan(stoch_k) and stoch_k > 70:
                if not np.isnan(stoch_cross_down) and stoch_cross_down > 0.5:
                    score += 0.30 * direction  # Full confirmation
                elif not np.isnan(stoch_d) and stoch_k < stoch_d:
                    score += 0.15 * direction  # %K below %D but no fresh cross
                else:
                    score += 0.08 * direction  # Overbought, no cross yet

        # --- Regime bonus ---
        score += regime_bonus * direction

        # --- Price position filter ---
        price_range_pct = row.get("price_range_pct", np.nan)
        if not np.isnan(price_range_pct):
            if direction > 0 and price_range_pct < 0.30:
                score += 0.15 * direction  # Low in range — good for long
            elif direction < 0 and price_range_pct > 0.70:
                score += 0.15 * direction  # High in range — good for short

        return float(np.clip(score, -1.0, 1.0))

    def predict_signal(self, features: pd.DataFrame, atr: float,
                       current_price: float) -> Signal:
        """
        Override to use Keltner midline as TP target instead of fixed ATR multiple.
        """
        prob_up = self.predict_proba(features)

        # Get Keltner basis for midline target
        row = features.iloc[-1]
        keltner_basis = row.get("keltner_basis", np.nan)

        if prob_up >= self.risk.signal_threshold:
            direction = "BUY"
            confidence = prob_up
            sl = current_price - atr * self.risk.sl_atr_mult
            # TP = Keltner midline (mean reversion target)
            if not np.isnan(keltner_basis) and keltner_basis > current_price:
                tp = keltner_basis
            else:
                tp = current_price + atr * self.risk.tp_atr_mult
        elif prob_up <= (1.0 - self.risk.signal_threshold):
            direction = "SELL"
            confidence = 1.0 - prob_up
            sl = current_price + atr * self.risk.sl_atr_mult
            # TP = Keltner midline
            if not np.isnan(keltner_basis) and keltner_basis < current_price:
                tp = keltner_basis
            else:
                tp = current_price - atr * self.risk.tp_atr_mult
        else:
            return Signal(direction="HOLD", confidence=0.0)

        return Signal(
            direction=direction,
            confidence=confidence,
            strategies_agreed=[self.name],
            entry=current_price,
            stop_loss=sl,
            take_profit=tp,
            rationale=f"{self.name}: P(up)={prob_up:.3f}, target=keltner_midline",
        )
