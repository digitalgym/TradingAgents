"""
FVG Rebalance Strategy (Rule-Based)

Designed for crypto markets. Trades Fair Value Gap midpoint rebalancing
after a Change of Character (CHOCH) confirms potential reversal.

Core thesis:
  - FVGs are price inefficiencies that tend to get rebalanced
  - CHOCH signals trend shift (buyers/sellers losing control)
  - Enter at FVG midpoint AFTER CHOCH confirms the reversal
  - High R:R (3-5:1) with ~40% win rate = profitable

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.35  Unfilled FVG nearby (< 1.5 ATR) with price approaching midpoint
  +0.25  CHOCH detected confirming reversal direction
  +0.15  FVG is fresh (< 30 bars old, not yet tested)
  +0.10  Momentum supports rebalancing (RSI reversing from extreme)
  +0.10  Volume declining into the gap (exhaustion)
  +0.05  Premium/discount zone alignment

Key difference from SMC Zones:
  - Focuses specifically on FVG midpoint, not general zone proximity
  - Requires CHOCH confirmation (structure shift)
  - Higher R:R targets (3-5x ATR TP vs 2x ATR SL)
  - Designed for crypto's sharp moves and rebalancing tendency
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class FVGRebalanceFeatures(CompositeFeatures):
    """Technical + SMC features plus FVG-specific derived features."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[
                TechnicalFeatures(windows=windows),
                SMCFeatures(windows=windows),
            ],
            windows=windows,
        )
        self._extra_names = [
            "fvg_midpoint_proximity",      # How close price is to nearest FVG midpoint (ATR units)
            "fvg_direction_match",         # +1 if bullish FVG below price, -1 if bearish above
            "choch_with_fvg",              # CHOCH detected AND unfilled FVG exists
            "rsi_reversal",                # RSI reversing from extreme (oversold bounce or overbought rejection)
            "volume_exhaustion",           # Volume declining (exhaustion before reversal)
            "momentum_divergence",         # MACD histogram diverging from price
            "fvg_freshness",              # Inverse of FVG age (1.0 = brand new, 0.0 = old)
            "structure_strength",          # Combined BOS + CHOCH signal strength
        ]

    @property
    def feature_names(self) -> list:
        return super().feature_names + self._extra_names

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)

        n = len(features)
        close = df["close"].values.astype(float) if "close" in df.columns else np.zeros(n)
        high = df["high"].values.astype(float) if "high" in df.columns else np.zeros(n)
        low = df["low"].values.astype(float) if "low" in df.columns else np.zeros(n)

        # Initialize extra columns
        for col in self._extra_names:
            features[col] = np.nan

        for i in range(self.warmup_bars, n):
            atr = features.iloc[i].get("atr_14", np.nan)
            if np.isnan(atr) or atr <= 0:
                continue

            # FVG proximity — how close to midpoint (already in ATR units, closer = better)
            fvg_dist = features.iloc[i].get("nearest_fvg_dist_atr", np.nan)
            if not np.isnan(fvg_dist):
                # Convert distance to proximity score (1.0 = at midpoint, 0.0 = far away)
                features.iloc[i, features.columns.get_loc("fvg_midpoint_proximity")] = max(0, 1.0 - abs(fvg_dist) / 2.0)
            else:
                features.iloc[i, features.columns.get_loc("fvg_midpoint_proximity")] = 0.0

            # FVG direction match
            bull_fvg_fill = features.iloc[i].get("nearest_bull_fvg_fill_pct", 1.0)
            bear_fvg_fill = features.iloc[i].get("nearest_bear_fvg_fill_pct", 1.0)
            has_bull_fvg = bull_fvg_fill < 0.8  # Unfilled bullish FVG exists
            has_bear_fvg = bear_fvg_fill < 0.8  # Unfilled bearish FVG exists

            if has_bull_fvg and not has_bear_fvg:
                features.iloc[i, features.columns.get_loc("fvg_direction_match")] = 1.0
            elif has_bear_fvg and not has_bull_fvg:
                features.iloc[i, features.columns.get_loc("fvg_direction_match")] = -1.0
            elif has_bull_fvg and has_bear_fvg:
                # Both exist — use the closer one
                features.iloc[i, features.columns.get_loc("fvg_direction_match")] = 1.0 if fvg_dist and fvg_dist > 0 else -1.0
            else:
                features.iloc[i, features.columns.get_loc("fvg_direction_match")] = 0.0

            # CHOCH + FVG confluence
            choch = features.iloc[i].get("choch_detected", 0)
            unfilled_count = features.iloc[i].get("unfilled_fvg_count", 0)
            features.iloc[i, features.columns.get_loc("choch_with_fvg")] = 1.0 if (choch and unfilled_count > 0) else 0.0

            # RSI reversal detection
            rsi = features.iloc[i].get("rsi_14", 50)
            if not np.isnan(rsi):
                if rsi < 35:
                    # Oversold — potential bullish reversal
                    features.iloc[i, features.columns.get_loc("rsi_reversal")] = (35 - rsi) / 35  # 0 to 1
                elif rsi > 65:
                    # Overbought — potential bearish reversal
                    features.iloc[i, features.columns.get_loc("rsi_reversal")] = -(rsi - 65) / 35  # 0 to -1
                else:
                    features.iloc[i, features.columns.get_loc("rsi_reversal")] = 0.0

            # Volume exhaustion — volume declining over last 5 bars
            vol_ratio = features.iloc[i].get("volume_ratio", 1.0)
            if not np.isnan(vol_ratio):
                # Volume below average = exhaustion (favorable for reversal)
                features.iloc[i, features.columns.get_loc("volume_exhaustion")] = max(0, 1.0 - vol_ratio)
            else:
                features.iloc[i, features.columns.get_loc("volume_exhaustion")] = 0.0

            # Momentum divergence — MACD histogram changing direction
            macd_hist = features.iloc[i].get("macd_hist_norm", 0)
            if not np.isnan(macd_hist):
                # Magnitude of MACD histogram (direction change = reversal signal)
                features.iloc[i, features.columns.get_loc("momentum_divergence")] = macd_hist

            # FVG freshness
            zone_age = features.iloc[i].get("nearest_zone_age_bars", 100)
            if not np.isnan(zone_age):
                features.iloc[i, features.columns.get_loc("fvg_freshness")] = max(0, 1.0 - zone_age / 50)
            else:
                features.iloc[i, features.columns.get_loc("fvg_freshness")] = 0.0

            # Structure strength — combined BOS + CHOCH
            bos_bull = features.iloc[i].get("bos_bullish_recent", 0)
            bos_bear = features.iloc[i].get("bos_bearish_recent", 0)
            choch_val = 1 if choch else 0
            features.iloc[i, features.columns.get_loc("structure_strength")] = (
                bos_bull - bos_bear + choch_val * 0.5
            )

        return features


class FVGRebalanceStrategy(BaseStrategy):
    """
    FVG Rebalance: Enter at FVG midpoint after CHOCH confirms reversal.

    Designed for crypto's sharp moves and rebalancing tendency.
    Higher R:R (4:1) with ~40% expected win rate.
    """

    @property
    def name(self) -> str:
        return "fvg_rebalance"

    def get_feature_set(self) -> BaseFeatureSet:
        return FVGRebalanceFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # === GATE: Must have unfilled FVG nearby ===
        fvg_proximity = row.get("fvg_midpoint_proximity", 0)
        if np.isnan(fvg_proximity) or fvg_proximity < 0.25:
            return 0.0  # No FVG nearby or too far

        fvg_direction = row.get("fvg_direction_match", 0)
        if np.isnan(fvg_direction) or fvg_direction == 0:
            return 0.0  # No clear FVG direction

        # Direction multiplier: +1 for bullish, -1 for bearish
        direction = 1.0 if fvg_direction > 0 else -1.0

        # === COMPONENT 1: FVG proximity (35% weight) ===
        # Closer to midpoint = stronger signal
        score += 0.35 * fvg_proximity

        # === COMPONENT 2: CHOCH confirmation (25% weight) ===
        choch_with_fvg = row.get("choch_with_fvg", 0)
        choch_detected = row.get("choch_detected", 0)
        if choch_with_fvg:
            score += 0.25
        elif choch_detected:
            score += 0.10  # CHOCH without FVG confluence = weaker

        # === COMPONENT 3: FVG freshness (15% weight) ===
        freshness = row.get("fvg_freshness", 0)
        if not np.isnan(freshness):
            score += 0.15 * freshness

        # === COMPONENT 4: RSI reversal (10% weight) ===
        rsi_rev = row.get("rsi_reversal", 0)
        if not np.isnan(rsi_rev):
            # For bullish: positive rsi_reversal (oversold) adds to score
            # For bearish: negative rsi_reversal (overbought) adds to score
            if direction > 0 and rsi_rev > 0:
                score += 0.10 * rsi_rev
            elif direction < 0 and rsi_rev < 0:
                score += 0.10 * abs(rsi_rev)

        # === COMPONENT 5: Volume exhaustion (10% weight) ===
        vol_exhaustion = row.get("volume_exhaustion", 0)
        if not np.isnan(vol_exhaustion):
            score += 0.10 * vol_exhaustion

        # === COMPONENT 6: Premium/discount alignment (5% weight) ===
        pd_pct = row.get("premium_discount_pct", 0)
        if not np.isnan(pd_pct):
            # Bullish: want to be in discount (pd_pct < 0)
            # Bearish: want to be in premium (pd_pct > 0)
            if direction > 0 and pd_pct < -0.2:
                score += 0.05
            elif direction < 0 and pd_pct > 0.2:
                score += 0.05

        # === BONUS: OTE zone overlap ===
        in_ote = row.get("in_ote_zone", 0)
        if not np.isnan(in_ote) and in_ote > 0:
            score *= 1.15  # 15% bonus — FVG in OTE = highest probability

        # === BONUS: Liquidity sweep just occurred ===
        has_sweep = row.get("has_strong_sweep", 0)
        if not np.isnan(has_sweep) and has_sweep > 0:
            score *= 1.10  # 10% bonus — sweep before FVG fill = smart money trapped retail

        # === BONUS: CHOCH has strong displacement ===
        choch_disp = row.get("choch_displacement_strength", 0)
        if not np.isnan(choch_disp) and choch_disp > 1.5:
            score *= 1.10  # 10% bonus — strong displacement = genuine structure shift

        # === BONUS: Kill zone session ===
        in_kill = row.get("in_kill_zone", 0)
        if not np.isnan(in_kill) and in_kill > 0:
            score *= 1.05  # 5% bonus — London/NY overlap = highest liquidity

        # === PENALTY: Counter-trend without CHOCH ===
        adx = row.get("adx_14", 0)
        di_diff = row.get("di_diff", 0)
        if not np.isnan(adx) and not np.isnan(di_diff) and adx > 30:
            # Strong trend — penalize counter-trend without CHOCH
            trend_bullish = di_diff > 0
            if direction > 0 and not trend_bullish and not choch_detected:
                score *= 0.3  # Heavy penalty
            elif direction < 0 and trend_bullish and not choch_detected:
                score *= 0.3

        # === PENALTY: Weak displacement on BOS (unreliable structure) ===
        bos_disp = row.get("bos_displacement_strength", 0)
        if not np.isnan(bos_disp) and bos_disp > 0 and bos_disp < 0.8:
            score *= 0.7  # Weak BOS displacement = less reliable zone

        return min(1.0, score) * direction

    @property
    def default_risk(self):
        """Optuna-optimized defaults for FVG rebalance.

        Backtested 2026-04-04 on XAUUSD D1: PF=3.67, Sharpe=0.791, 88 trades.
        Key insight: wider stops (2.5x ATR) give FVG time to work.
        TP at 7.5x ATR with 3:1 R:R means ~25% WR is profitable.
        """
        from tradingagents.quant_strats.config import RiskDefaults
        return RiskDefaults(
            sl_atr_mult=2.5,       # Wide stop — FVGs need room to work
            tp_atr_mult=7.5,       # 3:1 R:R (7.5 / 2.5)
            signal_threshold=0.75,  # High conviction only
            max_hold_bars=20,       # Quick resolution expected
        )
