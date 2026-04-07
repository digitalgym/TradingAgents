"""
SMC Zones Strategy (Rule-Based)

Uses Smart Money Concept zone proximity and quality to generate signals.
Enters when price approaches a strong zone with trend alignment.

Scoring rules (max +1.0 bullish, -1.0 bearish):
  +0.30  Price near strong bullish zone (OB/FVG)
  +0.25  Zone is fresh (not yet tested)
  +0.20  Higher timeframe trend aligned
  +0.15  RSI not overbought (room to move)
  +0.10  Volume confirmation
"""

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures


class SMCZonesStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "smc_zones"

    def get_feature_set(self) -> BaseFeatureSet:
        return CompositeFeatures(
            providers=[
                TechnicalFeatures(windows=self.windows),
                SMCFeatures(windows=self.windows),
            ],
            windows=self.windows,
        )

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        row = features.iloc[idx]
        score = 0.0

        # Zone proximity and direction
        zone_dist = row.get("nearest_zone_dist_atr", np.nan)
        zone_type = row.get("nearest_zone_type", np.nan)  # 1=bullish, -1=bearish
        zone_strength = row.get("nearest_zone_strength", np.nan)
        zone_age = row.get("nearest_zone_age_bars", np.nan)

        # No zone nearby = no signal
        if np.isnan(zone_dist) or zone_dist > 2.0:
            return 0.0

        # Determine direction from zone type
        if np.isnan(zone_type):
            return 0.0
        direction = 1.0 if zone_type > 0 else -1.0

        # 1. Zone proximity (closer = stronger signal)
        if zone_dist < 0.5:
            score += 0.30 * direction  # Very close
        elif zone_dist < 1.0:
            score += 0.20 * direction  # Close
        else:
            score += 0.10 * direction  # Approaching

        # 2. Zone strength
        if not np.isnan(zone_strength):
            if zone_strength > 0.7:
                score += 0.25 * direction
            elif zone_strength > 0.4:
                score += 0.15 * direction

        # 3. Zone freshness (untested zones are stronger)
        if not np.isnan(zone_age) and zone_age < 50:
            score += 0.10 * direction  # Fresh zone

        # 4. Trend alignment (EMA cross)
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            if (direction > 0 and ema_cross > 0.5) or \
               (direction < 0 and ema_cross < 0.5):
                score += 0.15 * direction  # With trend

        # 5. RSI filter — don't buy overbought, don't sell oversold
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if direction > 0 and rsi < 65:
                score += 0.10 * direction  # Room to move up
            elif direction < 0 and rsi > 35:
                score += 0.10 * direction  # Room to move down
            elif (direction > 0 and rsi > 75) or (direction < 0 and rsi < 25):
                score *= 0.5  # Cut signal in half — overextended

        # 6. OTE zone overlap (highest probability entries)
        in_ote = row.get("in_ote_zone", 0)
        if not np.isnan(in_ote) and in_ote > 0:
            score *= 1.15  # 15% bonus

        # 7. Liquidity sweep confirmation
        has_sweep = row.get("has_strong_sweep", 0)
        if not np.isnan(has_sweep) and has_sweep > 0:
            score *= 1.10  # 10% bonus — sweep trapped retail before reversal

        # 8. Kill zone session bonus
        in_kill = row.get("in_kill_zone", 0)
        if not np.isnan(in_kill) and in_kill > 0:
            score *= 1.05  # 5% bonus — institutional session

        # 9. Strong displacement penalty/bonus
        bos_disp = row.get("bos_displacement_strength", 0)
        if not np.isnan(bos_disp) and bos_disp > 1.5:
            score *= 1.10  # Strong structure = reliable zone
        elif not np.isnan(bos_disp) and 0 < bos_disp < 0.8:
            score *= 0.7  # Weak displacement = less reliable

        return float(np.clip(score, -1.0, 1.0))
