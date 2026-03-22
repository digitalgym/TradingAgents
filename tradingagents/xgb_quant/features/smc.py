"""
SMC (Smart Money Concepts) Features for XGBoost.

Wraps the existing SmartMoneyAnalyzer to extract numerical features
from SMC analysis for use as XGBoost model inputs.

For training: computes features on rolling windows (expensive, cached).
For live prediction: computes on latest data only.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List

from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.config import FeatureWindows

logger = logging.getLogger(__name__)


class SMCFeatures(BaseFeatureSet):
    """Extract numerical features from SmartMoneyAnalyzer output."""

    def __init__(self, windows: Optional[FeatureWindows] = None, smc_window: int = 200):
        super().__init__(windows)
        self.smc_window = smc_window

    @property
    def feature_names(self) -> list:
        return [
            # Order Blocks
            "nearest_bull_ob_strength",
            "nearest_bear_ob_strength",
            "nearest_ob_dist_atr",
            "ob_count_bull",
            "ob_count_bear",
            # Fair Value Gaps
            "nearest_bull_fvg_fill_pct",
            "nearest_bear_fvg_fill_pct",
            "nearest_fvg_dist_atr",
            "unfilled_fvg_count",
            # Confluence
            "confluence_score",
            "confluence_bullish_factors",
            "confluence_bearish_factors",
            # Premium/Discount
            "premium_discount_pct",
            # Structure
            "bos_bullish_recent",
            "bos_bearish_recent",
            "choch_detected",
            # Zone freshness
            "nearest_zone_age_bars",
        ]

    @property
    def warmup_bars(self) -> int:
        return self.smc_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SMC features for every bar from smc_window onwards.

        This is expensive — calls SmartMoneyAnalyzer for each bar's trailing window.
        Use feature caching for training data.
        """
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer

        analyzer = SmartMoneyAnalyzer()
        n = len(df)
        features = pd.DataFrame(index=df.index, columns=self.feature_names, dtype=float)
        features[:] = np.nan

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        # Compute ATR for distance normalisation
        atr = self._compute_atr_simple(high, low, close)

        success_count = 0
        error_count = 0
        last_error = None
        total_bars = n - self.smc_window

        logger.info(f"SMC features: computing for {total_bars} bars (window={self.smc_window})...")

        for i in range(self.smc_window, n):
            window_df = df.iloc[i - self.smc_window:i + 1].copy()
            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else 1.0

            try:
                smc = analyzer.analyze_full_smc(
                    window_df,
                    current_price=current_price,
                    use_structural_obs=True,
                    include_equal_levels=True,
                    include_breakers=False,
                    include_ote=False,
                    include_sweeps=False,
                    include_inducements=False,
                    include_rejections=False,
                    include_turtle_soup=False,
                )
                self._extract_features(features, i, smc, current_price, current_atr)
                success_count += 1
            except Exception as e:
                error_count += 1
                last_error = str(e)
                # Leave as NaN — XGBoost handles missing values

            if (i - self.smc_window) % 100 == 0 and i > self.smc_window:
                logger.debug(
                    f"  SMC progress: {i - self.smc_window}/{total_bars} bars, "
                    f"{success_count} ok, {error_count} errors"
                )

        logger.info(
            f"SMC features done: {success_count}/{total_bars} succeeded, "
            f"{error_count} errors"
        )
        if error_count > 0 and last_error:
            logger.warning(f"  Last SMC error: {last_error}")

        return features

    def compute_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMC features for only the latest bar (fast, for live prediction)."""
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer

        analyzer = SmartMoneyAnalyzer()
        n = len(df)
        features = pd.DataFrame(index=[df.index[-1]], columns=self.feature_names, dtype=float)
        features[:] = np.nan

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        current_price = close[-1]
        atr = self._compute_atr_simple(high, low, close)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 1.0

        start = max(0, n - self.smc_window)
        window_df = df.iloc[start:].copy()

        try:
            smc = analyzer.analyze_full_smc(
                window_df,
                current_price=current_price,
                use_structural_obs=True,
                include_equal_levels=True,
            )
            self._extract_features(features, 0, smc, current_price, current_atr)
        except Exception:
            pass

        return features

    def _extract_features(self, features: pd.DataFrame, idx: int,
                          smc: dict, price: float, atr: float):
        """Extract numerical features from SMC analysis dict.

        The SMC analyzer returns nested dicts:
          order_blocks: {bullish: [...], bearish: [...], total: int, ...}
          fair_value_gaps: {bullish: [...], bearish: [...], total: int, ...}
          structure: {all_bos: [...], all_choc: [...], ...}
          current_confluence: ConfluenceScore object
          premium_discount: PremiumDiscountZone object
        """
        # Order Blocks — stored as dict with bullish/bearish lists
        obs_data = smc.get("order_blocks", {})
        if isinstance(obs_data, dict):
            bull_obs = [ob for ob in obs_data.get("bullish", []) if not ob.mitigated]
            bear_obs = [ob for ob in obs_data.get("bearish", []) if not ob.mitigated]
        else:
            bull_obs = []
            bear_obs = []

        features.iloc[idx, features.columns.get_loc("ob_count_bull")] = len(bull_obs)
        features.iloc[idx, features.columns.get_loc("ob_count_bear")] = len(bear_obs)

        if bull_obs:
            nearest = min(bull_obs, key=lambda ob: abs(ob.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bull_ob_strength")] = nearest.strength
            features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")] = (price - nearest.midpoint) / max(atr, 1e-10)
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bull_ob_strength")] = 0.0

        if bear_obs:
            nearest = min(bear_obs, key=lambda ob: abs(ob.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bear_ob_strength")] = nearest.strength
            if np.isnan(features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")]):
                features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")] = (nearest.midpoint - price) / max(atr, 1e-10)
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bear_ob_strength")] = 0.0

        # FVGs — stored as dict with bullish/bearish lists, key is "fair_value_gaps"
        fvg_data = smc.get("fair_value_gaps", {})
        if isinstance(fvg_data, dict):
            bull_fvgs = [f for f in fvg_data.get("bullish", []) if not f.mitigated]
            bear_fvgs = [f for f in fvg_data.get("bearish", []) if not f.mitigated]
        else:
            bull_fvgs = []
            bear_fvgs = []

        features.iloc[idx, features.columns.get_loc("unfilled_fvg_count")] = len(bull_fvgs) + len(bear_fvgs)

        if bull_fvgs:
            nearest = min(bull_fvgs, key=lambda f: abs(f.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bull_fvg_fill_pct")] = nearest.fill_percentage / 100.0
            features.iloc[idx, features.columns.get_loc("nearest_fvg_dist_atr")] = (price - nearest.midpoint) / max(atr, 1e-10)
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bull_fvg_fill_pct")] = 1.0

        if bear_fvgs:
            nearest = min(bear_fvgs, key=lambda f: abs(f.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bear_fvg_fill_pct")] = nearest.fill_percentage / 100.0
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bear_fvg_fill_pct")] = 1.0

        # Confluence — key is "current_confluence"
        confluence = smc.get("current_confluence", None) or smc.get("confluence", None)
        if confluence is not None and hasattr(confluence, "total_score"):
            features.iloc[idx, features.columns.get_loc("confluence_score")] = confluence.total_score / 100.0
            features.iloc[idx, features.columns.get_loc("confluence_bullish_factors")] = confluence.bullish_factors
            features.iloc[idx, features.columns.get_loc("confluence_bearish_factors")] = confluence.bearish_factors
        else:
            features.iloc[idx, features.columns.get_loc("confluence_score")] = 0.0
            features.iloc[idx, features.columns.get_loc("confluence_bullish_factors")] = 0
            features.iloc[idx, features.columns.get_loc("confluence_bearish_factors")] = 0

        # Premium/Discount
        pd_zone = smc.get("premium_discount", None)
        if pd_zone is not None and hasattr(pd_zone, "position_pct"):
            features.iloc[idx, features.columns.get_loc("premium_discount_pct")] = pd_zone.position_pct / 100.0
        else:
            features.iloc[idx, features.columns.get_loc("premium_discount_pct")] = 0.5

        # Structure — nested in structure dict with all_bos/all_choc keys
        structure = smc.get("structure", {})
        if isinstance(structure, dict):
            bos_list = structure.get("all_bos", []) or structure.get("recent_bos", [])
            choch_list = structure.get("all_choc", []) or structure.get("recent_choc", [])
        else:
            bos_list = []
            choch_list = []

        recent_bull_bos = sum(1 for b in bos_list if hasattr(b, "type") and b.type == "bullish")
        recent_bear_bos = sum(1 for b in bos_list if hasattr(b, "type") and b.type == "bearish")
        features.iloc[idx, features.columns.get_loc("bos_bullish_recent")] = min(recent_bull_bos, 5)
        features.iloc[idx, features.columns.get_loc("bos_bearish_recent")] = min(recent_bear_bos, 5)
        features.iloc[idx, features.columns.get_loc("choch_detected")] = 1.0 if len(choch_list) > 0 else 0.0

        # Zone age
        all_zones = bull_obs + bear_obs
        if all_zones:
            nearest_zone = min(all_zones, key=lambda z: abs(z.midpoint - price))
            age = abs(idx - nearest_zone.candle_index) if hasattr(nearest_zone, "candle_index") else 50
            features.iloc[idx, features.columns.get_loc("nearest_zone_age_bars")] = min(age, 200)
        else:
            features.iloc[idx, features.columns.get_loc("nearest_zone_age_bars")] = 200

    @staticmethod
    def _compute_atr_simple(high, low, close, period: int = 14) -> np.ndarray:
        """Simple ATR for normalisation."""
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
