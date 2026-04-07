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

from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.config import FeatureWindows

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
            # --- Liquidity Sweeps ---
            "sweep_bullish_recent",
            "sweep_bearish_recent",
            "nearest_sweep_rejection_strength",
            "nearest_sweep_atr_penetration",
            "has_strong_sweep",
            # --- Breaker Blocks ---
            "breaker_bull_count",
            "breaker_bear_count",
            "nearest_breaker_dist_atr",
            "nearest_breaker_strength",
            # --- OTE Zones ---
            "in_ote_zone",
            "ote_dist_atr",
            # --- Equal Levels ---
            "equal_highs_count",
            "equal_lows_count",
            "nearest_equal_level_dist_atr",
            "max_equal_touches",
            # --- Inducements & Rejections ---
            "inducement_recent_count",
            "rejection_block_count",
            "nearest_rejection_wick_atr",
            # --- Displacement Strength ---
            "bos_displacement_strength",    # Body size of BOS candle / ATR (>1.5 = strong)
            "choch_displacement_strength",  # Body size of CHoCH candle / ATR
            # --- Session Filter ---
            "in_london_session",            # 1.0 if bar is during London session (07-16 UTC)
            "in_ny_session",                # 1.0 if bar is during NY session (12-21 UTC)
            "in_kill_zone",                 # 1.0 if bar is in London/NY overlap (12-16 UTC)
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

        # Pre-compute session features from bar timestamps
        bar_hours = None
        if "time" in df.columns:
            try:
                bar_hours = pd.to_datetime(df["time"]).dt.hour.values
            except Exception:
                pass

        # Pre-compute candle body sizes for displacement strength
        body_sizes = np.abs(close - df["open"].values.astype(float)) if "open" in df.columns else np.zeros(n)

        for i in range(self.smc_window, n):
            window_df = df.iloc[i - self.smc_window:i + 1].copy()
            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else 1.0

            # Session features (UTC hours)
            if bar_hours is not None:
                h = bar_hours[i]
                features.iloc[i, features.columns.get_loc("in_london_session")] = 1.0 if 7 <= h < 16 else 0.0
                features.iloc[i, features.columns.get_loc("in_ny_session")] = 1.0 if 12 <= h < 21 else 0.0
                features.iloc[i, features.columns.get_loc("in_kill_zone")] = 1.0 if 12 <= h < 16 else 0.0
            else:
                features.iloc[i, features.columns.get_loc("in_london_session")] = 0.5
                features.iloc[i, features.columns.get_loc("in_ny_session")] = 0.5
                features.iloc[i, features.columns.get_loc("in_kill_zone")] = 0.5

            try:
                smc = analyzer.analyze_full_smc(
                    window_df,
                    current_price=current_price,
                    use_structural_obs=True,
                    include_equal_levels=True,
                    include_breakers=True,
                    include_ote=True,
                    include_sweeps=True,
                    include_inducements=True,
                    include_rejections=True,
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
                include_breakers=True,
                include_ote=True,
                include_sweeps=True,
                include_inducements=True,
                include_rejections=True,
                include_turtle_soup=False,
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
          liquidity_sweeps: {bullish: [...], bearish: [...], ...}
          breaker_blocks: {bullish: [...], bearish: [...], ...}
          ote_zones: list of OTEZone
          equal_levels: {equal_highs: [...], equal_lows: [...], ...}
          inducements: {bullish: [...], bearish: [...], ...}
          rejection_blocks: {bullish: [...], bearish: [...], ...}
        """
        safe_atr = max(atr, 1e-10)

        # ─── Order Blocks ───
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
            features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")] = (price - nearest.midpoint) / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bull_ob_strength")] = 0.0

        if bear_obs:
            nearest = min(bear_obs, key=lambda ob: abs(ob.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bear_ob_strength")] = nearest.strength
            if np.isnan(features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")]):
                features.iloc[idx, features.columns.get_loc("nearest_ob_dist_atr")] = (nearest.midpoint - price) / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bear_ob_strength")] = 0.0

        # ─── Fair Value Gaps ───
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
            features.iloc[idx, features.columns.get_loc("nearest_fvg_dist_atr")] = (price - nearest.midpoint) / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bull_fvg_fill_pct")] = 1.0

        if bear_fvgs:
            nearest = min(bear_fvgs, key=lambda f: abs(f.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_bear_fvg_fill_pct")] = nearest.fill_percentage / 100.0
        else:
            features.iloc[idx, features.columns.get_loc("nearest_bear_fvg_fill_pct")] = 1.0

        # ─── Confluence ───
        confluence = smc.get("current_confluence", None) or smc.get("confluence", None)
        if confluence is not None and hasattr(confluence, "total_score"):
            features.iloc[idx, features.columns.get_loc("confluence_score")] = confluence.total_score / 100.0
            features.iloc[idx, features.columns.get_loc("confluence_bullish_factors")] = confluence.bullish_factors
            features.iloc[idx, features.columns.get_loc("confluence_bearish_factors")] = confluence.bearish_factors
        else:
            features.iloc[idx, features.columns.get_loc("confluence_score")] = 0.0
            features.iloc[idx, features.columns.get_loc("confluence_bullish_factors")] = 0
            features.iloc[idx, features.columns.get_loc("confluence_bearish_factors")] = 0

        # ─── Premium/Discount ───
        pd_zone = smc.get("premium_discount", None)
        if pd_zone is not None and hasattr(pd_zone, "position_pct"):
            features.iloc[idx, features.columns.get_loc("premium_discount_pct")] = pd_zone.position_pct / 100.0
        else:
            features.iloc[idx, features.columns.get_loc("premium_discount_pct")] = 0.5

        # ─── Structure (BOS / CHoCH) ───
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

        # ─── Zone Age ───
        all_zones = bull_obs + bear_obs
        if all_zones:
            nearest_zone = min(all_zones, key=lambda z: abs(z.midpoint - price))
            age = abs(idx - nearest_zone.candle_index) if hasattr(nearest_zone, "candle_index") else 50
            features.iloc[idx, features.columns.get_loc("nearest_zone_age_bars")] = min(age, 200)
        else:
            features.iloc[idx, features.columns.get_loc("nearest_zone_age_bars")] = 200

        # ─── Liquidity Sweeps ───
        sweep_data = smc.get("liquidity_sweeps", {})
        if isinstance(sweep_data, dict):
            bull_sweeps = sweep_data.get("bullish", [])
            bear_sweeps = sweep_data.get("bearish", [])
        else:
            bull_sweeps = []
            bear_sweeps = []

        features.iloc[idx, features.columns.get_loc("sweep_bullish_recent")] = len(bull_sweeps)
        features.iloc[idx, features.columns.get_loc("sweep_bearish_recent")] = len(bear_sweeps)

        all_sweeps = bull_sweeps + bear_sweeps
        if all_sweeps:
            # Most recent sweep (highest candle index)
            nearest = max(all_sweeps, key=lambda s: s.sweep_candle_index)
            features.iloc[idx, features.columns.get_loc("nearest_sweep_rejection_strength")] = nearest.rejection_strength
            features.iloc[idx, features.columns.get_loc("nearest_sweep_atr_penetration")] = nearest.atr_penetration
            features.iloc[idx, features.columns.get_loc("has_strong_sweep")] = 1.0 if nearest.is_strong else 0.0
        else:
            features.iloc[idx, features.columns.get_loc("nearest_sweep_rejection_strength")] = 0.0
            features.iloc[idx, features.columns.get_loc("nearest_sweep_atr_penetration")] = 0.0
            features.iloc[idx, features.columns.get_loc("has_strong_sweep")] = 0.0

        # ─── Breaker Blocks ───
        breaker_data = smc.get("breaker_blocks", {})
        if isinstance(breaker_data, dict):
            bull_breakers = [b for b in breaker_data.get("bullish", []) if not b.mitigated]
            bear_breakers = [b for b in breaker_data.get("bearish", []) if not b.mitigated]
        else:
            bull_breakers = []
            bear_breakers = []

        features.iloc[idx, features.columns.get_loc("breaker_bull_count")] = len(bull_breakers)
        features.iloc[idx, features.columns.get_loc("breaker_bear_count")] = len(bear_breakers)

        all_breakers = bull_breakers + bear_breakers
        if all_breakers:
            nearest = min(all_breakers, key=lambda b: abs(b.midpoint - price))
            features.iloc[idx, features.columns.get_loc("nearest_breaker_dist_atr")] = abs(nearest.midpoint - price) / safe_atr
            features.iloc[idx, features.columns.get_loc("nearest_breaker_strength")] = nearest.strength
        else:
            features.iloc[idx, features.columns.get_loc("nearest_breaker_dist_atr")] = 10.0  # far away
            features.iloc[idx, features.columns.get_loc("nearest_breaker_strength")] = 0.0

        # ─── OTE Zones ───
        ote_data = smc.get("ote_zones", [])
        if not isinstance(ote_data, list):
            ote_data = []

        in_ote = False
        min_ote_dist = float("inf")
        for ote in ote_data:
            mid = ote.midpoint
            dist = abs(price - mid)
            if dist < min_ote_dist:
                min_ote_dist = dist
            if ote.bottom <= price <= ote.top:
                in_ote = True

        features.iloc[idx, features.columns.get_loc("in_ote_zone")] = 1.0 if in_ote else 0.0
        if min_ote_dist < float("inf"):
            features.iloc[idx, features.columns.get_loc("ote_dist_atr")] = min_ote_dist / safe_atr
        else:
            features.iloc[idx, features.columns.get_loc("ote_dist_atr")] = 10.0

        # ─── Equal Levels ───
        eq_data = smc.get("equal_levels", {})
        if isinstance(eq_data, dict):
            eq_highs = [e for e in eq_data.get("equal_highs", []) if not e.swept]
            eq_lows = [e for e in eq_data.get("equal_lows", []) if not e.swept]
        else:
            eq_highs = []
            eq_lows = []

        features.iloc[idx, features.columns.get_loc("equal_highs_count")] = len(eq_highs)
        features.iloc[idx, features.columns.get_loc("equal_lows_count")] = len(eq_lows)

        all_eq = eq_highs + eq_lows
        if all_eq:
            nearest = min(all_eq, key=lambda e: abs(e.price - price))
            features.iloc[idx, features.columns.get_loc("nearest_equal_level_dist_atr")] = abs(nearest.price - price) / safe_atr
            features.iloc[idx, features.columns.get_loc("max_equal_touches")] = nearest.touches
        else:
            features.iloc[idx, features.columns.get_loc("nearest_equal_level_dist_atr")] = 10.0
            features.iloc[idx, features.columns.get_loc("max_equal_touches")] = 0

        # ─── Inducements ───
        ind_data = smc.get("inducements", {})
        if isinstance(ind_data, dict):
            all_inds = ind_data.get("bullish", []) + ind_data.get("bearish", [])
        else:
            all_inds = []
        features.iloc[idx, features.columns.get_loc("inducement_recent_count")] = len(all_inds)

        # ─── Rejection Blocks ───
        rej_data = smc.get("rejection_blocks", {})
        if isinstance(rej_data, dict):
            unmit_rejs = [r for r in rej_data.get("bullish", []) + rej_data.get("bearish", []) if not r.mitigated]
        else:
            unmit_rejs = []

        features.iloc[idx, features.columns.get_loc("rejection_block_count")] = len(unmit_rejs)

        if unmit_rejs:
            nearest = min(unmit_rejs, key=lambda r: abs(r.rejection_price - price))
            features.iloc[idx, features.columns.get_loc("nearest_rejection_wick_atr")] = nearest.wick_atr_ratio
        else:
            features.iloc[idx, features.columns.get_loc("nearest_rejection_wick_atr")] = 0.0

        # ─── Displacement Strength ───
        # Measure candle body size of BOS/CHoCH events relative to ATR
        # Strong displacement (> 1.5x ATR) = more reliable structure break
        bos_disp = 0.0
        if bos_list:
            for b in bos_list:
                if hasattr(b, "displacement_atr"):
                    bos_disp = max(bos_disp, b.displacement_atr)
                elif hasattr(b, "candle_index"):
                    # Estimate from candle body at the BOS bar
                    pass  # Will be 0.0 if no displacement_atr attr
        features.iloc[idx, features.columns.get_loc("bos_displacement_strength")] = bos_disp

        choch_disp = 0.0
        if choch_list:
            for c in choch_list:
                if hasattr(c, "displacement_atr"):
                    choch_disp = max(choch_disp, c.displacement_atr)
        features.iloc[idx, features.columns.get_loc("choch_displacement_strength")] = choch_disp

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
