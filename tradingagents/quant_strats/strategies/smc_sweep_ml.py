"""
SMC Sweep ML Strategy (Focused, Lean)

Focused on a single high-edge SMC concept: liquidity sweep reversals.

Thesis: When price sweeps a liquidity level (equal highs/lows, swing
points) and gets rejected, smart money has grabbed liquidity and price
reverses. Enter on the rejection with tight SL beyond the sweep.

Feature set is deliberately lean (~25 features) to avoid overfitting:
  - Sweep detection (5): counts, rejection strength, ATR penetration
  - Equal levels / liquidity pools (4): counts, distance, touches
  - OTE zone (2): in zone, distance
  - Zone context (4): OB proximity/strength, FVG proximity, premium/discount
  - Trend filter (4): EMA cross, ADX, DI diff, regime
  - Volatility (3): ATR%, BB width, squeeze
  - Volume (2): volume ratio, POC distance
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.volume_profile import VolumeProfileFeatures
from tradingagents.quant_strats.features.regime import RegimeFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures
from tradingagents.quant_strats.config import RiskDefaults, MODELS_DIR

logger = logging.getLogger(__name__)

# The lean feature subset — only what matters for sweep reversals
_SWEEP_FEATURES = [
    # Sweep detection
    "sweep_bullish_recent",
    "sweep_bearish_recent",
    "nearest_sweep_rejection_strength",
    "nearest_sweep_atr_penetration",
    "has_strong_sweep",
    # Equal levels (liquidity pools)
    "equal_highs_count",
    "equal_lows_count",
    "nearest_equal_level_dist_atr",
    "max_equal_touches",
    # OTE
    "in_ote_zone",
    "ote_dist_atr",
    # Zone context
    "nearest_ob_dist_atr",
    "nearest_bull_ob_strength",
    "nearest_bear_ob_strength",
    "nearest_fvg_dist_atr",
    "premium_discount_pct",
    # Trend filter
    "ema_cross",
    "adx_14",
    "di_diff",
    # Volatility
    "atr_pct",
    "bb_width",
    "squeeze_strength",
    # Volume
    "volume_ratio",
    "poc_dist_atr",
]


class SweepFeatureSelector(CompositeFeatures):
    """Computes all features then selects only the sweep-relevant subset."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[
                TechnicalFeatures(windows=windows),
                SMCFeatures(windows=windows),
                VolumeProfileFeatures(windows=windows),
                RegimeFeatures(windows=windows),
            ],
            windows=windows,
        )

    @property
    def feature_names(self) -> list:
        return _SWEEP_FEATURES

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        full = super().compute(df)
        # Select only sweep-relevant features, fill missing with NaN
        selected = pd.DataFrame(index=full.index, dtype=float)
        for col in _SWEEP_FEATURES:
            if col in full.columns:
                selected[col] = full[col]
            else:
                selected[col] = np.nan
        return selected

    def compute_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        full = super().compute_latest(df)
        selected = pd.DataFrame(index=full.index, dtype=float)
        for col in _SWEEP_FEATURES:
            if col in full.columns:
                selected[col] = full[col]
            else:
                selected[col] = np.nan
        return selected


class SMCSweepMLStrategy(BaseStrategy):
    """
    Focused sweep-reversal strategy with lean XGBoost model.

    Only 24 features — reduces overfitting risk on limited data.
    """

    def __init__(self, risk=None, windows=None):
        super().__init__(risk=risk, windows=windows)
        self._model = None
        self._feature_names = None

    @property
    def name(self):
        return "smc_sweep_ml"

    def get_feature_set(self) -> BaseFeatureSet:
        return SweepFeatureSelector(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        """Rule-based sweep scoring as fallback."""
        row = features.iloc[idx]
        score = 0.0

        # Gate: must have a recent sweep
        bull_sweeps = row.get("sweep_bullish_recent", 0)
        bear_sweeps = row.get("sweep_bearish_recent", 0)
        has_sweep = row.get("has_strong_sweep", 0)
        rejection = row.get("nearest_sweep_rejection_strength", 0)

        if bull_sweeps == 0 and bear_sweeps == 0:
            return 0.0  # No sweep = no trade

        # Direction from sweep type (bullish sweep = buy signal)
        if bull_sweeps > bear_sweeps:
            direction = 1.0
        elif bear_sweeps > bull_sweeps:
            direction = -1.0
        else:
            return 0.0

        # Strong sweep bonus
        if has_sweep > 0.5:
            score += 0.30 * direction
        elif rejection > 0.3:
            score += 0.15 * direction
        else:
            score += 0.08 * direction

        # Liquidity pool depth (more touches = more stops grabbed)
        touches = row.get("max_equal_touches", 0)
        if touches >= 3:
            score += 0.15 * direction
        elif touches >= 2:
            score += 0.08 * direction

        # OTE alignment
        in_ote = row.get("in_ote_zone", 0)
        if in_ote > 0.5:
            score += 0.12 * direction

        # OB nearby (additional support/resistance)
        ob_dist = abs(row.get("nearest_ob_dist_atr", 10))
        if ob_dist < 1.5:
            score += 0.08 * direction

        # Premium/discount alignment
        pd_pct = row.get("premium_discount_pct", 0.5)
        if direction > 0 and pd_pct < 0.4:
            score += 0.08 * direction
        elif direction < 0 and pd_pct > 0.6:
            score += 0.08 * direction

        # Trend alignment
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            if (direction > 0 and ema_cross > 0.5) or (direction < 0 and ema_cross < 0.5):
                score += 0.10 * direction

        # Squeeze = pending breakout — sweep more meaningful
        squeeze = row.get("squeeze_strength", 0)
        if squeeze > 0.5:
            score *= 1.10

        return float(np.clip(score, -1.0, 1.0))

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is not None:
            try:
                feature_cols = self._feature_names or _SWEEP_FEATURES
                available = [c for c in feature_cols if c in features.columns]
                X = features[available].values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return self._model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.debug(f"ML prediction failed, falling back to rules: {e}")

        return super().predict_proba_batch(features)

    def train_model(self, features: pd.DataFrame, labels: np.ndarray):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost not installed, using rule-based only")
            return

        feature_cols = [c for c in features.columns if c in _SWEEP_FEATURES]
        self._feature_names = feature_cols

        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        valid_mask = ~np.isnan(labels)
        X_train = X[valid_mask]
        y_train = labels[valid_mask].astype(int)

        if len(X_train) < 30 or y_train.sum() < 5 or (1 - y_train).sum() < 5:
            logger.debug(f"Not enough training data ({len(X_train)} samples, {y_train.sum()} positive)")
            return

        # Lean model for lean features — less regularization needed
        self._model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,     # 24 features — less subsampling needed
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X_train, y_train)

        importances = self._model.feature_importances_
        top_features = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        logger.info(
            f"SMC Sweep ML trained on {len(X_train)} samples. "
            f"Top features: {', '.join(f'{n}={v:.3f}' for n, v in top_features)}"
        )

    def save_model(self, symbol: str, timeframe: str):
        if self._model is None:
            return
        model_dir = MODELS_DIR / self.name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{symbol}_{timeframe}_xgb.json"
        self._model.save_model(str(model_path))

        meta_path = model_dir / f"{symbol}_{timeframe}.json"
        meta = {
            "strategy": self.name,
            "risk": {
                "sl_atr_mult": self.risk.sl_atr_mult,
                "tp_atr_mult": self.risk.tp_atr_mult,
                "signal_threshold": self.risk.signal_threshold,
                "max_hold_bars": self.risk.max_hold_bars,
            },
            "feature_names": self._feature_names,
            "model_file": f"{symbol}_{timeframe}_xgb.json",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved SMC Sweep ML model to {model_path}")

    def load_model(self, symbol: str, timeframe: str) -> bool:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return False
        model_dir = MODELS_DIR / self.name
        model_path = model_dir / f"{symbol}_{timeframe}_xgb.json"
        meta_path = model_dir / f"{symbol}_{timeframe}.json"
        if not model_path.exists():
            return False
        try:
            self._model = XGBClassifier()
            self._model.load_model(str(model_path))
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._feature_names = meta.get("feature_names")
            logger.info(f"Loaded SMC Sweep ML model from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SMC Sweep ML model: {e}")
            self._model = None
            return False

    @property
    def default_risk(self):
        """Sweep reversals: tight SL beyond sweep, wide TP."""
        return RiskDefaults(
            sl_atr_mult=1.5,
            tp_atr_mult=4.5,
            signal_threshold=0.65,
            max_hold_bars=20,
        )
