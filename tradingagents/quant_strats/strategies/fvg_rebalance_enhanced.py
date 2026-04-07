"""
FVG Rebalance Enhanced ML Strategy

Same FVG midpoint rebalancing thesis as fvg_rebalance_ml, but with
additional Volume Profile and session features for better context.

Feature composition:
  FVGRebalanceFeatures  ~67 features  (Technical + expanded SMC + 8 FVG-derived)
  VolumeProfileFeatures ~10 features  (POC, VAH, VAL, HVN, LVN)
  ─────────────────────────────────
  Total                 ~77 features

The hypothesis: FVG rebalancing works better when you know WHERE the
FVG sits relative to value area (POC/VAH/VAL) and volume nodes.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal
from tradingagents.quant_strats.strategies.fvg_rebalance import FVGRebalanceFeatures, FVGRebalanceStrategy
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.volume_profile import VolumeProfileFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures
from tradingagents.quant_strats.config import RiskDefaults, MODELS_DIR

logger = logging.getLogger(__name__)


class FVGRebalanceEnhancedFeatures(CompositeFeatures):
    """FVG Rebalance features + Volume Profile."""

    def __init__(self, windows=None):
        super().__init__(
            providers=[
                FVGRebalanceFeatures(windows=windows),
                VolumeProfileFeatures(windows=windows),
            ],
            windows=windows,
        )


class FVGRebalanceEnhancedStrategy(BaseStrategy):
    """
    Enhanced FVG Rebalance with Volume Profile context.

    Same core thesis — FVG midpoint entry after CHOCH — but the XGBoost
    model also sees VP features to learn whether FVGs near POC/VAH/VAL
    have higher fill probability.
    """

    def __init__(self, risk=None, windows=None):
        super().__init__(risk=risk, windows=windows)
        self._model = None
        self._feature_names = None
        self._rule_based = FVGRebalanceStrategy(risk=risk, windows=windows)

    @property
    def name(self):
        return "fvg_rebalance_enhanced"

    def get_feature_set(self) -> BaseFeatureSet:
        return FVGRebalanceEnhancedFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        """Fall back to rule-based FVG scoring."""
        return self._rule_based.score_bar(features, idx)

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Use XGBoost if trained, otherwise rule-based."""
        if self._model is not None:
            try:
                feature_cols = self._feature_names or [
                    c for c in features.columns
                    if c not in ("time", "open", "high", "low", "close", "volume", "tick_volume")
                ]
                X = features[feature_cols].values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return self._model.predict_proba(X)[:, 1]
            except Exception as e:
                logger.debug(f"ML prediction failed, falling back to rules: {e}")

        return super().predict_proba_batch(features)

    def train_model(self, features: pd.DataFrame, labels: np.ndarray):
        """Train XGBoost on FVG + VP features."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("xgboost not installed, using rule-based only")
            return

        feature_cols = [
            c for c in features.columns
            if c not in ("time", "open", "high", "low", "close", "volume", "tick_volume")
        ]
        self._feature_names = feature_cols

        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        valid_mask = ~np.isnan(labels)
        X_train = X[valid_mask]
        y_train = labels[valid_mask].astype(int)

        if len(X_train) < 30 or y_train.sum() < 5 or (1 - y_train).sum() < 5:
            logger.debug(f"Not enough training data ({len(X_train)} samples, {y_train.sum()} positive)")
            return

        self._model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=1.5,
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
        )[:12]
        logger.info(
            f"FVG Enhanced ML trained on {len(X_train)} samples. "
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
        logger.info(f"Saved FVG Enhanced ML model to {model_path}")

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
            logger.info(f"Loaded FVG Enhanced ML model from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load FVG Enhanced ML model: {e}")
            self._model = None
            return False

    @property
    def default_risk(self):
        """Same as fvg_rebalance_ml — proven defaults."""
        return RiskDefaults(
            sl_atr_mult=2.5,
            tp_atr_mult=7.5,
            signal_threshold=0.60,
            max_hold_bars=20,
        )
