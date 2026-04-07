"""
FVG Rebalance ML Strategy (XGBoost Model)

Same features as FVGRebalanceStrategy but uses an XGBoost classifier
to learn which FVG setups actually work, rather than fixed scoring weights.

The model learns:
- Which feature combinations predict successful FVG rebalances
- Non-linear interactions (e.g., CHOCH + low volume + fresh zone = high probability)
- Per-pair differences in FVG behavior
- Optimal weighting of each feature

Falls back to rule-based scoring if no trained model is available.
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
from tradingagents.quant_strats.config import RiskDefaults, MODELS_DIR

logger = logging.getLogger(__name__)


class FVGRebalanceMLStrategy(BaseStrategy):
    """
    XGBoost-powered FVG Rebalance strategy.

    Uses the same FVGRebalanceFeatures but trains an XGBClassifier to predict
    which setups will be profitable. Walk-forward training prevents lookahead bias.
    """

    def __init__(self, risk=None, windows=None):
        super().__init__(risk=risk, windows=windows)
        self._model = None
        self._feature_names = None
        self._rule_based = FVGRebalanceStrategy(risk=risk, windows=windows)

    @property
    def name(self):
        return "fvg_rebalance_ml"

    def get_feature_set(self) -> BaseFeatureSet:
        return FVGRebalanceFeatures(windows=self.windows)

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        """Fall back to rule-based scoring if no model."""
        return self._rule_based.score_bar(features, idx)

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """
        If a trained model exists, use it. Otherwise fall back to rule-based.

        This is called by the trainer during walk-forward backtesting.
        The trainer handles train/test splits — this method just predicts.
        """
        if self._model is not None:
            try:
                feature_cols = self._feature_names or [
                    c for c in features.columns
                    if c not in ("time", "open", "high", "low", "close", "volume", "tick_volume")
                ]
                X = features[feature_cols].values
                # Replace NaN/inf with 0 for XGBoost
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return self._model.predict_proba(X)[:, 1]  # P(up)
            except Exception as e:
                logger.debug(f"ML prediction failed, falling back to rules: {e}")

        # Fall back to rule-based
        return super().predict_proba_batch(features)

    def train_model(self, features: pd.DataFrame, labels: np.ndarray):
        """
        Train XGBoost classifier on features → direction labels.

        Args:
            features: Feature DataFrame (from compute())
            labels: Binary labels (1 = profitable trade, 0 = losing trade)
        """
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

        # Filter out rows where we don't have labels
        valid_mask = ~np.isnan(labels)
        X_train = X[valid_mask]
        y_train = labels[valid_mask].astype(int)

        if len(X_train) < 30 or y_train.sum() < 5 or (1 - y_train).sum() < 5:
            logger.debug(f"Not enough training data ({len(X_train)} samples, {y_train.sum()} positive)")
            return

        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X_train, y_train)

        # Log feature importance
        importances = self._model.feature_importances_
        top_features = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        logger.info(f"FVG ML model trained on {len(X_train)} samples. "
                     f"Top features: {', '.join(f'{n}={v:.3f}' for n, v in top_features)}")

    def save_model(self, symbol: str, timeframe: str):
        """Save trained XGBoost model."""
        if self._model is None:
            return

        model_dir = MODELS_DIR / self.name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        model_path = model_dir / f"{symbol}_{timeframe}_xgb.json"
        self._model.save_model(str(model_path))

        # Save metadata
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

        logger.info(f"Saved FVG ML model to {model_path}")

    def load_model(self, symbol: str, timeframe: str) -> bool:
        """Load a trained XGBoost model."""
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

            logger.info(f"Loaded FVG ML model from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load FVG ML model: {e}")
            self._model = None
            return False

    @property
    def default_risk(self):
        """Same optimized defaults as rule-based FVG rebalance."""
        return RiskDefaults(
            sl_atr_mult=2.5,
            tp_atr_mult=7.5,
            signal_threshold=0.60,  # Lower threshold — ML model is more calibrated
            max_hold_bars=20,
        )
