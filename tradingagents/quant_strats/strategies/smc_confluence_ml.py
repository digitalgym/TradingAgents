"""
SMC Confluence ML Strategy (XGBoost Model)

Unified SMC strategy that combines ALL Smart Money Concepts with Volume Profile,
Regime, and Session features into a single XGBoost classifier.

Feature composition (~86 features):
  TechnicalFeatures     ~24 features  (EMA, RSI, MACD, BB, ADX, volume, etc.)
  SMCFeatures           ~35 features  (OB, FVG, BOS, CHoCH, sweeps, breakers,
                                       OTE, equal levels, inducements, rejections)
  VolumeProfileFeatures ~10 features  (POC, VAH, VAL, HVN, LVN)
  RegimeFeatures        ~9 features   (trend, volatility, squeeze)
  SessionFeatures       ~8 features   (killzone, session flags, day encoding)

The model learns non-linear interactions between ALL SMC concepts —
e.g., liquidity sweep + OTE zone + killzone + trending regime = high probability.

Falls back to enhanced rule-based scoring if no trained model is available.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal
from tradingagents.quant_strats.features.base import BaseFeatureSet
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.features.smc import SMCFeatures
from tradingagents.quant_strats.features.volume_profile import VolumeProfileFeatures
from tradingagents.quant_strats.features.regime import RegimeFeatures
from tradingagents.quant_strats.features.session import SessionFeatures
from tradingagents.quant_strats.features.composite import CompositeFeatures
from tradingagents.quant_strats.config import RiskDefaults, MODELS_DIR

logger = logging.getLogger(__name__)


class SMCConfluenceMLStrategy(BaseStrategy):
    """
    XGBoost-powered SMC Confluence strategy.

    Combines all available feature sets and trains an XGBClassifier to predict
    which SMC setups will be profitable. Walk-forward training prevents lookahead bias.
    """

    def __init__(self, risk=None, windows=None):
        super().__init__(risk=risk, windows=windows)
        self._model = None
        self._feature_names = None

    @property
    def name(self):
        return "smc_confluence_ml"

    def get_feature_set(self) -> BaseFeatureSet:
        return CompositeFeatures(
            providers=[
                TechnicalFeatures(windows=self.windows),
                SMCFeatures(windows=self.windows),
                VolumeProfileFeatures(windows=self.windows),
                RegimeFeatures(windows=self.windows),
                SessionFeatures(windows=self.windows),
            ],
            windows=self.windows,
        )

    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        """
        Enhanced rule-based scoring as fallback when no model is trained.

        Extends smc_zones logic with sweep, breaker, OTE, and VP confluence.
        """
        row = features.iloc[idx]
        score = 0.0

        # ─── Determine directional bias from structure ───
        bull_bos = row.get("bos_bullish_recent", 0)
        bear_bos = row.get("bos_bearish_recent", 0)
        choch = row.get("choch_detected", 0)
        bull_conf = row.get("confluence_bullish_factors", 0)
        bear_conf = row.get("confluence_bearish_factors", 0)

        # Net structural direction
        struct_bias = 0.0
        if bull_bos > bear_bos:
            struct_bias = 1.0
        elif bear_bos > bull_bos:
            struct_bias = -1.0
        elif bull_conf > bear_conf:
            struct_bias = 1.0
        elif bear_conf > bull_conf:
            struct_bias = -1.0

        if struct_bias == 0.0:
            return 0.0  # No clear direction

        direction = struct_bias

        # ─── Zone proximity (OB or FVG nearby) ───
        ob_dist = row.get("nearest_ob_dist_atr", np.nan)
        fvg_dist = abs(row.get("nearest_fvg_dist_atr", np.nan)) if not np.isnan(row.get("nearest_fvg_dist_atr", np.nan)) else np.nan

        zone_nearby = False
        if not np.isnan(ob_dist) and abs(ob_dist) < 2.0:
            zone_nearby = True
            if abs(ob_dist) < 0.5:
                score += 0.20 * direction
            elif abs(ob_dist) < 1.0:
                score += 0.12 * direction
            else:
                score += 0.06 * direction

        if not np.isnan(fvg_dist) and fvg_dist < 2.0:
            zone_nearby = True
            score += 0.10 * direction

        if not zone_nearby:
            return 0.0  # No POI nearby — skip

        # ─── Zone strength ───
        bull_ob_str = row.get("nearest_bull_ob_strength", 0)
        bear_ob_str = row.get("nearest_bear_ob_strength", 0)
        zone_str = bull_ob_str if direction > 0 else bear_ob_str
        if zone_str > 0.7:
            score += 0.10 * direction
        elif zone_str > 0.4:
            score += 0.05 * direction

        # ─── Zone freshness ───
        zone_age = row.get("nearest_zone_age_bars", 200)
        if zone_age < 30:
            score += 0.08 * direction

        # ─── Liquidity sweep (recent sweep = strong reversal signal) ───
        has_sweep = row.get("has_strong_sweep", 0)
        sweep_rejection = row.get("nearest_sweep_rejection_strength", 0)
        if has_sweep > 0.5:
            score += 0.12 * direction
        elif sweep_rejection > 0.3:
            score += 0.06 * direction

        # ─── Breaker block confluence ───
        breaker_dist = row.get("nearest_breaker_dist_atr", 10)
        breaker_str = row.get("nearest_breaker_strength", 0)
        if breaker_dist < 1.5 and breaker_str > 0.5:
            score += 0.08 * direction

        # ─── OTE zone (price in optimal entry) ───
        in_ote = row.get("in_ote_zone", 0)
        if in_ote > 0.5:
            score += 0.10 * direction

        # ─── Premium/Discount alignment ───
        pd_pct = row.get("premium_discount_pct", 0.5)
        if direction > 0 and pd_pct < 0.4:
            score += 0.06 * direction  # Buying in discount
        elif direction < 0 and pd_pct > 0.6:
            score += 0.06 * direction  # Selling in premium

        # ─── Volume Profile confluence ───
        poc_dist = abs(row.get("poc_dist_atr", 10))
        val_dist = abs(row.get("val_dist_atr", 10))
        vah_dist = abs(row.get("vah_dist_atr", 10))
        if poc_dist < 1.0 or val_dist < 0.5 or vah_dist < 0.5:
            score += 0.06 * direction

        # ─── Trend alignment ───
        ema_cross = row.get("ema_cross", np.nan)
        if not np.isnan(ema_cross):
            if (direction > 0 and ema_cross > 0.5) or (direction < 0 and ema_cross < 0.5):
                score += 0.08 * direction

        # ─── RSI filter ───
        rsi = row.get("rsi_14", np.nan)
        if not np.isnan(rsi):
            if (direction > 0 and rsi > 75) or (direction < 0 and rsi < 25):
                score *= 0.5  # Overextended — cut signal

        # ─── Killzone bonus ───
        is_kz = row.get("is_killzone", 0)
        if is_kz > 0.5:
            score *= 1.15  # 15% boost during killzone sessions

        # ─── CHoCH confirmation bonus ───
        if choch > 0.5:
            score *= 1.10

        return float(np.clip(score, -1.0, 1.0))

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Use XGBoost model if trained, otherwise fall back to rule-based."""
        if self._model is not None:
            try:
                feature_cols = self._feature_names or [
                    c for c in features.columns
                    if c not in ("time", "open", "high", "low", "close", "volume", "tick_volume")
                ]
                X = features[feature_cols].values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return self._model.predict_proba(X)[:, 1]  # P(up)
            except Exception as e:
                logger.debug(f"ML prediction failed, falling back to rules: {e}")

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

        # Conservative hyperparams for ~86 features — strong regularization
        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.6,     # Each tree sees ~52 of 86 features
            reg_alpha=0.3,            # L1 regularization
            reg_lambda=2.0,           # L2 regularization
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
        )[:15]
        logger.info(
            f"SMC Confluence ML trained on {len(X_train)} samples "
            f"({y_train.sum()} positive). "
            f"Top features: {', '.join(f'{n}={v:.3f}' for n, v in top_features)}"
        )

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

        logger.info(f"Saved SMC Confluence ML model to {model_path}")

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

            logger.info(f"Loaded SMC Confluence ML model from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SMC Confluence ML model: {e}")
            self._model = None
            return False

    @property
    def default_risk(self):
        """Optimized risk parameters from XAUUSD H4 grid search (192 combos).

        Best: SL=2.5, TP=6.25 (2.5:1 R:R), thresh=0.60, hold=20.
        Pattern: wider SL + shorter hold + lower threshold = higher Sharpe.
        """
        return RiskDefaults(
            sl_atr_mult=2.5,
            tp_atr_mult=6.25,
            signal_threshold=0.60,
            max_hold_bars=20,
        )
