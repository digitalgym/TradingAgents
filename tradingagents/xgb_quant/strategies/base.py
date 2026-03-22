"""
Base Strategy — abstract class for all XGBoost strategies.

Each strategy defines its feature set, default XGBoost parameters,
and risk parameters. The trainer handles training and evaluation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from tradingagents.xgb_quant.config import (
    STRATEGY_XGB_DEFAULTS, RiskDefaults, FeatureWindows, MODELS_DIR,
)
from tradingagents.xgb_quant.features.base import BaseFeatureSet

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal output from a strategy."""
    direction: str         # "BUY", "SELL", or "HOLD"
    confidence: float      # 0.0 - 1.0 (probability from XGBoost)
    strategies_agreed: List[str] = field(default_factory=list)
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    rationale: str = ""


class BaseStrategy(ABC):
    """Abstract base class for XGBoost trading strategies."""

    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        risk: Optional[RiskDefaults] = None,
        windows: Optional[FeatureWindows] = None,
    ):
        self.risk = risk or RiskDefaults()
        self.windows = windows or FeatureWindows()
        self.model = None

        # Merge provided params with strategy defaults
        defaults = STRATEGY_XGB_DEFAULTS.get(self.name, {})
        self.xgb_params = {**defaults, **(xgb_params or {})}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name (used for file paths and config keys)."""
        ...

    @abstractmethod
    def get_feature_set(self) -> BaseFeatureSet:
        """Return the feature set this strategy uses."""
        ...

    def model_path(self, symbol: str, timeframe: str) -> Path:
        """Path to saved model file."""
        d = MODELS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}_{timeframe}.json"

    def save_model(self, symbol: str, timeframe: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save — train first")
        path = self.model_path(symbol, timeframe)
        self.model.save_model(str(path))
        logger.info(f"Saved {self.name} model to {path}")

    def load_model(self, symbol: str, timeframe: str) -> bool:
        """Load model from disk. Returns True if successful."""
        import xgboost as xgb
        path = self.model_path(symbol, timeframe)
        if not path.exists():
            logger.warning(f"No saved model at {path}")
            return False
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        logger.info(f"Loaded {self.name} model from {path}")
        return True

    def predict_proba(self, features: pd.DataFrame) -> float:
        """Predict probability of UP move for the latest bar."""
        if self.model is None:
            raise ValueError("No model loaded — train or load first")

        # Get last row, drop NaN columns (XGBoost handles missing but be safe)
        X = features.iloc[[-1]].copy()
        proba = self.model.predict_proba(X)[:, 1][0]
        return float(proba)

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of UP move for all bars. Returns array with NaN for invalid rows."""
        if self.model is None:
            raise ValueError("No model loaded — train or load first")

        n = len(features)
        probs = np.full(n, np.nan)

        # Find rows that have at least some data
        valid_mask = features.notna().any(axis=1)
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
            return probs

        X = features.iloc[valid_idx].copy()
        try:
            batch_probs = self.model.predict_proba(X)[:, 1]
            probs[valid_idx] = batch_probs
        except Exception:
            pass

        return probs

    def predict_signal(self, features: pd.DataFrame, atr: float,
                       current_price: float) -> Signal:
        """Generate a trading signal from model prediction."""
        prob_up = self.predict_proba(features)

        if prob_up >= self.risk.signal_threshold:
            direction = "BUY"
            confidence = prob_up
            sl = current_price - atr * self.risk.sl_atr_mult
            tp = current_price + atr * self.risk.tp_atr_mult
        elif prob_up <= (1.0 - self.risk.signal_threshold):
            direction = "SELL"
            confidence = 1.0 - prob_up
            sl = current_price + atr * self.risk.sl_atr_mult
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
            rationale=f"{self.name}: P(up)={prob_up:.3f}",
        )

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels: 1 if next bar closes higher, 0 otherwise.

        Shifted by -1 (look ahead 1 bar) — only valid during training.
        The last row will be NaN and should be excluded.
        """
        return (df["close"].shift(-1) > df["close"]).astype(float)

    def create_trade_labels(
        self, df: pd.DataFrame,
        sl_atr_mult: Optional[float] = None,
        tp_atr_mult: Optional[float] = None,
        max_hold: Optional[int] = None,
    ) -> pd.Series:
        """
        Create trade-outcome labels using walk-forward exit simulation.

        For each bar, simulates a BUY trade with ATR-based SL/TP.
        1 = TP hit (win), 0 = SL hit or timeout (loss).

        More realistic than simple next-bar direction.
        """
        from tradingagents.automation.auto_tuner import _simulate_exit, _compute_atr

        sl_mult = sl_atr_mult or self.risk.sl_atr_mult
        tp_mult = tp_atr_mult or self.risk.tp_atr_mult
        hold = max_hold or self.risk.max_hold_bars

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr = _compute_atr(high, low, close)

        labels = pd.Series(np.nan, index=df.index, dtype=float)

        for i in range(len(df) - hold - 1):
            if np.isnan(atr[i]):
                continue

            entry = close[i]
            sl = entry - atr[i] * sl_mult
            tp = entry + atr[i] * tp_mult

            result = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold)
            labels.iloc[i] = 1.0 if result["reason"] == "tp" else 0.0

        return labels
