"""
Base Strategy — abstract class for all rule-based quant strategies.

Each strategy defines its feature set and a score_bar() method that
returns a directional score from its computed features. No ML model
is involved — signals come from explicit, interpretable rules.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from tradingagents.quant_strats.config import (
    RiskDefaults, FeatureWindows, MODELS_DIR,
)
from tradingagents.quant_strats.features.base import BaseFeatureSet

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal output from a strategy."""
    direction: str         # "BUY", "SELL", or "HOLD"
    confidence: float      # 0.0 - 1.0
    strategies_agreed: List[str] = field(default_factory=list)
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    rationale: str = ""


class BaseStrategy(ABC):
    """Abstract base class for rule-based trading strategies."""

    def __init__(
        self,
        risk: Optional[RiskDefaults] = None,
        windows: Optional[FeatureWindows] = None,
    ):
        self.risk = risk or RiskDefaults()
        self.windows = windows or FeatureWindows()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name (used for file paths and config keys)."""
        ...

    @abstractmethod
    def get_feature_set(self) -> BaseFeatureSet:
        """Return the feature set this strategy uses."""
        ...

    @abstractmethod
    def score_bar(self, features: pd.DataFrame, idx: int) -> float:
        """
        Score a single bar using rule-based logic.

        Args:
            features: Full feature DataFrame (use features.iloc[idx] for current bar)
            idx: Positional index of the bar to score

        Returns:
            Score in range [-1.0, +1.0]:
              Positive = bullish (higher = stronger)
              Negative = bearish (lower = stronger)
              Near zero = no signal
        """
        ...

    # ------------------------------------------------------------------
    # Prediction interface (same API as before, now rule-based)
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> float:
        """
        Predict probability of UP move for the latest bar.

        Converts score_bar() output to a probability-like value in [0, 1].
        """
        score = self.score_bar(features, len(features) - 1)
        # Map [-1, +1] → [0, 1]
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

    def predict_proba_batch(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability of UP move for all bars."""
        n = len(features)
        probs = np.full(n, np.nan)

        for i in range(n):
            row = features.iloc[i]
            if row.isna().all():
                continue
            score = self.score_bar(features, i)
            probs[i] = np.clip((score + 1.0) / 2.0, 0.0, 1.0)

        return probs

    def predict_signal(self, features: pd.DataFrame, atr: float,
                       current_price: float) -> Signal:
        """Generate a trading signal from rule-based score."""
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

    # ------------------------------------------------------------------
    # Model persistence (no-op for rule-based, kept for interface compat)
    # ------------------------------------------------------------------

    def model_path(self, symbol: str, timeframe: str) -> Path:
        """Path to saved model/params file."""
        d = MODELS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}_{timeframe}.json"

    def save_model(self, symbol: str, timeframe: str):
        """Save strategy params to disk (rule-based strategies have no model)."""
        path = self.model_path(symbol, timeframe)
        params = {
            "strategy": self.name,
            "risk": {
                "sl_atr_mult": self.risk.sl_atr_mult,
                "tp_atr_mult": self.risk.tp_atr_mult,
                "signal_threshold": self.risk.signal_threshold,
                "max_hold_bars": self.risk.max_hold_bars,
            },
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved {self.name} params to {path}")

    def load_model(self, symbol: str, timeframe: str) -> bool:
        """Load strategy params from disk. Returns True if successful."""
        path = self.model_path(symbol, timeframe)
        if not path.exists():
            logger.warning(f"No saved params at {path}")
            return False
        try:
            with open(path) as f:
                params = json.load(f)
            if "risk" in params:
                r = params["risk"]
                self.risk = RiskDefaults(
                    sl_atr_mult=r.get("sl_atr_mult", self.risk.sl_atr_mult),
                    tp_atr_mult=r.get("tp_atr_mult", self.risk.tp_atr_mult),
                    signal_threshold=r.get("signal_threshold", self.risk.signal_threshold),
                    max_hold_bars=r.get("max_hold_bars", self.risk.max_hold_bars),
                )
            logger.info(f"Loaded {self.name} params from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load params from {path}: {e}")
            return False

    # ------------------------------------------------------------------
    # Labels (unchanged — used by trainer for backtesting)
    # ------------------------------------------------------------------

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels: 1 if next bar closes higher, 0 otherwise.
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
