"""
Mean Reversion Strategy

Fades overextended price moves back to the mean using
Bollinger Band position, Z-score, RSI extremes, and Stochastic.
Best suited for range-bound, low-volatility pairs (EURGBP, AUDNZD, USDCHF).

IMPORTANT: This strategy has a hard regime gate — it will only fire signals
when the market is confirmed ranging (is_ranging=True, MR score >45, ADX <20).
Without this gate, the model fires ~750+ trades with ~40% WR and catastrophic
drawdowns on trending pairs.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from tradingagents.xgb_quant.strategies.base import BaseStrategy, Signal
from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.features.technical import TechnicalFeatures
from tradingagents.xgb_quant.config import FeatureWindows, RiskDefaults

logger = logging.getLogger(__name__)

# Pairs that should NEVER use mean reversion — too volatile / trending
MR_EXCLUDED_PAIRS = frozenset({"XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "GBPJPY"})


class MeanReversionFeatures(TechnicalFeatures):
    """Technical features + mean reversion specific features."""

    @property
    def feature_names(self) -> list:
        base = super().feature_names
        return base + [
            "zscore_20", "zscore_50",
            "stoch_k", "stoch_d",
            "close_vs_sma20", "close_vs_sma50",
            "bb_squeeze",  # BB width percentile (low = squeeze)
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        features = super().compute(df)
        close = df["close"].values.astype(float)
        w = self.windows

        # Z-scores
        cs = pd.Series(close)
        sma20 = cs.rolling(w.mid).mean()
        std20 = cs.rolling(w.mid).std()
        sma50 = cs.rolling(w.long).mean()
        std50 = cs.rolling(w.long).std()

        features["zscore_20"] = ((cs - sma20) / (std20 + 1e-10)).values
        features["zscore_50"] = ((cs - sma50) / (std50 + 1e-10)).values

        # Stochastic %K / %D
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        period = w.stoch_period

        lowest_low = pd.Series(low).rolling(period).min()
        highest_high = pd.Series(high).rolling(period).max()
        stoch_k = 100 * (cs - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()

        features["stoch_k"] = stoch_k.values
        features["stoch_d"] = stoch_d.values

        # Close vs SMA (normalised)
        features["close_vs_sma20"] = ((cs - sma20) / (sma20 + 1e-10)).values
        features["close_vs_sma50"] = ((cs - sma50) / (sma50 + 1e-10)).values

        # BB squeeze (rolling percentile of BB width)
        bb_width = features["bb_width"].values
        bb_series = pd.Series(bb_width)
        bb_rank = bb_series.rolling(100, min_periods=20).rank(pct=True)
        features["bb_squeeze"] = bb_rank.values

        return features


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    def get_feature_set(self) -> BaseFeatureSet:
        return MeanReversionFeatures(windows=self.windows)

    def check_regime_gate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        symbol: str = "",
    ) -> dict:
        """
        Hard regime gate — returns range analysis dict.
        Signal is blocked unless is_ranging=True.

        Uses the same analyze_range() from range_quant.py so both
        the XGBoost and LLM paths share identical range detection.
        """
        from tradingagents.agents.analysts.range_quant import analyze_range

        if symbol in MR_EXCLUDED_PAIRS:
            return {
                "is_ranging": False,
                "blocked_reason": f"{symbol} excluded from mean reversion (too volatile)",
                "mean_reversion_score": 0.0,
            }

        analysis = analyze_range(high, low, close)

        if not analysis["is_ranging"]:
            reasons = []
            if analysis.get("trend_strength", 0) >= 0.35:
                reasons.append(f"trend_strength={analysis['trend_strength']:.2f} (>=0.35)")
            if analysis.get("adx_proxy", 0) >= 20:
                reasons.append(f"adx_proxy={analysis['adx_proxy']:.1f} (>=20)")
            if analysis.get("mean_reversion_score", 0) <= 45:
                reasons.append(f"mr_score={analysis['mean_reversion_score']:.0f} (<=45)")
            analysis["blocked_reason"] = "Not ranging: " + ", ".join(reasons) if reasons else "Not ranging"

        return analysis

    def predict_signal(
        self,
        features: pd.DataFrame,
        atr: float,
        current_price: float,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        close: Optional[np.ndarray] = None,
        symbol: str = "",
    ) -> Signal:
        """
        Generate signal with regime gate.

        If high/low/close arrays are provided, the regime gate is checked
        first — signal is blocked unless market is confirmed ranging.
        Falls back to base predict_signal if no price arrays provided
        (e.g. during walk-forward training where gating isn't wanted).
        """
        if high is not None and low is not None and close is not None:
            gate = self.check_regime_gate(high, low, close, symbol)
            if not gate.get("is_ranging", False):
                reason = gate.get("blocked_reason", "Market not ranging")
                logger.info(f"MR regime gate BLOCKED for {symbol}: {reason}")
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    rationale=f"mean_reversion: regime gate blocked — {reason}",
                )

        return super().predict_signal(features, atr, current_price)
