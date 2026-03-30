"""
Ensemble / Confluence Voting System.

Combines predictions from multiple XGBoost strategies using
majority vote with minimum probability threshold.
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from tradingagents.quant_strats.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """Combines predictions from multiple strategies via voting."""

    def __init__(
        self,
        strategies: List[BaseStrategy],
        min_agree: int = 2,
        min_prob: float = 0.60,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            strategies: List of trained strategy instances
            min_agree: Minimum strategies that must agree for a signal
            min_prob: Minimum probability threshold per strategy
            weights: Optional per-strategy weights (by name). If None, equal weight.
        """
        self.strategies = strategies
        self.min_agree = min_agree
        self.min_prob = min_prob
        self.weights = weights or {s.name: 1.0 for s in strategies}

    def predict(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
    ) -> Signal:
        """
        Generate ensemble signal from all strategies.

        1. Each strategy predicts P(up) for latest bar
        2. Count votes: UP if prob >= min_prob, DOWN if prob <= (1 - min_prob)
        3. If >= min_agree strategies agree → fire signal
        4. Otherwise → HOLD

        Returns:
            Signal with direction, confidence, and contributing strategies
        """
        votes_up = []
        votes_down = []

        for strategy in self.strategies:
            try:
                feature_set = strategy.get_feature_set()
                features = feature_set.compute(df)
                prob_up = strategy.predict_proba(features)
                weight = self.weights.get(strategy.name, 1.0)

                if prob_up >= self.min_prob:
                    votes_up.append((strategy.name, prob_up, weight))
                elif prob_up <= (1.0 - self.min_prob):
                    votes_down.append((strategy.name, 1.0 - prob_up, weight))
                else:
                    logger.debug(f"{strategy.name}: no signal (P(up)={prob_up:.3f})")

            except Exception as e:
                logger.warning(f"Strategy {strategy.name} prediction failed: {e}")
                continue

        # Evaluate votes
        if len(votes_up) >= self.min_agree:
            total_weight = sum(v[2] for v in votes_up)
            weighted_conf = sum(v[1] * v[2] for v in votes_up) / total_weight
            names = [v[0] for v in votes_up]

            sl = current_price - atr * 1.5  # Default, can be overridden
            tp = current_price + atr * 2.5

            return Signal(
                direction="BUY",
                confidence=weighted_conf,
                strategies_agreed=names,
                entry=current_price,
                stop_loss=sl,
                take_profit=tp,
                rationale=f"Ensemble BUY: {', '.join(names)} agree (conf={weighted_conf:.3f})",
            )

        if len(votes_down) >= self.min_agree:
            total_weight = sum(v[2] for v in votes_down)
            weighted_conf = sum(v[1] * v[2] for v in votes_down) / total_weight
            names = [v[0] for v in votes_down]

            sl = current_price + atr * 1.5
            tp = current_price - atr * 2.5

            return Signal(
                direction="SELL",
                confidence=weighted_conf,
                strategies_agreed=names,
                entry=current_price,
                stop_loss=sl,
                take_profit=tp,
                rationale=f"Ensemble SELL: {', '.join(names)} agree (conf={weighted_conf:.3f})",
            )

        return Signal(
            direction="HOLD",
            confidence=0.0,
            rationale=f"No consensus: {len(votes_up)} UP, {len(votes_down)} DOWN, need {self.min_agree}",
        )
