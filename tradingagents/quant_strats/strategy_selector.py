"""
Strategy Selector — picks the best strategy for a pair.

Given a pair and market context, scores each strategy in the library
and recommends the best one (or best ensemble combination).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from tradingagents.quant_strats.config import (
    REGIME_SUITABILITY, PAIR_STRATEGY_DEFAULTS, RESULTS_DIR, TrainingConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of strategy selection for a pair."""
    symbol: str
    recommended_strategy: str
    recommended_timeframe: Optional[str] = None
    confidence: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    reasoning: str = ""
    fallback_strategy: Optional[str] = None
    ensemble_candidates: Optional[List[str]] = None


class StrategySelector:
    """Select the best strategy for a given pair and market context."""

    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self._results_cache: Dict[str, Dict] = {}
        self._min_trades = (training_config or TrainingConfig()).min_trades

    def select(
        self,
        symbol: str,
        direction: Optional[str] = None,
        regime: Optional[str] = None,
        volatility: Optional[str] = None,
    ) -> SelectionResult:
        """
        Score each strategy and pick the best for this pair.

        Args:
            symbol: Trading pair
            direction: "LONG" or "SHORT" (from scanner)
            regime: Market regime ("trending-up", "trending-down", "ranging")
            volatility: Volatility regime ("low", "normal", "high", "extreme")
        """
        candidates = []
        strategy_names = list(REGIME_SUITABILITY.keys())

        # Respect per-strategy exclusion lists
        from tradingagents.quant_strats.strategies.mean_reversion import MR_EXCLUDED_PAIRS
        excluded_map = {
            "mean_reversion": MR_EXCLUDED_PAIRS,
        }

        for strategy_name in strategy_names:
            # Skip strategies that exclude this symbol
            if symbol in excluded_map.get(strategy_name, set()):
                logger.debug(f"Skipping {strategy_name} for {symbol} (excluded pair)")
                continue

            bt = self._get_backtest_result(strategy_name, symbol)

            # 1. Backtest performance score (0-1)
            if bt and bt.get("total_trades", 0) >= self._min_trades:
                wr = bt.get("win_rate", 0) / 100.0  # stored as %
                pf = min(bt.get("profit_factor", 0) / 2.0, 1.0)
                sh = min(max(bt.get("sharpe", 0), 0) / 2.0, 1.0)
                bt_score = wr * 0.4 + pf * 0.3 + sh * 0.3
            else:
                bt_score = 0.3  # Cold start neutral

            # 2. Regime suitability (0-1)
            regime_scores = REGIME_SUITABILITY.get(strategy_name, {})
            regime_score = regime_scores.get(regime, 0.5) if regime else 0.5

            # 3. Composite
            score = bt_score * 0.55 + regime_score * 0.45

            candidates.append({
                "name": strategy_name,
                "score": score,
                "bt_score": bt_score,
                "regime_score": regime_score,
                "win_rate": bt.get("win_rate", 0) if bt else 0,
                "profit_factor": bt.get("profit_factor", 0) if bt else 0,
                "sharpe": bt.get("sharpe", 0) if bt else 0,
                "trades": bt.get("total_trades", 0) if bt else 0,
                "timeframe": bt.get("timeframe") if bt else None,
            })

        # Sort by score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        if not candidates:
            # Fall back to default
            default = PAIR_STRATEGY_DEFAULTS.get(symbol, "trend_following")
            return SelectionResult(
                symbol=symbol,
                recommended_strategy=default,
                confidence=0.3,
                win_rate=0,
                profit_factor=0,
                sharpe=0,
                reasoning=f"No data — using default ({default}) for {symbol}",
            )

        best = candidates[0]
        fallback = candidates[1] if len(candidates) > 1 else None

        # Select top 3 for ensemble (if they have decent scores)
        ensemble = [c["name"] for c in candidates[:3] if c["score"] > 0.3]

        reasoning_parts = [
            f"Best: {best['name']} (score={best['score']:.3f})",
            f"  Backtest: {best['trades']} trades, {best['win_rate']:.1f}% WR, PF={best['profit_factor']:.2f}",
            f"  Regime fit: {best['regime_score']:.2f} ({regime or 'unknown'})",
        ]
        if fallback:
            reasoning_parts.append(
                f"Fallback: {fallback['name']} (score={fallback['score']:.3f})"
            )

        return SelectionResult(
            symbol=symbol,
            recommended_strategy=best["name"],
            recommended_timeframe=best.get("timeframe"),
            confidence=best["score"],
            win_rate=best["win_rate"],
            profit_factor=best["profit_factor"],
            sharpe=best["sharpe"],
            reasoning="\n".join(reasoning_parts),
            fallback_strategy=fallback["name"] if fallback else None,
            ensemble_candidates=ensemble,
        )

    def get_performance_matrix(self) -> Dict[str, Dict[str, Dict]]:
        """
        Build the full performance matrix: symbol → strategy → metrics.

        Reads all saved backtest results from disk.
        """
        matrix = {}

        if not RESULTS_DIR.exists():
            return matrix

        for symbol_dir in RESULTS_DIR.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            matrix[symbol] = {}

            for result_file in symbol_dir.glob("*.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    strategy = data.get("strategy", result_file.stem.rsplit("_", 1)[0])
                    timeframe = data.get("timeframe", result_file.stem.rsplit("_", 1)[-1])
                    # Use strategy_timeframe as key to support multiple TFs
                    key = f"{strategy}_{timeframe}" if timeframe else strategy
                    matrix[symbol][key] = {
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "win_rate": data.get("win_rate", 0),
                        "profit_factor": data.get("profit_factor", 0),
                        "sharpe": data.get("sharpe", 0),
                        "total_trades": data.get("total_trades", 0),
                        "total_pnl_pct": data.get("total_pnl_pct", 0),
                        "max_drawdown_pct": data.get("max_drawdown_pct", 0),
                    }
                except Exception:
                    continue

        return matrix

    def _get_backtest_result(self, strategy_name: str, symbol: str) -> Optional[Dict]:
        """Load the best backtest result for a strategy/symbol across all timeframes."""
        cache_key = f"{strategy_name}_{symbol}"
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        symbol_dir = RESULTS_DIR / symbol
        if not symbol_dir.exists():
            return None

        # Load all timeframe results and pick the one with best Sharpe
        best = None
        for result_file in symbol_dir.glob(f"{strategy_name}_*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("total_trades", 0) < self._min_trades:
                    continue
                if best is None or data.get("sharpe", 0) > best.get("sharpe", 0):
                    best = data
            except Exception:
                continue

        if best:
            self._results_cache[cache_key] = best
        return best
