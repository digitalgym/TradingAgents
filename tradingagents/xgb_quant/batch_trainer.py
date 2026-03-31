"""
Batch Trainer — trains all XGBoost models across pairs × timeframes × strategies.

Designed to run weekly (or on-demand) to ensure all models are fresh when
the scanner routes pairs to XGBoost pipelines.

Usage:
    # From API / automation:
    trainer = BatchTrainer()
    result = trainer.run(symbols=DEFAULT_WATCHLIST, timeframes=["D1", "H4"])

    # Or run only stale models:
    result = trainer.run(skip_fresh_days=7)
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from tradingagents.xgb_quant.config import (
    DEFAULT_WATCHLIST,
    MODELS_DIR,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# All strategies to train
STRATEGY_NAMES = [
    "trend_following",
    "mean_reversion",
    "breakout",
    "smc_zones",
    "volume_profile_strat",
    "donchian_breakout",
    "flag_continuation",
]

# Default timeframes to train across
DEFAULT_TIMEFRAMES = ["D1", "H4"]


@dataclass
class TrainJobResult:
    """Result of a batch training run."""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0
    total_tasks: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    best_per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    blacklist: List[Dict[str, str]] = field(default_factory=list)  # Bad combos to avoid


class BatchTrainer:
    """Trains all XGBoost models for the full watchlist."""

    def __init__(self):
        self._progress_callback = None
        self._cancelled = False

    def on_progress(self, callback):
        """Register a progress callback: callback(current, total, message)."""
        self._progress_callback = callback

    def cancel(self):
        self._cancelled = True

    def _report(self, current: int, total: int, msg: str):
        logger.info(f"[{current}/{total}] {msg}")
        if self._progress_callback:
            self._progress_callback(current, total, msg)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        bars: int = 2000,
        skip_fresh_days: int = 0,
    ) -> TrainJobResult:
        """
        Train models for all symbol × timeframe × strategy combinations.

        Args:
            symbols: Pairs to train. Default: full watchlist (17 pairs).
            timeframes: TFs to train. Default: D1, H4.
            strategies: Strategy names. Default: all 5.
            bars: History bars to fetch per symbol/TF.
            skip_fresh_days: Skip models trained within N days (0 = retrain all).
        """
        from tradingagents.xgb_quant.trainer import WalkForwardTrainer

        symbols = symbols or DEFAULT_WATCHLIST
        timeframes = timeframes or DEFAULT_TIMEFRAMES
        strategies = strategies or STRATEGY_NAMES

        strategy_instances = self._load_strategies(strategies)

        # Build task list
        tasks = []
        for symbol in symbols:
            for tf in timeframes:
                for name in strategies:
                    if name not in strategy_instances:
                        continue
                    tasks.append((symbol, tf, name))

        result = TrainJobResult(
            started_at=datetime.utcnow().isoformat(),
            total_tasks=len(tasks),
        )

        start = time.time()
        trainer = WalkForwardTrainer()

        for i, (symbol, tf, strat_name) in enumerate(tasks):
            if self._cancelled:
                logger.info("Batch training cancelled.")
                break

            # Check if model is fresh enough to skip
            if skip_fresh_days > 0 and self._is_fresh(strat_name, symbol, tf, skip_fresh_days):
                result.skipped += 1
                result.results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "strategy": strat_name,
                    "status": "skipped",
                    "reason": f"Model < {skip_fresh_days} days old",
                })
                self._report(i + 1, len(tasks), f"{symbol} {tf} {strat_name}: skipped (fresh)")
                continue

            self._report(i + 1, len(tasks), f"{symbol} {tf} {strat_name}: training...")

            try:
                df = self._load_data(symbol, tf, bars)
                if df is None or len(df) < 200:
                    result.failed += 1
                    result.results.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "strategy": strat_name,
                        "status": "failed",
                        "error": f"Insufficient data ({len(df) if df is not None else 0} bars)",
                    })
                    continue

                strategy = strategy_instances[strat_name]()
                bt = trainer.train_and_evaluate(
                    strategy=strategy,
                    df=df,
                    symbol=symbol,
                    timeframe=tf,
                )

                result.completed += 1
                # Mark viability: negative Sharpe or <40% WR or PF<0.8 = bad combo
                viable = (
                    bt.sharpe > 0
                    and bt.win_rate >= 40
                    and bt.profit_factor >= 0.8
                    and bt.total_trades >= 5
                )
                entry = {
                    "symbol": symbol,
                    "timeframe": tf,
                    "strategy": strat_name,
                    "status": "success",
                    "viable": viable,
                    "win_rate": bt.win_rate,
                    "sharpe": bt.sharpe,
                    "profit_factor": bt.profit_factor,
                    "total_trades": bt.total_trades,
                    "max_drawdown_pct": bt.max_drawdown_pct,
                }
                result.results.append(entry)

                self._report(
                    i + 1, len(tasks),
                    f"{symbol} {tf} {strat_name}: WR={bt.win_rate:.1%} "
                    f"Sharpe={bt.sharpe:.2f} PF={bt.profit_factor:.2f} ({bt.total_trades} trades)"
                )

            except Exception as e:
                result.failed += 1
                result.results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "strategy": strat_name,
                    "status": "failed",
                    "error": str(e),
                })
                logger.warning(f"{symbol} {tf} {strat_name}: FAILED — {e}")

        result.finished_at = datetime.utcnow().isoformat()
        result.duration_seconds = round(time.time() - start, 1)

        # Compute best strategy per symbol (for strategy selector)
        result.best_per_symbol = self._compute_best_per_symbol(result.results)

        # Build blacklist: bad combos the scanner should avoid
        result.blacklist = self._build_blacklist(result.results)

        logger.info(
            f"Batch training complete: {result.completed} trained, "
            f"{result.skipped} skipped, {result.failed} failed, "
            f"{len(result.blacklist)} blacklisted "
            f"in {result.duration_seconds:.0f}s"
        )

        # Save summary + blacklist
        self._save_summary(result)
        self._save_blacklist(result.blacklist)

        return result

    @staticmethod
    def _load_strategies(names: List[str]) -> Dict[str, type]:
        """Lazy-load strategy classes by name."""
        from tradingagents.xgb_quant.strategies.trend_following import TrendFollowingStrategy
        from tradingagents.xgb_quant.strategies.mean_reversion import MeanReversionStrategy
        from tradingagents.xgb_quant.strategies.breakout import BreakoutStrategy
        from tradingagents.xgb_quant.strategies.smc_zones import SMCZonesStrategy
        from tradingagents.xgb_quant.strategies.volume_profile_strat import VolumeProfileStrategy
        from tradingagents.xgb_quant.strategies.donchian_breakout import DonchianBreakoutStrategy
        from tradingagents.xgb_quant.strategies.flag_continuation import FlagContinuationStrategy

        registry = {
            "trend_following": TrendFollowingStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "smc_zones": SMCZonesStrategy,
            "volume_profile_strat": VolumeProfileStrategy,
            "donchian_breakout": DonchianBreakoutStrategy,
            "flag_continuation": FlagContinuationStrategy,
        }
        return {k: v for k, v in registry.items() if k in names}

    @staticmethod
    def _load_data(symbol: str, timeframe: str, bars: int):
        """Load market data from MT5."""
        try:
            from tradingagents.automation.auto_tuner import load_mt5_data
            return load_mt5_data(symbol, timeframe, bars)
        except Exception as e:
            logger.warning(f"Failed to load data for {symbol} {timeframe}: {e}")
            return None

    @staticmethod
    def _is_fresh(strategy: str, symbol: str, timeframe: str, days: int) -> bool:
        """Check if a model was trained recently enough to skip."""
        model_path = MODELS_DIR / strategy / f"{symbol}_{timeframe}.json"
        if not model_path.exists():
            return False
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        return (datetime.now() - mtime) < timedelta(days=days)

    @staticmethod
    def _compute_best_per_symbol(results: List[Dict]) -> Dict[str, Dict]:
        """Find the best strategy+TF combo per symbol by Sharpe."""
        best = {}
        for r in results:
            if r["status"] != "success" or r.get("total_trades", 0) == 0:
                continue
            symbol = r["symbol"]
            sharpe = r.get("sharpe", 0)
            if symbol not in best or sharpe > best[symbol].get("sharpe", 0):
                best[symbol] = {
                    "strategy": r["strategy"],
                    "timeframe": r["timeframe"],
                    "sharpe": sharpe,
                    "win_rate": r.get("win_rate", 0),
                    "profit_factor": r.get("profit_factor", 0),
                }
        return best

    @staticmethod
    def _build_blacklist(results: List[Dict]) -> List[Dict[str, str]]:
        """Build list of bad symbol+strategy combos from backtest results."""
        blacklist = []
        for r in results:
            if r["status"] != "success":
                continue
            if not r.get("viable", True):
                blacklist.append({
                    "symbol": r["symbol"],
                    "strategy": r["strategy"],
                    "timeframe": r["timeframe"],
                    "reason": (
                        f"WR={r.get('win_rate', 0):.0f}% "
                        f"Sharpe={r.get('sharpe', 0):.2f} "
                        f"PF={r.get('profit_factor', 0):.2f}"
                    ),
                })
        return blacklist

    @staticmethod
    def _save_blacklist(blacklist: List[Dict[str, str]]):
        """Save blacklist so scanner can load it at runtime."""
        blacklist_file = RESULTS_DIR / "blacklist.json"
        blacklist_file.write_text(json.dumps(blacklist, indent=2))
        logger.info(f"Blacklist saved: {len(blacklist)} bad combos -> {blacklist_file}")

    @staticmethod
    def load_blacklist() -> List[Dict[str, str]]:
        """Load the blacklist from disk (called by scanner at runtime)."""
        blacklist_file = RESULTS_DIR / "blacklist.json"
        if not blacklist_file.exists():
            return []
        try:
            return json.loads(blacklist_file.read_text())
        except Exception:
            return []

    @staticmethod
    def _save_summary(result: TrainJobResult):
        """Save training summary to results dir."""
        summary_dir = RESULTS_DIR / "_batch"
        summary_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"batch_{ts}.json"
        summary_file.write_text(json.dumps(asdict(result), indent=2, default=str))
        logger.info(f"Batch summary saved to {summary_file}")
