"""
Walk-Forward Backtester for Rule-Based Strategies.

Evaluates strategies using walk-forward validation:
- No look-ahead bias (features computed only from past data)
- Rolling test windows for out-of-sample evaluation
- Uses existing _simulate_exit() for realistic trade outcome simulation
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from tradingagents.quant_strats.config import (
    TrainingConfig, RiskDefaults, RESULTS_DIR,
)
from tradingagents.quant_strats.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result from walk-forward backtest of a strategy."""
    strategy: str
    symbol: str
    timeframe: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    profit_factor: float
    sharpe: float
    max_drawdown_pct: float
    avg_hold_bars: float
    params: Dict[str, Any]
    # Per-fold diagnostics
    fold_win_rates: List[float] = None
    fold_win_rate_std: float = 0.0
    n_folds: int = 0
    # Kept for backward compat (rule-based has no train/test gap)
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    train_test_gap: float = 0.0

    def __post_init__(self):
        if self.fold_win_rates is None:
            self.fold_win_rates = []

    def to_dict(self) -> dict:
        return asdict(self)


class WalkForwardTrainer:
    """Evaluate rule-based strategies with walk-forward backtesting."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        risk: Optional[RiskDefaults] = None,
    ):
        self.config = config or TrainingConfig()
        self.risk = risk or RiskDefaults()

    def train_and_evaluate(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        use_trade_labels: bool = False,
    ) -> BacktestResult:
        """
        Walk-forward evaluate a rule-based strategy.

        No training step — strategies use fixed rules.
        Walk-forward structure is kept to measure stability across time windows.
        """
        feature_set = strategy.get_feature_set()
        cfg = self.config

        logger.info(
            f"Computing features for {strategy.name} on {symbol} {timeframe} "
            f"({len(df)} bars, warmup={feature_set.warmup_bars})..."
        )
        features = feature_set.compute(df)

        total_rows = len(features)
        valid_rows = features.notna().all(axis=1).sum()
        logger.info(
            f"  Features: {total_rows} rows, {valid_rows} fully valid "
            f"({valid_rows / total_rows * 100:.0f}%), {len(features.columns)} columns"
        )

        # Walk-forward folds (for stability measurement, not training)
        all_trades = []
        n = len(df)
        fold_count = 0
        fold_win_rates: List[float] = []

        total_folds = max(1, (n - cfg.train_window - cfg.test_window) // cfg.test_window + 1)
        logger.info(
            f"  Walk-forward: {total_folds} potential folds "
            f"(window={cfg.test_window})"
        )

        for fold_start in range(cfg.train_window, n - cfg.test_window, cfg.test_window):
            fold_end = min(fold_start + cfg.test_window, n - 1)

            # Score and simulate trades for this fold
            fold_trades = self._simulate_trades_from_rules(
                features, fold_start, fold_end, df, strategy,
            )
            all_trades.extend(fold_trades)

            # Fold-level win rate
            if fold_trades:
                fold_wr = sum(1 for t in fold_trades if t["pnl_pct"] > 0) / len(fold_trades)
                fold_win_rates.append(fold_wr)

            fold_count += 1
            logger.info(
                f"  Fold {fold_count}: bars={fold_start}-{fold_end}, "
                f"trades={len(fold_trades)}"
                + (f", wr={fold_wr:.1%}" if fold_trades else "")
            )

        logger.info(
            f"  Evaluation complete: {fold_count} folds, "
            f"{len(all_trades)} total trades"
        )

        # Save params (no model to save for rule-based)
        strategy.save_model(symbol, timeframe)
        logger.info(f"  Params saved for {strategy.name} {symbol} {timeframe}")

        # Compute result
        result = self._compute_result(
            all_trades, strategy.name, symbol, timeframe,
            {"risk": asdict(strategy.risk) if hasattr(strategy.risk, '__dataclass_fields__') else {}},
        )

        # Attach fold diagnostics
        result.fold_win_rates = fold_win_rates
        result.n_folds = fold_count
        if fold_win_rates:
            result.test_win_rate = float(np.mean(fold_win_rates))
            result.fold_win_rate_std = float(np.std(fold_win_rates))

        # Stability warning
        if result.fold_win_rate_std > cfg.max_fold_wr_std:
            logger.warning(
                f"  INSTABILITY WARNING: fold WR std={result.fold_win_rate_std:.1%} "
                f"(threshold {cfg.max_fold_wr_std:.0%}). Results vary across folds."
            )
        if result.total_trades < cfg.min_trades:
            logger.warning(
                f"  LOW TRADES WARNING: {result.total_trades} trades "
                f"(minimum {cfg.min_trades})."
            )

        # Save result
        self._save_result(result)

        logger.info(
            f"{strategy.name} on {symbol} {timeframe}: "
            f"{result.total_trades} trades, {result.win_rate:.1f}% WR, "
            f"PF={result.profit_factor:.2f}, Sharpe={result.sharpe:.2f}, "
            f"fold std={result.fold_win_rate_std:.1%}"
        )

        return result

    def _simulate_trades_from_rules(
        self,
        features: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> List[Dict[str, Any]]:
        """Score each bar with rules and simulate trades."""
        from tradingagents.automation.auto_tuner import _simulate_exit, _compute_atr

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr = _compute_atr(high, low, close)

        trades = []
        threshold = strategy.risk.signal_threshold
        max_hold = strategy.risk.max_hold_bars

        for i in range(start_idx, end_idx):
            if np.isnan(atr[i]) or atr[i] < 1e-10:
                continue

            row = features.iloc[i]
            if row.isna().all():
                continue

            entry = close[i]

            # Use strategy's predict_signal() to get direction, SL, TP
            # This lets strategies like Keltner MR use custom targets (midline)
            signal = strategy.predict_signal(features.iloc[:i + 1], atr[i], entry)

            if signal.direction == "HOLD":
                continue

            direction = signal.direction
            sl = signal.stop_loss
            tp = signal.take_profit
            prob = signal.confidence

            # Simulate exit
            result = _simulate_exit(direction, entry, sl, tp, high, low, close, i, max_hold)

            trades.append({
                "bar": i,
                "direction": direction,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": result["exit"],
                "pnl_pct": result["pnl_pct"],
                "reason": result["reason"],
                "bars_held": result["bars"],
                "prob": float(prob),
            })

        return trades

    def _compute_result(
        self,
        trades: List[Dict],
        strategy_name: str,
        symbol: str,
        timeframe: str,
        params: Dict,
    ) -> BacktestResult:
        """Compute backtest statistics from trades."""
        if not trades:
            return BacktestResult(
                strategy=strategy_name, symbol=symbol, timeframe=timeframe,
                total_trades=0, wins=0, losses=0, win_rate=0.0,
                avg_pnl_pct=0.0, total_pnl_pct=0.0, profit_factor=0.0,
                sharpe=0.0, max_drawdown_pct=0.0, avg_hold_bars=0.0,
                params=params,
            )

        pnls = [t["pnl_pct"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        avg_pnl = float(np.mean(pnls))
        total_pnl = float(np.sum(pnls))

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.001
        profit_factor = gross_profit / gross_loss

        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 / max(1, len(pnls))) if std_pnl > 0 else 0

        # Max drawdown
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        avg_hold = float(np.mean([t["bars_held"] for t in trades]))

        return BacktestResult(
            strategy=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            total_trades=len(trades),
            wins=len(winners),
            losses=len(losers),
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl,
            total_pnl_pct=total_pnl,
            profit_factor=profit_factor,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            avg_hold_bars=avg_hold,
            params=params,
        )

    def _save_result(self, result: BacktestResult):
        """Save backtest result to JSON."""
        symbol_dir = RESULTS_DIR / result.symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        path = symbol_dir / f"{result.strategy}_{result.timeframe}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
