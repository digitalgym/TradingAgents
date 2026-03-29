"""
Walk-Forward Trainer for XGBoost Strategies.

Trains and evaluates strategies using walk-forward validation:
- No look-ahead bias (train only on past data)
- Rolling train/test windows
- Uses existing _simulate_exit() for realistic trade outcome simulation
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from tradingagents.xgb_quant.config import (
    TrainingConfig, RiskDefaults, RESULTS_DIR,
)
from tradingagents.xgb_quant.strategies.base import BaseStrategy

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

    def to_dict(self) -> dict:
        return asdict(self)


class WalkForwardTrainer:
    """Train and evaluate XGBoost strategies with walk-forward validation."""

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
        Walk-forward train and evaluate a strategy.

        Args:
            strategy: Strategy instance with feature set and params
            df: Full OHLCV DataFrame
            symbol: Symbol name (for logging/saving)
            timeframe: Timeframe string
            use_trade_labels: If True, use trade-outcome labels instead of direction

        Returns:
            BacktestResult with performance metrics
        """
        feature_set = strategy.get_feature_set()
        cfg = self.config

        logger.info(
            f"Computing features for {strategy.name} on {symbol} {timeframe} "
            f"({len(df)} bars, warmup={feature_set.warmup_bars})..."
        )
        features = feature_set.compute(df)

        # Log feature quality
        total_rows = len(features)
        valid_rows = features.notna().all(axis=1).sum()
        nan_pct_per_col = features.isna().mean()
        worst_cols = nan_pct_per_col[nan_pct_per_col > 0.5]
        logger.info(
            f"  Features: {total_rows} rows, {valid_rows} fully valid "
            f"({valid_rows / total_rows * 100:.0f}%), {len(features.columns)} columns"
        )
        if len(worst_cols) > 0:
            logger.warning(
                f"  High-NaN columns (>50%): "
                f"{', '.join(f'{c}={v:.0%}' for c, v in worst_cols.items())}"
            )

        # Create labels
        if use_trade_labels:
            labels = strategy.create_trade_labels(df)
        else:
            labels = strategy.create_labels(df)

        valid_labels = labels.notna().sum()
        logger.info(f"  Labels: {valid_labels} valid out of {len(labels)}")

        # Generate fold splits
        n = len(df)
        purge = cfg.purge_bars

        if cfg.n_splits > 0:
            # Purged K-fold: splits data into n_splits sequential folds,
            # with a purge gap between train and test to prevent leakage
            # from labels that look ahead (e.g. next-bar direction).
            folds = self._purged_kfold_splits(n, cfg.n_splits, purge)
            fold_mode = f"purged {cfg.n_splits}-fold"
        else:
            # Classic walk-forward with purge gap
            folds = []
            for fold_start in range(0, n - cfg.train_window - cfg.test_window, cfg.test_window):
                train_end = fold_start + cfg.train_window
                test_start = train_end + purge
                test_end = min(test_start + cfg.test_window, n - 1)
                if test_start >= test_end:
                    continue
                folds.append((fold_start, train_end, test_start, test_end))
            fold_mode = "walk-forward"

        logger.info(
            f"  {fold_mode}: {len(folds)} folds "
            f"(train={cfg.train_window}, test={cfg.test_window}, "
            f"purge={purge}, min_train={cfg.min_train_bars})"
        )

        all_trades = []
        fold_count = 0
        skipped_folds = 0
        last_trained_model = None

        for train_start, train_end, test_start, test_end in folds:
            X_train = features.iloc[train_start:train_end].copy()
            y_train = labels.iloc[train_start:train_end].copy()
            X_test = features.iloc[test_start:test_end].copy()
            y_test = labels.iloc[test_start:test_end].copy()

            # Drop NaN rows — but use XGBoost's native NaN handling for features
            # Only drop rows where ALL feature columns are NaN (completely empty)
            # or where the label is NaN
            train_label_valid = y_train.notna()
            train_has_data = features.iloc[train_start:train_end].notna().any(axis=1)
            train_mask = train_label_valid & train_has_data
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]

            test_label_valid = y_test.notna()
            test_has_data = features.iloc[test_start:test_end].notna().any(axis=1)
            test_mask = test_label_valid & test_has_data
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

            if len(X_train) < cfg.min_train_bars or len(X_test) < 10:
                logger.debug(
                    f"  Fold {fold_count + skipped_folds}: skipped — "
                    f"train={len(X_train)} (need {cfg.min_train_bars}), "
                    f"test={len(X_test)} (need 10)"
                )
                skipped_folds += 1
                continue

            # Train XGBoost
            model = xgb.XGBClassifier(**strategy.xgb_params)

            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"  Fold {fold_count}: training failed — {e}")
                skipped_folds += 1
                continue

            last_trained_model = model

            # Predict on test set
            probs = model.predict_proba(X_test)[:, 1]

            # Simulate trades from predictions
            fold_trades = self._simulate_trades_from_predictions(
                probs, X_test.index, df, strategy,
            )
            all_trades.extend(fold_trades)
            fold_count += 1

            logger.info(
                f"  Fold {fold_count}: train={len(X_train)}, test={len(X_test)}, "
                f"trades={len(fold_trades)}"
            )

        logger.info(
            f"  Training complete: {fold_count} folds used, {skipped_folds} skipped, "
            f"{len(all_trades)} total trades"
        )

        # Save the last trained model
        if last_trained_model is not None:
            strategy.model = last_trained_model
            strategy.save_model(symbol, timeframe)
            logger.info(f"  Model saved for {strategy.name} {symbol} {timeframe}")
        else:
            logger.warning(
                f"  No model trained for {strategy.name} — "
                f"all {skipped_folds} folds skipped (not enough valid data)"
            )

        # Compute result
        result = self._compute_result(all_trades, strategy.name, symbol, timeframe, strategy.xgb_params)

        # Save result
        self._save_result(result)

        logger.info(
            f"{strategy.name} on {symbol} {timeframe}: "
            f"{result.total_trades} trades, {result.win_rate:.1f}% WR, "
            f"PF={result.profit_factor:.2f}, Sharpe={result.sharpe:.2f}"
        )

        return result

    @staticmethod
    def _purged_kfold_splits(
        n: int, n_splits: int, purge: int,
    ) -> list:
        """
        Generate purged K-fold splits for time series data.

        Each split uses all data before the test fold as training,
        with a purge gap between train end and test start to prevent
        label leakage (e.g. when labels look 1+ bars ahead).

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        fold_size = n // n_splits
        splits = []

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min(test_start + fold_size, n)

            # Train on everything BEFORE the test fold (with purge gap)
            train_end = max(0, test_start - purge)

            if train_end < 100:
                # Not enough training data for early folds — skip
                continue

            splits.append((0, train_end, test_start, test_end))

        return splits

    def _simulate_trades_from_predictions(
        self,
        probs: np.ndarray,
        bar_indices: pd.Index,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> List[Dict[str, Any]]:
        """Convert model predictions into simulated trades."""
        from tradingagents.automation.auto_tuner import _simulate_exit, _compute_atr

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr = _compute_atr(high, low, close)

        trades = []
        threshold = strategy.risk.signal_threshold
        max_hold = strategy.risk.max_hold_bars

        for prob, bar_idx in zip(probs, bar_indices):
            i = df.index.get_loc(bar_idx)

            if np.isnan(atr[i]) or atr[i] < 1e-10:
                continue

            entry = close[i]

            if prob >= threshold:
                direction = "BUY"
                sl = entry - atr[i] * strategy.risk.sl_atr_mult
                tp = entry + atr[i] * strategy.risk.tp_atr_mult
            elif prob <= (1.0 - threshold):
                direction = "SELL"
                sl = entry + atr[i] * strategy.risk.sl_atr_mult
                tp = entry - atr[i] * strategy.risk.tp_atr_mult
            else:
                continue  # No signal

            # Simulate exit (with trailing stop if configured)
            result = _simulate_exit(
                direction, entry, sl, tp, high, low, close, i, max_hold,
                trailing_atr_mult=strategy.risk.trailing_atr_mult,
                atr=atr,
            )

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
