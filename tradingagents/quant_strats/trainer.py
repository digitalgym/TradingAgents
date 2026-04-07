"""
Walk-Forward Backtester for Rule-Based Strategies.

Evaluates strategies using walk-forward validation:
- No look-ahead bias (features computed only from past data)
- Rolling test windows for out-of-sample evaluation
- Uses existing _simulate_exit() for realistic trade outcome simulation
- Regime-filtered training: strategies can restrict training to specific regimes
- Regime-gated backtesting: trades only count when regime gate passes
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


def _compute_regime_mask(
    df: pd.DataFrame,
    regime_type: str = "ranging",
    lookback: int = 25,
) -> np.ndarray:
    """
    Pre-compute a boolean regime mask for every bar in the DataFrame.

    For each bar i (where i >= lookback), runs analyze_range() on bars [i-lookback:i+1]
    and stores whether the market is in the requested regime.

    Args:
        df: OHLCV DataFrame
        regime_type: "ranging" (for mean reversion) or "trending" (for trend/breakout)
        lookback: Window size for regime detection

    Returns:
        Boolean array of length len(df). True = bar is in the requested regime.
    """
    from tradingagents.agents.analysts.range_quant import analyze_range

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n = len(close)

    mask = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        analysis = analyze_range(
            high[:i + 1],
            low[:i + 1],
            close[:i + 1],
            lookback=lookback,
        )

        if regime_type == "ranging":
            mask[i] = analysis.get("is_ranging", False)
        elif regime_type == "trending":
            mask[i] = not analysis.get("is_ranging", True)

    ranging_pct = mask.sum() / max(n, 1) * 100
    logger.info(
        f"  Regime mask ({regime_type}): {mask.sum()}/{n} bars "
        f"({ranging_pct:.1f}%) pass filter"
    )

    return mask


# Map strategy names to their required regime
STRATEGY_REGIME_FILTER: Dict[str, str] = {
    "mean_reversion": "ranging",
    # Add more as needed:
    # "trend_following": "trending",
    # "donchian_breakout": "trending",
}


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
    # Regime filtering info
    regime_filtered: bool = False       # Was regime filtering applied?
    regime_type: Optional[str] = None   # "ranging", "trending", or None
    regime_bars_pct: float = 0.0        # % of bars that passed the regime filter

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
        regime_filter: bool = True,
        precomputed_features: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Walk-forward evaluate a rule-based strategy.

        No training step — strategies use fixed rules.
        Walk-forward structure is kept to measure stability across time windows.

        Args:
            strategy: Strategy instance with feature set and params
            df: Full OHLCV DataFrame
            symbol: Symbol name (for logging/saving)
            timeframe: Timeframe string
            use_trade_labels: If True, use trade-outcome labels instead of direction
            regime_filter: If True, apply regime filtering to training and backtesting.
                           Mean reversion only trains/trades on ranging bars.
                           Set False to see unfiltered performance.
            precomputed_features: Pre-computed feature DataFrame to skip expensive
                                  recomputation (e.g., during parameter grid search).

        Returns:
            BacktestResult with performance metrics
        """
        feature_set = strategy.get_feature_set()
        cfg = self.config

        if precomputed_features is not None:
            features = precomputed_features
            logger.info(
                f"Using precomputed features for {strategy.name} on {symbol} {timeframe} "
                f"({len(features)} rows, {len(features.columns)} columns)"
            )
        else:
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

        # Regime mask — filter training and backtest bars to the correct regime
        regime_mask = None
        required_regime = STRATEGY_REGIME_FILTER.get(strategy.name)
        if regime_filter and required_regime:
            logger.info(
                f"  Computing regime mask for {strategy.name} "
                f"(required: {required_regime})..."
            )
            regime_mask = _compute_regime_mask(df, regime_type=required_regime)

        # Walk-forward folds (for stability measurement, not training)
        all_trades = []
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

        fold_count = 0
        fold_win_rates: List[float] = []

        for train_start, train_end, test_start, test_end in folds:
            # Train ML model on training fold if strategy supports it
            if hasattr(strategy, 'train_model') and callable(strategy.train_model):
                try:
                    # Generate trade labels from training data
                    train_labels = self._generate_trade_labels(
                        features, train_start, train_end, df, strategy,
                    )
                    if train_labels is not None and np.nansum(train_labels >= 0) > 20:
                        strategy.train_model(
                            features.iloc[train_start:train_end],
                            train_labels,
                        )
                except Exception as e:
                    logger.debug(f"ML training failed for fold, using rule-based: {e}")

            # Score and simulate trades for this fold
            fold_trades = self._simulate_trades_from_rules(
                features, test_start, test_end, df, strategy,
                regime_mask=regime_mask,
            )
            all_trades.extend(fold_trades)

            # Fold-level win rate
            if fold_trades:
                fold_wr = sum(1 for t in fold_trades if t["pnl_pct"] > 0) / len(fold_trades)
                fold_win_rates.append(fold_wr)

            fold_count += 1
            logger.info(
                f"  Fold {fold_count}: bars={test_start}-{test_end}, "
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

        # Tag with regime info
        if regime_mask is not None:
            result.regime_filtered = True
            result.regime_type = required_regime
            result.regime_bars_pct = float(regime_mask.sum() / max(len(regime_mask), 1) * 100)

        # Save result
        self._save_result(result)

        regime_tag = f" [regime={required_regime}, {result.regime_bars_pct:.0f}% bars]" if result.regime_filtered else ""
        logger.info(
            f"{strategy.name} on {symbol} {timeframe}{regime_tag}: "
            f"{result.total_trades} trades, {result.win_rate:.1f}% WR, "
            f"PF={result.profit_factor:.2f}, Sharpe={result.sharpe:.2f}, "
            f"fold std={result.fold_win_rate_std:.1%}"
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

    def _generate_trade_labels(
        self,
        features: pd.DataFrame,
        start: int,
        end: int,
        df: pd.DataFrame,
        strategy: 'BaseStrategy',
    ) -> Optional[np.ndarray]:
        """
        Generate binary trade labels for ML training.

        Simulates trades on training data and labels each bar:
        - 1.0 if a trade from that bar would have been profitable
        - 0.0 if it would have lost
        - NaN if no trade signal
        """
        from tradingagents.quant_strats.backtest_utils import simulate_exit as _simulate_exit, compute_atr as _compute_atr

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        n = end - start
        labels = np.full(n, np.nan)

        atr_arr = _compute_atr(high, low, close)
        risk = strategy.risk

        for i in range(start, end):
            idx_in_labels = i - start
            atr = atr_arr[i] if i < len(atr_arr) and not np.isnan(atr_arr[i]) else 0
            if atr <= 0:
                continue

            signal = strategy.predict_signal(features.iloc[:i + 1], atr, close[i])
            if signal.direction == "HOLD":
                continue

            entry = close[i]
            sl = signal.stop_loss
            tp = signal.take_profit

            exit_info = _simulate_exit(
                signal.direction, entry, sl, tp,
                high, low, close, i + 1, risk.max_hold_bars,
            )

            if exit_info and exit_info.get("pnl_pct", 0) > 0:
                labels[idx_in_labels] = 1.0
            elif exit_info:
                labels[idx_in_labels] = 0.0

        return labels

    def _simulate_trades_from_rules(
        self,
        features: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        regime_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each bar with rules and simulate trades.

        If regime_mask is provided, trades are only taken on bars where
        the mask is True — matching what the live regime gate does.
        """
        from tradingagents.quant_strats.backtest_utils import simulate_exit as _simulate_exit, compute_atr as _compute_atr

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr = _compute_atr(high, low, close)

        trades = []
        gated_count = 0
        threshold = strategy.risk.signal_threshold
        max_hold = strategy.risk.max_hold_bars

        for i in range(start_idx, end_idx):
            if np.isnan(atr[i]) or atr[i] < 1e-10:
                continue

            # Regime gate: skip bars where regime doesn't match
            if regime_mask is not None and not regime_mask[i]:
                gated_count += 1
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

        if gated_count > 0:
            logger.info(f"    Regime gate blocked {gated_count} trades in this fold")

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
