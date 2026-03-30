"""
Parameter Tuner -- Optuna-based risk parameter optimisation.

Fine-tunes risk parameters (SL/TP multipliers, signal threshold,
max hold bars) per pair at deployment time. Uses walk-forward
backtest as objective.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from tradingagents.quant_strats.config import RiskDefaults, TrainingConfig
from tradingagents.quant_strats.strategies.base import BaseStrategy
from tradingagents.quant_strats.trainer import WalkForwardTrainer, BacktestResult

logger = logging.getLogger(__name__)


class TuneResult:
    """Result of hyperparameter tuning."""

    def __init__(self, best_params: Dict[str, Any], best_sharpe: float,
                 best_result: Optional[BacktestResult] = None, n_trials: int = 0):
        self.best_params = best_params
        self.best_sharpe = best_sharpe
        self.best_result = best_result
        self.n_trials = n_trials


class ParameterTuner:
    """Optuna-based risk parameter optimisation for rule-based strategies."""

    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()

    def tune(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        n_trials: int = 50,
        timeout: Optional[int] = 600,
    ) -> TuneResult:
        """
        Tune risk parameters using Optuna.

        Args:
            strategy: Strategy to tune
            df: Full OHLCV data
            symbol: Symbol name
            timeframe: Timeframe
            n_trials: Number of Optuna trials
            timeout: Max seconds for tuning (default 10 min)

        Returns:
            TuneResult with best parameters and performance
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_result_holder = [None]
        cfg = self.training_config

        def objective(trial):
            # Risk parameters only (no ML hyperparams to tune)
            risk = RiskDefaults(
                sl_atr_mult=trial.suggest_float("sl_atr_mult", 1.0, 3.0),
                tp_atr_mult=trial.suggest_float("tp_atr_mult", 1.5, 4.0),
                signal_threshold=trial.suggest_float("signal_threshold", 0.55, 0.75),
                max_hold_bars=trial.suggest_int("max_hold_bars", 10, 30, step=5),
            )

            strategy.risk = risk

            trainer = WalkForwardTrainer(config=cfg, risk=risk)

            try:
                result = trainer.train_and_evaluate(
                    strategy=strategy,
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -999

            # Penalty: not enough trades
            if result.total_trades < cfg.min_trades:
                logger.debug(
                    f"Trial pruned: {result.total_trades} trades < min {cfg.min_trades}"
                )
                return -999

            # Penalty: fold instability
            penalty = 0.0
            if result.fold_win_rate_std > cfg.max_fold_wr_std:
                penalty += (result.fold_win_rate_std - cfg.max_fold_wr_std) * 3.0

            adjusted_sharpe = result.sharpe - penalty

            # Track best result
            if best_result_holder[0] is None or adjusted_sharpe > (best_result_holder[0].sharpe or -999):
                best_result_holder[0] = result

            return adjusted_sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        logger.info(
            f"Tuning complete for {strategy.name} on {symbol}: "
            f"best Sharpe={study.best_value:.3f} in {len(study.trials)} trials"
        )

        return TuneResult(
            best_params=study.best_params,
            best_sharpe=study.best_value,
            best_result=best_result_holder[0],
            n_trials=len(study.trials),
        )
