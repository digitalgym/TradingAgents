"""
Parameter Tuner — Optuna-based hyperparameter optimisation.

Fine-tunes XGBoost hyperparameters and risk parameters per pair
at deployment time. Uses walk-forward backtest as objective.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from tradingagents.xgb_quant.config import RiskDefaults, TrainingConfig
from tradingagents.xgb_quant.strategies.base import BaseStrategy
from tradingagents.xgb_quant.trainer import WalkForwardTrainer, BacktestResult

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
    """Optuna-based hyperparameter optimisation for XGBoost strategies."""

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
        Tune strategy hyperparameters using Optuna.

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

        def objective(trial):
            # XGBoost hyperparameters
            xgb_params = {
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }

            # Risk parameters
            risk = RiskDefaults(
                sl_atr_mult=trial.suggest_float("sl_atr_mult", 1.0, 3.0),
                tp_atr_mult=trial.suggest_float("tp_atr_mult", 1.5, 4.0),
                signal_threshold=trial.suggest_float("signal_threshold", 0.55, 0.75),
                max_hold_bars=trial.suggest_int("max_hold_bars", 10, 30, step=5),
            )

            # Create strategy copy with trial params
            strategy.xgb_params = xgb_params
            strategy.risk = risk

            trainer = WalkForwardTrainer(config=self.training_config, risk=risk)

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

            # Track best result
            if best_result_holder[0] is None or result.sharpe > (best_result_holder[0].sharpe or -999):
                best_result_holder[0] = result

            return result.sharpe

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
