"""
Rule-Based Strategy Backtester & Optimizer

Runs walk-forward backtests for all (or selected) strategies on given symbols,
with optional Optuna risk-parameter optimisation.

Includes quality checks:
  - Minimum trade count filter (default 30)
  - Fold-to-fold stability check

Usage:
    # Backtest all strategies on XAUUSD H4 (default)
    python run_strategy_backtest.py

    # Backtest specific strategy and symbol
    python run_strategy_backtest.py --symbols XAUUSD BTCUSD --strategies donchian_breakout trend_following

    # Run with Optuna optimisation (slower, better risk params)
    python run_strategy_backtest.py --optimize --trials 30 --timeout 300

    # Use more data
    python run_strategy_backtest.py --bars 5000 --timeframe H1

    # Adjust minimum trades threshold
    python run_strategy_backtest.py --min-trades 50
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.quant_strats.config import (
    TrainingConfig, RiskDefaults, RESULTS_DIR,
)
from tradingagents.quant_strats.trainer import WalkForwardTrainer, BacktestResult
from tradingagents.quant_strats.parameter_tuner import ParameterTuner, TuneResult
from tradingagents.quant_strats.strategies.trend_following import TrendFollowingStrategy
from tradingagents.quant_strats.strategies.mean_reversion import MeanReversionStrategy
from tradingagents.quant_strats.strategies.breakout import BreakoutStrategy
from tradingagents.quant_strats.strategies.smc_zones import SMCZonesStrategy
from tradingagents.quant_strats.strategies.volume_profile_strat import VolumeProfileStrategy
from tradingagents.quant_strats.strategies.keltner_mean_reversion import KeltnerMeanReversionStrategy
from tradingagents.quant_strats.strategies.copper_ema_pullback import CopperEMAPullbackStrategy
from tradingagents.quant_strats.strategies.gold_platinum_ratio import GoldPlatinumRatioStrategy

logger = logging.getLogger("strategy_backtest")

STRATEGY_CLASSES = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "donchian_breakout": BreakoutStrategy,
    "smc_zones": SMCZonesStrategy,
    "volume_profile_strat": VolumeProfileStrategy,
    "keltner_mean_reversion": KeltnerMeanReversionStrategy,
    "copper_ema_pullback": CopperEMAPullbackStrategy,
    "gold_platinum_ratio": GoldPlatinumRatioStrategy,
}


def load_data(symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
    """Load OHLCV data from MT5."""
    try:
        from tradingagents.automation.auto_tuner import load_mt5_data
        df = load_mt5_data(symbol, timeframe, bars)
        logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data for {symbol} {timeframe}: {e}")
        return None


def run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    training_config: TrainingConfig,
) -> Optional[BacktestResult]:
    """Run a single walk-forward backtest."""
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None

    # Use strategy-specific risk defaults where available
    from tradingagents.quant_strats.config import (
        KELTNER_MR_DEFAULTS, COPPER_EMA_DEFAULTS, GOLD_PLAT_RATIO_DEFAULTS,
    )
    risk_map = {
        "keltner_mean_reversion": KELTNER_MR_DEFAULTS,
        "copper_ema_pullback": COPPER_EMA_DEFAULTS,
        "gold_platinum_ratio": GOLD_PLAT_RATIO_DEFAULTS,
    }
    risk = risk_map.get(strategy_name)
    strategy = cls(risk=risk) if risk else cls()
    trainer = WalkForwardTrainer(config=training_config)

    try:
        result = trainer.train_and_evaluate(
            strategy=strategy,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
        )
        return result
    except Exception as e:
        logger.error(f"Backtest failed for {strategy_name} on {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_optimize(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    training_config: TrainingConfig,
    n_trials: int = 30,
    timeout: int = 300,
) -> Optional[TuneResult]:
    """Run Optuna optimisation for a strategy."""
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None

    strategy = cls()
    tuner = ParameterTuner(training_config=training_config)

    try:
        result = tuner.tune(
            strategy=strategy,
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            n_trials=n_trials,
            timeout=timeout,
        )
        return result
    except Exception as e:
        logger.error(f"Optimization failed for {strategy_name} on {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(result: BacktestResult, min_trades: int):
    """Pretty-print a backtest result with stability diagnostics."""
    # Status flags
    flags = []
    if result.total_trades < min_trades:
        flags.append(f"LOW TRADES ({result.total_trades}<{min_trades})")
    if result.fold_win_rate_std > 0.25:
        flags.append(f"UNSTABLE (std={result.fold_win_rate_std:.0%})")

    status = " | ".join(flags) if flags else "OK"

    print(f"\n{'-'*70}")
    print(f"  {result.strategy} on {result.symbol} {result.timeframe}  [{status}]")
    print(f"{'-'*70}")
    print(f"  Trades:       {result.total_trades:>6}   (min required: {min_trades})")
    print(f"  Win Rate:     {result.win_rate:>6.1f}%  ({result.wins}W / {result.losses}L)")
    print(f"  Avg P&L:      {result.avg_pnl_pct:>+6.3f}%")
    print(f"  Total P&L:    {result.total_pnl_pct:>+6.2f}%")
    print(f"  Profit Factor:{result.profit_factor:>6.2f}")
    print(f"  Sharpe:       {result.sharpe:>6.2f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:>6.2f}%")
    print(f"  Avg Hold:     {result.avg_hold_bars:>6.1f} bars")
    print(f"  -- Stability Diagnostics --")
    print(f"  Folds:        {result.n_folds:>6}")
    print(f"  Avg Fold WR:  {result.test_win_rate:>6.1%}")
    print(f"  Fold WR Std:  {result.fold_win_rate_std:>6.1%}  (threshold: 25%)")


def print_summary_table(results: List[BacktestResult], min_trades: int):
    """Print a summary comparison table."""
    if not results:
        return

    print(f"\n{'='*85}")
    print(f"  SUMMARY")
    print(f"{'='*85}")
    print(f"  {'Strategy':<22} {'Symbol':<8} {'TF':<4} {'Trades':>6} {'WR%':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'FoldStd':>7} {'Status':<12}")
    print(f"  {'-'*22} {'-'*8} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*12}")

    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        if r.total_trades < min_trades:
            status = "LOW TRADES"
        elif r.fold_win_rate_std > 0.25:
            status = "UNSTABLE"
        else:
            status = "OK"

        print(f"  {r.strategy:<22} {r.symbol:<8} {r.timeframe:<4} "
              f"{r.total_trades:>6} {r.win_rate:>5.1f}% "
              f"{r.profit_factor:>6.2f} {r.sharpe:>7.2f} "
              f"{r.fold_win_rate_std:>6.0%} {status:<12}")

    print(f"{'='*85}")

    valid = [r for r in results
             if r.total_trades >= min_trades
             and r.fold_win_rate_std <= 0.25]
    print(f"\n  {len(valid)}/{len(results)} results passed all checks "
          f"(min_trades={min_trades}, max_std=25%)")


def main():
    parser = argparse.ArgumentParser(
        description="Rule-Based Strategy Backtester & Optimizer"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["XAUUSD"],
        help="Symbols to backtest (default: XAUUSD)",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGY_CLASSES.keys()),
        help=f"Strategies to test (default: all). Choices: {list(STRATEGY_CLASSES.keys())}",
    )
    parser.add_argument(
        "--timeframe", default="H4",
        help="Timeframe (default: H4)",
    )
    parser.add_argument(
        "--bars", type=int, default=2000,
        help="Number of bars to fetch (default: 2000)",
    )
    parser.add_argument(
        "--min-trades", type=int, default=30,
        help="Minimum trades for a valid backtest (default: 30)",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run Optuna hyperparameter optimisation",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials (default: 30)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Max seconds per optimisation run (default: 300)",
    )
    parser.add_argument(
        "--train-window", type=int, default=500,
        help="Walk-forward train window size (default: 500)",
    )
    parser.add_argument(
        "--test-window", type=int, default=100,
        help="Walk-forward test window size (default: 100)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    training_config = TrainingConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        total_bars=args.bars,
        min_trades=args.min_trades,
    )

    print(f"\n{'='*70}")
    print(f"  Rule-Based Backtest {'& Optimize ' if args.optimize else ''}Runner")
    print(f"{'='*70}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Bars:       {args.bars}")
    print(f"  Min Trades: {args.min_trades}")
    print(f"  Train/Test: {args.train_window}/{args.test_window}")
    if args.optimize:
        print(f"  Optuna:     {args.trials} trials, {args.timeout}s timeout")
    print(f"{'='*70}\n")

    all_results: List[BacktestResult] = []

    for symbol in args.symbols:
        # Load data once per symbol
        df = load_data(symbol, args.timeframe, args.bars)
        if df is None:
            continue

        for strategy_name in args.strategies:
            if strategy_name not in STRATEGY_CLASSES:
                logger.warning(f"Skipping unknown strategy: {strategy_name}")
                continue

            print(f"\n>>> {strategy_name} on {symbol} {args.timeframe}")

            if args.optimize:
                tune_result = run_optimize(
                    strategy_name, symbol, args.timeframe, df,
                    training_config, args.trials, args.timeout,
                )
                if tune_result and tune_result.best_result:
                    result = tune_result.best_result
                    print(f"  Best params from {tune_result.n_trials} trials "
                          f"(Sharpe={tune_result.best_sharpe:.3f})")
                    # Save optimised params
                    params_path = RESULTS_DIR / symbol
                    params_path.mkdir(parents=True, exist_ok=True)
                    with open(params_path / f"{strategy_name}_{args.timeframe}_optimised.json", "w") as f:
                        json.dump(tune_result.best_params, f, indent=2)
                else:
                    logger.warning(f"  Optimisation produced no valid result")
                    continue
            else:
                result = run_backtest(
                    strategy_name, symbol, args.timeframe, df, training_config,
                )

            if result:
                all_results.append(result)
                print_result(result, args.min_trades)

    # Summary
    print_summary_table(all_results, args.min_trades)


if __name__ == "__main__":
    main()
