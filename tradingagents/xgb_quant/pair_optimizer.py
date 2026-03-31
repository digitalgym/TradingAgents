"""
Pair Optimizer — iterative per-pair strategy optimization.

After batch training produces baseline results for all combos,
this module reviews each pair, identifies improvable combos,
and runs staged optimization (risk tune → full tune → window profiles)
with holdout validation to catch overfitting.

Usage:
    optimizer = PairOptimizer()
    results = optimizer.run()  # Optimize all pairs
    results = optimizer.run(symbols=["EURUSD", "XAUUSD"])  # Specific pairs
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import pandas as pd

from tradingagents.xgb_quant.config import (
    DEFAULT_WATCHLIST,
    RESULTS_DIR,
    MODELS_DIR,
    OptimizationConfig,
    RiskDefaults,
    TrainingConfig,
    WINDOW_PROFILES,
    FeatureWindows,
)

logger = logging.getLogger(__name__)

OPTIMIZATION_DIR = RESULTS_DIR / "_optimization"


@dataclass
class ComboResult:
    """Tracks optimization result for one strategy+timeframe combo."""
    strategy: str
    timeframe: str
    tier_before: str = "C"
    tier_after: str = "C"
    phases_run: List[str] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    optimized: Dict[str, Any] = field(default_factory=dict)
    holdout: Dict[str, Any] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    window_profile: str = "default"
    overfit_flag: bool = False
    verdict: str = ""  # "IMPROVED", "ALREADY_GOOD", "NON_VIABLE", "OVERFIT"


@dataclass
class PairOptimizationResult:
    """Full optimization result for one pair."""
    symbol: str
    combos: List[ComboResult] = field(default_factory=list)
    best_combo: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0


@dataclass
class OptimizationReport:
    """Full report across all pairs."""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0
    total_pairs: int = 0
    total_combos: int = 0
    tier_a_count: int = 0
    tier_b_improved: int = 0
    tier_c_count: int = 0
    overfit_count: int = 0
    pairs: Dict[str, PairOptimizationResult] = field(default_factory=dict)


class PairOptimizer:
    """Iterative per-pair strategy optimization with holdout validation."""

    def __init__(self, config: Optional[OptimizationConfig] = None, max_hours: float = 6.0):
        self.config = config or OptimizationConfig()
        self.max_hours = max_hours
        self._progress_callback: Optional[Callable] = None
        self._cancelled = False
        self._global_start = 0.0

    def on_progress(self, callback: Callable):
        self._progress_callback = callback

    def cancel(self):
        self._cancelled = True

    def _report(self, msg: str, pair_idx: int = 0, total_pairs: int = 0):
        logger.info(msg)
        if self._progress_callback:
            self._progress_callback(pair_idx, total_pairs, msg)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        bars: int = 2000,
    ) -> OptimizationReport:
        """Run optimization for all pairs."""
        symbols = symbols or DEFAULT_WATCHLIST
        timeframes = timeframes or ["D1", "H4"]
        strategies = strategies or [
            "trend_following", "mean_reversion", "breakout",
            "smc_zones", "volume_profile_strat",
        ]

        report = OptimizationReport(
            started_at=datetime.utcnow().isoformat(),
            total_pairs=len(symbols),
        )
        self._global_start = time.time()

        for i, symbol in enumerate(symbols):
            if self._cancelled:
                logger.info("Optimization cancelled.")
                break

            # Check global timeout
            elapsed = time.time() - self._global_start
            if elapsed > self.max_hours * 3600:
                logger.info(f"Global timeout ({self.max_hours}h) reached after {i} pairs.")
                break

            self._report(f"Optimizing {symbol} ({i+1}/{len(symbols)})...", i + 1, len(symbols))

            try:
                pair_result = self._optimize_pair(symbol, timeframes, strategies, bars)
                report.pairs[symbol] = pair_result

                # Aggregate counts
                for combo in pair_result.combos:
                    report.total_combos += 1
                    if combo.tier_before == "A":
                        report.tier_a_count += 1
                    elif combo.verdict == "IMPROVED":
                        report.tier_b_improved += 1
                    elif combo.tier_before == "C":
                        report.tier_c_count += 1
                    if combo.overfit_flag:
                        report.overfit_count += 1

            except Exception as e:
                logger.error(f"Failed to optimize {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        report.finished_at = datetime.utcnow().isoformat()
        report.duration_seconds = round(time.time() - self._global_start, 1)

        self._save_reports(report)
        self._update_blacklist(report)

        self._report(
            f"Optimization complete: {report.tier_b_improved} improved, "
            f"{report.tier_a_count} already good, {report.tier_c_count} non-viable, "
            f"{report.overfit_count} overfit in {report.duration_seconds:.0f}s",
            len(symbols), len(symbols),
        )

        return report

    def _optimize_pair(
        self,
        symbol: str,
        timeframes: List[str],
        strategies: List[str],
        bars: int,
    ) -> PairOptimizationResult:
        """Optimize all combos for a single pair."""
        start = time.time()
        result = PairOptimizationResult(symbol=symbol)

        # Load data for each timeframe
        dfs = {}
        for tf in timeframes:
            df = self._load_data(symbol, tf, bars)
            if df is not None and len(df) >= 400:
                dfs[tf] = df

        if not dfs:
            logger.warning(f"{symbol}: insufficient data for any timeframe")
            return result

        strategy_classes = self._load_strategies(strategies)

        for tf in timeframes:
            if tf not in dfs:
                continue
            df_full = dfs[tf]

            # Split into optimization and holdout
            df_opt, df_holdout = self._split_data(df_full)

            for strat_name in strategies:
                if strat_name not in strategy_classes:
                    continue
                if self._cancelled:
                    break

                combo = self._optimize_combo(
                    symbol, tf, strat_name, strategy_classes[strat_name],
                    df_opt, df_holdout,
                )
                result.combos.append(combo)

        # Find best combo
        viable = [c for c in result.combos if c.tier_after == "A" and not c.overfit_flag]
        if viable:
            best = max(viable, key=lambda c: c.optimized.get("sharpe", c.baseline.get("sharpe", 0)))
            result.best_combo = {
                "strategy": best.strategy,
                "timeframe": best.timeframe,
                "sharpe": best.optimized.get("sharpe", best.baseline.get("sharpe", 0)),
                "holdout_validated": best.holdout.get("status") == "validated",
            }

        result.duration_seconds = round(time.time() - start, 1)
        return result

    def _optimize_combo(
        self,
        symbol: str,
        tf: str,
        strat_name: str,
        strategy_cls: type,
        df_opt: pd.DataFrame,
        df_holdout: pd.DataFrame,
    ) -> ComboResult:
        """Optimize a single strategy+timeframe combo through the phases."""
        combo = ComboResult(strategy=strat_name, timeframe=tf)

        # Load baseline result
        baseline = self._load_result(symbol, strat_name, tf)
        if baseline:
            combo.baseline = baseline
        else:
            # No baseline — run one to get it
            try:
                from tradingagents.xgb_quant.trainer import WalkForwardTrainer
                trainer = WalkForwardTrainer()
                strategy = strategy_cls()
                bt = trainer.train_and_evaluate(strategy, df_opt, symbol, tf)
                combo.baseline = {
                    "sharpe": bt.sharpe, "win_rate": bt.win_rate,
                    "profit_factor": bt.profit_factor, "total_trades": bt.total_trades,
                    "max_drawdown_pct": bt.max_drawdown_pct,
                }
            except Exception as e:
                combo.verdict = "NON_VIABLE"
                combo.baseline = {"error": str(e)}
                return combo

        # Classify tier
        combo.tier_before = self._classify_tier(combo.baseline)

        if combo.tier_before == "C":
            combo.verdict = "NON_VIABLE"
            combo.tier_after = "C"
            self._report(f"  {symbol} {tf} {strat_name}: TIER C — skipped (non-viable)")
            return combo

        # Get current XGB params from model file
        current_xgb_params = self._load_model_params(strat_name, symbol, tf)

        best_sharpe = combo.baseline.get("sharpe", -999)
        best_params = dict(current_xgb_params) if current_xgb_params else {}
        best_risk = None

        # --- Phase 1: Risk-only tune ---
        if combo.tier_before == "B" or combo.tier_before == "A":
            phase_label = "Phase 1 (risk tune)" if combo.tier_before == "B" else "Light tune (risk)"
            self._report(f"  {symbol} {tf} {strat_name}: {phase_label}...")

            try:
                from tradingagents.xgb_quant.parameter_tuner import ParameterTuner
                tuner = ParameterTuner()
                tune_result = tuner.tune(
                    strategy=strategy_cls(),
                    df=df_opt,
                    symbol=symbol,
                    timeframe=tf,
                    n_trials=self.config.phase1_trials,
                    timeout=self.config.phase1_timeout,
                    search_mode="risk_only",
                    fixed_xgb_params=current_xgb_params,
                )
                combo.phases_run.append("risk_tune")

                if tune_result.best_result and tune_result.best_sharpe > best_sharpe:
                    best_sharpe = tune_result.best_sharpe
                    best_params.update(tune_result.best_params)
                    best_risk = RiskDefaults(
                        sl_atr_mult=tune_result.best_params.get("sl_atr_mult", 1.5),
                        tp_atr_mult=tune_result.best_params.get("tp_atr_mult", 2.5),
                        signal_threshold=tune_result.best_params.get("signal_threshold", 0.60),
                        max_hold_bars=tune_result.best_params.get("max_hold_bars", 20),
                    )
                    combo.optimized = {
                        "sharpe": tune_result.best_result.sharpe,
                        "win_rate": tune_result.best_result.win_rate,
                        "profit_factor": tune_result.best_result.profit_factor,
                        "total_trades": tune_result.best_result.total_trades,
                        "max_drawdown_pct": tune_result.best_result.max_drawdown_pct,
                    }

                    self._report(
                        f"    Risk tune: Sharpe {combo.baseline.get('sharpe', 0):.2f} → "
                        f"{tune_result.best_sharpe:.2f}"
                    )
            except Exception as e:
                logger.warning(f"    Phase 1 failed: {e}")

            # Check if we've reached Tier A
            current_result = combo.optimized if combo.optimized else combo.baseline
            if self._classify_tier(current_result) == "A":
                combo.tier_after = "A"
                if combo.tier_before == "A":
                    combo.verdict = "ALREADY_GOOD"
                else:
                    combo.verdict = "IMPROVED"
                combo.best_params = best_params
                # Still do holdout validation
                self._run_holdout(combo, strategy_cls, df_holdout, symbol, tf, best_params)
                return combo

            # Tier A combos stop here
            if combo.tier_before == "A":
                combo.tier_after = "A"
                combo.verdict = "ALREADY_GOOD"
                combo.best_params = best_params
                self._run_holdout(combo, strategy_cls, df_holdout, symbol, tf, best_params)
                return combo

        # --- Phase 2: Full XGB tune (Tier B only) ---
        self._report(f"  {symbol} {tf} {strat_name}: Phase 2 (full XGB tune)...")

        try:
            from tradingagents.xgb_quant.parameter_tuner import ParameterTuner
            tuner = ParameterTuner()
            tune_result = tuner.tune(
                strategy=strategy_cls(),
                df=df_opt,
                symbol=symbol,
                timeframe=tf,
                n_trials=self.config.phase2_trials,
                timeout=self.config.phase2_timeout,
                search_mode="full",
            )
            combo.phases_run.append("xgb_tune")

            if tune_result.best_result and tune_result.best_sharpe > best_sharpe:
                improvement = tune_result.best_sharpe - best_sharpe
                best_sharpe = tune_result.best_sharpe
                best_params = tune_result.best_params
                combo.optimized = {
                    "sharpe": tune_result.best_result.sharpe,
                    "win_rate": tune_result.best_result.win_rate,
                    "profit_factor": tune_result.best_result.profit_factor,
                    "total_trades": tune_result.best_result.total_trades,
                    "max_drawdown_pct": tune_result.best_result.max_drawdown_pct,
                }
                self._report(
                    f"    XGB tune: Sharpe → {tune_result.best_sharpe:.2f} "
                    f"(+{improvement:.2f})"
                )

                # Check plateau
                if improvement < self.config.plateau_threshold:
                    self._report(f"    Plateau reached (delta={improvement:.3f})")
        except Exception as e:
            logger.warning(f"    Phase 2 failed: {e}")

        # Check tier
        current_result = combo.optimized if combo.optimized else combo.baseline
        if self._classify_tier(current_result) == "A":
            combo.tier_after = "A"
            combo.verdict = "IMPROVED"
            combo.best_params = best_params
            self._run_holdout(combo, strategy_cls, df_holdout, symbol, tf, best_params)
            return combo

        # --- Phase 3: Feature window profiles ---
        self._report(f"  {symbol} {tf} {strat_name}: Phase 3 (window profiles)...")

        best_window = "default"
        try:
            from tradingagents.xgb_quant.trainer import WalkForwardTrainer

            for profile_name, windows in WINDOW_PROFILES.items():
                if profile_name == "default":
                    continue  # Already tested
                strategy = strategy_cls()
                strategy.feature_windows = windows
                if best_params:
                    xgb_p = {k: v for k, v in best_params.items()
                             if k not in ("sl_atr_mult", "tp_atr_mult", "signal_threshold", "max_hold_bars")}
                    if xgb_p:
                        strategy.xgb_params = xgb_p

                trainer = WalkForwardTrainer()
                bt = trainer.train_and_evaluate(strategy, df_opt, symbol, tf)

                if bt.sharpe > best_sharpe:
                    best_sharpe = bt.sharpe
                    best_window = profile_name
                    combo.optimized = {
                        "sharpe": bt.sharpe, "win_rate": bt.win_rate,
                        "profit_factor": bt.profit_factor,
                        "total_trades": bt.total_trades,
                        "max_drawdown_pct": bt.max_drawdown_pct,
                    }
                    self._report(f"    Window '{profile_name}': Sharpe → {bt.sharpe:.2f}")

            combo.phases_run.append("window_profiles")
            combo.window_profile = best_window
        except Exception as e:
            logger.warning(f"    Phase 3 failed: {e}")

        # Final tier classification
        current_result = combo.optimized if combo.optimized else combo.baseline
        combo.tier_after = self._classify_tier(current_result)
        combo.best_params = best_params

        if combo.tier_after in ("A", "B") and combo.optimized:
            combo.verdict = "IMPROVED" if combo.tier_after == "A" else "MARGINAL"
        else:
            combo.verdict = "NON_VIABLE"

        # Holdout validation
        self._run_holdout(combo, strategy_cls, df_holdout, symbol, tf, best_params)

        return combo

    def _run_holdout(
        self,
        combo: ComboResult,
        strategy_cls: type,
        df_holdout: pd.DataFrame,
        symbol: str,
        tf: str,
        best_params: Dict[str, Any],
    ):
        """Run holdout validation and check for overfitting."""
        if df_holdout is None or len(df_holdout) < self.config.min_holdout_bars:
            combo.holdout = {"status": "insufficient_data", "bars": len(df_holdout) if df_holdout is not None else 0}
            return

        try:
            from tradingagents.xgb_quant.trainer import WalkForwardTrainer

            strategy = strategy_cls()
            if best_params:
                xgb_p = {k: v for k, v in best_params.items()
                         if k not in ("sl_atr_mult", "tp_atr_mult", "signal_threshold", "max_hold_bars")}
                if xgb_p:
                    strategy.xgb_params = xgb_p

            trainer = WalkForwardTrainer()
            bt = trainer.train_and_evaluate(strategy, df_holdout, symbol, tf)

            opt_sharpe = combo.optimized.get("sharpe", combo.baseline.get("sharpe", 0))
            holdout_sharpe = bt.sharpe

            # Check for overfitting
            if opt_sharpe > 0.5 and holdout_sharpe < 0:
                status = "overfit"
                combo.overfit_flag = True
                combo.verdict = "OVERFIT"
            elif opt_sharpe > 0 and holdout_sharpe >= opt_sharpe * self.config.overfit_sharpe_ratio:
                status = "validated"
            elif bt.total_trades < 5:
                status = "insufficient_trades"
            else:
                status = "degraded"

            combo.holdout = {
                "status": status,
                "sharpe": holdout_sharpe,
                "win_rate": bt.win_rate,
                "profit_factor": bt.profit_factor,
                "total_trades": bt.total_trades,
            }

            if combo.overfit_flag:
                self._report(
                    f"    OVERFIT: opt Sharpe={opt_sharpe:.2f}, "
                    f"holdout Sharpe={holdout_sharpe:.2f}"
                )

        except Exception as e:
            combo.holdout = {"status": "error", "error": str(e)}

    def _classify_tier(self, result: Dict[str, Any]) -> str:
        """Classify a backtest result into tier A/B/C."""
        sharpe = result.get("sharpe", -999)
        wr = result.get("win_rate", 0)
        pf = result.get("profit_factor", 0)
        trades = result.get("total_trades", 0)

        if (sharpe >= self.config.tier_a_sharpe
                and wr >= self.config.tier_a_win_rate
                and pf >= self.config.tier_a_pf
                and trades >= self.config.tier_a_min_trades):
            return "A"

        if (sharpe >= self.config.tier_b_sharpe
                and wr >= self.config.tier_b_win_rate
                and pf >= self.config.tier_b_pf
                and trades >= self.config.tier_b_min_trades):
            return "B"

        return "C"

    def _split_data(self, df: pd.DataFrame):
        """Split data into optimization (80%) and holdout (20%)."""
        holdout_start = int(len(df) * (1 - self.config.holdout_frac))
        return df.iloc[:holdout_start].copy(), df.iloc[holdout_start:].copy()

    @staticmethod
    def _load_data(symbol: str, timeframe: str, bars: int):
        try:
            from tradingagents.automation.auto_tuner import load_mt5_data
            return load_mt5_data(symbol, timeframe, bars)
        except Exception as e:
            logger.warning(f"Failed to load {symbol} {timeframe}: {e}")
            return None

    @staticmethod
    def _load_strategies(names: List[str]) -> Dict[str, type]:
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
    def _load_result(symbol: str, strategy: str, timeframe: str) -> Optional[Dict]:
        result_file = RESULTS_DIR / symbol / f"{strategy}_{timeframe}.json"
        if not result_file.exists():
            return None
        try:
            return json.loads(result_file.read_text())
        except Exception:
            return None

    @staticmethod
    def _load_model_params(strategy: str, symbol: str, timeframe: str) -> Optional[Dict]:
        result = PairOptimizer._load_result(symbol, strategy, timeframe)
        if result and "params" in result:
            return result["params"]
        return None

    def _save_reports(self, report: OptimizationReport):
        """Save JSON + human-readable reports."""
        OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_file = OPTIMIZATION_DIR / f"report_{ts}.json"
        json_file.write_text(json.dumps(asdict(report), indent=2, default=str))

        # Human-readable viability report
        lines = [
            f"PAIR OPTIMIZATION REPORT — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            "",
        ]
        for symbol, pair_result in report.pairs.items():
            lines.append(f"{symbol}:")
            for combo in pair_result.combos:
                sharpe_before = combo.baseline.get("sharpe", 0)
                sharpe_after = combo.optimized.get("sharpe", sharpe_before)
                wr = combo.optimized.get("win_rate", combo.baseline.get("win_rate", 0))
                holdout_status = combo.holdout.get("status", "")

                if combo.verdict == "IMPROVED":
                    mark = "validated" if holdout_status == "validated" else holdout_status
                    lines.append(
                        f"  IMPROVED: {combo.strategy} {combo.timeframe} "
                        f"Sharpe {sharpe_before:.2f} -> {sharpe_after:.2f} "
                        f"WR={wr:.0f}% [{mark}]"
                    )
                elif combo.verdict == "ALREADY_GOOD":
                    lines.append(
                        f"  GOOD: {combo.strategy} {combo.timeframe} "
                        f"Sharpe {sharpe_after:.2f} WR={wr:.0f}%"
                    )
                elif combo.verdict == "OVERFIT":
                    h_sharpe = combo.holdout.get("sharpe", 0)
                    lines.append(
                        f"  OVERFIT: {combo.strategy} {combo.timeframe} "
                        f"opt={sharpe_after:.2f} holdout={h_sharpe:.2f} !!"
                    )
                elif combo.verdict == "NON_VIABLE":
                    lines.append(
                        f"  NON-VIABLE: {combo.strategy} {combo.timeframe} "
                        f"Sharpe={sharpe_before:.2f} WR={combo.baseline.get('win_rate', 0):.0f}%"
                    )
                else:
                    lines.append(
                        f"  {combo.verdict}: {combo.strategy} {combo.timeframe} "
                        f"Sharpe={sharpe_after:.2f}"
                    )

            if pair_result.best_combo:
                b = pair_result.best_combo
                lines.append(
                    f"  >>> BEST: {b['strategy']} {b['timeframe']} "
                    f"Sharpe={b['sharpe']:.2f}"
                )
            lines.append("")

        lines.extend([
            "=" * 60,
            f"Total combos: {report.total_combos}",
            f"Already good (Tier A): {report.tier_a_count}",
            f"Improved: {report.tier_b_improved}",
            f"Non-viable: {report.tier_c_count}",
            f"Overfit warnings: {report.overfit_count}",
            f"Runtime: {report.duration_seconds:.0f}s",
        ])

        txt_file = OPTIMIZATION_DIR / f"viability_{ts}.txt"
        txt_file.write_text("\n".join(lines))

        # Save optimized params per pair
        params_dir = OPTIMIZATION_DIR / "optimized_params"
        params_dir.mkdir(exist_ok=True)
        for symbol, pair_result in report.pairs.items():
            pair_file = params_dir / f"{symbol}.json"
            pair_file.write_text(json.dumps(asdict(pair_result), indent=2, default=str))

        logger.info(f"Reports saved to {OPTIMIZATION_DIR}")

    def _update_blacklist(self, report: OptimizationReport):
        """Update blacklist with refined non-viable + overfit combos."""
        from tradingagents.xgb_quant.batch_trainer import BatchTrainer

        blacklist = BatchTrainer.load_blacklist()
        existing_keys = {f"{e['symbol']}:{e['strategy']}:{e.get('timeframe', '')}" for e in blacklist}

        for symbol, pair_result in report.pairs.items():
            for combo in pair_result.combos:
                key = f"{symbol}:{combo.strategy}:{combo.timeframe}"
                if key in existing_keys:
                    continue
                if combo.verdict in ("NON_VIABLE", "OVERFIT"):
                    blacklist.append({
                        "symbol": symbol,
                        "strategy": combo.strategy,
                        "timeframe": combo.timeframe,
                        "reason": combo.verdict,
                    })

        # Save updated blacklist
        blacklist_file = RESULTS_DIR / "blacklist.json"
        blacklist_file.write_text(json.dumps(blacklist, indent=2))
        logger.info(f"Blacklist updated: {len(blacklist)} entries")
