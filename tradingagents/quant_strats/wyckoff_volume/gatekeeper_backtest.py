"""
Wyckoff Gatekeeper Backtest — replay historical signals through the gatekeeper.

Measures the filter's impact on win rate and profit factor before live deployment.

Usage:
    python -m tradingagents.quant_strats.wyckoff_volume.gatekeeper_backtest \
        --symbol XAUUSD --timeframe D1 --strategy breakout \
        --max-signals 50

    # Dry run (no LLM calls, just shows what would be sent):
    python -m tradingagents.quant_strats.wyckoff_volume.gatekeeper_backtest \
        --symbol XAUUSD --timeframe D1 --strategy breakout --dry-run
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Strategy name → module.class mapping (avoids broken global registry)
_STRATEGY_MAP = {
    "trend_following": ("tradingagents.quant_strats.strategies.trend_following", "TrendFollowingStrategy"),
    "mean_reversion": ("tradingagents.quant_strats.strategies.mean_reversion", "MeanReversionStrategy"),
    "breakout": ("tradingagents.quant_strats.strategies.breakout", "BreakoutStrategy"),
    "smc_zones": ("tradingagents.quant_strats.strategies.smc_zones", "SMCZonesStrategy"),
    "volume_profile_strat": ("tradingagents.quant_strats.strategies.volume_profile_strat", "VolumeProfileStrategy"),
    "keltner_mean_reversion": ("tradingagents.quant_strats.strategies.keltner_mean_reversion", "KeltnerMeanReversionStrategy"),
    "copper_ema_pullback": ("tradingagents.quant_strats.strategies.copper_ema_pullback", "CopperEMAPullbackStrategy"),
    "gold_platinum_ratio": ("tradingagents.quant_strats.strategies.gold_platinum_ratio", "GoldPlatinumRatioStrategy"),
}


def _load_strategy(strategy_name: str):
    """Load a strategy class by name without triggering the global registry."""
    import importlib
    if strategy_name not in _STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(_STRATEGY_MAP.keys())}")
    module_path, class_name = _STRATEGY_MAP[strategy_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def run_gatekeeper_backtest(
    symbol: str = "XAUUSD",
    timeframe: str = "D1",
    strategy_name: str = "breakout",
    bars: int = 500,
    min_prob: float = 0.60,
    max_signals: Optional[int] = None,
    dry_run: bool = False,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    max_hold_bars: int = 20,
) -> Dict:
    """
    Replay historical bars through strategy + gatekeeper, measure filter impact.

    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe
        strategy_name: Strategy to generate signals with
        bars: Number of historical bars to load
        min_prob: Minimum probability to consider a signal
        max_signals: Limit LLM calls (None = unlimited)
        dry_run: If True, compute features but skip LLM calls
        sl_atr_mult: ATR multiplier for stop loss
        tp_atr_mult: ATR multiplier for take profit
        max_hold_bars: Max bars to wait for TP/SL hit

    Returns:
        Dict with backtest metrics
    """
    from tradingagents.automation.auto_tuner import load_mt5_data
    from tradingagents.quant_strats.features.wyckoff import WyckoffFeatures
    from tradingagents.quant_strats.features.technical import TechnicalFeatures
    from tradingagents.quant_strats.wyckoff_volume.gatekeeper_prompts import (
        build_context_snapshot,
    )

    print(f"\n{'='*60}")
    print(f"Wyckoff Gatekeeper Backtest")
    print(f"Symbol: {symbol} | TF: {timeframe} | Strategy: {strategy_name}")
    print(f"Bars: {bars} | Min Prob: {min_prob} | Max Signals: {max_signals or 'unlimited'}")
    print(f"Dry Run: {dry_run}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading MT5 data...")
    df = load_mt5_data(symbol, timeframe, bars=bars)
    print(f"Loaded {len(df)} bars from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    # Compute ATR for SL/TP
    from tradingagents.automation.auto_tuner import _compute_atr
    atr_arr = _compute_atr(high, low, close)

    # Load strategy directly (avoids global registry which may have broken imports)
    strategy = _load_strategy(strategy_name)
    strategy.load_model(symbol, timeframe)  # Load trained model if available

    # Compute features for context snapshots
    wyckoff_feat = WyckoffFeatures()
    technical_feat = TechnicalFeatures()
    wyckoff_df = wyckoff_feat.compute(df)
    technical_df = technical_feat.compute(df)

    # Init gatekeeper (unless dry run)
    gatekeeper = None
    if not dry_run:
        from tradingagents.quant_strats.wyckoff_volume.llm_gatekeeper import (
            WyckoffGatekeeper,
        )
        gatekeeper = WyckoffGatekeeper()

    # Walk-forward: generate signals from bar 100 onwards (warmup)
    warmup = 100
    results: List[Dict] = []
    signal_count = 0

    for i in range(warmup, len(df) - max_hold_bars):
        if max_signals and signal_count >= max_signals:
            break

        # Use bars up to and including i for prediction
        sub_df = df.iloc[:i + 1].copy()
        current_atr = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 1.0
        current_price = float(close[i])

        # Get signal from strategy
        signal = strategy.predict_signal(
            strategy.get_feature_set().compute(sub_df),
            current_atr, current_price,
        )

        if signal.direction == "HOLD":
            continue

        prob = signal.confidence
        if prob < min_prob:
            continue

        signal_count += 1
        direction = signal.direction

        # Calculate SL/TP
        if direction == "BUY":
            sl = current_price - sl_atr_mult * current_atr
            tp = current_price + tp_atr_mult * current_atr
        else:
            sl = current_price + sl_atr_mult * current_atr
            tp = current_price - tp_atr_mult * current_atr

        # Determine actual outcome: walk forward up to max_hold_bars
        outcome = _evaluate_outcome(
            high, low, close, i, direction, sl, tp, max_hold_bars
        )

        # Get gatekeeper verdict
        verdict_data = None
        if dry_run:
            # Show what would be sent
            wyckoff_row = wyckoff_df.iloc[i]
            technical_row = technical_df.iloc[i]
            context = build_context_snapshot(
                wyckoff_row, technical_row, prob, direction, symbol, timeframe,
            )
            if signal_count <= 3:
                print(f"\n--- Signal #{signal_count} Context Preview ---")
                print(context[:500])
                print("...\n")
            verdict_data = {"verdict": "DRY_RUN", "confidence": 0.0}
        else:
            verdict = gatekeeper.evaluate(
                sub_df, prob, direction, symbol, timeframe,
            )
            verdict_data = verdict.model_dump()
            print(
                f"  Signal #{signal_count}: {direction} prob={prob:.2f} "
                f"-> {verdict_data['verdict']} (conf={verdict_data['confidence']:.2f}) "
                f"| Outcome: {outcome['result']}"
            )

        results.append({
            "bar_index": i,
            "date": str(df["date"].iloc[i]),
            "direction": direction,
            "xgb_prob": round(prob, 4),
            "entry_price": round(current_price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "atr": round(current_atr, 2),
            "outcome": outcome,
            "verdict": verdict_data,
        })

    # Compute metrics
    metrics = _compute_metrics(results)

    # Print summary
    _print_summary(metrics, dry_run)

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"wyckoff_backtest_{symbol}_{timeframe}.json"
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "trades": results}, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return metrics


def _evaluate_outcome(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: int,
    direction: str,
    sl: float,
    tp: float,
    max_hold: int,
) -> Dict:
    """Walk forward from entry bar to determine if TP or SL was hit first."""
    entry_price = close[entry_idx]
    n = len(close)

    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
        if direction == "BUY":
            if low[j] <= sl:
                return {
                    "result": "loss",
                    "exit_type": "sl_hit",
                    "exit_bar": j - entry_idx,
                    "pnl_pct": round((sl - entry_price) / entry_price * 100, 4),
                }
            if high[j] >= tp:
                return {
                    "result": "win",
                    "exit_type": "tp_hit",
                    "exit_bar": j - entry_idx,
                    "pnl_pct": round((tp - entry_price) / entry_price * 100, 4),
                }
        else:  # SELL
            if high[j] >= sl:
                return {
                    "result": "loss",
                    "exit_type": "sl_hit",
                    "exit_bar": j - entry_idx,
                    "pnl_pct": round((entry_price - sl) / entry_price * 100, 4),
                }
            if low[j] <= tp:
                return {
                    "result": "win",
                    "exit_type": "tp_hit",
                    "exit_bar": j - entry_idx,
                    "pnl_pct": round((entry_price - tp) / entry_price * 100, 4),
                }

    # Time exit — neither TP nor SL hit
    exit_price = close[min(entry_idx + max_hold, n - 1)]
    if direction == "BUY":
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100

    return {
        "result": "win" if pnl > 0 else "loss",
        "exit_type": "time_exit",
        "exit_bar": max_hold,
        "pnl_pct": round(pnl, 4),
    }


def _compute_metrics(results: List[Dict]) -> Dict:
    """Compute comparison metrics: all signals vs gatekeeper-approved only."""
    if not results:
        return {"total_signals": 0}

    all_trades = results
    approved = [r for r in results if r["verdict"].get("verdict") == "APPROVE"]
    rejected = [r for r in results if r["verdict"].get("verdict") == "REJECT"]
    held = [r for r in results if r["verdict"].get("verdict") == "HOLD"]

    def _stats(trades: List[Dict]) -> Dict:
        if not trades:
            return {"count": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0}
        wins = [t for t in trades if t["outcome"]["result"] == "win"]
        losses = [t for t in trades if t["outcome"]["result"] == "loss"]
        gross_profit = sum(t["outcome"]["pnl_pct"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["outcome"]["pnl_pct"] for t in losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_pnl = sum(t["outcome"]["pnl_pct"] for t in trades) / len(trades)
        return {
            "count": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(pf, 2),
            "avg_pnl": round(avg_pnl, 4),
            "total_pnl": round(sum(t["outcome"]["pnl_pct"] for t in trades), 4),
        }

    # Phase distribution of approved trades
    phase_dist = {}
    for r in approved:
        phase = r["verdict"].get("wyckoff_phase", "unknown")
        phase_dist[phase] = phase_dist.get(phase, 0) + 1

    # Common rejection reasons
    rejection_reasons = [r["verdict"].get("reasoning", "") for r in rejected]

    # Confidence distribution
    confidences = [r["verdict"].get("confidence", 0) for r in results
                   if r["verdict"].get("confidence") is not None]

    return {
        "total_signals": len(all_trades),
        "approved": len(approved),
        "rejected": len(rejected),
        "held": len(held),
        "acceptance_rate": round(len(approved) / len(all_trades) * 100, 1) if all_trades else 0,
        "all_signals": _stats(all_trades),
        "approved_only": _stats(approved),
        "rejected_only": _stats(rejected),
        "phase_distribution": phase_dist,
        "rejection_reasons": rejection_reasons[:10],
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0,
        "confidence_std": round(np.std(confidences), 3) if confidences else 0,
    }


def _print_summary(metrics: Dict, dry_run: bool) -> None:
    """Print a human-readable backtest summary."""
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")

    if dry_run:
        print(f"Total signals found: {metrics.get('total_signals', 0)}")
        print("(Dry run — no LLM verdicts computed)")
        return

    print(f"Total Signals: {metrics['total_signals']}")
    print(f"  Approved: {metrics['approved']} ({metrics['acceptance_rate']:.0f}%)")
    print(f"  Rejected: {metrics['rejected']}")
    print(f"  Held: {metrics['held']}")

    print(f"\n--- All Signals (unfiltered) ---")
    all_s = metrics["all_signals"]
    print(f"  Count: {all_s['count']} | Win Rate: {all_s['win_rate']}% | PF: {all_s['profit_factor']} | Avg PnL: {all_s['avg_pnl']}%")

    print(f"\n--- Approved Only (gatekeeper-filtered) ---")
    app_s = metrics["approved_only"]
    print(f"  Count: {app_s['count']} | Win Rate: {app_s['win_rate']}% | PF: {app_s['profit_factor']} | Avg PnL: {app_s['avg_pnl']}%")

    print(f"\n--- Rejected Signals (would-have-been trades) ---")
    rej_s = metrics["rejected_only"]
    print(f"  Count: {rej_s['count']} | Win Rate: {rej_s['win_rate']}% | PF: {rej_s['profit_factor']} | Avg PnL: {rej_s['avg_pnl']}%")

    if metrics.get("phase_distribution"):
        print(f"\n--- Phase Distribution (approved trades) ---")
        for phase, count in sorted(metrics["phase_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {phase}: {count}")

    print(f"\n--- LLM Confidence ---")
    print(f"  Mean: {metrics['avg_confidence']} | Std: {metrics['confidence_std']}")

    print(f"\n{'='*60}")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wyckoff Gatekeeper Backtest — measure LLM filter impact"
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="D1", help="Chart timeframe")
    parser.add_argument("--strategy", default="breakout", help="Strategy name")
    parser.add_argument("--bars", type=int, default=500, help="Historical bars to load")
    parser.add_argument("--min-prob", type=float, default=0.60, help="Min signal probability")
    parser.add_argument("--max-signals", type=int, default=None, help="Limit LLM calls")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, show context only")
    parser.add_argument("--sl-atr", type=float, default=2.0, help="SL ATR multiplier")
    parser.add_argument("--tp-atr", type=float, default=3.0, help="TP ATR multiplier")
    parser.add_argument("--max-hold", type=int, default=20, help="Max bars to hold")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_gatekeeper_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        bars=args.bars,
        min_prob=args.min_prob,
        max_signals=args.max_signals,
        dry_run=args.dry_run,
        sl_atr_mult=args.sl_atr,
        tp_atr_mult=args.tp_atr,
        max_hold_bars=args.max_hold,
    )


if __name__ == "__main__":
    main()
