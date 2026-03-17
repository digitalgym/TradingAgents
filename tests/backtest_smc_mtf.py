"""
Backtest SMC MTF pipeline across timeframe pairs and parameter combinations.

Uses the auto-tuner's run_tune() function which handles:
- Loading dual-timeframe MT5 data
- Pre-computing MTF alignment signals (OTE, channels, weekend gaps, protected flips)
- Parameter sweep: hold, min_alignment, require_confirmation, require_channel, atr_sl_mult, rr_ratio
- Ranking by profit_factor * win_rate

Usage:
    python tests/backtest_smc_mtf.py [--symbol XAUUSD] [--bars 800] [--min-trades 5]
"""

import asyncio
import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_result(r, rank=None):
    """Print a result dict from run_tune. Win rates are stored as 0-100."""
    prefix = f"  #{rank}" if rank else " "
    tf = r["timeframe"]
    wr = r["win_rate"]  # Already 0-100
    sharpe = r["sharpe"]
    pf = r["profit_factor"]
    trades = r["total_trades"]
    buy_t = r["buy_trades"]
    sell_t = r["sell_trades"]
    buy_wr = r["buy_win_rate"]  # Already 0-100
    sell_wr = r["sell_win_rate"]  # Already 0-100
    params = r["params"]
    print(f"{prefix}  TF={tf:8s}  WR={wr:.1f}%  Sharpe={sharpe:.2f}  "
          f"PF={pf:.2f}  Trades={trades:3d}  "
          f"BUY={buy_t}({buy_wr:.0f}%) SELL={sell_t}({sell_wr:.0f}%)  "
          f"Params={params}")


async def main():
    parser = argparse.ArgumentParser(description="Backtest SMC MTF pipeline")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to backtest")
    parser.add_argument("--bars", type=int, default=800, help="Number of bars per timeframe")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades to qualify")
    args = parser.parse_args()

    from tradingagents.automation.auto_tuner import run_tune, get_parameter_grid, get_tunable_timeframes

    grid = get_parameter_grid("smc_mtf")
    timeframes = get_tunable_timeframes("smc_mtf")

    # Calculate total combos
    total = 1
    for v in grid.values():
        total *= len(v)
    total *= len(timeframes)

    print(f"=" * 80)
    print(f"SMC MTF Backtest: {args.symbol}")
    print(f"Timeframe pairs: {timeframes}")
    print(f"Parameter grid: {grid}")
    print(f"Total combinations: {total}")
    print(f"Min trades: {args.min_trades}")
    print(f"Bars per timeframe: {args.bars}")
    print(f"=" * 80)

    start = time.time()

    last_phase = [None]
    def progress(phase, current, total_p, message, steps=None):
        # Only print phase changes, precompute milestones, and sweep progress every 100
        if phase != last_phase[0]:
            last_phase[0] = phase
            print(f"\n  [{phase}] {message}")
        elif phase == "precompute" and current % 100 == 0:
            print(f"  [{phase}] {message}")
        elif phase == "done":
            print(f"  [{phase}] {message}")

    result = await run_tune(
        symbol=args.symbol,
        pipeline="smc_mtf",
        timeframes=timeframes,
        bars=args.bars,
        min_trades=args.min_trades,
        progress_callback=progress,
    )

    duration = time.time() - start

    if result.get("error") and not result.get("best"):
        print(f"\nERROR: {result['error']}")
        print(f"Timeframes tested: {result.get('timeframes_tested', [])}")
        print(f"Bars loaded: {result.get('bars_per_tf', {})}")
        return

    print(f"\n{'=' * 80}")
    print(f"RESULTS ({duration:.1f}s)")
    print(f"{'=' * 80}")
    print(f"Total valid configs: {result.get('all_count', 0)}")
    print(f"Timeframes tested: {result.get('timeframes_tested', [])}")
    print(f"Bars per TF: {result.get('bars_per_tf', {})}")

    best = result.get("best")
    if best:
        print(f"\n--- BEST CONFIG ---")
        print_result(best, rank=1)

    top5 = result.get("top_5", [])
    if len(top5) > 1:
        print(f"\n--- TOP 5 ---")
        for i, r in enumerate(top5):
            print_result(r, rank=i + 1)

    config_updates = result.get("config_updates", {})
    if config_updates:
        print(f"\n--- CONFIG UPDATES (to apply to automation) ---")
        for k, v in config_updates.items():
            print(f"  {k}: {v}")

    # Print per-timeframe breakdown
    if top5:
        print(f"\n--- PER-TIMEFRAME BREAKDOWN ---")
        tf_groups = {}
        for r in top5:
            tf_groups.setdefault(r["timeframe"], []).append(r)
        for tf, results in sorted(tf_groups.items()):
            print(f"\n  {tf}:")
            for r in results:
                print(f"    WR={r['win_rate']:.1f}%  PF={r['profit_factor']:.2f}  Sharpe={r['sharpe']:.2f}  "
                      f"Trades={r['total_trades']}  Params={r['params']}")


if __name__ == "__main__":
    asyncio.run(main())
