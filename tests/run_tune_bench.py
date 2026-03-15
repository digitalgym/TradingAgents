"""Quick benchmark: run auto-tuner for each pipeline on XAUUSD."""
import asyncio
import sys
import time

async def run_one(pipeline):
    from tradingagents.automation.auto_tuner import run_tune

    print(f"\n{'='*60}", flush=True)
    print(f"  {pipeline.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    def prog(phase, cur, tot, msg, steps=None):
        if phase == "sweeping" and cur > 0 and cur % 200 == 0:
            print(f"  sweep: {cur}/{tot}", flush=True)
        elif phase == "precompute":
            print(f"  precompute: {msg}", flush=True)
        elif phase == "loading_data":
            print(f"  load: {msg}", flush=True)

    t0 = time.time()
    r = await run_tune("XAUUSD", pipeline, bars=800, min_trades=5, progress_callback=prog)
    elapsed = time.time() - t0

    if r.get("error"):
        print(f"  ERROR: {r['error']}", flush=True)
        return

    b = r["best"]
    if not b:
        print("  No valid results", flush=True)
        return

    print(f"\n  Duration: {elapsed:.1f}s | Valid configs: {r['all_count']}", flush=True)
    print(f"  BEST ({b['timeframe']}): WR={b['win_rate']:.1f}% Sharpe={b['sharpe']:.2f} PF={b['profit_factor']:.2f} Trades={b['total_trades']}", flush=True)
    print(f"    BUY: {b['buy_trades']} ({b['buy_win_rate']:.0f}%) | SELL: {b['sell_trades']} ({b['sell_win_rate']:.0f}%)", flush=True)
    print(f"    Avg PnL: {b['avg_pnl']:.3f}% | Total PnL: {b['total_pnl']:.1f}%", flush=True)
    print(f"    Params: {b['params']}", flush=True)
    print(f"    Config updates: {r['config_updates']}", flush=True)
    print(f"  TOP 5:", flush=True)
    for i, x in enumerate(r["top_5"][:5]):
        ep = {k: v for k, v in x["params"].items() if k not in ("atr_sl_mult", "rr_ratio")}
        print(f"    #{i+1} {x['timeframe']} WR={x['win_rate']:.1f}% S={x['sharpe']:.2f} PF={x['profit_factor']:.2f} t={x['total_trades']} atr={x['params'].get('atr_sl_mult')} rr={x['params'].get('rr_ratio')} {ep}", flush=True)


async def main():
    pipelines = sys.argv[1:] if len(sys.argv) > 1 else ["breakout_quant", "range_quant", "volume_profile", "rule_based"]
    for p in pipelines:
        await run_one(p)

if __name__ == "__main__":
    asyncio.run(main())
