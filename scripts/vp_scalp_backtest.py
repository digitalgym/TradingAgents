"""VP Scalp Mode Parameter Sweep Backtest"""
import os, json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

mt5.initialize()

# Load all closed VP decisions
vp = []
for f in sorted(glob.glob("examples/trade_decisions/*.json")):
    with open(f) as fh:
        d = json.load(fh)
    if d.get("status") == "closed" and (
        "volume" in (d.get("pipeline") or "").lower()
        or "volume" in (d.get("source") or "").lower()
    ):
        vp.append(d)


def get_entry_context(d):
    symbol = d.get("symbol", "XAUUSD")
    direction = d.get("action", "BUY")
    entry = d.get("entry_price", 0) or 0
    actual_pnl = d.get("pnl_percent", 0) or 0
    created = d.get("created_at", "")
    if not entry or not created:
        return None
    try:
        entry_time = datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None

    h1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, entry_time - timedelta(days=5), entry_time + timedelta(hours=1))
    if h1_rates is None or len(h1_rates) < 30:
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
    if h1_rates is None:
        return None

    h1 = pd.DataFrame(h1_rates)
    h = h1["high"].values.astype(float)
    l = h1["low"].values.astype(float)
    c = h1["close"].values.astype(float)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

    plus_dm = np.maximum(h[1:] - h[:-1], 0)
    minus_dm = np.maximum(l[:-1] - l[1:], 0)
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0
    atr_s = pd.Series(tr).rolling(14).mean().values
    pdi = 100 * pd.Series(plus_dm).rolling(14).mean().values / (atr_s + 1e-10)
    mdi = 100 * pd.Series(minus_dm).rolling(14).mean().values / (atr_s + 1e-10)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-10)
    adx = float(pd.Series(dx).rolling(14).mean().values[-1])
    ema20 = float(pd.Series(c).ewm(span=20).mean().iloc[-1])
    ema50 = float(pd.Series(c).ewm(span=50).mean().iloc[-1])
    regime = "trending-up" if adx > 25 and ema20 > ema50 else ("trending-down" if adx > 25 else "ranging")

    is_counter = False
    if adx > 30:
        if regime == "trending-up" and direction == "SELL":
            is_counter = True
        elif regime == "trending-down" and direction == "BUY":
            is_counter = True

    bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, entry_time, entry_time + timedelta(hours=12))
    if bars is None or len(bars) < 3:
        bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, entry_time, entry_time + timedelta(hours=12))
    if bars is None or len(bars) < 2:
        return None

    bars_df = pd.DataFrame(bars)
    bars_df["time"] = pd.to_datetime(bars_df["time"], unit="s")

    return {
        "id": d["decision_id"],
        "direction": direction,
        "entry": entry,
        "actual_pnl": actual_pnl,
        "atr": atr,
        "adx": adx,
        "regime": regime,
        "is_counter": is_counter,
        "bars": bars_df,
        "entry_time": entry_time,
    }


def simulate_scalp(ctx, tp_mult, be_mult, max_hours, use_trailing=False, trail_mult=0.5):
    if ctx["is_counter"]:
        return 0.0, "blocked"

    entry = ctx["entry"]
    direction = ctx["direction"]
    atr = ctx["atr"]
    bars = ctx["bars"]
    entry_time = ctx["entry_time"]

    scalp_tp = tp_mult * atr
    scalp_be = be_mult * atr

    exit_price = None
    reason = None
    be_set = False
    best_profit = 0

    for _, bar in bars.iterrows():
        hrs = (bar["time"] - entry_time).total_seconds() / 3600
        if hrs < 0:
            continue

        if direction == "BUY":
            max_p = bar["high"] - entry
            min_p = bar["low"] - entry
            close_p = bar["close"] - entry
        else:
            max_p = entry - bar["low"]
            min_p = entry - bar["high"]
            close_p = entry - bar["close"]

        best_profit = max(best_profit, max_p)

        # TP hit
        if max_p >= scalp_tp:
            exit_price = entry + scalp_tp if direction == "BUY" else entry - scalp_tp
            reason = "tp"
            break

        # Trailing stop
        if use_trailing and best_profit >= scalp_be:
            trail_dist = trail_mult * atr
            trail_level = best_profit - trail_dist
            if close_p <= trail_level and close_p > 0:
                exit_price = bar["close"]
                reason = "trail"
                break

        # BE stop hit
        if be_set and min_p <= 0:
            exit_price = entry
            reason = "be"
            break

        # Set BE
        if not be_set and max_p >= scalp_be:
            be_set = True

        # Time limit
        if hrs >= max_hours:
            exit_price = bar["close"]
            reason = "time"
            break

    if exit_price is None:
        exit_price = bars.iloc[-1]["close"]
        reason = "end"

    if direction == "BUY":
        pnl = (exit_price - entry) / entry * 100
    else:
        pnl = (entry - exit_price) / entry * 100

    return pnl, reason


# Precompute contexts
contexts = [ctx for ctx in (get_entry_context(d) for d in vp) if ctx is not None]

# Parameter sweep
configs = [
    # (name, tp_mult, be_mult, max_hours, use_trailing, trail_mult)
    ("Baseline (actual)", None, None, None, False, None),
    ("Filter only", None, None, None, False, None),
    ("S 0.75/0.3/4h", 0.75, 0.3, 4.0, False, None),
    ("S 0.50/0.2/3h", 0.50, 0.2, 3.0, False, None),
    ("S 0.50/0.2/2h", 0.50, 0.2, 2.0, False, None),
    ("S 0.50/0.15/1.5h", 0.50, 0.15, 1.5, False, None),
    ("S 0.30/0.15/1h", 0.30, 0.15, 1.0, False, None),
    ("S 0.30/0.15/2h", 0.30, 0.15, 2.0, False, None),
    ("S 1.0/0.3/4h", 1.0, 0.3, 4.0, False, None),
    ("S 1.0/0.3/6h", 1.0, 0.3, 6.0, False, None),
    ("S 0.50/0.2/6h", 0.50, 0.2, 6.0, False, None),
    ("T 0.75/0.3/4h/0.5t", 0.75, 0.3, 4.0, True, 0.5),
    ("T 0.50/0.2/3h/0.3t", 0.50, 0.2, 3.0, True, 0.3),
    ("T 0.50/0.2/2h/0.25t", 0.50, 0.2, 2.0, True, 0.25),
    ("T 0.30/0.15/2h/0.2t", 0.30, 0.15, 2.0, True, 0.2),
    ("T 0.30/0.15/1h/0.15t", 0.30, 0.15, 1.0, True, 0.15),
    ("T 1.0/0.3/6h/0.5t", 1.0, 0.3, 6.0, True, 0.5),
    ("T 1.0/0.5/4h/0.5t", 1.0, 0.5, 4.0, True, 0.5),
    ("Wide 1.5/0.5/6h", 1.5, 0.5, 6.0, False, None),
    ("Wide T 1.5/0.5/6h/0.75t", 1.5, 0.5, 6.0, True, 0.75),
]

print(f"PARAMETER SWEEP: {len(contexts)} VP trades, {len(configs)} configs")
print("=" * 100)
header = f"{'Config':<30s} {'Total%':>8s} {'Taken':>5s} {'Wins':>4s} {'WR%':>5s} {'TP':>3s} {'BE':>3s} {'Time':>4s} {'Trail':>5s} {'Blk':>3s}"
print(header)
print("-" * 100)

results = []

for name, tp_m, be_m, max_h, use_t, trail_m in configs:
    total = 0
    taken = 0
    wins = 0
    reasons = {"tp": 0, "be": 0, "time": 0, "trail": 0, "end": 0, "blocked": 0}

    for ctx in contexts:
        if name == "Baseline (actual)":
            total += ctx["actual_pnl"]
            taken += 1
            if ctx["actual_pnl"] > 0:
                wins += 1
        elif name == "Filter only":
            if ctx["is_counter"]:
                reasons["blocked"] += 1
            else:
                total += ctx["actual_pnl"]
                taken += 1
                if ctx["actual_pnl"] > 0:
                    wins += 1
        else:
            pnl, reason = simulate_scalp(ctx, tp_m, be_m, max_h, use_t, trail_m or 0)
            if reason == "blocked":
                reasons["blocked"] += 1
            else:
                total += pnl
                taken += 1
                if pnl > 0:
                    wins += 1
                reasons[reason] = reasons.get(reason, 0) + 1

    wr = (wins / taken * 100) if taken > 0 else 0
    results.append((name, total, taken, wins, wr, reasons))
    blk = reasons.get("blocked", 0)
    tp_c = reasons.get("tp", 0)
    be_c = reasons.get("be", 0)
    time_c = reasons.get("time", 0) + reasons.get("end", 0)
    trail_c = reasons.get("trail", 0)
    print(f"{name:<30s} {total:>+7.2f}% {taken:>5d} {wins:>4d} {wr:>4.0f}% {tp_c:>3d} {be_c:>3d} {time_c:>4d} {trail_c:>5d} {blk:>3d}")

print("=" * 100)

# Find best
best = max(results[2:], key=lambda r: r[1])  # Skip baseline and filter-only
print(f"\nBEST CONFIG: {best[0]} -> {best[1]:+.2f}% ({best[3]}/{best[2]} wins, {best[4]:.0f}% WR)")

# Detail best config per-trade
print(f"\n{'=' * 90}")
print(f"DETAIL: {best[0]}")
print(f"{'=' * 90}")

best_cfg = [c for c in configs if c[0] == best[0]][0]
_, tp_m, be_m, max_h, use_t, trail_m = best_cfg

for ctx in contexts:
    pnl, reason = simulate_scalp(ctx, tp_m, be_m, max_h, use_t, trail_m or 0) if tp_m else (ctx["actual_pnl"], "actual")
    tag = "BETTER" if pnl > ctx["actual_pnl"] + 0.01 else ("WORSE" if pnl < ctx["actual_pnl"] - 0.01 else "SAME")
    if reason == "blocked":
        print(f"  {ctx['id']}: {ctx['direction']} ADX={ctx['adx']:.0f} -> BLOCKED (was {ctx['actual_pnl']:+.2f}%)")
    else:
        print(f"  {ctx['id']}: {ctx['direction']} ADX={ctx['adx']:.0f} -> {reason} pnl={pnl:+.2f}% (was {ctx['actual_pnl']:+.2f}%) [{tag}]")

mt5.shutdown()
