"""
Lightweight backtest utilities — no heavy dependencies.

Extracted from automation/auto_tuner.py to avoid importing
langgraph/torch/transformers via the automation package chain.
"""
from typing import Dict, Any
import numpy as np


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR for the entire series. Returns array same length as input (NaN for first `period` bars)."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def simulate_exit(
    direction: str, entry: float, sl: float, tp: float,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_bar: int, max_hold: int,
    trailing_atr_mult: float = 0.0, atr: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """Walk forward bar-by-bar to find SL/TP hit or max-hold expiry."""
    end_bar = min(entry_bar + max_hold, len(close) - 1)
    current_sl = sl
    best_price = entry

    for j in range(entry_bar + 1, end_bar + 1):
        if trailing_atr_mult > 0 and atr is not None and not np.isnan(atr[j]):
            trail_dist = atr[j] * trailing_atr_mult
            if direction == "BUY":
                if high[j] > best_price:
                    best_price = high[j]
                new_sl = best_price - trail_dist
                if new_sl > current_sl:
                    current_sl = new_sl
            else:
                if low[j] < best_price:
                    best_price = low[j]
                new_sl = best_price + trail_dist
                if new_sl < current_sl:
                    current_sl = new_sl

        if direction == "BUY":
            if low[j] <= current_sl:
                pnl = (current_sl - entry) / entry * 100
                reason = "trailing_sl" if current_sl > sl else "sl"
                return {"exit": current_sl, "pnl_pct": pnl, "reason": reason, "bars": j - entry_bar}
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {"exit": tp, "pnl_pct": pnl, "reason": "tp", "bars": j - entry_bar}
        else:
            if high[j] >= current_sl:
                pnl = (entry - current_sl) / entry * 100
                reason = "trailing_sl" if current_sl < sl else "sl"
                return {"exit": current_sl, "pnl_pct": pnl, "reason": reason, "bars": j - entry_bar}
            if low[j] <= tp:
                pnl = (entry - tp) / entry * 100
                return {"exit": tp, "pnl_pct": pnl, "reason": "tp", "bars": j - entry_bar}

    exit_p = close[end_bar]
    if direction == "BUY":
        pnl = (exit_p - entry) / entry * 100
    else:
        pnl = (entry - exit_p) / entry * 100
    return {"exit": exit_p, "pnl_pct": pnl, "reason": "expire", "bars": end_bar - entry_bar}
