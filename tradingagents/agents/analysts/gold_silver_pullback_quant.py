"""
Gold/Silver Pullback Quant Analyst

Mechanical pullback strategy for gold using silver momentum confirmation.

Clear Pullback Definition (Longs):
1. Trend intact: price > SMA_fast AND SMA_fast > SMA_slow
2. Pullback depth: 0.5-2.5 ATR from recent swing high (or 23.6%-61.8% Fib retrace)
3. Reversal candle: higher low formed + bullish candle closing above prior candle's high
4. Silver leads: silver makes new N-bar high or shows strong momentum
5. Ratio filter: Au/Ag ratio not at extremes

Sells mirror the logic in downtrends.

Confluence-scored (0-7). Signal fires when score >= min_confluence.
"""

from typing import Optional, Dict, Any
import numpy as np
from tradingagents.agents.analysts.quant_utils import create_quant_logger


def _get_gs_logger():
    """Get or create the gold/silver pullback quant prompt logger."""
    return create_quant_logger("gold_silver_pullback_prompts", "gold_silver_pullback_prompts")


def _find_swing_high(high: np.ndarray, lookback: int) -> tuple:
    """Find the most recent swing high (local max) and its index."""
    segment = high[-lookback:]
    idx = int(np.argmax(segment))
    return float(segment[idx]), len(high) - lookback + idx


def _find_swing_low(low: np.ndarray, lookback: int) -> tuple:
    """Find the most recent swing low (local min) and its index."""
    segment = low[-lookback:]
    idx = int(np.argmin(segment))
    return float(segment[idx]), len(low) - lookback + idx


def _impulse_leg_size(close: np.ndarray, high: np.ndarray, low: np.ndarray, lookback: int) -> tuple:
    """Estimate the prior impulse leg size for Fibonacci retracement.

    Returns (leg_size, swing_high, swing_low) for the most recent impulse.
    """
    seg_high = high[-lookback:]
    seg_low = low[-lookback:]
    hi_idx = int(np.argmax(seg_high))
    lo_idx = int(np.argmin(seg_low))
    swing_hi = float(seg_high[hi_idx])
    swing_lo = float(seg_low[lo_idx])
    return swing_hi - swing_lo, swing_hi, swing_lo


def analyze_gold_silver_pullback(
    gold_high: np.ndarray,
    gold_low: np.ndarray,
    gold_close: np.ndarray,
    gold_open: np.ndarray,
    silver_close: np.ndarray,
    silver_high: "np.ndarray | None" = None,
    sma_fast: int = 50,
    sma_slow: int = 200,
    pullback_atr_mult: float = 1.0,
    silver_roc_period: int = 5,
    ratio_z_filter: float = 2.0,
    pullback_lookback: int = 20,
    min_confluence: int = 4,
) -> Dict[str, Any]:
    """
    Analyze gold and silver data for pullback entry conditions.

    Confluence factors (1 point each, max 7):
      1. Trend: price > SMA_fast AND SMA_fast > SMA_slow
      2. Pullback depth: retrace >= pullback_atr_mult * ATR from swing high
      3. Fib zone: retrace within 23.6%-61.8% of impulse leg
      4. Reversal candle: higher low + bullish close above prior candle high
      5. Silver confirmation: positive ROC or new N-bar high
      6. Ratio filter: Au/Ag ratio Z-score within bounds
      7. Structure: higher low confirmed (low > prior swing low)
    """
    min_bars = max(sma_slow + 10, 260)
    empty = {
        "direction": None,
        "confluence": 0,
        "buy_score": 0,
        "sell_score": 0,
        "signal": False,
        "uptrend": False,
        "downtrend": False,
        "pullback_depth_atr": 0.0,
        "fib_retrace_pct": 0.0,
        "silver_roc": 0.0,
        "ratio_z": 0.0,
        "ratio_val": 0.0,
        "atr": 0.0,
        "swing_low": 0.0,
        "swing_high": 0.0,
        "sma_fast_val": None,
        "sma_slow_val": None,
    }
    if len(gold_close) < min_bars or len(silver_close) < min_bars:
        return empty

    n = len(gold_close)

    # --- ATR (14-period) ---
    tr = np.empty(n)
    tr[0] = gold_high[0] - gold_low[0]
    for i in range(1, n):
        tr[i] = max(
            gold_high[i] - gold_low[i],
            abs(gold_high[i] - gold_close[i - 1]),
            abs(gold_low[i] - gold_close[i - 1]),
        )
    atr_period = 14
    atr_arr = np.full(n, np.nan)
    if n >= atr_period:
        atr_arr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr_arr[i] = (atr_arr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    current_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    if current_atr == 0:
        return empty

    current_price = float(gold_close[-1])

    # --- Trend filter (double condition: price > fast AND fast > slow) ---
    sma_f = float(np.mean(gold_close[-sma_fast:])) if n >= sma_fast else 0.0
    sma_s = float(np.mean(gold_close[-sma_slow:])) if n >= sma_slow else 0.0
    uptrend = current_price > sma_f and sma_f > sma_s
    downtrend = current_price < sma_f and sma_f < sma_s

    # --- Swing points ---
    lb = min(pullback_lookback, n - 1)
    swing_hi, _ = _find_swing_high(gold_high, lb)
    swing_lo, _ = _find_swing_low(gold_low, lb)

    # --- Pullback depth (ATR-based) ---
    buy_depth_atr = (swing_hi - current_price) / current_atr
    sell_depth_atr = (current_price - swing_lo) / current_atr
    is_pullback_buy = buy_depth_atr >= pullback_atr_mult
    is_pullback_sell = sell_depth_atr >= pullback_atr_mult

    # --- Fibonacci retracement (23.6%-61.8% of impulse leg) ---
    impulse_size, imp_hi, imp_lo = _impulse_leg_size(gold_close, gold_high, gold_low, lb)
    fib_retrace_buy = 0.0
    fib_retrace_sell = 0.0
    fib_ok_buy = False
    fib_ok_sell = False
    if impulse_size > 0:
        fib_retrace_buy = (swing_hi - current_price) / impulse_size
        fib_retrace_sell = (current_price - swing_lo) / impulse_size
        fib_ok_buy = 0.236 <= fib_retrace_buy <= 0.618
        fib_ok_sell = 0.236 <= fib_retrace_sell <= 0.618

    # --- Reversal candle ---
    # Buy: bullish candle (close > open) that closes above prior candle's high
    candle_bullish = float(gold_close[-1]) > float(gold_open[-1])
    closes_above_prev_high = float(gold_close[-1]) > float(gold_high[-2]) if n > 1 else False
    reversal_buy = candle_bullish and closes_above_prev_high

    # Sell: bearish candle that closes below prior candle's low
    candle_bearish = float(gold_close[-1]) < float(gold_open[-1])
    closes_below_prev_low = float(gold_close[-1]) < float(gold_low[-2]) if n > 1 else False
    reversal_sell = candle_bearish and closes_below_prev_low

    # --- Silver confirmation ---
    silver_roc = 0.0
    silver_confirms_buy = False
    silver_confirms_sell = False
    if len(silver_close) > silver_roc_period:
        roc = (float(silver_close[-1]) - float(silver_close[-1 - silver_roc_period])) / float(silver_close[-1 - silver_roc_period])
        silver_roc = roc
        silver_confirms_buy = roc > 0

        # Silver breakout: new N-bar high
        if silver_high is not None and len(silver_high) > silver_roc_period:
            silver_new_high = float(silver_high[-1]) >= float(np.max(silver_high[-silver_roc_period - 1:-1]))
            silver_confirms_buy = silver_confirms_buy or silver_new_high
        else:
            # Fallback: silver close is highest in N bars
            silver_new_high = float(silver_close[-1]) >= float(np.max(silver_close[-silver_roc_period - 1:-1]))
            silver_confirms_buy = silver_confirms_buy or silver_new_high

        silver_confirms_sell = roc < 0

    # --- Ratio filter ---
    ratio = gold_close / silver_close
    ratio_val = float(ratio[-1])
    ratio_z = 0.0
    ratio_ok = True
    if len(ratio) >= 252:
        ratio_mean = float(np.mean(ratio[-252:]))
        ratio_std = float(np.std(ratio[-252:]))
        if ratio_std > 0:
            ratio_z = (ratio_val - ratio_mean) / ratio_std
            ratio_ok = abs(ratio_z) < ratio_z_filter

    # --- Structure confirmation ---
    # Higher low: current low > lowest low in last 5 bars (excluding current)
    higher_low = False
    lower_high = False
    if n > 5:
        higher_low = float(gold_low[-1]) > float(np.min(gold_low[-5:-1]))
        lower_high = float(gold_high[-1]) < float(np.max(gold_high[-5:-1]))

    # --- Confluence scoring (7 factors) ---
    buy_score = sum([
        uptrend,              # 1. Trend
        is_pullback_buy,      # 2. Pullback depth
        fib_ok_buy,           # 3. Fib zone
        reversal_buy,         # 4. Reversal candle
        silver_confirms_buy,  # 5. Silver
        ratio_ok,             # 6. Ratio
        higher_low,           # 7. Structure
    ])
    sell_score = sum([
        downtrend,            # 1. Trend
        is_pullback_sell,     # 2. Pullback depth
        fib_ok_sell,          # 3. Fib zone
        reversal_sell,        # 4. Reversal candle
        silver_confirms_sell, # 5. Silver
        ratio_ok,             # 6. Ratio
        lower_high,           # 7. Structure
    ])

    # Pick stronger direction
    direction = None
    confluence = 0
    signal = False

    if buy_score >= sell_score and buy_score >= min_confluence:
        direction = "BUY"
        confluence = buy_score
        signal = True
    elif sell_score > buy_score and sell_score >= min_confluence:
        direction = "SELL"
        confluence = sell_score
        signal = True

    return {
        "direction": direction,
        "confluence": confluence,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "signal": signal,
        "uptrend": uptrend,
        "downtrend": downtrend,
        "is_pullback_buy": is_pullback_buy,
        "is_pullback_sell": is_pullback_sell,
        "pullback_depth_atr": buy_depth_atr if direction == "BUY" else sell_depth_atr,
        "fib_retrace_pct": fib_retrace_buy if direction == "BUY" else fib_retrace_sell,
        "reversal_buy": reversal_buy,
        "reversal_sell": reversal_sell,
        "silver_confirms_buy": silver_confirms_buy,
        "silver_confirms_sell": silver_confirms_sell,
        "silver_roc": silver_roc,
        "ratio_ok": ratio_ok,
        "ratio_z": ratio_z,
        "ratio_val": ratio_val,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "atr": current_atr,
        "swing_low": swing_lo,
        "swing_high": swing_hi,
        "sma_fast_val": sma_f,
        "sma_slow_val": sma_s,
    }


def analyze_gold_silver_pullback_mtf(
    d1_gold_close: np.ndarray,
    d1_silver_close: np.ndarray,
    h4_gold_high: np.ndarray,
    h4_gold_low: np.ndarray,
    h4_gold_close: np.ndarray,
    h4_gold_open: np.ndarray,
    h4_silver_close: np.ndarray,
    sma_fast: int = 50,
    sma_slow: int = 200,
    pullback_atr_mult: float = 1.0,
    silver_roc_period: int = 5,
    ratio_z_filter: float = 2.0,
    pullback_lookback: int = 20,
    min_confluence: int = 4,
) -> Dict[str, Any]:
    """
    Multi-timeframe gold/silver pullback analysis.

    D1 provides (slow conditions):
      1. Trend filter (SMA fast > slow on D1)
      6. Ratio filter (Au/Ag Z-score on D1)

    H4 provides (entry triggers):
      2. Pullback depth (ATR-based on H4)
      3. Fib zone (23.6%-61.8% on H4)
      4. Reversal candle (H4 candle pattern)
      5. Silver confirmation (H4 ROC / breakout)
      7. Structure (higher low / lower high on H4)

    Returns same format as single-TF version.
    """
    empty = {
        "direction": None, "confluence": 0, "buy_score": 0, "sell_score": 0,
        "signal": False, "uptrend": False, "downtrend": False,
        "pullback_depth_atr": 0.0, "fib_retrace_pct": 0.0,
        "silver_roc": 0.0, "ratio_z": 0.0, "ratio_val": 0.0,
        "atr": 0.0, "swing_low": 0.0, "swing_high": 0.0,
        "sma_fast_val": None, "sma_slow_val": None,
    }

    # Need enough D1 bars for SMAs and ratio
    if len(d1_gold_close) < max(sma_slow, 252) or len(h4_gold_close) < 50:
        return empty

    # ===== D1 CONDITIONS =====

    # Trend filter on D1
    d1_sma_f = float(np.mean(d1_gold_close[-sma_fast:]))
    d1_sma_s = float(np.mean(d1_gold_close[-sma_slow:]))
    d1_price = float(d1_gold_close[-1])
    uptrend = d1_price > d1_sma_f and d1_sma_f > d1_sma_s
    downtrend = d1_price < d1_sma_f and d1_sma_f < d1_sma_s

    # Ratio filter on D1
    d1_ratio = d1_gold_close / d1_silver_close
    ratio_val = float(d1_ratio[-1])
    ratio_z = 0.0
    ratio_ok = True
    if len(d1_ratio) >= 252:
        ratio_mean = float(np.mean(d1_ratio[-252:]))
        ratio_std = float(np.std(d1_ratio[-252:]))
        if ratio_std > 0:
            ratio_z = (ratio_val - ratio_mean) / ratio_std
            ratio_ok = abs(ratio_z) < ratio_z_filter

    # ===== H4 CONDITIONS =====
    n = len(h4_gold_close)

    # ATR on H4
    tr = np.empty(n)
    tr[0] = h4_gold_high[0] - h4_gold_low[0]
    for i in range(1, n):
        tr[i] = max(
            h4_gold_high[i] - h4_gold_low[i],
            abs(h4_gold_high[i] - h4_gold_close[i - 1]),
            abs(h4_gold_low[i] - h4_gold_close[i - 1]),
        )
    atr_period = 14
    atr_arr = np.full(n, np.nan)
    if n >= atr_period:
        atr_arr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr_arr[i] = (atr_arr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    h4_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    if h4_atr == 0:
        return empty

    h4_price = float(h4_gold_close[-1])
    lb = min(pullback_lookback, n - 1)

    # Swing points on H4
    swing_hi, _ = _find_swing_high(h4_gold_high, lb)
    swing_lo, _ = _find_swing_low(h4_gold_low, lb)

    # Pullback depth on H4
    buy_depth = (swing_hi - h4_price) / h4_atr
    sell_depth = (h4_price - swing_lo) / h4_atr
    is_pullback_buy = buy_depth >= pullback_atr_mult
    is_pullback_sell = sell_depth >= pullback_atr_mult

    # Fib retracement on H4
    impulse_size, _, _ = _impulse_leg_size(h4_gold_close, h4_gold_high, h4_gold_low, lb)
    fib_buy = (swing_hi - h4_price) / impulse_size if impulse_size > 0 else 0.0
    fib_sell = (h4_price - swing_lo) / impulse_size if impulse_size > 0 else 0.0
    fib_ok_buy = 0.236 <= fib_buy <= 0.618
    fib_ok_sell = 0.236 <= fib_sell <= 0.618

    # Reversal candle on H4
    candle_bullish = h4_gold_close[-1] > h4_gold_open[-1]
    closes_above_prev = float(h4_gold_close[-1]) > float(h4_gold_high[-2]) if n > 1 else False
    reversal_buy = candle_bullish and closes_above_prev

    candle_bearish = h4_gold_close[-1] < h4_gold_open[-1]
    closes_below_prev = float(h4_gold_close[-1]) < float(h4_gold_low[-2]) if n > 1 else False
    reversal_sell = candle_bearish and closes_below_prev

    # Silver confirmation on H4
    silver_roc = 0.0
    silver_buy = False
    silver_sell = False
    if len(h4_silver_close) > silver_roc_period:
        roc = (float(h4_silver_close[-1]) - float(h4_silver_close[-1 - silver_roc_period])) / float(h4_silver_close[-1 - silver_roc_period])
        silver_roc = roc
        silver_buy = roc > 0
        # Silver breakout: new N-bar high
        silver_new_hi = float(h4_silver_close[-1]) >= float(np.max(h4_silver_close[-silver_roc_period - 1:-1]))
        silver_buy = silver_buy or silver_new_hi
        silver_sell = roc < 0

    # Structure on H4
    higher_low = float(h4_gold_low[-1]) > float(np.min(h4_gold_low[-5:-1])) if n > 5 else False
    lower_high = float(h4_gold_high[-1]) < float(np.max(h4_gold_high[-5:-1])) if n > 5 else False

    # ===== SCORING =====
    buy_score = sum([
        uptrend,          # 1. D1 trend
        is_pullback_buy,  # 2. H4 pullback
        fib_ok_buy,       # 3. H4 fib zone
        reversal_buy,     # 4. H4 reversal candle
        silver_buy,       # 5. H4 silver
        ratio_ok,         # 6. D1 ratio
        higher_low,       # 7. H4 structure
    ])
    sell_score = sum([
        downtrend,         # 1. D1 trend
        is_pullback_sell,  # 2. H4 pullback
        fib_ok_sell,       # 3. H4 fib zone
        reversal_sell,     # 4. H4 reversal candle
        silver_sell,       # 5. H4 silver
        ratio_ok,          # 6. D1 ratio
        lower_high,        # 7. H4 structure
    ])

    direction = None
    confluence = 0
    signal = False
    if buy_score >= sell_score and buy_score >= min_confluence:
        direction = "BUY"
        confluence = buy_score
        signal = True
    elif sell_score > buy_score and sell_score >= min_confluence:
        direction = "SELL"
        confluence = sell_score
        signal = True

    return {
        "direction": direction,
        "confluence": confluence,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "signal": signal,
        "uptrend": uptrend,
        "downtrend": downtrend,
        "pullback_depth_atr": buy_depth if direction == "BUY" else sell_depth,
        "fib_retrace_pct": fib_buy if direction == "BUY" else fib_sell,
        "silver_roc": silver_roc,
        "ratio_ok": ratio_ok,
        "ratio_z": ratio_z,
        "ratio_val": ratio_val,
        "atr": h4_atr,
        "swing_low": swing_lo,
        "swing_high": swing_hi,
        "sma_fast_val": d1_sma_f,
        "sma_slow_val": d1_sma_s,
    }
