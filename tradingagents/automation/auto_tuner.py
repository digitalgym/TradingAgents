"""
Auto-Tuner: Per-symbol parameter optimization for automation pipelines.

Runs backtests using pure signal generation (no LLM) to find optimal parameters
for a given symbol/pipeline combination. Can be triggered manually or periodically.

Usage:
    from tradingagents.automation.auto_tuner import run_tune
    result = await run_tune("XAUUSD", "range_quant", progress_callback=my_callback)
    # result["best"] = TuneResult with optimal params
    # result["config_updates"] = dict to apply to automation config
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import pandas as pd

from tradingagents.agents.analysts.range_quant import analyze_range
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class TuneResult:
    """Result from a single parameter combination backtest."""
    strategy: str
    timeframe: str
    params: Dict[str, Any]
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    sharpe: float
    profit_factor: float
    buy_trades: int
    sell_trades: int
    buy_win_rate: float
    sell_win_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# MT5 data loading
# ------------------------------------------------------------------

def load_mt5_data(symbol: str, timeframe: str, bars: int = 800) -> pd.DataFrame:
    """Fetch historical OHLCV data from MT5 for backtesting."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")

    tf_map = {
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        mt5.shutdown()
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned from MT5 for {symbol} {timeframe}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "date", "tick_volume": "volume"}, inplace=True)
    return df


# ------------------------------------------------------------------
# Parameter grids
# ------------------------------------------------------------------

def get_parameter_grid(pipeline: str) -> Dict[str, List[Any]]:
    """Return parameter grid for a given pipeline type.

    All pipelines include atr_sl_mult and rr_ratio for SL/TP tuning.
    """
    # Shared exit params — tested across all pipelines
    exit_params = {
        "atr_sl_mult": [1.0, 1.5, 2.0, 2.5],
        "rr_ratio": [1.5, 2.0, 3.0],
    }

    if pipeline == "range_quant":
        return {
            "lookback": [20, 25, 30, 40],
            "mr_threshold": [45, 55, 65],
            "hold": [3, 5, 7, 10],
            "bias_filter": [True, False],
            **exit_params,
        }
    elif pipeline == "breakout_quant":
        return {
            "lookback": [15, 20, 25, 30],
            "squeeze_threshold": [60, 70, 80],
            "hold": [3, 5, 7, 10],
            "volume_confirm": [True, False],
            **exit_params,
        }
    elif pipeline in ("rule_based", "smc_quant_basic", "smc_quant"):
        return {
            "hold": [3, 5, 7, 10],
            "ob_proximity_pct": [0.3, 0.5, 0.8, 1.0],
            **exit_params,
        }
    elif pipeline == "volume_profile":
        return {
            "lookback": [30, 50, 75, 100],
            "hold": [3, 5, 7, 10],
            **exit_params,
        }
    elif pipeline == "smc_mtf":
        return {
            "hold": [3, 5, 7, 10],
            "min_alignment": [40, 50, 60],
            "require_confirmation": [True, False],
            "require_channel": [True, False],
            **exit_params,
        }
    elif pipeline in ("xgboost", "xgboost_ensemble"):
        return {
            "signal_threshold": [0.55, 0.60, 0.65, 0.70],
            "hold": [5, 10, 15, 20],
            **exit_params,
        }
    else:
        return {}


def get_tunable_timeframes(pipeline: str) -> List[str]:
    """Return timeframes to test for a pipeline.

    For smc_mtf, returns TF pairs as 'HTF+LTF' strings (e.g., 'D1+H4').
    """
    if pipeline == "smc_mtf":
        return ["D1+H4", "D1+H1", "H4+H1"]
    elif pipeline in ("rule_based", "smc_quant_basic", "smc_quant"):
        return ["D1", "H4", "H1"]
    elif pipeline in ("range_quant", "breakout_quant", "volume_profile"):
        return ["D1", "H4", "H1"]
    elif pipeline in ("xgboost", "xgboost_ensemble"):
        return ["D1", "H4"]
    else:
        return ["D1", "H4", "H1"]


def scale_hold_days(hold_d1: int, timeframe: str) -> int:
    """Scale hold period from D1 bars to equivalent in other timeframes."""
    multipliers = {"D1": 1, "H4": 6, "H1": 24, "M30": 48, "M15": 96}
    return hold_d1 * multipliers.get(timeframe, 1)


def _count_grid_combos(grid: Dict[str, List], timeframes: List[str]) -> int:
    """Count total parameter combinations across timeframes."""
    combos = 1
    for values in grid.values():
        combos *= len(values)
    return combos * len(timeframes)


# ------------------------------------------------------------------
# ATR calculation & SL/TP exit simulation
# ------------------------------------------------------------------

def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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


def _simulate_exit(
    direction: str, entry: float, sl: float, tp: float,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_bar: int, max_hold: int,
) -> Dict[str, Any]:
    """Walk forward bar-by-bar to find SL/TP hit or max-hold expiry.

    Returns dict with exit_price, pnl_pct, exit_reason, bars_held.
    """
    end_bar = min(entry_bar + max_hold, len(close) - 1)
    for j in range(entry_bar + 1, end_bar + 1):
        if direction == "BUY":
            # Check SL first (conservative: assume worst case hit first)
            if low[j] <= sl:
                pnl = (sl - entry) / entry * 100
                return {"exit": sl, "pnl_pct": pnl, "reason": "sl", "bars": j - entry_bar}
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {"exit": tp, "pnl_pct": pnl, "reason": "tp", "bars": j - entry_bar}
        else:  # SELL
            if high[j] >= sl:
                pnl = (entry - sl) / entry * 100
                return {"exit": sl, "pnl_pct": pnl, "reason": "sl", "bars": j - entry_bar}
            if low[j] <= tp:
                pnl = (entry - tp) / entry * 100
                return {"exit": tp, "pnl_pct": pnl, "reason": "tp", "bars": j - entry_bar}

    # Max hold expiry
    exit_p = close[end_bar]
    if direction == "BUY":
        pnl = (exit_p - entry) / entry * 100
    else:
        pnl = (entry - exit_p) / entry * 100
    return {"exit": exit_p, "pnl_pct": pnl, "reason": "expire", "bars": end_bar - entry_bar}


# ------------------------------------------------------------------
# Backtest functions (extracted from tests/backtest_all_pipelines.py)
# ------------------------------------------------------------------

def _compute_stats(trades: List[Dict], strategy: str, timeframe: str, params: Dict) -> TuneResult:
    """Compute backtest statistics with buy/sell split."""
    if not trades:
        return TuneResult(
            strategy=strategy, timeframe=timeframe, params=params,
            total_trades=0, winners=0, losers=0, win_rate=0,
            avg_pnl=0, total_pnl=0, sharpe=0, profit_factor=0,
            buy_trades=0, sell_trades=0, buy_win_rate=0, sell_win_rate=0,
        )

    pnls = [t["pnl_pct"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252 / max(1, len(pnls))) if std_pnl > 0 else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    buys = [t for t in trades if t["direction"] == "BUY"]
    sells = [t for t in trades if t["direction"] == "SELL"]
    buy_wins = [t for t in buys if t["pnl_pct"] > 0]
    sell_wins = [t for t in sells if t["pnl_pct"] > 0]

    return TuneResult(
        strategy=strategy, timeframe=timeframe, params=params,
        total_trades=len(trades), winners=len(winners), losers=len(losers),
        win_rate=len(winners) / len(trades) * 100,
        avg_pnl=avg_pnl, total_pnl=float(np.sum(pnls)),
        sharpe=sharpe, profit_factor=profit_factor,
        buy_trades=len(buys), sell_trades=len(sells),
        buy_win_rate=len(buy_wins) / len(buys) * 100 if buys else 0,
        sell_win_rate=len(sell_wins) / len(sells) * 100 if sells else 0,
    )


def _precompute_range_signals(df, lookback=25):
    """Pre-compute range analysis signals once per lookback.

    Returns list of dicts (one per bar from min_bars onward) with range state.
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    atr = _compute_atr(high, low, close)
    signals = []
    min_bars = max(lookback + 10, 15)
    for i in range(min_bars, len(close)):
        if np.isnan(atr[i]):
            signals.append(None)
            continue
        result = analyze_range(high[:i + 1], low[:i + 1], close[:i + 1], lookback=lookback)
        signals.append({
            "bar": i,
            "price": close[i],
            "atr": atr[i],
            "is_ranging": result["is_ranging"],
            "mr_score": result["mean_reversion_score"],
            "position": result["price_position"],
            "bias": result.get("structural_bias", "neutral"),
        })
    return signals


def _backtest_range_quant_from_cache(signals, df, hold_days=5, mr_threshold=55,
                                      use_bias_filter=True, atr_sl_mult=1.5, rr_ratio=2.0):
    """Backtest range_quant using pre-computed signals (fast sweep)."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trades = []
    for sig in signals:
        if sig is None:
            continue
        if not sig["is_ranging"] or sig["mr_score"] < mr_threshold:
            continue
        i = sig["bar"]
        if i >= len(close) - 1:
            continue
        entry = sig["price"]
        sl_dist = sig["atr"] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio
        pos, bias = sig["position"], sig["bias"]
        if pos == "discount":
            if use_bias_filter and bias == "bearish":
                continue
            sl, tp = entry - sl_dist, entry + tp_dist
            exit_info = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "BUY", "entry": entry, **exit_info})
        elif pos == "premium":
            if use_bias_filter and bias == "bullish":
                continue
            sl, tp = entry + sl_dist, entry - tp_dist
            exit_info = _simulate_exit("SELL", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "SELL", "entry": entry, **exit_info})
    return trades


def _precompute_bb_widths(close, lookback):
    n = len(close)
    widths = np.full(n, np.nan)
    for i in range(lookback, n):
        window = close[i - lookback:i]
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        widths[i] = (4 * std / sma) * 100 if sma > 0 else 0
    return widths


def _precompute_breakout_signals(df, lookback=20):
    """Pre-compute breakout/squeeze signals once per lookback, including volume analysis."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    volume = df["tick_volume"].values if "tick_volume" in df.columns else (
        df["volume"].values if "volume" in df.columns else None
    )
    atr = _compute_atr(high, low, close)
    bb_widths = _precompute_bb_widths(close, lookback)
    signals = []
    min_bars = max(lookback * 2, 50)

    for i in range(min_bars, len(close)):
        if np.isnan(atr[i]):
            signals.append(None)
            continue

        # Range boundaries (exclude current candle for breakout detection)
        rh = high[i - lookback + 1:i]
        rl = low[i - lookback + 1:i]
        range_high, range_low = float(np.max(rh)), float(np.min(rl))
        mid = (range_high + range_low) / 2
        range_pct = ((range_high - range_low) / mid) * 100 if mid > 0 else 0

        cw = bb_widths[i]
        if np.isnan(cw):
            signals.append(None)
            continue
        hist = bb_widths[lookback:i]
        hist = hist[~np.isnan(hist)]
        squeeze_str = float((np.sum(hist > cw) / len(hist)) * 100) if len(hist) > 0 else 50.0

        half = lookback // 2
        rh_full = high[i - lookback + 1:i + 1]
        rl_full = low[i - lookback + 1:i + 1]
        hl = float(np.min(rl_full[:half])) < float(np.min(rl_full[half:]))
        lh = float(np.max(rh_full[:half])) > float(np.max(rh_full[half:]))
        if hl and not lh:
            bias = "bullish"
        elif lh and not hl:
            bias = "bearish"
        else:
            bias = "neutral"

        # Volume analysis
        vol_contracting = False
        breakout_detected = False
        breakout_dir = None
        vol_surge = False
        breakout_confirmed = False

        if volume is not None:
            recent_vol = volume[i - lookback + 1:i]  # exclude current candle
            avg_vol_in_range = float(np.mean(recent_vol)) if len(recent_vol) > 0 else 0
            current_vol = float(volume[i])

            # Volume contraction vs prior period
            if i >= lookback * 2:
                prior_vol = volume[i - lookback * 2:i - lookback]
                avg_vol_before = float(np.mean(prior_vol))
                vol_contracting = avg_vol_in_range < avg_vol_before * 0.8 if avg_vol_before > 0 else False

            # Breakout detection (close-based)
            if close[i] > range_high:
                breakout_detected = True
                breakout_dir = "up"
            elif close[i] < range_low:
                breakout_detected = True
                breakout_dir = "down"

            # Volume surge on breakout (1.5x average)
            if avg_vol_in_range > 0:
                vol_surge = current_vol >= avg_vol_in_range * 1.5

            breakout_confirmed = breakout_detected and vol_surge

        signals.append({
            "bar": i, "price": close[i], "atr": atr[i],
            "range_high": range_high, "range_low": range_low,
            "range_pct": range_pct, "squeeze_str": squeeze_str, "bias": bias,
            "vol_contracting": vol_contracting,
            "breakout_detected": breakout_detected, "breakout_dir": breakout_dir,
            "vol_surge": vol_surge, "breakout_confirmed": breakout_confirmed,
        })
    return signals


def _backtest_breakout_from_cache(signals, df, hold_days=5, squeeze_threshold=70,
                                   atr_sl_mult=1.5, rr_ratio=2.0, volume_confirm=True):
    """Backtest breakout_quant using pre-computed signals (fast sweep).

    Two entry modes:
    - volume_confirm=True: Only enter on confirmed breakouts (close outside range + 1.5x volume surge)
    - volume_confirm=False: Enter on squeeze + bias alone (original behavior, no volume needed)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trades = []
    for sig in signals:
        if sig is None:
            continue
        # Base filter: consolidation + squeeze + directional bias
        if not (sig["range_pct"] < 3.0 and sig["squeeze_str"] > 60 and sig["bias"] != "neutral"):
            continue
        if sig["squeeze_str"] < squeeze_threshold:
            continue

        i = sig["bar"]
        if i >= len(close) - 1:
            continue

        if volume_confirm:
            # Volume-confirmed breakout: must have breakout_confirmed=True
            if not sig.get("breakout_confirmed"):
                continue
            # Direction from actual breakout, not just structure bias
            direction = "BUY" if sig["breakout_dir"] == "up" else "SELL"
        else:
            # Original mode: enter on squeeze + bias, no volume needed
            direction = "BUY" if sig["bias"] == "bullish" else "SELL"

        entry = sig["price"]
        sl_dist = sig["atr"] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio
        if direction == "BUY":
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist
        exit_info = _simulate_exit(direction, entry, sl, tp, high, low, close, i, hold_days)
        trades.append({"bar": i, "direction": direction, "entry": entry, **exit_info})
    return trades


def _precompute_smc_signals(df, lookback=100, progress_callback=None):
    analyzer = SmartMoneyAnalyzer()
    signals = []
    min_bars = max(lookback, 100)
    total = len(df) - min_bars

    for i in range(min_bars, len(df)):
        window = df.iloc[max(0, i - lookback):i + 1].copy()
        current_price = window.iloc[-1]["close"]

        try:
            smc = analyzer.analyze_full_smc(
                window, current_price,
                include_equal_levels=False, include_breakers=False,
                include_ote=False, include_sweeps=False,
                include_inducements=False, include_rejections=False,
                include_turtle_soup=False,
            )
        except Exception:
            signals.append(None)
            continue

        bias = smc.get("bias", "neutral")
        bullish_ob_dist = float('inf')
        bearish_ob_dist = float('inf')
        for ob in smc["order_blocks"]["bullish"]:
            if not ob.mitigated:
                bullish_ob_dist = min(bullish_ob_dist, abs(current_price - ob.top) / current_price * 100)
        for ob in smc["order_blocks"]["bearish"]:
            if not ob.mitigated:
                bearish_ob_dist = min(bearish_ob_dist, abs(current_price - ob.bottom) / current_price * 100)

        bullish_fvg_near = any(
            not fvg.mitigated and abs(current_price - fvg.top) / current_price * 100 < 1.0
            for fvg in smc["fair_value_gaps"]["bullish"]
        )
        bearish_fvg_near = any(
            not fvg.mitigated and abs(current_price - fvg.bottom) / current_price * 100 < 1.0
            for fvg in smc["fair_value_gaps"]["bearish"]
        )

        pd_zone = smc.get("premium_discount", {})
        zone = pd_zone.get("zone", "neutral") if isinstance(pd_zone, dict) else "neutral"

        signals.append({
            "bar": i, "price": current_price, "bias": bias,
            "bullish_ob_dist": bullish_ob_dist, "bearish_ob_dist": bearish_ob_dist,
            "bullish_fvg_near": bullish_fvg_near, "bearish_fvg_near": bearish_fvg_near,
            "pd_zone": zone,
        })

        done = i - min_bars + 1
        if progress_callback and done % 10 == 0:
            progress_callback("smc_precompute", done, total, f"SMC analysis: {done}/{total} bars")

    return signals


def _backtest_smc_rule_based(signals, df, hold_days=5, ob_proximity_pct=0.5,
                             atr_sl_mult=1.5, rr_ratio=2.0):
    trades = []
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    atr = _compute_atr(high, low, close)
    for sig in signals:
        if sig is None:
            continue
        i = sig["bar"]
        if i >= len(close) - 1 or np.isnan(atr[i]):
            continue
        bias = sig["bias"]
        if bias == "neutral":
            continue
        entry = sig["price"]
        sl_dist = atr[i] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio
        if bias == "bullish" and sig["bullish_ob_dist"] < ob_proximity_pct:
            sl, tp = entry - sl_dist, entry + tp_dist
            exit_info = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "BUY", "entry": entry, **exit_info})
        elif bias == "bearish" and sig["bearish_ob_dist"] < ob_proximity_pct:
            sl, tp = entry + sl_dist, entry - tp_dist
            exit_info = _simulate_exit("SELL", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "SELL", "entry": entry, **exit_info})
    return trades


def _backtest_smc_confluence(signals, df, hold_days=5, min_confluence=2,
                             atr_sl_mult=1.5, rr_ratio=2.0):
    trades = []
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    atr = _compute_atr(high, low, close)
    for sig in signals:
        if sig is None:
            continue
        i = sig["bar"]
        if i >= len(close) - 1 or np.isnan(atr[i]):
            continue
        bias = sig["bias"]
        if bias == "neutral":
            continue
        entry = sig["price"]
        sl_dist = atr[i] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio
        confluence = 0
        if bias == "bullish":
            if sig["bullish_ob_dist"] < 0.5:
                confluence += 1
            if sig["bullish_fvg_near"]:
                confluence += 1
            if sig["pd_zone"] == "discount":
                confluence += 1
            if confluence >= min_confluence:
                sl, tp = entry - sl_dist, entry + tp_dist
                exit_info = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold_days)
                trades.append({"bar": i, "direction": "BUY", "entry": entry, **exit_info, "confluence": confluence})
        elif bias == "bearish":
            if sig["bearish_ob_dist"] < 0.5:
                confluence += 1
            if sig["bearish_fvg_near"]:
                confluence += 1
            if sig["pd_zone"] == "premium":
                confluence += 1
            if confluence >= min_confluence:
                sl, tp = entry + sl_dist, entry - tp_dist
                exit_info = _simulate_exit("SELL", entry, sl, tp, high, low, close, i, hold_days)
                trades.append({"bar": i, "direction": "SELL", "entry": entry, **exit_info, "confluence": confluence})
    return trades


def _precompute_vp_signals(df, lookback=50):
    """Pre-compute volume profile signals once per lookback."""
    vp = VolumeProfileAnalyzer()
    high_arr, low_arr, close_arr = df["high"].values, df["low"].values, df["close"].values
    atr = _compute_atr(high_arr, low_arr, close_arr)
    signals = []
    min_bars = max(lookback, 50)
    for i in range(min_bars, len(df)):
        if np.isnan(atr[i]):
            signals.append(None)
            continue
        window = df.iloc[max(0, i - lookback):i + 1].copy()
        current_price = window.iloc[-1]["close"]
        try:
            profile = vp.calculate_volume_profile(window, num_bins=30, lookback=lookback)
        except Exception:
            signals.append(None)
            continue
        if profile.value_area_high == 0 or profile.value_area_low == 0:
            signals.append(None)
            continue
        signals.append({
            "bar": i, "price": current_price, "atr": atr[i],
            "va_high": profile.value_area_high, "va_low": profile.value_area_low,
        })
    return signals


def _backtest_vp_from_cache(signals, df, hold_days=5, atr_sl_mult=1.5, rr_ratio=2.0):
    """Backtest volume_profile using pre-computed signals (fast sweep)."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trades = []
    for sig in signals:
        if sig is None:
            continue
        i = sig["bar"]
        if i >= len(close) - 1:
            continue
        entry = sig["price"]
        sl_dist = sig["atr"] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio
        if entry < sig["va_low"]:
            sl, tp = entry - sl_dist, entry + tp_dist
            exit_info = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "BUY", "entry": entry, **exit_info})
        elif entry > sig["va_high"]:
            sl, tp = entry + sl_dist, entry - tp_dist
            exit_info = _simulate_exit("SELL", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "SELL", "entry": entry, **exit_info})
    return trades


# ------------------------------------------------------------------
# SMC MTF (Multi-Timeframe) backtest
# ------------------------------------------------------------------

def _precompute_mtf_signals(htf_df, ltf_df, swing_lookback=5, channel_lookback=50,
                            progress_callback=None):
    """Pre-compute MTF alignment signals using higher + lower TF data.

    Returns list of signal dicts (one per lower TF bar) with alignment score,
    trade bias, and entry confirmation status.
    """
    from tradingagents.agents.analysts.smc_mtf_quant import analyze_timeframe, run_mtf_analysis

    ltf_high = ltf_df["high"].values
    ltf_low = ltf_df["low"].values
    ltf_close = ltf_df["close"].values
    atr = _compute_atr(ltf_high, ltf_low, ltf_close)

    signals = []
    min_bars = max(channel_lookback + 20, 100)
    total = len(ltf_df) - min_bars

    for i in range(min_bars, len(ltf_df)):
        if np.isnan(atr[i]):
            signals.append(None)
            continue

        # Use a rolling window of lower TF data up to bar i
        ltf_window = ltf_df.iloc[:i + 1].copy()
        current_price = ltf_close[i]

        try:
            result = run_mtf_analysis(
                higher_tf_df=htf_df,
                lower_tf_df=ltf_window,
                current_price=current_price,
                swing_lookback=swing_lookback,
                channel_lookback=channel_lookback,
            )

            bias = result.trade_bias
            # Normalize weak biases
            if bias.startswith("weak_"):
                bias = bias[5:]  # "weak_bullish" -> "bullish"

            signals.append({
                "bar": i,
                "price": current_price,
                "atr": atr[i],
                "htf_bias": result.higher_tf_bias,
                "ltf_bias": result.lower_tf_bias,
                "alignment_score": result.alignment_score,
                "trade_bias": result.trade_bias,
                "has_confirmation": result.has_entry_confirmation,
                "confirmation_type": result.confirmation_type,
                "price_in_ote": result.price_in_ote,
                "price_in_channel": result.price_in_or_touches_channel,
                "bias": bias,
            })
        except Exception:
            signals.append(None)

        if progress_callback and i % 20 == 0:
            progress_callback("precompute", i - min_bars, total,
                              f"MTF analysis: bar {i - min_bars}/{total}")

    return signals


def _backtest_mtf_from_cache(signals, df, hold_days=5, min_alignment=60,
                              require_confirmation=True, require_channel=True,
                              atr_sl_mult=1.5, rr_ratio=2.0):
    """Backtest smc_mtf using pre-computed alignment signals."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trades = []
    for sig in signals:
        if sig is None:
            continue
        if sig["bias"] == "neutral":
            continue
        if sig["alignment_score"] < min_alignment:
            continue
        if require_confirmation and not sig["has_confirmation"]:
            continue
        if require_channel and not sig["price_in_channel"]:
            continue

        i = sig["bar"]
        if i >= len(close) - 1:
            continue

        entry = sig["price"]
        sl_dist = sig["atr"] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio

        if sig["bias"] == "bullish":
            sl, tp = entry - sl_dist, entry + tp_dist
            exit_info = _simulate_exit("BUY", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "BUY", "entry": entry, **exit_info})
        elif sig["bias"] == "bearish":
            sl, tp = entry + sl_dist, entry - tp_dist
            exit_info = _simulate_exit("SELL", entry, sl, tp, high, low, close, i, hold_days)
            trades.append({"bar": i, "direction": "SELL", "entry": entry, **exit_info})
    return trades


# ------------------------------------------------------------------
# XGBoost backtest
# ------------------------------------------------------------------

def _precompute_xgboost_signals(df, pipeline, symbol, timeframe):
    """Run XGBoost model(s) on historical data to generate signals.

    Uses batch prediction: computes features once, then predicts all bars at once.
    Returns list of dicts: {bar, direction, probability}.
    """
    from tradingagents.xgb_quant.predictor import LivePredictor
    from tradingagents.xgb_quant.strategy_selector import StrategySelector

    predictor = LivePredictor()
    signals = []

    if pipeline == "xgboost_ensemble":
        # Get all available models for this symbol/TF
        strategy_names = predictor.get_available_models(symbol, timeframe)
        if len(strategy_names) < 2:
            logger.warning(f"XGBoost ensemble tune: only {len(strategy_names)} models, need 2+")
            return signals

        # Compute predictions from each model, then find consensus
        all_probs = {}
        for name in strategy_names:
            try:
                if not predictor.load_strategy(name, symbol, timeframe):
                    continue
                key = f"{name}_{symbol}_{timeframe}"
                strategy = predictor._loaded_strategies[key]
                features = strategy.get_feature_set().compute(df)
                probs = strategy.predict_proba_batch(features)
                all_probs[name] = probs
            except Exception as e:
                logger.debug(f"XGBoost tune: {name} failed: {e}")
                continue

        if len(all_probs) < 2:
            return signals

        # For each bar, check consensus
        n = len(df)
        for i in range(n):
            buy_votes = 0
            sell_votes = 0
            total_prob = 0.0
            voted = 0

            for name, probs in all_probs.items():
                if i >= len(probs) or np.isnan(probs[i]):
                    continue
                if probs[i] >= 0.55:
                    buy_votes += 1
                    total_prob += probs[i]
                    voted += 1
                elif probs[i] <= 0.45:
                    sell_votes += 1
                    total_prob += (1.0 - probs[i])
                    voted += 1

            if buy_votes >= 2:
                signals.append({"bar": i, "direction": "BUY", "probability": total_prob / voted})
            elif sell_votes >= 2:
                signals.append({"bar": i, "direction": "SELL", "probability": total_prob / voted})

    else:
        # Single best strategy
        selector = StrategySelector()
        selection = selector.select(symbol)
        strategy_name = selection.recommended_strategy

        if not predictor.load_strategy(strategy_name, symbol, timeframe):
            logger.warning(f"XGBoost tune: no model for {strategy_name} on {symbol} {timeframe}")
            return signals

        key = f"{strategy_name}_{symbol}_{timeframe}"
        strategy = predictor._loaded_strategies[key]

        # Compute features once (batch)
        features = strategy.get_feature_set().compute(df)

        # Get probabilities for all bars (batch prediction)
        probs = strategy.predict_proba_batch(features)

        for i in range(len(probs)):
            if np.isnan(probs[i]):
                continue
            if probs[i] >= 0.55:
                signals.append({"bar": i, "direction": "BUY", "probability": float(probs[i])})
            elif probs[i] <= 0.45:
                signals.append({"bar": i, "direction": "SELL", "probability": float(1.0 - probs[i])})

    logger.info(f"XGBoost precompute: {len(signals)} signals from {len(df)} bars")
    return signals


def _backtest_xgboost_from_cache(
    signals, df, hold_days=10, signal_threshold=0.60,
    atr_sl_mult=1.5, rr_ratio=2.0,
):
    """Backtest XGBoost signals with given exit parameters."""
    if not signals:
        return []

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = _compute_atr(high, low, close)

    trades = []
    last_exit_bar = -1

    for sig in signals:
        i = sig["bar"]
        if i <= last_exit_bar or i >= len(close) - 1:
            continue
        if sig["probability"] < signal_threshold:
            continue

        entry = close[i]
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        sl_dist = atr[i] * atr_sl_mult
        tp_dist = sl_dist * rr_ratio

        direction = sig["direction"]
        if direction == "BUY":
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist

        exit_info = _simulate_exit(direction, entry, sl, tp, high, low, close, i, hold_days)
        trades.append({"bar": i, "direction": direction, "entry": entry, **exit_info})
        last_exit_bar = i + exit_info["bars"]

    return trades


# ------------------------------------------------------------------
# Pipeline-to-backtest dispatch
# ------------------------------------------------------------------

def _run_pipeline_sweep(pipeline, df, timeframe, grid, precomputed_cache=None,
                        progress_callback=None, progress_offset=0, progress_total=1, min_trades=10):
    """Run parameter sweep for a pipeline on a single timeframe's data.

    precomputed_cache: dict keyed by lookback (for range/breakout) or None (for SMC signals list).
      - range_quant / breakout_quant: {lookback: signals_list}
      - rule_based / smc_quant*: signals_list (single precompute, no lookback key)
      - volume_profile: None (VP is computed inline, fast enough)
    """
    from itertools import product

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))
    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Scale hold for non-D1 timeframes
        if timeframe != "D1" and "hold" in params:
            params["hold"] = scale_hold_days(params["hold"], timeframe)

        # Common exit params
        atr_sl = params.get("atr_sl_mult", 1.5)
        rr = params.get("rr_ratio", 2.0)

        if pipeline == "range_quant":
            cache_key = params["lookback"]
            signals = precomputed_cache.get(cache_key) if precomputed_cache else None
            if signals is None:
                continue
            trades = _backtest_range_quant_from_cache(
                signals, df, hold_days=params["hold"],
                mr_threshold=params["mr_threshold"], use_bias_filter=params["bias_filter"],
                atr_sl_mult=atr_sl, rr_ratio=rr,
            )
        elif pipeline == "breakout_quant":
            cache_key = params["lookback"]
            signals = precomputed_cache.get(cache_key) if precomputed_cache else None
            if signals is None:
                continue
            trades = _backtest_breakout_from_cache(
                signals, df, hold_days=params["hold"],
                squeeze_threshold=params["squeeze_threshold"],
                atr_sl_mult=atr_sl, rr_ratio=rr,
                volume_confirm=params.get("volume_confirm", True),
            )
        elif pipeline in ("rule_based", "smc_quant_basic", "smc_quant"):
            if precomputed_cache is None:
                continue
            trades = _backtest_smc_rule_based(
                precomputed_cache, df, hold_days=params["hold"],
                ob_proximity_pct=params["ob_proximity_pct"],
                atr_sl_mult=atr_sl, rr_ratio=rr,
            )
        elif pipeline == "volume_profile":
            cache_key = params["lookback"]
            signals = precomputed_cache.get(cache_key) if precomputed_cache else None
            if signals is None:
                continue
            trades = _backtest_vp_from_cache(
                signals, df, hold_days=params["hold"],
                atr_sl_mult=atr_sl, rr_ratio=rr,
            )
        elif pipeline == "smc_mtf":
            if precomputed_cache is None:
                continue
            trades = _backtest_mtf_from_cache(
                precomputed_cache, df, hold_days=params["hold"],
                min_alignment=params.get("min_alignment", 60),
                require_confirmation=params.get("require_confirmation", True),
                require_channel=params.get("require_channel", True),
                atr_sl_mult=atr_sl, rr_ratio=rr,
            )
        elif pipeline in ("xgboost", "xgboost_ensemble"):
            if precomputed_cache is None:
                continue
            trades = _backtest_xgboost_from_cache(
                precomputed_cache, df, hold_days=params["hold"],
                signal_threshold=params.get("signal_threshold", 0.60),
                atr_sl_mult=atr_sl, rr_ratio=rr,
            )
        else:
            continue

        result = _compute_stats(trades, pipeline, timeframe, params)
        if result.total_trades >= min_trades:
            results.append(result)

        if progress_callback:
            current = progress_offset + idx + 1
            progress_callback(
                "sweeping", current, progress_total,
                f"{timeframe}: testing config {idx + 1}/{len(combos)}"
            )

    return results


# ------------------------------------------------------------------
# Config mapping
# ------------------------------------------------------------------

_TIMEFRAME_INTERVAL_MINUTES = {
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 120,
    "D1": 240,
}


def best_params_to_config_updates(pipeline: str, best: TuneResult) -> Dict[str, Any]:
    """Map best backtest params to automation config field updates."""
    updates = {"timeframe": best.timeframe}

    # Analysis interval based on timeframe — match candle period for sub-H4,
    # check key levels periodically for H4/D1
    # For MTF pairs like "D1+H4", use the lower TF for interval
    interval_tf = best.timeframe.split("+")[-1] if "+" in best.timeframe else best.timeframe
    updates["analysis_interval_minutes"] = _TIMEFRAME_INTERVAL_MINUTES.get(interval_tf, 60)

    # ATR SL multiplier — directly maps to trailing_stop_atr_multiplier
    if "atr_sl_mult" in best.params:
        updates["trailing_stop_atr_multiplier"] = best.params["atr_sl_mult"]

    # Pipeline-specific confidence mapping
    if pipeline == "range_quant":
        mr = best.params.get("mr_threshold", 55)
        updates["min_confidence"] = round(max(0.50, min(0.90, mr / 100 + 0.05)), 2)
    elif pipeline == "breakout_quant":
        sq = best.params.get("squeeze_threshold", 70)
        updates["min_confidence"] = round(max(0.50, min(0.90, sq / 100)), 2)
    elif pipeline in ("xgboost", "xgboost_ensemble"):
        st = best.params.get("signal_threshold", 0.60)
        updates["min_confidence"] = round(st, 2)

    return updates


# ------------------------------------------------------------------
# Main tune function
# ------------------------------------------------------------------

async def run_tune(
    symbol: str,
    pipeline: str,
    timeframes: Optional[List[str]] = None,
    bars: int = 800,
    min_trades: int = 5,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Run parameter sweep for a symbol/pipeline combination.

    Args:
        symbol: MT5 symbol (e.g., "XAUUSD", "EURUSD")
        pipeline: Pipeline type (e.g., "range_quant", "breakout_quant")
        timeframes: Timeframes to test (None = auto-select based on pipeline)
        bars: Number of historical bars to load per timeframe
        min_trades: Minimum trades required for a config to be valid
        progress_callback: Optional callback(phase, current, total, message, steps)
            steps: list of {"name": str, "status": "pending"|"running"|"done"|"skipped"}

    Returns:
        Dict with best result, top 5, config_updates, etc.
    """
    start_time = time.time()

    if pipeline == "multi_agent":
        return {"error": "multi_agent pipeline requires LLM and cannot be auto-tuned"}

    if timeframes is None:
        timeframes = get_tunable_timeframes(pipeline)

    grid = get_parameter_grid(pipeline)
    if not grid:
        return {"error": f"No parameter grid defined for pipeline: {pipeline}"}

    total_combos = _count_grid_combos(grid, timeframes)
    needs_smc = pipeline in ("rule_based", "smc_quant_basic", "smc_quant")

    # Build step tracker
    precompute_label = (
        "Pre-compute SMC signals" if needs_smc
        else "Pre-compute range signals" if pipeline == "range_quant"
        else "Pre-compute breakout signals" if pipeline == "breakout_quant"
        else "Pre-compute VP signals" if pipeline == "volume_profile"
        else "Pre-compute MTF alignment" if pipeline == "smc_mtf"
        else "Pre-compute XGBoost predictions" if pipeline in ("xgboost", "xgboost_ensemble")
        else "Pre-compute signals"
    )
    steps = [
        {"name": "Connect to MT5", "status": "pending"},
        {"name": f"Load {symbol} data ({', '.join(timeframes)})", "status": "pending"},
        {"name": precompute_label, "status": "pending"},
        {"name": f"Sweep {total_combos} parameter combos", "status": "pending"},
        {"name": "Rank results", "status": "pending"},
    ]

    def _progress(phase, current, total, message):
        if progress_callback:
            progress_callback(phase, current, total, message, steps)

    # Phase 1: Load data
    steps[0]["status"] = "running"
    _progress("loading_data", 0, total_combos, f"Connecting to MT5...")

    is_mtf = pipeline == "smc_mtf"
    data = {}       # {tf_key: df} for single-TF, {tf_pair: (htf_df, ltf_df)} for MTF
    raw_data = {}   # cache individual TF loads to avoid duplicates

    for idx_tf, tf in enumerate(timeframes):
        steps[0]["status"] = "done"
        steps[1]["status"] = "running"

        if is_mtf and "+" in tf:
            # MTF pair: load both timeframes
            htf, ltf = tf.split("+")
            for sub_tf in (htf, ltf):
                if sub_tf not in raw_data:
                    _progress("loading_data", 0, total_combos,
                              f"Loading {symbol} {sub_tf} ({bars} bars)...")
                    try:
                        raw_data[sub_tf] = await asyncio.to_thread(load_mt5_data, symbol, sub_tf, bars)
                        logger.info(f"Loaded {len(raw_data[sub_tf])} {sub_tf} bars for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to load {sub_tf} data for {symbol}: {e}")
            if htf in raw_data and ltf in raw_data:
                data[tf] = (raw_data[htf], raw_data[ltf])
                _progress("loading_data", 0, total_combos,
                          f"Loaded {tf}: {len(raw_data[htf])} {htf} + {len(raw_data[ltf])} {ltf} bars")
        else:
            _progress("loading_data", 0, total_combos,
                      f"Loading {symbol} {tf} ({bars} bars)...")
            try:
                data[tf] = await asyncio.to_thread(load_mt5_data, symbol, tf, bars)
                logger.info(f"Loaded {len(data[tf])} {tf} bars for {symbol}")
                _progress("loading_data", 0, total_combos,
                          f"Loaded {len(data[tf])} {tf} bars")
            except Exception as e:
                logger.warning(f"Failed to load {tf} data for {symbol}: {e}")
                _progress("loading_data", 0, total_combos,
                          f"Failed to load {tf}: {e}")

    steps[1]["status"] = "done"

    if not data:
        return {"error": f"Failed to load data for {symbol} on any timeframe"}

    # Phase 2: Pre-compute signals (once per lookback/timeframe)
    # This is the key optimization: expensive signal generation runs once,
    # then the sweep just filters/exits on cached signals (instant).
    precompute_step_idx = 2  # always index 2 in steps list
    steps[precompute_step_idx]["status"] = "running"

    # Per-timeframe cache: {tf: cache_for_sweep}
    tf_caches: Dict[str, Any] = {}

    if needs_smc:
        for tf, df in data.items():
            total_smc_bars = len(df) - 100
            _progress("precompute", 0, total_smc_bars,
                      f"Pre-computing SMC signals for {tf} ({total_smc_bars} bars)...")
            tf_caches[tf] = await asyncio.to_thread(
                _precompute_smc_signals, df, 100,
                lambda ph, cur, tot, msg: _progress("precompute", cur, tot, msg),
            )
    elif pipeline == "range_quant":
        lookbacks = grid.get("lookback", [25])
        for tf, df in data.items():
            cache = {}
            for lb_idx, lb in enumerate(lookbacks):
                _progress("precompute", lb_idx, len(lookbacks),
                          f"Pre-computing range signals for {tf} (lookback={lb})...")
                cache[lb] = await asyncio.to_thread(_precompute_range_signals, df, lb)
            tf_caches[tf] = cache
    elif pipeline == "breakout_quant":
        lookbacks = grid.get("lookback", [20])
        for tf, df in data.items():
            cache = {}
            for lb_idx, lb in enumerate(lookbacks):
                _progress("precompute", lb_idx, len(lookbacks),
                          f"Pre-computing breakout signals for {tf} (lookback={lb})...")
                cache[lb] = await asyncio.to_thread(_precompute_breakout_signals, df, lb)
            tf_caches[tf] = cache
    elif pipeline == "volume_profile":
        lookbacks = grid.get("lookback", [50])
        for tf, df in data.items():
            cache = {}
            for lb_idx, lb in enumerate(lookbacks):
                _progress("precompute", lb_idx, len(lookbacks),
                          f"Pre-computing VP signals for {tf} (lookback={lb})...")
                cache[lb] = await asyncio.to_thread(_precompute_vp_signals, df, lb)
            tf_caches[tf] = cache
    elif pipeline == "smc_mtf":
        for tf_pair, (htf_df, ltf_df) in data.items():
            total_bars = len(ltf_df) - 100
            _progress("precompute", 0, total_bars,
                      f"Pre-computing MTF signals for {tf_pair} ({total_bars} bars)...")
            tf_caches[tf_pair] = await asyncio.to_thread(
                _precompute_mtf_signals, htf_df, ltf_df, 5, 50,
                lambda ph, cur, tot, msg: _progress("precompute", cur, tot, msg),
            )
    elif pipeline in ("xgboost", "xgboost_ensemble"):
        for tf, df in data.items():
            total_bars = len(df) - 200
            _progress("precompute", 0, total_bars,
                      f"Running XGBoost predictions for {tf} ({total_bars} bars)...")
            tf_caches[tf] = await asyncio.to_thread(
                _precompute_xgboost_signals, df, pipeline, symbol, tf,
            )
            _progress("precompute", total_bars, total_bars,
                      f"Got {len(tf_caches[tf])} XGBoost signals for {tf}")
    else:
        for tf in data:
            tf_caches[tf] = None

    steps[precompute_step_idx]["status"] = "done"

    # Phase 3: Sweep parameters
    sweep_step_idx = precompute_step_idx + 1
    steps[sweep_step_idx]["status"] = "running"
    all_results = []
    offset = 0
    combos_per_tf = total_combos // len(timeframes) if timeframes else 1

    for tf in data:
        cache = tf_caches.get(tf)

        if is_mtf:
            # MTF: sweep on lower TF data, precomputed cache has alignment signals
            _, ltf_df = data[tf]
            sweep_df = ltf_df.reset_index(drop=True)
            # Use the LTF name for hold scaling (e.g., "H4" from "D1+H4")
            sweep_tf = tf.split("+")[1] if "+" in tf else tf
        else:
            sweep_df = data[tf].reset_index(drop=True)
            sweep_tf = tf

        tf_results = await asyncio.to_thread(
            _run_pipeline_sweep, pipeline, sweep_df, sweep_tf, grid, cache,
            lambda ph, cur, tot, msg: _progress("sweeping", cur, tot, msg),
            offset, total_combos, min_trades,
        )
        # For MTF results, store the full pair name as timeframe
        if is_mtf:
            for r in tf_results:
                r.timeframe = tf
        all_results.extend(tf_results)
        offset += combos_per_tf

    steps[sweep_step_idx]["status"] = "done"

    # Phase 4: Find best
    rank_step_idx = sweep_step_idx + 1
    steps[rank_step_idx]["status"] = "running"
    _progress("ranking", total_combos, total_combos,
              f"Ranking {len(all_results)} valid configs...")

    duration = time.time() - start_time
    steps[rank_step_idx]["status"] = "done"
    _progress("done", total_combos, total_combos,
              f"Done in {round(duration, 1)}s")

    if not all_results:
        return {
            "best": None,
            "top_5": [],
            "all_count": 0,
            "symbol": symbol,
            "pipeline": pipeline,
            "timeframes_tested": list(data.keys()),
            "bars_per_tf": {tf: (len(v[1]) if isinstance(v, tuple) else len(v)) for tf, v in data.items()},
            "duration_seconds": round(duration, 1),
            "config_updates": {},
            "error": "No parameter combinations produced enough trades",
        }

    sorted_results = sorted(all_results, key=lambda r: r.sharpe, reverse=True)
    best = sorted_results[0]
    top_5 = sorted_results[:5]

    config_updates = best_params_to_config_updates(pipeline, best)

    return {
        "best": best.to_dict(),
        "top_5": [r.to_dict() for r in top_5],
        "all_count": len(all_results),
        "symbol": symbol,
        "pipeline": pipeline,
        "timeframes_tested": list(data.keys()),
        "bars_per_tf": {tf: (len(v[1]) if isinstance(v, tuple) else len(v)) for tf, v in data.items()},
        "duration_seconds": round(duration, 1),
        "config_updates": config_updates,
    }
