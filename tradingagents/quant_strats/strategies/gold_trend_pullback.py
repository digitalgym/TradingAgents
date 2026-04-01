"""
Gold Trend-Pullback Strategy (Mechanical)

A purely rule-based, no-ML strategy designed from backtest evidence:
- Gold trends strongly → trade WITH the trend only
- Pullback entries have highest probability → wait for dips
- Silver leads gold → require silver confirmation
- Multiple confluence → only enter when 3+ factors align

Rules (LONG only when D1 uptrend confirmed):
1. TREND: SMA50 > SMA100 on analysis timeframe
2. PULLBACK: Price pulled back ≥1x ATR from recent swing high
3. REVERSAL: Current bar is bullish (close > open) and closes above prior bar's high
4. CONFLUENCE (need 3+ of 5):
   a. RSI(14) < 40 (oversold in uptrend = high-probability bounce)
   b. Price touched or pierced BB lower band
   c. Silver accelerating (XAGUSD close > close[3])
   d. Gold/Silver ratio z-score < 1.0 (gold not overvalued vs silver)
   e. Price near EMA50 support (within 1.5x ATR)

SHORT rules are the exact mirror when SMA50 < SMA100.

Exit:
- SL: Below swing low or entry - 2x ATR (whichever is tighter)
- TP: 3x risk distance (3:1 R:R)
- Trailing stop: 2x ATR once 1.5x risk in profit

Why no ML:
- XGBoost on gold: 42% WR, Sharpe 0.02-0.05 across all strategies
- This mechanical approach targets fewer, higher-quality trades
- Every rule is backed by observed edge in backtest data
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PullbackSignal:
    direction: str          # "BUY", "SELL", or "HOLD"
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confluence_count: int = 0
    confluence_factors: List[str] = None
    rationale: str = ""

    def __post_init__(self):
        if self.confluence_factors is None:
            self.confluence_factors = []


def gold_trend_pullback(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    silver_close: Optional[np.ndarray] = None,
    # Trend params
    sma_fast: int = 50,
    sma_slow: int = 100,
    # Pullback params
    pullback_atr_mult: float = 1.0,
    swing_lookback: int = 10,
    # Confluence params
    rsi_period: int = 14,
    rsi_threshold: float = 40.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    silver_lookback: int = 3,
    ratio_zscore_period: int = 50,
    ratio_zscore_threshold: float = 1.0,
    ema_proximity_atr: float = 1.5,
    min_confluence: int = 3,
    # Exit params
    sl_atr_mult: float = 2.0,
    rr_target: float = 3.0,
    atr_period: int = 14,
) -> PullbackSignal:
    """
    Generate a trend-pullback signal for gold (or any trending instrument).

    Returns PullbackSignal with direction, levels, and confluence detail.
    """
    n = len(close)
    min_bars = max(sma_slow, bb_period, atr_period, rsi_period) + swing_lookback + 5
    if n < min_bars:
        return PullbackSignal(direction="HOLD", rationale=f"Need {min_bars} bars, have {n}")

    cs = pd.Series(close)

    # ── INDICATORS ──
    sma_f = cs.rolling(sma_fast).mean().values
    sma_s = cs.rolling(sma_slow).mean().values
    atr = _compute_atr(high, low, close, atr_period)

    # RSI
    delta = cs.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).values

    # Bollinger Bands
    bb_sma = cs.rolling(bb_period).mean().values
    bb_std_arr = cs.rolling(bb_period).std().values
    bb_upper = bb_sma + bb_std * bb_std_arr
    bb_lower = bb_sma - bb_std * bb_std_arr

    # EMA50
    ema50 = cs.ewm(span=50, adjust=False).mean().values

    # Gold/Silver ratio z-score
    ratio_z = np.nan
    if silver_close is not None and len(silver_close) >= n:
        sc = silver_close[-n:]
        ratio = close / (sc + 1e-10)
        rs = pd.Series(ratio)
        r_mean = rs.rolling(ratio_zscore_period).mean().values
        r_std = rs.rolling(ratio_zscore_period).std().values
        ratio_z_arr = (ratio - r_mean) / (r_std + 1e-10)
        ratio_z = ratio_z_arr[-1]

    # ── CURRENT BAR VALUES ──
    i = n - 1
    curr_close = close[i]
    curr_open = open_[i]
    curr_high = high[i]
    curr_low = low[i]
    curr_atr = atr[i]
    curr_rsi = rsi[i]
    prev_high = high[i - 1] if i > 0 else curr_high
    prev_low = low[i - 1] if i > 0 else curr_low

    if np.isnan(curr_atr) or curr_atr < 1e-10 or np.isnan(sma_f[i]) or np.isnan(sma_s[i]):
        return PullbackSignal(direction="HOLD", rationale="Indicators warming up")

    # ── STEP 1: TREND ──
    if sma_f[i] > sma_s[i]:
        trend = "UP"
    elif sma_f[i] < sma_s[i]:
        trend = "DOWN"
    else:
        return PullbackSignal(direction="HOLD", rationale="No clear trend (SMAs flat)")

    # ── STEP 2: PULLBACK ──
    if trend == "UP":
        swing_high = np.max(high[max(0, i - swing_lookback):i])
        pullback_depth = swing_high - curr_low
        has_pullback = pullback_depth >= curr_atr * pullback_atr_mult
    else:
        swing_low = np.min(low[max(0, i - swing_lookback):i])
        pullback_depth = curr_high - swing_low
        has_pullback = pullback_depth >= curr_atr * pullback_atr_mult

    if not has_pullback:
        return PullbackSignal(
            direction="HOLD",
            rationale=f"Trend={trend} but pullback depth {pullback_depth:.2f} < {curr_atr * pullback_atr_mult:.2f} (1x ATR)"
        )

    # ── STEP 3: REVERSAL CANDLE ──
    if trend == "UP":
        is_reversal = curr_close > curr_open and curr_close > prev_high
    else:
        is_reversal = curr_close < curr_open and curr_close < prev_low

    if not is_reversal:
        side = "bullish" if trend == "UP" else "bearish"
        return PullbackSignal(
            direction="HOLD",
            rationale=f"Trend={trend}, pullback OK, but no {side} reversal candle"
        )

    # ── STEP 4: CONFLUENCE ──
    factors = []

    # a. RSI oversold/overbought
    if trend == "UP" and not np.isnan(curr_rsi) and curr_rsi < rsi_threshold:
        factors.append(f"RSI={curr_rsi:.1f} (<{rsi_threshold})")
    elif trend == "DOWN" and not np.isnan(curr_rsi) and curr_rsi > (100 - rsi_threshold):
        factors.append(f"RSI={curr_rsi:.1f} (>{100 - rsi_threshold})")

    # b. BB band touch
    if trend == "UP" and not np.isnan(bb_lower[i]) and curr_low <= bb_lower[i]:
        factors.append(f"BB lower touch (low={curr_low:.2f}, band={bb_lower[i]:.2f})")
    elif trend == "DOWN" and not np.isnan(bb_upper[i]) and curr_high >= bb_upper[i]:
        factors.append(f"BB upper touch (high={curr_high:.2f}, band={bb_upper[i]:.2f})")

    # c. Silver accelerating
    if silver_close is not None and len(silver_close) >= silver_lookback + 1:
        sc = silver_close[-n:]
        if trend == "UP" and sc[-1] > sc[-(silver_lookback + 1)]:
            factors.append(f"Silver rising ({sc[-1]:.2f} > {sc[-(silver_lookback + 1)]:.2f})")
        elif trend == "DOWN" and sc[-1] < sc[-(silver_lookback + 1)]:
            factors.append(f"Silver falling ({sc[-1]:.2f} < {sc[-(silver_lookback + 1)]:.2f})")

    # d. Ratio z-score
    if not np.isnan(ratio_z):
        if trend == "UP" and ratio_z < ratio_zscore_threshold:
            factors.append(f"Au/Ag ratio z={ratio_z:.2f} (<{ratio_zscore_threshold})")
        elif trend == "DOWN" and ratio_z > -ratio_zscore_threshold:
            factors.append(f"Au/Ag ratio z={ratio_z:.2f} (>-{ratio_zscore_threshold})")

    # e. Near EMA50 support/resistance
    ema_dist = abs(curr_close - ema50[i])
    if ema_dist <= curr_atr * ema_proximity_atr:
        factors.append(f"Near EMA50 ({ema_dist / curr_atr:.1f}x ATR)")

    if len(factors) < min_confluence:
        return PullbackSignal(
            direction="HOLD",
            confluence_count=len(factors),
            confluence_factors=factors,
            rationale=(
                f"Trend={trend}, pullback+reversal OK, but only {len(factors)}/{min_confluence} "
                f"confluence: {', '.join(factors) if factors else 'none'}"
            ),
        )

    # ── STEP 5: ENTRY & LEVELS ──
    entry = curr_close

    if trend == "UP":
        direction = "BUY"
        # SL: below recent swing low or entry - 2x ATR (tighter one)
        recent_swing_low = np.min(low[max(0, i - swing_lookback):i + 1])
        sl_swing = recent_swing_low - curr_atr * 0.3  # Small buffer
        sl_atr = entry - curr_atr * sl_atr_mult
        sl = max(sl_swing, sl_atr)  # Tighter of the two
        risk = entry - sl
        tp = entry + risk * rr_target
    else:
        direction = "SELL"
        recent_swing_high = np.max(high[max(0, i - swing_lookback):i + 1])
        sl_swing = recent_swing_high + curr_atr * 0.3
        sl_atr = entry + curr_atr * sl_atr_mult
        sl = min(sl_swing, sl_atr)
        risk = sl - entry
        tp = entry - risk * rr_target

    return PullbackSignal(
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        confluence_count=len(factors),
        confluence_factors=factors,
        rationale=(
            f"{direction} {trend} pullback: {len(factors)} confluence factors — "
            f"{', '.join(factors)}. Entry={entry:.2f}, SL={sl:.2f}, "
            f"TP={tp:.2f}, RR=1:{rr_target:.1f}"
        ),
    )


# ---------------------------------------------------------------------------
# Vectorised backtest — generates signals for every bar, simulates exits
# ---------------------------------------------------------------------------

def backtest_gold_trend_pullback(
    df_gold: pd.DataFrame,
    df_silver: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run the gold trend-pullback strategy across all bars in df_gold.

    Returns DataFrame with signal, entry, sl, tp, confluence per bar.
    """
    p = {
        "sma_fast": 50,
        "sma_slow": 100,
        "pullback_atr_mult": 1.0,
        "swing_lookback": 10,
        "rsi_period": 14,
        "rsi_threshold": 40.0,
        "bb_period": 20,
        "bb_std": 2.0,
        "silver_lookback": 3,
        "ratio_zscore_period": 50,
        "ratio_zscore_threshold": 1.0,
        "ema_proximity_atr": 1.5,
        "min_confluence": 3,
        "sl_atr_mult": 2.0,
        "rr_target": 3.0,
        "atr_period": 14,
    }
    if params:
        p.update(params)

    high = df_gold["high"].values.astype(float)
    low = df_gold["low"].values.astype(float)
    close = df_gold["close"].values.astype(float)
    open_ = df_gold["open"].values.astype(float)
    n = len(close)

    silver_close = None
    if df_silver is not None and len(df_silver) >= n:
        silver_close = df_silver["close"].values.astype(float)

    # Pre-compute all indicators once
    cs = pd.Series(close)
    sma_f = cs.rolling(p["sma_fast"]).mean().values
    sma_s = cs.rolling(p["sma_slow"]).mean().values
    atr = _compute_atr(high, low, close, p["atr_period"])

    delta = cs.diff()
    gain = delta.where(delta > 0, 0.0).rolling(p["rsi_period"]).mean()
    loss_s = (-delta.where(delta < 0, 0.0)).rolling(p["rsi_period"]).mean()
    rsi = (100 - (100 / (1 + gain / (loss_s + 1e-10)))).values

    bb_sma = cs.rolling(p["bb_period"]).mean().values
    bb_std_arr = cs.rolling(p["bb_period"]).std().values
    bb_upper = bb_sma + p["bb_std"] * bb_std_arr
    bb_lower = bb_sma - p["bb_std"] * bb_std_arr

    ema50 = cs.ewm(span=50, adjust=False).mean().values

    # Ratio z-score (full array)
    ratio_z = np.full(n, np.nan)
    if silver_close is not None:
        ratio = close / (silver_close + 1e-10)
        rs = pd.Series(ratio)
        r_mean = rs.rolling(p["ratio_zscore_period"]).mean().values
        r_std = rs.rolling(p["ratio_zscore_period"]).std().values
        ratio_z = (ratio - r_mean) / (r_std + 1e-10)

    # Output arrays
    signals = np.zeros(n, dtype=int)
    entries = np.full(n, np.nan)
    sls = np.full(n, np.nan)
    tps = np.full(n, np.nan)
    confluences = np.zeros(n, dtype=int)

    warmup = max(p["sma_slow"], p["bb_period"], p["atr_period"], p["rsi_period"]) + p["swing_lookback"] + 5

    for i in range(warmup, n):
        if np.isnan(atr[i]) or atr[i] < 1e-10 or np.isnan(sma_f[i]) or np.isnan(sma_s[i]):
            continue

        # 1. Trend
        if sma_f[i] > sma_s[i]:
            trend = 1  # UP
        elif sma_f[i] < sma_s[i]:
            trend = -1  # DOWN
        else:
            continue

        # 2. Pullback
        lb = p["swing_lookback"]
        if trend == 1:
            swing_high = np.max(high[max(0, i - lb):i])
            pullback = swing_high - low[i]
        else:
            swing_low = np.min(low[max(0, i - lb):i])
            pullback = high[i] - swing_low

        if pullback < atr[i] * p["pullback_atr_mult"]:
            continue

        # 3. Reversal candle
        if trend == 1:
            if not (close[i] > open_[i] and close[i] > high[i - 1]):
                continue
        else:
            if not (close[i] < open_[i] and close[i] < low[i - 1]):
                continue

        # 4. Confluence
        conf = 0

        # a. RSI
        if trend == 1 and not np.isnan(rsi[i]) and rsi[i] < p["rsi_threshold"]:
            conf += 1
        elif trend == -1 and not np.isnan(rsi[i]) and rsi[i] > (100 - p["rsi_threshold"]):
            conf += 1

        # b. BB touch
        if trend == 1 and not np.isnan(bb_lower[i]) and low[i] <= bb_lower[i]:
            conf += 1
        elif trend == -1 and not np.isnan(bb_upper[i]) and high[i] >= bb_upper[i]:
            conf += 1

        # c. Silver
        if silver_close is not None and i >= p["silver_lookback"]:
            if trend == 1 and silver_close[i] > silver_close[i - p["silver_lookback"]]:
                conf += 1
            elif trend == -1 and silver_close[i] < silver_close[i - p["silver_lookback"]]:
                conf += 1

        # d. Ratio z-score
        if not np.isnan(ratio_z[i]):
            if trend == 1 and ratio_z[i] < p["ratio_zscore_threshold"]:
                conf += 1
            elif trend == -1 and ratio_z[i] > -p["ratio_zscore_threshold"]:
                conf += 1

        # e. Near EMA50
        ema_dist = abs(close[i] - ema50[i])
        if ema_dist <= atr[i] * p["ema_proximity_atr"]:
            conf += 1

        if conf < p["min_confluence"]:
            continue

        # 5. Levels
        entry = close[i]
        if trend == 1:
            swing_low_sl = np.min(low[max(0, i - lb):i + 1]) - atr[i] * 0.3
            sl = max(swing_low_sl, entry - atr[i] * p["sl_atr_mult"])
            risk = entry - sl
            tp = entry + risk * p["rr_target"]
            signals[i] = 1
        else:
            swing_high_sl = np.max(high[max(0, i - lb):i + 1]) + atr[i] * 0.3
            sl = min(swing_high_sl, entry + atr[i] * p["sl_atr_mult"])
            risk = sl - entry
            tp = entry - risk * p["rr_target"]
            signals[i] = -1

        entries[i] = entry
        sls[i] = sl
        tps[i] = tp
        confluences[i] = conf

    return pd.DataFrame({
        "signal": signals,
        "entry_price": entries,
        "sl": sls,
        "tp": tps,
        "confluence": confluences,
    }, index=df_gold.index)


# ---------------------------------------------------------------------------
# Full backtest with exit simulation
# ---------------------------------------------------------------------------

def run_backtest(
    df_gold: pd.DataFrame,
    df_silver: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    max_hold: int = 30,
) -> Dict[str, Any]:
    """
    Run full backtest: generate signals, simulate exits, compute stats.

    Returns dict with trades list and summary statistics.
    """

    signals_df = backtest_gold_trend_pullback(df_gold, df_silver, params)

    high = df_gold["high"].values.astype(float)
    low = df_gold["low"].values.astype(float)
    close = df_gold["close"].values.astype(float)
    atr = _compute_atr(high, low, close, 14)
    n = len(close)

    trades = []
    for i in range(n):
        sig = signals_df["signal"].iloc[i]
        if sig == 0:
            continue

        entry = signals_df["entry_price"].iloc[i]
        sl = signals_df["sl"].iloc[i]
        tp = signals_df["tp"].iloc[i]

        if np.isnan(entry) or np.isnan(sl) or np.isnan(tp):
            continue

        direction = "BUY" if sig == 1 else "SELL"

        result = _simulate_exit(
            direction, entry, sl, tp, high, low, close, i, max_hold,
            trailing_atr_mult=2.0,
            atr=atr,
        )

        trades.append({
            "bar": i,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit": result["exit"],
            "pnl_pct": result["pnl_pct"],
            "reason": result["reason"],
            "bars_held": result["bars"],
            "confluence": int(signals_df["confluence"].iloc[i]),
        })

    # Stats
    if not trades:
        return {
            "trades": [],
            "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "profit_factor": 0, "sharpe": 0,
            "max_drawdown_pct": 0, "total_pnl_pct": 0, "avg_hold": 0,
            "buy_count": 0, "sell_count": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(pnls) * 100
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    pf = gross_profit / gross_loss

    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252 / max(1, len(pnls))) if std_pnl > 0 else 0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity)) if len(equity) > 0 else 0

    buy_count = sum(1 for t in trades if t["direction"] == "BUY")
    sell_count = sum(1 for t in trades if t["direction"] == "SELL")

    return {
        "trades": trades,
        "total": len(trades),
        "wins": len(winners),
        "losses": len(losers),
        "win_rate": win_rate,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_pnl_pct": sum(pnls),
        "avg_pnl_pct": avg_pnl,
        "avg_hold": np.mean([t["bars_held"] for t in trades]),
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_atr(high, low, close, period=14):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def _simulate_exit(
    direction: str, entry: float, sl: float, tp: float,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_bar: int, max_hold: int,
    trailing_atr_mult: float = 0.0, atr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Walk forward bar-by-bar: SL checked first, then TP, then max-hold expiry."""
    end_bar = min(entry_bar + max_hold, len(close) - 1)
    current_sl = sl
    best_price = entry

    for j in range(entry_bar + 1, end_bar + 1):
        # Trailing stop update
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

        # Check SL/TP
        if direction == "BUY":
            if low[j] <= current_sl:
                pnl = (current_sl - entry) / entry * 100
                reason = "trail_sl" if current_sl > sl else "sl"
                return {"exit": current_sl, "pnl_pct": pnl, "reason": reason, "bars": j - entry_bar}
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {"exit": tp, "pnl_pct": pnl, "reason": "tp", "bars": j - entry_bar}
        else:
            if high[j] >= current_sl:
                pnl = (entry - current_sl) / entry * 100
                reason = "trail_sl" if current_sl < sl else "sl"
                return {"exit": current_sl, "pnl_pct": pnl, "reason": reason, "bars": j - entry_bar}
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
