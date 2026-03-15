"""
Backtest all pipeline strategies on XAUUSD daily data.

Tests pure signal generation (no LLM) for each strategy type:
1. Range Quant - mean reversion at range extremes
2. Breakout Quant - breakout from consolidation
3. SMC Rule-Based - order block / FVG / structure-based entries
4. Volume Profile - POC / value area mean reversion
5. SMC Quant (basic/deep) - same signals as rule-based but tested at different timeframe params

Outputs optimal parameters per pipeline for use in frontend defaults.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingagents.agents.analysts.range_quant import analyze_range
from tradingagents.agents.analysts.breakout_quant import analyze_consolidation
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer
from tradingagents.indicators.regime import RegimeDetector


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_xauusd(timeframe: str = "D1") -> pd.DataFrame:
    """Load XAUUSD data for a given timeframe from MT5 cache."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "tradingagents", "dataflows", "data_cache")
    path = os.path.join(cache_dir, f"XAUUSD-MT5-{timeframe}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached data for {timeframe}: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if 'adj close' in df.columns:
        df.rename(columns={'adj close': 'adj_close'}, inplace=True)
    return df


def load_all_timeframes() -> Dict[str, pd.DataFrame]:
    """Load XAUUSD data for all available timeframes."""
    frames = {}
    for tf in ["D1", "H4", "H1"]:
        try:
            frames[tf] = load_xauusd(tf)
        except FileNotFoundError:
            print(f"  Warning: No data for {tf}, skipping")
    return frames


# ------------------------------------------------------------------
# Backtest statistics
# ------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy: str
    timeframe: str
    params: Dict[str, Any]
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    total_pnl: float
    max_win: float
    max_loss: float
    sharpe: float
    profit_factor: float
    trades: List[Dict]


def compute_stats(trades: List[Dict], strategy: str, timeframe: str, params: Dict) -> BacktestResult:
    """Compute backtest statistics from trade list."""
    if not trades:
        return BacktestResult(
            strategy=strategy, timeframe=timeframe, params=params,
            total_trades=0, winners=0, losers=0, win_rate=0,
            avg_pnl=0, avg_winner=0, avg_loser=0, total_pnl=0,
            max_win=0, max_loss=0, sharpe=0, profit_factor=0, trades=[]
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

    return BacktestResult(
        strategy=strategy,
        timeframe=timeframe,
        params=params,
        total_trades=len(trades),
        winners=len(winners),
        losers=len(losers),
        win_rate=len(winners) / len(trades) * 100 if trades else 0,
        avg_pnl=avg_pnl,
        avg_winner=float(np.mean(winners)) if winners else 0,
        avg_loser=float(np.mean(losers)) if losers else 0,
        total_pnl=float(np.sum(pnls)),
        max_win=max(pnls) if pnls else 0,
        max_loss=min(pnls) if pnls else 0,
        sharpe=sharpe,
        profit_factor=profit_factor,
        trades=trades,
    )


# ------------------------------------------------------------------
# Strategy 1: Range Quant
# ------------------------------------------------------------------

def backtest_range_quant(df: pd.DataFrame, lookback: int = 25, hold_days: int = 5,
                          mr_threshold: int = 55, use_bias_filter: bool = True) -> List[Dict]:
    """Backtest range quant mean-reversion strategy."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    trades = []
    min_bars = lookback + 10

    for i in range(min_bars, len(close) - hold_days):
        result = analyze_range(high[:i + 1], low[:i + 1], close[:i + 1], lookback=lookback)

        if not result["is_ranging"]:
            continue
        if result["mean_reversion_score"] < mr_threshold:
            continue

        entry_price = close[i]
        exit_price = close[i + hold_days]
        position = result["price_position"]
        bias = result.get("structural_bias", "neutral")

        if position == "discount":
            if use_bias_filter and bias == "bearish":
                continue  # Skip counter-bias
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({"bar": i, "direction": "BUY", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})
        elif position == "premium":
            if use_bias_filter and bias == "bullish":
                continue  # Skip counter-bias
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({"bar": i, "direction": "SELL", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})

    return trades


# ------------------------------------------------------------------
# Strategy 2: Breakout Quant
# ------------------------------------------------------------------

def _precompute_bb_widths(close: np.ndarray, lookback: int) -> np.ndarray:
    """Pre-compute BB width for all bars in O(n) using rolling stats."""
    n = len(close)
    widths = np.full(n, np.nan)
    for i in range(lookback, n):
        window = close[i - lookback:i]
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        widths[i] = (4 * std / sma) * 100 if sma > 0 else 0
    return widths


def backtest_breakout_quant(df: pd.DataFrame, lookback: int = 20, hold_days: int = 5,
                             squeeze_threshold: int = 70) -> List[Dict]:
    """
    Backtest breakout strategy (fast version with pre-computed BB widths).

    Entry: When consolidation detected with squeeze + directional bias.
    Exit: Hold for hold_days.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    trades = []
    min_bars = max(lookback * 2, 50)

    # Pre-compute BB widths for O(n) instead of O(n^2)
    bb_widths = _precompute_bb_widths(close, lookback)

    for i in range(min_bars, len(close) - hold_days):
        recent_high = high[i - lookback + 1:i + 1]
        recent_low = low[i - lookback + 1:i + 1]
        recent_close = close[i - lookback + 1:i + 1]

        range_high = float(np.max(recent_high))
        range_low = float(np.min(recent_low))
        range_midpoint = (range_high + range_low) / 2
        range_percent = ((range_high - range_low) / range_midpoint) * 100 if range_midpoint > 0 else 0

        # BB squeeze: compare current width to all historical widths
        current_width = bb_widths[i]
        if np.isnan(current_width):
            continue
        historical = bb_widths[lookback:i]
        historical = historical[~np.isnan(historical)]
        if len(historical) > 0:
            squeeze_strength = float((np.sum(historical > current_width) / len(historical)) * 100)
        else:
            squeeze_strength = 50.0

        bb_squeeze = squeeze_strength > 70

        # Structure bias
        half = lookback // 2
        first_half_low = np.min(recent_low[:half])
        second_half_low = np.min(recent_low[half:])
        first_half_high = np.max(recent_high[:half])
        second_half_high = np.max(recent_high[half:])
        higher_lows = second_half_low > first_half_low
        lower_highs = second_half_high < first_half_high
        if higher_lows and not lower_highs:
            structure_bias = "bullish"
        elif lower_highs and not higher_lows:
            structure_bias = "bearish"
        else:
            structure_bias = "neutral"

        is_consolidating = range_percent < 3.0 and squeeze_strength > 60
        breakout_ready = is_consolidating and structure_bias != "neutral"

        if not breakout_ready:
            continue
        if squeeze_strength < squeeze_threshold:
            continue

        entry_price = close[i]
        exit_price = close[i + hold_days]

        if structure_bias == "bullish":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({"bar": i, "direction": "BUY", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})
        elif structure_bias == "bearish":
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({"bar": i, "direction": "SELL", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})

    return trades


# ------------------------------------------------------------------
# Strategy 3: SMC Rule-Based (Order Block + bias)
# ------------------------------------------------------------------

def _precompute_smc_signals(df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
    """
    Pre-compute SMC signals for all bars (run once, cache results).
    Returns list of dicts with bar index, price, bias, and OB distances.
    """
    analyzer = SmartMoneyAnalyzer()
    signals = []
    min_bars = max(lookback, 100)

    for i in range(min_bars, len(df)):
        window = df.iloc[max(0, i - lookback):i + 1].copy()
        current_price = window.iloc[-1]["close"]

        try:
            smc = analyzer.analyze_full_smc(
                window, current_price,
                include_equal_levels=False,
                include_breakers=False,
                include_ote=False,
                include_sweeps=False,
                include_inducements=False,
                include_rejections=False,
                include_turtle_soup=False,
            )
        except Exception:
            signals.append(None)
            continue

        bias = smc.get("bias", "neutral")
        # Find closest unmitigated OB distances
        bullish_ob_dist = float('inf')
        bearish_ob_dist = float('inf')
        for ob in smc["order_blocks"]["bullish"]:
            if not ob.mitigated:
                dist = abs(current_price - ob.top) / current_price * 100
                bullish_ob_dist = min(bullish_ob_dist, dist)
        for ob in smc["order_blocks"]["bearish"]:
            if not ob.mitigated:
                dist = abs(current_price - ob.bottom) / current_price * 100
                bearish_ob_dist = min(bearish_ob_dist, dist)

        # Also check FVG proximity for confluence scoring
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

        if (i - min_bars) % 100 == 0:
            print(f"    SMC pre-compute: {i - min_bars}/{len(df) - min_bars} bars...", flush=True)

    return signals


def backtest_smc_rule_based(signals: List[Dict], df: pd.DataFrame,
                             hold_days: int = 5, ob_proximity_pct: float = 0.5) -> List[Dict]:
    """Backtest SMC rule-based using pre-computed signals."""
    trades = []
    close = df["close"].values

    for sig in signals:
        if sig is None:
            continue
        i = sig["bar"]
        if i + hold_days >= len(close):
            continue

        bias = sig["bias"]
        if bias == "neutral":
            continue

        entry_price = sig["price"]
        exit_price = close[i + hold_days]

        if bias == "bullish" and sig["bullish_ob_dist"] < ob_proximity_pct:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({"bar": i, "direction": "BUY", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})
        elif bias == "bearish" and sig["bearish_ob_dist"] < ob_proximity_pct:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({"bar": i, "direction": "SELL", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})

    return trades


# ------------------------------------------------------------------
# Strategy 4: Volume Profile (POC / Value Area reversion)
# ------------------------------------------------------------------

def backtest_volume_profile(df: pd.DataFrame, lookback: int = 50, hold_days: int = 5) -> List[Dict]:
    """
    Backtest volume profile mean-reversion strategy.

    Entry: Price outside value area, expect reversion to POC.
    - BUY when price below value_area_low
    - SELL when price above value_area_high
    Exit: Hold for hold_days.
    """
    vp_analyzer = VolumeProfileAnalyzer()
    trades = []
    min_bars = max(lookback, 50)

    for i in range(min_bars, len(df) - hold_days):
        window = df.iloc[max(0, i - lookback):i + 1].copy()
        current_price = window.iloc[-1]["close"]

        try:
            profile = vp_analyzer.calculate_volume_profile(window, num_bins=30, lookback=lookback)
        except Exception:
            continue

        if profile.value_area_high == 0 or profile.value_area_low == 0:
            continue

        entry_price = current_price
        exit_price = df.iloc[i + hold_days]["close"]

        if current_price < profile.value_area_low:
            # Below value area - expect reversion up to POC
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({"bar": i, "direction": "BUY", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})
        elif current_price > profile.value_area_high:
            # Above value area - expect reversion down to POC
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({"bar": i, "direction": "SELL", "entry": entry_price,
                          "exit": exit_price, "pnl_pct": pnl_pct})

    return trades


# ------------------------------------------------------------------
# Strategy 5: SMC + FVG confluence (deeper analysis for smc_quant)
# ------------------------------------------------------------------

def backtest_smc_confluence(signals: List[Dict], df: pd.DataFrame,
                             hold_days: int = 5, min_confluence: int = 2) -> List[Dict]:
    """Backtest deeper SMC using pre-computed signals with confluence counting."""
    trades = []
    close = df["close"].values

    for sig in signals:
        if sig is None:
            continue
        i = sig["bar"]
        if i + hold_days >= len(close):
            continue

        bias = sig["bias"]
        if bias == "neutral":
            continue

        entry_price = sig["price"]
        exit_price = close[i + hold_days]
        confluence = 0

        if bias == "bullish":
            if sig["bullish_ob_dist"] < 0.5:  # OB within 0.5%
                confluence += 1
            if sig["bullish_fvg_near"]:
                confluence += 1
            if sig["pd_zone"] == "discount":
                confluence += 1
            if confluence >= min_confluence:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({"bar": i, "direction": "BUY", "entry": entry_price,
                              "exit": exit_price, "pnl_pct": pnl_pct, "confluence": confluence})

        elif bias == "bearish":
            if sig["bearish_ob_dist"] < 0.5:
                confluence += 1
            if sig["bearish_fvg_near"]:
                confluence += 1
            if sig["pd_zone"] == "premium":
                confluence += 1
            if confluence >= min_confluence:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({"bar": i, "direction": "SELL", "entry": entry_price,
                              "exit": exit_price, "pnl_pct": pnl_pct, "confluence": confluence})

    return trades


# ------------------------------------------------------------------
# Parameter sweep and main
# ------------------------------------------------------------------

def print_strategy_results(all_results: List[BacktestResult], strategy: str, label: str):
    """Print top results for a strategy."""
    results = [r for r in all_results if r.strategy == strategy]
    if not results:
        print(f"  No trades with enough samples")
        return

    best = max(results, key=lambda r: r.sharpe)
    print(f"  Best: {best.timeframe} {best.params}")
    print(f"  Trades: {best.total_trades} | WR: {best.win_rate:.1f}% | Sharpe: {best.sharpe:.2f}")
    print(f"  Avg P/L: {best.avg_pnl:.3f}% | Total P/L: {best.total_pnl:.1f}% | PF: {best.profit_factor:.2f}")

    # Best per timeframe
    for tf in ["D1", "H4", "H1"]:
        tf_results = [r for r in results if r.timeframe == tf]
        if tf_results:
            best_tf = max(tf_results, key=lambda r: r.sharpe)
            print(f"    {tf}: WR={best_tf.win_rate:.1f}% Sharpe={best_tf.sharpe:.2f} "
                  f"PF={best_tf.profit_factor:.2f} Trades={best_tf.total_trades} {best_tf.params}")


def run_parameter_sweep():
    """Run parameter sweeps for all strategies across all timeframes."""

    data = load_all_timeframes()

    print("=" * 80)
    print("XAUUSD PIPELINE BACKTEST -- ALL STRATEGIES x ALL TIMEFRAMES")
    print("=" * 80)
    for tf, tdf in data.items():
        print(f"  {tf}: {len(tdf)} bars, {tdf.iloc[0]['date']} to {tdf.iloc[-1]['date']}")

    all_results: List[BacktestResult] = []

    # -- 1. Range Quant -------------------------------------------
    print("\n" + "-" * 80)
    print("1. RANGE QUANT -- Mean reversion at range extremes")
    print("-" * 80)

    for tf, df in data.items():
        # Scale hold period by timeframe (5 D1 bars ~ 30 H4 bars ~ 120 H1 bars)
        hold_options = {"D1": [3, 5, 7, 10], "H4": [6, 12, 24, 36], "H1": [24, 48, 72, 120]}
        for lookback in [15, 20, 25, 30, 40]:
            for hold in hold_options.get(tf, [5]):
                for mr_thresh in [45, 55, 65]:
                    for bias_filter in [True, False]:
                        trades = backtest_range_quant(df, lookback=lookback, hold_days=hold,
                                                      mr_threshold=mr_thresh, use_bias_filter=bias_filter)
                        params = {"lookback": lookback, "hold": hold, "mr_threshold": mr_thresh,
                                  "bias_filter": bias_filter}
                        result = compute_stats(trades, "range_quant", tf, params)
                        if result.total_trades >= 10:
                            all_results.append(result)
        print(f"  {tf}: {len([r for r in all_results if r.strategy == 'range_quant' and r.timeframe == tf])} configs tested")

    print_strategy_results(all_results, "range_quant", "Range Quant")

    # -- 2. Breakout Quant ----------------------------------------
    print("\n" + "-" * 80)
    print("2. BREAKOUT QUANT -- Consolidation breakout")
    print("-" * 80)

    # Fast pre-computed BB widths -- can do D1 + H4
    for tf in ["D1", "H4"]:
        if tf not in data:
            continue
        df = data[tf].reset_index(drop=True)
        hold_options = {"D1": [3, 5, 7, 10], "H4": [6, 12, 24]}
        for lookback in [15, 20, 25, 30]:
            for hold in hold_options.get(tf, [5]):
                for squeeze in [60, 70, 80]:
                    trades = backtest_breakout_quant(df, lookback=lookback, hold_days=hold,
                                                     squeeze_threshold=squeeze)
                    params = {"lookback": lookback, "hold": hold, "squeeze_threshold": squeeze}
                    result = compute_stats(trades, "breakout_quant", tf, params)
                    if result.total_trades >= 10:
                        all_results.append(result)
        print(f"  {tf}: {len([r for r in all_results if r.strategy == 'breakout_quant' and r.timeframe == tf])} configs tested")

    print_strategy_results(all_results, "breakout_quant", "Breakout Quant")

    # -- 3 & 5. SMC Rule-Based + Confluence -------------------------
    # Pre-compute SMC signals once, then sweep hold/proximity params
    print("\n" + "-" * 80)
    print("3. SMC RULE-BASED + 5. SMC CONFLUENCE -- Pre-computing signals...")
    print("-" * 80)

    df_d1 = data["D1"].reset_index(drop=True)
    smc_signals = _precompute_smc_signals(df_d1, lookback=100)
    print(f"  Pre-computed {len([s for s in smc_signals if s is not None])} valid signals from {len(smc_signals)} bars")

    # 3. Rule-based: sweep hold + proximity on cached signals
    for hold in [3, 5, 7, 10]:
        for prox in [0.3, 0.5, 0.8, 1.0]:
            trades = backtest_smc_rule_based(smc_signals, df_d1, hold_days=hold, ob_proximity_pct=prox)
            params = {"lookback": 100, "hold": hold, "ob_proximity_pct": prox}
            result = compute_stats(trades, "smc_rule_based", "D1", params)
            if result.total_trades >= 10:
                all_results.append(result)
    print(f"  Rule-based: {len([r for r in all_results if r.strategy == 'smc_rule_based'])} configs tested")
    print_strategy_results(all_results, "smc_rule_based", "SMC Rule-Based")

    # 5. Confluence: sweep hold + min_confluence on cached signals
    for hold in [3, 5, 7, 10]:
        for min_conf in [2, 3]:
            trades = backtest_smc_confluence(smc_signals, df_d1, hold_days=hold, min_confluence=min_conf)
            params = {"lookback": 100, "hold": hold, "min_confluence": min_conf}
            result = compute_stats(trades, "smc_confluence", "D1", params)
            if result.total_trades >= 5:
                all_results.append(result)
    print(f"  Confluence: {len([r for r in all_results if r.strategy == 'smc_confluence'])} configs tested")
    print_strategy_results(all_results, "smc_confluence", "SMC Confluence")

    # -- 4. Volume Profile ----------------------------------------
    print("\n" + "-" * 80)
    print("4. VOLUME PROFILE -- Value area reversion")
    print("-" * 80)

    for tf in ["D1"]:
        if tf not in data:
            continue
        df = data[tf].reset_index(drop=True)
        for lookback in [30, 50, 75, 100]:
            for hold in [3, 5, 7, 10]:
                trades = backtest_volume_profile(df, lookback=lookback, hold_days=hold)
                params = {"lookback": lookback, "hold": hold}
                result = compute_stats(trades, "volume_profile", tf, params)
                if result.total_trades >= 10:
                    all_results.append(result)
        print(f"  {tf}: {len([r for r in all_results if r.strategy == 'volume_profile' and r.timeframe == tf])} configs tested")

    print_strategy_results(all_results, "volume_profile", "Volume Profile")

    print_strategy_results(all_results, "smc_confluence", "SMC Confluence")

    # --------------------------------------------------------------
    # Summary: Best config per strategy + best timeframe
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY -- BEST CONFIG PER STRATEGY (across all timeframes)")
    print("=" * 80)
    print(f"{'Strategy':<20} {'TF':<4} {'WR%':>6} {'Sharpe':>7} {'PF':>6} {'AvgPnL':>8} {'Trades':>7} {'Params'}")
    print("-" * 100)

    strategies = ["range_quant", "breakout_quant", "smc_rule_based", "volume_profile", "smc_confluence"]
    for strategy_name in strategies:
        strat_results = [r for r in all_results if r.strategy == strategy_name]
        if strat_results:
            best = max(strat_results, key=lambda r: r.sharpe)
            print(f"{best.strategy:<20} {best.timeframe:<4} {best.win_rate:>5.1f}% {best.sharpe:>7.2f} "
                  f"{best.profit_factor:>5.2f} {best.avg_pnl:>7.3f}% {best.total_trades:>7d}  {best.params}")
        else:
            print(f"{strategy_name:<20}  No trades with enough samples")

    # Best timeframe per strategy
    print("\n" + "=" * 80)
    print("BEST TIMEFRAME PER STRATEGY")
    print("=" * 80)
    for strategy_name in strategies:
        print(f"\n  {strategy_name}:")
        for tf in ["D1", "H4", "H1"]:
            tf_results = [r for r in all_results if r.strategy == strategy_name and r.timeframe == tf]
            if tf_results:
                best = max(tf_results, key=lambda r: r.sharpe)
                print(f"    {tf}: Sharpe={best.sharpe:.2f} WR={best.win_rate:.1f}% PF={best.profit_factor:.2f} "
                      f"Trades={best.total_trades} {best.params}")
            else:
                print(f"    {tf}: No results")

    # BUY vs SELL split
    print("\n" + "=" * 80)
    print("BUY vs SELL SPLIT -- BEST CONFIG PER STRATEGY")
    print("=" * 80)

    for strategy_name in strategies:
        strat_results = [r for r in all_results if r.strategy == strategy_name]
        if not strat_results:
            continue
        best = max(strat_results, key=lambda r: r.sharpe)
        buys = [t for t in best.trades if t["direction"] == "BUY"]
        sells = [t for t in best.trades if t["direction"] == "SELL"]
        buy_wr = len([t for t in buys if t["pnl_pct"] > 0]) / len(buys) * 100 if buys else 0
        sell_wr = len([t for t in sells if t["pnl_pct"] > 0]) / len(sells) * 100 if sells else 0
        buy_avg = np.mean([t["pnl_pct"] for t in buys]) if buys else 0
        sell_avg = np.mean([t["pnl_pct"] for t in sells]) if sells else 0
        print(f"\n  {strategy_name} ({best.timeframe}):")
        print(f"    BUY:  {len(buys):4d} trades, WR={buy_wr:.1f}%, Avg={buy_avg:.3f}%")
        print(f"    SELL: {len(sells):4d} trades, WR={sell_wr:.1f}%, Avg={sell_avg:.3f}%")

    return all_results


if __name__ == "__main__":
    results = run_parameter_sweep()
