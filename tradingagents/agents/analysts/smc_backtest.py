"""
SMC Quant Analyst Prompt Backtest Harness.

Replays historical bars through the SMC quant analyst LLM, captures decisions
+ reasoning, and compares against actual price outcomes. Use this to iteratively
tune the SMC system prompt.

Usage:
    python -m tradingagents.agents.analysts.smc_backtest \
        --symbol XAUUSD --timeframe D1 --max-signals 20

    # Dry run (no LLM calls, shows prompt previews):
    python -m tradingagents.agents.analysts.smc_backtest \
        --symbol XAUUSD --timeframe D1 --dry-run

    # Save reasoning for analysis:
    python -m tradingagents.agents.analysts.smc_backtest \
        --symbol XAUUSD --timeframe D1 --max-signals 30 --save-reasoning
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent / "quant_strats" / "results"
VERDICTS_DIR = Path(__file__).parent.parent.parent.parent / "logs" / "smc_backtest" / "verdicts"


def _compute_indicators(df: pd.DataFrame) -> Dict:
    """Compute technical indicators from OHLCV DataFrame (same as backend API)."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ATR
    high_low = high - low
    atr = high_low.rolling(14).mean().iloc[-1]

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.0001)
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)

    # EMA
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)
    atr_14 = high_low.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 0.0001))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 0.0001))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(14).mean().iloc[-1]

    # Regimes
    avg_atr = high_low.rolling(50).mean().iloc[-1]
    volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"
    if adx > 25:
        market_regime = "trending-up" if plus_di.iloc[-1] > minus_di.iloc[-1] else "trending-down"
    else:
        market_regime = "ranging"

    # Volume
    volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol = df[volume_col] if volume_col in df.columns else pd.Series(0, index=df.index)
    current_volume = vol.iloc[-1]
    avg_volume_20 = vol.rolling(20).mean().iloc[-1]
    volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
    volume_spike = current_volume > avg_volume_20 * 1.5

    current_price = float(close.iloc[-1])

    indicators_context = f"""## Technical Indicators

### Momentum
- **RSI(14)**: {rsi_value:.1f} {"(Overbought)" if rsi_value > 70 else "(Oversold)" if rsi_value < 30 else "(Neutral)"}
- **MACD**: {macd_line.iloc[-1]:.5f} | Signal: {signal_line.iloc[-1]:.5f} | Histogram: {macd_hist.iloc[-1]:.5f}
  - {"Bullish (MACD > Signal)" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish (MACD < Signal)"}

### Trend
- **EMA20**: {ema20:.5f} {"(Price above)" if current_price > ema20 else "(Price below)"}
- **EMA50**: {ema50:.5f} {"(Price above)" if current_price > ema50 else "(Price below)"}
- **ADX**: {adx:.1f} {"(Strong trend)" if adx > 25 else "(Weak/No trend)"}
- **Regime**: {market_regime}

### Volatility
- **ATR(14)**: {atr:.5f}
- **Volatility**: {volatility_regime}
- **Bollinger Bands**: Upper={bb_upper.iloc[-1]:.5f} | Middle={sma20.iloc[-1]:.5f} | Lower={bb_lower.iloc[-1]:.5f}

### Volume
- **Volume Ratio**: {volume_ratio:.2f}x average {"SPIKE" if volume_spike else ""}
"""

    return {
        "indicators_context": indicators_context,
        "market_regime": market_regime,
        "volatility_regime": volatility_regime,
        "atr": float(atr),
        "rsi": float(rsi_value),
        "adx": float(adx),
    }


def _evaluate_outcome(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: int,
    direction: str,
    entry_price: float,
    sl: float,
    tp: float,
    max_hold: int,
) -> Dict:
    """Walk forward to determine TP/SL hit."""
    n = len(close)

    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
        if direction == "BUY":
            if low[j] <= sl:
                return {"result": "loss", "exit_type": "sl_hit", "exit_bar": j - entry_idx,
                        "pnl_pct": round((sl - entry_price) / entry_price * 100, 4)}
            if high[j] >= tp:
                return {"result": "win", "exit_type": "tp_hit", "exit_bar": j - entry_idx,
                        "pnl_pct": round((tp - entry_price) / entry_price * 100, 4)}
        else:
            if high[j] >= sl:
                return {"result": "loss", "exit_type": "sl_hit", "exit_bar": j - entry_idx,
                        "pnl_pct": round((entry_price - sl) / entry_price * 100, 4)}
            if low[j] <= tp:
                return {"result": "win", "exit_type": "tp_hit", "exit_bar": j - entry_idx,
                        "pnl_pct": round((entry_price - tp) / entry_price * 100, 4)}

    exit_price = close[min(entry_idx + max_hold, n - 1)]
    pnl = ((exit_price - entry_price) / entry_price * 100) if direction == "BUY" else (
        (entry_price - exit_price) / entry_price * 100)
    return {"result": "win" if pnl > 0 else "loss", "exit_type": "time_exit",
            "exit_bar": max_hold, "pnl_pct": round(pnl, 4)}


def run_smc_backtest(
    symbol: str = "XAUUSD",
    timeframe: str = "D1",
    bars: int = 500,
    max_signals: Optional[int] = None,
    dry_run: bool = False,
    max_hold_bars: int = 20,
    skip_bars: int = 3,
    save_reasoning: bool = False,
) -> Dict:
    """
    Replay historical bars through the SMC quant analyst LLM.

    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe
        bars: Historical bars to load
        max_signals: Limit LLM calls (None = unlimited)
        dry_run: If True, skip LLM calls, show prompt previews
        max_hold_bars: Max bars to wait for TP/SL hit
        skip_bars: Evaluate every N bars (reduces cost, avoids correlated signals)
        save_reasoning: Write full reasoning to JSONL file
    """
    from tradingagents.automation.auto_tuner import load_mt5_data
    from tradingagents.agents.analysts.smc_quant import (
        analyze_smc_for_quant,
        _build_smc_data_context,
        _build_smc_quant_prompt,
    )

    print(f"\n{'='*60}")
    print(f"SMC Quant Analyst Prompt Backtest")
    print(f"Symbol: {symbol} | TF: {timeframe} | Bars: {bars}")
    print(f"Max Signals: {max_signals or 'unlimited'} | Skip: every {skip_bars} bars")
    print(f"Dry Run: {dry_run} | Save Reasoning: {save_reasoning}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading MT5 data...")
    df = load_mt5_data(symbol, timeframe, bars=bars)
    print(f"Loaded {len(df)} bars from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    # Init LLM (unless dry run)
    structured_llm = None
    if not dry_run:
        from tradingagents.llm_factory import get_llm
        from tradingagents.schemas.trade_analysis import QuantAnalystDecision
        llm = get_llm(tier="deep")
        structured_llm = llm.with_structured_output(QuantAnalystDecision)

    # Setup reasoning logger
    if save_reasoning:
        VERDICTS_DIR.mkdir(parents=True, exist_ok=True)

    warmup = 100
    results: List[Dict] = []
    signal_count = 0
    llm_call_count = 0

    for i in range(warmup, len(df) - max_hold_bars, skip_bars):
        if max_signals and llm_call_count >= max_signals:
            break

        # Slice data up to bar i
        sub_df = df.iloc[:i + 1].copy()
        current_price = float(close[i])
        bar_date = str(df["date"].iloc[i])[:10]

        # Step 1: Compute SMC analysis
        try:
            smc_data = analyze_smc_for_quant(sub_df, current_price)
        except Exception as e:
            logger.debug(f"SMC analysis failed at bar {i}: {e}")
            continue

        smc_context = smc_data["smc_context"]
        smc_analysis = smc_data["smc_analysis"]

        # Step 2: Compute indicators
        indicators = _compute_indicators(sub_df)

        # Step 3: Build full prompt
        data_context = _build_smc_data_context(
            ticker=symbol,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=indicators["indicators_context"],
            market_regime=indicators["market_regime"],
            volatility_regime=indicators["volatility_regime"],
            trading_session="london",
            current_date=bar_date,
        )
        full_prompt = _build_smc_quant_prompt(data_context)

        if dry_run:
            signal_count += 1
            if signal_count <= 3:
                print(f"\n--- Bar #{signal_count} ({bar_date}) Prompt Preview ---")
                # Show just the data context, not the full system prompt
                print(data_context[:600])
                print("...\n")

            results.append({
                "bar_index": i,
                "date": bar_date,
                "price": round(current_price, 2),
                "market_regime": indicators["market_regime"],
                "volatility_regime": indicators["volatility_regime"],
                "smc_bias": smc_analysis.get("bias", "neutral"),
                "verdict": {"signal": "DRY_RUN"},
                "outcome": {"result": "unknown"},
            })
            continue

        # Step 4: Call LLM
        llm_call_count += 1
        llm_start = time.time()

        try:
            from langchain_core.messages import HumanMessage
            decision = structured_llm.invoke([HumanMessage(content=full_prompt)])
            decision_dict = decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()
            latency_ms = (time.time() - llm_start) * 1000
        except Exception as e:
            logger.error(f"LLM call failed at bar {i}: {e}")
            decision_dict = {"signal": "hold", "confidence": 0, "justification": f"LLM error: {e}"}
            latency_ms = (time.time() - llm_start) * 1000

        # Parse signal
        raw_signal = decision_dict.get("signal", "hold")
        if isinstance(raw_signal, dict):
            raw_signal = raw_signal.get("value", "hold")
        signal_map = {"buy_to_enter": "BUY", "sell_to_enter": "SELL", "hold": "HOLD", "close": "HOLD"}
        direction = signal_map.get(raw_signal, "HOLD")
        confidence = decision_dict.get("confidence", 0)
        entry = decision_dict.get("entry_price") or current_price
        sl = decision_dict.get("stop_loss")
        tp = decision_dict.get("profit_target")
        justification = decision_dict.get("justification", "")

        # Evaluate outcome if we have a trade signal with valid SL/TP
        outcome = {"result": "no_trade", "pnl_pct": 0}
        if direction in ("BUY", "SELL") and sl and tp:
            # Validate SL/TP on correct side
            valid = True
            if direction == "BUY" and (sl >= entry or tp <= entry):
                valid = False
            if direction == "SELL" and (sl <= entry or tp >= entry):
                valid = False

            if valid:
                outcome = _evaluate_outcome(
                    high, low, close, i, direction, entry, sl, tp, max_hold_bars
                )
            else:
                outcome = {"result": "invalid_sl_tp", "pnl_pct": 0}

        print(
            f"  #{llm_call_count} {bar_date} {direction} conf={confidence:.2f} "
            f"-> {outcome['result']} ({outcome.get('pnl_pct', 0):+.2f}%) "
            f"regime={indicators['market_regime']} bias={smc_analysis.get('bias', '?')} "
            f"[{latency_ms:.0f}ms]"
        )

        trade_record = {
            "bar_index": i,
            "date": bar_date,
            "price": round(current_price, 2),
            "direction": direction,
            "confidence": round(confidence, 4),
            "entry": round(entry, 2) if entry else None,
            "sl": round(sl, 2) if sl else None,
            "tp": round(tp, 2) if tp else None,
            "market_regime": indicators["market_regime"],
            "volatility_regime": indicators["volatility_regime"],
            "smc_bias": smc_analysis.get("bias", "neutral"),
            "atr": round(indicators["atr"], 2),
            "rsi": round(indicators["rsi"], 1),
            "justification": justification,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 1),
        }
        results.append(trade_record)

        # Save reasoning to JSONL
        if save_reasoning:
            jsonl_path = VERDICTS_DIR / f"smc_backtest_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade_record, ensure_ascii=False) + "\n")

    # Compute metrics
    metrics = _compute_metrics(results)
    _print_summary(metrics, dry_run)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"smc_backtest_{symbol}_{timeframe}.json"
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "trades": results}, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return metrics


def _compute_metrics(results: List[Dict]) -> Dict:
    """Compute backtest metrics."""
    if not results:
        return {"total_bars": 0}

    # Separate by signal type
    trades = [r for r in results if r.get("direction") in ("BUY", "SELL")]
    buys = [r for r in trades if r["direction"] == "BUY"]
    sells = [r for r in trades if r["direction"] == "SELL"]
    holds = [r for r in results if r.get("direction", r.get("verdict", {}).get("signal")) == "HOLD"]
    invalid = [r for r in trades if r["outcome"].get("result") == "invalid_sl_tp"]

    def _stats(trade_list: List[Dict]) -> Dict:
        valid = [t for t in trade_list if t["outcome"]["result"] in ("win", "loss")]
        if not valid:
            return {"count": len(trade_list), "valid_count": 0, "win_rate": 0, "profit_factor": 0, "avg_pnl": 0}
        wins = [t for t in valid if t["outcome"]["result"] == "win"]
        losses = [t for t in valid if t["outcome"]["result"] == "loss"]
        gross_profit = sum(t["outcome"]["pnl_pct"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["outcome"]["pnl_pct"] for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_pnl = sum(t["outcome"]["pnl_pct"] for t in valid) / len(valid)
        return {
            "count": len(trade_list),
            "valid_count": len(valid),
            "win_rate": round(len(wins) / len(valid) * 100, 1),
            "profit_factor": round(pf, 2),
            "avg_pnl": round(avg_pnl, 4),
            "total_pnl": round(sum(t["outcome"]["pnl_pct"] for t in valid), 4),
        }

    # Regime breakdown
    regime_stats = {}
    for r in trades:
        regime = r.get("market_regime", "unknown")
        if regime not in regime_stats:
            regime_stats[regime] = []
        regime_stats[regime].append(r)
    regime_breakdown = {k: _stats(v) for k, v in regime_stats.items()}

    # Bias alignment: did the LLM agree with SMC structural bias?
    aligned = [r for r in trades if (
        (r["direction"] == "BUY" and r.get("smc_bias") == "bullish") or
        (r["direction"] == "SELL" and r.get("smc_bias") == "bearish")
    )]
    counter_bias = [r for r in trades if (
        (r["direction"] == "BUY" and r.get("smc_bias") == "bearish") or
        (r["direction"] == "SELL" and r.get("smc_bias") == "bullish")
    )]

    # Confidence buckets
    high_conf = [r for r in trades if r.get("confidence", 0) >= 0.7]
    low_conf = [r for r in trades if r.get("confidence", 0) < 0.7]

    # Common justifications on losses
    losing_justifications = [
        r.get("justification", "")[:200]
        for r in trades
        if r["outcome"].get("result") == "loss"
    ][:10]

    return {
        "total_bars_evaluated": len(results),
        "total_signals": len(trades),
        "holds": len(holds),
        "invalid_sl_tp": len(invalid),
        "buy_signals": len(buys),
        "sell_signals": len(sells),
        "all_trades": _stats(trades),
        "buys_only": _stats(buys),
        "sells_only": _stats(sells),
        "regime_breakdown": regime_breakdown,
        "aligned_with_bias": _stats(aligned),
        "counter_to_bias": _stats(counter_bias),
        "high_confidence": _stats(high_conf),
        "low_confidence": _stats(low_conf),
        "losing_justifications": losing_justifications,
    }


def _print_summary(metrics: Dict, dry_run: bool) -> None:
    """Print human-readable backtest summary."""
    print(f"\n{'='*60}")
    print("SMC BACKTEST SUMMARY")
    print(f"{'='*60}")

    if dry_run:
        print(f"Total bars evaluated: {metrics.get('total_bars_evaluated', 0)}")
        print("(Dry run - no LLM calls)")
        return

    print(f"Bars evaluated: {metrics['total_bars_evaluated']}")
    print(f"Signals: {metrics['total_signals']} (BUY: {metrics['buy_signals']}, SELL: {metrics['sell_signals']})")
    print(f"Holds: {metrics['holds']} | Invalid SL/TP: {metrics['invalid_sl_tp']}")

    all_t = metrics["all_trades"]
    print(f"\n--- All Trades ---")
    print(f"  Valid: {all_t['valid_count']} | WR: {all_t['win_rate']}% | PF: {all_t['profit_factor']} | Avg PnL: {all_t['avg_pnl']}%")

    buys = metrics["buys_only"]
    sells = metrics["sells_only"]
    print(f"\n--- BUY Trades ---")
    print(f"  Valid: {buys['valid_count']} | WR: {buys['win_rate']}% | PF: {buys['profit_factor']} | Avg PnL: {buys['avg_pnl']}%")
    print(f"\n--- SELL Trades ---")
    print(f"  Valid: {sells['valid_count']} | WR: {sells['win_rate']}% | PF: {sells['profit_factor']} | Avg PnL: {sells['avg_pnl']}%")

    print(f"\n--- By Regime ---")
    for regime, stats in metrics.get("regime_breakdown", {}).items():
        print(f"  {regime}: {stats['valid_count']} trades, WR={stats['win_rate']}%, PF={stats['profit_factor']}")

    aligned = metrics["aligned_with_bias"]
    counter = metrics["counter_to_bias"]
    print(f"\n--- Bias Alignment ---")
    print(f"  With SMC bias:    {aligned['valid_count']} trades, WR={aligned['win_rate']}%, PF={aligned['profit_factor']}")
    print(f"  Counter SMC bias: {counter['valid_count']} trades, WR={counter['win_rate']}%, PF={counter['profit_factor']}")

    hi = metrics["high_confidence"]
    lo = metrics["low_confidence"]
    print(f"\n--- By Confidence ---")
    print(f"  High (>=0.7): {hi['valid_count']} trades, WR={hi['win_rate']}%, PF={hi['profit_factor']}")
    print(f"  Low  (<0.7):  {lo['valid_count']} trades, WR={lo['win_rate']}%, PF={lo['profit_factor']}")

    if metrics.get("losing_justifications"):
        print(f"\n--- Sample Losing Justifications ---")
        for j in metrics["losing_justifications"][:5]:
            print(f"  - {j}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SMC Quant Analyst Prompt Backtest")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="D1", help="Chart timeframe")
    parser.add_argument("--bars", type=int, default=500, help="Historical bars to load")
    parser.add_argument("--max-signals", type=int, default=None, help="Limit LLM calls")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls")
    parser.add_argument("--max-hold", type=int, default=20, help="Max bars to hold")
    parser.add_argument("--skip-bars", type=int, default=3, help="Evaluate every N bars")
    parser.add_argument("--save-reasoning", action="store_true", help="Save reasoning to JSONL")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_smc_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        max_signals=args.max_signals,
        dry_run=args.dry_run,
        max_hold_bars=args.max_hold,
        skip_bars=args.skip_bars,
        save_reasoning=args.save_reasoning,
    )


if __name__ == "__main__":
    main()
