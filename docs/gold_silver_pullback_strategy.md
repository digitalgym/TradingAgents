# Gold/Silver Pullback Strategy — Implementation Plan

## Overview

A systematic pullback strategy on XAUUSD using the gold-silver ratio and silver momentum as confirmation filters. Designed for daily/H4 timeframes with vectorbt backtesting.

---

## Architecture

```
Data from MT5 (copy_rates_range)
    ↓
Indicator Computation (pandas/numpy)
    ↓
Signal Generation (trend + pullback + silver lead)
    ↓
Backtest Engine (vectorbt)
    ↓
Performance Report (equity curve, stats)
```

---

## 1. Data Layer

### Sources
- **XAUUSD** daily + H4 (5+ years)
- **XAGUSD** daily + H4 (5+ years)
- **Gold-Silver Ratio** = XAUUSD / XAGUSD

### Implementation

Uses the existing MT5 data layer (`tradingagents/dataflows/mt5_data.py`).

```python
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tradingagents.dataflows.mt5_data import _ensure_mt5_initialized, TIMEFRAMES

_ensure_mt5_initialized()

# Download 5+ years of daily data
utc_to = datetime.now()
utc_from = utc_to - timedelta(days=365 * 5)

gold_rates = mt5.copy_rates_range("XAUUSD", TIMEFRAMES["D1"], utc_from, utc_to)
silver_rates = mt5.copy_rates_range("XAGUSD", TIMEFRAMES["D1"], utc_from, utc_to)

gold = pd.DataFrame(gold_rates)
gold["time"] = pd.to_datetime(gold["time"], unit="s")
gold.set_index("time", inplace=True)
gold.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)

silver = pd.DataFrame(silver_rates)
silver["time"] = pd.to_datetime(silver["time"], unit="s")
silver.set_index("time", inplace=True)
silver.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)

# Align indices and compute ratio
gold, silver = gold.align(silver, join="inner")
ratio = gold["Close"] / silver["Close"]
```

### H4 Data (optional higher resolution)

```python
gold_h4 = mt5.copy_rates_range("XAUUSD", TIMEFRAMES["H4"], utc_from, utc_to)
silver_h4 = mt5.copy_rates_range("XAGUSD", TIMEFRAMES["H4"], utc_from, utc_to)
```

### Data Validation
- Drop NaN rows from ratio alignment
- Verify OHLC integrity (high >= low, open/close within range)
- Minimum 1000 daily bars required
- MT5 uses `tick_volume` — not true volume, but sufficient for relative comparisons

---

## 2. Trend Filter

### Logic
Uptrend confirmed when SMA50 > SMA200 on gold daily.

```python
gold["sma50"] = gold["Close"].rolling(50).mean()
gold["sma200"] = gold["Close"].rolling(200).mean()
gold["uptrend"] = gold["sma50"] > gold["sma200"]
```

### Rules
- **Long only** when `uptrend == True`
- No shorts in this strategy (pullback-to-trend only)
- Trend must have been active for at least 5 bars (avoid whipsaws at crossover)

---

## 3. Pullback Detection

### Method A: ATR-Based Retrace

```python
gold["atr14"] = ta.atr(gold["High"], gold["Low"], gold["Close"], length=14)

# Pullback = price dropped at least 1.5 ATR from recent high
gold["recent_high"] = gold["High"].rolling(20).max()
gold["pullback_depth"] = gold["recent_high"] - gold["Low"]
gold["is_pullback"] = gold["pullback_depth"] >= 1.5 * gold["atr14"]
```

### Method B: Percentage Drop

```python
# Alternative: 2-5% retrace from 20-bar high
gold["pct_drop"] = (gold["recent_high"] - gold["Low"]) / gold["recent_high"]
gold["is_pullback"] = gold["pct_drop"].between(0.02, 0.05)
```

### Higher Low Confirmation

```python
def has_higher_low(df, idx, lookback=5):
    """Check if current low is higher than the lowest low in pullback."""
    recent_lows = df["Low"].iloc[idx - lookback:idx]
    return df["Low"].iloc[idx] > recent_lows.min()
```

### Bullish Candle Confirmation

```python
gold["bullish_candle"] = gold["Close"] > gold["Open"]
gold["body_size"] = abs(gold["Close"] - gold["Open"])
gold["avg_body"] = gold["body_size"].rolling(20).mean()
gold["strong_bullish"] = (gold["bullish_candle"]) & (gold["body_size"] > gold["avg_body"])
```

---

## 4. Silver Lead / Confirmation

### Concept
Silver often leads gold in breakouts. Use silver acceleration as entry confirmation.

```python
silver["roc5"] = silver["Close"].pct_change(5)  # 5-bar rate of change
silver["roc5_sma"] = silver["roc5"].rolling(10).mean()

# Silver acceleration = ROC above its own average
silver["accelerating"] = silver["roc5"] > silver["roc5_sma"]

# Silver breakout = closing above 20-bar high
silver["breakout"] = silver["Close"] > silver["High"].rolling(20).max().shift(1)
```

### Confirmation Modes (test both)
1. **Silver accelerating**: `silver["accelerating"] == True`
2. **Silver breakout**: `silver["breakout"] == True`
3. **Either**: acceleration OR breakout

---

## 5. Gold-Silver Ratio Filter

### Concept
Extreme ratio values indicate dislocation — avoid entries at extremes.

```python
ratio_mean = ratio.rolling(252).mean()  # 1-year mean
ratio_std = ratio.rolling(252).std()

# Avoid when ratio is >2 std devs from mean (extreme fear/greed)
ratio_z = (ratio - ratio_mean) / ratio_std
ratio_ok = ratio_z.abs() < 2.0
```

### Interpretation
| Ratio Zone | Meaning | Action |
|-----------|---------|--------|
| < -2σ | Silver outperforming heavily | Skip — reversion risk |
| -2σ to +2σ | Normal range | Trade allowed |
| > +2σ | Gold outperforming heavily | Skip — silver lagging = weak rally |

---

## 6. Signal Generation

### Combined Entry Signal

```python
entry_signal = (
    gold["uptrend"] &              # Trend filter
    gold["is_pullback"] &          # Price pulled back
    gold["strong_bullish"] &       # Bullish candle confirmation
    silver["accelerating"] &       # Silver leading
    ratio_ok                       # Ratio not extreme
)
```

### Stop Loss
- Below the recent swing low (lowest low in pullback window)
- Minimum 1 ATR below entry

```python
gold["stop_loss"] = gold["Low"].rolling(10).min()  # 10-bar low
gold["stop_distance"] = gold["Close"] - gold["stop_loss"]

# Enforce minimum stop distance
gold["stop_distance"] = gold["stop_distance"].clip(lower=gold["atr14"])
gold["stop_loss"] = gold["Close"] - gold["stop_distance"]
```

### Take Profit
- 1:2 risk-reward ratio (2x stop distance)

```python
gold["take_profit"] = gold["Close"] + 2 * gold["stop_distance"]
```

---

## 7. Backtest Engine

### Use Existing Auto-Tuner

This strategy should plug into the existing backtest infrastructure in `tradingagents/automation/auto_tuner.py` rather than introducing a new engine.

The auto-tuner already provides:
- **Walk-forward bar-by-bar SL/TP** via `_simulate_exit()` with trailing stops
- **Signal pre-computation + cached parameter sweeps** for fast iteration
- **ATR-based SL**, configurable RR ratio, max hold expiry
- **Position sizing** via `risk/position_sizing.py` (Kelly, fixed fractional, volatility)
- **Risk metrics** via `risk/metrics.py` (Sharpe, Sortino, Calmar, VaR, max drawdown)

### Integration Approach

Add a new pipeline type `"gold_silver_pullback"` following the existing pattern:

```python
# 1. Pre-compute signals (like _precompute_range_signals, _precompute_breakout_signals)
def _precompute_gold_silver_signals(gold_df, silver_df, params):
    """Pre-compute all indicators for the gold/silver pullback strategy."""
    sma_fast = gold_df["close"].rolling(params["sma_fast"]).mean()
    sma_slow = gold_df["close"].rolling(params["sma_slow"]).mean()
    uptrend = sma_fast > sma_slow

    atr = compute_atr(gold_df, 14)
    pullback_depth = gold_df["high"].rolling(20).max() - gold_df["low"]
    is_pullback = pullback_depth >= params["pullback_atr_mult"] * atr

    silver_roc = silver_df["close"].pct_change(params["silver_roc_period"])
    silver_accelerating = silver_roc > silver_roc.rolling(10).mean()

    ratio = gold_df["close"] / silver_df["close"]
    ratio_z = (ratio - ratio.rolling(252).mean()) / ratio.rolling(252).std()
    ratio_ok = ratio_z.abs() < params["ratio_z_filter"]

    bullish_candle = gold_df["close"] > gold_df["open"]

    return {
        "signal": uptrend & is_pullback & bullish_candle & silver_accelerating & ratio_ok,
        "atr": atr,
    }

# 2. Feed signals into _simulate_exit() — already handles SL/TP/trailing/expiry
# 3. Collect trades → _compute_stats() — already computes all metrics
# 4. Parameter sweep via get_parameter_grid() — add grid for this pipeline
```

### Parameter Grid

```python
GOLD_SILVER_GRID = {
    "sma_fast": [20, 50, 100],
    "sma_slow": [100, 200],
    "pullback_atr_mult": [1.0, 1.5, 2.0, 2.5],
    "silver_roc_period": [3, 5, 10],
    "atr_sl_mult": [1.0, 1.5, 2.0, 2.5],
    "rr_ratio": [1.5, 2.0, 3.0],
    "max_hold_days": [30, 60, 90],
    "ratio_z_filter": [1.5, 2.0, 2.5],
}
```

### Metrics (already computed by auto-tuner)

- win_rate, profit_factor, sharpe, max_drawdown
- avg_pnl, avg_winner, avg_loser, total_pnl
- Per-direction stats (BUY/SELL split)

---

## 9. Parameter Sensitivity

Test these parameter ranges to find robust values:

| Parameter | Default | Range to Test |
|-----------|---------|---------------|
| SMA fast | 50 | 20, 50, 100 |
| SMA slow | 200 | 100, 200 |
| Pullback ATR mult | 1.5 | 1.0, 1.5, 2.0, 2.5 |
| Pullback % range | 2-5% | 1-3%, 2-5%, 3-7% |
| Silver ROC period | 5 | 3, 5, 10 |
| RR ratio | 2.0 | 1.5, 2.0, 3.0 |
| Max hold bars | 60 | 30, 60, 90 |
| Ratio Z-score filter | 2.0 | 1.5, 2.0, 2.5 |

### Walk-Forward Validation

Use the auto-tuner's existing in-sample/out-of-sample split or run `run_tune()` on different bar ranges:

```python
# Optimize on first 600 bars, validate on last 200
result_is = await run_tune("XAUUSD", "gold_silver_pullback", bars=600)
result_oos = await run_tune("XAUUSD", "gold_silver_pullback", bars=200)
# If OOS degrades >30% vs IS, parameters are overfit
```

---

## 10. Integration Notes

### Relationship to Existing System
- Plugs into `auto_tuner.py` as a new pipeline type alongside range_quant, breakout, etc.
- Uses existing `_simulate_exit()`, `_compute_stats()`, position sizing, and risk metrics
- Could feed signals into the XGBoost quant system as an additional strategy type
- Gold-silver ratio could become a feature in XGBoost models

### File Location
- Signal logic: `tradingagents/agents/analysts/gold_silver_pullback_quant.py` (new analyst)
- Pre-compute + grid: add to `tradingagents/automation/auto_tuner.py`
- Config: add `"gold_silver_pullback"` pipeline to existing parameter grid system

---

## Risks & Caveats

1. **Survivorship bias**: Gold has been in a secular uptrend — long-only strategies look great in hindsight
2. **Regime dependency**: Strategy assumes trending regime; will underperform in range-bound gold
3. **Data quality**: MT5 data may have gaps around rollover or maintenance windows
4. **Execution gap**: Backtest assumes fill at close; real execution has slippage and spread
5. **Silver correlation**: Au/Ag correlation is not constant — may decouple during stress events
6. **No guarantee**: Past performance does not predict future results
