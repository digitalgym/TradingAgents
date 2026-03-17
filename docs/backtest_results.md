# Backtest Results — Auto-Tuner Parameter Optimization

## Overview
- Symbols: XAUUSD, BTCUSD
- Bars: 800 per timeframe
- Exit method: ATR-based SL/TP (entry ± ATR(14) × multiplier, TP = SL distance × RR ratio)
- Parameters tuned: indicator-specific params + shared exit params (atr_sl_mult, rr_ratio)
- Timeframes tested: D1, H4, H1 (single-TF); D1+H4, D1+H1, H4+H1 (MTF pairs)
- Date: 2026-03-17

## Summary Table — XAUUSD

| Pipeline | WR% | Sharpe | PF | Trades | Best TF | ATR Mult | RR Ratio |
|----------|------|--------|------|--------|---------|----------|----------|
| rule_based | 75.0% | 2.62 | 4.41 | 12 | D1 | 1.5 | 1.5 |
| range_quant | 63.6% | 1.50 | 3.03 | 22 | H4 | 2.5 | 3.0 |
| breakout_quant | 100% | 162.78 | 12301 | 7 | D1 | 1.5 | 1.5 |
| volume_profile | 51.7% | 0.42 | 3.23 | 286 | H1 | 2.5 | 3.0 |
| **smc_mtf** | **70.7%** | **0.50** | **2.60** | **215** | **D1+H4** | **2.5** | **1.5** |

## Summary Table — BTCUSD

| Pipeline | WR% | Sharpe | PF | Trades | Best TF | ATR Mult | RR Ratio |
|----------|------|--------|------|--------|---------|----------|----------|
| rule_based | 80.0% | 4.19 | 4.61 | 10 | H1 | 1.5 | 1.5 |
| range_quant | 94.7% | 5.58 | 14.08 | 19 | H4 | 1.5 | 1.5 |
| smc_mtf | 40.6% | 0.42 | 2.34 | 180 | D1+H1 | 2.5 | 3.0 |

## Detailed Results

### XAUUSD

#### rule_based (SMC Rule-Based)
- **Best Config**: TF=D1, hold=3, ob_proximity=0.8%, atr_sl_mult=1.5, rr_ratio=1.5
- **Stats**: WR=75.0%, Sharpe=2.62, PF=4.41, 12 trades
- **Notes**: Strong directional accuracy with tight ATR stops. BUY signals dominant. D1 is clearly the best timeframe — picks up major structural levels accurately.

#### range_quant (Mean Reversion)
- **Best Config**: TF=H4, lookback=30, mr_threshold=65, hold=18, bias_filter=on, atr_sl_mult=2.5, rr_ratio=3.0
- **Stats**: WR=63.6%, Sharpe=1.50, PF=3.03, 22 trades
- **Notes**: H4 outperforms D1 (previous best) with wider exits. The 3:1 RR allows winners to run. Bias filter improves quality.

#### breakout_quant (Bollinger Band Squeeze)
- **Best Config**: TF=D1, lookback=20, squeeze_threshold=60, hold=7, volume_confirm=off, atr_sl_mult=1.5, rr_ratio=1.5
- **Stats**: WR=100%, Sharpe=162.78, PF=12301, 7 trades
- **Caution**: Only 7 trades — likely overfitted. Results should not be trusted without more data. The strategy is very selective on D1.

#### volume_profile (Volume Profile)
- **Best Config**: TF=H1, lookback=75, hold=120, atr_sl_mult=2.5, rr_ratio=3.0
- **Stats**: WR=51.7%, Sharpe=0.42, PF=3.23, 286 trades
- **Notes**: VP generates many signals on H1. The high PF (3.23) comes from the 3:1 RR — winners are 3x larger than losers despite ~50% WR. This is a viable strategy with proper risk management.

#### smc_mtf (Multi-Timeframe OTE + Channel)
- **Best Config**: TF=D1+H4, hold=60 (H4 bars), min_alignment=50, require_confirmation=off, require_channel=on, atr_sl_mult=2.5, rr_ratio=1.5
- **Stats**: WR=70.7%, Sharpe=0.50, PF=2.60, 215 trades
- **Notes**: D1+H4 is the clear winner. All 215 signals are BUY — gold's strong uptrend means HTF bias is exclusively bullish. No weekend gap or protected flip confirmations found in the data window, so require_confirmation must be off. High trade count gives statistical significance.
- **Per-pair breakdown**:
  - D1+H4: 67.8% WR, PF 2.53, all BUY (646 raw signals)
  - D1+H1: 44.2% WR, PF 0.77, all BUY (too noisy at H1)
  - H4+H1: 43.1% WR, PF 1.01, all SELL (fighting D1 uptrend)

### BTCUSD

#### rule_based (SMC Rule-Based)
- **Best Config**: TF=H1, hold=72 (H1 bars), ob_proximity=0.5%, atr_sl_mult=1.5, rr_ratio=1.5
- **Stats**: WR=80.0%, Sharpe=4.19, PF=4.61, 10 trades
- **Notes**: H1 outperforms D1/H4 for BTC. Tighter OB proximity (0.5% vs 0.8% for gold) reflects BTC's more precise OB reactions. Excellent WR but limited sample.

#### range_quant (Mean Reversion)
- **Best Config**: TF=H4, lookback=40, mr_threshold=65, hold=30, bias_filter=on, atr_sl_mult=1.5, rr_ratio=1.5
- **Stats**: WR=94.7%, Sharpe=5.58, PF=14.08, 19 trades
- **Notes**: Outstanding results — BTC's well-defined ranges at H4 level produce extremely high-quality mean reversion signals. The tight 1.5 RR works because the WR is so high.

#### smc_mtf (Multi-Timeframe OTE + Channel)
- **Best Config**: TF=D1+H1, hold=168 (H1 bars), min_alignment=50, require_confirmation=off, require_channel=on, atr_sl_mult=2.5, rr_ratio=3.0
- **Stats**: WR=40.6%, Sharpe=0.42, PF=2.34, 180 trades
- **Notes**: Lower WR than XAUUSD but profitable due to 3:1 RR. D1+H1 pair works better for BTC than D1+H4. Wide ATR stops needed for BTC's higher volatility.

## Active Automation Instances (as of 2026-03-17)

All instances configured with optimal backtest params, auto_execute=ON, trailing stops=ON.

| Instance | Symbol | Pipeline | TF | Interval | Confidence | ATR Mult | Backtest WR |
|----------|--------|----------|-----|----------|------------|----------|-------------|
| xauusd_rule_based | XAUUSD | rule_based | D1 | 4h | 0.65 | 1.5 | 75.0% |
| xauusd_smc_quant | XAUUSD | smc_quant | D1 | 4h | 0.65 | 1.5 | ~75%* |
| xauusd_breakout | XAUUSD | breakout_quant | D1 | 4h | 0.60 | 1.5 | 100%** |
| xauusd_range_quant | XAUUSD | range_quant | H4 | 2h | 0.70 | 2.5 | 63.6% |
| xauusd_volume_profile | XAUUSD | volume_profile | H1 | 1h | 0.65 | 2.5 | 51.7% |
| xauusd_smc_mtf | XAUUSD | smc_mtf | D1 | 2h | 0.60 | 2.5 | 70.7% |
| btcusd_rule_based | BTCUSD | rule_based | H1 | 1h | 0.65 | 1.5 | 80.0% |
| btcusd_smc_quant | BTCUSD | smc_quant | H1 | 1h | 0.65 | 1.5 | ~80%* |
| btcusd_range_quant | BTCUSD | range_quant | H4 | 2h | 0.70 | 1.5 | 94.7% |
| btcusd_smc_mtf | BTCUSD | smc_mtf | D1 | 1h | 0.60 | 2.5 | 40.6% |

\* smc_quant uses same base signals as rule_based + LLM reasoning — backtest approximated from rule_based results
\*\* Only 7 trades — low statistical confidence

## Parameter Grids

### Shared Exit Parameters (all pipelines)
- atr_sl_mult: [1.0, 1.5, 2.0, 2.5]
- rr_ratio: [1.5, 2.0, 3.0]

### Pipeline-Specific Parameters
- **rule_based/smc_quant**: hold[4], ob_proximity[4] → 192 combos/TF
- **range_quant**: lookback[5], mr_threshold[3], hold[4], bias_filter[2] → 1440 combos/TF
- **breakout_quant**: lookback[4], squeeze_threshold[3], hold[4], volume_confirm[2] → 1152 combos/TF
- **volume_profile**: lookback[4], hold[4] → 192 combos/TF
- **smc_mtf**: hold[4], min_alignment[3], require_confirmation[2], require_channel[2] → 576 combos/TF pair

## Methodology
- Pre-compute signals once per lookback/timeframe, then sweep parameters on cached results
- ATR(14) with Wilder smoothing for stop loss distance
- Walk-forward bar-by-bar exit simulation (check SL first, then TP, fall back to max hold)
- Statistics: win rate, Sharpe ratio, profit factor, avg PnL%, total PnL%, buy/sell split
- Ranking: sorted by Sharpe ratio (risk-adjusted returns)
- Minimum 5 trades required for a config to qualify
