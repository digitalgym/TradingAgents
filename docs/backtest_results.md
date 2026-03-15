# Backtest Results — Auto-Tuner Parameter Optimization

## Overview
- Symbol: XAUUSD
- Bars: 800 (D1)
- Exit method: ATR-based SL/TP (entry ± ATR(14) × multiplier, TP = SL distance × RR ratio)
- Parameters tuned: indicator-specific params + shared exit params (atr_sl_mult, rr_ratio)

## Summary Table

| Pipeline | WR% | Sharpe | PF | Trades | Best TF | ATR Mult | RR Ratio | Duration |
|----------|------|--------|------|--------|---------|----------|----------|----------|
| rule_based | ~75% | 2.62 | 4.41 | varies | D1 | 1.5 | 1.5 | ~3s |
| range_quant | ~62% | 1.48 | 2.48 | varies | D1 | 1.0 | 1.5 | ~3s |
| breakout_quant | 100% | high | high | 5 | D1 | 2.5 | 2.0 | ~1s |
| volume_profile | ~52% | 0.06 | 1.18 | varies | D1 | 2.5 | 3.0 | ~38s |

## Detailed Results

### rule_based (SMC Rule-Based)
- **Best Config**: TF=D1, hold=varies, ob_proximity=varies, atr_sl_mult=1.5, rr_ratio=1.5
- **Stats**: WR≈75%, Sharpe=2.62, PF=4.41
- **Previous (fixed hold)**: Sharpe=1.49, PF=2.22
- **Improvement**: Sharpe +76%, PF +99%
- **Notes**: Strong improvement with ATR exits. The tighter ATR multiplier (1.5) and symmetric RR (1.5) work well with SMC signals which tend to have high directional accuracy.

### range_quant (Mean Reversion)
- **Best Config**: TF=D1, lookback=varies, mr_threshold=varies, atr_sl_mult=1.0, rr_ratio=1.5
- **Stats**: WR≈62%, Sharpe=1.48, PF=2.48
- **Previous (fixed hold)**: Sharpe=0.74, PF=1.87
- **Improvement**: Sharpe +100%, PF +33%
- **Notes**: Tight ATR (1.0) makes sense for mean reversion — small moves, quick exits. The 1.5 RR provides decent reward without requiring large moves.

### breakout_quant (Bollinger Band Squeeze)
- **Best Config**: TF=D1, lookback=varies, squeeze_threshold=varies, atr_sl_mult=2.5, rr_ratio=2.0
- **Stats**: WR=100%, Sharpe=high, PF=high, Trades=5
- **Previous (fixed hold)**: Similar limited trades
- **Notes**: Only 5 trades — likely overfitted. The wide ATR (2.5) and high RR (2.0) filter for only the strongest breakouts. Need more data or shorter timeframes to validate. Results should not be trusted without more trades.

### volume_profile (Volume Profile)
- **Best Config**: TF=D1, lookback=varies, atr_sl_mult=2.5, rr_ratio=3.0
- **Stats**: WR≈52%, Sharpe=0.06, PF=1.18
- **Previous (fixed hold)**: Sharpe negative, PF<1
- **Improvement**: From losing to marginally profitable
- **Notes**: Volume profile signals are weaker on XAUUSD. Wide ATR (2.5) and high RR (3.0) suggest the strategy only works with generous exits. May perform better on other symbols with clearer volume profiles.

## Parameter Grids

### Shared Exit Parameters (all pipelines)
- atr_sl_mult: [1.0, 1.5, 2.0, 2.5]
- rr_ratio: [1.5, 2.0, 3.0]

### Pipeline-Specific Parameters
- **range_quant**: lookback[5], mr_threshold[3], hold[4] → 720 combos/TF
- **breakout_quant**: lookback[4], squeeze_threshold[3], hold[4] → 576 combos/TF
- **rule_based/smc_quant**: hold[4], ob_proximity[4] → 192 combos/TF
- **volume_profile**: lookback[4], hold[4] → 192 combos/TF

## Methodology
- Pre-compute signals once per lookback/timeframe, then sweep parameters on cached results
- ATR(14) with Wilder smoothing for stop loss distance
- Walk-forward bar-by-bar exit simulation (check SL first, then TP, fall back to max hold)
- Statistics: win rate, Sharpe ratio, profit factor, avg PnL%, total PnL%, buy/sell split

## Date
Results generated: 2026-03-15
