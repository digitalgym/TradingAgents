# Backtest Results

This document consolidates all backtest results across both the LLM pipeline system and the XGBoost quant system.

---

## Table of Contents

1. [LLM Pipeline Results (2026-03-17)](#llm-pipeline-results-2026-03-17)
2. [XGBoost Full Watchlist (2026-03-29)](#xgboost-full-watchlist-batch-run-2026-03-29)
3. [Donchian Breakout — Rule-Based vs XGBoost (2026-03-31)](#donchian-breakout--rule-based-vs-xgboost-2026-03-31)
4. [Flag Continuation — New Strategy (2026-03-31)](#flag-continuation--rule-based-vs-xgboost--optuna-2026-03-31)
5. [Cross-Strategy Rankings](#cross-strategy-rankings)
6. [Methodology](#methodology)

---

## LLM Pipeline Results (2026-03-17)

Original backtest of LLM-based analysis pipelines using the auto-tuner parameter sweep.

- **Data**: 800 bars per timeframe
- **Exit**: ATR(14)-based SL/TP (entry +/- ATR * mult, TP = SL * RR ratio)
- **Timeframes**: D1, H4, H1 (single-TF); D1+H4, D1+H1, H4+H1 (MTF pairs)
- **Tuning**: Grid sweep over indicator-specific + shared exit params

### XAUUSD — LLM Pipelines

| Pipeline | WR% | Sharpe | PF | Trades | Best TF | ATR Mult | RR Ratio |
|----------|------|--------|------|--------|---------|----------|----------|
| rule_based | 75.0% | 2.62 | 4.41 | 12 | D1 | 1.5 | 1.5 |
| range_quant | 63.6% | 1.50 | 3.03 | 22 | H4 | 2.5 | 3.0 |
| breakout_quant | 100% | 162.78 | 12301 | 7 | D1 | 1.5 | 1.5 |
| volume_profile | 51.7% | 0.42 | 3.23 | 286 | H1 | 2.5 | 3.0 |
| **smc_mtf** | **70.7%** | **0.50** | **2.60** | **215** | **D1+H4** | **2.5** | **1.5** |

### BTCUSD — LLM Pipelines

| Pipeline | WR% | Sharpe | PF | Trades | Best TF | ATR Mult | RR Ratio |
|----------|------|--------|------|--------|---------|----------|----------|
| rule_based | 80.0% | 4.19 | 4.61 | 10 | H1 | 1.5 | 1.5 |
| range_quant | 94.7% | 5.58 | 14.08 | 19 | H4 | 1.5 | 1.5 |
| smc_mtf | 40.6% | 0.42 | 2.34 | 180 | D1+H1 | 2.5 | 3.0 |

### LLM Pipeline Detail

#### XAUUSD

- **rule_based**: TF=D1, hold=3, ob_proximity=0.8%, SL=1.5 ATR, RR=1.5. BUY dominant. D1 picks up major structural levels.
- **range_quant**: TF=H4, lookback=30, mr_threshold=65, hold=18, bias_filter=on, SL=2.5 ATR, RR=3.0. 3:1 RR lets winners run.
- **breakout_quant**: TF=D1, lookback=20, squeeze_threshold=60, hold=7, volume_confirm=off, SL=1.5 ATR, RR=1.5. **Caution**: Only 7 trades — likely overfitted.
- **volume_profile**: TF=H1, lookback=75, hold=120, SL=2.5 ATR, RR=3.0. Many signals on H1; PF 3.23 comes from asymmetric RR.
- **smc_mtf**: TF=D1+H4, hold=60 (H4 bars), min_alignment=50, require_channel=on, SL=2.5 ATR, RR=1.5. All 215 signals BUY (gold uptrend). High trade count gives statistical significance. D1+H1: 44.2% WR (too noisy). H4+H1: 43.1% WR (fighting D1 uptrend).

#### BTCUSD

- **rule_based**: TF=H1, hold=72, ob_proximity=0.5%, SL=1.5 ATR, RR=1.5. Tighter OB proximity than gold. Limited sample (10 trades).
- **range_quant**: TF=H4, lookback=40, mr_threshold=65, hold=30, bias_filter=on, SL=1.5 ATR, RR=1.5. Outstanding — BTC ranges well at H4.
- **smc_mtf**: TF=D1+H1, hold=168 (H1 bars), min_alignment=50, require_channel=on, SL=2.5 ATR, RR=3.0. Lower WR but profitable via 3:1 RR.

### LLM Pipeline Parameter Grids

- **Shared exit**: atr_sl_mult [1.0, 1.5, 2.0, 2.5], rr_ratio [1.5, 2.0, 3.0]
- **rule_based/smc_quant**: hold[4], ob_proximity[4] -> 192 combos/TF
- **range_quant**: lookback[5], mr_threshold[3], hold[4], bias_filter[2] -> 1440 combos/TF
- **breakout_quant**: lookback[4], squeeze_threshold[3], hold[4], volume_confirm[2] -> 1152 combos/TF
- **volume_profile**: lookback[4], hold[4] -> 192 combos/TF
- **smc_mtf**: hold[4], min_alignment[3], require_confirmation[2], require_channel[2] -> 576 combos/TF pair

---

## XGBoost Full Watchlist Batch Run (2026-03-29)

First full batch training across 17 pairs x 2 timeframes x 5 strategies (donchian_breakout not yet created).

- **Data**: 2000 bars (H4), 890 bars (D1)
- **Training**: Walk-forward purged K-fold (5 folds, 10-bar purge gap)
- **Strategies**: trend_following, mean_reversion, breakout, smc_zones, volume_profile_strat
- **Viable threshold**: Sharpe > 0, WR >= 40%, PF >= 0.8, trades >= 5

### Metals — XGBoost Default Params

#### XAUUSD

| Strategy | TF | WR% | PF | Sharpe | Trades | Viable |
|----------|-----|-----|----|--------|--------|--------|
| trend_following | H4 | 39.4% | 0.95 | -0.012 | 989 | No |
| mean_reversion | H4 | 40.9% | 0.98 | -0.006 | 775 | No |
| breakout | H4 | 39.5% | 0.92 | -0.017 | 1019 | No |
| smc_zones | H4 | 38.2% | 0.89 | -0.024 | 987 | No |
| volume_profile_strat | H4 | 37.9% | 0.87 | -0.031 | 962 | No |

**Note**: XAUUSD D1 was not included in this batch run. All H4 results are non-viable — the base XGBoost strategies with default params struggle on gold H4.

#### XAGUSD

| Strategy | TF | WR% | PF | Sharpe | Trades | Viable |
|----------|-----|-----|----|--------|--------|--------|
| trend_following | D1 | 40.7% | 1.14 | 0.065 | 204 | Yes |
| **mean_reversion** | **D1** | **43.0%** | **1.28** | **0.140** | 165 | **Yes** |
| breakout | D1 | 39.7% | 1.07 | 0.034 | 214 | No |
| smc_zones | D1 | 40.9% | 1.09 | 0.045 | 186 | Yes |
| volume_profile_strat | D1 | 37.2% | 1.02 | 0.012 | 199 | No |
| trend_following | H4 | 40.0% | 1.07 | 0.013 | 956 | No |
| mean_reversion | H4 | 40.1% | 0.94 | -0.014 | 769 | No |
| breakout | H4 | 38.5% | 1.02 | 0.004 | 1000 | No |
| smc_zones | H4 | 43.0% | 1.11 | 0.021 | 967 | Yes |
| volume_profile_strat | H4 | 38.5% | 0.92 | -0.016 | 951 | No |

**Note**: Mean reversion D1 shows highest Sharpe for silver despite XAGUSD being in the exclusion list for live trading. The model was trained but `predict_signal()` will return HOLD in production.

### Forex Majors — XGBoost Default Params (Best per Pair)

| Pair | Best Strategy | TF | WR% | PF | Sharpe | Trades |
|------|---------------|-----|-----|----|--------|--------|
| **EURJPY** | mean_reversion | D1 | 51.6% | 1.35 | 0.160 | 184 |
| **EURJPY** | smc_zones | D1 | 53.2% | 1.37 | 0.153 | 218 |
| EURJPY | breakout | H4 | 44.9% | 1.23 | 0.047 | 1005 |
| GBPJPY | mean_reversion | H4 | 46.5% | 1.21 | 0.051 | 757 |
| GBPJPY | trend_following | H4 | 46.2% | 1.21 | 0.045 | 930 |
| EURGBP | volume_profile_strat | D1 | 43.4% | 1.26 | 0.119 | 205 |
| EURGBP | mean_reversion | H4 | 43.2% | 1.09 | 0.021 | 812 |
| **AUDNZD** | breakout | D1 | 50.5% | 1.35 | 0.150 | 206 |
| **AUDNZD** | trend_following | D1 | 49.0% | 1.31 | 0.135 | 206 |
| CADJPY | volume_profile_strat | H4 | 44.7% | 1.24 | 0.051 | 971 |
| CADJPY | mean_reversion | H4 | 43.9% | 1.17 | 0.042 | 788 |
| EURAUD | trend_following | D1 | 43.0% | 1.09 | 0.043 | 200 |

### EURJPY Deep Dive — All 7 Strategies (2026-03-31)

EURJPY is the most consistent pair: all original 5 strategies viable on D1, and the 2 new strategies also viable. Full results:

#### EURJPY D1 — XGBoost Default Params

| Strategy | WR% | PF | Sharpe | Trades | Viable |
|----------|-----|----|--------|--------|--------|
| **flag_continuation** | **53.6%** | **1.59** | **0.207** | 252 | **Yes** |
| mean_reversion | 51.6% | 1.35 | 0.160 | 184 | Yes |
| smc_zones | 53.2% | 1.37 | 0.153 | 218 | Yes |
| trend_following | 50.5% | 1.26 | 0.114 | 214 | Yes |
| donchian_breakout | 50.4% | 1.29 | 0.111 | 258 | Yes |
| breakout | 50.9% | 1.22 | 0.096 | 222 | Yes |
| volume_profile_strat | 48.0% | 1.08 | 0.041 | 200 | Yes |

**7 out of 7 viable on D1.** Flag continuation leads with highest Sharpe (0.207) and PF (1.59).

#### EURJPY H4 — XGBoost Default Params

| Strategy | WR% | PF | Sharpe | Trades | Viable |
|----------|-----|----|--------|--------|--------|
| flag_continuation | 47.6% | 1.34 | 0.079 | 757 | Yes |
| donchian_breakout | 44.8% | 1.21 | 0.054 | 694 | Yes |
| smc_zones | 45.7% | 1.24 | 0.051 | 946 | Yes |
| breakout | 44.9% | 1.23 | 0.047 | 1005 | Yes |
| trend_following | 44.5% | 1.15 | 0.034 | 931 | Yes |
| volume_profile_strat | 44.4% | 1.15 | 0.034 | 966 | Yes |
| mean_reversion | 42.6% | 1.06 | 0.015 | 753 | Yes |

**7 out of 7 viable on H4.** 14/14 total — no other pair achieves this.

#### EURJPY Rule-Based

| Strategy | TF | Trades | W/L | WR% | PF | P/L |
|----------|-----|--------|-----|-----|----|-----|
| Donchian | D1 | 5 | 3/2 | 60.0% | 4.83 | +15.41 |
| Donchian | H4 | 89 | 20/69 | 22.5% | 0.81 | -12.05 |
| Flag | D1 | 2 | 0/2 | 0.0% | 0.00 | -3.77 |
| Flag | H4 | 5 | 0/5 | 0.0% | 0.00 | -2.41 |

Rule-based struggles on EURJPY — not enough directional bias for mechanical breakout rules (unlike metals which have a strong uptrend). XGBoost adds significant value here by learning which setups work.

#### EURJPY Optuna-Tuned (40 trials each)

| Strategy | TF | Trades | WR% | PF | Sharpe | Key Params |
|----------|-----|--------|-----|----|--------|------------|
| **flag_continuation** | **H4** | 23 | 73.9% | 22.78 | **3.376** | depth=3, SL=1.25 ATR, TP=2.75 ATR, threshold=0.73, hold=20 |
| **flag_continuation** | **D1** | 21 | 61.9% | 6.17 | **2.125** | depth=4, SL=1.89 ATR, TP=3.46 ATR, threshold=0.75, hold=30 |
| donchian_breakout | D1 | 241 | 95.9% | 631 | 1.157 | depth=5, SL=2.84 ATR, TP=2.36 ATR, threshold=0.58, hold=25 |
| smc_zones | D1 | 177 | 42.4% | 1.15 | 0.071 | depth=7, SL=1.35 ATR, TP=3.32 ATR, threshold=0.64, hold=15 |
| mean_reversion | D1 | — | — | — | — | No valid trials (regime filter blocks all folds) |

**Note**: Flag continuation tuned H4 achieved Sharpe 3.376 but only 23 trades — low sample. Donchian D1 shows 95.9% WR which is extreme overfitting. SMC zones barely improved from default. Mean reversion D1 could not be tuned — the regime filter (requires ranging market) blocked all folds during Optuna search with varying risk params.

#### EURJPY Summary

EURJPY is the ideal XGBoost pair because it has clean trends AND well-defined ranges, making every strategy archetype work:
- **Best default**: Flag Continuation D1 (Sharpe 0.207, PF 1.59)
- **Best tuned**: Flag Continuation H4 (Sharpe 3.376, but only 23 trades)
- **Most reliable**: SMC Zones D1 or Mean Reversion D1 (150-220 trades, proven in batch run)
- **Rule-based**: Not recommended — EURJPY lacks the persistent directional bias that makes mechanical breakout rules profitable on metals

### Crypto — XGBoost Default Params

| Pair | Best Strategy | TF | WR% | PF | Sharpe | Trades |
|------|---------------|-----|-----|----|--------|--------|
| BTCUSD | breakout | H4 | 39.4% | 1.05 | 0.012 | 980 |
| BTCUSD | smc_zones | H4 | 39.3% | 1.05 | 0.013 | 893 |
| **ETHUSD** | breakout | H4 | 43.6% | 1.21 | 0.042 | 984 |
| **ETHUSD** | smc_zones | H4 | 43.0% | 1.19 | 0.041 | 900 |
| ETHUSD | trend_following | H4 | 42.4% | 1.15 | 0.033 | 893 |

**Observation**: BTCUSD is difficult for XGBoost across the board (no viable D1 results). ETHUSD performs better on H4 with multiple viable strategies.

### Full Watchlist — D1 vs H4 Pattern

Across all pairs, a clear pattern emerges:

- **D1**: Higher Sharpe but fewer trades (150-230 per strategy). Better for selective, higher-quality signals.
- **H4**: More trades (750-1000+) but lower Sharpe. Most H4 combos cluster around breakeven (Sharpe -0.03 to +0.05).
- **GBPJPY D1**: All 5 strategies non-viable. GBPJPY only works on H4.
- **EURJPY D1**: All 5 strategies viable — the most consistent pair in the watchlist.

---

## Donchian Breakout — Rule-Based vs XGBoost (2026-03-31)

Comparison of the Donchian Channel Breakout strategy in two modes:
- **Rule-based**: Pure mechanical signals — Donchian channel break + SMA trend bias + regime gate (ADX > 25 or BB squeeze expanding). No ML.
- **XGBoost**: Walk-forward trained model with 13 Donchian-specific features on top of 24 base technicals. Regime gate as hard filter on ML predictions.

Data: 2000 bars (H4), 890 bars (D1). Exit: ATR-based SL, 3:1 RR target.

### XAUUSD (Gold)

| Mode | TF | Trades | W/L | WR% | PF | Sharpe | P/L | Viable |
|------|----|--------|-----|-----|----|--------|-----|--------|
| **Rule-based** | **D1** | 32 | 19/13 | **59.4%** | **3.25** | — | +$3,910 | **Yes** |
| Rule-based | H4 | 58 | 21/37 | 36.2% | 1.22 | — | +$585 | Marginal |
| XGBoost | D1 | 231 | 108/123 | 46.8% | 1.24 | 0.10 | — | Yes |
| XGBoost | H4 | 637 | 257/380 | 40.3% | 0.94 | -0.02 | — | No |

### XAGUSD (Silver)

| Mode | TF | Trades | W/L | WR% | PF | Sharpe | P/L | Viable |
|------|----|--------|-----|-----|----|--------|-----|--------|
| **Rule-based** | **D1** | 19 | 14/5 | **73.7%** | **7.85** | — | +$126 | **Yes** |
| Rule-based | H4 | 27 | 7/20 | 25.9% | 0.75 | — | -$11 | No |
| XGBoost | D1 | 222 | 80/142 | 36.0% | 0.82 | -0.08 | — | No |
| XGBoost | H4 | 734 | 330/404 | 45.0% | 1.27 | 0.05 | — | Yes |

### Donchian Observations

1. **Rule-based D1 dominates both metals** — simple Donchian breakouts with trend bias capture the precious metals uptrend effectively. High WR, high PF, low trade count.
2. **XGBoost helps on H4, hurts on D1** — ML filtering improves signal quality on the noisier H4 timeframe. On D1, the model overfits (only 2 usable folds from 890 bars).
3. **Recommendation**: Rule-based on D1, XGBoost on H4.
4. **Silver-lead confirmation** not tested (no cross-pair data passed). May improve H4 rule-based results.
5. **Strong long bias**: Gold D1 = 33L/1S, Silver D1 = 19L/0S — reflects sustained uptrend in precious metals.

---

## Flag Continuation — Rule-Based vs XGBoost + Optuna (2026-03-31)

New strategy detecting: strong impulse move ("pole") -> consolidation ("flag") -> breakout in trend direction.

- **Features**: 40 total (24 base technical + 16 flag-specific: impulse size/direction/linearity/body ratio, consolidation range/retrace/ATR ratio/slope/narrowing, breakout distance/volume/bar strength, trend alignment, ADX proxy)
- **Regime gate**: ADX proxy > 25 (trending only)
- **Direction lock**: Only enters in impulse direction (continuation only, no reversals)

### Default XGBoost Results (before tuning)

| Symbol | TF | Trades | WR% | PF | Sharpe | Viable |
|--------|-----|--------|-----|----|--------|--------|
| XAUUSD | D1 | 224 | 46.4% | 1.30 | 0.12 | Yes |
| XAUUSD | H4 | 699 | 42.1% | 1.03 | 0.01 | Marginal |
| XAGUSD | D1 | 245 | 35.5% | 0.84 | -0.07 | No |
| XAGUSD | H4 | 782 | 45.4% | 1.15 | 0.03 | Yes |

### Rule-Based Results (no ML)

Flag patterns are rare — the mechanical version is very selective.

| Symbol | TF | Trades | W/L | WR% | PF | P/L |
|--------|-----|--------|-----|-----|----|-----|
| XAUUSD | D1 | 2 | 0/2 | 0.0% | 0.00 | -$124 |
| XAUUSD | H4 | 6 | 4/2 | 66.7% | 4.87 | +$255 |
| XAGUSD | D1 | 5 | 0/5 | 0.0% | 0.00 | -$3 |
| XAGUSD | H4 | 1 | 0/1 | 0.0% | 0.00 | -$1 |

**Rule-based verdict**: Too few signals with default params. XAUUSD H4 promising (4/6 wins, PF 4.87) but insufficient sample size. Needs parameter relaxation for more signals.

### Optuna-Tuned XGBoost Results (40 trials each)

| Symbol | TF | Trades | WR% | PF | Sharpe | Key Params |
|--------|-----|--------|-----|----|--------|------------|
| **XAUUSD** | **D1** | 86 | 95.3% | 552 | **1.57** | depth=2, lr=0.06, n=400, SL=2.67 ATR, TP=2.96 ATR, threshold=0.69 |
| **XAUUSD** | **H4** | 439 | 91.3% | 147 | **0.53** | depth=6, lr=0.05, n=450, SL=2.65 ATR, TP=3.49 ATR, threshold=0.71 |
| **XAGUSD** | **H4** | 231 | 97.8% | 13889 | **0.76** | depth=7, lr=0.08, n=150, SL=1.97 ATR, TP=3.42 ATR, threshold=0.73 |

**Caution**: Very high WR/PF likely reflects in-sample overfitting. Walk-forward mitigates but doesn't eliminate. Sharpe values (0.5-1.6) are more trustworthy. Monitor live performance closely.

### Tuned Parameter Patterns

Optuna converged on consistent patterns across all three combos:
- **Wide SL** (1.97-2.67 ATR) — flags need room to breathe during consolidation
- **Wide TP** (2.96-3.49 ATR) — continuation moves tend to be large
- **High signal threshold** (0.69-0.73) — only take high-conviction flag patterns
- **Short max hold** (10-25 bars) — flags should resolve quickly once breakout starts
- **Near-zero trailing** — let RR-based TP work, don't trail prematurely

---

## Cross-Strategy Rankings

### Best Breakout Strategies for Precious Metals (all tests, 2026-03-31)

Ranked by confidence level (Sharpe, trade count, overfitting risk).

#### XAUUSD (Gold)

| Rank | Strategy | TF | WR% | PF | Sharpe | Trades | Confidence |
|------|----------|-----|-----|----|--------|--------|------------|
| 1 | Flag Continuation (Optuna) | D1 | 95.3% | 552 | 1.57 | 86 | Low (overfit risk) |
| 2 | Donchian Rule-Based | D1 | 59.4% | 3.25 | — | 32 | **High** (no ML) |
| 3 | Flag Continuation (Optuna) | H4 | 91.3% | 147 | 0.53 | 439 | Medium |
| 4 | Flag Continuation (default) | D1 | 46.4% | 1.30 | 0.12 | 224 | Medium |
| 5 | Donchian XGBoost | D1 | 46.8% | 1.24 | 0.10 | 231 | Medium |
| 6 | Donchian Rule-Based | H4 | 36.2% | 1.22 | — | 58 | Marginal |
| 7 | Donchian XGBoost | H4 | 40.3% | 0.94 | -0.02 | 637 | Not viable |

#### XAGUSD (Silver)

| Rank | Strategy | TF | WR% | PF | Sharpe | Trades | Confidence |
|------|----------|-----|-----|----|--------|--------|------------|
| 1 | Flag Continuation (Optuna) | H4 | 97.8% | 13889 | 0.76 | 231 | Low (overfit risk) |
| 2 | Donchian Rule-Based | D1 | 73.7% | 7.85 | — | 19 | **High** (no ML, small sample) |
| 3 | Donchian XGBoost | H4 | 45.0% | 1.27 | 0.05 | 734 | Medium |
| 4 | Flag Continuation (default) | H4 | 45.4% | 1.15 | 0.03 | 782 | Medium |
| 5 | Trend Following XGBoost | D1 | 40.7% | 1.14 | 0.07 | 204 | Medium |

### Best Strategy per Pair (XGBoost, all pairs)

Highest Sharpe viable combo from the full batch run:

| Pair | Strategy | TF | WR% | PF | Sharpe | Trades |
|------|----------|-----|-----|----|--------|--------|
| EURJPY | mean_reversion | D1 | 51.6% | 1.35 | 0.160 | 184 |
| AUDNZD | breakout | D1 | 50.5% | 1.35 | 0.150 | 206 |
| EURJPY | smc_zones | D1 | 53.2% | 1.37 | 0.153 | 218 |
| XAGUSD | mean_reversion* | D1 | 43.0% | 1.28 | 0.140 | 165 |
| AUDNZD | trend_following | D1 | 49.0% | 1.31 | 0.135 | 206 |
| EURGBP | volume_profile_strat | D1 | 43.4% | 1.26 | 0.119 | 205 |
| XAUUSD | flag_continuation | D1 | 46.4% | 1.30 | 0.122 | 224 |
| XAGUSD | flag_continuation | H4 | 45.4% | 1.15 | 0.031 | 782 |
| CADJPY | volume_profile_strat | H4 | 44.7% | 1.24 | 0.051 | 971 |
| ETHUSD | breakout | H4 | 43.6% | 1.21 | 0.042 | 984 |

\* XAGUSD mean_reversion is excluded from live trading (too volatile) despite good backtest.

### Practical Deployment Recommendation

| Symbol | Primary (High Confidence) | Secondary (Needs Live Validation) |
|--------|--------------------------|-----------------------------------|
| XAUUSD | Donchian Rule-Based D1 | Flag Continuation (tuned) D1 + H4 |
| XAGUSD | Donchian Rule-Based D1 | Flag Continuation (tuned) H4, Donchian XGBoost H4 |
| EURJPY | Flag Continuation D1, SMC Zones D1, Mean Reversion D1 | Flag Continuation (tuned) H4 |
| AUDNZD | Breakout D1, Trend Following D1 | — |
| ETHUSD | Breakout H4, SMC Zones H4 | — |
| BTCUSD | Range Quant H4 (LLM pipeline) | XGBoost not viable |

---

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

\* smc_quant uses same base signals as rule_based + LLM reasoning
\*\* Only 7 trades — low statistical confidence

---

## Methodology

### LLM Pipeline Backtesting (2026-03-17)
- Pre-compute signals once per lookback/timeframe, then sweep parameters on cached results
- ATR(14) with Wilder smoothing for stop loss distance
- Walk-forward bar-by-bar exit simulation (check SL first, then TP, fall back to max hold)
- Statistics: win rate, Sharpe ratio, profit factor, avg PnL%, total PnL%, buy/sell split
- Ranking: sorted by Sharpe ratio (risk-adjusted returns)
- Minimum 5 trades required for a config to qualify

### XGBoost Walk-Forward Training (2026-03-29+)
- **Purged K-fold**: 5 sequential folds with 10-bar purge gap between train/test to prevent leakage
- Each fold trains only on past data, tests on future data — no look-ahead bias
- D1 (890 bars): 2 usable folds (train=524/702, test=178 each)
- H4 (2000 bars): 3 usable folds (train=790/1190/1590, test=400 each)
- Model saved after final fold using all available data
- Viable threshold: Sharpe > 0, WR >= 40%, PF >= 0.8, trades >= 5

### Rule-Based Backtesting (Donchian, Flag)
- Bar-by-bar signal generation with indicator warmup
- For each signal: walk forward checking SL/TP hit (max 100 bars hold)
- SL checked before TP on each bar (conservative)
- No ML, no optimization — pure mechanical rules

### Optuna Parameter Tuning
- 40 trials per combo, 300s timeout
- Objective: maximize Sharpe ratio
- Minimum 15 trades per trial to prevent overfitting to handful of "perfect" trades
- Search space: 8 XGBoost hyperparams + 4 risk params (SL mult, TP mult, signal threshold, max hold)
- Walk-forward trainer as objective function (same purged K-fold as batch training)

### Overfitting Risk Levels
- **High confidence**: Rule-based strategies (no ML, no optimization)
- **Medium confidence**: Default XGBoost params (walk-forward validated, no param search)
- **Low confidence**: Optuna-tuned (param search over same data folds — selection bias possible despite walk-forward)

---

## Trained Model Inventory

### Metals Models (as of 2026-03-31)

| Strategy | XAUUSD D1 | XAUUSD H4 | XAGUSD D1 | XAGUSD H4 |
|----------|-----------|-----------|-----------|-----------|
| trend_following | Y | Y | Y | Y |
| mean_reversion | Y | Y | Y* | Y* |
| breakout | Y | Y | Y | Y |
| smc_zones | Y | Y | Y | Y |
| volume_profile_strat | Y | Y | Y | Y |
| donchian_breakout | Y | Y | Y | Y |
| flag_continuation | Y (tuned) | Y (tuned) | Y | Y (tuned) |

\* Mean reversion models exist but XAGUSD excluded from live trading (too volatile).

### All Pairs with Trained Models

Full models trained for: EURJPY, GBPJPY, EURGBP, AUDNZD, CADJPY, EURAUD, XAUUSD, XAGUSD, BTCUSD, ETHUSD, NZDUSD (H4 only for smc_zones + volume_profile_strat).
