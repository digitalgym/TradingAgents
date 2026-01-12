# CLI Commands for Continuous Learning System

## New Commands Available

The continuous learning system adds 5 new CLI commands to your existing `tradingagents` CLI:

### 1. `learning-status` - View System Status

Shows the complete status of the continuous learning system.

```bash
tradingagents learning-status
```

**Output:**

```
═══ CONTINUOUS LEARNING SYSTEM STATUS ═══

Risk Guardrails
┌──────────────────────┬────────┬────────┬─────────────────┐
│ Metric               │  Value │  Limit │ Status          │
├──────────────────────┼────────┼────────┼─────────────────┤
│ Trading Allowed      │     ✅ │      - │ OK              │
│ Consecutive Losses   │      0 │      2 │ ✓               │
│ Daily Loss           │  0.00% │  3.00% │ ✓               │
│ Total Breaches       │      0 │      - │ 📊              │
└──────────────────────┴────────┴────────┴─────────────────┘

Agent Weights (Adaptive)
┌──────────┬────────┬────────┬───────────┐
│ Agent    │ Weight │ Status │ Influence │
├──────────┼────────┼────────┼───────────┤
│ Bull     │  0.450 │ 🔥     │ High      │
│ Bear     │  0.250 │ ❄️     │ Low       │
│ Market   │  0.300 │ →      │ Normal    │
└──────────┴────────┴────────┴───────────┘

Pattern Analysis
┌────────────────────┬─────────┐
│ Metric             │   Value │
├────────────────────┼─────────┤
│ Trades Since Update│   15/30 │
│ Update Needed      │    No ✓ │
└────────────────────┴─────────┘

Portfolio Performance
┌───────────────┬──────────┐
│ Metric        │    Value │
├───────────────┼──────────┤
│ Total Trades  │       42 │
│ Win Rate      │    65.5% │
│ Sharpe Ratio  │     1.85 │
│ Max Drawdown  │     8.50%│
│ Current Equity│ $10,425  │
└───────────────┴──────────┘
```

### 2. `update-patterns` - Run Pattern Analysis

Analyzes trade patterns and updates agent weights.

```bash
tradingagents update-patterns
```

**What it does:**

- Analyzes last 30 days of trades
- Groups trades by setup type, regime, time, confluence
- Identifies winning and losing patterns
- Updates agent weights based on performance
- Shows recommendations

**Output:**

```
═══ PATTERN ANALYSIS & WEIGHT UPDATE ═══

Analyzing Trade Patterns...

Total Trades Analyzed: 42
Overall Win Rate: 65.5%
Overall Avg RR: 1.85
Patterns Found: 8

Pattern Quality:
  🌟 Excellent: 2
  ✓ Good: 3
  → Neutral: 2
  ✗ Poor: 1

Top 5 Patterns:
┌─────────────────────┬────────────┬──────────┬────────┬────────┬─────────┐
│ Pattern             │ Type       │ Win Rate │ Avg RR │ Sample │ Quality │
├─────────────────────┼────────────┼──────────┼────────┼────────┼─────────┤
│ breaker-block       │ setup_type │    75.0% │   2.30 │     12 │ 🌟      │
│ trending-up/normal  │ regime     │    70.0% │   2.10 │     20 │ 🌟      │
│ FVG                 │ setup_type │    60.0% │   1.50 │     10 │ ✓       │
│ high-confluence     │ confluence │    66.7% │   1.80 │      9 │ ✓       │
│ resistance-reject   │ setup_type │    30.0% │   0.50 │     10 │ ✗       │
└─────────────────────┴────────────┴──────────┴────────┴────────┴─────────┘

Recommendations:
  ✓ INCREASE focus on breaker-block (75% win rate, 2.3R avg)
  ✓ INCREASE focus on trending-up/normal (70% win rate, 2.1R avg)
  ✗ AVOID resistance-rejection (30% win rate, 0.5R avg)

Updating Agent Weights...

Agent Weight Changes
┌──────────┬────────────┬────────────┬─────────┬───────────┐
│ Agent    │ Old Weight │ New Weight │  Change │ Direction │
├──────────┼────────────┼────────────┼─────────┼───────────┤
│ Bull     │      0.330 │      0.450 │  +0.120 │ ↑         │
│ Bear     │      0.330 │      0.250 │  -0.080 │ ↓         │
│ Market   │      0.340 │      0.300 │  -0.040 │ ↓         │
└──────────┴────────────┴────────────┴─────────┴───────────┘

Reasoning:
  BULL: INCREASED from 0.33 to 0.45 (win rate: 70%, avg reward: +1.5)
  BEAR: DECREASED from 0.33 to 0.25 (win rate: 45%, avg reward: -0.3)

✅ Learning system updated!
```

### 3. `risk-status` - View Risk Guardrails

Shows detailed risk guardrails status and breach history.

```bash
tradingagents risk-status
```

**Output:**

```
╭─────────────────── Risk Guardrails Status ───────────────────╮
│                                                               │
│ RISK GUARDRAILS STATUS                                        │
│                                                               │
│ Trading Allowed: ✅ YES                                       │
│ Reason: OK                                                    │
│                                                               │
│ Current Metrics:                                              │
│ - Consecutive Losses: 0 / 2                                   │
│ - Daily Loss: 0.00% / 3.00%                                   │
│ - Total Breaches: 0                                           │
│                                                               │
│ Limits:                                                       │
│ - Daily Loss Limit: 3.0%                                      │
│ - Max Consecutive Losses: 2                                   │
│ - Max Position Size: 2.0%                                     │
│ - Cooldown Period: 24 hours                                   │
│                                                               │
│ Status: ✅ All systems normal                                 │
│                                                               │
╰───────────────────────────────────────────────────────────────╯
```

### 4. `regime` - Detect Market Regime

Detects the current market regime for a symbol.

```bash
# Default (XAUUSD)
tradingagents regime

# Specific symbol
tradingagents regime --symbol XAGUSD

# More historical data
tradingagents regime --symbol XAUUSD --days 200
```

**Output:**

```
═══ REGIME DETECTION: XAUUSD ═══

Fetching price data from MT5...
Loaded 100 bars

Current Regime: XAUUSD
┌───────────────┬──────────────┬─────────────┐
│ Component     │        Value │ Description │
├───────────────┼──────────────┼─────────────┤
│ Market Trend  │  trending-up │ 📈          │
│ Volatility    │       normal │ →           │
│ Expansion     │    expansion │ 📊          │
└───────────────┴──────────────┴─────────────┘

Description: Trending upward with normal volatility in expansion phase

Trading Implications:
  Trend Trading: ✅ Favorable
  Range Trading: ❌ Not Favorable
  Position Size Adjustment: 1.00x
```

### 5. `similar-trades` - Find Similar Historical Trades

Finds similar historical trades based on setup and regime.

```bash
# Basic search
tradingagents similar-trades --symbol XAUUSD --direction BUY

# With setup type
tradingagents similar-trades --symbol XAUUSD --direction BUY --setup breaker-block

# With regime filter
tradingagents similar-trades --symbol XAUUSD --direction BUY --regime trending-up

# More results
tradingagents similar-trades --symbol XAUUSD --direction BUY --limit 10
```

**Output:**

```
═══ SIMILAR TRADES: XAUUSD BUY ═══

Searching historical trades...

Historical Performance
┌─────────────────┬─────────┐
│ Metric          │   Value │
├─────────────────┼─────────┤
│ Similar Trades  │       8 │
│ Win Rate        │   75.0% │
│ Avg RR          │    2.30 │
│ Best Trade      │   +3.50R│
│ Worst Trade     │   -1.00R│
│ Confidence Adj  │   +0.15 │
└─────────────────┴─────────┘

Top 5 Similar Trades:
┌───┬─────────┬─────────┬───────────────┬──────────────────┬────────────┐
│ # │ Outcome │      RR │ Setup         │ Regime           │ Similarity │
├───┼─────────┼─────────┼───────────────┼──────────────────┼────────────┤
│ 1 │ ✅ WIN  │  +2.50R │ breaker-block │ trending-up/norm │       0.92 │
│ 2 │ ✅ WIN  │  +3.00R │ breaker-block │ trending-up/norm │       0.89 │
│ 3 │ ✅ WIN  │  +2.00R │ breaker-block │ trending-up/high │       0.85 │
│ 4 │ ❌ LOSS │  -1.00R │ breaker-block │ trending-up/extr │       0.82 │
│ 5 │ ✅ WIN  │  +1.80R │ FVG           │ trending-up/norm │       0.78 │
└───┴─────────┴─────────┴───────────────┴──────────────────┴────────────┘

Recommendation:
  Found 8 similar trades in trending-up / normal. STRONG historical
  performance: 75% win rate (2.30R avg). INCREASE confidence by +0.1 to +0.2.
```

## Integration with Existing Commands

### Enhanced `analyze` Command

Your existing `analyze` command can be enhanced to use continuous learning:

```bash
# Your existing analyze command
tradingagents analyze

# Now internally uses:
# 1. Risk guardrails check (Phase 5)
# 2. Regime detection (Phase 2)
# 3. RAG similar trades (Phase 3)
# 4. Agent weights (Phase 4)
# 5. Reward calculation after close (Phase 1)
```

### Enhanced `decisions` Command

Your existing `decisions` command now stores more data:

```bash
# List decisions (unchanged)
tradingagents decisions list

# Close decision (now calculates reward automatically)
tradingagents decisions close XAUUSD_20260111_140000

# Output now includes:
# ✅ Decision closed: XAUUSD_20260111_140000
#    Entry: 2650.0 → Exit: 2690.0
#    P&L: +1.51% (✓ Correct)
#    Risk-Reward: +2.00R (planned: 2.00R)
#    Reward Signal: +1.85  ← NEW!

# Stats (unchanged)
tradingagents decisions stats
```

## Typical Workflow

### Morning Routine

```bash
# Check system status
tradingagents learning-status

# Check risk status
tradingagents risk-status

# Detect regime for your symbols
tradingagents regime --symbol XAUUSD
tradingagents regime --symbol XAGUSD
```

### Before Trading

```bash
# Find similar historical trades
tradingagents similar-trades --symbol XAUUSD --direction BUY --setup breaker-block

# Run analysis (uses learning automatically)
tradingagents analyze
```

### After 30 Trades

```bash
# Update patterns and weights
tradingagents update-patterns

# Check new status
tradingagents learning-status
```

### After Circuit Breaker

```bash
# Check why trading was halted
tradingagents risk-status

# View breach history
# (shown in risk-status output)

# Wait for cooldown to expire
# System will automatically resume after 24 hours
```

## Command Options Summary

| Command           | Options                                                     | Description                           |
| ----------------- | ----------------------------------------------------------- | ------------------------------------- |
| `learning-status` | None                                                        | View complete system status           |
| `update-patterns` | None                                                        | Run pattern analysis & update weights |
| `risk-status`     | None                                                        | View risk guardrails details          |
| `regime`          | `--symbol`, `--days`                                        | Detect market regime                  |
| `similar-trades`  | `--symbol`, `--direction`, `--setup`, `--regime`, `--limit` | Find similar trades                   |

## Help

Get help for any command:

```bash
tradingagents learning-status --help
tradingagents update-patterns --help
tradingagents regime --help
tradingagents similar-trades --help
tradingagents risk-status --help
```

## Examples

### Example 1: Check Before Trading

```bash
# Morning check
$ tradingagents learning-status

# Output shows:
# - Trading allowed: ✅
# - Agent weights: Bull 0.45, Bear 0.25
# - Pattern update: Not needed (15/30 trades)
# - Win rate: 65.5%

# Check regime
$ tradingagents regime --symbol XAUUSD

# Output shows:
# - Trending up / normal volatility
# - Favorable for trend trading
# - Position size: 1.00x (no adjustment)

# Find similar trades
$ tradingagents similar-trades --symbol XAUUSD --direction BUY

# Output shows:
# - 8 similar trades found
# - 75% win rate
# - Confidence adjustment: +0.15
```

### Example 2: After 30 Trades

```bash
# Update patterns
$ tradingagents update-patterns

# Output shows:
# - Breaker blocks: 75% win rate (excellent)
# - Resistance rejections: 30% win rate (poor)
# - Bull agent weight increased to 0.45
# - Bear agent weight decreased to 0.25

# Check new status
$ tradingagents learning-status

# Output shows updated weights and stats
```

### Example 3: After Circuit Breaker

```bash
# Check what happened
$ tradingagents risk-status

# Output shows:
# ⛔ Trading blocked: CONSECUTIVE LOSSES
# - Consecutive losses: 2/2
# - Cooldown until: 2026-01-12 14:30:00
# - Recent breaches: 2026-01-11 14:30: consecutive_losses

# Try to trade
$ tradingagents analyze

# Output:
# ⛔ Trading Blocked: CONSECUTIVE LOSSES
# System in cooldown. No analysis will be performed.
```

## Notes

- All commands work with your existing MT5 connection
- Data is stored in `examples/` directory
- Commands are safe to run anytime (read-only except `update-patterns`)
- `update-patterns` can be run manually or waits for 30 trades
- Risk guardrails are always active and cannot be bypassed

---

**You now have full CLI access to the continuous learning system!** 🚀
