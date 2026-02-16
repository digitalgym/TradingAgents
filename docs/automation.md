# TradingAgents Automation System

This document explains how the automated trading system works, including triggers, actions, configuration, and safety controls.

## Overview

The automation system has three layers:

1. **Orchestration Layer** (`PortfolioAutomation`) - coordinates all workflows
2. **Scheduling Layer** (`DailyScheduler`) - manages time-based execution
3. **API Layer** - REST endpoints and CLI commands for user control

## Daily Cycles

The system runs three automated cycles each day:

| Cycle | Default Time (UTC) | Purpose |
|-------|-------------------|---------|
| Morning Analysis | 08:00 | Analyze markets and execute new trades |
| Midday Review | 13:00 | Adjust stop losses on open positions |
| Evening Reflection | 20:00 | Process closed trades and learn from outcomes |

## What Each Cycle Does

### Morning Analysis

1. Validates MT5 connection and checks guardrails (daily loss limit, position limits)
2. For each enabled symbol:
   - Checks position limits and correlation group limits
   - Runs multi-agent analysis (market, news, fundamentals, technicals)
   - Calculates entry, stop loss, and take profit using ATR
   - Determines position size based on risk budget
3. Ranks opportunities by confidence score
4. Executes top opportunities (in FULL_AUTO mode)

### Midday Review

1. Reviews all open positions
2. For each position:
   - Calculates ATR-based trailing stop
   - Checks breakeven conditions (moves SL to entry if 1%+ profit)
   - Applies stop loss adjustments automatically (FULL_AUTO) or queues for approval (SEMI_AUTO)

### Evening Reflection

1. Checks MT5 history for closed trades
2. For each closed trade:
   - Records exit price and P&L
   - Updates decision record with outcome
   - Feeds results into learning system
3. Stores patterns in memory for future analysis

## Execution Modes

| Mode | Behavior |
|------|----------|
| `FULL_AUTO` | Executes trades automatically without user confirmation |
| `SEMI_AUTO` | Generates signals, requires user approval before execution |
| `PAPER` | Simulation mode - logs decisions without placing real orders |

## Configuration

Configuration is stored in `portfolio_config.yaml`:

```yaml
# Symbols to trade
symbols:
  - symbol: XAUUSD
    max_positions: 1           # Max concurrent positions for this symbol
    risk_budget_pct: 2.0       # Max risk % per trade
    correlation_group: metals  # For correlation-aware limits
    timeframes: [1H, 4H, D1]   # Analysis timeframes
    enabled: true
    min_confidence: 0.6        # Minimum confidence to execute

# Portfolio-level limits
max_total_positions: 4         # Total positions across all symbols
max_daily_trades: 3            # Max new trades per day
max_correlation_group_positions: 2

# Risk management
total_risk_budget_pct: 6.0     # Total portfolio risk %
daily_loss_limit_pct: 3.0      # Circuit breaker threshold
max_consecutive_losses: 2
cooldown_hours: 24             # Cooldown after circuit breaker trips

# Execution settings
execution_mode: full_auto
use_atr_stops: true
atr_stop_multiplier: 2.0       # ATR multiplier for stop loss
atr_trailing_multiplier: 1.5   # ATR multiplier for trailing stops
risk_reward_ratio: 2.0

# Schedule (24-hour format, UTC)
schedule:
  morning_analysis_hour: 8
  midday_review_hour: 13
  evening_reflect_hour: 20
  timezone: UTC
```

## Safety Controls

### Circuit Breakers

- **Daily Loss Limit**: If daily losses exceed `daily_loss_limit_pct`, trading stops for 24 hours
- **Consecutive Loss Limit**: If `max_consecutive_losses` losses occur in a row, trading pauses

### Position Limits

- **Total Positions**: `max_total_positions` caps concurrent positions across all symbols
- **Per-Symbol Limit**: Each symbol has its own `max_positions` setting
- **Correlation Groups**: `max_correlation_group_positions` prevents over-concentration in correlated assets (e.g., multiple metals)

### Pre-Execution Validation

Before any trade executes, the system validates:

- Stop loss is on the correct side of entry (below for BUY, above for SELL)
- Position size is within broker limits
- Risk amount doesn't exceed budget
- Confidence score meets minimum threshold

## Triggers

### Automatic (Time-Based)

The scheduler daemon runs continuously and triggers cycles at their scheduled times. It checks every 60 seconds and ensures each cycle runs only once per day.

### Manual

Trigger cycles manually via API or CLI:

```bash
# CLI
portfolio trigger morning
portfolio trigger midday
portfolio trigger evening

# API
POST /api/portfolio/trigger?cycle_type=morning
```

## Starting and Stopping

### Start Automation

```bash
# CLI
portfolio start

# API
POST /api/portfolio/start
```

### Stop Automation

```bash
# CLI
portfolio stop

# API
POST /api/portfolio/stop
```

### Check Status

```bash
# CLI
portfolio status

# API
GET /api/portfolio/status
```

Returns:
```json
{
  "running": true,
  "pid": 12345,
  "enabled": true,
  "last_start": "2026-01-23T10:30:00"
}
```

## Decision Tracking

Every automated trade is recorded with:

- Symbol, direction, entry/SL/TP prices
- Volume and execution time
- Analysis context and confidence score
- Exit price and P&L when closed
- Outcome (win/loss)

Query decisions via:
```
GET /api/decisions?status=open
GET /api/decisions/stats
```

## Architecture

```
┌─────────────────────────────────────────────┐
│           API / CLI / Frontend              │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│         DailyScheduler (Daemon)             │
│   Polls every 60s, triggers at set times    │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│       PortfolioAutomation Orchestrator      │
│   Morning → Midday → Evening cycles         │
└─────────────────────┬───────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐   ┌────▼────┐  ┌────▼────┐
   │  Agent  │   │  Risk   │  │   MT5   │
   │ Analysis│   │Guardrails│  │ Broker  │
   └─────────┘   └─────────┘  └─────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `tradingagents/automation/portfolio_automation.py` | Main orchestrator |
| `tradingagents/automation/daily_scheduler.py` | Scheduler daemon |
| `tradingagents/automation/portfolio_config.py` | Configuration schema |
| `tradingagents/automation/correlation_manager.py` | Correlation tracking |
| `portfolio_config.yaml` | Live configuration |
| `portfolio_state.json` | Runtime state |
