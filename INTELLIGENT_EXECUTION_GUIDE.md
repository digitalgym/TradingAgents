# Intelligent Trade Execution System

## Problem Solved

**Before:** CLI `analyze` gives you a detailed trading plan with staged entries, dynamic stops, and scale-out targets, but when you go to execute, you only get "buy at market" - losing all the intelligence from the plan.

**After:** The system now:

1. **Parses the plan** - Extracts entry levels, stops, targets, conditions
2. **Decides order types** - Market vs limit based on current price and urgency
3. **Executes staged entries** - Places multiple tranches with proper limits
4. **Manages dynamic stops** - Trails stops based on plan rules
5. **Reviews and adapts** - Monitors conditions and recommends adjustments

## Complete Workflow

### Step 1: Get Analysis from CLI

```bash
tradingagents analyze
# Enter symbol: COPPER-C
# Follow prompts...
```

**Output includes detailed plan:**

```
3. Refined Trader's Plan

• Entry (Staged on $75-83 Dips):
   • Tranche 1: 2% at $77-80 (volume>80M + RSI<65)
   • Tranche 2: 1.5% at $75 (MACD bullish + DXY<104)
   • Tranche 3: 1% at $83 retest (geo escalation)

• Stop-Loss: Dynamic trailing, 1.5-2x ATR (~$72 initial,
  trails to $75 post-$82 close/$78 post-$87)

• Profit-Taking:
   • 30% at $87
   • 30% at $100
   • Trail 40% to $108-125
```

**Save this output to a file:**

```bash
# Copy the plan section to a file
# examples/plans/copper_plan_20260113.txt
```

### Step 2: Parse and Execute the Plan

```bash
# Dry run first (no real orders)
tradingagents execute-plan examples/plans/copper_plan_20260113.txt --dry-run

# When ready, execute live
tradingagents execute-plan examples/plans/copper_plan_20260113.txt --live
```

**What happens:**

```
═══ EXECUTE TRADING PLAN ═══

PARSED TRADING PLAN: COPPER-C BUY
═══════════════════════════════════════════════════════════════════

POSITION SIZING:
  Total: 3.0% | Max: 5.0%

ENTRY STRATEGY (STAGED):
  Tranche 1: 2.0% at $77.00-$80.00
    Conditions: volume>80M, RSI<65 cooldown
  Tranche 2: 1.5% at $75.00
    Conditions: MACD bullish hold, DXY<104
  Tranche 3: 1.0% at $83.00
    Conditions: geo escalation confirm

STOP LOSS (TRAILING):
  Initial: $72.00
  ATR Multiple: 1.5x
  Trail to $75.00 after $82.00
  Trail to $78.00 after $87.00

TAKE PROFIT TARGETS:
  TP1: 30% at $87.00 (first resistance)
  TP2: 30% at $100.00 (consensus short-term)
  TP3: 40% at $108.00 (trailing target)

═══════════════════════════════════════════════════════════════════

Current Price: $78.50

Order Type Decisions:

┌──────────┬──────┬──────────────┬────────────┬─────────┐
│ Tranche  │ Size │ Target Price │ Order Type │ Urgency │
├──────────┼──────┼──────────────┼────────────┼─────────┤
│ 1        │ 2.0% │ $78.50       │ MARKET     │ high    │
│ 2        │ 1.5% │ $75.00       │ LIMIT      │ low     │
│ 3        │ 1.0% │ $83.00       │ LIMIT      │ low     │
└──────────┴──────┴──────────────┴────────────┴─────────┘

Executing Tranches:

Tranche 1: 2.0% at $78.50
  ✓ Order placed: #123456 (MARKET)
  ✓ Filled at $78.52

Tranche 2: 1.5% at $75.00
  ✓ Order placed: #123457 (LIMIT)
  ⏳ Pending fill

Tranche 3: 1.0% at $83.00
  ✓ Order placed: #123458 (LIMIT)
  ⏳ Pending fill

✓ Plan execution initiated: COPPER-C_BUY_20260113_143000
Monitor with: tradingagents monitor-plan COPPER-C_BUY_20260113_143000
```

### Step 3: Monitor the Plan

```bash
tradingagents monitor-plan COPPER-C_BUY_20260113_143000
```

**Output:**

```
═══ MONITORING PLAN: COPPER-C_BUY_20260113_143000 ═══

═══════════════════════════════════════════════════════════════════
STAGED ENTRY STATUS: COPPER-C_BUY_20260113_143000
═══════════════════════════════════════════════════════════════════

Progress: 1/3 tranches filled
Position: 2.0% (0.020 lots)
Avg Entry: $78.52

Status Breakdown:
  Filled:    1
  Active:    2 (orders placed)
  Pending:   0
  Cancelled: 0
  Skipped:   0

Tranche Details:
  ✅ Tranche 1: 2.0% at $78.50
     Filled at $78.52 (0.020 lots)
  ⏳ Tranche 2: 1.5% at $75.00
     Order #123457 active
  ⏳ Tranche 3: 1.0% at $83.00
     Order #123458 active

═══════════════════════════════════════════════════════════════════

Current Price: $79.20

✓ No adjustments needed
```

### Step 4: Review and Adapt

```bash
tradingagents review-plan COPPER-C_BUY_20260113_143000
```

**Output:**

```
═══ REVIEWING PLAN: COPPER-C_BUY_20260113_143000 ═══

Current Price: $82.00

═══════════════════════════════════════════════════════════════════
PLAN REVIEW - 2026-01-13T14:35:00
═══════════════════════════════════════════════════════════════════

Total Adjustments: 2
  High Priority: 1
  Medium Priority: 1
  Low Priority: 0

🔴 HIGH PRIORITY ADJUSTMENTS:

  STOP: Position up 4.4%. Move stop to breakeven to protect profit.
    Current: 72.0
    Recommended: 78.60
    Confidence: 85%

🟡 MEDIUM PRIORITY ADJUSTMENTS:

  ENTRY: Price moved 9.3% above Tranche 2 entry. Consider cancelling
    or adjusting limit up.
    Current: 75.0
    Recommended: 81.0

═══════════════════════════════════════════════════════════════════
```

## Key Features

### 1. Intelligent Order Type Selection

The system decides **market vs limit** based on:

- **Current price vs target** - How far away is the entry?
- **Price range** - Are we in the acceptable range?
- **Urgency** - Is this a time-sensitive opportunity?
- **Spread** - Is the spread too wide for market orders?

**Decision Logic:**

```
Market Order:
- Price at or better than target
- In favorable part of range (lower 30% for buys)
- Within 0.5% of target with good conditions

Limit Order:
- Price >0.5% from target
- In range but waiting for better entry
- High spread (>0.1%)
```

### 2. Staged Entry Management

Tracks multiple tranches:

- **Pending** - Not yet placed
- **Active** - Order placed, waiting for fill
- **Filled** - Executed successfully
- **Cancelled** - Order cancelled
- **Skipped** - Conditions not met

**Automatic adjustments:**

- If price moves >2% away from remaining tranches → recommend cancelling/adjusting
- If first tranche fills favorably → accelerate remaining tranches
- Calculates average entry price across all fills

### 3. Dynamic Stop Management

Implements plan's stop rules:

- **Initial stop** - ATR-based or explicit price
- **Trailing rules** - "Trail to $75 after $82"
- **Breakeven moves** - Auto-suggest when in profit
- **Tightness warnings** - Alert if stop too close

**Auto-updates:**

```
Price: $78 → $82 → $87
Stop:  $72 → $75 → $78
       (initial) (trail 1) (trail 2)
```

### 4. Plan Review System

Monitors and recommends adjustments:

**Entry Review:**

- Price moved significantly from planned entries?
- Should adjust limits or enter at market?

**Stop Review:**

- In profit >2%? → Move to breakeven
- Stop too tight? → Widen to avoid premature exit

**Target Review:**

- Approaching TP? → Prepare to scale out
- Moved past TP? → Take profit now

**Time Review:**

- In trade >7 days with minimal profit? → Consider exit
- In trade >14 days with good profit? → Take partial

## CLI Commands

### Execute Plan

```bash
# Dry run (simulated)
tradingagents execute-plan <plan_file> --dry-run

# Live execution
tradingagents execute-plan <plan_file> --live

# With symbol override
tradingagents execute-plan <plan_file> --symbol XAUUSD --live
```

### Monitor Plan

```bash
tradingagents monitor-plan <plan_id>
```

Shows:

- Tranche fill status
- Current position size
- Average entry price
- Pending orders
- Adjustment recommendations

### Review Plan

```bash
tradingagents review-plan <plan_id>

# With manual price
tradingagents review-plan <plan_id> --price 82.50
```

Shows:

- High/medium/low priority adjustments
- Stop loss recommendations
- Entry adjustments
- Target adjustments
- Time-based suggestions

## Python API

### Parse a Plan

```python
from tradingagents.execution import TradingPlanParser

parser = TradingPlanParser()
plan = parser.parse_plan(plan_text, symbol="COPPER-C")

print(f"Direction: {plan.direction}")
print(f"Entry tranches: {len(plan.entry_tranches)}")
print(f"Stop: ${plan.stop_loss.initial_price:.2f}")
print(f"Targets: {len(plan.take_profits)}")
```

### Decide Order Type

```python
from tradingagents.execution import OrderExecutor

executor = OrderExecutor("COPPER-C")

decision = executor.decide_order_type(
    direction="BUY",
    target_price=78.50,
    price_range=(77.0, 80.0),
    conditions=["volume>80M", "RSI<65"]
)

print(f"Order type: {decision.order_type.value}")
print(f"Urgency: {decision.urgency}")
print(f"Reason: {decision.reason}")
```

### Execute Order

```python
result = executor.execute_order(
    direction="BUY",
    volume=0.02,
    order_decision=decision,
    stop_loss=72.00,
    take_profit=87.00,
    comment="Tranche 1"
)

if result['success']:
    print(f"Order placed: #{result['ticket']}")
```

### Manage Staged Entries

```python
from tradingagents.execution import StagedEntryManager

manager = StagedEntryManager("COPPER-C_BUY_20260113")
manager.initialize_tranches(plan.entry_tranches)

# Mark tranche as filled
manager.mark_tranche_filled(
    tranche_number=1,
    filled_price=78.52,
    filled_volume=0.02
)

# Get status
status = manager.get_status_summary()
print(f"Filled: {status['filled']}/{status['total_tranches']}")
print(f"Avg entry: ${status['avg_entry_price']:.2f}")

# Check for adjustments
adjustment = manager.should_adjust_remaining(current_price=82.00)
if adjustment['should_adjust']:
    print(adjustment['recommendation'])
```

### Dynamic Stops

```python
from tradingagents.execution import DynamicStopManager

stop_mgr = DynamicStopManager(symbol="COPPER-C", direction="BUY")

# Set initial stop
stop_mgr.set_initial_stop(entry_price=78.50, atr_multiple=1.5)

# Add trailing rules
stop_mgr.add_trailing_rule(trigger_price=82.00, new_stop=75.00)
stop_mgr.add_trailing_rule(trigger_price=87.00, new_stop=78.00)

# Check for updates
result = stop_mgr.check_and_update(current_price=82.50)

if result['updated']:
    for update in result['updates']:
        print(f"Stop updated: ${update['old_stop']:.2f} → ${update['new_stop']:.2f}")

    # Apply to MT5 position
    stop_mgr.apply_stop_to_position(ticket=123456)
```

### Review Plan

```python
from tradingagents.execution import PlanReviewer

reviewer = PlanReviewer()

review = reviewer.review_plan(
    plan=plan,
    current_price=82.00,
    entry_price=78.52,
    position_size=2.0,
    time_in_trade=2
)

print(f"High priority: {len(review['high_priority'])}")
for adj in review['high_priority']:
    print(f"  {adj.component}: {adj.reason}")
```

## Example: Complete Workflow

```python
from tradingagents.execution import *

# 1. Parse plan
parser = TradingPlanParser()
plan = parser.parse_plan(plan_text, symbol="COPPER-C")

# 2. Initialize managers
executor = OrderExecutor(plan.symbol)
staged_mgr = StagedEntryManager("COPPER-C_BUY_20260113")
staged_mgr.initialize_tranches(plan.entry_tranches)
stop_mgr = DynamicStopManager(plan.symbol, plan.direction)

# 3. Execute first tranche
tranche = staged_mgr.get_next_tranche()
decision = executor.decide_order_type(
    direction=plan.direction,
    target_price=tranche.price_level,
    price_range=tranche.price_range
)

result = executor.execute_order(
    direction=plan.direction,
    volume=0.02,
    order_decision=decision,
    stop_loss=plan.stop_loss.initial_price
)

if result['success']:
    staged_mgr.mark_tranche_active(tranche.tranche_number, result['ticket'])
    staged_mgr.mark_tranche_filled(tranche.tranche_number, result['price'], 0.02)

# 4. Setup dynamic stops
stop_mgr.set_initial_stop(result['price'], plan.stop_loss.initial_price)
stop_mgr.parse_trailing_rules(plan.stop_loss.trail_rules)

# 5. Monitor and update (in a loop)
current_price = 82.00

# Check stop updates
stop_result = stop_mgr.check_and_update(current_price)
if stop_result['updated']:
    stop_mgr.apply_stop_to_position(result['ticket'])

# Check staged entry adjustments
adjustment = staged_mgr.should_adjust_remaining(current_price)
if adjustment['should_adjust']:
    print(adjustment['recommendation'])

# Review plan
reviewer = PlanReviewer()
review = reviewer.review_plan(
    plan=plan,
    current_price=current_price,
    entry_price=result['price'],
    position_size=staged_mgr.total_filled_pct,
    time_in_trade=2
)

for adj in review['high_priority']:
    print(f"⚠️  {adj.reason}")
```

## Benefits

### Before

- ❌ Manual order placement
- ❌ Lost plan intelligence
- ❌ Simple "buy at market"
- ❌ Manual stop adjustments
- ❌ No systematic review

### After

- ✅ Automatic plan parsing
- ✅ Intelligent order types
- ✅ Staged entry execution
- ✅ Dynamic stop management
- ✅ Systematic plan review
- ✅ Adaptation recommendations

## Summary

The intelligent execution system bridges the gap between **analysis** and **execution**:

1. **CLI analyze** → Detailed trading plan
2. **Save plan** → Text file
3. **execute-plan** → Parse and execute with intelligence
4. **monitor-plan** → Track staged entries
5. **review-plan** → Get adaptation recommendations

No more losing the plan's intelligence when you execute. The system implements exactly what the analysts recommended, with proper order types, staged entries, dynamic stops, and ongoing review! 🎯
