# TradingAgents Implementation Standards

This document defines mandatory implementation standards for the TradingAgents financial trading system. These rules are **NON-NEGOTIABLE**. Financial systems require high accuracy and robust risk management.

## Core Principle

**Never implement financial functionality without proper validation.** In trading systems, bugs lose money. Every price, every order, every calculation must be validated before execution.

---

## 1. Price Validation

### Always Validate Against Current Market Price

Never trust cached or stale prices for order execution. Always fetch and validate against live market data.

```typescript
// WRONG - using potentially stale price
const entry = suggestedEntry || cachedPrice

// CORRECT - always use current market price for execution
const { data } = await getSymbolInfo(symbol)
const entry = signal === "BUY" ? data.ask : data.bid
```

### Validate Price Reasonableness

Reject prices that deviate significantly from market (hallucinated or corrupted data):

```typescript
const deviation = Math.abs(suggestedPrice - marketPrice) / marketPrice
if (deviation > 0.2) {
  // Price is more than 20% from market - reject or warn
}
```

---

## 2. Stop Loss Validation

### MANDATORY: Stop Loss Side Validation

Stop loss MUST be validated to be on the correct side of entry. This is a critical safety check.

```typescript
// ALWAYS implement this validation for ANY stop loss input
function validateStopLoss(sl: number, entry: number, direction: "BUY" | "SELL"): boolean {
  if (direction === "BUY") {
    // For BUY: SL must be BELOW entry
    return sl < entry
  } else {
    // For SELL: SL must be ABOVE entry
    return sl > entry
  }
}
```

### Validate Stop Loss Distance

Stop loss should not be:
- Too close (within spread or minimum stop distance)
- Too far (more than reasonable % of price)

```typescript
const slDistance = Math.abs(entry - sl)
const slPercent = slDistance / entry

// Too close - likely to be hit immediately
if (slPercent < 0.001) reject("SL too close to entry")

// Too far - unreasonable risk
if (slPercent > 0.10) warn("SL is more than 10% from entry")
```

---

## 3. Take Profit Validation

### MANDATORY: Take Profit Side Validation

```typescript
function validateTakeProfit(tp: number, entry: number, direction: "BUY" | "SELL"): boolean {
  if (direction === "BUY") {
    // For BUY: TP must be ABOVE entry
    return tp > entry
  } else {
    // For SELL: TP must be BELOW entry
    return tp < entry
  }
}
```

---

## 4. Position Size Validation

### Never Allow Oversized Positions

```typescript
// Validate against broker limits
if (volume < symbolInfo.volume_min) reject("Below minimum lot size")
if (volume > symbolInfo.volume_max) reject("Exceeds maximum lot size")

// Validate against account risk
const riskAmount = calculateRisk(entry, sl, volume)
const maxRiskAmount = accountBalance * (maxRiskPercent / 100)
if (riskAmount > maxRiskAmount) reject("Position size exceeds risk limit")
```

### Validate Lot Step

```typescript
// Ensure volume matches broker's lot step
const step = symbolInfo.volume_step || 0.01
const adjustedVolume = Math.round(volume / step) * step
```

---

## 5. Order Execution Rules

### Pre-Execution Checklist

Before ANY order execution, validate ALL of the following:

1. Entry price is current (not stale)
2. Stop loss is on correct side of entry
3. Stop loss is at valid distance
4. Take profit is on correct side of entry (if set)
5. Position size is within limits
6. Account has sufficient margin
7. No circuit breaker is active
8. Symbol is tradeable (market open, no halt)

### Disable Execute Button Until Valid

```typescript
// ALWAYS disable execution until ALL validations pass
<Button
  onClick={handleExecute}
  disabled={
    loading ||
    !entryPrice ||
    !stopLoss ||
    !volume ||
    !!stopLossError ||      // SL validation failed
    !!takeProfitError ||    // TP validation failed
    !!volumeError ||        // Volume validation failed
    !!marginError           // Margin check failed
  }
>
```

---

## 6. Error Handling

### Never Silently Fail

All financial operations MUST have explicit error handling:

```typescript
// WRONG
try {
  await placeOrder(params)
} catch (e) {
  console.log(e)  // Silent failure
}

// CORRECT
try {
  const result = await placeOrder(params)
  if (!result.success) {
    setError(result.error)
    logTradeError(params, result.error)
  }
} catch (e) {
  setError(`Order failed: ${e.message}`)
  logTradeError(params, e)
}
```

### Return Meaningful Errors

```typescript
// WRONG
throw new Error("Invalid order")

// CORRECT
throw new Error(`Invalid stop loss: ${sl} must be below entry ${entry} for BUY orders`)
```

---

## 7. Data Integrity

### Never Modify Price Data In-Place

```typescript
// WRONG - mutating original data
prices[i] = prices[i] * adjustment

// CORRECT - create new array
const adjustedPrices = prices.map(p => p * adjustment)
```

### Validate All External Data

Data from APIs, broker feeds, or user input must be validated:

```typescript
function validateOHLC(candle: OHLC): boolean {
  if (candle.high < candle.low) return false
  if (candle.open > candle.high || candle.open < candle.low) return false
  if (candle.close > candle.high || candle.close < candle.low) return false
  if (candle.volume < 0) return false
  return true
}
```

---

## 8. Risk Management Integration

### Circuit Breaker Checks

Before any trade execution, check circuit breaker status:

```typescript
const circuitBreaker = await getCircuitBreakerStatus()
if (circuitBreaker.tripped) {
  reject(`Trading halted: ${circuitBreaker.reason}`)
}
```

### Position Correlation Checks

Before opening new positions, check for over-exposure:

```typescript
const existingExposure = await getSymbolExposure(symbol)
const newExposure = calculateExposure(volume, entry)
const totalExposure = existingExposure + newExposure

if (totalExposure > maxExposureLimit) {
  warn("Combined exposure exceeds limit")
}
```

---

## 9. Audit Trail

### Log All Trade-Related Actions

```typescript
interface TradeLog {
  timestamp: string
  action: "order_placed" | "order_modified" | "order_cancelled" | "position_closed"
  symbol: string
  direction: "BUY" | "SELL"
  volume: number
  entry: number
  sl?: number
  tp?: number
  result: "success" | "failure"
  error?: string
}
```

### Never Delete Trade History

Trade decisions and outcomes must be preserved for learning and analysis.

---

## 10. UI/UX for Financial Safety

### Clear Visual Indicators

- **RED** for sell/short/loss/stop-loss
- **GREEN** for buy/long/profit/take-profit
- **YELLOW** for warnings
- Disable buttons with clear visual feedback when invalid

### Confirmation for Destructive Actions

Always require confirmation for:
- Closing positions
- Cancelling orders
- Modifying stop losses (especially widening)

### Show Calculations Transparently

```typescript
<div>
  <span>Risk Amount: ${riskAmount.toFixed(2)}</span>
  <span>= {riskPercent}% of ${accountBalance.toFixed(2)}</span>
</div>
```

---

## Implementation Checklist

When implementing ANY trading-related feature:

- [ ] Identified all inputs that need validation
- [ ] Implemented validation for each input
- [ ] Validation errors displayed clearly to user
- [ ] Execute buttons disabled until all validations pass
- [ ] Error handling covers all failure modes
- [ ] Audit logging implemented
- [ ] Tested with edge cases (negative, zero, very large values)
- [ ] Tested with invalid data (wrong side SL, stale prices)
