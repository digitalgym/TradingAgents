# Investigation: BTCUSD SELL Signal Not Executed (2026-03-24)

## Signal Details
- **Instance**: btcusd_range_quant
- **Signal**: SELL @ 71256.91, 75% confidence
- **Time**: 2026-03-24T07:29:44
- **SL/TP**: None (ATR fallback would calculate them)
- **Auto Execute**: ON
- **Result**: Not executed, no error logged

## Root Cause
**Global guardrails circuit breaker** blocked the trade.

The shared risk state (`examples/risk_state.pkl`) had:
- `daily_loss_pct`: 4.65% (exceeds 3% limit)
- `cooldown_until`: 2026-03-24T21:07:25 (24h cooldown from breach)
- `consecutive_losses`: triggered 3x limit on March 22 and daily loss limit on March 23

The `check_can_trade()` returned `False, "COOLDOWN: X hours remaining"` at line 1512-1515 in `quant_automation.py`, causing a silent early return without setting `execution_error`.

## Issues Found

### 1. Silent blocking (no feedback to user)
When guardrails block a trade, the result is returned without any indication:
```python
# Line 1513-1515
if not can_trade:
    self.logger.warning(f"Trading blocked by guardrails: {reason}")
    return result  # No execution_error set, no rationale append
```
**Fix**: Should set `result.execution_error = f"Guardrails: {reason}"` or append to rationale so the UI shows WHY a signal wasn't executed.

### 2. Shared guardrails state across all automations
All instances share `examples/risk_state.pkl`. Losses from `btcusd_smc_quant` and `xauusd_*` instances triggered the circuit breaker, blocking `btcusd_range_quant` which had no losses.

**Consider**: Per-instance or per-symbol guardrails state, or at minimum a way to reset/override from the UI.

### 3. Daily loss not resetting
`daily_loss_pct` is 4.65% from March 23 but the date is now March 24. The daily loss should reset at day boundaries but the cooldown persists for 24h regardless.

## Action Items
- [ ] Add `execution_error` or rationale annotation when guardrails block a trade
- [ ] Consider per-instance guardrails state (or at least per-symbol)
- [ ] Add guardrails status to the automation UI panel (show cooldown timer)
- [ ] Add manual guardrails reset button in the UI
- [ ] Consider reducing cooldown from 24h or making it configurable per instance
