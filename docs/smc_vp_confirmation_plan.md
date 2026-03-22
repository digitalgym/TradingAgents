# SMC + Volume Profile Confirmation Pipeline

## Context

Currently each quant pipeline (SMC Quant, Volume Profile, Rule-Based, etc.) trades independently. Today we saw opposing positions on both XAUUSD and BTCUSD — different pipelines taking opposite sides. A confirmation pipeline solves this by requiring **both SMC and VP to agree on direction** before placing a trade. This should produce fewer but higher-quality signals.

The rule-based XAUUSD SELL targeting 3332 (28% drop) highlighted how individual pipelines can latch onto unrealistic levels. Dual confirmation naturally filters these out — VP wouldn't confirm a sell when price is deep in discount below POC.

## Architecture

```
quant_automation (smc_vp_confirmation pipeline)
    │
    ├── asyncio.gather()
    │   ├── _run_smc_quant_analysis(symbol)  → POST /api/analysis/smc-quant
    │   └── _run_vp_analysis(symbol)         → POST /api/analysis/vp-quant
    │
    ├── Agreement Check
    │   ├── Both BUY or both SELL → proceed to merge
    │   ├── Either HOLD → combined HOLD
    │   └── Disagree (BUY vs SELL) → combined HOLD
    │
    ├── Conservative Merge (when agreed)
    │   ├── Entry: closer to market (more conservative)
    │   ├── SL: tighter of the two (less risk)
    │   └── TP: more conservative (closer to entry, higher win rate)
    │
    └── Combined AnalysisCycleResult → existing _execute_trade() flow
```

## Merge Logic

**Entry**: For BUY, take the higher entry (harder to fill = more conservative). For SELL, take the lower.

**Stop Loss**: For BUY, take the higher SL (tighter). For SELL, take the lower SL (tighter). Reasoning: if either analyst thinks the closer invalidation level matters, respect it.

**Take Profit**: For BUY, take the lower TP (closer). For SELL, take the higher TP. This avoids unrealistic distant targets (like the 3332 XAUUSD SELL TP). If VP says POC is at 5000 and SMC says liquidity at 4900, we target 4900.

**Confidence**: `min((smc_conf + vp_conf) / 2 * 1.15, 1.0)` — average with 15% boost for dual agreement.

## Files to Modify

### 1. MODIFY: `tradingagents/automation/quant_automation.py`

**A) Add to PipelineType enum** (~line 113):
```python
SMC_VP_CONFIRMATION = "smc_vp_confirmation"
```

**B) New method `_run_smc_vp_confirmation_analysis(self, symbol)`** (after `_run_vp_analysis`):
- `asyncio.gather()` both existing analysis methods in parallel
- Handle errors: if either fails/exceptions, return HOLD
- Check agreement on direction
- If agreed: merge entry/SL/TP conservatively, combine confidence
- Build combined rationale showing both agents' analysis
- Log: `[CONFIRM] SMC: BUY (0.75), VP: BUY (0.65) → CONFIRMED BUY (0.81)`
- Return single `AnalysisCycleResult` with `pipeline="smc_vp_confirmation"`

**C) Wire into 3 dispatch locations**:
- Main analysis loop (~line 2258): `elif pipeline == SMC_VP_CONFIRMATION`
- `run_single_analysis` (~line 2827): same dispatch
- Reversal close (~line 2010): add to supported pipelines tuple + dispatch

### 2. MODIFY: `web/frontend/src/app/automation/page.tsx`

Add to the 4 lookup objects:
- `pipelineLabels`: `"SMC + VP Confirmation"`
- `pipelineColors`: `"text-amber-500"`
- `pipelineDescriptions`: summary + details explaining dual confirmation
- `pipelineDefaults`: `{ timeframe: "D1", interval: 3600, confidence: 0.65, atrMultiplier: 2.0 }`
- Add to pipeline dropdown options

### 3. MODIFY: `automation_configs.json`

Add example config entry for `xauusd_smc_vp_confirmation` instance.

## What We DON'T Change

- No new backend endpoints (reuse existing SMC + VP endpoints)
- No new LLM calls (just the existing two)
- No changes to `_execute_trade`, `store_decision`, position management, or guardrails
- No changes to reflection/learning (combined rationale includes SMC text, so setup_type extraction still works)
- No new dataclass fields

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| One analysis times out | Existing methods return HOLD on timeout → combined = HOLD |
| Both agree, one has no entry | Fall back to other's entry. If neither, execution validation rejects |
| Both agree, wildly different SLs | Conservative merge picks tighter SL, execution validates min distance |
| Timeframe | Both use same `config.timeframe` — consistent analysis window |

## Implementation Order

1. `quant_automation.py` — PipelineType + new method + 3 dispatch points
2. `automation/page.tsx` — frontend labels/descriptions/dropdown
3. `automation_configs.json` — example config
4. Test: create instance via web UI, run test analysis, verify both agents called and merged

## Verification

1. `python -c "from tradingagents.automation.quant_automation import PipelineType; print(PipelineType.SMC_VP_CONFIRMATION)"` — enum exists
2. Create instance via web UI with `smc_vp_confirmation` pipeline
3. Run "Test Analysis" → verify logs show `[CONFIRM] SMC: X, VP: Y` and correct merge
4. Test disagreement: if SMC=BUY, VP=SELL → should return HOLD
5. Test agreement: if both BUY → should return BUY with merged params
6. `npx tsc --noEmit` in frontend (only pre-existing errors)
