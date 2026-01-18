# Debug SMC State Flow

## Issue

After multiple attempts, SMC is still not appearing in analyst reasoning despite:

1. Adding SMC fields to AgentState as Optional
2. Initializing SMC in create_initial_state
3. Adding SMC to state in CLI
4. Injecting SMC into all analyst prompts

## New Debug Logging

I've added explicit debug logging to track the state flow:

### In CLI (cli/main.py)

```
[DEBUG] Added SMC to state: X chars
[DEBUG] First 200 chars: ...
[DEBUG] State keys after SMC: ['messages', 'company_of_interest', ..., 'smc_context', 'smc_analysis']
```

### In Trader (trader.py)

```
[TRADER DEBUG] State keys: [...]
[TRADER DEBUG] SMC in state: True/False
[TRADER DEBUG] SMC context length: X
```

## What to Check

Run analysis again and look for:

1. **In Messages panel:** Do you see `[DEBUG] State keys after SMC: ...`?

   - Should include `'smc_context'` and `'smc_analysis'`

2. **In terminal stderr:** Do you see `[TRADER DEBUG]` messages?
   - Check if trader receives smc_context key
   - Check if it's empty or has content

## Possible Issues

If state keys show SMC but trader doesn't receive it:

- LangGraph is dropping the fields during stream
- MessagesState parent class is overriding
- State schema mismatch

If state keys don't show SMC:

- SMC analysis is failing silently
- Condition check is wrong (asset_type/data_vendor)

## Next Steps

Based on debug output, we'll know:

1. Is SMC being added to initial state? (CLI debug)
2. Is trader receiving it? (Trader debug)
3. Where is it being lost?

Then we can fix the exact point of failure.
