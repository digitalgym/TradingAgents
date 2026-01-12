# Integration with Existing CLI and Trade Commodities System

## Overview

The continuous learning system (Phases 1-5) integrates seamlessly with your existing `analyze` CLI command and `trade_commodities.py` workflow. This guide shows exactly where and how to add the new features.

## Your Existing System

### 1. CLI Analyze Command (`cli/main.py`)

Your CLI has an `analyze` command that:

- Takes a ticker and date
- Runs `TradingAgentsGraph.propagate()` to analyze
- Returns a signal (BUY/SELL/HOLD) and final_state
- Displays results in a rich terminal UI

### 2. Trade Commodities (`examples/trade_commodities.py`)

Your commodity trading script:

- Analyzes commodities (XAUUSD, XAGUSD, etc.) using MT5 data
- Uses xAI Grok for LLM reasoning
- Saves trade state for later reflection
- Runs reflection after trade closes to store lessons

## Integration Points

### Where the Continuous Learning System Fits

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR EXISTING FLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. CLI: tradingagents analyze XAUUSD 2026-01-11           │
│     │                                                        │
│     ├──> TradingAgentsGraph.propagate()                    │
│     │    ├─> Market Analyst                                │
│     │    ├─> News Analyst                                  │
│     │    ├─> Bull/Bear Researchers                         │
│     │    └─> Trader → Signal (BUY/SELL/HOLD)              │
│     │                                                        │
│     └──> Display Results                                    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│              ADD CONTINUOUS LEARNING HERE ↓                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              ENHANCED FLOW WITH LEARNING                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. CLI: tradingagents analyze XAUUSD 2026-01-11           │
│     │                                                        │
│     ├──> [PHASE 5] Check Risk Guardrails                   │
│     │    └─> Can we trade? (daily loss, consecutive loss)  │
│     │                                                        │
│     ├──> [PHASE 2] Detect Market Regime                    │
│     │    └─> trending-up / normal volatility               │
│     │                                                        │
│     ├──> [PHASE 3] RAG: Find Similar Trades                │
│     │    └─> 5 similar trades, 75% win rate                │
│     │                                                        │
│     ├──> [PHASE 4] Get Current Agent Weights               │
│     │    └─> Bull: 0.45, Bear: 0.25, Market: 0.30         │
│     │                                                        │
│     ├──> TradingAgentsGraph.propagate()                    │
│     │    ├─> [Enhanced] Agents get RAG context in prompts  │
│     │    ├─> Market Analyst                                │
│     │    ├─> News Analyst                                  │
│     │    ├─> Bull/Bear Researchers                         │
│     │    └─> Trader → Signal                               │
│     │                                                        │
│     ├──> [PHASE 3] Apply RAG Confidence Adjustment         │
│     │    └─> Base 0.75 → Adjusted 0.85                     │
│     │                                                        │
│     ├──> [PHASE 4] Apply Agent Weights to Consensus        │
│     │    └─> Weighted decision based on performance        │
│     │                                                        │
│     ├──> [PHASE 5] Validate Position Size                  │
│     │    └─> Cap at 2% max                                 │
│     │                                                        │
│     ├──> [PHASE 1] Store Decision with Metadata            │
│     │    └─> Save regime, setup_type, confluence           │
│     │                                                        │
│     └──> Display Enhanced Results                           │
│                                                              │
│  2. After Trade Closes:                                     │
│     │                                                        │
│     ├──> [PHASE 1] Calculate Reward Signal                 │
│     │    └─> Multi-factor: RR + Sharpe + Drawdown         │
│     │                                                        │
│     ├──> [PHASE 5] Record in Risk Guardrails               │
│     │    └─> Update consecutive losses, daily loss         │
│     │                                                        │
│     ├──> [PHASE 4] Check if Pattern Update Needed          │
│     │    └─> Every 30 trades: analyze patterns             │
│     │                                                        │
│     └──> Existing Reflection (stores lessons in memory)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Integration

### Step 1: Enhance CLI Analyze Command

Modify `cli/main.py` to add continuous learning features:

```python
# In cli/main.py

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    date: Optional[str] = typer.Option(None, help="Analysis date (YYYY-MM-DD)"),
    # ... existing parameters ...
):
    """Analyze a stock with continuous learning enhancements."""

    # ========================================================================
    # PHASE 5: Risk Guardrails - CHECK FIRST
    # ========================================================================
    from tradingagents.risk import RiskGuardrails

    guardrails = RiskGuardrails()
    account_balance = 10000  # Get from your account

    can_trade, reason = guardrails.check_can_trade(account_balance)

    if not can_trade:
        console.print(f"[bold red]⛔ Trading Blocked: {reason}[/bold red]")
        console.print("[yellow]System in cooldown. No analysis will be performed.[/yellow]")
        return

    console.print(f"[green]✅ Risk Check: {reason}[/green]")

    # ========================================================================
    # PHASE 2: Regime Detection
    # ========================================================================
    from tradingagents.indicators import RegimeDetector
    from tradingagents.dataflows.mt5_data import get_mt5_data
    import pandas as pd
    import numpy as np

    console.print("[cyan]📊 Detecting market regime...[/cyan]")

    # Get price data from MT5
    end_date = date or datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") -
                  datetime.timedelta(days=100)).strftime("%Y-%m-%d")

    price_data = get_mt5_data(ticker, start_date, end_date, timeframe="D1")
    df = pd.read_csv(pd.StringIO(price_data), comment='#')

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)

    console.print(f"[cyan]   Market: {regime['market_regime']}[/cyan]")
    console.print(f"[cyan]   Volatility: {regime['volatility_regime']}[/cyan]")

    # ========================================================================
    # PHASE 3: RAG - Find Similar Historical Trades
    # ========================================================================
    from tradingagents.learning import TradeSimilaritySearch

    console.print("[cyan]🔍 Searching for similar historical trades...[/cyan]")

    current_setup = {
        "symbol": ticker,
        "direction": "BUY",  # Will be refined after analysis
        **regime
    }

    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup, n_results=5)

    stats = rag_result['statistics']
    if stats['sample_size'] > 0:
        console.print(f"[cyan]   Found {stats['sample_size']} similar trades[/cyan]")
        console.print(f"[cyan]   Historical win rate: {stats['win_rate']*100:.1f}%[/cyan]")
        console.print(f"[cyan]   Confidence adjustment: {stats['confidence_adjustment']:+.2f}[/cyan]")

    # ========================================================================
    # PHASE 4: Get Current Agent Weights
    # ========================================================================
    from tradingagents.learning import OnlineRLUpdater

    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    console.print(f"[cyan]🤖 Agent Weights: Bull={weights['bull']:.2f}, "
                  f"Bear={weights['bear']:.2f}, Market={weights['market']:.2f}[/cyan]")

    # Check if pattern update needed
    should_update, trades_since = updater.should_update()
    if should_update:
        console.print(f"[yellow]⚠️  Pattern analysis due ({trades_since} trades)[/yellow]")

    # ========================================================================
    # EXISTING: Run TradingAgentsGraph Analysis
    # ========================================================================
    console.print("[bold cyan]Running multi-agent analysis...[/bold cyan]")

    # Create config with your selections
    config = create_config_from_selections(...)  # Your existing code

    # Add regime and RAG context to config for agents to use
    config['current_regime'] = regime
    config['rag_context'] = rag_result
    config['agent_weights'] = weights

    # Run analysis (your existing code)
    graph = TradingAgentsGraph(config=config, debug=debug)
    final_state, signal = graph.propagate(ticker, date)

    # ========================================================================
    # PHASE 3: Apply RAG Confidence Adjustment
    # ========================================================================
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment

    # Extract base confidence from final_state
    base_confidence = final_state.get('trader_confidence', 0.5)

    # Apply RAG adjustment
    final_confidence = apply_confidence_adjustment(
        base_confidence,
        stats['confidence_adjustment']
    )

    console.print(f"[cyan]Confidence: {base_confidence:.2f} → {final_confidence:.2f} "
                  f"(RAG adjusted)[/cyan]")

    # ========================================================================
    # PHASE 5: Validate Position Size
    # ========================================================================
    requested_size = 1.5  # Your base position size
    risk_adj = detector.get_risk_adjustment_factor(regime)
    regime_adjusted = requested_size * risk_adj

    is_valid, reason, final_size = guardrails.validate_position_size(
        regime_adjusted,
        account_balance
    )

    console.print(f"[cyan]Position Size: {requested_size}% → {final_size}% "
                  f"(regime + guardrail adjusted)[/cyan]")

    # ========================================================================
    # Display Results (your existing code)
    # ========================================================================
    display_results(final_state, signal, final_confidence, final_size)

    # ========================================================================
    # PHASE 1: Store Decision for Later Tracking
    # ========================================================================
    from tradingagents.trade_decisions import store_decision, set_decision_regime

    if signal in ["BUY", "SELL"]:
        decision_id = store_decision(
            symbol=ticker,
            decision_type="OPEN",
            action=signal,
            rationale=final_state.get('final_trade_decision', ''),
            entry_price=close[-1],
            stop_loss=close[-1] * 0.98,  # Example
            take_profit=close[-1] * 1.04,  # Example
            volume=final_size / 100
        )

        # Add regime context
        set_decision_regime(decision_id, regime)

        # Add setup metadata
        from tradingagents.trade_decisions import load_decision
        import json
        decision = load_decision(decision_id)
        decision['setup_type'] = final_state.get('setup_type', 'unknown')
        decision['confluence_score'] = final_state.get('confluence_score', 5)
        decision['higher_tf_bias'] = final_state.get('higher_tf_bias', 'neutral')

        # Save
        from tradingagents.trade_decisions import DECISIONS_DIR
        import os
        decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
        with open(decision_file, 'w') as f:
            json.dump(decision, f, indent=2, default=str)

        console.print(f"[green]✅ Decision stored: {decision_id}[/green]")
```

### Step 2: Enhance Trade Commodities Workflow

Modify `examples/trade_commodities.py`:

```python
# In examples/trade_commodities.py

def analyze_commodity_with_learning(
    symbol: str,
    trade_date: str,
    entry_price: float = None,
    save_for_reflection: bool = False
):
    """
    Enhanced commodity analysis with continuous learning.

    This wraps your existing analyze_commodity() with learning features.
    """

    # ========================================================================
    # PHASE 5: Risk Check
    # ========================================================================
    from tradingagents.risk import RiskGuardrails

    guardrails = RiskGuardrails()
    account_balance = 10000  # Get from MT5 account

    can_trade, reason = guardrails.check_can_trade(account_balance)

    if not can_trade:
        print(f"⛔ Trading blocked: {reason}")
        return None, "BLOCKED", None

    print(f"✅ Risk check: {reason}")

    # ========================================================================
    # PHASE 2: Regime Detection
    # ========================================================================
    from tradingagents.indicators import RegimeDetector
    from tradingagents.dataflows.mt5_data import get_mt5_data
    import pandas as pd

    print("📊 Detecting regime...")

    # Get price data
    end_date = trade_date
    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") -
                  timedelta(days=100)).strftime("%Y-%m-%d")

    price_data = get_mt5_data(symbol, start_date, end_date, timeframe="D1")
    df = pd.read_csv(pd.StringIO(price_data), comment='#')

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)

    print(f"   Market: {regime['market_regime']}")
    print(f"   Volatility: {regime['volatility_regime']}")

    # ========================================================================
    # PHASE 3 & 4: RAG + Agent Weights
    # ========================================================================
    from tradingagents.learning import TradeSimilaritySearch, OnlineRLUpdater

    print("🔍 Checking historical performance...")

    current_setup = {
        "symbol": symbol,
        "direction": "BUY",
        **regime
    }

    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup)

    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    print(f"   Similar trades: {rag_result['statistics']['sample_size']}")
    print(f"   Win rate: {rag_result['statistics']['win_rate']*100:.1f}%")
    print(f"   Agent weights: Bull={weights['bull']:.2f}")

    # ========================================================================
    # EXISTING: Run Analysis
    # ========================================================================
    print("\n🤖 Running multi-agent analysis...")

    # Add learning context to config
    enhanced_config = COMMODITY_CONFIG.copy()
    enhanced_config['current_regime'] = regime
    enhanced_config['rag_context'] = rag_result
    enhanced_config['agent_weights'] = weights

    graph = TradingAgentsGraph(config=enhanced_config, debug=True)
    final_state, signal = graph.propagate(symbol, trade_date)

    # ========================================================================
    # PHASE 3: Apply RAG Adjustment
    # ========================================================================
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment

    base_conf = final_state.get('trader_confidence', 0.5)
    final_conf = apply_confidence_adjustment(
        base_conf,
        rag_result['statistics']['confidence_adjustment']
    )

    print(f"\n✨ Confidence: {base_conf:.2f} → {final_conf:.2f}")

    # ========================================================================
    # PHASE 1: Store with Full Metadata
    # ========================================================================
    if save_for_reflection:
        from tradingagents.trade_decisions import store_decision, set_decision_regime

        decision_id = store_decision(
            symbol=symbol,
            decision_type="OPEN",
            action=signal,
            rationale=final_state.get('final_trade_decision', ''),
            entry_price=entry_price or close[-1],
            stop_loss=(entry_price or close[-1]) * 0.98,
            take_profit=(entry_price or close[-1]) * 1.04
        )

        # Add regime
        set_decision_regime(decision_id, regime)

        # Add setup metadata
        decision = load_decision(decision_id)
        decision['setup_type'] = final_state.get('setup_type', 'unknown')
        decision['confluence_score'] = final_state.get('confluence_score', 5)

        # Save
        decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
        with open(decision_file, 'w') as f:
            json.dump(decision, f, indent=2, default=str)

        return final_state, signal, decision_id

    return final_state, signal, None


def complete_trade_with_learning(trade_id: str, exit_price: float):
    """
    Enhanced trade completion with learning features.

    This wraps your existing complete_trade() with learning.
    """

    # ========================================================================
    # PHASE 1: Calculate Reward Signal
    # ========================================================================
    from tradingagents.learning import RewardCalculator, PortfolioStateTracker
    from tradingagents.trade_decisions import close_decision

    print(f"\n{'='*60}")
    print(f"COMPLETING TRADE WITH LEARNING: {trade_id}")
    print(f"{'='*60}")

    # Close decision (automatically calculates reward)
    close_decision(
        decision_id=trade_id,
        exit_price=exit_price,
        exit_reason="manual-close"
    )

    # Load decision to get reward
    decision = load_decision(trade_id)
    reward = decision.get('reward_signal', 0)
    was_correct = decision.get('was_correct', False)
    pnl_pct = decision.get('pnl_pct', 0)

    print(f"   Reward Signal: {reward:+.2f}")
    print(f"   Was Correct: {was_correct}")
    print(f"   P&L: {pnl_pct:+.2f}%")

    # ========================================================================
    # PHASE 5: Update Risk Guardrails
    # ========================================================================
    from tradingagents.risk import RiskGuardrails

    guardrails = RiskGuardrails()
    account_balance = 10000  # Get from MT5

    result = guardrails.record_trade_result(was_correct, pnl_pct, account_balance)

    if result['breach_triggered']:
        print(f"\n⛔ CIRCUIT BREAKER TRIGGERED!")
        print(f"   Type: {result['breach_type']}")
        print(f"   Cooldown until: {result['cooldown_until']}")

    # ========================================================================
    # PHASE 4: Check if Pattern Update Needed
    # ========================================================================
    from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

    updater = OnlineRLUpdater()
    should_update, trades_since = updater.should_update()

    if should_update:
        print(f"\n📊 Running pattern analysis ({trades_since} trades)...")

        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30)

        print(analyzer.format_report(analysis))

        # Update weights
        performances = updater.calculate_agent_performances()
        weight_result = updater.update_weights(performances)

        print("\n" + updater.format_report(weight_result))

    # ========================================================================
    # EXISTING: Run Reflection
    # ========================================================================
    print(f"\n🧠 Running reflection...")

    # Your existing reflection code
    trade_data = load_trade_state(trade_id)
    graph = TradingAgentsGraph(config=COMMODITY_CONFIG, debug=False)
    graph.reflect(trade_data['final_state'], pnl_pct)

    print("✅ Complete! All learning systems updated.")
```

### Step 3: Add Learning Dashboard Command

Add a new CLI command to view learning status:

```python
# In cli/main.py

@app.command()
def learning_status():
    """Display continuous learning system status."""

    from tradingagents.risk import RiskGuardrails
    from tradingagents.learning import OnlineRLUpdater, PatternAnalyzer

    console.print("\n[bold cyan]CONTINUOUS LEARNING SYSTEM STATUS[/bold cyan]\n")

    # Risk Guardrails
    guardrails = RiskGuardrails()
    console.print(Panel(guardrails.format_report(), title="Risk Guardrails"))

    # Agent Weights
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()

    weight_table = Table(title="Agent Weights")
    weight_table.add_column("Agent")
    weight_table.add_column("Weight")
    weight_table.add_column("Status")

    for agent, weight in weights.items():
        status = "🔥" if weight > 0.4 else "❄️" if weight < 0.25 else "→"
        weight_table.add_row(agent.capitalize(), f"{weight:.3f}", status)

    console.print(weight_table)

    # Pattern Analysis Status
    should_update, trades_since = updater.should_update()

    if should_update:
        console.print(f"\n[yellow]⚠️  Pattern analysis due ({trades_since}/30 trades)[/yellow]")
        console.print("[yellow]Run: tradingagents update-patterns[/yellow]")
    else:
        console.print(f"\n[green]✅ Pattern analysis current ({trades_since}/30 trades)[/green]")


@app.command()
def update_patterns():
    """Run pattern analysis and update agent weights."""

    from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater

    console.print("\n[bold cyan]RUNNING PATTERN ANALYSIS[/bold cyan]\n")

    with console.status("[cyan]Analyzing trade patterns..."):
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30)

    console.print(analyzer.format_report(analysis))

    console.print("\n[bold cyan]UPDATING AGENT WEIGHTS[/bold cyan]\n")

    with console.status("[cyan]Calculating agent performances..."):
        updater = OnlineRLUpdater()
        performances = updater.calculate_agent_performances()
        result = updater.update_weights(performances)

    console.print(updater.format_report(result))

    console.print("\n[green]✅ Learning system updated![/green]")
```

## Usage Examples

### Example 1: CLI Analysis with Learning

```bash
# Analyze with all learning features
tradingagents analyze XAUUSD --date 2026-01-11

# Output includes:
# ✅ Risk Check: OK
# 📊 Market: trending-up / normal volatility
# 🔍 Found 8 similar trades, 75% win rate
# 🤖 Agent Weights: Bull=0.45, Bear=0.25, Market=0.30
# ... (normal analysis) ...
# Confidence: 0.75 → 0.85 (RAG adjusted)
# Position Size: 1.5% → 1.5% (validated)
# ✅ Decision stored: XAUUSD_20260111_140000
```

### Example 2: Commodity Trading with Learning

```python
# In examples/trade_commodities.py or your script

# Analyze with learning
final_state, signal, decision_id = analyze_commodity_with_learning(
    symbol="XAUUSD",
    trade_date="2026-01-11",
    entry_price=2650.0,
    save_for_reflection=True
)

# ... execute trade ...

# After trade closes
complete_trade_with_learning(decision_id, exit_price=2690.0)

# Output includes:
# ✅ Reward Signal: +1.85
# ✅ Risk guardrails updated
# 📊 Pattern analysis (if 30 trades reached)
# 🔄 Agent weights updated
# 🧠 Reflection complete
```

### Example 3: Check Learning Status

```bash
# View system status
tradingagents learning-status

# Output shows:
# - Risk guardrails status
# - Agent weights
# - Pattern analysis status
# - Breach history

# Update patterns manually
tradingagents update-patterns
```

## Key Benefits

### 1. **Seamless Integration**

- No changes to your core `TradingAgentsGraph` logic
- Wraps existing `analyze` and `trade_commodities` functions
- All learning happens before/after existing workflow

### 2. **Backward Compatible**

- Existing code still works without modifications
- Learning features are additive, not breaking
- Can enable/disable features individually

### 3. **Enhanced Decision Quality**

- Risk guardrails prevent catastrophic losses
- RAG provides historical context
- Regime detection filters by market conditions
- Agent weights adapt to performance

### 4. **Automatic Learning**

- Pattern analysis runs every 30 trades
- Agent weights update automatically
- No manual intervention needed

## Migration Checklist

- [ ] Add risk check before `graph.propagate()`
- [ ] Add regime detection before analysis
- [ ] Add RAG similarity search
- [ ] Get current agent weights
- [ ] Apply RAG confidence adjustment
- [ ] Validate position size with guardrails
- [ ] Store decision with full metadata (regime, setup_type, etc.)
- [ ] Calculate reward signal after trade closes
- [ ] Update risk guardrails after each trade
- [ ] Check for pattern update trigger (every 30 trades)
- [ ] Add `learning-status` and `update-patterns` CLI commands

## Next Steps

1. **Start with Risk Guardrails** - Protect your account first
2. **Add Regime Detection** - Filter trades by market conditions
3. **Enable RAG** - Use historical performance
4. **Let it Learn** - After 30 trades, patterns emerge
5. **Monitor & Adjust** - Use `learning-status` to track

The system learns and improves automatically while your existing workflow remains unchanged!
