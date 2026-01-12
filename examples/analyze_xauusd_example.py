"""
Example: Analyze XAUUSD with Continuous Learning

This shows how to analyze XAUUSD (Gold) to determine if you should open a trade,
using all 5 phases of the continuous learning system.

Prerequisites:
1. MT5 terminal running and logged in
2. XAI_API_KEY set in .env file

Usage:
    python examples/analyze_xauusd_example.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def analyze_xauusd_with_learning(
    trade_date: str = None,
    entry_price: float = None,
    setup_type: str = None
):
    """
    Complete analysis of XAUUSD with continuous learning.
    
    Args:
        trade_date: Date to analyze (YYYY-MM-DD), defaults to today
        entry_price: Current price (optional, will fetch from MT5)
        setup_type: Your identified setup (e.g., 'breaker-block', 'FVG', 'support-bounce')
    
    Returns:
        dict with decision, confidence, position_size, and all analysis data
    """
    
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\n" + "="*70)
    print(f"XAUUSD ANALYSIS WITH CONTINUOUS LEARNING - {trade_date}")
    print("="*70 + "\n")
    
    # ========================================================================
    # PHASE 5: Risk Guardrails - CHECK FIRST
    # ========================================================================
    print("1️⃣  RISK GUARDRAILS CHECK")
    print("-" * 70)
    
    from tradingagents.risk import RiskGuardrails
    
    guardrails = RiskGuardrails()
    account_balance = 10000  # TODO: Get from MT5 account_info()
    
    can_trade, reason = guardrails.check_can_trade(account_balance)
    
    if not can_trade:
        print(f"⛔ TRADING BLOCKED: {reason}")
        print("\nSystem in cooldown. No analysis will be performed.")
        print("Check status with: tradingagents risk-status")
        return {
            "decision": "BLOCKED",
            "reason": reason,
            "can_trade": False
        }
    
    print(f"✅ {reason}")
    print(f"   Consecutive losses: {guardrails.state['consecutive_losses']}/{guardrails.max_consecutive_losses}")
    print(f"   Daily loss: {guardrails.state['daily_loss_pct']:.2f}%/{guardrails.daily_loss_limit_pct}%")
    
    # ========================================================================
    # PHASE 2: Regime Detection
    # ========================================================================
    print("\n2️⃣  REGIME DETECTION")
    print("-" * 70)
    
    from tradingagents.indicators import RegimeDetector
    from tradingagents.dataflows.mt5_data import get_mt5_data
    import pandas as pd
    from io import StringIO
    
    # Get price data from MT5
    end_date = trade_date
    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")
    
    print(f"Fetching XAUUSD data from MT5 ({start_date} to {end_date})...")
    
    try:
        price_data = get_mt5_data("XAUUSD", start_date, end_date, timeframe="D1")
        df = pd.read_csv(StringIO(price_data), comment='#')
        
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        current_price = close[-1]
        if entry_price is None:
            entry_price = current_price
        
        print(f"Loaded {len(close)} bars, Current price: ${current_price:.2f}")
        
        # Detect regime
        detector = RegimeDetector()
        regime = detector.get_full_regime(high, low, close)
        
        print(f"\n📊 Current Regime:")
        print(f"   Market: {regime['market_regime']}")
        print(f"   Volatility: {regime['volatility_regime']}")
        print(f"   Expansion: {regime['expansion_regime']}")
        
        # Get description
        description = detector.get_regime_description(regime)
        print(f"\n   {description}")
        
        # Trading implications
        trend_favorable = detector.is_favorable_for_trend_trading(regime)
        range_favorable = detector.is_favorable_for_range_trading(regime)
        risk_adj = detector.get_risk_adjustment_factor(regime)
        
        print(f"\n   Trend Trading: {'✅ Favorable' if trend_favorable else '❌ Not Favorable'}")
        print(f"   Range Trading: {'✅ Favorable' if range_favorable else '❌ Not Favorable'}")
        print(f"   Position Size Adjustment: {risk_adj:.2f}x")
        
    except Exception as e:
        print(f"⚠️  Could not fetch MT5 data: {e}")
        print("Using default regime...")
        regime = {
            "market_regime": "unknown",
            "volatility_regime": "normal",
            "expansion_regime": "normal"
        }
        risk_adj = 1.0
    
    # ========================================================================
    # PHASE 3: RAG - Find Similar Historical Trades
    # ========================================================================
    print("\n3️⃣  HISTORICAL CONTEXT (RAG)")
    print("-" * 70)
    
    from tradingagents.learning import TradeSimilaritySearch
    
    # Build current setup
    current_setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",  # Change to SELL if you're looking for shorts
        **regime
    }
    
    if setup_type:
        current_setup["setup_type"] = setup_type
        print(f"Setup type: {setup_type}")
    
    print("Searching for similar historical trades...")
    
    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup, n_results=5)
    
    stats = rag_result['statistics']
    
    if stats['sample_size'] > 0:
        print(f"\n📚 Found {stats['sample_size']} similar trades:")
        print(f"   Win rate: {stats['win_rate']*100:.1f}%")
        print(f"   Avg RR: {stats['avg_rr']:.2f}")
        print(f"   Best: {stats['best_rr']:+.2f}R | Worst: {stats['worst_rr']:+.2f}R")
        print(f"   Confidence adjustment: {stats['confidence_adjustment']:+.2f}")
        
        # Show top 3 similar trades
        if rag_result['similar_trades']:
            print(f"\n   Top 3 Similar Trades:")
            for i, trade in enumerate(rag_result['similar_trades'][:3], 1):
                outcome = "✅ WIN" if trade.get('was_correct') else "❌ LOSS"
                rr = trade.get('rr_realized', 0)
                setup = trade.get('setup_type', 'unknown')
                print(f"   {i}. {outcome} {rr:+.2f}R ({setup})")
    else:
        print("⚠️  No similar trades found in history")
        print("   This is a new pattern - proceed with caution")
    
    # ========================================================================
    # PHASE 4: Get Current Agent Weights
    # ========================================================================
    print("\n4️⃣  AGENT WEIGHTS (ADAPTIVE)")
    print("-" * 70)
    
    from tradingagents.learning import OnlineRLUpdater
    
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()
    
    print(f"Current agent weights (based on recent performance):")
    print(f"   Bull:   {weights['bull']:.3f} {'🔥' if weights['bull'] > 0.4 else '❄️' if weights['bull'] < 0.25 else '→'}")
    print(f"   Bear:   {weights['bear']:.3f} {'🔥' if weights['bear'] > 0.4 else '❄️' if weights['bear'] < 0.25 else '→'}")
    print(f"   Market: {weights['market']:.3f} {'🔥' if weights['market'] > 0.4 else '❄️' if weights['market'] < 0.25 else '→'}")
    
    # Check if pattern update needed
    should_update, trades_since = updater.should_update()
    if should_update:
        print(f"\n⚠️  Pattern analysis due ({trades_since} trades)")
        print("   Run: tradingagents update-patterns")
    
    # ========================================================================
    # PHASE 1-4: Run Multi-Agent Analysis
    # ========================================================================
    print("\n5️⃣  MULTI-AGENT ANALYSIS")
    print("-" * 70)
    
    # Create config (using commodity config as base)
    from examples.trade_commodities import COMMODITY_CONFIG
    
    config = COMMODITY_CONFIG.copy()
    config['current_regime'] = regime
    config['rag_context'] = rag_result
    config['agent_weights'] = weights
    
    print("Running TradingAgentsGraph analysis...")
    print("(This will query market data, news, sentiment, and run agent debate)\n")
    
    try:
        graph = TradingAgentsGraph(config=config, debug=False)
        final_state, signal = graph.propagate("XAUUSD", trade_date)
        
        print(f"\n📊 Analysis Complete!")
        print(f"   Signal: {signal}")
        
        # Extract key information
        market_report = final_state.get("market_report", "N/A")
        news_report = final_state.get("news_report", "N/A")
        final_decision = final_state.get("final_trade_decision", "N/A")
        
        # Get detailed Portfolio Manager decision from risk debate state
        risk_debate_state = final_state.get("risk_debate_state", {})
        portfolio_manager_decision = risk_debate_state.get("judge_decision", "N/A")
        
        print(f"\n--- Market Report (excerpt) ---")
        print(market_report[:300] + "..." if len(market_report) > 300 else market_report)
        
        print(f"\n--- News Report (excerpt) ---")
        print(news_report[:300] + "..." if len(news_report) > 300 else news_report)
        
        print(f"\n--- Portfolio Manager Decision (FULL DETAIL) ---")
        print(portfolio_manager_decision)
        
        print(f"\n--- Final Decision Summary ---")
        print(final_decision[:500] + "..." if len(final_decision) > 500 else final_decision)
        
    except Exception as e:
        print(f"⚠️  Analysis error: {e}")
        print("Using simplified decision...")
        signal = "HOLD"
        final_state = {}
    
    # ========================================================================
    # PHASE 3: Apply RAG Confidence Adjustment
    # ========================================================================
    print("\n6️⃣  CONFIDENCE CALCULATION")
    print("-" * 70)
    
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment
    
    # Base confidence from agents (simplified - normally extracted from final_state)
    base_confidence = 0.65  # Example
    
    # Apply RAG adjustment
    final_confidence = apply_confidence_adjustment(
        base_confidence,
        stats['confidence_adjustment']
    )
    
    print(f"Base confidence: {base_confidence:.2f}")
    print(f"RAG adjustment: {stats['confidence_adjustment']:+.2f}")
    print(f"Final confidence: {final_confidence:.2f}")
    
    # ========================================================================
    # PHASE 5: Position Sizing
    # ========================================================================
    print("\n7️⃣  POSITION SIZING")
    print("-" * 70)
    
    base_size = 1.5  # 1.5% base risk
    regime_adjusted = base_size * risk_adj
    
    is_valid, reason, final_size = guardrails.validate_position_size(
        regime_adjusted,
        account_balance
    )
    
    print(f"Base position size: {base_size}%")
    print(f"Regime adjusted: {regime_adjusted:.2f}% (×{risk_adj:.2f})")
    print(f"Final position size: {final_size:.2f}%")
    
    if not is_valid:
        print(f"⚠️  {reason}")
    
    # ========================================================================
    # FINAL DECISION
    # ========================================================================
    print("\n8️⃣  FINAL DECISION")
    print("=" * 70)
    
    # Decision logic
    if signal == "BUY" and final_confidence > 0.65:
        decision = "ENTER LONG"
        decision_color = "✅"
    elif signal == "SELL" and final_confidence > 0.65:
        decision = "ENTER SHORT"
        decision_color = "✅"
    elif final_confidence > 0.5:
        decision = "CONSIDER (Lower confidence)"
        decision_color = "⚠️"
    else:
        decision = "SKIP"
        decision_color = "❌"
    
    print(f"\n{decision_color} DECISION: {decision}")
    print(f"   Signal: {signal}")
    print(f"   Confidence: {final_confidence:.2f}")
    print(f"   Position Size: {final_size:.2f}% of account")
    print(f"   Entry Price: ${entry_price:.2f}")
    
    # Calculate stop loss and take profit (example)
    if signal == "BUY":
        stop_loss = entry_price * 0.985  # 1.5% stop
        take_profit = entry_price * 1.03  # 3% target (2R)
    elif signal == "SELL":
        stop_loss = entry_price * 1.015
        take_profit = entry_price * 0.97
    else:
        stop_loss = None
        take_profit = None
    
    if stop_loss and take_profit:
        print(f"   Stop Loss: ${stop_loss:.2f}")
        print(f"   Take Profit: ${take_profit:.2f}")
        
        # Calculate RR
        if signal == "BUY":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        rr = reward / risk if risk > 0 else 0
        print(f"   Risk-Reward: {rr:.2f}R")
    
    # ========================================================================
    # NEXT STEPS
    # ========================================================================
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if decision.startswith("ENTER"):
        print("\n✅ Trade Setup Approved!")
        print("\n1. Store the decision:")
        print(f"   from tradingagents.trade_decisions import store_decision, set_decision_regime")
        print(f"   decision_id = store_decision(")
        print(f"       symbol='XAUUSD',")
        print(f"       decision_type='OPEN',")
        print(f"       action='{signal}',")
        print(f"       entry_price={entry_price:.2f},")
        print(f"       stop_loss={stop_loss:.2f},")
        print(f"       take_profit={take_profit:.2f},")
        print(f"       volume={final_size/100:.4f}")
        print(f"   )")
        print(f"   set_decision_regime(decision_id, regime)")
        
        print("\n2. Execute trade in MT5")
        
        print("\n3. After trade closes:")
        print(f"   from tradingagents.trade_decisions import close_decision")
        print(f"   close_decision(decision_id, exit_price=<actual_exit>, exit_reason='tp-hit')")
        
    elif decision == "CONSIDER (Lower confidence)":
        print("\n⚠️  Marginal Setup")
        print("   - Consider reducing position size further")
        print("   - Wait for better confluence")
        print("   - Or skip this trade")
        
    else:
        print("\n❌ Trade Not Recommended")
        print("   - Signal not strong enough")
        print("   - Wait for better setup")
        print("   - Check again later")
    
    print("\n" + "=" * 70 + "\n")
    
    # Return complete analysis
    return {
        "decision": decision,
        "signal": signal,
        "confidence": final_confidence,
        "position_size_pct": final_size,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "regime": regime,
        "rag_stats": stats,
        "agent_weights": weights,
        "can_trade": True,
        "final_state": final_state
    }


def main():
    """Run XAUUSD analysis"""
    
    # Check for API key
    if not os.getenv("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in environment or .env file")
        print("Please set your xAI API key to use TradingAgents with Grok")
        return
    
    # Run analysis
    print("\n🔍 Analyzing XAUUSD for trading opportunity...\n")
    
    result = analyze_xauusd_with_learning(
        trade_date=None,  # Today
        entry_price=None,  # Will fetch from MT5
        setup_type="breaker-block"  # Change to your identified setup
    )
    
    # Summary
    if result['can_trade']:
        print(f"✅ Analysis complete!")
        print(f"   Decision: {result['decision']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Position: {result['position_size_pct']:.2f}%")


if __name__ == "__main__":
    main()
