"""
Example: Analyze Multiple Precious Metals with Continuous Learning

This script analyzes Gold (XAUUSD), Silver (XAGUSD), Copper (COPPER), and Platinum (XPTUSD)
to determine which commodities present the best trading opportunities.

Prerequisites:
1. MT5 terminal running and logged in
2. XAI_API_KEY set in .env file

Usage:
    python examples/analyze_precious_metals.py
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


# Commodity symbols mapping
COMMODITIES = {
    "Gold": "XAUUSD",
    "Silver": "XAGUSD", 
    "Copper": "COPPER",
    "Platinum": "XPTUSD"
}


def analyze_commodity(
    commodity_name: str,
    symbol: str,
    trade_date: str = None,
    setup_type: str = None,
    verbose: bool = False
):
    """
    Analyze a single commodity with continuous learning.
    
    Args:
        commodity_name: Display name (e.g., "Gold")
        symbol: MT5 symbol (e.g., "XAUUSD")
        trade_date: Date to analyze (YYYY-MM-DD), defaults to today
        setup_type: Your identified setup (e.g., 'breaker-block', 'FVG', 'support-bounce')
        verbose: Show detailed output
    
    Returns:
        dict with decision, confidence, position_size, and all analysis data
    """
    
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    if verbose:
        print("\n" + "="*70)
        print(f"{commodity_name} ({symbol}) ANALYSIS - {trade_date}")
        print("="*70 + "\n")
    else:
        print(f"\n🔍 Analyzing {commodity_name} ({symbol})...")
    
    # ========================================================================
    # PHASE 5: Risk Guardrails - CHECK FIRST
    # ========================================================================
    from tradingagents.risk import RiskGuardrails
    
    guardrails = RiskGuardrails()
    account_balance = 10000  # TODO: Get from MT5 account_info()
    
    can_trade, reason = guardrails.check_can_trade(account_balance)
    
    if not can_trade:
        if verbose:
            print(f"⛔ TRADING BLOCKED: {reason}")
        return {
            "commodity": commodity_name,
            "symbol": symbol,
            "decision": "BLOCKED",
            "reason": reason,
            "can_trade": False,
            "score": 0
        }
    
    if verbose:
        print(f"1️⃣  RISK GUARDRAILS: ✅ {reason}")
    
    # ========================================================================
    # PHASE 2: Regime Detection
    # ========================================================================
    from tradingagents.indicators import RegimeDetector
    from tradingagents.dataflows.mt5_data import get_mt5_data
    import pandas as pd
    from io import StringIO
    
    end_date = trade_date
    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")
    
    try:
        price_data = get_mt5_data(symbol, start_date, end_date, timeframe="D1")
        df = pd.read_csv(StringIO(price_data), comment='#')
        
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        current_price = close[-1]
        
        if verbose:
            print(f"   Loaded {len(close)} bars, Current price: ${current_price:.2f}")
        
        # Detect regime
        detector = RegimeDetector()
        regime = detector.get_full_regime(high, low, close)
        
        if verbose:
            print(f"2️⃣  REGIME: {regime['market_regime']} / {regime['volatility_regime']}")
        
        # Trading implications
        trend_favorable = detector.is_favorable_for_trend_trading(regime)
        range_favorable = detector.is_favorable_for_range_trading(regime)
        risk_adj = detector.get_risk_adjustment_factor(regime)
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Could not fetch MT5 data: {e}")
        regime = {
            "market_regime": "unknown",
            "volatility_regime": "normal",
            "expansion_regime": "normal"
        }
        risk_adj = 1.0
        current_price = 0
        trend_favorable = False
        range_favorable = False
    
    # ========================================================================
    # PHASE 3: RAG - Find Similar Historical Trades
    # ========================================================================
    from tradingagents.learning import TradeSimilaritySearch
    
    current_setup = {
        "symbol": symbol,
        "direction": "BUY",
        **regime
    }
    
    if setup_type:
        current_setup["setup_type"] = setup_type
    
    searcher = TradeSimilaritySearch()
    rag_result = searcher.find_similar_trades(current_setup, n_results=5)
    
    stats = rag_result['statistics']
    
    if verbose and stats['sample_size'] > 0:
        print(f"3️⃣  RAG: {stats['sample_size']} similar trades, {stats['win_rate']*100:.1f}% win rate")
    
    # ========================================================================
    # PHASE 4: Get Current Agent Weights
    # ========================================================================
    from tradingagents.learning import OnlineRLUpdater
    
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()
    
    if verbose:
        print(f"4️⃣  WEIGHTS: Bull={weights['bull']:.2f}, Bear={weights['bear']:.2f}, Market={weights['market']:.2f}")
    
    # ========================================================================
    # PHASE 1-4: Run Multi-Agent Analysis
    # ========================================================================
    from examples.trade_commodities import COMMODITY_CONFIG
    
    config = COMMODITY_CONFIG.copy()
    config['current_regime'] = regime
    config['rag_context'] = rag_result
    config['agent_weights'] = weights
    
    try:
        graph = TradingAgentsGraph(config=config, debug=False)
        final_state, signal = graph.propagate(symbol, trade_date)
        
        # Extract detailed Portfolio Manager decision
        risk_debate_state = final_state.get("risk_debate_state", {})
        portfolio_manager_decision = risk_debate_state.get("judge_decision", "")
        
        if verbose:
            print(f"5️⃣  SIGNAL: {signal}")
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Analysis error: {e}")
        signal = "HOLD"
        final_state = {}
    
    # ========================================================================
    # PHASE 3: Apply RAG Confidence Adjustment
    # ========================================================================
    from tradingagents.learning.rag_prompts import apply_confidence_adjustment
    
    base_confidence = 0.65
    final_confidence = apply_confidence_adjustment(
        base_confidence,
        stats['confidence_adjustment']
    )
    
    # ========================================================================
    # PHASE 5: Position Sizing
    # ========================================================================
    base_size = 1.5
    regime_adjusted = base_size * risk_adj
    
    is_valid, reason_msg, final_size = guardrails.validate_position_size(
        regime_adjusted,
        account_balance
    )
    
    # ========================================================================
    # FINAL DECISION
    # ========================================================================
    if signal == "BUY" and final_confidence > 0.65:
        decision = "ENTER LONG"
        score = final_confidence * 100
    elif signal == "SELL" and final_confidence > 0.65:
        decision = "ENTER SHORT"
        score = final_confidence * 100
    elif final_confidence > 0.5:
        decision = "CONSIDER"
        score = final_confidence * 50
    else:
        decision = "SKIP"
        score = 0
    
    # Calculate stop loss and take profit
    if signal == "BUY":
        stop_loss = current_price * 0.985
        take_profit = current_price * 1.03
        risk = current_price - stop_loss
        reward = take_profit - current_price
    elif signal == "SELL":
        stop_loss = current_price * 1.015
        take_profit = current_price * 0.97
        risk = stop_loss - current_price
        reward = current_price - take_profit
    else:
        stop_loss = None
        take_profit = None
        risk = 0
        reward = 0
    
    rr = reward / risk if risk > 0 else 0
    
    if verbose:
        print(f"\n{'✅' if decision.startswith('ENTER') else '⚠️' if decision == 'CONSIDER' else '❌'} DECISION: {decision}")
        print(f"   Confidence: {final_confidence:.2f}")
        print(f"   Position Size: {final_size:.2f}%")
        if stop_loss and take_profit:
            print(f"   Entry: ${current_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
            print(f"   Risk-Reward: {rr:.2f}R")
    
    return {
        "commodity": commodity_name,
        "symbol": symbol,
        "decision": decision,
        "signal": signal,
        "confidence": final_confidence,
        "position_size_pct": final_size,
        "entry_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "rr": rr,
        "regime": regime,
        "rag_stats": stats,
        "agent_weights": weights,
        "can_trade": True,
        "score": score,
        "trend_favorable": trend_favorable,
        "range_favorable": range_favorable,
        "portfolio_manager_decision": portfolio_manager_decision
    }


def analyze_all_commodities(trade_date: str = None, setup_type: str = None):
    """
    Analyze all precious metals and rank them by opportunity score.
    
    Args:
        trade_date: Date to analyze (YYYY-MM-DD), defaults to today
        setup_type: Your identified setup type
    
    Returns:
        list of analysis results sorted by score
    """
    
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\n" + "="*70)
    print(f"PRECIOUS METALS ANALYSIS - {trade_date}")
    print("="*70)
    
    results = []
    
    for commodity_name, symbol in COMMODITIES.items():
        result = analyze_commodity(
            commodity_name,
            symbol,
            trade_date,
            setup_type,
            verbose=False
        )
        results.append(result)
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display summary
    print("\n" + "="*70)
    print("RANKING BY OPPORTUNITY")
    print("="*70 + "\n")
    
    for i, result in enumerate(results, 1):
        commodity = result['commodity']
        symbol = result['symbol']
        decision = result['decision']
        confidence = result['confidence']
        score = result['score']
        
        if decision == "BLOCKED":
            icon = "⛔"
            status = "BLOCKED"
        elif decision.startswith("ENTER"):
            icon = "✅"
            status = f"{decision} (Conf: {confidence:.2f})"
        elif decision == "CONSIDER":
            icon = "⚠️"
            status = f"CONSIDER (Conf: {confidence:.2f})"
        else:
            icon = "❌"
            status = "SKIP"
        
        print(f"{i}. {icon} {commodity:10s} ({symbol:8s}) - {status}")
        
        if decision.startswith("ENTER") and result['entry_price'] > 0:
            print(f"   Entry: ${result['entry_price']:.2f} | SL: ${result['stop_loss']:.2f} | TP: ${result['take_profit']:.2f} | RR: {result['rr']:.2f}R")
            print(f"   Position: {result['position_size_pct']:.2f}% | Regime: {result['regime']['market_regime']}")
    
    # Show detailed analysis for top opportunity
    if results and results[0]['decision'].startswith("ENTER"):
        print("\n" + "="*70)
        print("DETAILED ANALYSIS - TOP OPPORTUNITY")
        print("="*70)
        
        top = results[0]
        print(f"\n{top['commodity']} ({top['symbol']})")
        print(f"Signal: {top['signal']}")
        print(f"Confidence: {top['confidence']:.2f}")
        print(f"Entry Price: ${top['entry_price']:.2f}")
        print(f"Stop Loss: ${top['stop_loss']:.2f}")
        print(f"Take Profit: ${top['take_profit']:.2f}")
        print(f"Risk-Reward: {top['rr']:.2f}R")
        print(f"Position Size: {top['position_size_pct']:.2f}%")
        print(f"\nRegime:")
        print(f"  Market: {top['regime']['market_regime']}")
        print(f"  Volatility: {top['regime']['volatility_regime']}")
        print(f"  Trend Trading: {'✅' if top['trend_favorable'] else '❌'}")
        print(f"  Range Trading: {'✅' if top['range_favorable'] else '❌'}")
        
        if top['rag_stats']['sample_size'] > 0:
            print(f"\nHistorical Context:")
            print(f"  Similar trades: {top['rag_stats']['sample_size']}")
            print(f"  Win rate: {top['rag_stats']['win_rate']*100:.1f}%")
            print(f"  Avg RR: {top['rag_stats']['avg_rr']:.2f}")
        
        print(f"\nAgent Weights:")
        print(f"  Bull:   {top['agent_weights']['bull']:.3f}")
        print(f"  Bear:   {top['agent_weights']['bear']:.3f}")
        print(f"  Market: {top['agent_weights']['market']:.3f}")
        
        # Show detailed Portfolio Manager decision
        if top.get('portfolio_manager_decision'):
            print("\n" + "="*70)
            print("PORTFOLIO MANAGER DETAILED ANALYSIS")
            print("="*70)
            print(top['portfolio_manager_decision'])
    
    print("\n" + "="*70 + "\n")
    
    return results


def main():
    """Run multi-commodity analysis"""
    
    # Check for API key
    if not os.getenv("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in environment or .env file")
        print("Please set your xAI API key to use TradingAgents with Grok")
        return
    
    # Run analysis for all commodities
    results = analyze_all_commodities(
        trade_date=None,  # Today
        setup_type="breaker-block"  # Change to your identified setup
    )
    
    # Count opportunities
    enter_count = sum(1 for r in results if r['decision'].startswith('ENTER'))
    consider_count = sum(1 for r in results if r['decision'] == 'CONSIDER')
    
    print(f"✅ Analysis complete!")
    print(f"   {enter_count} strong opportunities")
    print(f"   {consider_count} marginal setups")
    print(f"   {len(results) - enter_count - consider_count} skipped")


if __name__ == "__main__":
    main()
