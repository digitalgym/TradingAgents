"""
Example: Trading commodities (Gold, Silver, Platinum, Copper) with TradingAgents.

This example shows how to use MT5 data from Vantage Australia to analyze commodities,
with xAI Grok for LLM, news (web search), and X (Twitter) sentiment analysis.

Prerequisites:
1. MT5 terminal must be running and logged into Vantage account
2. Set your XAI_API_KEY in .env file

Features:
- MT5 for OHLCV price data
- xAI Grok for LLM reasoning
- xAI web_search for market news
- xAI x_search for X/Twitter sentiment
- Local embeddings (sentence-transformers) for memory

Usage:
    python examples/trade_commodities.py
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph


# =============================================================================
# TRADE STATE PERSISTENCE
# =============================================================================
# Since reflection happens AFTER a trade completes (days/weeks later),
# we need to save the final_state from analysis to use later.

TRADES_DIR = os.path.join(os.path.dirname(__file__), "pending_trades")

def save_trade_state(symbol: str, trade_date: str, final_state: dict, signal: str, entry_price: float = None):
    """
    Save trade state for later reflection when trade completes.
    
    Call this RIGHT AFTER analysis, before executing the trade.
    The final_state contains all the context needed for reflection.
    """
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    trade_id = f"{symbol}_{trade_date}_{datetime.now().strftime('%H%M%S')}"
    trade_file = os.path.join(TRADES_DIR, f"{trade_id}.pkl")
    
    trade_data = {
        "trade_id": trade_id,
        "symbol": symbol,
        "trade_date": trade_date,
        "signal": signal,
        "entry_price": entry_price,
        "final_state": final_state,
        "created_at": datetime.now().isoformat(),
        "status": "pending",  # pending, closed, reflected
    }
    
    with open(trade_file, "wb") as f:
        pickle.dump(trade_data, f)
    
    print(f"💾 Trade state saved: {trade_file}")
    print(f"   Trade ID: {trade_id}")
    print(f"   When trade closes, run: complete_trade('{trade_id}', exit_price)")
    
    return trade_id


def load_trade_state(trade_id: str) -> dict:
    """Load a saved trade state."""
    trade_file = os.path.join(TRADES_DIR, f"{trade_id}.pkl")
    
    if not os.path.exists(trade_file):
        raise FileNotFoundError(f"Trade not found: {trade_id}")
    
    with open(trade_file, "rb") as f:
        return pickle.load(f)


def list_pending_trades():
    """List all pending trades waiting for reflection."""
    if not os.path.exists(TRADES_DIR):
        print("No pending trades.")
        return []
    
    trades = []
    for f in os.listdir(TRADES_DIR):
        if f.endswith(".pkl"):
            trade_data = load_trade_state(f.replace(".pkl", ""))
            if trade_data["status"] == "pending":
                trades.append(trade_data)
                print(f"📋 {trade_data['trade_id']}: {trade_data['symbol']} {trade_data['signal']} @ {trade_data.get('entry_price', 'N/A')}")
    
    if not trades:
        print("No pending trades.")
    return trades


def complete_trade(trade_id: str, exit_price: float, graph: TradingAgentsGraph = None):
    """
    Complete a trade and run reflection to store lessons.
    
    Call this AFTER the trade closes with the actual exit price.
    
    Args:
        trade_id: The ID returned from save_trade_state()
        exit_price: The actual exit price of the trade
        graph: TradingAgentsGraph instance (will create one if not provided)
    """
    trade_data = load_trade_state(trade_id)
    
    if trade_data["status"] != "pending":
        print(f"⚠️ Trade {trade_id} already {trade_data['status']}")
        return
    
    entry_price = trade_data.get("entry_price")
    if entry_price is None:
        print("⚠️ No entry price recorded. Please provide manually.")
        return
    
    # Calculate returns
    returns_losses = ((exit_price - entry_price) / entry_price) * 100
    
    print(f"\n{'='*60}")
    print(f"COMPLETING TRADE: {trade_id}")
    print(f"{'='*60}")
    print(f"Symbol: {trade_data['symbol']}")
    print(f"Signal: {trade_data['signal']}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Exit: ${exit_price:.2f}")
    print(f"Returns: {returns_losses:+.2f}%")
    
    # Create graph if not provided
    if graph is None:
        graph = TradingAgentsGraph(config=COMMODITY_CONFIG, debug=False)
    
    # Run reflection with the ORIGINAL final_state + actual outcome
    final_state = trade_data["final_state"]
    graph.reflect(final_state, returns_losses)
    
    # Update trade status
    trade_data["status"] = "reflected"
    trade_data["exit_price"] = exit_price
    trade_data["returns_losses"] = returns_losses
    trade_data["reflected_at"] = datetime.now().isoformat()
    
    trade_file = os.path.join(TRADES_DIR, f"{trade_id}.pkl")
    with open(trade_file, "wb") as f:
        pickle.dump(trade_data, f)
    
    print(f"\n✅ Reflection complete! Lessons stored in memory.")
    print(f"   Trade status updated to: reflected")

# Configuration for commodity trading with MT5
# Start with default config and override what we need
from tradingagents.default_config import DEFAULT_CONFIG

COMMODITY_CONFIG = DEFAULT_CONFIG.copy()
COMMODITY_CONFIG.update({
    # Use MT5 for price data and indicators
    "data_vendors": {
        "core_stock_apis": "mt5",           # Use MT5 for OHLCV data
        "technical_indicators": "mt5",       # MT5 calculates indicators locally
        "fundamental_data": "openai",        # Not used for commodities
        "news_data": "xai",                  # xAI Grok web search for news
    },
    # Use xAI for sentiment analysis from X (Twitter)
    "tool_vendors": {
        "get_insider_sentiment": "xai",      # X sentiment via Grok
    },
    # Set asset type to commodity (auto-excludes fundamentals analyst)
    "asset_type": "commodity",
    # LLM settings - Using xAI Grok (latest models)
    "llm_provider": "xai",
    "deep_think_llm": "grok-4-1-fast-reasoning",   # Best tool-calling with 2M context
    "quick_think_llm": "grok-4-fast-non-reasoning", # Fast 2M context without reasoning
    "backend_url": "https://api.x.ai/v1",
    # Enable memory with local embeddings (no API needed)
    "use_memory": True,
    "embedding_provider": "local",  # Uses sentence-transformers locally
    # Commodity symbol mappings
    "commodity_symbols": {
        "gold": "XAUUSD",
        "silver": "XAGUSD",
        "platinum": "XPTUSD",
        "copper": "COPPER-C",
    },
})


def analyze_commodity(symbol: str, trade_date: str, entry_price: float = None, save_for_reflection: bool = False):
    """
    Analyze a commodity using TradingAgents.

    Args:
        symbol: Commodity symbol (XAUUSD, XAGUSD, XPTUSD, COPPER-C) or alias (gold, silver, etc.)
        trade_date: Date to analyze in YYYY-MM-DD format
        entry_price: Optional entry price if you're going to execute the trade
        save_for_reflection: If True, saves final_state for later reflection

    Returns:
        final_state, signal, trade_id (if save_for_reflection=True)

    WORKFLOW:
        1. Run analysis: final_state, signal, trade_id = analyze_commodity("XAUUSD", "2025-12-26", entry_price=2650.00, save_for_reflection=True)
        2. Execute trade based on signal
        3. Wait for trade to close
        4. Complete trade: complete_trade(trade_id, exit_price=2720.00)
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {symbol} for {trade_date}")
    print(f"{'='*60}\n")

    # Create the trading graph with commodity config
    # Note: fundamentals analyst is auto-excluded for commodities
    graph = TradingAgentsGraph(
        config=COMMODITY_CONFIG,
        debug=True,  # Set to False for less verbose output
    )

    print(f"Selected analysts: {graph.selected_analysts}")

    # Run SMC analysis for commodities with MT5
    smc_analysis = None
    smc_context = None
    if COMMODITY_CONFIG.get("data_vendors", {}).get("core_stock_apis") == "mt5":
        try:
            from tradingagents.dataflows.smc_utils import (
                analyze_multi_timeframe_smc,
                format_smc_for_prompt,
                get_htf_bias_alignment
            )

            print("Running Smart Money Concepts analysis...")
            smc_analysis = analyze_multi_timeframe_smc(
                symbol=symbol,
                timeframes=['1H', '4H', 'D1']
            )

            if smc_analysis:
                alignment = get_htf_bias_alignment(smc_analysis)
                print(f"SMC Analysis: {alignment['message']}")
                print(f"  - Timeframes with unmitigated OBs: {len([a for a in smc_analysis.values() if a['order_blocks']['unmitigated'] > 0])}")
                smc_context = format_smc_for_prompt(smc_analysis, symbol)
        except Exception as e:
            print(f"SMC analysis skipped: {e}")

    # Run analysis with SMC context
    final_state, signal = graph.propagate(symbol, trade_date, smc_context=smc_context)
    
    # Print results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nSymbol: {symbol}")
    print(f"Date: {trade_date}")
    print(f"\n--- Market Report ---")
    print(final_state.get("market_report", "N/A")[:500])
    print(f"\n--- News Report ---")
    print(final_state.get("news_report", "N/A")[:500])
    print(f"\n--- Final Decision ---")
    print(final_state.get("final_trade_decision", "N/A"))
    print(f"\n--- Signal ---")
    print(f"Recommendation: {signal}")
    
    # Optionally save for later reflection
    trade_id = None
    if save_for_reflection:
        trade_id = save_trade_state(symbol, trade_date, final_state, signal, entry_price)
    
    if save_for_reflection:
        return final_state, signal, trade_id, smc_analysis
    return final_state, signal, smc_analysis


def display_smc_levels(symbol: str, signal: str, smc_analysis: dict, trade_date: str):
    """Display SMC-suggested stop loss and take profit levels."""
    if not smc_analysis or signal == "HOLD":
        return None, None

    from tradingagents.dataflows.smc_utils import (
        suggest_smc_stop_loss,
        suggest_smc_take_profits,
        suggest_smc_entry_strategy
    )
    from tradingagents.dataflows.interface import route_to_vendor

    # Use full multi-timeframe analysis directly
    if not smc_analysis:
        print("No SMC data available for level suggestions")
        return None, None

    # Get current price from any available timeframe (prefer 1H)
    current_price = None
    for tf in ['1H', '4H', 'H4', 'D1']:
        if tf in smc_analysis and 'current_price' in smc_analysis[tf]:
            current_price = smc_analysis[tf]['current_price']
            break

    if not current_price:
        print("Could not determine current price from SMC data")
        return None, None

    print(f"\n{'='*60}")
    print("SMC-BASED TRADE LEVELS (Multi-Timeframe Confluence)")
    print(f"{'='*60}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Signal: {signal}")
    print(f"Available timeframes: {list(smc_analysis.keys())}")

    # Get entry strategy suggestion
    entry_strategy = suggest_smc_entry_strategy(
        smc_analysis=smc_analysis,
        direction=signal,
        current_price=current_price,
        primary_timeframe='1H'
    )

    print(f"\n{'='*60}")
    print("ENTRY STRATEGY")
    print(f"{'='*60}")

    # Display Market Entry option
    print(f"\n[OPTION 1: MARKET ORDER]")
    print(f"  Entry Price: ${entry_strategy['market_entry']['price']:.2f}")
    print(f"  Type: {entry_strategy['market_entry']['type']}")
    print(f"  Pros: Immediate execution, no risk of missing the trade")
    print(f"  Cons: Entering away from optimal SMC zone, higher risk")

    # Display Limit Entry option
    print(f"\n[OPTION 2: LIMIT ORDER AT ORDER BLOCK]")
    if entry_strategy['limit_entry'] and 'price' in entry_strategy['limit_entry']:
        limit_entry = entry_strategy['limit_entry']
        print(f"  Entry Price: ${limit_entry['price']:.2f}")
        print(f"  Entry Zone: ${limit_entry['zone_bottom']:.2f} - ${limit_entry['zone_top']:.2f}")
        print(f"  Type: {limit_entry['type']}")
        print(f"  Confluence: {limit_entry['confluence_score']:.1f} ({', '.join(limit_entry['aligned_timeframes'])})")
        print(f"  Pros: Better risk/reward, entering at institutional zone")
        print(f"  Cons: Price may not return to zone, could miss the trade")
        print(f"  Reason: {limit_entry['reason']}")
    else:
        print(f"  {entry_strategy['limit_entry']['reason']}")

    # Display Recommendation
    print(f"\n[RECOMMENDATION: {entry_strategy['recommendation']}]")
    if 'recommendation_reason' in entry_strategy:
        print(f"  {entry_strategy['recommendation_reason']}")
        if entry_strategy['distance_to_zone_pct'] is not None:
            print(f"  Distance to optimal zone: {entry_strategy['distance_to_zone_pct']:.2f}%")

    print(f"\n{'='*60}")

    # Get ATR for stop loss fallback
    atr_data = route_to_vendor("get_indicators", symbol, "atr", trade_date, 14)
    atr_value = None
    if "ATR(14):" in atr_data:
        try:
            atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
            print(f"ATR(14): {atr_value:.2f}")
        except:
            pass

    # Determine which entry price to use for SL/TP calculations
    # If limit entry is available and recommended, show both perspectives
    show_limit_levels = (entry_strategy['limit_entry'] and
                        'price' in entry_strategy['limit_entry'] and
                        entry_strategy['recommendation'] in ['LIMIT', 'LIMIT_OR_MARKET'])

    # Get SMC stop loss suggestion with multi-timeframe confluence (for market entry)
    stop_suggestion_market = suggest_smc_stop_loss(
        smc_analysis=smc_analysis,
        direction=signal,
        entry_price=current_price,
        atr=atr_value,
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    print(f"\n{'='*60}")
    print("RISK MANAGEMENT LEVELS")
    print(f"{'='*60}")

    # Show Stop Loss for Market Entry
    print(f"\n--- Stop Loss (if entering at MARKET: ${current_price:.2f}) ---")
    if stop_suggestion_market:
        print(f"  SL Price: ${stop_suggestion_market['price']:.2f}")
        print(f"  Zone: ${stop_suggestion_market['zone_bottom']:.2f} - ${stop_suggestion_market['zone_top']:.2f}")
        print(f"  Risk: ${abs(current_price - stop_suggestion_market['price']):.2f} ({stop_suggestion_market['distance_pct']:.2f}%)")
        if 'confluence_score' in stop_suggestion_market and stop_suggestion_market['confluence_score'] > 1.0:
            print(f"  Confluence: {stop_suggestion_market['confluence_score']:.1f} ({', '.join(stop_suggestion_market['aligned_timeframes'])})")
    else:
        print(f"  Could not calculate stop loss")

    # Show Stop Loss for Limit Entry (if applicable)
    if show_limit_levels:
        limit_entry_price = entry_strategy['limit_entry']['price']
        stop_suggestion_limit = suggest_smc_stop_loss(
            smc_analysis=smc_analysis,
            direction=signal,
            entry_price=limit_entry_price,
            atr=atr_value,
            atr_multiplier=2.0,
            primary_timeframe='1H'
        )

        print(f"\n--- Stop Loss (if entering at LIMIT: ${limit_entry_price:.2f}) ---")
        if stop_suggestion_limit:
            print(f"  SL Price: ${stop_suggestion_limit['price']:.2f}")
            print(f"  Zone: ${stop_suggestion_limit['zone_bottom']:.2f} - ${stop_suggestion_limit['zone_top']:.2f}")
            print(f"  Risk: ${abs(limit_entry_price - stop_suggestion_limit['price']):.2f} ({stop_suggestion_limit['distance_pct']:.2f}%)")
            if 'confluence_score' in stop_suggestion_limit and stop_suggestion_limit['confluence_score'] > 1.0:
                print(f"  Confluence: {stop_suggestion_limit['confluence_score']:.1f} ({', '.join(stop_suggestion_limit['aligned_timeframes'])})")

            # Show risk comparison
            market_risk = abs(current_price - stop_suggestion_market['price']) if stop_suggestion_market else 0
            limit_risk = abs(limit_entry_price - stop_suggestion_limit['price'])
            if market_risk > 0:
                risk_reduction = ((market_risk - limit_risk) / market_risk * 100)
                print(f"  Risk Reduction: {risk_reduction:.1f}% vs market entry")

    # Get SMC take profit suggestions (for market entry)
    first_tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
    tp_suggestions_market = suggest_smc_take_profits(
        smc_analysis=first_tf_data if first_tf_data else {},
        direction=signal,
        entry_price=current_price,
        num_targets=3
    )

    print(f"\n--- Take Profit Targets (if entering at MARKET: ${current_price:.2f}) ---")
    if tp_suggestions_market:
        for i, tp in enumerate(tp_suggestions_market, 1):
            print(f"  TP{i}: ${tp['price']:.2f} ({tp['source']}) | +{tp['distance_pct']:.2f}%")
    else:
        print("  No suitable SMC zones found for take profits")

    # Show TP for Limit Entry (if applicable)
    if show_limit_levels:
        limit_entry_price = entry_strategy['limit_entry']['price']
        tp_suggestions_limit = suggest_smc_take_profits(
            smc_analysis=first_tf_data if first_tf_data else {},
            direction=signal,
            entry_price=limit_entry_price,
            num_targets=3
        )

        print(f"\n--- Take Profit Targets (if entering at LIMIT: ${limit_entry_price:.2f}) ---")
        if tp_suggestions_limit:
            for i, tp in enumerate(tp_suggestions_limit, 1):
                print(f"  TP{i}: ${tp['price']:.2f} ({tp['source']}) | +{tp['distance_pct']:.2f}%")

                # Show reward comparison
                if i <= len(tp_suggestions_market):
                    market_reward = abs(tp_suggestions_market[i-1]['price'] - current_price)
                    limit_reward = abs(tp['price'] - limit_entry_price)
                    reward_increase = ((limit_reward - market_reward) / market_reward * 100) if market_reward > 0 else 0
                    if reward_increase > 0:
                        print(f"       Reward Increase: +{reward_increase:.1f}% vs market entry")

    # Store the suggestions for later use
    stop_suggestion = stop_suggestion_market
    tp_suggestions = tp_suggestions_market

    return stop_suggestion, tp_suggestions


def display_comprehensive_smc_plan(symbol: str, signal: str, smc_analysis: dict, trade_date: str, final_state: dict = None):
    """
    Display comprehensive multi-scenario SMC trading plan with LLM enhancement.

    Analyzes current price position relative to SMC zones and generates:
    - Position analysis (at resistance, at support, or in between)
    - Order block strength assessments (retests, confluence, breakout probability)
    - Primary setup based on current position
    - Alternative setup for opposite scenario
    - LLM-enhanced contextual intelligence (market regime, sentiment, risks)
    - Clear recommendation with confidence level
    """
    if not smc_analysis or signal == "HOLD":
        return None

    from tradingagents.dataflows.smc_utils import generate_smc_trading_plan
    from tradingagents.dataflows.interface import route_to_vendor
    from tradingagents.dataflows.llm_smc_enhancer import enhance_plan_with_llm
    from langchain_openai import ChatOpenAI
    import os

    # Get current price
    current_price = None
    for tf in ['1H', '4H', 'H4', 'D1']:
        if tf in smc_analysis and 'current_price' in smc_analysis[tf]:
            current_price = smc_analysis[tf]['current_price']
            break

    if not current_price:
        print("Could not determine current price from SMC data")
        return None

    # Get ATR
    atr_data = route_to_vendor("get_indicators", symbol, "atr", trade_date, 14)
    atr_value = None
    if "ATR(14):" in atr_data:
        try:
            atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
        except:
            pass

    # Generate comprehensive plan
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias=signal,
        primary_timeframe='1H',
        atr=atr_value
    )

    if 'error' in plan:
        print(f"\nError generating trading plan: {plan['error']}")
        return None

    # Enhance plan with LLM reasoning (if API key available)
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        try:
            print("\n[Enhancing plan with LLM contextual intelligence...]")
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.3,
                api_key=xai_api_key,
                base_url="https://api.x.ai/v1"
            )

            plan = enhance_plan_with_llm(
                plan=plan,
                smc_analysis=smc_analysis,
                llm=llm,
                atr=atr_value,
                final_state=final_state
            )
            print("[LLM enhancement complete]")
        except Exception as e:
            print(f"[LLM enhancement skipped: {e}]")
    else:
        print("[LLM enhancement skipped: XAI_API_KEY not set]")

    print(f"\n{'='*70}")
    print("COMPREHENSIVE SMC TRADING PLAN")
    print(f"{'='*70}")

    # Position Analysis
    pos = plan['position_analysis']
    print(f"\n[POSITION ANALYSIS]")
    print(f"Current Price: ${pos['current_price']:.2f}")

    if pos['at_resistance']:
        print(f"Position: AT RESISTANCE")
    elif pos['at_support']:
        print(f"Position: AT SUPPORT")
    else:
        print(f"Position: BETWEEN ZONES")

    # Show nearest resistance
    if pos['nearest_resistance']:
        res = pos['nearest_resistance']
        print(f"\nNearest Resistance OB: ${res['price_range'][0]:.2f} - ${res['price_range'][1]:.2f}")
        print(f"  Distance: {res['distance_pct']:.2f}% away")
        if res['assessment']:
            assess = res['assessment']
            print(f"  Strength: {assess['strength_score']}/10 ({assess['strength_category']})")
            print(f"  {assess['assessment']}")
            print(f"  Hold Probability: {assess['hold_probability']:.0%} | Breakout Probability: {assess['breakout_probability']:.0%}")

    # Show nearest support
    if pos['nearest_support']:
        sup = pos['nearest_support']
        print(f"\nNearest Support OB: ${sup['price_range'][0]:.2f} - ${sup['price_range'][1]:.2f}")
        print(f"  Distance: {sup['distance_pct']:.2f}% away")
        if sup['assessment']:
            assess = sup['assessment']
            print(f"  Strength: {assess['strength_score']}/10 ({assess['strength_category']})")
            print(f"  {assess['assessment']}")
            print(f"  Hold Probability: {assess['hold_probability']:.0%} | Breakout Probability: {assess['breakout_probability']:.0%}")

    # Recommendation
    rec = plan['recommendation']
    print(f"\n{'='*70}")
    print(f"[RECOMMENDATION: {rec['action']}] - Confidence: {rec['confidence']}")
    print(f"{'='*70}")
    print(f"{rec['reason']}")
    if rec.get('alternative'):
        print(f"\nAlternative: {rec['alternative']}")

    # Primary Setup
    if plan['primary_setup']:
        setup = plan['primary_setup']
        print(f"\n{'='*70}")
        print(f"[PRIMARY SETUP: {setup['direction']}]")
        print(f"{'='*70}")
        print(f"Entry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        if setup['entry_zone']:
            print(f"Entry Zone: ${setup['entry_zone'][0]:.2f} - ${setup['entry_zone'][1]:.2f}")
        print(f"Stop Loss: ${setup['stop_loss']:.2f} ({setup['stop_loss_reason']})")
        print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"TP Reason: {setup['tp_reason']}")
        print(f"\nRisk/Reward:")
        print(f"  Risk: {setup['risk_pct']:.2f}%")
        print(f"  Reward (TP1): {setup['reward_pct_tp1']:.2f}%")
        print(f"  Reward (TP2): {setup['reward_pct_tp2']:.2f}%")
        print(f"  R:R Ratio (TP1): 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f}")
        print(f"  R:R Ratio (TP2): 1:{(setup['reward_pct_tp2']/setup['risk_pct']):.2f}")
        print(f"\nRationale: {setup['rationale']}")

        # Display LLM enhancement if available
        if 'llm_enhancement' in setup:
            llm = setup['llm_enhancement']
            print(f"\n--- LLM Contextual Intelligence ---")
            print(f"Confidence Level: {llm['confidence_level']}")
            print(f"Adjusted Hold Probability: {llm['adjusted_hold_probability']:.0%} (Rule-based: {llm['rule_based_hold_prob']:.0%})")
            print(f"Probability Adjustment: {llm['probability_adjustment']:+.0%}")
            print(f"\nKey Reasoning:")
            print(f"  {llm['key_reasoning']}")
            print(f"\nContextual Factors:")
            for factor in llm['contextual_factors']:
                print(f"  • {factor}")
            print(f"\nTop Risks:")
            for risk in llm['top_risks']:
                print(f"  ⚠️  {risk}")
            print(f"\nRecommended Action: {llm['recommended_action']}")
            print(f"Assessment vs Rules: {llm['adjustment_vs_rules']}")

    # Alternative Setup
    if plan['alternative_setup']:
        setup = plan['alternative_setup']
        print(f"\n{'='*70}")
        print(f"[ALTERNATIVE SETUP: {setup['direction']}]")
        print(f"{'='*70}")
        print(f"Trigger: {setup.get('trigger_condition', 'After primary setup closes')}")
        print(f"Entry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        if setup['entry_zone']:
            print(f"Entry Zone: ${setup['entry_zone'][0]:.2f} - ${setup['entry_zone'][1]:.2f}")
        print(f"Stop Loss: ${setup['stop_loss']:.2f}")
        print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"\nRisk/Reward:")
        print(f"  Risk: {setup['risk_pct']:.2f}%")
        print(f"  Reward (TP1): {setup['reward_pct_tp1']:.2f}%")
        print(f"  Reward (TP2): {setup['reward_pct_tp2']:.2f}%")
        print(f"  R:R Ratio (TP1): 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f}")
        print(f"\nRationale: {setup['rationale']}")

        # Display LLM enhancement if available
        if 'llm_enhancement' in setup:
            llm = setup['llm_enhancement']
            print(f"\n--- LLM Contextual Intelligence ---")
            print(f"Confidence Level: {llm['confidence_level']}")
            print(f"Adjusted Hold Probability: {llm['adjusted_hold_probability']:.0%} (Rule-based: {llm['rule_based_hold_prob']:.0%})")
            print(f"Probability Adjustment: {llm['probability_adjustment']:+.0%}")
            print(f"\nKey Reasoning:")
            print(f"  {llm['key_reasoning']}")
            print(f"\nContextual Factors:")
            for factor in llm['contextual_factors']:
                print(f"  • {factor}")
            print(f"\nTop Risks:")
            for risk in llm['top_risks']:
                print(f"  ⚠️  {risk}")
            print(f"\nRecommended Action: {llm['recommended_action']}")

    # Display LLM adjustment to recommendation if significant
    if 'llm_adjustment' in plan['recommendation']:
        llm_adj = plan['recommendation']['llm_adjustment']
        print(f"\n{'='*70}")
        print("[LLM RECOMMENDATION ADJUSTMENT]")
        print(f"{'='*70}")
        print(f"Rule-based: {llm_adj['original_action']}")
        print(f"LLM Suggests: {llm_adj['llm_recommended_action']}")
        print(f"Probability Change: {llm_adj['probability_adjustment']}")
        print(f"\nReason: {llm_adj['reason']}")

    # Display market context if available
    if 'market_context' in plan:
        ctx = plan['market_context']
        print(f"\n{'='*70}")
        print("[MARKET CONTEXT]")
        print(f"{'='*70}")
        print(f"Volatility: {ctx['volatility']['description']}")
        print(f"Trend: {ctx['trend']['description']}")
        print(f"Structure: {ctx['structure']['description']}")
        if ctx.get('news_sentiment'):
            print(f"News: {ctx['news_sentiment']}")
        if ctx.get('social_sentiment'):
            print(f"Social: {ctx['social_sentiment']}")

    print(f"\n{'='*70}\n")

    return plan


def prompt_trade_execution(symbol: str, signal: str, smc_analysis: dict = None, final_state: dict = None):
    """
    Prompt user to execute the trade via MT5 with SMC-based levels.

    Args:
        symbol: Trading symbol (e.g., XAUUSD)
        signal: Trade signal (BUY, SELL, HOLD)
        smc_analysis: SMC analysis dict with timeframe data
        final_state: Final state from analysis for reflection
    """
    if signal == "HOLD":
        print("\n[HOLD] No trade to execute.")
        return

    from tradingagents.dataflows.mt5_data import (
        execute_trade_signal,
        get_mt5_current_price,
        get_mt5_symbol_info,
        get_open_positions,
    )
    from tradingagents.dataflows.smc_utils import (
        suggest_smc_stop_loss,
        suggest_smc_take_profits
    )

    print(f"\n{'='*60}")
    print(f"TRADE EXECUTION: {signal} {symbol}")
    print(f"{'='*60}")

    # Check for open positions when SELL signal
    if signal == "SELL":
        try:
            open_positions = get_open_positions()
            long_positions = [p for p in open_positions if p['type'] == 'BUY' and symbol in p['symbol']]

            if long_positions:
                print("\n⚠️  WARNING: You have open LONG positions!")
                print("Consider closing them before entering a SHORT:")
                for p in long_positions:
                    profit_sign = "+" if p['profit'] >= 0 else ""
                    print(f"  {p['symbol']} BUY {p['volume']} lots @ {p['price_open']} | P/L: {profit_sign}{p['profit']:.2f}")

                response = input("\nContinue with SELL order? (y/N): ").strip().lower()
                if response != 'y':
                    print("Trade execution cancelled.")
                    return
        except Exception as e:
            pass  # Continue if we can't check positions

    # Get current price and symbol info
    try:
        price_info = get_mt5_current_price(symbol)
        symbol_info = get_mt5_symbol_info(symbol)
    except Exception as e:
        print(f"Error getting MT5 data: {e}")
        return

    current_price = price_info["ask"] if signal == "BUY" else price_info["bid"]
    digits = symbol_info.get("digits", 2)

    print(f"\nCurrent Price: ${current_price:.{digits}f}")
    print(f"Spread: {symbol_info['spread']} | Min lot: {symbol_info['volume_min']}")

    # Get SMC-based levels and entry strategy
    from tradingagents.dataflows.smc_utils import suggest_smc_entry_strategy

    smc_sl = None
    smc_tp = None
    smc_entry = None

    if smc_analysis:
        # Get ATR for stop loss fallback
        from tradingagents.risk.stop_loss import get_atr_for_symbol
        atr = get_atr_for_symbol(symbol, period=14)

        # Get SMC entry strategy
        smc_entry = suggest_smc_entry_strategy(
            smc_analysis=smc_analysis,
            direction=signal,
            current_price=current_price,
            primary_timeframe='1H'
        )

        # Use full multi-timeframe analysis for stop loss
        smc_sl = suggest_smc_stop_loss(
            smc_analysis=smc_analysis,  # Pass full multi-TF dict
            direction=signal,
            entry_price=current_price,
            atr=atr if atr > 0 else None,
            atr_multiplier=2.0,
            primary_timeframe='1H'
        )

        # Use first available timeframe for take profits
        first_tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
        if first_tf_data:
            smc_tps = suggest_smc_take_profits(
                smc_analysis=first_tf_data,
                direction=signal,
                entry_price=current_price,
                num_targets=3
            )
            if smc_tps:
                smc_tp = smc_tps[0]  # Use first TP target

    # Calculate fallback ATR-based levels (use already imported get_atr_for_symbol)
    from tradingagents.risk.stop_loss import DynamicStopLoss

    # ATR already fetched above if smc_analysis exists, otherwise get it now
    if not smc_analysis:
        from tradingagents.risk.stop_loss import get_atr_for_symbol
        atr = get_atr_for_symbol(symbol, period=14)

    dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5, risk_reward_ratio=2.0)

    if atr > 0:
        atr_levels = dsl.calculate_levels(current_price, atr, signal)
        atr_sl = round(atr_levels.stop_loss, digits)
        atr_tp = round(atr_levels.take_profit, digits)
    else:
        # Percentage fallback
        if signal == "BUY":
            atr_sl = round(current_price * 0.98, digits)
            atr_tp = round(current_price * 1.04, digits)
        else:
            atr_sl = round(current_price * 1.02, digits)
            atr_tp = round(current_price * 0.96, digits)

    # Display level options
    print("\n--- Stop Loss Options ---")
    if smc_sl:
        print(f"  1. SMC-based: ${smc_sl['price']:.{digits}f} ({smc_sl['reason']})")
    print(f"  2. ATR-based (2x): ${atr_sl:.{digits}f}")
    print(f"  3. Manual entry")

    sl_choice = input("\nSelect SL option (1/2/3) [1 if SMC available, else 2]: ").strip()

    if sl_choice == "1" and smc_sl:
        stop_loss = round(smc_sl['price'], digits)
        print(f"  ✓ Using SMC stop loss: ${stop_loss:.{digits}f}")
    elif sl_choice == "3":
        sl_input = input(f"  Enter stop loss price: ").strip()
        stop_loss = float(sl_input) if sl_input else atr_sl
    else:
        stop_loss = atr_sl
        print(f"  ✓ Using ATR stop loss: ${stop_loss:.{digits}f}")

    print("\n--- Take Profit Options ---")
    if smc_tp:
        print(f"  1. SMC-based: ${smc_tp['price']:.{digits}f} ({smc_tp['source']} zone)")
    print(f"  2. ATR-based (2:1 R:R): ${atr_tp:.{digits}f}")
    print(f"  3. Manual entry")

    tp_choice = input("\nSelect TP option (1/2/3) [1 if SMC available, else 2]: ").strip()

    if tp_choice == "1" and smc_tp:
        take_profit = round(smc_tp['price'], digits)
        print(f"  ✓ Using SMC take profit: ${take_profit:.{digits}f}")
    elif tp_choice == "3":
        tp_input = input(f"  Enter take profit price: ").strip()
        take_profit = float(tp_input) if tp_input else atr_tp
    else:
        take_profit = atr_tp
        print(f"  ✓ Using ATR take profit: ${take_profit:.{digits}f}")

    # Get lot size
    volume_input = input(f"\nLot size (min: {symbol_info['volume_min']}) [0.01]: ").strip()
    volume = float(volume_input) if volume_input else 0.01

    # Entry Strategy Options
    print("\n--- Entry Strategy Options ---")

    # Option 1: Market order at current price
    print(f"  1. MARKET order at ${current_price:.{digits}f}")
    print(f"     Immediate execution, no risk of missing trade")

    # Option 2: Limit order at SMC zone (if available)
    smc_limit_available = False
    if smc_entry and smc_entry.get('limit_entry') and 'price' in smc_entry['limit_entry']:
        smc_limit_available = True
        limit_data = smc_entry['limit_entry']
        print(f"  2. LIMIT order at ${limit_data['price']:.{digits}f} (SMC Order Block)")
        print(f"     Zone: ${limit_data['zone_bottom']:.{digits}f} - ${limit_data['zone_top']:.{digits}f}")
        print(f"     Confluence: {limit_data['confluence_score']:.1f} ({', '.join(limit_data['aligned_timeframes'])})")
        print(f"     Better R:R, may not fill")
    else:
        print(f"  2. LIMIT order (manual entry price)")

    print(f"  3. Manual (specify custom entry price and type)")

    # Show recommendation if available
    if smc_entry:
        rec = smc_entry['recommendation']
        if rec == 'MARKET':
            print(f"\n  Recommendation: MARKET (Option 1) - Price far from optimal zone")
        elif rec == 'LIMIT':
            print(f"\n  Recommendation: LIMIT (Option 2) - Wait for pullback to OB")
        elif rec == 'LIMIT_OR_MARKET':
            print(f"\n  Recommendation: Either MARKET or LIMIT - Price near optimal zone")

    entry_choice = input("\nSelect entry strategy (1/2/3) [1]: ").strip()

    if entry_choice == "2" and smc_limit_available:
        # Use SMC limit order
        use_limit = True
        entry_price = round(limit_data['price'], digits)
        print(f"  ✓ Using SMC limit order: ${entry_price:.{digits}f}")
    elif entry_choice == "2" and not smc_limit_available:
        # Manual limit order
        use_limit = True
        entry_input = input(f"  Enter limit price [current: ${current_price:.{digits}f}]: ").strip()
        entry_price = float(entry_input) if entry_input else current_price
        print(f"  ✓ Using limit order: ${entry_price:.{digits}f}")
    elif entry_choice == "3":
        # Manual entry
        order_type_input = input("  Order type - (M)arket or (L)imit? [M]: ").strip().upper()
        use_limit = order_type_input == "L"
        if use_limit:
            entry_input = input(f"  Entry price [current: ${current_price:.{digits}f}]: ").strip()
            entry_price = float(entry_input) if entry_input else current_price
        else:
            entry_price = current_price
        print(f"  ✓ Using {'limit' if use_limit else 'market'} order: ${entry_price:.{digits}f}")
    else:
        # Default: Market order
        use_limit = False
        entry_price = current_price
        print(f"  ✓ Using market order: ${entry_price:.{digits}f}")

    # Normalize prices to tick size
    tick_size = symbol_info.get('trade_tick_size', 0.01)
    entry_price = round(round(entry_price / tick_size) * tick_size, digits)
    stop_loss = round(round(stop_loss / tick_size) * tick_size, digits)
    take_profit = round(round(take_profit / tick_size) * tick_size, digits)

    # Order summary
    print(f"\n{'='*60}")
    print("ORDER SUMMARY")
    print(f"{'='*60}")
    print(f"  Symbol:      {symbol}")
    print(f"  Type:        {signal} {'LIMIT' if use_limit else 'MARKET'}")
    print(f"  Volume:      {volume} lots")
    print(f"  Entry:       ${entry_price:.{digits}f}")
    print(f"  Stop Loss:   ${stop_loss:.{digits}f}")
    print(f"  Take Profit: ${take_profit:.{digits}f}")

    # Calculate risk/reward
    if signal == "BUY":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk > 0:
        rr_ratio = reward / risk
        print(f"  Risk:Reward: 1:{rr_ratio:.2f}")

    # Confirm
    confirm = input("\nConfirm and place order? (y/N): ").strip().lower()

    if confirm != 'y':
        print("Order cancelled.")
        return

    # Execute the trade
    try:
        result = execute_trade_signal(
            symbol=symbol,
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=volume,
            use_limit_order=use_limit,
            comment=f"TradingAgents SMC {signal}",
        )

        if result.get("success"):
            print(f"\n✅ Order placed successfully!")
            print(f"  Order ID: {result.get('order_id')}")
            print(f"  Price: {result.get('price')}")
        else:
            print(f"\n❌ Order failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Error executing trade: {e}")


def reflect_on_trade(graph, final_state, returns_losses: float):
    """
    Reflect on a trade outcome and store lessons learned in memory.
    
    This is the key to the learning system:
    1. final_state contains all the analysis/decisions made
    2. returns_losses is the ACTUAL outcome (e.g., +5.2 for 5.2% profit)
    3. The reflector analyzes what went right/wrong
    4. Lessons are stored in memory, keyed by situation embedding
    5. Next time a similar situation occurs, lessons are retrieved
    
    Args:
        graph: The TradingAgentsGraph instance (must have memory enabled)
        final_state: The state dict from graph.propagate()
        returns_losses: Actual return percentage (positive=profit, negative=loss)
    """
    print(f"\n{'='*60}")
    print(f"REFLECTING ON TRADE OUTCOME: {returns_losses:+.2f}%")
    print(f"{'='*60}\n")
    
    if not graph.config.get("use_memory", False):
        print("⚠️ Memory is disabled. Enable use_memory=True to store lessons.")
        return
    
    # The reflect method analyzes the trade and stores lessons
    # It uses the LLM to generate insights like:
    # - "BUY decision was correct because news showed strong demand"
    # - "Should have been more cautious given the overbought RSI"
    graph.reflect(final_state, returns_losses)
    
    print("✅ Reflection complete! Lessons stored in memory.")
    print("   Next time a similar situation occurs, these lessons will be retrieved.")


def simulate_trade_lifecycle():
    """
    Demonstrate the full trade lifecycle with memory/reflection.
    
    This shows how the system learns from past trades:
    
    WORKFLOW:
    1. Analyze asset → Get BUY/SELL/HOLD signal
    2. Execute trade (simulated or real via MT5)
    3. Wait for outcome (next day, week, etc.)
    4. Calculate returns/losses
    5. Reflect on the trade → Store lessons in memory
    6. Future analyses will retrieve relevant lessons
    
    MEMORY RELATIONSHIP:
    - Situation = market_report + sentiment + news + fundamentals
    - Decision = what the agents decided (BUY/SELL/HOLD)
    - Outcome = actual returns/losses you provide
    - Lesson = LLM-generated analysis of what went right/wrong
    
    The lesson is stored with the situation embedding, so when a 
    similar situation occurs, the system retrieves past lessons.
    """
    print("\n" + "="*70)
    print("TRADE LIFECYCLE DEMONSTRATION WITH MEMORY")
    print("="*70)
    
    # Check for API key
    if not os.getenv("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set")
        return
    
    # Create graph with memory enabled
    graph = TradingAgentsGraph(config=COMMODITY_CONFIG, debug=True)
    
    # =========================================================================
    # STEP 1: Analyze and get signal
    # =========================================================================
    symbol = "XAUUSD"
    trade_date = "2025-12-26"
    
    print(f"\n📊 STEP 1: Analyzing {symbol} for {trade_date}...")
    final_state, signal = graph.propagate(symbol, trade_date)
    
    print(f"\n   Signal: {signal}")
    decision = final_state.get('final_trade_decision', 'N/A')
    if decision and decision != 'N/A':
        print(f"   Decision: {decision[:200]}...")
    else:
        print(f"   Decision: N/A")
    
    # =========================================================================
    # STEP 2: Execute trade (simulated)
    # =========================================================================
    print(f"\n💰 STEP 2: Executing trade based on signal: {signal}")
    entry_price = 2650.00  # Example entry price
    print(f"   Entry price: ${entry_price}")
    
    # =========================================================================
    # STEP 3: Wait for outcome (in real trading, this is days/weeks later)
    # =========================================================================
    print(f"\n⏳ STEP 3: Waiting for trade outcome...")
    print("   (In real trading, you'd wait and check the actual P&L)")
    
    # Simulated outcome - in reality you'd calculate this from actual prices
    exit_price = 2720.00  # Example exit price
    returns_losses = ((exit_price - entry_price) / entry_price) * 100
    
    print(f"   Exit price: ${exit_price}")
    print(f"   Returns: {returns_losses:+.2f}%")
    
    # =========================================================================
    # STEP 4: Reflect on the trade and store lessons
    # =========================================================================
    print(f"\n🧠 STEP 4: Reflecting on trade outcome...")
    reflect_on_trade(graph, final_state, returns_losses)
    
    # =========================================================================
    # STEP 5: Show how memory will be used in future
    # =========================================================================
    print(f"\n📚 STEP 5: How memory helps future trades...")
    print("""
    When you analyze a similar situation in the future:
    
    1. The system extracts the current situation (market + news + sentiment)
    2. It creates an embedding of this situation
    3. It searches memory for similar past situations
    4. Retrieved lessons are injected into agent prompts:
       - "Learn from Past Mistakes: {past_memory_str}"
       - "Here are your past reflections: {past_memory_str}"
    
    This allows agents to say things like:
    - "Last time gold was overbought with similar news, we lost money by buying"
    - "In a similar situation, holding was the right call"
    """)
    
    return final_state, signal, returns_losses


def main():
    """Main function to demonstrate commodity trading."""

    # Check for API key
    if not os.getenv("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in environment or .env file")
        print("Please set your xAI API key to use TradingAgents with Grok")
        return

    # Prompt for symbol/forex pair
    print("\n" + "="*70)
    print("TRADINGAGENTS - COMMODITY & FOREX ANALYSIS")
    print("="*70)
    print("\nAvailable symbols:")
    print("  Commodities: XAUUSD (Gold), XAGUSD (Silver), XPTUSD (Platinum), COPPER-C")
    print("  Forex Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, etc.")

    symbol_input = input("\nEnter symbol to analyze [XAGUSD]: ").strip().upper()
    symbol = symbol_input if symbol_input else "XAGUSD"

    print(f"\n✓ Analyzing: {symbol}")

    # Get most recent trading day (Friday if weekend)
    today = datetime.now()
    weekday = today.weekday()  # Monday=0, Sunday=6

    if weekday == 5:  # Saturday
        trade_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")  # Friday
    elif weekday == 6:  # Sunday
        trade_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")  # Friday
    else:  # Monday-Friday
        trade_date = today.strftime("%Y-%m-%d")

    print(f"✓ Trade date: {trade_date}")

    try:
        final_state, signal, smc_analysis = analyze_commodity(symbol, trade_date)
        print(f"\n[OK] Analysis complete! Signal: {signal}")

        # Display SMC levels if available
        if smc_analysis and signal != "HOLD":
            display_smc_levels(symbol, signal, smc_analysis, trade_date)

            # Display comprehensive multi-scenario SMC trading plan with LLM enhancement
            display_comprehensive_smc_plan(symbol, signal, smc_analysis, trade_date, final_state)

        # Prompt for trade execution
        if signal in ["BUY", "SELL"]:
            execute = input("\nWould you like to execute this trade? (y/N): ").strip().lower()
            if execute == 'y':
                prompt_trade_execution(symbol, signal, smc_analysis, final_state)

        # Uncomment to demonstrate the full trade lifecycle with reflection:
        # simulate_trade_lifecycle()

    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
