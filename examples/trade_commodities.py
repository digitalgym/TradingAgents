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
from datetime import datetime
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
    
    # Run analysis
    final_state, signal = graph.propagate(symbol, trade_date)
    
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
        return final_state, signal, trade_id
    return final_state, signal


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
    print(f"   Decision: {final_state.get('final_trade_decision', 'N/A')[:200]}...")
    
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
    
    # Example: Analyze Silver
    # You can change this to any of: XAUUSD, XAGUSD, XPTUSD, COPPER-C
    # Or use aliases: gold, silver, platinum, copper
    
    symbol = "XAGUSD"  # Silver
    trade_date = "2025-12-26"  # Recent trading date
    
    try:
        final_state, signal = analyze_commodity(symbol, trade_date)
        print(f"\n✅ Analysis complete! Signal: {signal}")
        
        # Uncomment to demonstrate the full trade lifecycle with reflection:
        # simulate_trade_lifecycle()
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
