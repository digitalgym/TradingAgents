"""
Example: Trading commodities (Gold, Silver, Platinum, Copper) with TradingAgents.

This example shows how to use MT5 data from Vantage Australia to analyze commodities.

Prerequisites:
1. MT5 terminal must be running and logged into Vantage account
2. Set your OPENAI_API_KEY in .env file

Usage:
    python examples/trade_commodities.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph

# Configuration for commodity trading with MT5
# Start with default config and override what we need
from tradingagents.default_config import DEFAULT_CONFIG

COMMODITY_CONFIG = DEFAULT_CONFIG.copy()
COMMODITY_CONFIG.update({
    # Use MT5 for price data
    "data_vendors": {
        "core_stock_apis": "mt5",           # Use MT5 for OHLCV data
        "technical_indicators": "yfinance",  # yfinance works for indicators
        "fundamental_data": "openai",        # Not used for commodities
        "news_data": "google",               # Google news for commodity news
    },
    # Set asset type to commodity (auto-excludes fundamentals analyst)
    "asset_type": "commodity",
    # LLM settings - Using xAI Grok
    "llm_provider": "xai",
    "deep_think_llm": "grok-3-latest",
    "quick_think_llm": "grok-3-mini-latest",
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


def analyze_commodity(symbol: str, trade_date: str):
    """
    Analyze a commodity using TradingAgents.
    
    Args:
        symbol: Commodity symbol (XAUUSD, XAGUSD, XPTUSD, COPPER-C) or alias (gold, silver, etc.)
        trade_date: Date to analyze in YYYY-MM-DD format
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
    
    return final_state, signal


def main():
    """Main function to demonstrate commodity trading."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        print("Please set your OpenAI API key to use TradingAgents")
        return
    
    # Example: Analyze Gold
    # You can change this to any of: XAUUSD, XAGUSD, XPTUSD, COPPER-C
    # Or use aliases: gold, silver, platinum, copper
    
    symbol = "XAGUSD"  # Gold
    trade_date = "2025-12-26"  # Recent trading date
    
    try:
        final_state, signal = analyze_commodity(symbol, trade_date)
        print(f"\n✅ Analysis complete! Signal: {signal}")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
