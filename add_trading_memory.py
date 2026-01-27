"""
Add permanent trading memories to the TradingAgents memory system.
These memories will be retrieved during analysis when similar situations arise.
"""

from tradingagents.agents.utils.memory import FinancialSituationMemory

# Config for fastembed (lightweight local embeddings, no PyTorch needed)
config = {
    "llm_provider": "xai",
    "embedding_provider": "fastembed",  # Uses fastembed library (no PyTorch)
    "fastembed_model": "BAAI/bge-small-en-v1.5",  # Good quality, small size
}

# Trading memories to add - these are personal trading discipline reminders
TRADING_MEMORIES = [
    # Cover longs before shorts reminder
    (
        """SELL signal detected. Bearish analysis. Short opportunity. Price pullback expected. 
        Overbought conditions. RSI divergence. Sentiment turning bearish. 
        Reversal pattern forming. Distribution phase. Smart money selling.""",
        
        """⚠️ CRITICAL REMINDER - COVER YOUR LONGS FIRST!
        
Before entering any SHORT position or when a SELL signal is detected:

1. CHECK FOR OPEN LONG POSITIONS - You often forget to close longs before pullbacks
2. PLACE SELL ORDERS to cover existing longs BEFORE the pullback occurs
3. Don't miss the opportunity to protect profits on your long positions

This applies especially to:
- Gold (XAUUSD) and Silver (XAGUSD) which move together
- Any asset showing overbought RSI + bearish divergence
- When sentiment shifts from bullish to bearish

ACTION: Run 'python -m cli.main positions' to check and close your longs!

You always regret not covering longs when you see a short coming. Don't repeat this mistake."""
    ),
    
    # Additional memory for pullback scenarios
    (
        """Price correction imminent. Taking profits. Reducing exposure. 
        Market topping. Resistance rejection. Failed breakout. 
        Bearish engulfing. Evening star pattern. Head and shoulders.""",
        
        """🔔 PULLBACK ALERT - PROTECT YOUR LONG POSITIONS!
        
When a pullback or correction is signaled:
1. Set trailing stops on profitable longs
2. Consider taking partial profits (50%) at key resistance
3. Place limit sell orders at current levels to lock in gains
4. Don't hold through the entire pullback hoping it reverses

Remember: A profit taken is better than a profit lost waiting for more."""
    ),
]


def add_memories():
    """Add trading memories to the trader's memory collection."""
    print("Initializing memory system with local embeddings...")
    
    # Add to trader memory (this is what gets checked during trading decisions)
    trader_memory = FinancialSituationMemory("trader_memory", config, persistent=True)
    
    # Check existing count
    existing_count = trader_memory.situation_collection.count()
    print(f"Existing memories in trader_memory: {existing_count}")
    
    # Add the trading memories
    print("\nAdding trading discipline memories...")
    trader_memory.add_situations(TRADING_MEMORIES)
    
    new_count = trader_memory.situation_collection.count()
    print(f"Total memories after adding: {new_count}")
    print(f"Added {new_count - existing_count} new memories")
    
    # Verify by querying
    print("\n--- Verification ---")
    test_query = "SELL signal for XAUUSD, bearish outlook, short opportunity"
    results = trader_memory.get_memories(test_query, n_matches=2)
    
    print(f"\nTest query: '{test_query}'")
    for i, result in enumerate(results):
        print(f"\nMatch {i+1} (similarity: {result['similarity_score']:.3f}):")
        print(f"Recommendation: {result['recommendation'][:200]}...")
    
    print("\n✅ Trading memories added successfully!")
    print("These will be retrieved when analyzing SELL signals or pullback scenarios.")


if __name__ == "__main__":
    add_memories()
