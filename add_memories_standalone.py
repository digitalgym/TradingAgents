"""
Standalone script to add trading memories without loading the full tradingagents module.
Uses sentence-transformers for embeddings (available in .venv).
"""

import chromadb
from pathlib import Path

# Use sentence-transformers (available in .venv)
from sentence_transformers import SentenceTransformer

MEMORY_DB_PATH = Path(__file__).parent / "memory_db"

# Trading memories to add - personal trading discipline reminders
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
    """Add trading memories using sentence-transformers."""
    print("Initializing sentence-transformers model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to ChromaDB at {MEMORY_DB_PATH}...")
    MEMORY_DB_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(MEMORY_DB_PATH))

    # Add to trader_memory collection (used during trading decisions)
    collection = chroma_client.get_or_create_collection(name="trader_memory")

    existing_count = collection.count()
    print(f"Existing memories in trader_memory: {existing_count}")

    # Prepare data for insertion
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    offset = existing_count

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print("\nGenerating embeddings for trading memories...")
    for i, (situation, recommendation) in enumerate(TRADING_MEMORIES):
        documents.append(situation)
        metadatas.append({
            "recommendation": recommendation,
            "tier": "long",  # These are high-value permanent lessons
            "confidence": 0.9,
            "outcome_quality": 0.9,
            "timestamp": timestamp,
            "reference_count": 0,
            "prediction_correct": "True",
        })
        ids.append(str(offset + i))

        # Generate embedding
        emb = embedding_model.encode(situation, convert_to_numpy=True)
        embeddings.append(emb.tolist())
        print(f"  [OK] Memory {i+1}/{len(TRADING_MEMORIES)}")

    # Add to collection
    print("\nAdding memories to database...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids,
    )

    new_count = collection.count()
    print(f"Total memories after adding: {new_count}")
    print(f"Added {new_count - existing_count} new memories")

    # Verify by querying
    print("\n--- Verification ---")
    test_query = "SELL signal for XAUUSD, bearish outlook, short opportunity"
    query_embedding = embedding_model.encode(test_query, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["metadatas", "documents", "distances"],
    )

    print(f"\nTest query: '{test_query}'")
    for i in range(len(results["documents"][0])):
        similarity = 1 - results["distances"][0][i]
        rec = results["metadatas"][0][i].get("recommendation", "")
        print(f"\nMatch {i+1} (similarity: {similarity:.3f}):")
        print(f"Recommendation: {rec[:200]}...")

    print("\n[SUCCESS] Trading memories added successfully!")
    print("These will be retrieved when analyzing SELL signals or pullback scenarios.")


if __name__ == "__main__":
    add_memories()
