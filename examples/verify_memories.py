"""
Memory Verification Script

Use this script to verify that the Daily Learning Cycle is creating
memories correctly in the ChromaDB database.

Usage:
    python examples/verify_memories.py
    python examples/verify_memories.py --collection prediction_accuracy
    python examples/verify_memories.py --search "XAUUSD BUY"
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings

# Memory database path
MEMORY_DB_PATH = Path(__file__).parent.parent / "memory_db"


def list_collections():
    """List all collections in the memory database."""
    if not MEMORY_DB_PATH.exists():
        print(f"Memory database not found at: {MEMORY_DB_PATH}")
        return []

    client = chromadb.PersistentClient(path=str(MEMORY_DB_PATH))
    collections = client.list_collections()

    print(f"\n{'='*60}")
    print(f"MEMORY DATABASE COLLECTIONS")
    print(f"Path: {MEMORY_DB_PATH}")
    print(f"{'='*60}\n")

    for coll in collections:
        count = coll.count()
        print(f"  [{coll.name}] - {count} memories")

    print()
    return [c.name for c in collections]


def inspect_collection(collection_name: str, limit: int = 5):
    """Inspect memories in a specific collection."""
    client = chromadb.PersistentClient(path=str(MEMORY_DB_PATH))

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return

    count = collection.count()

    print(f"\n{'='*60}")
    print(f"COLLECTION: {collection_name}")
    print(f"Total memories: {count}")
    print(f"{'='*60}\n")

    if count == 0:
        print("  (empty collection)")
        return

    # Get recent memories
    results = collection.get(
        limit=limit,
        include=["metadatas", "documents"]
    )

    for i, (doc, meta, id) in enumerate(zip(
        results["documents"],
        results["metadatas"],
        results["ids"]
    ), 1):
        print(f"--- Memory {i} (ID: {id[:20]}...) ---")

        # Parse metadata
        tier = meta.get("tier", "unknown")
        confidence = meta.get("confidence", "N/A")
        timestamp = meta.get("timestamp", "N/A")
        outcome_quality = meta.get("outcome_quality", "N/A")
        market_regime = meta.get("market_regime", "N/A")

        print(f"  Tier: {tier}")
        print(f"  Confidence: {confidence}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Outcome Quality: {outcome_quality}")
        print(f"  Market Regime: {market_regime}")

        # Show recommendation snippet
        rec = meta.get("recommendation", "")
        if rec:
            rec_preview = rec[:200].replace('\n', ' ')
            print(f"  Recommendation: {rec_preview}...")

        print()


def search_memories(collection_name: str, query: str, n_results: int = 3):
    """Search for similar memories in a collection."""
    from tradingagents.agents.utils.memory import FinancialSituationMemory
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["embedding_provider"] = "local"  # Use local embeddings

    print(f"\n{'='*60}")
    print(f"SEARCHING: '{query}'")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}\n")

    try:
        memory = FinancialSituationMemory(collection_name, config)
        results = memory.get_memories(query, n_matches=n_results)

        if not results:
            print("  No matching memories found.")
            return

        for i, mem in enumerate(results, 1):
            print(f"--- Match {i} (Score: {mem.get('composite_score', 0):.4f}) ---")
            print(f"  Tier: {mem.get('tier', 'unknown')}")
            print(f"  Confidence: {mem.get('confidence', 'N/A')}")
            print(f"  Similarity: {mem.get('similarity_score', 0):.4f}")

            rec = mem.get("recommendation", "")
            if rec:
                rec_preview = rec[:300].replace('\n', ' ')
                print(f"  Recommendation: {rec_preview}...")

            print()

    except Exception as e:
        print(f"Error searching memories: {e}")
        import traceback
        traceback.print_exc()


def get_memory_stats():
    """Get statistics about memory usage."""
    client = chromadb.PersistentClient(path=str(MEMORY_DB_PATH))

    print(f"\n{'='*60}")
    print(f"MEMORY STATISTICS")
    print(f"{'='*60}\n")

    collections = client.list_collections()

    total_memories = 0
    stats_by_collection = {}

    for coll in collections:
        count = coll.count()
        total_memories += count

        # Get tier distribution
        try:
            results = coll.get(include=["metadatas"])
            tiers = {"short": 0, "mid": 0, "long": 0, "unknown": 0}

            for meta in results.get("metadatas", []):
                tier = meta.get("tier", "unknown")
                tiers[tier] = tiers.get(tier, 0) + 1

            stats_by_collection[coll.name] = {
                "total": count,
                "tiers": tiers
            }
        except Exception:
            stats_by_collection[coll.name] = {"total": count, "tiers": {}}

    print(f"Total memories across all collections: {total_memories}\n")

    for name, stats in stats_by_collection.items():
        print(f"[{name}]")
        print(f"  Total: {stats['total']}")
        if stats['tiers']:
            print(f"  Tiers: short={stats['tiers'].get('short', 0)}, "
                  f"mid={stats['tiers'].get('mid', 0)}, "
                  f"long={stats['tiers'].get('long', 0)}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify Daily Cycle Memories")
    parser.add_argument("--collection", "-c", type=str, help="Collection to inspect")
    parser.add_argument("--search", "-s", type=str, help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Number of results")
    parser.add_argument("--stats", action="store_true", help="Show memory statistics")

    args = parser.parse_args()

    if not MEMORY_DB_PATH.exists():
        print(f"Memory database not found at: {MEMORY_DB_PATH}")
        print("Run the daily cycle first to create memories.")
        return

    # List collections
    collections = list_collections()

    if args.stats:
        get_memory_stats()
        return

    if args.collection:
        if args.search:
            search_memories(args.collection, args.search, args.limit)
        else:
            inspect_collection(args.collection, args.limit)
    elif args.search and collections:
        # Search in prediction_accuracy if it exists, otherwise first collection
        search_coll = "prediction_accuracy" if "prediction_accuracy" in collections else collections[0]
        search_memories(search_coll, args.search, args.limit)


if __name__ == "__main__":
    main()
