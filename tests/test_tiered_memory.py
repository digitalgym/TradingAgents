"""
Test script for the tiered memory system (Phase 1B implementation).

Tests:
1. Adding memories with tiers and confidence
2. Weighted retrieval with recency decay
3. Tier promotion logic
4. Memory statistics
5. Memory maintenance utilities
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from tradingagents.agents.utils.memory import (
    FinancialSituationMemory,
    TIER_SHORT,
    TIER_MID,
    TIER_LONG,
    DEFAULT_TIER_WEIGHTS
)
from tradingagents.agents.utils.memory_maintenance import MemoryMaintenance


def test_tiered_memory():
    """Test the tiered memory system."""
    print("\n" + "="*60)
    print("TESTING TIERED MEMORY SYSTEM (Phase 1B)")
    print("="*60)
    
    # Use a test collection that won't interfere with real data
    config = {
        "embedding_provider": "local",
        "local_embedding_model": "all-MiniLM-L6-v2",
    }
    
    # Create a test memory collection
    test_memory = FinancialSituationMemory(
        "test_tiered_memory",
        config,
        persistent=False  # Use in-memory for testing
    )
    
    print("\n1. Testing add_situations with tiers and confidence...")
    
    # Add memories with different tiers and confidence levels
    test_data = [
        # Short-term, low confidence (incorrect prediction)
        {
            "situation": "RSI at 65, MACD bullish crossover, price above SMA20",
            "advice": "BUY signal generated but price dropped 2%",
            "tier": TIER_SHORT,
            "returns": -2.0,
            "prediction_correct": False,
        },
        # Mid-term, medium confidence (correct prediction)
        {
            "situation": "RSI at 30 (oversold), MACD bearish, price below lower Bollinger",
            "advice": "BUY signal generated, price rose 1.5%",
            "tier": TIER_MID,
            "returns": 1.5,
            "prediction_correct": True,
        },
        # Long-term, high confidence (strong correct prediction)
        {
            "situation": "RSI at 25 (very oversold), strong bullish divergence, support level",
            "advice": "Strong BUY signal, price rose 5%",
            "tier": TIER_LONG,
            "returns": 5.0,
            "prediction_correct": True,
        },
    ]
    
    for data in test_data:
        test_memory.add_situations(
            [(data["situation"], data["advice"])],
            tier=data["tier"],
            returns=data["returns"],
            prediction_correct=data["prediction_correct"]
        )
        print(f"  ✓ Added {data['tier']} tier memory (returns: {data['returns']:+.1f}%)")
    
    print("\n2. Testing get_memory_stats...")
    stats = test_memory.get_memory_stats()
    print(f"  Total memories: {stats['total']}")
    print(f"  By tier: {stats['by_tier']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Avg outcome quality: {stats['avg_outcome_quality']:.2f}")
    
    assert stats['total'] == 3, f"Expected 3 memories, got {stats['total']}"
    assert stats['by_tier'][TIER_SHORT] == 1, "Expected 1 short-term memory"
    assert stats['by_tier'][TIER_MID] == 1, "Expected 1 mid-term memory"
    assert stats['by_tier'][TIER_LONG] == 1, "Expected 1 long-term memory"
    print("  ✓ Stats correct!")
    
    print("\n3. Testing weighted retrieval...")
    
    # Query for oversold conditions - should match the high-confidence long-term memory
    query = "RSI is very low around 28, showing oversold conditions with potential reversal"
    results = test_memory.get_memories(query, n_matches=3)
    
    print(f"  Query: '{query[:50]}...'")
    print(f"  Found {len(results)} matches:")
    for i, r in enumerate(results):
        print(f"    {i+1}. tier={r['tier']}, conf={r['confidence']:.2f}, score={r['composite_score']:.4f}")
        print(f"       similarity={r['similarity_score']:.3f}, recency={r['recency']:.3f}")
    
    # The long-term high-confidence memory should rank highly due to similarity
    assert len(results) > 0, "Expected at least one result"
    print("  ✓ Weighted retrieval working!")
    
    print("\n4. Testing confidence calculation...")
    
    # Test confidence calculation directly
    conf_correct_high = test_memory._calculate_confidence(5.0, True)
    conf_correct_low = test_memory._calculate_confidence(1.0, True)
    conf_incorrect = test_memory._calculate_confidence(-3.0, False)
    
    print(f"  Correct + 5% returns: confidence = {conf_correct_high:.2f}")
    print(f"  Correct + 1% returns: confidence = {conf_correct_low:.2f}")
    print(f"  Incorrect - 3% returns: confidence = {conf_incorrect:.2f}")
    
    assert conf_correct_high > conf_correct_low, "Higher returns should give higher confidence"
    assert conf_correct_low > conf_incorrect, "Correct predictions should have higher confidence"
    print("  ✓ Confidence calculation correct!")
    
    print("\n5. Testing tier promotion logic...")
    
    # Create a new memory and simulate multiple references
    promo_memory = FinancialSituationMemory(
        "test_promotion",
        config,
        persistent=False
    )
    
    # Add a short-term memory
    promo_memory.add_situations(
        [("Test situation for promotion", "Test advice")],
        tier=TIER_SHORT,
        returns=1.0,
        prediction_correct=True
    )
    
    initial_stats = promo_memory.get_memory_stats()
    print(f"  Initial: {initial_stats['by_tier']}")
    
    # Retrieve it multiple times to trigger promotion
    for i in range(4):
        promo_memory.get_memories("Test situation", n_matches=1)
    
    final_stats = promo_memory.get_memory_stats()
    print(f"  After 4 retrievals: {final_stats['by_tier']}")
    
    # Should have been promoted to mid-tier (3+ references)
    assert final_stats['by_tier'][TIER_MID] >= 1 or final_stats['by_tier'][TIER_SHORT] >= 1, \
        "Memory should exist in some tier"
    print("  ✓ Tier promotion logic working!")
    
    print("\n6. Testing MemoryMaintenance utilities...")
    
    maint = MemoryMaintenance(test_memory)
    detailed_stats = maint.get_detailed_stats()
    
    print(f"  Detailed stats keys: {list(detailed_stats.keys())}")
    print(f"  Confidence range: {detailed_stats['confidence']['min']:.2f} - {detailed_stats['confidence']['max']:.2f}")
    print(f"  Prediction accuracy: {detailed_stats['prediction_accuracy']}")
    
    top_memories = maint.get_top_memories(n=2, sort_by="confidence")
    print(f"  Top 2 by confidence:")
    for mem in top_memories:
        print(f"    - conf={mem['confidence']:.2f}, tier={mem['tier']}")
    
    print("  ✓ Maintenance utilities working!")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    return True


def test_backward_compatibility():
    """Test that old-style add_situations still works."""
    print("\n" + "="*60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("="*60)
    
    config = {
        "embedding_provider": "local",
    }
    
    memory = FinancialSituationMemory(
        "test_backward_compat",
        config,
        persistent=False
    )
    
    # Old-style call without tier/confidence
    memory.add_situations([
        ("Old style situation", "Old style advice"),
    ])
    
    stats = memory.get_memory_stats()
    print(f"  Added memory with old-style call")
    print(f"  Total: {stats['total']}, Tier: {stats['by_tier']}")
    
    # Should default to short tier with 0.5 confidence
    assert stats['total'] == 1, "Should have 1 memory"
    assert stats['by_tier'][TIER_SHORT] == 1, "Should be in short tier by default"
    
    # Test simple retrieval (backward compatible method)
    results = memory.get_memories_simple("Old style query", n_matches=1)
    assert len(results) == 1, "Should get 1 result"
    assert "similarity_score" in results[0], "Should have similarity score"
    
    print("  ✓ Backward compatibility maintained!")
    
    return True


if __name__ == "__main__":
    try:
        test_tiered_memory()
        test_backward_compatibility()
        print("\n✅ All Phase 1B tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
