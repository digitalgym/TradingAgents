"""Test local embeddings with sentence-transformers."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_local_embeddings():
    """Test local sentence-transformers embeddings."""
    print("Testing local embeddings with sentence-transformers...")
    
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    test_texts = ["Hello world", "This is a test about gold prices"]
    
    print(f"\nEncoding: {test_texts}")
    
    embeddings = model.encode(test_texts, convert_to_numpy=True)
    
    print(f"✅ Got {len(embeddings)} embeddings")
    print(f"   Embedding dimensions: {len(embeddings[0])}")
    
    return embeddings


def test_memory_with_local():
    """Test FinancialSituationMemory with local embeddings."""
    print("\n" + "="*60)
    print("Testing FinancialSituationMemory with local embeddings...")
    print("="*60)
    
    from tradingagents.agents.utils.memory import FinancialSituationMemory
    
    config = {
        "llm_provider": "xai",
        "embedding_provider": "local",
    }
    
    memory = FinancialSituationMemory("test_memory", config)
    
    # Add some test situations
    test_data = [
        ("Gold prices rising due to inflation fears", "Consider buying gold ETFs"),
        ("Market volatility increasing", "Reduce position sizes"),
    ]
    
    print("\nAdding test situations...")
    memory.add_situations(test_data)
    print(f"✅ Added {len(test_data)} situations")
    
    # Query
    query = "Inflation is causing gold to go up"
    print(f"\nQuerying: '{query}'")
    
    results = memory.get_memories(query, n_matches=1)
    
    print(f"✅ Got {len(results)} results")
    for r in results:
        print(f"   Match: {r['matched_situation'][:50]}...")
        print(f"   Score: {r['similarity_score']:.3f}")
        print(f"   Recommendation: {r['recommendation']}")


if __name__ == "__main__":
    test_local_embeddings()
    test_memory_with_local()
    print("\n✅ All tests passed!")
