import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Default persistent storage path
MEMORY_DB_PATH = Path(__file__).parent.parent.parent.parent / "memory_db"

# Memory tier constants
TIER_SHORT = "short"   # Recent memories (last 30 days, decay quickly)
TIER_MID = "mid"       # Medium-term (promoted after 3+ references or good outcomes)
TIER_LONG = "long"     # Long-term (high-impact lessons, persist indefinitely)

# Default tier weights for retrieval
DEFAULT_TIER_WEIGHTS = {
    TIER_SHORT: 0.5,
    TIER_MID: 0.3,
    TIER_LONG: 0.2,
}


class FinancialSituationMemory:
    def __init__(self, name, config, persistent=True):
        self.llm_provider = config.get("llm_provider", "openai").lower()
        backend_url = config.get("backend_url", "https://api.openai.com/v1")
        embedding_provider = config.get("embedding_provider", "auto").lower()
        self.name = name
        
        # Determine embedding provider
        # "auto" = use local for xAI/grok, OpenAI for others
        # "local" = always use sentence-transformers
        # "openai" = always use OpenAI API
        # "ollama" = use Ollama local embeddings
        
        if embedding_provider == "local" or (embedding_provider == "auto" and self.llm_provider in ["xai", "grok"]):
            # Use local sentence-transformers (no API needed)
            self._use_local = True
            self._local_model = None  # Lazy load
            self._local_model_name = config.get("local_embedding_model", "all-MiniLM-L6-v2")
            self.client = None
        elif embedding_provider == "ollama" or backend_url == "http://localhost:11434/v1":
            # Ollama local
            self._use_local = False
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url="http://localhost:11434/v1")
        else:
            # Default to OpenAI for embeddings
            self._use_local = False
            self.embedding = "text-embedding-3-small"
            self.client = OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Use persistent storage if enabled
        if persistent:
            db_path = config.get("memory_db_path", MEMORY_DB_PATH)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        
        # Get or create collection (persistent collections persist across restarts)
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    def _get_local_model(self):
        """Lazy load the sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._local_model_name)
        return self._local_model

    def get_embedding(self, text):
        """Get embedding for a text"""
        if self._use_local:
            model = self._get_local_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            response = self.client.embeddings.create(
                model=self.embedding, input=text
            )
            return response.data[0].embedding

    def add_situations(
        self, 
        situations_and_advice: List[Tuple[str, str]], 
        tier: str = TIER_SHORT,
        confidence: float = 0.5,
        outcome_quality: float = 0.5,
        prediction_correct: bool = None,
        returns: float = None,
        regime: Optional[Dict[str, str]] = None
    ):
        """
        Add financial situations and their corresponding advice with tier and confidence metadata.
        
        Args:
            situations_and_advice: List of tuples (situation, recommendation)
            tier: Memory tier - "short", "mid", or "long" (default: "short")
            confidence: Confidence score 0.0-1.0 based on outcome quality (default: 0.5)
            outcome_quality: Quality of the outcome 0.0-1.0 (default: 0.5)
            prediction_correct: Whether the prediction was correct (optional)
            returns: The returns/P&L from this prediction (optional, used to calculate confidence)
            regime: Optional regime dict (e.g., {"market_regime": "trending-up", "volatility_regime": "high"})
        """
        # Calculate confidence from returns if provided
        if returns is not None and prediction_correct is not None:
            confidence = self._calculate_confidence(returns, prediction_correct)
            outcome_quality = self._calculate_outcome_quality(returns, prediction_correct)
        
        situations = []
        advice = []
        ids = []
        embeddings = []
        metadatas = []

        offset = self.situation_collection.count()
        timestamp = datetime.now().isoformat()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))
            metadata_dict = {
                "recommendation": recommendation,
                "tier": tier,
                "confidence": confidence,
                "outcome_quality": outcome_quality,
                "timestamp": timestamp,
                "reference_count": 0,
                "prediction_correct": str(prediction_correct) if prediction_correct is not None else "unknown",
            }
            
            # Add regime metadata if provided
            if regime:
                metadata_dict["market_regime"] = regime.get("market_regime")
                metadata_dict["volatility_regime"] = regime.get("volatility_regime")
                metadata_dict["expansion_regime"] = regime.get("expansion_regime")
            
            metadatas.append(metadata_dict)

        self.situation_collection.add(
            documents=situations,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
    
    def _calculate_confidence(self, returns: float, prediction_correct: bool) -> float:
        """
        Calculate confidence score based on outcome.
        
        High confidence = correct prediction + strong returns
        Low confidence = incorrect prediction or weak returns
        
        Args:
            returns: The percentage returns from the trade
            prediction_correct: Whether the prediction direction was correct
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if prediction_correct:
            # Correct prediction: confidence based on magnitude of returns
            # 0% returns = 0.5 confidence, 5% returns = 1.0 confidence
            return min(0.5 + abs(returns) / 10, 1.0)
        else:
            # Incorrect prediction: low confidence
            # More wrong = lower confidence
            return max(0.2, 0.5 - abs(returns) / 10)
    
    def _calculate_outcome_quality(self, returns: float, prediction_correct: bool) -> float:
        """
        Calculate outcome quality score.
        
        Args:
            returns: The percentage returns from the trade
            prediction_correct: Whether the prediction direction was correct
            
        Returns:
            Outcome quality score between 0.0 and 1.0
        """
        if prediction_correct:
            # Correct prediction: quality based on returns magnitude
            return min(0.6 + abs(returns) / 10, 1.0)
        else:
            # Incorrect: quality inversely related to loss magnitude
            return max(0.1, 0.4 - abs(returns) / 10)

    def get_memories(
        self, 
        current_situation: str, 
        n_matches: int = 5,
        tier_weights: Dict[str, float] = None,
        min_confidence: float = 0.0,
        recency_half_life_days: float = 30.0,
        regime_filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find matching recommendations with weighted scoring.
        
        Score = similarity × tier_weight × confidence × recency_decay
        
        Args:
            current_situation: The current market situation to match against
            n_matches: Number of matches to return (default: 5)
            tier_weights: Custom tier weights dict (default: {"short": 0.5, "mid": 0.3, "long": 0.2})
            min_confidence: Minimum confidence threshold (default: 0.0)
            recency_half_life_days: Half-life for recency decay in days (default: 30)
            regime_filter: Optional regime filter dict (e.g., {"market_regime": "trending-up", "volatility_regime": "high"})
                          Only return memories from similar regimes
            
        Returns:
            List of matched results sorted by composite score
        """
        if tier_weights is None:
            tier_weights = DEFAULT_TIER_WEIGHTS
        
        query_embedding = self.get_embedding(current_situation)
        
        # Get more candidates than needed for re-ranking
        n_candidates = min(max(n_matches * 4, 20), self.situation_collection.count())
        if n_candidates == 0:
            return []

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["metadatas", "documents", "distances"],
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []

        # Re-rank by composite score
        scored_results = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            similarity = 1 - results["distances"][0][i]
            
            # Apply regime filter if specified
            if regime_filter:
                regime_match = self._check_regime_match(metadata, regime_filter)
                if not regime_match:
                    continue
            
            # Get confidence (handle legacy memories without confidence)
            confidence = metadata.get("confidence", 0.5)
            if confidence < min_confidence:
                continue
            
            # Get tier weight
            tier = metadata.get("tier", TIER_SHORT)
            tier_weight = tier_weights.get(tier, 0.3)
            
            # Calculate recency decay (exponential)
            timestamp_str = metadata.get("timestamp")
            if timestamp_str:
                try:
                    memory_time = datetime.fromisoformat(timestamp_str)
                    age_days = (datetime.now() - memory_time).days
                    recency = np.exp(-age_days / recency_half_life_days)
                except (ValueError, TypeError):
                    recency = 0.5  # Default for malformed timestamps
            else:
                recency = 0.5  # Default for legacy memories without timestamp
            
            # Composite score
            score = similarity * tier_weight * confidence * recency
            
            scored_results.append({
                "matched_situation": results["documents"][0][i],
                "recommendation": metadata.get("recommendation", ""),
                "similarity_score": similarity,
                "composite_score": score,
                "tier": tier,
                "confidence": confidence,
                "recency": recency,
                "timestamp": timestamp_str,
                "outcome_quality": metadata.get("outcome_quality", 0.5),
                "reference_count": metadata.get("reference_count", 0),
                "id": results["ids"][0][i] if results.get("ids") else str(i),
                "market_regime": metadata.get("market_regime"),
                "volatility_regime": metadata.get("volatility_regime"),
                "expansion_regime": metadata.get("expansion_regime"),
            })
        
        # Sort by composite score (highest first)
        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Increment reference count for returned memories
        self._increment_reference_counts([r["id"] for r in scored_results[:n_matches]])
        
        return scored_results[:n_matches]
    
    def _increment_reference_counts(self, ids: List[str]):
        """
        Increment reference count for retrieved memories.
        This is used for tier promotion decisions.
        """
        for memory_id in ids:
            try:
                # Get current metadata
                result = self.situation_collection.get(ids=[memory_id], include=["metadatas"])
                if result["metadatas"]:
                    metadata = result["metadatas"][0]
                    new_count = metadata.get("reference_count", 0) + 1
                    metadata["reference_count"] = new_count
                    
                    # Check for tier promotion
                    self._check_tier_promotion(memory_id, metadata)
                    
                    # Update metadata
                    self.situation_collection.update(
                        ids=[memory_id],
                        metadatas=[metadata]
                    )
            except Exception:
                pass  # Silently ignore update failures
    
    def _check_regime_match(self, metadata: Dict[str, Any], regime_filter: Dict[str, str]) -> bool:
        """
        Check if memory's regime matches the filter criteria.
        
        Args:
            metadata: Memory metadata containing regime fields
            regime_filter: Dict with regime criteria (e.g., {"market_regime": "trending-up"})
        
        Returns:
            True if regime matches (or no regime data in memory), False otherwise
        """
        for regime_key, regime_value in regime_filter.items():
            memory_regime = metadata.get(regime_key)
            
            # If memory has no regime data (legacy), include it
            if memory_regime is None:
                continue
            
            # If regime doesn't match, exclude this memory
            if memory_regime != regime_value:
                return False
        
        return True
    
    def _check_tier_promotion(self, memory_id: str, metadata: Dict[str, Any]):
        """
        Check if a memory should be promoted to a higher tier.
        
        Promotion rules:
        - Short -> Mid: Referenced 3+ times OR outcome_quality > 0.7
        - Mid -> Long: Confidence > 0.8 AND outcome_quality > 0.8
        """
        current_tier = metadata.get("tier", TIER_SHORT)
        reference_count = metadata.get("reference_count", 0)
        confidence = metadata.get("confidence", 0.5)
        outcome_quality = metadata.get("outcome_quality", 0.5)
        
        if current_tier == TIER_SHORT:
            # Promote to mid-term if frequently referenced or high quality
            if reference_count >= 3 or outcome_quality > 0.7:
                metadata["tier"] = TIER_MID
        elif current_tier == TIER_MID:
            # Promote to long-term if high confidence and high quality
            if confidence > 0.8 and outcome_quality > 0.8:
                metadata["tier"] = TIER_LONG
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory collection.
        
        Returns:
            Dictionary with memory statistics
        """
        total = self.situation_collection.count()
        if total == 0:
            return {
                "total": 0,
                "by_tier": {TIER_SHORT: 0, TIER_MID: 0, TIER_LONG: 0},
                "avg_confidence": 0.0,
                "avg_outcome_quality": 0.0,
            }
        
        # Get all memories
        all_memories = self.situation_collection.get(include=["metadatas"])
        
        tier_counts = {TIER_SHORT: 0, TIER_MID: 0, TIER_LONG: 0}
        confidences = []
        outcome_qualities = []
        reference_counts = []
        
        for metadata in all_memories["metadatas"]:
            tier = metadata.get("tier", TIER_SHORT)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            if "confidence" in metadata:
                confidences.append(metadata["confidence"])
            if "outcome_quality" in metadata:
                outcome_qualities.append(metadata["outcome_quality"])
            if "reference_count" in metadata:
                reference_counts.append(metadata["reference_count"])
        
        return {
            "total": total,
            "by_tier": tier_counts,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "avg_outcome_quality": np.mean(outcome_qualities) if outcome_qualities else 0.0,
            "avg_reference_count": np.mean(reference_counts) if reference_counts else 0.0,
            "most_referenced": max(reference_counts) if reference_counts else 0,
        }
    
    def update_memory_confidence(
        self, 
        memory_id: str, 
        returns: float, 
        prediction_correct: bool
    ):
        """
        Update confidence and outcome quality for an existing memory after evaluation.
        
        Args:
            memory_id: The ID of the memory to update
            returns: The returns from the trade
            prediction_correct: Whether the prediction was correct
        """
        try:
            result = self.situation_collection.get(ids=[memory_id], include=["metadatas"])
            if result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["confidence"] = self._calculate_confidence(returns, prediction_correct)
                metadata["outcome_quality"] = self._calculate_outcome_quality(returns, prediction_correct)
                metadata["prediction_correct"] = str(prediction_correct)
                
                # Check for tier promotion
                self._check_tier_promotion(memory_id, metadata)
                
                self.situation_collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
        except Exception as e:
            print(f"Warning: Failed to update memory confidence: {e}")
    
    def get_memories_simple(self, current_situation: str, n_matches: int = 1) -> List[Dict[str, Any]]:
        """
        Simple memory retrieval without weighted scoring (backward compatible).
        
        Args:
            current_situation: The current market situation to match against
            n_matches: Number of matches to return
            
        Returns:
            List of matched results
        """
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i].get("recommendation", ""),
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
