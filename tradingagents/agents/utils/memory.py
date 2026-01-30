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

# PHASE 3: Agent-specific memory retrieval configuration
# Different agents need different numbers and types of memories
AGENT_MEMORY_CONFIG = {
    "bull_researcher": {
        "n_matches": 3,
        "min_confidence": 0.4,
        "tier_weights": {TIER_SHORT: 0.4, TIER_MID: 0.35, TIER_LONG: 0.25},
        "recency_half_life_days": 45.0,  # Bulls benefit from longer-term patterns
    },
    "bear_researcher": {
        "n_matches": 3,
        "min_confidence": 0.4,
        "tier_weights": {TIER_SHORT: 0.4, TIER_MID: 0.35, TIER_LONG: 0.25},
        "recency_half_life_days": 45.0,
    },
    "trader": {
        "n_matches": 4,  # Trader needs more context for decision-making
        "min_confidence": 0.5,  # Higher bar for execution decisions
        "tier_weights": {TIER_SHORT: 0.5, TIER_MID: 0.3, TIER_LONG: 0.2},
        "recency_half_life_days": 30.0,
        "include_smc": True,  # Trader should see SMC pattern insights
    },
    "invest_judge": {
        "n_matches": 3,
        "min_confidence": 0.5,
        "tier_weights": {TIER_SHORT: 0.35, TIER_MID: 0.35, TIER_LONG: 0.30},
        "recency_half_life_days": 60.0,  # Judges benefit from longer-term wisdom
    },
    "risk_manager": {
        "n_matches": 2,  # Fewer but high-quality memories
        "min_confidence": 0.6,  # High bar - only validated lessons
        "tier_weights": {TIER_SHORT: 0.3, TIER_MID: 0.35, TIER_LONG: 0.35},  # Prefer validated wisdom
        "recency_half_life_days": 90.0,  # Risk lessons are often timeless
        "focus": "losses",  # Learn from mistakes
    },
}


def get_agent_memory_config(agent_name: str) -> Dict[str, Any]:
    """Get memory retrieval configuration for a specific agent."""
    return AGENT_MEMORY_CONFIG.get(agent_name, {
        "n_matches": 2,
        "min_confidence": 0.0,
        "tier_weights": DEFAULT_TIER_WEIGHTS,
        "recency_half_life_days": 30.0,
    })


class FinancialSituationMemory:
    def __init__(self, name, config, persistent=True):
        self.llm_provider = config.get("llm_provider", "openai").lower()
        backend_url = config.get("backend_url", "https://api.openai.com/v1")
        embedding_provider = config.get("embedding_provider", "auto").lower()
        self.name = name
        
        # Determine embedding provider
        # "auto" = use fastembed for xAI/grok (lightweight, no PyTorch), OpenAI for others
        # "local" = sentence-transformers (requires PyTorch)
        # "fastembed" = fastembed library (lightweight, no PyTorch needed)
        # "openai" = always use OpenAI API
        # "ollama" = use Ollama local embeddings

        self._embedding_provider = embedding_provider
        self._use_local = False
        self._use_fastembed = False
        self._local_model = None
        self._fastembed_model = None
        self.client = None

        if embedding_provider == "fastembed" or (embedding_provider == "auto" and self.llm_provider in ["xai", "grok"]):
            # Use fastembed (lightweight, no PyTorch needed)
            self._use_fastembed = True
            self._fastembed_model_name = config.get("fastembed_model", "BAAI/bge-small-en-v1.5")
        elif embedding_provider == "local":
            # Use local sentence-transformers (requires PyTorch)
            self._use_local = True
            self._local_model_name = config.get("local_embedding_model", "all-MiniLM-L6-v2")
        elif embedding_provider == "ollama" or backend_url == "http://localhost:11434/v1":
            # Ollama local
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url="http://localhost:11434/v1")
        else:
            # Default to OpenAI for embeddings
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

    def _get_fastembed_model(self):
        """Lazy load the fastembed model (lightweight, no PyTorch needed)."""
        if self._fastembed_model is None:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name=self._fastembed_model_name)
        return self._fastembed_model

    def get_embedding(self, text):
        """Get embedding for a text"""
        if self._use_fastembed:
            model = self._get_fastembed_model()
            # fastembed returns a generator, get first result
            embeddings = list(model.embed([text]))
            return embeddings[0].tolist()
        elif self._use_local:
            model = self._get_local_model()
            # Disable progress bar to avoid tqdm errors when running as subprocess
            embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
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

        # Ensure defaults for ChromaDB (it doesn't accept None values)
        if confidence is None:
            confidence = 0.5
        if outcome_quality is None:
            outcome_quality = 0.5

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
            
            # Add regime metadata if provided (filter out None values - ChromaDB doesn't accept them)
            if regime:
                for key in ["market_regime", "volatility_regime", "expansion_regime"]:
                    if regime.get(key) is not None:
                        metadata_dict[key] = regime[key]
            
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

    def get_memories_for_agent(
        self,
        agent_name: str,
        current_situation: str,
        regime_filter: Optional[Dict[str, str]] = None,
        setup_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        PHASE 3: Enhanced memory retrieval tailored for specific agents.

        Uses agent-specific configuration for optimal retrieval:
        - Different n_matches per agent role
        - Different confidence thresholds
        - Different tier weights
        - Different recency decay

        Args:
            agent_name: Name of the agent (bull_researcher, bear_researcher, trader, etc.)
            current_situation: Current market situation for similarity matching
            regime_filter: Optional regime filter for regime-aware retrieval
            setup_type: Optional SMC setup type for setup-specific matching

        Returns:
            List of matched memories with enhanced scoring
        """
        config = get_agent_memory_config(agent_name)

        query_embedding = self.get_embedding(current_situation)

        # Get more candidates for re-ranking
        n_candidates = min(max(config["n_matches"] * 5, 25), self.situation_collection.count())
        if n_candidates == 0:
            return []

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["metadatas", "documents", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # Enhanced scoring with Phase 3 improvements
        scored_results = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            similarity = 1 - results["distances"][0][i]

            # Apply regime filter if specified
            if regime_filter:
                regime_match = self._check_regime_match(metadata, regime_filter)
                if not regime_match:
                    continue

            # Confidence check
            confidence = metadata.get("confidence", 0.5)
            if confidence < config.get("min_confidence", 0.0):
                continue

            # Risk manager focuses on losses
            if config.get("focus") == "losses":
                prediction_correct = metadata.get("prediction_correct", "unknown")
                if prediction_correct == "True":
                    continue  # Skip wins for risk manager

            # Tier weight
            tier = metadata.get("tier", TIER_SHORT)
            tier_weight = config.get("tier_weights", DEFAULT_TIER_WEIGHTS).get(tier, 0.3)

            # Recency decay
            timestamp_str = metadata.get("timestamp")
            recency_half_life = config.get("recency_half_life_days", 30.0)
            if timestamp_str:
                try:
                    memory_time = datetime.fromisoformat(timestamp_str)
                    age_days = (datetime.now() - memory_time).days
                    recency = np.exp(-age_days / recency_half_life)
                except (ValueError, TypeError):
                    recency = 0.5
            else:
                recency = 0.5

            # PHASE 3: Enhanced scoring formula
            # New formula weights validated memories higher
            validated_count = metadata.get("validated_count", 0)
            invalidated_count = metadata.get("invalidated_count", 0)

            # Validation bonus: memories that have been validated get a boost
            if validated_count > 0:
                validation_ratio = validated_count / (validated_count + invalidated_count + 1)
                validation_bonus = 1.0 + (validation_ratio * 0.5)  # Up to 50% boost
            else:
                validation_bonus = 1.0

            # Regime match bonus
            regime_bonus = 1.0
            if regime_filter:
                # Already filtered, so all passing memories get a small boost
                regime_bonus = 1.15

            # Setup type match bonus (if specified)
            setup_bonus = 1.0
            if setup_type and metadata.get("setup_type") == setup_type:
                setup_bonus = 1.25

            # PHASE 3 SCORING:
            # score = similarity(0.3) + regime_match(0.2) + validation(0.2) + confidence(0.15) + tier(0.1) + recency(0.05)
            # Then multiply by bonuses
            base_score = (
                similarity * 0.30 +
                (0.2 if regime_filter else 0.1) +  # Regime contributes more if filtered
                (validation_bonus - 1.0) * 0.4 +  # Validation contribution (0-0.2)
                confidence * 0.15 +
                tier_weight * 0.10 +
                recency * 0.05
            )

            final_score = base_score * validation_bonus * regime_bonus * setup_bonus

            scored_results.append({
                "matched_situation": results["documents"][0][i],
                "recommendation": metadata.get("recommendation", ""),
                "similarity_score": similarity,
                "composite_score": final_score,
                "tier": tier,
                "confidence": confidence,
                "recency": recency,
                "timestamp": timestamp_str,
                "outcome_quality": metadata.get("outcome_quality", 0.5),
                "reference_count": metadata.get("reference_count", 0),
                "validated_count": validated_count,
                "invalidated_count": invalidated_count,
                "id": results["ids"][0][i] if results.get("ids") else str(i),
                "market_regime": metadata.get("market_regime"),
                "volatility_regime": metadata.get("volatility_regime"),
            })

        # Sort by composite score
        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Increment reference counts
        n_matches = config["n_matches"]
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


class MemoryUsageTracker:
    """
    Tracks which memories were used for which trades.

    This enables the feedback loop:
    1. Trade starts → memories retrieved → usage recorded
    2. Trade closes → update memory confidence based on outcome
    3. Memories that help win trades get higher confidence
    4. Memories that correlate with losses get lower confidence
    """

    def __init__(self, config: Dict[str, Any], persistent: bool = True):
        # Use persistent storage
        if persistent:
            db_path = config.get("memory_db_path", MEMORY_DB_PATH)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.chroma_client = chromadb.Client(Settings(allow_reset=True))

        # Collection for tracking memory usage
        self.usage_collection = self.chroma_client.get_or_create_collection(name="memory_usage")

    def track_usage(
        self,
        trade_id: str,
        memory_ids: List[str],
        memory_collection: str,
        agent_name: str,
    ):
        """
        Record that specific memories were used for a trade.

        Args:
            trade_id: The trade/decision ID
            memory_ids: List of memory IDs that were retrieved
            memory_collection: Which collection the memories came from
            agent_name: Which agent used the memories (bull, bear, trader, etc.)
        """
        timestamp = datetime.now().isoformat()

        for memory_id in memory_ids:
            usage_id = f"{trade_id}_{memory_collection}_{memory_id}"

            self.usage_collection.add(
                documents=[f"Trade {trade_id} used memory {memory_id}"],
                metadatas=[{
                    "trade_id": trade_id,
                    "memory_id": memory_id,
                    "memory_collection": memory_collection,
                    "agent_name": agent_name,
                    "timestamp": timestamp,
                    "outcome_processed": "False",
                }],
                ids=[usage_id],
            )

    def get_memories_used_for_trade(self, trade_id: str) -> List[Dict[str, Any]]:
        """Get all memories that were used for a specific trade."""
        results = self.usage_collection.get(
            where={"trade_id": {"$eq": trade_id}},
            include=["metadatas"],
        )

        usages = []
        for i, metadata in enumerate(results.get("metadatas", [])):
            usages.append({
                "usage_id": results["ids"][i],
                "memory_id": metadata.get("memory_id"),
                "memory_collection": metadata.get("memory_collection"),
                "agent_name": metadata.get("agent_name"),
                "timestamp": metadata.get("timestamp"),
                "outcome_processed": metadata.get("outcome_processed") == "True",
            })

        return usages

    def update_on_outcome(
        self,
        trade_id: str,
        was_successful: bool,
        returns_pct: float,
        agent_memories: Dict[str, "FinancialSituationMemory"],
    ) -> Dict[str, int]:
        """
        Update all memories used for a trade based on its outcome.

        Args:
            trade_id: The trade/decision ID
            was_successful: Whether the trade was profitable
            returns_pct: The percentage returns
            agent_memories: Dict mapping collection names to memory instances

        Returns:
            Dict with counts of memories updated
        """
        usages = self.get_memories_used_for_trade(trade_id)
        updated = 0
        failed = 0

        for usage in usages:
            if usage.get("outcome_processed"):
                continue  # Already processed

            memory_id = usage.get("memory_id")
            collection_name = usage.get("memory_collection")

            if collection_name not in agent_memories:
                continue

            memory_instance = agent_memories[collection_name]

            try:
                # Get current metadata
                result = memory_instance.situation_collection.get(
                    ids=[memory_id],
                    include=["metadatas"]
                )

                if not result["metadatas"]:
                    continue

                metadata = result["metadatas"][0]

                # Update confidence based on outcome
                current_conf = metadata.get("confidence", 0.5)
                validated = metadata.get("validated_count", 0)
                invalidated = metadata.get("invalidated_count", 0)

                if was_successful:
                    # Memory was helpful → increase confidence
                    validated += 1
                    new_conf = min(1.0, current_conf + 0.05)
                else:
                    # Memory was not helpful → decrease confidence
                    invalidated += 1
                    # Only decrease if invalidated > validated
                    if invalidated > validated:
                        new_conf = max(0.1, current_conf - 0.1)
                    else:
                        new_conf = current_conf

                metadata["confidence"] = new_conf
                metadata["validated_count"] = validated
                metadata["invalidated_count"] = invalidated
                metadata["last_validation"] = datetime.now().isoformat()

                # Check for tier promotion/demotion
                memory_instance._check_tier_promotion(memory_id, metadata)

                # Update memory
                memory_instance.situation_collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )

                # Mark usage as processed
                self.usage_collection.update(
                    ids=[usage["usage_id"]],
                    metadatas=[{"outcome_processed": "True"}]
                )

                updated += 1

            except Exception as e:
                print(f"Warning: Failed to update memory {memory_id}: {e}")
                failed += 1

        return {"updated": updated, "failed": failed}

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics on memory validation."""
        results = self.usage_collection.get(include=["metadatas"])

        total = len(results.get("metadatas", []))
        processed = sum(1 for m in results.get("metadatas", []) if m.get("outcome_processed") == "True")

        by_collection: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}

        for metadata in results.get("metadatas", []):
            coll = metadata.get("memory_collection", "unknown")
            agent = metadata.get("agent_name", "unknown")
            by_collection[coll] = by_collection.get(coll, 0) + 1
            by_agent[agent] = by_agent.get(agent, 0) + 1

        return {
            "total_usages": total,
            "processed": processed,
            "pending": total - processed,
            "by_collection": by_collection,
            "by_agent": by_agent,
        }


class SMCPatternMemory:
    """
    Specialized memory for SMC (Smart Money Concepts) patterns.

    Tracks outcomes by setup type to learn which SMC setups work best.
    Separate from general situation memory for more targeted retrieval.

    Collections:
    - smc_patterns: Individual pattern instances with outcomes
    - smc_stats: Aggregated statistics by setup type + symbol
    """

    SETUP_TYPES = ["fvg_bounce", "ob_bounce", "liquidity_sweep", "choch", "bos", "trend_continuation"]

    def __init__(self, config: Dict[str, Any], persistent: bool = True):
        self.llm_provider = config.get("llm_provider", "openai").lower()
        backend_url = config.get("backend_url", "https://api.openai.com/v1")
        embedding_provider = config.get("embedding_provider", "auto").lower()

        # Embedding setup (same logic as FinancialSituationMemory)
        self._use_fastembed = False
        self._use_local = False
        self._fastembed_model = None
        self._local_model = None
        self.client = None

        if embedding_provider == "fastembed" or (embedding_provider == "auto" and self.llm_provider in ["xai", "grok"]):
            self._use_fastembed = True
            self._fastembed_model_name = config.get("fastembed_model", "BAAI/bge-small-en-v1.5")
        elif embedding_provider == "local":
            self._use_local = True
            self._local_model_name = config.get("local_embedding_model", "all-MiniLM-L6-v2")
        elif embedding_provider == "ollama" or backend_url == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
            self.client = OpenAI(base_url="http://localhost:11434/v1")
        else:
            self.embedding = "text-embedding-3-small"
            self.client = OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # Use persistent storage
        if persistent:
            db_path = config.get("memory_db_path", MEMORY_DB_PATH)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.chroma_client = chromadb.Client(Settings(allow_reset=True))

        # SMC pattern collection
        self.pattern_collection = self.chroma_client.get_or_create_collection(name="smc_patterns")

    def _get_fastembed_model(self):
        """Lazy load fastembed model."""
        if self._fastembed_model is None:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(model_name=self._fastembed_model_name)
        return self._fastembed_model

    def _get_local_model(self):
        """Lazy load sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._local_model_name)
        return self._local_model

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self._use_fastembed:
            model = self._get_fastembed_model()
            embeddings = list(model.embed([text]))
            return embeddings[0].tolist()
        elif self._use_local:
            model = self._get_local_model()
            embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        else:
            response = self.client.embeddings.create(model=self.embedding, input=text)
            return response.data[0].embedding

    def store_pattern(
        self,
        decision_id: str,
        symbol: str,
        setup_type: str,
        direction: str,  # "BUY" or "SELL"
        smc_context: Dict[str, Any],
        situation_text: str,
        outcome: Dict[str, Any],
        lesson: str = "",
    ):
        """
        Store an SMC pattern with its outcome for learning.

        Args:
            decision_id: ID of the trade decision
            symbol: Trading symbol
            setup_type: Type of SMC setup (fvg_bounce, ob_bounce, etc.)
            direction: Trade direction (BUY/SELL)
            smc_context: Full SMC context dict from decision
            situation_text: Description of the market situation
            outcome: Structured outcome dict from trade close
            lesson: Generated lesson from this trade
        """
        timestamp = datetime.now().isoformat()
        pattern_id = f"{decision_id}_{setup_type}"

        # Build searchable text for embedding
        search_text = f"{symbol} {setup_type} {direction} {situation_text}"

        # Metadata for filtering and stats
        metadata = {
            "decision_id": decision_id,
            "symbol": symbol,
            "setup_type": setup_type,
            "direction": direction,
            "timestamp": timestamp,
            # SMC context (flattened for ChromaDB)
            "entry_zone": smc_context.get("entry_zone", "unknown"),
            "entry_zone_strength": float(smc_context.get("entry_zone_strength", 0.5)),
            "with_trend": str(smc_context.get("with_trend", "unknown")),
            "higher_tf_aligned": str(smc_context.get("higher_tf_aligned", "unknown")),
            "confluences": json.dumps(smc_context.get("confluences", [])),
            "zone_tested_before": str(smc_context.get("zone_tested_before", "unknown")),
            # Outcome
            "was_win": str(outcome.get("result") == "win"),
            "returns_pct": float(outcome.get("returns_pct", 0)),
            "direction_correct": str(outcome.get("direction_correct", "unknown")),
            "sl_placement": outcome.get("sl_placement", "unknown"),
            "tp_placement": outcome.get("tp_placement", "unknown"),
            "exit_type": outcome.get("exit_type", "unknown"),
            # Lesson
            "lesson": lesson[:500] if lesson else "",  # Truncate for metadata
        }

        embedding = self.get_embedding(search_text)

        self.pattern_collection.add(
            documents=[situation_text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[pattern_id],
        )

        print(f"📝 SMC pattern stored: {setup_type} for {symbol} ({'win' if outcome.get('result') == 'win' else 'loss'})")

    def get_patterns_by_setup(
        self,
        setup_type: str,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get all patterns for a specific setup type.

        Args:
            setup_type: Type of SMC setup to retrieve
            symbol: Optional symbol filter
            direction: Optional direction filter (BUY/SELL)
            limit: Maximum patterns to return

        Returns:
            List of pattern dicts with metadata
        """
        # Build where clause for filtering
        where_clauses = [{"setup_type": {"$eq": setup_type}}]
        if symbol:
            where_clauses.append({"symbol": {"$eq": symbol}})
        if direction:
            where_clauses.append({"direction": {"$eq": direction}})

        where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]

        results = self.pattern_collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

        patterns = []
        for i, doc in enumerate(results.get("documents", [])):
            metadata = results["metadatas"][i]
            patterns.append({
                "situation": doc,
                "setup_type": metadata.get("setup_type"),
                "symbol": metadata.get("symbol"),
                "direction": metadata.get("direction"),
                "was_win": metadata.get("was_win") == "True",
                "returns_pct": metadata.get("returns_pct", 0),
                "direction_correct": metadata.get("direction_correct") == "True",
                "sl_placement": metadata.get("sl_placement"),
                "tp_placement": metadata.get("tp_placement"),
                "entry_zone": metadata.get("entry_zone"),
                "entry_zone_strength": metadata.get("entry_zone_strength"),
                "with_trend": metadata.get("with_trend") == "True",
                "confluences": json.loads(metadata.get("confluences", "[]")),
                "lesson": metadata.get("lesson", ""),
                "timestamp": metadata.get("timestamp"),
            })

        return patterns

    def get_similar_patterns(
        self,
        situation: str,
        setup_type: Optional[str] = None,
        symbol: Optional[str] = None,
        n_matches: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar SMC patterns by situation similarity.

        Args:
            situation: Current market situation text
            setup_type: Optional setup type filter
            symbol: Optional symbol filter
            n_matches: Number of matches to return

        Returns:
            List of similar patterns with similarity scores
        """
        embedding = self.get_embedding(situation)

        # Build where clause
        where = None
        where_clauses = []
        if setup_type:
            where_clauses.append({"setup_type": {"$eq": setup_type}})
        if symbol:
            where_clauses.append({"symbol": {"$eq": symbol}})
        if where_clauses:
            where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]

        results = self.pattern_collection.query(
            query_embeddings=[embedding],
            where=where,
            n_results=n_matches,
            include=["documents", "metadatas", "distances"],
        )

        patterns = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance

            patterns.append({
                "situation": doc,
                "similarity": similarity,
                "setup_type": metadata.get("setup_type"),
                "symbol": metadata.get("symbol"),
                "direction": metadata.get("direction"),
                "was_win": metadata.get("was_win") == "True",
                "returns_pct": metadata.get("returns_pct", 0),
                "direction_correct": metadata.get("direction_correct") == "True",
                "sl_placement": metadata.get("sl_placement"),
                "tp_placement": metadata.get("tp_placement"),
                "with_trend": metadata.get("with_trend") == "True",
                "confluences": json.loads(metadata.get("confluences", "[]")),
                "lesson": metadata.get("lesson", ""),
            })

        return patterns

    def get_setup_stats(
        self,
        symbol: Optional[str] = None,
        min_samples: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated statistics for each setup type.

        Args:
            symbol: Optional symbol filter
            min_samples: Minimum samples required to include in stats

        Returns:
            Dict of setup_type -> stats dict
        """
        where = {"symbol": {"$eq": symbol}} if symbol else None

        results = self.pattern_collection.get(
            where=where,
            include=["metadatas"],
        )

        # Group by setup type
        by_setup: Dict[str, List[Dict]] = {}
        for metadata in results.get("metadatas", []):
            st = metadata.get("setup_type")
            if st:
                if st not in by_setup:
                    by_setup[st] = []
                by_setup[st].append(metadata)

        stats = {}
        for setup_type, patterns in by_setup.items():
            if len(patterns) < min_samples:
                continue

            wins = sum(1 for p in patterns if p.get("was_win") == "True")
            returns = [p.get("returns_pct", 0) for p in patterns]
            direction_correct = sum(1 for p in patterns if p.get("direction_correct") == "True")

            # Analyze by confluence
            confluence_wins: Dict[str, List[bool]] = {}
            for p in patterns:
                confs = json.loads(p.get("confluences", "[]"))
                was_win = p.get("was_win") == "True"
                for c in confs:
                    if c not in confluence_wins:
                        confluence_wins[c] = []
                    confluence_wins[c].append(was_win)

            best_confluences = []
            for conf, outcomes in confluence_wins.items():
                if len(outcomes) >= 2:
                    win_rate = sum(outcomes) / len(outcomes)
                    best_confluences.append({
                        "confluence": conf,
                        "win_rate": win_rate,
                        "samples": len(outcomes),
                    })
            best_confluences.sort(key=lambda x: x["win_rate"], reverse=True)

            # With-trend analysis
            with_trend = [p for p in patterns if p.get("with_trend") == "True"]
            counter_trend = [p for p in patterns if p.get("with_trend") == "False"]

            stats[setup_type] = {
                "win_rate": wins / len(patterns),
                "sample_size": len(patterns),
                "avg_returns_pct": sum(returns) / len(returns) if returns else 0,
                "direction_accuracy": direction_correct / len(patterns),
                "best_confluences": best_confluences[:5],
                "with_trend_win_rate": sum(1 for p in with_trend if p.get("was_win") == "True") / len(with_trend) if with_trend else None,
                "counter_trend_win_rate": sum(1 for p in counter_trend if p.get("was_win") == "True") / len(counter_trend) if counter_trend else None,
                "sl_analysis": {
                    "too_tight": sum(1 for p in patterns if p.get("sl_placement") == "too_tight"),
                    "appropriate": sum(1 for p in patterns if p.get("sl_placement") == "appropriate"),
                    "too_wide": sum(1 for p in patterns if p.get("sl_placement") == "too_wide"),
                },
            }

        return stats

    def get_best_setups(
        self,
        symbol: Optional[str] = None,
        min_samples: int = 3,
        min_win_rate: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Get the best performing setups for trading recommendations.

        Returns setups sorted by win rate, filtered by minimum criteria.
        """
        stats = self.get_setup_stats(symbol, min_samples)

        best = []
        for setup_type, s in stats.items():
            if s["win_rate"] >= min_win_rate:
                best.append({
                    "setup_type": setup_type,
                    "win_rate": s["win_rate"],
                    "sample_size": s["sample_size"],
                    "avg_returns_pct": s["avg_returns_pct"],
                    "direction_accuracy": s["direction_accuracy"],
                    "best_confluences": s["best_confluences"][:3],
                    "recommendation": self._generate_setup_recommendation(setup_type, s),
                })

        best.sort(key=lambda x: x["win_rate"], reverse=True)
        return best

    def _generate_setup_recommendation(self, setup_type: str, stats: Dict) -> str:
        """Generate a recommendation based on setup stats."""
        parts = [f"{setup_type}: {stats['win_rate']*100:.0f}% win rate ({stats['sample_size']} trades)."]

        if stats.get("with_trend_win_rate") and stats.get("counter_trend_win_rate"):
            if stats["with_trend_win_rate"] > stats["counter_trend_win_rate"] + 0.1:
                parts.append("Works better WITH trend.")
            elif stats["counter_trend_win_rate"] > stats["with_trend_win_rate"] + 0.1:
                parts.append("Works better COUNTER trend.")

        if stats.get("best_confluences"):
            top_conf = stats["best_confluences"][0]
            if top_conf["win_rate"] >= 0.6:
                parts.append(f"Best with {top_conf['confluence']} ({top_conf['win_rate']*100:.0f}% win).")

        sl_analysis = stats.get("sl_analysis", {})
        if sl_analysis.get("too_tight", 0) > stats["sample_size"] * 0.3:
            parts.append("SL often too tight - consider wider stops.")

        return " ".join(parts)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall SMC pattern memory statistics."""
        total = self.pattern_collection.count()
        if total == 0:
            return {"total": 0, "by_setup": {}, "by_symbol": {}}

        results = self.pattern_collection.get(include=["metadatas"])

        by_setup: Dict[str, int] = {}
        by_symbol: Dict[str, int] = {}
        wins = 0

        for metadata in results.get("metadatas", []):
            st = metadata.get("setup_type", "unknown")
            sym = metadata.get("symbol", "unknown")
            by_setup[st] = by_setup.get(st, 0) + 1
            by_symbol[sym] = by_symbol.get(sym, 0) + 1
            if metadata.get("was_win") == "True":
                wins += 1

        return {
            "total": total,
            "overall_win_rate": wins / total if total > 0 else 0,
            "by_setup": by_setup,
            "by_symbol": by_symbol,
        }


class MetaPatternLearning:
    """
    PHASE 5: Meta-Learning Across Agents

    Learns higher-level patterns that emerge from cross-agent analysis:
    - Bull/bear agreement patterns (both agree vs disagree)
    - Which analysts are most accurate
    - Performance by market regime
    - Disagreement patterns (when bull/bear disagree, who is usually right?)

    This gives the Risk Manager additional context for decision-making.
    """

    def __init__(self, config: Dict[str, Any], persistent: bool = True):
        # Use persistent storage
        if persistent:
            db_path = config.get("memory_db_path", MEMORY_DB_PATH)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.chroma_client = chromadb.Client(Settings(allow_reset=True))

        # Collection for meta-pattern tracking
        self.meta_collection = self.chroma_client.get_or_create_collection(name="meta_patterns")

    def record_trade_outcome(
        self,
        decision_id: str,
        symbol: str,
        bull_signal: str,  # "bullish", "bearish", "neutral"
        bear_signal: str,  # "bullish", "bearish", "neutral"
        final_action: str,  # "BUY", "SELL", "HOLD"
        was_successful: bool,
        returns_pct: float,
        market_regime: Optional[str] = None,
        volatility_regime: Optional[str] = None,
    ):
        """
        Record a trade outcome for meta-pattern learning.

        Args:
            decision_id: Unique trade identifier
            symbol: Trading symbol
            bull_signal: What the bull researcher recommended
            bear_signal: What the bear researcher recommended
            final_action: Final action taken
            was_successful: Whether the trade was profitable
            returns_pct: Percentage returns
            market_regime: Current market regime
            volatility_regime: Current volatility regime
        """
        timestamp = datetime.now().isoformat()

        # Determine agreement pattern
        if bull_signal == bear_signal:
            if bull_signal == "bullish":
                agreement = "both_bullish"
            elif bull_signal == "bearish":
                agreement = "both_bearish"
            else:
                agreement = "both_neutral"
        else:
            agreement = "disagree"

        # Document for semantic search
        doc = f"{symbol} {agreement} {final_action} {market_regime or 'unknown'} {volatility_regime or 'unknown'}"

        metadata = {
            "decision_id": decision_id,
            "symbol": symbol,
            "bull_signal": bull_signal,
            "bear_signal": bear_signal,
            "agreement": agreement,
            "final_action": final_action,
            "was_successful": str(was_successful),
            "returns_pct": float(returns_pct),
            "market_regime": market_regime or "unknown",
            "volatility_regime": volatility_regime or "unknown",
            "timestamp": timestamp,
        }

        self.meta_collection.add(
            documents=[doc],
            metadatas=[metadata],
            ids=[decision_id],
        )

    def get_agreement_stats(self, min_samples: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Get win rates based on bull/bear agreement patterns.

        Returns stats like:
        - "both_bullish": {"win_rate": 0.72, "sample_size": 50, "avg_returns": 2.3}
        - "disagree": {"win_rate": 0.45, "sample_size": 80, "avg_returns": -0.5}
        """
        results = self.meta_collection.get(include=["metadatas"])

        # Group by agreement pattern
        by_agreement: Dict[str, List[Dict]] = {}
        for metadata in results.get("metadatas", []):
            agreement = metadata.get("agreement", "unknown")
            if agreement not in by_agreement:
                by_agreement[agreement] = []
            by_agreement[agreement].append(metadata)

        stats = {}
        for agreement, trades in by_agreement.items():
            if len(trades) < min_samples:
                continue

            wins = sum(1 for t in trades if t.get("was_successful") == "True")
            returns = [t.get("returns_pct", 0) for t in trades]

            stats[agreement] = {
                "win_rate": wins / len(trades),
                "sample_size": len(trades),
                "avg_returns_pct": sum(returns) / len(returns) if returns else 0,
            }

        return stats

    def get_disagreement_stats(self, min_samples: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        When bull and bear disagree, which one is usually right?

        Returns stats like:
        - "bull_correct_when_disagree": 0.55
        - "bear_correct_when_disagree": 0.45
        """
        results = self.meta_collection.get(
            where={"agreement": {"$eq": "disagree"}},
            include=["metadatas"],
        )

        trades = results.get("metadatas", [])
        if len(trades) < min_samples:
            return {"message": f"Not enough disagreement samples (have {len(trades)}, need {min_samples})"}

        bull_correct = 0
        bear_correct = 0

        for t in trades:
            was_win = t.get("was_successful") == "True"
            action = t.get("final_action", "").upper()
            bull = t.get("bull_signal", "")
            bear = t.get("bear_signal", "")

            # Bull is correct if bullish signal and trade won (for BUY)
            # or bearish signal and trade won (for SELL)
            if action == "BUY" and was_win:
                if bull == "bullish":
                    bull_correct += 1
                if bear == "bullish":
                    bear_correct += 1
            elif action == "SELL" and was_win:
                if bull == "bearish":
                    bull_correct += 1
                if bear == "bearish":
                    bear_correct += 1

        total = len(trades)
        return {
            "total_disagreements": total,
            "bull_correct_pct": bull_correct / total if total > 0 else 0,
            "bear_correct_pct": bear_correct / total if total > 0 else 0,
            "recommendation": self._generate_disagreement_recommendation(bull_correct, bear_correct, total),
        }

    def _generate_disagreement_recommendation(self, bull_correct: int, bear_correct: int, total: int) -> str:
        """Generate a recommendation based on disagreement patterns."""
        if total < 5:
            return "Not enough data. Default to HOLD when analysts disagree."

        bull_rate = bull_correct / total
        bear_rate = bear_correct / total

        if bull_rate > bear_rate + 0.1:
            return f"Bull is usually right when they disagree ({bull_rate:.0%}). Lean toward bull's view."
        elif bear_rate > bull_rate + 0.1:
            return f"Bear is usually right when they disagree ({bear_rate:.0%}). Lean toward bear's view."
        else:
            return f"Neither consistently right ({bull_rate:.0%} vs {bear_rate:.0%}). Use extra caution."

    def get_regime_performance(self, min_samples: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics by market regime.

        Returns stats like:
        - "trending_bullish": {"long_win_rate": 0.70, "short_win_rate": 0.35}
        - "ranging": {"long_win_rate": 0.48, "short_win_rate": 0.50}
        """
        results = self.meta_collection.get(include=["metadatas"])

        # Group by regime
        by_regime: Dict[str, Dict[str, List]] = {}
        for metadata in results.get("metadatas", []):
            regime = metadata.get("market_regime", "unknown")
            action = metadata.get("final_action", "").upper()
            was_win = metadata.get("was_successful") == "True"

            if regime not in by_regime:
                by_regime[regime] = {"long": [], "short": []}

            if action == "BUY":
                by_regime[regime]["long"].append(was_win)
            elif action == "SELL":
                by_regime[regime]["short"].append(was_win)

        stats = {}
        for regime, trades in by_regime.items():
            long_trades = trades["long"]
            short_trades = trades["short"]

            stats[regime] = {
                "long_win_rate": sum(long_trades) / len(long_trades) if len(long_trades) >= min_samples else None,
                "long_samples": len(long_trades),
                "short_win_rate": sum(short_trades) / len(short_trades) if len(short_trades) >= min_samples else None,
                "short_samples": len(short_trades),
            }

        return stats

    def get_meta_insights_for_decision(
        self,
        bull_signal: str,
        bear_signal: str,
        market_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get meta-pattern insights to help with a trading decision.

        This is called by the Risk Manager to get additional context.

        Returns:
            Dict with recommendations and statistics
        """
        insights = {
            "agreement_insight": None,
            "regime_insight": None,
            "overall_recommendation": None,
        }

        # Check agreement pattern
        agreement_stats = self.get_agreement_stats()

        if bull_signal == bear_signal:
            pattern = "both_bullish" if bull_signal == "bullish" else "both_bearish" if bull_signal == "bearish" else "both_neutral"
        else:
            pattern = "disagree"

        if pattern in agreement_stats:
            stats = agreement_stats[pattern]
            win_rate = stats["win_rate"]
            sample = stats["sample_size"]

            if pattern == "disagree":
                disagreement_stats = self.get_disagreement_stats()
                insights["agreement_insight"] = (
                    f"Bull/Bear disagree. Historical win rate: {win_rate:.0%} ({sample} trades). "
                    f"{disagreement_stats.get('recommendation', '')}"
                )
            else:
                insights["agreement_insight"] = (
                    f"Bull/Bear both {bull_signal}. Historical win rate: {win_rate:.0%} ({sample} trades)."
                )

            # Generate recommendation
            if win_rate < 0.45:
                insights["overall_recommendation"] = "CAUTION: Pattern has poor historical performance."
            elif win_rate >= 0.6:
                insights["overall_recommendation"] = "Pattern has good historical performance. Consider proceeding."
            else:
                insights["overall_recommendation"] = "Pattern has mixed results. Proceed with normal risk management."

        # Check regime performance
        if market_regime:
            regime_stats = self.get_regime_performance()
            if market_regime in regime_stats:
                r_stats = regime_stats[market_regime]
                long_wr = r_stats.get("long_win_rate")
                short_wr = r_stats.get("short_win_rate")

                insights["regime_insight"] = f"In {market_regime} regime: "
                if long_wr is not None:
                    insights["regime_insight"] += f"Long trades win {long_wr:.0%} ({r_stats['long_samples']} samples). "
                if short_wr is not None:
                    insights["regime_insight"] += f"Short trades win {short_wr:.0%} ({r_stats['short_samples']} samples)."

        return insights

    def get_stats(self) -> Dict[str, Any]:
        """Get overall meta-pattern statistics."""
        total = self.meta_collection.count()
        if total == 0:
            return {"total": 0, "message": "No meta-patterns recorded yet"}

        results = self.meta_collection.get(include=["metadatas"])
        wins = sum(1 for m in results.get("metadatas", []) if m.get("was_successful") == "True")

        return {
            "total_trades": total,
            "overall_win_rate": wins / total if total > 0 else 0,
            "agreement_stats": self.get_agreement_stats(),
            "regime_stats": self.get_regime_performance(),
        }


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test", {})

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
