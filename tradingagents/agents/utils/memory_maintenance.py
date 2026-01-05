"""
Memory Maintenance Utilities

This module provides utilities for maintaining the memory system:
- Pruning low-quality memories
- Deduplicating similar memories
- Archiving old memories
- Memory statistics and reporting
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .memory import (
    FinancialSituationMemory, 
    MEMORY_DB_PATH,
    TIER_SHORT, 
    TIER_MID, 
    TIER_LONG,
    DEFAULT_TIER_WEIGHTS
)


# Archive directory for pruned memories
ARCHIVE_DIR = MEMORY_DB_PATH / "archive"


class MemoryMaintenance:
    """Handles memory cleanup, deduplication, and maintenance tasks."""
    
    def __init__(self, memory: FinancialSituationMemory):
        """
        Initialize maintenance utilities for a memory collection.
        
        Args:
            memory: The FinancialSituationMemory instance to maintain
        """
        self.memory = memory
        self.collection = memory.situation_collection
    
    def prune_low_quality_memories(
        self, 
        min_confidence: float = 0.3,
        max_age_days: int = 60,
        archive: bool = True
    ) -> Dict[str, Any]:
        """
        Remove memories with low confidence that are older than max_age_days.
        
        Args:
            min_confidence: Minimum confidence threshold (default: 0.3)
            max_age_days: Only prune memories older than this (default: 60 days)
            archive: If True, archive pruned memories instead of deleting (default: True)
            
        Returns:
            Dictionary with pruning statistics
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Get all memories
        all_memories = self.collection.get(include=["metadatas", "documents"])
        
        if not all_memories["ids"]:
            return {"pruned": 0, "archived": 0, "total_before": 0}
        
        ids_to_prune = []
        memories_to_archive = []
        
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            document = all_memories["documents"][i]
            
            # Check confidence
            confidence = metadata.get("confidence", 0.5)
            if confidence >= min_confidence:
                continue
            
            # Check age
            timestamp_str = metadata.get("timestamp")
            if timestamp_str:
                try:
                    memory_time = datetime.fromisoformat(timestamp_str)
                    if memory_time > cutoff_date:
                        continue  # Too recent to prune
                except (ValueError, TypeError):
                    pass
            
            # Mark for pruning
            ids_to_prune.append(memory_id)
            if archive:
                memories_to_archive.append({
                    "id": memory_id,
                    "document": document,
                    "metadata": metadata,
                    "pruned_at": datetime.now().isoformat(),
                    "reason": f"low_confidence ({confidence:.2f} < {min_confidence})"
                })
        
        # Archive before deleting
        if archive and memories_to_archive:
            self._archive_memories(memories_to_archive)
        
        # Delete from collection
        if ids_to_prune:
            self.collection.delete(ids=ids_to_prune)
        
        return {
            "pruned": len(ids_to_prune),
            "archived": len(memories_to_archive) if archive else 0,
            "total_before": len(all_memories["ids"]),
            "total_after": len(all_memories["ids"]) - len(ids_to_prune),
        }
    
    def deduplicate_memories(
        self, 
        similarity_threshold: float = 0.95,
        keep_highest_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Find and merge near-duplicate memories.
        
        Args:
            similarity_threshold: Similarity threshold for considering duplicates (default: 0.95)
            keep_highest_confidence: Keep the version with highest confidence (default: True)
            
        Returns:
            Dictionary with deduplication statistics
        """
        # Get all memories with embeddings
        all_memories = self.collection.get(
            include=["metadatas", "documents", "embeddings"]
        )
        
        if not all_memories["ids"] or len(all_memories["ids"]) < 2:
            return {"duplicates_found": 0, "merged": 0}
        
        embeddings = np.array(all_memories["embeddings"])
        n_memories = len(embeddings)
        
        # Calculate pairwise similarities
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Find duplicates (upper triangle only to avoid double counting)
        duplicates = []
        processed = set()
        
        for i in range(n_memories):
            if i in processed:
                continue
            
            # Find all memories similar to this one
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_indices = [j for j in similar_indices if j != i and j not in processed]
            
            if similar_indices:
                # Group this memory with its duplicates
                group = [i] + list(similar_indices)
                duplicates.append(group)
                processed.update(group)
        
        # Merge duplicate groups
        merged_count = 0
        ids_to_delete = []
        
        for group in duplicates:
            if len(group) < 2:
                continue
            
            # Find the best memory to keep
            best_idx = group[0]
            best_confidence = all_memories["metadatas"][best_idx].get("confidence", 0.5)
            
            if keep_highest_confidence:
                for idx in group[1:]:
                    conf = all_memories["metadatas"][idx].get("confidence", 0.5)
                    if conf > best_confidence:
                        best_idx = idx
                        best_confidence = conf
            
            # Mark others for deletion
            for idx in group:
                if idx != best_idx:
                    ids_to_delete.append(all_memories["ids"][idx])
                    merged_count += 1
        
        # Delete duplicates
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        
        return {
            "duplicates_found": len(duplicates),
            "merged": merged_count,
            "groups": len(duplicates),
        }
    
    def promote_high_quality_memories(self) -> Dict[str, Any]:
        """
        Scan all memories and promote those that qualify for higher tiers.
        
        Returns:
            Dictionary with promotion statistics
        """
        all_memories = self.collection.get(include=["metadatas"])
        
        if not all_memories["ids"]:
            return {"promoted_to_mid": 0, "promoted_to_long": 0}
        
        promoted_to_mid = 0
        promoted_to_long = 0
        
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            current_tier = metadata.get("tier", TIER_SHORT)
            reference_count = metadata.get("reference_count", 0)
            confidence = metadata.get("confidence", 0.5)
            outcome_quality = metadata.get("outcome_quality", 0.5)
            
            new_tier = current_tier
            
            if current_tier == TIER_SHORT:
                if reference_count >= 3 or outcome_quality > 0.7:
                    new_tier = TIER_MID
                    promoted_to_mid += 1
            elif current_tier == TIER_MID:
                if confidence > 0.8 and outcome_quality > 0.8:
                    new_tier = TIER_LONG
                    promoted_to_long += 1
            
            if new_tier != current_tier:
                metadata["tier"] = new_tier
                self.collection.update(ids=[memory_id], metadatas=[metadata])
        
        return {
            "promoted_to_mid": promoted_to_mid,
            "promoted_to_long": promoted_to_long,
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the memory collection.
        
        Returns:
            Comprehensive statistics dictionary
        """
        all_memories = self.collection.get(include=["metadatas"])
        
        if not all_memories["ids"]:
            return {
                "total": 0,
                "by_tier": {TIER_SHORT: 0, TIER_MID: 0, TIER_LONG: 0},
                "confidence": {"min": 0, "max": 0, "avg": 0, "median": 0},
                "outcome_quality": {"min": 0, "max": 0, "avg": 0, "median": 0},
                "age_days": {"min": 0, "max": 0, "avg": 0},
                "reference_counts": {"min": 0, "max": 0, "avg": 0, "total": 0},
                "prediction_accuracy": {"correct": 0, "incorrect": 0, "unknown": 0},
            }
        
        tier_counts = {TIER_SHORT: 0, TIER_MID: 0, TIER_LONG: 0}
        confidences = []
        outcome_qualities = []
        ages = []
        reference_counts = []
        prediction_results = {"correct": 0, "incorrect": 0, "unknown": 0}
        
        now = datetime.now()
        
        for metadata in all_memories["metadatas"]:
            # Tier counts
            tier = metadata.get("tier", TIER_SHORT)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            # Confidence
            if "confidence" in metadata:
                confidences.append(metadata["confidence"])
            
            # Outcome quality
            if "outcome_quality" in metadata:
                outcome_qualities.append(metadata["outcome_quality"])
            
            # Age
            timestamp_str = metadata.get("timestamp")
            if timestamp_str:
                try:
                    memory_time = datetime.fromisoformat(timestamp_str)
                    age_days = (now - memory_time).days
                    ages.append(age_days)
                except (ValueError, TypeError):
                    pass
            
            # Reference counts
            if "reference_count" in metadata:
                reference_counts.append(metadata["reference_count"])
            
            # Prediction results
            pred_correct = metadata.get("prediction_correct", "unknown")
            if pred_correct == "True":
                prediction_results["correct"] += 1
            elif pred_correct == "False":
                prediction_results["incorrect"] += 1
            else:
                prediction_results["unknown"] += 1
        
        def safe_stats(arr):
            if not arr:
                return {"min": 0, "max": 0, "avg": 0, "median": 0}
            return {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "avg": float(np.mean(arr)),
                "median": float(np.median(arr)),
            }
        
        return {
            "total": len(all_memories["ids"]),
            "by_tier": tier_counts,
            "confidence": safe_stats(confidences),
            "outcome_quality": safe_stats(outcome_qualities),
            "age_days": {
                "min": min(ages) if ages else 0,
                "max": max(ages) if ages else 0,
                "avg": np.mean(ages) if ages else 0,
            },
            "reference_counts": {
                "min": min(reference_counts) if reference_counts else 0,
                "max": max(reference_counts) if reference_counts else 0,
                "avg": np.mean(reference_counts) if reference_counts else 0,
                "total": sum(reference_counts) if reference_counts else 0,
            },
            "prediction_accuracy": prediction_results,
        }
    
    def get_top_memories(
        self, 
        n: int = 10, 
        sort_by: str = "reference_count"
    ) -> List[Dict[str, Any]]:
        """
        Get the top N memories sorted by a specific metric.
        
        Args:
            n: Number of memories to return (default: 10)
            sort_by: Metric to sort by - "reference_count", "confidence", "outcome_quality"
            
        Returns:
            List of top memories with their metadata
        """
        all_memories = self.collection.get(include=["metadatas", "documents"])
        
        if not all_memories["ids"]:
            return []
        
        memories = []
        for i, memory_id in enumerate(all_memories["ids"]):
            metadata = all_memories["metadatas"][i]
            memories.append({
                "id": memory_id,
                "document": all_memories["documents"][i][:200] + "...",  # Truncate
                "tier": metadata.get("tier", TIER_SHORT),
                "confidence": metadata.get("confidence", 0.5),
                "outcome_quality": metadata.get("outcome_quality", 0.5),
                "reference_count": metadata.get("reference_count", 0),
                "timestamp": metadata.get("timestamp", ""),
                "prediction_correct": metadata.get("prediction_correct", "unknown"),
            })
        
        # Sort by specified metric
        sort_key = sort_by if sort_by in ["reference_count", "confidence", "outcome_quality"] else "reference_count"
        memories.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
        
        return memories[:n]
    
    def _archive_memories(self, memories: List[Dict[str, Any]]):
        """
        Archive memories to a JSON file for audit trail.
        
        Args:
            memories: List of memory dictionaries to archive
        """
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create archive filename with collection name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"{self.memory.name}_archive_{timestamp}.json"
        
        # Load existing archive if it exists for today
        today_pattern = datetime.now().strftime("%Y%m%d")
        existing_archives = list(ARCHIVE_DIR.glob(f"{self.memory.name}_archive_{today_pattern}*.json"))
        
        if existing_archives:
            # Append to most recent today's archive
            archive_file = existing_archives[-1]
            try:
                with open(archive_file, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        else:
            existing_data = []
        
        # Append new memories
        existing_data.extend(memories)
        
        # Save archive
        with open(archive_file, "w") as f:
            json.dump(existing_data, f, indent=2)
    
    def run_full_maintenance(
        self,
        prune_min_confidence: float = 0.3,
        prune_max_age_days: int = 60,
        dedupe_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run all maintenance tasks in sequence.
        
        Args:
            prune_min_confidence: Minimum confidence for pruning
            prune_max_age_days: Maximum age for pruning
            dedupe_threshold: Similarity threshold for deduplication
            
        Returns:
            Combined results from all maintenance tasks
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "collection": self.memory.name,
        }
        
        # 1. Promote high-quality memories first
        results["promotion"] = self.promote_high_quality_memories()
        
        # 2. Deduplicate
        results["deduplication"] = self.deduplicate_memories(
            similarity_threshold=dedupe_threshold
        )
        
        # 3. Prune low-quality old memories
        results["pruning"] = self.prune_low_quality_memories(
            min_confidence=prune_min_confidence,
            max_age_days=prune_max_age_days
        )
        
        # 4. Get final stats
        results["final_stats"] = self.get_detailed_stats()
        
        return results


def run_maintenance_on_all_collections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run maintenance on all standard memory collections.
    
    Args:
        config: Configuration dictionary for memory initialization
        
    Returns:
        Results for each collection
    """
    collection_names = [
        "bull_memory",
        "bear_memory", 
        "trader_memory",
        "invest_judge_memory",
        "risk_manager_memory",
        "prediction_accuracy",
        "technical_backtest",
    ]
    
    results = {}
    
    for name in collection_names:
        try:
            memory = FinancialSituationMemory(name, config)
            maintenance = MemoryMaintenance(memory)
            results[name] = maintenance.run_full_maintenance()
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results
