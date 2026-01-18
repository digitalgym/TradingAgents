"""
Trade Similarity Search for RAG-based Decision Support

Finds similar historical trades based on setup characteristics and market regime.
Used to provide historical context to agents before making recommendations.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np


class TradeSimilaritySearch:
    """Find similar historical trades for decision support"""
    
    def __init__(self, decisions_dir: Optional[str] = None):
        """
        Initialize trade similarity search.
        
        Args:
            decisions_dir: Optional custom path to decisions directory
        """
        if decisions_dir is None:
            from tradingagents.trade_decisions import DECISIONS_DIR
            self.decisions_dir = DECISIONS_DIR
        else:
            self.decisions_dir = decisions_dir
    
    def find_similar_trades(
        self,
        current_setup: Dict[str, Any],
        n_results: int = 5,
        min_confidence: float = 0.0,
        regime_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Find similar past trades based on setup characteristics.
        
        Args:
            current_setup: {
                "symbol": "XAUUSD",
                "direction": "BUY",
                "setup_type": "breaker-block",  # Optional
                "market_regime": "trending-up",
                "volatility_regime": "normal",
                "confluence_score": 7,  # Optional
                "higher_tf_bias": "bullish"  # Optional
            }
            n_results: Number of similar trades to return
            min_confidence: Minimum similarity score (0.0-1.0)
            regime_weight: Weight for regime matching (0.0-1.0)
        
        Returns:
            {
                "similar_trades": [...],
                "statistics": {...},
                "recommendation": "..."
            }
        """
        # Load all closed trades
        closed_trades = self._load_closed_trades()
        
        if not closed_trades:
            return {
                "similar_trades": [],
                "statistics": self._empty_statistics(),
                "recommendation": "No historical data available for comparison."
            }
        
        # Calculate similarity scores
        scored_trades = []
        for trade in closed_trades:
            similarity = self._calculate_similarity(current_setup, trade, regime_weight)
            
            if similarity >= min_confidence:
                scored_trades.append({
                    "trade": trade,
                    "similarity": similarity
                })
        
        # Sort by similarity (highest first)
        scored_trades.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top N
        top_trades = scored_trades[:n_results]
        
        # Calculate statistics
        stats = self._calculate_similarity_stats(top_trades, current_setup)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(stats, current_setup)
        
        return {
            "similar_trades": [t["trade"] for t in top_trades],
            "similarity_scores": [t["similarity"] for t in top_trades],
            "statistics": stats,
            "recommendation": recommendation
        }
    
    def _load_closed_trades(self) -> List[Dict[str, Any]]:
        """Load all closed trade decisions."""
        if not os.path.exists(self.decisions_dir):
            return []
        
        trades = []
        for filename in os.listdir(self.decisions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.decisions_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        trade = json.load(f)
                        if trade.get("status") == "closed":
                            trades.append(trade)
                except Exception:
                    continue
        
        return trades
    
    def _calculate_similarity(
        self,
        current: Dict[str, Any],
        historical: Dict[str, Any],
        regime_weight: float
    ) -> float:
        """
        Calculate similarity score between current setup and historical trade.
        
        Score components:
        - Symbol match: 0.2
        - Direction match: 0.2
        - Regime match: regime_weight (default 0.5)
        - Setup characteristics: (1.0 - regime_weight) (default 0.5)
        """
        score = 0.0
        max_score = 0.0
        
        # Symbol match (20% weight)
        max_score += 0.2
        if current.get("symbol") == historical.get("symbol"):
            score += 0.2
        
        # Direction match (20% weight)
        max_score += 0.2
        current_direction = current.get("direction", "").upper()
        historical_action = historical.get("action", "").upper()
        if current_direction in historical_action or historical_action in current_direction:
            score += 0.2
        
        # Regime match (regime_weight, default 50%)
        regime_score = self._calculate_regime_similarity(current, historical)
        score += regime_score * regime_weight
        max_score += regime_weight
        
        # Setup characteristics (1.0 - regime_weight, default 50%)
        setup_weight = 1.0 - regime_weight
        setup_score = self._calculate_setup_similarity(current, historical)
        score += setup_score * setup_weight
        max_score += setup_weight
        
        # Normalize to 0-1 range
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_regime_similarity(
        self,
        current: Dict[str, Any],
        historical: Dict[str, Any]
    ) -> float:
        """Calculate regime similarity (0.0-1.0)."""
        score = 0.0
        count = 0
        
        # Market regime (most important)
        if current.get("market_regime") and historical.get("market_regime"):
            count += 1
            if current["market_regime"] == historical["market_regime"]:
                score += 1.0
        
        # Volatility regime
        if current.get("volatility_regime") and historical.get("volatility_regime"):
            count += 1
            if current["volatility_regime"] == historical["volatility_regime"]:
                score += 1.0
            elif self._adjacent_volatility(current["volatility_regime"], historical["volatility_regime"]):
                score += 0.5  # Partial match for adjacent levels
        
        # Expansion regime (less important)
        if current.get("expansion_regime") and historical.get("expansion_regime"):
            count += 0.5  # Lower weight
            if current["expansion_regime"] == historical["expansion_regime"]:
                score += 0.5
        
        return score / count if count > 0 else 0.5  # Default 0.5 if no regime data
    
    def _adjacent_volatility(self, vol1: str, vol2: str) -> bool:
        """Check if volatility levels are adjacent."""
        levels = ["low", "normal", "high", "extreme"]
        try:
            idx1 = levels.index(vol1)
            idx2 = levels.index(vol2)
            return abs(idx1 - idx2) == 1
        except ValueError:
            return False
    
    def _calculate_setup_similarity(
        self,
        current: Dict[str, Any],
        historical: Dict[str, Any]
    ) -> float:
        """Calculate setup characteristics similarity (0.0-1.0)."""
        score = 0.0
        count = 0
        
        # Setup type (if available)
        if current.get("setup_type") and historical.get("setup_type"):
            count += 1
            if current["setup_type"] == historical["setup_type"]:
                score += 1.0
        
        # Higher timeframe bias
        if current.get("higher_tf_bias") and historical.get("higher_tf_bias"):
            count += 1
            if current["higher_tf_bias"] == historical["higher_tf_bias"]:
                score += 1.0
        
        # Confluence score (within ±2 points)
        if current.get("confluence_score") is not None and historical.get("confluence_score") is not None:
            count += 1
            diff = abs(current["confluence_score"] - historical["confluence_score"])
            if diff == 0:
                score += 1.0
            elif diff <= 2:
                score += 0.7
            elif diff <= 4:
                score += 0.3
        
        return score / count if count > 0 else 0.5  # Default 0.5 if no setup data
    
    def _calculate_similarity_stats(
        self,
        similar_trades: List[Dict[str, Any]],
        current_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from similar trades.
        
        Returns:
            {
                "sample_size": 5,
                "win_rate": 0.65,
                "avg_rr": 2.3,
                "avg_reward": 1.8,
                "best_rr": 3.5,
                "worst_rr": -1.0,
                "confidence_adjustment": +0.15,
                "regime": "trending-up / high volatility"
            }
        """
        if not similar_trades:
            return self._empty_statistics()
        
        trades = [t["trade"] for t in similar_trades]
        
        # Win rate
        wins = sum(1 for t in trades if t.get("was_correct"))
        win_rate = wins / len(trades)
        
        # Risk-reward stats
        rr_values = [t.get("rr_realized") for t in trades if t.get("rr_realized") is not None]
        avg_rr = np.mean(rr_values) if rr_values else 0.0
        best_rr = max(rr_values) if rr_values else 0.0
        worst_rr = min(rr_values) if rr_values else 0.0
        
        # Reward signal stats
        rewards = [t.get("reward_signal") for t in trades if t.get("reward_signal") is not None]
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Confidence adjustment based on performance
        confidence_adjustment = self._calculate_confidence_adjustment(win_rate, avg_rr, len(trades))
        
        # Regime summary
        regime = f"{current_setup.get('market_regime', 'unknown')} / {current_setup.get('volatility_regime', 'unknown')}"
        
        return {
            "sample_size": len(trades),
            "win_rate": float(win_rate),
            "avg_rr": float(avg_rr),
            "avg_reward": float(avg_reward),
            "best_rr": float(best_rr),
            "worst_rr": float(worst_rr),
            "confidence_adjustment": float(confidence_adjustment),
            "regime": regime,
            "avg_similarity": float(np.mean([t["similarity"] for t in similar_trades]))
        }
    
    def _calculate_confidence_adjustment(
        self,
        win_rate: float,
        avg_rr: float,
        sample_size: int
    ) -> float:
        """
        Calculate confidence adjustment based on historical performance.
        
        Returns:
            Adjustment factor (-0.3 to +0.3)
        """
        # Base adjustment on win rate
        if win_rate >= 0.70:
            adjustment = 0.2
        elif win_rate >= 0.60:
            adjustment = 0.1
        elif win_rate >= 0.50:
            adjustment = 0.0
        elif win_rate >= 0.40:
            adjustment = -0.1
        else:
            adjustment = -0.2
        
        # Bonus for good risk-reward
        if avg_rr > 2.0:
            adjustment += 0.1
        elif avg_rr < 0.5:
            adjustment -= 0.1
        
        # Reduce adjustment for small sample sizes
        if sample_size < 5:
            adjustment *= 0.5
        elif sample_size < 10:
            adjustment *= 0.75
        
        # Clip to range
        return np.clip(adjustment, -0.3, 0.3)
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "avg_reward": 0.0,
            "best_rr": 0.0,
            "worst_rr": 0.0,
            "confidence_adjustment": 0.0,
            "regime": "unknown",
            "avg_similarity": 0.0
        }
    
    def _generate_recommendation(
        self,
        stats: Dict[str, Any],
        current_setup: Dict[str, Any]
    ) -> str:
        """
        Generate actionable recommendation based on statistics.
        
        Returns:
            Human-readable recommendation string
        """
        if stats["sample_size"] == 0:
            return "No similar historical trades found. Proceed with caution and use base confidence."
        
        win_rate = stats["win_rate"]
        avg_rr = stats["avg_rr"]
        sample_size = stats["sample_size"]
        
        # Build recommendation
        parts = []
        
        # Sample size context
        if sample_size < 5:
            parts.append(f"Limited historical data ({sample_size} similar trades).")
        else:
            parts.append(f"Found {sample_size} similar trades in {stats['regime']}.")
        
        # Win rate assessment
        if win_rate >= 0.70:
            parts.append(f"STRONG historical performance: {win_rate*100:.0f}% win rate.")
            parts.append("INCREASE confidence by +0.1 to +0.2.")
        elif win_rate >= 0.60:
            parts.append(f"Good historical performance: {win_rate*100:.0f}% win rate.")
            parts.append("Slight confidence boost (+0.1) warranted.")
        elif win_rate >= 0.50:
            parts.append(f"Neutral historical performance: {win_rate*100:.0f}% win rate.")
            parts.append("Use base confidence level.")
        elif win_rate >= 0.40:
            parts.append(f"Below-average historical performance: {win_rate*100:.0f}% win rate.")
            parts.append("REDUCE confidence by -0.1.")
        else:
            parts.append(f"POOR historical performance: {win_rate*100:.0f}% win rate.")
            parts.append("REDUCE confidence by -0.2 or SKIP this setup.")
        
        # Risk-reward context
        if avg_rr > 2.0:
            parts.append(f"Average RR: {avg_rr:.2f} (excellent).")
        elif avg_rr > 1.0:
            parts.append(f"Average RR: {avg_rr:.2f} (acceptable).")
        elif avg_rr > 0:
            parts.append(f"Average RR: {avg_rr:.2f} (poor - wins are small).")
        else:
            parts.append(f"Average RR: {avg_rr:.2f} (losses exceed wins).")
        
        return " ".join(parts)
    
    def format_for_prompt(
        self,
        similar_trades: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        max_trades: int = 3
    ) -> str:
        """
        Format similar trades for inclusion in agent prompts.
        
        Args:
            similar_trades: List of similar trade dicts
            statistics: Statistics dict from find_similar_trades
            max_trades: Maximum number of trades to include in detail
        
        Returns:
            Formatted string for prompt inclusion
        """
        if not similar_trades:
            return "No similar historical trades found."
        
        output = f"""HISTORICAL CONTEXT ({statistics['sample_size']} similar trades):

Performance in {statistics['regime']}:
- Win Rate: {statistics['win_rate']*100:.1f}%
- Avg Risk-Reward: {statistics['avg_rr']:.2f}R
- Best Trade: {statistics['best_rr']:+.2f}R
- Worst Trade: {statistics['worst_rr']:+.2f}R
- Avg Reward Signal: {statistics['avg_reward']:+.2f}

Top {min(max_trades, len(similar_trades))} Similar Trades:
"""
        
        for i, trade in enumerate(similar_trades[:max_trades], 1):
            outcome = "✓ WIN" if trade.get("was_correct") else "✗ LOSS"
            rr = trade.get("rr_realized", 0)
            setup = trade.get("setup_type", "unknown")
            
            output += f"\n{i}. {outcome} | {rr:+.2f}R | Setup: {setup}"
            
            # Add brief rationale if available
            rationale = trade.get("rationale", "")
            if rationale and len(rationale) < 100:
                output += f"\n   Rationale: {rationale[:100]}"
        
        output += f"\n\nRECOMMENDATION: {statistics.get('confidence_adjustment', 0):+.2f} confidence adjustment"
        
        return output
