"""
RAG-Enhanced Prompt Utilities

Helper functions to enhance agent prompts with historical context from similar trades.
"""

from typing import Dict, Any, Optional
from .trade_similarity import TradeSimilaritySearch


def enhance_prompt_with_rag(
    base_prompt: str,
    current_setup: Dict[str, Any],
    decisions_dir: Optional[str] = None,
    n_similar: int = 5,
    include_details: bool = True
) -> tuple[str, float]:
    """
    Enhance an agent prompt with RAG context from similar historical trades.
    
    Args:
        base_prompt: Original agent prompt
        current_setup: Current trade setup dict
        decisions_dir: Optional custom decisions directory
        n_similar: Number of similar trades to retrieve
        include_details: Whether to include detailed trade examples
    
    Returns:
        (enhanced_prompt, confidence_adjustment)
    """
    searcher = TradeSimilaritySearch(decisions_dir)
    
    # Find similar trades
    result = searcher.find_similar_trades(current_setup, n_results=n_similar)
    
    # Format historical context
    if result["similar_trades"]:
        historical_context = searcher.format_for_prompt(
            result["similar_trades"],
            result["statistics"],
            max_trades=3 if include_details else 0
        )
    else:
        historical_context = "No similar historical trades found."
    
    # Build enhanced prompt
    enhanced = f"""{base_prompt}

---

{historical_context}

IMPORTANT: Consider this historical performance when forming your recommendation.
- If historical win rate is strong (>65%), you may INCREASE confidence by up to +0.2
- If historical win rate is weak (<45%), you must REDUCE confidence by up to -0.2
- Provide strong justification if recommending against historical trends

Recommended confidence adjustment: {result['statistics']['confidence_adjustment']:+.2f}

---

"""
    
    return enhanced, result['statistics']['confidence_adjustment']


def create_rag_context_section(
    current_setup: Dict[str, Any],
    decisions_dir: Optional[str] = None,
    n_similar: int = 5
) -> Dict[str, Any]:
    """
    Create a structured RAG context section for agent state.
    
    Args:
        current_setup: Current trade setup dict
        decisions_dir: Optional custom decisions directory
        n_similar: Number of similar trades to retrieve
    
    Returns:
        {
            "similar_trades": [...],
            "statistics": {...},
            "recommendation": "...",
            "confidence_adjustment": float,
            "formatted_context": "..."
        }
    """
    searcher = TradeSimilaritySearch(decisions_dir)
    result = searcher.find_similar_trades(current_setup, n_results=n_similar)
    
    # Add formatted context
    result["formatted_context"] = searcher.format_for_prompt(
        result["similar_trades"],
        result["statistics"]
    )
    
    return result


def apply_confidence_adjustment(
    base_confidence: float,
    adjustment: float,
    min_confidence: float = 0.1,
    max_confidence: float = 0.95
) -> float:
    """
    Apply confidence adjustment with bounds.
    
    Args:
        base_confidence: Original confidence (0.0-1.0)
        adjustment: Adjustment from RAG (-0.3 to +0.3)
        min_confidence: Minimum allowed confidence
        max_confidence: Maximum allowed confidence
    
    Returns:
        Adjusted confidence clipped to bounds
    """
    adjusted = base_confidence + adjustment
    return max(min_confidence, min(max_confidence, adjusted))


def format_confidence_explanation(
    base_confidence: float,
    adjustment: float,
    statistics: Dict[str, Any]
) -> str:
    """
    Format explanation of confidence adjustment.
    
    Args:
        base_confidence: Original confidence
        adjustment: Applied adjustment
        statistics: Statistics from similar trades
    
    Returns:
        Human-readable explanation
    """
    final_confidence = apply_confidence_adjustment(base_confidence, adjustment)
    
    explanation = f"""Confidence Analysis:
- Base Confidence: {base_confidence:.2f}
- Historical Adjustment: {adjustment:+.2f}
- Final Confidence: {final_confidence:.2f}

Reasoning:
- Found {statistics['sample_size']} similar trades in {statistics['regime']}
- Historical win rate: {statistics['win_rate']*100:.1f}%
- Average RR: {statistics['avg_rr']:.2f}
"""
    
    if adjustment > 0.1:
        explanation += "- BOOST: Strong historical performance supports this setup"
    elif adjustment < -0.1:
        explanation += "- CAUTION: Weak historical performance suggests lower confidence"
    else:
        explanation += "- NEUTRAL: Historical performance is mixed or limited"
    
    return explanation
