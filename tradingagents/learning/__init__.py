"""
Learning module for continuous improvement and online learning.

Components:
- reward: Multi-factor reward signal calculation
- portfolio_state: Portfolio tracking for Sharpe/drawdown calculations
- online_rl: Agent weight updates (future)
- pattern_analyzer: Trade pattern clustering (future)
- trade_similarity: Similar trade search for RAG (future)
"""

from .reward import RewardCalculator
from .portfolio_state import PortfolioStateTracker
from .trade_similarity import TradeSimilaritySearch
from .pattern_analyzer import PatternAnalyzer
from .online_rl import OnlineRLUpdater

__all__ = [
    "RewardCalculator",
    "PortfolioStateTracker",
    "TradeSimilaritySearch",
    "PatternAnalyzer",
    "OnlineRLUpdater",
]
