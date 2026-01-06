# TradingAgents Risk Management Module
"""
Quantitative risk metrics and portfolio management.

This module provides:
- RiskMetrics: Sharpe, Sortino, VaR, max drawdown, Calmar ratio
- Portfolio: Position tracking and equity curve management
- PositionSizer: Kelly criterion and risk-based position sizing
"""

from .metrics import RiskMetrics, RiskReport
from .portfolio import Portfolio
from .position_sizing import (
    PositionSizer, 
    PositionSizeResult,
    calculate_kelly_from_history,
    recommend_position_size
)

__all__ = [
    "RiskMetrics", 
    "RiskReport",
    "Portfolio", 
    "PositionSizer",
    "PositionSizeResult",
    "calculate_kelly_from_history",
    "recommend_position_size"
]
