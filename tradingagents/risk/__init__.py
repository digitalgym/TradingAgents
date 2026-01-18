# TradingAgents Risk Management Module
"""
Quantitative risk metrics and portfolio management.

This module provides:
- RiskMetrics: Sharpe, Sortino, VaR, max drawdown, Calmar ratio
- Portfolio: Position tracking and equity curve management
- PositionSizer: Kelly criterion and risk-based position sizing
- DynamicStopLoss: ATR-based stop-loss and trailing stops
"""

from .metrics import RiskMetrics, RiskReport
from .portfolio import Portfolio
from .position_sizing import (
    PositionSizer, 
    PositionSizeResult,
    calculate_kelly_from_history,
    recommend_position_size
)
from .stop_loss import (
    DynamicStopLoss,
    StopLossLevels,
    calculate_atr,
    get_atr_for_symbol,
)
from .guardrails import RiskGuardrails

__all__ = [
    "RiskMetrics", 
    "RiskReport",
    "Portfolio", 
    "PositionSizer",
    "PositionSizeResult",
    "calculate_kelly_from_history",
    "recommend_position_size",
    "DynamicStopLoss",
    "StopLossLevels",
    "calculate_atr",
    "get_atr_for_symbol",
    "RiskGuardrails",
]
