"""
Trade Execution Module

Intelligent execution system that:
- Parses trading plans from analyst output
- Decides market vs limit entry
- Manages staged entries (tranches)
- Sets dynamic stops and take-profits
- Reviews and adapts plans
"""

from .plan_parser import TradingPlanParser
from .order_executor import OrderExecutor, OrderType, OrderStatus
from .staged_entry import StagedEntryManager, TrancheStatus
from .dynamic_stops import DynamicStopManager
from .plan_reviewer import PlanReviewer

__all__ = [
    "TradingPlanParser",
    "OrderExecutor",
    "OrderType",
    "OrderStatus",
    "StagedEntryManager",
    "TrancheStatus",
    "DynamicStopManager",
    "PlanReviewer",
]
