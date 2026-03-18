"""
Storage abstraction layer for TradingAgents.

Supports multiple backends:
- FileStore: Local JSON/pickle files (default, current behavior)
- PostgresStore: Remote PostgreSQL database (Vercel Postgres / Neon)

Configure via environment variable:
    TRADING_AGENTS_STORAGE=postgres  # Use Postgres
    POSTGRES_URL=postgresql://...    # Connection string

If not set, defaults to file-based storage for backward compatibility.
"""

from .base import DecisionStore, AutomationStateStore
from .file_store import FileDecisionStore, FileAutomationStateStore
from .trade_queue import TradeQueueStore, get_trade_queue, TradeCommand
from .automation_control import AutomationControlStore, get_automation_control, ControlCommand

__all__ = [
    "DecisionStore",
    "AutomationStateStore",
    "FileDecisionStore",
    "FileAutomationStateStore",
    "get_decision_store",
    "get_automation_state_store",
    "TradeQueueStore",
    "get_trade_queue",
    "TradeCommand",
    "AutomationControlStore",
    "get_automation_control",
    "ControlCommand",
]

# Singleton instances
_decision_store = None
_automation_state_store = None


def get_decision_store() -> DecisionStore:
    """Get the configured decision store instance (singleton)."""
    global _decision_store
    if _decision_store is not None:
        return _decision_store

    import os

    storage_type = os.environ.get("TRADING_AGENTS_STORAGE", "file").lower()

    if storage_type == "postgres":
        from .postgres_store import PostgresDecisionStore

        _decision_store = PostgresDecisionStore()
    else:
        _decision_store = FileDecisionStore()

    return _decision_store


def get_automation_state_store() -> AutomationStateStore:
    """Get the configured automation state store instance (singleton)."""
    global _automation_state_store
    if _automation_state_store is not None:
        return _automation_state_store

    import os

    storage_type = os.environ.get("TRADING_AGENTS_STORAGE", "file").lower()

    if storage_type == "postgres":
        from .postgres_store import PostgresAutomationStateStore

        _automation_state_store = PostgresAutomationStateStore()
    else:
        _automation_state_store = FileAutomationStateStore()

    return _automation_state_store
