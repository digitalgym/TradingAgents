"""
Abstract base classes for storage backends.

These define the interface that all storage implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class DecisionStore(ABC):
    """Abstract interface for trade decision storage."""

    @abstractmethod
    def store(self, decision: Dict[str, Any]) -> str:
        """
        Store a trade decision.

        Args:
            decision: Decision dict with all fields

        Returns:
            decision_id
        """
        pass

    @abstractmethod
    def load(self, decision_id: str) -> Dict[str, Any]:
        """
        Load a decision by ID.

        Args:
            decision_id: Unique decision identifier

        Returns:
            Decision dict

        Raises:
            KeyError: If decision not found
        """
        pass

    @abstractmethod
    def update(self, decision_id: str, updates: Dict[str, Any]) -> None:
        """
        Update specific fields of a decision.

        Args:
            decision_id: Unique decision identifier
            updates: Dict of field -> new value
        """
        pass

    @abstractmethod
    def list_active(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all active (unclosed) decisions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of decision dicts, sorted by created_at desc
        """
        pass

    @abstractmethod
    def list_closed(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List closed decisions.

        Args:
            symbol: Optional symbol filter
            limit: Max results to return

        Returns:
            List of decision dicts, sorted by exit_date desc
        """
        pass

    @abstractmethod
    def find_by_ticket(self, mt5_ticket: int) -> Optional[Dict[str, Any]]:
        """
        Find a decision by MT5 ticket number.

        Args:
            mt5_ticket: MT5 position ticket

        Returns:
            Decision dict or None
        """
        pass

    @abstractmethod
    def store_context(self, decision_id: str, context: Dict[str, Any]) -> None:
        """
        Store analysis context for a decision (can be large).

        Args:
            decision_id: Decision to attach context to
            context: Full analysis context dict
        """
        pass

    @abstractmethod
    def load_context(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis context for a decision.

        Args:
            decision_id: Decision ID

        Returns:
            Context dict or None if not stored
        """
        pass

    @abstractmethod
    def add_event(
        self,
        decision_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        """
        Add an event to decision's event log.

        Args:
            decision_id: Decision ID
            event_type: Type of event (executed, sl_hit, tp_hit, etc.)
            details: Optional event details
            source: Source of event
        """
        pass


class AutomationStateStore(ABC):
    """Abstract interface for automation state storage."""

    @abstractmethod
    def save_state(self, instance_name: str, state: Dict[str, Any]) -> None:
        """
        Save automation state.

        Args:
            instance_name: Automation instance identifier
            state: State dict to persist
        """
        pass

    @abstractmethod
    def load_state(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """
        Load automation state.

        Args:
            instance_name: Automation instance identifier

        Returns:
            State dict or None if not found
        """
        pass

    @abstractmethod
    def save_guardrails(self, instance_name: str, guardrails: Dict[str, Any]) -> None:
        """
        Save risk guardrails state.

        Args:
            instance_name: Automation instance identifier
            guardrails: Guardrails state dict
        """
        pass

    @abstractmethod
    def load_guardrails(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """
        Load risk guardrails state.

        Args:
            instance_name: Automation instance identifier

        Returns:
            Guardrails state dict or None
        """
        pass
