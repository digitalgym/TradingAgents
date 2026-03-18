"""
File-based storage implementation.

This preserves the existing behavior using JSON files and pickle for contexts.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import DecisionStore, AutomationStateStore


# Default directories
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DECISIONS_DIR = _PROJECT_ROOT / "examples" / "trade_decisions"
AUTOMATION_STATE_DIR = _PROJECT_ROOT / "automation_state"


class FileDecisionStore(DecisionStore):
    """File-based decision storage using JSON files."""

    def __init__(self, decisions_dir: Optional[Path] = None):
        self.decisions_dir = Path(decisions_dir) if decisions_dir else DECISIONS_DIR
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

        # Index files for fast lookup
        self._ticket_index_file = self.decisions_dir / "_ticket_index.json"
        self._active_index_file = self.decisions_dir / "_active_index.json"

    def _decision_file(self, decision_id: str) -> Path:
        return self.decisions_dir / f"{decision_id}.json"

    def _context_file(self, decision_id: str) -> Path:
        return self.decisions_dir / f"{decision_id}_context.pkl"

    # --- Index management ---

    def _load_ticket_index(self) -> Dict[int, str]:
        if not self._ticket_index_file.exists():
            return {}
        try:
            with open(self._ticket_index_file, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            return {}

    def _save_ticket_index(self, index: Dict[int, str]) -> None:
        try:
            with open(self._ticket_index_file, "w") as f:
                json.dump({str(k): v for k, v in index.items()}, f)
        except Exception:
            pass

    def _update_ticket_index(self, mt5_ticket: int, decision_id: str) -> None:
        if mt5_ticket is None:
            return
        index = self._load_ticket_index()
        index[mt5_ticket] = decision_id
        self._save_ticket_index(index)

    def _remove_from_ticket_index(self, mt5_ticket: int) -> None:
        if mt5_ticket is None:
            return
        index = self._load_ticket_index()
        if mt5_ticket in index:
            del index[mt5_ticket]
            self._save_ticket_index(index)

    def _load_active_index(self) -> Dict[str, Dict[str, Any]]:
        if not self._active_index_file.exists():
            return {}
        try:
            with open(self._active_index_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_active_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        try:
            with open(self._active_index_file, "w") as f:
                json.dump(index, f)
        except Exception:
            pass

    def _add_to_active_index(
        self, decision_id: str, symbol: str, created_at: str
    ) -> None:
        index = self._load_active_index()
        index[decision_id] = {"symbol": symbol, "created_at": created_at}
        self._save_active_index(index)

    def _remove_from_active_index(self, decision_id: str) -> None:
        index = self._load_active_index()
        if decision_id in index:
            del index[decision_id]
            self._save_active_index(index)

    # --- DecisionStore implementation ---

    def store(self, decision: Dict[str, Any]) -> str:
        decision_id = decision["decision_id"]

        # Save decision JSON
        with open(self._decision_file(decision_id), "w") as f:
            json.dump(decision, f, indent=2, default=str)

        # Update indexes
        if decision.get("mt5_ticket"):
            self._update_ticket_index(decision["mt5_ticket"], decision_id)

        if decision.get("status") == "active":
            self._add_to_active_index(
                decision_id, decision["symbol"], decision["created_at"]
            )

        return decision_id

    def load(self, decision_id: str) -> Dict[str, Any]:
        filepath = self._decision_file(decision_id)
        if not filepath.exists():
            raise KeyError(f"Decision not found: {decision_id}")

        with open(filepath, "r") as f:
            return json.load(f)

    def update(self, decision_id: str, updates: Dict[str, Any]) -> None:
        decision = self.load(decision_id)
        old_status = decision.get("status")

        decision.update(updates)

        with open(self._decision_file(decision_id), "w") as f:
            json.dump(decision, f, indent=2, default=str)

        # Update indexes if status changed
        new_status = decision.get("status")
        if old_status == "active" and new_status != "active":
            self._remove_from_active_index(decision_id)

        # Update ticket index if ticket was added
        if "mt5_ticket" in updates and updates["mt5_ticket"]:
            self._update_ticket_index(updates["mt5_ticket"], decision_id)

    def list_active(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        # Try fast index lookup
        index = self._load_active_index()
        if index:
            decisions = []
            stale_ids = []

            for decision_id, meta in index.items():
                if symbol is not None and meta.get("symbol") != symbol:
                    continue
                try:
                    decision = self.load(decision_id)
                    if decision["status"] == "active":
                        decisions.append(decision)
                    else:
                        stale_ids.append(decision_id)
                except KeyError:
                    stale_ids.append(decision_id)
                except Exception:
                    continue

            # Clean up stale entries
            for stale_id in stale_ids:
                self._remove_from_active_index(stale_id)

            if decisions:
                return sorted(
                    decisions, key=lambda d: d["created_at"], reverse=True
                )

        # Fallback: scan all files
        decisions = []
        for f in self.decisions_dir.iterdir():
            if f.suffix == ".json" and not f.name.startswith("_"):
                decision_id = f.stem
                try:
                    decision = self.load(decision_id)
                    if decision["status"] == "active":
                        # Rebuild index
                        self._add_to_active_index(
                            decision_id,
                            decision["symbol"],
                            decision["created_at"],
                        )
                        if symbol is None or decision["symbol"] == symbol:
                            decisions.append(decision)
                except Exception:
                    continue

        return sorted(decisions, key=lambda d: d["created_at"], reverse=True)

    def list_closed(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        decisions = []
        for f in self.decisions_dir.iterdir():
            if f.suffix == ".json" and not f.name.startswith("_"):
                decision_id = f.stem
                try:
                    decision = self.load(decision_id)
                    if decision["status"] == "closed":
                        if symbol is None or decision["symbol"] == symbol:
                            decisions.append(decision)
                except Exception:
                    continue

        decisions = sorted(
            decisions, key=lambda d: d.get("exit_date", ""), reverse=True
        )
        return decisions[:limit]

    def find_by_ticket(self, mt5_ticket: int) -> Optional[Dict[str, Any]]:
        # Try fast index lookup
        index = self._load_ticket_index()
        if mt5_ticket in index:
            decision_id = index[mt5_ticket]
            try:
                return self.load(decision_id)
            except KeyError:
                self._remove_from_ticket_index(mt5_ticket)

        # Fallback: scan all files
        for f in self.decisions_dir.iterdir():
            if f.suffix == ".json" and not f.name.startswith("_"):
                decision_id = f.stem
                try:
                    decision = self.load(decision_id)
                    if decision.get("mt5_ticket") == mt5_ticket:
                        self._update_ticket_index(mt5_ticket, decision_id)
                        return decision
                except Exception:
                    continue

        return None

    def store_context(self, decision_id: str, context: Dict[str, Any]) -> None:
        with open(self._context_file(decision_id), "wb") as f:
            pickle.dump(context, f)

        # Update decision to indicate context exists
        try:
            decision = self.load(decision_id)
            decision["has_context"] = True
            with open(self._decision_file(decision_id), "w") as f:
                json.dump(decision, f, indent=2, default=str)
        except Exception:
            pass

    def load_context(self, decision_id: str) -> Optional[Dict[str, Any]]:
        context_file = self._context_file(decision_id)
        if not context_file.exists():
            return None

        with open(context_file, "rb") as f:
            return pickle.load(f)

    def add_event(
        self,
        decision_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        try:
            decision = self.load(decision_id)

            if "events" not in decision:
                decision["events"] = []

            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "source": source,
            }
            if details:
                event.update(details)

            decision["events"].append(event)

            with open(self._decision_file(decision_id), "w") as f:
                json.dump(decision, f, indent=2, default=str)
        except Exception:
            pass  # Don't let event logging break trade flow


class FileAutomationStateStore(AutomationStateStore):
    """File-based automation state storage."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = Path(state_dir) if state_dir else AUTOMATION_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _state_file(self, instance_name: str) -> Path:
        return self.state_dir / f"{instance_name}_state.json"

    def _guardrails_file(self, instance_name: str) -> Path:
        return self.state_dir / f"{instance_name}_guardrails.json"

    def save_state(self, instance_name: str, state: Dict[str, Any]) -> None:
        with open(self._state_file(instance_name), "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, instance_name: str) -> Optional[Dict[str, Any]]:
        filepath = self._state_file(instance_name)
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def save_guardrails(self, instance_name: str, guardrails: Dict[str, Any]) -> None:
        with open(self._guardrails_file(instance_name), "w") as f:
            json.dump(guardrails, f, indent=2, default=str)

    def load_guardrails(self, instance_name: str) -> Optional[Dict[str, Any]]:
        filepath = self._guardrails_file(instance_name)
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)
