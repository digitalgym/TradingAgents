"""
Staged Entry Manager

Manages multiple entry tranches:
- Tracks which tranches are filled
- Monitors conditions for each tranche
- Adjusts remaining tranches based on fills
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os


class TrancheStatus(Enum):
    """Status of an entry tranche"""
    PENDING = "pending"
    ACTIVE = "active"  # Order placed
    FILLED = "filled"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class TrancheState:
    """State of a single tranche"""
    tranche_number: int
    size_pct: float
    target_price: float
    price_range: Optional[tuple]
    conditions: List[str]
    status: TrancheStatus
    order_ticket: Optional[int] = None
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    filled_volume: float = 0.0


class StagedEntryManager:
    """
    Manages staged entry execution.
    
    Features:
    - Tracks multiple tranches
    - Places orders for each tranche
    - Monitors fills
    - Adjusts remaining tranches
    - Persists state
    """
    
    def __init__(self, plan_id: str, storage_dir: str = "examples/staged_entries"):
        self.plan_id = plan_id
        self.storage_dir = storage_dir
        self.tranches: List[TrancheState] = []
        self.total_filled_pct = 0.0
        self.total_filled_volume = 0.0
        self.avg_entry_price = 0.0
        
        os.makedirs(storage_dir, exist_ok=True)
    
    def initialize_tranches(self, entry_tranches: list):
        """
        Initialize tranches from parsed plan.
        
        Args:
            entry_tranches: List of EntryTranche from plan parser
        """
        self.tranches = []
        
        for tranche in entry_tranches:
            self.tranches.append(TrancheState(
                tranche_number=tranche.tranche_number,
                size_pct=tranche.size_pct,
                target_price=tranche.price_level,
                price_range=tranche.price_range,
                conditions=tranche.conditions,
                status=TrancheStatus.PENDING
            ))
        
        self.save_state()
    
    def get_next_tranche(self) -> Optional[TrancheState]:
        """Get the next pending tranche to execute"""
        for tranche in self.tranches:
            if tranche.status == TrancheStatus.PENDING:
                return tranche
        return None
    
    def mark_tranche_active(self, tranche_number: int, order_ticket: int):
        """Mark a tranche as active (order placed)"""
        for tranche in self.tranches:
            if tranche.tranche_number == tranche_number:
                tranche.status = TrancheStatus.ACTIVE
                tranche.order_ticket = order_ticket
                self.save_state()
                break
    
    def mark_tranche_filled(
        self,
        tranche_number: int,
        filled_price: float,
        filled_volume: float
    ):
        """
        Mark a tranche as filled and update totals.
        
        Args:
            tranche_number: Tranche number
            filled_price: Actual fill price
            filled_volume: Actual fill volume
        """
        for tranche in self.tranches:
            if tranche.tranche_number == tranche_number:
                tranche.status = TrancheStatus.FILLED
                tranche.filled_price = filled_price
                tranche.filled_time = datetime.now()
                tranche.filled_volume = filled_volume
                
                # Update totals
                self.total_filled_pct += tranche.size_pct
                self.total_filled_volume += filled_volume
                
                # Recalculate average entry
                self._recalculate_avg_entry()
                
                self.save_state()
                break
    
    def mark_tranche_cancelled(self, tranche_number: int):
        """Mark a tranche as cancelled"""
        for tranche in self.tranches:
            if tranche.tranche_number == tranche_number:
                tranche.status = TrancheStatus.CANCELLED
                self.save_state()
                break
    
    def mark_tranche_skipped(self, tranche_number: int, reason: str = ""):
        """Skip a tranche (conditions not met, price moved away, etc.)"""
        for tranche in self.tranches:
            if tranche.tranche_number == tranche_number:
                tranche.status = TrancheStatus.SKIPPED
                self.save_state()
                break
    
    def _recalculate_avg_entry(self):
        """Recalculate average entry price from filled tranches"""
        total_cost = 0.0
        total_volume = 0.0
        
        for tranche in self.tranches:
            if tranche.status == TrancheStatus.FILLED:
                total_cost += tranche.filled_price * tranche.filled_volume
                total_volume += tranche.filled_volume
        
        if total_volume > 0:
            self.avg_entry_price = total_cost / total_volume
        else:
            self.avg_entry_price = 0.0
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of staged entry status"""
        pending = sum(1 for t in self.tranches if t.status == TrancheStatus.PENDING)
        active = sum(1 for t in self.tranches if t.status == TrancheStatus.ACTIVE)
        filled = sum(1 for t in self.tranches if t.status == TrancheStatus.FILLED)
        cancelled = sum(1 for t in self.tranches if t.status == TrancheStatus.CANCELLED)
        skipped = sum(1 for t in self.tranches if t.status == TrancheStatus.SKIPPED)
        
        return {
            "plan_id": self.plan_id,
            "total_tranches": len(self.tranches),
            "pending": pending,
            "active": active,
            "filled": filled,
            "cancelled": cancelled,
            "skipped": skipped,
            "total_filled_pct": self.total_filled_pct,
            "total_filled_volume": self.total_filled_volume,
            "avg_entry_price": self.avg_entry_price,
            "is_complete": pending == 0 and active == 0
        }
    
    def get_tranche_details(self) -> List[Dict[str, Any]]:
        """Get detailed info for each tranche"""
        details = []
        
        for tranche in self.tranches:
            details.append({
                "tranche_number": tranche.tranche_number,
                "size_pct": tranche.size_pct,
                "target_price": tranche.target_price,
                "price_range": tranche.price_range,
                "status": tranche.status.value,
                "order_ticket": tranche.order_ticket,
                "filled_price": tranche.filled_price,
                "filled_volume": tranche.filled_volume,
                "filled_time": tranche.filled_time.isoformat() if tranche.filled_time else None
            })
        
        return details
    
    def should_adjust_remaining(self, current_price: float) -> Dict[str, Any]:
        """
        Check if remaining tranches should be adjusted.
        
        Args:
            current_price: Current market price
        
        Returns:
            dict with adjustment recommendation
        """
        # Get filled tranches
        filled_tranches = [t for t in self.tranches if t.status == TrancheStatus.FILLED]
        pending_tranches = [t for t in self.tranches if t.status == TrancheStatus.PENDING]
        
        if not filled_tranches or not pending_tranches:
            return {"should_adjust": False}
        
        # Check if price has moved significantly from filled tranches
        avg_filled_price = sum(t.filled_price for t in filled_tranches) / len(filled_tranches)
        price_move_pct = ((current_price - avg_filled_price) / avg_filled_price) * 100
        
        # If price moved >2% in favorable direction, consider adjusting
        if abs(price_move_pct) > 2.0:
            # Price moved away from remaining tranches
            if price_move_pct > 0:  # Price went up
                # For BUY orders, this is unfavorable
                return {
                    "should_adjust": True,
                    "reason": f"Price moved {price_move_pct:+.2f}% away from remaining tranches",
                    "recommendation": "Consider cancelling higher tranches or adjusting limits up"
                }
            else:  # Price went down
                # For BUY orders, this is favorable
                return {
                    "should_adjust": True,
                    "reason": f"Price moved {price_move_pct:+.2f}% favorably",
                    "recommendation": "Consider accelerating remaining tranches"
                }
        
        return {"should_adjust": False}
    
    def save_state(self):
        """Save staged entry state to disk"""
        state = {
            "plan_id": self.plan_id,
            "total_filled_pct": self.total_filled_pct,
            "total_filled_volume": self.total_filled_volume,
            "avg_entry_price": self.avg_entry_price,
            "tranches": []
        }
        
        for tranche in self.tranches:
            state["tranches"].append({
                "tranche_number": tranche.tranche_number,
                "size_pct": tranche.size_pct,
                "target_price": tranche.target_price,
                "price_range": tranche.price_range,
                "conditions": tranche.conditions,
                "status": tranche.status.value,
                "order_ticket": tranche.order_ticket,
                "filled_price": tranche.filled_price,
                "filled_time": tranche.filled_time.isoformat() if tranche.filled_time else None,
                "filled_volume": tranche.filled_volume
            })
        
        filepath = os.path.join(self.storage_dir, f"{self.plan_id}.json")
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, plan_id: str, storage_dir: str = "examples/staged_entries"):
        """Load staged entry state from disk"""
        filepath = os.path.join(storage_dir, f"{plan_id}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Staged entry state not found: {plan_id}")
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        manager = cls(plan_id, storage_dir)
        manager.total_filled_pct = state["total_filled_pct"]
        manager.total_filled_volume = state["total_filled_volume"]
        manager.avg_entry_price = state["avg_entry_price"]
        
        for t_data in state["tranches"]:
            tranche = TrancheState(
                tranche_number=t_data["tranche_number"],
                size_pct=t_data["size_pct"],
                target_price=t_data["target_price"],
                price_range=tuple(t_data["price_range"]) if t_data["price_range"] else None,
                conditions=t_data["conditions"],
                status=TrancheStatus(t_data["status"]),
                order_ticket=t_data.get("order_ticket"),
                filled_price=t_data.get("filled_price"),
                filled_time=datetime.fromisoformat(t_data["filled_time"]) if t_data.get("filled_time") else None,
                filled_volume=t_data.get("filled_volume", 0.0)
            )
            manager.tranches.append(tranche)
        
        return manager
    
    def format_status_report(self) -> str:
        """Format a readable status report"""
        summary = self.get_status_summary()
        
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"STAGED ENTRY STATUS: {self.plan_id}")
        lines.append(f"{'='*70}\n")
        
        lines.append(f"Progress: {summary['filled']}/{summary['total_tranches']} tranches filled")
        lines.append(f"Position: {summary['total_filled_pct']:.1f}% ({summary['total_filled_volume']:.3f} lots)")
        
        if summary['avg_entry_price'] > 0:
            lines.append(f"Avg Entry: ${summary['avg_entry_price']:.2f}\n")
        
        lines.append(f"Status Breakdown:")
        lines.append(f"  Filled:    {summary['filled']}")
        lines.append(f"  Active:    {summary['active']} (orders placed)")
        lines.append(f"  Pending:   {summary['pending']}")
        lines.append(f"  Cancelled: {summary['cancelled']}")
        lines.append(f"  Skipped:   {summary['skipped']}\n")
        
        lines.append(f"Tranche Details:")
        for tranche in self.tranches:
            status_icon = {
                TrancheStatus.FILLED: "✅",
                TrancheStatus.ACTIVE: "⏳",
                TrancheStatus.PENDING: "⏸️",
                TrancheStatus.CANCELLED: "❌",
                TrancheStatus.SKIPPED: "⏭️"
            }.get(tranche.status, "?")
            
            lines.append(f"  {status_icon} Tranche {tranche.tranche_number}: {tranche.size_pct:.1f}% at ${tranche.target_price:.2f}")
            
            if tranche.status == TrancheStatus.FILLED:
                lines.append(f"     Filled at ${tranche.filled_price:.2f} ({tranche.filled_volume:.3f} lots)")
            elif tranche.status == TrancheStatus.ACTIVE:
                lines.append(f"     Order #{tranche.order_ticket} active")
        
        lines.append(f"\n{'='*70}")
        
        return '\n'.join(lines)
