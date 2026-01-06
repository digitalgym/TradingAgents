"""
Trade Decision Tracking System

Tracks trade decisions from analysis/review and links them to outcomes for learning.

Workflow:
1. Analysis/Review generates recommendation
2. User accepts recommendation -> store_decision() saves decision with rationale
3. Trade executes (manually or via MT5)
4. Trade closes -> close_decision() records outcome
5. System learns from decision quality

Decision Types:
- OPEN: New position based on analysis (BUY/SELL)
- ADJUST: Modify existing position (change SL/TP, partial close)
- CLOSE: Close existing position
- HOLD: Keep current position unchanged
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


# Directory for storing decisions
DECISIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "trade_decisions")


def store_decision(
    symbol: str,
    decision_type: str,  # OPEN, ADJUST, CLOSE, HOLD
    action: str,  # BUY, SELL, MODIFY_SL, MODIFY_TP, PARTIAL_CLOSE, etc.
    rationale: str,  # Why this decision was made
    source: str = "analysis",  # analysis, review, manual
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    volume: Optional[float] = None,
    mt5_ticket: Optional[int] = None,
    analysis_context: Optional[Dict[str, Any]] = None,
    position_sizing: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Store a trade decision for later outcome tracking.
    
    Args:
        symbol: Trading symbol (e.g., XAUUSD)
        decision_type: Type of decision (OPEN, ADJUST, CLOSE, HOLD)
        action: Specific action taken (BUY, SELL, MODIFY_SL, etc.)
        rationale: Summary of why this decision was made
        source: Where the decision came from (analysis, review, manual)
        entry_price: Entry price for new positions
        stop_loss: Stop loss level
        take_profit: Take profit level
        volume: Position size in lots
        mt5_ticket: MT5 position ticket if applicable
        analysis_context: Full analysis context (market report, news, etc.)
        position_sizing: Position sizing recommendation
        
    Returns:
        decision_id: Unique identifier for this decision
    """
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    
    timestamp = datetime.now()
    decision_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    decision = {
        "decision_id": decision_id,
        "symbol": symbol,
        "decision_type": decision_type,
        "action": action,
        "rationale": rationale,
        "source": source,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "volume": volume,
        "mt5_ticket": mt5_ticket,
        "created_at": timestamp.isoformat(),
        "status": "active",  # active, closed, cancelled
        # Outcome fields (filled when closed)
        "exit_price": None,
        "exit_date": None,
        "pnl": None,
        "pnl_percent": None,
        "outcome_notes": None,
        "was_correct": None,  # True if decision led to profit or avoided loss
    }
    
    # Store analysis context separately (can be large)
    if analysis_context:
        decision["has_context"] = True
        context_file = os.path.join(DECISIONS_DIR, f"{decision_id}_context.pkl")
        with open(context_file, "wb") as f:
            pickle.dump(analysis_context, f)
    else:
        decision["has_context"] = False
    
    if position_sizing:
        decision["position_sizing"] = position_sizing
    
    # Save decision as JSON for easy viewing
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    
    print(f"💾 Decision stored: {decision_id}")
    print(f"   Type: {decision_type} | Action: {action}")
    print(f"   Symbol: {symbol} @ {entry_price or 'market'}")
    
    return decision_id


def load_decision(decision_id: str) -> Dict[str, Any]:
    """Load a stored decision."""
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    
    if not os.path.exists(decision_file):
        raise FileNotFoundError(f"Decision not found: {decision_id}")
    
    with open(decision_file, "r") as f:
        return json.load(f)


def load_decision_context(decision_id: str) -> Optional[Dict[str, Any]]:
    """Load the full analysis context for a decision."""
    context_file = os.path.join(DECISIONS_DIR, f"{decision_id}_context.pkl")
    
    if not os.path.exists(context_file):
        return None
    
    with open(context_file, "rb") as f:
        return pickle.load(f)


def close_decision(
    decision_id: str,
    exit_price: float,
    outcome_notes: str = "",
    was_correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Close a decision and record the outcome.
    
    Args:
        decision_id: The decision to close
        exit_price: The exit price
        outcome_notes: Notes about what happened
        was_correct: Override auto-calculation of correctness
        
    Returns:
        Updated decision with outcome
    """
    decision = load_decision(decision_id)
    
    if decision["status"] != "active":
        print(f"⚠️ Decision {decision_id} already {decision['status']}")
        return decision
    
    entry_price = decision.get("entry_price")
    action = decision.get("action", "").upper()
    
    # Calculate P&L
    if entry_price and entry_price > 0:
        if action in ["BUY", "LONG"]:
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        elif action in ["SELL", "SHORT"]:
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
        else:
            pnl_percent = 0
        
        volume = decision.get("volume", 0.01)
        # Rough P&L calculation (actual depends on contract size)
        pnl = (exit_price - entry_price) * volume * 100 if action in ["BUY", "LONG"] else (entry_price - exit_price) * volume * 100
    else:
        pnl = 0
        pnl_percent = 0
    
    # Determine if decision was correct
    if was_correct is None:
        was_correct = pnl_percent > 0
    
    # Update decision
    decision["exit_price"] = exit_price
    decision["exit_date"] = datetime.now().isoformat()
    decision["pnl"] = pnl
    decision["pnl_percent"] = pnl_percent
    decision["outcome_notes"] = outcome_notes
    decision["was_correct"] = was_correct
    decision["status"] = "closed"
    
    # Save updated decision
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    
    print(f"✅ Decision closed: {decision_id}")
    print(f"   Entry: {entry_price} → Exit: {exit_price}")
    print(f"   P&L: {pnl_percent:+.2f}% ({'✓ Correct' if was_correct else '✗ Incorrect'})")
    
    return decision


def list_active_decisions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all active (unclosed) decisions."""
    if not os.path.exists(DECISIONS_DIR):
        return []
    
    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision["status"] == "active":
                    if symbol is None or decision["symbol"] == symbol:
                        decisions.append(decision)
            except Exception:
                continue
    
    return sorted(decisions, key=lambda d: d["created_at"], reverse=True)


def list_closed_decisions(
    symbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List closed decisions for analysis."""
    if not os.path.exists(DECISIONS_DIR):
        return []
    
    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision["status"] == "closed":
                    if symbol is None or decision["symbol"] == symbol:
                        decisions.append(decision)
            except Exception:
                continue
    
    # Sort by exit date, most recent first
    decisions = sorted(decisions, key=lambda d: d.get("exit_date", ""), reverse=True)
    return decisions[:limit]


def get_decision_stats(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics on decision quality."""
    closed = list_closed_decisions(symbol, limit=1000)
    
    if not closed:
        return {
            "total_decisions": 0,
            "correct_rate": 0,
            "avg_pnl_percent": 0,
            "total_pnl": 0,
        }
    
    correct = sum(1 for d in closed if d.get("was_correct"))
    pnl_percents = [d.get("pnl_percent", 0) for d in closed]
    total_pnl = sum(d.get("pnl", 0) for d in closed)
    
    return {
        "total_decisions": len(closed),
        "correct_decisions": correct,
        "correct_rate": correct / len(closed) if closed else 0,
        "avg_pnl_percent": sum(pnl_percents) / len(pnl_percents) if pnl_percents else 0,
        "total_pnl": total_pnl,
        "best_decision": max(closed, key=lambda d: d.get("pnl_percent", 0)) if closed else None,
        "worst_decision": min(closed, key=lambda d: d.get("pnl_percent", 0)) if closed else None,
    }


def link_decision_to_ticket(decision_id: str, mt5_ticket: int):
    """Link a decision to an MT5 ticket after execution."""
    decision = load_decision(decision_id)
    decision["mt5_ticket"] = mt5_ticket
    
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    
    print(f"🔗 Decision {decision_id} linked to MT5 ticket {mt5_ticket}")


def find_decision_by_ticket(mt5_ticket: int) -> Optional[Dict[str, Any]]:
    """Find a decision by its MT5 ticket."""
    if not os.path.exists(DECISIONS_DIR):
        return None
    
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision.get("mt5_ticket") == mt5_ticket:
                    return decision
            except Exception:
                continue
    
    return None


def cancel_decision(decision_id: str, reason: str = ""):
    """Cancel an active decision (e.g., order not filled)."""
    decision = load_decision(decision_id)
    
    if decision["status"] != "active":
        print(f"⚠️ Decision {decision_id} already {decision['status']}")
        return
    
    decision["status"] = "cancelled"
    decision["outcome_notes"] = f"Cancelled: {reason}"
    decision["exit_date"] = datetime.now().isoformat()
    
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    
    print(f"❌ Decision cancelled: {decision_id}")
