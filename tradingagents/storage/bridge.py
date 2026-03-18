"""
Bridge module for gradual migration from trade_decisions.py to storage abstraction.

This provides drop-in replacements for key functions that use the configured storage
backend. Import these instead of the originals to use Postgres when configured.

Example:
    # Instead of:
    from tradingagents.trade_decisions import store_decision, list_active_decisions

    # Use:
    from tradingagents.storage.bridge import store_decision, list_active_decisions
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from tradingagents.storage import get_decision_store


def _get_store():
    """Get the configured decision store."""
    return get_decision_store()


def store_decision(
    symbol: str,
    decision_type: str,
    action: str,
    rationale: str,
    source: str = "analysis",
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    volume: Optional[float] = None,
    mt5_ticket: Optional[int] = None,
    analysis_context: Optional[Dict[str, Any]] = None,
    position_sizing: Optional[Dict[str, Any]] = None,
    status: str = "active",
    execution_error: Optional[str] = None,
) -> str:
    """
    Store a trade decision using the configured storage backend.

    This is a drop-in replacement for tradingagents.trade_decisions.store_decision
    """
    store = _get_store()

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
        "status": status,
        "execution_error": execution_error,
        # Setup classification
        "setup_type": None,
        "higher_tf_bias": None,
        "confluence_score": None,
        "confluence_factors": [],
        # SMC context
        "smc_context": {
            "setup_type": None,
            "entry_zone": None,
            "entry_zone_strength": None,
            "with_trend": None,
            "higher_tf_aligned": None,
            "confluences": [],
            "zone_tested_before": None,
        },
        # Market context
        "volatility_regime": None,
        "market_regime": None,
        "session": None,
        # Outcome fields
        "exit_price": None,
        "exit_date": None,
        "pnl": None,
        "pnl_percent": None,
        "outcome_notes": None,
        "was_correct": None,
        # Exit analysis
        "trailing_stop_atr_multiplier": None,
        "exit_reason": None,
        "rr_planned": None,
        "rr_realized": None,
        # Learning signals
        "reward_signal": None,
        "sharpe_contribution": None,
        "drawdown_impact": None,
        "pattern_tags": [],
        "events": [],
        "has_context": bool(analysis_context),
    }

    if position_sizing:
        decision["position_sizing"] = position_sizing

    # Store decision
    store.store(decision)

    # Store context separately if provided
    if analysis_context:
        store.store_context(decision_id, analysis_context)

    print(f"💾 Decision stored: {decision_id}")
    print(f"   Type: {decision_type} | Action: {action}")
    print(f"   Symbol: {symbol} @ {entry_price or 'market'}")

    return decision_id


def load_decision(decision_id: str) -> Dict[str, Any]:
    """Load a stored decision."""
    return _get_store().load(decision_id)


def load_decision_context(decision_id: str) -> Optional[Dict[str, Any]]:
    """Load the full analysis context for a decision."""
    return _get_store().load_context(decision_id)


def list_active_decisions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all active (unclosed) decisions."""
    return _get_store().list_active(symbol)


def list_closed_decisions(
    symbol: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """List closed decisions."""
    return _get_store().list_closed(symbol, limit)


def find_decision_by_ticket(mt5_ticket: int) -> Optional[Dict[str, Any]]:
    """Find a decision by its MT5 ticket."""
    return _get_store().find_by_ticket(mt5_ticket)


def link_decision_to_ticket(decision_id: str, mt5_ticket: int):
    """Link a decision to an MT5 ticket after execution."""
    store = _get_store()
    store.update(decision_id, {"mt5_ticket": mt5_ticket})
    print(f"🔗 Decision {decision_id} linked to MT5 ticket {mt5_ticket}")


def add_trade_event(
    decision_id: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> None:
    """Append a timestamped event to a trade decision's event log."""
    _get_store().add_event(decision_id, event_type, details, source)


def close_decision(
    decision_id: str,
    exit_price: float,
    outcome_notes: str = "",
    was_correct: Optional[bool] = None,
    exit_reason: Optional[str] = None,
    calculate_reward: bool = True,
    max_favorable_price: Optional[float] = None,
    max_adverse_price: Optional[float] = None,
    held_duration_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Close a decision and record the outcome.

    This delegates to the original implementation for complex outcome analysis,
    but uses the storage abstraction for persistence.
    """
    # Import the original analysis functions
    from tradingagents.trade_decisions import (
        analyze_trade_outcome,
    )

    store = _get_store()
    decision = store.load(decision_id)

    if decision["status"] != "active":
        print(f"[WARN] Decision {decision_id} already {decision['status']}")
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
        pnl = (
            (exit_price - entry_price) * volume * 100
            if action in ["BUY", "LONG"]
            else (entry_price - exit_price) * volume * 100
        )
    else:
        pnl = 0
        pnl_percent = 0

    # Structured outcome analysis
    structured_outcome = None
    if entry_price and entry_price > 0:
        structured_outcome = analyze_trade_outcome(
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=decision.get("stop_loss"),
            take_profit=decision.get("take_profit"),
            direction=action,
            max_favorable_price=max_favorable_price,
            max_adverse_price=max_adverse_price,
            planned_entry=decision.get("planned_entry") or entry_price,
            exit_reason=exit_reason,
            held_duration_hours=held_duration_hours,
        )

        if was_correct is None:
            was_correct = structured_outcome["result"] == "win"
    else:
        if was_correct is None:
            was_correct = pnl_percent > 0

    # Calculate RR
    rr_realized = None
    rr_planned = None

    if decision.get("stop_loss") and entry_price:
        try:
            from tradingagents.learning.reward import RewardCalculator

            direction = "BUY" if action in ["BUY", "LONG"] else "SELL"
            rr_realized = RewardCalculator.calculate_realized_rr(
                entry_price, exit_price, decision["stop_loss"], direction
            )
        except Exception:
            pass

    if decision.get("stop_loss") and decision.get("take_profit") and entry_price:
        if action in ["BUY", "LONG"]:
            risk = abs(entry_price - decision["stop_loss"])
            reward = abs(decision["take_profit"] - entry_price)
        else:
            risk = abs(decision["stop_loss"] - entry_price)
            reward = abs(entry_price - decision["take_profit"])
        rr_planned = reward / risk if risk > 0 else None

    # Build updates
    updates = {
        "exit_price": exit_price,
        "exit_date": datetime.now().isoformat(),
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "outcome_notes": outcome_notes,
        "was_correct": was_correct,
        "status": "closed",
        "exit_reason": exit_reason,
        "rr_planned": rr_planned,
        "rr_realized": rr_realized,
    }

    if structured_outcome:
        updates["structured_outcome"] = structured_outcome
        updates["direction_correct"] = structured_outcome.get("direction_correct")
        updates["sl_placement"] = structured_outcome.get("sl_placement")
        updates["tp_placement"] = structured_outcome.get("tp_placement")
        updates["entry_quality"] = structured_outcome.get("entry_quality")
        updates["outcome_lessons"] = structured_outcome.get("lessons", [])

    # Update via storage
    store.update(decision_id, updates)

    # Reload to get full decision
    decision = store.load(decision_id)

    print(f"[OK] Decision closed: {decision_id}")
    print(f"   Entry: {entry_price} -> Exit: {exit_price}")
    print(f"   P&L: {pnl_percent:+.2f}% ({'Correct' if was_correct else 'Incorrect'})")

    return decision
