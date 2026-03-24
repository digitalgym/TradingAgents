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

Structured Outcome Analysis:
The system analyzes WHY trades won or lost, not just the P&L:
- Direction correct: Did price move in the predicted direction?
- Entry quality: Was the entry well-timed?
- SL placement: Was stop loss too tight, appropriate, or too wide?
- TP placement: Was take profit realistic?
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Database storage toggle - use DB if POSTGRES_URL is set
_USE_DB = bool(os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL"))
_db_store = None


def _get_db_store():
    """Get or create database store singleton."""
    global _db_store
    if _db_store is None and _USE_DB:
        try:
            from .storage.postgres_store import get_decision_store
            _db_store = get_decision_store()
        except Exception as e:
            logger.warning(f"Failed to initialize DB store: {e}")
    return _db_store


# Structured outcome types
OutcomeResult = Literal["win", "loss", "breakeven"]
ExitType = Literal["tp_hit", "sl_hit", "manual_close", "trailing_stop", "time_exit", "unknown"]
EntryQuality = Literal["good", "poor", "neutral"]
EntryTiming = Literal["early", "on_time", "late"]
SlPlacement = Literal["too_tight", "appropriate", "too_wide"]
TpPlacement = Literal["too_ambitious", "appropriate", "too_conservative"]


def analyze_trade_outcome(
    entry_price: float,
    exit_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    direction: str,  # BUY or SELL
    max_favorable_price: Optional[float] = None,
    max_adverse_price: Optional[float] = None,
    planned_entry: Optional[float] = None,
    exit_reason: Optional[str] = None,
    held_duration_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze trade outcome to understand WHY it won or lost.

    This provides structured data for learning, not just P&L.

    Args:
        entry_price: Actual entry price
        exit_price: Actual exit price
        stop_loss: Stop loss level
        take_profit: Take profit level
        direction: "BUY" or "SELL"
        max_favorable_price: Best price reached during trade (highest for BUY, lowest for SELL)
        max_adverse_price: Worst price reached during trade
        planned_entry: Originally planned entry price (for entry quality analysis)
        exit_reason: How the trade exited (tp_hit, sl_hit, manual_close, etc.)
        held_duration_hours: How long the trade was held

    Returns:
        Structured outcome dict with analysis of why the trade won or lost
    """
    is_long = direction.upper() in ["BUY", "LONG"]

    # Basic outcome
    if is_long:
        pnl_pips = exit_price - entry_price
        max_favorable_pips = (max_favorable_price - entry_price) if max_favorable_price else None
        max_adverse_pips = (entry_price - max_adverse_price) if max_adverse_price else None
    else:
        pnl_pips = entry_price - exit_price
        max_favorable_pips = (entry_price - max_favorable_price) if max_favorable_price else None
        max_adverse_pips = (max_adverse_price - entry_price) if max_adverse_price else None

    # Determine result
    if abs(pnl_pips) < 0.0001:  # Essentially breakeven
        result: OutcomeResult = "breakeven"
    elif pnl_pips > 0:
        result = "win"
    else:
        result = "loss"

    # Calculate returns percentage
    returns_pct = (pnl_pips / entry_price) * 100 if entry_price > 0 else 0

    # Direction correct analysis
    # Direction was correct if price moved favorably at some point, even if trade ended in loss
    direction_correct = True
    if max_favorable_pips is not None:
        direction_correct = max_favorable_pips > 0
    elif result == "win":
        direction_correct = True
    elif result == "loss":
        # If we lost and don't have max favorable, assume direction was wrong
        direction_correct = False

    # SL placement analysis
    sl_placement: SlPlacement = "appropriate"
    if stop_loss and result == "loss" and exit_reason in ["sl_hit", "SL"]:
        # SL was hit
        if direction_correct and max_favorable_pips and max_favorable_pips > 0:
            # Direction was right but SL got hit - SL was too tight
            sl_placement = "too_tight"
        else:
            # Direction was wrong - SL was appropriate (saved us from worse loss)
            sl_placement = "appropriate"
    elif stop_loss and result == "win":
        # We won - SL placement was fine
        sl_placement = "appropriate"

    # TP placement analysis
    tp_placement: TpPlacement = "appropriate"
    if take_profit and result == "win" and exit_reason in ["tp_hit", "TP"]:
        # TP was hit - check if we left money on the table
        if max_favorable_pips and max_favorable_price:
            # If price went significantly beyond TP, we were too conservative
            if is_long:
                beyond_tp = max_favorable_price - take_profit
            else:
                beyond_tp = take_profit - max_favorable_price

            if beyond_tp > abs(take_profit - entry_price) * 0.5:
                tp_placement = "too_conservative"
    elif take_profit and result == "loss":
        # Check if TP was too ambitious
        if max_favorable_pips and take_profit:
            if is_long:
                tp_distance = take_profit - entry_price
                achieved_distance = max_favorable_pips
            else:
                tp_distance = entry_price - take_profit
                achieved_distance = max_favorable_pips

            # If we got more than 50% to TP but then reversed, TP might be ok
            # If we never got close to TP, it was too ambitious
            if achieved_distance < tp_distance * 0.3:
                tp_placement = "too_ambitious"

    # Entry quality analysis
    entry_quality: EntryQuality = "neutral"
    entry_timing: EntryTiming = "on_time"

    if planned_entry and entry_price:
        entry_deviation = abs(entry_price - planned_entry) / planned_entry * 100
        if entry_deviation < 0.1:  # Within 0.1%
            entry_quality = "good"
            entry_timing = "on_time"
        elif entry_deviation < 0.5:  # Within 0.5%
            entry_quality = "neutral"
            # Determine if early or late
            if is_long:
                entry_timing = "early" if entry_price > planned_entry else "late"
            else:
                entry_timing = "early" if entry_price < planned_entry else "late"
        else:
            entry_quality = "poor"
            if is_long:
                entry_timing = "early" if entry_price > planned_entry else "late"
            else:
                entry_timing = "early" if entry_price < planned_entry else "late"
    elif result == "win" and max_adverse_pips is not None and max_adverse_pips < abs(pnl_pips) * 0.3:
        # Won without much drawdown - good entry
        entry_quality = "good"
    elif result == "loss" and sl_placement == "too_tight":
        # Lost due to tight SL after going in right direction - entry timing might be issue
        entry_quality = "neutral"  # Can't tell for sure

    # Determine exit type if not provided
    if exit_reason:
        exit_type = exit_reason.lower().replace("-", "_").replace(" ", "_")
        if exit_type not in ["tp_hit", "sl_hit", "manual_close", "trailing_stop", "time_exit"]:
            if "tp" in exit_type or "profit" in exit_type:
                exit_type = "tp_hit"
            elif "sl" in exit_type or "stop" in exit_type:
                exit_type = "sl_hit"
            elif "trail" in exit_type:
                exit_type = "trailing_stop"
            else:
                exit_type = "unknown"
    else:
        # Infer from prices
        if take_profit and abs(exit_price - take_profit) < abs(exit_price - entry_price) * 0.05:
            exit_type = "tp_hit"
        elif stop_loss and abs(exit_price - stop_loss) < abs(exit_price - entry_price) * 0.05:
            exit_type = "sl_hit"
        else:
            exit_type = "manual_close"

    return {
        # Basic outcome
        "result": result,
        "returns_pct": round(returns_pct, 4),
        "pnl_pips": round(pnl_pips, 5),

        # Excursion analysis
        "max_favorable_pips": round(max_favorable_pips, 5) if max_favorable_pips else None,
        "max_adverse_pips": round(max_adverse_pips, 5) if max_adverse_pips else None,

        # Entry analysis
        "entry_quality": entry_quality,
        "entry_timing": entry_timing,
        "entry_vs_plan_pct": round((entry_price - planned_entry) / planned_entry * 100, 4) if planned_entry else None,

        # Exit analysis
        "exit_type": exit_type,

        # Decision quality (separate from outcome!)
        "direction_correct": direction_correct,
        "sl_placement": sl_placement,
        "tp_placement": tp_placement,

        # Context
        "held_duration_hours": held_duration_hours,

        # Lessons to extract
        "lessons": _generate_outcome_lessons(
            result, direction_correct, sl_placement, tp_placement,
            entry_quality, exit_type, max_favorable_pips, pnl_pips
        )
    }


def _generate_outcome_lessons(
    result: str,
    direction_correct: bool,
    sl_placement: str,
    tp_placement: str,
    entry_quality: str,
    exit_type: str,
    max_favorable_pips: Optional[float],
    pnl_pips: float,
) -> List[str]:
    """Generate human-readable lessons from outcome analysis."""
    lessons = []

    if result == "loss":
        if direction_correct and sl_placement == "too_tight":
            lessons.append("Direction was correct but SL was too tight. Consider wider stop or trailing.")
        elif not direction_correct:
            lessons.append("Direction was wrong. Review analysis for signal quality.")
        if entry_quality == "poor":
            lessons.append("Entry quality was poor. Wait for better entry or use limit orders.")

    elif result == "win":
        if tp_placement == "too_conservative" and max_favorable_pips:
            lessons.append(f"TP was conservative. Price went further. Consider trailing or wider TP.")
        if entry_quality == "good":
            lessons.append("Good entry timing contributed to the win.")

    if exit_type == "manual_close" and result == "loss":
        lessons.append("Manual exit resulted in loss. Consider letting SL/TP play out.")

    return lessons


# Directory for storing decisions
DECISIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "trade_decisions")

# Index file for fast ticket lookup
TICKET_INDEX_FILE = os.path.join(DECISIONS_DIR, "_ticket_index.json")


def _load_ticket_index() -> Dict[int, str]:
    """Load the ticket -> decision_id index."""
    if not os.path.exists(TICKET_INDEX_FILE):
        return {}
    try:
        with open(TICKET_INDEX_FILE, "r") as f:
            # Keys are stored as strings in JSON, convert back to int
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    except Exception:
        return {}


def _save_ticket_index(index: Dict[int, str]) -> None:
    """Save the ticket -> decision_id index."""
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    try:
        with open(TICKET_INDEX_FILE, "w") as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in index.items()}, f)
    except Exception:
        pass


def _update_ticket_index(mt5_ticket: int, decision_id: str) -> None:
    """Add or update a ticket in the index."""
    if mt5_ticket is None:
        return
    index = _load_ticket_index()
    index[mt5_ticket] = decision_id
    _save_ticket_index(index)


def _remove_from_ticket_index(mt5_ticket: int) -> None:
    """Remove a ticket from the index (when decision is closed)."""
    if mt5_ticket is None:
        return
    index = _load_ticket_index()
    if mt5_ticket in index:
        del index[mt5_ticket]
        _save_ticket_index(index)


# Active decisions index for fast listing
ACTIVE_INDEX_FILE = os.path.join(DECISIONS_DIR, "_active_index.json")


def _load_active_index() -> Dict[str, Dict[str, Any]]:
    """Load active decisions index: {decision_id: {symbol, created_at}}."""
    if not os.path.exists(ACTIVE_INDEX_FILE):
        return {}
    try:
        with open(ACTIVE_INDEX_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_active_index(index: Dict[str, Dict[str, Any]]) -> None:
    """Save active decisions index."""
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    try:
        with open(ACTIVE_INDEX_FILE, "w") as f:
            json.dump(index, f)
    except Exception:
        pass


def _add_to_active_index(decision_id: str, symbol: str, created_at: str) -> None:
    """Add a decision to the active index."""
    index = _load_active_index()
    index[decision_id] = {"symbol": symbol, "created_at": created_at}
    _save_active_index(index)


def _remove_from_active_index(decision_id: str) -> None:
    """Remove a decision from the active index (when closed)."""
    index = _load_active_index()
    if decision_id in index:
        del index[decision_id]
        _save_active_index(index)


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
    status: str = "active",  # active, failed, closed, cancelled
    execution_error: Optional[str] = None,  # Error message if trade failed
    confidence: Optional[float] = None,  # Signal confidence 0.0-1.0
    pipeline: Optional[str] = None,  # Pipeline that generated the signal
    trailing_stop_atr_multiplier: Optional[float] = None,  # LLM-suggested trailing
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
        "status": status,  # active, failed, closed, cancelled, retried
        "execution_error": execution_error,
        
        # Setup classification
        "setup_type": None,  # "fvg_bounce", "ob_bounce", "liquidity_sweep", "choch", "bos", "trend_continuation"
        "higher_tf_bias": None,  # "bullish", "bearish", "neutral" from H4/D1
        "confluence_score": None,  # 0-10 based on number of confirming factors
        "confluence_factors": [],  # ["ob_fvg_overlap", "pdh_pdl", "session_high", "with_trend"]

        # SMC-specific context (for pattern learning)
        "smc_context": {
            "setup_type": None,  # "fvg_bounce", "ob_bounce", "liquidity_sweep", "choch", "bos"
            "entry_zone": None,  # "fvg", "ob", "liquidity", "market"
            "entry_zone_strength": None,  # 0.0-1.0 strength of the zone
            "with_trend": None,  # True if trade aligned with higher TF trend
            "higher_tf_aligned": None,  # True if H4/D1 structure supports direction
            "confluences": [],  # List of confluence factors present
            "zone_tested_before": None,  # True if zone was previously tested
        },
        
        # Market context
        "volatility_regime": None,  # "low", "normal", "high", "extreme"
        "market_regime": None,  # "trending-up", "trending-down", "ranging", "expansion"
        "session": None,  # "asian", "london", "ny", "overlap"
        
        # Outcome fields (filled when closed)
        "exit_price": None,
        "exit_date": None,
        "pnl": None,
        "pnl_percent": None,
        "outcome_notes": None,
        "was_correct": None,  # True if decision led to profit or avoided loss
        
        # Exit analysis (filled when closed)
        "confidence": confidence,  # Signal confidence 0.0-1.0
        "pipeline": pipeline,  # Pipeline that generated the signal
        "trailing_stop_atr_multiplier": trailing_stop_atr_multiplier,  # LLM-suggested trailing stop distance as ATR multiple
        "exit_reason": None,  # "tp-hit", "sl-hit", "manual", "trailing-stop", "time-exit"
        "rr_planned": None,  # Planned risk-reward ratio
        "rr_realized": None,  # Actual risk-reward achieved
        
        # Learning signals (filled when closed)
        "reward_signal": None,  # Calculated reward for RL
        "sharpe_contribution": None,  # Impact on portfolio Sharpe
        "drawdown_impact": None,  # Contribution to drawdown
        "pattern_tags": [],  # Auto-generated tags for pattern clustering

        # Trade event log — timestamped audit trail of every action on this trade
        "events": [],
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
    
    # Save to database first (primary storage)
    db_store = _get_db_store()
    if db_store:
        try:
            db_store.store(decision)
            if analysis_context:
                db_store.store_context(decision_id, analysis_context)
            logger.info(f"Decision stored in DB: {decision_id}")
        except Exception as e:
            logger.error(f"Failed to store decision in DB: {e}")
            # Continue to file storage as fallback

    # Save decision as JSON for easy viewing (backup/local access)
    try:
        decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
        with open(decision_file, "w") as f:
            json.dump(decision, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to write decision file: {e}")
        # If DB write succeeded, this is OK

    # Update ticket index for fast lookup
    if mt5_ticket:
        _update_ticket_index(mt5_ticket, decision_id)

    # Add to active index if status is active
    if status == "active":
        _add_to_active_index(decision_id, symbol, decision["created_at"])

    print(f"💾 Decision stored: {decision_id}")
    print(f"   Type: {decision_type} | Action: {action}")
    print(f"   Symbol: {symbol} @ {entry_price or 'market'}")

    return decision_id


def load_decision(decision_id: str) -> Dict[str, Any]:
    """Load a stored decision. Reads from DB first, falls back to file."""
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            return db_store.load(decision_id)
        except KeyError:
            pass  # Not in DB, try file
        except Exception as e:
            logger.warning(f"DB load failed, trying file: {e}")

    # Fall back to file
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")

    if not os.path.exists(decision_file):
        raise FileNotFoundError(f"Decision not found: {decision_id}")

    with open(decision_file, "r") as f:
        return json.load(f)


def load_decision_context(decision_id: str) -> Optional[Dict[str, Any]]:
    """Load the full analysis context for a decision."""
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            result = db_store.load_context(decision_id)
            if result:
                return result
        except Exception as e:
            logger.warning(f"DB load_context failed: {e}")

    # Fall back to file
    context_file = os.path.join(DECISIONS_DIR, f"{decision_id}_context.pkl")

    if not os.path.exists(context_file):
        return None

    with open(context_file, "rb") as f:
        return pickle.load(f)


def add_trade_event(
    decision_id: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> None:
    """Append a timestamped event to a trade decision's event log.

    Event types:
        executed        - Trade placed in MT5
        breakeven_set   - SL moved to breakeven
        trailing_stop   - SL trailed to new level
        reversal_signal - Reversal detected (may or may not close)
        closed          - Position closed (by automation, MT5, or externally)
        sl_hit          - Stop loss hit
        tp_hit          - Take profit hit
        external_close  - Position disappeared from MT5
        modify_sl       - SL modified
        modify_tp       - TP modified
        error           - Something went wrong
    """
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            db_store.add_event(decision_id, event_type, details, source)
            return  # DB handles both event table and decision JSON update
        except Exception as e:
            logger.warning(f"DB add_event failed: {e}")

    # Fall back to file
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    if not os.path.exists(decision_file):
        return

    try:
        with open(decision_file, "r") as f:
            decision = json.load(f)

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

        with open(decision_file, "w") as f:
            json.dump(decision, f, indent=2, default=str)
    except Exception:
        pass  # Don't let event logging break trade flow


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
    Close a decision and record the outcome with structured analysis.

    Args:
        decision_id: The decision to close
        exit_price: The exit price
        outcome_notes: Notes about what happened
        was_correct: Override auto-calculation of correctness
        exit_reason: How the trade exited (tp_hit, sl_hit, manual_close, etc.)
        calculate_reward: Whether to calculate reward signals
        max_favorable_price: Best price reached during trade (for excursion analysis)
        max_adverse_price: Worst price reached during trade (for excursion analysis)
        held_duration_hours: How long the trade was held

    Returns:
        Updated decision with structured outcome analysis
    """
    decision = load_decision(decision_id)

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
        # Rough P&L calculation (actual depends on contract size)
        pnl = (exit_price - entry_price) * volume * 100 if action in ["BUY", "LONG"] else (entry_price - exit_price) * volume * 100
    else:
        pnl = 0
        pnl_percent = 0

    # === STRUCTURED OUTCOME ANALYSIS ===
    # This tells us WHY the trade won or lost, not just the P&L
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
            planned_entry=decision.get("planned_entry") or entry_price,  # Use entry as planned if not specified
            exit_reason=exit_reason,
            held_duration_hours=held_duration_hours,
        )

        # Use structured outcome for was_correct if not overridden
        if was_correct is None:
            was_correct = structured_outcome["result"] == "win"
    else:
        if was_correct is None:
            was_correct = pnl_percent > 0

    # Calculate realized RR if we have stop loss
    rr_realized = None
    if decision.get("stop_loss") and entry_price:
        try:
            from tradingagents.learning.reward import RewardCalculator
            direction = "BUY" if action in ["BUY", "LONG"] else "SELL"
            rr_realized = RewardCalculator.calculate_realized_rr(
                entry_price, exit_price, decision["stop_loss"], direction
            )
        except Exception:
            pass

    # Calculate planned RR if we have SL and TP
    rr_planned = None
    if decision.get("stop_loss") and decision.get("take_profit") and entry_price:
        if action in ["BUY", "LONG"]:
            risk = abs(entry_price - decision["stop_loss"])
            reward = abs(decision["take_profit"] - entry_price)
        else:
            risk = abs(decision["stop_loss"] - entry_price)
            reward = abs(entry_price - decision["take_profit"])
        rr_planned = reward / risk if risk > 0 else None

    # Calculate reward signal if enabled
    reward_signal = None
    sharpe_contribution = None
    drawdown_impact = None

    if calculate_reward and rr_realized is not None:
        try:
            from tradingagents.learning.portfolio_state import PortfolioStateTracker
            from tradingagents.learning.reward import RewardCalculator

            # Load portfolio state
            portfolio = PortfolioStateTracker.load_state()

            # Calculate reward components
            reward_components = RewardCalculator.calculate_all_components(
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=decision["stop_loss"],
                direction="BUY" if action in ["BUY", "LONG"] else "SELL",
                trade_pnl=pnl,
                portfolio_returns=portfolio.returns[-50:] if len(portfolio.returns) > 0 else [],
                equity_curve=portfolio.equity_curve,
                peak_equity=portfolio.peak_equity,
                position_size_pct=0.01  # Default 1% risk
            )

            reward_signal = reward_components["reward"]
            sharpe_contribution = reward_components["sharpe_contribution"]
            drawdown_impact = reward_components["drawdown_impact"]

            # Update portfolio state
            portfolio.update(pnl, win=was_correct)
            portfolio.save_state()

        except Exception as e:
            print(f"[WARN] Could not calculate reward signal: {e}")

    # Update decision with basic outcome
    decision["exit_price"] = exit_price
    decision["exit_date"] = datetime.now().isoformat()
    decision["pnl"] = pnl
    decision["pnl_percent"] = pnl_percent
    decision["outcome_notes"] = outcome_notes
    decision["was_correct"] = was_correct
    # Use distinct status for orders that were placed but never filled
    if exit_reason == "order_unfilled":
        decision["status"] = "order_unfilled"
    else:
        decision["status"] = "closed"
    decision["exit_reason"] = exit_reason
    decision["rr_planned"] = rr_planned
    decision["rr_realized"] = rr_realized
    decision["reward_signal"] = reward_signal
    decision["sharpe_contribution"] = sharpe_contribution
    decision["drawdown_impact"] = drawdown_impact

    # === Store structured outcome analysis ===
    # This is the key data for learning WHY trades succeed or fail
    if structured_outcome:
        decision["structured_outcome"] = structured_outcome
        # Also store key fields at top level for easy access
        decision["direction_correct"] = structured_outcome.get("direction_correct")
        decision["sl_placement"] = structured_outcome.get("sl_placement")
        decision["tp_placement"] = structured_outcome.get("tp_placement")
        decision["entry_quality"] = structured_outcome.get("entry_quality")
        decision["outcome_lessons"] = structured_outcome.get("lessons", [])

    # Save to database first (primary storage)
    db_store = _get_db_store()
    if db_store:
        try:
            db_store.update(decision_id, decision)
            logger.info(f"Decision closed in DB: {decision_id}")
        except Exception as e:
            logger.error(f"Failed to update decision in DB: {e}")

    # Save updated decision to file (backup)
    try:
        decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
        with open(decision_file, "w") as f:
            json.dump(decision, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to write decision file: {e}")

    # Remove from active index
    _remove_from_active_index(decision_id)

    print(f"[OK] Decision closed: {decision_id}")
    print(f"   Entry: {entry_price} -> Exit: {exit_price}")
    print(f"   P&L: {pnl_percent:+.2f}% ({'Correct' if was_correct else 'Incorrect'})")
    if rr_realized is not None:
        print(f"   Risk-Reward: {rr_realized:+.2f}R (planned: {rr_planned:.2f}R)" if rr_planned else f"   Risk-Reward: {rr_realized:+.2f}R")

    # Print structured outcome insights
    if structured_outcome:
        print(f"   Direction: {'✓ Correct' if structured_outcome.get('direction_correct') else '✗ Wrong'}")
        print(f"   SL: {structured_outcome.get('sl_placement')} | TP: {structured_outcome.get('tp_placement')}")
        if structured_outcome.get("lessons"):
            print(f"   Lessons: {'; '.join(structured_outcome['lessons'][:2])}")

    if reward_signal is not None:
        print(f"   Reward Signal: {reward_signal:+.2f}")

    return decision


def list_decisions(symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List all decisions regardless of status. Reads from DB first."""
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            return db_store.list_all(symbol=symbol, limit=limit)
        except Exception as e:
            logger.warning(f"DB list failed, falling back to files: {e}")

    # Fall back to files
    if not os.path.exists(DECISIONS_DIR):
        return []

    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json") and not f.startswith("_"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if symbol is None or decision.get("symbol") == symbol:
                    decisions.append(decision)
            except Exception:
                continue

    return sorted(decisions, key=lambda d: d.get("created_at", ""), reverse=True)[:limit]


def list_active_decisions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all active (unclosed) decisions.

    Uses DB first, then index for faster lookup. Falls back to full scan if
    index is missing or inconsistent (handles legacy data).
    """
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            return db_store.list_active(symbol=symbol)
        except Exception as e:
            logger.warning(f"DB list_active failed, falling back to files: {e}")

    if not os.path.exists(DECISIONS_DIR):
        return []

    # Try fast index lookup
    index = _load_active_index()
    if index:
        decisions = []
        stale_ids = []
        for decision_id, meta in index.items():
            # Filter by symbol if specified
            if symbol is not None and meta.get("symbol") != symbol:
                continue
            try:
                decision = load_decision(decision_id)
                # Verify still active (index could be stale)
                if decision["status"] == "active":
                    decisions.append(decision)
                else:
                    stale_ids.append(decision_id)
            except FileNotFoundError:
                stale_ids.append(decision_id)
            except Exception:
                continue

        # Clean up stale entries
        if stale_ids:
            for stale_id in stale_ids:
                _remove_from_active_index(stale_id)

        if decisions:
            return sorted(decisions, key=lambda d: d["created_at"], reverse=True)

    # Fallback: scan all files (handles pre-index data, rebuilds index)
    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json") and not f.startswith("_"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision["status"] == "active":
                    # Rebuild index entry
                    _add_to_active_index(
                        decision_id,
                        decision["symbol"],
                        decision["created_at"],
                    )
                    if symbol is None or decision["symbol"] == symbol:
                        decisions.append(decision)
            except Exception:
                continue

    return sorted(decisions, key=lambda d: d["created_at"], reverse=True)


def cleanup_stale_decisions(dry_run: bool = True) -> List[Dict[str, Any]]:
    """Find and close active decisions whose MT5 positions no longer exist.

    When a position is closed externally (SL/TP hit, manual close) but the
    automation doesn't detect it, the decision stays 'active' forever.
    This function reconciles by checking MT5 for each active decision's ticket.

    Args:
        dry_run: If True, only report stale decisions without closing them.

    Returns:
        List of stale decisions that were found (and closed if dry_run=False).
    """
    try:
        from tradingagents.dataflows.mt5_data import get_open_positions, get_closed_deal_by_ticket
    except ImportError:
        return []

    active = list_active_decisions()
    if not active:
        return []

    # Get currently open MT5 tickets
    try:
        positions = get_open_positions()
    except Exception:
        return []

    open_tickets = {p.get("ticket") for p in positions}

    stale = []
    for dec in active:
        ticket = dec.get("mt5_ticket")
        if not ticket:
            continue  # No ticket = never executed, not stale

        if ticket in open_tickets:
            continue  # Position still open, not stale

        # Position is gone from MT5 — this decision is stale
        stale.append(dec)

        if not dry_run:
            # Try to get exit info from MT5 history
            exit_price = dec.get("entry_price", 0)
            pnl = 0.0
            exit_reason = "stale_cleanup"
            try:
                deal = get_closed_deal_by_ticket(ticket, days_back=14)
                if deal:
                    exit_price = deal.get("price", exit_price)
                    pnl = deal.get("profit", 0)
                    exit_reason = "sl_hit" if pnl < 0 else "tp_hit"
            except Exception:
                pass

            try:
                close_decision(
                    dec["decision_id"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    outcome_notes=f"Auto-cleaned: MT5 position #{ticket} no longer open (pnl={pnl:.2f})",
                )
            except Exception as e:
                print(f"Failed to close stale decision {dec['decision_id']}: {e}")

    return stale


def list_failed_decisions(symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """List failed trade decisions."""
    if not os.path.exists(DECISIONS_DIR):
        return []

    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision["status"] in ("failed", "retried", "order_unfilled"):
                    if symbol is None or decision["symbol"] == symbol:
                        decisions.append(decision)
            except Exception:
                continue

    return sorted(decisions, key=lambda d: d.get("created_at", ""), reverse=True)[:limit]


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
    # Update in DB first
    db_store = _get_db_store()
    if db_store:
        try:
            db_store.link_ticket(decision_id, mt5_ticket)
            logger.info(f"Decision {decision_id} linked to ticket {mt5_ticket} in DB")
        except Exception as e:
            logger.error(f"Failed to link ticket in DB: {e}")

    # Update file (backup)
    try:
        decision = load_decision(decision_id)
        decision["mt5_ticket"] = mt5_ticket

        decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
        with open(decision_file, "w") as f:
            json.dump(decision, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to update decision file: {e}")

    # Update ticket index for fast lookup
    _update_ticket_index(mt5_ticket, decision_id)

    print(f"🔗 Decision {decision_id} linked to MT5 ticket {mt5_ticket}")


def find_decision_by_ticket(mt5_ticket: int) -> Optional[Dict[str, Any]]:
    """Find a decision by its MT5 ticket.

    Uses DB first, then ticket index for O(1) lookup.
    Falls back to full scan if index miss (handles legacy data).
    """
    # Try DB first
    db_store = _get_db_store()
    if db_store:
        try:
            result = db_store.find_by_ticket(mt5_ticket)
            if result:
                return result
        except Exception as e:
            logger.warning(f"DB find_by_ticket failed: {e}")

    if not os.path.exists(DECISIONS_DIR):
        return None

    # Try fast index lookup first
    index = _load_ticket_index()
    if mt5_ticket in index:
        decision_id = index[mt5_ticket]
        try:
            return load_decision(decision_id)
        except FileNotFoundError:
            # Index stale, remove entry and fall through to scan
            _remove_from_ticket_index(mt5_ticket)

    # Fallback: scan all files (handles pre-index decisions)
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json") and not f.startswith("_"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision.get("mt5_ticket") == mt5_ticket:
                    # Update index for future lookups
                    _update_ticket_index(mt5_ticket, decision_id)
                    return decision
            except Exception:
                continue

    return None


def mark_decision_reviewed(decision_id: str):
    """Mark a decision as reviewed."""
    decision = load_decision(decision_id)
    decision["reviewed"] = True
    decision["reviewed_at"] = datetime.now().isoformat()
    
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)


def list_unreviewed_decisions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """List closed decisions that haven't been reviewed yet."""
    if not os.path.exists(DECISIONS_DIR):
        return []
    
    decisions = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".json"):
            decision_id = f.replace(".json", "")
            try:
                decision = load_decision(decision_id)
                if decision["status"] == "closed" and not decision.get("reviewed", False):
                    if symbol is None or decision["symbol"] == symbol:
                        decisions.append(decision)
            except Exception:
                continue
    
    return sorted(decisions, key=lambda d: d.get("exit_date", ""), reverse=True)


def group_decisions_by_symbol(decisions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group decisions by symbol."""
    grouped = {}
    for decision in decisions:
        symbol = decision["symbol"]
        if symbol not in grouped:
            grouped[symbol] = []
        grouped[symbol].append(decision)
    return grouped


def set_decision_regime(decision_id: str, regime: Dict[str, str]):
    """
    Set regime context for a decision.
    
    Args:
        decision_id: The decision to update
        regime: Regime dict with market_regime, volatility_regime, expansion_regime
    """
    decision = load_decision(decision_id)
    
    decision["market_regime"] = regime.get("market_regime")
    decision["volatility_regime"] = regime.get("volatility_regime")
    decision["expansion_regime"] = regime.get("expansion_regime")
    
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)



def cancel_decision(decision_id: str, reason: str = ""):
    """Cancel an active decision (e.g., order not filled)."""
    decision = load_decision(decision_id)

    if decision["status"] != "active":
        print(f"[WARN] Decision {decision_id} already {decision['status']}")
        return

    decision["status"] = "cancelled"
    decision["outcome_notes"] = f"Cancelled: {reason}"
    decision["exit_date"] = datetime.now().isoformat()

    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)

    print(f"❌ Decision cancelled: {decision_id}")


def set_smc_context(
    decision_id: str,
    setup_type: Optional[str] = None,  # "fvg_bounce", "ob_bounce", "liquidity_sweep", "choch", "bos"
    entry_zone: Optional[str] = None,  # "fvg", "ob", "liquidity", "market"
    entry_zone_strength: Optional[float] = None,  # 0.0-1.0 strength of the zone
    with_trend: Optional[bool] = None,  # True if trade aligned with higher TF trend
    higher_tf_aligned: Optional[bool] = None,  # True if H4/D1 structure supports direction
    confluences: Optional[List[str]] = None,  # ["ob_fvg_overlap", "pdh_pdl", "session_high", "with_trend"]
    zone_tested_before: Optional[bool] = None,  # True if zone was previously tested
):
    """
    Set SMC context for a decision.

    This context is critical for learning which SMC setups work best.

    Args:
        decision_id: The decision to update
        setup_type: Type of SMC setup (fvg_bounce, ob_bounce, liquidity_sweep, choch, bos)
        entry_zone: Where entry was placed (fvg, ob, liquidity, market)
        entry_zone_strength: Strength of the zone (0.0-1.0)
        with_trend: Whether trade aligned with higher TF trend
        higher_tf_aligned: Whether H4/D1 structure supports direction
        confluences: List of confluence factors present
        zone_tested_before: Whether zone was previously tested

    Example:
        set_smc_context(
            decision_id,
            setup_type="fvg_bounce",
            entry_zone="fvg",
            entry_zone_strength=0.85,
            with_trend=True,
            higher_tf_aligned=True,
            confluences=["ob_fvg_overlap", "pdh_pdl"],
            zone_tested_before=False
        )
    """
    decision = load_decision(decision_id)

    # Initialize smc_context if not present
    if "smc_context" not in decision:
        decision["smc_context"] = {}

    # Update only provided fields
    if setup_type is not None:
        decision["smc_context"]["setup_type"] = setup_type
        decision["setup_type"] = setup_type  # Also set top-level for quick access
    if entry_zone is not None:
        decision["smc_context"]["entry_zone"] = entry_zone
    if entry_zone_strength is not None:
        decision["smc_context"]["entry_zone_strength"] = entry_zone_strength
    if with_trend is not None:
        decision["smc_context"]["with_trend"] = with_trend
    if higher_tf_aligned is not None:
        decision["smc_context"]["higher_tf_aligned"] = higher_tf_aligned
    if confluences is not None:
        decision["smc_context"]["confluences"] = confluences
        decision["confluence_factors"] = confluences  # Also set top-level
    if zone_tested_before is not None:
        decision["smc_context"]["zone_tested_before"] = zone_tested_before

    # Calculate confluence score based on factors
    if confluences:
        decision["confluence_score"] = len(confluences)

    # Save updated decision
    decision_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)

    print(f"📊 SMC context set for {decision_id}: {setup_type or 'N/A'} @ {entry_zone or 'N/A'}")


def get_smc_pattern_stats(
    symbol: Optional[str] = None,
    setup_type: Optional[str] = None,
    min_samples: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Get win rate statistics for SMC patterns.

    This helps learn which setups work best for which symbols.

    Args:
        symbol: Filter by symbol (None for all)
        setup_type: Filter by setup type (None for all)
        min_samples: Minimum trades to include in stats

    Returns:
        Dict of setup_type -> {win_rate, sample_size, avg_returns, best_confluences}
    """
    closed = list_closed_decisions(symbol, limit=1000)

    # Group by setup type
    by_setup: Dict[str, List[Dict]] = {}
    for d in closed:
        smc = d.get("smc_context", {})
        st = smc.get("setup_type") or d.get("setup_type")
        if st:
            if setup_type and st != setup_type:
                continue
            if st not in by_setup:
                by_setup[st] = []
            by_setup[st].append(d)

    stats = {}
    for st, trades in by_setup.items():
        if len(trades) < min_samples:
            continue

        wins = sum(1 for t in trades if t.get("was_correct"))
        pnl_pcts = [t.get("pnl_percent", 0) for t in trades]

        # Find best confluences
        confluence_wins: Dict[str, List[bool]] = {}
        for t in trades:
            smc = t.get("smc_context", {})
            confs = smc.get("confluences", []) or t.get("confluence_factors", [])
            for c in confs:
                if c not in confluence_wins:
                    confluence_wins[c] = []
                confluence_wins[c].append(t.get("was_correct", False))

        best_confluences = []
        for c, results in confluence_wins.items():
            if len(results) >= 2:  # Need at least 2 samples
                win_rate = sum(results) / len(results)
                if win_rate >= 0.5:  # Only include positive confluences
                    best_confluences.append({
                        "confluence": c,
                        "win_rate": win_rate,
                        "samples": len(results)
                    })
        best_confluences.sort(key=lambda x: x["win_rate"], reverse=True)

        # With-trend vs counter-trend analysis
        with_trend_trades = [t for t in trades if t.get("smc_context", {}).get("with_trend")]
        counter_trend_trades = [t for t in trades if t.get("smc_context", {}).get("with_trend") == False]

        stats[st] = {
            "win_rate": wins / len(trades),
            "sample_size": len(trades),
            "avg_returns_pct": sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0,
            "best_confluences": best_confluences[:5],  # Top 5
            "with_trend_win_rate": sum(1 for t in with_trend_trades if t.get("was_correct")) / len(with_trend_trades) if with_trend_trades else None,
            "counter_trend_win_rate": sum(1 for t in counter_trend_trades if t.get("was_correct")) / len(counter_trend_trades) if counter_trend_trades else None,
        }

    return stats


def get_structured_outcome_stats(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics on structured outcomes to understand systematic issues.

    This helps identify patterns like "SL too tight" or "direction often correct but timing poor".
    """
    closed = list_closed_decisions(symbol, limit=1000)

    # Filter to those with structured outcomes
    with_outcomes = [d for d in closed if d.get("structured_outcome")]

    if not with_outcomes:
        return {"message": "No trades with structured outcome analysis yet"}

    # Direction analysis
    direction_correct = sum(1 for d in with_outcomes if d.get("direction_correct"))
    direction_wrong = len(with_outcomes) - direction_correct

    # SL placement analysis
    sl_too_tight = sum(1 for d in with_outcomes if d.get("sl_placement") == "too_tight")
    sl_appropriate = sum(1 for d in with_outcomes if d.get("sl_placement") == "appropriate")
    sl_too_wide = sum(1 for d in with_outcomes if d.get("sl_placement") == "too_wide")

    # TP placement analysis
    tp_too_ambitious = sum(1 for d in with_outcomes if d.get("tp_placement") == "too_ambitious")
    tp_appropriate = sum(1 for d in with_outcomes if d.get("tp_placement") == "appropriate")
    tp_too_conservative = sum(1 for d in with_outcomes if d.get("tp_placement") == "too_conservative")

    # Entry quality
    entry_good = sum(1 for d in with_outcomes if d.get("entry_quality") == "good")
    entry_neutral = sum(1 for d in with_outcomes if d.get("entry_quality") == "neutral")
    entry_poor = sum(1 for d in with_outcomes if d.get("entry_quality") == "poor")

    # Lessons extracted
    all_lessons = []
    for d in with_outcomes:
        all_lessons.extend(d.get("outcome_lessons", []))

    # Count lesson frequency
    lesson_counts: Dict[str, int] = {}
    for lesson in all_lessons:
        # Normalize lesson text
        key = lesson[:50]  # First 50 chars as key
        lesson_counts[key] = lesson_counts.get(key, 0) + 1

    top_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_analyzed": len(with_outcomes),
        "direction_analysis": {
            "correct": direction_correct,
            "wrong": direction_wrong,
            "accuracy": direction_correct / len(with_outcomes) if with_outcomes else 0,
        },
        "sl_placement": {
            "too_tight": sl_too_tight,
            "appropriate": sl_appropriate,
            "too_wide": sl_too_wide,
            "too_tight_pct": sl_too_tight / len(with_outcomes) if with_outcomes else 0,
        },
        "tp_placement": {
            "too_ambitious": tp_too_ambitious,
            "appropriate": tp_appropriate,
            "too_conservative": tp_too_conservative,
        },
        "entry_quality": {
            "good": entry_good,
            "neutral": entry_neutral,
            "poor": entry_poor,
        },
        "top_lessons": top_lessons,
        "key_insight": _generate_key_insight(
            direction_correct / len(with_outcomes) if with_outcomes else 0,
            sl_too_tight / len(with_outcomes) if with_outcomes else 0,
            tp_too_ambitious / len(with_outcomes) if with_outcomes else 0,
        )
    }


def _generate_key_insight(direction_accuracy: float, sl_tight_pct: float, tp_ambitious_pct: float) -> str:
    """Generate a key insight from the stats."""
    insights = []

    if direction_accuracy >= 0.6 and sl_tight_pct >= 0.3:
        insights.append("Direction is often correct but SL is too tight. Consider wider stops or trailing.")

    if direction_accuracy < 0.5:
        insights.append("Direction prediction needs improvement. Review signal quality and regime alignment.")

    if tp_ambitious_pct >= 0.3:
        insights.append("TP is often too ambitious. Consider more conservative targets or scaling out.")

    if direction_accuracy >= 0.65 and sl_tight_pct < 0.2:
        insights.append("Good direction accuracy with appropriate stops. Focus on TP optimization.")

    return " ".join(insights) if insights else "Not enough data for key insights yet."


def _get_entry_price_from_history(ticket: int, days_back: int = 30) -> Optional[float]:
    """Get the entry price for a position from MT5 deal history (opening deal)."""
    try:
        import MetaTrader5 as mt5
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=days_back)
        to_date = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return None
        for deal in deals:
            if deal.position_id == ticket and deal.entry == 0:  # entry=0 means opening deal
                return deal.price
    except Exception:
        pass
    return None


def reconcile_decisions(days_back: int = 14) -> List[Dict[str, Any]]:
    """
    Reconcile active decisions against MT5 closed positions.
    Finds decisions with mt5_ticket whose positions are no longer open and closes them.
    Returns list of decisions that were reconciled.
    """
    from tradingagents.dataflows.mt5_data import get_open_positions, get_closed_deal_by_ticket

    active = list_active_decisions()
    if not active:
        return []

    # Get all currently open tickets
    try:
        positions = get_open_positions()
        open_tickets = {p.get("ticket") for p in positions}
    except Exception:
        return []  # Can't reconcile without MT5

    reconciled = []
    for dec in active:
        ticket = dec.get("mt5_ticket")
        if not ticket or ticket in open_tickets:
            continue  # No ticket or still open

        # Position is gone — try to get exit info from MT5 deal history
        deal_info = get_closed_deal_by_ticket(ticket, days_back=days_back)
        if not deal_info:
            continue

        exit_price = deal_info["price"]
        profit = deal_info.get("profit", 0)
        entry = dec.get("entry_price", 0)
        sl = dec.get("stop_loss")
        tp = dec.get("take_profit")

        # Backfill entry_price if stored as 0
        if not entry:
            entry = _get_entry_price_from_history(ticket, days_back)
            if entry:
                dec["entry_price"] = entry
                # Persist fix to decision file
                dec_file = os.path.join(DECISIONS_DIR, f"{dec['decision_id']}.json")
                if os.path.exists(dec_file):
                    with open(dec_file, "r") as f:
                        data = json.load(f)
                    data["entry_price"] = entry
                    with open(dec_file, "w") as f:
                        json.dump(data, f, indent=2, default=str)

        # Infer exit reason
        tolerance = abs(entry * 0.0005) if entry else 0.5
        if tp and abs(exit_price - tp) <= tolerance:
            exit_reason = "tp_hit"
        elif sl and abs(exit_price - sl) <= tolerance:
            exit_reason = "sl_hit"
        else:
            exit_reason = "unknown"

        try:
            result = close_decision(
                dec["decision_id"],
                exit_price=exit_price,
                outcome_notes=f"Reconciled from MT5 history. Profit: {profit:.2f}",
                exit_reason=exit_reason,
            )
            reconciled.append(result)
        except Exception:
            continue

    # Also backfill entry_price for already-closed decisions that have entry_price=0
    _backfill_entry_prices(days_back)

    return reconciled


def _backfill_entry_prices(days_back: int = 30) -> int:
    """Fix closed decisions that have entry_price=0 by looking up MT5 deal history."""
    if not os.path.exists(DECISIONS_DIR):
        return 0
    fixed = 0
    for f in os.listdir(DECISIONS_DIR):
        if not f.endswith(".json"):
            continue
        filepath = os.path.join(DECISIONS_DIR, f)
        try:
            with open(filepath, "r") as fh:
                dec = json.load(fh)
            if dec.get("entry_price") or not dec.get("mt5_ticket"):
                continue  # Already has entry price or no ticket
            ticket = dec["mt5_ticket"]
            entry = _get_entry_price_from_history(ticket, days_back)
            if entry:
                dec["entry_price"] = entry
                # Recalculate P/L if we have exit_price
                exit_price = dec.get("exit_price")
                action = dec.get("action", "").upper()
                if exit_price and exit_price > 0:
                    volume = dec.get("volume", 0.01)
                    if action in ["BUY", "LONG"]:
                        dec["pnl"] = (exit_price - entry) * volume * 100
                        dec["pnl_percent"] = ((exit_price - entry) / entry) * 100
                    elif action in ["SELL", "SHORT"]:
                        dec["pnl"] = (entry - exit_price) * volume * 100
                        dec["pnl_percent"] = ((entry - exit_price) / entry) * 100
                with open(filepath, "w") as fh:
                    json.dump(dec, fh, indent=2, default=str)
                fixed += 1
        except Exception:
            continue
    return fixed


def get_trade_memories(symbol: str, limit: int = 10) -> str:
    """
    Build a concise text summary of past trade outcomes for a symbol,
    suitable for injection into the LLM analysis prompt.

    Returns empty string if no closed trades exist for the symbol.
    """
    closed = list_closed_decisions(symbol, limit=limit)
    if not closed:
        return ""

    # --- Aggregate stats ---
    total = len(closed)
    wins = sum(1 for d in closed if d.get("pnl", 0) > 0)
    losses = total - wins
    win_rate = wins / total if total else 0
    total_pnl = sum(d.get("pnl", 0) for d in closed)
    avg_pnl = total_pnl / total if total else 0

    # Direction accuracy (from structured outcomes)
    with_direction = [d for d in closed if d.get("direction_correct") is not None]
    dir_correct = sum(1 for d in with_direction if d["direction_correct"])
    dir_accuracy = dir_correct / len(with_direction) if with_direction else None

    # SL/TP placement stats
    sl_stats = {}
    tp_stats = {}
    for d in closed:
        sl_p = d.get("sl_placement")
        tp_p = d.get("tp_placement")
        if sl_p:
            sl_stats[sl_p] = sl_stats.get(sl_p, 0) + 1
        if tp_p:
            tp_stats[tp_p] = tp_stats.get(tp_p, 0) + 1

    # --- Build text ---
    lines = [f"## LESSONS FROM YOUR PAST TRADES ({symbol})", ""]
    lines.append(f"### Performance (last {total} trades)")
    lines.append(f"- Win rate: {win_rate:.0%} ({wins}/{total}) | Total P/L: ${total_pnl:.2f} | Avg: ${avg_pnl:.2f}")

    if dir_accuracy is not None:
        lines.append(f"- Direction accuracy: {dir_accuracy:.0%} ({dir_correct}/{len(with_direction)})"
                      + (" — you often get direction right but lose on execution" if dir_accuracy > 0.5 and win_rate < 0.5 else ""))

    # SL placement issues
    if sl_stats:
        dominant_sl = max(sl_stats, key=sl_stats.get)
        sl_pct = sl_stats[dominant_sl] / sum(sl_stats.values())
        if dominant_sl == "too_tight" and sl_pct >= 0.3:
            lines.append(f"- SL placement: {sl_pct:.0%} too tight — stops get hit before the move plays out")
        elif dominant_sl == "too_wide" and sl_pct >= 0.3:
            lines.append(f"- SL placement: {sl_pct:.0%} too wide — giving back too much on losers")

    # TP placement issues
    if tp_stats:
        dominant_tp = max(tp_stats, key=tp_stats.get)
        tp_pct = tp_stats[dominant_tp] / sum(tp_stats.values())
        if dominant_tp == "too_ambitious" and tp_pct >= 0.3:
            lines.append(f"- TP placement: {tp_pct:.0%} too ambitious — price reverses before reaching target")
        elif dominant_tp == "too_conservative" and tp_pct >= 0.3:
            lines.append(f"- TP placement: {tp_pct:.0%} too conservative — leaving profit on the table")

    # Key insight
    outcome_stats = get_structured_outcome_stats(symbol)
    if isinstance(outcome_stats, dict) and outcome_stats.get("key_insight"):
        lines.append("")
        lines.append(f"### Key Insight")
        lines.append(outcome_stats["key_insight"])

    # Recent trade details (last 5)
    lines.append("")
    lines.append("### Recent Outcomes")
    tight_sl_count = 0
    tight_tp_count = 0
    for i, d in enumerate(closed[:5], 1):
        action = d.get("action", "?")
        entry = d.get("entry_price", 0)
        exit_p = d.get("exit_price", 0)
        sl = d.get("stop_loss", 0)
        tp = d.get("take_profit", 0)
        pnl = d.get("pnl", 0)
        reason = d.get("exit_reason", "unknown")

        # Format exit reason
        reason_text = {
            "tp_hit": "TP hit",
            "sl_hit": "SL hit",
            "manual_close": "Manual close",
            "reversal_signal": "Reversal signal",
        }.get(reason, reason)

        line = f"{i}. {action} @ {entry:.2f}" if entry else f"{i}. {action}"
        line += f" -> {reason_text} @ {exit_p:.2f}" if exit_p else ""
        line += f" | {'+'if pnl>=0 else ''}{pnl:.2f}"

        # Add lesson if available
        lessons = d.get("outcome_lessons", [])
        sl_placement = d.get("sl_placement")
        tp_placement = d.get("tp_placement")
        if sl_placement and sl_placement != "appropriate":
            line += f" | SL {sl_placement.replace('_', ' ')}"
        elif tp_placement and tp_placement != "appropriate":
            line += f" | TP {tp_placement.replace('_', ' ')}"

        # Always show SL/TP distance and flag if too tight
        if entry and sl:
            sl_dist_pct = abs(entry - sl) / entry * 100
            sl_dist_label = ""
            if sl_dist_pct < 0.3:
                sl_dist_label = " [WARN] EXTREMELY TIGHT"
                tight_sl_count += 1
            elif sl_dist_pct < 0.5:
                sl_dist_label = " [WARN] VERY TIGHT"
                tight_sl_count += 1
            elif sl_dist_pct < 1.0:
                sl_dist_label = " (tight)"
                tight_sl_count += 1
            line += f" | SL {sl_dist_pct:.1f}% from entry{sl_dist_label}"

        if entry and tp:
            tp_dist_pct = abs(tp - entry) / entry * 100
            if tp_dist_pct < 0.3:
                line += f" | TP {tp_dist_pct:.1f}% from entry [WARN] TOO CLOSE"
                tight_tp_count += 1

        lines.append(line)

    # Add explicit warning if SL/TP placement is a pattern
    if tight_sl_count >= 2:
        lines.append("")
        lines.append(f"[WARN] **CRITICAL PATTERN**: {tight_sl_count}/5 recent trades had SL too tight (<1% from entry).")
        lines.append("Your stops are getting hit or positions closed at a loss because SL is too close to entry.")
        lines.append("**ACTION REQUIRED**: Place SL at least 1-1.5x ATR from entry, or beyond the nearest OB/FVG zone. Do NOT use SL within 0.5% of entry.")
    if tight_tp_count >= 2:
        lines.append(f"[WARN] **PATTERN**: {tight_tp_count}/5 recent trades had TP too close (<0.3% from entry). Use wider targets.")

    # Top lessons from structured outcomes
    all_lessons = []
    for d in closed:
        all_lessons.extend(d.get("outcome_lessons", []))
    if all_lessons:
        # Count and deduplicate
        lesson_counts: Dict[str, int] = {}
        for lesson in all_lessons:
            key = lesson[:80]
            lesson_counts[key] = lesson_counts.get(key, 0) + 1
        top = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append("")
        lines.append("### Apply These Lessons")
        for lesson, count in top:
            lines.append(f"- {lesson}" + (f" (occurred {count}x)" if count > 1 else ""))

    return "\n".join(lines)
