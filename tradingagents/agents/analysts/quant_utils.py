"""
Shared utilities for quant analyst agents.

This module consolidates common functionality used across all quant analysts:
- Signal normalization (convert various signal formats to BUY/SELL/HOLD)
- Decision to modal format conversion (for TradeExecutionWizard)
- Shared logger creation
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional


# Signal mapping constant - maps various signal formats to standard BUY/SELL/HOLD
SIGNAL_MAP = {
    "buy_to_enter": "BUY",
    "sell_to_enter": "SELL",
    "hold": "HOLD",
    "close": "HOLD",
    "buy": "BUY",
    "sell": "SELL",
}


def normalize_signal(signal) -> str:
    """
    Normalize a signal value from various formats to BUY/SELL/HOLD.

    Handles:
    - Dict format: {"value": "buy_to_enter"}
    - String format: "buy_to_enter", "BUY_TO_ENTER", "buy", "BUY"
    - None

    Returns: "BUY", "SELL", or "HOLD"
    """
    if signal is None:
        return "HOLD"
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")
    signal_lower = str(signal).lower()
    return SIGNAL_MAP.get(signal_lower, "HOLD")


def normalize_order_type(order_type, default: str = "market") -> str:
    """
    Normalize order type from various formats.

    Handles:
    - Dict format: {"value": "limit"}
    - String format: "limit", "market"
    - None

    Returns: "market" or "limit"
    """
    if order_type is None:
        return default
    if isinstance(order_type, dict):
        order_type = order_type.get("value", default)
    return str(order_type).lower()


def decision_to_modal_format(
    decision: Dict[str, Any],
    strategy_prefix: str = "",
    default_order_type: str = "market",
) -> Dict[str, Any]:
    """
    Convert a quant decision dict to trade modal format.

    This is the shared implementation used by all quant analysts to format
    their decisions for the TradeExecutionWizard component.

    Args:
        decision: The quant decision dict from agent state
        strategy_prefix: Optional prefix for rationale (e.g., "BREAKOUT", "RANGE")
        default_order_type: Default order type if not specified ("market" or "limit")

    Returns:
        Dict formatted for TradeExecutionWizard props:
        - symbol: Trading symbol
        - signal: "BUY", "SELL", or "HOLD"
        - orderType: "market" or "limit"
        - suggestedEntry: Entry price
        - suggestedStopLoss: Stop loss price
        - suggestedTakeProfit: Take profit price
        - rationale: Combined justification and invalidation
        - confidence: Confidence score 0-1
    """
    if not decision:
        return {}

    signal = normalize_signal(decision.get("signal"))
    order_type = normalize_order_type(decision.get("order_type"), default_order_type)

    justification = decision.get("justification", "")
    invalidation = decision.get("invalidation_condition", "")

    # Build rationale with optional prefix
    if strategy_prefix:
        rationale = f"{strategy_prefix}: {justification}. Invalidation: {invalidation}"
    else:
        rationale = f"{justification}. Invalidation: {invalidation}"

    return {
        "symbol": decision.get("symbol", ""),
        "signal": signal,
        "orderType": order_type,
        "suggestedEntry": decision.get("entry_price"),
        "suggestedStopLoss": decision.get("stop_loss"),
        "suggestedTakeProfit": decision.get("profit_target"),
        "rationale": rationale,
        "confidence": decision.get("confidence", 0.5),
    }


def create_quant_logger(logger_name: str, log_subdir: str) -> logging.Logger:
    """
    Create a configured logger for a quant analyst.

    All quant analysts use the same logging pattern:
    - File-based logging with date-stamped filenames
    - DEBUG level for comprehensive prompt/response logging
    - UTF-8 encoding for special characters

    Args:
        logger_name: Name for the logger (e.g., "smc_quant_prompts")
        log_subdir: Subdirectory in logs/ (e.g., "smc_quant_prompts")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)

    # Only configure if not already set up
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Create logs directory
    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "logs", log_subdir
    )
    os.makedirs(log_dir, exist_ok=True)

    # Date-stamped log file
    log_file = os.path.join(
        log_dir, f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# Convenience wrappers for each quant type - maintains backward compatibility
def get_smc_quant_decision_for_modal(decision: dict) -> dict:
    """Convert SMC quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="")


def get_breakout_decision_for_modal(decision: dict) -> dict:
    """Convert breakout quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="BREAKOUT", default_order_type="limit")


def get_range_quant_decision_for_modal(decision: dict) -> dict:
    """Convert range quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="RANGE", default_order_type="limit")


def get_vp_quant_decision_for_modal(decision: dict) -> dict:
    """Convert volume profile quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="")


def get_mtf_quant_decision_for_modal(decision: dict) -> dict:
    """Convert MTF quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="")


def get_quant_decision_for_modal(decision: dict) -> dict:
    """Convert generic quant decision to modal format."""
    return decision_to_modal_format(decision, strategy_prefix="")
