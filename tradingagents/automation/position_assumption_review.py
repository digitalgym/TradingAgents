"""
Position Assumption Review

Lightweight review of open positions against current market structure.
Two-step process:
1. Rule-based SMC structure check (fast, no LLM):
   - Has the entry zone (OB/FVG) been mitigated or disappeared?
   - Is the TP still realistic (no new zones blocking it)?
   - Is the SL still protected behind a valid zone?
   - Has market bias shifted (CHOCH against position)?
   - Have new OBs/FVGs emerged that change the picture?

2. LLM assessment (optional):
   - Takes the rule-based findings + original rationale + current SMC context
   - Provides a nuanced interpretation and actionable recommendation
   - Same style as the quant analyst output
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from tradingagents.dataflows.mt5_data import get_open_positions
from tradingagents.trade_decisions import (
    list_active_decisions,
    load_decision,
    find_decision_by_ticket,
)

logger = logging.getLogger("PositionAssumptionReview")


@dataclass
class AssumptionFinding:
    """A single finding from the assumption review."""
    category: str  # "bias_shift", "sl_risk", "tp_blocked", "zone_mitigated", "structure_break", "zone_emerged"
    severity: str  # "critical", "warning", "info"
    message: str
    suggested_action: Optional[str] = None  # "adjust_sl", "adjust_tp", "close", "monitor"
    suggested_value: Optional[float] = None  # New SL or TP value if applicable


@dataclass
class PositionAssumptionReport:
    """Report from reviewing one position's assumptions."""
    decision_id: str
    symbol: str
    direction: str
    ticket: int
    entry_price: float
    current_price: float
    current_sl: float
    current_tp: float
    pnl_pct: float
    findings: List[AssumptionFinding] = field(default_factory=list)
    recommended_action: str = "hold"  # "hold", "adjust_sl", "adjust_tp", "close"
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    llm_assessment: Optional[str] = None  # LLM's nuanced interpretation
    review_strategy: Optional[str] = None  # "smc", "mean_reversion", "breakout", etc.
    review_timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


def review_position_assumptions(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Review a single position's original assumptions against current market structure.

    Args:
        decision: The trade decision record (from trade_decisions)
        position: The MT5 position dict (from get_open_positions)
        timeframe: Timeframe for SMC analysis

    Returns:
        PositionAssumptionReport with findings and recommendations
    """
    symbol = decision.get("symbol", "")
    direction = decision.get("action", "BUY")
    ticket = decision.get("mt5_ticket", 0)
    entry_price = position.get("price_open", decision.get("entry_price", 0))
    current_price = position.get("price_current", 0)
    current_sl = position.get("sl", 0)
    current_tp = position.get("tp", 0)

    # Calculate P&L
    if entry_price > 0 and current_price > 0:
        if direction == "BUY":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
    else:
        pnl_pct = 0

    report = PositionAssumptionReport(
        decision_id=decision.get("decision_id", ""),
        symbol=symbol,
        direction=direction,
        ticket=ticket,
        entry_price=entry_price,
        current_price=current_price,
        current_sl=current_sl,
        current_tp=current_tp,
        pnl_pct=pnl_pct,
    )

    # Route to the correct strategy-specific reviewer
    from tradingagents.automation.strategy_reviewers import route_to_reviewer
    reviewer = route_to_reviewer(decision)
    logger.info(
        f"  Reviewer: {reviewer.__name__} "
        f"(pipeline={decision.get('pipeline')}, setup_type={decision.get('setup_type')})"
    )

    report = reviewer(decision, position, report, timeframe=timeframe)
    return report


def _get_llm_system_prompt(decision: Dict[str, Any]) -> str:
    """Get strategy-aware system prompt for LLM assessment."""
    pipeline = (decision.get("pipeline") or "").lower()
    setup_type = (decision.get("setup_type") or "").lower()

    if pipeline in ("xgboost", "xgboost_ensemble"):
        if "mean_reversion" in setup_type:
            return ("You are a mean reversion trading analyst. Focus on z-scores, range boundaries, "
                    "volatility regimes, and whether price is reverting to mean. Be concise and actionable.")
        if "breakout" in setup_type:
            return ("You are a breakout/momentum trading analyst. Focus on momentum indicators, "
                    "volume, ADX, and whether the breakout is holding. Be concise and actionable.")
    if pipeline in ("smc_quant", "smc_quant_basic", "smc_mtf"):
        return ("You are an SMC (Smart Money Concepts) trading analyst reviewing open positions. "
                "Be concise and actionable.")
    if pipeline == "volume_profile":
        return ("You are a volume profile trading analyst. Focus on POC, VAH, VAL shifts "
                "and value area changes. Be concise and actionable.")
    if pipeline == "rule_based":
        return ("You are a trend-following trading analyst. Focus on EMA alignment, ADX strength, "
                "and trend integrity. Be concise and actionable.")

    return "You are a trading analyst reviewing open positions. Be concise and actionable."


def get_llm_assessment(
    report: PositionAssumptionReport,
    decision: Dict[str, Any],
    smc_context_str: str,
) -> str:
    """
    Get an LLM assessment of the position based on the rule-based findings.

    Takes the structured findings from the rule-based review plus the original
    trade rationale and current SMC context, and asks the LLM for a nuanced
    interpretation — similar to what the quant analyst produces for new entries.

    Args:
        report: The rule-based assumption review report
        decision: The original trade decision record
        smc_context_str: The formatted SMC context string from get_smc_position_review_context

    Returns:
        LLM assessment text
    """
    try:
        from tradingagents.dataflows.llm_client import get_llm_client, chat_completion
    except ImportError:
        logger.warning("LLM client not available, skipping LLM assessment")
        return ""

    try:
        client, model, uses_responses_api = get_llm_client()
    except Exception as e:
        logger.warning(f"Could not initialize LLM client: {e}")
        return ""

    # Build findings summary for prompt
    findings_text = ""
    if report.findings:
        findings_text = "\n".join(
            f"- [{f.severity.upper()}] {f.category}: {f.message}"
            for f in report.findings
        )
    else:
        findings_text = "No structural issues detected. Original assumptions appear intact."

    original_rationale = decision.get("rationale", "No rationale recorded")

    sl_str = f"{report.current_sl:.5f}" if report.current_sl else "NONE"
    tp_str = f"{report.current_tp:.5f}" if report.current_tp else "NONE"

    prompt = f"""You are reviewing an OPEN {report.direction} position on {report.symbol}.
Your job is to assess whether the original trade thesis still holds based on updated market structure.

## Original Trade
- Direction: {report.direction}
- Entry: {report.entry_price:.5f}
- Current Price: {report.current_price:.5f}
- Current SL: {sl_str}
- Current TP: {tp_str}
- P&L: {report.pnl_pct:+.2f}%

## Original Rationale
{original_rationale[:800]}

## Current Market Structure (SMC)
{smc_context_str[:1500]}

## Rule-Based Findings
{findings_text}

## Your Assessment
Based on the above, provide a concise assessment (3-5 sentences) covering:
1. Do the original assumptions still hold? Which have been validated or invalidated?
2. Are the SL and TP levels still appropriate given current structure?
3. Have any new order blocks or FVGs appeared that change the picture?
4. Clear recommendation: HOLD (no changes), ADJUST (with specific SL/TP values), or CLOSE (with reason).

Be specific with price levels. Be direct — this is for an automated trading system, not a report."""

    try:
        logger.info(f"Calling LLM for assessment: model={model}, responses_api={uses_responses_api}")
        assessment = chat_completion(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": _get_llm_system_prompt(decision)},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
            use_responses_api=uses_responses_api,
        )
        logger.info(f"LLM assessment received: {len(assessment)} chars")
        return assessment.strip()
    except Exception as e:
        import traceback
        logger.warning(f"LLM assessment failed: {e}\n{traceback.format_exc()}")
        return ""


def review_all_positions(
    source_filter: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    timeframe: str = "H1",
    auto_apply: bool = False,
    use_llm: bool = True,
) -> List[PositionAssumptionReport]:
    """
    Review all open positions owned by a specific automation source.

    Two-step process per position:
    1. Rule-based SMC structure check (fast)
    2. LLM assessment interpreting the findings (optional)

    Args:
        source_filter: Only review decisions from this source (e.g. automation instance name)
        symbols: Only review these symbols (optional)
        timeframe: Timeframe for SMC analysis
        auto_apply: If True, apply SL/TP adjustments automatically via MT5
        use_llm: If True, get LLM assessment for each position (default True)

    Returns:
        List of PositionAssumptionReport
    """
    from tradingagents.dataflows.mt5_data import modify_position

    reports = []

    try:
        positions = get_open_positions()
    except Exception as e:
        logger.error(f"Failed to get open positions: {e}")
        return reports

    try:
        active_decisions = list_active_decisions()
    except Exception as e:
        logger.error(f"Failed to get active decisions: {e}")
        return reports

    # Build ticket -> position lookup (ensure int keys for consistent matching)
    ticket_to_position = {}
    for p in positions:
        t = p.get("ticket")
        ticket_to_position[int(t) if t else t] = p

    logger.info(
        f"review_all_positions: {len(positions)} positions, {len(active_decisions)} active decisions, "
        f"source_filter='{source_filter}', symbols={symbols}, "
        f"position_tickets={list(ticket_to_position.keys())}"
    )

    for decision in active_decisions:
        # Filter by source
        if source_filter and decision.get("source") != source_filter:
            continue

        # Filter by symbol
        dec_symbol = decision.get("symbol", "")
        if symbols and dec_symbol not in symbols:
            continue

        ticket = decision.get("mt5_ticket")
        if ticket is not None:
            ticket = int(ticket)
        if not ticket or ticket not in ticket_to_position:
            logger.debug(f"  Skipping decision {decision.get('decision_id')}: ticket={ticket} not in open positions")
            continue

        position = ticket_to_position[ticket]

        logger.info(f"Reviewing assumptions: {dec_symbol} #{ticket} ({decision.get('action')}) "
                    f"pipeline={decision.get('pipeline')}")

        try:
            report = review_position_assumptions(decision, position, timeframe=timeframe)

            # Step 2: LLM assessment (only for SMC strategies that have SMC context)
            if use_llm and report.review_strategy == "smc":
                logger.warning(f"Step 2: Getting LLM assessment for {dec_symbol} #{ticket}")
                try:
                    smc = get_smc_position_review_context(
                        symbol=dec_symbol,
                        direction=decision.get("action", "BUY"),
                        entry_price=position.get("price_open", decision.get("entry_price", 0)),
                        current_price=position.get("price_current", 0),
                        sl=position.get("sl", 0),
                        tp=position.get("tp", 0),
                        timeframe=timeframe,
                    )
                    smc_context_str = smc.get("smc_context", "")
                    assessment = get_llm_assessment(report, decision, smc_context_str)
                    if assessment:
                        report.llm_assessment = assessment
                        logger.info(f"  LLM Assessment: {assessment[:200]}...")
                except Exception as e:
                    logger.warning(f"  LLM assessment skipped: {e}")

            reports.append(report)

            if report.findings:
                for f in report.findings:
                    log_fn = logger.warning if f.severity in ("critical", "warning") else logger.info
                    log_fn(f"  [{f.severity.upper()}] {f.category}: {f.message}")

                logger.info(f"  Recommendation: {report.recommended_action}")

                # Auto-apply adjustments if enabled
                if auto_apply and report.recommended_action in ("adjust_sl", "adjust_tp"):
                    _apply_adjustment(ticket, report)

            else:
                logger.info(f"  No issues found — assumptions still hold")

        except Exception as e:
            logger.error(f"Error reviewing {dec_symbol} #{ticket}: {e}")
            reports.append(PositionAssumptionReport(
                decision_id=decision.get("decision_id", ""),
                symbol=dec_symbol,
                direction=decision.get("action", ""),
                ticket=ticket,
                entry_price=decision.get("entry_price", 0),
                current_price=position.get("price_current", 0),
                current_sl=position.get("sl", 0),
                current_tp=position.get("tp", 0),
                pnl_pct=0,
                error=str(e),
            ))

    return reports


def _apply_adjustment(ticket: int, report: PositionAssumptionReport):
    """Apply SL/TP adjustments from a review report."""
    from tradingagents.dataflows.mt5_data import modify_position

    new_sl = report.suggested_sl
    new_tp = report.suggested_tp

    # Validate: only tighten SL (move in favorable direction), never widen
    if new_sl and report.current_sl > 0:
        if report.direction == "BUY" and new_sl < report.current_sl:
            logger.info(f"  Skipping SL adjustment: {new_sl:.5f} is WORSE than current {report.current_sl:.5f} for BUY")
            new_sl = None
        elif report.direction == "SELL" and new_sl > report.current_sl:
            logger.info(f"  Skipping SL adjustment: {new_sl:.5f} is WORSE than current {report.current_sl:.5f} for SELL")
            new_sl = None

    if new_sl or new_tp:
        kwargs = {}
        if new_sl:
            kwargs["sl"] = new_sl
        if new_tp:
            kwargs["tp"] = new_tp

        result = modify_position(ticket, **kwargs)
        if result.get("success"):
            parts = []
            if new_sl:
                parts.append(f"SL {report.current_sl:.5f} -> {new_sl:.5f}")
            if new_tp:
                parts.append(f"TP {report.current_tp:.5f} -> {new_tp:.5f}")
            logger.info(f"  APPLIED: #{ticket} {', '.join(parts)}")
        else:
            logger.error(f"  Failed to apply adjustment for #{ticket}: {result.get('error')}")


def format_review_summary(reports: List[PositionAssumptionReport]) -> str:
    """Format review reports as a human-readable summary."""
    if not reports:
        return "No positions to review."

    lines = [
        "=" * 60,
        "POSITION ASSUMPTION REVIEW",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Positions reviewed: {len(reports)}",
        "=" * 60,
    ]

    critical_count = sum(1 for r in reports if r.has_critical)
    warning_count = sum(1 for r in reports if r.has_warnings and not r.has_critical)
    ok_count = sum(1 for r in reports if not r.findings and not r.error)

    lines.append(f"Critical: {critical_count}  Warnings: {warning_count}  OK: {ok_count}")
    lines.append("")

    for report in reports:
        status = "CRITICAL" if report.has_critical else ("WARNING" if report.has_warnings else "OK")
        lines.append(
            f"  {report.symbol} {report.direction} #{report.ticket} "
            f"[{status}] P&L: {report.pnl_pct:+.2f}% -> {report.recommended_action.upper()}"
        )
        if report.error:
            lines.append(f"    Error: {report.error}")
        for f in report.findings:
            lines.append(f"    [{f.severity}] {f.message}")
        if report.suggested_sl:
            lines.append(f"    Suggested SL: {report.suggested_sl:.5f}")
        if report.suggested_tp:
            lines.append(f"    Suggested TP: {report.suggested_tp:.5f}")
        if report.llm_assessment:
            lines.append(f"    LLM: {report.llm_assessment}")

    lines.append("=" * 60)
    return "\n".join(lines)
