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

from tradingagents.dataflows.smc_utils import get_smc_position_review_context
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

    # Get current SMC structure
    try:
        smc = get_smc_position_review_context(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            sl=current_sl,
            tp=current_tp,
            timeframe=timeframe,
        )
    except Exception as e:
        report.error = f"SMC analysis failed: {e}"
        return report

    if smc.get("error"):
        report.error = f"SMC analysis error: {smc['error']}"
        return report

    # Extract original assumptions from decision
    original_rationale = decision.get("rationale", "")
    original_smc = decision.get("smc_context", {})
    original_sl = decision.get("stop_loss", 0)
    original_tp = decision.get("take_profit", 0)

    # === CHECK 1: Bias alignment ===
    bias = smc.get("bias", "neutral")
    bias_aligns = smc.get("bias_aligns", True)

    if not bias_aligns:
        report.findings.append(AssumptionFinding(
            category="bias_shift",
            severity="critical",
            message=f"Market bias is now {bias.upper()} — against your {direction} position. "
                    f"Original trade assumed {'bullish' if direction == 'BUY' else 'bearish'} conditions.",
            suggested_action="close" if pnl_pct < 0 else "monitor",
        ))

    # === CHECK 2: Structure break (CHOCH) against position ===
    if smc.get("structure_shift"):
        report.findings.append(AssumptionFinding(
            category="structure_break",
            severity="critical",
            message=f"Change of Character (CHOCH) detected AGAINST your {direction} position. "
                    f"Market structure has shifted — the original trend assumption may be invalidated.",
            suggested_action="close",
        ))

    # === CHECK 3: SL placement risk ===
    if smc.get("sl_at_risk") and current_sl > 0:
        report.findings.append(AssumptionFinding(
            category="sl_risk",
            severity="warning",
            message=smc.get("sl_risk_reason", "SL is in a vulnerable location"),
            suggested_action="adjust_sl",
            suggested_value=smc.get("suggested_sl"),
        ))

    # Check if SL is no longer behind a valid zone
    if current_sl > 0:
        support_levels = smc.get("support_levels", [])
        resistance_levels = smc.get("resistance_levels", [])

        if direction == "BUY" and support_levels:
            # SL should be below support. If nearest support is now ABOVE SL,
            # that support may have disappeared or shifted
            nearest_support_bottom = support_levels[0].get("bottom", 0)
            if nearest_support_bottom > 0 and current_sl > nearest_support_bottom:
                # SL is above the nearest support — it's exposed
                report.findings.append(AssumptionFinding(
                    category="sl_risk",
                    severity="warning",
                    message=f"SL at {current_sl:.5f} is above nearest support zone at {nearest_support_bottom:.5f}. "
                            f"Your stop may get hit before the support zone can defend the position.",
                    suggested_action="adjust_sl",
                    suggested_value=nearest_support_bottom * 0.998,
                ))
        elif direction == "SELL" and resistance_levels:
            nearest_resistance_top = resistance_levels[0].get("top", 0)
            if nearest_resistance_top > 0 and current_sl < nearest_resistance_top:
                report.findings.append(AssumptionFinding(
                    category="sl_risk",
                    severity="warning",
                    message=f"SL at {current_sl:.5f} is below nearest resistance zone at {nearest_resistance_top:.5f}. "
                            f"Your stop may get hit before the resistance zone can defend the position.",
                    suggested_action="adjust_sl",
                    suggested_value=nearest_resistance_top * 1.002,
                ))

    # === CHECK 4: TP still realistic? ===
    if current_tp > 0:
        support_levels = smc.get("support_levels", [])
        resistance_levels = smc.get("resistance_levels", [])

        if direction == "BUY" and resistance_levels:
            # Check if a new resistance zone has appeared BEFORE our TP
            for zone in resistance_levels:
                zone_price = zone.get("price", 0)
                zone_strength = zone.get("strength", 0)
                if 0 < zone_price < current_tp and zone_price > current_price:
                    # Resistance zone between current price and TP
                    if zone_strength >= 0.6:
                        report.findings.append(AssumptionFinding(
                            category="tp_blocked",
                            severity="warning",
                            message=f"Resistance zone ({zone.get('type', 'zone')}) at {zone_price:.5f} "
                                    f"(strength {zone_strength:.0%}) sits between current price and TP at {current_tp:.5f}. "
                                    f"Price may stall or reverse here before reaching TP.",
                            suggested_action="adjust_tp",
                            suggested_value=zone.get("bottom", zone_price),
                        ))
                        break  # Only report the nearest blocking zone

        elif direction == "SELL" and support_levels:
            for zone in support_levels:
                zone_price = zone.get("price", 0)
                zone_strength = zone.get("strength", 0)
                if 0 < zone_price > current_tp and zone_price < current_price:
                    if zone_strength >= 0.6:
                        report.findings.append(AssumptionFinding(
                            category="tp_blocked",
                            severity="warning",
                            message=f"Support zone ({zone.get('type', 'zone')}) at {zone_price:.5f} "
                                    f"(strength {zone_strength:.0%}) sits between current price and TP at {current_tp:.5f}. "
                                    f"Price may bounce here before reaching TP.",
                            suggested_action="adjust_tp",
                            suggested_value=zone.get("top", zone_price),
                        ))
                        break

    # === CHECK 5: Original entry zone status ===
    # Check if the OB/FVG count has changed significantly
    unmitigated_obs = smc.get("unmitigated_obs", 0)
    unmitigated_fvgs = smc.get("unmitigated_fvgs", 0)

    if direction == "BUY":
        # For a BUY, we want bullish OBs/FVGs (support). If they've all been mitigated...
        support_zones = smc.get("support_levels", [])
        if not support_zones:
            report.findings.append(AssumptionFinding(
                category="zone_mitigated",
                severity="warning",
                message="No unmitigated support zones remain below current price. "
                        "The order blocks or FVGs that supported the original entry may have been filled.",
                suggested_action="monitor",
            ))
    else:
        resistance_zones = smc.get("resistance_levels", [])
        if not resistance_zones:
            report.findings.append(AssumptionFinding(
                category="zone_mitigated",
                severity="warning",
                message="No unmitigated resistance zones remain above current price. "
                        "The order blocks or FVGs that supported the original entry may have been filled.",
                suggested_action="monitor",
            ))

    # === CHECK 6: New zones emerged that help or hurt ===
    # If a strong OB/FVG has appeared between entry and current price in the direction
    # of the trade, that's positive (trailing stop opportunity). Report it as info.
    if smc.get("trailing_sl") and current_sl > 0:
        trailing_sl = smc["trailing_sl"]
        trailing_source = smc.get("trailing_sl_source", "SMC zone")
        is_better = (
            (direction == "BUY" and trailing_sl > current_sl) or
            (direction == "SELL" and trailing_sl < current_sl)
        )
        if is_better:
            report.findings.append(AssumptionFinding(
                category="zone_emerged",
                severity="info",
                message=f"New {trailing_source} has formed between entry and current price. "
                        f"Trailing stop can be tightened from {current_sl:.5f} to {trailing_sl:.5f}.",
                suggested_action="adjust_sl",
                suggested_value=trailing_sl,
            ))

    # === Determine overall recommendation ===
    _determine_recommendation(report, smc)

    return report


def _determine_recommendation(report: PositionAssumptionReport, smc: Dict[str, Any]):
    """Set the overall recommendation based on findings."""
    has_bias_shift = any(f.category == "bias_shift" for f in report.findings)
    has_structure_break = any(f.category == "structure_break" for f in report.findings)
    has_sl_risk = any(f.category == "sl_risk" for f in report.findings)
    has_tp_blocked = any(f.category == "tp_blocked" for f in report.findings)
    has_trailing_opportunity = any(
        f.category == "zone_emerged" and f.suggested_action == "adjust_sl"
        for f in report.findings
    )

    # Critical: bias shift + structure break = close
    if has_bias_shift and has_structure_break:
        report.recommended_action = "close"
        return

    # Critical: structure break alone in losing position = close
    if has_structure_break and report.pnl_pct < 0:
        report.recommended_action = "close"
        return

    # SL at risk — suggest adjustment
    if has_sl_risk:
        sl_findings = [f for f in report.findings if f.category == "sl_risk" and f.suggested_value]
        if sl_findings:
            report.recommended_action = "adjust_sl"
            report.suggested_sl = sl_findings[0].suggested_value

    # TP blocked — suggest adjustment
    if has_tp_blocked:
        tp_findings = [f for f in report.findings if f.category == "tp_blocked" and f.suggested_value]
        if tp_findings:
            if report.recommended_action == "hold":
                report.recommended_action = "adjust_tp"
            report.suggested_tp = tp_findings[0].suggested_value

    # Trailing stop opportunity
    if has_trailing_opportunity and report.recommended_action == "hold":
        trail_findings = [f for f in report.findings if f.category == "zone_emerged" and f.suggested_value]
        if trail_findings:
            report.recommended_action = "adjust_sl"
            report.suggested_sl = trail_findings[0].suggested_value

    # If no findings, keep hold
    if not report.findings:
        report.recommended_action = "hold"


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
                {"role": "system", "content": "You are an SMC (Smart Money Concepts) trading analyst reviewing open positions. Be concise and actionable."},
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

        logger.info(f"Reviewing assumptions: {dec_symbol} #{ticket} ({decision.get('action')})")

        try:
            report = review_position_assumptions(decision, position, timeframe=timeframe)

            # Step 2: LLM assessment
            if use_llm:
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
