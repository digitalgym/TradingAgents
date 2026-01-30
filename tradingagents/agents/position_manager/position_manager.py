"""
Position Manager Agent

A dedicated agent for managing existing trading positions.
Unlike the trader agent (which focuses on new trade entries), this agent
specifically analyzes whether to HOLD, ADJUST, or CLOSE existing positions.

It synthesizes:
- Market analysis from other agents (bias, sentiment, news)
- SMC structure analysis (support/resistance, structure breaks)
- Position context (entry, SL, TP, P/L, direction)

And outputs position-specific management recommendations.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum


class PositionAction(str, Enum):
    """Possible actions for position management."""
    HOLD = "HOLD"
    ADJUST = "ADJUST"
    CLOSE = "CLOSE"


class UrgencyLevel(str, Enum):
    """Urgency of the recommended action."""
    IMMEDIATE = "immediate"  # Act now - structure break, reversal signal
    HIGH = "high"           # Act soon - deteriorating conditions
    NORMAL = "normal"       # Standard recommendation
    LOW = "low"             # Optional optimization


class PositionManagerDecision(BaseModel):
    """Structured output for position management decisions."""

    action: PositionAction = Field(
        ...,
        description="Recommended action: HOLD, ADJUST, or CLOSE"
    )

    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.NORMAL,
        description="How urgently should this action be taken"
    )

    # SL/TP suggestions (only relevant for ADJUST)
    suggested_sl: Optional[float] = Field(
        None,
        description="Suggested stop loss price. Only set if recommending SL change."
    )

    suggested_tp: Optional[float] = Field(
        None,
        description="Suggested take profit price. Only set if recommending TP change."
    )

    trail_sl_to: Optional[float] = Field(
        None,
        description="Suggested trailing stop level for profitable positions."
    )

    # Close reasoning
    close_reason: Optional[str] = Field(
        None,
        description="If CLOSE, brief reason (e.g., 'structure break against position', 'bias reversal')"
    )

    # Analysis summary
    bias_assessment: str = Field(
        ...,
        description="Current market bias assessment relative to position"
    )

    risk_assessment: str = Field(
        ...,
        description="Current risk level for this position (low/medium/high/critical)"
    )

    key_factors: list[str] = Field(
        default_factory=list,
        description="Top 3 factors influencing this decision"
    )

    reasoning: str = Field(
        ...,
        description="Detailed reasoning for the recommendation"
    )


POSITION_MANAGER_SYSTEM_PROMPT = """You are an expert Position Manager specializing in trade management.

Your ONLY job is to analyze EXISTING positions and recommend:
1. HOLD - Keep the position as-is, current SL/TP are appropriate
2. ADJUST - Modify SL and/or TP to optimize the position
3. CLOSE - Exit the position due to adverse conditions

CRITICAL RULES:
- You are NOT recommending new trades. The position already exists.
- For a LONG position: SL must be BELOW entry price, TP must be ABOVE entry price
- For a SHORT position: SL must be ABOVE entry price, TP must be BELOW entry price
- Never suggest SL/TP values that would instantly close the position
- Consider the trader's risk tolerance and position size

WHEN TO RECOMMEND EACH ACTION:

HOLD when:
- Market bias aligns with position direction
- No structure breaks against the position
- Current SL/TP levels are well-placed relative to support/resistance
- Position is working as expected

ADJUST when:
- Position is profitable and can trail SL to lock in gains
- Current SL is at risk (near liquidity pool, above support for longs)
- Better TP level identified based on resistance/support
- Risk can be reduced without sacrificing position potential

CLOSE when:
- Clear structure break against position (CHOCH)
- Market bias has reversed against position
- Multiple confluence factors suggest reversal
- Position is at high risk with no reasonable SL placement

POSITION MANAGEMENT PRIORITIES:
1. Protect capital - If structure breaks against position, recommend CLOSE
2. Lock in profits - Trail SL to breakeven or better when in profit
3. Optimize risk/reward - Suggest better SL/TP based on SMC zones
4. Let winners run - Don't close profitable positions prematurely unless there's a clear reversal signal"""


def create_position_manager_prompt(
    position_context: Dict[str, Any],
    smc_context: Dict[str, Any],
    market_analysis: str,
    research_summary: str,
) -> str:
    """
    Create the prompt for the position manager agent.

    Args:
        position_context: Current position details (direction, entry, SL, TP, P/L)
        smc_context: SMC analysis (support/resistance zones, structure breaks, bias)
        market_analysis: Summary from market analyst
        research_summary: Bull/bear debate summary from research manager

    Returns:
        Formatted prompt string
    """
    direction = position_context.get('direction', 'UNKNOWN')
    entry_price = position_context.get('entry_price', 0)
    current_price = position_context.get('current_price', 0)
    current_sl = position_context.get('current_sl', 0)
    current_tp = position_context.get('current_tp', 0)
    pnl_pct = position_context.get('pnl_pct', 0)
    profit = position_context.get('profit', 0)
    volume = position_context.get('volume', 0)

    # SMC details
    bias = smc_context.get('bias', 'neutral')
    bias_aligns = smc_context.get('bias_aligns', False)
    structure_shift = smc_context.get('structure_shift', False)
    sl_at_risk = smc_context.get('sl_at_risk', False)
    sl_risk_reason = smc_context.get('sl_risk_reason', '')
    suggested_sl = smc_context.get('suggested_sl')
    suggested_tp = smc_context.get('suggested_tp')
    trailing_sl = smc_context.get('trailing_sl')

    # Format support/resistance zones
    support_str = ""
    if smc_context.get('support_levels'):
        support_str = "\n".join([
            f"  - {s.get('type', 'zone')} @ {s.get('price', 0):.5f} (strength: {s.get('strength', 0):.0%})"
            for s in smc_context.get('support_levels', [])[:3]
        ])
    else:
        support_str = "  None identified"

    resistance_str = ""
    if smc_context.get('resistance_levels'):
        resistance_str = "\n".join([
            f"  - {r.get('type', 'zone')} @ {r.get('price', 0):.5f} (strength: {r.get('strength', 0):.0%})"
            for r in smc_context.get('resistance_levels', [])[:3]
        ])
    else:
        resistance_str = "  None identified"

    prompt = f"""EXISTING POSITION TO MANAGE:
=====================================
Direction: {direction}
Entry Price: {entry_price:.5f}
Current Price: {current_price:.5f}
Volume: {volume} lots
Current P/L: {pnl_pct:+.2f}% (${profit:.2f})

Current Stop Loss: {f'{current_sl:.5f}' if current_sl else 'NOT SET - RISK EXPOSURE'}
Current Take Profit: {f'{current_tp:.5f}' if current_tp else 'NOT SET'}

SMC STRUCTURE ANALYSIS:
=====================================
Market Bias: {bias.upper()}
Bias Aligns with Position: {'YES' if bias_aligns else 'NO - CAUTION'}
Structure Shift Detected: {'YES - CHOCH AGAINST POSITION' if structure_shift else 'No'}

SL Assessment: {f'WARNING - {sl_risk_reason}' if sl_at_risk else 'Current SL placement appears safe'}

Support Zones (for longs - place SL below):
{support_str}

Resistance Zones (for shorts - place SL above):
{resistance_str}

SMC-Calculated Suggestions:
- Suggested SL: {f'{suggested_sl:.5f}' if suggested_sl else 'N/A'}
- Suggested TP: {f'{suggested_tp:.5f}' if suggested_tp else 'N/A'}
- Trailing SL (if profitable): {f'{trailing_sl:.5f}' if trailing_sl else 'N/A'}

MARKET ANALYSIS SUMMARY:
=====================================
{market_analysis[:1500] if market_analysis else 'No market analysis available'}

RESEARCH DEBATE SUMMARY:
=====================================
{research_summary[:1500] if research_summary else 'No research summary available'}

YOUR TASK:
Based on all the above information, provide your position management recommendation.
Remember: You are managing an EXISTING {direction} position, not entering a new trade.

Consider:
1. Does the market bias still support this position?
2. Is there a structure break (CHOCH) that invalidates the position?
3. Is the current SL at risk of being hit before the move plays out?
4. Can we trail SL to lock in profits if position is profitable?
5. Are there better SL/TP levels based on the SMC zones?

Provide your recommendation as a structured response."""

    return prompt


def create_position_manager_agent(
    llm_client,
    model: str,
    use_responses_api: bool = False,
):
    """
    Create a position manager agent function.

    Args:
        llm_client: OpenAI-compatible client
        model: Model name to use
        use_responses_api: Whether to use xAI Responses API

    Returns:
        Function that takes position context and returns management decision
    """
    from tradingagents.dataflows.llm_client import structured_output

    def manage_position(
        position_context: Dict[str, Any],
        smc_context: Dict[str, Any],
        market_analysis: str = "",
        research_summary: str = "",
    ) -> PositionManagerDecision:
        """
        Analyze a position and return management recommendation.

        Args:
            position_context: Position details (direction, entry, SL, TP, P/L)
            smc_context: SMC analysis results
            market_analysis: Market analyst output
            research_summary: Research manager output

        Returns:
            PositionManagerDecision with recommendation
        """
        prompt = create_position_manager_prompt(
            position_context=position_context,
            smc_context=smc_context,
            market_analysis=market_analysis,
            research_summary=research_summary,
        )

        messages = [
            {"role": "system", "content": POSITION_MANAGER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            decision = structured_output(
                client=llm_client,
                model=model,
                messages=messages,
                response_schema=PositionManagerDecision,
                max_tokens=1000,
                temperature=0.3,
                use_responses_api=use_responses_api,
            )

            # Validate SL/TP directions
            direction = position_context.get('direction', 'BUY').upper()
            entry_price = position_context.get('entry_price', 0)

            if decision.suggested_sl:
                # For LONG: SL must be below entry
                # For SHORT: SL must be above entry
                if direction == 'BUY' and decision.suggested_sl >= entry_price:
                    decision.suggested_sl = None
                elif direction == 'SELL' and decision.suggested_sl <= entry_price:
                    decision.suggested_sl = None

            if decision.suggested_tp:
                # For LONG: TP must be above entry
                # For SHORT: TP must be below entry
                if direction == 'BUY' and decision.suggested_tp <= entry_price:
                    decision.suggested_tp = None
                elif direction == 'SELL' and decision.suggested_tp >= entry_price:
                    decision.suggested_tp = None

            if decision.trail_sl_to:
                # Trailing SL must be better than entry (locking in profit)
                if direction == 'BUY' and decision.trail_sl_to <= position_context.get('entry_price', 0):
                    pass  # Trailing below entry is valid for longs not yet at breakeven
                elif direction == 'SELL' and decision.trail_sl_to >= position_context.get('entry_price', 0):
                    pass  # Trailing above entry is valid for shorts not yet at breakeven

            return decision

        except Exception as e:
            # Return a safe default on error
            return PositionManagerDecision(
                action=PositionAction.HOLD,
                urgency=UrgencyLevel.NORMAL,
                bias_assessment="Unable to assess - analysis error",
                risk_assessment="unknown",
                key_factors=["Analysis error occurred"],
                reasoning=f"Error during analysis: {str(e)}. Defaulting to HOLD for safety.",
            )

    return manage_position
