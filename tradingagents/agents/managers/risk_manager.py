import time
import json
from typing import Optional

from tradingagents.schemas import FinalTradingDecision
from tradingagents.agents.utils.agent_utils import record_memory_usage, extract_memory_ids_from_results


def create_risk_manager(llm, memory, use_structured_output: bool = True):
    """
    Create a risk manager node that evaluates risk debates and produces trading decisions.

    Args:
        llm: The language model to use for decision making
        memory: Memory store for past decisions and lessons learned
        use_structured_output: If True, uses LLM structured output for guaranteed JSON schema compliance

    Returns:
        A function that processes state and returns the final trading decision
    """

    # Create structured output LLM wrapper if supported
    structured_llm = None
    if use_structured_output:
        try:
            structured_llm = llm.with_structured_output(FinalTradingDecision)
        except Exception as e:
            # Fallback if LLM doesn't support structured output
            print(f"Warning: Structured output not supported, falling back to free-form: {e}")
            structured_llm = None

    def risk_manager_node(state) -> dict:
        company_name = state["company_of_interest"]

        # Get current broker price - CRITICAL for accurate price levels
        current_price = state.get("current_price")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"

        # Track memory IDs for feedback loop
        memory_ids_used = []

        past_memory_str = ""
        if memory is not None:
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
            # Track which memories were used
            if past_memories:
                memory_ids_used = extract_memory_ids_from_results(past_memories)

        # Build broker price context - CRITICAL
        price_context = ""
        if current_price:
            price_context = f"""
**CRITICAL - BROKER PRICE**: {current_price:.5f}
This is the ACTUAL trading price from the broker. News sources may report different prices (e.g., spot vs CFD).
When specifying entry_price, stop_loss, and take_profit, they MUST be based on this broker price ({current_price:.5f}), not news-reported prices.

"""

        # Build SMC validation guidance for final decision
        smc_guidance = ""
        if smc_context:
            smc_guidance = f"""

{smc_context}

**SMC VALIDATION REQUIREMENTS**:
Your final decision MUST be validated against the Smart Money Concepts analysis above:
1. **Entry Price**: Should be AT or NEAR an institutional zone (Order Block or FVG), not in empty space
2. **Stop Loss**: MUST be placed BEYOND the nearest institutional zone (below OB for buys, above OB for sells) - NOT inside the zone
3. **Take Profit**: Should target the next institutional zone in the trade direction (resistance for buys, support for sells)
4. **Confluence Score**: If the score is below 50, increase caution; above 70 supports conviction
5. **Price Zone**: Favor BUYS in discount zone, SELLS in premium zone; be cautious at equilibrium
6. **Multi-Timeframe Alignment**: Strong trades have all timeframes aligned; mixed signals warrant reduced size
7. **Liquidity Targets**: Consider EQH/EQL as potential stop-hunt levels - don't place stops at these levels
"""

        # Base prompt for both structured and unstructured output
        base_context = f"""{price_context}As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader.

**Asset**: {company_name}

**Trader's Original Plan**:
{trader_plan}

**Past Lessons Learned** (use these to avoid repeating mistakes):
{past_memory_str if past_memory_str else "No past lessons available."}
{smc_guidance}
**Analysts Debate History**:
{history}

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Adjust entry, stop loss, and take profit levels based on the analysts' risk insights AND SMC analysis.
4. **Learn from Past Mistakes**: Apply lessons from past decisions to avoid repeating errors.
5. **Be Decisive**: Choose Hold only if strongly justified by specific arguments, not as a fallback.
6. **Validate Against SMC**: Ensure all price levels align with institutional zones from the SMC analysis."""

        if structured_llm is not None:
            # Use structured output - LLM returns FinalTradingDecision directly
            structured_prompt = f"""{base_context}

Your decision must include:
- signal: BUY, SELL, or HOLD
- confidence: Your confidence level (0.0 to 1.0)
- entry_price, stop_loss, take_profit: Refined price levels (use null if not explicitly determined)
- risk_level: Low, Medium, High, or Extreme
- risk_reward_ratio: Calculated R:R if prices are available
- rationale: Detailed explanation of your decision
- key_risks: Main risks that could invalidate this trade
- key_catalysts: Factors that could drive the trade in the expected direction"""

            try:
                decision: FinalTradingDecision = structured_llm.invoke(structured_prompt)

                # Convert structured decision to dict for state
                decision_dict = decision.model_dump()

                # Create a human-readable text version for display and logging
                text_decision = _format_decision_text(decision)

                new_risk_debate_state = {
                    "judge_decision": text_decision,
                    "history": risk_debate_state["history"],
                    "risky_history": risk_debate_state["risky_history"],
                    "safe_history": risk_debate_state["safe_history"],
                    "neutral_history": risk_debate_state["neutral_history"],
                    "latest_speaker": "Judge",
                    "current_risky_response": risk_debate_state["current_risky_response"],
                    "current_safe_response": risk_debate_state["current_safe_response"],
                    "current_neutral_response": risk_debate_state["current_neutral_response"],
                    "count": risk_debate_state["count"],
                }

                result = {
                    "risk_debate_state": new_risk_debate_state,
                    "final_trade_decision": text_decision,
                    "final_trade_decision_structured": decision_dict,
                }

                # Include memory tracking if memories were used
                if memory_ids_used:
                    result.update(record_memory_usage(state, "risk_manager", memory_ids_used))

                return result

            except Exception as e:
                # Fallback to unstructured if structured output fails
                print(f"Structured output failed, falling back to unstructured: {e}")

        # Unstructured fallback
        unstructured_prompt = f"""{base_context}

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Entry price, stop loss, and take profit levels if recommending Buy or Sell.
- Detailed reasoning anchored in the debate and past reflections.
- Assessment of risk level (Low, Medium, High, or Extreme).

End your response with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"""

        response = llm.invoke(unstructured_prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        result = {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
            "final_trade_decision_structured": None,  # Will be parsed by SignalProcessor
        }

        # Include memory tracking if memories were used
        if memory_ids_used:
            result.update(record_memory_usage(state, "risk_manager", memory_ids_used))

        return result

    return risk_manager_node


def _format_decision_text(decision: FinalTradingDecision) -> str:
    """Format a structured decision into human-readable text."""
    lines = [
        f"## FINAL TRADING DECISION: **{decision.signal}**",
        f"**Confidence**: {decision.confidence:.0%}",
        f"**Risk Level**: {decision.risk_level}",
        "",
    ]

    # Trade parameters
    if decision.signal != "HOLD":
        lines.append("### Trade Parameters")
        if decision.entry_price:
            lines.append(f"- **Entry Price**: {decision.entry_price}")
        if decision.stop_loss:
            lines.append(f"- **Stop Loss**: {decision.stop_loss}")
        if decision.take_profit:
            lines.append(f"- **Take Profit**: {decision.take_profit}")
        if decision.take_profit_2:
            lines.append(f"- **Take Profit 2**: {decision.take_profit_2}")
        if decision.risk_reward_ratio:
            lines.append(f"- **Risk/Reward Ratio**: {decision.risk_reward_ratio:.2f}")
        if decision.position_size_recommendation:
            lines.append(f"- **Position Sizing**: {decision.position_size_recommendation}")
        lines.append("")

    # Rationale
    lines.extend([
        "### Rationale",
        decision.rationale,
        "",
    ])

    # Risks and catalysts
    if decision.key_risks:
        lines.extend([
            "### Key Risks",
            decision.key_risks,
            "",
        ])

    if decision.key_catalysts:
        lines.extend([
            "### Key Catalysts",
            decision.key_catalysts,
            "",
        ])

    lines.append(f"FINAL TRANSACTION PROPOSAL: **{decision.signal}**")

    return "\n".join(lines)
