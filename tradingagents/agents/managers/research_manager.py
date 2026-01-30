import time
import json
from typing import Optional

from tradingagents.agents.utils.memory import SMCPatternMemory
from tradingagents.agents.utils.agent_utils import record_memory_usage, extract_memory_ids_from_results


def create_research_manager(llm, memory, smc_memory: Optional[SMCPatternMemory] = None, meta_pattern_learning=None):
    """
    Create a research manager node that synthesizes bull/bear debates into investment decisions.

    Args:
        llm: The language model to use
        memory: General memory store for past decisions
        smc_memory: SMC pattern memory for setup-specific insights
        meta_pattern_learning: Meta-pattern learning for cross-agent insights

    Returns:
        A function that processes state and returns the investment plan
    """

    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get current broker price - CRITICAL for accurate price references
        current_price = state.get("current_price")
        ticker = state.get("company_of_interest", "")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""

        investment_debate_state = state["investment_debate_state"]

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

        # Query SMC pattern memory for setup-specific insights
        smc_pattern_insights = ""
        if smc_memory is not None:
            try:
                # Get best performing setups for this symbol
                best_setups = smc_memory.get_best_setups(symbol=ticker, min_samples=3, min_win_rate=0.5)
                if best_setups:
                    smc_pattern_insights = "\n\nSMC PATTERN INSIGHTS FROM PAST TRADES:\n"
                    for setup in best_setups[:3]:
                        smc_pattern_insights += f"- {setup['recommendation']}\n"

                # Get similar pattern outcomes
                similar_patterns = smc_memory.get_similar_patterns(
                    curr_situation, symbol=ticker, n_matches=2
                )
                if similar_patterns:
                    smc_pattern_insights += "\nSIMILAR PAST MARKET CONDITIONS:\n"
                    for p in similar_patterns:
                        outcome = "WON" if p.get("was_win") else "LOST"
                        returns = p.get("returns_pct", 0)
                        smc_pattern_insights += f"- {p.get('setup_type', 'Unknown')} setup {outcome} ({returns:+.2f}%). "
                        if p.get("lesson"):
                            smc_pattern_insights += f"Lesson: {p['lesson'][:100]}\n"
                        else:
                            smc_pattern_insights += "\n"
            except Exception:
                pass  # Silently ignore SMC memory errors

        # Query meta-pattern learning for cross-agent insights
        meta_insights_str = ""
        if meta_pattern_learning is not None:
            try:
                # Infer bull/bear signals from debate history
                bull_history = investment_debate_state.get("bull_history", "").lower()
                bear_history = investment_debate_state.get("bear_history", "").lower()

                bull_signal = "neutral"
                if "strongly bullish" in bull_history or "highly confident" in bull_history:
                    bull_signal = "bullish"
                elif "cautious" in bull_history or "uncertain" in bull_history:
                    bull_signal = "bearish"
                elif "bullish" in bull_history or "buy" in bull_history:
                    bull_signal = "bullish"

                bear_signal = "neutral"
                if "strongly bearish" in bear_history or "significant risk" in bear_history:
                    bear_signal = "bearish"
                elif "limited downside" in bear_history or "overblown" in bear_history:
                    bear_signal = "bullish"
                elif "bearish" in bear_history or "sell" in bear_history:
                    bear_signal = "bearish"

                meta_insights = meta_pattern_learning.get_meta_insights_for_decision(
                    bull_signal=bull_signal,
                    bear_signal=bear_signal,
                    market_regime=None  # Could be enhanced with regime detection
                )

                if meta_insights and meta_insights.get("pattern_win_rate"):
                    agreement = "agree" if bull_signal == bear_signal else "disagree"
                    win_rate = meta_insights.get("pattern_win_rate", 0)
                    sample_size = meta_insights.get("sample_size", 0)
                    if sample_size >= 3:
                        meta_insights_str = f"\n\nMETA-PATTERN INSIGHT:\nWhen bull and bear analysts {agreement} (as they do now), historical win rate is {win_rate:.0%} (based on {sample_size} trades)."
                        if meta_insights.get("recommendation"):
                            meta_insights_str += f"\nHistorical pattern suggests: {meta_insights['recommendation']}"
            except Exception:
                pass  # Silently ignore meta-pattern errors

        # Build broker price context
        price_context = ""
        if current_price:
            price_context = f"""
CRITICAL - BROKER PRICE CONTEXT:
The trader's broker quotes {ticker} at {current_price:.5f}. This is the ACTUAL trading price.
News sources may report different prices (e.g., spot prices vs CFD prices). When discussing price levels, targets, or stop losses, use the broker price ({current_price:.5f}) as the reference point, not news-reported prices.

"""

        # Build SMC guidance for decision-making
        smc_guidance = ""
        if smc_context:
            smc_guidance = f"""

{smc_context}

SMC DECISION GUIDANCE:
When making your recommendation, consider:
- Does the market bias (from structure breaks) align with the bull or bear argument?
- Are there strong confluence scores supporting one direction?
- Is price in premium (favor sells) or discount (favor buys) zone?
- Are there unmitigated institutional zones that could act as targets or reversals?
"""

        prompt = f"""{price_context}As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"{smc_pattern_insights}{meta_insights_str}{smc_guidance}

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        result = {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

        # Include memory tracking if memories were used
        if memory_ids_used:
            result.update(record_memory_usage(state, "invest_judge", memory_ids_used))

        return result

    return research_manager_node
