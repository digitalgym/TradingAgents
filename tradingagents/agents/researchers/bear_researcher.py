from langchain_core.messages import AIMessage
import time
import json

from tradingagents.agents.utils.agent_utils import record_memory_usage, extract_memory_ids_from_results


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get current broker price for context
        current_price = state.get("current_price")
        ticker = state.get("company_of_interest", "")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""

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

        # Build broker price context
        price_context = ""
        if current_price:
            price_context = f"""BROKER PRICE: The trader's broker quotes {ticker} at {current_price:.5f}. Use this price (not news prices) when discussing entry levels, targets, or stop losses.

"""

        # Build SMC instruction for bearish perspective
        smc_instruction = ""
        if smc_context:
            smc_instruction = f"""

{smc_context}

BEARISH SMC ANALYSIS GUIDANCE:
Use the Smart Money Concepts data above to strengthen your bearish argument:
- Reference RESISTANCE zones (bearish Order Blocks, bearish FVGs) where institutional sellers are likely positioned
- Highlight if price is in a PREMIUM zone (above equilibrium) - unfavorable for buying, favorable for shorting
- Note any bearish CHOCH (Change of Character) signaling potential trend reversal
- Point to unswept EQH (Equal Highs) as liquidity that may be swept before price reverses down
- Reference bearish OTE zones as optimal entry areas for short positions
- Note if confluence score is low - indicating weak setup quality
- Highlight if multi-timeframe biases are conflicting - increased uncertainty
"""

        prompt = f"""{price_context}You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}{smc_instruction}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        result = {"investment_debate_state": new_investment_debate_state}

        # Include memory tracking if memories were used
        if memory_ids_used:
            result.update(record_memory_usage(state, "bear_researcher", memory_ids_used))

        return result

    return bear_node
