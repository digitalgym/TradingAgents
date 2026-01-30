from langchain_core.messages import AIMessage
import time
import json

from tradingagents.agents.utils.agent_utils import record_memory_usage, extract_memory_ids_from_results


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        # Build SMC instruction for bullish perspective
        smc_instruction = ""
        if smc_context:
            smc_instruction = f"""

{smc_context}

BULLISH SMC ANALYSIS GUIDANCE:
Use the Smart Money Concepts data above to strengthen your bullish argument:
- Reference SUPPORT zones (bullish Order Blocks, bullish FVGs) where institutional buyers are likely positioned
- Highlight if price is in a DISCOUNT zone (below equilibrium) - favorable for buying
- Note any bullish BOS (Break of Structure) confirming uptrend continuation
- Point to unswept EQL (Equal Lows) as liquidity that may fuel upward moves after collection
- Reference bullish OTE zones as optimal entry areas for long positions
- Use confluence scores to support the strength of bullish setups
"""

        prompt = f"""{price_context}You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}{smc_instruction}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        result = {"investment_debate_state": new_investment_debate_state}

        # Include memory tracking if memories were used
        if memory_ids_used:
            result.update(record_memory_usage(state, "bull_researcher", memory_ids_used))

        return result

    return bull_node
