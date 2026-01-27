import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get current broker price - CRITICAL for accurate price levels
        current_price = state.get("current_price")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"

        past_memory_str = "No past memories found."
        if memory is not None:
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            if past_memories:
                past_memory_str = ""
                for i, rec in enumerate(past_memories, 1):
                    past_memory_str += rec["recommendation"] + "\n\n"

        # Build broker price context - CRITICAL
        price_context = ""
        if current_price:
            price_context = f"""
CRITICAL - BROKER PRICE:
The broker quotes {company_name} at {current_price:.5f}. This is the ACTUAL trading price.
NEWS SOURCES MAY REPORT DIFFERENT PRICES (e.g., spot vs CFD). Always use {current_price:.5f} as your reference when suggesting:
- Entry price (should be near {current_price:.5f})
- Stop loss (must be a realistic distance from {current_price:.5f})
- Take profit (must be a realistic distance from {current_price:.5f})
Do NOT use prices from news articles - they may use different quote conventions.

"""

        context = {
            "role": "user",
            "content": f"{price_context}Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        # Build system prompt with SMC context if available
        system_prompt = f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}"""
        
        # Add SMC context if available
        if smc_context:
            system_prompt += f"\n\n{smc_context}\n\nIMPORTANT: When suggesting entry, stop loss, and take profit levels, align them with the Smart Money Concepts zones shown above. Place stops BELOW support zones (for buys) or ABOVE resistance zones (for sells) to avoid premature stop-outs. Target take profits AT resistance zones (for buys) or support zones (for sells) where institutional orders are likely to be filled."
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
