from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get current broker price for context
        current_price = state.get("current_price")
        ticker = state.get("company_of_interest", "")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""

        trader_decision = state["trader_investment_plan"]

        # Build price context
        price_context = ""
        if current_price:
            price_context = f"[BROKER PRICE: {ticker} at {current_price:.5f}. Use this for price level discussions.]\n\n"

        # Build SMC instruction with conservative focus
        smc_instruction = ""
        if smc_context:
            smc_instruction = f"""

{smc_context}

CONSERVATIVE SMC RISK ASSESSMENT:
Use the Smart Money Concepts data above to identify and highlight risks:
- If price is in PREMIUM zone (above equilibrium), warn against buying - institutions are selling there
- Point out nearby RESISTANCE zones (bearish OBs, FVGs) that could reject price
- Highlight unswept EQH (Equal Highs) above price as potential stop-hunt targets
- Critique stop loss placements that are INSIDE order blocks or FVGs - they will likely get hit
- Recommend WIDER stops beyond institutional zones for safety
- Note any bearish CHOCH or structure shifts that increase downside risk
- If confluence score is LOW, emphasize the weak setup quality as additional risk
- If multi-timeframe biases CONFLICT, highlight this uncertainty as a reason for caution"""

        prompt = f"""{price_context}As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Risky and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}{smc_instruction}
Here is the current conversation history: {history} Here is the last response from the risky analyst: {current_risky_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints, do not halluncinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
