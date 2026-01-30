from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


# ============================================================================
# Report Summarization Utilities
# ============================================================================
# These functions create condensed versions of analyst reports to reduce
# context bloat for downstream agents (e.g., risk debators, risk manager)

def extract_market_summary(market_report: str, max_length: int = 500) -> str:
    """
    Extract key points from market analyst report.

    Args:
        market_report: Full market analyst report
        max_length: Maximum character length for summary

    Returns:
        Condensed market summary focusing on bias, key levels, and indicators
    """
    if not market_report or len(market_report.strip()) == 0:
        return "No market analysis available."

    # Extract key sections if they exist
    summary_parts = []

    # Look for bias/trend indicators
    lower_report = market_report.lower()
    if "bullish" in lower_report:
        summary_parts.append("Market Bias: BULLISH")
    elif "bearish" in lower_report:
        summary_parts.append("Market Bias: BEARISH")
    elif "neutral" in lower_report or "mixed" in lower_report:
        summary_parts.append("Market Bias: NEUTRAL/MIXED")

    # Look for RSI mentions
    if "overbought" in lower_report:
        summary_parts.append("RSI: Overbought conditions")
    elif "oversold" in lower_report:
        summary_parts.append("RSI: Oversold conditions")

    # Look for MACD
    if "macd" in lower_report:
        if "bullish crossover" in lower_report:
            summary_parts.append("MACD: Bullish crossover")
        elif "bearish crossover" in lower_report:
            summary_parts.append("MACD: Bearish crossover")

    # If we found structured points, use them
    if summary_parts:
        return "MARKET SUMMARY: " + " | ".join(summary_parts) + f"\n\n[Full analysis truncated - {len(market_report)} chars]"

    # Otherwise truncate intelligently
    if len(market_report) <= max_length:
        return market_report

    # Find a good break point
    truncated = market_report[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length // 2:
        return truncated[:last_period + 1] + f"\n\n[...truncated, {len(market_report) - last_period} more chars]"

    return truncated + "..."


def extract_sentiment_summary(sentiment_report: str, max_length: int = 400) -> str:
    """
    Extract key sentiment indicators from social media analyst report.

    Args:
        sentiment_report: Full sentiment report
        max_length: Maximum character length

    Returns:
        Condensed sentiment summary
    """
    if not sentiment_report or len(sentiment_report.strip()) == 0:
        return "No sentiment analysis available."

    summary_parts = []
    lower_report = sentiment_report.lower()

    # Extract sentiment direction
    if "positive" in lower_report or "bullish sentiment" in lower_report:
        summary_parts.append("Sentiment: POSITIVE")
    elif "negative" in lower_report or "bearish sentiment" in lower_report:
        summary_parts.append("Sentiment: NEGATIVE")
    elif "neutral" in lower_report or "mixed" in lower_report:
        summary_parts.append("Sentiment: NEUTRAL/MIXED")

    # Look for specific mentions
    if "trending" in lower_report:
        summary_parts.append("Topic is trending")
    if "fear" in lower_report:
        summary_parts.append("Fear detected")
    if "greed" in lower_report:
        summary_parts.append("Greed detected")

    if summary_parts:
        return "SENTIMENT SUMMARY: " + " | ".join(summary_parts)

    # Truncate if needed
    if len(sentiment_report) <= max_length:
        return sentiment_report

    truncated = sentiment_report[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length // 2:
        return truncated[:last_period + 1] + "..."

    return truncated + "..."


def extract_news_summary(news_report: str, max_length: int = 400) -> str:
    """
    Extract key news points from news analyst report.

    Args:
        news_report: Full news report
        max_length: Maximum character length

    Returns:
        Condensed news summary
    """
    if not news_report or len(news_report.strip()) == 0:
        return "No news analysis available."

    summary_parts = []
    lower_report = news_report.lower()

    # Look for impact indicators
    if "breaking" in lower_report or "major" in lower_report:
        summary_parts.append("Major news event detected")
    if "fed" in lower_report or "central bank" in lower_report or "interest rate" in lower_report:
        summary_parts.append("Central bank/rates news")
    if "earnings" in lower_report:
        summary_parts.append("Earnings-related news")
    if "geopolitical" in lower_report or "war" in lower_report or "conflict" in lower_report:
        summary_parts.append("Geopolitical factors")
    if "inflation" in lower_report:
        summary_parts.append("Inflation-related news")

    if summary_parts:
        return "NEWS SUMMARY: " + " | ".join(summary_parts)

    # Truncate if needed
    if len(news_report) <= max_length:
        return news_report

    truncated = news_report[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length // 2:
        return truncated[:last_period + 1] + "..."

    return truncated + "..."


def extract_fundamentals_summary(fundamentals_report: str, max_length: int = 400) -> str:
    """
    Extract key fundamentals from fundamentals analyst report.

    Args:
        fundamentals_report: Full fundamentals report
        max_length: Maximum character length

    Returns:
        Condensed fundamentals summary
    """
    if not fundamentals_report or len(fundamentals_report.strip()) == 0:
        return "No fundamentals analysis available."

    summary_parts = []
    lower_report = fundamentals_report.lower()

    # Look for valuation indicators
    if "undervalued" in lower_report:
        summary_parts.append("Valuation: UNDERVALUED")
    elif "overvalued" in lower_report:
        summary_parts.append("Valuation: OVERVALUED")
    elif "fair value" in lower_report:
        summary_parts.append("Valuation: FAIR VALUE")

    # Financial health
    if "strong balance sheet" in lower_report or "healthy" in lower_report:
        summary_parts.append("Financial health: STRONG")
    elif "weak" in lower_report or "debt" in lower_report:
        summary_parts.append("Financial health: CONCERNS")

    # Growth
    if "growth" in lower_report:
        if "high growth" in lower_report or "strong growth" in lower_report:
            summary_parts.append("Growth: STRONG")
        elif "slow growth" in lower_report or "declining" in lower_report:
            summary_parts.append("Growth: WEAK")

    if summary_parts:
        return "FUNDAMENTALS SUMMARY: " + " | ".join(summary_parts)

    # Truncate if needed
    if len(fundamentals_report) <= max_length:
        return fundamentals_report

    truncated = fundamentals_report[:max_length]
    last_period = truncated.rfind('.')
    if last_period > max_length // 2:
        return truncated[:last_period + 1] + "..."

    return truncated + "..."


def create_condensed_context(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    max_total_length: int = 1500
) -> str:
    """
    Create a condensed context from all analyst reports.

    This is useful for downstream agents that don't need full reports.

    Args:
        market_report: Full market analyst report
        sentiment_report: Full sentiment report
        news_report: Full news report
        fundamentals_report: Full fundamentals report
        max_total_length: Maximum total length for combined summary

    Returns:
        Single condensed string with key points from all reports
    """
    # Allocate space proportionally
    market_len = max_total_length // 3
    other_len = (max_total_length - market_len) // 3

    parts = [
        extract_market_summary(market_report, market_len),
        extract_sentiment_summary(sentiment_report, other_len),
        extract_news_summary(news_report, other_len),
        extract_fundamentals_summary(fundamentals_report, other_len),
    ]

    return "\n\n".join(parts)


# ============================================================================
# Memory Tracking Utilities
# ============================================================================
# These functions help track which memories were used by which agents
# for feedback loop validation

def record_memory_usage(
    state: dict,
    agent_name: str,
    memory_ids: list
) -> dict:
    """
    Record memory IDs used by an agent in the state.

    This enables the feedback loop - when a trade closes, we can
    update memory confidence based on whether it helped or not.

    Args:
        state: Current agent state dict
        agent_name: Name of the agent using memories (e.g., "bull_researcher")
        memory_ids: List of memory IDs that were retrieved and used

    Returns:
        Updated memory_ids_used dict to merge into state
    """
    current = state.get("memory_ids_used") or {}
    current[agent_name] = memory_ids
    return {"memory_ids_used": current}


def extract_memory_ids_from_results(memory_results: list) -> list:
    """
    Extract memory IDs from memory retrieval results.

    Args:
        memory_results: List of results from memory.get_memories()

    Returns:
        List of memory IDs
    """
    return [r.get("id") for r in memory_results if r.get("id")]


        