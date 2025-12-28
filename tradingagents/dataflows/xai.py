"""
xAI/Grok dataflow module for news and sentiment from X (Twitter) and web search.

Uses Grok's Agent Tools API (Responses API) with:
- web_search: Search the web for news
- x_search: Search X/Twitter for posts

Requires Grok 4 models and the /v1/responses endpoint.
"""

import os
import requests
from typing import Annotated
from datetime import datetime, timedelta


def _call_responses_api(prompt: str, tools: list, from_date: str = None, to_date: str = None) -> str:
    """
    Call xAI's Responses API with agent tools.
    
    The Responses API is required for web_search and x_search tools.
    Only works with Grok 4 models.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")
    
    # Build tools config with date filters if provided
    tools_config = []
    for tool in tools:
        tool_config = {"type": tool}
        if tool == "x_search" and from_date:
            tool_config["from_date"] = from_date
            if to_date:
                tool_config["to_date"] = to_date
        tools_config.append(tool_config)
    
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "input": prompt,
        "tools": tools_config,
        "tool_choice": "auto",
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(
        "https://api.x.ai/v1/responses",
        json=payload,
        headers=headers,
        timeout=120,
    )
    
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    
    data = response.json()
    
    # Extract the output text from the response
    if "output" in data:
        # Handle different output formats
        output = data["output"]
        if isinstance(output, list):
            # Find message content in output items
            for item in output:
                if item.get("type") == "message" and "content" in item:
                    content = item["content"]
                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") == "output_text":
                                return c.get("text", "")
                    elif isinstance(content, str):
                        return content
        elif isinstance(output, str):
            return output
    
    # Fallback: try to get any text content
    return data.get("output_text", str(data))


def get_xai_news(
    ticker: Annotated[str, "Ticker symbol or query to search"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Get news for a ticker using Grok's web search capability.
    
    Uses Grok 4 with web_search tool via Responses API.
    """
    prompt = f"""Search the web for recent news about {ticker} between {start_date} and {end_date}.
    
Focus on:
- Price movements and market analysis
- Major announcements or events
- Analyst opinions and forecasts

Provide a summary of the most relevant news articles found."""

    try:
        return _call_responses_api(prompt, ["web_search"])
    except Exception as e:
        print(f"xAI web search failed: {e}")
        return f"Error fetching news: {e}"


def get_xai_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"],
    limit: Annotated[int, "Max number of results"] = 10,
) -> str:
    """
    Get global market news using Grok's web search capability.
    """
    prompt = f"""Search the web for global market news from the past {look_back_days} days (up to {curr_date}).
    
Focus on:
- Major market movements (stocks, commodities, forex)
- Economic indicators and central bank decisions
- Geopolitical events affecting markets
- Commodity prices (gold, oil, etc.)

Provide a summary of the {limit} most important market news items."""

    try:
        return _call_responses_api(prompt, ["web_search"])
    except Exception as e:
        print(f"xAI global news search failed: {e}")
        return f"Error fetching global news: {e}"


def get_x_sentiment(
    ticker: Annotated[str, "Ticker symbol or asset name to analyze"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"] = 7,
) -> str:
    """
    Get sentiment analysis from X (Twitter) posts about a ticker/asset.
    
    Uses Grok 4 with x_search tool via Responses API.
    """
    # Calculate from_date
    from_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    
    prompt = f"""Search X (Twitter) for posts about {ticker}.

Analyze the sentiment of the posts and provide:
1. Overall sentiment (Bullish/Bearish/Neutral) with a score from -1 to 1
2. Key themes and topics being discussed
3. Notable posts from influential accounts (if any)
4. Volume/engagement level (high/medium/low activity)
5. Any emerging trends or concerns

Format the response as a structured sentiment report."""

    try:
        return _call_responses_api(prompt, ["x_search"], from_date, curr_date)
    except Exception as e:
        print(f"xAI X sentiment search failed: {e}")
        return f"Error fetching X sentiment: {e}"


def get_x_news(
    ticker: Annotated[str, "Ticker symbol or query to search"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Get news/updates from X (Twitter) about a ticker.
    
    Uses Grok 4 with x_search tool via Responses API.
    """
    prompt = f"""Search X (Twitter) for posts about {ticker}.

Find:
1. Breaking news and announcements
2. Analysis and commentary from traders/analysts
3. Market reactions and price discussions
4. Any viral or highly-engaged posts about {ticker}

Summarize the key information from X posts."""

    try:
        return _call_responses_api(prompt, ["x_search"], start_date, end_date)
    except Exception as e:
        print(f"xAI X news search failed: {e}")
        return f"Error fetching X news: {e}"
