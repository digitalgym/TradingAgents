import os
import logging
from openai import OpenAI
from .config import get_config

logger = logging.getLogger(__name__)


def _get_client():
    """Get an OpenAI client configured for xAI."""
    config = get_config()
    api_key = os.environ.get("XAI_API_KEY") or ""
    base_url = config.get("backend_url", "https://api.x.ai/v1")
    logger.debug(f"Creating OpenAI client with base_url={base_url}")
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def get_stock_news_openai(query, start_date, end_date):
    """Get stock news using xAI API."""
    logger.info(f"get_stock_news_openai called: query={query}, start_date={start_date}, end_date={end_date}")
    try:
        config = get_config()
        client = _get_client()

        response = client.chat.completions.create(
            model=config["quick_think_llm"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial news analyst. Provide recent news and social media sentiment for the requested asset."
                },
                {
                    "role": "user",
                    "content": f"Search for recent news and social media discussions about {query} from {start_date} to {end_date}. Summarize the key points and sentiment."
                }
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        result = response.choices[0].message.content
        logger.info(f"get_stock_news_openai success: {len(result)} chars returned")
        return result
    except Exception as e:
        logger.error(f"get_stock_news_openai failed: {type(e).__name__}: {e}")
        raise


def get_global_news_openai(curr_date, look_back_days=7, limit=5):
    """Get global/macro news using xAI API."""
    logger.info(f"get_global_news_openai called: curr_date={curr_date}, look_back_days={look_back_days}, limit={limit}")
    try:
        config = get_config()
        client = _get_client()

        response = client.chat.completions.create(
            model=config["quick_think_llm"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a macroeconomic analyst. Provide recent global economic and geopolitical news relevant to trading."
                },
                {
                    "role": "user",
                    "content": f"Summarize the most important global and macroeconomic news from the {look_back_days} days before {curr_date} that would be relevant for trading. Focus on central bank decisions, geopolitical events, economic data releases, and market-moving events. Limit to {limit} key items."
                }
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        result = response.choices[0].message.content
        logger.info(f"get_global_news_openai success: {len(result)} chars returned")
        return result
    except Exception as e:
        logger.error(f"get_global_news_openai failed: {type(e).__name__}: {e}")
        raise


def get_fundamentals_openai(ticker, curr_date):
    """Get fundamental analysis using xAI API."""
    logger.info(f"get_fundamentals_openai called: ticker={ticker}, curr_date={curr_date}")
    try:
        config = get_config()
        client = _get_client()
        model = config["quick_think_llm"]
        logger.debug(f"Using model: {model}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fundamental analyst specializing in commodities and forex. For commodities like gold (XAUUSD), silver, etc., focus on supply/demand dynamics, central bank activity, inflation data, and macro drivers rather than traditional equity metrics."
                },
                {
                    "role": "user",
                    "content": f"""Provide a fundamental analysis for {ticker} as of {curr_date}. Include:
- Key supply/demand factors
- Central bank activity (buying/selling)
- Inflation and interest rate outlook
- USD strength/weakness impact
- Geopolitical risk factors
- Recent price drivers
- Analyst consensus and price targets

Format as a structured summary."""
                }
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        result = response.choices[0].message.content
        logger.info(f"get_fundamentals_openai success: {len(result)} chars returned")
        return result
    except Exception as e:
        logger.error(f"get_fundamentals_openai failed: {type(e).__name__}: {e}")
        raise
