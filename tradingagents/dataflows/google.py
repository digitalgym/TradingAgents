from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"],
    limit: Annotated[int, "Max number of results"] = 10,
) -> str:
    """Get global market news from Google."""
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")
    
    queries = ["global markets", "economy news", "commodities market"]
    all_news = []
    
    for query in queries:
        try:
            news_results = getNewsData(query.replace(" ", "+"), before_str, curr_date)
            all_news.extend(news_results[:limit // len(queries)])
        except Exception as e:
            print(f"Google news query '{query}' failed: {e}")
            continue
    
    if not all_news:
        return "No global news found."
    
    news_str = ""
    for news in all_news[:limit]:
        news_str += f"### {news['title']} (source: {news['source']})\n{news['snippet']}\n\n"
    
    return f"## Global Market News from {before_str} to {curr_date}:\n\n{news_str}"


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"