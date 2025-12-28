from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from datetime import datetime
import re

def normalize_date(date_str: str) -> str:
    """Normalize date string to YYYY-MM-DD format, handling common LLM mistakes."""
    # Already correct format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            # Check if day/month might be swapped (day > 12)
            parts = date_str.split('-')
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if month > 12 and day <= 12:
                # Likely swapped: 2025-26-12 should be 2025-12-26
                return f"{year:04d}-{day:02d}-{month:02d}"
            return date_str
        except ValueError:
            # Invalid date, try to fix
            parts = date_str.split('-')
            if len(parts) == 3:
                year, a, b = int(parts[0]), int(parts[1]), int(parts[2])
                # Try swapping if one is > 12
                if a > 12 and b <= 12:
                    return f"{year:04d}-{b:02d}-{a:02d}"
                elif b > 12 and a <= 12:
                    return f"{year:04d}-{a:02d}-{b:02d}"
    return date_str

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # Normalize date to handle LLM formatting mistakes
    curr_date = normalize_date(curr_date)
    return route_to_vendor("get_indicators", symbol, indicator, curr_date, look_back_days)