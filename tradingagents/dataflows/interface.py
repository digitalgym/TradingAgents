import logging
from typing import Annotated

# Import from vendor-specific modules
from .local import get_finnhub_news, get_finnhub_company_insider_sentiment, get_finnhub_company_insider_transactions, get_reddit_global_news, get_reddit_company_news
from .google import get_google_news, get_google_global_news
from .openai import get_stock_news_openai, get_global_news_openai, get_fundamentals_openai
from .xai import get_xai_news, get_xai_global_news, get_x_sentiment, get_x_news
from .mt5_data import get_mt5_data, get_asset_type, get_mt5_indicator

# Configuration and routing logic
from .config import get_config

logger = logging.getLogger(__name__)


# Stub implementations for equity-specific tools (not applicable to commodities/crypto)
def _not_applicable_for_commodities(ticker, *args, **kwargs):
    """Return N/A message for equity-specific tools when trading commodities."""
    logger.info(f"Financial statement requested for {ticker} - not applicable for commodities/crypto")
    return f"Financial statements (balance sheet, income statement, cash flow) are not applicable for {ticker}. This is a commodity/crypto asset, not an equity. Use get_fundamentals for macro analysis instead."

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Fundamental analysis (macro factors for commodities)",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News (public/insiders, original/processed)",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_sentiment",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "local",
    "openai",
    "google",
    "mt5",
    "xai",
]

# Mapping of methods to their vendor-specific implementations
# Simplified for MT5-based commodity trading (removed Alpha Vantage, yfinance)
VENDOR_METHODS = {
    # core_stock_apis - MT5 only for commodities
    "get_stock_data": {
        "mt5": get_mt5_data,
    },
    # technical_indicators - MT5 calculates locally from OHLCV
    "get_indicators": {
        "mt5": get_mt5_indicator,
    },
    # fundamental_data - OpenAI for commodities (no balance sheets for gold/crypto)
    "get_fundamentals": {
        "openai": get_fundamentals_openai,
    },
    # Equity-specific tools - return N/A for commodities/crypto
    "get_balance_sheet": {
        "openai": _not_applicable_for_commodities,
    },
    "get_cashflow": {
        "openai": _not_applicable_for_commodities,
    },
    "get_income_statement": {
        "openai": _not_applicable_for_commodities,
    },
    # news_data
    "get_news": {
        "xai": get_xai_news,
        "openai": get_stock_news_openai,
        "google": get_google_news,
        "x": get_x_news,
        "local": [get_finnhub_news, get_reddit_company_news, get_google_news],
    },
    "get_global_news": {
        "xai": get_xai_global_news,
        "google": get_google_global_news,
        "openai": get_global_news_openai,
        "local": get_reddit_global_news
    },
    "get_insider_sentiment": {
        "xai": get_x_sentiment,
        "local": get_finnhub_company_insider_sentiment,
    },
    "get_insider_transactions": {
        "local": get_finnhub_company_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    logger.info(f"route_to_vendor called: method={method}, args={args}")

    try:
        category = get_category_for_method(method)
    except ValueError as e:
        logger.error(f"Method category lookup failed: {e}")
        raise

    vendor_config = get_vendor(category, method)
    logger.debug(f"Vendor config for {method}: category={category}, vendor_config={vendor_config}")

    # Handle comma-separated vendors
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        logger.error(f"Method '{method}' not in VENDOR_METHODS. Available: {list(VENDOR_METHODS.keys())}")
        raise ValueError(f"Method '{method}' not supported")

    # Get all available vendors for this method for fallback
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    
    # Create fallback vendor list: primary vendors first, then remaining vendors as fallbacks
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    # Track results and execution state
    results = []
    vendor_attempt_count = 0
    any_primary_vendor_attempted = False
    successful_vendor = None

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        is_primary_vendor = vendor in primary_vendors
        vendor_attempt_count += 1

        # Track if we attempted any primary vendor
        if is_primary_vendor:
            any_primary_vendor_attempted = True

        # Handle list of methods for a vendor
        if isinstance(vendor_impl, list):
            vendor_methods = [(impl, vendor) for impl in vendor_impl]
        else:
            vendor_methods = [(vendor_impl, vendor)]

        # Run methods for this vendor
        vendor_results = []
        for impl_func, vendor_name in vendor_methods:
            try:
                result = impl_func(*args, **kwargs)
                vendor_results.append(result)
                    
            except Exception as e:
                # Log error but continue with other implementations
                logger.error(f"Vendor implementation failed: {impl_func.__name__} from '{vendor_name}': {type(e).__name__}: {e}")
                continue

        # Add this vendor's results
        if vendor_results:
            results.extend(vendor_results)
            successful_vendor = vendor
            
            # Stopping logic: Stop after first successful vendor for single-vendor configs
            # Multiple vendor configs (comma-separated) may want to collect from multiple sources
            if len(primary_vendors) == 1:
                break

    # Final result summary
    if not results:
        logger.error(f"All vendor implementations failed for method '{method}'. Attempted vendors: {fallback_vendors}")
        raise RuntimeError(f"All vendor implementations failed for method '{method}'")

    # Return single result if only one, otherwise concatenate as string
    logger.info(f"route_to_vendor success: method={method}, vendor={successful_vendor}, results_count={len(results)}")
    if len(results) == 1:
        return results[0]
    else:
        # Convert all results to strings and concatenate
        return '\n'.join(str(result) for result in results)