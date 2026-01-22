"""
Test script for news data fetching from various vendors.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.dataflows.interface import route_to_vendor, VENDOR_METHODS
from tradingagents.dataflows.config import set_config

# Test configuration
TEST_CONFIG = {
    "data_vendors": {
        "news_data": "google",  # Test with google first
    },
    "tool_vendors": {},
}

def test_get_news(ticker="XAUUSD", days_back=7):
    """Test get_news for a specific ticker."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Testing get_news for {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    try:
        result = route_to_vendor("get_news", ticker, start_date, end_date)
        if result:
            print(f"✅ Success! Got {len(result)} characters of news")
            print(f"\nPreview (first 500 chars):\n{result[:500]}...")
        else:
            print("⚠️ No news returned (empty result)")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_get_global_news(days_back=7, limit=5):
    """Test get_global_news."""
    from datetime import datetime
    
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Testing get_global_news")
    print(f"Current date: {curr_date}, Look back: {days_back} days, Limit: {limit}")
    print(f"{'='*60}")
    
    try:
        result = route_to_vendor("get_global_news", curr_date, days_back, limit)
        if result and result != "No global news found.":
            print(f"✅ Success! Got {len(result)} characters of news")
            print(f"\nPreview (first 500 chars):\n{result[:500]}...")
        else:
            print("⚠️ No global news returned")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_google_news_direct():
    """Test Google news scraping directly."""
    print(f"\n{'='*60}")
    print("Testing Google News scraping directly")
    print(f"{'='*60}")
    
    try:
        from tradingagents.dataflows.google import get_google_news, get_google_global_news
        from datetime import datetime
        
        curr_date = datetime.now().strftime("%Y-%m-%d")
        
        # Test ticker news
        print("\n1. Testing get_google_news('gold price', curr_date, 7)...")
        result = get_google_news("gold price", curr_date, 7)
        if result:
            print(f"   ✅ Got {len(result)} chars")
            print(f"   Preview: {result[:200]}...")
        else:
            print("   ⚠️ Empty result")
        
        # Test global news
        print("\n2. Testing get_google_global_news(curr_date, 7, 5)...")
        result = get_google_global_news(curr_date, 7, 5)
        if result and result != "No global news found.":
            print(f"   ✅ Got {len(result)} chars")
            print(f"   Preview: {result[:200]}...")
        else:
            print("   ⚠️ Empty result")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_alpha_vantage_news():
    """Test Alpha Vantage news (requires ALPHA_VANTAGE_API_KEY)."""
    print(f"\n{'='*60}")
    print("Testing Alpha Vantage News")
    print(f"{'='*60}")
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("⚠️ ALPHA_VANTAGE_API_KEY not set, skipping")
        return
    
    try:
        from tradingagents.dataflows.alpha_vantage import get_news
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        print(f"Testing get_news('XAUUSD', '{start_date}', '{end_date}')...")
        result = get_news("XAUUSD", start_date, end_date)
        if result:
            print(f"✅ Got {len(result)} chars")
            print(f"Preview: {result[:300]}...")
        else:
            print("⚠️ Empty result")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def list_available_vendors():
    """List all available news vendors."""
    print(f"\n{'='*60}")
    print("Available News Vendors")
    print(f"{'='*60}")
    
    print("\nget_news vendors:")
    for vendor in VENDOR_METHODS.get("get_news", {}).keys():
        print(f"  - {vendor}")
    
    print("\nget_global_news vendors:")
    for vendor in VENDOR_METHODS.get("get_global_news", {}).keys():
        print(f"  - {vendor}")


def main():
    print("News Data Test Script")
    print("=" * 60)
    
    # Set test config
    set_config(TEST_CONFIG)
    
    # List available vendors
    list_available_vendors()
    
    # Test Google news directly
    test_google_news_direct()
    
    # Test via routing
    test_get_news("gold", 7)
    test_get_global_news(7, 5)
    
    # Test Alpha Vantage if available
    test_alpha_vantage_news()
    
    print(f"\n{'='*60}")
    print("Tests complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
