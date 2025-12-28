"""Test xAI news and X sentiment functions."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta


def test_xai_news():
    """Test xAI web search news."""
    print("\n" + "="*60)
    print("Testing xAI Web Search News")
    print("="*60)
    
    from tradingagents.dataflows.xai import get_xai_news
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"Searching news for 'gold' from {start_date} to {end_date}...")
    
    try:
        result = get_xai_news("gold", start_date, end_date)
        print(f"✅ Got {len(result)} characters")
        print(f"\nPreview:\n{result[:500]}...")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_xai_global_news():
    """Test xAI global news."""
    print("\n" + "="*60)
    print("Testing xAI Global News")
    print("="*60)
    
    from tradingagents.dataflows.xai import get_xai_global_news
    
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Searching global news for past 7 days...")
    
    try:
        result = get_xai_global_news(curr_date, 7, 5)
        print(f"✅ Got {len(result)} characters")
        print(f"\nPreview:\n{result[:500]}...")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_x_sentiment():
    """Test X (Twitter) sentiment analysis."""
    print("\n" + "="*60)
    print("Testing X (Twitter) Sentiment Analysis")
    print("="*60)
    
    from tradingagents.dataflows.xai import get_x_sentiment
    
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Analyzing X sentiment for 'XAUUSD' (gold)...")
    
    try:
        result = get_x_sentiment("XAUUSD gold", curr_date, 7)
        print(f"✅ Got {len(result)} characters")
        print(f"\nSentiment Report:\n{result[:800]}...")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_x_news():
    """Test X (Twitter) news."""
    print("\n" + "="*60)
    print("Testing X (Twitter) News")
    print("="*60)
    
    from tradingagents.dataflows.xai import get_x_news
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"Searching X for 'silver' posts...")
    
    try:
        result = get_x_news("silver XAGUSD", start_date, end_date)
        print(f"✅ Got {len(result)} characters")
        print(f"\nPreview:\n{result[:500]}...")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_via_routing():
    """Test via the routing interface."""
    print("\n" + "="*60)
    print("Testing via Routing Interface (vendor=xai)")
    print("="*60)
    
    from tradingagents.dataflows.interface import route_to_vendor
    from tradingagents.dataflows.config import set_config
    
    # Set xai as news vendor
    set_config({
        "data_vendors": {
            "news_data": "xai",
        },
        "tool_vendors": {},
    })
    
    curr_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\nTesting get_global_news via xai vendor...")
    try:
        result = route_to_vendor("get_global_news", curr_date, 7, 5)
        print(f"✅ Got {len(result)} characters")
        print(f"Preview: {result[:300]}...")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY not set")
        exit(1)
    
    print(f"Using XAI_API_KEY: {api_key[:10]}...")
    
    # Test individual functions
    test_xai_news()
    test_xai_global_news()
    test_x_sentiment()
    test_x_news()
    
    # Test via routing
    test_via_routing()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)
