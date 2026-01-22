"""
Test script for MT5 integration with TradingAgents.
Tests commodity data fetching from Vantage MT5.
"""

import sys
sys.path.insert(0, '.')

from tradingagents.dataflows.mt5_data import (
    get_mt5_data,
    get_mt5_symbol_info,
    get_mt5_current_price,
    list_mt5_symbols,
    get_asset_type,
)

def test_mt5_connection():
    """Test basic MT5 data fetching."""
    print("=" * 60)
    print("Testing MT5 Integration")
    print("=" * 60)
    
    # Test 1: List commodity symbols
    print("\n1. Available commodity symbols:")
    commodities = list_mt5_symbols("XAU")
    print(f"   Gold symbols: {commodities}")
    
    silver = list_mt5_symbols("XAG")
    print(f"   Silver symbols: {silver}")
    
    platinum = list_mt5_symbols("XPT")
    print(f"   Platinum symbols: {platinum}")
    
    copper = list_mt5_symbols("COPPER")
    print(f"   Copper symbols: {copper}")
    
    # Test 2: Get symbol info
    print("\n2. Symbol info for XAUUSD:")
    try:
        info = get_mt5_symbol_info("XAUUSD")
        for key, value in info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Get current price
    print("\n3. Current prices:")
    for symbol in ["XAUUSD", "XAGUSD", "XPTUSD", "COPPER-C"]:
        try:
            price = get_mt5_current_price(symbol)
            print(f"   {symbol}: Bid={price['bid']}, Ask={price['ask']}")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
    
    # Test 4: Get historical OHLCV data
    print("\n4. Historical data for XAUUSD (last 5 days):")
    try:
        data = get_mt5_data("XAUUSD", "2024-12-20", "2024-12-27")
        print(data[:1000])  # Print first 1000 chars
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Test asset type detection
    print("\n5. Asset type detection:")
    test_symbols = ["XAUUSD", "XAGUSD", "EURUSD", "AAPL", "BTCUSD"]
    for symbol in test_symbols:
        asset_type = get_asset_type(symbol)
        print(f"   {symbol}: {asset_type}")
    
    # Test 6: Test alias resolution
    print("\n6. Alias resolution (using 'gold' instead of 'XAUUSD'):")
    try:
        data = get_mt5_data("gold", "2024-12-20", "2024-12-27")
        lines = data.split('\n')[:10]
        print('\n'.join(lines))
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("MT5 Integration Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_mt5_connection()
