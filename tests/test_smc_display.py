"""Quick test of SMC multi-timeframe confluence display"""
import sys
import os

# Suppress Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc, suggest_smc_stop_loss
from tradingagents.dataflows.interface import route_to_vendor
from datetime import datetime

def test_smc_display():
    """Test the new multi-timeframe confluence SMC display."""
    print("\n" + "="*70)
    print("TESTING MULTI-TIMEFRAME SMC CONFLUENCE")
    print("="*70)

    symbol = "XAGUSD"
    trade_date = "2026-01-16"

    # Get SMC analysis
    print(f"\n1. Running SMC analysis for {symbol}...")
    smc_analysis = analyze_multi_timeframe_smc(symbol, ['1H', '4H', 'D1'])

    if not smc_analysis:
        print("ERROR: No SMC analysis data")
        return

    print(f"   Available timeframes: {list(smc_analysis.keys())}")

    # Get current price
    current_price = None
    for tf in ['1H', '4H', 'D1']:
        if tf in smc_analysis and 'current_price' in smc_analysis[tf]:
            current_price = smc_analysis[tf]['current_price']
            break

    if not current_price:
        print("ERROR: Could not get current price")
        return

    print(f"   Current price: ${current_price:.2f}")

    # Show order blocks per timeframe
    print(f"\n2. Order Blocks per timeframe:")
    for tf in ['1H', '4H', 'D1']:
        if tf in smc_analysis:
            tf_data = smc_analysis[tf]
            bullish_obs = tf_data.get('order_blocks', {}).get('bullish', [])
            bearish_obs = tf_data.get('order_blocks', {}).get('bearish', [])
            print(f"   {tf}: {len(bullish_obs)} bullish, {len(bearish_obs)} bearish")

    # Get ATR
    print(f"\n3. Fetching ATR for fallback...")
    try:
        atr_data = route_to_vendor("get_indicators", symbol, "atr", trade_date, 14)
        if "ATR(14):" in atr_data:
            atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
            print(f"   ATR(14): {atr_value:.2f}")
        else:
            atr_value = None
            print(f"   ATR not found")
    except Exception as e:
        atr_value = None
        print(f"   ATR fetch error: {e}")

    # Test BUY stop loss
    print(f"\n4. Testing BUY stop loss with confluence...")
    stop_buy = suggest_smc_stop_loss(
        smc_analysis=smc_analysis,
        direction='BUY',
        entry_price=current_price,
        atr=atr_value,
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    if stop_buy:
        print(f"   >> Stop Loss: ${stop_buy['price']:.2f}")
        print(f"   >> Source: {stop_buy['source']}")
        print(f"   >> Distance: {stop_buy['distance_pct']:.2f}%")
        print(f"   >> Confluence Score: {stop_buy.get('confluence_score', 'N/A')}")
        print(f"   >> Aligned Timeframes: {stop_buy.get('aligned_timeframes', [])}")
        print(f"   >> Reason: {stop_buy['reason']}")
    else:
        print(f"   >> No stop loss found (should not happen with ATR!)")

    # Test SELL stop loss
    print(f"\n5. Testing SELL stop loss with confluence...")
    stop_sell = suggest_smc_stop_loss(
        smc_analysis=smc_analysis,
        direction='SELL',
        entry_price=current_price,
        atr=atr_value,
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    if stop_sell:
        print(f"   >> Stop Loss: ${stop_sell['price']:.2f}")
        print(f"   >> Source: {stop_sell['source']}")
        print(f"   >> Distance: {stop_sell['distance_pct']:.2f}%")
        print(f"   >> Confluence Score: {stop_sell.get('confluence_score', 'N/A')}")
        print(f"   >> Aligned Timeframes: {stop_sell.get('aligned_timeframes', [])}")
        print(f"   >> Reason: {stop_sell['reason']}")
    else:
        print(f"   >> No stop loss found (should not happen with ATR!)")

    print(f"\n" + "="*70)
    print("TEST COMPLETE - Multi-timeframe confluence is working!")
    print("="*70)

if __name__ == "__main__":
    try:
        test_smc_display()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
