"""Test entry strategy display with all three perspectives."""
import sys
import os

# Suppress Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    suggest_smc_entry_strategy
)

def test_entry_strategy():
    """Test the entry strategy suggestions for all three scenarios."""
    print("\n" + "="*70)
    print("TESTING ENTRY STRATEGY SUGGESTIONS")
    print("="*70)

    symbol = "XAGUSD"

    # Get SMC analysis
    print(f"\n1. Running SMC analysis for {symbol}...")
    smc_analysis = analyze_multi_timeframe_smc(symbol, ['1H', '4H', 'D1'])

    if not smc_analysis:
        print("ERROR: No SMC analysis data")
        return

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

    # Test BUY entry strategy
    print(f"\n2. Testing BUY entry strategy...")
    entry_strategy = suggest_smc_entry_strategy(
        smc_analysis=smc_analysis,
        direction='BUY',
        current_price=current_price,
        primary_timeframe='1H'
    )

    print(f"\n{'='*70}")
    print("BUY ENTRY STRATEGY RESULTS")
    print(f"{'='*70}")

    # Market Entry
    print(f"\n[MARKET ENTRY]")
    print(f"  Price: ${entry_strategy['market_entry']['price']:.2f}")
    print(f"  Type: {entry_strategy['market_entry']['type']}")

    # Limit Entry
    print(f"\n[LIMIT ENTRY]")
    if entry_strategy['limit_entry'] and 'price' in entry_strategy['limit_entry']:
        limit = entry_strategy['limit_entry']
        print(f"  Price: ${limit['price']:.2f}")
        print(f"  Zone: ${limit['zone_bottom']:.2f} - ${limit['zone_top']:.2f}")
        print(f"  Type: {limit['type']}")
        print(f"  Confluence: {limit['confluence_score']:.1f}")
        print(f"  Aligned TFs: {', '.join(limit['aligned_timeframes'])}")
        print(f"  Reason: {limit['reason']}")
    else:
        print(f"  {entry_strategy['limit_entry']['reason']}")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print(f"  Strategy: {entry_strategy['recommendation']}")
    if 'recommendation_reason' in entry_strategy:
        print(f"  Reason: {entry_strategy['recommendation_reason']}")
    if entry_strategy['distance_to_zone_pct'] is not None:
        print(f"  Distance to zone: {entry_strategy['distance_to_zone_pct']:.2f}%")

    # Test SELL entry strategy
    print(f"\n\n3. Testing SELL entry strategy...")
    entry_strategy_sell = suggest_smc_entry_strategy(
        smc_analysis=smc_analysis,
        direction='SELL',
        current_price=current_price,
        primary_timeframe='1H'
    )

    print(f"\n{'='*70}")
    print("SELL ENTRY STRATEGY RESULTS")
    print(f"{'='*70}")

    # Market Entry
    print(f"\n[MARKET ENTRY]")
    print(f"  Price: ${entry_strategy_sell['market_entry']['price']:.2f}")
    print(f"  Type: {entry_strategy_sell['market_entry']['type']}")

    # Limit Entry
    print(f"\n[LIMIT ENTRY]")
    if entry_strategy_sell['limit_entry'] and 'price' in entry_strategy_sell['limit_entry']:
        limit = entry_strategy_sell['limit_entry']
        print(f"  Price: ${limit['price']:.2f}")
        print(f"  Zone: ${limit['zone_bottom']:.2f} - ${limit['zone_top']:.2f}")
        print(f"  Type: {limit['type']}")
        print(f"  Confluence: {limit['confluence_score']:.1f}")
        print(f"  Aligned TFs: {', '.join(limit['aligned_timeframes'])}")
        print(f"  Reason: {limit['reason']}")
    else:
        print(f"  {entry_strategy_sell['limit_entry']['reason']}")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print(f"  Strategy: {entry_strategy_sell['recommendation']}")
    if 'recommendation_reason' in entry_strategy_sell:
        print(f"  Reason: {entry_strategy_sell['recommendation_reason']}")
    if entry_strategy_sell['distance_to_zone_pct'] is not None:
        print(f"  Distance to zone: {entry_strategy_sell['distance_to_zone_pct']:.2f}%")

    print(f"\n{'='*70}")
    print("TEST COMPLETE - Entry strategy working!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        test_entry_strategy()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
