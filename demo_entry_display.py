"""Demonstrate the full entry strategy display with risk/reward comparison."""
import sys
import os

# Suppress Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    suggest_smc_entry_strategy,
    suggest_smc_stop_loss,
    suggest_smc_take_profits
)
from tradingagents.dataflows.interface import route_to_vendor

def demo_full_display():
    """Demonstrate complete entry strategy display with SL/TP for both entry types."""
    symbol = "XAGUSD"
    signal = "BUY"
    trade_date = "2026-01-19"

    print("\n" + "="*70)
    print(f"SMC ENTRY STRATEGY DEMONSTRATION - {symbol}")
    print("="*70)

    # Get SMC analysis
    print(f"\nAnalyzing {symbol}...")
    smc_analysis = analyze_multi_timeframe_smc(symbol, ['1H', '4H', 'D1'])

    # Get current price
    current_price = None
    for tf in ['1H', '4H', 'D1']:
        if tf in smc_analysis and 'current_price' in smc_analysis[tf]:
            current_price = smc_analysis[tf]['current_price']
            break

    print(f"\n{'='*60}")
    print("SMC-BASED TRADE LEVELS (Multi-Timeframe Confluence)")
    print(f"{'='*60}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Signal: {signal}")
    print(f"Available timeframes: {list(smc_analysis.keys())}")

    # Get entry strategy
    entry_strategy = suggest_smc_entry_strategy(
        smc_analysis=smc_analysis,
        direction=signal,
        current_price=current_price,
        primary_timeframe='1H'
    )

    print(f"\n{'='*60}")
    print("ENTRY STRATEGY")
    print(f"{'='*60}")

    # Display Market Entry option
    print(f"\n[OPTION 1: MARKET ORDER]")
    print(f"  Entry Price: ${entry_strategy['market_entry']['price']:.2f}")
    print(f"  Type: {entry_strategy['market_entry']['type']}")
    print(f"  Pros: Immediate execution, no risk of missing the trade")
    print(f"  Cons: Entering away from optimal SMC zone, higher risk")

    # Display Limit Entry option
    print(f"\n[OPTION 2: LIMIT ORDER AT ORDER BLOCK]")
    if entry_strategy['limit_entry'] and 'price' in entry_strategy['limit_entry']:
        limit_entry = entry_strategy['limit_entry']
        print(f"  Entry Price: ${limit_entry['price']:.2f}")
        print(f"  Entry Zone: ${limit_entry['zone_bottom']:.2f} - ${limit_entry['zone_top']:.2f}")
        print(f"  Type: {limit_entry['type']}")
        print(f"  Confluence: {limit_entry['confluence_score']:.1f} ({', '.join(limit_entry['aligned_timeframes'])})")
        print(f"  Pros: Better risk/reward, entering at institutional zone")
        print(f"  Cons: Price may not return to zone, could miss the trade")
        print(f"  Reason: {limit_entry['reason']}")
    else:
        print(f"  {entry_strategy['limit_entry']['reason']}")

    # Display Recommendation
    print(f"\n[RECOMMENDATION: {entry_strategy['recommendation']}]")
    if 'recommendation_reason' in entry_strategy:
        print(f"  {entry_strategy['recommendation_reason']}")
        if entry_strategy['distance_to_zone_pct'] is not None:
            print(f"  Distance to optimal zone: {entry_strategy['distance_to_zone_pct']:.2f}%")

    # Get ATR
    print(f"\n{'='*60}")
    atr_data = route_to_vendor("get_indicators", symbol, "atr", trade_date, 14)
    atr_value = None
    if "ATR(14):" in atr_data:
        try:
            atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
            print(f"ATR(14): {atr_value:.2f}")
        except:
            pass

    # Show risk management levels for both entry types
    show_limit_levels = (entry_strategy['limit_entry'] and
                        'price' in entry_strategy['limit_entry'] and
                        entry_strategy['recommendation'] in ['LIMIT', 'LIMIT_OR_MARKET'])

    # Get stop loss for market entry
    stop_suggestion_market = suggest_smc_stop_loss(
        smc_analysis=smc_analysis,
        direction=signal,
        entry_price=current_price,
        atr=atr_value,
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    print(f"\n{'='*60}")
    print("RISK MANAGEMENT LEVELS")
    print(f"{'='*60}")

    # Show Stop Loss for Market Entry
    print(f"\n--- Stop Loss (if entering at MARKET: ${current_price:.2f}) ---")
    if stop_suggestion_market:
        print(f"  SL Price: ${stop_suggestion_market['price']:.2f}")
        print(f"  Zone: ${stop_suggestion_market['zone_bottom']:.2f} - ${stop_suggestion_market['zone_top']:.2f}")
        print(f"  Risk: ${abs(current_price - stop_suggestion_market['price']):.2f} ({stop_suggestion_market['distance_pct']:.2f}%)")
        if 'confluence_score' in stop_suggestion_market and stop_suggestion_market['confluence_score'] > 1.0:
            print(f"  Confluence: {stop_suggestion_market['confluence_score']:.1f} ({', '.join(stop_suggestion_market['aligned_timeframes'])})")

    # Show Stop Loss for Limit Entry
    if show_limit_levels:
        limit_entry_price = entry_strategy['limit_entry']['price']
        stop_suggestion_limit = suggest_smc_stop_loss(
            smc_analysis=smc_analysis,
            direction=signal,
            entry_price=limit_entry_price,
            atr=atr_value,
            atr_multiplier=2.0,
            primary_timeframe='1H'
        )

        print(f"\n--- Stop Loss (if entering at LIMIT: ${limit_entry_price:.2f}) ---")
        if stop_suggestion_limit:
            print(f"  SL Price: ${stop_suggestion_limit['price']:.2f}")
            print(f"  Zone: ${stop_suggestion_limit['zone_bottom']:.2f} - ${stop_suggestion_limit['zone_top']:.2f}")
            print(f"  Risk: ${abs(limit_entry_price - stop_suggestion_limit['price']):.2f} ({stop_suggestion_limit['distance_pct']:.2f}%)")
            if 'confluence_score' in stop_suggestion_limit and stop_suggestion_limit['confluence_score'] > 1.0:
                print(f"  Confluence: {stop_suggestion_limit['confluence_score']:.1f} ({', '.join(stop_suggestion_limit['aligned_timeframes'])})")

            # Show risk comparison
            market_risk = abs(current_price - stop_suggestion_market['price']) if stop_suggestion_market else 0
            limit_risk = abs(limit_entry_price - stop_suggestion_limit['price'])
            if market_risk > 0:
                risk_reduction = ((market_risk - limit_risk) / market_risk * 100)
                print(f"  >>> RISK REDUCTION: {risk_reduction:.1f}% vs market entry <<<")

    # Get take profit suggestions
    first_tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
    tp_suggestions_market = suggest_smc_take_profits(
        smc_analysis=first_tf_data if first_tf_data else {},
        direction=signal,
        entry_price=current_price,
        num_targets=3
    )

    print(f"\n--- Take Profit Targets (if entering at MARKET: ${current_price:.2f}) ---")
    if tp_suggestions_market:
        for i, tp in enumerate(tp_suggestions_market, 1):
            print(f"  TP{i}: ${tp['price']:.2f} ({tp['source']}) | +{tp['distance_pct']:.2f}%")
    else:
        print("  No suitable SMC zones found")

    # Show TP for Limit Entry
    if show_limit_levels:
        limit_entry_price = entry_strategy['limit_entry']['price']
        tp_suggestions_limit = suggest_smc_take_profits(
            smc_analysis=first_tf_data if first_tf_data else {},
            direction=signal,
            entry_price=limit_entry_price,
            num_targets=3
        )

        print(f"\n--- Take Profit Targets (if entering at LIMIT: ${limit_entry_price:.2f}) ---")
        if tp_suggestions_limit:
            for i, tp in enumerate(tp_suggestions_limit, 1):
                print(f"  TP{i}: ${tp['price']:.2f} ({tp['source']}) | +{tp['distance_pct']:.2f}%")

                # Show reward comparison
                if i <= len(tp_suggestions_market):
                    market_reward = abs(tp_suggestions_market[i-1]['price'] - current_price)
                    limit_reward = abs(tp['price'] - limit_entry_price)
                    reward_increase = ((limit_reward - market_reward) / market_reward * 100) if market_reward > 0 else 0
                    if reward_increase > 0:
                        print(f"       >>> REWARD INCREASE: +{reward_increase:.1f}% vs market entry <<<")

    # Summary comparison
    if show_limit_levels and stop_suggestion_market and stop_suggestion_limit:
        print(f"\n{'='*60}")
        print("RISK/REWARD COMPARISON SUMMARY")
        print(f"{'='*60}")

        market_risk = abs(current_price - stop_suggestion_market['price'])
        limit_risk = abs(limit_entry_price - stop_suggestion_limit['price'])
        risk_reduction = ((market_risk - limit_risk) / market_risk * 100)

        if tp_suggestions_market and tp_suggestions_limit:
            market_reward_tp1 = abs(tp_suggestions_market[0]['price'] - current_price)
            limit_reward_tp1 = abs(tp_suggestions_limit[0]['price'] - limit_entry_price)
            reward_increase_tp1 = ((limit_reward_tp1 - market_reward_tp1) / market_reward_tp1 * 100)

            market_rr = market_reward_tp1 / market_risk if market_risk > 0 else 0
            limit_rr = limit_reward_tp1 / limit_risk if limit_risk > 0 else 0

            print(f"\nMARKET ENTRY (${current_price:.2f}):")
            print(f"  Risk: ${market_risk:.2f} | Reward (TP1): ${market_reward_tp1:.2f}")
            print(f"  Risk/Reward Ratio: 1:{market_rr:.2f}")

            print(f"\nLIMIT ENTRY (${limit_entry_price:.2f}):")
            print(f"  Risk: ${limit_risk:.2f} | Reward (TP1): ${limit_reward_tp1:.2f}")
            print(f"  Risk/Reward Ratio: 1:{limit_rr:.2f}")

            print(f"\nIMPROVEMENT WITH LIMIT ORDER:")
            print(f"  Risk Reduction: {risk_reduction:.1f}%")
            print(f"  Reward Increase: +{reward_increase_tp1:.1f}%")
            print(f"  R/R Improvement: {((limit_rr - market_rr) / market_rr * 100):.1f}%")

    print(f"\n{'='*70}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        demo_full_display()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
