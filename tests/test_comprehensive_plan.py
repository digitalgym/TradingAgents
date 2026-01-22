"""Test comprehensive SMC trading plan generation."""
import sys
import os

# Suppress Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    generate_smc_trading_plan
)
from tradingagents.dataflows.interface import route_to_vendor

def test_comprehensive_plan():
    """Test comprehensive trading plan with XAGUSD."""
    symbol = "XAGUSD"
    trade_date = "2026-01-19"

    print("\n" + "="*70)
    print(f"TESTING COMPREHENSIVE SMC TRADING PLAN - {symbol}")
    print("="*70)

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
    print(f"   Available timeframes: {list(smc_analysis.keys())}")

    # Get ATR
    print(f"\n2. Fetching ATR...")
    atr_data = route_to_vendor("get_indicators", symbol, "atr", trade_date, 14)
    atr_value = None
    if "ATR(14):" in atr_data:
        try:
            atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
            print(f"   ATR(14): {atr_value:.2f}")
        except:
            print(f"   ATR not found")

    # Test with BUY bias (overall bullish but may be at resistance)
    print(f"\n3. Generating comprehensive trading plan (BUY bias)...")
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias='BUY',
        primary_timeframe='1H',
        atr=atr_value
    )

    if 'error' in plan:
        print(f"ERROR: {plan['error']}")
        return

    print(f"\n{'='*70}")
    print("COMPREHENSIVE TRADING PLAN")
    print(f"{'='*70}")

    # Position Analysis
    pos = plan['position_analysis']
    print(f"\n[POSITION ANALYSIS]")
    print(f"Current Price: ${pos['current_price']:.2f}")

    if pos['at_resistance']:
        print(f"Position: AT RESISTANCE (This is the scenario from your chart!)")
    elif pos['at_support']:
        print(f"Position: AT SUPPORT")
    else:
        print(f"Position: BETWEEN ZONES")

    # Show nearest resistance
    if pos['nearest_resistance']:
        res = pos['nearest_resistance']
        print(f"\nNearest Resistance OB: ${res['price_range'][0]:.2f} - ${res['price_range'][1]:.2f}")
        print(f"  Distance: {res['distance_pct']:.2f}% away")
        if res['assessment']:
            assess = res['assessment']
            print(f"  Strength: {assess['strength_score']}/10 ({assess['strength_category']})")
            print(f"  Retests: {assess['retests']}")
            print(f"  Confluence: {assess['confluence_score']} ({', '.join(assess['aligned_timeframes'])})")
            print(f"  Volume: {assess['volume_profile']}")
            print(f"  Hold Probability: {assess['hold_probability']:.0%}")
            print(f"  Breakout Probability: {assess['breakout_probability']:.0%}")
            print(f"  Assessment: {assess['assessment']}")

    # Show nearest support
    if pos['nearest_support']:
        sup = pos['nearest_support']
        print(f"\nNearest Support OB: ${sup['price_range'][0]:.2f} - ${sup['price_range'][1]:.2f}")
        print(f"  Distance: {sup['distance_pct']:.2f}% away")
        if sup['assessment']:
            assess = sup['assessment']
            print(f"  Strength: {assess['strength_score']}/10 ({assess['strength_category']})")
            print(f"  Retests: {assess['retests']}")
            print(f"  Confluence: {assess['confluence_score']} ({', '.join(assess['aligned_timeframes'])})")
            print(f"  Volume: {assess['volume_profile']}")
            print(f"  Hold Probability: {assess['hold_probability']:.0%}")
            print(f"  Breakout Probability: {assess['breakout_probability']:.0%}")
            print(f"  Assessment: {assess['assessment']}")

    # Recommendation
    rec = plan['recommendation']
    print(f"\n{'='*70}")
    print(f"[RECOMMENDATION: {rec['action']}]")
    print(f"Confidence: {rec['confidence']}")
    print(f"{'='*70}")
    print(f"\nReason: {rec['reason']}")
    if rec.get('alternative'):
        print(f"\nAlternative: {rec['alternative']}")

    # Primary Setup
    if plan['primary_setup']:
        setup = plan['primary_setup']
        print(f"\n{'='*70}")
        print(f"[PRIMARY SETUP: {setup['direction']}]")
        print(f"{'='*70}")
        print(f"\nEntry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        if setup['entry_zone']:
            print(f"Entry Zone: ${setup['entry_zone'][0]:.2f} - ${setup['entry_zone'][1]:.2f}")
        print(f"\nStop Loss: ${setup['stop_loss']:.2f}")
        print(f"Reason: {setup['stop_loss_reason']}")
        print(f"\nTake Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"TP Reason: {setup['tp_reason']}")
        print(f"\nRISK/REWARD:")
        print(f"  Risk: {setup['risk_pct']:.2f}%")
        print(f"  Reward (TP1): {setup['reward_pct_tp1']:.2f}%")
        print(f"  Reward (TP2): {setup['reward_pct_tp2']:.2f}%")
        print(f"  R:R Ratio (TP1): 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f}")
        print(f"  R:R Ratio (TP2): 1:{(setup['reward_pct_tp2']/setup['risk_pct']):.2f}")
        print(f"\nRationale: {setup['rationale']}")

        if setup['ob_strength']:
            ob = setup['ob_strength']
            print(f"\nOrder Block Strength Details:")
            print(f"  Score: {ob['strength_score']}/10")
            print(f"  Category: {ob['strength_category']}")
            print(f"  {ob['assessment']}")

    # Alternative Setup
    if plan['alternative_setup']:
        setup = plan['alternative_setup']
        print(f"\n{'='*70}")
        print(f"[ALTERNATIVE SETUP: {setup['direction']}]")
        print(f"{'='*70}")

        if 'trigger_condition' in setup:
            print(f"\nTrigger Condition:")
            print(f"  {setup['trigger_condition']}")

        print(f"\nEntry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        if setup['entry_zone']:
            print(f"Entry Zone: ${setup['entry_zone'][0]:.2f} - ${setup['entry_zone'][1]:.2f}")
        print(f"\nStop Loss: ${setup['stop_loss']:.2f}")
        print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"\nRISK/REWARD:")
        print(f"  Risk: {setup['risk_pct']:.2f}%")
        print(f"  Reward (TP1): {setup['reward_pct_tp1']:.2f}%")
        print(f"  Reward (TP2): {setup['reward_pct_tp2']:.2f}%")
        print(f"  R:R Ratio (TP1): 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f}")
        print(f"\nRationale: {setup['rationale']}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\nThis plan addresses your exact scenario:")
    print("- Analyzes current price position relative to SMC zones")
    print("- Assesses order block strength (retests, confluence, breakout probability)")
    print("- Provides PRIMARY setup (e.g., SHORT at resistance)")
    print("- Provides ALTERNATIVE setup (e.g., LONG re-entry after pullback)")
    print("- Gives clear recommendation with confidence level")
    print("\nNow you have a complete plan, not just a single order!")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        test_comprehensive_plan()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
