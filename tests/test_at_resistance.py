"""Test comprehensive plan when price is AT resistance (like user's chart)."""
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    generate_smc_trading_plan
)

def test_at_resistance_scenario():
    """
    Simulate the exact scenario from user's chart:
    - Price at $93.18 (inside bearish order block)
    - Overall bias is BUY (bullish)
    - But current position is AT RESISTANCE
    - Need both SHORT (now) and LONG (re-entry) setups
    """
    symbol = "XAGUSD"

    print("\n" + "="*70)
    print("SCENARIO: PRICE AT RESISTANCE (User's Chart Scenario)")
    print("="*70)
    print("\nSituation:")
    print("- Overall bias: BULLISH (long-term)")
    print("- Current price: Inside bearish order block (resistance)")
    print("- User says: 'I agree with buy bias BUT we are at resistance'")
    print("- Need plan that addresses BOTH scenarios")
    print("\n" + "="*70)

    # Get real SMC data
    smc_analysis = analyze_multi_timeframe_smc(symbol, ['1H', '4H', 'D1'])

    if not smc_analysis or '1H' not in smc_analysis:
        print("ERROR: No SMC data")
        return

    # Get actual current price
    current_price = smc_analysis['1H']['current_price']
    print(f"\nActual current price: ${current_price:.2f}")

    # Check if we're at resistance
    bearish_obs = smc_analysis['1H'].get('order_blocks', {}).get('bearish', [])
    at_resistance = False

    for ob in bearish_obs:
        if ob.bottom <= current_price <= ob.top:
            at_resistance = True
            print(f"Price IS inside bearish OB: ${ob.bottom:.2f} - ${ob.top:.2f}")
            print(f"  Retests: {ob.retests if hasattr(ob, 'retests') else 'N/A'}")
            break

    if not at_resistance:
        # Find nearest resistance
        resistances_above = [ob for ob in bearish_obs if ob.bottom > current_price]
        if resistances_above:
            nearest = min(resistances_above, key=lambda ob: ob.bottom - current_price)
            dist_pct = ((nearest.bottom - current_price) / current_price * 100)
            print(f"Price is {dist_pct:.2f}% below nearest resistance: ${nearest.bottom:.2f} - ${nearest.top:.2f}")

    # Generate plan with BULLISH bias (overall)
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias='BUY',  # Overall bullish, BUT...
        primary_timeframe='1H',
        atr=5.0
    )

    if 'error' in plan:
        print(f"ERROR: {plan['error']}")
        return

    print(f"\n{'='*70}")
    print("COMPREHENSIVE MULTI-SCENARIO PLAN")
    print(f"{'='*70}")

    # Position
    pos = plan['position_analysis']
    print(f"\n[CURRENT POSITION]")
    print(f"Price: ${pos['current_price']:.2f}")

    if pos['at_resistance']:
        print("Status: AT RESISTANCE ZONE")
        print("  -> This validates user's observation!")
    elif pos['at_support']:
        print("Status: AT SUPPORT ZONE")
    else:
        print("Status: BETWEEN ZONES")

    # Show order block assessments
    print(f"\n[ORDER BLOCK STRENGTH ASSESSMENT]")

    if pos['nearest_resistance'] and pos['nearest_resistance']['assessment']:
        res = pos['nearest_resistance']['assessment']
        print(f"\nResistance OB Strength:")
        print(f"  Score: {res['strength_score']}/10 ({res['strength_category']})")
        print(f"  Retests: {res['retests']}")
        print(f"  Confluence: {res['confluence_score']} ({', '.join(res['aligned_timeframes'])})")
        print(f"  Hold Probability: {res['hold_probability']:.0%}")
        print(f"  Breakout Probability: {res['breakout_probability']:.0%}")
        print(f"\n  -> If hold probability >= 65%, system recommends SHORT")
        print(f"  -> If breakout probability >= 60%, system warns to wait")

    if pos['nearest_support'] and pos['nearest_support']['assessment']:
        sup = pos['nearest_support']['assessment']
        print(f"\nSupport OB Strength:")
        print(f"  Score: {sup['strength_score']}/10 ({sup['strength_category']})")
        print(f"  Retests: {sup['retests']}")
        print(f"  Confluence: {sup['confluence_score']} ({', '.join(sup['aligned_timeframes'])})")
        print(f"  Hold Probability: {sup['hold_probability']:.0%}")

    # Recommendation
    rec = plan['recommendation']
    print(f"\n{'='*70}")
    print(f"[SYSTEM RECOMMENDATION: {rec['action']}]")
    print(f"Confidence: {rec['confidence']}")
    print(f"{'='*70}")
    print(f"\n{rec['reason']}")
    if rec.get('alternative'):
        print(f"\nAlternative: {rec['alternative']}")

    # Show both setups
    print(f"\n{'='*70}")
    print("SCENARIO 1: PRIMARY SETUP")
    print(f"{'='*70}")

    if plan['primary_setup']:
        setup = plan['primary_setup']
        print(f"\nDirection: {setup['direction']}")
        print(f"Entry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        print(f"Stop Loss: ${setup['stop_loss']:.2f}")
        print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"\nRisk: {setup['risk_pct']:.2f}%")
        print(f"Reward (TP1): {setup['reward_pct_tp1']:.2f}% (R:R = 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f})")
        print(f"Reward (TP2): {setup['reward_pct_tp2']:.2f}% (R:R = 1:{(setup['reward_pct_tp2']/setup['risk_pct']):.2f})")
        print(f"\nRationale: {setup['rationale']}")

    print(f"\n{'='*70}")
    print("SCENARIO 2: ALTERNATIVE SETUP (Conditional)")
    print(f"{'='*70}")

    if plan['alternative_setup']:
        setup = plan['alternative_setup']
        print(f"\nDirection: {setup['direction']}")

        if 'trigger_condition' in setup:
            print(f"\nWHEN TO ENTER:")
            print(f"  {setup['trigger_condition']}")

        print(f"\nEntry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
        print(f"Stop Loss: ${setup['stop_loss']:.2f}")
        print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
        print(f"Take Profit 2: ${setup['take_profit_2']:.2f}")
        print(f"\nRisk: {setup['risk_pct']:.2f}%")
        print(f"Reward (TP1): {setup['reward_pct_tp1']:.2f}% (R:R = 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f})")
        print(f"\nRationale: {setup['rationale']}")
    else:
        print("\nNo alternative setup (only primary setup valid)")

    print(f"\n{'='*70}")
    print("WHAT THIS SOLVES")
    print(f"{'='*70}")
    print("\nUser's original problem:")
    print('  "Im between two minds, the overall bias is buy which i agree with,')
    print('   but the order suggested is based on the 1h smc order blocks,')
    print('   we need a plan not just an order"')
    print("\nSystem now provides:")
    print("  1. Position analysis (at resistance/support/between)")
    print("  2. Order block strength (retests, confluence, breakout probability)")
    print("  3. PRIMARY setup based on CURRENT position")
    print("  4. ALTERNATIVE setup for opposite scenario")
    print("  5. Clear recommendation with confidence level")
    print("\nExample complete plan:")
    print('  "SHORT now at resistance (TP at support)')
    print('   THEN re-enter LONG at support (TP at next resistance)"')
    print("\n  -> Addresses both short-term (SHORT) and long-term (LONG) views")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        test_at_resistance_scenario()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
