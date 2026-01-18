"""
Example: Execute Trading Plan with Intelligent Order Management

This demonstrates the complete workflow:
1. Parse trading plan from analyst output
2. Decide market vs limit orders
3. Execute staged entries
4. Manage dynamic stops
5. Review and adapt plan

Usage:
    python examples/execute_plan_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.execution import (
    TradingPlanParser,
    OrderExecutor,
    StagedEntryManager,
    DynamicStopManager,
    PlanReviewer
)


# Example plan text (from your CLI analyze output)
EXAMPLE_PLAN = """
3. Refined Trader's Plan

Starting from the original plan (3-5% sizing to 7-8% max, staged $75-83
entries on RSI<65/MACD bullish/volume>80M, dynamic trailing stops from
$70/2-2.5x ATR, scale-outs at $87/$100/$108+, monitor
DXY/sentiment/ETF/geo/industrials, full exit <$70/RSI<40, 1-2% gold
hedge), refinements integrate analysts' insights (Neutral's
hedges/sizing cap, Safe's vol buffers, Risky's macro triggers) + gold
lessons (vol-adjusted sizing down from 7-8% max, tighter DXY/geo
monitoring, enhanced technicals):

 • Position Sizing: Cap at 4-5% total (Neutral's moderate vs. original
   7-8%/Risky's all-in; Safe's 2% too timid). Vol-adjusted: Start
   2.5-3% initial (gold's holiday tweak for 60%+ vol), scale +1-1.5%
   per tranche on confirms. Lessons: Prevents whipsaws like gold's
   near-stop-out.
 • Entry (Staged on $75-83 Dips):
    • Tranche 1: 2% at $77-80 (current test, volume>80M + RSI<65
      cooldown).
    • Tranche 2: 1.5% at $75 (MACD bullish hold + DXY<104, per
      Neutral).
    • Tranche 3: 1% at $83 retest/breakout (geo escalation confirm,
      e.g., Iran/Venezuela via Kitco).
    • No chase: Wait post-overbought like gold dip-timing win.
 • Stop-Loss: Dynamic trailing, 1.5-2x ATR (~$72 initial from $75-83,
   trails to $75 post-$82 close/$78 post-$87). Firmer than Risky's
   loose but wider than Safe's rigid $70 (avoids gold's "razor-edge"
   clip). Full exit <$72 high-volume or RSI<40 sustained.
 • Profit-Taking: Surgical scale-out for 15-25% avg return (realistic
   per Neutral):
    • 30% at $87 (10-15% quick win, first resistance).
    • 30% at $100 (consensus short-term).
    • Trail 40% to $108-125 (Risky's stretch, but trim 20% if
      sentiment>95% euphoria or DXY+1%).
    • Partial pre-events: 25-50% trim on Fed speeches/geo de-escalation
      (Iran fizzle/Powell clear).
 • Monitoring Triggers (Enhanced from gold lessons: Real-time alerts,
   quantitative sentiment):
    • Daily: DXY (trim 25% on +1% surge, tightened from 1.5%); ETF
      flows/COMEX (bullish if drops); sentiment heatmaps (exit partial
      >95% euphoria).
    • Geo/Events: Kitco/Investing.com alerts—add on Venezuela/Iran;
      scale out on de-escalation.
    • Fundamentals: Weekly EV/solar data (scale out on H2 softening);
      industrial installs for demand cracks.
    • Hedge: 1.5-2% gold (XAUUSD ETF) from start (Neutral/Safe boost
      vs. original 1-2%).
 • Exit Contingencies: Full out on $72 break/high vol (bear confirm);
   RSI<40 sustained; or industrial reversal (e.g., China supply dump).

This plan turns Safe's "vol tax" into edge (buffers/hedges), captures
Risky's upside (macro scaling), and Neutral's balance—projected 20%+
portfolio lift by Q2 if catalysts hold, like gold's geo-fueled 18.2%.
"""


def main():
    """Demonstrate intelligent plan execution"""
    
    print("\n" + "="*70)
    print("INTELLIGENT TRADING PLAN EXECUTION")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: Parse the Trading Plan
    # ========================================================================
    print("STEP 1: PARSING TRADING PLAN")
    print("-" * 70)
    
    parser = TradingPlanParser()
    plan = parser.parse_plan(EXAMPLE_PLAN, symbol="COPPER-C")
    
    # Display parsed plan
    print(parser.format_plan_summary(plan))
    
    # ========================================================================
    # STEP 2: Decide Order Types for Each Tranche
    # ========================================================================
    print("\nSTEP 2: DECIDING ORDER TYPES")
    print("-" * 70)
    
    executor = OrderExecutor("COPPER-C")
    
    # Assume current price is $78.50
    current_price = 78.50
    print(f"Current Market Price: ${current_price:.2f}\n")
    
    for tranche in plan.entry_tranches:
        decision = executor.decide_order_type(
            direction=plan.direction,
            target_price=tranche.price_level,
            price_range=tranche.price_range,
            conditions=tranche.conditions
        )
        
        print(f"Tranche {tranche.tranche_number}:")
        print(f"  Target: ${tranche.price_level:.2f}")
        if tranche.price_range:
            print(f"  Range: ${tranche.price_range[0]:.2f}-${tranche.price_range[1]:.2f}")
        print(f"  Order Type: {decision.order_type.value.upper()}")
        print(f"  Urgency: {decision.urgency}")
        print(f"  Reason: {decision.reason}\n")
    
    # ========================================================================
    # STEP 3: Initialize Staged Entry Manager
    # ========================================================================
    print("\nSTEP 3: INITIALIZING STAGED ENTRY MANAGER")
    print("-" * 70)
    
    plan_id = f"COPPER-C_{plan.direction}_20260113"
    staged_manager = StagedEntryManager(plan_id)
    staged_manager.initialize_tranches(plan.entry_tranches)
    
    print(staged_manager.format_status_report())
    
    # ========================================================================
    # STEP 4: Execute First Tranche (Example)
    # ========================================================================
    print("\nSTEP 4: EXECUTING FIRST TRANCHE")
    print("-" * 70)
    
    next_tranche = staged_manager.get_next_tranche()
    if next_tranche:
        print(f"Executing Tranche {next_tranche.tranche_number}:")
        print(f"  Size: {next_tranche.size_pct}%")
        print(f"  Target: ${next_tranche.target_price:.2f}")
        
        # Get order decision
        decision = executor.decide_order_type(
            direction=plan.direction,
            target_price=next_tranche.target_price,
            price_range=next_tranche.price_range
        )
        
        print(f"  Order Type: {decision.order_type.value.upper()}")
        print(f"  Price: ${decision.price:.2f}")
        
        # In real execution, you would call:
        # account_balance = 10000
        # volume = (next_tranche.size_pct / 100) * account_balance / decision.price
        # result = executor.execute_order(
        #     direction=plan.direction,
        #     volume=volume,
        #     order_decision=decision,
        #     stop_loss=plan.stop_loss.initial_price,
        #     comment=f"Tranche {next_tranche.tranche_number}"
        # )
        
        # Simulate fill
        print(f"\n  ✅ Order placed (simulated)")
        print(f"  Order Ticket: #12345")
        
        # Mark as active
        staged_manager.mark_tranche_active(next_tranche.tranche_number, 12345)
        
        # Simulate fill
        staged_manager.mark_tranche_filled(
            next_tranche.tranche_number,
            filled_price=78.50,
            filled_volume=0.02
        )
        
        print(f"  ✅ Tranche filled at ${78.50:.2f}")
    
    print("\n" + staged_manager.format_status_report())
    
    # ========================================================================
    # STEP 5: Setup Dynamic Stop Management
    # ========================================================================
    print("\nSTEP 5: SETTING UP DYNAMIC STOPS")
    print("-" * 70)
    
    stop_manager = DynamicStopManager(symbol="COPPER-C", direction=plan.direction)
    
    # Set initial stop
    entry_price = 78.50
    stop_manager.set_initial_stop(
        entry_price=entry_price,
        stop_price=plan.stop_loss.initial_price
    )
    
    # Parse and add trailing rules
    stop_manager.parse_trailing_rules(plan.stop_loss.trail_rules)
    
    print(stop_manager.format_status_report(current_price, entry_price))
    
    # ========================================================================
    # STEP 6: Monitor and Update (Simulated Price Movement)
    # ========================================================================
    print("\nSTEP 6: MONITORING POSITION (Simulated)")
    print("-" * 70)
    
    # Simulate price moving to $82
    new_price = 82.00
    print(f"\nPrice moved to ${new_price:.2f}")
    
    # Check for stop updates
    update_result = stop_manager.check_and_update(new_price)
    
    if update_result['updated']:
        print(f"\n🔄 Stop Loss Updated!")
        for update in update_result['updates']:
            print(f"  Rule: {update['rule']}")
            print(f"  Old Stop: ${update['old_stop']:.2f}")
            print(f"  New Stop: ${update['new_stop']:.2f}")
            
            # In real execution:
            # stop_manager.apply_stop_to_position(ticket=12345)
    else:
        print(f"\n✓ No stop updates triggered yet")
    
    print(stop_manager.format_status_report(new_price, entry_price))
    
    # ========================================================================
    # STEP 7: Review Plan
    # ========================================================================
    print("\nSTEP 7: REVIEWING PLAN")
    print("-" * 70)
    
    reviewer = PlanReviewer()
    
    review = reviewer.review_plan(
        plan=plan,
        current_price=new_price,
        entry_price=entry_price,
        position_size=staged_manager.total_filled_pct,
        time_in_trade=2  # 2 days in trade
    )
    
    print(reviewer.format_review_report(review))
    
    # ========================================================================
    # STEP 8: Check Staged Entry Adjustments
    # ========================================================================
    print("\nSTEP 8: CHECKING STAGED ENTRY ADJUSTMENTS")
    print("-" * 70)
    
    adjustment_check = staged_manager.should_adjust_remaining(new_price)
    
    if adjustment_check['should_adjust']:
        print(f"⚠️  {adjustment_check['reason']}")
        print(f"   {adjustment_check['recommendation']}")
    else:
        print(f"✓ Remaining tranches on track")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    
    summary = staged_manager.get_status_summary()
    
    print(f"\nPosition Status:")
    print(f"  Symbol: {plan.symbol} {plan.direction}")
    print(f"  Tranches Filled: {summary['filled']}/{summary['total_tranches']}")
    print(f"  Position Size: {summary['total_filled_pct']:.1f}%")
    print(f"  Avg Entry: ${summary['avg_entry_price']:.2f}")
    print(f"  Current Price: ${new_price:.2f}")
    print(f"  Current Stop: ${stop_manager.current_stop:.2f}")
    
    # Calculate profit
    profit_pct = ((new_price - entry_price) / entry_price) * 100
    print(f"  Unrealized P&L: {profit_pct:+.2f}%")
    
    print(f"\nNext Actions:")
    print(f"  • Monitor for Tranche 2 entry at ${plan.entry_tranches[1].price_level:.2f}")
    print(f"  • Watch for stop trail trigger at ${82.00:.2f}")
    print(f"  • Prepare for TP1 at ${plan.take_profits[0].price_level:.2f}")
    
    if review['high_priority']:
        print(f"\n⚠️  High Priority Adjustments: {len(review['high_priority'])}")
        for adj in review['high_priority']:
            print(f"  • {adj.reason}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
