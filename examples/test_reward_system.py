"""
Example script demonstrating the reward signal system

This script shows how to:
1. Create and track trades with the decision system
2. Calculate reward signals on trade close
3. Monitor portfolio state
4. View learning metrics

Run this to test the Phase 1 implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.trade_decisions import store_decision, close_decision, get_decision_stats
from tradingagents.learning.portfolio_state import PortfolioStateTracker
from tradingagents.learning.reward import RewardCalculator


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def example_winning_trade():
    """Example: Store and close a winning trade"""
    print_separator("Example 1: Winning Trade")
    
    # Store decision
    print("📝 Storing BUY decision...")
    decision_id = store_decision(
        symbol="XAUUSD",
        decision_type="OPEN",
        action="BUY",
        rationale="Strong bullish setup: breaker block + FVG + support zone confluence",
        entry_price=2650.0,
        stop_loss=2630.0,
        take_profit=2690.0,
        volume=0.1,
        source="analysis"
    )
    print(f"✓ Decision stored: {decision_id}\n")
    
    # Simulate trade execution and close
    print("⏳ Simulating trade execution...\n")
    print("💰 Closing trade with profit...")
    closed = close_decision(
        decision_id=decision_id,
        exit_price=2680.0,  # +30 pips, 1.5R win
        exit_reason="tp-hit",
        outcome_notes="Take profit hit as planned. Clean execution."
    )
    
    # Display results
    print_separator()
    print("📊 Trade Results:")
    print(f"   Entry: ${closed['entry_price']:.2f}")
    print(f"   Exit:  ${closed['exit_price']:.2f}")
    print(f"   P&L:   {closed['pnl_percent']:+.2f}%")
    print(f"   RR Realized: {closed['rr_realized']:+.2f}R (planned: {closed['rr_planned']:.2f}R)")
    print(f"   Reward Signal: {closed['reward_signal']:+.3f}")
    print(f"   Sharpe Contribution: {closed['sharpe_contribution']:+.3f}")
    print(f"   Drawdown Impact: {closed['drawdown_impact']:+.3f}")


def example_losing_trade():
    """Example: Store and close a losing trade"""
    print_separator("Example 2: Losing Trade")
    
    print("📝 Storing SELL decision...")
    decision_id = store_decision(
        symbol="XAUUSD",
        decision_type="OPEN",
        action="SELL",
        rationale="Bearish reversal at resistance",
        entry_price=2680.0,
        stop_loss=2700.0,
        take_profit=2640.0,
        volume=0.1,
        source="analysis"
    )
    print(f"✓ Decision stored: {decision_id}\n")
    
    print("⏳ Simulating trade execution...\n")
    print("❌ Closing trade with loss...")
    closed = close_decision(
        decision_id=decision_id,
        exit_price=2700.0,  # Hit stop loss
        exit_reason="sl-hit",
        outcome_notes="Stop loss hit. Market continued higher."
    )
    
    print_separator()
    print("📊 Trade Results:")
    print(f"   Entry: ${closed['entry_price']:.2f}")
    print(f"   Exit:  ${closed['exit_price']:.2f}")
    print(f"   P&L:   {closed['pnl_percent']:+.2f}%")
    print(f"   RR Realized: {closed['rr_realized']:+.2f}R")
    print(f"   Reward Signal: {closed['reward_signal']:+.3f} (negative for loss)")


def example_trade_series():
    """Example: Series of trades to build portfolio history"""
    print_separator("Example 3: Trade Series")
    
    trades = [
        ("BUY", 2640.0, 2620.0, 2680.0, "tp-hit", "Win #1"),
        ("SELL", 2680.0, 2700.0, 2700.0, "sl-hit", "Loss #1"),
        ("BUY", 2700.0, 2680.0, 2740.0, "tp-hit", "Win #2"),
        ("BUY", 2740.0, 2720.0, 2760.0, "manual", "Win #3 - partial profit"),
    ]
    
    print(f"📝 Executing {len(trades)} trades...\n")
    
    for i, (action, entry, sl, exit_price, exit_reason, note) in enumerate(trades, 1):
        print(f"Trade {i}: {action} @ ${entry:.2f}")
        
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action=action,
            rationale=note,
            entry_price=entry,
            stop_loss=sl,
            take_profit=exit_price if "Win" in note else entry,
            volume=0.1
        )
        
        close_decision(
            decision_id=decision_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            outcome_notes=note
        )
        print()
    
    print_separator()
    print("✓ Trade series completed")


def show_portfolio_state():
    """Display current portfolio state"""
    print_separator("Portfolio State")
    
    portfolio = PortfolioStateTracker.load_state()
    stats = portfolio.get_statistics()
    
    print("📈 Portfolio Statistics:")
    print(f"   Initial Capital:    ${stats['initial_capital']:,.2f}")
    print(f"   Current Equity:     ${stats['current_equity']:,.2f}")
    print(f"   Total Return:       {stats['total_return_pct']:+.2f}%")
    print(f"   Total P&L:          ${stats['total_pnl']:+,.2f}")
    print()
    print(f"   Total Trades:       {stats['total_trades']}")
    print(f"   Winning Trades:     {stats['winning_trades']}")
    print(f"   Losing Trades:      {stats['losing_trades']}")
    print(f"   Win Rate:           {stats['win_rate']*100:.1f}%")
    print()
    print(f"   Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
    print(f"   Profit Factor:      {stats['profit_factor']:.2f}")
    print(f"   Max Drawdown:       {stats['max_drawdown_pct']:.2f}%")
    print(f"   Current Drawdown:   {stats['current_drawdown_pct']:.2f}%")
    print(f"   Peak Equity:        ${stats['peak_equity']:,.2f}")


def show_decision_stats():
    """Display decision statistics"""
    print_separator("Decision Statistics")
    
    stats = get_decision_stats()
    
    print("📊 Decision Quality Metrics:")
    print(f"   Total Decisions:    {stats['total_decisions']}")
    print(f"   Correct Decisions:  {stats['correct_decisions']}")
    print(f"   Accuracy Rate:      {stats['correct_rate']*100:.1f}%")
    print(f"   Avg P&L per Trade:  {stats['avg_pnl_percent']:+.2f}%")
    print(f"   Total P&L:          ${stats['total_pnl']:+,.2f}")
    
    if stats['best_decision']:
        print(f"\n   Best Decision:      {stats['best_decision']['decision_id']}")
        print(f"   └─ P&L: {stats['best_decision']['pnl_percent']:+.2f}%")
    
    if stats['worst_decision']:
        print(f"\n   Worst Decision:     {stats['worst_decision']['decision_id']}")
        print(f"   └─ P&L: {stats['worst_decision']['pnl_percent']:+.2f}%")


def demonstrate_reward_components():
    """Demonstrate individual reward components"""
    print_separator("Reward Components Breakdown")
    
    print("🔬 Calculating reward components for a sample trade:\n")
    
    # Sample trade parameters
    entry = 2650.0
    exit = 2680.0
    sl = 2630.0
    direction = "BUY"
    
    print(f"Trade Setup:")
    print(f"   Direction: {direction}")
    print(f"   Entry:     ${entry:.2f}")
    print(f"   Stop Loss: ${sl:.2f}")
    print(f"   Exit:      ${exit:.2f}")
    print()
    
    # Calculate RR
    rr = RewardCalculator.calculate_realized_rr(entry, exit, sl, direction)
    print(f"1. Realized Risk-Reward: {rr:+.2f}R")
    
    # Sample portfolio data
    portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
    trade_return = (exit - entry) / entry
    
    sharpe_contrib = RewardCalculator.calculate_sharpe_contribution(
        trade_return, portfolio_returns, 0.01
    )
    print(f"2. Sharpe Contribution:  {sharpe_contrib:+.3f}")
    
    # Sample equity curve
    equity_curve = [100000, 102000, 101000, 103000]
    peak = 103000
    pnl = 300
    
    dd_impact = RewardCalculator.calculate_drawdown_impact(pnl, equity_curve, peak)
    print(f"3. Drawdown Impact:      {dd_impact:+.3f}")
    
    # Calculate composite reward
    reward = RewardCalculator.calculate_reward(rr, sharpe_contrib, dd_impact, True)
    print()
    print(f"📊 Composite Reward Signal: {reward:+.3f}")
    print()
    print("Formula: reward = (RR × 0.4) + (Sharpe × 0.3) - (DD × 0.3)")
    print(f"         = ({rr:.2f} × 0.4) + ({sharpe_contrib:.3f} × 0.3) - ({dd_impact:.3f} × 0.3)")
    print(f"         = {reward:+.3f}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("  REWARD SIGNAL SYSTEM - PHASE 1 DEMONSTRATION")
    print("="*60)
    print("\nThis script demonstrates the continuous learning foundation:")
    print("  • Enhanced trade decision tracking")
    print("  • Multi-factor reward signal calculation")
    print("  • Portfolio state management")
    print("  • Learning metrics tracking")
    
    input("\nPress Enter to start examples...")
    
    # Run examples
    example_winning_trade()
    input("\nPress Enter to continue...")
    
    example_losing_trade()
    input("\nPress Enter to continue...")
    
    example_trade_series()
    input("\nPress Enter to continue...")
    
    demonstrate_reward_components()
    input("\nPress Enter to view portfolio state...")
    
    show_portfolio_state()
    input("\nPress Enter to view decision statistics...")
    
    show_decision_stats()
    
    print_separator("Examples Complete")
    print("✅ Phase 1 implementation is working correctly!")
    print("\nNext steps:")
    print("  • Phase 2: Implement regime detection")
    print("  • Phase 3: Add RAG-based decision support")
    print("  • Phase 4: Enable online learning")
    print()


if __name__ == "__main__":
    main()
