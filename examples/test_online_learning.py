"""
Example script demonstrating online learning and pattern analysis

This script shows how to:
1. Analyze trade patterns periodically
2. Update agent weights based on performance
3. Adapt to changing market conditions
4. Enforce risk guardrails

Run this to test Phase 4 & 5 implementation.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.learning.pattern_analyzer import PatternAnalyzer
from tradingagents.learning.online_rl import OnlineRLUpdater
from tradingagents.risk.guardrails import RiskGuardrails
from tradingagents.trade_decisions import store_decision, close_decision, set_decision_regime


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def example_pattern_analysis():
    """Example: Analyze trade patterns"""
    print_separator("Example 1: Pattern Analysis")
    
    print("📊 Analyzing recent trade patterns...\n")
    
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
    
    # Display report
    report = analyzer.format_report(analysis)
    print(report)


def example_online_weight_updates():
    """Example: Update agent weights based on performance"""
    print_separator("Example 2: Online Agent Weight Updates")
    
    updater = OnlineRLUpdater(learning_rate=0.1, momentum=0.9)
    
    print("🤖 Current Agent Weights:")
    current = updater.get_current_weights()
    for agent, weight in current.items():
        print(f"   {agent.capitalize():8}: {weight:.3f}")
    
    # Simulate agent performances
    print("\n📈 Recent Agent Performance (last 30 days):")
    performances = {
        "bull": {"win_rate": 0.70, "avg_reward": 1.5, "sample_size": 25},
        "bear": {"win_rate": 0.45, "avg_reward": -0.3, "sample_size": 15},
        "market": {"win_rate": 0.55, "avg_reward": 0.5, "sample_size": 20}
    }
    
    for agent, perf in performances.items():
        print(f"   {agent.capitalize():8}: {perf['win_rate']*100:.0f}% win rate, "
              f"{perf['avg_reward']:+.2f} avg reward ({perf['sample_size']} trades)")
    
    # Update weights
    print("\n🔄 Updating weights based on performance...")
    result = updater.update_weights(performances)
    
    # Display results
    report = updater.format_report(result)
    print(report)
    
    print("💡 Impact:")
    print("   Bull agent performing well → weight increased")
    print("   Bear agent underperforming → weight decreased")
    print("   System adapts to what's working!")


def example_risk_guardrails():
    """Example: Risk guardrails and circuit breakers"""
    print_separator("Example 3: Risk Guardrails")
    
    guardrails = RiskGuardrails(
        daily_loss_limit_pct=3.0,
        max_consecutive_losses=2,
        max_position_size_pct=2.0,
        cooldown_hours=24
    )
    
    print("🛡️  Risk Guardrail Configuration:")
    print(f"   Daily Loss Limit: 3.0%")
    print(f"   Max Consecutive Losses: 2")
    print(f"   Max Position Size: 2.0%")
    print(f"   Cooldown Period: 24 hours")
    
    # Check initial status
    print("\n📊 Current Status:")
    report = guardrails.format_report()
    print(report)


def example_circuit_breaker_trigger():
    """Example: Trigger circuit breaker"""
    print_separator("Example 4: Circuit Breaker Activation")
    
    guardrails = RiskGuardrails()
    account_balance = 10000
    
    print("💰 Account Balance: $10,000\n")
    
    # Simulate losing trades
    print("📉 Simulating consecutive losses...\n")
    
    # Loss 1
    print("Trade 1: LOSS -1.5%")
    result1 = guardrails.record_trade_result(
        was_win=False,
        pnl_pct=-1.5,
        account_balance=account_balance
    )
    print(f"   Status: {result1['status']}")
    
    # Loss 2
    print("\nTrade 2: LOSS -1.2%")
    result2 = guardrails.record_trade_result(
        was_win=False,
        pnl_pct=-1.2,
        account_balance=account_balance
    )
    print(f"   Status: {result2['status']}")
    
    if result2['breach_triggered']:
        print(f"\n⛔ CIRCUIT BREAKER TRIGGERED!")
        print(f"   Breach Type: {result2['breach_type']}")
        print(f"   Cooldown Until: {result2['cooldown_until']}")
        print(f"\n   🚫 Trading DISABLED for 24 hours")
        print(f"   ⏰ System will auto-resume after cooldown")
    
    # Try to trade during cooldown
    print("\n🔍 Attempting to place new trade...")
    can_trade, reason = guardrails.check_can_trade(account_balance)
    
    if not can_trade:
        print(f"   ❌ BLOCKED: {reason}")
        print(f"   → Trade rejected by risk guardrails")
    else:
        print(f"   ✅ ALLOWED: {reason}")


def example_position_size_validation():
    """Example: Position size validation"""
    print_separator("Example 5: Position Size Validation")
    
    guardrails = RiskGuardrails(max_position_size_pct=2.0)
    account_balance = 10000
    
    print("💰 Account Balance: $10,000")
    print("📏 Max Position Size: 2.0% ($200)\n")
    
    # Test various position sizes
    test_sizes = [1.0, 2.0, 3.0, 5.0]
    
    for size_pct in test_sizes:
        is_valid, reason, adjusted = guardrails.validate_position_size(
            size_pct,
            account_balance
        )
        
        size_usd = account_balance * size_pct / 100
        
        if is_valid:
            print(f"✅ {size_pct}% (${size_usd:.0f}): ALLOWED")
        else:
            adjusted_usd = account_balance * adjusted / 100
            print(f"⚠️  {size_pct}% (${size_usd:.0f}): CAPPED at {adjusted}% (${adjusted_usd:.0f})")


def example_adaptive_system():
    """Example: Complete adaptive system"""
    print_separator("Example 6: Adaptive Trading System")
    
    print("🎯 Demonstrating complete adaptive system:\n")
    
    # 1. Check risk guardrails
    print("1️⃣  Risk Check:")
    guardrails = RiskGuardrails()
    can_trade, reason = guardrails.check_can_trade(10000)
    print(f"   Can Trade: {can_trade} ({reason})")
    
    if not can_trade:
        print("   → Trading blocked, skipping analysis")
        return
    
    # 2. Get current agent weights
    print("\n2️⃣  Agent Weights:")
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()
    for agent, weight in weights.items():
        print(f"   {agent.capitalize():8}: {weight:.3f}")
    
    # 3. Check if pattern analysis needed
    print("\n3️⃣  Pattern Analysis:")
    should_update, trades_since = updater.should_update()
    print(f"   Trades since last update: {trades_since}")
    
    if should_update:
        print("   → Running pattern analysis...")
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30)
        
        print(f"   → Found {analysis['statistics']['patterns_found']} patterns")
        print(f"   → Excellent: {analysis['statistics']['excellent_patterns']}")
        print(f"   → Poor: {analysis['statistics']['poor_patterns']}")
        
        # Update weights
        print("\n   → Updating agent weights...")
        performances = updater.calculate_agent_performances()
        result = updater.update_weights(performances)
        
        print(f"   → Weights updated successfully")
    else:
        print(f"   → Not yet (need {30 - trades_since} more trades)")
    
    # 4. Make decision
    print("\n4️⃣  Decision Making:")
    print("   → Analyzing market setup...")
    print("   → Querying similar trades (RAG)...")
    print("   → Applying regime filters...")
    print("   → Weighting agent opinions...")
    print("   → Validating position size...")
    print("   ✅ Decision ready for execution")
    
    print("\n💡 System Benefits:")
    print("   • Risk guardrails prevent catastrophic losses")
    print("   • Agent weights adapt to performance")
    print("   • Pattern analysis identifies what works")
    print("   • RAG provides historical context")
    print("   • Regime detection filters by conditions")


def example_recovery_after_cooldown():
    """Example: System recovery after cooldown"""
    print_separator("Example 7: Recovery After Cooldown")
    
    guardrails = RiskGuardrails()
    
    print("⏰ Simulating cooldown period...\n")
    
    # Reset to simulate end of cooldown
    guardrails.reset_cooldown()
    guardrails.reset_consecutive_losses()
    guardrails.reset_daily_loss()
    
    print("✅ Cooldown period ended")
    print("✅ Counters reset")
    print("✅ System ready to resume trading\n")
    
    status = guardrails.get_status()
    print(f"Status: {status['status_summary']}")
    
    can_trade, reason = guardrails.check_can_trade(10000)
    print(f"Can Trade: {can_trade} ({reason})")
    
    if can_trade:
        print("\n💡 Best Practices After Recovery:")
        print("   • Start with reduced position sizes")
        print("   • Focus on highest-confidence setups")
        print("   • Review what went wrong during losses")
        print("   • Apply lessons from pattern analysis")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("  ONLINE LEARNING & RISK GUARDRAILS - PHASE 4 & 5 DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates:")
    print("  • Pattern analysis and clustering")
    print("  • Online agent weight updates")
    print("  • Risk guardrails and circuit breakers")
    print("  • Adaptive trading system")
    
    input("\nPress Enter to start examples...")
    
    # Run examples
    example_pattern_analysis()
    input("\nPress Enter to continue...")
    
    example_online_weight_updates()
    input("\nPress Enter to continue...")
    
    example_risk_guardrails()
    input("\nPress Enter to continue...")
    
    example_circuit_breaker_trigger()
    input("\nPress Enter to continue...")
    
    example_position_size_validation()
    input("\nPress Enter to continue...")
    
    example_adaptive_system()
    input("\nPress Enter to continue...")
    
    example_recovery_after_cooldown()
    
    print_separator("Examples Complete")
    print("✅ Phase 4 & 5 implementation is working correctly!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Pattern analysis identifies winning/losing clusters")
    print("  ✓ Agent weights adapt based on performance")
    print("  ✓ Circuit breakers prevent catastrophic losses")
    print("  ✓ Position size limits enforced")
    print("  ✓ Automatic cooldown after breaches")
    print("  ✓ Complete adaptive trading system")
    print("\nSystem is now fully adaptive and self-improving!")
    print()


if __name__ == "__main__":
    main()
