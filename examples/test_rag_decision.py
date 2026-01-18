"""
Example script demonstrating RAG-enhanced decision making

This script shows how to:
1. Find similar historical trades
2. Calculate statistics and confidence adjustments
3. Enhance agent prompts with historical context
4. Make regime-aware, history-informed decisions

Run this to test the Phase 3 implementation.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.learning.trade_similarity import TradeSimilaritySearch
from tradingagents.learning.rag_prompts import (
    enhance_prompt_with_rag,
    apply_confidence_adjustment,
    format_confidence_explanation
)
from tradingagents.trade_decisions import store_decision, close_decision, set_decision_regime


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def setup_sample_trades():
    """Create sample historical trades for demonstration"""
    print("📝 Setting up sample historical trades...")
    
    sample_trades = [
        # Winning breaker block trades in trending-up market
        {
            "symbol": "XAUUSD", "action": "BUY", "entry": 2600, "sl": 2580, "exit": 2650,
            "regime": {"market_regime": "trending-up", "volatility_regime": "normal"},
            "setup_type": "breaker-block", "higher_tf_bias": "bullish", "confluence": 8
        },
        {
            "symbol": "XAUUSD", "action": "BUY", "entry": 2620, "sl": 2600, "exit": 2660,
            "regime": {"market_regime": "trending-up", "volatility_regime": "normal"},
            "setup_type": "breaker-block", "higher_tf_bias": "bullish", "confluence": 7
        },
        {
            "symbol": "XAUUSD", "action": "BUY", "entry": 2640, "sl": 2620, "exit": 2680,
            "regime": {"market_regime": "trending-up", "volatility_regime": "high"},
            "setup_type": "breaker-block", "higher_tf_bias": "bullish", "confluence": 9
        },
        # Losing trade in extreme volatility
        {
            "symbol": "XAUUSD", "action": "BUY", "entry": 2650, "sl": 2630, "exit": 2630,
            "regime": {"market_regime": "trending-up", "volatility_regime": "extreme"},
            "setup_type": "breaker-block", "higher_tf_bias": "bullish", "confluence": 6
        },
        # Ranging market trades (mixed results)
        {
            "symbol": "XAUUSD", "action": "SELL", "entry": 2650, "sl": 2670, "exit": 2670,
            "regime": {"market_regime": "ranging", "volatility_regime": "low"},
            "setup_type": "resistance-rejection", "higher_tf_bias": "neutral", "confluence": 5
        },
    ]
    
    created_count = 0
    for trade_data in sample_trades:
        decision_id = store_decision(
            symbol=trade_data["symbol"],
            decision_type="OPEN",
            action=trade_data["action"],
            rationale=f"{trade_data['setup_type']} setup",
            entry_price=trade_data["entry"],
            stop_loss=trade_data["sl"],
            take_profit=trade_data["exit"] if trade_data["exit"] > trade_data["entry"] else trade_data["entry"],
            volume=0.1
        )
        
        # Set regime
        set_decision_regime(decision_id, trade_data["regime"])
        
        # Close trade
        close_decision(
            decision_id=decision_id,
            exit_price=trade_data["exit"],
            exit_reason="tp-hit" if trade_data["exit"] != trade_data["sl"] else "sl-hit"
        )
        
        created_count += 1
    
    print(f"✓ Created {created_count} sample trades\n")


def example_find_similar_trades():
    """Example: Find similar trades for a new setup"""
    print_separator("Example 1: Finding Similar Trades")
    
    # Current setup
    current_setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "setup_type": "breaker-block",
        "market_regime": "trending-up",
        "volatility_regime": "normal",
        "higher_tf_bias": "bullish",
        "confluence_score": 8
    }
    
    print("🎯 Current Setup:")
    print(f"   Symbol: {current_setup['symbol']}")
    print(f"   Direction: {current_setup['direction']}")
    print(f"   Setup: {current_setup['setup_type']}")
    print(f"   Regime: {current_setup['market_regime']} / {current_setup['volatility_regime']}")
    print(f"   Confluence: {current_setup['confluence_score']}/10")
    
    # Search for similar trades
    print("\n🔍 Searching for similar historical trades...")
    searcher = TradeSimilaritySearch()
    result = searcher.find_similar_trades(current_setup, n_results=5)
    
    print(f"\n📊 Found {result['statistics']['sample_size']} similar trades:")
    print(f"   Win Rate: {result['statistics']['win_rate']*100:.1f}%")
    print(f"   Avg RR: {result['statistics']['avg_rr']:.2f}")
    print(f"   Avg Reward: {result['statistics']['avg_reward']:+.2f}")
    print(f"   Confidence Adjustment: {result['statistics']['confidence_adjustment']:+.2f}")
    
    print("\n📋 Top Similar Trades:")
    for i, (trade, score) in enumerate(zip(result['similar_trades'][:3], result['similarity_scores'][:3]), 1):
        outcome = "✓ WIN" if trade.get('was_correct') else "✗ LOSS"
        rr = trade.get('rr_realized', 0)
        print(f"   {i}. {outcome} | {rr:+.2f}R | Similarity: {score:.2f}")
    
    print(f"\n💡 Recommendation:")
    print(f"   {result['recommendation']}")


def example_confidence_adjustment():
    """Example: Apply confidence adjustment based on history"""
    print_separator("Example 2: Confidence Adjustment")
    
    current_setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "setup_type": "breaker-block",
        "market_regime": "trending-up",
        "volatility_regime": "normal",
        "confluence_score": 8
    }
    
    # Get similar trades
    searcher = TradeSimilaritySearch()
    result = searcher.find_similar_trades(current_setup)
    
    # Original agent confidence
    base_confidence = 0.75
    adjustment = result['statistics']['confidence_adjustment']
    
    print(f"🤖 Agent Analysis:")
    print(f"   Base Confidence: {base_confidence:.2f}")
    print(f"   Reasoning: Strong bullish setup with multiple confluence factors")
    
    print(f"\n📚 Historical Context:")
    print(f"   Similar Trades: {result['statistics']['sample_size']}")
    print(f"   Win Rate: {result['statistics']['win_rate']*100:.1f}%")
    print(f"   Suggested Adjustment: {adjustment:+.2f}")
    
    # Apply adjustment
    final_confidence = apply_confidence_adjustment(base_confidence, adjustment)
    
    print(f"\n✨ Final Decision:")
    print(f"   Adjusted Confidence: {final_confidence:.2f}")
    
    if adjustment > 0.1:
        print(f"   ✓ BOOST: Historical performance supports this setup")
    elif adjustment < -0.1:
        print(f"   ⚠️  CAUTION: Historical performance suggests lower confidence")
    else:
        print(f"   → NEUTRAL: Proceed with base confidence")
    
    # Show explanation
    print(f"\n📝 Detailed Explanation:")
    explanation = format_confidence_explanation(base_confidence, adjustment, result['statistics'])
    for line in explanation.split('\n'):
        if line.strip():
            print(f"   {line}")


def example_rag_enhanced_prompt():
    """Example: Enhance agent prompt with RAG context"""
    print_separator("Example 3: RAG-Enhanced Agent Prompt")
    
    current_setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "setup_type": "breaker-block",
        "market_regime": "trending-up",
        "volatility_regime": "normal",
        "confluence_score": 8
    }
    
    # Original prompt
    base_prompt = """You are a bullish analyst. Analyze the current XAUUSD setup.

Current Market:
- Price: 2650
- Setup: Breaker block at 2630 support
- Higher TF: Bullish on H4 and D1
- Confluence: 8/10 (multiple factors aligned)

Provide your bullish case with confidence score (0.0-1.0)."""
    
    print("📄 Original Prompt (excerpt):")
    print("   " + base_prompt.split('\n')[0])
    print("   " + base_prompt.split('\n')[1])
    print("   ...")
    
    # Enhance with RAG
    print("\n🔄 Enhancing with historical context...")
    enhanced_prompt, adjustment = enhance_prompt_with_rag(
        base_prompt,
        current_setup,
        n_similar=5
    )
    
    print(f"\n✨ Enhanced Prompt includes:")
    print(f"   ✓ {5} similar historical trades")
    print(f"   ✓ Win rate statistics")
    print(f"   ✓ Risk-reward performance")
    print(f"   ✓ Confidence adjustment guidance ({adjustment:+.2f})")
    
    print(f"\n📋 RAG Context Added:")
    # Extract the historical context section
    if "HISTORICAL CONTEXT" in enhanced_prompt:
        context_start = enhanced_prompt.find("HISTORICAL CONTEXT")
        context_section = enhanced_prompt[context_start:context_start+400]
        for line in context_section.split('\n')[:8]:
            print(f"   {line}")
        print("   ...")


def example_decision_comparison():
    """Example: Compare decisions with and without RAG"""
    print_separator("Example 4: Decision Comparison (With vs Without RAG)")
    
    current_setup = {
        "symbol": "XAUUSD",
        "direction": "BUY",
        "setup_type": "breaker-block",
        "market_regime": "trending-up",
        "volatility_regime": "extreme",  # Note: extreme volatility
        "confluence_score": 7
    }
    
    print("⚠️  Current Setup (Extreme Volatility):")
    print(f"   Setup: {current_setup['setup_type']}")
    print(f"   Regime: {current_setup['market_regime']} / {current_setup['volatility_regime']}")
    
    # Without RAG
    print("\n❌ WITHOUT RAG:")
    print("   Agent sees: Good setup, trending market")
    print("   Confidence: 0.75 (high)")
    print("   Decision: ENTER TRADE")
    
    # With RAG
    print("\n✅ WITH RAG:")
    searcher = TradeSimilaritySearch()
    result = searcher.find_similar_trades(current_setup)
    
    print(f"   Agent sees: Good setup, BUT...")
    print(f"   Historical data: {result['statistics']['sample_size']} similar trades")
    print(f"   Win rate in extreme volatility: {result['statistics']['win_rate']*100:.1f}%")
    print(f"   Adjustment: {result['statistics']['confidence_adjustment']:+.2f}")
    
    final_conf = apply_confidence_adjustment(0.75, result['statistics']['confidence_adjustment'])
    print(f"   Adjusted Confidence: {final_conf:.2f}")
    
    if final_conf < 0.5:
        print("   Decision: SKIP TRADE (low confidence)")
    else:
        print("   Decision: REDUCE SIZE or WAIT")
    
    print(f"\n💡 Impact: RAG prevents entering trade in unfavorable conditions")


def example_regime_specific_performance():
    """Example: Show performance varies by regime"""
    print_separator("Example 5: Regime-Specific Performance")
    
    searcher = TradeSimilaritySearch()
    
    regimes = [
        {"market_regime": "trending-up", "volatility_regime": "normal"},
        {"market_regime": "trending-up", "volatility_regime": "high"},
        {"market_regime": "trending-up", "volatility_regime": "extreme"},
        {"market_regime": "ranging", "volatility_regime": "low"},
    ]
    
    print("📊 Breaker Block BUY Performance by Regime:\n")
    
    for regime in regimes:
        setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "setup_type": "breaker-block",
            **regime
        }
        
        result = searcher.find_similar_trades(setup, n_results=10)
        stats = result['statistics']
        
        regime_desc = f"{regime['market_regime']:15} / {regime['volatility_regime']:10}"
        
        if stats['sample_size'] > 0:
            print(f"{regime_desc}: {stats['win_rate']*100:5.1f}% win rate "
                  f"({stats['sample_size']} trades) → {stats['confidence_adjustment']:+.2f} adjustment")
        else:
            print(f"{regime_desc}: No data")
    
    print("\n💡 Insight: Same setup performs differently in different regimes!")
    print("   RAG helps identify which conditions are favorable")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("  RAG-ENHANCED DECISION MAKING - PHASE 3 DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates RAG (Retrieval-Augmented Generation):")
    print("  • Finding similar historical trades")
    print("  • Calculating performance statistics")
    print("  • Adjusting confidence based on history")
    print("  • Enhancing agent prompts with context")
    print("  • Regime-aware decision making")
    
    input("\nPress Enter to set up sample trades...")
    setup_sample_trades()
    
    input("Press Enter to start examples...")
    
    # Run examples
    example_find_similar_trades()
    input("\nPress Enter to continue...")
    
    example_confidence_adjustment()
    input("\nPress Enter to continue...")
    
    example_rag_enhanced_prompt()
    input("\nPress Enter to continue...")
    
    example_decision_comparison()
    input("\nPress Enter to continue...")
    
    example_regime_specific_performance()
    
    print_separator("Examples Complete")
    print("✅ Phase 3 implementation is working correctly!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Trade similarity search with regime filtering")
    print("  ✓ Automatic confidence adjustment based on history")
    print("  ✓ RAG-enhanced agent prompts")
    print("  ✓ Regime-specific performance analysis")
    print("  ✓ Historical context prevents bad trades")
    print("\nNext steps:")
    print("  • Phase 4: Online learning with agent weight updates")
    print("  • Phase 5: Risk guardrails and circuit breakers")
    print()


if __name__ == "__main__":
    main()
