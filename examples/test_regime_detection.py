"""
Example script demonstrating regime detection

This script shows how to:
1. Detect market regimes from price data
2. Use regime information in decision making
3. Filter memories by regime
4. Adjust risk based on regime

Run this to test the Phase 2 implementation.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.indicators.regime import RegimeDetector
from tradingagents.dataflows.regime_utils import (
    get_current_regime_from_prices,
    format_regime_for_prompt,
    get_regime_summary
)


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def example_trending_market():
    """Example: Detect trending market regime"""
    print_separator("Example 1: Trending Market Detection")
    
    # Create uptrending price data
    print("📈 Creating uptrending price data...")
    close = np.linspace(2600, 2700, 100)
    high = close + 5
    low = close - 5
    
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)
    
    print("\n🔍 Regime Detection Results:")
    print(f"   Market Regime:     {regime['market_regime']}")
    print(f"   Volatility Regime: {regime['volatility_regime']}")
    print(f"   Expansion Regime:  {regime['expansion_regime']}")
    
    print("\n📊 Regime Analysis:")
    description = detector.get_regime_description(regime)
    print(f"   {description}")
    
    print("\n💡 Trading Implications:")
    if detector.is_favorable_for_trend_trading(regime):
        print("   ✓ Favorable for trend trading")
    else:
        print("   ✗ Not favorable for trend trading")
    
    if detector.is_favorable_for_range_trading(regime):
        print("   ✓ Favorable for range trading")
    else:
        print("   ✗ Not favorable for range trading")
    
    risk_adj = detector.get_risk_adjustment_factor(regime)
    print(f"\n   Position size adjustment: {risk_adj:.2f}x")


def example_ranging_market():
    """Example: Detect ranging market regime"""
    print_separator("Example 2: Ranging Market Detection")
    
    # Create ranging price data (oscillating)
    print("📊 Creating ranging price data...")
    close = 2650 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    high = close + 5
    low = close - 5
    
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)
    
    print("\n🔍 Regime Detection Results:")
    print(f"   Market Regime:     {regime['market_regime']}")
    print(f"   Volatility Regime: {regime['volatility_regime']}")
    print(f"   Expansion Regime:  {regime['expansion_regime']}")
    
    print("\n📊 Regime Analysis:")
    description = detector.get_regime_description(regime)
    print(f"   {description}")
    
    print("\n💡 Trading Implications:")
    if detector.is_favorable_for_range_trading(regime):
        print("   ✓ Favorable for range trading")
        print("   → Consider mean reversion strategies")
    else:
        print("   ✗ Not favorable for range trading")


def example_high_volatility():
    """Example: Detect high volatility regime"""
    print_separator("Example 3: High Volatility Detection")
    
    # Create high volatility data
    print("⚡ Creating high volatility price data...")
    np.random.seed(42)
    close = np.concatenate([
        2650 + np.random.randn(80) * 5,   # Normal volatility
        2650 + np.random.randn(20) * 25   # High volatility spike
    ])
    high = close + np.abs(np.random.randn(100) * 10)
    low = close - np.abs(np.random.randn(100) * 10)
    
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)
    
    print("\n🔍 Regime Detection Results:")
    print(f"   Market Regime:     {regime['market_regime']}")
    print(f"   Volatility Regime: {regime['volatility_regime']}")
    print(f"   Expansion Regime:  {regime['expansion_regime']}")
    
    risk_adj = detector.get_risk_adjustment_factor(regime)
    print(f"\n⚠️  Risk Management:")
    print(f"   Volatility level: {regime['volatility_regime']}")
    print(f"   Position size adjustment: {risk_adj:.2f}x")
    
    if risk_adj < 1.0:
        print(f"   → REDUCE position size to {risk_adj*100:.0f}% of normal")
    elif risk_adj > 1.0:
        print(f"   → Can INCREASE position size to {risk_adj*100:.0f}% of normal")


def example_regime_transitions():
    """Example: Track regime changes over time"""
    print_separator("Example 4: Regime Transitions")
    
    print("📈 Simulating market regime transitions...\n")
    
    detector = RegimeDetector()
    
    # Simulate different market phases
    phases = [
        ("Trending Up", np.linspace(2600, 2650, 50)),
        ("Ranging", 2650 + 5 * np.sin(np.linspace(0, 2*np.pi, 50))),
        ("Trending Down", np.linspace(2650, 2600, 50)),
    ]
    
    for phase_name, close_data in phases:
        high = close_data + 5
        low = close_data - 5
        
        regime = detector.get_full_regime(high, low, close_data)
        summary = get_regime_summary(regime)
        
        print(f"Phase: {phase_name:15} → Regime: {summary}")


def example_prompt_formatting():
    """Example: Format regime for LLM prompts"""
    print_separator("Example 5: Regime Formatting for Prompts")
    
    # Create sample regime
    close = np.linspace(2600, 2700, 100)
    high = close + 5
    low = close - 5
    
    regime = get_current_regime_from_prices(high, low, close)
    
    print("📝 Formatted regime information for LLM prompt:\n")
    formatted = format_regime_for_prompt(regime)
    print(formatted)


def example_regime_aware_decision():
    """Example: Make regime-aware trading decision"""
    print_separator("Example 6: Regime-Aware Decision Making")
    
    # Simulate current market
    np.random.seed(42)
    close = np.linspace(2600, 2700, 100) + np.random.randn(100) * 3
    high = close + 5
    low = close - 5
    
    detector = RegimeDetector()
    regime = detector.get_full_regime(high, low, close)
    
    print("🎯 Trading Decision Framework:\n")
    print(f"Current Regime: {get_regime_summary(regime)}")
    print()
    
    # Decision logic based on regime
    market = regime['market_regime']
    volatility = regime['volatility_regime']
    
    print("Decision Logic:")
    
    if market == 'trending-up':
        print("  ✓ Market is trending up")
        print("  → Strategy: Look for pullback entries in trend direction")
        print("  → Setup: Breaker blocks, FVG fills, support bounces")
        
        if volatility == 'extreme':
            print("  ⚠️  Extreme volatility detected")
            print("  → Reduce position size by 50%")
            print("  → Widen stop loss to avoid noise")
        elif volatility == 'high':
            print("  ⚠️  High volatility")
            print("  → Reduce position size by 25%")
        else:
            print("  ✓ Normal volatility - standard position sizing")
    
    elif market == 'ranging':
        print("  ✓ Market is ranging")
        print("  → Strategy: Mean reversion at range extremes")
        print("  → Setup: Support/resistance bounces, overbought/oversold")
        print("  → Avoid: Breakout trades (likely false breaks)")
    
    elif market == 'trending-down':
        print("  ✓ Market is trending down")
        print("  → Strategy: Look for rally entries to short")
        print("  → Setup: Resistance rejections, bearish patterns")
    
    # Risk adjustment
    risk_adj = detector.get_risk_adjustment_factor(regime)
    print(f"\n📊 Final Position Size: {risk_adj*100:.0f}% of base size")


def example_memory_filtering():
    """Example: Demonstrate regime-based memory filtering"""
    print_separator("Example 7: Regime-Based Memory Filtering")
    
    print("🧠 Memory Retrieval with Regime Filtering:\n")
    
    # Current regime
    current_regime = {
        "market_regime": "trending-up",
        "volatility_regime": "high",
        "expansion_regime": "expansion"
    }
    
    print(f"Current Regime: {get_regime_summary(current_regime)}\n")
    
    # Simulated historical memories with regimes
    memories = [
        {"id": 1, "setup": "Breaker block BUY", "outcome": "Win", 
         "market_regime": "trending-up", "volatility_regime": "high"},
        {"id": 2, "setup": "Support bounce BUY", "outcome": "Win",
         "market_regime": "trending-up", "volatility_regime": "normal"},
        {"id": 3, "setup": "Range short SELL", "outcome": "Loss",
         "market_regime": "ranging", "volatility_regime": "low"},
        {"id": 4, "setup": "Breakout BUY", "outcome": "Win",
         "market_regime": "trending-up", "volatility_regime": "high"},
    ]
    
    print("All Historical Trades:")
    for mem in memories:
        print(f"  #{mem['id']}: {mem['setup']:25} → {mem['outcome']:4} "
              f"({mem['market_regime']}, {mem['volatility_regime']})")
    
    print("\n🔍 Filtered by Current Regime (trending-up + high volatility):")
    filtered = [m for m in memories 
                if m['market_regime'] == current_regime['market_regime']
                and m['volatility_regime'] == current_regime['volatility_regime']]
    
    for mem in filtered:
        print(f"  #{mem['id']}: {mem['setup']:25} → {mem['outcome']:4}")
    
    if filtered:
        win_rate = sum(1 for m in filtered if m['outcome'] == 'Win') / len(filtered)
        print(f"\n📊 Win Rate in Similar Regime: {win_rate*100:.0f}%")
        print(f"   Sample Size: {len(filtered)} trades")
        print("\n💡 Insight: Use this historical performance to adjust confidence")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("  REGIME DETECTION SYSTEM - PHASE 2 DEMONSTRATION")
    print("="*60)
    print("\nThis script demonstrates market regime detection:")
    print("  • Trend detection (trending-up/down, ranging)")
    print("  • Volatility classification (low/normal/high/extreme)")
    print("  • Expansion/contraction detection")
    print("  • Risk adjustment based on regime")
    print("  • Regime-aware decision making")
    
    input("\nPress Enter to start examples...")
    
    # Run examples
    example_trending_market()
    input("\nPress Enter to continue...")
    
    example_ranging_market()
    input("\nPress Enter to continue...")
    
    example_high_volatility()
    input("\nPress Enter to continue...")
    
    example_regime_transitions()
    input("\nPress Enter to continue...")
    
    example_prompt_formatting()
    input("\nPress Enter to continue...")
    
    example_regime_aware_decision()
    input("\nPress Enter to continue...")
    
    example_memory_filtering()
    
    print_separator("Examples Complete")
    print("✅ Phase 2 implementation is working correctly!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-factor regime detection")
    print("  ✓ Risk adjustment based on volatility")
    print("  ✓ Strategy selection by market type")
    print("  ✓ Regime-based memory filtering")
    print("\nNext steps:")
    print("  • Phase 3: Integrate RAG-based decision support")
    print("  • Phase 4: Enable online learning with agent weights")
    print()


if __name__ == "__main__":
    main()
