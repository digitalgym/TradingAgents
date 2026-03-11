"""
Test script for Breakout Quant Analyst

Run this to see the generated prompt and test consolidation detection.
Usage: python scripts/test_breakout_prompt.py
"""

import numpy as np
from datetime import datetime
import json

# Import the breakout quant functions
from tradingagents.agents.analysts.breakout_quant import (
    analyze_consolidation,
    _build_breakout_data_context,
    _build_breakout_prompt,
)
from tradingagents.indicators.regime import RegimeDetector


def generate_sample_data(scenario: str = "squeeze"):
    """Generate sample price data for different scenarios."""
    np.random.seed(42)

    if scenario == "squeeze":
        # Consolidation with bullish structure (higher lows)
        base = 2700
        # First half: wider range
        first_half = base + np.random.randn(50) * 15
        # Second half: tight range with higher lows
        second_half = base + 10 + np.random.randn(50) * 5
        close = np.concatenate([first_half, second_half])

    elif scenario == "trending":
        # Strong uptrend (no consolidation)
        close = np.linspace(2600, 2800, 100) + np.random.randn(100) * 10

    elif scenario == "bearish_squeeze":
        # Consolidation with bearish structure (lower highs)
        base = 2700
        first_half = base + np.random.randn(50) * 15
        second_half = base - 10 + np.random.randn(50) * 5
        close = np.concatenate([first_half, second_half])

    else:
        # Random choppy
        close = 2700 + np.random.randn(100) * 20

    high = close + np.abs(np.random.randn(len(close)) * 5)
    low = close - np.abs(np.random.randn(len(close)) * 5)

    return high, low, close


def test_consolidation_analysis():
    """Test the consolidation analysis function."""
    print("=" * 80)
    print("CONSOLIDATION ANALYSIS TEST")
    print("=" * 80)

    scenarios = ["squeeze", "trending", "bearish_squeeze", "choppy"]

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario.upper()} ---")
        high, low, close = generate_sample_data(scenario)

        result = analyze_consolidation(high, low, close, lookback=20)

        print(f"  Is Consolidating: {result['is_consolidating']}")
        print(f"  Squeeze Strength: {result['squeeze_strength']:.1f}%")
        print(f"  BB Squeeze: {result['bb_squeeze']}")
        print(f"  Structure Bias: {result['structure_bias']}")
        print(f"  Breakout Ready: {result['breakout_ready']}")
        print(f"  Range: {result['range_low']:.2f} - {result['range_high']:.2f}")
        print(f"  Range %: {result['range_percent']:.2f}%")


def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def test_regime_detector():
    """Test the RegimeDetector consolidation method."""
    print("\n" + "=" * 80)
    print("REGIME DETECTOR CONSOLIDATION TEST")
    print("=" * 80)

    detector = RegimeDetector()
    high, low, close = generate_sample_data("squeeze")

    # Full regime
    regime = detector.get_full_regime(high, low, close)
    print(f"\nFull Regime: {json.dumps(regime, indent=2)}")

    # Consolidation specific
    consolidation = detector.detect_consolidation(high, low, close)
    consolidation = make_json_serializable(consolidation)
    print(f"\nConsolidation Analysis: {json.dumps(consolidation, indent=2)}")

    # Favorability
    print(f"\nFavorable for Breakout: {detector.is_favorable_for_breakout_trading(regime)}")
    print(f"Favorable for Trend: {detector.is_favorable_for_trend_trading(regime)}")
    print(f"Favorable for Range: {detector.is_favorable_for_range_trading(regime)}")


def test_prompt_generation():
    """Test the full prompt generation."""
    print("\n" + "=" * 80)
    print("PROMPT GENERATION TEST")
    print("=" * 80)

    # Generate sample data
    high, low, close = generate_sample_data("squeeze")

    # Run consolidation analysis
    consolidation = analyze_consolidation(high, low, close, lookback=20)

    # Build data context
    data_context = _build_breakout_data_context(
        ticker="XAUUSD",
        current_price=2710.50,
        smc_context="""## SMART MONEY CONCEPTS
- Bullish OB at 2695-2700 (strength: 75%)
- Bearish OB at 2725-2730 (strength: 60%)
- Bullish FVG at 2702-2705 (unfilled)
- Bias: BULLISH
""",
        smc_analysis={},
        market_report="""## TECHNICAL INDICATORS
- RSI(14): 55 (neutral)
- MACD: Bullish crossover forming
- ATR(14): 12.5 (low volatility)
- SMA50: 2695 (price above)
- SMA200: 2650 (price above)
""",
        market_regime="ranging",
        volatility_regime="low",
        expansion_regime="contraction",
        trading_session="London",
        current_date="2026-03-11",
        consolidation=consolidation,
    )

    print("\n--- DATA CONTEXT ---")
    print(data_context)

    # Build full prompt
    trade_memories = """## LESSONS FROM PAST BREAKOUT TRADES
1. [2026-03-05] False breakout on XAUUSD - waited for candle close confirmation
2. [2026-03-08] Missed entry by 5 pips - use limit orders at range extremes
"""

    full_prompt = _build_breakout_prompt(data_context, trade_memories=trade_memories)

    print("\n--- FULL PROMPT ---")
    print(full_prompt)

    print("\n--- PROMPT STATS ---")
    print(f"Total characters: {len(full_prompt)}")
    print(f"Estimated tokens: ~{len(full_prompt) // 4}")


def test_with_mock_llm():
    """Test with a mock LLM to see the flow without API calls."""
    print("\n" + "=" * 80)
    print("MOCK LLM TEST (No API call)")
    print("=" * 80)

    from unittest.mock import MagicMock
    from tradingagents.agents.analysts.breakout_quant import create_breakout_quant
    from tradingagents.schemas import QuantAnalystDecision

    # Create mock LLM
    mock_llm = MagicMock()

    # Mock structured output response
    mock_decision = QuantAnalystDecision(
        symbol="XAUUSD",
        signal="buy_to_enter",
        order_type="limit",
        entry_price=2700.0,
        stop_loss=2685.0,
        profit_target=2740.0,
        quantity=0.1,
        leverage=10,
        risk_usd=150.0,
        risk_reward_ratio=2.67,
        invalidation_condition="Price closes below 2685",
        justification="Bullish structure in squeeze, higher lows forming, entry at range low",
        confidence=0.75,
        risk_level="Medium"
    )

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # Create the breakout quant
    breakout_node = create_breakout_quant(mock_llm, use_structured_output=True)

    # Prepare state
    high, low, close = generate_sample_data("squeeze")
    state = {
        "company_of_interest": "XAUUSD",
        "trade_date": "2026-03-11",
        "current_price": 2710.50,
        "market_regime": "ranging",
        "volatility_regime": "low",
        "expansion_regime": "contraction",
        "trading_session": "London",
        "smc_context": "## SMC: Bullish OB at 2695",
        "smc_analysis": {},
        "market_report": "## Indicators: RSI 55, MACD bullish",
        "price_data": {
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
        },
        "trade_memories": "",
    }

    # Run the node
    result = breakout_node(state)

    print("\n--- RESULT ---")
    print(f"Report:\n{result.get('breakout_quant_report', 'N/A')[:500]}...")
    print(f"\nDecision: {json.dumps(result.get('breakout_quant_decision'), indent=2)}")
    consolidation_result = make_json_serializable(result.get('consolidation_analysis'))
    print(f"\nConsolidation Analysis: {json.dumps(consolidation_result, indent=2)}")

    print("\n--- LLM was called with prompt ---")
    if mock_structured_llm.invoke.called:
        prompt_sent = mock_structured_llm.invoke.call_args[0][0]
        print(f"Prompt length: {len(prompt_sent)} chars")
        print(f"First 500 chars:\n{prompt_sent[:500]}...")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BREAKOUT QUANT PROMPT TEST")
    print("=" * 80 + "\n")

    # Run tests
    test_consolidation_analysis()
    test_regime_detector()
    test_prompt_generation()
    test_with_mock_llm()

    print("\n" + "=" * 80)
    print("LOG FILE LOCATION")
    print("=" * 80)
    print("\nPrompts are also logged to:")
    print("  logs/quant_prompts/breakout_quant_YYYYMMDD.log")
    print("\nRun with actual LLM to see real responses logged there.")


if __name__ == "__main__":
    main()
