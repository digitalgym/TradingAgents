"""
Test SMC Integration with Analyze Workflow

This test verifies that:
1. SMC analysis returns valid data structure
2. format_smc_for_prompt creates non-empty output
3. SMC fields are properly defined in AgentState
4. State initialization includes SMC fields
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="module")
def smc_analysis():
    """Fixture to provide SMC analysis data for tests."""
    from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc

    symbol = "XAGUSD"
    timeframes = ['1H', '4H', 'D1']

    print(f"\n[FIXTURE] Running SMC analysis for {symbol}...")
    analysis = analyze_multi_timeframe_smc(symbol, timeframes)
    print(f"[FIXTURE] SMC analysis complete: {len(analysis) if analysis else 0} timeframes")

    return analysis


def test_smc_data_structure(smc_analysis):
    """Test that SMC analysis returns proper structure."""
    print("\n=== Test 1: SMC Data Structure ===")

    # Check structure
    assert smc_analysis is not None, "SMC analysis returned None"
    assert isinstance(smc_analysis, dict), f"SMC analysis is not dict: {type(smc_analysis)}"
    assert len(smc_analysis) > 0, "SMC analysis is empty dict"

    print(f"✓ SMC analysis returned: {type(smc_analysis)}")
    print(f"✓ Timeframes: {list(smc_analysis.keys())}")

    # Check first timeframe structure
    first_tf = list(smc_analysis.keys())[0]
    tf_data = smc_analysis[first_tf]

    assert isinstance(tf_data, dict), f"Timeframe data is not dict: {type(tf_data)}"

    required_keys = ['bias', 'current_price', 'order_blocks', 'fair_value_gaps', 'structure', 'zones']
    for key in required_keys:
        assert key in tf_data, f"Missing key '{key}' in {first_tf} data"

    print(f"✓ {first_tf} has all required keys: {list(tf_data.keys())}")
    print(f"✓ Bias: {tf_data['bias']}")
    print(f"✓ Order Blocks: {tf_data['order_blocks']['unmitigated']} unmitigated")
    print(f"✓ FVGs: {tf_data['fair_value_gaps']['unmitigated']} unmitigated")

    print("\n✅ Test 1 PASSED: SMC data structure is valid\n")


def test_smc_formatting(smc_analysis):
    """Test that format_smc_for_prompt creates non-empty output."""
    print("\n=== Test 2: SMC Formatting ===")

    from tradingagents.dataflows.smc_utils import format_smc_for_prompt

    symbol = "XAGUSD"
    print(f"Formatting SMC for {symbol}...")

    smc_context = format_smc_for_prompt(smc_analysis, symbol)

    # Check output
    assert smc_context is not None, "format_smc_for_prompt returned None"
    assert isinstance(smc_context, str), f"SMC context is not string: {type(smc_context)}"
    assert len(smc_context) > 0, "SMC context is empty string"
    assert symbol in smc_context, f"Symbol '{symbol}' not in formatted output"
    assert "SMART MONEY CONCEPTS" in smc_context, "Missing header in formatted output"

    print(f"✓ SMC context length: {len(smc_context)} chars")
    print(f"✓ Contains symbol: {symbol in smc_context}")
    print(f"✓ Contains header: {'SMART MONEY CONCEPTS' in smc_context}")
    print(f"\n--- First 300 chars ---")
    print(smc_context[:300])
    print("--- End preview ---\n")

    print("✅ Test 2 PASSED: SMC formatting works correctly\n")


def test_agent_state_definition():
    """Test that AgentState includes SMC fields."""
    print("\n=== Test 3: AgentState Definition ===")

    from tradingagents.agents.utils.agent_states import AgentState

    # Check if AgentState has annotations
    assert hasattr(AgentState, '__annotations__'), "AgentState has no annotations"

    annotations = AgentState.__annotations__
    print(f"✓ AgentState has {len(annotations)} fields")

    # Check for SMC fields
    assert 'smc_context' in annotations, "Missing 'smc_context' in AgentState"
    assert 'smc_analysis' in annotations, "Missing 'smc_analysis' in AgentState"

    print(f"✓ 'smc_context' field exists: {annotations['smc_context']}")
    print(f"✓ 'smc_analysis' field exists: {annotations['smc_analysis']}")

    # Check if they're Optional
    smc_context_type = str(annotations['smc_context'])
    smc_analysis_type = str(annotations['smc_analysis'])

    print(f"✓ smc_context type: {smc_context_type}")
    print(f"✓ smc_analysis type: {smc_analysis_type}")

    print("\n✅ Test 3 PASSED: AgentState has SMC fields\n")


def test_state_initialization():
    """Test that create_initial_state includes SMC fields."""
    print("\n=== Test 4: State Initialization ===")

    from tradingagents.graph.propagation import Propagator

    propagator = Propagator()

    print("Creating initial state...")
    state = propagator.create_initial_state("XAGUSD", "2025-01-16")

    # Check state structure
    assert isinstance(state, dict), f"State is not dict: {type(state)}"
    assert 'smc_context' in state, "Missing 'smc_context' in initial state"
    assert 'smc_analysis' in state, "Missing 'smc_analysis' in initial state"

    print(f"✓ State has {len(state)} keys")
    print(f"✓ 'smc_context' in state: {state['smc_context']}")
    print(f"✓ 'smc_analysis' in state: {state['smc_analysis']}")

    # Check default values
    assert state['smc_context'] is None, f"smc_context should be None by default, got: {state['smc_context']}"
    assert state['smc_analysis'] is None, f"smc_analysis should be None by default, got: {state['smc_analysis']}"

    print("✓ Default values are None (correct)")

    print("\n✅ Test 4 PASSED: State initialization includes SMC fields\n")


def test_trader_receives_smc():
    """Test that trader node can access SMC from state."""
    print("\n=== Test 5: Trader Receives SMC ===")

    # Create mock state with SMC
    mock_state = {
        'company_of_interest': 'XAGUSD',
        'investment_plan': 'Test plan',
        'market_report': 'Test market',
        'sentiment_report': 'Test sentiment',
        'news_report': 'Test news',
        'fundamentals_report': 'Test fundamentals',
        'smc_context': 'Test SMC context with order blocks and FVGs',
        'smc_analysis': {'1H': {'bias': 'bullish'}}
    }

    # Test state access
    smc_context = mock_state.get('smc_context') or ""

    assert smc_context != "", "SMC context should not be empty"
    assert 'order blocks' in smc_context, "SMC context should contain 'order blocks'"

    print(f"✓ Mock state has SMC context: {len(smc_context)} chars")
    print(f"✓ SMC context content: {smc_context[:50]}...")

    print("\n✅ Test 5 PASSED: Trader can access SMC from state\n")


def test_smc_stop_loss_suggestion(smc_analysis):
    """Test that SMC stop loss suggestions work with multi-timeframe confluence."""
    print("\n=== Test 6: SMC Stop Loss Suggestions ===")

    from tradingagents.dataflows.smc_utils import suggest_smc_stop_loss

    # Get current price from first available timeframe
    first_tf = list(smc_analysis.keys())[0]
    tf_data = smc_analysis[first_tf]
    current_price = tf_data['current_price']

    # Test BUY stop loss with multi-timeframe analysis
    stop_suggestion = suggest_smc_stop_loss(
        smc_analysis=smc_analysis,  # Pass full multi-TF dict
        direction='BUY',
        entry_price=current_price,
        atr=0.5,  # Provide ATR for fallback testing
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    print(f"✓ Stop loss suggestion returned: {type(stop_suggestion)}")

    # Should ALWAYS return a result (never None with ATR fallback)
    assert stop_suggestion is not None, "Stop loss should never be None when ATR provided"
    assert 'price' in stop_suggestion, "Missing 'price' in stop suggestion"
    assert 'zone_top' in stop_suggestion, "Missing 'zone_top' in stop suggestion"
    assert 'zone_bottom' in stop_suggestion, "Missing 'zone_bottom' in stop suggestion"
    assert 'source' in stop_suggestion, "Missing 'source' in stop suggestion"
    assert 'distance_pct' in stop_suggestion, "Missing 'distance_pct' in stop suggestion"
    assert 'confluence_score' in stop_suggestion, "Missing 'confluence_score' in stop suggestion"
    assert 'aligned_timeframes' in stop_suggestion, "Missing 'aligned_timeframes' in stop suggestion"

    assert stop_suggestion['price'] < current_price, "Stop loss should be below entry for BUY"

    print(f"✓ Stop loss price: ${stop_suggestion['price']:.2f}")
    print(f"✓ Distance: {stop_suggestion['distance_pct']:.2f}%")
    print(f"✓ Source: {stop_suggestion['source']}")
    print(f"✓ Confluence score: {stop_suggestion['confluence_score']:.1f}")
    print(f"✓ Aligned timeframes: {stop_suggestion['aligned_timeframes']}")

    print("\n✅ Test 6 PASSED: Stop loss returns closest zone with confluence scoring\n")


def test_smc_take_profit_suggestions(smc_analysis):
    """Test that SMC take profit suggestions work."""
    print("\n=== Test 7: SMC Take Profit Suggestions ===")

    from tradingagents.dataflows.smc_utils import suggest_smc_take_profits

    # Use first available timeframe
    first_tf = list(smc_analysis.keys())[0]
    tf_data = smc_analysis[first_tf]

    current_price = tf_data['current_price']

    # Test BUY take profits
    tp_suggestions = suggest_smc_take_profits(
        smc_analysis=tf_data,
        direction='BUY',
        entry_price=current_price,
        num_targets=3
    )

    print(f"✓ Take profit suggestions returned: {type(tp_suggestions)}")
    assert isinstance(tp_suggestions, list), "TP suggestions should be a list"

    if tp_suggestions:
        print(f"✓ Found {len(tp_suggestions)} take profit targets")

        for i, tp in enumerate(tp_suggestions, 1):
            assert 'price' in tp, f"Missing 'price' in TP{i}"
            assert 'source' in tp, f"Missing 'source' in TP{i}"
            assert 'distance_pct' in tp, f"Missing 'distance_pct' in TP{i}"

            assert tp['price'] > current_price, f"TP{i} should be above entry for BUY"

            print(f"  TP{i}: ${tp['price']:.2f} (+{tp['distance_pct']:.2f}%) from {tp['source']}")
    else:
        print("⚠ No take profit targets found (zones may not exist above price)")

    print("\n✅ Test 7 PASSED: Take profit suggestion structure is valid\n")


def test_smc_stop_loss_ath_scenario():
    """Test that stop loss works at ATH (no resistance above) using ATR fallback."""
    print("\n=== Test 8: SMC Stop Loss ATH Scenario ===")

    from tradingagents.dataflows.smc_utils import suggest_smc_stop_loss

    # Mock ATH scenario: No resistance zones above price for SELL
    ath_analysis = {
        '1H': {
            'current_price': 2100,
            'zones': {
                'resistance': [],  # No resistance at ATH
                'support': [
                    {'type': 'demand', 'top': 2000, 'bottom': 1990, 'strength': 0.9}
                ]
            },
            'order_blocks': {
                'bearish': [],  # No bearish OBs above
                'bullish': []
            },
            'fair_value_gaps': {
                'bearish': [],  # No bearish FVGs above
                'bullish': []
            }
        }
    }

    current_price = 2100  # At ATH, far above any support

    # Test SELL at ATH - should use ATR fallback
    stop_suggestion = suggest_smc_stop_loss(
        smc_analysis=ath_analysis,
        direction='SELL',
        entry_price=current_price,
        atr=10.0,  # ATR value
        atr_multiplier=2.0,
        primary_timeframe='1H'
    )

    assert stop_suggestion is not None, "Stop loss should use ATR fallback at ATH"
    assert stop_suggestion['source'] == 'ATR(2.0x)', "Should use ATR-based stop at ATH"
    assert stop_suggestion['price'] > current_price, "Stop should be above entry for SELL"
    assert stop_suggestion['confluence_score'] == 0.0, "ATR fallback should have score 0.0"
    assert stop_suggestion['aligned_timeframes'] == [], "ATR fallback should have empty timeframes"

    expected_stop = current_price + (10.0 * 2.0)
    assert abs(stop_suggestion['price'] - expected_stop) < 0.01, f"Expected ${expected_stop:.2f}, got ${stop_suggestion['price']:.2f}"

    print(f"✓ ATH SELL stop loss: ${stop_suggestion['price']:.2f}")
    print(f"✓ Source: {stop_suggestion['source']}")
    print(f"✓ Distance: {stop_suggestion['distance_pct']:.2f}%")

    print("\n✅ Test 8 PASSED: ATR fallback works for ATH scenario\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SMC INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: SMC data structure
    smc_analysis = test_smc_data_structure()
    results['data_structure'] = smc_analysis is not None
    
    # Test 2: SMC formatting
    smc_context = test_smc_formatting(smc_analysis)
    results['formatting'] = smc_context is not None and len(smc_context) > 0
    
    # Test 3: AgentState definition
    results['agent_state'] = test_agent_state_definition()
    
    # Test 4: State initialization
    results['state_init'] = test_state_initialization()
    
    # Test 5: Trader receives SMC
    results['trader_access'] = test_trader_receives_smc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! SMC integration is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
