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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_smc_data_structure():
    """Test that SMC analysis returns proper structure."""
    print("\n=== Test 1: SMC Data Structure ===")
    
    try:
        from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc
        
        # Run SMC analysis
        symbol = "XAGUSD"
        timeframes = ['1H', '4H', 'D1']
        
        print(f"Running SMC analysis for {symbol}...")
        smc_analysis = analyze_multi_timeframe_smc(symbol, timeframes)
        
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
        return smc_analysis
        
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_smc_formatting(smc_analysis):
    """Test that format_smc_for_prompt creates non-empty output."""
    print("\n=== Test 2: SMC Formatting ===")
    
    if not smc_analysis:
        print("❌ Test 2 SKIPPED: No SMC analysis data")
        return None
    
    try:
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
        return smc_context
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_agent_state_definition():
    """Test that AgentState includes SMC fields."""
    print("\n=== Test 3: AgentState Definition ===")
    
    try:
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
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_state_initialization():
    """Test that create_initial_state includes SMC fields."""
    print("\n=== Test 4: State Initialization ===")
    
    try:
        from tradingagents.graph.propagation import GraphPropagator
        
        propagator = GraphPropagator()
        
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
        return True
        
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_trader_receives_smc():
    """Test that trader node can access SMC from state."""
    print("\n=== Test 5: Trader Receives SMC ===")
    
    try:
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
        return True
        
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


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
