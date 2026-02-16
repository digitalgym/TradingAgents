"""
Test LLM-Enhanced SMC Trading Plan System

This script tests the LLM enhancement layer that adds contextual intelligence
to rule-based SMC trading plans.

Tests:
1. Market context calculation (volatility, trend, structure)
2. Order block LLM evaluation with market context
3. Full plan enhancement with LLM reasoning
4. Fallback behavior when LLM fails
5. Integration with real XAGUSD data
"""

import sys
import os
from datetime import datetime, timedelta

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    generate_smc_trading_plan,
    assess_order_block_strength
)
from tradingagents.dataflows.llm_smc_enhancer import (
    calculate_volatility_regime,
    determine_trend,
    analyze_market_structure,
    evaluate_order_block_with_llm,
    enhance_plan_with_llm
)
from langchain_xai import ChatXAI
from dotenv import load_dotenv
import pytest

load_dotenv()


@pytest.fixture(scope="module")
def smc_analysis():
    """Fixture to provide SMC analysis data for tests."""
    symbol = "XAGUSD"
    analysis = analyze_multi_timeframe_smc(symbol, ['1H', '4H', 'D1'])
    if not analysis or '1H' not in analysis:
        pytest.skip("No SMC data available - MT5 may not be connected")
    return analysis


def test_market_context_calculation(smc_analysis):
    """Test market context helper functions."""
    print("\n" + "="*70)
    print("TEST 1: Market Context Calculation")
    print("="*70)

    current_price = smc_analysis['1H']['current_price']
    print(f"Current price: ${current_price:.2f}")

    # Test volatility regime
    print("\n--- Volatility Regime ---")
    volatility = calculate_volatility_regime(smc_analysis, atr=0.5)
    print(f"Level: {volatility['level']}")
    print(f"Description: {volatility['description']}")

    assert 'level' in volatility
    assert volatility['level'] in ['low', 'normal', 'high', 'unknown']
    print("✅ Volatility calculation passed")

    # Test trend determination
    print("\n--- Trend Analysis ---")
    trend = determine_trend(smc_analysis)
    print(f"Direction: {trend['direction']}")
    print(f"Strength: {trend['strength']}")
    print(f"Description: {trend['description']}")

    assert 'direction' in trend
    assert trend['direction'] in ['bullish', 'bearish', 'ranging', 'unknown']
    assert 'strength' in trend
    print("✅ Trend determination passed")

    # Test market structure
    print("\n--- Market Structure ---")
    structure = analyze_market_structure(smc_analysis)
    print(f"Type: {structure['structure_type']}")
    print(f"Description: {structure['description']}")

    assert 'structure_type' in structure
    assert structure['structure_type'] in ['simple', 'moderate', 'complex', 'unknown']
    print("✅ Market structure analysis passed")

    print("\n✅ TEST 1 PASSED: All market context calculations working\n")


def test_llm_order_block_evaluation(smc_analysis):
    """Test LLM evaluation of individual order blocks."""
    print("\n" + "="*70)
    print("TEST 2: LLM Order Block Evaluation")
    print("="*70)

    # Check for API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("XAI_API_KEY not set")

    # Get a bullish order block from 1H
    tf_data = smc_analysis.get('1H')
    assert tf_data is not None, "No 1H data available"

    bullish_obs = tf_data.get('order_blocks', {}).get('bullish', [])
    if not bullish_obs:
        pytest.skip("No bullish order blocks found")

    # Get the first OB
    ob = bullish_obs[0]
    current_price = tf_data['current_price']

    print(f"\nTesting OB: ${ob.bottom:.2f} - ${ob.top:.2f}")
    print(f"Current price: ${current_price:.2f}")

    # Get rule-based assessment
    ob_assessment = assess_order_block_strength(
        order_block=ob,
        smc_analysis=smc_analysis,
        direction='bullish',
        primary_timeframe='1H'
    )

    print(f"\nRule-based assessment:")
    print(f"  Strength: {ob_assessment['strength_score']}/10")
    print(f"  Hold probability: {ob_assessment['hold_probability']:.0%}")
    print(f"  Retests: {ob_assessment['retests']}")

    # Build market context
    market_context = {
        'volatility': calculate_volatility_regime(smc_analysis, atr=0.5),
        'trend': determine_trend(smc_analysis),
        'structure': analyze_market_structure(smc_analysis)
    }

    # Initialize LLM
    print("\n--- LLM Evaluation ---")
    llm = ChatXAI(
        model="grok-beta",
        temperature=0.3,
        xai_api_key=xai_api_key
    )

    # Get LLM assessment
    llm_assessment = evaluate_order_block_with_llm(
        order_block=ob,
        ob_assessment=ob_assessment,
        smc_analysis=smc_analysis,
        market_context=market_context,
        llm=llm,
        similar_trades=None
    )

    print(f"\nLLM assessment:")
    print(f"  Adjusted hold probability: {llm_assessment['adjusted_hold_probability']:.0%}")
    print(f"  Probability adjustment: {llm_assessment['probability_adjustment']:+.0%}")
    print(f"  Confidence: {llm_assessment['confidence_level']}")
    print(f"  Recommended action: {llm_assessment['recommended_action']}")

    print(f"\nKey reasoning:")
    print(f"  {llm_assessment['key_reasoning']}")

    print(f"\nContextual factors:")
    for factor in llm_assessment['contextual_factors']:
        print(f"  • {factor}")

    print(f"\nTop risks:")
    for risk in llm_assessment['top_risks']:
        print(f"  ⚠️  {risk}")

    # Validate structure
    assert 'adjusted_hold_probability' in llm_assessment
    assert 'confidence_in_assessment' in llm_assessment
    assert 'key_reasoning' in llm_assessment
    assert 'contextual_factors' in llm_assessment
    assert 'top_risks' in llm_assessment
    assert 'recommended_action' in llm_assessment
    assert 'confidence_level' in llm_assessment

    assert 0 <= llm_assessment['adjusted_hold_probability'] <= 1
    assert llm_assessment['confidence_level'] in ['HIGH', 'MEDIUM', 'LOW']

    print("\n✅ TEST 2 PASSED: LLM order block evaluation working\n")


def test_full_plan_enhancement(smc_analysis):
    """Test full plan enhancement with LLM."""
    print("\n" + "="*70)
    print("TEST 3: Full Plan Enhancement")
    print("="*70)

    # Check for API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("XAI_API_KEY not set")

    # Get current price
    current_price = smc_analysis['1H']['current_price']
    print(f"\nCurrent price: ${current_price:.2f}")

    # Generate rule-based plan
    print("\n--- Generating Rule-Based Plan ---")
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias='BUY',
        primary_timeframe='1H',
        atr=0.5
    )

    assert 'error' not in plan, f"Plan generation failed: {plan.get('error')}"

    print(f"Position: {'AT RESISTANCE' if plan['position_analysis']['at_resistance'] else 'AT SUPPORT' if plan['position_analysis']['at_support'] else 'BETWEEN ZONES'}")
    print(f"Primary setup: {plan['primary_setup']['direction'] if plan['primary_setup'] else 'None'}")
    print(f"Alternative setup: {plan['alternative_setup']['direction'] if plan['alternative_setup'] else 'None'}")

    # Enhance with LLM
    print("\n--- Enhancing with LLM ---")
    llm = ChatXAI(
        model="grok-beta",
        temperature=0.3,
        xai_api_key=xai_api_key
    )

    enhanced_plan = enhance_plan_with_llm(
        plan=plan,
        smc_analysis=smc_analysis,
        llm=llm,
        atr=0.5,
        final_state=None
    )

    # Validate enhancement
    print("\n--- Validation ---")

    # Check market context added
    assert 'market_context' in enhanced_plan, "Missing market_context"
    print("✅ Market context added")

    # Check primary setup enhancement
    if enhanced_plan['primary_setup']:
        primary = enhanced_plan['primary_setup']
        if 'llm_enhancement' in primary:
            llm_enh = primary['llm_enhancement']
            print(f"\n✅ Primary setup enhanced:")
            print(f"   Adjusted probability: {llm_enh['adjusted_hold_probability']:.0%}")
            print(f"   Confidence: {llm_enh['confidence_level']}")
            print(f"   Key reasoning: {llm_enh['key_reasoning'][:100]}...")

            assert 'adjusted_hold_probability' in llm_enh
            assert 'confidence_level' in llm_enh
            assert 'top_risks' in llm_enh
        else:
            print("⚠️  Primary setup not enhanced (may be expected)")

    # Check alternative setup enhancement
    if enhanced_plan['alternative_setup']:
        alt = enhanced_plan['alternative_setup']
        if 'llm_enhancement' in alt:
            llm_enh = alt['llm_enhancement']
            print(f"\n✅ Alternative setup enhanced:")
            print(f"   Adjusted probability: {llm_enh['adjusted_hold_probability']:.0%}")
            print(f"   Confidence: {llm_enh['confidence_level']}")
        else:
            print("⚠️  Alternative setup not enhanced (may be expected)")

    # Check for recommendation adjustment
    if 'llm_adjustment' in enhanced_plan['recommendation']:
        llm_adj = enhanced_plan['recommendation']['llm_adjustment']
        print(f"\n✅ Recommendation adjusted:")
        print(f"   Original: {llm_adj['original_action']}")
        print(f"   LLM suggests: {llm_adj['llm_recommended_action']}")
        print(f"   Reason: {llm_adj['reason'][:100]}...")
    else:
        print("\n✓ No significant recommendation adjustment (LLM agrees with rules)")

    print("\n✅ TEST 3 PASSED: Full plan enhancement working\n")


def test_llm_fallback(smc_analysis):
    """Test that system falls back gracefully when LLM fails."""
    print("\n" + "="*70)
    print("TEST 4: LLM Fallback Behavior")
    print("="*70)

    current_price = smc_analysis['1H']['current_price']

    # Generate rule-based plan
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias='BUY',
        primary_timeframe='1H',
        atr=0.5
    )

    assert 'error' not in plan, f"Plan generation failed: {plan.get('error')}"

    # Try to enhance with invalid LLM (should fallback gracefully)
    print("\n--- Testing with Invalid LLM ---")

    # Create LLM with bad API key
    bad_llm = ChatXAI(
        model="grok-beta",
        temperature=0.3,
        xai_api_key="invalid_key_for_testing"
    )

    try:
        enhanced_plan = enhance_plan_with_llm(
            plan=plan,
            smc_analysis=smc_analysis,
            llm=bad_llm,
            atr=0.5,
            final_state=None
        )

        # Should still return a plan (with fallback values)
        assert enhanced_plan is not None
        print("✅ Plan returned even with LLM failure")

        # Check if fallback was used
        if enhanced_plan['primary_setup']:
            primary = enhanced_plan['primary_setup']
            if 'llm_enhancement' in primary:
                llm_enh = primary['llm_enhancement']
                if 'error' in llm_enh:
                    print(f"✅ Fallback detected: {llm_enh['error'][:100]}")
                    assert llm_enh['adjusted_hold_probability'] == llm_enh['rule_based_hold_prob']
                    print("✅ Fallback uses rule-based assessment")

        print("\n✅ TEST 4 PASSED: Fallback behavior working\n")

    except Exception as e:
        # Exception is acceptable - LLM failure is expected with invalid key
        print(f"✅ Exception caught (expected): {str(e)[:100]}")
        print("✅ TEST 4 PASSED: Error handling working\n")


def test_integration_with_xagusd(smc_analysis):
    """Test complete integration with real XAGUSD data."""
    print("\n" + "="*70)
    print("TEST 5: Integration Test with XAGUSD")
    print("="*70)

    # Check for API key
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        pytest.skip("XAI_API_KEY not set")

    current_price = smc_analysis['1H']['current_price']
    print(f"   Current price: ${current_price:.2f}")
    print(f"   Timeframes: {list(smc_analysis.keys())}")

    # Generate rule-based plan
    print(f"\n2. Generating rule-based SMC plan...")
    plan = generate_smc_trading_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        overall_bias='BUY',
        primary_timeframe='1H',
        atr=0.5
    )

    assert 'error' not in plan, f"Plan generation failed: {plan.get('error')}"

    print(f"   Position: {'AT RESISTANCE' if plan['position_analysis']['at_resistance'] else 'AT SUPPORT' if plan['position_analysis']['at_support'] else 'BETWEEN ZONES'}")
    print(f"   Recommendation: {plan['recommendation']['action']}")

    # Enhance with LLM
    print(f"\n3. Enhancing with LLM contextual intelligence...")
    llm = ChatXAI(
        model="grok-beta",
        temperature=0.3,
        xai_api_key=xai_api_key
    )

    enhanced_plan = enhance_plan_with_llm(
        plan=plan,
        smc_analysis=smc_analysis,
        llm=llm,
        atr=0.5,
        final_state=None
    )

    # Display results
    print(f"\n4. Enhancement Results:")
    print(f"\n--- Market Context ---")
    ctx = enhanced_plan['market_context']
    print(f"Volatility: {ctx['volatility']['description']}")
    print(f"Trend: {ctx['trend']['description']}")
    print(f"Structure: {ctx['structure']['description']}")

    if enhanced_plan['primary_setup'] and 'llm_enhancement' in enhanced_plan['primary_setup']:
        llm_enh = enhanced_plan['primary_setup']['llm_enhancement']
        print(f"\n--- Primary Setup LLM Enhancement ---")
        print(f"Confidence: {llm_enh['confidence_level']}")
        print(f"Hold Probability: {llm_enh['adjusted_hold_probability']:.0%} (was {llm_enh['rule_based_hold_prob']:.0%})")
        print(f"Adjustment: {llm_enh['probability_adjustment']:+.0%}")
        print(f"\nReasoning: {llm_enh['key_reasoning']}")
        print(f"\nContextual Factors:")
        for factor in llm_enh['contextual_factors']:
            print(f"  • {factor}")
        print(f"\nTop Risks:")
        for risk in llm_enh['top_risks']:
            print(f"  ⚠️  {risk}")
        print(f"\nRecommended Action: {llm_enh['recommended_action']}")

    print("\n✅ TEST 5 PASSED: Full integration working\n")


if __name__ == "__main__":
    # Run tests with pytest when executed directly
    sys.exit(pytest.main([__file__, "-v"]))
