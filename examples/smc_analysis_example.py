"""
Example: Smart Money Concepts Analysis

Demonstrates multi-timeframe SMC analysis and TP/SL suggestions
based on order blocks, FVGs, and unmitigated zones.

Usage:
    python examples/smc_analysis_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.dataflows.smc_utils import (
    analyze_multi_timeframe_smc,
    suggest_smc_stop_loss,
    suggest_smc_take_profits,
    format_smc_for_prompt,
    get_htf_bias_alignment,
    validate_trade_against_smc
)
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer


def main():
    """Demonstrate SMC analysis"""
    
    print("\n" + "="*70)
    print("SMART MONEY CONCEPTS ANALYSIS EXAMPLE")
    print("="*70 + "\n")
    
    symbol = "XAUUSD"
    
    # ========================================================================
    # STEP 1: Multi-Timeframe SMC Analysis
    # ========================================================================
    print("STEP 1: ANALYZING MULTIPLE TIMEFRAMES")
    print("-" * 70)
    
    mtf_analysis = analyze_multi_timeframe_smc(
        symbol=symbol,
        timeframes=['1H', '4H', 'D1']
    )
    
    if not mtf_analysis:
        print("❌ Could not fetch data. Ensure MT5 is running and logged in.")
        return
    
    # Display each timeframe
    for tf, analysis in mtf_analysis.items():
        print(f"\n[{tf} TIMEFRAME]")
        print(f"  Current Price: ${analysis['current_price']:.2f}")
        print(f"  Market Bias: {analysis['bias'].upper()}")
        print(f"  Order Blocks: {analysis['order_blocks']['unmitigated']} unmitigated")
        print(f"  Fair Value Gaps: {analysis['fair_value_gaps']['unmitigated']} unmitigated")
        
        if analysis['structure']['recent_choc']:
            print(f"  ⚠️  CHOC: {len(analysis['structure']['recent_choc'])} recent changes of character")
        
        if analysis['structure']['recent_bos']:
            print(f"  ✓ BOS: {len(analysis['structure']['recent_bos'])} recent breaks of structure")
        
        if analysis['nearest_support']:
            s = analysis['nearest_support']
            dist = ((analysis['current_price'] - s['top']) / analysis['current_price'] * 100)
            print(f"  Support: ${s['bottom']:.2f}-${s['top']:.2f} ({s['type']}) | -{dist:.2f}%")
        
        if analysis['nearest_resistance']:
            r = analysis['nearest_resistance']
            dist = ((r['bottom'] - analysis['current_price']) / analysis['current_price'] * 100)
            print(f"  Resistance: ${r['bottom']:.2f}-${r['top']:.2f} ({r['type']}) | +{dist:.2f}%")
    
    # ========================================================================
    # STEP 2: Check Higher Timeframe Alignment
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: HIGHER TIMEFRAME BIAS ALIGNMENT")
    print("-" * 70)
    
    alignment = get_htf_bias_alignment(mtf_analysis)
    
    print(f"\nAlignment: {'✓ ALIGNED' if alignment['aligned'] else '✗ NOT ALIGNED'}")
    print(f"Bias: {alignment['bias'].upper()}")
    print(f"Strength: {alignment['strength'].upper()}")
    print(f"Message: {alignment['message']}")
    
    # ========================================================================
    # STEP 3: Suggest Stop Loss Based on SMC
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: SMC-BASED STOP LOSS SUGGESTION")
    print("-" * 70)
    
    # Use D1 timeframe for stop loss (higher TF = stronger levels)
    d1_analysis = mtf_analysis.get('D1')
    
    if d1_analysis:
        direction = "BUY" if alignment['bias'] == 'bullish' else "SELL"
        entry_price = d1_analysis['current_price']
        
        stop_suggestion = suggest_smc_stop_loss(
            smc_analysis=d1_analysis,
            direction=direction,
            entry_price=entry_price,
            max_distance_pct=3.0
        )
        
        if stop_suggestion:
            print(f"\nDirection: {direction}")
            print(f"Entry: ${entry_price:.2f}")
            print(f"\nSuggested Stop Loss: ${stop_suggestion['price']:.2f}")
            print(f"  Zone: ${stop_suggestion['zone_bottom']:.2f}-${stop_suggestion['zone_top']:.2f}")
            print(f"  Source: {stop_suggestion['source']}")
            print(f"  Strength: {stop_suggestion['strength']:.0%}")
            print(f"  Distance: {stop_suggestion['distance_pct']:.2f}%")
            print(f"  Reason: {stop_suggestion['reason']}")
        else:
            print(f"\n⚠️  No suitable SMC stop loss found within 3% distance")
    
    # ========================================================================
    # STEP 4: Suggest Take Profit Targets
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: SMC-BASED TAKE PROFIT TARGETS")
    print("-" * 70)
    
    if d1_analysis:
        tp_suggestions = suggest_smc_take_profits(
            smc_analysis=d1_analysis,
            direction=direction,
            entry_price=entry_price,
            num_targets=3
        )
        
        if tp_suggestions:
            print(f"\nSuggested Take Profit Targets:\n")
            for tp in tp_suggestions:
                print(f"TP{tp['number']}: ${tp['price']:.2f}")
                print(f"  Zone: ${tp['zone_bottom']:.2f}-${tp['zone_top']:.2f}")
                print(f"  Source: {tp['source']}")
                print(f"  Distance: +{tp['distance_pct']:.2f}%")
                print(f"  Reason: {tp['reason']}\n")
        else:
            print(f"\n⚠️  No suitable SMC take profit targets found")
    
    # ========================================================================
    # STEP 5: Validate Example Trade Plan
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: VALIDATE TRADE PLAN AGAINST SMC")
    print("-" * 70)
    
    if d1_analysis and stop_suggestion and tp_suggestions:
        # Example trade plan
        example_stop = entry_price * 0.98  # 2% stop
        example_tp = entry_price * 1.05  # 5% target
        
        print(f"\nExample Trade Plan:")
        print(f"  Direction: {direction}")
        print(f"  Entry: ${entry_price:.2f}")
        print(f"  Stop Loss: ${example_stop:.2f}")
        print(f"  Take Profit: ${example_tp:.2f}")
        
        validation = validate_trade_against_smc(
            direction=direction,
            entry_price=entry_price,
            stop_loss=example_stop,
            take_profit=example_tp,
            smc_analysis=d1_analysis
        )
        
        print(f"\nValidation Score: {validation['score']}/100")
        print(f"Valid: {'✓ YES' if validation['valid'] else '✗ NO'}")
        
        if validation['issues']:
            print(f"\nIssues Found:")
            for issue in validation['issues']:
                print(f"  {issue}")
        
        if validation['suggestions']:
            print(f"\nSuggestions:")
            for suggestion in validation['suggestions']:
                print(f"  {suggestion}")
    
    # ========================================================================
    # STEP 6: Format for LLM Prompt
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: FORMATTED FOR LLM PROMPT")
    print("-" * 70)
    
    prompt_text = format_smc_for_prompt(mtf_analysis, symbol)
    print(prompt_text)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nSymbol: {symbol}")
    print(f"HTF Bias: {alignment['bias'].upper()} ({alignment['strength']})")
    
    if stop_suggestion:
        print(f"\nRecommended Stop: ${stop_suggestion['price']:.2f}")
        print(f"  Based on: {stop_suggestion['source']} at ${stop_suggestion['zone_bottom']:.2f}")
    
    if tp_suggestions:
        print(f"\nRecommended Targets:")
        for tp in tp_suggestions[:3]:
            print(f"  TP{tp['number']}: ${tp['price']:.2f} (+{tp['distance_pct']:.1f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
