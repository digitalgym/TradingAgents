"""
Smart Money Concepts Utilities

Helper functions for multi-timeframe SMC analysis and integration
with trading decisions.
"""

import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Any, Optional
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer


def get_mt5_data_for_smc(
    symbol: str,
    timeframe: int,
    bars: int = 500
) -> Optional[pd.DataFrame]:
    """
    Get MT5 data formatted for SMC analysis.
    
    Args:
        symbol: Trading symbol
        timeframe: MT5 timeframe constant (mt5.TIMEFRAME_H1, etc.)
        bars: Number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not mt5.initialize():
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
        columns={'tick_volume': 'volume'}
    )


def analyze_multi_timeframe_smc(
    symbol: str,
    timeframes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze SMC across multiple timeframes.
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes to analyze (default: ['1H', '4H', 'D1'])
    
    Returns:
        dict with analysis for each timeframe
    """
    if timeframes is None:
        timeframes = ['1H', '4H', 'D1']
    
    # Map timeframe strings to MT5 constants
    tf_map = {
        '1H': mt5.TIMEFRAME_H1,
        '4H': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1
    }
    
    analyzer = SmartMoneyAnalyzer(
        swing_lookback=5,
        ob_strength_threshold=0.5,
        fvg_min_size_atr=0.3
    )
    
    results = {}
    
    for tf_name in timeframes:
        if tf_name not in tf_map:
            continue
        
        df = get_mt5_data_for_smc(symbol, tf_map[tf_name], bars=500)
        
        if df is not None and len(df) > 50:
            analysis = analyzer.analyze_full_smc(df)
            results[tf_name] = analysis
    
    return results


def get_key_smc_levels(
    smc_analysis: Dict[str, Any],
    level_type: str = 'both'
) -> List[Dict[str, Any]]:
    """
    Extract key SMC levels for TP/SL placement.
    
    Args:
        smc_analysis: SMC analysis from analyze_full_smc
        level_type: 'support', 'resistance', or 'both'
    
    Returns:
        List of key levels with prices and metadata
    """
    levels = []
    
    if level_type in ['support', 'both']:
        for zone in smc_analysis['zones']['support']:
            levels.append({
                'type': 'support',
                'price': zone['bottom'],  # Use bottom of support zone
                'zone_top': zone['top'],
                'zone_bottom': zone['bottom'],
                'source': zone['type'],
                'strength': zone['strength']
            })
    
    if level_type in ['resistance', 'both']:
        for zone in smc_analysis['zones']['resistance']:
            levels.append({
                'type': 'resistance',
                'price': zone['top'],  # Use top of resistance zone
                'zone_top': zone['top'],
                'zone_bottom': zone['bottom'],
                'source': zone['type'],
                'strength': zone['strength']
            })
    
    return levels


def suggest_smc_stop_loss(
    smc_analysis: Dict[str, Any],
    direction: str,
    entry_price: float,
    max_distance_pct: float = 3.0
) -> Optional[Dict[str, Any]]:
    """
    Suggest stop loss based on SMC levels.
    
    Args:
        smc_analysis: SMC analysis from analyze_full_smc
        direction: 'BUY' or 'SELL'
        entry_price: Planned entry price
        max_distance_pct: Maximum stop distance as % of entry
    
    Returns:
        dict with stop loss suggestion or None
    """
    if direction == 'BUY':
        # For buys, look for support below entry
        support_zones = smc_analysis['zones']['support']
        
        # Find nearest support below entry
        valid_supports = [
            z for z in support_zones
            if z['top'] < entry_price and
            ((entry_price - z['bottom']) / entry_price * 100) <= max_distance_pct
        ]
        
        if valid_supports:
            # Use strongest support
            best_support = max(valid_supports, key=lambda z: z['strength'])
            
            # Place stop just below the zone
            stop_price = best_support['bottom'] * 0.998  # 0.2% below zone
            
            return {
                'price': stop_price,
                'zone_top': best_support['top'],
                'zone_bottom': best_support['bottom'],
                'source': best_support['type'],
                'strength': best_support['strength'],
                'distance_pct': ((entry_price - stop_price) / entry_price * 100),
                'reason': f"Below {best_support['type']} zone at ${best_support['bottom']:.2f}"
            }
    
    else:  # SELL
        # For sells, look for resistance above entry
        resistance_zones = smc_analysis['zones']['resistance']
        
        valid_resistances = [
            z for z in resistance_zones
            if z['bottom'] > entry_price and
            ((z['top'] - entry_price) / entry_price * 100) <= max_distance_pct
        ]
        
        if valid_resistances:
            best_resistance = max(valid_resistances, key=lambda z: z['strength'])
            
            # Place stop just above the zone
            stop_price = best_resistance['top'] * 1.002  # 0.2% above zone
            
            return {
                'price': stop_price,
                'zone_top': best_resistance['top'],
                'zone_bottom': best_resistance['bottom'],
                'source': best_resistance['type'],
                'strength': best_resistance['strength'],
                'distance_pct': ((stop_price - entry_price) / entry_price * 100),
                'reason': f"Above {best_resistance['type']} zone at ${best_resistance['top']:.2f}"
            }
    
    return None


def suggest_smc_take_profits(
    smc_analysis: Dict[str, Any],
    direction: str,
    entry_price: float,
    num_targets: int = 3
) -> List[Dict[str, Any]]:
    """
    Suggest take profit targets based on SMC levels.
    
    Args:
        smc_analysis: SMC analysis from analyze_full_smc
        direction: 'BUY' or 'SELL'
        entry_price: Planned entry price
        num_targets: Number of TP targets to suggest
    
    Returns:
        List of TP suggestions
    """
    targets = []
    
    if direction == 'BUY':
        # For buys, look for resistance above entry
        resistance_zones = smc_analysis['zones']['resistance']
        
        valid_resistances = [
            z for z in resistance_zones
            if z['bottom'] > entry_price
        ]
        
        # Sort by distance from entry
        valid_resistances.sort(key=lambda z: z['bottom'] - entry_price)
        
        for i, zone in enumerate(valid_resistances[:num_targets], 1):
            # Target the bottom of resistance zone (conservative)
            target_price = zone['bottom']
            
            distance_pct = ((target_price - entry_price) / entry_price * 100)
            
            targets.append({
                'number': i,
                'price': target_price,
                'zone_top': zone['top'],
                'zone_bottom': zone['bottom'],
                'source': zone['type'],
                'strength': zone['strength'],
                'distance_pct': distance_pct,
                'reason': f"At {zone['type']} zone ${zone['bottom']:.2f}-${zone['top']:.2f}"
            })
    
    else:  # SELL
        # For sells, look for support below entry
        support_zones = smc_analysis['zones']['support']
        
        valid_supports = [
            z for z in support_zones
            if z['top'] < entry_price
        ]
        
        # Sort by distance from entry
        valid_supports.sort(key=lambda z: entry_price - z['top'], reverse=False)
        
        for i, zone in enumerate(valid_supports[:num_targets], 1):
            # Target the top of support zone (conservative)
            target_price = zone['top']
            
            distance_pct = ((entry_price - target_price) / entry_price * 100)
            
            targets.append({
                'number': i,
                'price': target_price,
                'zone_top': zone['top'],
                'zone_bottom': zone['bottom'],
                'source': zone['type'],
                'strength': zone['strength'],
                'distance_pct': distance_pct,
                'reason': f"At {zone['type']} zone ${zone['bottom']:.2f}-${zone['top']:.2f}"
            })
    
    return targets


def format_smc_for_prompt(
    mtf_analysis: Dict[str, Any],
    symbol: str
) -> str:
    """
    Format multi-timeframe SMC analysis for LLM prompt.
    
    Args:
        mtf_analysis: Multi-timeframe SMC analysis
        symbol: Trading symbol
    
    Returns:
        Formatted string for prompt injection
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SMART MONEY CONCEPTS ANALYSIS - {symbol}")
    lines.append(f"{'='*70}\n")
    
    for tf, analysis in mtf_analysis.items():
        lines.append(f"[{tf} TIMEFRAME]")
        lines.append(f"  Bias: {analysis['bias'].upper()}")
        lines.append(f"  Order Blocks: {analysis['order_blocks']['unmitigated']} unmitigated")
        lines.append(f"  Fair Value Gaps: {analysis['fair_value_gaps']['unmitigated']} unmitigated")
        
        if analysis['structure']['recent_choc']:
            lines.append(f"  ⚠️  CHOC detected: {len(analysis['structure']['recent_choc'])} recent")
        
        if analysis['structure']['recent_bos']:
            lines.append(f"  ✓ BOS detected: {len(analysis['structure']['recent_bos'])} recent")
        
        # Key levels
        if analysis['nearest_support']:
            s = analysis['nearest_support']
            dist = ((analysis['current_price'] - s['top']) / analysis['current_price'] * 100)
            lines.append(f"  Nearest Support: ${s['bottom']:.2f}-${s['top']:.2f} ({s['type']}) | -{dist:.2f}%")
        
        if analysis['nearest_resistance']:
            r = analysis['nearest_resistance']
            dist = ((r['bottom'] - analysis['current_price']) / analysis['current_price'] * 100)
            lines.append(f"  Nearest Resistance: ${r['bottom']:.2f}-${r['top']:.2f} ({r['type']}) | +{dist:.2f}%")
        
        lines.append("")
    
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


def get_htf_bias_alignment(mtf_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if higher timeframe biases are aligned.
    
    Args:
        mtf_analysis: Multi-timeframe SMC analysis
    
    Returns:
        dict with alignment info
    """
    biases = {tf: analysis['bias'] for tf, analysis in mtf_analysis.items()}
    
    # Check alignment
    unique_biases = set(biases.values())
    
    if len(unique_biases) == 1 and 'neutral' not in unique_biases:
        # All aligned
        return {
            'aligned': True,
            'bias': list(unique_biases)[0],
            'strength': 'strong',
            'message': f"All timeframes aligned {list(unique_biases)[0].upper()}"
        }
    
    # Check if majority aligned
    bullish_count = sum(1 for b in biases.values() if b == 'bullish')
    bearish_count = sum(1 for b in biases.values() if b == 'bearish')
    
    if bullish_count > bearish_count:
        return {
            'aligned': True,
            'bias': 'bullish',
            'strength': 'moderate',
            'message': f"{bullish_count}/{len(biases)} timeframes bullish"
        }
    elif bearish_count > bullish_count:
        return {
            'aligned': True,
            'bias': 'bearish',
            'strength': 'moderate',
            'message': f"{bearish_count}/{len(biases)} timeframes bearish"
        }
    
    return {
        'aligned': False,
        'bias': 'neutral',
        'strength': 'weak',
        'message': "Timeframes not aligned - mixed signals"
    }


def validate_trade_against_smc(
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    smc_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate a trade plan against SMC levels.
    
    Args:
        direction: 'BUY' or 'SELL'
        entry_price: Planned entry
        stop_loss: Planned stop loss
        take_profit: Planned take profit
        smc_analysis: SMC analysis
    
    Returns:
        dict with validation results and suggestions
    """
    issues = []
    suggestions = []
    
    current_price = smc_analysis['current_price']
    
    # Check if entry is near unmitigated zone
    if direction == 'BUY':
        # Check if entry is near support
        nearest_support = smc_analysis['nearest_support']
        if nearest_support:
            if abs(entry_price - nearest_support['top']) / entry_price < 0.01:
                suggestions.append(f"✓ Entry aligns with {nearest_support['type']} at ${nearest_support['top']:.2f}")
            elif entry_price < nearest_support['bottom']:
                issues.append(f"⚠️  Entry below support zone (${nearest_support['bottom']:.2f}-${nearest_support['top']:.2f})")
        
        # Check stop loss
        if stop_loss > nearest_support['bottom'] if nearest_support else False:
            issues.append(f"⚠️  Stop loss (${stop_loss:.2f}) above support zone - may get stopped out prematurely")
            suggestions.append(f"Consider moving stop to ${nearest_support['bottom'] * 0.998:.2f} (below support)")
        
        # Check take profit
        nearest_resistance = smc_analysis['nearest_resistance']
        if nearest_resistance:
            if take_profit > nearest_resistance['top']:
                suggestions.append(f"✓ TP beyond resistance zone (${nearest_resistance['bottom']:.2f}-${nearest_resistance['top']:.2f})")
            elif take_profit < nearest_resistance['bottom']:
                issues.append(f"⚠️  TP (${take_profit:.2f}) before resistance - leaving profit on table")
                suggestions.append(f"Consider TP at ${nearest_resistance['bottom']:.2f} (resistance zone)")
    
    else:  # SELL
        # Check if entry is near resistance
        nearest_resistance = smc_analysis['nearest_resistance']
        if nearest_resistance:
            if abs(entry_price - nearest_resistance['bottom']) / entry_price < 0.01:
                suggestions.append(f"✓ Entry aligns with {nearest_resistance['type']} at ${nearest_resistance['bottom']:.2f}")
            elif entry_price > nearest_resistance['top']:
                issues.append(f"⚠️  Entry above resistance zone (${nearest_resistance['bottom']:.2f}-${nearest_resistance['top']:.2f})")
        
        # Check stop loss
        if stop_loss < nearest_resistance['top'] if nearest_resistance else False:
            issues.append(f"⚠️  Stop loss (${stop_loss:.2f}) below resistance zone - may get stopped out prematurely")
            suggestions.append(f"Consider moving stop to ${nearest_resistance['top'] * 1.002:.2f} (above resistance)")
        
        # Check take profit
        nearest_support = smc_analysis['nearest_support']
        if nearest_support:
            if take_profit < nearest_support['bottom']:
                suggestions.append(f"✓ TP beyond support zone (${nearest_support['bottom']:.2f}-${nearest_support['top']:.2f})")
            elif take_profit > nearest_support['top']:
                issues.append(f"⚠️  TP (${take_profit:.2f}) before support - leaving profit on table")
                suggestions.append(f"Consider TP at ${nearest_support['top']:.2f} (support zone)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'suggestions': suggestions,
        'score': max(0, 100 - (len(issues) * 20))
    }
