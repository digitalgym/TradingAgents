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


def calculate_ob_confluence(
    order_blocks: List,
    all_timeframes_data: Dict[str, Any],
    direction: str,
    entry_price: float,
    price_tolerance_pct: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Calculate confluence scores for order blocks across timeframes.

    Args:
        order_blocks: Order blocks from primary timeframe (1H)
        all_timeframes_data: Full SMC analysis with all timeframes
        direction: 'BUY' or 'SELL'
        entry_price: Entry price for distance calculation
        price_tolerance_pct: Price tolerance for alignment detection

    Returns:
        List of dicts with:
        - ob: The order block object
        - score: Confluence score (1.0 to 2.0)
        - aligned_timeframes: List of timeframes where OB appears
        - distance: Distance from entry price
    """
    if price_tolerance_pct is None:
        price_tolerance_pct = {'4H': 5.0, 'D1': 10.0}

    scored_obs = []

    for ob in order_blocks:
        score = 1.0  # Base score for appearing in primary timeframe (1H)
        aligned_tfs = ['1H']
        ob_mid = (ob.top + ob.bottom) / 2

        # Check for alignment in 4H
        if '4H' in all_timeframes_data:
            htf_obs = all_timeframes_data['4H'].get('order_blocks', {}).get(
                'bullish' if direction == 'BUY' else 'bearish', []
            )
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100

                if price_diff_pct <= price_tolerance_pct['4H']:
                    score += 0.5
                    aligned_tfs.append('4H')
                    break  # Only count once per timeframe

        # Check for alignment in D1
        if 'D1' in all_timeframes_data:
            htf_obs = all_timeframes_data['D1'].get('order_blocks', {}).get(
                'bullish' if direction == 'BUY' else 'bearish', []
            )
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100

                if price_diff_pct <= price_tolerance_pct['D1']:
                    score += 0.5
                    aligned_tfs.append('D1')
                    break

        # Calculate distance from entry
        if direction == 'BUY':
            distance = entry_price - ob.top
        else:  # SELL
            distance = ob.bottom - entry_price

        scored_obs.append({
            'ob': ob,
            'score': score,
            'aligned_timeframes': aligned_tfs,
            'distance': distance
        })

    return scored_obs


def suggest_smc_stop_loss(
    smc_analysis: Dict[str, Any],
    direction: str,
    entry_price: float,
    atr: Optional[float] = None,
    atr_multiplier: float = 2.0,
    primary_timeframe: str = '1H'
) -> Optional[Dict[str, Any]]:
    """
    Suggest stop loss using multi-timeframe confluence.

    Strategy:
    1. Get 1H order blocks (user's preferred timeframe)
    2. Score them by alignment with 4H and D1
    3. Select best (highest score, then closest)
    4. Fallback to any HTF OBs if no 1H OBs
    5. Final fallback to ATR

    Args:
        smc_analysis: Multi-timeframe SMC analysis (dict with all TFs)
        direction: 'BUY' or 'SELL'
        entry_price: Planned entry price
        atr: ATR value for fallback stop loss
        atr_multiplier: Multiplier for ATR-based stop
        primary_timeframe: Primary timeframe for OBs (default: '1H')

    Returns:
        dict with stop loss suggestion or None
    """
    if direction == 'BUY':
        # 1. PRIMARY: 1H order blocks with multi-timeframe confluence
        if primary_timeframe in smc_analysis:
            primary_tf_data = smc_analysis[primary_timeframe]
            order_blocks = primary_tf_data.get('order_blocks', {}).get('bullish', [])
            valid_obs = [ob for ob in order_blocks if ob.top < entry_price]

            if valid_obs:
                # Score by confluence with higher timeframes
                scored_obs = calculate_ob_confluence(
                    order_blocks=valid_obs,
                    all_timeframes_data=smc_analysis,
                    direction='BUY',
                    entry_price=entry_price
                )

                # Sort by: 1) score (desc), 2) distance (asc)
                scored_obs.sort(key=lambda x: (-x['score'], x['distance']))
                best_ob_data = scored_obs[0]
                best_ob = best_ob_data['ob']

                stop_price = best_ob.bottom * 0.998
                distance_pct = ((entry_price - stop_price) / entry_price * 100)

                return {
                    'price': stop_price,
                    'zone_top': best_ob.top,
                    'zone_bottom': best_ob.bottom,
                    'source': f"1H Bullish OB (Confluence: {best_ob_data['score']:.1f})",
                    'strength': best_ob.strength if hasattr(best_ob, 'strength') else 0.8,
                    'confluence_score': best_ob_data['score'],
                    'aligned_timeframes': best_ob_data['aligned_timeframes'],
                    'distance_pct': distance_pct,
                    'reason': f"Below 1H bullish OB at ${best_ob.bottom:.2f} (aligned: {', '.join(best_ob_data['aligned_timeframes'])})"
                }

        # 2. FALLBACK: Check HTF order blocks (4H, D1) if no 1H OBs found
        for tf in ['4H', 'D1']:
            if tf not in smc_analysis:
                continue

            tf_data = smc_analysis[tf]
            order_blocks = tf_data.get('order_blocks', {}).get('bullish', [])
            valid_obs = [ob for ob in order_blocks if ob.top < entry_price]

            if valid_obs:
                closest_ob = min(valid_obs, key=lambda ob: entry_price - ob.top)
                stop_price = closest_ob.bottom * 0.998
                distance_pct = ((entry_price - stop_price) / entry_price * 100)

                return {
                    'price': stop_price,
                    'zone_top': closest_ob.top,
                    'zone_bottom': closest_ob.bottom,
                    'source': f'{tf} Bullish OB (fallback)',
                    'strength': closest_ob.strength if hasattr(closest_ob, 'strength') else 0.8,
                    'confluence_score': 1.0,
                    'aligned_timeframes': [tf],
                    'distance_pct': distance_pct,
                    'reason': f"Below {tf} bullish Order Block at ${closest_ob.bottom:.2f}"
                }

        # 3. FINAL FALLBACK: ATR-based stop loss
        if atr:
            stop_price = entry_price - (atr * atr_multiplier)
            distance_pct = ((entry_price - stop_price) / entry_price * 100)

            return {
                'price': stop_price,
                'zone_top': entry_price,
                'zone_bottom': stop_price,
                'source': f'ATR({atr_multiplier}x)',
                'strength': 0.5,
                'confluence_score': 0.0,
                'aligned_timeframes': [],
                'distance_pct': distance_pct,
                'reason': f"ATR-based stop: {atr_multiplier}x ATR below entry"
            }

        # Only return None if absolutely no data available
        return None

    elif direction == 'SELL':
        # 1. PRIMARY: 1H order blocks with multi-timeframe confluence
        if primary_timeframe in smc_analysis:
            primary_tf_data = smc_analysis[primary_timeframe]
            order_blocks = primary_tf_data.get('order_blocks', {}).get('bearish', [])
            valid_obs = [ob for ob in order_blocks if ob.bottom > entry_price]

            if valid_obs:
                # Score by confluence with higher timeframes
                scored_obs = calculate_ob_confluence(
                    order_blocks=valid_obs,
                    all_timeframes_data=smc_analysis,
                    direction='SELL',
                    entry_price=entry_price
                )

                # Sort by: 1) score (desc), 2) distance (asc)
                scored_obs.sort(key=lambda x: (-x['score'], x['distance']))
                best_ob_data = scored_obs[0]
                best_ob = best_ob_data['ob']

                stop_price = best_ob.top * 1.002
                distance_pct = ((stop_price - entry_price) / entry_price * 100)

                return {
                    'price': stop_price,
                    'zone_top': best_ob.top,
                    'zone_bottom': best_ob.bottom,
                    'source': f"1H Bearish OB (Confluence: {best_ob_data['score']:.1f})",
                    'strength': best_ob.strength if hasattr(best_ob, 'strength') else 0.8,
                    'confluence_score': best_ob_data['score'],
                    'aligned_timeframes': best_ob_data['aligned_timeframes'],
                    'distance_pct': distance_pct,
                    'reason': f"Above 1H bearish OB at ${best_ob.top:.2f} (aligned: {', '.join(best_ob_data['aligned_timeframes'])})"
                }

        # 2. FALLBACK: Check HTF order blocks (4H, D1) if no 1H OBs found
        for tf in ['4H', 'D1']:
            if tf not in smc_analysis:
                continue

            tf_data = smc_analysis[tf]
            order_blocks = tf_data.get('order_blocks', {}).get('bearish', [])
            valid_obs = [ob for ob in order_blocks if ob.bottom > entry_price]

            if valid_obs:
                closest_ob = min(valid_obs, key=lambda ob: ob.bottom - entry_price)
                stop_price = closest_ob.top * 1.002
                distance_pct = ((stop_price - entry_price) / entry_price * 100)

                return {
                    'price': stop_price,
                    'zone_top': closest_ob.top,
                    'zone_bottom': closest_ob.bottom,
                    'source': f'{tf} Bearish OB (fallback)',
                    'strength': closest_ob.strength if hasattr(closest_ob, 'strength') else 0.8,
                    'confluence_score': 1.0,
                    'aligned_timeframes': [tf],
                    'distance_pct': distance_pct,
                    'reason': f"Above {tf} bearish Order Block at ${closest_ob.top:.2f}"
                }

        # 3. FINAL FALLBACK: ATR-based stop loss (handles ATH scenario)
        if atr:
            stop_price = entry_price + (atr * atr_multiplier)
            distance_pct = ((stop_price - entry_price) / entry_price * 100)

            return {
                'price': stop_price,
                'zone_top': stop_price,
                'zone_bottom': entry_price,
                'source': f'ATR({atr_multiplier}x)',
                'strength': 0.5,
                'confluence_score': 0.0,
                'aligned_timeframes': [],
                'distance_pct': distance_pct,
                'reason': f"ATR-based stop: {atr_multiplier}x ATR above entry (ATH scenario)"
            }

        # Only return None if absolutely no data available
        return None

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
        resistance_zones = smc_analysis.get('zones', {}).get('resistance', [])

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

        # Fallback: Use FVGs if not enough zones
        if len(targets) < num_targets:
            fvgs = smc_analysis.get('fair_value_gaps', {}).get('bearish', [])
            valid_fvgs = [fvg for fvg in fvgs if fvg.bottom > entry_price]
            valid_fvgs.sort(key=lambda fvg: fvg.bottom - entry_price)

            for fvg in valid_fvgs[:num_targets - len(targets)]:
                distance_pct = ((fvg.bottom - entry_price) / entry_price * 100)
                targets.append({
                    'number': len(targets) + 1,
                    'price': fvg.bottom,
                    'zone_top': fvg.top,
                    'zone_bottom': fvg.bottom,
                    'source': 'Bearish FVG',
                    'strength': 0.7,
                    'distance_pct': distance_pct,
                    'reason': f"At Bearish FVG ${fvg.bottom:.2f}-${fvg.top:.2f}"
                })
    
    else:  # SELL
        # For sells, look for support below entry
        support_zones = smc_analysis.get('zones', {}).get('support', [])

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

        # Fallback: Use FVGs if not enough zones
        if len(targets) < num_targets:
            fvgs = smc_analysis.get('fair_value_gaps', {}).get('bullish', [])
            valid_fvgs = [fvg for fvg in fvgs if fvg.top < entry_price]
            valid_fvgs.sort(key=lambda fvg: entry_price - fvg.top)

            for fvg in valid_fvgs[:num_targets - len(targets)]:
                distance_pct = ((entry_price - fvg.top) / entry_price * 100)
                targets.append({
                    'number': len(targets) + 1,
                    'price': fvg.top,
                    'zone_top': fvg.top,
                    'zone_bottom': fvg.bottom,
                    'source': 'Bullish FVG',
                    'strength': 0.7,
                    'distance_pct': distance_pct,
                    'reason': f"At Bullish FVG ${fvg.bottom:.2f}-${fvg.top:.2f}"
                })

    return targets


def suggest_smc_entry_strategy(
    smc_analysis: Dict[str, Any],
    direction: str,
    current_price: float,
    primary_timeframe: str = '1H'
) -> Dict[str, Any]:
    """
    Suggest optimal entry strategy based on SMC principles.

    Returns both market entry and limit order entry at order blocks,
    with guidance on which strategy to use based on current price position.

    Args:
        smc_analysis: Multi-timeframe SMC analysis
        direction: 'BUY' or 'SELL'
        current_price: Current market price
        primary_timeframe: Preferred timeframe for order blocks (default: '1H')

    Returns:
        Dict with:
        - market_entry: Dict with market order details
        - limit_entry: Dict with limit order details at order block
        - recommendation: Which strategy to use
        - distance_to_zone: Distance from current price to optimal entry zone
    """
    result = {
        'market_entry': {
            'price': current_price,
            'type': 'MARKET',
            'reason': 'Immediate execution at current market price'
        },
        'limit_entry': None,
        'recommendation': None,
        'distance_to_zone_pct': None
    }

    # Get primary timeframe data
    if primary_timeframe not in smc_analysis:
        result['recommendation'] = 'MARKET'
        result['limit_entry'] = {'reason': f'No {primary_timeframe} data available'}
        return result

    primary_tf_data = smc_analysis[primary_timeframe]

    if direction == 'BUY':
        # Find bullish order blocks below current price
        order_blocks = primary_tf_data.get('order_blocks', {}).get('bullish', [])
        valid_obs = [ob for ob in order_blocks if ob.top < current_price]

        if not valid_obs:
            result['recommendation'] = 'MARKET'
            result['limit_entry'] = {'reason': 'No bullish order blocks below current price - enter at market'}
            return result

        # Find closest order block
        closest_ob = min(valid_obs, key=lambda ob: current_price - ob.top)

        # Calculate distance
        distance_pct = ((current_price - closest_ob.top) / current_price * 100)
        result['distance_to_zone_pct'] = distance_pct

        # Suggest limit entry in the order block zone
        # Entry at top of OB (best price in zone)
        limit_price = closest_ob.top

        # Check if aligned with higher timeframes
        confluence_score = 1.0
        aligned_tfs = [primary_timeframe]

        ob_mid = (closest_ob.top + closest_ob.bottom) / 2

        # Check 4H alignment
        if '4H' in smc_analysis:
            htf_obs = smc_analysis['4H'].get('order_blocks', {}).get('bullish', [])
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100
                if price_diff_pct <= 5.0:
                    confluence_score += 0.5
                    aligned_tfs.append('4H')
                    break

        # Check D1 alignment
        if 'D1' in smc_analysis:
            htf_obs = smc_analysis['D1'].get('order_blocks', {}).get('bullish', [])
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100
                if price_diff_pct <= 10.0:
                    confluence_score += 0.5
                    aligned_tfs.append('D1')
                    break

        result['limit_entry'] = {
            'price': limit_price,
            'zone_top': closest_ob.top,
            'zone_bottom': closest_ob.bottom,
            'type': 'BUY_LIMIT',
            'confluence_score': confluence_score,
            'aligned_timeframes': aligned_tfs,
            'reason': f"Wait for pullback to {primary_timeframe} bullish OB at ${closest_ob.bottom:.2f}-${closest_ob.top:.2f}"
        }

        # Recommendation logic
        if distance_pct < 0.5:
            # Very close to order block (< 0.5%)
            result['recommendation'] = 'MARKET'
            result['recommendation_reason'] = f"Price is already at the order block (only {distance_pct:.2f}% away) - enter now"
        elif distance_pct < 2.0:
            # Close to order block (0.5-2%)
            result['recommendation'] = 'LIMIT_OR_MARKET'
            result['recommendation_reason'] = f"Price is near the order block ({distance_pct:.2f}% away) - either strategy works"
        else:
            # Far from order block (>2%)
            result['recommendation'] = 'LIMIT'
            result['recommendation_reason'] = f"Price is {distance_pct:.2f}% above optimal entry - wait for pullback to order block"

    elif direction == 'SELL':
        # Find bearish order blocks above current price
        order_blocks = primary_tf_data.get('order_blocks', {}).get('bearish', [])
        valid_obs = [ob for ob in order_blocks if ob.bottom > current_price]

        if not valid_obs:
            result['recommendation'] = 'MARKET'
            result['limit_entry'] = {'reason': 'No bearish order blocks above current price - enter at market'}
            return result

        # Find closest order block
        closest_ob = min(valid_obs, key=lambda ob: ob.bottom - current_price)

        # Calculate distance
        distance_pct = ((closest_ob.bottom - current_price) / current_price * 100)
        result['distance_to_zone_pct'] = distance_pct

        # Suggest limit entry in the order block zone
        # Entry at bottom of OB (best price in zone)
        limit_price = closest_ob.bottom

        # Check if aligned with higher timeframes
        confluence_score = 1.0
        aligned_tfs = [primary_timeframe]

        ob_mid = (closest_ob.top + closest_ob.bottom) / 2

        # Check 4H alignment
        if '4H' in smc_analysis:
            htf_obs = smc_analysis['4H'].get('order_blocks', {}).get('bearish', [])
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100
                if price_diff_pct <= 5.0:
                    confluence_score += 0.5
                    aligned_tfs.append('4H')
                    break

        # Check D1 alignment
        if 'D1' in smc_analysis:
            htf_obs = smc_analysis['D1'].get('order_blocks', {}).get('bearish', [])
            for htf_ob in htf_obs:
                htf_mid = (htf_ob.top + htf_ob.bottom) / 2
                price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100
                if price_diff_pct <= 10.0:
                    confluence_score += 0.5
                    aligned_tfs.append('D1')
                    break

        result['limit_entry'] = {
            'price': limit_price,
            'zone_top': closest_ob.top,
            'zone_bottom': closest_ob.bottom,
            'type': 'SELL_LIMIT',
            'confluence_score': confluence_score,
            'aligned_timeframes': aligned_tfs,
            'reason': f"Wait for rally to {primary_timeframe} bearish OB at ${closest_ob.bottom:.2f}-${closest_ob.top:.2f}"
        }

        # Recommendation logic
        if distance_pct < 0.5:
            # Very close to order block (< 0.5%)
            result['recommendation'] = 'MARKET'
            result['recommendation_reason'] = f"Price is already at the order block (only {distance_pct:.2f}% away) - enter now"
        elif distance_pct < 2.0:
            # Close to order block (0.5-2%)
            result['recommendation'] = 'LIMIT_OR_MARKET'
            result['recommendation_reason'] = f"Price is near the order block ({distance_pct:.2f}% away) - either strategy works"
        else:
            # Far from order block (>2%)
            result['recommendation'] = 'LIMIT'
            result['recommendation_reason'] = f"Price is {distance_pct:.2f}% below optimal entry - wait for rally to order block"

    return result


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
