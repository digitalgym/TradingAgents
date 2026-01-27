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


def assess_order_block_strength(
    order_block: Any,
    smc_analysis: Dict[str, Any],
    direction: str,
    primary_timeframe: str = '1H'
) -> Dict[str, Any]:
    """
    Assess the strength of an order block based on multiple factors.

    Args:
        order_block: Order block object to assess
        smc_analysis: Full multi-timeframe SMC analysis
        direction: 'BUY' or 'SELL'
        primary_timeframe: Primary timeframe for the order block

    Returns:
        dict with strength assessment:
        - strength_score: 0.0-10.0 (10 = strongest)
        - retests: Number of times price returned to zone
        - confluence_score: Multi-timeframe alignment (1.0-2.0)
        - aligned_timeframes: List of timeframes confirming this OB
        - volume_profile: 'high', 'medium', 'low'
        - breakout_probability: 0.0-1.0 (probability zone breaks)
        - hold_probability: 0.0-1.0 (probability zone holds)
        - assessment: Text description
    """
    # Base strength from order block object
    base_strength = order_block.strength if hasattr(order_block, 'strength') else 0.5

    # Count retests (touches after formation)
    retests = order_block.retests if hasattr(order_block, 'retests') else 0

    # Calculate multi-timeframe confluence
    ob_mid = (order_block.top + order_block.bottom) / 2
    confluence_score = 1.0
    aligned_tfs = [primary_timeframe]

    # Check alignment with higher timeframes
    price_tolerance_pct = {'4H': 5.0, 'D1': 10.0}

    for tf in ['4H', 'D1']:
        if tf == primary_timeframe or tf not in smc_analysis:
            continue

        tf_data = smc_analysis[tf]
        htf_obs = tf_data.get('order_blocks', {}).get(
            'bullish' if direction == 'BUY' else 'bearish', []
        )

        for htf_ob in htf_obs:
            htf_mid = (htf_ob.top + htf_ob.bottom) / 2
            price_diff_pct = abs(htf_mid - ob_mid) / ob_mid * 100

            if price_diff_pct <= price_tolerance_pct.get(tf, 5.0):
                confluence_score += 0.5
                aligned_tfs.append(tf)
                break

    # Volume assessment
    volume_profile = 'medium'
    if hasattr(order_block, 'volume_ratio'):
        if order_block.volume_ratio > 1.5:
            volume_profile = 'high'
        elif order_block.volume_ratio < 0.8:
            volume_profile = 'low'

    # Calculate strength score (0-10)
    strength_score = 0.0

    # Base strength (0-3 points)
    strength_score += base_strength * 3

    # Retests (0-3 points) - more retests = stronger zone
    # But diminishing returns after 3 retests
    if retests >= 3:
        strength_score += 3.0
    elif retests == 2:
        strength_score += 2.5
    elif retests == 1:
        strength_score += 1.5

    # Multi-timeframe confluence (0-3 points)
    if confluence_score >= 2.0:  # Triple alignment
        strength_score += 3.0
    elif confluence_score >= 1.5:  # Double alignment
        strength_score += 2.0
    else:  # Single timeframe
        strength_score += 1.0

    # Volume profile (0-1 point)
    if volume_profile == 'high':
        strength_score += 1.0
    elif volume_profile == 'medium':
        strength_score += 0.5

    # Cap at 10
    strength_score = min(10.0, strength_score)

    # Calculate breakout/hold probability
    # Strong zones (>7.0) are likely to hold (80%+)
    # Weak zones (<4.0) are likely to break (60%+)
    # Retests reduce hold probability (zone getting weaker)

    if strength_score >= 7.0:
        base_hold_prob = 0.85
    elif strength_score >= 5.0:
        base_hold_prob = 0.65
    elif strength_score >= 3.0:
        base_hold_prob = 0.50
    else:
        base_hold_prob = 0.35

    # Each retest reduces hold probability by 5%
    retest_penalty = min(retests * 0.05, 0.20)
    hold_probability = max(0.2, base_hold_prob - retest_penalty)
    breakout_probability = 1.0 - hold_probability

    # Generate assessment text
    if strength_score >= 7.5:
        strength_desc = "VERY STRONG"
    elif strength_score >= 6.0:
        strength_desc = "STRONG"
    elif strength_score >= 4.0:
        strength_desc = "MODERATE"
    elif strength_score >= 2.0:
        strength_desc = "WEAK"
    else:
        strength_desc = "VERY WEAK"

    retest_desc = f"{retests} retest{'s' if retests != 1 else ''}"
    confluence_desc = f"{confluence_score:.1f}x confluence ({', '.join(aligned_tfs)})"

    assessment = f"{strength_desc} order block | {retest_desc} | {confluence_desc} | {volume_profile} volume"

    return {
        'strength_score': round(strength_score, 1),
        'retests': retests,
        'confluence_score': confluence_score,
        'aligned_timeframes': aligned_tfs,
        'volume_profile': volume_profile,
        'breakout_probability': round(breakout_probability, 2),
        'hold_probability': round(hold_probability, 2),
        'assessment': assessment,
        'strength_category': strength_desc
    }


def generate_smc_trading_plan(
    smc_analysis: Dict[str, Any],
    current_price: float,
    overall_bias: str,
    primary_timeframe: str = '1H',
    atr: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive multi-scenario SMC trading plan.

    Analyzes current price position relative to SMC zones and generates
    appropriate setups:
    - At resistance: SHORT setup + conditional LONG re-entry
    - At support: LONG setup + conditional SHORT re-entry
    - Between zones: PRIMARY setup based on bias + alternative scenario

    Args:
        smc_analysis: Full multi-timeframe SMC analysis
        current_price: Current market price
        overall_bias: Overall market bias ('BUY' or 'SELL')
        primary_timeframe: Primary timeframe (default: '1H')
        atr: ATR value for risk management

    Returns:
        dict with:
        - position_analysis: Where price is relative to SMC zones
        - primary_setup: Main trading setup (SHORT at resistance or LONG at support)
        - alternative_setup: Conditional setup for opposite direction
        - order_block_assessments: Strength analysis of relevant OBs
        - recommendation: Overall trading recommendation
    """
    if primary_timeframe not in smc_analysis:
        return {
            'error': f'No {primary_timeframe} data available',
            'position_analysis': None,
            'primary_setup': None,
            'alternative_setup': None
        }

    primary_tf_data = smc_analysis[primary_timeframe]

    # Get order blocks
    bullish_obs = primary_tf_data.get('order_blocks', {}).get('bullish', [])
    bearish_obs = primary_tf_data.get('order_blocks', {}).get('bearish', [])

    # Find nearest order blocks above and below current price
    nearest_resistance_ob = None
    nearest_support_ob = None

    resistances_above = [ob for ob in bearish_obs if ob.bottom > current_price]
    supports_below = [ob for ob in bullish_obs if ob.top < current_price]

    if resistances_above:
        nearest_resistance_ob = min(resistances_above, key=lambda ob: ob.bottom - current_price)

    if supports_below:
        nearest_support_ob = max(supports_below, key=lambda ob: ob.top)

    # Determine price position
    at_resistance = False
    at_support = False
    in_between = True

    if nearest_resistance_ob:
        distance_to_resistance_pct = ((nearest_resistance_ob.bottom - current_price) / current_price * 100)
        if distance_to_resistance_pct < 0.5 or (current_price >= nearest_resistance_ob.bottom and current_price <= nearest_resistance_ob.top):
            at_resistance = True
            in_between = False

    if nearest_support_ob:
        distance_to_support_pct = ((current_price - nearest_support_ob.top) / current_price * 100)
        if distance_to_support_pct < 0.5 or (current_price >= nearest_support_ob.bottom and current_price <= nearest_support_ob.top):
            at_support = True
            in_between = False

    # Assess order block strengths
    resistance_assessment = None
    support_assessment = None

    if nearest_resistance_ob:
        resistance_assessment = assess_order_block_strength(
            nearest_resistance_ob,
            smc_analysis,
            'SELL',
            primary_timeframe
        )

    if nearest_support_ob:
        support_assessment = assess_order_block_strength(
            nearest_support_ob,
            smc_analysis,
            'BUY',
            primary_timeframe
        )

    # Generate position analysis
    position_analysis = {
        'current_price': current_price,
        'at_resistance': at_resistance,
        'at_support': at_support,
        'in_between': in_between,
        'nearest_resistance': {
            'price_range': (nearest_resistance_ob.bottom, nearest_resistance_ob.top) if nearest_resistance_ob else None,
            'distance_pct': distance_to_resistance_pct if nearest_resistance_ob else None,
            'assessment': resistance_assessment
        } if nearest_resistance_ob else None,
        'nearest_support': {
            'price_range': (nearest_support_ob.bottom, nearest_support_ob.top) if nearest_support_ob else None,
            'distance_pct': distance_to_support_pct if nearest_support_ob else None,
            'assessment': support_assessment
        } if nearest_support_ob else None
    }

    # Generate primary setup based on position
    primary_setup = None
    alternative_setup = None
    recommendation = None

    if at_resistance:
        # Price at resistance - PRIMARY: SHORT setup
        primary_setup = _generate_short_setup(
            current_price,
            nearest_resistance_ob,
            resistance_assessment,
            nearest_support_ob,
            smc_analysis,
            primary_timeframe,
            atr
        )

        # ALTERNATIVE: LONG re-entry after pullback
        if nearest_support_ob:
            alternative_setup = _generate_long_reentry_setup(
                nearest_support_ob,
                support_assessment,
                nearest_resistance_ob,
                smc_analysis,
                primary_timeframe,
                atr
            )

        # Recommendation based on OB strength
        if resistance_assessment and resistance_assessment['hold_probability'] >= 0.65:
            recommendation = {
                'action': 'SHORT NOW',
                'confidence': 'HIGH' if resistance_assessment['hold_probability'] >= 0.75 else 'MEDIUM',
                'reason': f"Price at {resistance_assessment['strength_category']} resistance OB ({resistance_assessment['hold_probability']:.0%} hold probability). {resistance_assessment['assessment']}.",
                'alternative': f"If resistance breaks, wait for pullback to ${nearest_resistance_ob.top:.2f} to re-enter LONG" if nearest_resistance_ob else None
            }
        else:
            recommendation = {
                'action': 'WAIT FOR CONFIRMATION',
                'confidence': 'LOW',
                'reason': f"At WEAK resistance OB ({resistance_assessment['breakout_probability']:.0%} breakout probability). Wait for rejection confirmation or breakout.",
                'alternative': f"On breakout, consider LONG above ${nearest_resistance_ob.top:.2f}" if nearest_resistance_ob else None
            }

    elif at_support:
        # Price at support - PRIMARY: LONG setup
        primary_setup = _generate_long_setup(
            current_price,
            nearest_support_ob,
            support_assessment,
            nearest_resistance_ob,
            smc_analysis,
            primary_timeframe,
            atr
        )

        # ALTERNATIVE: SHORT re-entry after rejection
        if nearest_resistance_ob:
            alternative_setup = _generate_short_reentry_setup(
                nearest_resistance_ob,
                resistance_assessment,
                nearest_support_ob,
                smc_analysis,
                primary_timeframe,
                atr
            )

        # Recommendation based on OB strength
        if support_assessment and support_assessment['hold_probability'] >= 0.65:
            recommendation = {
                'action': 'LONG NOW',
                'confidence': 'HIGH' if support_assessment['hold_probability'] >= 0.75 else 'MEDIUM',
                'reason': f"Price at {support_assessment['strength_category']} support OB ({support_assessment['hold_probability']:.0%} hold probability). {support_assessment['assessment']}.",
                'alternative': f"If support breaks, wait for retest of ${nearest_support_ob.bottom:.2f} to re-enter SHORT" if nearest_support_ob else None
            }
        else:
            recommendation = {
                'action': 'WAIT FOR CONFIRMATION',
                'confidence': 'LOW',
                'reason': f"At WEAK support OB ({support_assessment['breakout_probability']:.0%} breakout probability). Wait for bounce confirmation or breakdown.",
                'alternative': f"On breakdown, consider SHORT below ${nearest_support_ob.bottom:.2f}" if nearest_support_ob else None
            }

    else:  # in_between
        # Price between zones - use overall bias
        if overall_bias == 'BUY':
            # PRIMARY: LONG at support
            if nearest_support_ob:
                primary_setup = _generate_long_setup(
                    nearest_support_ob.top,  # Entry at top of support zone
                    nearest_support_ob,
                    support_assessment,
                    nearest_resistance_ob,
                    smc_analysis,
                    primary_timeframe,
                    atr
                )

            recommendation = {
                'action': 'WAIT FOR PULLBACK',
                'confidence': 'MEDIUM',
                'reason': f"Price {distance_to_support_pct:.1f}% above support OB. Overall bias is BULLISH. Wait for pullback to ${nearest_support_ob.top:.2f} for better entry." if nearest_support_ob else "Overall bias is BULLISH but no clear support zone nearby.",
                'alternative': f"If price reaches ${nearest_resistance_ob.bottom:.2f} resistance, consider SHORT" if nearest_resistance_ob else None
            }

        else:  # SELL bias
            # PRIMARY: SHORT at resistance
            if nearest_resistance_ob:
                primary_setup = _generate_short_setup(
                    nearest_resistance_ob.bottom,  # Entry at bottom of resistance zone
                    nearest_resistance_ob,
                    resistance_assessment,
                    nearest_support_ob,
                    smc_analysis,
                    primary_timeframe,
                    atr
                )

            recommendation = {
                'action': 'WAIT FOR RALLY',
                'confidence': 'MEDIUM',
                'reason': f"Price {distance_to_resistance_pct:.1f}% below resistance OB. Overall bias is BEARISH. Wait for rally to ${nearest_resistance_ob.bottom:.2f} for better entry." if nearest_resistance_ob else "Overall bias is BEARISH but no clear resistance zone nearby.",
                'alternative': f"If price reaches ${nearest_support_ob.top:.2f} support, consider LONG" if nearest_support_ob else None
            }

    return {
        'position_analysis': position_analysis,
        'primary_setup': primary_setup,
        'alternative_setup': alternative_setup,
        'order_block_assessments': {
            'resistance': resistance_assessment,
            'support': support_assessment
        },
        'recommendation': recommendation
    }


def _generate_short_setup(
    entry_price: float,
    resistance_ob: Any,
    resistance_assessment: Dict[str, Any],
    nearest_support_ob: Any,
    smc_analysis: Dict[str, Any],
    primary_timeframe: str,
    atr: Optional[float]
) -> Dict[str, Any]:
    """Generate SHORT setup details."""
    # Entry at current price or top of resistance zone
    entry = entry_price

    # Stop loss above resistance OB
    stop_loss = resistance_ob.top * 1.002 if resistance_ob else entry * 1.02

    # Take profit at support OB or using R:R
    if nearest_support_ob:
        tp1 = nearest_support_ob.top
        tp2 = nearest_support_ob.bottom * 0.998
    else:
        # Use 2:1 R:R if no support found
        risk = stop_loss - entry
        tp1 = entry - (risk * 1.5)
        tp2 = entry - (risk * 2.5)

    return {
        'direction': 'SELL',
        'entry_price': entry,
        'entry_type': 'MARKET' if abs(entry - entry_price) < entry * 0.002 else 'LIMIT',
        'entry_zone': (resistance_ob.bottom, resistance_ob.top) if resistance_ob else None,
        'stop_loss': stop_loss,
        'stop_loss_reason': f"Above {primary_timeframe} resistance OB" if resistance_ob else "ATR-based",
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'tp_reason': f"At support OB ${nearest_support_ob.bottom:.2f}-${nearest_support_ob.top:.2f}" if nearest_support_ob else "2:1 R:R",
        'risk_pct': ((stop_loss - entry) / entry * 100),
        'reward_pct_tp1': ((entry - tp1) / entry * 100),
        'reward_pct_tp2': ((entry - tp2) / entry * 100),
        'ob_strength': resistance_assessment,
        'rationale': f"SHORT at resistance. OB strength: {resistance_assessment['strength_score']}/10. Hold probability: {resistance_assessment['hold_probability']:.0%}."
    }


def _generate_long_setup(
    entry_price: float,
    support_ob: Any,
    support_assessment: Dict[str, Any],
    nearest_resistance_ob: Any,
    smc_analysis: Dict[str, Any],
    primary_timeframe: str,
    atr: Optional[float]
) -> Dict[str, Any]:
    """Generate LONG setup details."""
    # Entry at current price or bottom of support zone
    entry = entry_price

    # Stop loss below support OB
    stop_loss = support_ob.bottom * 0.998 if support_ob else entry * 0.98

    # Take profit at resistance OB or using R:R
    if nearest_resistance_ob:
        tp1 = nearest_resistance_ob.bottom
        tp2 = nearest_resistance_ob.top * 1.002
    else:
        # Use 2:1 R:R if no resistance found
        risk = entry - stop_loss
        tp1 = entry + (risk * 1.5)
        tp2 = entry + (risk * 2.5)

    return {
        'direction': 'BUY',
        'entry_price': entry,
        'entry_type': 'MARKET' if abs(entry - entry_price) < entry * 0.002 else 'LIMIT',
        'entry_zone': (support_ob.bottom, support_ob.top) if support_ob else None,
        'stop_loss': stop_loss,
        'stop_loss_reason': f"Below {primary_timeframe} support OB" if support_ob else "ATR-based",
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'tp_reason': f"At resistance OB ${nearest_resistance_ob.bottom:.2f}-${nearest_resistance_ob.top:.2f}" if nearest_resistance_ob else "2:1 R:R",
        'risk_pct': ((entry - stop_loss) / entry * 100),
        'reward_pct_tp1': ((tp1 - entry) / entry * 100),
        'reward_pct_tp2': ((tp2 - entry) / entry * 100),
        'ob_strength': support_assessment,
        'rationale': f"LONG at support. OB strength: {support_assessment['strength_score']}/10. Hold probability: {support_assessment['hold_probability']:.0%}."
    }


def _generate_long_reentry_setup(
    support_ob: Any,
    support_assessment: Dict[str, Any],
    nearest_resistance_ob: Any,
    smc_analysis: Dict[str, Any],
    primary_timeframe: str,
    atr: Optional[float]
) -> Dict[str, Any]:
    """Generate LONG re-entry setup after SHORT closes."""
    entry = support_ob.top  # Enter at top of support zone on pullback
    stop_loss = support_ob.bottom * 0.998

    if nearest_resistance_ob:
        tp1 = nearest_resistance_ob.bottom
        tp2 = nearest_resistance_ob.top * 1.002
    else:
        risk = entry - stop_loss
        tp1 = entry + (risk * 1.5)
        tp2 = entry + (risk * 2.5)

    return {
        'direction': 'BUY',
        'entry_price': entry,
        'entry_type': 'LIMIT',
        'entry_zone': (support_ob.bottom, support_ob.top),
        'stop_loss': stop_loss,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'trigger_condition': f"After SHORT position closes at TP1 (${tp1:.2f}), wait for pullback to support",
        'risk_pct': ((entry - stop_loss) / entry * 100),
        'reward_pct_tp1': ((tp1 - entry) / entry * 100),
        'reward_pct_tp2': ((tp2 - entry) / entry * 100),
        'ob_strength': support_assessment,
        'rationale': f"Re-enter LONG at support after SHORT completes. Support strength: {support_assessment['strength_score']}/10."
    }


def get_smc_position_review_context(
    symbol: str,
    direction: str,
    entry_price: float,
    current_price: float,
    sl: float = 0,
    tp: float = 0,
    timeframe: str = 'H1',
    lookback: int = 100
) -> Dict[str, Any]:
    """
    Generate SMC context for reviewing an open position.

    Reusable function for both web API and automation.
    Analyzes current market structure relative to the position.

    Args:
        symbol: Trading symbol
        direction: 'BUY' or 'SELL'
        entry_price: Position entry price
        current_price: Current market price
        sl: Current stop loss (0 if none)
        tp: Current take profit (0 if none)
        timeframe: Primary timeframe for analysis (default: 'H1')
        lookback: Bars to analyze (default: 100)

    Returns:
        dict with:
        - smc_context: Formatted string for LLM prompt
        - bias: Current market bias
        - bias_aligns: Whether bias aligns with position direction
        - support_levels: List of support zones
        - resistance_levels: List of resistance zones
        - sl_at_risk: Whether SL is near liquidity/weak zone
        - suggested_sl: SMC-based SL suggestion
        - suggested_tp: SMC-based TP suggestion
        - structure_shift: Whether there's been a CHOCH against position
        - raw_analysis: Full SMC analysis data
    """
    try:
        # Get OHLCV data
        if not mt5.initialize():
            return {'error': 'MT5 not initialized', 'smc_context': ''}

        # Map timeframe string to MT5 constant
        tf_map = {
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            '1H': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            '4H': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, lookback)
        if rates is None or len(rates) < 50:
            return {'error': 'Insufficient price data', 'smc_context': ''}

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Run SMC analysis
        analyzer = SmartMoneyAnalyzer(fvg_min_size_atr=0.1)
        order_blocks = analyzer.detect_order_blocks(df, lookback=lookback)
        fvgs = analyzer.detect_fair_value_gaps(df, lookback=lookback)
        swing_points = analyzer.detect_swing_points(df, lookback=lookback)
        structure_breaks = analyzer.detect_structure_breaks(df, swing_points)
        zones = analyzer.get_unmitigated_zones(order_blocks, fvgs, current_price)

        # Determine market bias
        bias = analyzer._determine_bias(
            structure_breaks.get('recent_bos', []),
            structure_breaks.get('recent_choc', []),
            zones,
            current_price
        )

        # Check if bias aligns with position
        if direction.upper() == 'BUY':
            bias_aligns = bias == 'bullish'
        else:
            bias_aligns = bias == 'bearish'

        # Check for structure shift against position (CHOCH)
        recent_choc = structure_breaks.get('choc', [])[-3:]
        structure_shift = False
        for choc in recent_choc:
            # CHOCH against a BUY = bearish CHOCH, against SELL = bullish CHOCH
            if direction.upper() == 'BUY' and hasattr(choc, 'type') and choc.type == 'bearish':
                structure_shift = True
            elif direction.upper() == 'SELL' and hasattr(choc, 'type') and choc.type == 'bullish':
                structure_shift = True

        # Format support/resistance levels
        support_levels = []
        resistance_levels = []

        for z in zones.get('support', [])[:3]:
            mid = (z.get('top', 0) + z.get('bottom', 0)) / 2
            support_levels.append({
                'type': z.get('type', 'zone'),
                'price': mid,
                'top': z.get('top'),
                'bottom': z.get('bottom'),
                'strength': z.get('strength', 0.5)
            })

        for z in zones.get('resistance', [])[:3]:
            mid = (z.get('top', 0) + z.get('bottom', 0)) / 2
            resistance_levels.append({
                'type': z.get('type', 'zone'),
                'price': mid,
                'top': z.get('top'),
                'bottom': z.get('bottom'),
                'strength': z.get('strength', 0.5)
            })

        # Check if SL is at risk (near liquidity pool or weak)
        sl_at_risk = False
        sl_risk_reason = None
        if sl > 0:
            # Check if SL is above support (for BUY) or below resistance (for SELL)
            if direction.upper() == 'BUY' and support_levels:
                nearest_support = support_levels[0]
                if sl > nearest_support['bottom']:
                    sl_at_risk = True
                    sl_risk_reason = f"SL ({sl:.5f}) is above support zone ({nearest_support['bottom']:.5f}) - may get stopped out before bounce"
            elif direction.upper() == 'SELL' and resistance_levels:
                nearest_resistance = resistance_levels[0]
                if sl < nearest_resistance['top']:
                    sl_at_risk = True
                    sl_risk_reason = f"SL ({sl:.5f}) is below resistance zone ({nearest_resistance['top']:.5f}) - may get stopped out before rejection"

        # Calculate ATR for fallback trailing SL
        atr = None
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            tr_list = []
            for i in range(1, len(df)):
                tr_list.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                ))
            if len(tr_list) >= 14:
                atr = sum(tr_list[-14:]) / 14
        except Exception:
            pass

        # Generate SMC-based SL/TP suggestions
        suggested_sl = None
        suggested_tp = None
        suggested_tp_source = None
        trailing_sl = None
        trailing_sl_source = None

        if direction.upper() == 'BUY':
            # SL below nearest support
            if support_levels:
                suggested_sl = support_levels[0]['bottom'] * 0.998
            # TP at nearest resistance
            if resistance_levels:
                suggested_tp = resistance_levels[0]['bottom']
                suggested_tp_source = f"{resistance_levels[0]['type']} @ {resistance_levels[0]['price']:.5f}"
            elif atr and sl > 0:
                # Fallback: ATR-based TP (2:1 R:R from current SL)
                risk = entry_price - sl
                suggested_tp = current_price + (risk * 2)
                suggested_tp_source = "ATR 2:1 R:R"
            elif atr:
                # No SL set - use 2x ATR from current price
                suggested_tp = current_price + (atr * 2)
                suggested_tp_source = "ATR (2x)"

            # Trailing SL: Find most recent bullish OB/FVG between entry and current price
            # (levels price has already cleared that can now act as support)
            if current_price > entry_price:  # Only trail if in profit
                # Look for bullish zones between entry and current price
                cleared_supports = [
                    s for s in support_levels
                    if s['top'] > entry_price and s['top'] < current_price
                ]
                if cleared_supports:
                    # Use the highest cleared support (closest to current price)
                    best_trail = max(cleared_supports, key=lambda s: s['top'])
                    trailing_sl = best_trail['bottom'] * 0.998
                    trailing_sl_source = f"{best_trail['type']} @ {best_trail['price']:.5f}"
                elif atr:
                    # Fallback: ATR-based trailing (1.5 ATR from current price)
                    trailing_sl = current_price - (atr * 1.5)
                    trailing_sl_source = "ATR (1.5x)"
        else:  # SELL
            # SL above nearest resistance
            if resistance_levels:
                suggested_sl = resistance_levels[0]['top'] * 1.002
            # TP at nearest support
            if support_levels:
                suggested_tp = support_levels[0]['top']
                suggested_tp_source = f"{support_levels[0]['type']} @ {support_levels[0]['price']:.5f}"
            elif atr and sl > 0:
                # Fallback: ATR-based TP (2:1 R:R from current SL)
                risk = sl - entry_price
                suggested_tp = current_price - (risk * 2)
                suggested_tp_source = "ATR 2:1 R:R"
            elif atr:
                # No SL set - use 2x ATR from current price
                suggested_tp = current_price - (atr * 2)
                suggested_tp_source = "ATR (2x)"

            # Trailing SL: Find most recent bearish OB/FVG between entry and current price
            if current_price < entry_price:  # Only trail if in profit
                # Look for bearish zones between current price and entry
                cleared_resistances = [
                    r for r in resistance_levels
                    if r['bottom'] < entry_price and r['bottom'] > current_price
                ]
                if cleared_resistances:
                    # Use the lowest cleared resistance (closest to current price)
                    best_trail = min(cleared_resistances, key=lambda r: r['bottom'])
                    trailing_sl = best_trail['top'] * 1.002
                    trailing_sl_source = f"{best_trail['type']} @ {best_trail['price']:.5f}"
                elif atr:
                    # Fallback: ATR-based trailing (1.5 ATR from current price)
                    trailing_sl = current_price + (atr * 1.5)
                    trailing_sl_source = "ATR (1.5x)"

        # Build context string for LLM
        unmitigated_obs = [ob for ob in order_blocks if not ob.mitigated]
        unmitigated_fvgs = [fvg for fvg in fvgs if not fvg.mitigated]

        support_str = '\n'.join([
            f"  - {s['type']} @ {s['price']:.5f} (strength: {s['strength']:.0%})"
            for s in support_levels
        ]) if support_levels else '  None identified'

        resistance_str = '\n'.join([
            f"  - {r['type']} @ {r['price']:.5f} (strength: {r['strength']:.0%})"
            for r in resistance_levels
        ]) if resistance_levels else '  None identified'

        smc_context = f"""
CURRENT MARKET STRUCTURE (SMC {timeframe}):
Market Bias: {bias.upper()}
Bias Alignment: {'✓ ALIGNED with position' if bias_aligns else '⚠️ AGAINST position direction'}
Structure Shift: {'⚠️ CHOCH detected AGAINST position' if structure_shift else 'No recent shift against position'}

Unmitigated Order Blocks: {len(unmitigated_obs)} ({len([ob for ob in unmitigated_obs if ob.type == 'bullish'])} bullish, {len([ob for ob in unmitigated_obs if ob.type == 'bearish'])} bearish)
Unmitigated FVGs: {len(unmitigated_fvgs)} ({len([fvg for fvg in unmitigated_fvgs if fvg.type == 'bullish'])} bullish, {len([fvg for fvg in unmitigated_fvgs if fvg.type == 'bearish'])} bearish)

KEY SUPPORT LEVELS (below price):
{support_str}

KEY RESISTANCE LEVELS (above price):
{resistance_str}

SL ASSESSMENT: {'⚠️ ' + sl_risk_reason if sl_at_risk else '✓ SL placement appears safe'}
"""

        return {
            'smc_context': smc_context,
            'bias': bias,
            'bias_aligns': bias_aligns,
            'structure_shift': structure_shift,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'sl_at_risk': sl_at_risk,
            'sl_risk_reason': sl_risk_reason,
            'suggested_sl': suggested_sl,
            'suggested_tp': suggested_tp,
            'suggested_tp_source': suggested_tp_source,
            'trailing_sl': trailing_sl,
            'trailing_sl_source': trailing_sl_source,
            'unmitigated_obs': len(unmitigated_obs),
            'unmitigated_fvgs': len(unmitigated_fvgs),
            'raw_analysis': {
                'order_blocks': order_blocks,
                'fvgs': fvgs,
                'zones': zones,
                'structure_breaks': structure_breaks
            }
        }

    except Exception as e:
        return {
            'error': str(e),
            'smc_context': f'(SMC analysis unavailable: {str(e)[:50]})'
        }


def _generate_short_reentry_setup(
    resistance_ob: Any,
    resistance_assessment: Dict[str, Any],
    nearest_support_ob: Any,
    smc_analysis: Dict[str, Any],
    primary_timeframe: str,
    atr: Optional[float]
) -> Dict[str, Any]:
    """Generate SHORT re-entry setup after LONG closes."""
    entry = resistance_ob.bottom  # Enter at bottom of resistance zone on rally
    stop_loss = resistance_ob.top * 1.002

    if nearest_support_ob:
        tp1 = nearest_support_ob.top
        tp2 = nearest_support_ob.bottom * 0.998
    else:
        risk = stop_loss - entry
        tp1 = entry - (risk * 1.5)
        tp2 = entry - (risk * 2.5)

    return {
        'direction': 'SELL',
        'entry_price': entry,
        'entry_type': 'LIMIT',
        'entry_zone': (resistance_ob.bottom, resistance_ob.top),
        'stop_loss': stop_loss,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'trigger_condition': f"After LONG position closes at TP1 (${tp1:.2f}), wait for rally to resistance",
        'risk_pct': ((stop_loss - entry) / entry * 100),
        'reward_pct_tp1': ((entry - tp1) / entry * 100),
        'reward_pct_tp2': ((entry - tp2) / entry * 100),
        'ob_strength': resistance_assessment,
        'rationale': f"Re-enter SHORT at resistance after LONG completes. Resistance strength: {resistance_assessment['strength_score']}/10."
    }
