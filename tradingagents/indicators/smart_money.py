"""
Smart Money Concepts (SMC) Indicators

Implements institutional trading concepts:
- Order Blocks (OB): Last up/down candle before strong move
- Change of Character (CHOC): Momentum shift, breaks previous high/low
- Break of Structure (BOS): Continuation, breaks in trend direction
- Fair Value Gaps (FVG): 3-candle imbalance zones
- Support/Resistance with mitigation tracking

Used for better TP/SL placement aligned with institutional levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderBlock:
    """An order block level"""
    type: str  # bullish or bearish
    top: float
    bottom: float
    candle_index: int
    timestamp: str
    strength: float  # 0-1, based on volume and size
    mitigated: bool = False
    mitigation_index: Optional[int] = None


@dataclass
class FairValueGap:
    """A fair value gap (imbalance)"""
    type: str  # bullish or bearish
    top: float
    bottom: float
    start_index: int
    timestamp: str
    size: float  # gap size in price
    mitigated: bool = False
    mitigation_index: Optional[int] = None


@dataclass
class StructurePoint:
    """A market structure point (swing high/low)"""
    type: str  # high or low
    price: float
    index: int
    timestamp: str
    broken: bool = False
    break_index: Optional[int] = None
    break_type: Optional[str] = None  # BOS or CHOC


class SmartMoneyAnalyzer:
    """
    Analyzes price action for smart money concepts.
    
    Features:
    - Order block detection (last opposing candle before strong move)
    - CHOC detection (momentum shift, counter-trend break)
    - BOS detection (trend continuation break)
    - FVG detection (3-candle imbalance)
    - Unmitigated zone tracking
    """
    
    def __init__(
        self,
        swing_lookback: int = 5,
        ob_strength_threshold: float = 0.5,
        fvg_min_size_atr: float = 0.3,
        structure_break_confirm: int = 1
    ):
        """
        Initialize SMC analyzer.
        
        Args:
            swing_lookback: Periods to look back for swing highs/lows
            ob_strength_threshold: Minimum strength for valid OB (0-1)
            fvg_min_size_atr: Minimum FVG size as multiple of ATR
            structure_break_confirm: Candles to confirm structure break
        """
        self.swing_lookback = swing_lookback
        self.ob_strength_threshold = ob_strength_threshold
        self.fvg_min_size_atr = fvg_min_size_atr
        self.structure_break_confirm = structure_break_confirm
    
    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[OrderBlock]:
        """
        Detect order blocks (last opposing candle before strong move).
        
        An order block is the last up/down candle before a strong opposite move.
        These represent institutional entry zones.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: How many recent candles to analyze
        
        Returns:
            List of OrderBlock objects
        """
        if len(df) < 10:
            return []
        
        order_blocks = []
        
        # Calculate ATR for strength measurement
        df = df.copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Analyze recent candles
        start_idx = max(0, len(df) - lookback)
        
        for i in range(start_idx + 3, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            prev3 = df.iloc[i-3]
            
            # Bullish OB: Last down candle before strong up move
            # Look for: down candle followed by strong bullish move
            if (prev['close'] < prev['open'] and  # Previous was bearish
                current['close'] > current['open'] and  # Current is bullish
                current['close'] > prev['high']):  # Breaks previous high
                
                # Check if move is strong enough
                move_size = current['close'] - prev['low']
                atr = current['atr']
                
                if pd.notna(atr) and atr > 0:
                    strength = min(move_size / (atr * 2), 1.0)
                    
                    if strength >= self.ob_strength_threshold:
                        # Volume confirmation (if available)
                        if 'volume' in df.columns and pd.notna(current['volume']):
                            avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
                            if current['volume'] > avg_vol * 1.2:
                                strength = min(strength * 1.2, 1.0)
                        
                        ob = OrderBlock(
                            type='bullish',
                            top=prev['high'],
                            bottom=prev['low'],
                            candle_index=i-1,
                            timestamp=str(prev.name) if hasattr(prev, 'name') else '',
                            strength=strength
                        )
                        order_blocks.append(ob)
            
            # Bearish OB: Last up candle before strong down move
            elif (prev['close'] > prev['open'] and  # Previous was bullish
                  current['close'] < current['open'] and  # Current is bearish
                  current['close'] < prev['low']):  # Breaks previous low
                
                move_size = prev['high'] - current['close']
                atr = current['atr']
                
                if pd.notna(atr) and atr > 0:
                    strength = min(move_size / (atr * 2), 1.0)
                    
                    if strength >= self.ob_strength_threshold:
                        if 'volume' in df.columns and pd.notna(current['volume']):
                            avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
                            if current['volume'] > avg_vol * 1.2:
                                strength = min(strength * 1.2, 1.0)
                        
                        ob = OrderBlock(
                            type='bearish',
                            top=prev['high'],
                            bottom=prev['low'],
                            candle_index=i-1,
                            timestamp=str(prev.name) if hasattr(prev, 'name') else '',
                            strength=strength
                        )
                        order_blocks.append(ob)
        
        # Check for mitigation (price revisiting OB)
        if order_blocks:
            current_price = df.iloc[-1]['close']
            
            for ob in order_blocks:
                ob_idx = ob.candle_index
                
                # Check if price has returned to OB zone
                for j in range(ob_idx + 1, len(df)):
                    candle = df.iloc[j]
                    
                    if ob.type == 'bullish':
                        # Mitigated if price goes back into OB zone
                        if candle['low'] <= ob.top:
                            ob.mitigated = True
                            ob.mitigation_index = j
                            break
                    else:  # bearish
                        if candle['high'] >= ob.bottom:
                            ob.mitigated = True
                            ob.mitigation_index = j
                            break
        
        return order_blocks
    
    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[FairValueGap]:
        """
        Detect fair value gaps (3-candle imbalances).
        
        FVG = gap between candle 1 high and candle 3 low (bullish)
              or candle 1 low and candle 3 high (bearish)
        
        Args:
            df: DataFrame with OHLC data
            lookback: How many recent candles to analyze
        
        Returns:
            List of FairValueGap objects
        """
        if len(df) < 10:
            return []
        
        fvgs = []
        
        # Calculate ATR for minimum size threshold
        df = df.copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        start_idx = max(0, len(df) - lookback)
        
        for i in range(start_idx + 2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            atr = candle3['atr']
            if pd.isna(atr) or atr == 0:
                continue
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if candle3['low'] > candle1['high']:
                gap_size = candle3['low'] - candle1['high']
                
                # Must be significant gap
                if gap_size >= atr * self.fvg_min_size_atr:
                    fvg = FairValueGap(
                        type='bullish',
                        top=candle3['low'],
                        bottom=candle1['high'],
                        start_index=i-2,
                        timestamp=str(candle1.name) if hasattr(candle1, 'name') else '',
                        size=gap_size
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            elif candle3['high'] < candle1['low']:
                gap_size = candle1['low'] - candle3['high']
                
                if gap_size >= atr * self.fvg_min_size_atr:
                    fvg = FairValueGap(
                        type='bearish',
                        top=candle1['low'],
                        bottom=candle3['high'],
                        start_index=i-2,
                        timestamp=str(candle1.name) if hasattr(candle1, 'name') else '',
                        size=gap_size
                    )
                    fvgs.append(fvg)
        
        # Check for mitigation (FVG filled)
        if fvgs:
            for fvg in fvgs:
                start_idx = fvg.start_index
                
                for j in range(start_idx + 3, len(df)):
                    candle = df.iloc[j]
                    
                    if fvg.type == 'bullish':
                        # Mitigated if price fills gap (goes back down into it)
                        if candle['low'] <= fvg.bottom:
                            fvg.mitigated = True
                            fvg.mitigation_index = j
                            break
                    else:  # bearish
                        # Mitigated if price fills gap (goes back up into it)
                        if candle['high'] >= fvg.top:
                            fvg.mitigated = True
                            fvg.mitigation_index = j
                            break
        
        return fvgs
    
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> List[StructurePoint]:
        """
        Detect swing highs and lows for structure analysis.
        
        Args:
            df: DataFrame with OHLC data
            lookback: How many candles to analyze
        
        Returns:
            List of StructurePoint objects
        """
        if len(df) < self.swing_lookback * 2:
            return []
        
        swing_points = []
        
        start_idx = max(0, len(df) - lookback)
        
        for i in range(start_idx + self.swing_lookback, len(df) - self.swing_lookback):
            candle = df.iloc[i]
            
            # Check for swing high
            is_swing_high = True
            for j in range(1, self.swing_lookback + 1):
                if df.iloc[i-j]['high'] >= candle['high'] or df.iloc[i+j]['high'] >= candle['high']:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_points.append(StructurePoint(
                    type='high',
                    price=candle['high'],
                    index=i,
                    timestamp=str(candle.name) if hasattr(candle, 'name') else ''
                ))
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, self.swing_lookback + 1):
                if df.iloc[i-j]['low'] <= candle['low'] or df.iloc[i+j]['low'] <= candle['low']:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_points.append(StructurePoint(
                    type='low',
                    price=candle['low'],
                    index=i,
                    timestamp=str(candle.name) if hasattr(candle, 'name') else ''
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def detect_structure_breaks(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint]
    ) -> Dict[str, List[StructurePoint]]:
        """
        Detect BOS (Break of Structure) and CHOC (Change of Character).
        
        BOS = Break in trend direction (continuation)
        CHOC = Break against trend direction (reversal signal)
        
        Args:
            df: DataFrame with OHLC data
            swing_points: List of swing highs/lows
        
        Returns:
            dict with 'bos' and 'choc' lists
        """
        if len(swing_points) < 3:
            return {'bos': [], 'choc': []}
        
        bos_points = []
        choc_points = []
        
        # Determine trend from swing points
        highs = [sp for sp in swing_points if sp.type == 'high']
        lows = [sp for sp in swing_points if sp.type == 'low']
        
        # Check each swing point for breaks
        for i in range(len(swing_points) - 1):
            current_sp = swing_points[i]
            
            # Look for breaks after this swing point
            for j in range(current_sp.index + self.structure_break_confirm, len(df)):
                candle = df.iloc[j]
                
                if current_sp.type == 'high' and not current_sp.broken:
                    # Check if high is broken
                    if candle['close'] > current_sp.price:
                        # Determine if BOS or CHOC based on trend
                        # If previous swing low is higher (uptrend), breaking high = BOS
                        # If previous swing low is lower (downtrend), breaking high = CHOC
                        
                        prev_lows = [sp for sp in lows if sp.index < current_sp.index]
                        if len(prev_lows) >= 2:
                            if prev_lows[-1].price > prev_lows[-2].price:
                                # Uptrend, breaking high = BOS
                                current_sp.broken = True
                                current_sp.break_index = j
                                current_sp.break_type = 'BOS'
                                bos_points.append(current_sp)
                            else:
                                # Downtrend, breaking high = CHOC
                                current_sp.broken = True
                                current_sp.break_index = j
                                current_sp.break_type = 'CHOC'
                                choc_points.append(current_sp)
                        break
                
                elif current_sp.type == 'low' and not current_sp.broken:
                    # Check if low is broken
                    if candle['close'] < current_sp.price:
                        prev_highs = [sp for sp in highs if sp.index < current_sp.index]
                        if len(prev_highs) >= 2:
                            if prev_highs[-1].price < prev_highs[-2].price:
                                # Downtrend, breaking low = BOS
                                current_sp.broken = True
                                current_sp.break_index = j
                                current_sp.break_type = 'BOS'
                                bos_points.append(current_sp)
                            else:
                                # Uptrend, breaking low = CHOC
                                current_sp.broken = True
                                current_sp.break_index = j
                                current_sp.break_type = 'CHOC'
                                choc_points.append(current_sp)
                        break
        
        return {'bos': bos_points, 'choc': choc_points}
    
    def get_unmitigated_zones(
        self,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        current_price: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get unmitigated support and resistance zones.
        
        Args:
            order_blocks: List of order blocks
            fvgs: List of fair value gaps
            current_price: Current market price
        
        Returns:
            dict with 'support' and 'resistance' zone lists
        """
        support_zones = []
        resistance_zones = []
        
        # Unmitigated bullish OBs = support
        for ob in order_blocks:
            if ob.type == 'bullish' and not ob.mitigated and ob.top < current_price:
                support_zones.append({
                    'type': 'order_block',
                    'top': ob.top,
                    'bottom': ob.bottom,
                    'strength': ob.strength,
                    'timestamp': ob.timestamp
                })
        
        # Unmitigated bearish OBs = resistance
        for ob in order_blocks:
            if ob.type == 'bearish' and not ob.mitigated and ob.bottom > current_price:
                resistance_zones.append({
                    'type': 'order_block',
                    'top': ob.top,
                    'bottom': ob.bottom,
                    'strength': ob.strength,
                    'timestamp': ob.timestamp
                })
        
        # Unmitigated bullish FVGs = support
        for fvg in fvgs:
            if fvg.type == 'bullish' and not fvg.mitigated and fvg.top < current_price:
                support_zones.append({
                    'type': 'fvg',
                    'top': fvg.top,
                    'bottom': fvg.bottom,
                    'strength': min(fvg.size / current_price * 100, 1.0),
                    'timestamp': fvg.timestamp
                })
        
        # Unmitigated bearish FVGs = resistance
        for fvg in fvgs:
            if fvg.type == 'bearish' and not fvg.mitigated and fvg.bottom > current_price:
                resistance_zones.append({
                    'type': 'fvg',
                    'top': fvg.top,
                    'bottom': fvg.bottom,
                    'strength': min(fvg.size / current_price * 100, 1.0),
                    'timestamp': fvg.timestamp
                })
        
        # Sort by proximity to current price
        support_zones.sort(key=lambda z: current_price - z['top'], reverse=False)
        resistance_zones.sort(key=lambda z: z['bottom'] - current_price, reverse=False)
        
        return {
            'support': support_zones,
            'resistance': resistance_zones
        }
    
    def analyze_full_smc(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run full smart money analysis.
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current price (uses last close if not provided)
        
        Returns:
            dict with all SMC analysis
        """
        if current_price is None:
            current_price = df.iloc[-1]['close']
        
        # Detect all components
        order_blocks = self.detect_order_blocks(df)
        fvgs = self.detect_fair_value_gaps(df)
        swing_points = self.detect_swing_points(df)
        structure_breaks = self.detect_structure_breaks(df, swing_points)
        zones = self.get_unmitigated_zones(order_blocks, fvgs, current_price)
        
        # Find nearest levels
        nearest_support = zones['support'][0] if zones['support'] else None
        nearest_resistance = zones['resistance'][0] if zones['resistance'] else None
        
        # Recent structure breaks
        recent_bos = [sp for sp in structure_breaks['bos'] if sp.break_index and sp.break_index >= len(df) - 20]
        recent_choc = [sp for sp in structure_breaks['choc'] if sp.break_index and sp.break_index >= len(df) - 20]
        
        return {
            'current_price': current_price,
            'order_blocks': {
                'total': len(order_blocks),
                'unmitigated': len([ob for ob in order_blocks if not ob.mitigated]),
                'bullish': [ob for ob in order_blocks if ob.type == 'bullish'],
                'bearish': [ob for ob in order_blocks if ob.type == 'bearish']
            },
            'fair_value_gaps': {
                'total': len(fvgs),
                'unmitigated': len([fvg for fvg in fvgs if not fvg.mitigated]),
                'bullish': [fvg for fvg in fvgs if fvg.type == 'bullish'],
                'bearish': [fvg for fvg in fvgs if fvg.type == 'bearish']
            },
            'structure': {
                'swing_points': len(swing_points),
                'bos_count': len(structure_breaks['bos']),
                'choc_count': len(structure_breaks['choc']),
                'recent_bos': recent_bos,
                'recent_choc': recent_choc
            },
            'zones': zones,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'bias': self._determine_bias(recent_bos, recent_choc, zones, current_price)
        }
    
    def _determine_bias(
        self,
        recent_bos: List[StructurePoint],
        recent_choc: List[StructurePoint],
        zones: Dict[str, List[Dict[str, Any]]],
        current_price: float
    ) -> str:
        """Determine market bias from structure"""
        if recent_choc:
            last_choc = recent_choc[-1]
            if last_choc.type == 'high':
                return 'bearish'  # Broke high in downtrend = bearish CHOC
            else:
                return 'bullish'  # Broke low in uptrend = bullish CHOC
        
        if recent_bos:
            last_bos = recent_bos[-1]
            if last_bos.type == 'high':
                return 'bullish'  # Broke high in uptrend = bullish BOS
            else:
                return 'bearish'  # Broke low in downtrend = bearish BOS
        
        # Fallback to zone analysis
        support_count = len([z for z in zones['support'] if z['top'] < current_price])
        resistance_count = len([z for z in zones['resistance'] if z['bottom'] > current_price])
        
        if support_count > resistance_count:
            return 'bullish'
        elif resistance_count > support_count:
            return 'bearish'
        
        return 'neutral'
    
    def format_smc_report(self, analysis: Dict[str, Any]) -> str:
        """Format SMC analysis as readable report"""
        lines = []
        lines.append("="*70)
        lines.append("SMART MONEY CONCEPTS ANALYSIS")
        lines.append("="*70)
        
        lines.append(f"\nCurrent Price: ${analysis['current_price']:.2f}")
        lines.append(f"Market Bias: {analysis['bias'].upper()}")
        
        # Order Blocks
        lines.append(f"\n{'─'*70}")
        lines.append("ORDER BLOCKS:")
        lines.append(f"  Total: {analysis['order_blocks']['total']} | Unmitigated: {analysis['order_blocks']['unmitigated']}")
        
        if analysis['nearest_support'] and analysis['nearest_support']['type'] == 'order_block':
            s = analysis['nearest_support']
            lines.append(f"\n  Nearest Support OB: ${s['bottom']:.2f} - ${s['top']:.2f}")
            lines.append(f"    Strength: {s['strength']:.0%} | Distance: {((analysis['current_price'] - s['top']) / analysis['current_price'] * 100):.2f}%")
        
        if analysis['nearest_resistance'] and analysis['nearest_resistance']['type'] == 'order_block':
            r = analysis['nearest_resistance']
            lines.append(f"\n  Nearest Resistance OB: ${r['bottom']:.2f} - ${r['top']:.2f}")
            lines.append(f"    Strength: {r['strength']:.0%} | Distance: {((r['bottom'] - analysis['current_price']) / analysis['current_price'] * 100):.2f}%")
        
        # Fair Value Gaps
        lines.append(f"\n{'─'*70}")
        lines.append("FAIR VALUE GAPS:")
        lines.append(f"  Total: {analysis['fair_value_gaps']['total']} | Unmitigated: {analysis['fair_value_gaps']['unmitigated']}")
        
        # Structure
        lines.append(f"\n{'─'*70}")
        lines.append("MARKET STRUCTURE:")
        lines.append(f"  BOS (Break of Structure): {analysis['structure']['bos_count']}")
        lines.append(f"  CHOC (Change of Character): {analysis['structure']['choc_count']}")
        
        if analysis['structure']['recent_bos']:
            lines.append(f"\n  Recent BOS: {len(analysis['structure']['recent_bos'])} in last 20 candles")
        
        if analysis['structure']['recent_choc']:
            lines.append(f"  Recent CHOC: {len(analysis['structure']['recent_choc'])} in last 20 candles ⚠️")
        
        # Key Zones
        lines.append(f"\n{'─'*70}")
        lines.append("KEY SUPPORT/RESISTANCE ZONES:")
        
        lines.append(f"\n  Support Zones: {len(analysis['zones']['support'])}")
        for i, zone in enumerate(analysis['zones']['support'][:3], 1):
            dist = ((analysis['current_price'] - zone['top']) / analysis['current_price'] * 100)
            lines.append(f"    {i}. ${zone['bottom']:.2f}-${zone['top']:.2f} ({zone['type']}) | -{dist:.2f}%")
        
        lines.append(f"\n  Resistance Zones: {len(analysis['zones']['resistance'])}")
        for i, zone in enumerate(analysis['zones']['resistance'][:3], 1):
            dist = ((zone['bottom'] - analysis['current_price']) / analysis['current_price'] * 100)
            lines.append(f"    {i}. ${zone['bottom']:.2f}-${zone['top']:.2f} ({zone['type']}) | +{dist:.2f}%")
        
        lines.append(f"\n{'='*70}")
        
        return '\n'.join(lines)
