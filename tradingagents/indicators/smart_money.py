"""
Smart Money Concepts (SMC) Indicators

Implements institutional trading concepts:
- Order Blocks (OB): Last up/down candle before strong move
- Change of Character (CHOC): Momentum shift, breaks previous high/low
- Break of Structure (BOS): Continuation, breaks in trend direction
- Fair Value Gaps (FVG): 3-candle imbalance zones
- Support/Resistance with mitigation tracking
- Equal Highs/Lows (EQH/EQL): Liquidity targets
- Breaker Blocks: Failed OBs that flip polarity
- Premium/Discount Zones: Position relative to range
- OTE (Optimal Trade Entry): Fibonacci-based entry zones
- Session Analysis: Kill zone identification
- Multi-Timeframe Alignment: Confluence across timeframes

Used for better TP/SL placement aligned with institutional levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum


class TradingSession(Enum):
    """Trading session identifiers"""
    ASIAN = "asian"
    LONDON_OPEN = "london_open"
    LONDON = "london"
    NY_OPEN = "ny_open"
    NY = "ny"
    OFF_SESSION = "off_session"


class SMCConstants:
    """Constants for SMC analysis - eliminates magic numbers"""
    # ATR settings
    ATR_PERIOD = 14

    # Order Block detection
    OB_VOLUME_MULTIPLIER = 1.2
    OB_MOVE_ATR_DIVISOR = 2.0
    OB_OVERLAP_THRESHOLD = 0.3  # 30% overlap = duplicate

    # Structural OB detection
    STRUCTURAL_OB_MIN_IMPULSE_ATR = 2.0
    STRUCTURAL_OB_CONSOLIDATION_CANDLES = 5
    STRUCTURAL_OB_STRENGTH_ATR_DIVISOR = 3.0

    # FVG detection
    FVG_MIN_SIZE_ATR_DEFAULT = 0.3

    # Liquidity zone detection
    LIQUIDITY_TEST_TOLERANCE_PCT = 0.002  # 0.2%
    LIQUIDITY_SWEEP_AGE_CUTOFF = 50
    LIQUIDITY_MOVE_WEIGHT = 40
    LIQUIDITY_TEST_WEIGHT = 30
    LIQUIDITY_RECENCY_WEIGHT = 30
    LIQUIDITY_MOVE_ATR_DIVISOR = 3.0

    # Equal levels detection
    EQUAL_LEVEL_TOLERANCE_ATR = 0.1
    EQUAL_LEVEL_MIN_TOUCHES = 2

    # OTE (Optimal Trade Entry) Fibonacci levels
    OTE_FIB_START = 0.62
    OTE_FIB_END = 0.79

    # Premium/Discount
    PREMIUM_DISCOUNT_LOOKBACK = 50

    # Confluence scoring weights
    CONFLUENCE_OB_WEIGHT = 30
    CONFLUENCE_FVG_WEIGHT = 20
    CONFLUENCE_LIQUIDITY_WEIGHT = 25
    CONFLUENCE_PREMIUM_DISCOUNT_WEIGHT = 15
    CONFLUENCE_OTE_WEIGHT = 20

    # Session times (UTC)
    ASIAN_START = time(0, 0)
    ASIAN_END = time(6, 0)
    LONDON_OPEN_START = time(7, 0)
    LONDON_OPEN_END = time(10, 0)
    LONDON_END = time(12, 0)
    NY_OPEN_START = time(12, 0)
    NY_OPEN_END = time(15, 0)
    NY_END = time(21, 0)

    # Regime adjustment factors
    # In trending markets: directional zones get boost, counter-trend zones get penalty
    REGIME_TRENDING_ALIGNED_BOOST = 1.2  # 20% boost for trend-aligned zones
    REGIME_TRENDING_COUNTER_PENALTY = 0.7  # 30% penalty for counter-trend zones
    REGIME_RANGING_FVG_BOOST = 1.15  # FVGs more important in ranging markets
    REGIME_RANGING_EQL_BOOST = 1.2  # Equal levels more important in ranging markets

    # Time decay for zone strength (candles since formation)
    TIME_DECAY_FRESH_THRESHOLD = 20  # < 20 candles = fresh (100%)
    TIME_DECAY_AGING_THRESHOLD = 50  # 20-50 candles = aging (75%)
    TIME_DECAY_OLD_THRESHOLD = 100  # 50-100 candles = old (50%)
    TIME_DECAY_AGING_FACTOR = 0.75
    TIME_DECAY_OLD_FACTOR = 0.50
    TIME_DECAY_VERY_OLD_FACTOR = 0.25  # > 100 candles

    # Confluence regime weight
    CONFLUENCE_REGIME_ALIGNMENT_WEIGHT = 15


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
    detection_method: str = "candle"  # "candle" or "structural"
    zone_candles: int = 1  # Number of candles in the zone
    invalidated: bool = False  # Price closed beyond zone (stronger than mitigation)
    invalidation_index: Optional[int] = None
    session: Optional[str] = None  # Trading session when formed

    @property
    def midpoint(self) -> float:
        """Get 50% level of OB (optimal entry)"""
        return (self.top + self.bottom) / 2

    @property
    def invalidation_price(self) -> float:
        """Get price that invalidates this OB"""
        if self.type == 'bullish':
            return self.bottom  # Below OB invalidates bullish
        return self.top  # Above OB invalidates bearish


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
    fill_percentage: float = 0.0  # How much of gap has been filled (0-100)
    session: Optional[str] = None  # Trading session when formed

    @property
    def midpoint(self) -> float:
        """Get 50% level of FVG"""
        return (self.top + self.bottom) / 2

    @property
    def remaining_size(self) -> float:
        """Get unfilled portion of FVG"""
        return self.size * (1 - self.fill_percentage / 100)


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
    session: Optional[str] = None  # Trading session when formed


@dataclass
class LiquidityZone:
    """A liquidity zone (cluster of stop losses)"""
    type: str  # "buy-side" (above highs) or "sell-side" (below lows)
    price: float  # The swing high/low price where stops cluster
    strength: float  # 0-100, based on significance of the swing
    touched: bool = False  # Whether price has swept this liquidity
    sweep_index: Optional[int] = None
    swing_index: int = 0
    timestamp: str = ""
    session: Optional[str] = None


@dataclass
class EqualLevel:
    """Equal highs or lows - major liquidity targets"""
    type: str  # "equal_highs" or "equal_lows"
    price: float  # The price level where multiple swings meet
    touches: int  # How many times price touched this level
    indices: List[int] = field(default_factory=list)  # Candle indices of touches
    timestamps: List[str] = field(default_factory=list)
    swept: bool = False  # Whether liquidity has been taken
    sweep_index: Optional[int] = None

    @property
    def liquidity_side(self) -> str:
        """Where stops cluster relative to this level"""
        if self.type == "equal_highs":
            return "above"  # Buy stops above equal highs
        return "below"  # Sell stops below equal lows


@dataclass
class BreakerBlock:
    """A failed order block that flipped polarity"""
    original_type: str  # Was bullish/bearish
    current_type: str  # Now opposite (bearish/bullish)
    top: float
    bottom: float
    original_index: int  # When OB was formed
    break_index: int  # When it failed/flipped
    timestamp: str
    strength: float  # Inherited from original OB, adjusted
    mitigated: bool = False
    mitigation_index: Optional[int] = None

    @property
    def midpoint(self) -> float:
        """Get 50% level of breaker"""
        return (self.top + self.bottom) / 2


@dataclass
class OTEZone:
    """Optimal Trade Entry zone based on Fibonacci retracement"""
    direction: str  # "bullish" or "bearish"
    top: float
    bottom: float
    swing_high: float
    swing_low: float
    fib_start: float  # e.g., 0.62
    fib_end: float  # e.g., 0.79

    @property
    def midpoint(self) -> float:
        """Get middle of OTE zone"""
        return (self.top + self.bottom) / 2


@dataclass
class PremiumDiscountZone:
    """Premium/Discount zone analysis"""
    range_high: float
    range_low: float
    equilibrium: float  # 50% level
    current_price: float
    position_pct: float  # 0-100, where in range
    zone: str  # "premium", "discount", or "equilibrium"

    @property
    def is_premium(self) -> bool:
        return self.zone == "premium"

    @property
    def is_discount(self) -> bool:
        return self.zone == "discount"


@dataclass
class ConfluenceScore:
    """Confluence analysis at a price level"""
    price: float
    total_score: int  # 0-100
    factors: List[str] = field(default_factory=list)
    bullish_factors: int = 0
    bearish_factors: int = 0

    @property
    def bias(self) -> str:
        if self.bullish_factors > self.bearish_factors:
            return "bullish"
        elif self.bearish_factors > self.bullish_factors:
            return "bearish"
        return "neutral"


class SmartMoneyAnalyzer:
    """
    Analyzes price action for smart money concepts.

    Features:
    - Order block detection (last opposing candle before strong move)
    - CHOC detection (momentum shift, counter-trend break)
    - BOS detection (trend continuation break)
    - FVG detection (3-candle imbalance)
    - Equal Highs/Lows detection (liquidity targets)
    - Breaker block detection (failed OBs)
    - Premium/Discount zone analysis
    - OTE (Optimal Trade Entry) calculation
    - Session-based analysis
    - Confluence scoring
    - Multi-timeframe alignment
    - Unmitigated zone tracking
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        ob_strength_threshold: float = 0.5,
        fvg_min_size_atr: float = SMCConstants.FVG_MIN_SIZE_ATR_DEFAULT,
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

        # ATR cache to avoid recalculation
        self._atr_cache: Dict[int, pd.Series] = {}
        self._cache_df_hash: Optional[int] = None

    def _get_df_hash(self, df: pd.DataFrame) -> int:
        """Get a hash to identify if DataFrame changed"""
        return hash((len(df), df.iloc[-1]['close'] if len(df) > 0 else 0))

    def _get_atr(self, df: pd.DataFrame, period: int = SMCConstants.ATR_PERIOD) -> pd.Series:
        """
        Calculate ATR with caching to avoid repeated computation.

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            Series with ATR values
        """
        df_hash = self._get_df_hash(df)

        # Clear cache if DataFrame changed
        if df_hash != self._cache_df_hash:
            self._atr_cache = {}
            self._cache_df_hash = df_hash

        # Return cached if available
        if period in self._atr_cache:
            return self._atr_cache[period]

        # Calculate ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = tr.rolling(period).mean()

        # Cache and return
        self._atr_cache[period] = atr
        return atr

    def _identify_session(self, timestamp: Union[str, datetime, pd.Timestamp]) -> str:
        """
        Identify trading session from timestamp.

        Args:
            timestamp: The timestamp to check

        Returns:
            Session name string
        """
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp)
            except (ValueError, TypeError):
                return TradingSession.OFF_SESSION.value

        if not isinstance(timestamp, (datetime, pd.Timestamp)):
            return TradingSession.OFF_SESSION.value

        t = timestamp.time() if hasattr(timestamp, 'time') else time(0, 0)

        if SMCConstants.ASIAN_START <= t < SMCConstants.ASIAN_END:
            return TradingSession.ASIAN.value
        elif SMCConstants.LONDON_OPEN_START <= t < SMCConstants.LONDON_OPEN_END:
            return TradingSession.LONDON_OPEN.value
        elif SMCConstants.LONDON_OPEN_END <= t < SMCConstants.LONDON_END:
            return TradingSession.LONDON.value
        elif SMCConstants.NY_OPEN_START <= t < SMCConstants.NY_OPEN_END:
            return TradingSession.NY_OPEN.value
        elif SMCConstants.NY_OPEN_END <= t < SMCConstants.NY_END:
            return TradingSession.NY.value
        else:
            return TradingSession.OFF_SESSION.value

    def _get_session_multiplier(self, session: str) -> float:
        """
        Get strength multiplier based on session.
        High-volume sessions get higher multiplier.
        """
        multipliers = {
            TradingSession.LONDON_OPEN.value: 1.3,
            TradingSession.NY_OPEN.value: 1.3,
            TradingSession.LONDON.value: 1.1,
            TradingSession.NY.value: 1.1,
            TradingSession.ASIAN.value: 0.9,
            TradingSession.OFF_SESSION.value: 0.8,
        }
        return multipliers.get(session, 1.0)

    def get_regime_adjusted_strength(
        self,
        base_strength: float,
        zone_type: str,
        market_regime: Optional[str] = None,
        zone_category: str = "ob"
    ) -> float:
        """
        Adjust zone strength based on market regime.

        In trending markets:
        - Directional zones aligned with trend get boosted
        - Counter-trend zones get penalized

        In ranging markets:
        - FVGs and Equal Highs/Lows get boosted (mean reversion)

        Args:
            base_strength: Original strength (0-1)
            zone_type: "bullish" or "bearish"
            market_regime: "trending-up", "trending-down", "ranging", or None
            zone_category: "ob", "fvg", "liquidity", "eql" (equal level)

        Returns:
            Adjusted strength (clamped 0-1)
        """
        if not market_regime or base_strength <= 0:
            return base_strength

        adjusted = base_strength

        if market_regime == "trending-up":
            if zone_type == "bullish":
                # Bullish zones align with uptrend
                adjusted *= SMCConstants.REGIME_TRENDING_ALIGNED_BOOST
            else:
                # Bearish zones are counter-trend
                adjusted *= SMCConstants.REGIME_TRENDING_COUNTER_PENALTY

        elif market_regime == "trending-down":
            if zone_type == "bearish":
                # Bearish zones align with downtrend
                adjusted *= SMCConstants.REGIME_TRENDING_ALIGNED_BOOST
            else:
                # Bullish zones are counter-trend
                adjusted *= SMCConstants.REGIME_TRENDING_COUNTER_PENALTY

        elif market_regime == "ranging":
            # In ranging markets, FVGs and equal levels are more important
            if zone_category == "fvg":
                adjusted *= SMCConstants.REGIME_RANGING_FVG_BOOST
            elif zone_category == "eql":
                adjusted *= SMCConstants.REGIME_RANGING_EQL_BOOST

        return min(max(adjusted, 0.0), 1.0)

    def get_time_decay_factor(
        self,
        candles_since_formation: int
    ) -> float:
        """
        Calculate time decay factor for zone strength.

        Older zones are less reliable and should have reduced strength.
        Fresh zones are more likely to produce reactions.

        Args:
            candles_since_formation: How many candles since zone formed

        Returns:
            Decay factor (0.25 to 1.0)
        """
        if candles_since_formation < SMCConstants.TIME_DECAY_FRESH_THRESHOLD:
            return 1.0  # Fresh zone - full strength
        elif candles_since_formation < SMCConstants.TIME_DECAY_AGING_THRESHOLD:
            return SMCConstants.TIME_DECAY_AGING_FACTOR  # Aging - 75%
        elif candles_since_formation < SMCConstants.TIME_DECAY_OLD_THRESHOLD:
            return SMCConstants.TIME_DECAY_OLD_FACTOR  # Old - 50%
        else:
            return SMCConstants.TIME_DECAY_VERY_OLD_FACTOR  # Very old - 25%

    def apply_zone_adjustments(
        self,
        zones: List[Any],
        current_candle_index: int,
        market_regime: Optional[str] = None,
        zone_category: str = "ob"
    ) -> List[Any]:
        """
        Apply regime and time adjustments to a list of zones.

        Args:
            zones: List of OrderBlock, FairValueGap, or similar zone objects
            current_candle_index: Current candle index for time decay calculation
            market_regime: Current market regime
            zone_category: Type of zone ("ob", "fvg", "liquidity", "eql")

        Returns:
            Same zones with adjusted strength values
        """
        for zone in zones:
            if not hasattr(zone, 'strength') or not hasattr(zone, 'type'):
                continue

            base_strength = zone.strength

            # Apply time decay
            if hasattr(zone, 'candle_index'):
                candles_since = current_candle_index - zone.candle_index
                time_factor = self.get_time_decay_factor(candles_since)
                base_strength *= time_factor
            elif hasattr(zone, 'start_index'):
                candles_since = current_candle_index - zone.start_index
                time_factor = self.get_time_decay_factor(candles_since)
                base_strength *= time_factor

            # Apply regime adjustment
            zone_type = zone.type
            adjusted = self.get_regime_adjusted_strength(
                base_strength, zone_type, market_regime, zone_category
            )

            zone.strength = adjusted

        return zones

    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        track_invalidation: bool = True
    ) -> List[OrderBlock]:
        """
        Detect order blocks (last opposing candle before strong move).

        An order block is the last up/down candle before a strong opposite move.
        These represent institutional entry zones.

        Args:
            df: DataFrame with OHLCV data
            lookback: How many recent candles to analyze
            track_invalidation: Whether to track zone invalidation

        Returns:
            List of OrderBlock objects
        """
        if len(df) < 10:
            return []

        order_blocks = []

        # Use cached ATR
        atr_series = self._get_atr(df)

        # Analyze recent candles
        start_idx = max(0, len(df) - lookback)

        for i in range(start_idx + 3, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            atr = atr_series.iloc[i] if i < len(atr_series) else None
            if pd.isna(atr) or atr <= 0:
                continue

            # Get session for this candle
            timestamp_str = str(prev.name) if hasattr(prev, 'name') else ''
            session = self._identify_session(prev.name) if hasattr(prev, 'name') else None

            # Bullish OB: Last down candle before strong up move
            if (prev['close'] < prev['open'] and  # Previous was bearish
                current['close'] > current['open'] and  # Current is bullish
                current['close'] > prev['high']):  # Breaks previous high

                move_size = current['close'] - prev['low']
                strength = min(move_size / (atr * SMCConstants.OB_MOVE_ATR_DIVISOR), 1.0)

                if strength >= self.ob_strength_threshold:
                    # Volume confirmation (if available)
                    if 'volume' in df.columns and pd.notna(current['volume']):
                        avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
                        if avg_vol > 0 and current['volume'] > avg_vol * SMCConstants.OB_VOLUME_MULTIPLIER:
                            strength = min(strength * SMCConstants.OB_VOLUME_MULTIPLIER, 1.0)

                    # Session multiplier
                    if session:
                        strength = min(strength * self._get_session_multiplier(session), 1.0)

                    ob = OrderBlock(
                        type='bullish',
                        top=prev['high'],
                        bottom=prev['low'],
                        candle_index=i-1,
                        timestamp=timestamp_str,
                        strength=strength,
                        session=session
                    )
                    order_blocks.append(ob)

            # Bearish OB: Last up candle before strong down move
            elif (prev['close'] > prev['open'] and  # Previous was bullish
                  current['close'] < current['open'] and  # Current is bearish
                  current['close'] < prev['low']):  # Breaks previous low

                move_size = prev['high'] - current['close']
                strength = min(move_size / (atr * SMCConstants.OB_MOVE_ATR_DIVISOR), 1.0)

                if strength >= self.ob_strength_threshold:
                    if 'volume' in df.columns and pd.notna(current['volume']):
                        avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
                        if avg_vol > 0 and current['volume'] > avg_vol * SMCConstants.OB_VOLUME_MULTIPLIER:
                            strength = min(strength * SMCConstants.OB_VOLUME_MULTIPLIER, 1.0)

                    if session:
                        strength = min(strength * self._get_session_multiplier(session), 1.0)

                    ob = OrderBlock(
                        type='bearish',
                        top=prev['high'],
                        bottom=prev['low'],
                        candle_index=i-1,
                        timestamp=timestamp_str,
                        strength=strength,
                        session=session
                    )
                    order_blocks.append(ob)

        # Check for mitigation and invalidation
        if order_blocks:
            for ob in order_blocks:
                ob_idx = ob.candle_index

                for j in range(ob_idx + 1, len(df)):
                    candle = df.iloc[j]

                    if ob.type == 'bullish':
                        # Mitigated if price goes back into OB zone
                        if not ob.mitigated and candle['low'] <= ob.top:
                            ob.mitigated = True
                            ob.mitigation_index = j

                        # Invalidated if price closes below OB (stronger signal)
                        if track_invalidation and not ob.invalidated and candle['close'] < ob.bottom:
                            ob.invalidated = True
                            ob.invalidation_index = j
                            break
                    else:  # bearish
                        if not ob.mitigated and candle['high'] >= ob.bottom:
                            ob.mitigated = True
                            ob.mitigation_index = j

                        if track_invalidation and not ob.invalidated and candle['close'] > ob.top:
                            ob.invalidated = True
                            ob.invalidation_index = j
                            break

        return order_blocks

    def detect_structural_order_blocks(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        lookback: int = 100,
        min_impulse_atr: float = SMCConstants.STRUCTURAL_OB_MIN_IMPULSE_ATR,
        consolidation_candles: int = SMCConstants.STRUCTURAL_OB_CONSOLIDATION_CANDLES
    ) -> List[OrderBlock]:
        """
        Detect structural order blocks based on swing points and impulsive moves.

        This method identifies significant supply/demand zones where institutions
        accumulated positions before major moves. It looks for:
        1. Significant swing highs/lows
        2. Consolidation/base before the impulsive move from that swing
        3. The zone from which price launched

        Args:
            df: DataFrame with OHLCV data
            swing_points: Previously detected swing points
            lookback: How many candles to analyze
            min_impulse_atr: Minimum move size in ATR multiples to qualify as impulsive
            consolidation_candles: Max candles to look back for consolidation zone

        Returns:
            List of structural OrderBlock objects
        """
        if len(df) < 20 or not swing_points:
            return []

        order_blocks = []

        # Use cached ATR
        atr_series = self._get_atr(df)
        current_price = df.iloc[-1]['close']

        # Process each significant swing point
        for sp in swing_points:
            if sp.index < consolidation_candles or sp.index >= len(df) - 3:
                continue

            atr_at_swing = atr_series.iloc[sp.index] if sp.index < len(atr_series) else None
            if pd.isna(atr_at_swing) or atr_at_swing == 0:
                continue

            # Get session info
            timestamp_str = str(df.iloc[sp.index].name) if hasattr(df.iloc[sp.index], 'name') else ''
            session = self._identify_session(df.iloc[sp.index].name) if hasattr(df.iloc[sp.index], 'name') else None

            if sp.type == 'low':
                # Bullish OB: Find consolidation zone before swing low
                future_candles = min(20, len(df) - sp.index - 1)
                if future_candles < 3:
                    continue

                max_high_after = df.iloc[sp.index + 1:sp.index + future_candles + 1]['high'].max()
                move_size = max_high_after - sp.price

                if move_size < atr_at_swing * min_impulse_atr:
                    continue

                zone_start_idx = max(0, sp.index - consolidation_candles)
                zone_candles = df.iloc[zone_start_idx:sp.index + 1]

                zone_low = zone_candles['low'].min()
                zone_high = zone_candles['high'].max()

                ob_bottom = zone_low
                ob_top = min(zone_high, sp.price + (zone_high - zone_low) * 0.5)

                strength = min(move_size / (atr_at_swing * SMCConstants.STRUCTURAL_OB_STRENGTH_ATR_DIVISOR), 1.0)

                # Apply session multiplier
                if session:
                    strength = min(strength * self._get_session_multiplier(session), 1.0)

                # Check mitigation and invalidation
                mitigated = False
                mitigation_idx = None
                invalidated = False
                invalidation_idx = None

                for j in range(sp.index + 5, len(df)):
                    candle = df.iloc[j]
                    if not mitigated and candle['low'] <= ob_top:
                        mitigated = True
                        mitigation_idx = j
                    if not invalidated and candle['close'] < ob_bottom:
                        invalidated = True
                        invalidation_idx = j
                        break

                if ob_top < current_price or not mitigated:
                    ob = OrderBlock(
                        type='bullish',
                        top=ob_top,
                        bottom=ob_bottom,
                        candle_index=sp.index,
                        timestamp=timestamp_str,
                        strength=strength,
                        mitigated=mitigated,
                        mitigation_index=mitigation_idx,
                        detection_method='structural',
                        zone_candles=sp.index - zone_start_idx + 1,
                        invalidated=invalidated,
                        invalidation_index=invalidation_idx,
                        session=session
                    )
                    order_blocks.append(ob)

            elif sp.type == 'high':
                future_candles = min(20, len(df) - sp.index - 1)
                if future_candles < 3:
                    continue

                min_low_after = df.iloc[sp.index + 1:sp.index + future_candles + 1]['low'].min()
                move_size = sp.price - min_low_after

                if move_size < atr_at_swing * min_impulse_atr:
                    continue

                zone_start_idx = max(0, sp.index - consolidation_candles)
                zone_candles = df.iloc[zone_start_idx:sp.index + 1]

                zone_low = zone_candles['low'].min()
                zone_high = zone_candles['high'].max()

                ob_top = zone_high
                ob_bottom = max(zone_low, sp.price - (zone_high - zone_low) * 0.5)

                strength = min(move_size / (atr_at_swing * SMCConstants.STRUCTURAL_OB_STRENGTH_ATR_DIVISOR), 1.0)

                if session:
                    strength = min(strength * self._get_session_multiplier(session), 1.0)

                mitigated = False
                mitigation_idx = None
                invalidated = False
                invalidation_idx = None

                for j in range(sp.index + 5, len(df)):
                    candle = df.iloc[j]
                    if not mitigated and candle['high'] >= ob_bottom:
                        mitigated = True
                        mitigation_idx = j
                    if not invalidated and candle['close'] > ob_top:
                        invalidated = True
                        invalidation_idx = j
                        break

                if ob_bottom > current_price or not mitigated:
                    ob = OrderBlock(
                        type='bearish',
                        top=ob_top,
                        bottom=ob_bottom,
                        candle_index=sp.index,
                        timestamp=timestamp_str,
                        strength=strength,
                        mitigated=mitigated,
                        mitigation_index=mitigation_idx,
                        detection_method='structural',
                        zone_candles=sp.index - zone_start_idx + 1,
                        invalidated=invalidated,
                        invalidation_index=invalidation_idx,
                        session=session
                    )
                    order_blocks.append(ob)

        # Remove duplicates (zones that overlap significantly)
        filtered_obs = []
        for ob in order_blocks:
            is_duplicate = False
            for existing in filtered_obs:
                if existing.type != ob.type:
                    continue
                overlap_top = min(ob.top, existing.top)
                overlap_bottom = max(ob.bottom, existing.bottom)
                if overlap_top > overlap_bottom:
                    overlap_pct = (overlap_top - overlap_bottom) / (ob.top - ob.bottom)
                    if overlap_pct > 0.5:
                        if ob.strength > existing.strength:
                            filtered_obs.remove(existing)
                        else:
                            is_duplicate = True
                        break
            if not is_duplicate:
                filtered_obs.append(ob)

        return filtered_obs

    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        track_partial_fill: bool = True
    ) -> List[FairValueGap]:
        """
        Detect fair value gaps (3-candle imbalances).

        FVG = gap between candle 1 high and candle 3 low (bullish)
              or candle 1 low and candle 3 high (bearish)

        Args:
            df: DataFrame with OHLC data
            lookback: How many recent candles to analyze
            track_partial_fill: Whether to track partial FVG fills

        Returns:
            List of FairValueGap objects
        """
        if len(df) < 10:
            return []

        fvgs = []

        # Use cached ATR
        atr_series = self._get_atr(df)

        start_idx = max(0, len(df) - lookback)

        for i in range(start_idx + 2, len(df)):
            candle1 = df.iloc[i-2]
            candle3 = df.iloc[i]

            atr = atr_series.iloc[i] if i < len(atr_series) else None
            if pd.isna(atr) or atr == 0:
                continue

            # Get session info
            timestamp_str = str(candle1.name) if hasattr(candle1, 'name') else ''
            session = self._identify_session(candle1.name) if hasattr(candle1, 'name') else None

            # Bullish FVG: Gap between candle1 high and candle3 low
            if candle3['low'] > candle1['high']:
                gap_size = candle3['low'] - candle1['high']

                if gap_size >= atr * self.fvg_min_size_atr:
                    fvg = FairValueGap(
                        type='bullish',
                        top=candle3['low'],
                        bottom=candle1['high'],
                        start_index=i-2,
                        timestamp=timestamp_str,
                        size=gap_size,
                        session=session
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
                        timestamp=timestamp_str,
                        size=gap_size,
                        session=session
                    )
                    fvgs.append(fvg)

        # Check for mitigation and partial fill
        if fvgs:
            for fvg in fvgs:
                fvg_start_idx = fvg.start_index
                max_fill_depth = 0.0

                for j in range(fvg_start_idx + 3, len(df)):
                    candle = df.iloc[j]

                    if fvg.type == 'bullish':
                        # Track how deep price went into the FVG
                        if candle['low'] < fvg.top:
                            fill_depth = fvg.top - max(candle['low'], fvg.bottom)
                            max_fill_depth = max(max_fill_depth, fill_depth)

                        # Fully mitigated if price fills entire gap
                        if candle['low'] <= fvg.bottom:
                            fvg.mitigated = True
                            fvg.mitigation_index = j
                            fvg.fill_percentage = 100.0
                            break
                    else:  # bearish
                        if candle['high'] > fvg.bottom:
                            fill_depth = min(candle['high'], fvg.top) - fvg.bottom
                            max_fill_depth = max(max_fill_depth, fill_depth)

                        if candle['high'] >= fvg.top:
                            fvg.mitigated = True
                            fvg.mitigation_index = j
                            fvg.fill_percentage = 100.0
                            break

                # Calculate partial fill percentage if not fully mitigated
                if track_partial_fill and not fvg.mitigated and fvg.size > 0:
                    fvg.fill_percentage = min((max_fill_depth / fvg.size) * 100, 100.0)

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
            timestamp_str = str(candle.name) if hasattr(candle, 'name') else ''
            session = self._identify_session(candle.name) if hasattr(candle, 'name') else None

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
                    timestamp=timestamp_str,
                    session=session
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
                    timestamp=timestamp_str,
                    session=session
                ))

        return sorted(swing_points, key=lambda x: x.index)
    
    def detect_liquidity_zones(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        current_price: float,
        max_zones: int = 6
    ) -> List[LiquidityZone]:
        """
        Detect liquidity zones based on swing highs and lows.

        In SMC, liquidity zones are where stop losses cluster:
        - Buy-side liquidity: Above swing highs (shorts' stops)
        - Sell-side liquidity: Below swing lows (longs' stops)

        The strength is based on:
        1. How prominent the swing is (larger move = more stops)
        2. How many times the level has been tested
        3. Age (more recent = more relevant)

        Args:
            df: DataFrame with OHLC data
            swing_points: List of swing highs/lows
            current_price: Current market price
            max_zones: Maximum number of zones to return

        Returns:
            List of LiquidityZone objects
        """
        if len(swing_points) < 2 or len(df) < 20:
            return []

        liquidity_zones = []

        # Use cached ATR
        atr_series = self._get_atr(df)
        recent_atr = atr_series.iloc[-1] if pd.notna(atr_series.iloc[-1]) else 1.0

        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']

        # Process swing highs -> Buy-side liquidity (BSL)
        for sh in swing_highs:
            touched = False
            sweep_idx = None
            for j in range(sh.index + 1, len(df)):
                if df.iloc[j]['high'] > sh.price:
                    touched = True
                    sweep_idx = j
                    break

            if touched and sweep_idx and (len(df) - sweep_idx) > SMCConstants.LIQUIDITY_SWEEP_AGE_CUTOFF:
                continue

            # Calculate strength using constants
            lookback_range = min(10, sh.index)
            if lookback_range > 0:
                move_to_high = sh.price - df.iloc[sh.index - lookback_range:sh.index + 1]['low'].min()
                move_strength = min(move_to_high / (recent_atr * SMCConstants.LIQUIDITY_MOVE_ATR_DIVISOR), 1.0) * SMCConstants.LIQUIDITY_MOVE_WEIGHT
            else:
                move_strength = SMCConstants.LIQUIDITY_MOVE_WEIGHT / 2

            test_count = 0
            tolerance = 1 - SMCConstants.LIQUIDITY_TEST_TOLERANCE_PCT
            for j in range(sh.index + 1, len(df)):
                high_j = df.iloc[j]['high']
                if high_j >= sh.price * tolerance and high_j <= sh.price:
                    test_count += 1
            test_strength = min(test_count * 10, SMCConstants.LIQUIDITY_TEST_WEIGHT)

            candles_ago = len(df) - sh.index
            recency_strength = max(0, SMCConstants.LIQUIDITY_RECENCY_WEIGHT - (candles_ago / 5))

            total_strength = min(100, move_strength + test_strength + recency_strength)

            lz = LiquidityZone(
                type='buy-side',
                price=sh.price,
                strength=round(total_strength, 1),
                touched=touched,
                sweep_index=sweep_idx,
                swing_index=sh.index,
                timestamp=sh.timestamp,
                session=sh.session
            )
            liquidity_zones.append(lz)

        # Process swing lows -> Sell-side liquidity (SSL)
        for sl in swing_lows:
            touched = False
            sweep_idx = None
            for j in range(sl.index + 1, len(df)):
                if df.iloc[j]['low'] < sl.price:
                    touched = True
                    sweep_idx = j
                    break

            if touched and sweep_idx and (len(df) - sweep_idx) > SMCConstants.LIQUIDITY_SWEEP_AGE_CUTOFF:
                continue

            lookback_range = min(10, sl.index)
            if lookback_range > 0:
                move_to_low = df.iloc[sl.index - lookback_range:sl.index + 1]['high'].max() - sl.price
                move_strength = min(move_to_low / (recent_atr * SMCConstants.LIQUIDITY_MOVE_ATR_DIVISOR), 1.0) * SMCConstants.LIQUIDITY_MOVE_WEIGHT
            else:
                move_strength = SMCConstants.LIQUIDITY_MOVE_WEIGHT / 2

            test_count = 0
            tolerance = 1 + SMCConstants.LIQUIDITY_TEST_TOLERANCE_PCT
            for j in range(sl.index + 1, len(df)):
                low_j = df.iloc[j]['low']
                if low_j <= sl.price * tolerance and low_j >= sl.price:
                    test_count += 1
            test_strength = min(test_count * 10, SMCConstants.LIQUIDITY_TEST_WEIGHT)

            candles_ago = len(df) - sl.index
            recency_strength = max(0, SMCConstants.LIQUIDITY_RECENCY_WEIGHT - (candles_ago / 5))

            total_strength = min(100, move_strength + test_strength + recency_strength)

            lz = LiquidityZone(
                type='sell-side',
                price=sl.price,
                strength=round(total_strength, 1),
                touched=touched,
                sweep_index=sweep_idx,
                swing_index=sl.index,
                timestamp=sl.timestamp,
                session=sl.session
            )
            liquidity_zones.append(lz)

        liquidity_zones.sort(key=lambda x: abs(x.price - current_price))
        return liquidity_zones[:max_zones]

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
                    'timestamp': ob.timestamp,
                    'detection_method': getattr(ob, 'detection_method', 'candle'),
                    'zone_candles': getattr(ob, 'zone_candles', 1)
                })

        # Unmitigated bearish OBs = resistance
        for ob in order_blocks:
            if ob.type == 'bearish' and not ob.mitigated and ob.bottom > current_price:
                resistance_zones.append({
                    'type': 'order_block',
                    'top': ob.top,
                    'bottom': ob.bottom,
                    'strength': ob.strength,
                    'timestamp': ob.timestamp,
                    'detection_method': getattr(ob, 'detection_method', 'candle'),
                    'zone_candles': getattr(ob, 'zone_candles', 1)
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
        current_price: Optional[float] = None,
        use_structural_obs: bool = True
    ) -> Dict[str, Any]:
        """
        Run full smart money analysis.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price (uses last close if not provided)
            use_structural_obs: Whether to include structural OB detection (default: True)

        Returns:
            dict with all SMC analysis
        """
        if current_price is None:
            current_price = df.iloc[-1]['close']

        # Detect swing points first (needed for structural OBs)
        swing_points = self.detect_swing_points(df)

        # Detect order blocks using both methods
        candle_obs = self.detect_order_blocks(df)

        # Also detect structural order blocks (larger institutional zones)
        structural_obs = []
        if use_structural_obs and swing_points:
            structural_obs = self.detect_structural_order_blocks(df, swing_points)

        # Merge order blocks, prioritizing structural ones (they're more significant)
        # Remove candle OBs that overlap significantly with structural OBs
        merged_obs = list(structural_obs)  # Start with structural
        for candle_ob in candle_obs:
            is_duplicate = False
            for struct_ob in structural_obs:
                if candle_ob.type != struct_ob.type:
                    continue
                # Check for overlap
                overlap_top = min(candle_ob.top, struct_ob.top)
                overlap_bottom = max(candle_ob.bottom, struct_ob.bottom)
                if overlap_top > overlap_bottom:
                    # There's overlap
                    candle_range = candle_ob.top - candle_ob.bottom
                    if candle_range > 0:
                        overlap_pct = (overlap_top - overlap_bottom) / candle_range
                        if overlap_pct > 0.3:  # 30% overlap = duplicate
                            is_duplicate = True
                            break
            if not is_duplicate:
                merged_obs.append(candle_ob)

        order_blocks = merged_obs

        fvgs = self.detect_fair_value_gaps(df)
        structure_breaks = self.detect_structure_breaks(df, swing_points)
        zones = self.get_unmitigated_zones(order_blocks, fvgs, current_price)

        # Detect liquidity zones (stop loss clusters above highs / below lows)
        liquidity_zones = self.detect_liquidity_zones(df, swing_points, current_price)

        # Find nearest levels
        nearest_support = zones['support'][0] if zones['support'] else None
        nearest_resistance = zones['resistance'][0] if zones['resistance'] else None

        # Recent structure breaks
        recent_bos = [sp for sp in structure_breaks['bos'] if sp.break_index and sp.break_index >= len(df) - 20]
        recent_choc = [sp for sp in structure_breaks['choc'] if sp.break_index and sp.break_index >= len(df) - 20]

        # Separate structural and candle-based OBs for reporting
        structural_count = len([ob for ob in order_blocks if ob.detection_method == 'structural'])
        candle_count = len([ob for ob in order_blocks if ob.detection_method == 'candle'])

        return {
            'current_price': current_price,
            'order_blocks': {
                'total': len(order_blocks),
                'unmitigated': len([ob for ob in order_blocks if not ob.mitigated]),
                'structural': structural_count,
                'candle_based': candle_count,
                'bullish': [ob for ob in order_blocks if ob.type == 'bullish'],
                'bearish': [ob for ob in order_blocks if ob.type == 'bearish']
            },
            'fair_value_gaps': {
                'total': len(fvgs),
                'unmitigated': len([fvg for fvg in fvgs if not fvg.mitigated]),
                'bullish': [fvg for fvg in fvgs if fvg.type == 'bullish'],
                'bearish': [fvg for fvg in fvgs if fvg.type == 'bearish']
            },
            'liquidity_zones': liquidity_zones,
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

    # =========================================================================
    # NEW DETECTION METHODS
    # =========================================================================

    def detect_equal_levels(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        tolerance_atr: float = SMCConstants.EQUAL_LEVEL_TOLERANCE_ATR,
        min_touches: int = SMCConstants.EQUAL_LEVEL_MIN_TOUCHES
    ) -> List[EqualLevel]:
        """
        Detect equal highs and equal lows - major liquidity targets.

        Equal highs/lows occur when price creates multiple swing points at
        similar prices. These are significant because:
        - EQH: Buy stops cluster above (shorts' stops)
        - EQL: Sell stops cluster below (longs' stops)

        Args:
            df: DataFrame with OHLC data
            swing_points: List of detected swing points
            tolerance_atr: Price tolerance as ATR multiple
            min_touches: Minimum touches to qualify as equal level

        Returns:
            List of EqualLevel objects
        """
        if len(swing_points) < 2 or len(df) < 20:
            return []

        equal_levels = []
        atr_series = self._get_atr(df)
        recent_atr = atr_series.iloc[-1] if pd.notna(atr_series.iloc[-1]) else 1.0
        tolerance = recent_atr * tolerance_atr

        # Group swing highs
        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']

        # Find equal highs
        processed_highs = set()
        for i, sh1 in enumerate(swing_highs):
            if i in processed_highs:
                continue

            matching_highs = [sh1]
            matching_indices = [sh1.index]
            matching_timestamps = [sh1.timestamp]

            for j, sh2 in enumerate(swing_highs):
                if i == j or j in processed_highs:
                    continue
                if abs(sh1.price - sh2.price) <= tolerance:
                    matching_highs.append(sh2)
                    matching_indices.append(sh2.index)
                    matching_timestamps.append(sh2.timestamp)
                    processed_highs.add(j)

            if len(matching_highs) >= min_touches:
                avg_price = sum(h.price for h in matching_highs) / len(matching_highs)

                # Check if swept
                swept = False
                sweep_idx = None
                max_index = max(matching_indices)
                for k in range(max_index + 1, len(df)):
                    if df.iloc[k]['high'] > avg_price + tolerance:
                        swept = True
                        sweep_idx = k
                        break

                eq = EqualLevel(
                    type='equal_highs',
                    price=avg_price,
                    touches=len(matching_highs),
                    indices=matching_indices,
                    timestamps=matching_timestamps,
                    swept=swept,
                    sweep_index=sweep_idx
                )
                equal_levels.append(eq)
                processed_highs.add(i)

        # Find equal lows
        processed_lows = set()
        for i, sl1 in enumerate(swing_lows):
            if i in processed_lows:
                continue

            matching_lows = [sl1]
            matching_indices = [sl1.index]
            matching_timestamps = [sl1.timestamp]

            for j, sl2 in enumerate(swing_lows):
                if i == j or j in processed_lows:
                    continue
                if abs(sl1.price - sl2.price) <= tolerance:
                    matching_lows.append(sl2)
                    matching_indices.append(sl2.index)
                    matching_timestamps.append(sl2.timestamp)
                    processed_lows.add(j)

            if len(matching_lows) >= min_touches:
                avg_price = sum(l.price for l in matching_lows) / len(matching_lows)

                # Check if swept
                swept = False
                sweep_idx = None
                max_index = max(matching_indices)
                for k in range(max_index + 1, len(df)):
                    if df.iloc[k]['low'] < avg_price - tolerance:
                        swept = True
                        sweep_idx = k
                        break

                eq = EqualLevel(
                    type='equal_lows',
                    price=avg_price,
                    touches=len(matching_lows),
                    indices=matching_indices,
                    timestamps=matching_timestamps,
                    swept=swept,
                    sweep_index=sweep_idx
                )
                equal_levels.append(eq)
                processed_lows.add(i)

        return equal_levels

    def detect_breaker_blocks(
        self,
        df: pd.DataFrame,
        order_blocks: List[OrderBlock]
    ) -> List[BreakerBlock]:
        """
        Detect breaker blocks - failed order blocks that flip polarity.

        When an OB fails (price closes through it), it becomes a breaker:
        - Failed bullish OB becomes bearish breaker (resistance)
        - Failed bearish OB becomes bullish breaker (support)

        Breakers are strong reversal zones because they represent trapped traders.

        Args:
            df: DataFrame with OHLC data
            order_blocks: List of detected order blocks

        Returns:
            List of BreakerBlock objects
        """
        if not order_blocks or len(df) < 10:
            return []

        breaker_blocks = []

        for ob in order_blocks:
            # Only invalidated OBs become breakers
            if not ob.invalidated or ob.invalidation_index is None:
                continue

            # Bullish OB that got invalidated -> Bearish breaker
            if ob.type == 'bullish':
                breaker = BreakerBlock(
                    original_type='bullish',
                    current_type='bearish',  # Flipped
                    top=ob.top,
                    bottom=ob.bottom,
                    original_index=ob.candle_index,
                    break_index=ob.invalidation_index,
                    timestamp=ob.timestamp,
                    strength=ob.strength * 0.8  # Slightly reduced strength
                )

                # Check if breaker has been mitigated (price returned to zone)
                for j in range(ob.invalidation_index + 1, len(df)):
                    if df.iloc[j]['high'] >= breaker.bottom:
                        breaker.mitigated = True
                        breaker.mitigation_index = j
                        break

                breaker_blocks.append(breaker)

            # Bearish OB that got invalidated -> Bullish breaker
            elif ob.type == 'bearish':
                breaker = BreakerBlock(
                    original_type='bearish',
                    current_type='bullish',  # Flipped
                    top=ob.top,
                    bottom=ob.bottom,
                    original_index=ob.candle_index,
                    break_index=ob.invalidation_index,
                    timestamp=ob.timestamp,
                    strength=ob.strength * 0.8
                )

                for j in range(ob.invalidation_index + 1, len(df)):
                    if df.iloc[j]['low'] <= breaker.top:
                        breaker.mitigated = True
                        breaker.mitigation_index = j
                        break

                breaker_blocks.append(breaker)

        return breaker_blocks

    def calculate_premium_discount(
        self,
        df: pd.DataFrame,
        lookback: int = SMCConstants.PREMIUM_DISCOUNT_LOOKBACK
    ) -> PremiumDiscountZone:
        """
        Calculate premium/discount zone relative to recent range.

        In SMC:
        - Premium zone (above 50%): Look for sells
        - Discount zone (below 50%): Look for buys
        - Equilibrium (at 50%): Neutral

        Args:
            df: DataFrame with OHLC data
            lookback: Candles to calculate range from

        Returns:
            PremiumDiscountZone object
        """
        recent = df.iloc[-lookback:] if len(df) >= lookback else df
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        equilibrium = (range_high + range_low) / 2
        current_price = df.iloc[-1]['close']

        range_size = range_high - range_low
        if range_size > 0:
            position_pct = ((current_price - range_low) / range_size) * 100
        else:
            position_pct = 50.0

        # Determine zone
        if position_pct > 70:
            zone = "premium"
        elif position_pct < 30:
            zone = "discount"
        elif position_pct > 50:
            zone = "premium"  # Slight premium
        elif position_pct < 50:
            zone = "discount"  # Slight discount
        else:
            zone = "equilibrium"

        return PremiumDiscountZone(
            range_high=range_high,
            range_low=range_low,
            equilibrium=equilibrium,
            current_price=current_price,
            position_pct=round(position_pct, 2),
            zone=zone
        )

    def calculate_ote_zone(
        self,
        swing_high: float,
        swing_low: float,
        direction: str,
        fib_start: float = SMCConstants.OTE_FIB_START,
        fib_end: float = SMCConstants.OTE_FIB_END
    ) -> OTEZone:
        """
        Calculate Optimal Trade Entry (OTE) zone using Fibonacci retracement.

        The OTE zone (typically 62-79% retracement) is where institutions
        often enter trades after an impulse move.

        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            direction: "bullish" (looking for buy) or "bearish" (looking for sell)
            fib_start: Starting Fibonacci level (default 0.62)
            fib_end: Ending Fibonacci level (default 0.79)

        Returns:
            OTEZone object
        """
        range_size = swing_high - swing_low

        if direction == 'bullish':
            # For bullish OTE, measure retracement from high to low
            # OTE zone is where price pulls back before continuing up
            ote_top = swing_high - (range_size * fib_start)
            ote_bottom = swing_high - (range_size * fib_end)
        else:
            # For bearish OTE, measure retracement from low to high
            # OTE zone is where price pulls back before continuing down
            ote_bottom = swing_low + (range_size * fib_start)
            ote_top = swing_low + (range_size * fib_end)

        return OTEZone(
            direction=direction,
            top=ote_top,
            bottom=ote_bottom,
            swing_high=swing_high,
            swing_low=swing_low,
            fib_start=fib_start,
            fib_end=fib_end
        )

    def find_ote_zones(
        self,
        swing_points: List[StructurePoint],
        current_price: float
    ) -> List[OTEZone]:
        """
        Find relevant OTE zones from recent swing points.

        Args:
            swing_points: List of swing points
            current_price: Current market price

        Returns:
            List of OTEZone objects near current price
        """
        if len(swing_points) < 2:
            return []

        ote_zones = []
        highs = [sp for sp in swing_points if sp.type == 'high']
        lows = [sp for sp in swing_points if sp.type == 'low']

        # Find pairs of swing high/low for OTE calculation
        for high in highs:
            # Find the nearest preceding low
            preceding_lows = [low for low in lows if low.index < high.index]
            if preceding_lows:
                low = preceding_lows[-1]  # Most recent low before this high

                # Bullish OTE: After an up move, look for pullback zone
                ote = self.calculate_ote_zone(high.price, low.price, 'bullish')

                # Only include if price is near or in the OTE zone
                if ote.bottom <= current_price <= ote.top * 1.02:
                    ote_zones.append(ote)

        for low in lows:
            # Find the nearest preceding high
            preceding_highs = [high for high in highs if high.index < low.index]
            if preceding_highs:
                high = preceding_highs[-1]

                # Bearish OTE: After a down move, look for pullback zone
                ote = self.calculate_ote_zone(high.price, low.price, 'bearish')

                if ote.bottom * 0.98 <= current_price <= ote.top:
                    ote_zones.append(ote)

        return ote_zones

    def calculate_confluence_score(
        self,
        price: float,
        analysis: Dict[str, Any],
        tolerance_pct: float = 0.5,
        market_regime: Optional[str] = None,
        trade_direction: Optional[str] = None
    ) -> ConfluenceScore:
        """
        Calculate confluence score at a specific price level.

        Confluence occurs when multiple SMC concepts align at the same price.
        Higher confluence = higher probability trade setup.

        Args:
            price: The price level to check
            analysis: Full SMC analysis dict from analyze_full_smc
            tolerance_pct: Price tolerance as percentage
            market_regime: Current market regime ("trending-up", "trending-down", "ranging")
            trade_direction: Proposed trade direction ("bullish" or "bearish")

        Returns:
            ConfluenceScore object
        """
        score = 0
        factors = []
        bullish_factors = 0
        bearish_factors = 0
        tolerance = price * (tolerance_pct / 100)

        # Add regime alignment bonus
        if market_regime and trade_direction:
            regime_aligned = False
            if market_regime == "trending-up" and trade_direction == "bullish":
                regime_aligned = True
                factors.append("Regime aligned (bullish in uptrend)")
            elif market_regime == "trending-down" and trade_direction == "bearish":
                regime_aligned = True
                factors.append("Regime aligned (bearish in downtrend)")
            elif market_regime == "ranging":
                # In ranging, both directions are valid at extremes
                factors.append("Ranging market (mean reversion)")
                regime_aligned = True

            if regime_aligned:
                score += SMCConstants.CONFLUENCE_REGIME_ALIGNMENT_WEIGHT
                if trade_direction == "bullish":
                    bullish_factors += 1
                else:
                    bearish_factors += 1

        # Check Order Blocks
        for ob in analysis['order_blocks']['bullish']:
            if ob.bottom - tolerance <= price <= ob.top + tolerance and not ob.mitigated:
                score += SMCConstants.CONFLUENCE_OB_WEIGHT
                factors.append(f"Bullish OB ({ob.strength:.0%})")
                bullish_factors += 1

        for ob in analysis['order_blocks']['bearish']:
            if ob.bottom - tolerance <= price <= ob.top + tolerance and not ob.mitigated:
                score += SMCConstants.CONFLUENCE_OB_WEIGHT
                factors.append(f"Bearish OB ({ob.strength:.0%})")
                bearish_factors += 1

        # Check FVGs
        for fvg in analysis['fair_value_gaps']['bullish']:
            if fvg.bottom - tolerance <= price <= fvg.top + tolerance and not fvg.mitigated:
                score += SMCConstants.CONFLUENCE_FVG_WEIGHT
                factors.append(f"Bullish FVG ({fvg.fill_percentage:.0f}% filled)")
                bullish_factors += 1

        for fvg in analysis['fair_value_gaps']['bearish']:
            if fvg.bottom - tolerance <= price <= fvg.top + tolerance and not fvg.mitigated:
                score += SMCConstants.CONFLUENCE_FVG_WEIGHT
                factors.append(f"Bearish FVG ({fvg.fill_percentage:.0f}% filled)")
                bearish_factors += 1

        # Check Liquidity zones
        for lz in analysis['liquidity_zones']:
            if abs(lz.price - price) <= tolerance and not lz.touched:
                score += SMCConstants.CONFLUENCE_LIQUIDITY_WEIGHT
                factors.append(f"{lz.type} liquidity ({lz.strength:.0f}%)")
                if lz.type == 'buy-side':
                    bearish_factors += 1  # Buy-side liquidity = target for sells
                else:
                    bullish_factors += 1  # Sell-side liquidity = target for buys

        # Check Premium/Discount
        if 'premium_discount' in analysis:
            pd_zone = analysis['premium_discount']
            if pd_zone.zone == 'discount' and price < pd_zone.equilibrium:
                score += SMCConstants.CONFLUENCE_PREMIUM_DISCOUNT_WEIGHT
                factors.append("In discount zone (bullish)")
                bullish_factors += 1
            elif pd_zone.zone == 'premium' and price > pd_zone.equilibrium:
                score += SMCConstants.CONFLUENCE_PREMIUM_DISCOUNT_WEIGHT
                factors.append("In premium zone (bearish)")
                bearish_factors += 1

        # Check OTE zones
        if 'ote_zones' in analysis:
            for ote in analysis['ote_zones']:
                if ote.bottom <= price <= ote.top:
                    score += SMCConstants.CONFLUENCE_OTE_WEIGHT
                    factors.append(f"{ote.direction.capitalize()} OTE zone")
                    if ote.direction == 'bullish':
                        bullish_factors += 1
                    else:
                        bearish_factors += 1

        return ConfluenceScore(
            price=price,
            total_score=min(score, 100),
            factors=factors,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors
        )

    def align_mtf_zones(
        self,
        htf_analysis: Dict[str, Any],
        ltf_analysis: Dict[str, Any],
        tolerance_pct: float = 0.3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Align zones across multiple timeframes.

        Zones that exist on both higher and lower timeframes are stronger
        because they represent institutional interest on multiple levels.

        Args:
            htf_analysis: Higher timeframe analysis (e.g., H4, Daily)
            ltf_analysis: Lower timeframe analysis (e.g., M15, H1)
            tolerance_pct: Price tolerance for matching

        Returns:
            Dict with 'aligned_support' and 'aligned_resistance' lists
        """
        aligned_support = []
        aligned_resistance = []

        htf_support = htf_analysis['zones']['support']
        htf_resistance = htf_analysis['zones']['resistance']
        ltf_support = ltf_analysis['zones']['support']
        ltf_resistance = ltf_analysis['zones']['resistance']

        # Find aligned support zones
        for htf_zone in htf_support:
            htf_mid = (htf_zone['top'] + htf_zone['bottom']) / 2
            tolerance = htf_mid * (tolerance_pct / 100)

            for ltf_zone in ltf_support:
                ltf_mid = (ltf_zone['top'] + ltf_zone['bottom']) / 2

                if abs(htf_mid - ltf_mid) <= tolerance:
                    aligned_support.append({
                        'htf_zone': htf_zone,
                        'ltf_zone': ltf_zone,
                        'avg_price': (htf_mid + ltf_mid) / 2,
                        'combined_strength': (htf_zone['strength'] + ltf_zone.get('strength', 0.5)) / 2,
                        'mtf_confirmed': True
                    })
                    break

        # Find aligned resistance zones
        for htf_zone in htf_resistance:
            htf_mid = (htf_zone['top'] + htf_zone['bottom']) / 2
            tolerance = htf_mid * (tolerance_pct / 100)

            for ltf_zone in ltf_resistance:
                ltf_mid = (ltf_zone['top'] + ltf_zone['bottom']) / 2

                if abs(htf_mid - ltf_mid) <= tolerance:
                    aligned_resistance.append({
                        'htf_zone': htf_zone,
                        'ltf_zone': ltf_zone,
                        'avg_price': (htf_mid + ltf_mid) / 2,
                        'combined_strength': (htf_zone['strength'] + ltf_zone.get('strength', 0.5)) / 2,
                        'mtf_confirmed': True
                    })
                    break

        return {
            'aligned_support': aligned_support,
            'aligned_resistance': aligned_resistance
        }

    def analyze_full_smc_extended(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        use_structural_obs: bool = True,
        include_equal_levels: bool = True,
        include_breakers: bool = True,
        include_ote: bool = True,
        include_sweeps: bool = True,
        include_inducements: bool = True,
        include_rejections: bool = True,
        include_turtle_soup: bool = True
    ) -> Dict[str, Any]:
        """
        Run extended full smart money analysis with all features.

        This is an enhanced version of analyze_full_smc that includes:
        - Equal highs/lows detection
        - Breaker blocks
        - Premium/Discount zones
        - OTE zones
        - Liquidity Sweeps (NEW)
        - Inducement patterns (NEW)
        - Rejection Blocks (NEW)
        - Turtle Soup patterns (NEW)
        - Confluence scoring for key levels

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price (uses last close if not provided)
            use_structural_obs: Include structural OB detection
            include_equal_levels: Include equal highs/lows
            include_breakers: Include breaker block detection
            include_ote: Include OTE zone calculation
            include_sweeps: Include liquidity sweep detection
            include_inducements: Include inducement pattern detection
            include_rejections: Include rejection block detection
            include_turtle_soup: Include turtle soup detection

        Returns:
            Extended dict with all SMC analysis
        """
        # Start with base analysis
        analysis = self.analyze_full_smc(df, current_price, use_structural_obs)

        if current_price is None:
            current_price = df.iloc[-1]['close']

        swing_points = self.detect_swing_points(df)

        # Add Premium/Discount zone
        analysis['premium_discount'] = self.calculate_premium_discount(df)

        # Add Equal levels
        if include_equal_levels:
            equal_levels = self.detect_equal_levels(df, swing_points)
            analysis['equal_levels'] = {
                'equal_highs': [el for el in equal_levels if el.type == 'equal_highs'],
                'equal_lows': [el for el in equal_levels if el.type == 'equal_lows'],
                'total': len(equal_levels),
                'unswept': len([el for el in equal_levels if not el.swept])
            }

        # Add Breaker blocks
        if include_breakers:
            all_obs = analysis['order_blocks']['bullish'] + analysis['order_blocks']['bearish']
            breaker_blocks = self.detect_breaker_blocks(df, all_obs)
            analysis['breaker_blocks'] = {
                'total': len(breaker_blocks),
                'unmitigated': len([bb for bb in breaker_blocks if not bb.mitigated]),
                'bullish': [bb for bb in breaker_blocks if bb.current_type == 'bullish'],
                'bearish': [bb for bb in breaker_blocks if bb.current_type == 'bearish']
            }

        # Add OTE zones
        if include_ote:
            ote_zones = self.find_ote_zones(swing_points, current_price)
            analysis['ote_zones'] = ote_zones

        # Add Liquidity Sweeps (NEW)
        if include_sweeps:
            sweeps = self.detect_liquidity_sweeps(df, swing_points)
            recent_sweeps = [s for s in sweeps if s.sweep_candle_index >= len(df) - 10]
            analysis['liquidity_sweeps'] = {
                'total': len(sweeps),
                'recent': recent_sweeps,
                'bullish': [s for s in sweeps if s.type == 'bullish'],
                'bearish': [s for s in sweeps if s.type == 'bearish'],
                'strong_sweeps': [s for s in sweeps if s.is_strong]
            }

        # Add Inducements (NEW)
        if include_inducements:
            inducements = self.detect_inducements(df, swing_points)
            recent_inducements = [ind for ind in inducements if ind.reversal_index >= len(df) - 15]
            analysis['inducements'] = {
                'total': len(inducements),
                'recent': recent_inducements,
                'bullish': [ind for ind in inducements if ind.type == 'bullish'],
                'bearish': [ind for ind in inducements if ind.type == 'bearish']
            }

        # Add Rejection Blocks (NEW)
        if include_rejections:
            rejection_blocks = self.detect_rejection_blocks(df)
            unmitigated_rejections = [rb for rb in rejection_blocks if not rb.mitigated]
            analysis['rejection_blocks'] = {
                'total': len(rejection_blocks),
                'unmitigated': len(unmitigated_rejections),
                'bullish': [rb for rb in rejection_blocks if rb.type == 'bullish'],
                'bearish': [rb for rb in rejection_blocks if rb.type == 'bearish'],
                'held': [rb for rb in rejection_blocks if rb.held]
            }

        # Add Turtle Soup (NEW)
        if include_turtle_soup:
            turtle_soups = self.detect_turtle_soup(df, swing_points)
            recent_ts = [ts for ts in turtle_soups if ts.break_candle_index >= len(df) - 10]
            analysis['turtle_soup'] = {
                'total': len(turtle_soups),
                'recent': recent_ts,
                'bullish': [ts for ts in turtle_soups if ts.type == 'bullish'],
                'bearish': [ts for ts in turtle_soups if ts.type == 'bearish']
            }

        # Generate alerts for actionable patterns
        analysis['alerts'] = self._generate_pattern_alerts(analysis, current_price, df)

        # Calculate confluence for nearest support and resistance
        if analysis['nearest_support']:
            support_price = (analysis['nearest_support']['top'] + analysis['nearest_support']['bottom']) / 2
            analysis['support_confluence'] = self.calculate_confluence_score(support_price, analysis)

        if analysis['nearest_resistance']:
            resistance_price = (analysis['nearest_resistance']['top'] + analysis['nearest_resistance']['bottom']) / 2
            analysis['resistance_confluence'] = self.calculate_confluence_score(resistance_price, analysis)

        # Calculate confluence at current price
        analysis['current_confluence'] = self.calculate_confluence_score(current_price, analysis)

        return analysis

    def _generate_pattern_alerts(
        self,
        analysis: Dict[str, Any],
        current_price: float,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable alerts from detected patterns.

        Args:
            analysis: Full SMC analysis dict
            current_price: Current market price
            df: Price DataFrame

        Returns:
            List of alert dictionaries
        """
        alerts = []
        last_candle_idx = len(df) - 1

        # Alert on recent liquidity sweeps (high priority)
        if 'liquidity_sweeps' in analysis:
            for sweep in analysis['liquidity_sweeps'].get('recent', []):
                if sweep.sweep_candle_index >= last_candle_idx - 3:
                    alerts.append({
                        'type': 'LIQUIDITY_SWEEP',
                        'priority': 'HIGH' if sweep.is_strong else 'MEDIUM',
                        'direction': sweep.signal_direction,
                        'message': f"{'Strong ' if sweep.is_strong else ''}{sweep.type.upper()} sweep at {sweep.sweep_level:.5f}",
                        'details': {
                            'sweep_level': sweep.sweep_level,
                            'rejection_strength': sweep.rejection_strength,
                            'atr_penetration': sweep.atr_penetration,
                            'session': sweep.session
                        },
                        'candle_index': sweep.sweep_candle_index,
                        'timestamp': sweep.timestamp
                    })

        # Alert on recent inducements
        if 'inducements' in analysis:
            for ind in analysis['inducements'].get('recent', []):
                if ind.reversal_index >= last_candle_idx - 5:
                    alerts.append({
                        'type': 'INDUCEMENT',
                        'priority': 'HIGH',
                        'direction': 'BUY' if ind.type == 'bullish' else 'SELL',
                        'message': f"{ind.type.upper()} inducement - trapped {ind.trapped_direction}s at {ind.inducement_level:.5f}",
                        'details': {
                            'inducement_level': ind.inducement_level,
                            'target_liquidity': ind.target_liquidity,
                            'trapped_direction': ind.trapped_direction,
                            'candles_to_reversal': ind.candles_to_reversal
                        },
                        'candle_index': ind.reversal_index,
                        'timestamp': ind.timestamp
                    })

        # Alert on recent turtle soup patterns
        if 'turtle_soup' in analysis:
            for ts in analysis['turtle_soup'].get('recent', []):
                if ts.break_candle_index >= last_candle_idx - 3:
                    alerts.append({
                        'type': 'TURTLE_SOUP',
                        'priority': 'MEDIUM',
                        'direction': ts.trade_direction,
                        'message': f"{ts.type.upper()} Turtle Soup at {ts.level:.5f} (failed breakout)",
                        'details': {
                            'level': ts.level,
                            'penetration_atr': ts.penetration_atr
                        },
                        'candle_index': ts.break_candle_index,
                        'timestamp': ts.timestamp
                    })

        # Alert on strong rejection blocks that held
        if 'rejection_blocks' in analysis:
            for rb in analysis['rejection_blocks'].get('held', []):
                if rb.candle_index >= last_candle_idx - 5 and not rb.mitigated:
                    alerts.append({
                        'type': 'REJECTION_BLOCK',
                        'priority': 'MEDIUM' if rb.wick_atr_ratio < 1.5 else 'HIGH',
                        'direction': 'BUY' if rb.type == 'bullish' else 'SELL',
                        'message': f"{rb.type.upper()} rejection at {rb.rejection_price:.5f} ({rb.wick_atr_ratio:.1f}x ATR wick)",
                        'details': {
                            'rejection_price': rb.rejection_price,
                            'zone_top': rb.zone_top,
                            'zone_bottom': rb.zone_bottom,
                            'wick_atr_ratio': rb.wick_atr_ratio,
                            'session': rb.session
                        },
                        'candle_index': rb.candle_index,
                        'timestamp': rb.timestamp
                    })

        # Alert on CHOCH (Change of Character) - trend reversal signal
        if analysis['structure'].get('recent_choc'):
            for choc in analysis['structure']['recent_choc']:
                if choc.break_index and choc.break_index >= last_candle_idx - 10:
                    direction = 'SELL' if choc.type == 'high' else 'BUY'
                    alerts.append({
                        'type': 'CHOCH',
                        'priority': 'HIGH',
                        'direction': direction,
                        'message': f"Change of Character - broke {choc.type} at {choc.price:.5f}",
                        'details': {
                            'break_price': choc.price,
                            'break_type': choc.break_type
                        },
                        'candle_index': choc.break_index,
                        'timestamp': choc.timestamp
                    })

        # Sort alerts by priority and recency
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        alerts.sort(key=lambda x: (priority_order.get(x['priority'], 2), -x['candle_index']))

        return alerts

    # =========================================================================
    # LIQUIDITY SWEEPS & ADVANCED PATTERNS
    # =========================================================================

    def detect_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        lookback: int = 30,
        min_rejection_pct: float = 0.3
    ) -> List['LiquiditySweep']:
        """
        Detect liquidity sweeps - price grabs stops then reverses.

        A sweep is a high-probability reversal signal when:
        1. Price breaks through a swing high/low (takes stops)
        2. Price immediately reverses (rejection)
        3. Candle closes back inside the previous range

        Args:
            df: DataFrame with OHLC data
            swing_points: List of detected swing points
            lookback: How many recent candles to check
            min_rejection_pct: Minimum rejection (wick vs body ratio) to qualify

        Returns:
            List of LiquiditySweep objects
        """
        if len(df) < 10 or not swing_points:
            return []

        sweeps = []
        atr_series = self._get_atr(df)

        # Get recent swing highs and lows
        recent_highs = [sp for sp in swing_points if sp.type == 'high']
        recent_lows = [sp for sp in swing_points if sp.type == 'low']

        start_idx = max(0, len(df) - lookback)

        for i in range(start_idx, len(df)):
            candle = df.iloc[i]
            atr = atr_series.iloc[i] if i < len(atr_series) and pd.notna(atr_series.iloc[i]) else None
            if atr is None or atr == 0:
                continue

            timestamp_str = str(candle.name) if hasattr(candle, 'name') else ''
            session = self._identify_session(candle.name) if hasattr(candle, 'name') else None

            # Check for bullish sweep (swept below swing low, closed above)
            for sl in recent_lows:
                if sl.index >= i:  # Can only sweep past swing points
                    continue

                # Check if this candle swept the low
                if candle['low'] < sl.price and candle['close'] > sl.price:
                    # Calculate rejection strength
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    body_size = abs(candle['close'] - candle['open'])
                    total_range = candle['high'] - candle['low']

                    if total_range > 0:
                        rejection_pct = lower_wick / total_range

                        if rejection_pct >= min_rejection_pct:
                            sweep = LiquiditySweep(
                                type='bullish',
                                sweep_level=sl.price,
                                sweep_candle_index=i,
                                sweep_low=candle['low'],
                                sweep_high=candle['high'],
                                close_price=candle['close'],
                                rejection_strength=round(rejection_pct, 2),
                                atr_penetration=round((sl.price - candle['low']) / atr, 2),
                                timestamp=timestamp_str,
                                session=session,
                                swing_index=sl.index
                            )
                            sweeps.append(sweep)
                            break  # Only count one sweep per candle

            # Check for bearish sweep (swept above swing high, closed below)
            for sh in recent_highs:
                if sh.index >= i:
                    continue

                if candle['high'] > sh.price and candle['close'] < sh.price:
                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    body_size = abs(candle['close'] - candle['open'])
                    total_range = candle['high'] - candle['low']

                    if total_range > 0:
                        rejection_pct = upper_wick / total_range

                        if rejection_pct >= min_rejection_pct:
                            sweep = LiquiditySweep(
                                type='bearish',
                                sweep_level=sh.price,
                                sweep_candle_index=i,
                                sweep_low=candle['low'],
                                sweep_high=candle['high'],
                                close_price=candle['close'],
                                rejection_strength=round(rejection_pct, 2),
                                atr_penetration=round((candle['high'] - sh.price) / atr, 2),
                                timestamp=timestamp_str,
                                session=session,
                                swing_index=sh.index
                            )
                            sweeps.append(sweep)
                            break

        return sweeps

    def detect_inducements(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        lookback: int = 50
    ) -> List['Inducement']:
        """
        Detect inducement patterns - false breakouts to trap retail traders.

        Inducement occurs when:
        1. Price breaks a minor swing high/low
        2. Traps traders in the wrong direction
        3. Then reverses to take their stops

        Often precedes a move to grab major liquidity.

        Args:
            df: DataFrame with OHLC data
            swing_points: List of detected swing points
            lookback: How many candles to analyze

        Returns:
            List of Inducement objects
        """
        if len(df) < 20 or len(swing_points) < 4:
            return []

        inducements = []
        atr_series = self._get_atr(df)

        # Sort swing points by recency
        sorted_swings = sorted(swing_points, key=lambda x: x.index, reverse=True)

        for i, sp in enumerate(sorted_swings[:-2]):  # Need at least 2 more swings after
            if len(df) - sp.index > lookback:
                continue

            atr = atr_series.iloc[sp.index] if sp.index < len(atr_series) and pd.notna(atr_series.iloc[sp.index]) else None
            if atr is None or atr == 0:
                continue

            timestamp_str = str(df.iloc[sp.index].name) if hasattr(df.iloc[sp.index], 'name') else ''

            # Look for a break of this swing followed by reversal
            if sp.type == 'high':
                # Find if price broke above this high then reversed
                broke_high = False
                break_idx = None
                reversal_idx = None

                for j in range(sp.index + 1, min(sp.index + 15, len(df))):
                    if df.iloc[j]['high'] > sp.price:
                        broke_high = True
                        break_idx = j
                        break

                if broke_high and break_idx:
                    # Check for reversal within next 5 candles
                    for k in range(break_idx + 1, min(break_idx + 6, len(df))):
                        if df.iloc[k]['close'] < sp.price:
                            reversal_idx = k

                            # Find the low that was targeted (below current swing)
                            target_lows = [s for s in swing_points if s.type == 'low' and s.index < sp.index]
                            target_low = target_lows[-1].price if target_lows else None

                            inducement = Inducement(
                                type='bearish',  # Induces longs, then drops
                                inducement_level=sp.price,
                                inducement_index=sp.index,
                                break_index=break_idx,
                                reversal_index=reversal_idx,
                                target_liquidity=target_low,
                                timestamp=timestamp_str,
                                trapped_direction='long'
                            )
                            inducements.append(inducement)
                            break

            elif sp.type == 'low':
                # Find if price broke below this low then reversed
                broke_low = False
                break_idx = None
                reversal_idx = None

                for j in range(sp.index + 1, min(sp.index + 15, len(df))):
                    if df.iloc[j]['low'] < sp.price:
                        broke_low = True
                        break_idx = j
                        break

                if broke_low and break_idx:
                    for k in range(break_idx + 1, min(break_idx + 6, len(df))):
                        if df.iloc[k]['close'] > sp.price:
                            reversal_idx = k

                            target_highs = [s for s in swing_points if s.type == 'high' and s.index < sp.index]
                            target_high = target_highs[-1].price if target_highs else None

                            inducement = Inducement(
                                type='bullish',  # Induces shorts, then rallies
                                inducement_level=sp.price,
                                inducement_index=sp.index,
                                break_index=break_idx,
                                reversal_index=reversal_idx,
                                target_liquidity=target_high,
                                timestamp=timestamp_str,
                                trapped_direction='short'
                            )
                            inducements.append(inducement)
                            break

        return inducements

    def detect_rejection_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 30,
        min_wick_atr: float = 1.0,
        min_wick_body_ratio: float = 2.0
    ) -> List['RejectionBlock']:
        """
        Detect rejection blocks - strong rejection candles at key levels.

        A rejection block forms when:
        1. Price has a long wick (rejection)
        2. Body is small relative to wick
        3. Indicates strong buying/selling pressure

        These often mark significant support/resistance levels.

        Args:
            df: DataFrame with OHLC data
            lookback: How many candles to analyze
            min_wick_atr: Minimum wick size in ATR multiples
            min_wick_body_ratio: Minimum ratio of wick to body

        Returns:
            List of RejectionBlock objects
        """
        if len(df) < 10:
            return []

        rejection_blocks = []
        atr_series = self._get_atr(df)

        start_idx = max(0, len(df) - lookback)

        for i in range(start_idx, len(df)):
            candle = df.iloc[i]
            atr = atr_series.iloc[i] if i < len(atr_series) and pd.notna(atr_series.iloc[i]) else None
            if atr is None or atr == 0:
                continue

            timestamp_str = str(candle.name) if hasattr(candle, 'name') else ''
            session = self._identify_session(candle.name) if hasattr(candle, 'name') else None

            body_top = max(candle['open'], candle['close'])
            body_bottom = min(candle['open'], candle['close'])
            body_size = body_top - body_bottom

            upper_wick = candle['high'] - body_top
            lower_wick = body_bottom - candle['low']

            # Bullish rejection (long lower wick)
            if lower_wick >= atr * min_wick_atr:
                if body_size > 0 and lower_wick / body_size >= min_wick_body_ratio:
                    # Check if it held (next candle didn't break the low)
                    held = True
                    if i + 1 < len(df):
                        held = df.iloc[i + 1]['low'] > candle['low']

                    rb = RejectionBlock(
                        type='bullish',
                        rejection_price=candle['low'],
                        body_top=body_top,
                        body_bottom=body_bottom,
                        wick_size=lower_wick,
                        wick_atr_ratio=round(lower_wick / atr, 2),
                        candle_index=i,
                        timestamp=timestamp_str,
                        session=session,
                        held=held,
                        mitigated=False
                    )
                    rejection_blocks.append(rb)

            # Bearish rejection (long upper wick)
            if upper_wick >= atr * min_wick_atr:
                if body_size > 0 and upper_wick / body_size >= min_wick_body_ratio:
                    held = True
                    if i + 1 < len(df):
                        held = df.iloc[i + 1]['high'] < candle['high']

                    rb = RejectionBlock(
                        type='bearish',
                        rejection_price=candle['high'],
                        body_top=body_top,
                        body_bottom=body_bottom,
                        wick_size=upper_wick,
                        wick_atr_ratio=round(upper_wick / atr, 2),
                        candle_index=i,
                        timestamp=timestamp_str,
                        session=session,
                        held=held,
                        mitigated=False
                    )
                    rejection_blocks.append(rb)

        # Check for mitigation
        for rb in rejection_blocks:
            for j in range(rb.candle_index + 1, len(df)):
                if rb.type == 'bullish' and df.iloc[j]['low'] < rb.rejection_price:
                    rb.mitigated = True
                    rb.mitigation_index = j
                    break
                elif rb.type == 'bearish' and df.iloc[j]['high'] > rb.rejection_price:
                    rb.mitigated = True
                    rb.mitigation_index = j
                    break

        return rejection_blocks

    def detect_turtle_soup(
        self,
        df: pd.DataFrame,
        swing_points: List[StructurePoint],
        lookback: int = 30,
        max_penetration_atr: float = 0.5
    ) -> List['TurtleSoup']:
        """
        Detect Turtle Soup patterns - failed breakouts with quick reversals.

        Named after the classic trading strategy, this pattern occurs when:
        1. Price breaks a significant high/low
        2. Fails to hold (reverses quickly)
        3. Traps breakout traders

        Similar to sweeps but focuses on the failure aspect.

        Args:
            df: DataFrame with OHLC data
            swing_points: List of detected swing points
            lookback: Candles to analyze
            max_penetration_atr: Maximum break beyond level (in ATR)

        Returns:
            List of TurtleSoup objects
        """
        if len(df) < 20 or not swing_points:
            return []

        turtle_soups = []
        atr_series = self._get_atr(df)

        start_idx = max(0, len(df) - lookback)

        for sp in swing_points:
            if sp.index < start_idx or sp.index >= len(df) - 2:
                continue

            atr = atr_series.iloc[sp.index] if sp.index < len(atr_series) and pd.notna(atr_series.iloc[sp.index]) else None
            if atr is None or atr == 0:
                continue

            timestamp_str = str(df.iloc[sp.index].name) if hasattr(df.iloc[sp.index], 'name') else ''

            # Look for break and fail within next 3 candles
            for i in range(sp.index + 1, min(sp.index + 4, len(df))):
                candle = df.iloc[i]

                if sp.type == 'high':
                    # Bearish Turtle Soup: Breaks high, fails, reverses down
                    penetration = candle['high'] - sp.price

                    if penetration > 0 and penetration <= atr * max_penetration_atr:
                        # Check if closed below the high (failed breakout)
                        if candle['close'] < sp.price:
                            ts = TurtleSoup(
                                type='bearish',
                                level=sp.price,
                                break_candle_index=i,
                                penetration=round(penetration, 5),
                                penetration_atr=round(penetration / atr, 2),
                                timestamp=timestamp_str,
                                swing_index=sp.index
                            )
                            turtle_soups.append(ts)
                            break

                elif sp.type == 'low':
                    # Bullish Turtle Soup: Breaks low, fails, reverses up
                    penetration = sp.price - candle['low']

                    if penetration > 0 and penetration <= atr * max_penetration_atr:
                        if candle['close'] > sp.price:
                            ts = TurtleSoup(
                                type='bullish',
                                level=sp.price,
                                break_candle_index=i,
                                penetration=round(penetration, 5),
                                penetration_atr=round(penetration / atr, 2),
                                timestamp=timestamp_str,
                                swing_index=sp.index
                            )
                            turtle_soups.append(ts)
                            break

        return turtle_soups


# =========================================================================
# NEW DATACLASSES FOR ADVANCED PATTERNS
# =========================================================================

@dataclass
class LiquiditySweep:
    """A completed liquidity sweep (grab + reversal)"""
    type: str  # "bullish" (swept lows, reversed up) or "bearish" (swept highs, reversed down)
    sweep_level: float  # The swing high/low that was swept
    sweep_candle_index: int
    sweep_low: float  # Candle low
    sweep_high: float  # Candle high
    close_price: float  # Where candle closed
    rejection_strength: float  # Wick to range ratio (0-1)
    atr_penetration: float  # How far past the level in ATR multiples
    timestamp: str
    session: Optional[str] = None
    swing_index: int = 0  # Index of the swing that was swept

    @property
    def is_strong(self) -> bool:
        """Strong sweep has high rejection and penetration"""
        return self.rejection_strength >= 0.5 and self.atr_penetration >= 0.2

    @property
    def signal_direction(self) -> str:
        """Direction to trade after sweep"""
        return "BUY" if self.type == "bullish" else "SELL"


@dataclass
class Inducement:
    """Inducement pattern - false breakout to trap traders"""
    type: str  # "bullish" (induces shorts then rallies) or "bearish"
    inducement_level: float  # The level used to trap traders
    inducement_index: int  # Candle index of the inducement swing
    break_index: int  # When level was broken
    reversal_index: int  # When reversal occurred
    target_liquidity: Optional[float]  # The major liquidity target
    timestamp: str
    trapped_direction: str  # "long" or "short"

    @property
    def candles_to_reversal(self) -> int:
        """How many candles from break to reversal"""
        return self.reversal_index - self.break_index


@dataclass
class RejectionBlock:
    """Strong rejection candle marking support/resistance"""
    type: str  # "bullish" (rejection from below) or "bearish" (from above)
    rejection_price: float  # The wick extreme
    body_top: float
    body_bottom: float
    wick_size: float
    wick_atr_ratio: float  # Wick size in ATR multiples
    candle_index: int
    timestamp: str
    session: Optional[str] = None
    held: bool = True  # Did the level hold on next candle
    mitigated: bool = False
    mitigation_index: Optional[int] = None

    @property
    def zone_top(self) -> float:
        """Top of rejection zone (use for S/R)"""
        return self.body_top if self.type == 'bullish' else self.rejection_price

    @property
    def zone_bottom(self) -> float:
        """Bottom of rejection zone"""
        return self.rejection_price if self.type == 'bullish' else self.body_bottom


@dataclass
class TurtleSoup:
    """Turtle Soup - failed breakout pattern"""
    type: str  # "bullish" (failed breakdown) or "bearish" (failed breakout)
    level: float  # The level that was broken
    break_candle_index: int
    penetration: float  # How far price went past the level
    penetration_atr: float  # Penetration in ATR multiples
    timestamp: str
    swing_index: int  # The swing that was broken

    @property
    def trade_direction(self) -> str:
        """Direction to trade"""
        return "BUY" if self.type == "bullish" else "SELL"
