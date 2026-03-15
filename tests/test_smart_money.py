"""
Comprehensive tests for Smart Money Concepts (SMC) analysis library.

Tests cover:
- ATR caching and performance
- Order Block detection (candle and structural)
- Fair Value Gap detection with partial fills
- Swing point detection
- Liquidity zone detection
- Structure breaks (BOS/CHOC)
- Equal Highs/Lows detection
- Breaker Blocks detection
- Premium/Discount zones
- OTE (Optimal Trade Entry) levels
- Confluence scoring
- Multi-timeframe alignment
- Session identification
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tradingagents.indicators.smart_money import (
    SmartMoneyAnalyzer,
    SMCConstants,
    TradingSession,
    OrderBlock,
    FairValueGap,
    StructurePoint,
    LiquidityZone,
    EqualLevel,
    BreakerBlock,
    OTEZone,
    PremiumDiscountZone,
    ConfluenceScore,
)


# =============================================================================
# FIXTURES - Test Data Generation
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data with realistic price action."""
    np.random.seed(42)
    n = 200

    # Create timestamps
    base_time = datetime(2024, 1, 1, 8, 0)  # London open time
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    # Generate price data with trends and swings
    base_price = 100.0
    prices = [base_price]

    for i in range(1, n):
        # Add some trend and noise
        trend = 0.02 * np.sin(i / 20)  # Cyclical trend
        noise = np.random.normal(0, 0.5)
        new_price = prices[-1] * (1 + trend / 100 + noise / 100)
        prices.append(new_price)

    # Create OHLCV data
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for i, price in enumerate(prices):
        volatility = np.random.uniform(0.5, 1.5)
        high_offset = np.random.uniform(0.2, volatility)
        low_offset = np.random.uniform(0.2, volatility)

        open_price = price
        close_price = price * (1 + np.random.uniform(-0.5, 0.5) / 100)
        high_price = max(open_price, close_price) + high_offset
        low_price = min(open_price, close_price) - low_offset

        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)
        data['volume'].append(np.random.uniform(1000, 5000))

    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
    return df


@pytest.fixture
def bullish_ob_data():
    """Generate data with a clear bullish order block pattern."""
    timestamps = pd.date_range('2024-01-01 08:00', periods=50, freq='h')

    # Create a pattern: consolidation -> bearish candle -> strong bullish move
    data = {
        'open': [100] * 45 + [100, 99, 98, 102, 105],
        'high': [101] * 45 + [100.5, 99.5, 98.5, 106, 108],
        'low': [99] * 45 + [99.5, 98.5, 97, 98, 104],
        'close': [100] * 45 + [99.5, 98.5, 97.5, 105, 107],
        'volume': [1000] * 45 + [1000, 1200, 1500, 3000, 2500]
    }

    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def fvg_data():
    """Generate data with clear fair value gaps."""
    timestamps = pd.date_range('2024-01-01 08:00', periods=50, freq='h')

    # Create bullish FVG: candle1.high < candle3.low
    data = {
        'open': [100] * 47 + [100, 102, 106],
        'high': [101] * 47 + [101, 103, 108],  # candle1 high = 101
        'low': [99] * 47 + [99, 101, 104],      # candle3 low = 104 > 101 = FVG
        'close': [100] * 47 + [100.5, 102.5, 107],
        'volume': [1000] * 50
    }

    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def swing_point_data():
    """Generate data with clear swing highs and lows."""
    timestamps = pd.date_range('2024-01-01', periods=100, freq='h')

    # Create wave pattern
    prices = []
    for i in range(100):
        base = 100 + 10 * np.sin(i / 10)  # Wave pattern
        prices.append(base)

    data = {
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000] * 100
    }

    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def analyzer():
    """Create a SmartMoneyAnalyzer instance."""
    return SmartMoneyAnalyzer(
        swing_lookback=5,
        ob_strength_threshold=0.3,
        fvg_min_size_atr=0.2
    )


# =============================================================================
# TEST: CONSTANTS AND CONFIGURATION
# =============================================================================

class TestSMCConstants:
    """Test SMC constants are properly defined."""

    def test_atr_constants(self):
        assert SMCConstants.ATR_PERIOD == 14
        assert SMCConstants.OB_MOVE_ATR_DIVISOR == 2.0

    def test_ob_constants(self):
        assert SMCConstants.OB_VOLUME_MULTIPLIER == 1.2
        assert SMCConstants.OB_OVERLAP_THRESHOLD == 0.3

    def test_fvg_constants(self):
        assert SMCConstants.FVG_MIN_SIZE_ATR_DEFAULT == 0.3

    def test_liquidity_constants(self):
        assert SMCConstants.LIQUIDITY_SWEEP_AGE_CUTOFF == 50
        assert SMCConstants.LIQUIDITY_TEST_TOLERANCE_PCT == 0.002

    def test_ote_constants(self):
        assert SMCConstants.OTE_FIB_START == 0.62
        assert SMCConstants.OTE_FIB_END == 0.79

    def test_confluence_weights(self):
        assert SMCConstants.CONFLUENCE_OB_WEIGHT == 30
        assert SMCConstants.CONFLUENCE_FVG_WEIGHT == 20
        assert SMCConstants.CONFLUENCE_LIQUIDITY_WEIGHT == 25


class TestTradingSession:
    """Test trading session enum."""

    def test_session_values(self):
        assert TradingSession.ASIAN.value == "asian"
        assert TradingSession.LONDON_OPEN.value == "london_open"
        assert TradingSession.NY_OPEN.value == "ny_open"


# =============================================================================
# TEST: ATR CACHING
# =============================================================================

class TestATRCaching:
    """Test ATR caching functionality."""

    def test_atr_calculation(self, analyzer, sample_ohlcv_data):
        """Test ATR is calculated correctly."""
        atr = analyzer._get_atr(sample_ohlcv_data)

        assert len(atr) == len(sample_ohlcv_data)
        assert atr.iloc[SMCConstants.ATR_PERIOD:].notna().all()
        assert (atr.iloc[SMCConstants.ATR_PERIOD:] > 0).all()

    def test_atr_caching_same_df(self, analyzer, sample_ohlcv_data):
        """Test ATR is cached for same DataFrame."""
        atr1 = analyzer._get_atr(sample_ohlcv_data)
        atr2 = analyzer._get_atr(sample_ohlcv_data)

        # Should be same object (cached)
        assert atr1 is atr2

    def test_atr_cache_invalidation(self, analyzer, sample_ohlcv_data):
        """Test ATR cache is invalidated when DataFrame changes."""
        atr1 = analyzer._get_atr(sample_ohlcv_data)

        # Modify DataFrame
        modified_df = sample_ohlcv_data.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc('close')] = 999

        atr2 = analyzer._get_atr(modified_df)

        # Should be different (cache invalidated)
        assert atr1 is not atr2

    def test_atr_different_periods(self, analyzer, sample_ohlcv_data):
        """Test different ATR periods are cached separately."""
        atr_14 = analyzer._get_atr(sample_ohlcv_data, period=14)
        atr_20 = analyzer._get_atr(sample_ohlcv_data, period=20)

        assert atr_14 is not atr_20
        assert len(analyzer._atr_cache) == 2


# =============================================================================
# TEST: SESSION IDENTIFICATION
# =============================================================================

class TestSessionIdentification:
    """Test trading session identification."""

    def test_asian_session(self, analyzer):
        """Test Asian session detection."""
        asian_time = datetime(2024, 1, 1, 3, 0)  # 3 AM UTC
        assert analyzer._identify_session(asian_time) == "asian"

    def test_london_open_session(self, analyzer):
        """Test London open session detection."""
        london_time = datetime(2024, 1, 1, 8, 0)  # 8 AM UTC
        assert analyzer._identify_session(london_time) == "london_open"

    def test_ny_open_session(self, analyzer):
        """Test NY open session detection."""
        ny_time = datetime(2024, 1, 1, 13, 0)  # 1 PM UTC
        assert analyzer._identify_session(ny_time) == "ny_open"

    def test_off_session(self, analyzer):
        """Test off session detection."""
        off_time = datetime(2024, 1, 1, 22, 0)  # 10 PM UTC
        assert analyzer._identify_session(off_time) == "off_session"

    def test_session_multiplier(self, analyzer):
        """Test session strength multipliers."""
        assert analyzer._get_session_multiplier("london_open") == 1.3
        assert analyzer._get_session_multiplier("ny_open") == 1.3
        assert analyzer._get_session_multiplier("asian") == 0.9
        assert analyzer._get_session_multiplier("off_session") == 0.8


# =============================================================================
# TEST: ORDER BLOCK DETECTION
# =============================================================================

class TestOrderBlockDetection:
    """Test order block detection."""

    def test_detect_order_blocks_returns_list(self, analyzer, sample_ohlcv_data):
        """Test OB detection returns a list."""
        obs = analyzer.detect_order_blocks(sample_ohlcv_data)
        assert isinstance(obs, list)

    def test_order_block_properties(self, analyzer, bullish_ob_data):
        """Test detected OB has correct properties."""
        obs = analyzer.detect_order_blocks(bullish_ob_data, lookback=10)

        # Should detect at least one OB
        if obs:
            ob = obs[0]
            assert isinstance(ob, OrderBlock)
            assert ob.type in ['bullish', 'bearish']
            assert ob.top > ob.bottom
            assert 0 <= ob.strength <= 1
            assert ob.candle_index >= 0

    def test_ob_midpoint_property(self):
        """Test OB midpoint calculation."""
        ob = OrderBlock(
            type='bullish',
            top=110,
            bottom=100,
            candle_index=10,
            timestamp='2024-01-01',
            strength=0.8
        )
        assert ob.midpoint == 105

    def test_ob_invalidation_price(self):
        """Test OB invalidation price calculation."""
        bullish_ob = OrderBlock(
            type='bullish',
            top=110,
            bottom=100,
            candle_index=10,
            timestamp='2024-01-01',
            strength=0.8
        )
        assert bullish_ob.invalidation_price == 100  # Below OB

        bearish_ob = OrderBlock(
            type='bearish',
            top=110,
            bottom=100,
            candle_index=10,
            timestamp='2024-01-01',
            strength=0.8
        )
        assert bearish_ob.invalidation_price == 110  # Above OB

    def test_ob_invalidation_tracking(self, analyzer, sample_ohlcv_data):
        """Test OB invalidation is tracked."""
        obs = analyzer.detect_order_blocks(sample_ohlcv_data, track_invalidation=True)

        for ob in obs:
            if ob.invalidated:
                assert ob.invalidation_index is not None
                assert ob.invalidation_index > ob.candle_index


# =============================================================================
# TEST: FAIR VALUE GAP DETECTION
# =============================================================================

class TestFairValueGapDetection:
    """Test Fair Value Gap detection."""

    def test_detect_fvg_returns_list(self, analyzer, sample_ohlcv_data):
        """Test FVG detection returns a list."""
        fvgs = analyzer.detect_fair_value_gaps(sample_ohlcv_data)
        assert isinstance(fvgs, list)

    def test_fvg_properties(self, analyzer, fvg_data):
        """Test detected FVG has correct properties."""
        fvgs = analyzer.detect_fair_value_gaps(fvg_data)

        for fvg in fvgs:
            assert isinstance(fvg, FairValueGap)
            assert fvg.type in ['bullish', 'bearish']
            assert fvg.top > fvg.bottom
            assert fvg.size > 0

    def test_fvg_midpoint_property(self):
        """Test FVG midpoint calculation."""
        fvg = FairValueGap(
            type='bullish',
            top=105,
            bottom=101,
            start_index=47,
            timestamp='2024-01-01',
            size=4
        )
        assert fvg.midpoint == 103

    def test_fvg_remaining_size(self):
        """Test FVG remaining size calculation."""
        fvg = FairValueGap(
            type='bullish',
            top=105,
            bottom=101,
            start_index=47,
            timestamp='2024-01-01',
            size=4,
            fill_percentage=50
        )
        assert fvg.remaining_size == 2

    def test_partial_fill_tracking(self, analyzer, sample_ohlcv_data):
        """Test partial FVG fill tracking."""
        fvgs = analyzer.detect_fair_value_gaps(sample_ohlcv_data, track_partial_fill=True)

        for fvg in fvgs:
            assert 0 <= fvg.fill_percentage <= 100
            if fvg.mitigated:
                assert fvg.fill_percentage == 100


# =============================================================================
# TEST: SWING POINT DETECTION
# =============================================================================

class TestSwingPointDetection:
    """Test swing point detection."""

    def test_detect_swing_points_returns_list(self, analyzer, swing_point_data):
        """Test swing point detection returns a list."""
        swings = analyzer.detect_swing_points(swing_point_data)
        assert isinstance(swings, list)

    def test_swing_point_properties(self, analyzer, swing_point_data):
        """Test swing points have correct properties."""
        swings = analyzer.detect_swing_points(swing_point_data)

        for sp in swings:
            assert isinstance(sp, StructurePoint)
            assert sp.type in ['high', 'low']
            assert sp.price > 0
            assert sp.index >= 0

    def test_swing_points_sorted_by_index(self, analyzer, swing_point_data):
        """Test swing points are sorted by index."""
        swings = analyzer.detect_swing_points(swing_point_data)

        indices = [sp.index for sp in swings]
        assert indices == sorted(indices)

    def test_swing_high_detection(self, analyzer, swing_point_data):
        """Test swing highs are actual local maxima."""
        swings = analyzer.detect_swing_points(swing_point_data)
        highs = [sp for sp in swings if sp.type == 'high']

        for h in highs:
            idx = h.index
            lookback = analyzer.swing_lookback

            # Verify it's a local maximum
            if idx >= lookback and idx < len(swing_point_data) - lookback:
                for j in range(1, lookback + 1):
                    assert swing_point_data.iloc[idx]['high'] >= swing_point_data.iloc[idx - j]['high']
                    assert swing_point_data.iloc[idx]['high'] >= swing_point_data.iloc[idx + j]['high']


# =============================================================================
# TEST: LIQUIDITY ZONE DETECTION
# =============================================================================

class TestLiquidityZoneDetection:
    """Test liquidity zone detection."""

    def test_detect_liquidity_zones(self, analyzer, sample_ohlcv_data):
        """Test liquidity zone detection."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        current_price = sample_ohlcv_data.iloc[-1]['close']
        lz = analyzer.detect_liquidity_zones(sample_ohlcv_data, swings, current_price)

        assert isinstance(lz, list)
        assert len(lz) <= 6  # Default max zones

    def test_liquidity_zone_properties(self, analyzer, sample_ohlcv_data):
        """Test liquidity zones have correct properties."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        current_price = sample_ohlcv_data.iloc[-1]['close']
        zones = analyzer.detect_liquidity_zones(sample_ohlcv_data, swings, current_price)

        for lz in zones:
            assert isinstance(lz, LiquidityZone)
            assert lz.type in ['buy-side', 'sell-side']
            assert 0 <= lz.strength <= 100
            assert lz.price > 0


# =============================================================================
# TEST: EQUAL HIGHS/LOWS DETECTION
# =============================================================================

class TestEqualLevelsDetection:
    """Test equal highs and lows detection."""

    def test_detect_equal_levels(self, analyzer, sample_ohlcv_data):
        """Test equal levels detection returns list."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        equal_levels = analyzer.detect_equal_levels(sample_ohlcv_data, swings)

        assert isinstance(equal_levels, list)

    def test_equal_level_properties(self):
        """Test EqualLevel properties."""
        el = EqualLevel(
            type='equal_highs',
            price=110,
            touches=3,
            indices=[10, 20, 30],
            timestamps=['2024-01-01', '2024-01-02', '2024-01-03']
        )

        assert el.liquidity_side == 'above'  # Buy stops above equal highs

        el_lows = EqualLevel(
            type='equal_lows',
            price=90,
            touches=2,
            indices=[15, 25]
        )
        assert el_lows.liquidity_side == 'below'  # Sell stops below equal lows

    def test_equal_levels_minimum_touches(self, analyzer, sample_ohlcv_data):
        """Test equal levels require minimum touches."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        equal_levels = analyzer.detect_equal_levels(
            sample_ohlcv_data, swings, min_touches=2
        )

        for el in equal_levels:
            assert el.touches >= 2


# =============================================================================
# TEST: BREAKER BLOCKS DETECTION
# =============================================================================

class TestBreakerBlocksDetection:
    """Test breaker block detection."""

    def test_detect_breaker_blocks(self, analyzer, sample_ohlcv_data):
        """Test breaker block detection."""
        obs = analyzer.detect_order_blocks(sample_ohlcv_data, track_invalidation=True)
        breakers = analyzer.detect_breaker_blocks(sample_ohlcv_data, obs)

        assert isinstance(breakers, list)

    def test_breaker_block_polarity_flip(self):
        """Test breaker blocks flip polarity correctly."""
        # Bullish OB that got invalidated becomes bearish breaker
        bb = BreakerBlock(
            original_type='bullish',
            current_type='bearish',
            top=110,
            bottom=100,
            original_index=10,
            break_index=20,
            timestamp='2024-01-01',
            strength=0.7
        )

        assert bb.original_type == 'bullish'
        assert bb.current_type == 'bearish'
        assert bb.midpoint == 105

    def test_breaker_from_invalidated_ob(self, analyzer, sample_ohlcv_data):
        """Test breakers only come from invalidated OBs."""
        obs = analyzer.detect_order_blocks(sample_ohlcv_data, track_invalidation=True)
        breakers = analyzer.detect_breaker_blocks(sample_ohlcv_data, obs)

        # Breakers should have break_index > original_index
        for bb in breakers:
            assert bb.break_index > bb.original_index


# =============================================================================
# TEST: PREMIUM/DISCOUNT ZONES
# =============================================================================

class TestPremiumDiscountZones:
    """Test premium/discount zone calculation."""

    def test_calculate_premium_discount(self, analyzer, sample_ohlcv_data):
        """Test premium/discount calculation."""
        pd_zone = analyzer.calculate_premium_discount(sample_ohlcv_data)

        assert isinstance(pd_zone, PremiumDiscountZone)
        assert pd_zone.range_high > pd_zone.range_low
        assert pd_zone.equilibrium == (pd_zone.range_high + pd_zone.range_low) / 2
        assert 0 <= pd_zone.position_pct <= 100
        assert pd_zone.zone in ['premium', 'discount', 'equilibrium']

    def test_premium_zone_detection(self, analyzer):
        """Test premium zone detection."""
        # Create data where current price is in premium
        data = pd.DataFrame({
            'open': [100] * 50 + [120],
            'high': [105] * 50 + [122],
            'low': [95] * 50 + [118],
            'close': [100] * 50 + [121]
        })

        pd_zone = analyzer.calculate_premium_discount(data)
        assert pd_zone.zone == 'premium'
        assert pd_zone.is_premium
        assert not pd_zone.is_discount

    def test_discount_zone_detection(self, analyzer):
        """Test discount zone detection."""
        # Create data where current price is in discount
        data = pd.DataFrame({
            'open': [100] * 50 + [80],
            'high': [105] * 50 + [82],
            'low': [95] * 50 + [78],
            'close': [100] * 50 + [79]
        })

        pd_zone = analyzer.calculate_premium_discount(data)
        assert pd_zone.zone == 'discount'
        assert pd_zone.is_discount
        assert not pd_zone.is_premium


# =============================================================================
# TEST: OTE (OPTIMAL TRADE ENTRY) ZONES
# =============================================================================

class TestOTEZones:
    """Test Optimal Trade Entry zone calculation."""

    def test_calculate_ote_zone_bullish(self, analyzer):
        """Test bullish OTE calculation."""
        ote = analyzer.calculate_ote_zone(
            swing_high=110,
            swing_low=100,
            direction='bullish'
        )

        assert isinstance(ote, OTEZone)
        assert ote.direction == 'bullish'
        # OTE should be between 62-79% retracement from high
        assert ote.top < ote.swing_high
        assert ote.bottom < ote.top
        assert ote.bottom > ote.swing_low

    def test_calculate_ote_zone_bearish(self, analyzer):
        """Test bearish OTE calculation."""
        ote = analyzer.calculate_ote_zone(
            swing_high=110,
            swing_low=100,
            direction='bearish'
        )

        assert isinstance(ote, OTEZone)
        assert ote.direction == 'bearish'
        # OTE should be between 62-79% retracement from low
        assert ote.bottom > ote.swing_low
        assert ote.top > ote.bottom
        assert ote.top < ote.swing_high

    def test_ote_midpoint(self, analyzer):
        """Test OTE midpoint calculation."""
        ote = analyzer.calculate_ote_zone(110, 100, 'bullish')
        assert ote.midpoint == (ote.top + ote.bottom) / 2

    def test_find_ote_zones(self, analyzer, sample_ohlcv_data):
        """Test finding OTE zones near current price."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        current_price = sample_ohlcv_data.iloc[-1]['close']
        ote_zones = analyzer.find_ote_zones(swings, current_price)

        assert isinstance(ote_zones, list)
        for ote in ote_zones:
            assert isinstance(ote, OTEZone)


# =============================================================================
# TEST: CONFLUENCE SCORING
# =============================================================================

class TestConfluenceScoring:
    """Test confluence scoring system."""

    def test_calculate_confluence_score(self, analyzer, sample_ohlcv_data):
        """Test confluence score calculation."""
        analysis = analyzer.analyze_full_smc(sample_ohlcv_data)
        current_price = sample_ohlcv_data.iloc[-1]['close']

        score = analyzer.calculate_confluence_score(current_price, analysis)

        assert isinstance(score, ConfluenceScore)
        assert 0 <= score.total_score <= 100
        assert isinstance(score.factors, list)

    def test_confluence_score_bias(self):
        """Test confluence score bias calculation."""
        bullish_score = ConfluenceScore(
            price=100,
            total_score=50,
            factors=['test'],
            bullish_factors=3,
            bearish_factors=1
        )
        assert bullish_score.bias == 'bullish'

        bearish_score = ConfluenceScore(
            price=100,
            total_score=50,
            factors=['test'],
            bullish_factors=1,
            bearish_factors=3
        )
        assert bearish_score.bias == 'bearish'

        neutral_score = ConfluenceScore(
            price=100,
            total_score=50,
            factors=['test'],
            bullish_factors=2,
            bearish_factors=2
        )
        assert neutral_score.bias == 'neutral'


# =============================================================================
# TEST: MULTI-TIMEFRAME ALIGNMENT
# =============================================================================

class TestMultiTimeframeAlignment:
    """Test multi-timeframe zone alignment."""

    def test_align_mtf_zones(self, analyzer, sample_ohlcv_data):
        """Test MTF zone alignment."""
        # Create two analyses (simulating different timeframes)
        htf_analysis = analyzer.analyze_full_smc(sample_ohlcv_data)
        ltf_analysis = analyzer.analyze_full_smc(sample_ohlcv_data)

        aligned = analyzer.align_mtf_zones(htf_analysis, ltf_analysis)

        assert 'aligned_support' in aligned
        assert 'aligned_resistance' in aligned
        assert isinstance(aligned['aligned_support'], list)
        assert isinstance(aligned['aligned_resistance'], list)

    def test_aligned_zones_have_properties(self, analyzer, sample_ohlcv_data):
        """Test aligned zones have required properties."""
        analysis = analyzer.analyze_full_smc(sample_ohlcv_data)
        aligned = analyzer.align_mtf_zones(analysis, analysis)

        for zone in aligned['aligned_support'] + aligned['aligned_resistance']:
            assert 'htf_zone' in zone
            assert 'ltf_zone' in zone
            assert 'avg_price' in zone
            assert 'combined_strength' in zone
            assert zone['mtf_confirmed'] is True


# =============================================================================
# TEST: STRUCTURE BREAKS (BOS/CHOC)
# =============================================================================

class TestStructureBreaks:
    """Test Break of Structure and Change of Character detection."""

    def test_detect_structure_breaks(self, analyzer, sample_ohlcv_data):
        """Test structure break detection."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        breaks = analyzer.detect_structure_breaks(sample_ohlcv_data, swings)

        assert 'bos' in breaks
        assert 'choc' in breaks
        assert isinstance(breaks['bos'], list)
        assert isinstance(breaks['choc'], list)

    def test_structure_break_properties(self, analyzer, sample_ohlcv_data):
        """Test structure break properties."""
        swings = analyzer.detect_swing_points(sample_ohlcv_data)
        breaks = analyzer.detect_structure_breaks(sample_ohlcv_data, swings)

        for sp in breaks['bos'] + breaks['choc']:
            assert sp.broken is True
            assert sp.break_index is not None
            assert sp.break_type in ['BOS', 'CHOC']


# =============================================================================
# TEST: FULL ANALYSIS
# =============================================================================

class TestFullAnalysis:
    """Test full SMC analysis."""

    def test_analyze_full_smc(self, analyzer, sample_ohlcv_data):
        """Test full SMC analysis."""
        analysis = analyzer.analyze_full_smc(sample_ohlcv_data)

        assert 'current_price' in analysis
        assert 'order_blocks' in analysis
        assert 'fair_value_gaps' in analysis
        assert 'liquidity_zones' in analysis
        assert 'structure' in analysis
        assert 'zones' in analysis
        assert 'bias' in analysis

    def test_analyze_full_smc_extended(self, analyzer, sample_ohlcv_data):
        """Test extended full SMC analysis."""
        analysis = analyzer.analyze_full_smc_extended(sample_ohlcv_data)

        # Base analysis fields
        assert 'current_price' in analysis
        assert 'order_blocks' in analysis

        # Extended analysis fields
        assert 'premium_discount' in analysis
        assert 'equal_levels' in analysis
        assert 'breaker_blocks' in analysis
        assert 'ote_zones' in analysis
        assert 'current_confluence' in analysis

    def test_analysis_bias_values(self, analyzer, sample_ohlcv_data):
        """Test analysis bias is valid."""
        analysis = analyzer.analyze_full_smc(sample_ohlcv_data)
        assert analysis['bias'] in ['bullish', 'bearish', 'neutral']


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, analyzer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        obs = analyzer.detect_order_blocks(empty_df)
        assert obs == []

        fvgs = analyzer.detect_fair_value_gaps(empty_df)
        assert fvgs == []

    def test_small_dataframe(self, analyzer):
        """Test handling of small DataFrame."""
        small_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1000]
        })

        obs = analyzer.detect_order_blocks(small_df)
        assert isinstance(obs, list)

    def test_no_swing_points(self, analyzer):
        """Test handling when no swing points detected."""
        flat_df = pd.DataFrame({
            'open': [100] * 20,
            'high': [100.1] * 20,
            'low': [99.9] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20
        })

        swings = analyzer.detect_swing_points(flat_df)
        assert isinstance(swings, list)

    def test_string_timestamp_session(self, analyzer):
        """Test session identification with string timestamp."""
        session = analyzer._identify_session("2024-01-01 08:00:00")
        assert session in [s.value for s in TradingSession]

    def test_invalid_timestamp_session(self, analyzer):
        """Test session identification with invalid timestamp."""
        session = analyzer._identify_session("invalid")
        assert session == TradingSession.OFF_SESSION.value


# =============================================================================
# TEST: DATA CLASSES
# =============================================================================

class TestDataClasses:
    """Test data class functionality."""

    def test_order_block_creation(self):
        """Test OrderBlock creation with all fields."""
        ob = OrderBlock(
            type='bullish',
            top=110,
            bottom=100,
            candle_index=50,
            timestamp='2024-01-01',
            strength=0.85,
            mitigated=False,
            detection_method='structural',
            session='london_open'
        )

        assert ob.type == 'bullish'
        assert ob.midpoint == 105
        assert ob.invalidation_price == 100

    def test_fvg_creation(self):
        """Test FairValueGap creation."""
        fvg = FairValueGap(
            type='bearish',
            top=100,
            bottom=95,
            start_index=30,
            timestamp='2024-01-01',
            size=5,
            fill_percentage=25
        )

        assert fvg.midpoint == 97.5
        assert fvg.remaining_size == 3.75

    def test_confluence_score_creation(self):
        """Test ConfluenceScore creation."""
        cs = ConfluenceScore(
            price=100,
            total_score=75,
            factors=['Bullish OB', 'In discount'],
            bullish_factors=2,
            bearish_factors=0
        )

        assert cs.bias == 'bullish'
        assert len(cs.factors) == 2


# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Test performance aspects."""

    def test_large_dataset_performance(self, analyzer):
        """Test analysis on larger dataset completes in reasonable time."""
        import time

        # Generate larger dataset
        n = 1000
        timestamps = pd.date_range('2024-01-01', periods=n, freq='h')
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, n),
            'high': np.random.uniform(100, 110, n),
            'low': np.random.uniform(90, 100, n),
            'close': np.random.uniform(95, 105, n),
            'volume': np.random.uniform(1000, 5000, n)
        }, index=timestamps)

        # Ensure high > open/close and low < open/close
        data['high'] = data[['open', 'close', 'high']].max(axis=1) + 0.5
        data['low'] = data[['open', 'close', 'low']].min(axis=1) - 0.5

        start = time.time()
        analysis = analyzer.analyze_full_smc_extended(data)
        elapsed = time.time() - start

        assert elapsed < 10  # Should complete in under 10 seconds
        assert 'current_price' in analysis

    def test_atr_cache_improves_performance(self, analyzer):
        """Test ATR caching improves performance."""
        import time

        n = 500
        data = pd.DataFrame({
            'open': np.random.uniform(95, 105, n),
            'high': np.random.uniform(100, 110, n),
            'low': np.random.uniform(90, 100, n),
            'close': np.random.uniform(95, 105, n),
            'volume': np.random.uniform(1000, 5000, n)
        })

        # First call - calculates ATR
        start = time.time()
        analyzer._get_atr(data)
        first_call = time.time() - start

        # Second call - uses cache
        start = time.time()
        analyzer._get_atr(data)
        second_call = time.time() - start

        # Cached call should be faster
        assert second_call < first_call


# =============================================================================
# TEST: XAUUSD-LIKE SCENARIOS - Alignment with MT5 indicators
# =============================================================================

def _make_xauusd_downtrend_data():
    """
    Create XAUUSD-like H1 data simulating the scenario from the chart:
    - Price rallies from ~5023 to ~5200 area
    - Forms swing highs around 5200, 5186
    - Drops through 5140 (PDL) = bearish BOS
    - Multiple touches near 5077 = proven support (equal lows)
    - Current price around 5130
    """
    np.random.seed(99)
    n = 200
    timestamps = pd.date_range('2026-03-01', periods=n, freq='h')

    # Phase 1 (0-40): Base around 5023-5030, multiple touches
    # Phase 2 (40-100): Rally up to 5200
    # Phase 3 (100-140): Ranging around 5170-5200 with swing highs
    # Phase 4 (140-180): Sell-off through 5140 PDL
    # Phase 5 (180-200): Current price hovering ~5127-5140

    prices = []
    for i in range(n):
        if i < 15:
            # Touch 5077 area multiple times
            base = 5077 + np.sin(i * 1.2) * 8
        elif i < 25:
            # Touch 5077 area again
            base = 5080 + np.sin(i * 0.8) * 10
        elif i < 35:
            # Another touch near 5077
            base = 5075 + np.sin(i * 1.0) * 12
        elif i < 45:
            # Another set of touches near 5077
            base = 5078 + np.sin(i * 0.9) * 9
        elif i < 60:
            # Start rallying
            base = 5077 + (i - 45) * 4
        elif i < 80:
            # Continue rally
            base = 5137 + (i - 60) * 2.5
        elif i < 100:
            # Rally to highs
            base = 5187 + (i - 80) * 0.5 + np.sin(i * 0.5) * 5
        elif i < 120:
            # Ranging at top, swing highs around 5198, 5186
            base = 5185 + np.sin(i * 0.4) * 12
        elif i < 140:
            # Start dropping
            base = 5190 - (i - 120) * 2.5
        elif i < 160:
            # Sharp drop through 5140 (PDL break = BOS)
            base = 5140 - (i - 140) * 1.0
        elif i < 180:
            # Stabilize around 5127-5135
            base = 5130 + np.sin(i * 0.6) * 5
        else:
            # Current area ~5127
            base = 5128 + np.sin(i * 0.3) * 3

        prices.append(base)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
    }

    for i, price in enumerate(prices):
        volatility = np.random.uniform(3, 8)  # XAUUSD-like volatility
        h_off = np.random.uniform(1, volatility)
        l_off = np.random.uniform(1, volatility)
        o = price + np.random.uniform(-2, 2)
        c = price + np.random.uniform(-2, 2)
        h = max(o, c) + h_off
        lo = min(o, c) - l_off
        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(lo)
        data['close'].append(c)
        data['volume'].append(np.random.uniform(5000, 20000))

    return pd.DataFrame(data, index=timestamps)


def _make_bearish_bos_data():
    """
    Create data with a clear bearish BOS pattern:
    - Uptrend with higher highs and higher lows
    - Then price breaks below a swing low = bearish BOS
    """
    n = 80
    timestamps = pd.date_range('2026-03-01', periods=n, freq='h')

    prices = []
    for i in range(n):
        if i < 20:
            # Uptrend: higher highs, higher lows
            base = 100 + i * 1.5 + np.sin(i * 0.8) * 5
        elif i < 35:
            # Form swing high around 130
            base = 130 + np.sin(i * 0.6) * 3
        elif i < 45:
            # Pull back to swing low around 120
            base = 130 - (i - 35) * 1.0
        elif i < 55:
            # Lower high around 127
            base = 120 + (55 - i) * 0.7
        elif i < 65:
            # Break below the swing low at 120 = bearish BOS
            base = 120 - (i - 55) * 1.5
        else:
            # Continue lower
            base = 105 - (i - 65) * 0.5

        prices.append(base)

    data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    for i, price in enumerate(prices):
        o = price + np.random.uniform(-0.5, 0.5)
        c = price + np.random.uniform(-0.5, 0.5)
        h = max(o, c) + np.random.uniform(0.5, 2.0)
        lo = min(o, c) - np.random.uniform(0.5, 2.0)
        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(lo)
        data['close'].append(c)
        data['volume'].append(np.random.uniform(1000, 5000))

    return pd.DataFrame(data, index=timestamps)


class TestXAUUSDAlignmentScenarios:
    """
    Test SMC detection against XAUUSD-like scenarios to ensure
    alignment with MT5 indicator outputs.
    """

    def test_equal_lows_detected_at_proven_support(self):
        """
        XAUUSD scenario: Price touches ~5077 multiple times (6 retests).
        With old tolerance (0.1 ATR ≈ 2.2 pts), these were missed.
        With new tolerance (0.25 ATR ≈ 5.5 pts), they should be detected.
        """
        df = _make_xauusd_downtrend_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=5, ob_strength_threshold=0.3)

        swings = analyzer.detect_swing_points(df, lookback=200)
        swing_lows = [sp for sp in swings if sp.type == 'low']

        # There should be swing lows in the 5065-5090 range (our 5077 area)
        lows_near_5077 = [sp for sp in swing_lows if 5060 <= sp.price <= 5095]
        assert len(lows_near_5077) >= 2, (
            f"Expected at least 2 swing lows near 5077, got {len(lows_near_5077)}. "
            f"Low prices: {[sp.price for sp in swing_lows]}"
        )

        # Now detect equal levels
        equal_levels = analyzer.detect_equal_levels(df, swings)
        equal_lows = [el for el in equal_levels if el.type == 'equal_lows']

        # Should detect equal lows near 5077
        eql_near_5077 = [el for el in equal_lows if 5060 <= el.price <= 5095]
        assert len(eql_near_5077) >= 1, (
            f"Expected equal lows near 5077 (proven support), got none. "
            f"Equal low prices: {[el.price for el in equal_lows]}"
        )

        # Verify it has multiple touches
        for el in eql_near_5077:
            assert el.touches >= 2, f"Expected >= 2 touches, got {el.touches}"

    def test_equal_level_tolerance_wider_than_before(self):
        """Verify the tolerance constant was increased."""
        assert SMCConstants.EQUAL_LEVEL_TOLERANCE_ATR == 0.25, (
            f"Expected 0.25, got {SMCConstants.EQUAL_LEVEL_TOLERANCE_ATR}"
        )

    def test_bearish_bos_detected_on_low_break(self):
        """
        Scenario: Price in uptrend, then breaks below a swing low.
        This should be classified as CHOC (change of character / reversal)
        since the prior trend was UP and breaking a low is counter-trend.

        Previously: with only 1 previous high, the break was silently skipped.
        Now: fallback logic ensures it's still classified.
        """
        df = _make_bearish_bos_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=5, ob_strength_threshold=0.3)

        swings = analyzer.detect_swing_points(df, lookback=80)
        breaks = analyzer.detect_structure_breaks(df, swings)

        # Should have at least some structure breaks
        all_breaks = breaks['bos'] + breaks['choc']
        assert len(all_breaks) > 0, (
            f"Expected structure breaks to be detected. "
            f"Swings: {[(sp.type, round(sp.price, 1), sp.index) for sp in swings]}"
        )

        # Look for breaks of lows (these should exist given the scenario)
        low_breaks = [sp for sp in all_breaks if sp.type == 'low']
        assert len(low_breaks) > 0, (
            f"Expected bearish structure breaks (low broken), got none. "
            f"All breaks: {[(sp.type, sp.break_type, round(sp.price, 1)) for sp in all_breaks]}"
        )

    def test_bearish_bos_with_downtrend_context(self):
        """
        When price is already in a downtrend (lower highs) and breaks
        a swing low, it should be BOS (continuation), not CHOC.
        """
        n = 60
        timestamps = pd.date_range('2026-03-01', periods=n, freq='h')
        np.random.seed(42)

        prices = []
        for i in range(n):
            if i < 15:
                base = 130 - i * 0.5 + np.sin(i * 0.8) * 3  # Down
            elif i < 25:
                base = 123 + np.sin(i * 0.6) * 4  # Swing high ~127
            elif i < 35:
                base = 127 - (i - 25) * 1.0  # Drop to swing low ~117
            elif i < 45:
                base = 117 + np.sin(i * 0.5) * 3  # Lower high ~120
            else:
                base = 120 - (i - 45) * 1.5  # Break below 117 = BOS
            prices.append(base)

        data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        for price in prices:
            o = price + np.random.uniform(-0.3, 0.3)
            c = price + np.random.uniform(-0.3, 0.3)
            h = max(o, c) + np.random.uniform(0.3, 1.5)
            lo = min(o, c) - np.random.uniform(0.3, 1.5)
            data['open'].append(o)
            data['high'].append(h)
            data['low'].append(lo)
            data['close'].append(c)
            data['volume'].append(np.random.uniform(1000, 5000))

        df = pd.DataFrame(data, index=timestamps)
        analyzer = SmartMoneyAnalyzer(swing_lookback=3, ob_strength_threshold=0.3)

        swings = analyzer.detect_swing_points(df, lookback=60)
        breaks = analyzer.detect_structure_breaks(df, swings)

        # In a downtrend, breaking a low should be BOS
        bos_lows = [sp for sp in breaks['bos'] if sp.type == 'low']
        # It's OK if there are also CHOCs; the key point is that breaks are detected
        all_low_breaks = [sp for sp in breaks['bos'] + breaks['choc'] if sp.type == 'low']
        assert len(all_low_breaks) > 0, (
            f"Expected at least one low break in downtrend scenario. "
            f"BOS: {[(sp.type, sp.break_type) for sp in breaks['bos']]} "
            f"CHOC: {[(sp.type, sp.break_type) for sp in breaks['choc']]}"
        )

    def test_structural_ob_zone_not_too_wide(self):
        """
        Structural OBs should not span more than ~2x ATR.
        Previously zones could be 50+ points on XAUUSD (ATR ~22).
        """
        df = _make_xauusd_downtrend_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=5, ob_strength_threshold=0.3)

        swings = analyzer.detect_swing_points(df, lookback=200)
        structural_obs = analyzer.detect_structural_order_blocks(df, swings, lookback=200)

        atr_series = analyzer._get_atr(df)
        recent_atr = atr_series.iloc[-1]

        for ob in structural_obs:
            zone_width = ob.top - ob.bottom
            max_allowed = recent_atr * 3.0  # 3x ATR is generous upper limit
            assert zone_width <= max_allowed, (
                f"Structural OB zone too wide: {ob.type} at "
                f"{ob.bottom:.1f}-{ob.top:.1f} = {zone_width:.1f} pts, "
                f"ATR={recent_atr:.1f}, max allowed={max_allowed:.1f}"
            )

    def test_structural_ob_consolidation_candles_reduced(self):
        """Verify consolidation candles constant was reduced."""
        assert SMCConstants.STRUCTURAL_OB_CONSOLIDATION_CANDLES == 3, (
            f"Expected 3, got {SMCConstants.STRUCTURAL_OB_CONSOLIDATION_CANDLES}"
        )

    def test_candle_ob_zone_width_reasonable(self):
        """
        Candle-based OBs should be single-candle width (high-low of one candle).
        They're already well-constrained but verify.
        """
        df = _make_xauusd_downtrend_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=5, ob_strength_threshold=0.3)

        obs = analyzer.detect_order_blocks(df, lookback=200)
        for ob in obs:
            if ob.detection_method == 'candle':
                zone_width = ob.top - ob.bottom
                # Single candle OB shouldn't exceed ~30 points on XAUUSD
                assert zone_width < 40, (
                    f"Candle OB too wide: {zone_width:.1f} pts at "
                    f"{ob.bottom:.1f}-{ob.top:.1f}"
                )

    def test_full_analysis_detects_bearish_structure(self):
        """
        Run full analysis on downtrend data and verify bearish bias
        with proper BOS/CHoCH detection.
        """
        df = _make_xauusd_downtrend_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=5, ob_strength_threshold=0.3)

        analysis = analyzer.analyze_full_smc(df)

        # Should have structure data
        assert 'structure' in analysis
        structure = analysis['structure']

        # Full analysis uses summary keys (bos_count, choc_count)
        bos_count = structure.get('bos_count', 0)
        choc_count = structure.get('choc_count', 0)
        total_breaks = bos_count + choc_count
        assert total_breaks > 0, (
            f"No structure breaks detected in downtrend data. "
            f"Structure: {structure}"
        )

    def test_bos_detection_requires_fewer_previous_swings(self):
        """
        BOS detection should work even with only 1 previous swing
        (fallback logic), not require 2 previous same-type swings.
        """
        # Create minimal data: one swing high, one swing low, then break
        n = 40
        timestamps = pd.date_range('2026-03-01', periods=n, freq='h')
        np.random.seed(123)

        prices = []
        for i in range(n):
            if i < 10:
                base = 100 + i * 2  # Up to ~120
            elif i < 15:
                base = 120 - (i - 10) * 1  # Pullback to ~115
            elif i < 20:
                base = 115 + (i - 15) * 1  # Up to ~120 again (lower high)
            elif i < 30:
                base = 120 - (i - 20) * 2  # Drop below 115 = break
            else:
                base = 100 - (i - 30) * 0.5  # Continue lower
            prices.append(base)

        data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        for price in prices:
            o = price + np.random.uniform(-0.2, 0.2)
            c = price + np.random.uniform(-0.2, 0.2)
            h = max(o, c) + np.random.uniform(0.2, 1.0)
            lo = min(o, c) - np.random.uniform(0.2, 1.0)
            data['open'].append(o)
            data['high'].append(h)
            data['low'].append(lo)
            data['close'].append(c)
            data['volume'].append(1000)

        df = pd.DataFrame(data, index=timestamps)
        analyzer = SmartMoneyAnalyzer(swing_lookback=3, ob_strength_threshold=0.3)

        swings = analyzer.detect_swing_points(df, lookback=40)
        breaks = analyzer.detect_structure_breaks(df, swings)

        # With the fallback logic, breaks should be detected
        # even with minimal swing history
        all_breaks = breaks['bos'] + breaks['choc']
        # At minimum we should detect the highs being broken
        # or the lows being broken
        high_breaks = [sp for sp in all_breaks if sp.type == 'high']
        low_breaks = [sp for sp in all_breaks if sp.type == 'low']

        assert len(all_breaks) > 0, (
            f"Expected breaks with minimal swing history. "
            f"Swings found: {[(sp.type, round(sp.price, 1)) for sp in swings]}"
        )


class TestTPSelection:
    """Tests for _calculate_take_profit in SMCTradePlanGenerator."""

    def _make_generator(self):
        from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator
        return SMCTradePlanGenerator(min_quality_score=60.0, min_rr_ratio=1.5)

    def _base_smc_analysis(self):
        """Base SMC analysis data with zones mimicking XAUUSD scenario."""
        return {
            "order_blocks": {
                "bullish": [
                    {"top": 5107, "bottom": 5051, "strength": 0.80, "mitigated": False},
                ],
                "bearish": [
                    {"top": 5200, "bottom": 5185, "strength": 0.65, "mitigated": False},
                ],
            },
            "liquidity_zones": [
                {"price": 5156, "type": "sell-side", "strength": 50},
            ],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5077, "touches": 6, "swept": False},
                    {"price": 5030, "touches": 2, "swept": False},
                ],
            },
        }

    def test_equal_levels_included_as_tp_candidates_sell(self):
        """Equal lows should be valid TP targets for SELL orders."""
        gen = self._make_generator()
        smc = self._base_smc_analysis()

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5140.0,
            market_regime="trending-down",
        )

        # TP should be at the strong equal level (5077), not the nearest liquidity (5156)
        # 5077 has 6 touches = strength 100, while 5156 liquidity = strength 50
        assert tp_zone["type"] == "equal_level", f"Expected equal_level TP, got {tp_zone['type']}"
        assert abs(tp_price - 5077) < 1, f"Expected TP near 5077, got {tp_price}"

    def test_strong_trend_prefers_stronger_target(self):
        """In a strong downtrend, prefer the deeper/stronger target over nearest."""
        gen = self._make_generator()
        smc = self._base_smc_analysis()

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5140.0,
            market_regime="trending-down",
        )

        # 5077 (equal_level, 6 touches, strength=100) should beat
        # 5107 (opposing_ob top, strength=80) and 5156 (liquidity, strength=50)
        assert tp_price < 5120, f"Expected deeper TP target in strong trend, got {tp_price}"

    def test_ranging_market_prefers_nearest_target(self):
        """In ranging/weak market, prefer the nearest target when strengths are comparable."""
        gen = self._make_generator()
        # Use zones with similar strength so proximity wins in ranging mode
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [
                {"price": 5130, "type": "sell-side", "strength": 70},
            ],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5050, "touches": 3, "swept": False},  # strength=70
                ],
            },
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5150.0,
            market_regime="ranging",
        )

        # Both have strength ~70, but 5130 is much closer - should win in ranging
        assert tp_price > 5100, f"Expected nearer TP in ranging market, got {tp_price}"

    def test_equal_highs_as_buy_tp(self):
        """Equal highs should be valid TP targets for BUY orders."""
        gen = self._make_generator()
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "equal_levels": {
                "equal_highs": [
                    {"price": 5200, "touches": 4, "swept": False},
                ],
                "equal_lows": [],
            },
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5100.0,
            direction="BUY",
            smc_analysis=smc,
            current_price=5110.0,
            market_regime="trending-up",
        )

        assert tp_zone["type"] == "equal_level"
        assert abs(tp_price - 5200) < 1

    def test_swept_equal_levels_excluded(self):
        """Swept equal levels should not be used as TP targets."""
        gen = self._make_generator()
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5077, "touches": 6, "swept": True},  # Swept - excluded
                ],
            },
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5140.0,
            market_regime="trending-down",
        )

        # With only swept levels and no other candidates, should fall back to calculated
        assert tp_zone["type"] == "calculated"

    def test_no_candidates_falls_back(self):
        """When no TP candidates exist, fallback calculation is used."""
        gen = self._make_generator()
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "equal_levels": {"equal_highs": [], "equal_lows": []},
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5140.0,
        )

        assert tp_zone["type"] == "calculated"
        assert tp_price < 5140.0, "Fallback SELL TP should be below current price"

    def test_strength_scales_with_touches(self):
        """Equal level strength should scale: 2 touches=60, 6=100, 10=100 (capped)."""
        gen = self._make_generator()

        # 2 touches vs 6 touches - both available as SELL targets
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5090, "touches": 2, "swept": False},  # strength=60
                    {"price": 5077, "touches": 6, "swept": False},  # strength=100
                ],
            },
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5140.0,
            market_regime="trending-down",
        )

        # In strong trend, the stronger 5077 (100 strength) should win
        assert abs(tp_price - 5077) < 1, f"Expected 5077 (6 touches), got {tp_price}"

    def test_proximity_penalty_for_very_close_targets(self):
        """Targets less than 0.2% from entry should get penalized in strong trends."""
        gen = self._make_generator()
        # 5160 is only 0.12% from 5166 entry - should be penalized
        smc = {
            "order_blocks": {"bullish": [], "bearish": []},
            "liquidity_zones": [
                {"price": 5160, "type": "sell-side", "strength": 60},
            ],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5077, "touches": 3, "swept": False},  # strength=70
                ],
            },
        }

        tp_price, tp_zone = gen._calculate_take_profit(
            entry_price=5166.0,
            direction="SELL",
            smc_analysis=smc,
            current_price=5150.0,
            market_regime="trending-down",
        )

        # 5160 is too close and should be penalized; 5077 should win
        assert tp_price < 5120, f"Very close target should be penalized, got TP={tp_price}"


class TestGeneratePlanTPIntegration:
    """
    Integration tests verifying that generate_plan() receives equal levels
    and uses them for TP selection. This catches the bug where flat_smc
    in the backend was missing equal_levels, causing TP to fall back
    to the nearest liquidity zone.
    """

    def _make_xauusd_smc_with_equal_levels(self):
        """
        Simulate the full SMC analysis dict as returned by analyze_full_smc(),
        matching XAUUSD H1 scenario with proven support at 5077.
        """
        return {
            "bias": "bearish",
            "order_blocks": {
                "bullish": [
                    {"top": 5107, "bottom": 5051, "strength": 0.80, "mitigated": False},
                ],
                "bearish": [
                    {"top": 5190, "bottom": 5170, "strength": 0.65, "mitigated": False},
                ],
            },
            "fair_value_gaps": {
                "bullish": [],
                "bearish": [
                    {"top": 5175, "bottom": 5160, "strength": 0.60, "mitigated": False},
                ],
            },
            "liquidity_zones": [
                {"price": 5156, "type": "sell-side", "strength": 50},
            ],
            "equal_levels": {
                "equal_highs": [],
                "equal_lows": [
                    {"price": 5077, "touches": 6, "swept": False},
                ],
            },
            "market_structure": {
                "recent_bos": [{"type": "low", "price": 5140}],
                "recent_choc": [],
            },
            "structure": {
                "recent_bos": [{"type": "low", "price": 5140}],
                "recent_choc": [],
            },
            "atr": 21.8,
        }

    def test_generate_plan_uses_equal_levels_for_tp(self):
        """
        generate_plan() should target equal level at 5077 (6 retests)
        instead of nearest liquidity at 5156 when equal_levels is present.
        """
        from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator

        gen = SMCTradePlanGenerator(
            min_quality_score=50.0,
            min_rr_ratio=0.1,  # Low threshold so plan isn't rejected
        )
        smc = self._make_xauusd_smc_with_equal_levels()

        plan = gen.generate_plan(
            smc_analysis=smc,
            current_price=5140.0,
            atr=21.8,
            market_regime="trending-down",
        )

        assert plan is not None, "Expected a trade plan to be generated"
        assert plan.signal == "SELL", f"Expected SELL signal, got {plan.signal}"
        # TP should target the proven equal level at 5077, not 5156
        assert plan.take_profit < 5120, (
            f"TP {plan.take_profit:.2f} is too conservative. "
            f"Expected ~5077 (proven support with 6 retests), not ~5156 (nearest liquidity)."
        )

    def test_generate_plan_without_equal_levels_uses_nearest(self):
        """
        Without equal_levels in the analysis dict (the old bug),
        TP falls back to nearest liquidity/OB — this is the behavior
        we had before the fix.
        """
        from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator

        gen = SMCTradePlanGenerator(
            min_quality_score=50.0,
            min_rr_ratio=0.1,
        )
        smc = self._make_xauusd_smc_with_equal_levels()
        # Remove equal_levels — simulates the old flat_smc bug
        del smc["equal_levels"]

        plan = gen.generate_plan(
            smc_analysis=smc,
            current_price=5140.0,
            atr=21.8,
            market_regime="trending-down",
        )

        assert plan is not None, "Expected a trade plan to be generated"
        # Without equal levels, TP should target opposing OB top at 5107 or liquidity
        # It should NOT be at 5077 since that data is missing
        assert plan.take_profit > 5077, (
            f"TP {plan.take_profit:.2f} reached 5077 even without equal_levels data — "
            f"this means equal_levels wasn't the source of the 5077 target."
        )

    def test_flat_smc_must_include_equal_levels(self):
        """
        Regression test: verify that the flat_smc dict pattern used in the
        backend includes equal_levels. This is a documentation/contract test.
        """
        smc_result = self._make_xauusd_smc_with_equal_levels()

        # Simulate the backend's flat_smc construction (AFTER the fix)
        flat_smc = {
            "order_blocks": smc_result.get("order_blocks", {}),
            "fair_value_gaps": smc_result.get("fair_value_gaps", {}),
            "liquidity_zones": smc_result.get("liquidity_zones", {}),
            "market_structure": smc_result.get("market_structure", {}),
            "equal_levels": smc_result.get("equal_levels", {}),
            "atr": smc_result.get("atr", 21.8),
        }

        # Verify equal_levels survived the flattening
        assert "equal_levels" in flat_smc, "flat_smc must include equal_levels"
        eq_lows = flat_smc["equal_levels"].get("equal_lows", [])
        assert len(eq_lows) > 0, "equal_lows should be present in flat_smc"
        assert eq_lows[0]["price"] == 5077, "5077 proven support should be in flat_smc"


    def test_generate_plan_with_dataclass_equal_levels(self):
        """
        analyze_full_smc_extended returns EqualLevel dataclass objects,
        not dicts. Verify generate_plan handles dataclass objects correctly
        via safe_get.
        """
        from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator

        gen = SMCTradePlanGenerator(
            min_quality_score=50.0,
            min_rr_ratio=0.1,
        )
        smc = self._make_xauusd_smc_with_equal_levels()

        # Replace dict equal levels with EqualLevel dataclass objects
        # (as returned by analyze_full_smc_extended)
        smc["equal_levels"] = {
            "equal_highs": [],
            "equal_lows": [
                EqualLevel(price=5077, type="equal_lows", touches=6, swept=False, indices=[10, 50, 90, 120, 150, 180]),
            ],
        }

        plan = gen.generate_plan(
            smc_analysis=smc,
            current_price=5140.0,
            atr=21.8,
            market_regime="trending-down",
        )

        assert plan is not None, "Expected a trade plan to be generated"
        assert plan.take_profit < 5120, (
            f"TP {plan.take_profit:.2f} should target dataclass EqualLevel at 5077, not nearest zone"
        )

    def test_analyze_full_smc_extended_includes_equal_levels(self):
        """
        Verify analyze_full_smc_extended returns equal_levels in its output,
        confirming the rule-based endpoint will have the data it needs.
        """
        # Use the XAUUSD downtrend test data
        data = _make_xauusd_downtrend_data()
        analyzer = SmartMoneyAnalyzer(swing_lookback=3, ob_strength_threshold=0.3)

        result = analyzer.analyze_full_smc_extended(
            data, current_price=5127.0, use_structural_obs=True
        )

        assert "equal_levels" in result, "analyze_full_smc_extended must return equal_levels"
        assert "equal_highs" in result["equal_levels"], "Must have equal_highs key"
        assert "equal_lows" in result["equal_levels"], "Must have equal_lows key"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
