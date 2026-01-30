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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
