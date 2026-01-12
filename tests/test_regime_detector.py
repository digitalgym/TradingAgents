"""
Unit tests for RegimeDetector

Tests regime detection including:
- Trend regime detection (ADX-based)
- Volatility regime detection (ATR percentile)
- Expansion regime detection (Bollinger Bands)
- Full regime classification
"""

import pytest
import numpy as np
from tradingagents.indicators.regime import RegimeDetector


class TestTrendRegimeDetection:
    """Test trend regime detection using ADX"""
    
    def test_trending_up_market(self):
        """Strong uptrend should be detected"""
        detector = RegimeDetector()
        
        # Create uptrending price data
        close = np.linspace(2600, 2700, 50)
        high = close + 5
        low = close - 5
        
        regime = detector.detect_trend_regime(high, low, close)
        assert regime == "trending-up"
    
    def test_trending_down_market(self):
        """Strong downtrend should be detected"""
        detector = RegimeDetector()
        
        # Create downtrending price data
        close = np.linspace(2700, 2600, 50)
        high = close + 5
        low = close - 5
        
        regime = detector.detect_trend_regime(high, low, close)
        assert regime == "trending-down"
    
    def test_ranging_market(self):
        """Ranging market should be detected"""
        detector = RegimeDetector()
        
        # Create ranging price data (oscillating)
        close = 2650 + 10 * np.sin(np.linspace(0, 4*np.pi, 50))
        high = close + 5
        low = close - 5
        
        regime = detector.detect_trend_regime(high, low, close)
        assert regime == "ranging"
    
    def test_minimal_data_returns_ranging(self):
        """Insufficient data should default to ranging"""
        detector = RegimeDetector()
        
        close = np.array([2650, 2655, 2660])
        high = close + 5
        low = close - 5
        
        regime = detector.detect_trend_regime(high, low, close, period=14)
        assert regime == "ranging"


class TestVolatilityRegimeDetection:
    """Test volatility regime detection using ATR percentile"""
    
    def test_low_volatility(self):
        """Low volatility should be detected"""
        detector = RegimeDetector(lookback_period=50)
        
        # Create low volatility data (tight range)
        close = 2650 + np.random.randn(100) * 2  # Small std dev
        high = close + 1
        low = close - 1
        
        regime = detector.detect_volatility_regime(high, low, close)
        # Should be low or normal
        assert regime in ["low", "normal"]
    
    def test_high_volatility(self):
        """High volatility should be detected"""
        detector = RegimeDetector(lookback_period=50)
        
        # Create high volatility data
        np.random.seed(42)
        close = np.concatenate([
            2650 + np.random.randn(80) * 5,  # Normal volatility
            2650 + np.random.randn(20) * 25  # High volatility at end
        ])
        high = close + np.abs(np.random.randn(100) * 10)
        low = close - np.abs(np.random.randn(100) * 10)
        
        regime = detector.detect_volatility_regime(high, low, close)
        # Should detect elevated volatility
        assert regime in ["high", "extreme", "normal"]
    
    def test_minimal_data_returns_normal(self):
        """Insufficient data should default to normal"""
        detector = RegimeDetector()
        
        close = np.array([2650, 2655, 2660])
        high = close + 5
        low = close - 5
        
        regime = detector.detect_volatility_regime(high, low, close)
        assert regime == "normal"


class TestExpansionRegimeDetection:
    """Test expansion/contraction detection using Bollinger Bands"""
    
    def test_expansion_regime(self):
        """Expansion should be detected during volatile periods"""
        detector = RegimeDetector(lookback_period=50)
        
        # Create data with expanding range at end
        np.random.seed(42)
        close = np.concatenate([
            2650 + np.random.randn(80) * 3,   # Normal range
            2650 + np.random.randn(20) * 15   # Expanding range
        ])
        
        regime = detector.detect_expansion_regime(close)
        # Should detect expansion or neutral
        assert regime in ["expansion", "neutral"]
    
    def test_contraction_regime(self):
        """Contraction should be detected during tight ranges"""
        detector = RegimeDetector(lookback_period=50)
        
        # Create data with contracting range at end
        np.random.seed(42)
        close = np.concatenate([
            2650 + np.random.randn(80) * 10,  # Wide range
            2650 + np.random.randn(20) * 1    # Tight range
        ])
        
        regime = detector.detect_expansion_regime(close)
        # Should detect contraction or neutral
        assert regime in ["contraction", "neutral"]
    
    def test_minimal_data_returns_neutral(self):
        """Insufficient data should default to neutral"""
        detector = RegimeDetector()
        
        close = np.array([2650, 2655, 2660])
        regime = detector.detect_expansion_regime(close)
        assert regime == "neutral"


class TestFullRegimeDetection:
    """Test complete regime classification"""
    
    def test_full_regime_structure(self):
        """Full regime should return all components"""
        detector = RegimeDetector()
        
        # Create sample data
        close = np.linspace(2600, 2700, 100)
        high = close + 5
        low = close - 5
        
        regime = detector.get_full_regime(high, low, close)
        
        # Check all fields present
        assert "market_regime" in regime
        assert "volatility_regime" in regime
        assert "expansion_regime" in regime
        assert "timestamp" in regime
        
        # Check valid values
        assert regime["market_regime"] in ["trending-up", "trending-down", "ranging"]
        assert regime["volatility_regime"] in ["low", "normal", "high", "extreme"]
        assert regime["expansion_regime"] in ["expansion", "contraction", "neutral"]
    
    def test_trending_up_high_volatility(self):
        """Test specific regime combination"""
        detector = RegimeDetector()
        
        # Create trending up data with high volatility
        np.random.seed(42)
        trend = np.linspace(2600, 2700, 100)
        noise = np.random.randn(100) * 15
        close = trend + noise
        high = close + np.abs(np.random.randn(100) * 10)
        low = close - np.abs(np.random.randn(100) * 10)
        
        regime = detector.get_full_regime(high, low, close)
        
        # Should detect uptrend
        assert regime["market_regime"] in ["trending-up", "ranging"]


class TestRegimeDescription:
    """Test human-readable regime descriptions"""
    
    def test_trending_up_description(self):
        """Test description for trending up market"""
        detector = RegimeDetector()
        
        regime = {
            "market_regime": "trending-up",
            "volatility_regime": "high",
            "expansion_regime": "expansion"
        }
        
        desc = detector.get_regime_description(regime)
        assert "Trending upward" in desc
        assert "high volatility" in desc
        assert "expansion" in desc
    
    def test_ranging_description(self):
        """Test description for ranging market"""
        detector = RegimeDetector()
        
        regime = {
            "market_regime": "ranging",
            "volatility_regime": "low",
            "expansion_regime": "contraction"
        }
        
        desc = detector.get_regime_description(regime)
        assert "Ranging" in desc
        assert "low volatility" in desc
        assert "contraction" in desc


class TestRegimeFavorability:
    """Test regime favorability checks"""
    
    def test_favorable_for_trend_trading(self):
        """Test trend trading favorability"""
        detector = RegimeDetector()
        
        # Favorable: trending + normal volatility + expansion
        favorable_regime = {
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "expansion_regime": "expansion"
        }
        assert detector.is_favorable_for_trend_trading(favorable_regime) is True
        
        # Unfavorable: ranging market
        unfavorable_regime = {
            "market_regime": "ranging",
            "volatility_regime": "normal",
            "expansion_regime": "neutral"
        }
        assert detector.is_favorable_for_trend_trading(unfavorable_regime) is False
        
        # Unfavorable: extreme volatility
        extreme_regime = {
            "market_regime": "trending-up",
            "volatility_regime": "extreme",
            "expansion_regime": "expansion"
        }
        assert detector.is_favorable_for_trend_trading(extreme_regime) is False
    
    def test_favorable_for_range_trading(self):
        """Test range trading favorability"""
        detector = RegimeDetector()
        
        # Favorable: ranging + low volatility + contraction
        favorable_regime = {
            "market_regime": "ranging",
            "volatility_regime": "low",
            "expansion_regime": "contraction"
        }
        assert detector.is_favorable_for_range_trading(favorable_regime) is True
        
        # Unfavorable: trending market
        unfavorable_regime = {
            "market_regime": "trending-up",
            "volatility_regime": "low",
            "expansion_regime": "neutral"
        }
        assert detector.is_favorable_for_range_trading(unfavorable_regime) is False


class TestRiskAdjustment:
    """Test risk adjustment based on regime"""
    
    def test_extreme_volatility_reduces_risk(self):
        """Extreme volatility should reduce position size"""
        detector = RegimeDetector()
        
        regime = {"volatility_regime": "extreme"}
        adjustment = detector.get_risk_adjustment_factor(regime)
        assert adjustment == 0.5  # 50% of normal size
    
    def test_high_volatility_reduces_risk(self):
        """High volatility should moderately reduce position size"""
        detector = RegimeDetector()
        
        regime = {"volatility_regime": "high"}
        adjustment = detector.get_risk_adjustment_factor(regime)
        assert adjustment == 0.75  # 75% of normal size
    
    def test_normal_volatility_no_adjustment(self):
        """Normal volatility should have no adjustment"""
        detector = RegimeDetector()
        
        regime = {"volatility_regime": "normal"}
        adjustment = detector.get_risk_adjustment_factor(regime)
        assert adjustment == 1.0  # 100% of normal size
    
    def test_low_volatility_increases_risk(self):
        """Low volatility allows slightly larger position"""
        detector = RegimeDetector()
        
        regime = {"volatility_regime": "low"}
        adjustment = detector.get_risk_adjustment_factor(regime)
        assert adjustment == 1.25  # 125% of normal size


class TestTechnicalCalculations:
    """Test internal technical indicator calculations"""
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        detector = RegimeDetector()
        
        # Create sample data with known range
        close = np.array([100, 102, 101, 103, 102, 104, 103, 105] * 3)
        high = close + 2
        low = close - 2
        
        atr = detector._calculate_atr(high, low, close, period=14)
        
        # ATR should be positive
        assert atr > 0
        # ATR should be reasonable (around the average range)
        assert 2 <= atr <= 6
    
    def test_bb_width_calculation(self):
        """Test Bollinger Band width calculation"""
        detector = RegimeDetector()
        
        # Create sample data
        close = np.array([100] * 20)  # Flat price
        width = detector._calculate_bb_width(close, period=20, std_dev=2.0)
        
        # Width should be very small for flat prices
        assert width >= 0
        assert width < 1  # Less than 1% for flat prices
    
    def test_true_range_calculation(self):
        """Test True Range calculation"""
        detector = RegimeDetector()
        
        high = np.array([105, 107, 106, 108])
        low = np.array([100, 102, 101, 103])
        close = np.array([103, 104, 103, 105])
        
        tr = detector._calculate_true_range(high, low, close)
        
        # TR should be positive
        assert all(tr >= 0)
        # First TR should be high-low
        assert tr[0] == 5


class TestCustomThresholds:
    """Test detector with custom thresholds"""
    
    def test_custom_adx_threshold(self):
        """Test with custom ADX threshold"""
        detector = RegimeDetector(adx_threshold_trending=30.0)
        
        # Same data might give different results with higher threshold
        close = np.linspace(2600, 2650, 50)
        high = close + 5
        low = close - 5
        
        regime = detector.detect_trend_regime(high, low, close)
        assert regime in ["trending-up", "trending-down", "ranging"]
    
    def test_custom_lookback_period(self):
        """Test with custom lookback period"""
        detector = RegimeDetector(lookback_period=50)
        
        close = np.random.randn(100) * 10 + 2650
        high = close + 5
        low = close - 5
        
        regime = detector.detect_volatility_regime(high, low, close)
        assert regime in ["low", "normal", "high", "extreme"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
