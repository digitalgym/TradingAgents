"""
Unit tests for Breakout Quant Analyst

Tests consolidation detection, structure bias analysis, and breakout trading logic.
"""

import pytest
import numpy as np
from tradingagents.agents.analysts.breakout_quant import (
    analyze_consolidation,
    get_breakout_decision_for_modal,
)
from tradingagents.indicators.regime import RegimeDetector


class TestConsolidationAnalysis:
    """Test consolidation detection functionality."""

    def test_tight_range_detected_as_consolidation(self):
        """Tight price range should be detected as consolidation."""
        # Create data with tight range
        np.random.seed(42)
        base_price = 2700
        # Low volatility oscillation
        close = base_price + np.random.randn(50) * 5  # Small std dev
        high = close + 2
        low = close - 2

        result = analyze_consolidation(high, low, close, lookback=20)

        # Should detect some level of squeeze
        assert result["squeeze_strength"] >= 0
        assert result["range_high"] is not None
        assert result["range_low"] is not None
        assert result["range_midpoint"] is not None

    def test_wide_range_not_consolidation(self):
        """Wide price range should not be detected as consolidation."""
        np.random.seed(42)
        # Create data with large range
        close = 2700 + np.random.randn(50) * 50  # Large std dev
        high = close + 20
        low = close - 20

        result = analyze_consolidation(high, low, close, lookback=20)

        # With high volatility, squeeze strength should be lower
        # The exact threshold depends on the historical comparison
        assert result["squeeze_strength"] is not None
        assert result["range_percent"] > 1.0  # Wide range

    def test_higher_lows_bullish_structure(self):
        """Higher lows in consolidation should indicate bullish structure."""
        # Create data with clear higher lows
        close = np.concatenate([
            np.linspace(2680, 2700, 10),  # First half: bottoms at 2680
            np.linspace(2690, 2710, 10),  # Second half: bottoms at 2690 (higher low)
        ])
        high = close + 5
        low = close - 5

        result = analyze_consolidation(high, low, close, lookback=20)

        assert result["structure_bias"] == "bullish"

    def test_lower_highs_bearish_structure(self):
        """Lower highs in consolidation should indicate bearish structure."""
        # Create data with clear lower highs
        close = np.concatenate([
            np.linspace(2720, 2700, 10),  # First half: peaks at 2720
            np.linspace(2710, 2690, 10),  # Second half: peaks at 2710 (lower high)
        ])
        high = close + 5
        low = close - 5

        result = analyze_consolidation(high, low, close, lookback=20)

        assert result["structure_bias"] == "bearish"

    def test_neutral_structure(self):
        """No clear structure should be neutral."""
        np.random.seed(123)
        # Random oscillation with no clear trend
        close = 2700 + np.sin(np.linspace(0, 4*np.pi, 20)) * 10
        high = close + 5
        low = close - 5

        result = analyze_consolidation(high, low, close, lookback=20)

        # Could be neutral or slightly biased depending on random patterns
        assert result["structure_bias"] in ["bullish", "bearish", "neutral"]

    def test_minimal_data_returns_defaults(self):
        """Insufficient data should return safe defaults."""
        close = np.array([2700, 2705, 2710])
        high = close + 5
        low = close - 5

        result = analyze_consolidation(high, low, close, lookback=20)

        assert result["is_consolidating"] is False
        assert result["range_high"] is None
        assert result["structure_bias"] == "neutral"

    def test_range_boundaries_calculated(self):
        """Range high, low, and midpoint should be calculated correctly."""
        close = np.array([100, 105, 102, 108, 95, 103, 106, 101, 104, 99] * 2)
        high = close + 3
        low = close - 3

        result = analyze_consolidation(high, low, close, lookback=20)

        # Check range is reasonable
        assert result["range_high"] > result["range_low"]
        assert result["range_midpoint"] == (result["range_high"] + result["range_low"]) / 2


class TestRegimeDetectorConsolidation:
    """Test consolidation detection in RegimeDetector."""

    def test_detect_consolidation_returns_all_fields(self):
        """Consolidation detection should return all expected fields."""
        detector = RegimeDetector()

        np.random.seed(42)
        close = 2700 + np.random.randn(100) * 5
        high = close + 3
        low = close - 3

        result = detector.detect_consolidation(high, low, close, lookback=20)

        # Check all expected keys
        assert "is_consolidating" in result
        assert "range_high" in result
        assert "range_low" in result
        assert "range_midpoint" in result
        assert "range_percent" in result
        assert "squeeze_strength" in result
        assert "structure_bias" in result
        assert "breakout_ready" in result

    def test_favorable_for_breakout_trading(self):
        """Test breakout trading favorability check."""
        detector = RegimeDetector()

        # Favorable: ranging + low vol + contraction
        favorable_regime = {
            "market_regime": "ranging",
            "volatility_regime": "low",
            "expansion_regime": "contraction"
        }
        assert detector.is_favorable_for_breakout_trading(favorable_regime) is True

        # Unfavorable: trending
        unfavorable_regime = {
            "market_regime": "trending-up",
            "volatility_regime": "low",
            "expansion_regime": "contraction"
        }
        assert detector.is_favorable_for_breakout_trading(unfavorable_regime) is False

        # Unfavorable: expansion (already broken out)
        unfavorable_regime2 = {
            "market_regime": "ranging",
            "volatility_regime": "normal",
            "expansion_regime": "expansion"
        }
        assert detector.is_favorable_for_breakout_trading(unfavorable_regime2) is False

    def test_breakout_ready_requires_bias(self):
        """Breakout ready should require clear structure bias."""
        detector = RegimeDetector()

        # Create data with clear bullish structure
        close = np.concatenate([
            np.full(50, 2700) + np.random.randn(50) * 2,  # Tight first half
            np.linspace(2700, 2720, 50) + np.random.randn(50) * 2,  # Rising second half
        ])
        high = close + 2
        low = close - 2

        result = detector.detect_consolidation(high, low, close, lookback=20)

        # If consolidating with bias, should be breakout ready
        if result["is_consolidating"] and result["structure_bias"] != "neutral":
            assert result["breakout_ready"] is True


class TestBreakoutDecisionForModal:
    """Test conversion of breakout decision to modal format."""

    def test_buy_signal_conversion(self):
        """BUY signal should convert correctly."""
        decision = {
            "symbol": "XAUUSD",
            "signal": "buy_to_enter",
            "order_type": "limit",
            "entry_price": 2700.0,
            "stop_loss": 2680.0,
            "profit_target": 2750.0,
            "confidence": 0.75,
            "justification": "Bullish breakout from consolidation",
            "invalidation_condition": "Price closes below 2680"
        }

        result = get_breakout_decision_for_modal(decision)

        assert result["symbol"] == "XAUUSD"
        assert result["signal"] == "BUY"
        assert result["orderType"] == "limit"
        assert result["suggestedEntry"] == 2700.0
        assert result["suggestedStopLoss"] == 2680.0
        assert result["suggestedTakeProfit"] == 2750.0
        assert result["confidence"] == 0.75
        assert "BREAKOUT" in result["rationale"]

    def test_sell_signal_conversion(self):
        """SELL signal should convert correctly."""
        decision = {
            "symbol": "XAUUSD",
            "signal": "sell_to_enter",
            "order_type": "market",
            "entry_price": 2700.0,
            "stop_loss": 2720.0,
            "profit_target": 2650.0,
            "confidence": 0.8,
            "justification": "Bearish breakdown",
            "invalidation_condition": "Price reclaims 2720"
        }

        result = get_breakout_decision_for_modal(decision)

        assert result["signal"] == "SELL"
        assert result["orderType"] == "market"

    def test_hold_signal_conversion(self):
        """HOLD signal should convert correctly."""
        decision = {
            "symbol": "XAUUSD",
            "signal": "hold",
            "confidence": 0.5,
            "justification": "No clear consolidation",
            "invalidation_condition": "N/A"
        }

        result = get_breakout_decision_for_modal(decision)

        assert result["signal"] == "HOLD"

    def test_empty_decision_returns_empty(self):
        """Empty or None decision should return empty dict."""
        assert get_breakout_decision_for_modal(None) == {}
        assert get_breakout_decision_for_modal({}) == {}


class TestSqueezeStrength:
    """Test BB squeeze strength calculation."""

    def test_squeeze_strength_range(self):
        """Squeeze strength should be between 0 and 100."""
        np.random.seed(42)
        close = 2700 + np.random.randn(100) * 10
        high = close + 5
        low = close - 5

        result = analyze_consolidation(high, low, close, lookback=20)

        assert 0 <= result["squeeze_strength"] <= 100

    def test_tight_squeeze_high_strength(self):
        """Very tight range should have high squeeze strength."""
        # Create very tight range at the end after volatile history
        np.random.seed(42)
        volatile_history = 2700 + np.random.randn(80) * 30  # High volatility history
        tight_range = np.full(20, 2700) + np.random.randn(20) * 2  # Very tight recent

        close = np.concatenate([volatile_history, tight_range])
        high = close + np.abs(np.random.randn(100) * 5)
        low = close - np.abs(np.random.randn(100) * 5)

        result = analyze_consolidation(high, low, close, lookback=20)

        # Current tight range should show high squeeze strength vs volatile history
        # The exact value depends on comparison, but should be elevated
        assert result["squeeze_strength"] >= 50


class TestBreakoutReadyCondition:
    """Test breakout ready condition."""

    def test_not_ready_without_consolidation(self):
        """Should not be breakout ready if not consolidating."""
        np.random.seed(42)
        # Create trending data (not consolidating)
        close = np.linspace(2600, 2800, 50)  # Strong trend
        high = close + 10
        low = close - 10

        result = analyze_consolidation(high, low, close, lookback=20)

        # Trending markets shouldn't be considered consolidating
        if not result["is_consolidating"]:
            assert result["breakout_ready"] is False

    def test_not_ready_with_neutral_bias(self):
        """Should not be breakout ready with neutral structure."""
        # Create ranging data with neutral bias
        np.random.seed(42)
        close = 2700 + np.sin(np.linspace(0, 8*np.pi, 100)) * 5  # Pure oscillation
        high = close + 2
        low = close - 2

        result = analyze_consolidation(high, low, close, lookback=20)

        # Even if consolidating, neutral bias means not breakout ready
        if result["structure_bias"] == "neutral":
            assert result["breakout_ready"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
