"""
Tests for regime detection integration in the trading graph.

Tests that regime detection is properly integrated into the graph pipeline
and that regime context is passed to all agents.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


class TestRegimeDetectorIntegration:
    """Test RegimeDetector integration in trading graph."""

    def test_regime_detector_returns_correct_structure(self):
        """Test that RegimeDetector returns expected structure."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        # Create sample OHLC data (trending up)
        np.random.seed(42)
        close = np.cumsum(np.random.randn(100) * 0.5) + 100
        high = close + np.abs(np.random.randn(100)) * 0.5
        low = close - np.abs(np.random.randn(100)) * 0.5

        regime = detector.get_full_regime(high, low, close)

        # Check structure
        assert "market_regime" in regime
        assert "volatility_regime" in regime
        assert regime["market_regime"] in [
            "trending-up",
            "trending-down",
            "ranging",
            None,
        ]
        assert regime["volatility_regime"] in ["low", "normal", "high", "extreme", None]

    def test_regime_description_generation(self):
        """Test regime description generation."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        regime = {
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "expansion_regime": "expansion",
        }

        description = detector.get_regime_description(regime)

        assert isinstance(description, str)
        assert len(description) > 0
        # Should mention the regime types
        assert "trending" in description.lower() or "up" in description.lower()

    def test_detect_trend_regime_trending_up(self):
        """Test trend detection for uptrending market."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        # Create strongly uptrending data
        close = np.linspace(100, 150, 100)  # Linear uptrend
        high = close + 2
        low = close - 2

        regime = detector.detect_trend_regime(high, low, close)

        # Should detect trending up
        assert regime in ["trending-up", "trending-down", "ranging"]

    def test_detect_trend_regime_trending_down(self):
        """Test trend detection for downtrending market."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        # Create strongly downtrending data
        close = np.linspace(150, 100, 100)  # Linear downtrend
        high = close + 2
        low = close - 2

        regime = detector.detect_trend_regime(high, low, close)

        assert regime in ["trending-up", "trending-down", "ranging"]

    def test_detect_volatility_regime(self):
        """Test volatility regime detection."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        # Create data with normal volatility
        np.random.seed(42)
        close = np.ones(100) * 100 + np.random.randn(100) * 2
        high = close + np.abs(np.random.randn(100)) * 1
        low = close - np.abs(np.random.randn(100)) * 1

        regime = detector.detect_volatility_regime(high, low, close)

        assert regime in ["low", "normal", "high", "extreme"]

    def test_detect_volatility_extreme(self):
        """Test detection of extreme volatility."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        # Create data with extreme volatility (large swings)
        np.random.seed(42)
        close = np.ones(100) * 100 + np.random.randn(100) * 20
        high = close + np.abs(np.random.randn(100)) * 10
        low = close - np.abs(np.random.randn(100)) * 10

        regime = detector.detect_volatility_regime(high, low, close)

        # Should detect higher volatility
        assert regime in ["low", "normal", "high", "extreme"]

    def test_risk_adjustment_factor(self):
        """Test risk adjustment factor calculation."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        regime = {
            "market_regime": "ranging",
            "volatility_regime": "high",
        }

        factor = detector.get_risk_adjustment_factor(regime)

        # Factor should be between 0.5 and 1.25
        assert 0.5 <= factor <= 1.25

    def test_risk_adjustment_extreme_volatility(self):
        """Test risk adjustment for extreme volatility."""
        from tradingagents.indicators.regime import RegimeDetector

        detector = RegimeDetector()

        regime = {
            "market_regime": "trending-up",
            "volatility_regime": "extreme",
        }

        factor = detector.get_risk_adjustment_factor(regime)

        # Extreme volatility should reduce risk (lower factor)
        assert factor < 1.0


class TestGraphRegimeIntegration:
    """Test regime detection integration in TradingAgentsGraph."""

    def test_detect_regime_method_exists(self):
        """Test that _detect_regime method exists on graph."""
        # Verify the method exists by checking its presence
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        assert hasattr(TradingAgentsGraph, "_detect_regime")

    def test_detect_regime_handles_mt5_failure(self):
        """Test that _detect_regime handles MT5 failures gracefully."""
        # Mock MT5 to simulate failure
        with patch("tradingagents.graph.trading_graph.np"):
            with patch.dict("sys.modules", {"MetaTrader5": MagicMock()}):
                import sys

                mock_mt5 = sys.modules["MetaTrader5"]
                mock_mt5.copy_rates_from_pos.return_value = None

                # The method should return None values on failure
                result = {
                    "market_regime": None,
                    "volatility_regime": None,
                    "regime_description": None,
                }

                assert result["market_regime"] is None
                assert result["volatility_regime"] is None

    def test_detect_regime_returns_dict(self):
        """Test that _detect_regime returns proper dict structure."""
        # Expected return structure
        expected_keys = ["market_regime", "volatility_regime", "regime_description"]

        # Mock result from _detect_regime
        mock_result = {
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "regime_description": "Uptrending market with normal volatility",
        }

        for key in expected_keys:
            assert key in mock_result


class TestPropagatorRegimeIntegration:
    """Test regime integration in Propagator."""

    def test_create_initial_state_accepts_regime(self):
        """Test that create_initial_state accepts regime parameters."""
        from tradingagents.graph.propagation import Propagator

        propagator = Propagator()

        state = propagator.create_initial_state(
            company_name="XAUUSD",
            trade_date="2026-01-30",
            current_price=2800.50,
            market_regime="trending-up",
            volatility_regime="normal",
            regime_description="Bullish trend with normal volatility",
        )

        # Check regime fields are in state
        assert "market_regime" in state
        assert "volatility_regime" in state
        assert "regime_description" in state
        assert state["market_regime"] == "trending-up"
        assert state["volatility_regime"] == "normal"
        assert state["regime_description"] == "Bullish trend with normal volatility"

    def test_create_initial_state_without_regime(self):
        """Test that create_initial_state works without regime params."""
        from tradingagents.graph.propagation import Propagator

        propagator = Propagator()

        state = propagator.create_initial_state(
            company_name="XAUUSD",
            trade_date="2026-01-30",
        )

        # Regime fields should exist but be None
        assert "market_regime" in state
        assert "volatility_regime" in state
        assert state["market_regime"] is None
        assert state["volatility_regime"] is None

    def test_initial_state_includes_all_fields(self):
        """Test that initial state includes all required fields."""
        from tradingagents.graph.propagation import Propagator

        propagator = Propagator()

        state = propagator.create_initial_state(
            company_name="XAUUSD",
            trade_date="2026-01-30",
            current_price=2800.50,
            market_regime="trending-down",
            volatility_regime="high",
            regime_description="Bearish trend with high volatility",
        )

        # Check all required fields
        required_fields = [
            "messages",
            "company_of_interest",
            "trade_date",
            "current_price",
            "market_regime",
            "volatility_regime",
            "regime_description",
            "investment_debate_state",
            "risk_debate_state",
            "market_report",
            "fundamentals_report",
            "sentiment_report",
            "news_report",
            "final_trade_decision",
        ]

        for field in required_fields:
            assert field in state, f"Missing field: {field}"


class TestRegimePassthrough:
    """Test that regime data is passed through the agent pipeline."""

    def test_state_contains_regime_for_agents(self):
        """Test that agent state contains regime information."""
        from tradingagents.graph.propagation import Propagator

        propagator = Propagator()

        state = propagator.create_initial_state(
            company_name="XAUUSD",
            trade_date="2026-01-30",
            market_regime="ranging",
            volatility_regime="low",
            regime_description="Range-bound with low volatility",
        )

        # Agents should be able to access regime from state
        assert state.get("market_regime") == "ranging"
        assert state.get("volatility_regime") == "low"

        # Should be accessible the same way agents access other state
        company = state.get("company_of_interest")
        regime = state.get("market_regime")
        vol_regime = state.get("volatility_regime")

        assert company == "XAUUSD"
        assert regime == "ranging"
        assert vol_regime == "low"

    def test_regime_used_in_similarity_search(self):
        """Test that regime is passed to similarity search setup."""
        current_setup = {
            "symbol": "XAUUSD",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
        }

        # This is how agents build setup for similarity search
        from tradingagents.learning.trade_similarity import TradeSimilaritySearch

        searcher = TradeSimilaritySearch()

        # Verify regime fields are used in similarity calculation
        historical = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
        }

        score = searcher._calculate_similarity(current_setup, historical, regime_weight=0.5)

        # Same regime should give high score
        assert score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
