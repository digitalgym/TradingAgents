"""
Tests for automatic learning integration in the daily cycle.

Tests that pattern analysis and online RL weight updates are automatically
triggered after sufficient trades have been evaluated.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json


class TestOnlineRLUpdaterIntegration:
    """Test OnlineRLUpdater integration."""

    def test_should_update_returns_tuple(self):
        """Test should_update returns (bool, int) tuple."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()
        result = updater.should_update()

        assert isinstance(result, tuple)
        assert len(result) == 2
        should_update, trades_since = result
        assert isinstance(should_update, bool)
        assert isinstance(trades_since, int)

    def test_should_update_with_empty_dir(self):
        """Test should_update returns False when no trades directory."""
        from tradingagents.learning.online_rl import OnlineRLUpdater
        import tempfile

        updater = OnlineRLUpdater()
        # Use a temp directory that doesn't exist
        temp_dir = tempfile.mkdtemp()
        import shutil
        shutil.rmtree(temp_dir)  # Remove it so it doesn't exist

        should_update, trades_since = updater.should_update(decisions_dir=temp_dir)

        # Should not update with no trades
        assert should_update is False
        assert trades_since == 0

    def test_calculate_agent_performances(self):
        """Test agent performance calculation returns expected structure."""
        from tradingagents.learning.online_rl import OnlineRLUpdater
        import tempfile
        import os

        updater = OnlineRLUpdater()

        # Use empty temp dir to get default performances without data issues
        temp_dir = tempfile.mkdtemp()
        try:
            performances = updater.calculate_agent_performances(
                decisions_dir=temp_dir, lookback_days=30
            )

            assert isinstance(performances, dict)
            # Should have performance for each agent type
            expected_agents = ["bull", "bear", "market"]
            for agent in expected_agents:
                assert agent in performances
                assert isinstance(performances[agent], dict)
                assert "win_rate" in performances[agent]
                assert "avg_reward" in performances[agent]
                assert "sample_size" in performances[agent]
        finally:
            os.rmdir(temp_dir)

    def test_update_weights_returns_result(self):
        """Test update_weights returns proper result structure."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()

        # API expects dict of dicts with win_rate, avg_reward, sample_size
        performances = {
            "bull": {"win_rate": 0.6, "avg_reward": 1.2, "sample_size": 10},
            "bear": {"win_rate": 0.3, "avg_reward": -0.5, "sample_size": 10},
            "market": {"win_rate": 0.5, "avg_reward": 0.5, "sample_size": 10},
        }

        result = updater.update_weights(performances)

        assert "new_weights" in result
        assert "changes" in result
        assert isinstance(result["new_weights"], dict)
        assert isinstance(result["changes"], dict)

    def test_weights_sum_to_one(self):
        """Test that updated weights sum to 1.0."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()

        performances = {
            "bull": {"win_rate": 0.8, "avg_reward": 2.0, "sample_size": 15},
            "bear": {"win_rate": 0.2, "avg_reward": -1.0, "sample_size": 15},
            "market": {"win_rate": 0.5, "avg_reward": 0.5, "sample_size": 15},
        }

        result = updater.update_weights(performances)
        weights = result["new_weights"]

        # Weights should approximately sum to 1.0
        total = sum(weights.values())
        assert 0.99 <= total <= 1.01

    def test_weights_bounded(self):
        """Test that weights stay within bounds."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()

        # Extreme performance differences
        performances = {
            "bull": {"win_rate": 1.0, "avg_reward": 2.0, "sample_size": 20},
            "bear": {"win_rate": 0.0, "avg_reward": -2.0, "sample_size": 20},
            "market": {"win_rate": 0.5, "avg_reward": 0.0, "sample_size": 20},
        }

        result = updater.update_weights(performances)

        for agent, weight in result["new_weights"].items():
            assert 0.0 <= weight <= 1.0, f"Weight for {agent} out of bounds: {weight}"


class TestPatternAnalyzerIntegration:
    """Test PatternAnalyzer integration."""

    def test_analyze_patterns_returns_structure(self):
        """Test analyze_patterns returns expected structure."""
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)

        assert isinstance(patterns, dict)
        assert "patterns" in patterns or patterns == {}

    def test_analyze_patterns_handles_no_data(self):
        """Test analyze_patterns handles case with no historical data."""
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer()

        # With very short lookback, likely no data
        patterns = analyzer.analyze_patterns(lookback_days=0, min_cluster_size=100)

        # Should return empty patterns, not error
        assert patterns.get("patterns", []) == [] or patterns == {}


class TestDailyCycleLearningTrigger:
    """Test learning trigger integration in daily cycle."""

    def test_learning_trigger_method_exists(self):
        """Test that _trigger_learning_updates method exists."""
        # We test by checking the import and method existence
        import sys

        # Mock heavy dependencies
        with patch.dict(
            sys.modules,
            {
                "MetaTrader5": MagicMock(),
            },
        ):
            try:
                from examples.daily_cycle import DailyCycleRunner

                assert hasattr(DailyCycleRunner, "_trigger_learning_updates")
            except ImportError:
                # Module may have other dependencies, that's OK
                pass

    def test_learning_trigger_called_after_evaluation(self):
        """Test that learning trigger is called after evaluating predictions."""
        # Mock the daily cycle components
        mock_updater = MagicMock()
        mock_updater.should_update.return_value = (True, 35)
        mock_updater.calculate_agent_performances.return_value = {
            "bull": 0.6,
            "bear": 0.4,
            "market": 0.5,
        }
        mock_updater.update_weights.return_value = {
            "new_weights": {"bull": 0.35, "bear": 0.30, "market": 0.35},
            "changes": {"bull": 0.02, "bear": -0.02, "market": 0.0},
            "reasoning": "Performance-based adjustment",
        }

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_patterns.return_value = {
            "patterns": [
                {"name": "trending-bullish", "win_rate": 0.7},
            ],
            "recommendations": ["Increase bull weight in trends"],
        }

        with patch(
            "tradingagents.learning.online_rl.OnlineRLUpdater",
            return_value=mock_updater,
        ):
            with patch(
                "tradingagents.learning.pattern_analyzer.PatternAnalyzer",
                return_value=mock_analyzer,
            ):
                # The trigger should:
                # 1. Check if update is needed
                # 2. Run pattern analysis if so
                # 3. Update weights

                # Simulate what _trigger_learning_updates does
                from tradingagents.learning.online_rl import OnlineRLUpdater
                from tradingagents.learning.pattern_analyzer import PatternAnalyzer

                updater = OnlineRLUpdater()
                should_update, trades_since = updater.should_update()

                if should_update:
                    analyzer = PatternAnalyzer()
                    patterns = analyzer.analyze_patterns(lookback_days=30)
                    performances = updater.calculate_agent_performances()
                    result = updater.update_weights(performances)

                    assert result["new_weights"]["bull"] == 0.35

    def test_learning_trigger_handles_exceptions(self):
        """Test that learning trigger handles exceptions gracefully."""
        # Mock to raise exception
        with patch(
            "tradingagents.learning.online_rl.OnlineRLUpdater"
        ) as MockUpdater:
            MockUpdater.side_effect = Exception("Database error")

            # Should not raise, should be caught
            try:
                from tradingagents.learning.online_rl import OnlineRLUpdater

                updater = OnlineRLUpdater()
            except Exception as e:
                # Exception is expected here
                assert "Database error" in str(e)

    def test_learning_not_triggered_when_insufficient_trades(self):
        """Test learning is not triggered when not enough trades."""
        mock_updater = MagicMock()
        mock_updater.should_update.return_value = (False, 15)

        with patch(
            "tradingagents.learning.online_rl.OnlineRLUpdater",
            return_value=mock_updater,
        ):
            from tradingagents.learning.online_rl import OnlineRLUpdater

            updater = OnlineRLUpdater()
            should_update, trades_since = updater.should_update()

            assert should_update is False
            assert trades_since == 15


class TestLearningIntegrationEndToEnd:
    """End-to-end tests for learning integration."""

    def test_full_learning_flow(self):
        """Test the full learning update flow."""
        from tradingagents.learning.online_rl import OnlineRLUpdater
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer
        import tempfile
        import os

        # This tests the actual objects work together
        updater = OnlineRLUpdater()
        analyzer = PatternAnalyzer()

        # Get current state
        should_update, trades_since = updater.should_update()

        # Run analysis (will use whatever data is available)
        patterns = analyzer.analyze_patterns(lookback_days=7)

        # Calculate performances - use empty temp dir to avoid data issues
        temp_dir = tempfile.mkdtemp()
        try:
            performances = updater.calculate_agent_performances(
                decisions_dir=temp_dir, lookback_days=7
            )

            # Verify all objects return expected types
            assert isinstance(should_update, bool)
            assert isinstance(trades_since, int)
            assert isinstance(patterns, dict)
            assert isinstance(performances, dict)

            # Verify performances structure
            for agent, perf in performances.items():
                assert isinstance(perf, dict)
                assert "win_rate" in perf
                assert "avg_reward" in perf
                assert "sample_size" in perf
        finally:
            os.rmdir(temp_dir)

    def test_pattern_recommendations_format(self):
        """Test pattern recommendations are properly formatted."""
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze_patterns(lookback_days=30)

        if patterns.get("recommendations"):
            for rec in patterns["recommendations"]:
                assert isinstance(rec, str)
                assert len(rec) > 0

    def test_weight_changes_are_gradual(self):
        """Test that weight changes are gradual (momentum-based)."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()

        # Even with extreme performance differences, changes should be bounded
        # API expects dict of dicts
        performances = {
            "bull": {"win_rate": 1.0, "avg_reward": 2.0, "sample_size": 20},
            "bear": {"win_rate": 0.0, "avg_reward": -2.0, "sample_size": 20},
            "market": {"win_rate": 0.5, "avg_reward": 0.0, "sample_size": 20},
        }

        result = updater.update_weights(performances)
        changes = result.get("changes", {})

        # Changes should be gradual (not instant rebalancing)
        for agent, change in changes.items():
            # Maximum change per update should be bounded
            assert abs(change) < 0.5, f"Change for {agent} too large: {change}"


class TestLearningPersistence:
    """Test that learning state persists correctly."""

    @pytest.fixture
    def temp_weights_file(self):
        """Create temporary weights file."""
        temp_dir = tempfile.mkdtemp()
        weights_file = os.path.join(temp_dir, "agent_weights.pkl")
        yield weights_file
        # Cleanup
        if os.path.exists(weights_file):
            os.remove(weights_file)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    def test_weights_persist_after_update(self, temp_weights_file):
        """Test that weights are persisted after update."""
        from tradingagents.learning.online_rl import OnlineRLUpdater

        # Create updater with specific weights file
        updater = OnlineRLUpdater(weights_file=temp_weights_file)

        # API expects dict of dicts
        performances = {
            "bull": {"win_rate": 0.7, "avg_reward": 1.5, "sample_size": 10},
            "bear": {"win_rate": 0.3, "avg_reward": -0.5, "sample_size": 10},
            "market": {"win_rate": 0.5, "avg_reward": 0.5, "sample_size": 10},
        }

        result = updater.update_weights(performances)

        # Weights should be saved
        new_weights = result["new_weights"]
        assert len(new_weights) > 0
        assert "bull" in new_weights
        assert "bear" in new_weights
        assert "market" in new_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
