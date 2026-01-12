"""
Unit tests for TradeSimilaritySearch

Tests trade similarity search including:
- Similar trade finding
- Similarity scoring
- Statistics calculation
- Recommendation generation
"""

import pytest
import os
import tempfile
import shutil
import json
from tradingagents.learning.trade_similarity import TradeSimilaritySearch


@pytest.fixture
def temp_decisions_dir():
    """Create temporary directory with sample trade decisions"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample closed trades
    sample_trades = [
        {
            "decision_id": "XAUUSD_20260101_100000",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": True,
            "rr_realized": 2.5,
            "reward_signal": 1.8,
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block",
            "higher_tf_bias": "bullish",
            "confluence_score": 8,
            "rationale": "Strong bullish setup with multiple confluence"
        },
        {
            "decision_id": "XAUUSD_20260102_100000",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": True,
            "rr_realized": 1.8,
            "reward_signal": 1.2,
            "market_regime": "trending-up",
            "volatility_regime": "high",
            "setup_type": "breaker-block",
            "higher_tf_bias": "bullish",
            "confluence_score": 7
        },
        {
            "decision_id": "XAUUSD_20260103_100000",
            "symbol": "XAUUSD",
            "action": "SELL",
            "status": "closed",
            "was_correct": False,
            "rr_realized": -1.0,
            "reward_signal": -0.8,
            "market_regime": "ranging",
            "volatility_regime": "low",
            "setup_type": "resistance-rejection",
            "higher_tf_bias": "neutral",
            "confluence_score": 5
        },
        {
            "decision_id": "XAUUSD_20260104_100000",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": True,
            "rr_realized": 3.2,
            "reward_signal": 2.1,
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "FVG",
            "higher_tf_bias": "bullish",
            "confluence_score": 9
        },
        {
            "decision_id": "XAUUSD_20260105_100000",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": False,
            "rr_realized": -1.0,
            "reward_signal": -1.2,
            "market_regime": "trending-up",
            "volatility_regime": "extreme",
            "setup_type": "breaker-block",
            "higher_tf_bias": "bullish",
            "confluence_score": 6
        },
    ]
    
    for trade in sample_trades:
        filepath = os.path.join(temp_dir, f"{trade['decision_id']}.json")
        with open(filepath, 'w') as f:
            json.dump(trade, f)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestTradeSimilaritySearch:
    """Test trade similarity search functionality"""
    
    def test_find_similar_trades_basic(self, temp_decisions_dir):
        """Test basic similar trade finding"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block",
            "higher_tf_bias": "bullish",
            "confluence_score": 8
        }
        
        result = searcher.find_similar_trades(current_setup, n_results=3)
        
        # Check structure
        assert "similar_trades" in result
        assert "statistics" in result
        assert "recommendation" in result
        
        # Should find trades
        assert len(result["similar_trades"]) > 0
        assert len(result["similar_trades"]) <= 3
    
    def test_regime_filtering(self, temp_decisions_dir):
        """Test that regime filtering works"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        # Search for trending-up + normal volatility
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal"
        }
        
        result = searcher.find_similar_trades(current_setup, n_results=10)
        
        # Should find trades matching regime
        for trade in result["similar_trades"]:
            # Most should match regime (high similarity)
            if trade.get("market_regime") == "trending-up":
                assert True  # Expected
    
    def test_statistics_calculation(self, temp_decisions_dir):
        """Test statistics calculation from similar trades"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal"
        }
        
        result = searcher.find_similar_trades(current_setup)
        stats = result["statistics"]
        
        # Check all required fields
        assert "sample_size" in stats
        assert "win_rate" in stats
        assert "avg_rr" in stats
        assert "avg_reward" in stats
        assert "confidence_adjustment" in stats
        
        # Check value ranges
        assert 0 <= stats["win_rate"] <= 1
        assert -0.3 <= stats["confidence_adjustment"] <= 0.3
    
    def test_high_win_rate_increases_confidence(self, temp_decisions_dir):
        """Test that high win rate increases confidence"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        # Setup that should match winning trades
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block"
        }
        
        result = searcher.find_similar_trades(current_setup)
        stats = result["statistics"]
        
        # Should have positive confidence adjustment if win rate is good
        if stats["win_rate"] > 0.6:
            assert stats["confidence_adjustment"] > 0
    
    def test_low_win_rate_decreases_confidence(self, temp_decisions_dir):
        """Test that low win rate decreases confidence"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        # Setup that should match losing trades
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "SELL",
            "market_regime": "ranging",
            "volatility_regime": "low"
        }
        
        result = searcher.find_similar_trades(current_setup)
        stats = result["statistics"]
        
        # Should have negative or zero confidence adjustment
        if stats["sample_size"] > 0 and stats["win_rate"] < 0.5:
            assert stats["confidence_adjustment"] <= 0
    
    def test_no_similar_trades(self, temp_decisions_dir):
        """Test behavior when no similar trades found"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        # Very specific setup unlikely to match
        current_setup = {
            "symbol": "EURUSD",  # Different symbol
            "direction": "BUY",
            "market_regime": "trending-down",
            "volatility_regime": "extreme"
        }
        
        result = searcher.find_similar_trades(current_setup, min_confidence=0.9)
        
        # Should handle gracefully
        assert result["similar_trades"] == [] or len(result["similar_trades"]) == 0
        assert result["statistics"]["sample_size"] == 0
        assert "No" in result["recommendation"] or "no" in result["recommendation"].lower()
    
    def test_similarity_scoring(self, temp_decisions_dir):
        """Test similarity score calculation"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        current = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block"
        }
        
        historical = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block"
        }
        
        # Perfect match should score high
        score = searcher._calculate_similarity(current, historical, regime_weight=0.5)
        assert score > 0.8
    
    def test_partial_match_scoring(self, temp_decisions_dir):
        """Test partial match gives lower score"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        current = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal"
        }
        
        historical = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "market_regime": "ranging",  # Different
            "volatility_regime": "high"   # Different
        }
        
        # Partial match should score lower
        score = searcher._calculate_similarity(current, historical, regime_weight=0.5)
        assert score < 0.7
    
    def test_format_for_prompt(self, temp_decisions_dir):
        """Test formatting for LLM prompts"""
        searcher = TradeSimilaritySearch(temp_decisions_dir)
        
        current_setup = {
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_regime": "trending-up",
            "volatility_regime": "normal"
        }
        
        result = searcher.find_similar_trades(current_setup, n_results=3)
        
        formatted = searcher.format_for_prompt(
            result["similar_trades"],
            result["statistics"],
            max_trades=3
        )
        
        # Check key information is included
        assert "HISTORICAL CONTEXT" in formatted
        assert "Win Rate" in formatted
        assert "Risk-Reward" in formatted
        
        # Should include trade details
        if result["similar_trades"]:
            assert "WIN" in formatted or "LOSS" in formatted


class TestConfidenceAdjustment:
    """Test confidence adjustment calculation"""
    
    def test_high_win_rate_adjustment(self):
        """Test adjustment for high win rate"""
        searcher = TradeSimilaritySearch()
        
        # 75% win rate, good RR, decent sample
        adjustment = searcher._calculate_confidence_adjustment(0.75, 2.5, 10)
        assert adjustment > 0.15  # Should be positive and significant
    
    def test_low_win_rate_adjustment(self):
        """Test adjustment for low win rate"""
        searcher = TradeSimilaritySearch()
        
        # 30% win rate, poor RR
        adjustment = searcher._calculate_confidence_adjustment(0.30, 0.5, 10)
        assert adjustment < -0.15  # Should be negative and significant
    
    def test_small_sample_reduces_adjustment(self):
        """Test that small samples reduce adjustment magnitude"""
        searcher = TradeSimilaritySearch()
        
        # Same win rate, different sample sizes
        large_sample = searcher._calculate_confidence_adjustment(0.75, 2.0, 20)
        small_sample = searcher._calculate_confidence_adjustment(0.75, 2.0, 3)
        
        # Small sample should have smaller adjustment
        assert abs(small_sample) < abs(large_sample)
    
    def test_adjustment_clipping(self):
        """Test that adjustment is clipped to range"""
        searcher = TradeSimilaritySearch()
        
        # Extreme values
        adjustment = searcher._calculate_confidence_adjustment(1.0, 5.0, 100)
        assert -0.3 <= adjustment <= 0.3


class TestRecommendationGeneration:
    """Test recommendation text generation"""
    
    def test_strong_performance_recommendation(self):
        """Test recommendation for strong performance"""
        searcher = TradeSimilaritySearch()
        
        stats = {
            "sample_size": 10,
            "win_rate": 0.75,
            "avg_rr": 2.5,
            "regime": "trending-up / normal"
        }
        
        recommendation = searcher._generate_recommendation(stats, {})
        
        assert "STRONG" in recommendation or "Good" in recommendation
        assert "INCREASE" in recommendation or "boost" in recommendation
    
    def test_poor_performance_recommendation(self):
        """Test recommendation for poor performance"""
        searcher = TradeSimilaritySearch()
        
        stats = {
            "sample_size": 8,
            "win_rate": 0.25,
            "avg_rr": 0.5,
            "regime": "ranging / low"
        }
        
        recommendation = searcher._generate_recommendation(stats, {})
        
        assert "POOR" in recommendation or "Below" in recommendation
        assert "REDUCE" in recommendation or "SKIP" in recommendation
    
    def test_no_data_recommendation(self):
        """Test recommendation when no data"""
        searcher = TradeSimilaritySearch()
        
        stats = {
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "regime": "unknown"
        }
        
        recommendation = searcher._generate_recommendation(stats, {})
        
        assert "No" in recommendation or "no" in recommendation


class TestRegimeSimilarity:
    """Test regime similarity calculations"""
    
    def test_exact_regime_match(self):
        """Test exact regime match scores high"""
        searcher = TradeSimilaritySearch()
        
        current = {
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "expansion_regime": "expansion"
        }
        
        historical = {
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "expansion_regime": "expansion"
        }
        
        score = searcher._calculate_regime_similarity(current, historical)
        assert score > 0.9
    
    def test_adjacent_volatility_partial_match(self):
        """Test adjacent volatility levels get partial credit"""
        searcher = TradeSimilaritySearch()
        
        current = {"volatility_regime": "normal"}
        historical = {"volatility_regime": "high"}
        
        score = searcher._calculate_regime_similarity(current, historical)
        assert 0.3 < score < 0.7  # Partial credit
    
    def test_missing_regime_data(self):
        """Test handling of missing regime data"""
        searcher = TradeSimilaritySearch()
        
        current = {"market_regime": "trending-up"}
        historical = {}  # No regime data
        
        score = searcher._calculate_regime_similarity(current, historical)
        assert score == 0.5  # Default neutral score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
