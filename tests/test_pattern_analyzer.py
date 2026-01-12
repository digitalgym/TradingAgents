"""
Unit tests for PatternAnalyzer

Tests pattern analysis including:
- Pattern clustering by setup type
- Pattern clustering by regime
- Pattern clustering by time/confluence
- Recommendation generation
"""

import pytest
import os
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from tradingagents.learning.pattern_analyzer import PatternAnalyzer


@pytest.fixture
def temp_decisions_dir():
    """Create temporary directory with sample trade decisions"""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample closed trades with various patterns
    base_time = datetime.now() - timedelta(days=20)
    
    sample_trades = [
        # Excellent breaker block pattern
        *[{
            "decision_id": f"XAUUSD_BB_{i}",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": True,
            "rr_realized": 2.5 + i * 0.1,
            "reward_signal": 1.8,
            "market_regime": "trending-up",
            "volatility_regime": "normal",
            "setup_type": "breaker-block",
            "confluence_score": 8,
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "close_time": (base_time + timedelta(days=i, hours=4)).isoformat()
        } for i in range(5)],
        
        # Poor resistance rejection pattern
        *[{
            "decision_id": f"XAUUSD_RR_{i}",
            "symbol": "XAUUSD",
            "action": "SELL",
            "status": "closed",
            "was_correct": False,
            "rr_realized": -1.0,
            "reward_signal": -0.8,
            "market_regime": "ranging",
            "volatility_regime": "low",
            "setup_type": "resistance-rejection",
            "confluence_score": 5,
            "timestamp": (base_time + timedelta(days=i + 10)).isoformat(),
            "close_time": (base_time + timedelta(days=i + 10, hours=4)).isoformat()
        } for i in range(4)],
        
        # Mixed FVG pattern
        *[{
            "decision_id": f"XAUUSD_FVG_{i}",
            "symbol": "XAUUSD",
            "action": "BUY",
            "status": "closed",
            "was_correct": i % 2 == 0,
            "rr_realized": 1.5 if i % 2 == 0 else -1.0,
            "reward_signal": 1.0 if i % 2 == 0 else -0.5,
            "market_regime": "trending-up",
            "volatility_regime": "high",
            "setup_type": "FVG",
            "confluence_score": 7,
            "timestamp": (base_time + timedelta(days=i + 15)).isoformat(),
            "close_time": (base_time + timedelta(days=i + 15, hours=4)).isoformat()
        } for i in range(4)],
    ]
    
    for trade in sample_trades:
        filepath = os.path.join(temp_dir, f"{trade['decision_id']}.json")
        with open(filepath, 'w') as f:
            json.dump(trade, f)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestPatternAnalysis:
    """Test pattern analysis functionality"""
    
    def test_analyze_patterns_basic(self, temp_decisions_dir):
        """Test basic pattern analysis"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Check structure
        assert "patterns" in analysis
        assert "recommendations" in analysis
        assert "statistics" in analysis
        
        # Should find patterns
        assert len(analysis["patterns"]) > 0
    
    def test_setup_type_patterns(self, temp_decisions_dir):
        """Test pattern clustering by setup type"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Should find breaker-block, resistance-rejection, FVG patterns
        setup_patterns = [p for p in analysis["patterns"] if p["pattern_type"] == "setup_type"]
        assert len(setup_patterns) >= 3
        
        # Check pattern values
        setup_values = [p["pattern_value"] for p in setup_patterns]
        assert "breaker-block" in setup_values
        assert "resistance-rejection" in setup_values
        assert "FVG" in setup_values
    
    def test_regime_patterns(self, temp_decisions_dir):
        """Test pattern clustering by regime"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Should find regime patterns
        regime_patterns = [p for p in analysis["patterns"] if p["pattern_type"] == "regime"]
        assert len(regime_patterns) > 0
        
        # Check regime format
        for pattern in regime_patterns:
            assert "/" in pattern["pattern_value"]  # e.g., "trending-up/normal"
    
    def test_pattern_quality_classification(self, temp_decisions_dir):
        """Test pattern quality classification"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Should have patterns with different qualities
        qualities = [p["quality"] for p in analysis["patterns"]]
        assert "excellent" in qualities or "good" in qualities
        assert "poor" in qualities or "neutral" in qualities
    
    def test_excellent_pattern_criteria(self, temp_decisions_dir):
        """Test excellent pattern identification"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Breaker block should be excellent (100% win rate, good RR)
        bb_pattern = next((p for p in analysis["patterns"] 
                          if p.get("pattern_value") == "breaker-block"), None)
        
        if bb_pattern:
            assert bb_pattern["win_rate"] >= 0.65
            assert bb_pattern["quality"] in ["excellent", "good"]
    
    def test_poor_pattern_criteria(self, temp_decisions_dir):
        """Test poor pattern identification"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Resistance rejection should be poor (0% win rate)
        rr_pattern = next((p for p in analysis["patterns"] 
                          if p.get("pattern_value") == "resistance-rejection"), None)
        
        if rr_pattern:
            assert rr_pattern["win_rate"] < 0.5
            assert rr_pattern["quality"] in ["poor", "neutral"]
    
    def test_recommendations_generation(self, temp_decisions_dir):
        """Test recommendation generation"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        recommendations = analysis["recommendations"]
        assert len(recommendations) > 0
        
        # Should have both positive and negative recommendations
        rec_text = " ".join(recommendations)
        assert "INCREASE" in rec_text or "AVOID" in rec_text or "REDUCE" in rec_text
    
    def test_statistics_calculation(self, temp_decisions_dir):
        """Test overall statistics calculation"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        stats = analysis["statistics"]
        
        # Check required fields
        assert "total_trades" in stats
        assert "overall_win_rate" in stats
        assert "overall_avg_rr" in stats
        assert "patterns_found" in stats
        
        # Check value ranges
        assert 0 <= stats["overall_win_rate"] <= 1
        assert stats["total_trades"] > 0
        assert stats["patterns_found"] > 0
    
    def test_min_cluster_size(self, temp_decisions_dir):
        """Test minimum cluster size filtering"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        
        # With high min size, should find fewer patterns
        analysis_strict = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=5)
        analysis_lenient = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        assert len(analysis_strict["patterns"]) <= len(analysis_lenient["patterns"])
    
    def test_lookback_period(self, temp_decisions_dir):
        """Test lookback period filtering"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        
        # Short lookback should find fewer trades
        analysis_short = analyzer.analyze_patterns(lookback_days=5, min_cluster_size=2)
        analysis_long = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        assert analysis_short["statistics"]["total_trades"] <= analysis_long["statistics"]["total_trades"]
    
    def test_format_report(self, temp_decisions_dir):
        """Test report formatting"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        report = analyzer.format_report(analysis)
        
        # Check key sections present
        assert "PATTERN ANALYSIS REPORT" in report
        assert "OVERALL PERFORMANCE" in report
        assert "PATTERNS IDENTIFIED" in report
        assert "TOP PATTERNS" in report
        assert "RECOMMENDATIONS" in report
    
    def test_empty_directory(self):
        """Test behavior with no trades"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            analyzer = PatternAnalyzer(temp_dir)
            analysis = analyzer.analyze_patterns(lookback_days=30)
            
            # Should handle gracefully
            assert analysis["patterns"] == []
            assert "Insufficient data" in analysis["recommendations"][0]
        finally:
            shutil.rmtree(temp_dir)
    
    def test_confluence_patterns(self, temp_decisions_dir):
        """Test confluence score pattern analysis"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Should find confluence patterns
        conf_patterns = [p for p in analysis["patterns"] if p["pattern_type"] == "confluence"]
        
        if conf_patterns:
            # Check categories
            categories = [p["pattern_value"] for p in conf_patterns]
            assert any("high-confluence" in c or "medium-confluence" in c or "low-confluence" in c 
                      for c in categories)


class TestPatternImpact:
    """Test pattern impact scoring"""
    
    def test_impact_calculation(self, temp_decisions_dir):
        """Test impact score calculation"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # All patterns should have impact scores
        for pattern in analysis["patterns"]:
            assert "impact" in pattern
            assert pattern["impact"] >= 0
    
    def test_pattern_sorting_by_impact(self, temp_decisions_dir):
        """Test patterns are sorted by impact"""
        analyzer = PatternAnalyzer(temp_decisions_dir)
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
        
        # Check patterns are sorted descending by impact
        impacts = [p["impact"] for p in analysis["patterns"]]
        assert impacts == sorted(impacts, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
