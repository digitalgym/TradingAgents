"""
Pattern Analyzer for Trade Clustering

Analyzes historical trades to identify patterns and clusters.
Runs periodically (every 30 trades) to discover what's working and what's not.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta


class PatternAnalyzer:
    """Analyze trade patterns and identify successful/unsuccessful clusters"""
    
    def __init__(self, decisions_dir: Optional[str] = None):
        """
        Initialize pattern analyzer.
        
        Args:
            decisions_dir: Optional custom path to decisions directory
        """
        if decisions_dir is None:
            from tradingagents.trade_decisions import DECISIONS_DIR
            self.decisions_dir = DECISIONS_DIR
        else:
            self.decisions_dir = decisions_dir
    
    def analyze_patterns(
        self,
        lookback_days: int = 30,
        min_cluster_size: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze recent trade patterns and identify clusters.
        
        Args:
            lookback_days: Number of days to look back
            min_cluster_size: Minimum trades to form a pattern
        
        Returns:
            {
                "patterns": [...],
                "recommendations": [...],
                "statistics": {...}
            }
        """
        # Load recent trades
        trades = self._load_recent_trades(lookback_days)
        
        if len(trades) < min_cluster_size:
            return {
                "patterns": [],
                "recommendations": ["Insufficient data for pattern analysis"],
                "statistics": {"total_trades": len(trades)}
            }
        
        # Analyze by different dimensions
        patterns = []
        
        # 1. Setup type patterns
        setup_patterns = self._analyze_by_setup_type(trades, min_cluster_size)
        patterns.extend(setup_patterns)
        
        # 2. Regime patterns
        regime_patterns = self._analyze_by_regime(trades, min_cluster_size)
        patterns.extend(regime_patterns)
        
        # 3. Time-of-day patterns
        time_patterns = self._analyze_by_time(trades, min_cluster_size)
        patterns.extend(time_patterns)
        
        # 4. Confluence score patterns
        confluence_patterns = self._analyze_by_confluence(trades, min_cluster_size)
        patterns.extend(confluence_patterns)
        
        # Sort patterns by impact (win rate * sample size)
        patterns.sort(key=lambda p: p.get("impact", 0), reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)
        
        # Calculate overall statistics
        statistics = self._calculate_statistics(trades, patterns)
        
        return {
            "patterns": patterns,
            "recommendations": recommendations,
            "statistics": statistics,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _load_recent_trades(self, lookback_days: int) -> List[Dict[str, Any]]:
        """Load closed trades from recent period."""
        if not os.path.exists(self.decisions_dir):
            return []
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        trades = []
        
        for filename in os.listdir(self.decisions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.decisions_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        trade = json.load(f)
                        
                        if trade.get("status") != "closed":
                            continue
                        
                        # Check if within lookback period
                        close_time = trade.get("close_time")
                        if close_time:
                            trade_date = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                            if trade_date >= cutoff_date:
                                trades.append(trade)
                except Exception:
                    continue
        
        return trades
    
    def _analyze_by_setup_type(
        self,
        trades: List[Dict[str, Any]],
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Analyze patterns by setup type."""
        patterns = []
        
        # Group by setup type
        by_setup = defaultdict(list)
        for trade in trades:
            setup = trade.get("setup_type")
            if setup:
                by_setup[setup].append(trade)
        
        # Analyze each setup type
        for setup_type, setup_trades in by_setup.items():
            if len(setup_trades) >= min_size:
                pattern = self._create_pattern(
                    pattern_type="setup_type",
                    pattern_value=setup_type,
                    trades=setup_trades
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_by_regime(
        self,
        trades: List[Dict[str, Any]],
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Analyze patterns by market regime."""
        patterns = []
        
        # Group by regime combination
        by_regime = defaultdict(list)
        for trade in trades:
            market = trade.get("market_regime")
            volatility = trade.get("volatility_regime")
            if market and volatility:
                regime_key = f"{market}/{volatility}"
                by_regime[regime_key].append(trade)
        
        # Analyze each regime
        for regime_key, regime_trades in by_regime.items():
            if len(regime_trades) >= min_size:
                pattern = self._create_pattern(
                    pattern_type="regime",
                    pattern_value=regime_key,
                    trades=regime_trades
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_by_time(
        self,
        trades: List[Dict[str, Any]],
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Analyze patterns by time of day."""
        patterns = []
        
        # Group by hour of day
        by_hour = defaultdict(list)
        for trade in trades:
            timestamp = trade.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    session = self._get_trading_session(hour)
                    by_hour[session].append(trade)
                except Exception:
                    continue
        
        # Analyze each session
        for session, session_trades in by_hour.items():
            if len(session_trades) >= min_size:
                pattern = self._create_pattern(
                    pattern_type="time_session",
                    pattern_value=session,
                    trades=session_trades
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_by_confluence(
        self,
        trades: List[Dict[str, Any]],
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Analyze patterns by confluence score."""
        patterns = []
        
        # Group by confluence range
        by_confluence = defaultdict(list)
        for trade in trades:
            score = trade.get("confluence_score")
            if score is not None:
                if score >= 8:
                    category = "high-confluence (8-10)"
                elif score >= 6:
                    category = "medium-confluence (6-7)"
                else:
                    category = "low-confluence (0-5)"
                by_confluence[category].append(trade)
        
        # Analyze each category
        for category, conf_trades in by_confluence.items():
            if len(conf_trades) >= min_size:
                pattern = self._create_pattern(
                    pattern_type="confluence",
                    pattern_value=category,
                    trades=conf_trades
                )
                patterns.append(pattern)
        
        return patterns
    
    def _create_pattern(
        self,
        pattern_type: str,
        pattern_value: str,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create pattern summary from trades."""
        wins = sum(1 for t in trades if t.get("was_correct"))
        win_rate = wins / len(trades)
        
        # Calculate average RR
        rr_values = [t.get("rr_realized") for t in trades if t.get("rr_realized") is not None]
        avg_rr = np.mean(rr_values) if rr_values else 0.0
        
        # Calculate average reward
        rewards = [t.get("reward_signal") for t in trades if t.get("reward_signal") is not None]
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Calculate impact score (win_rate * sample_size * avg_rr)
        impact = win_rate * len(trades) * max(avg_rr, 0.5)
        
        # Determine quality
        if win_rate >= 0.65 and avg_rr > 1.5:
            quality = "excellent"
        elif win_rate >= 0.55 and avg_rr > 1.0:
            quality = "good"
        elif win_rate >= 0.45:
            quality = "neutral"
        else:
            quality = "poor"
        
        return {
            "pattern_type": pattern_type,
            "pattern_value": pattern_value,
            "sample_size": len(trades),
            "win_rate": float(win_rate),
            "avg_rr": float(avg_rr),
            "avg_reward": float(avg_reward),
            "impact": float(impact),
            "quality": quality
        }
    
    def _get_trading_session(self, hour: int) -> str:
        """Map hour to trading session."""
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "london"
        elif 16 <= hour < 24:
            return "newyork"
        else:
            return "unknown"
    
    def _generate_recommendations(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations from patterns."""
        recommendations = []
        
        # Find best and worst patterns
        excellent = [p for p in patterns if p["quality"] == "excellent"]
        poor = [p for p in patterns if p["quality"] == "poor"]
        
        # Recommendations for excellent patterns
        for pattern in excellent[:3]:  # Top 3
            rec = f"✓ INCREASE focus on {pattern['pattern_value']} "
            rec += f"({pattern['win_rate']*100:.0f}% win rate, {pattern['avg_rr']:.1f}R avg)"
            recommendations.append(rec)
        
        # Recommendations for poor patterns
        for pattern in poor[:3]:  # Top 3 worst
            rec = f"✗ AVOID or REDUCE {pattern['pattern_value']} "
            rec += f"({pattern['win_rate']*100:.0f}% win rate, {pattern['avg_rr']:.1f}R avg)"
            recommendations.append(rec)
        
        # Sample size warnings
        small_samples = [p for p in patterns if p["sample_size"] < 5]
        if small_samples:
            recommendations.append(f"⚠️  {len(small_samples)} patterns have <5 trades - need more data")
        
        return recommendations
    
    def _calculate_statistics(
        self,
        trades: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall statistics."""
        wins = sum(1 for t in trades if t.get("was_correct"))
        win_rate = wins / len(trades) if trades else 0.0
        
        rr_values = [t.get("rr_realized") for t in trades if t.get("rr_realized") is not None]
        avg_rr = np.mean(rr_values) if rr_values else 0.0
        
        rewards = [t.get("reward_signal") for t in trades if t.get("reward_signal") is not None]
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Pattern quality distribution
        quality_dist = defaultdict(int)
        for pattern in patterns:
            quality_dist[pattern["quality"]] += 1
        
        return {
            "total_trades": len(trades),
            "overall_win_rate": float(win_rate),
            "overall_avg_rr": float(avg_rr),
            "overall_avg_reward": float(avg_reward),
            "patterns_found": len(patterns),
            "excellent_patterns": quality_dist.get("excellent", 0),
            "good_patterns": quality_dist.get("good", 0),
            "neutral_patterns": quality_dist.get("neutral", 0),
            "poor_patterns": quality_dist.get("poor", 0)
        }
    
    def format_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as human-readable report."""
        stats = analysis["statistics"]
        patterns = analysis["patterns"]
        recommendations = analysis["recommendations"]
        
        report = f"""
PATTERN ANALYSIS REPORT
Generated: {analysis.get('analysis_date', 'N/A')}

OVERALL PERFORMANCE:
- Total Trades: {stats['total_trades']}
- Win Rate: {stats['overall_win_rate']*100:.1f}%
- Avg RR: {stats['overall_avg_rr']:.2f}
- Avg Reward: {stats['overall_avg_reward']:+.2f}

PATTERNS IDENTIFIED: {stats['patterns_found']}
- Excellent: {stats['excellent_patterns']}
- Good: {stats['good_patterns']}
- Neutral: {stats['neutral_patterns']}
- Poor: {stats['poor_patterns']}

TOP PATTERNS:
"""
        
        for i, pattern in enumerate(patterns[:5], 1):
            quality_emoji = {
                "excellent": "🌟",
                "good": "✓",
                "neutral": "→",
                "poor": "✗"
            }.get(pattern["quality"], "?")
            
            report += f"\n{i}. {quality_emoji} {pattern['pattern_value']} ({pattern['pattern_type']})\n"
            report += f"   Win Rate: {pattern['win_rate']*100:.1f}% | "
            report += f"Avg RR: {pattern['avg_rr']:.2f} | "
            report += f"Sample: {pattern['sample_size']} trades\n"
        
        report += "\nRECOMMENDATIONS:\n"
        for rec in recommendations:
            report += f"  {rec}\n"
        
        return report
