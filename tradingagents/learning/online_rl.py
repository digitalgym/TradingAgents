"""
Online Reinforcement Learning for Agent Weight Updates

Adjusts agent weights (bull/bear/market) based on recent performance.
Updates occur periodically (every 30 trades) to adapt to changing market conditions.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class OnlineRLUpdater:
    """Update agent weights based on performance using online learning"""
    
    DEFAULT_WEIGHTS = {
        "bull": 0.33,
        "bear": 0.33,
        "market": 0.34
    }
    
    def __init__(
        self,
        weights_file: Optional[str] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9
    ):
        """
        Initialize online RL updater.
        
        Args:
            weights_file: Path to save/load weights (default: tradingagents/examples/agent_weights.pkl)
            learning_rate: Learning rate for weight updates (0.0-1.0)
            momentum: Momentum factor for smoothing updates (0.0-1.0)
        """
        if weights_file is None:
            base_dir = Path(__file__).parent.parent.parent
            self.weights_file = base_dir / "examples" / "agent_weights.pkl"
        else:
            self.weights_file = Path(weights_file)
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Load or initialize weights
        self.weights = self._load_weights()
        self.weight_history = []
        self.velocity = {agent: 0.0 for agent in self.DEFAULT_WEIGHTS.keys()}
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from file or return defaults."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'rb') as f:
                    data = pickle.load(f)
                    return data.get("weights", self.DEFAULT_WEIGHTS.copy())
            except Exception:
                pass
        
        return self.DEFAULT_WEIGHTS.copy()
    
    def _save_weights(self):
        """Save weights to file."""
        self.weights_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "weights": self.weights,
            "weight_history": self.weight_history[-50:],  # Keep last 50
            "last_update": datetime.now().isoformat(),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        }
        
        with open(self.weights_file, 'wb') as f:
            pickle.dump(data, f)
    
    def update_weights(
        self,
        agent_performances: Dict[str, Dict[str, float]],
        decisions_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update agent weights based on recent performance.
        
        Args:
            agent_performances: {
                "bull": {"win_rate": 0.65, "avg_reward": 1.2, "sample_size": 20},
                "bear": {"win_rate": 0.45, "avg_reward": -0.3, "sample_size": 15},
                "market": {"win_rate": 0.55, "avg_reward": 0.5, "sample_size": 25}
            }
            decisions_dir: Optional decisions directory for analysis
        
        Returns:
            {
                "old_weights": {...},
                "new_weights": {...},
                "changes": {...},
                "reasoning": "..."
            }
        """
        old_weights = self.weights.copy()
        
        # Calculate performance scores for each agent
        scores = {}
        for agent, perf in agent_performances.items():
            # Score = win_rate * avg_reward * sqrt(sample_size)
            # This rewards both accuracy and positive outcomes, with sample size weighting
            win_rate = perf.get("win_rate", 0.5)
            avg_reward = perf.get("avg_reward", 0.0)
            sample_size = perf.get("sample_size", 1)
            
            # Normalize reward to 0-1 range (assuming -2 to +2 range)
            normalized_reward = (avg_reward + 2) / 4
            normalized_reward = np.clip(normalized_reward, 0, 1)
            
            # Calculate score with sample size weighting
            score = win_rate * normalized_reward * np.sqrt(sample_size)
            scores[agent] = score
        
        # Normalize scores to sum to 1
        total_score = sum(scores.values())
        if total_score > 0:
            target_weights = {agent: score / total_score for agent, score in scores.items()}
        else:
            target_weights = self.DEFAULT_WEIGHTS.copy()
        
        # Apply momentum-based update
        new_weights = {}
        changes = {}
        
        for agent in self.weights.keys():
            # Calculate gradient (difference between target and current)
            gradient = target_weights.get(agent, 0.33) - self.weights[agent]
            
            # Update velocity with momentum
            self.velocity[agent] = (
                self.momentum * self.velocity[agent] + 
                self.learning_rate * gradient
            )
            
            # Update weight
            new_weight = self.weights[agent] + self.velocity[agent]
            new_weights[agent] = new_weight
            changes[agent] = new_weight - self.weights[agent]
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        new_weights = {agent: w / total_weight for agent, w in new_weights.items()}
        
        # Recalculate changes after normalization
        changes = {agent: new_weights[agent] - old_weights[agent] for agent in new_weights.keys()}
        
        # Update stored weights
        self.weights = new_weights
        
        # Record in history
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": new_weights.copy(),
            "performances": agent_performances.copy()
        })
        
        # Save to file
        self._save_weights()
        
        # Generate reasoning
        reasoning = self._generate_reasoning(old_weights, new_weights, agent_performances)
        
        return {
            "old_weights": old_weights,
            "new_weights": new_weights,
            "changes": changes,
            "reasoning": reasoning,
            "agent_scores": scores
        }
    
    def _generate_reasoning(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        performances: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate human-readable reasoning for weight changes."""
        lines = []
        
        for agent in old_weights.keys():
            old = old_weights[agent]
            new = new_weights[agent]
            change = new - old
            perf = performances.get(agent, {})
            
            if abs(change) > 0.05:  # Significant change
                direction = "INCREASED" if change > 0 else "DECREASED"
                lines.append(
                    f"{agent.upper()}: {direction} from {old:.2f} to {new:.2f} "
                    f"(win rate: {perf.get('win_rate', 0)*100:.0f}%, "
                    f"avg reward: {perf.get('avg_reward', 0):+.2f})"
                )
        
        if not lines:
            return "No significant weight changes - all agents performing similarly"
        
        return "\n".join(lines)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current agent weights."""
        return self.weights.copy()
    
    def reset_weights(self):
        """Reset weights to defaults."""
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.velocity = {agent: 0.0 for agent in self.DEFAULT_WEIGHTS.keys()}
        self.weight_history = []
        self._save_weights()
    
    def get_weight_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent weight history."""
        return self.weight_history[-n:]
    
    def calculate_agent_performances(
        self,
        decisions_dir: Optional[str] = None,
        lookback_days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate agent performances from recent decisions.
        
        Args:
            decisions_dir: Path to decisions directory
            lookback_days: Number of days to analyze
        
        Returns:
            Agent performance dict
        """
        if decisions_dir is None:
            from tradingagents.trade_decisions import DECISIONS_DIR
            decisions_dir = DECISIONS_DIR
        
        if not os.path.exists(decisions_dir):
            return {agent: {"win_rate": 0.5, "avg_reward": 0.0, "sample_size": 0} 
                    for agent in self.DEFAULT_WEIGHTS.keys()}
        
        # Load recent closed trades
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Group trades by dominant agent
        agent_trades = {agent: [] for agent in self.DEFAULT_WEIGHTS.keys()}
        
        for filename in os.listdir(decisions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(decisions_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        trade = json.load(f)
                        
                        if trade.get("status") != "closed":
                            continue
                        
                        # Check if within lookback period
                        close_time = trade.get("close_time")
                        if close_time:
                            trade_date = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                            if trade_date < cutoff_date:
                                continue
                        
                        # Determine dominant agent (simplified - would need actual agent votes)
                        # For now, use action as proxy: BUY->bull, SELL->bear, HOLD->market
                        action = trade.get("action", "").upper()
                        if "BUY" in action:
                            dominant_agent = "bull"
                        elif "SELL" in action:
                            dominant_agent = "bear"
                        else:
                            dominant_agent = "market"
                        
                        agent_trades[dominant_agent].append(trade)
                
                except Exception:
                    continue
        
        # Calculate performance for each agent
        performances = {}
        for agent, trades in agent_trades.items():
            if not trades:
                performances[agent] = {
                    "win_rate": 0.5,
                    "avg_reward": 0.0,
                    "sample_size": 0
                }
                continue
            
            wins = sum(1 for t in trades if t.get("was_correct"))
            win_rate = wins / len(trades)
            
            rewards = [t.get("reward_signal", 0) for t in trades]
            avg_reward = np.mean(rewards) if rewards else 0.0
            
            performances[agent] = {
                "win_rate": float(win_rate),
                "avg_reward": float(avg_reward),
                "sample_size": len(trades)
            }
        
        return performances
    
    def should_update(self, decisions_dir: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if weights should be updated based on trade count.
        
        Args:
            decisions_dir: Path to decisions directory
        
        Returns:
            (should_update, trades_since_last_update)
        """
        if decisions_dir is None:
            from tradingagents.trade_decisions import DECISIONS_DIR
            decisions_dir = DECISIONS_DIR
        
        # Get last update time
        last_update = None
        if self.weight_history:
            last_update_str = self.weight_history[-1].get("timestamp")
            if last_update_str:
                last_update = datetime.fromisoformat(last_update_str)
        
        # Count trades since last update
        if not os.path.exists(decisions_dir):
            return False, 0
        
        trades_since = 0
        for filename in os.listdir(decisions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(decisions_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        trade = json.load(f)
                        
                        if trade.get("status") != "closed":
                            continue
                        
                        if last_update:
                            close_time = trade.get("close_time")
                            if close_time:
                                trade_date = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                                if trade_date > last_update:
                                    trades_since += 1
                        else:
                            trades_since += 1
                
                except Exception:
                    continue
        
        # Update every 30 trades
        return trades_since >= 30, trades_since
    
    def format_report(self, update_result: Dict[str, Any]) -> str:
        """Format weight update as human-readable report."""
        old = update_result["old_weights"]
        new = update_result["new_weights"]
        changes = update_result["changes"]
        
        report = """
AGENT WEIGHT UPDATE

Old Weights:
"""
        for agent, weight in old.items():
            report += f"  {agent.capitalize():8}: {weight:.3f}\n"
        
        report += "\nNew Weights:\n"
        for agent, weight in new.items():
            change = changes[agent]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            report += f"  {agent.capitalize():8}: {weight:.3f} {arrow} ({change:+.3f})\n"
        
        report += f"\nReasoning:\n{update_result['reasoning']}\n"
        
        return report
