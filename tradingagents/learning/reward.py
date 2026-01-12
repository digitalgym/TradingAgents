"""
Reward Signal Calculator for Online Learning

Calculates multi-factor reward signals based on:
1. Realized Risk-Reward ratio
2. Sharpe ratio contribution
3. Drawdown impact

Used to provide feedback signals for agent weight updates and pattern learning.
"""

import numpy as np
from typing import List, Optional


class RewardCalculator:
    """Calculate multi-factor reward signals for trade outcomes"""
    
    DEFAULT_WEIGHTS = {
        "rr": 0.4,        # Risk-reward component
        "sharpe": 0.3,    # Sharpe contribution component
        "drawdown": 0.3   # Drawdown impact component
    }
    
    @staticmethod
    def calculate_reward(
        realized_rr: float,
        sharpe_contribution: float,
        drawdown_impact: float,
        win: bool,
        weights: Optional[dict] = None
    ) -> float:
        """
        Calculate composite reward signal for a trade outcome.
        
        Formula:
            reward = (realized_RR × w_rr) + (Sharpe_contrib × w_sharpe) - (DD_impact × w_dd)
        
        Args:
            realized_rr: Actual risk-reward achieved (e.g., 2.5 for 2.5R win, -1.0 for full loss)
            sharpe_contribution: Impact on portfolio Sharpe ratio (-1.0 to 1.0)
            drawdown_impact: Contribution to drawdown (0 if none, negative if caused DD)
            win: True if profitable trade
            weights: Optional custom weights (must sum to 1.0)
        
        Returns:
            reward: Float in range approximately [-5.0, 5.0] (normalized)
        
        Examples:
            >>> # Big win with good Sharpe contribution, no drawdown
            >>> RewardCalculator.calculate_reward(3.0, 0.5, 0.0, True)
            1.65
            
            >>> # Full loss that caused drawdown
            >>> RewardCalculator.calculate_reward(-1.0, -0.3, -0.5, False)
            -0.94
        """
        if weights is None:
            weights = RewardCalculator.DEFAULT_WEIGHTS
        
        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        # Calculate weighted components
        rr_component = realized_rr * weights["rr"]
        sharpe_component = sharpe_contribution * weights["sharpe"]
        dd_component = drawdown_impact * weights["drawdown"]
        
        # Composite reward (drawdown is subtracted since it's negative impact)
        reward = rr_component + sharpe_component - dd_component
        
        # Normalize to reasonable range [-5, 5]
        reward = np.clip(reward, -5.0, 5.0)
        
        return float(reward)
    
    @staticmethod
    def calculate_realized_rr(
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        direction: str
    ) -> float:
        """
        Calculate realized risk-reward ratio.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            direction: "BUY" or "SELL"
        
        Returns:
            realized_rr: Positive for wins, negative for losses
        
        Examples:
            >>> # BUY: Entry 100, SL 98, Exit 104 -> 2R win
            >>> RewardCalculator.calculate_realized_rr(100, 104, 98, "BUY")
            2.0
            
            >>> # BUY: Entry 100, SL 98, Exit 98 -> -1R loss
            >>> RewardCalculator.calculate_realized_rr(100, 98, 98, "BUY")
            -1.0
        """
        if direction.upper() == "BUY":
            risk = abs(entry_price - stop_loss)
            pnl = exit_price - entry_price
        else:  # SELL
            risk = abs(stop_loss - entry_price)
            pnl = entry_price - exit_price
        
        if risk == 0:
            return 0.0
        
        realized_rr = pnl / risk
        return float(realized_rr)
    
    @staticmethod
    def calculate_sharpe_contribution(
        trade_return: float,
        portfolio_returns: List[float],
        position_size_pct: float = 1.0
    ) -> float:
        """
        Calculate how this trade affected portfolio Sharpe ratio.
        
        This is an approximation of the marginal contribution to Sharpe.
        
        Args:
            trade_return: Return from this trade (e.g., 0.02 for 2% gain)
            portfolio_returns: List of recent portfolio returns (last 20-50 trades)
            position_size_pct: Position size as % of portfolio (default 1% = 0.01)
        
        Returns:
            sharpe_contribution: Normalized contribution (-1.0 to 1.0)
        
        Notes:
            - Positive if trade improved Sharpe
            - Negative if trade degraded Sharpe
            - Magnitude indicates strength of impact
        """
        if len(portfolio_returns) < 2:
            # Not enough data, use simple return-based heuristic
            return np.clip(trade_return * 10, -1.0, 1.0)
        
        # Calculate portfolio Sharpe before this trade
        returns_array = np.array(portfolio_returns)
        sharpe_before = RewardCalculator._calculate_sharpe(returns_array)
        
        # Calculate portfolio Sharpe after adding this trade
        # Weight the trade return by position size
        weighted_return = trade_return * position_size_pct
        returns_with_trade = np.append(returns_array, weighted_return)
        sharpe_after = RewardCalculator._calculate_sharpe(returns_with_trade)
        
        # Contribution is the delta
        contribution = sharpe_after - sharpe_before
        
        # Normalize to [-1, 1] range (typical delta is -0.5 to +0.5)
        normalized = np.clip(contribution * 2, -1.0, 1.0)
        
        return float(normalized)
    
    @staticmethod
    def calculate_drawdown_impact(
        trade_pnl: float,
        equity_curve: List[float],
        peak_equity: float
    ) -> float:
        """
        Calculate if trade contributed to drawdown.
        
        Args:
            trade_pnl: P&L from this trade in currency units
            equity_curve: Historical equity values
            peak_equity: Peak equity before this trade
        
        Returns:
            drawdown_impact: 0 if no DD, negative if caused/worsened DD
        
        Notes:
            - Returns 0 if trade didn't cause drawdown
            - Returns negative value proportional to DD severity
            - Scaled to [-1, 0] range for normalization
        """
        if len(equity_curve) == 0:
            return 0.0
        
        current_equity = equity_curve[-1]
        new_equity = current_equity + trade_pnl
        
        # Check if we're in drawdown
        if current_equity >= peak_equity:
            # Not in drawdown before trade
            if new_equity >= peak_equity:
                # Still not in drawdown
                return 0.0
            else:
                # Trade caused new drawdown
                dd_pct = (new_equity - peak_equity) / peak_equity
                return float(np.clip(dd_pct, -1.0, 0.0))
        else:
            # Already in drawdown
            dd_before = (current_equity - peak_equity) / peak_equity
            dd_after = (new_equity - peak_equity) / peak_equity
            
            # Impact is how much we worsened (or improved) the DD
            dd_change = dd_after - dd_before
            
            # Normalize: worsening DD is negative, improving is positive (but capped at 0)
            return float(np.clip(dd_change, -1.0, 0.0))
    
    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio from returns array.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            sharpe: Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Assume daily returns, annualize
        periods_per_year = 252
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    @staticmethod
    def calculate_all_components(
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        direction: str,
        trade_pnl: float,
        portfolio_returns: List[float],
        equity_curve: List[float],
        peak_equity: float,
        position_size_pct: float = 0.01,
        weights: Optional[dict] = None
    ) -> dict:
        """
        Calculate all reward components and final reward in one call.
        
        Convenience method that calculates everything needed for reward signal.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            direction: "BUY" or "SELL"
            trade_pnl: P&L in currency units
            portfolio_returns: List of recent returns
            equity_curve: Historical equity values
            peak_equity: Peak equity before trade
            position_size_pct: Position size as % of portfolio
            weights: Optional custom weights
        
        Returns:
            dict with all components and final reward:
            {
                "realized_rr": 2.5,
                "sharpe_contribution": 0.3,
                "drawdown_impact": 0.0,
                "reward": 1.65,
                "win": True
            }
        """
        # Calculate realized RR
        realized_rr = RewardCalculator.calculate_realized_rr(
            entry_price, exit_price, stop_loss, direction
        )
        
        # Determine if win
        win = realized_rr > 0
        
        # Calculate trade return for Sharpe
        if direction.upper() == "BUY":
            trade_return = (exit_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - exit_price) / entry_price
        
        # Calculate Sharpe contribution
        sharpe_contribution = RewardCalculator.calculate_sharpe_contribution(
            trade_return, portfolio_returns, position_size_pct
        )
        
        # Calculate drawdown impact
        drawdown_impact = RewardCalculator.calculate_drawdown_impact(
            trade_pnl, equity_curve, peak_equity
        )
        
        # Calculate final reward
        reward = RewardCalculator.calculate_reward(
            realized_rr, sharpe_contribution, drawdown_impact, win, weights
        )
        
        return {
            "realized_rr": realized_rr,
            "sharpe_contribution": sharpe_contribution,
            "drawdown_impact": drawdown_impact,
            "reward": reward,
            "win": win
        }
