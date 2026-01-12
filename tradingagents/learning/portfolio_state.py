"""
Portfolio State Tracker

Maintains running portfolio state for reward signal calculations.
Tracks equity curve, returns, peak equity, and drawdowns.
"""

import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class PortfolioStateTracker:
    """Maintain running portfolio state for reward calculations"""
    
    DEFAULT_STATE_PATH = Path(__file__).parent.parent.parent / "examples" / "portfolio_state.pkl"
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize portfolio state tracker.
        
        Args:
            initial_capital: Starting capital in currency units
        """
        self.initial_capital = initial_capital
        self.equity_curve = [initial_capital]
        self.returns = []
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def update(self, trade_pnl: float, win: bool = None):
        """
        Update portfolio state after a trade.
        
        Args:
            trade_pnl: P&L from trade in currency units
            win: Optional flag indicating if trade was profitable
        """
        # Update equity
        self.current_equity += trade_pnl
        self.equity_curve.append(self.current_equity)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            previous_equity = self.equity_curve[-2]
            if previous_equity > 0:
                ret = trade_pnl / previous_equity
                self.returns.append(ret)
        
        # Update peak
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        # Update counters
        self.trade_count += 1
        self.total_pnl += trade_pnl
        
        if win is not None:
            if win:
                self.win_count += 1
            else:
                self.loss_count += 1
        elif trade_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        self.last_updated = datetime.now()
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate current portfolio Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            sharpe: Annualized Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        periods_per_year = 252  # Assume daily trading
        
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.
        
        Returns:
            drawdown_pct: Current drawdown as percentage (negative value)
        """
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = (self.current_equity - self.peak_equity) / self.peak_equity
        return float(drawdown)
    
    def get_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            max_dd_pct: Maximum drawdown as percentage (negative value)
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_dd = np.min(drawdown)
        
        return float(max_dd)
    
    def get_total_return(self) -> float:
        """
        Calculate total return since inception.
        
        Returns:
            total_return_pct: Total return as percentage
        """
        if self.initial_capital == 0:
            return 0.0
        
        return (self.current_equity - self.initial_capital) / self.initial_capital
    
    def get_win_rate(self) -> float:
        """
        Calculate win rate.
        
        Returns:
            win_rate: Percentage of winning trades (0.0 to 1.0)
        """
        total_trades = self.win_count + self.loss_count
        if total_trades == 0:
            return 0.0
        
        return self.win_count / total_trades
    
    def get_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Returns:
            profit_factor: Ratio of total wins to total losses
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate from equity changes
        equity_changes = np.diff(self.equity_curve)
        gross_profit = np.sum(equity_changes[equity_changes > 0])
        gross_loss = abs(np.sum(equity_changes[equity_changes < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive portfolio statistics.
        
        Returns:
            dict with all key metrics
        """
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "total_return_pct": self.get_total_return() * 100,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.get_sharpe_ratio(),
            "max_drawdown_pct": self.get_max_drawdown() * 100,
            "current_drawdown_pct": self.get_current_drawdown() * 100,
            "win_rate": self.get_win_rate(),
            "profit_factor": self.get_profit_factor(),
            "total_trades": self.trade_count,
            "winning_trades": self.win_count,
            "losing_trades": self.loss_count,
            "peak_equity": self.peak_equity,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }
    
    def save_state(self, path: Optional[str] = None):
        """
        Persist portfolio state to disk.
        
        Args:
            path: Optional custom path (default: examples/portfolio_state.pkl)
        """
        if path is None:
            path = self.DEFAULT_STATE_PATH
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_state(cls, path: Optional[str] = None) -> 'PortfolioStateTracker':
        """
        Load portfolio state from disk.
        
        Args:
            path: Optional custom path (default: examples/portfolio_state.pkl)
        
        Returns:
            PortfolioStateTracker instance
        """
        if path is None:
            path = cls.DEFAULT_STATE_PATH
        
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def reset(self, initial_capital: Optional[float] = None):
        """
        Reset portfolio state (use with caution).
        
        Args:
            initial_capital: Optional new starting capital
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.equity_curve = [self.initial_capital]
        self.returns = []
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def __repr__(self) -> str:
        return (
            f"PortfolioStateTracker("
            f"equity=${self.current_equity:,.2f}, "
            f"return={self.get_total_return()*100:.2f}%, "
            f"trades={self.trade_count}, "
            f"sharpe={self.get_sharpe_ratio():.2f})"
        )
