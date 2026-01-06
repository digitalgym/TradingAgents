# TradingAgents Risk Metrics Module
"""
Quantitative risk metrics for portfolio and trade analysis.

Provides standard financial risk metrics:
- Sharpe Ratio: Risk-adjusted return (excess return / volatility)
- Sortino Ratio: Downside risk-adjusted return
- Value at Risk (VaR): Maximum expected loss at confidence level
- Conditional VaR (CVaR): Expected loss beyond VaR threshold
- Maximum Drawdown: Largest peak-to-trough decline
- Calmar Ratio: Return / max drawdown
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profit / gross loss
"""

import numpy as np
from typing import List, Optional, Union, Dict
from dataclasses import dataclass


@dataclass
class RiskReport:
    """Container for risk metrics report."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    var_95: float
    var_99: float
    cvar_95: float
    calmar_ratio: float
    volatility: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "cvar_95": round(self.cvar_95, 4),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "volatility": round(self.volatility, 4),
            "annualized_return": round(self.annualized_return, 4),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 3),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "num_trades": self.num_trades,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"""Risk Metrics Report
{'='*40}
Sharpe Ratio:      {self.sharpe_ratio:>10.3f}
Sortino Ratio:     {self.sortino_ratio:>10.3f}
Calmar Ratio:      {self.calmar_ratio:>10.3f}
Max Drawdown:      {self.max_drawdown_pct:>9.2f}%
VaR (95%):         {self.var_95*100:>9.2f}%
CVaR (95%):        {self.cvar_95*100:>9.2f}%
Volatility:        {self.volatility*100:>9.2f}%
Annual Return:     {self.annualized_return*100:>9.2f}%
{'='*40}
Win Rate:          {self.win_rate:>9.2f}%
Profit Factor:     {self.profit_factor:>10.3f}
Avg Win:           {self.avg_win*100:>9.2f}%
Avg Loss:          {self.avg_loss*100:>9.2f}%
Num Trades:        {self.num_trades:>10d}
{'='*40}"""


class RiskMetrics:
    """
    Static methods for calculating financial risk metrics.
    
    All methods are designed to work with numpy arrays of returns
    or equity curves. Returns should be in decimal form (0.01 = 1%).
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.02, 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Sharpe = (mean_return - risk_free_rate) / std_dev * sqrt(periods)
        
        Args:
            returns: Array of periodic returns (decimal form)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year (252 for daily)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        excess_returns = returns - risk_free_rate / periods_per_year
        
        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0.0
            
        return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))
    
    @staticmethod
    def sortino_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.02, 
        periods_per_year: int = 252,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation only).
        
        Sortino = (mean_return - target) / downside_std * sqrt(periods)
        
        Args:
            returns: Array of periodic returns (decimal form)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Trading periods per year
            target_return: Minimum acceptable return (default 0)
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < target_return]
        if len(downside_returns) < 2:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))
    
    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> tuple:
        """
        Calculate maximum drawdown from peak.
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            Tuple of (max_drawdown_value, max_drawdown_percentage)
        """
        if len(equity_curve) < 2:
            return (0.0, 0.0)
        
        equity_curve = np.asarray(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve - peak
        drawdown_pct = drawdown / peak
        
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = float(drawdown[max_dd_idx])
        max_dd_pct = float(drawdown_pct[max_dd_idx]) * 100
        
        return (max_dd_value, max_dd_pct)
    
    @staticmethod
    def value_at_risk(
        returns: np.ndarray, 
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk at given confidence level.
        
        VaR represents the maximum expected loss over a period
        at a given confidence level.
        
        Args:
            returns: Array of periodic returns (decimal form)
            confidence: Confidence level (0.95 = 95%)
            method: "historical" or "parametric"
            
        Returns:
            VaR as a positive decimal (0.02 = 2% loss)
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        
        if method == "parametric":
            # Assume normal distribution
            from scipy import stats
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            var = stats.norm.ppf(1 - confidence, mean, std)
        else:
            # Historical simulation
            var = np.percentile(returns, (1 - confidence) * 100)
        
        return float(-var) if var < 0 else 0.0
    
    @staticmethod
    def conditional_var(
        returns: np.ndarray, 
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        CVaR is the expected loss given that the loss exceeds VaR.
        Also known as Expected Shortfall (ES) or Average VaR.
        
        Args:
            returns: Array of periodic returns (decimal form)
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            CVaR as a positive decimal
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        var = np.percentile(returns, (1 - confidence) * 100)
        
        # Average of returns below VaR
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return float(-var) if var < 0 else 0.0
            
        cvar = np.mean(tail_returns)
        return float(-cvar) if cvar < 0 else 0.0
    
    @staticmethod
    def calmar_ratio(
        returns: np.ndarray, 
        equity_curve: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Array of periodic returns
            equity_curve: Array of portfolio values
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2 or len(equity_curve) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        annual_return = np.mean(returns) * periods_per_year
        
        _, max_dd_pct = RiskMetrics.max_drawdown(equity_curve)
        max_dd_pct = abs(max_dd_pct) / 100  # Convert to decimal
        
        if max_dd_pct == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return float(annual_return / max_dd_pct)
    
    @staticmethod
    def volatility(
        returns: np.ndarray, 
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Array of periodic returns
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized volatility (standard deviation)
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    
    @staticmethod
    def win_rate(trade_returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of profitable trades).
        
        Args:
            trade_returns: Array of individual trade returns
            
        Returns:
            Win rate as percentage (0-100)
        """
        if len(trade_returns) == 0:
            return 0.0
        
        trade_returns = np.asarray(trade_returns)
        wins = np.sum(trade_returns > 0)
        return float(wins / len(trade_returns) * 100)
    
    @staticmethod
    def profit_factor(trade_returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            trade_returns: Array of individual trade returns
            
        Returns:
            Profit factor (>1 is profitable)
        """
        if len(trade_returns) == 0:
            return 0.0
        
        trade_returns = np.asarray(trade_returns)
        gross_profit = np.sum(trade_returns[trade_returns > 0])
        gross_loss = abs(np.sum(trade_returns[trade_returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return float(gross_profit / gross_loss)
    
    @staticmethod
    def avg_win_loss(trade_returns: np.ndarray) -> tuple:
        """
        Calculate average win and average loss.
        
        Args:
            trade_returns: Array of individual trade returns
            
        Returns:
            Tuple of (avg_win, avg_loss) as decimals
        """
        if len(trade_returns) == 0:
            return (0.0, 0.0)
        
        trade_returns = np.asarray(trade_returns)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        
        return (avg_win, avg_loss)
    
    @staticmethod
    def calculate_all(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trade_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> RiskReport:
        """
        Calculate all risk metrics and return a RiskReport.
        
        Args:
            returns: Array of periodic returns (e.g., daily)
            equity_curve: Array of portfolio values
            trade_returns: Optional array of individual trade returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            RiskReport dataclass with all metrics
        """
        returns = np.asarray(returns)
        equity_curve = np.asarray(equity_curve)
        
        if trade_returns is None:
            trade_returns = returns
        else:
            trade_returns = np.asarray(trade_returns)
        
        max_dd_value, max_dd_pct = RiskMetrics.max_drawdown(equity_curve)
        avg_win, avg_loss = RiskMetrics.avg_win_loss(trade_returns)
        
        return RiskReport(
            sharpe_ratio=RiskMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            sortino_ratio=RiskMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            max_drawdown=max_dd_value,
            max_drawdown_pct=max_dd_pct,
            var_95=RiskMetrics.value_at_risk(returns, 0.95),
            var_99=RiskMetrics.value_at_risk(returns, 0.99),
            cvar_95=RiskMetrics.conditional_var(returns, 0.95),
            calmar_ratio=RiskMetrics.calmar_ratio(returns, equity_curve, periods_per_year),
            volatility=RiskMetrics.volatility(returns, periods_per_year),
            annualized_return=float(np.mean(returns) * periods_per_year),
            win_rate=RiskMetrics.win_rate(trade_returns),
            profit_factor=RiskMetrics.profit_factor(trade_returns),
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=len(trade_returns),
        )


def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """
    Calculate returns from equity curve.
    
    Args:
        equity_curve: Array of portfolio values
        
    Returns:
        Array of periodic returns
    """
    equity_curve = np.asarray(equity_curve)
    if len(equity_curve) < 2:
        return np.array([])
    return np.diff(equity_curve) / equity_curve[:-1]
