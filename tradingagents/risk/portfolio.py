# TradingAgents Portfolio Management Module
"""
Portfolio tracking and position management with risk metrics integration.

Provides:
- Position tracking (long/short)
- Equity curve management
- Trade history logging
- Real-time risk metrics calculation
- Integration with MT5 positions
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .metrics import RiskMetrics, RiskReport, calculate_returns


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    shares: float
    avg_price: float
    current_price: float
    side: str = "long"  # "long" or "short"
    opened_at: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        if self.side == "long":
            return self.shares * self.current_price
        else:
            # Short position: profit when price falls
            return self.shares * (2 * self.avg_price - self.current_price)
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == "long":
            return self.shares * (self.current_price - self.avg_price)
        else:
            return self.shares * (self.avg_price - self.current_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        cost_basis = self.shares * self.avg_price
        if cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / cost_basis) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "side": self.side,
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "opened_at": self.opened_at.isoformat(),
        }


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    action: str  # "BUY", "SELL", "SHORT", "COVER"
    price: float
    shares: float
    timestamp: datetime = field(default_factory=datetime.now)
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "price": self.price,
            "shares": self.shares,
            "timestamp": self.timestamp.isoformat(),
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


class Portfolio:
    """
    Portfolio manager with position tracking and risk metrics.
    
    Tracks:
    - Cash balance
    - Open positions (long and short)
    - Equity curve over time
    - Trade history with P&L
    - Real-time risk metrics
    
    Example:
        portfolio = Portfolio(initial_capital=100000)
        portfolio.buy("XAUUSD", price=4450, shares=10)
        portfolio.update_price("XAUUSD", 4500)
        metrics = portfolio.get_risk_metrics()
        print(metrics)
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting cash balance
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            periods_per_year: Trading periods per year (252 for daily)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = [initial_capital]
        self.equity_timestamps: List[datetime] = [datetime.now()]
        self.trade_history: List[Trade] = []
        self.trade_returns: List[float] = []  # Individual trade returns
    
    @property
    def total_equity(self) -> float:
        """Current total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Total return since inception (percentage)."""
        return ((self.total_equity / self.initial_capital) - 1) * 100
    
    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def _record_equity(self):
        """Record current equity to curve."""
        self.equity_curve.append(self.total_equity)
        self.equity_timestamps.append(datetime.now())
    
    def buy(
        self, 
        ticker: str, 
        price: float, 
        shares: float,
        record: bool = True
    ) -> Trade:
        """
        Buy shares (open or add to long position).
        
        Args:
            ticker: Asset symbol
            price: Purchase price per share
            shares: Number of shares to buy
            record: Whether to record equity after trade
            
        Returns:
            Trade object
        """
        cost = price * shares
        
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}")
        
        if ticker in self.positions:
            pos = self.positions[ticker]
            if pos.side == "short":
                # Covering a short position
                return self.cover(ticker, price, shares, record)
            
            # Average up existing long position
            old_shares = pos.shares
            old_avg = pos.avg_price
            new_shares = old_shares + shares
            new_avg = (old_avg * old_shares + price * shares) / new_shares
            
            pos.shares = new_shares
            pos.avg_price = new_avg
            pos.current_price = price
        else:
            # New long position
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_price=price,
                current_price=price,
                side="long"
            )
        
        self.cash -= cost
        
        trade = Trade(
            ticker=ticker,
            action="BUY",
            price=price,
            shares=shares
        )
        self.trade_history.append(trade)
        
        if record:
            self._record_equity()
        
        return trade
    
    def sell(
        self, 
        ticker: str, 
        price: float, 
        shares: Optional[float] = None,
        record: bool = True
    ) -> Trade:
        """
        Sell shares (close or reduce long position).
        
        Args:
            ticker: Asset symbol
            price: Sale price per share
            shares: Number of shares to sell (None = all)
            record: Whether to record equity after trade
            
        Returns:
            Trade object with P&L
        """
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker}")
        
        pos = self.positions[ticker]
        
        if pos.side == "short":
            raise ValueError(f"Cannot sell short position in {ticker}, use cover()")
        
        if shares is None:
            shares = pos.shares
        
        if shares > pos.shares:
            raise ValueError(f"Cannot sell {shares} shares, only have {pos.shares}")
        
        # Calculate P&L
        proceeds = price * shares
        cost_basis = pos.avg_price * shares
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Update position
        pos.shares -= shares
        pos.current_price = price
        
        if pos.shares <= 0:
            del self.positions[ticker]
        
        self.cash += proceeds
        
        # Record trade return
        trade_return = pnl / cost_basis if cost_basis > 0 else 0
        self.trade_returns.append(trade_return)
        
        trade = Trade(
            ticker=ticker,
            action="SELL",
            price=price,
            shares=shares,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trade_history.append(trade)
        
        if record:
            self._record_equity()
        
        return trade
    
    def short(
        self, 
        ticker: str, 
        price: float, 
        shares: float,
        record: bool = True
    ) -> Trade:
        """
        Short sell shares (open short position).
        
        Args:
            ticker: Asset symbol
            price: Short sale price per share
            shares: Number of shares to short
            record: Whether to record equity after trade
            
        Returns:
            Trade object
        """
        if ticker in self.positions:
            pos = self.positions[ticker]
            if pos.side == "long":
                raise ValueError(f"Already have long position in {ticker}, sell first")
            
            # Add to existing short
            old_shares = pos.shares
            old_avg = pos.avg_price
            new_shares = old_shares + shares
            new_avg = (old_avg * old_shares + price * shares) / new_shares
            
            pos.shares = new_shares
            pos.avg_price = new_avg
            pos.current_price = price
        else:
            # New short position
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_price=price,
                current_price=price,
                side="short"
            )
        
        # Short sale proceeds go to cash (margin account)
        self.cash += price * shares
        
        trade = Trade(
            ticker=ticker,
            action="SHORT",
            price=price,
            shares=shares
        )
        self.trade_history.append(trade)
        
        if record:
            self._record_equity()
        
        return trade
    
    def cover(
        self, 
        ticker: str, 
        price: float, 
        shares: Optional[float] = None,
        record: bool = True
    ) -> Trade:
        """
        Cover short position (buy to close).
        
        Args:
            ticker: Asset symbol
            price: Cover price per share
            shares: Number of shares to cover (None = all)
            record: Whether to record equity after trade
            
        Returns:
            Trade object with P&L
        """
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker}")
        
        pos = self.positions[ticker]
        
        if pos.side == "long":
            raise ValueError(f"Cannot cover long position in {ticker}, use sell()")
        
        if shares is None:
            shares = pos.shares
        
        if shares > pos.shares:
            raise ValueError(f"Cannot cover {shares} shares, only short {pos.shares}")
        
        # Calculate P&L (profit when price falls)
        cost_to_cover = price * shares
        original_proceeds = pos.avg_price * shares
        pnl = original_proceeds - cost_to_cover
        pnl_pct = (pnl / original_proceeds) * 100 if original_proceeds > 0 else 0
        
        # Update position
        pos.shares -= shares
        pos.current_price = price
        
        if pos.shares <= 0:
            del self.positions[ticker]
        
        self.cash -= cost_to_cover
        
        # Record trade return
        trade_return = pnl / original_proceeds if original_proceeds > 0 else 0
        self.trade_returns.append(trade_return)
        
        trade = Trade(
            ticker=ticker,
            action="COVER",
            price=price,
            shares=shares,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trade_history.append(trade)
        
        if record:
            self._record_equity()
        
        return trade
    
    def update_price(self, ticker: str, price: float, record: bool = True):
        """
        Update current price for a position.
        
        Args:
            ticker: Asset symbol
            price: New current price
            record: Whether to record equity after update
        """
        if ticker in self.positions:
            self.positions[ticker].current_price = price
            if record:
                self._record_equity()
    
    def update_prices(self, prices: Dict[str, float], record: bool = True):
        """
        Update prices for multiple positions.
        
        Args:
            prices: Dict of {ticker: price}
            record: Whether to record equity after update
        """
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
        
        if record:
            self._record_equity()
    
    def get_risk_metrics(self) -> Optional[RiskReport]:
        """
        Calculate current risk metrics.
        
        Returns:
            RiskReport with all metrics, or None if insufficient data
        """
        if len(self.equity_curve) < 2:
            return None
        
        equity_array = np.array(self.equity_curve)
        returns = calculate_returns(equity_array)
        trade_returns = np.array(self.trade_returns) if self.trade_returns else returns
        
        return RiskMetrics.calculate_all(
            returns=returns,
            equity_curve=equity_array,
            trade_returns=trade_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
    
    def get_position_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        return [pos.to_dict() for pos in self.positions.values()]
    
    def get_trade_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            last_n: Return only last N trades (None = all)
            
        Returns:
            List of trade dictionaries
        """
        trades = self.trade_history[-last_n:] if last_n else self.trade_history
        return [t.to_dict() for t in trades]
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        metrics = self.get_risk_metrics()
        
        return {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value, 2),
            "total_equity": round(self.total_equity, 2),
            "total_return_pct": round(self.total_return, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "num_positions": len(self.positions),
            "num_trades": len(self.trade_history),
            "risk_metrics": metrics.to_dict() if metrics else None,
        }
    
    def save(self, filepath: str):
        """
        Save portfolio state to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "risk_free_rate": self.risk_free_rate,
            "periods_per_year": self.periods_per_year,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "equity_curve": self.equity_curve,
            "equity_timestamps": [t.isoformat() for t in self.equity_timestamps],
            "trade_history": [t.to_dict() for t in self.trade_history],
            "trade_returns": self.trade_returns,
            "saved_at": datetime.now().isoformat(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Portfolio":
        """
        Load portfolio state from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Portfolio instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        portfolio = cls(
            initial_capital=data["initial_capital"],
            risk_free_rate=data.get("risk_free_rate", 0.02),
            periods_per_year=data.get("periods_per_year", 252)
        )
        
        portfolio.cash = data["cash"]
        portfolio.equity_curve = data["equity_curve"]
        portfolio.equity_timestamps = [
            datetime.fromisoformat(t) for t in data["equity_timestamps"]
        ]
        portfolio.trade_returns = data.get("trade_returns", [])
        
        # Restore positions
        for ticker, pos_data in data.get("positions", {}).items():
            portfolio.positions[ticker] = Position(
                ticker=pos_data["ticker"],
                shares=pos_data["shares"],
                avg_price=pos_data["avg_price"],
                current_price=pos_data["current_price"],
                side=pos_data.get("side", "long"),
                opened_at=datetime.fromisoformat(pos_data["opened_at"])
            )
        
        # Restore trade history
        for trade_data in data.get("trade_history", []):
            portfolio.trade_history.append(Trade(
                ticker=trade_data["ticker"],
                action=trade_data["action"],
                price=trade_data["price"],
                shares=trade_data["shares"],
                timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                pnl=trade_data.get("pnl"),
                pnl_pct=trade_data.get("pnl_pct")
            ))
        
        return portfolio
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Portfolio Summary",
            f"{'='*40}",
            f"Initial Capital:   ${self.initial_capital:>12,.2f}",
            f"Cash:              ${self.cash:>12,.2f}",
            f"Positions Value:   ${self.positions_value:>12,.2f}",
            f"Total Equity:      ${self.total_equity:>12,.2f}",
            f"Total Return:      {self.total_return:>12.2f}%",
            f"Unrealized P&L:    ${self.unrealized_pnl:>12,.2f}",
            f"{'='*40}",
            f"Open Positions: {len(self.positions)}",
            f"Total Trades: {len(self.trade_history)}",
        ]
        
        if self.positions:
            lines.append(f"\nPositions:")
            for ticker, pos in self.positions.items():
                lines.append(
                    f"  {ticker}: {pos.shares:.2f} @ ${pos.avg_price:.2f} "
                    f"(now ${pos.current_price:.2f}, P&L: {pos.unrealized_pnl_pct:+.2f}%)"
                )
        
        return "\n".join(lines)
