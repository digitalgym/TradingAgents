"""
Unit tests for PortfolioStateTracker

Tests portfolio state management including:
- Equity tracking
- Returns calculation
- Sharpe ratio
- Drawdown tracking
- Statistics
"""

import pytest
import os
import tempfile
from pathlib import Path
from tradingagents.learning.portfolio_state import PortfolioStateTracker


class TestPortfolioInitialization:
    """Test portfolio initialization"""
    
    def test_default_initialization(self):
        """Test default initialization with 100k capital"""
        portfolio = PortfolioStateTracker()
        
        assert portfolio.initial_capital == 100000
        assert portfolio.current_equity == 100000
        assert portfolio.peak_equity == 100000
        assert portfolio.trade_count == 0
        assert len(portfolio.equity_curve) == 1
        assert len(portfolio.returns) == 0
    
    def test_custom_capital(self):
        """Test initialization with custom capital"""
        portfolio = PortfolioStateTracker(initial_capital=50000)
        
        assert portfolio.initial_capital == 50000
        assert portfolio.current_equity == 50000


class TestPortfolioUpdate:
    """Test portfolio state updates"""
    
    def test_winning_trade_update(self):
        """Test update with winning trade"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(trade_pnl=2000, win=True)
        
        assert portfolio.current_equity == 102000
        assert portfolio.peak_equity == 102000
        assert portfolio.trade_count == 1
        assert portfolio.win_count == 1
        assert portfolio.loss_count == 0
        assert len(portfolio.equity_curve) == 2
        assert len(portfolio.returns) == 1
    
    def test_losing_trade_update(self):
        """Test update with losing trade"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(trade_pnl=-1500, win=False)
        
        assert portfolio.current_equity == 98500
        assert portfolio.peak_equity == 100000  # Peak unchanged
        assert portfolio.trade_count == 1
        assert portfolio.win_count == 0
        assert portfolio.loss_count == 1
    
    def test_multiple_trades(self):
        """Test multiple trade updates"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(2000, win=True)   # 102000
        portfolio.update(-1000, win=False) # 101000
        portfolio.update(3000, win=True)   # 104000
        
        assert portfolio.current_equity == 104000
        assert portfolio.peak_equity == 104000
        assert portfolio.trade_count == 3
        assert portfolio.win_count == 2
        assert portfolio.loss_count == 1
        assert len(portfolio.equity_curve) == 4
        assert len(portfolio.returns) == 3
    
    def test_auto_detect_win_loss(self):
        """Test automatic win/loss detection from P&L"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(1000)  # Positive -> win
        portfolio.update(-500)  # Negative -> loss
        
        assert portfolio.win_count == 1
        assert portfolio.loss_count == 1


class TestSharpeRatio:
    """Test Sharpe ratio calculation"""
    
    def test_sharpe_with_positive_returns(self):
        """Positive returns should give positive Sharpe"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        # Simulate consistent wins
        for _ in range(10):
            portfolio.update(1000, win=True)
        
        sharpe = portfolio.get_sharpe_ratio()
        assert sharpe > 0
    
    def test_sharpe_with_negative_returns(self):
        """Negative returns should give negative Sharpe"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        # Simulate consistent losses
        for _ in range(10):
            portfolio.update(-1000, win=False)
        
        sharpe = portfolio.get_sharpe_ratio()
        assert sharpe < 0
    
    def test_sharpe_with_minimal_data(self):
        """Minimal data should return 0"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(1000, win=True)
        
        sharpe = portfolio.get_sharpe_ratio()
        assert sharpe == 0.0


class TestDrawdown:
    """Test drawdown calculations"""
    
    def test_no_drawdown_at_peak(self):
        """At peak equity, drawdown should be 0"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(5000, win=True)
        
        dd = portfolio.get_current_drawdown()
        assert dd == 0.0
    
    def test_drawdown_after_loss(self):
        """After loss from peak, should show drawdown"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(5000, win=True)   # Peak: 105000
        portfolio.update(-3000, win=False) # Current: 102000
        
        dd = portfolio.get_current_drawdown()
        expected_dd = (102000 - 105000) / 105000
        assert abs(dd - expected_dd) < 0.0001
        assert dd < 0
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(10000, win=True)  # 110000 (peak)
        portfolio.update(-5000, win=False) # 105000 (DD: -4.5%)
        portfolio.update(-8000, win=False) # 97000 (DD: -11.8%)
        portfolio.update(5000, win=True)   # 102000 (DD: -7.3%)
        
        max_dd = portfolio.get_max_drawdown()
        expected_max = (97000 - 110000) / 110000
        assert abs(max_dd - expected_max) < 0.0001
        assert max_dd < 0


class TestStatistics:
    """Test portfolio statistics"""
    
    def test_total_return(self):
        """Test total return calculation"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(10000, win=True)
        portfolio.update(5000, win=True)
        
        total_return = portfolio.get_total_return()
        assert total_return == 0.15  # 15% return
    
    def test_win_rate(self):
        """Test win rate calculation"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(1000, win=True)
        portfolio.update(1000, win=True)
        portfolio.update(-500, win=False)
        portfolio.update(1000, win=True)
        
        win_rate = portfolio.get_win_rate()
        assert win_rate == 0.75  # 3 wins out of 4 trades
    
    def test_profit_factor(self):
        """Test profit factor calculation"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(2000, win=True)   # +2000
        portfolio.update(3000, win=True)   # +3000
        portfolio.update(-1000, win=False) # -1000
        
        pf = portfolio.get_profit_factor()
        expected_pf = 5000 / 1000  # 5.0
        assert abs(pf - expected_pf) < 0.01
    
    def test_get_statistics(self):
        """Test comprehensive statistics"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        portfolio.update(5000, win=True)
        portfolio.update(-2000, win=False)
        portfolio.update(3000, win=True)
        
        stats = portfolio.get_statistics()
        
        # Check all expected fields
        assert "initial_capital" in stats
        assert "current_equity" in stats
        assert "total_return_pct" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown_pct" in stats
        assert "win_rate" in stats
        assert "profit_factor" in stats
        assert "total_trades" in stats
        
        # Verify values
        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 1
        assert stats["current_equity"] == 106000


class TestPersistence:
    """Test save/load functionality"""
    
    def test_save_and_load(self):
        """Test saving and loading portfolio state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_portfolio.pkl"
            
            # Create and update portfolio
            portfolio1 = PortfolioStateTracker(initial_capital=100000)
            portfolio1.update(5000, win=True)
            portfolio1.update(-2000, win=False)
            portfolio1.save_state(str(save_path))
            
            # Load portfolio
            portfolio2 = PortfolioStateTracker.load_state(str(save_path))
            
            # Verify state preserved
            assert portfolio2.initial_capital == portfolio1.initial_capital
            assert portfolio2.current_equity == portfolio1.current_equity
            assert portfolio2.trade_count == portfolio1.trade_count
            assert portfolio2.win_count == portfolio1.win_count
            assert len(portfolio2.equity_curve) == len(portfolio1.equity_curve)
    
    def test_load_nonexistent_creates_new(self):
        """Loading non-existent file should create new portfolio"""
        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "nonexistent.pkl"
            
            portfolio = PortfolioStateTracker.load_state(str(load_path))
            
            # Should be fresh portfolio
            assert portfolio.initial_capital == 100000
            assert portfolio.trade_count == 0


class TestReset:
    """Test portfolio reset functionality"""
    
    def test_reset_clears_state(self):
        """Reset should clear all state"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        
        # Add some trades
        portfolio.update(5000, win=True)
        portfolio.update(-2000, win=False)
        
        # Reset
        portfolio.reset()
        
        # Should be back to initial state
        assert portfolio.current_equity == 100000
        assert portfolio.peak_equity == 100000
        assert portfolio.trade_count == 0
        assert len(portfolio.equity_curve) == 1
        assert len(portfolio.returns) == 0
    
    def test_reset_with_new_capital(self):
        """Reset with new capital should update initial capital"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(5000, win=True)
        
        portfolio.reset(initial_capital=50000)
        
        assert portfolio.initial_capital == 50000
        assert portfolio.current_equity == 50000


class TestRepr:
    """Test string representation"""
    
    def test_repr_format(self):
        """Test __repr__ output"""
        portfolio = PortfolioStateTracker(initial_capital=100000)
        portfolio.update(5000, win=True)
        
        repr_str = repr(portfolio)
        
        assert "PortfolioStateTracker" in repr_str
        assert "equity=" in repr_str
        assert "return=" in repr_str
        assert "trades=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
