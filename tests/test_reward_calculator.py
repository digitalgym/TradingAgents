"""
Unit tests for RewardCalculator

Tests all reward signal calculation methods including:
- Realized RR calculation
- Sharpe contribution
- Drawdown impact
- Composite reward signal
"""

import pytest
import numpy as np
from tradingagents.learning.reward import RewardCalculator


class TestRealizedRR:
    """Test realized risk-reward ratio calculations"""
    
    def test_buy_win_2r(self):
        """BUY trade: Entry 100, SL 98, Exit 104 -> 2R win"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=104,
            stop_loss=98,
            direction="BUY"
        )
        assert rr == 2.0
    
    def test_buy_full_loss(self):
        """BUY trade: Entry 100, SL 98, Exit 98 -> -1R loss"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=98,
            stop_loss=98,
            direction="BUY"
        )
        assert rr == -1.0
    
    def test_buy_partial_loss(self):
        """BUY trade: Entry 100, SL 98, Exit 99 -> -0.5R loss"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=99,
            stop_loss=98,
            direction="BUY"
        )
        assert rr == -0.5
    
    def test_sell_win_3r(self):
        """SELL trade: Entry 100, SL 102, Exit 94 -> 3R win"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=94,
            stop_loss=102,
            direction="SELL"
        )
        assert rr == 3.0
    
    def test_sell_full_loss(self):
        """SELL trade: Entry 100, SL 102, Exit 102 -> -1R loss"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=102,
            stop_loss=102,
            direction="SELL"
        )
        assert rr == -1.0
    
    def test_zero_risk(self):
        """Edge case: Zero risk (SL = entry)"""
        rr = RewardCalculator.calculate_realized_rr(
            entry_price=100,
            exit_price=105,
            stop_loss=100,
            direction="BUY"
        )
        assert rr == 0.0


class TestSharpeContribution:
    """Test Sharpe ratio contribution calculations"""
    
    def test_positive_contribution_winning_trade(self):
        """Winning trade should improve Sharpe"""
        portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10  # 50 trades
        trade_return = 0.03  # 3% win
        
        contribution = RewardCalculator.calculate_sharpe_contribution(
            trade_return=trade_return,
            portfolio_returns=portfolio_returns,
            position_size_pct=0.01
        )
        
        # Should be positive for winning trade
        assert contribution > 0
        assert -1.0 <= contribution <= 1.0
    
    def test_negative_contribution_losing_trade(self):
        """Large losing trade should degrade Sharpe"""
        portfolio_returns = [0.01, 0.01, 0.01] * 10  # Consistent wins
        trade_return = -0.05  # 5% loss
        
        contribution = RewardCalculator.calculate_sharpe_contribution(
            trade_return=trade_return,
            portfolio_returns=portfolio_returns,
            position_size_pct=0.01
        )
        
        # Should be negative for losing trade
        assert contribution < 0
        assert -1.0 <= contribution <= 1.0
    
    def test_minimal_data_fallback(self):
        """With minimal data, should use simple heuristic"""
        contribution = RewardCalculator.calculate_sharpe_contribution(
            trade_return=0.02,
            portfolio_returns=[],
            position_size_pct=0.01
        )
        
        # Should still return normalized value
        assert -1.0 <= contribution <= 1.0


class TestDrawdownImpact:
    """Test drawdown impact calculations"""
    
    def test_no_drawdown_winning_trade(self):
        """Winning trade at peak -> no drawdown"""
        equity_curve = [100000, 102000, 105000]
        peak_equity = 105000
        trade_pnl = 2000
        
        impact = RewardCalculator.calculate_drawdown_impact(
            trade_pnl=trade_pnl,
            equity_curve=equity_curve,
            peak_equity=peak_equity
        )
        
        assert impact == 0.0
    
    def test_new_drawdown_from_peak(self):
        """Losing trade from peak creates new drawdown"""
        equity_curve = [100000, 102000, 105000]
        peak_equity = 105000
        trade_pnl = -3000  # Drop to 102000
        
        impact = RewardCalculator.calculate_drawdown_impact(
            trade_pnl=trade_pnl,
            equity_curve=equity_curve,
            peak_equity=peak_equity
        )
        
        # Should be negative (caused drawdown)
        assert impact < 0
        assert impact >= -1.0
    
    def test_worsening_existing_drawdown(self):
        """Losing trade during drawdown worsens it"""
        equity_curve = [100000, 102000, 105000, 103000]  # Already in DD
        peak_equity = 105000
        trade_pnl = -2000  # Drop to 101000
        
        impact = RewardCalculator.calculate_drawdown_impact(
            trade_pnl=trade_pnl,
            equity_curve=equity_curve,
            peak_equity=peak_equity
        )
        
        # Should be negative (worsened DD)
        assert impact < 0
    
    def test_improving_drawdown(self):
        """Winning trade during drawdown improves it"""
        equity_curve = [100000, 102000, 105000, 103000]  # In DD
        peak_equity = 105000
        trade_pnl = 1500  # Recover to 104500
        
        impact = RewardCalculator.calculate_drawdown_impact(
            trade_pnl=trade_pnl,
            equity_curve=equity_curve,
            peak_equity=peak_equity
        )
        
        # Should be 0 (improved but still capped at 0)
        assert impact == 0.0


class TestCompositeReward:
    """Test composite reward signal calculation"""
    
    def test_big_win_positive_reward(self):
        """Big win with good Sharpe, no DD -> high positive reward"""
        reward = RewardCalculator.calculate_reward(
            realized_rr=3.0,
            sharpe_contribution=0.5,
            drawdown_impact=0.0,
            win=True
        )
        
        # Should be positive
        assert reward > 0
        # Should be in reasonable range
        assert -5.0 <= reward <= 5.0
    
    def test_full_loss_negative_reward(self):
        """Full loss with DD impact -> negative reward"""
        reward = RewardCalculator.calculate_reward(
            realized_rr=-1.0,
            sharpe_contribution=-0.3,
            drawdown_impact=-0.5,
            win=False
        )
        
        # Should be negative
        assert reward < 0
        assert -5.0 <= reward <= 5.0
    
    def test_custom_weights(self):
        """Test with custom weight configuration"""
        custom_weights = {
            "rr": 0.5,
            "sharpe": 0.3,
            "drawdown": 0.2
        }
        
        reward = RewardCalculator.calculate_reward(
            realized_rr=2.0,
            sharpe_contribution=0.4,
            drawdown_impact=0.0,
            win=True,
            weights=custom_weights
        )
        
        # Should calculate with custom weights
        expected = (2.0 * 0.5) + (0.4 * 0.3) - (0.0 * 0.2)
        assert abs(reward - expected) < 0.01
    
    def test_invalid_weights_raises_error(self):
        """Weights must sum to 1.0"""
        invalid_weights = {
            "rr": 0.5,
            "sharpe": 0.3,
            "drawdown": 0.1  # Sum = 0.9, not 1.0
        }
        
        with pytest.raises(ValueError):
            RewardCalculator.calculate_reward(
                realized_rr=2.0,
                sharpe_contribution=0.4,
                drawdown_impact=0.0,
                win=True,
                weights=invalid_weights
            )
    
    def test_reward_clipping(self):
        """Extreme values should be clipped to [-5, 5]"""
        # Create extreme scenario
        reward = RewardCalculator.calculate_reward(
            realized_rr=100.0,  # Unrealistic but possible
            sharpe_contribution=1.0,
            drawdown_impact=0.0,
            win=True
        )
        
        # Should be clipped
        assert reward <= 5.0


class TestCalculateAllComponents:
    """Test the convenience method that calculates everything"""
    
    def test_complete_calculation_winning_trade(self):
        """Test full calculation for a winning trade"""
        result = RewardCalculator.calculate_all_components(
            entry_price=2650.0,
            exit_price=2680.0,
            stop_loss=2630.0,
            direction="BUY",
            trade_pnl=300.0,
            portfolio_returns=[0.01, -0.005, 0.02] * 10,
            equity_curve=[100000, 101000, 100500, 102000],
            peak_equity=102000,
            position_size_pct=0.01
        )
        
        # Check all components present
        assert "realized_rr" in result
        assert "sharpe_contribution" in result
        assert "drawdown_impact" in result
        assert "reward" in result
        assert "win" in result
        
        # Verify calculations
        assert result["win"] is True
        assert result["realized_rr"] == 1.5  # (2680-2650)/(2650-2630) = 30/20 = 1.5
        assert -1.0 <= result["sharpe_contribution"] <= 1.0
        assert -1.0 <= result["drawdown_impact"] <= 0.0
        assert -5.0 <= result["reward"] <= 5.0
    
    def test_complete_calculation_losing_trade(self):
        """Test full calculation for a losing trade"""
        result = RewardCalculator.calculate_all_components(
            entry_price=2650.0,
            exit_price=2630.0,
            stop_loss=2630.0,
            direction="BUY",
            trade_pnl=-200.0,
            portfolio_returns=[0.01, 0.01, 0.01] * 10,
            equity_curve=[100000, 101000, 102000, 103000],
            peak_equity=103000,
            position_size_pct=0.01
        )
        
        # Check loss detected
        assert result["win"] is False
        assert result["realized_rr"] == -1.0  # Full stop loss
        assert result["reward"] < 0  # Negative reward for loss


class TestSharpeCalculation:
    """Test internal Sharpe ratio calculation"""
    
    def test_sharpe_with_positive_returns(self):
        """Positive returns should give positive Sharpe"""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02] * 10)
        sharpe = RewardCalculator._calculate_sharpe(returns)
        
        assert sharpe > 0
    
    def test_sharpe_with_negative_returns(self):
        """Negative returns should give negative Sharpe"""
        returns = np.array([-0.01, -0.02, -0.015, -0.01] * 10)
        sharpe = RewardCalculator._calculate_sharpe(returns)
        
        assert sharpe < 0
    
    def test_sharpe_with_minimal_data(self):
        """Minimal data should return 0"""
        returns = np.array([0.01])
        sharpe = RewardCalculator._calculate_sharpe(returns)
        
        assert sharpe == 0.0
    
    def test_sharpe_with_zero_volatility(self):
        """Zero volatility should return 0"""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = RewardCalculator._calculate_sharpe(returns)
        
        assert sharpe == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
