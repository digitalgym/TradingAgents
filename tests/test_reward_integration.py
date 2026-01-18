"""
Integration test for reward signal calculation in trade decision flow

Tests the complete flow:
1. Store decision with trade details
2. Close decision with exit price
3. Verify reward signal calculated
4. Verify portfolio state updated
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from tradingagents.trade_decisions import store_decision, close_decision, load_decision
from tradingagents.learning.portfolio_state import PortfolioStateTracker


@pytest.fixture
def temp_decisions_dir():
    """Create temporary directory for test decisions"""
    temp_dir = tempfile.mkdtemp()
    
    # Temporarily override DECISIONS_DIR
    import tradingagents.trade_decisions as td
    original_dir = td.DECISIONS_DIR
    td.DECISIONS_DIR = temp_dir
    
    yield temp_dir
    
    # Restore and cleanup
    td.DECISIONS_DIR = original_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_portfolio_state():
    """Create temporary portfolio state file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_path = temp_file.name
    temp_file.close()
    
    # Create fresh portfolio
    portfolio = PortfolioStateTracker(initial_capital=100000)
    portfolio.save_state(temp_path)
    
    # Override default path
    import tradingagents.learning.portfolio_state as ps
    original_path = ps.PortfolioStateTracker.DEFAULT_STATE_PATH
    ps.PortfolioStateTracker.DEFAULT_STATE_PATH = Path(temp_path)
    
    yield temp_path
    
    # Restore and cleanup
    ps.PortfolioStateTracker.DEFAULT_STATE_PATH = original_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestRewardIntegration:
    """Test complete reward calculation flow"""
    
    def test_winning_trade_reward_flow(self, temp_decisions_dir, temp_portfolio_state):
        """Test complete flow for a winning trade"""
        # 1. Store decision
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="BUY",
            rationale="Strong bullish setup with multiple confluence factors",
            entry_price=2650.0,
            stop_loss=2630.0,
            take_profit=2690.0,
            volume=0.1
        )
        
        # 2. Verify decision stored
        decision = load_decision(decision_id)
        assert decision["status"] == "active"
        assert decision["entry_price"] == 2650.0
        
        # 3. Close decision with profit
        closed_decision = close_decision(
            decision_id=decision_id,
            exit_price=2680.0,  # +30 pips, 1.5R win
            exit_reason="tp-hit",
            outcome_notes="Take profit hit as expected"
        )
        
        # 4. Verify outcome calculated
        assert closed_decision["status"] == "closed"
        assert closed_decision["was_correct"] is True
        assert closed_decision["pnl_percent"] > 0
        
        # 5. Verify RR calculated
        assert closed_decision["rr_realized"] == 1.5  # (2680-2650)/(2650-2630)
        assert closed_decision["rr_planned"] == 2.0   # (2690-2650)/(2650-2630)
        
        # 6. Verify reward signal calculated
        assert closed_decision["reward_signal"] is not None
        assert closed_decision["reward_signal"] > 0  # Positive for win
        assert closed_decision["sharpe_contribution"] is not None
        assert closed_decision["drawdown_impact"] is not None
        
        # 7. Verify portfolio state updated
        portfolio = PortfolioStateTracker.load_state()
        assert portfolio.trade_count == 1
        assert portfolio.win_count == 1
        assert portfolio.current_equity > portfolio.initial_capital
    
    def test_losing_trade_reward_flow(self, temp_decisions_dir, temp_portfolio_state):
        """Test complete flow for a losing trade"""
        # 1. Store decision
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="SELL",
            rationale="Bearish reversal pattern",
            entry_price=2650.0,
            stop_loss=2670.0,
            take_profit=2610.0,
            volume=0.1
        )
        
        # 2. Close decision with loss
        closed_decision = close_decision(
            decision_id=decision_id,
            exit_price=2670.0,  # Hit stop loss
            exit_reason="sl-hit",
            outcome_notes="Stop loss hit, market continued up"
        )
        
        # 3. Verify outcome
        assert closed_decision["was_correct"] is False
        assert closed_decision["pnl_percent"] < 0
        
        # 4. Verify RR calculated
        assert closed_decision["rr_realized"] == -1.0  # Full stop loss
        
        # 5. Verify reward signal is negative
        assert closed_decision["reward_signal"] is not None
        assert closed_decision["reward_signal"] < 0  # Negative for loss
        
        # 6. Verify portfolio state updated
        portfolio = PortfolioStateTracker.load_state()
        assert portfolio.trade_count == 1
        assert portfolio.loss_count == 1
        assert portfolio.current_equity < portfolio.initial_capital
    
    def test_multiple_trades_portfolio_tracking(self, temp_decisions_dir, temp_portfolio_state):
        """Test portfolio state across multiple trades"""
        # Execute series of trades
        trades = [
            ("BUY", 2650.0, 2630.0, 2680.0, True),   # Win
            ("SELL", 2680.0, 2700.0, 2640.0, False), # Loss
            ("BUY", 2640.0, 2620.0, 2680.0, True),   # Win
        ]
        
        for i, (action, entry, sl, exit_price, is_win) in enumerate(trades):
            decision_id = store_decision(
                symbol="XAUUSD",
                decision_type="OPEN",
                action=action,
                rationale=f"Trade {i+1}",
                entry_price=entry,
                stop_loss=sl,
                take_profit=exit_price if is_win else entry,
                volume=0.1
            )
            
            close_decision(
                decision_id=decision_id,
                exit_price=exit_price,
                exit_reason="tp-hit" if is_win else "sl-hit"
            )
        
        # Verify portfolio state
        portfolio = PortfolioStateTracker.load_state()
        assert portfolio.trade_count == 3
        assert portfolio.win_count == 2
        assert portfolio.loss_count == 1
        assert len(portfolio.equity_curve) == 4  # Initial + 3 trades
        assert len(portfolio.returns) == 3
        
        # Verify Sharpe can be calculated
        sharpe = portfolio.get_sharpe_ratio()
        assert sharpe != 0.0  # Should have calculated value
    
    def test_reward_calculation_disabled(self, temp_decisions_dir, temp_portfolio_state):
        """Test closing decision without reward calculation"""
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="BUY",
            rationale="Test trade",
            entry_price=2650.0,
            stop_loss=2630.0,
            take_profit=2690.0,
            volume=0.1
        )
        
        # Close with reward calculation disabled
        closed_decision = close_decision(
            decision_id=decision_id,
            exit_price=2680.0,
            calculate_reward=False
        )
        
        # Verify basic outcome calculated but no reward
        assert closed_decision["was_correct"] is True
        assert closed_decision["rr_realized"] == 1.5
        assert closed_decision["reward_signal"] is None
        assert closed_decision["sharpe_contribution"] is None
        assert closed_decision["drawdown_impact"] is None
    
    def test_decision_without_stop_loss(self, temp_decisions_dir, temp_portfolio_state):
        """Test decision without stop loss (no RR calculation)"""
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="BUY",
            rationale="Test trade without SL",
            entry_price=2650.0,
            stop_loss=None,  # No stop loss
            take_profit=2690.0,
            volume=0.1
        )
        
        closed_decision = close_decision(
            decision_id=decision_id,
            exit_price=2680.0
        )
        
        # Verify outcome calculated but no RR/reward
        assert closed_decision["was_correct"] is True
        assert closed_decision["rr_realized"] is None
        assert closed_decision["reward_signal"] is None


class TestEnhancedDecisionFields:
    """Test new decision schema fields"""
    
    def test_setup_classification_fields(self, temp_decisions_dir):
        """Test setup classification fields are stored"""
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="BUY",
            rationale="Breaker block setup",
            entry_price=2650.0,
            stop_loss=2630.0,
            take_profit=2690.0
        )
        
        decision = load_decision(decision_id)
        
        # Verify new fields exist
        assert "setup_type" in decision
        assert "higher_tf_bias" in decision
        assert "confluence_score" in decision
        assert "confluence_factors" in decision
        assert "volatility_regime" in decision
        assert "market_regime" in decision
        assert "session" in decision
        assert "exit_reason" in decision
        assert "rr_planned" in decision
        assert "rr_realized" in decision
        assert "reward_signal" in decision
        assert "sharpe_contribution" in decision
        assert "drawdown_impact" in decision
        assert "pattern_tags" in decision
    
    def test_exit_analysis_fields_populated(self, temp_decisions_dir, temp_portfolio_state):
        """Test exit analysis fields are populated on close"""
        decision_id = store_decision(
            symbol="XAUUSD",
            decision_type="OPEN",
            action="BUY",
            rationale="Test",
            entry_price=2650.0,
            stop_loss=2630.0,
            take_profit=2690.0
        )
        
        closed_decision = close_decision(
            decision_id=decision_id,
            exit_price=2680.0,
            exit_reason="tp-hit"
        )
        
        # Verify exit analysis populated
        assert closed_decision["exit_reason"] == "tp-hit"
        assert closed_decision["rr_planned"] is not None
        assert closed_decision["rr_realized"] is not None
        assert closed_decision["reward_signal"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
