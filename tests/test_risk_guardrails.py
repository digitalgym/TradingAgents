"""
Unit tests for RiskGuardrails

Tests risk guardrails including:
- Daily loss limits
- Consecutive loss limits
- Position size validation
- Circuit breakers
- Cooldown periods
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from tradingagents.risk.guardrails import RiskGuardrails


@pytest.fixture
def temp_state_file():
    """Create temporary state file"""
    fd, path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestBasicFunctionality:
    """Test basic guardrail functionality"""
    
    def test_initialization(self, temp_state_file):
        """Test guardrail initialization"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            daily_loss_limit_pct=3.0,
            max_consecutive_losses=2
        )
        
        assert guardrails.daily_loss_limit_pct == 3.0
        assert guardrails.max_consecutive_losses == 2
    
    def test_initial_state(self, temp_state_file):
        """Test initial state allows trading"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        can_trade, reason = guardrails.check_can_trade(10000)
        assert can_trade is True
        assert reason == "OK"
    
    def test_state_persistence(self, temp_state_file):
        """Test state persists across instances"""
        # Create first instance and record loss
        guardrails1 = RiskGuardrails(state_file=temp_state_file)
        guardrails1.record_trade_result(False, -1.5, 10000)
        
        # Create second instance - should load saved state
        guardrails2 = RiskGuardrails(state_file=temp_state_file)
        assert guardrails2.state["consecutive_losses"] == 1
        assert guardrails2.state["daily_loss_pct"] == 1.5


class TestDailyLossLimit:
    """Test daily loss limit enforcement"""
    
    def test_daily_loss_accumulation(self, temp_state_file):
        """Test daily loss accumulates correctly"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            daily_loss_limit_pct=3.0
        )
        
        # Record first loss
        guardrails.record_trade_result(False, -1.5, 10000)
        assert guardrails.state["daily_loss_pct"] == 1.5
        
        # Record second loss
        guardrails.record_trade_result(False, -1.0, 10000)
        assert guardrails.state["daily_loss_pct"] == 2.5
    
    def test_daily_loss_limit_breach(self, temp_state_file):
        """Test daily loss limit triggers circuit breaker"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            daily_loss_limit_pct=3.0
        )
        
        # Exceed daily limit
        result = guardrails.record_trade_result(False, -3.5, 10000)
        
        assert result["breach_triggered"] is True
        assert result["breach_type"] == "daily_loss_limit"
        assert result["cooldown_until"] is not None
    
    def test_daily_loss_blocks_trading(self, temp_state_file):
        """Test trading blocked after daily loss breach"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            daily_loss_limit_pct=3.0
        )
        
        # Trigger breach
        guardrails.record_trade_result(False, -3.5, 10000)
        
        # Check trading blocked
        can_trade, reason = guardrails.check_can_trade(10000)
        assert can_trade is False
        assert "DAILY LOSS LIMIT" in reason
    
    def test_wins_dont_reduce_daily_loss(self, temp_state_file):
        """Test wins don't reduce daily loss counter"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # Record loss
        guardrails.record_trade_result(False, -2.0, 10000)
        assert guardrails.state["daily_loss_pct"] == 2.0
        
        # Record win - daily loss should stay same
        guardrails.record_trade_result(True, 1.5, 10000)
        assert guardrails.state["daily_loss_pct"] == 2.0


class TestConsecutiveLosses:
    """Test consecutive loss limit enforcement"""
    
    def test_consecutive_loss_counting(self, temp_state_file):
        """Test consecutive losses counted correctly"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            max_consecutive_losses=2
        )
        
        # First loss
        guardrails.record_trade_result(False, -1.0, 10000)
        assert guardrails.state["consecutive_losses"] == 1
        
        # Second loss
        guardrails.record_trade_result(False, -1.0, 10000)
        assert guardrails.state["consecutive_losses"] == 2
    
    def test_consecutive_loss_reset_on_win(self, temp_state_file):
        """Test consecutive losses reset on win"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # Two losses
        guardrails.record_trade_result(False, -1.0, 10000)
        guardrails.record_trade_result(False, -1.0, 10000)
        assert guardrails.state["consecutive_losses"] == 2
        
        # Win resets counter
        guardrails.record_trade_result(True, 2.0, 10000)
        assert guardrails.state["consecutive_losses"] == 0
    
    def test_consecutive_loss_breach(self, temp_state_file):
        """Test consecutive loss limit triggers circuit breaker"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            max_consecutive_losses=2
        )
        
        # First loss
        guardrails.record_trade_result(False, -1.0, 10000)
        
        # Second loss triggers breach
        result = guardrails.record_trade_result(False, -1.0, 10000)
        
        assert result["breach_triggered"] is True
        assert result["breach_type"] == "consecutive_losses"
    
    def test_consecutive_loss_blocks_trading(self, temp_state_file):
        """Test trading blocked after consecutive loss breach"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            max_consecutive_losses=2
        )
        
        # Trigger breach
        guardrails.record_trade_result(False, -1.0, 10000)
        guardrails.record_trade_result(False, -1.0, 10000)
        
        # Check trading blocked
        can_trade, reason = guardrails.check_can_trade(10000)
        assert can_trade is False
        assert "CONSECUTIVE LOSSES" in reason


class TestPositionSizing:
    """Test position size validation"""
    
    def test_valid_position_size(self, temp_state_file):
        """Test valid position size passes"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            max_position_size_pct=2.0
        )
        
        is_valid, reason, adjusted = guardrails.validate_position_size(1.5, 10000)
        
        assert is_valid is True
        assert reason == "OK"
        assert adjusted == 1.5
    
    def test_oversized_position_capped(self, temp_state_file):
        """Test oversized position is capped"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            max_position_size_pct=2.0
        )
        
        is_valid, reason, adjusted = guardrails.validate_position_size(5.0, 10000)
        
        assert is_valid is False
        assert "POSITION TOO LARGE" in reason
        assert adjusted == 2.0


class TestCooldownPeriod:
    """Test cooldown period functionality"""
    
    def test_cooldown_blocks_trading(self, temp_state_file):
        """Test cooldown blocks trading"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            cooldown_hours=24
        )
        
        # Trigger breach
        guardrails.record_trade_result(False, -3.5, 10000)
        
        # Should be in cooldown
        can_trade, reason = guardrails.check_can_trade(10000)
        assert can_trade is False
        assert "COOLDOWN" in reason
    
    def test_cooldown_duration(self, temp_state_file):
        """Test cooldown duration is correct"""
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            cooldown_hours=24
        )
        
        # Trigger breach
        result = guardrails.record_trade_result(False, -3.5, 10000)
        
        # Check cooldown end time
        cooldown_end = datetime.fromisoformat(result["cooldown_until"])
        expected_end = datetime.now() + timedelta(hours=24)
        
        # Allow 1 minute tolerance
        assert abs((cooldown_end - expected_end).total_seconds()) < 60
    
    def test_manual_cooldown_reset(self, temp_state_file):
        """Test manual cooldown reset"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # Trigger breach
        guardrails.record_trade_result(False, -3.5, 10000)
        
        # Reset cooldown
        guardrails.reset_cooldown()
        
        # Should allow trading
        can_trade, reason = guardrails.check_can_trade(10000)
        assert can_trade is True


class TestBreachHistory:
    """Test breach history tracking"""
    
    def test_breach_recorded(self, temp_state_file):
        """Test breaches are recorded in history"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # Trigger breach
        guardrails.record_trade_result(False, -3.5, 10000)
        
        # Check history
        history = guardrails.get_breach_history()
        assert len(history) == 1
        assert history[0]["type"] == "daily_loss_limit"
    
    def test_multiple_breaches_tracked(self, temp_state_file):
        """Test multiple breaches tracked"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # First breach
        guardrails.record_trade_result(False, -3.5, 10000)
        guardrails.reset_cooldown()
        guardrails.reset_daily_loss()
        
        # Second breach
        guardrails.record_trade_result(False, -1.0, 10000)
        guardrails.record_trade_result(False, -1.0, 10000)
        
        # Check history
        history = guardrails.get_breach_history()
        assert len(history) == 2
    
    def test_total_breaches_counter(self, temp_state_file):
        """Test total breaches counter"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        # Trigger breach
        guardrails.record_trade_result(False, -3.5, 10000)
        
        status = guardrails.get_status()
        assert status["total_breaches"] == 1


class TestStatusReporting:
    """Test status reporting"""
    
    def test_get_status(self, temp_state_file):
        """Test status retrieval"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        status = guardrails.get_status()
        
        assert "can_trade" in status
        assert "consecutive_losses" in status
        assert "daily_loss_pct" in status
        assert "status_summary" in status
    
    def test_format_report(self, temp_state_file):
        """Test report formatting"""
        guardrails = RiskGuardrails(state_file=temp_state_file)
        
        report = guardrails.format_report()
        
        assert "RISK GUARDRAILS STATUS" in report
        assert "Trading Allowed" in report
        assert "Current Metrics" in report
        assert "Limits" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
