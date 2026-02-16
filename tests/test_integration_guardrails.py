"""
Tests for guardrails integration in signal processing.

Tests that risk guardrails are properly checked in the signal processing
pipeline and can override BUY/SELL signals to HOLD when circuit breakers are active.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from datetime import datetime


class TestSignalProcessorGuardrails:
    """Test guardrails integration in SignalProcessor."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for SignalProcessor."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(
            content='{"signal": "BUY", "confidence": 0.8, "rationale": "Strong setup"}'
        )
        return mock

    def test_apply_guardrails_no_override_when_trading_allowed(self, mock_llm):
        """Test that signal passes through when trading is allowed."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "BUY",
            "confidence": 0.8,
            "rationale": "Strong bullish setup",
        }

        # Mock guardrails to allow trading
        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (True, None)
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            # Signal should remain BUY
            assert processed["signal"] == "BUY"
            assert "guardrail_override" not in processed or not processed.get(
                "guardrail_override"
            )

    def test_apply_guardrails_override_when_circuit_breaker_active(self, mock_llm):
        """Test that BUY signal is overridden to HOLD when circuit breaker is active."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "BUY",
            "confidence": 0.8,
            "rationale": "Strong bullish setup",
        }

        # Mock guardrails to block trading
        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Daily loss limit exceeded",
            )
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            # Signal should be overridden to HOLD
            assert processed["signal"] == "HOLD"
            assert processed["guardrail_override"] is True
            assert processed["original_signal"] == "BUY"
            assert "Daily loss limit" in processed["guardrail_reason"]
            assert "GUARDRAIL OVERRIDE" in processed["rationale"]

    def test_apply_guardrails_override_sell_signal(self, mock_llm):
        """Test that SELL signal is also overridden when circuit breaker is active."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "SELL",
            "confidence": 0.75,
            "rationale": "Bearish reversal pattern",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Max consecutive losses reached",
            )
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            assert processed["signal"] == "HOLD"
            assert processed["guardrail_override"] is True
            assert processed["original_signal"] == "SELL"

    def test_apply_guardrails_hold_not_affected(self, mock_llm):
        """Test that HOLD signals are not affected by guardrails."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "HOLD",
            "confidence": 0.5,
            "rationale": "Uncertain market conditions",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Circuit breaker active",
            )
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            # HOLD should remain HOLD, no override
            assert processed["signal"] == "HOLD"
            assert "guardrail_override" not in processed or not processed.get(
                "guardrail_override"
            )

    def test_apply_guardrails_handles_exception(self, mock_llm):
        """Test that guardrails exception is handled gracefully."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "BUY",
            "confidence": 0.8,
            "rationale": "Strong setup",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            MockGuardrails.side_effect = Exception("Guardrails database error")

            processed = processor._apply_guardrails(result)

            # Should continue without override on exception
            assert processed["signal"] == "BUY"

    def test_process_signal_applies_guardrails(self, mock_llm):
        """Test that process_signal calls _apply_guardrails."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        structured_decision = {
            "signal": "BUY",
            "confidence": 0.8,
            "entry_price": 2800.0,
            "stop_loss": 2780.0,
            "take_profit": 2850.0,
            "rationale": "Bullish breakout",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Cooldown period active",
            )
            MockGuardrails.return_value = mock_guardrails

            result = processor.process_signal(
                "Full signal text",
                current_price=2800.0,
                structured_decision=structured_decision,
            )

            # Should have applied guardrails
            assert result["signal"] == "HOLD"
            assert result["guardrail_override"] is True

    def test_guardrail_reason_in_rationale(self, mock_llm):
        """Test that guardrail reason is included in rationale."""
        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "BUY",
            "confidence": 0.8,
            "rationale": "Original bullish analysis",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Daily loss limit of 3% exceeded",
            )
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            # Rationale should include both override reason and original
            assert "GUARDRAIL OVERRIDE" in processed["rationale"]
            assert "Daily loss limit" in processed["rationale"]
            assert "BUY" in processed["rationale"]
            assert "Original bullish analysis" in processed["rationale"]


class TestRiskGuardrailsBasic:
    """Basic tests for RiskGuardrails functionality."""

    @pytest.fixture
    def temp_state_file(self):
        """Create temporary state file for guardrails."""
        temp_dir = tempfile.mkdtemp()
        state_file = os.path.join(temp_dir, "guardrails_state.pkl")
        yield state_file
        # Cleanup
        if os.path.exists(state_file):
            os.remove(state_file)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

    def test_guardrails_check_can_trade_default(self, temp_state_file):
        """Test check_can_trade returns True by default."""
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails(state_file=temp_state_file)

        # Reset state
        guardrails.reset_daily_loss()
        guardrails.reset_consecutive_losses()
        guardrails.reset_cooldown()

        can_trade, reason = guardrails.check_can_trade(account_balance=10000)

        assert can_trade is True
        # Reason can be None, "", or "OK" when trading is allowed
        assert reason is None or reason == "" or reason.upper() == "OK"

    def test_guardrails_blocks_after_daily_loss_limit(self, temp_state_file):
        """Test that trading is blocked after daily loss limit."""
        from tradingagents.risk.guardrails import RiskGuardrails

        # Use higher consecutive loss limit to avoid that trigger
        guardrails = RiskGuardrails(
            state_file=temp_state_file,
            daily_loss_limit_pct=3.0,
            max_consecutive_losses=10  # Higher to avoid consecutive loss trigger
        )

        # Reset state completely
        guardrails.reset_daily_loss()
        guardrails.reset_consecutive_losses()
        guardrails.reset_cooldown()

        # Record a large loss (4% of account - exceeds 3% limit)
        guardrails.record_trade_result(was_win=False, pnl_pct=-4.0, account_balance=10000)

        can_trade, reason = guardrails.check_can_trade(account_balance=9600)

        # Should be blocked - either by loss limit or cooldown that got triggered
        assert can_trade is False
        assert reason is not None and reason != ""

    def test_guardrails_blocks_after_consecutive_losses(self, temp_state_file):
        """Test that trading is blocked after max consecutive losses."""
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails(state_file=temp_state_file, max_consecutive_losses=2)

        # Reset and record consecutive losses
        guardrails.reset_daily_loss()
        guardrails.reset_consecutive_losses()
        guardrails.reset_cooldown()
        guardrails.record_trade_result(was_win=False, pnl_pct=-0.5, account_balance=10000)
        guardrails.record_trade_result(was_win=False, pnl_pct=-0.5, account_balance=9950)

        can_trade, reason = guardrails.check_can_trade(account_balance=9900)

        assert can_trade is False
        assert "consecutive" in reason.lower() or "losses" in reason.lower() or "cooldown" in reason.lower()

    def test_guardrails_resets_consecutive_on_win(self, temp_state_file):
        """Test that consecutive loss counter resets on win."""
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails(state_file=temp_state_file, max_consecutive_losses=3)

        # Reset and record loss then win
        guardrails.reset_daily_loss()
        guardrails.reset_consecutive_losses()
        guardrails.reset_cooldown()
        guardrails.record_trade_result(was_win=False, pnl_pct=-0.5, account_balance=10000)
        guardrails.record_trade_result(was_win=True, pnl_pct=1.0, account_balance=10100)

        can_trade, reason = guardrails.check_can_trade(account_balance=10100)

        # Should be able to trade after a win
        assert can_trade is True

    def test_guardrails_cooldown_period(self, temp_state_file):
        """Test cooldown period after circuit breaker trips."""
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails(state_file=temp_state_file, max_consecutive_losses=2, cooldown_hours=24)

        # Reset state
        guardrails.reset_daily_loss()
        guardrails.reset_consecutive_losses()
        guardrails.reset_cooldown()

        # Record enough losses to trip circuit breaker
        guardrails.record_trade_result(was_win=False, pnl_pct=-0.5, account_balance=10000)
        guardrails.record_trade_result(was_win=False, pnl_pct=-0.5, account_balance=9950)

        # Check immediately after - should be in cooldown
        can_trade, reason = guardrails.check_can_trade(account_balance=9900)

        # Either blocked by consecutive losses or cooldown
        assert can_trade is False


class TestGuardrailIntegrationWithGraph:
    """Test guardrails integration with the full graph pipeline."""

    def test_signal_processor_has_guardrails_method(self):
        """Test that SignalProcessor has _apply_guardrails method."""
        from tradingagents.graph.signal_processing import SignalProcessor

        assert hasattr(SignalProcessor, "_apply_guardrails")

    def test_guardrails_preserves_other_fields(self):
        """Test that guardrails override preserves other signal fields."""
        mock_llm = MagicMock()

        from tradingagents.graph.signal_processing import SignalProcessor

        processor = SignalProcessor(mock_llm)

        result = {
            "signal": "BUY",
            "confidence": 0.85,
            "entry_price": 2800.0,
            "stop_loss": 2780.0,
            "take_profit": 2850.0,
            "rationale": "Strong setup",
            "risk_level": "medium",
        }

        with patch(
            "tradingagents.graph.signal_processing.RiskGuardrails"
        ) as MockGuardrails:
            mock_guardrails = MagicMock()
            mock_guardrails.check_can_trade.return_value = (
                False,
                "Circuit breaker",
            )
            MockGuardrails.return_value = mock_guardrails

            processed = processor._apply_guardrails(result)

            # Other fields should be preserved
            assert processed["confidence"] == 0.85
            assert processed["entry_price"] == 2800.0
            assert processed["stop_loss"] == 2780.0
            assert processed["take_profit"] == 2850.0
            assert processed["risk_level"] == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
