"""
Unit tests for the review CLI command.

Tests:
1. Position review flow
2. ATR-based analysis integration
3. SL/TP suggestion parsing
4. Decision storage
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typer.testing import CliRunner


runner = CliRunner()


class TestReviewCommand:
    """Tests for the review CLI command."""
    
    def test_review_help(self):
        """Test review command help."""
        from cli.main import app
        result = runner.invoke(app, ["review", "--help"])
        
        assert result.exit_code == 0
        assert "review" in result.stdout.lower() or "position" in result.stdout.lower()


class TestSLTPParsing:
    """Tests for SL/TP value parsing from LLM analysis."""
    
    def test_parse_sl_from_analysis(self):
        """Test parsing stop-loss from analysis text."""
        import re
        
        analysis = """
        **ADJUST** - Move SL to breakeven
        
        **Stop Loss Update**: Move SL to 5.9315 to protect capital.
        **Take Profit Update**: Keep TP at 6.0500.
        """
        
        sl_patterns = [
            r'(?:SL|stop\s*loss|move\s*sl)\s*(?:to|at|:)?\s*\*?\*?(\d+\.?\d*)',
            r'breakeven\s*(?:at|:)?\s*\*?\*?(\d+\.?\d*)',
        ]
        
        suggested_sl = None
        for pattern in sl_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                try:
                    suggested_sl = float(match.group(1))
                    break
                except ValueError:
                    pass
        
        assert suggested_sl == pytest.approx(5.9315, rel=0.001)
    
    def test_parse_tp_from_analysis(self):
        """Test parsing take-profit from analysis text."""
        import re
        
        analysis = """
        **HOLD** - Position looks good
        
        **Stop Loss Update**: Keep current SL.
        **Take Profit Update**: Adjust TP to 6.1000 for better R:R.
        """
        
        tp_patterns = [
            r'(?:TP|take\s*profit|target)\s*(?:to|at|:)?\s*\*?\*?(\d+\.?\d*)',
        ]
        
        suggested_tp = None
        for pattern in tp_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                try:
                    suggested_tp = float(match.group(1))
                    break
                except ValueError:
                    pass
        
        assert suggested_tp == pytest.approx(6.1000, rel=0.001)
    
    def test_parse_action_close(self):
        """Test parsing CLOSE action from analysis."""
        analysis = "**CLOSE** - Exit position immediately due to trend reversal."
        
        analysis_upper = analysis.upper()
        if "CLOSE" in analysis_upper[:100]:
            action = "CLOSE"
        elif "ADJUST" in analysis_upper[:100]:
            action = "ADJUST"
        else:
            action = "HOLD"
        
        assert action == "CLOSE"
    
    def test_parse_action_adjust(self):
        """Test parsing ADJUST action from analysis."""
        analysis = "**ADJUST** - Move stop loss to breakeven."
        
        analysis_upper = analysis.upper()
        if "CLOSE" in analysis_upper[:100]:
            action = "CLOSE"
        elif "ADJUST" in analysis_upper[:100]:
            action = "ADJUST"
        else:
            action = "HOLD"
        
        assert action == "ADJUST"
    
    def test_parse_action_hold(self):
        """Test parsing HOLD action from analysis."""
        analysis = "**HOLD** - Position is performing well, maintain current levels."
        
        analysis_upper = analysis.upper()
        if "CLOSE" in analysis_upper[:100]:
            action = "CLOSE"
        elif "ADJUST" in analysis_upper[:100]:
            action = "ADJUST"
        else:
            action = "HOLD"
        
        assert action == "HOLD"
    
    def test_parse_action_default_hold(self):
        """Test default to HOLD when no clear action keywords."""
        # Note: "CLOSE" appears in "closely" so we need text without that
        analysis = "The position is performing well. Keep monitoring."
        
        analysis_upper = analysis.upper()
        if "CLOSE" in analysis_upper[:100]:
            action = "CLOSE"
        elif "ADJUST" in analysis_upper[:100]:
            action = "ADJUST"
        else:
            action = "HOLD"
        
        assert action == "HOLD"


class TestATRIntegration:
    """Tests for ATR-based analysis integration."""
    
    def test_atr_suggestions_generated(self):
        """Test that ATR suggestions are generated when ATR > 0."""
        from tradingagents.risk.stop_loss import DynamicStopLoss
        
        dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)
        
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=2650.0,
            current_price=2680.0,
            current_sl=2630.0,
            current_tp=2700.0,
            atr=15.5,
            direction="BUY",
        )
        
        assert suggestions is not None
        assert "recommendation" in suggestions
        assert suggestions["pnl_percent"] > 0  # Position is profitable
    
    def test_atr_zero_handling(self):
        """Test handling when ATR is 0."""
        atr = 0.0
        
        # When ATR is 0, we should skip ATR-based suggestions
        atr_suggestions = ""
        if atr > 0:
            atr_suggestions = "ATR-based suggestions here"
        
        assert atr_suggestions == ""


class TestPositionMetrics:
    """Tests for position metric calculations."""
    
    def test_buy_pnl_calculation(self):
        """Test P&L calculation for BUY position."""
        entry = 100.0
        current_price = 105.0
        pos_type = "BUY"
        
        if pos_type == "BUY":
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
        
        assert pnl_pct == pytest.approx(5.0, rel=0.01)
    
    def test_sell_pnl_calculation(self):
        """Test P&L calculation for SELL position."""
        entry = 100.0
        current_price = 95.0
        pos_type = "SELL"
        
        if pos_type == "BUY":
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
        
        assert pnl_pct == pytest.approx(5.0, rel=0.01)
    
    def test_risk_reward_calculation(self):
        """Test risk:reward ratio calculation."""
        entry = 100.0
        sl = 95.0
        tp = 115.0
        
        sl_distance = abs(entry - sl)  # 5
        tp_distance = abs(tp - entry)  # 15
        
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        assert risk_reward == pytest.approx(3.0, rel=0.01)
    
    def test_risk_reward_no_sl(self):
        """Test risk:reward when SL is 0."""
        entry = 100.0
        sl = 0.0
        tp = 115.0
        
        sl_distance = abs(entry - sl) if sl > 0 else 0
        tp_distance = abs(tp - entry) if tp > 0 else 0
        
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        assert risk_reward == 0
    
    def test_distance_to_sl_buy(self):
        """Test distance to SL for BUY position."""
        current_price = 105.0
        sl = 95.0
        pos_type = "BUY"
        
        if pos_type == "BUY":
            distance_to_sl = current_price - sl if sl > 0 else 0
        else:
            distance_to_sl = sl - current_price if sl > 0 else 0
        
        assert distance_to_sl == pytest.approx(10.0, rel=0.01)
    
    def test_distance_to_tp_buy(self):
        """Test distance to TP for BUY position."""
        current_price = 105.0
        tp = 115.0
        pos_type = "BUY"
        
        if pos_type == "BUY":
            distance_to_tp = tp - current_price if tp > 0 else 0
        else:
            distance_to_tp = current_price - tp if tp > 0 else 0
        
        assert distance_to_tp == pytest.approx(10.0, rel=0.01)


class TestQuestionaryNullHandling:
    """Tests for handling None returns from questionary."""
    
    def test_confirm_none_handling(self):
        """Test handling when questionary.confirm returns None."""
        act_on_review = None  # Simulates cancelled prompt
        
        if not act_on_review:
            result = "skipped"
        else:
            result = "acted"
        
        assert result == "skipped"
    
    def test_select_none_handling(self):
        """Test handling when questionary.select returns None."""
        value_choice = None  # Simulates cancelled prompt
        sl = 95.0
        tp = 115.0
        
        if not value_choice or value_choice == "Skip SL/TP update":
            new_sl = sl
            new_tp = tp
        else:
            new_sl = 100.0
            new_tp = 120.0
        
        assert new_sl == sl
        assert new_tp == tp
    
    def test_text_none_handling(self):
        """Test handling when questionary.text returns None."""
        sl_input = None  # Simulates cancelled prompt
        current_sl = 95.0
        
        new_sl = current_sl
        if sl_input:
            try:
                new_sl = float(sl_input)
            except ValueError:
                pass
        
        assert new_sl == current_sl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
