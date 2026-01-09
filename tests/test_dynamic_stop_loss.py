"""
Unit tests for the Dynamic Stop-Loss module.

Tests:
1. ATR calculation
2. Initial stop-loss calculation
3. Take-profit calculation based on risk:reward
4. Trailing stop logic
5. Breakeven stop logic
6. Stop adjustment suggestions
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.risk.stop_loss import (
    DynamicStopLoss,
    StopLossLevels,
    calculate_atr,
)


class TestCalculateATR:
    """Tests for ATR calculation function."""
    
    def test_basic_atr_calculation(self):
        """Test ATR calculation with simple data."""
        high = np.array([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14])
        low = np.array([9, 10, 11, 10, 9, 10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
        close = np.array([9.5, 10.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 11.5, 10.5, 9.5, 10.5, 11.5, 12.5, 13.5])
        
        atr = calculate_atr(high, low, close, period=14)
        
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_atr_with_insufficient_data(self):
        """Test ATR calculation with less data than period."""
        high = np.array([10, 11, 12])
        low = np.array([9, 10, 11])
        close = np.array([9.5, 10.5, 11.5])
        
        atr = calculate_atr(high, low, close, period=14)
        
        # Should return simple range average
        assert atr > 0
        assert atr == pytest.approx(1.0, rel=0.1)  # high - low = 1 for each bar
    
    def test_atr_with_gaps(self):
        """Test ATR calculation with price gaps."""
        # Create data with a gap
        high = np.array([10, 11, 15, 16, 15, 14, 13, 14, 15, 16, 15, 14, 13, 14, 15])
        low = np.array([9, 10, 14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14])
        close = np.array([9.5, 10.5, 14.5, 15.5, 14.5, 13.5, 12.5, 13.5, 14.5, 15.5, 14.5, 13.5, 12.5, 13.5, 14.5])
        
        atr = calculate_atr(high, low, close, period=14)
        
        # ATR should be higher due to the gap
        assert atr > 1.0


class TestDynamicStopLoss:
    """Tests for DynamicStopLoss class."""
    
    @pytest.fixture
    def dsl(self):
        """Create a DynamicStopLoss instance with default settings."""
        return DynamicStopLoss(
            atr_multiplier=2.0,
            trailing_multiplier=1.5,
            risk_reward_ratio=2.0,
        )
    
    @pytest.fixture
    def dsl_custom(self):
        """Create a DynamicStopLoss instance with custom settings."""
        return DynamicStopLoss(
            atr_multiplier=3.0,
            trailing_multiplier=2.0,
            risk_reward_ratio=3.0,
            min_stop_percent=1.0,
            max_stop_percent=10.0,
        )


class TestInitialStopLoss(TestDynamicStopLoss):
    """Tests for initial stop-loss calculation."""
    
    def test_buy_stop_loss(self, dsl):
        """Test stop-loss for BUY position."""
        entry = 100.0
        atr = 2.0
        
        sl = dsl.calculate_initial_sl(entry, atr, "BUY")
        
        # SL should be below entry for BUY
        assert sl < entry
        # SL should be 2x ATR below entry
        assert sl == pytest.approx(96.0, rel=0.01)
    
    def test_sell_stop_loss(self, dsl):
        """Test stop-loss for SELL position."""
        entry = 100.0
        atr = 2.0
        
        sl = dsl.calculate_initial_sl(entry, atr, "SELL")
        
        # SL should be above entry for SELL
        assert sl > entry
        # SL should be 2x ATR above entry
        assert sl == pytest.approx(104.0, rel=0.01)
    
    def test_stop_loss_min_constraint(self, dsl):
        """Test that stop-loss respects minimum distance."""
        entry = 100.0
        atr = 0.1  # Very small ATR
        
        sl = dsl.calculate_initial_sl(entry, atr, "BUY")
        
        # SL should be at least min_stop_percent (0.5%) from entry
        min_distance = entry * 0.005
        assert abs(entry - sl) >= min_distance
    
    def test_stop_loss_max_constraint(self, dsl):
        """Test that stop-loss respects maximum distance."""
        entry = 100.0
        atr = 10.0  # Very large ATR
        
        sl = dsl.calculate_initial_sl(entry, atr, "BUY")
        
        # SL should be at most max_stop_percent (5%) from entry
        max_distance = entry * 0.05
        assert abs(entry - sl) <= max_distance
    
    def test_long_direction_alias(self, dsl):
        """Test that LONG works same as BUY."""
        entry = 100.0
        atr = 2.0
        
        sl_buy = dsl.calculate_initial_sl(entry, atr, "BUY")
        sl_long = dsl.calculate_initial_sl(entry, atr, "LONG")
        
        assert sl_buy == sl_long
    
    def test_short_direction_alias(self, dsl):
        """Test that SHORT works same as SELL."""
        entry = 100.0
        atr = 2.0
        
        sl_sell = dsl.calculate_initial_sl(entry, atr, "SELL")
        sl_short = dsl.calculate_initial_sl(entry, atr, "SHORT")
        
        assert sl_sell == sl_short


class TestTakeProfit(TestDynamicStopLoss):
    """Tests for take-profit calculation."""
    
    def test_buy_take_profit(self, dsl):
        """Test take-profit for BUY position."""
        entry = 100.0
        sl = 96.0  # 4 points risk
        
        tp = dsl.calculate_initial_tp(entry, sl, "BUY")
        
        # TP should be above entry for BUY
        assert tp > entry
        # TP should be 2x the risk distance (8 points)
        assert tp == pytest.approx(108.0, rel=0.01)
    
    def test_sell_take_profit(self, dsl):
        """Test take-profit for SELL position."""
        entry = 100.0
        sl = 104.0  # 4 points risk
        
        tp = dsl.calculate_initial_tp(entry, sl, "SELL")
        
        # TP should be below entry for SELL
        assert tp < entry
        # TP should be 2x the risk distance (8 points)
        assert tp == pytest.approx(92.0, rel=0.01)
    
    def test_custom_risk_reward(self, dsl):
        """Test take-profit with custom risk:reward ratio."""
        entry = 100.0
        sl = 96.0  # 4 points risk
        
        tp = dsl.calculate_initial_tp(entry, sl, "BUY", risk_reward=3.0)
        
        # TP should be 3x the risk distance (12 points)
        assert tp == pytest.approx(112.0, rel=0.01)


class TestCalculateLevels(TestDynamicStopLoss):
    """Tests for combined SL/TP calculation."""
    
    def test_calculate_levels_buy(self, dsl):
        """Test full level calculation for BUY."""
        entry = 2650.0
        atr = 15.5
        
        levels = dsl.calculate_levels(entry, atr, "BUY")
        
        assert isinstance(levels, StopLossLevels)
        assert levels.stop_loss < entry
        assert levels.take_profit > entry
        assert levels.atr == atr
        assert levels.risk_reward_ratio == 2.0
    
    def test_calculate_levels_sell(self, dsl):
        """Test full level calculation for SELL."""
        entry = 2650.0
        atr = 15.5
        
        levels = dsl.calculate_levels(entry, atr, "SELL")
        
        assert levels.stop_loss > entry
        assert levels.take_profit < entry
    
    def test_levels_rounding(self, dsl):
        """Test that levels are properly rounded."""
        entry = 2650.123456
        atr = 15.5
        
        levels = dsl.calculate_levels(entry, atr, "BUY")
        
        # Should be rounded to 5 decimal places
        assert len(str(levels.stop_loss).split('.')[-1]) <= 5
        assert len(str(levels.take_profit).split('.')[-1]) <= 5


class TestTrailingStop(TestDynamicStopLoss):
    """Tests for trailing stop calculation."""
    
    def test_trailing_stop_buy_moves_up(self, dsl):
        """Test trailing stop moves up for profitable BUY."""
        current_price = 110.0
        current_sl = 95.0
        atr = 2.0
        
        new_sl, should_update = dsl.calculate_trailing_stop(
            current_price, current_sl, atr, "BUY"
        )
        
        # Trailing SL should be 1.5x ATR below current price
        expected_sl = 110.0 - (1.5 * 2.0)  # 107.0
        assert should_update is True
        assert new_sl == pytest.approx(expected_sl, rel=0.01)
    
    def test_trailing_stop_buy_no_move_down(self, dsl):
        """Test trailing stop doesn't move down for BUY."""
        current_price = 98.0  # Price dropped
        current_sl = 95.0
        atr = 2.0
        
        new_sl, should_update = dsl.calculate_trailing_stop(
            current_price, current_sl, atr, "BUY"
        )
        
        # New calculated SL (95.0) would be same as current, no update
        assert should_update is False
        assert new_sl == current_sl
    
    def test_trailing_stop_sell_moves_down(self, dsl):
        """Test trailing stop moves down for profitable SELL."""
        current_price = 90.0
        current_sl = 105.0
        atr = 2.0
        
        new_sl, should_update = dsl.calculate_trailing_stop(
            current_price, current_sl, atr, "SELL"
        )
        
        # Trailing SL should be 1.5x ATR above current price
        expected_sl = 90.0 + (1.5 * 2.0)  # 93.0
        assert should_update is True
        assert new_sl == pytest.approx(expected_sl, rel=0.01)
    
    def test_trailing_stop_sell_no_move_up(self, dsl):
        """Test trailing stop doesn't move up for SELL."""
        current_price = 102.0  # Price went up (loss for short)
        current_sl = 105.0
        atr = 2.0
        
        new_sl, should_update = dsl.calculate_trailing_stop(
            current_price, current_sl, atr, "SELL"
        )
        
        # New calculated SL (105.0) would be same as current, no update
        assert should_update is False
        assert new_sl == current_sl


class TestBreakevenStop(TestDynamicStopLoss):
    """Tests for breakeven stop calculation."""
    
    def test_breakeven_buy_profitable(self, dsl):
        """Test breakeven for profitable BUY position."""
        entry = 100.0
        current_price = 102.0  # 2% profit
        atr = 1.0
        
        be_sl, is_eligible = dsl.calculate_breakeven_stop(
            entry, current_price, "BUY", buffer_atr=0.1, atr=atr
        )
        
        assert is_eligible is True
        # Breakeven should be entry + small buffer
        assert be_sl >= entry
        assert be_sl == pytest.approx(100.1, rel=0.01)
    
    def test_breakeven_buy_not_profitable_enough(self, dsl):
        """Test breakeven not available for small profit."""
        entry = 100.0
        current_price = 100.2  # Only 0.2% profit (less than 0.5% threshold)
        atr = 1.0
        
        be_sl, is_eligible = dsl.calculate_breakeven_stop(
            entry, current_price, "BUY", buffer_atr=0.1, atr=atr
        )
        
        assert is_eligible is False
    
    def test_breakeven_sell_profitable(self, dsl):
        """Test breakeven for profitable SELL position."""
        entry = 100.0
        current_price = 98.0  # 2% profit for short
        atr = 1.0
        
        be_sl, is_eligible = dsl.calculate_breakeven_stop(
            entry, current_price, "SELL", buffer_atr=0.1, atr=atr
        )
        
        assert is_eligible is True
        # Breakeven should be entry - small buffer
        assert be_sl <= entry
        assert be_sl == pytest.approx(99.9, rel=0.01)


class TestStopAdjustmentSuggestions(TestDynamicStopLoss):
    """Tests for stop adjustment suggestions."""
    
    def test_suggestions_profitable_buy(self, dsl):
        """Test suggestions for profitable BUY position."""
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=100.0,
            current_price=105.0,  # 5% profit
            current_sl=95.0,
            current_tp=110.0,
            atr=2.0,
            direction="BUY",
        )
        
        assert "current" in suggestions
        assert "pnl_percent" in suggestions
        assert suggestions["pnl_percent"] == pytest.approx(5.0, rel=0.1)
        assert suggestions["breakeven"] is not None
        assert suggestions["trailing"] is not None
        assert "recommendation" in suggestions
    
    def test_suggestions_losing_position(self, dsl):
        """Test suggestions for losing position."""
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=100.0,
            current_price=97.0,  # 3% loss
            current_sl=95.0,
            current_tp=110.0,
            atr=2.0,
            direction="BUY",
        )
        
        assert suggestions["pnl_percent"] == pytest.approx(-3.0, rel=0.1)
        assert suggestions["breakeven"] is None  # Not profitable
        assert "loss" in suggestions["recommendation"].lower()
    
    def test_suggestions_partial_tp_levels(self, dsl):
        """Test that partial TP levels are calculated."""
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=100.0,
            current_price=105.0,
            current_sl=95.0,
            current_tp=120.0,  # 20 point TP
            atr=2.0,
            direction="BUY",
        )
        
        assert len(suggestions["partial_tp"]) == 2
        # First partial at 50% to TP (110)
        assert suggestions["partial_tp"][0]["level"] == pytest.approx(110.0, rel=0.01)
        assert suggestions["partial_tp"][0]["percent"] == 50
        # Second partial at 75% to TP (115)
        assert suggestions["partial_tp"][1]["level"] == pytest.approx(115.0, rel=0.01)
        assert suggestions["partial_tp"][1]["percent"] == 25


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_atr(self):
        """Test handling of zero ATR."""
        dsl = DynamicStopLoss()
        
        # Should use minimum stop distance
        sl = dsl.calculate_initial_sl(100.0, 0.0, "BUY")
        assert sl < 100.0
        assert abs(100.0 - sl) >= 100.0 * 0.005  # At least min_stop_percent
    
    def test_negative_atr(self):
        """Test handling of negative ATR (shouldn't happen but handle gracefully)."""
        dsl = DynamicStopLoss()
        
        # Should use minimum stop distance
        sl = dsl.calculate_initial_sl(100.0, -1.0, "BUY")
        assert sl < 100.0
    
    def test_zero_stop_loss_in_suggestions(self):
        """Test suggestions when current SL is 0."""
        dsl = DynamicStopLoss()
        
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=100.0,
            current_price=105.0,
            current_sl=0.0,  # No SL set
            current_tp=110.0,
            atr=2.0,
            direction="BUY",
        )
        
        # Should still provide suggestions
        assert suggestions is not None
        assert "recommendation" in suggestions
    
    def test_zero_take_profit_in_suggestions(self):
        """Test suggestions when current TP is 0."""
        dsl = DynamicStopLoss()
        
        suggestions = dsl.suggest_stop_adjustment(
            entry_price=100.0,
            current_price=105.0,
            current_sl=95.0,
            current_tp=0.0,  # No TP set
            atr=2.0,
            direction="BUY",
        )
        
        # Should still provide suggestions but no partial TP
        assert suggestions is not None
        assert len(suggestions["partial_tp"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
