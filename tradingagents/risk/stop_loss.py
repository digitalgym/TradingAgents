"""
Dynamic Stop-Loss Management

Provides ATR-based stop-loss calculation and trailing stop functionality.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class StopLossLevels:
    """Container for calculated stop-loss and take-profit levels."""
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    atr: float = 0.0
    risk_reward_ratio: float = 2.0


class DynamicStopLoss:
    """
    Dynamic stop-loss calculator using ATR (Average True Range).
    
    ATR-based stops adapt to market volatility:
    - High volatility = wider stops (avoid premature stop-outs)
    - Low volatility = tighter stops (protect profits)
    
    Usage:
        dsl = DynamicStopLoss(atr_multiplier=2.0)
        levels = dsl.calculate_levels(entry=2650, atr=15.5, direction="BUY")
        print(f"SL: {levels.stop_loss}, TP: {levels.take_profit}")
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.0,
        trailing_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        min_stop_percent: float = 0.5,
        max_stop_percent: float = 5.0,
    ):
        """
        Initialize DynamicStopLoss calculator.
        
        Args:
            atr_multiplier: Multiplier for ATR to set initial stop distance
            trailing_multiplier: Multiplier for ATR to set trailing stop distance
            risk_reward_ratio: Target risk:reward ratio for take profit
            min_stop_percent: Minimum stop distance as % of entry (floor)
            max_stop_percent: Maximum stop distance as % of entry (ceiling)
        """
        self.atr_mult = atr_multiplier
        self.trail_mult = trailing_multiplier
        self.risk_reward = risk_reward_ratio
        self.min_stop_pct = min_stop_percent / 100
        self.max_stop_pct = max_stop_percent / 100
    
    def calculate_initial_sl(
        self,
        entry_price: float,
        atr: float,
        direction: str,
    ) -> float:
        """
        Calculate initial stop-loss based on ATR.
        
        Args:
            entry_price: Entry price of the position
            atr: Current ATR value
            direction: "BUY" or "SELL"
            
        Returns:
            Stop-loss price
        """
        sl_distance = self.atr_mult * atr
        
        # Apply min/max constraints
        min_distance = entry_price * self.min_stop_pct
        max_distance = entry_price * self.max_stop_pct
        sl_distance = max(min_distance, min(max_distance, sl_distance))
        
        if direction.upper() in ["BUY", "LONG"]:
            return entry_price - sl_distance
        else:  # SELL/SHORT
            return entry_price + sl_distance
    
    def calculate_initial_tp(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_reward: Optional[float] = None,
    ) -> float:
        """
        Calculate take-profit based on risk-reward ratio.
        
        Args:
            entry_price: Entry price of the position
            stop_loss: Stop-loss price
            direction: "BUY" or "SELL"
            risk_reward: Override default risk:reward ratio
            
        Returns:
            Take-profit price
        """
        rr = risk_reward or self.risk_reward
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * rr
        
        if direction.upper() in ["BUY", "LONG"]:
            return entry_price + tp_distance
        else:  # SELL/SHORT
            return entry_price - tp_distance
    
    def calculate_levels(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        risk_reward: Optional[float] = None,
    ) -> StopLossLevels:
        """
        Calculate all stop-loss and take-profit levels.
        
        Args:
            entry_price: Entry price of the position
            atr: Current ATR value
            direction: "BUY" or "SELL"
            risk_reward: Override default risk:reward ratio
            
        Returns:
            StopLossLevels with SL, TP, and metadata
        """
        sl = self.calculate_initial_sl(entry_price, atr, direction)
        tp = self.calculate_initial_tp(entry_price, sl, direction, risk_reward)
        
        return StopLossLevels(
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            atr=atr,
            risk_reward_ratio=risk_reward or self.risk_reward,
        )
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        current_sl: float,
        atr: float,
        direction: str,
    ) -> Tuple[float, bool]:
        """
        Calculate updated trailing stop-loss.
        
        The trailing stop only moves in the profitable direction:
        - BUY: SL can only move UP (never down)
        - SELL: SL can only move DOWN (never up)
        
        Args:
            current_price: Current market price
            current_sl: Current stop-loss level
            atr: Current ATR value
            direction: "BUY" or "SELL"
            
        Returns:
            Tuple of (new_sl, should_update)
        """
        trail_distance = self.trail_mult * atr
        
        if direction.upper() in ["BUY", "LONG"]:
            # For long positions, trail below current price
            new_sl = current_price - trail_distance
            # Only move up, never down
            if new_sl > current_sl:
                return round(new_sl, 5), True
            return current_sl, False
        else:  # SELL/SHORT
            # For short positions, trail above current price
            new_sl = current_price + trail_distance
            # Only move down, never up
            if new_sl < current_sl:
                return round(new_sl, 5), True
            return current_sl, False
    
    def calculate_breakeven_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        buffer_atr: float = 0.0,
        atr: float = 0.0,
    ) -> Tuple[float, bool]:
        """
        Calculate breakeven stop-loss (entry price + small buffer).
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            direction: "BUY" or "SELL"
            buffer_atr: ATR multiplier for buffer above breakeven
            atr: Current ATR value (for buffer calculation)
            
        Returns:
            Tuple of (breakeven_sl, is_profitable_enough)
        """
        buffer = buffer_atr * atr if atr > 0 else 0
        
        if direction.upper() in ["BUY", "LONG"]:
            breakeven_sl = entry_price + buffer
            # Only move to breakeven if price is sufficiently above entry
            min_profit = entry_price * 0.005  # At least 0.5% profit
            is_profitable = current_price > entry_price + min_profit
            return round(breakeven_sl, 5), is_profitable
        else:  # SELL/SHORT
            breakeven_sl = entry_price - buffer
            min_profit = entry_price * 0.005
            is_profitable = current_price < entry_price - min_profit
            return round(breakeven_sl, 5), is_profitable
    
    def suggest_stop_adjustment(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        current_tp: float,
        atr: float,
        direction: str,
    ) -> dict:
        """
        Suggest stop-loss adjustments based on current position state.
        
        Returns a dict with suggestions for:
        - breakeven: Move SL to breakeven if profitable
        - trailing: Trail SL behind price
        - partial_tp: Partial take-profit levels
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            current_sl: Current stop-loss
            current_tp: Current take-profit
            atr: Current ATR value
            direction: "BUY" or "SELL"
            
        Returns:
            Dict with adjustment suggestions
        """
        suggestions = {
            "current": {
                "entry": entry_price,
                "price": current_price,
                "sl": current_sl,
                "tp": current_tp,
                "atr": atr,
            },
            "breakeven": None,
            "trailing": None,
            "partial_tp": [],
            "recommendation": "",
        }
        
        # Calculate P&L percentage
        if direction.upper() in ["BUY", "LONG"]:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        suggestions["pnl_percent"] = round(pnl_pct, 2)
        
        # Breakeven suggestion
        be_sl, be_eligible = self.calculate_breakeven_stop(
            entry_price, current_price, direction, buffer_atr=0.1, atr=atr
        )
        if be_eligible:
            suggestions["breakeven"] = {
                "new_sl": be_sl,
                "reason": f"Position is +{pnl_pct:.2f}% - move SL to breakeven to eliminate risk",
            }
        
        # Trailing stop suggestion
        trail_sl, should_trail = self.calculate_trailing_stop(
            current_price, current_sl, atr, direction
        )
        if should_trail:
            suggestions["trailing"] = {
                "new_sl": trail_sl,
                "distance": round(abs(current_price - trail_sl), 5),
                "reason": f"Trail SL to {trail_sl} ({self.trail_mult}x ATR from current price)",
            }
        
        # Partial take-profit suggestions
        if current_tp and current_tp > 0:
            if direction.upper() in ["BUY", "LONG"]:
                tp_distance = current_tp - entry_price
                partial_1 = entry_price + (tp_distance * 0.5)
                partial_2 = entry_price + (tp_distance * 0.75)
            else:
                tp_distance = entry_price - current_tp
                partial_1 = entry_price - (tp_distance * 0.5)
                partial_2 = entry_price - (tp_distance * 0.75)
            
            suggestions["partial_tp"] = [
                {"level": round(partial_1, 5), "percent": 50, "reason": "Take 50% at halfway to TP"},
                {"level": round(partial_2, 5), "percent": 25, "reason": "Take 25% at 75% to TP"},
            ]
        
        # Generate recommendation
        if pnl_pct < 0:
            suggestions["recommendation"] = "Position is in loss. Consider holding if thesis intact, or cut loss if invalidated."
        elif pnl_pct < 0.5:
            suggestions["recommendation"] = "Position is near breakeven. Wait for clearer direction."
        elif be_eligible and (current_sl == 0 or current_sl < be_sl if direction.upper() in ["BUY", "LONG"] else current_sl > be_sl):
            suggestions["recommendation"] = f"MOVE SL TO BREAKEVEN at {be_sl} to protect capital."
        elif should_trail:
            suggestions["recommendation"] = f"TRAIL SL to {trail_sl} to lock in profits."
        else:
            suggestions["recommendation"] = "SL is well-positioned. Monitor for trailing opportunities."
        
        return suggestions


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period (default 14)
        
    Returns:
        Current ATR value
    """
    if len(close) < period + 1:
        # Not enough data, use simple range
        return float(np.mean(high - low))
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Simple moving average of TR
    atr = np.mean(true_range[-period:])
    
    return float(atr)


def get_atr_for_symbol(symbol: str, period: int = 14, timeframe: str = "D1") -> float:
    """
    Get current ATR for a symbol from MT5.
    
    Args:
        symbol: MT5 symbol (e.g., XAUUSD)
        period: ATR period
        timeframe: Timeframe for ATR calculation
        
    Returns:
        Current ATR value
    """
    try:
        from tradingagents.dataflows.mt5_data import get_mt5_indicator
        from datetime import datetime
        
        result = get_mt5_indicator(
            symbol=symbol,
            indicator="atr",
            curr_date=datetime.now().strftime("%Y-%m-%d"),
            look_back_days=period * 2,
            period=period,
            timeframe=timeframe,
        )
        
        # Parse ATR from result string
        for line in result.split("\n"):
            if "ATR" in line and ":" in line:
                try:
                    atr_value = float(line.split(":")[-1].strip())
                    return atr_value
                except ValueError:
                    continue
        
        return 0.0
        
    except Exception as e:
        print(f"Error getting ATR for {symbol}: {e}")
        return 0.0
