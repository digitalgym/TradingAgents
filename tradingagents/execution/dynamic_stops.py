"""
Dynamic Stop Manager

Manages dynamic and trailing stops based on plan rules:
- ATR-based stops
- Trailing stops with price triggers
- Adjusts stops as price moves
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import MetaTrader5 as mt5
import numpy as np


@dataclass
class StopRule:
    """A stop loss adjustment rule"""
    trigger_price: float
    new_stop: float
    description: str
    triggered: bool = False


class DynamicStopManager:
    """
    Manages dynamic stop loss adjustments.
    
    Features:
    - ATR-based initial stops
    - Trailing stops with triggers
    - Automatic adjustment as price moves
    - Monitors and applies rules
    """
    
    def __init__(self, symbol: str, direction: str):
        self.symbol = symbol
        self.direction = direction  # BUY or SELL
        self.current_stop = 0.0
        self.rules: List[StopRule] = []
        self.mt5_initialized = False
    
    def _ensure_mt5(self):
        """Ensure MT5 is initialized"""
        if not self.mt5_initialized:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")
            self.mt5_initialized = True
    
    def calculate_atr_stop(
        self,
        entry_price: float,
        atr_multiple: float = 2.0,
        period: int = 14
    ) -> float:
        """
        Calculate ATR-based stop loss.
        
        Args:
            entry_price: Entry price
            atr_multiple: ATR multiplier (e.g., 2.0 for 2x ATR)
            period: ATR period
        
        Returns:
            Stop loss price
        """
        self._ensure_mt5()
        
        # Get recent price data
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, period + 1)
        
        if rates is None or len(rates) < period:
            # Fallback: Use 2% stop
            fallback_pct = 0.02
            if self.direction == "BUY":
                return entry_price * (1 - fallback_pct)
            else:
                return entry_price * (1 + fallback_pct)
        
        # Calculate ATR
        high = np.array([r['high'] for r in rates])
        low = np.array([r['low'] for r in rates])
        close = np.array([r['close'] for r in rates])
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        # Calculate stop
        stop_distance = atr * atr_multiple
        
        if self.direction == "BUY":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        return stop_price
    
    def set_initial_stop(
        self,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr_multiple: Optional[float] = None
    ):
        """
        Set initial stop loss.
        
        Args:
            entry_price: Entry price
            stop_price: Explicit stop price (if provided)
            atr_multiple: ATR multiple (if stop_price not provided)
        """
        if stop_price:
            self.current_stop = stop_price
        elif atr_multiple:
            self.current_stop = self.calculate_atr_stop(entry_price, atr_multiple)
        else:
            # Default: 2x ATR
            self.current_stop = self.calculate_atr_stop(entry_price, 2.0)
    
    def add_trailing_rule(
        self,
        trigger_price: float,
        new_stop: float,
        description: str = ""
    ):
        """
        Add a trailing stop rule.
        
        Args:
            trigger_price: Price that triggers the stop adjustment
            new_stop: New stop loss price when triggered
            description: Description of the rule
        """
        self.rules.append(StopRule(
            trigger_price=trigger_price,
            new_stop=new_stop,
            description=description or f"Trail to ${new_stop:.2f} after ${trigger_price:.2f}"
        ))
    
    def parse_trailing_rules(self, trail_rules: List[str]):
        """
        Parse trailing rules from plan text.
        
        Args:
            trail_rules: List of rule strings like "Trail to $75 after $82"
        """
        import re
        
        for rule_text in trail_rules:
            # Extract prices
            # "Trail to $75 post-$82 close"
            # "trails to $75 after $82"
            match = re.search(r'to\s+\$(\d+(?:\.\d+)?)\s+(?:post|after)[- ]?\$(\d+(?:\.\d+)?)', rule_text, re.IGNORECASE)
            
            if match:
                new_stop = float(match.group(1))
                trigger_price = float(match.group(2))
                
                self.add_trailing_rule(
                    trigger_price=trigger_price,
                    new_stop=new_stop,
                    description=rule_text
                )
    
    def check_and_update(self, current_price: float) -> Dict[str, Any]:
        """
        Check if any rules should trigger and update stop.
        
        Args:
            current_price: Current market price
        
        Returns:
            dict with update info
        """
        updates = []
        
        for rule in self.rules:
            if rule.triggered:
                continue
            
            # Check if rule should trigger
            should_trigger = False
            
            if self.direction == "BUY":
                # For BUY, trigger when price goes above trigger
                should_trigger = current_price >= rule.trigger_price
            else:
                # For SELL, trigger when price goes below trigger
                should_trigger = current_price <= rule.trigger_price
            
            if should_trigger:
                # Check if new stop is better than current
                is_better = False
                
                if self.direction == "BUY":
                    # For BUY, better stop is higher
                    is_better = rule.new_stop > self.current_stop
                else:
                    # For SELL, better stop is lower
                    is_better = rule.new_stop < self.current_stop
                
                if is_better:
                    old_stop = self.current_stop
                    self.current_stop = rule.new_stop
                    rule.triggered = True
                    
                    updates.append({
                        "rule": rule.description,
                        "old_stop": old_stop,
                        "new_stop": self.current_stop,
                        "trigger_price": rule.trigger_price,
                        "current_price": current_price
                    })
        
        return {
            "updated": len(updates) > 0,
            "updates": updates,
            "current_stop": self.current_stop
        }
    
    def calculate_breakeven_stop(self, entry_price: float, buffer_pct: float = 0.001) -> float:
        """
        Calculate breakeven stop (entry + small buffer).
        
        Args:
            entry_price: Entry price
            buffer_pct: Buffer percentage (default 0.1%)
        
        Returns:
            Breakeven stop price
        """
        if self.direction == "BUY":
            return entry_price * (1 + buffer_pct)
        else:
            return entry_price * (1 - buffer_pct)
    
    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        threshold_pct: float = 1.0
    ) -> bool:
        """
        Check if stop should be moved to breakeven.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            threshold_pct: % move required to go to breakeven
        
        Returns:
            True if should move to breakeven
        """
        if self.direction == "BUY":
            move_pct = ((current_price - entry_price) / entry_price) * 100
            return move_pct >= threshold_pct
        else:
            move_pct = ((entry_price - current_price) / entry_price) * 100
            return move_pct >= threshold_pct
    
    def get_stop_status(self, current_price: float, entry_price: float) -> Dict[str, Any]:
        """
        Get current stop status and recommendations.
        
        Args:
            current_price: Current price
            entry_price: Entry price
        
        Returns:
            dict with stop status
        """
        # Calculate distances
        if self.direction == "BUY":
            stop_distance = current_price - self.current_stop
            stop_distance_pct = (stop_distance / current_price) * 100
            profit_distance = current_price - entry_price
            profit_distance_pct = (profit_distance / entry_price) * 100
        else:
            stop_distance = self.current_stop - current_price
            stop_distance_pct = (stop_distance / current_price) * 100
            profit_distance = entry_price - current_price
            profit_distance_pct = (profit_distance / entry_price) * 100
        
        # Check for pending rules
        pending_rules = [r for r in self.rules if not r.triggered]
        next_rule = None
        
        if pending_rules:
            if self.direction == "BUY":
                # Next rule is the one with lowest trigger above current
                candidates = [r for r in pending_rules if r.trigger_price > current_price]
                if candidates:
                    next_rule = min(candidates, key=lambda r: r.trigger_price)
            else:
                # Next rule is the one with highest trigger below current
                candidates = [r for r in pending_rules if r.trigger_price < current_price]
                if candidates:
                    next_rule = max(candidates, key=lambda r: r.trigger_price)
        
        # Recommendations
        recommendations = []
        
        # Check if should move to breakeven
        if self.should_move_to_breakeven(entry_price, current_price, 1.0):
            be_stop = self.calculate_breakeven_stop(entry_price)
            if self.direction == "BUY":
                if be_stop > self.current_stop:
                    recommendations.append(f"Consider moving stop to breakeven (${be_stop:.2f})")
            else:
                if be_stop < self.current_stop:
                    recommendations.append(f"Consider moving stop to breakeven (${be_stop:.2f})")
        
        # Check if stop is too tight
        if abs(stop_distance_pct) < 0.5:
            recommendations.append("⚠️  Stop very tight (<0.5%), risk of premature exit")
        
        return {
            "current_stop": self.current_stop,
            "stop_distance": stop_distance,
            "stop_distance_pct": stop_distance_pct,
            "profit_distance": profit_distance,
            "profit_distance_pct": profit_distance_pct,
            "triggered_rules": len([r for r in self.rules if r.triggered]),
            "pending_rules": len(pending_rules),
            "next_rule": {
                "trigger_price": next_rule.trigger_price,
                "new_stop": next_rule.new_stop,
                "description": next_rule.description
            } if next_rule else None,
            "recommendations": recommendations
        }
    
    def apply_stop_to_position(self, ticket: int) -> Dict[str, Any]:
        """
        Apply current stop to an MT5 position.
        
        Args:
            ticket: Position ticket
        
        Returns:
            dict with result
        """
        self._ensure_mt5()
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {
                "success": False,
                "error": f"Position {ticket} not found"
            }
        
        pos = position[0]
        
        # Check if stop needs updating
        if abs(pos.sl - self.current_stop) < 0.01:
            return {
                "success": True,
                "updated": False,
                "reason": "Stop already at target"
            }
        
        # Modify position
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": self.current_stop,
            "tp": pos.tp,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Failed to update stop: {result.comment if result else 'None'}"
            }
        
        return {
            "success": True,
            "updated": True,
            "old_stop": pos.sl,
            "new_stop": self.current_stop
        }
    
    def format_status_report(self, current_price: float, entry_price: float) -> str:
        """Format a readable status report"""
        status = self.get_stop_status(current_price, entry_price)
        
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"DYNAMIC STOP STATUS: {self.symbol} {self.direction}")
        lines.append(f"{'='*70}\n")
        
        lines.append(f"Current Price: ${current_price:.2f}")
        lines.append(f"Entry Price: ${entry_price:.2f}")
        lines.append(f"Current Stop: ${status['current_stop']:.2f}\n")
        
        lines.append(f"Position Status:")
        lines.append(f"  Profit: ${status['profit_distance']:.2f} ({status['profit_distance_pct']:+.2f}%)")
        lines.append(f"  Stop Distance: ${status['stop_distance']:.2f} ({status['stop_distance_pct']:.2f}%)\n")
        
        lines.append(f"Trailing Rules:")
        lines.append(f"  Triggered: {status['triggered_rules']}")
        lines.append(f"  Pending: {status['pending_rules']}")
        
        if status['next_rule']:
            nr = status['next_rule']
            lines.append(f"\n  Next Rule:")
            lines.append(f"    Trigger: ${nr['trigger_price']:.2f}")
            lines.append(f"    New Stop: ${nr['new_stop']:.2f}")
            lines.append(f"    {nr['description']}")
        
        if status['recommendations']:
            lines.append(f"\nRecommendations:")
            for rec in status['recommendations']:
                lines.append(f"  {rec}")
        
        lines.append(f"\n{'='*70}")
        
        return '\n'.join(lines)
