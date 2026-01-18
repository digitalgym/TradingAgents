"""
Order Executor

Intelligent order execution that decides:
- Market vs Limit order type
- Order placement timing
- Order management (modify, cancel)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import MetaTrader5 as mt5


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderDecision:
    """Decision on how to execute an order"""
    order_type: OrderType
    price: float
    reason: str
    urgency: str  # low, medium, high
    expiration: Optional[datetime] = None


class OrderExecutor:
    """
    Intelligent order executor that decides market vs limit.
    
    Decision logic:
    - Market: When price is favorable and urgency is high
    - Limit: When price needs improvement or urgency is low
    - Considers spread, volatility, and distance to target
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.mt5_initialized = False
    
    def _ensure_mt5(self):
        """Ensure MT5 is initialized"""
        if not self.mt5_initialized:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")
            self.mt5_initialized = True
    
    def decide_order_type(
        self,
        direction: str,
        target_price: float,
        price_range: Optional[tuple] = None,
        conditions: list = None
    ) -> OrderDecision:
        """
        Decide whether to use market or limit order.
        
        Args:
            direction: BUY or SELL
            target_price: Target entry price
            price_range: Optional (min, max) acceptable range
            conditions: List of entry conditions
        
        Returns:
            OrderDecision with order type and reasoning
        """
        self._ensure_mt5()
        
        # Get current market data
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise ValueError(f"Could not get tick data for {self.symbol}")
        
        current_price = tick.ask if direction == "BUY" else tick.bid
        spread = tick.ask - tick.bid
        spread_pct = (spread / current_price) * 100
        
        # Calculate distance to target
        if direction == "BUY":
            distance = target_price - current_price
            distance_pct = (distance / current_price) * 100
        else:
            distance = current_price - target_price
            distance_pct = (distance / current_price) * 100
        
        # Decision logic
        urgency = "low"
        order_type = OrderType.LIMIT
        reason = ""
        
        # HIGH URGENCY: Use market order
        # 1. Price already at or better than target
        if (direction == "BUY" and current_price <= target_price) or \
           (direction == "SELL" and current_price >= target_price):
            urgency = "high"
            order_type = OrderType.MARKET
            reason = f"Price at target (current: ${current_price:.2f}, target: ${target_price:.2f})"
        
        # 2. Price within range and conditions likely met
        elif price_range:
            min_price, max_price = price_range
            if min_price <= current_price <= max_price:
                # Check if we're in favorable part of range
                if direction == "BUY":
                    # In buy range - closer to min is better
                    range_position = (current_price - min_price) / (max_price - min_price)
                    if range_position < 0.3:  # In lower 30% of range
                        urgency = "high"
                        order_type = OrderType.MARKET
                        reason = f"Price in favorable range (${min_price:.2f}-${max_price:.2f})"
                    else:
                        urgency = "medium"
                        order_type = OrderType.LIMIT
                        reason = f"Price in range but wait for better entry (target: ${min_price:.2f})"
                else:
                    # In sell range - closer to max is better
                    range_position = (current_price - min_price) / (max_price - min_price)
                    if range_position > 0.7:  # In upper 30% of range
                        urgency = "high"
                        order_type = OrderType.MARKET
                        reason = f"Price in favorable range (${min_price:.2f}-${max_price:.2f})"
                    else:
                        urgency = "medium"
                        order_type = OrderType.LIMIT
                        reason = f"Price in range but wait for better entry (target: ${max_price:.2f})"
        
        # MEDIUM URGENCY: Use limit order close to market
        # 3. Price within 0.5% of target
        elif abs(distance_pct) < 0.5:
            urgency = "medium"
            order_type = OrderType.LIMIT
            reason = f"Price close to target ({distance_pct:+.2f}%), use limit at ${target_price:.2f}"
        
        # LOW URGENCY: Use limit order at target
        # 4. Price more than 0.5% away
        else:
            urgency = "low"
            order_type = OrderType.LIMIT
            reason = f"Price {abs(distance_pct):.2f}% from target, use limit at ${target_price:.2f}"
        
        # Adjust for high spread
        if spread_pct > 0.1:  # Spread > 0.1%
            if order_type == OrderType.MARKET:
                order_type = OrderType.LIMIT
                reason += f" (High spread {spread_pct:.3f}%, using limit instead)"
        
        return OrderDecision(
            order_type=order_type,
            price=target_price if order_type == OrderType.LIMIT else current_price,
            reason=reason,
            urgency=urgency
        )
    
    def execute_order(
        self,
        direction: str,
        volume: float,
        order_decision: OrderDecision,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Execute an order in MT5.
        
        Args:
            direction: BUY or SELL
            volume: Position size in lots
            order_decision: OrderDecision from decide_order_type()
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            comment: Order comment
        
        Returns:
            dict with order result
        """
        self._ensure_mt5()
        
        # Get symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return {
                "success": False,
                "error": f"Symbol {self.symbol} not found"
            }
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_decision.order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "deviation": 20,
            "magic": 234000,
            "comment": comment or f"TradingAgents {order_decision.order_type.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Set price based on order type
        if order_decision.order_type == OrderType.MARKET:
            tick = mt5.symbol_info_tick(self.symbol)
            request["price"] = tick.ask if direction == "BUY" else tick.bid
        else:
            request["price"] = order_decision.price
            # For limit orders, set the correct order type
            if direction == "BUY":
                request["type"] = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                request["type"] = mt5.ORDER_TYPE_SELL_LIMIT
        
        # Add SL/TP if provided
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            return {
                "success": False,
                "error": "order_send returned None"
            }
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Order failed: {result.comment}",
                "retcode": result.retcode
            }
        
        return {
            "success": True,
            "order_id": result.order,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "order_type": order_decision.order_type.value,
            "comment": result.comment
        }
    
    def modify_order(
        self,
        ticket: int,
        new_price: Optional[float] = None,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            ticket: Order ticket number
            new_price: New entry price (for pending orders)
            new_sl: New stop loss
            new_tp: New take profit
        
        Returns:
            dict with result
        """
        self._ensure_mt5()
        
        # Get current order
        order = mt5.orders_get(ticket=ticket)
        if not order:
            # Check if it's a position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {
                    "success": False,
                    "error": f"Order/position {ticket} not found"
                }
            
            # Modify position (SL/TP only)
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position[0].symbol,
                "position": ticket,
                "sl": new_sl or position[0].sl,
                "tp": new_tp or position[0].tp,
            }
        else:
            # Modify pending order
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "price": new_price or order[0].price_open,
                "sl": new_sl or order[0].sl,
                "tp": new_tp or order[0].tp,
            }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Modify failed: {result.comment if result else 'None'}"
            }
        
        return {
            "success": True,
            "ticket": ticket,
            "modified": True
        }
    
    def cancel_order(self, ticket: int) -> Dict[str, Any]:
        """
        Cancel a pending order.
        
        Args:
            ticket: Order ticket number
        
        Returns:
            dict with result
        """
        self._ensure_mt5()
        
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Cancel failed: {result.comment if result else 'None'}"
            }
        
        return {
            "success": True,
            "ticket": ticket,
            "cancelled": True
        }
    
    def get_order_status(self, ticket: int) -> OrderStatus:
        """
        Get current status of an order.
        
        Args:
            ticket: Order ticket number
        
        Returns:
            OrderStatus
        """
        self._ensure_mt5()
        
        # Check pending orders
        order = mt5.orders_get(ticket=ticket)
        if order:
            return OrderStatus.PENDING
        
        # Check positions (filled orders)
        position = mt5.positions_get(ticket=ticket)
        if position:
            return OrderStatus.FILLED
        
        # Check history
        from datetime import datetime, timedelta
        history = mt5.history_orders_get(
            datetime.now() - timedelta(days=7),
            datetime.now(),
            ticket=ticket
        )
        
        if history:
            state = history[0].state
            if state == mt5.ORDER_STATE_FILLED:
                return OrderStatus.FILLED
            elif state == mt5.ORDER_STATE_CANCELED:
                return OrderStatus.CANCELLED
            elif state == mt5.ORDER_STATE_REJECTED:
                return OrderStatus.REJECTED
            elif state == mt5.ORDER_STATE_PARTIAL:
                return OrderStatus.PARTIAL
        
        return OrderStatus.CANCELLED  # Assume cancelled if not found
