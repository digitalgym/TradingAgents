# TradingAgents Position Sizing Module
"""
Position sizing strategies for risk-adjusted trade sizing.

Provides:
- Kelly Criterion: Optimal fraction of capital to risk
- Fixed Fractional: Risk fixed percentage per trade
- Volatility-Based: Size based on ATR/volatility
- Risk Parity: Equal risk contribution per position
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    recommended_size: float  # Number of units/shares/lots
    risk_amount: float  # Dollar amount at risk
    risk_percent: float  # Percentage of capital at risk
    position_value: float  # Total position value
    method: str  # Sizing method used
    kelly_fraction: Optional[float] = None  # Raw Kelly fraction if applicable
    confidence_adjustment: float = 1.0  # Confidence-based adjustment factor
    
    def to_dict(self) -> Dict:
        return {
            "recommended_size": round(self.recommended_size, 4),
            "risk_amount": round(self.risk_amount, 2),
            "risk_percent": round(self.risk_percent, 4),
            "position_value": round(self.position_value, 2),
            "method": self.method,
            "kelly_fraction": round(self.kelly_fraction, 4) if self.kelly_fraction else None,
            "confidence_adjustment": round(self.confidence_adjustment, 4),
        }
    
    def __str__(self) -> str:
        return f"""Position Size Recommendation
{'='*40}
Method:            {self.method}
Recommended Size:  {self.recommended_size:.4f} units
Position Value:    ${self.position_value:,.2f}
Risk Amount:       ${self.risk_amount:,.2f}
Risk Percent:      {self.risk_percent*100:.2f}%
{'='*40}"""


class PositionSizer:
    """
    Position sizing calculator with multiple strategies.
    
    Supports:
    - Kelly Criterion (optimal growth)
    - Half-Kelly (conservative Kelly)
    - Fixed Fractional (fixed % risk per trade)
    - Volatility-Based (ATR-adjusted sizing)
    
    Example:
        sizer = PositionSizer(account_balance=100000)
        
        # Kelly-based sizing
        result = sizer.kelly_size(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            entry_price=4500,
            stop_loss=4450
        )
        
        # Fixed fractional sizing
        result = sizer.fixed_fractional_size(
            risk_percent=0.02,
            entry_price=4500,
            stop_loss=4450
        )
    """
    
    def __init__(
        self,
        account_balance: float = 100000,
        max_risk_per_trade: float = 0.02,  # 2% max risk per trade
        max_position_size: float = 0.20,   # 20% max position size
        kelly_fraction: float = 0.5,       # Use half-Kelly by default
    ):
        """
        Initialize position sizer.
        
        Args:
            account_balance: Total account value
            max_risk_per_trade: Maximum risk per trade (0.02 = 2%)
            max_position_size: Maximum position as fraction of account
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        """
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion optimal bet fraction.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win probability
            R = Win/Loss ratio (avg_win / avg_loss)
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning return (decimal)
            avg_loss: Average losing return (decimal, positive value)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Ensure avg_loss is positive
        avg_loss = abs(avg_loss)
        
        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Kelly can be negative (don't bet) or very high
        # Clamp to reasonable range
        return max(0.0, min(kelly, 1.0))
    
    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning return (decimal)
            avg_loss: Average losing return (decimal)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            confidence: Confidence adjustment (0-1), reduces size if < 1
            
        Returns:
            PositionSizeResult with recommended size
        """
        # Calculate raw Kelly fraction
        raw_kelly = self.kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Apply fractional Kelly (e.g., half-Kelly)
        adjusted_kelly = raw_kelly * self.kelly_fraction
        
        # Apply confidence adjustment
        adjusted_kelly *= confidence
        
        # Cap at max risk per trade
        risk_fraction = min(adjusted_kelly, self.max_risk_per_trade)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return PositionSizeResult(
                recommended_size=0,
                risk_amount=0,
                risk_percent=0,
                position_value=0,
                method="kelly",
                kelly_fraction=raw_kelly,
                confidence_adjustment=confidence
            )
        
        # Calculate position size
        risk_amount = self.account_balance * risk_fraction
        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price
        
        # Cap at max position size
        max_position_value = self.account_balance * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value
            risk_amount = position_size * risk_per_unit
            risk_fraction = risk_amount / self.account_balance
        
        return PositionSizeResult(
            recommended_size=position_size,
            risk_amount=risk_amount,
            risk_percent=risk_fraction,
            position_value=position_value,
            method="kelly",
            kelly_fraction=raw_kelly,
            confidence_adjustment=confidence
        )
    
    def fixed_fractional_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None,
        confidence: float = 1.0
    ) -> PositionSizeResult:
        """
        Calculate position size using fixed fractional method.
        
        Risk a fixed percentage of account on each trade.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_percent: Risk per trade (default: max_risk_per_trade)
            confidence: Confidence adjustment (0-1)
            
        Returns:
            PositionSizeResult with recommended size
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        # Apply confidence adjustment
        adjusted_risk = risk_percent * confidence
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return PositionSizeResult(
                recommended_size=0,
                risk_amount=0,
                risk_percent=0,
                position_value=0,
                method="fixed_fractional",
                confidence_adjustment=confidence
            )
        
        # Calculate position size
        risk_amount = self.account_balance * adjusted_risk
        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price
        
        # Cap at max position size
        max_position_value = self.account_balance * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value
            risk_amount = position_size * risk_per_unit
            adjusted_risk = risk_amount / self.account_balance
        
        return PositionSizeResult(
            recommended_size=position_size,
            risk_amount=risk_amount,
            risk_percent=adjusted_risk,
            position_value=position_value,
            method="fixed_fractional",
            confidence_adjustment=confidence
        )
    
    def volatility_size(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        target_risk_percent: Optional[float] = None,
        confidence: float = 1.0
    ) -> PositionSizeResult:
        """
        Calculate position size based on volatility (ATR).
        
        Stop loss is set at entry ± (ATR × multiplier).
        Position sized to risk target_risk_percent.
        
        Args:
            entry_price: Entry price for the trade
            atr: Average True Range value
            atr_multiplier: Multiplier for ATR to set stop distance
            target_risk_percent: Target risk per trade
            confidence: Confidence adjustment (0-1)
            
        Returns:
            PositionSizeResult with recommended size
        """
        if target_risk_percent is None:
            target_risk_percent = self.max_risk_per_trade
        
        # Calculate stop distance based on ATR
        stop_distance = atr * atr_multiplier
        
        # Implied stop loss (for long position)
        stop_loss = entry_price - stop_distance
        
        # Use fixed fractional with ATR-based stop
        return self.fixed_fractional_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=target_risk_percent,
            confidence=confidence
        )
    
    def calculate_lots(
        self,
        position_size: float,
        contract_size: float = 100,
        min_lot: float = 0.01,
        lot_step: float = 0.01
    ) -> float:
        """
        Convert position size to MT5 lot size.
        
        Args:
            position_size: Number of units
            contract_size: Units per lot (100 for forex, 100 oz for gold)
            min_lot: Minimum lot size
            lot_step: Lot size increment
            
        Returns:
            Lot size rounded to valid increment
        """
        lots = position_size / contract_size
        
        # Round to lot step
        lots = round(lots / lot_step) * lot_step
        
        # Ensure minimum
        lots = max(lots, min_lot)
        
        return lots
    
    def size_for_trade(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str = "BUY",
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        atr: Optional[float] = None,
        confidence: float = 1.0,
        method: str = "auto"
    ) -> PositionSizeResult:
        """
        Calculate optimal position size for a trade.
        
        Automatically selects best method based on available data.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: "BUY" or "SELL"
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win (for Kelly)
            avg_loss: Average loss (for Kelly)
            atr: ATR value (for volatility sizing)
            confidence: Trade confidence (0-1)
            method: "kelly", "fixed", "volatility", or "auto"
            
        Returns:
            PositionSizeResult with recommended size
        """
        # Validate stop loss direction
        if direction.upper() == "BUY" and stop_loss >= entry_price:
            raise ValueError("Stop loss must be below entry for BUY")
        if direction.upper() == "SELL" and stop_loss <= entry_price:
            raise ValueError("Stop loss must be above entry for SELL")
        
        # Auto-select method
        if method == "auto":
            if win_rate is not None and avg_win is not None and avg_loss is not None:
                method = "kelly"
            elif atr is not None:
                method = "volatility"
            else:
                method = "fixed"
        
        # Calculate size based on method
        if method == "kelly":
            if win_rate is None or avg_win is None or avg_loss is None:
                raise ValueError("Kelly method requires win_rate, avg_win, avg_loss")
            return self.kelly_size(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence
            )
        elif method == "volatility":
            if atr is None:
                raise ValueError("Volatility method requires atr")
            return self.volatility_size(
                entry_price=entry_price,
                atr=atr,
                confidence=confidence
            )
        else:  # fixed
            return self.fixed_fractional_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence
            )


def calculate_kelly_from_history(trade_returns: list) -> Tuple[float, float, float, float]:
    """
    Calculate Kelly parameters from trade history.
    
    Args:
        trade_returns: List of trade returns (decimals)
        
    Returns:
        Tuple of (win_rate, avg_win, avg_loss, kelly_fraction)
    """
    if not trade_returns:
        return (0.0, 0.0, 0.0, 0.0)
    
    returns = np.array(trade_returns)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0
    
    # Calculate Kelly
    if avg_loss > 0 and win_rate > 0:
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        kelly = max(0.0, min(kelly, 1.0))
    else:
        kelly = 0.0
    
    return (win_rate, avg_win, avg_loss, kelly)


def recommend_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    direction: str,
    trade_history: Optional[list] = None,
    atr: Optional[float] = None,
    confidence: float = 1.0,
    max_risk: float = 0.02
) -> PositionSizeResult:
    """
    Convenience function to get position size recommendation.
    
    Args:
        account_balance: Account value
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: "BUY" or "SELL"
        trade_history: Optional list of past trade returns
        atr: Optional ATR value
        confidence: Trade confidence (0-1)
        max_risk: Maximum risk per trade
        
    Returns:
        PositionSizeResult with recommendation
    """
    sizer = PositionSizer(
        account_balance=account_balance,
        max_risk_per_trade=max_risk
    )
    
    # Get Kelly parameters from history if available
    win_rate, avg_win, avg_loss = None, None, None
    if trade_history and len(trade_history) >= 10:
        win_rate, avg_win, avg_loss, _ = calculate_kelly_from_history(trade_history)
    
    return sizer.size_for_trade(
        entry_price=entry_price,
        stop_loss=stop_loss,
        direction=direction,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        atr=atr,
        confidence=confidence
    )
