"""Schema for position review recommendations."""

from typing import Optional, List, Dict, Any
from pydantic import Field

from .base import (
    BaseSchema,
    Recommendation,
    RiskLevel,
    SignalType,
    Urgency,
)


class StopLossAdjustment(BaseSchema):
    """Recommended stop loss adjustment."""

    current_sl: Optional[float] = Field(None, description="Current stop loss")
    suggested_sl: float = Field(..., description="Suggested new stop loss")
    adjustment_type: str = Field(
        ..., description="Type: breakeven, trailing, or manual"
    )
    rationale: str = Field(..., description="Why this adjustment is recommended")


class TakeProfitAdjustment(BaseSchema):
    """Recommended take profit adjustment."""

    current_tp: Optional[float] = Field(None, description="Current take profit")
    suggested_tp: float = Field(..., description="Suggested new take profit")
    partial_close_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Percentage to close at this level"
    )
    rationale: str = Field(..., description="Why this adjustment is recommended")


class PositionReview(BaseSchema):
    """
    Position review result with HOLD/CLOSE/ADJUST recommendation.

    This schema is used for LLM-based position reviews that analyze
    open positions and provide management recommendations.
    """

    # Position identification
    symbol: str = Field(..., description="Trading symbol")
    ticket: Optional[int] = Field(None, description="MT5 ticket number")
    direction: SignalType = Field(..., description="Position direction: BUY or SELL")

    # Core recommendation
    recommendation: Recommendation = Field(
        ..., description="HOLD, CLOSE, or ADJUST"
    )
    urgency: Urgency = Field(
        Urgency.NORMAL, description="Urgency: immediate, high, normal, low"
    )

    # Current position metrics
    entry_price: float = Field(..., description="Position entry price")
    current_price: float = Field(..., description="Current market price")
    current_pnl_percent: float = Field(..., description="Current P&L as percentage")
    current_pnl_amount: Optional[float] = Field(
        None, description="Current P&L in account currency"
    )

    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Current risk assessment")
    distance_to_sl_pct: Optional[float] = Field(
        None, description="Distance to SL as percentage"
    )
    distance_to_tp_pct: Optional[float] = Field(
        None, description="Distance to TP as percentage"
    )

    # Adjustments (if recommendation is ADJUST)
    suggested_sl: Optional[float] = Field(
        None, description="Suggested stop loss, null if no change"
    )
    suggested_tp: Optional[float] = Field(
        None, description="Suggested take profit, null if no change"
    )
    sl_adjustment: Optional[StopLossAdjustment] = Field(
        None, description="Detailed SL adjustment"
    )
    tp_adjustment: Optional[TakeProfitAdjustment] = Field(
        None, description="Detailed TP adjustment"
    )

    # Close reasoning (if recommendation is CLOSE)
    close_reason: Optional[str] = Field(
        None, description="Reason for closing if CLOSE recommended"
    )

    # Supporting analysis
    reasoning: str = Field(
        ..., description="Detailed reasoning for the recommendation"
    )
    key_observations: List[str] = Field(
        default_factory=list, description="Key market observations"
    )

    # SMC context
    smc_context: Optional[str] = Field(None, description="SMC-based analysis context")


class QuickPositionReview(BaseSchema):
    """
    Simplified position review schema for quick LLM analysis.

    Compatible with existing TradeReviewSchema in llm_client.py.
    """

    recommendation: Recommendation = Field(
        ..., description="HOLD, CLOSE, or ADJUST"
    )
    suggested_sl: Optional[float] = Field(
        None, description="Suggested stop loss, null if no change"
    )
    suggested_tp: Optional[float] = Field(
        None, description="Suggested take profit, null if no change"
    )
    risk_level: RiskLevel = Field(..., description="Current risk assessment")
    reasoning: str = Field(
        ..., description="Brief explanation of the recommendation"
    )


class BatchPositionReview(BaseSchema):
    """
    Result of reviewing multiple positions in batch.
    """

    reviews: List[QuickPositionReview] = Field(
        ..., description="Individual position reviews"
    )
    portfolio_summary: Optional[str] = Field(
        None, description="Overall portfolio assessment"
    )
    total_positions: int = Field(..., description="Total number of positions reviewed")
    positions_to_adjust: int = Field(
        0, description="Number of positions needing adjustment"
    )
    positions_to_close: int = Field(
        0, description="Number of positions recommended to close"
    )
