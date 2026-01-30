"""Schema for trade analysis results from the multi-agent system."""

from typing import Optional, List, Dict, Any
from pydantic import Field

from .base import (
    BaseSchema,
    SignalType,
    RiskLevel,
    MarketRegime,
    VolatilityRegime,
    KeyFactor,
)


class SMCZone(BaseSchema):
    """Smart Money Concept zone information."""

    zone_type: str = Field(
        ..., description="Zone type: FVG, OB, BOS, CHOCH, liquidity"
    )
    direction: SignalType = Field(..., description="Bullish or bearish zone")
    price_start: float = Field(..., description="Zone start price")
    price_end: float = Field(..., description="Zone end price")
    strength: Optional[float] = Field(
        None, ge=0, le=100, description="Zone strength 0-100%"
    )
    mitigated: bool = Field(False, description="Whether zone has been mitigated")
    timeframe: Optional[str] = Field(
        None, description="Timeframe this zone was identified on"
    )


class SMCAnalysis(BaseSchema):
    """Smart Money Concepts analysis summary."""

    bias: SignalType = Field(..., description="Overall SMC bias")
    key_zones: List[SMCZone] = Field(
        default_factory=list, description="Key SMC zones"
    )
    structure: Optional[str] = Field(
        None, description="Current market structure description"
    )
    liquidity_levels: List[float] = Field(
        default_factory=list, description="Key liquidity levels"
    )


class TradeAnalysisResult(BaseSchema):
    """
    Complete trade analysis result from the multi-agent system.

    This schema captures the output of a full analysis run including
    signal, confidence, entry/SL/TP levels, and supporting rationale.
    """

    # Core signal
    symbol: str = Field(..., description="Trading symbol analyzed")
    signal: SignalType = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence 0-1")

    # Trade parameters
    entry_price: Optional[float] = Field(None, description="Recommended entry price")
    stop_loss: Optional[float] = Field(None, description="Recommended stop loss")
    take_profit: Optional[float] = Field(None, description="Recommended take profit")
    take_profit_2: Optional[float] = Field(
        None, description="Second take profit target"
    )
    take_profit_3: Optional[float] = Field(None, description="Third take profit target")

    # Position sizing
    recommended_volume: Optional[float] = Field(
        None, description="Recommended position size in lots"
    )
    risk_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Risk as percentage of account"
    )

    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Overall risk assessment")
    risk_reward_ratio: Optional[float] = Field(
        None, description="Risk-to-reward ratio"
    )

    # Market context
    market_regime: Optional[MarketRegime] = Field(
        None, description="Current market regime"
    )
    volatility_regime: Optional[VolatilityRegime] = Field(
        None, description="Current volatility regime"
    )
    atr: Optional[float] = Field(None, description="Average True Range value")

    # SMC context (optional)
    smc_analysis: Optional[SMCAnalysis] = Field(
        None, description="Smart Money Concepts analysis"
    )

    # Reasoning
    rationale: str = Field(
        ..., description="Summary rationale for the recommendation"
    )
    key_factors: List[KeyFactor] = Field(
        default_factory=list, description="Key factors influencing the decision"
    )

    # Agent reports (summary versions)
    market_summary: Optional[str] = Field(None, description="Market analyst summary")
    news_summary: Optional[str] = Field(None, description="News analyst summary")
    sentiment_summary: Optional[str] = Field(
        None, description="Sentiment analyst summary"
    )

    # Metadata
    analysis_date: str = Field(..., description="Date of analysis (YYYY-MM-DD)")
    timeframe: str = Field("H1", description="Primary timeframe analyzed")
    analysis_duration_seconds: Optional[float] = Field(
        None, description="Time taken for analysis"
    )


class QuickTradeAnalysis(BaseSchema):
    """
    Simplified trade analysis schema for quick LLM analysis.

    Use this for rapid signal extraction without full analysis context.
    """

    signal: SignalType = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence 0-1")
    entry_price: Optional[float] = Field(None, description="Recommended entry price")
    stop_loss: Optional[float] = Field(None, description="Recommended stop loss")
    take_profit: Optional[float] = Field(None, description="Recommended take profit")
    rationale: str = Field(..., description="Brief explanation of the signal")


class FinalTradingDecision(BaseSchema):
    """
    Structured output from the Risk Manager's final trading decision.

    This schema is used for LLM structured outputs to guarantee consistent
    JSON format without needing a secondary parsing step.
    """

    # Core decision
    signal: SignalType = Field(
        ..., description="Final trading decision: BUY, SELL, or HOLD"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in this decision (0-1)"
    )

    # Trade parameters (refined from trader's plan)
    entry_price: Optional[float] = Field(
        None, description="Recommended entry price. Use null if not explicitly determined."
    )
    stop_loss: Optional[float] = Field(
        None, description="Recommended stop loss price. Use null if not explicitly determined."
    )
    take_profit: Optional[float] = Field(
        None, description="Primary take profit target. Use null if not explicitly determined."
    )
    take_profit_2: Optional[float] = Field(
        None, description="Secondary take profit target for scaling out"
    )

    # Risk assessment
    risk_level: RiskLevel = Field(
        ..., description="Overall risk level: Low, Medium, High, or Extreme"
    )
    risk_reward_ratio: Optional[float] = Field(
        None, ge=0, description="Calculated risk-to-reward ratio"
    )

    # Position sizing guidance
    position_size_recommendation: Optional[str] = Field(
        None, description="Guidance on position sizing (e.g., 'reduced', 'standard', 'aggressive')"
    )

    # Reasoning
    rationale: str = Field(
        ..., description="Detailed rationale for the decision, including key arguments from analysts"
    )
    key_risks: Optional[str] = Field(
        None, description="Primary risks identified that could invalidate this trade"
    )
    key_catalysts: Optional[str] = Field(
        None, description="Key factors that could drive the trade in the expected direction"
    )


class PredictionLesson(BaseSchema):
    """
    Structured lesson from evaluating a prediction against actual outcome.

    Used by daily_cycle.py to extract actionable lessons from predictions.
    """

    analysis: str = Field(
        ...,
        description="Analysis of what happened and why the prediction was correct/incorrect"
    )
    predictive_factors: List[str] = Field(
        default_factory=list,
        description="Factors from the analysis that correctly predicted the outcome"
    )
    misleading_factors: List[str] = Field(
        default_factory=list,
        description="Factors that led to incorrect conclusions or were overweighted"
    )
    lesson: str = Field(
        ...,
        description="Concise, actionable lesson (2-3 sentences) for improving future predictions"
    )
    confidence_adjustment: float = Field(
        0.0,
        ge=-0.5,
        le=0.5,
        description="Suggested adjustment to confidence scoring for similar situations (-0.5 to +0.5)"
    )
    similar_pattern_advice: Optional[str] = Field(
        None,
        description="Specific advice for when similar market patterns occur in the future"
    )
