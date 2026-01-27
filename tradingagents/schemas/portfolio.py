"""Schema for portfolio diversification suggestions."""

from typing import Optional, List, Dict, Any
from pydantic import Field

from .base import BaseSchema, Priority, SignalType


class CorrelationPair(BaseSchema):
    """Correlation between two symbols."""

    symbol_a: str = Field(..., description="First symbol")
    symbol_b: str = Field(..., description="Second symbol")
    correlation: float = Field(
        ..., ge=-1, le=1, description="Correlation coefficient -1 to 1"
    )
    risk_note: Optional[str] = Field(
        None, description="Risk implication of this correlation"
    )


class SymbolSuggestion(BaseSchema):
    """A single symbol suggestion for portfolio diversification."""

    symbol: str = Field(..., description="Symbol to consider adding")
    reason: str = Field(..., description="Why this symbol is suggested")
    correlation_group: str = Field(
        ...,
        description="Correlation group: precious_metals, currencies, indices, etc.",
    )
    priority: Priority = Field(..., description="Suggestion priority")
    expected_direction: Optional[SignalType] = Field(
        None, description="Expected trade direction"
    )
    correlation_with_portfolio: Optional[float] = Field(
        None, ge=-1, le=1, description="Correlation with current portfolio"
    )


class PortfolioSuggestion(BaseSchema):
    """
    Portfolio diversification suggestions with correlation analysis.

    This schema provides suggestions for portfolio diversification
    based on current holdings and correlation analysis.
    """

    # Suggestions
    suggestions: List[SymbolSuggestion] = Field(
        ..., description="Suggested symbols to add"
    )

    # Portfolio analysis
    portfolio_analysis: str = Field(
        ..., description="Analysis of current portfolio balance"
    )
    concentration_risk: Optional[str] = Field(
        None, description="Assessment of concentration risk"
    )

    # Correlation insights
    high_correlations: List[CorrelationPair] = Field(
        default_factory=list, description="Highly correlated pairs in portfolio"
    )
    diversification_score: Optional[float] = Field(
        None, ge=0, le=100, description="Portfolio diversification score 0-100"
    )

    # Risk notes
    risk_notes: str = Field(..., description="Overall risk considerations")
    recommended_actions: List[str] = Field(
        default_factory=list, description="Specific recommended actions"
    )


class QuickPortfolioSuggestion(BaseSchema):
    """
    Simplified portfolio suggestion schema.

    Compatible with existing PortfolioSuggestionSchema in llm_client.py.
    """

    suggestions: List[SymbolSuggestion] = Field(..., description="Suggested symbols")
    portfolio_analysis: str = Field(
        ..., description="Analysis of current portfolio balance"
    )
    risk_notes: str = Field(..., description="Risk considerations")


class PortfolioRiskAssessment(BaseSchema):
    """
    Overall portfolio risk assessment.
    """

    total_exposure: float = Field(..., description="Total portfolio exposure in lots")
    margin_used_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Margin used as percentage"
    )
    max_drawdown_percent: Optional[float] = Field(
        None, description="Maximum drawdown as percentage"
    )
    correlation_risk: str = Field(
        ..., description="Assessment of correlation risk: low, medium, high"
    )
    concentration_risk: str = Field(
        ..., description="Assessment of concentration risk: low, medium, high"
    )
    overall_risk_level: str = Field(
        ..., description="Overall portfolio risk level: low, medium, high, extreme"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Risk mitigation recommendations"
    )
