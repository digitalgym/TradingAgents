"""
TradingAgents Schemas Module

Centralized Pydantic models for structured LLM outputs and API responses.
Supports both xAI Responses API and OpenAI Chat Completions structured outputs.

Usage with llm_client.py:
    from tradingagents.schemas import TradeAnalysisResult, PositionReview
    from tradingagents.dataflows.llm_client import structured_output

    # Using Pydantic model directly
    result = structured_output(
        client, model, messages,
        response_schema=PositionReview,
        use_responses_api=True
    )
    print(result.recommendation)  # Typed access

    # Using JSON schema
    result = structured_output(
        client, model, messages,
        response_schema=PositionReview.get_json_schema(),
        use_responses_api=True
    )
    print(result["recommendation"])  # Dict access
"""

from .base import (
    SignalType,
    RiskLevel,
    Recommendation,
    Priority,
    MarketRegime,
    VolatilityRegime,
    Urgency,
    BaseSchema,
    PriceLevel,
    KeyFactor,
)

from .trade_analysis import (
    TradeAnalysisResult,
    QuickTradeAnalysis,
    FinalTradingDecision,
    SMCZone,
    SMCAnalysis,
    PredictionLesson,
    QuantSignalType,
    QuantAnalystDecision,
)

from .position_review import (
    PositionReview,
    QuickPositionReview,
    BatchPositionReview,
    StopLossAdjustment,
    TakeProfitAdjustment,
)

from .portfolio import (
    PortfolioSuggestion,
    QuickPortfolioSuggestion,
    PortfolioRiskAssessment,
    SymbolSuggestion,
    CorrelationPair,
)

# Backward compatibility aliases
TradeReviewResult = QuickPositionReview


def get_all_schemas():
    """Return all schema classes for TypeScript generation."""
    return [
        # Enums
        SignalType,
        RiskLevel,
        Recommendation,
        Priority,
        MarketRegime,
        VolatilityRegime,
        Urgency,
        QuantSignalType,
        # Base
        PriceLevel,
        KeyFactor,
        # Trade Analysis
        SMCZone,
        SMCAnalysis,
        TradeAnalysisResult,
        QuickTradeAnalysis,
        FinalTradingDecision,
        PredictionLesson,
        QuantAnalystDecision,
        # Position Review
        StopLossAdjustment,
        TakeProfitAdjustment,
        PositionReview,
        QuickPositionReview,
        BatchPositionReview,
        # Portfolio
        CorrelationPair,
        SymbolSuggestion,
        PortfolioSuggestion,
        QuickPortfolioSuggestion,
        PortfolioRiskAssessment,
    ]


def generate_json_schemas():
    """Generate JSON schemas for all models (for documentation or tooling)."""
    return {
        schema.__name__: schema.model_json_schema()
        for schema in get_all_schemas()
        if hasattr(schema, "model_json_schema")
    }


__all__ = [
    # Enums
    "SignalType",
    "RiskLevel",
    "Recommendation",
    "Priority",
    "MarketRegime",
    "VolatilityRegime",
    "Urgency",
    "QuantSignalType",
    # Base
    "BaseSchema",
    "PriceLevel",
    "KeyFactor",
    # Trade Analysis
    "TradeAnalysisResult",
    "QuickTradeAnalysis",
    "FinalTradingDecision",
    "SMCZone",
    "SMCAnalysis",
    "PredictionLesson",
    "QuantAnalystDecision",
    # Position Review
    "PositionReview",
    "QuickPositionReview",
    "BatchPositionReview",
    "StopLossAdjustment",
    "TakeProfitAdjustment",
    # Portfolio
    "PortfolioSuggestion",
    "QuickPortfolioSuggestion",
    "PortfolioRiskAssessment",
    "SymbolSuggestion",
    "CorrelationPair",
    # Utilities
    "get_all_schemas",
    "generate_json_schemas",
    # Backward compatibility
    "TradeReviewResult",
]
