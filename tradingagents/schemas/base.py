"""Base models and common types for TradingAgents schemas."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Trading signal direction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"


class Recommendation(str, Enum):
    """Position management recommendation."""

    HOLD = "HOLD"
    CLOSE = "CLOSE"
    ADJUST = "ADJUST"


class Priority(str, Enum):
    """Priority levels for suggestions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "trending-up"
    TRENDING_DOWN = "trending-down"
    RANGING = "ranging"
    EXPANSION = "expansion"


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class Urgency(str, Enum):
    """Urgency levels for recommendations."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class BaseSchema(BaseModel):
    """Base class for all TradingAgents schemas with common config."""

    model_config = {
        "json_schema_extra": {"additionalProperties": False},
        "use_enum_values": True,
    }

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for LLM structured output."""
        schema = cls.model_json_schema()
        schema["title"] = cls.__name__
        return schema


class PriceLevel(BaseSchema):
    """A price level with optional rationale."""

    price: float = Field(..., description="Price level")
    rationale: Optional[str] = Field(None, description="Why this level was chosen")


class KeyFactor(BaseSchema):
    """A key factor influencing a trading decision."""

    factor: str = Field(..., description="The factor description")
    weight: Optional[float] = Field(
        None, ge=0, le=1, description="Factor weight 0-1"
    )
    direction: Optional[SignalType] = Field(
        None, description="Bias direction this factor suggests"
    )
