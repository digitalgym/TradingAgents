"""
LLM Client module for TradingAgents.

Provides a unified interface for making LLM API calls, supporting:
- xAI Responses API (recommended for xAI/Grok models)
- OpenAI Chat Completions API (fallback for OpenAI models)
- Structured Outputs with JSON schema validation

The xAI Chat Completions API is being deprecated in favor of the Responses API.
See: https://docs.x.ai/docs/guides/chat
See: https://docs.x.ai/docs/guides/structured-outputs
"""

import os
import json
from typing import Optional, List, Dict, Any, Type, TypeVar, Union
from openai import OpenAI

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

# Import central schemas for structured outputs
from tradingagents.schemas import (
    QuickPositionReview,
    QuickPortfolioSuggestion,
    TradeAnalysisResult,
    PositionReview,
    PortfolioSuggestion,
)

T = TypeVar('T')


def get_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[OpenAI, str, bool]:
    """
    Create an LLM client with appropriate configuration.

    Returns:
        Tuple of (client, model_name, uses_responses_api)
    """
    # Determine API key and base URL
    if api_key is None:
        if os.getenv("XAI_API_KEY"):
            api_key = os.getenv("XAI_API_KEY")
            base_url = base_url or "https://api.x.ai/v1"
        elif os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = base_url or "https://api.openai.com/v1"
        else:
            raise ValueError("No API key found. Set XAI_API_KEY or OPENAI_API_KEY environment variable.")

    # Determine if we should use Responses API (xAI) or Chat Completions (OpenAI)
    uses_responses_api = base_url and "x.ai" in base_url

    # Determine default model
    if uses_responses_api:
        model = "grok-3-mini-fast"
    else:
        model = "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)

    return client, model, uses_responses_api


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1000,
    temperature: float = 0.7,
    use_responses_api: bool = False,
) -> str:
    """
    Make an LLM chat completion call using the appropriate API.

    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        use_responses_api: If True, use xAI Responses API; otherwise use Chat Completions

    Returns:
        The assistant's response content as a string
    """
    if use_responses_api:
        return _call_responses_api(client, model, messages, max_tokens, temperature)
    else:
        return _call_chat_completions(client, model, messages, max_tokens, temperature)


def _call_responses_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call xAI's Responses API.

    The Responses API uses a different format than Chat Completions:
    - Uses 'input' instead of 'messages'
    - Uses 'max_output_tokens' instead of 'max_tokens'
    - Response structure differs
    """
    # Convert messages to Responses API format
    # The Responses API accepts messages in a similar format
    input_messages = []
    for msg in messages:
        input_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = client.responses.create(
        model=model,
        input=input_messages,
        max_output_tokens=max_tokens,
        temperature=temperature,
        store=False,  # Don't store conversations server-side
    )

    # Extract text from response
    # Response format: response.output is a list of output items
    # Each item can be a message with content
    return _extract_response_text(response)


def _extract_response_text(response) -> str:
    """
    Extract text content from a Responses API response.

    The response structure can vary:
    - response.output[N].content[M].text for structured content
    - response.output_text for simple text responses
    """
    # Try direct output_text first
    if hasattr(response, 'output_text') and response.output_text:
        return response.output_text

    # Try to find text in output array
    if hasattr(response, 'output') and response.output:
        for item in response.output:
            # Check for message type with content
            if hasattr(item, 'type') and item.type == 'message':
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'type') and content_item.type == 'output_text':
                            if hasattr(content_item, 'text'):
                                return content_item.text
                        # Also check for direct text attribute
                        if hasattr(content_item, 'text'):
                            return content_item.text
            # Direct text content
            if hasattr(item, 'text'):
                return item.text
            # String content
            if isinstance(item, str):
                return item

    # Fallback: convert response to string
    return str(response)


def _call_chat_completions(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call OpenAI's Chat Completions API.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response and response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        return message.content if message else ""

    return ""


def structured_output(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    response_schema: Union[Type[T], Dict[str, Any]],
    max_tokens: int = 1000,
    temperature: float = 0.7,
    use_responses_api: bool = False,
) -> Union[T, Dict[str, Any]]:
    """
    Make an LLM call with structured output (guaranteed JSON schema compliance).

    The response is guaranteed to match the provided schema, eliminating the need
    for manual JSON parsing and validation.

    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: List of message dicts with 'role' and 'content'
        response_schema: Either a Pydantic model class or a JSON schema dict
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        use_responses_api: If True, use xAI Responses API; otherwise use Chat Completions

    Returns:
        If response_schema is a Pydantic model: returns an instance of that model
        If response_schema is a dict: returns a parsed dict matching the schema

    Example with Pydantic:
        class TradeAnalysis(BaseModel):
            recommendation: str  # "HOLD", "CLOSE", or "ADJUST"
            suggested_sl: Optional[float]
            suggested_tp: Optional[float]
            risk_level: str
            reasoning: str

        result = structured_output(
            client, model, messages,
            response_schema=TradeAnalysis,
            use_responses_api=True
        )
        print(result.recommendation)  # Typed access

    Example with JSON schema dict:
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"type": "string", "enum": ["HOLD", "CLOSE", "ADJUST"]},
                "risk_level": {"type": "string"}
            },
            "required": ["recommendation", "risk_level"]
        }

        result = structured_output(
            client, model, messages,
            response_schema=schema,
            use_responses_api=True
        )
        print(result["recommendation"])  # Dict access
    """
    # Convert Pydantic model to JSON schema if needed
    if PYDANTIC_AVAILABLE and isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        json_schema = response_schema.model_json_schema()
        is_pydantic = True
        pydantic_class = response_schema
    else:
        json_schema = response_schema
        is_pydantic = False
        pydantic_class = None

    if use_responses_api:
        response_text = _call_responses_api_structured(
            client, model, messages, json_schema, max_tokens, temperature
        )
    else:
        response_text = _call_chat_completions_structured(
            client, model, messages, json_schema, max_tokens, temperature
        )

    # Parse the response
    parsed = json.loads(response_text)

    # Convert to Pydantic model if applicable
    if is_pydantic and pydantic_class:
        return pydantic_class.model_validate(parsed)

    return parsed


def _call_responses_api_structured(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call xAI's Responses API with structured output.
    """
    input_messages = []
    for msg in messages:
        input_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = client.responses.create(
        model=model,
        input=input_messages,
        max_output_tokens=max_tokens,
        temperature=temperature,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                    "strict": True,
                }
            }
        },
    )

    return _extract_response_text(response)


def _call_chat_completions_structured(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Call OpenAI's Chat Completions API with structured output.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.get("title", "response"),
                "schema": json_schema,
                "strict": True,
            }
        },
    )

    if response and response.choices and len(response.choices) > 0:
        message = response.choices[0].message
        return message.content if message else "{}"

    return "{}"


__all__ = [
    "get_llm_client",
    "chat_completion",
    "structured_output",
    # Re-exported from tradingagents.schemas for convenience
    "QuickPositionReview",
    "QuickPortfolioSuggestion",
    "TradeAnalysisResult",
    "PositionReview",
    "PortfolioSuggestion",
]
