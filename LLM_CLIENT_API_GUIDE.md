# LLM Client API Guide

This document describes the LLM client module used in TradingAgents for making API calls to language models.

## Overview

The `tradingagents/dataflows/llm_client.py` module provides a unified interface for making LLM API calls, supporting:

- **xAI Responses API** (recommended for xAI/Grok models)
- **OpenAI Chat Completions API** (fallback for OpenAI models)
- **Structured Outputs** with guaranteed JSON schema compliance

> **Note:** The xAI Chat Completions API (`/v1/chat/completions`) is being deprecated in favor of the Responses API (`/v1/responses`).

## API Reference

### `get_llm_client()`

Creates an LLM client with appropriate configuration based on available API keys.

```python
from tradingagents.dataflows.llm_client import get_llm_client

client, model, uses_responses = get_llm_client()
```

**Parameters:**
- `api_key` (optional): API key to use. If not provided, checks `XAI_API_KEY` then `OPENAI_API_KEY` environment variables.
- `base_url` (optional): Base URL for the API. Defaults based on provider.

**Returns:**
- `client`: OpenAI client instance
- `model`: Default model name (e.g., `grok-3-mini-fast` or `gpt-4o-mini`)
- `uses_responses`: Boolean indicating whether to use Responses API

### `chat_completion()`

Makes an LLM chat completion call using the appropriate API.

```python
from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

client, model, uses_responses = get_llm_client()

response = chat_completion(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=500,
    temperature=0.7,
    use_responses_api=uses_responses,
)
```

**Parameters:**
- `client`: OpenAI client instance
- `model`: Model name to use
- `messages`: List of message dicts with `role` and `content`
- `max_tokens`: Maximum tokens in response (default: 1000)
- `temperature`: Sampling temperature (default: 0.7)
- `use_responses_api`: If True, uses xAI Responses API; otherwise uses Chat Completions

**Returns:**
- String containing the assistant's response

### `quick_llm_call()`

Convenience function for simple LLM calls with sensible defaults.

```python
from tradingagents.dataflows.llm_client import quick_llm_call

response = quick_llm_call(
    prompt="What is the capital of France?",
    system_prompt="Answer concisely.",
    max_tokens=100,
    temperature=0.3,
)
```

### `structured_output()`

Makes an LLM call with **guaranteed** JSON schema compliance. The response is guaranteed to match your schema, eliminating manual JSON parsing and validation errors.

```python
from tradingagents.dataflows.llm_client import get_llm_client, structured_output

client, model, uses_responses = get_llm_client()

# Using a JSON schema dict
schema = {
    "type": "object",
    "properties": {
        "recommendation": {"type": "string", "enum": ["HOLD", "CLOSE", "ADJUST"]},
        "risk_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "reasoning": {"type": "string"}
    },
    "required": ["recommendation", "risk_level", "reasoning"]
}

result = structured_output(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "You are a trade analyst."},
        {"role": "user", "content": "Analyze this position..."},
    ],
    response_schema=schema,
    use_responses_api=uses_responses,
)

print(result["recommendation"])  # Guaranteed to be "HOLD", "CLOSE", or "ADJUST"
```

**Parameters:**
- `client`: OpenAI client instance
- `model`: Model name to use
- `messages`: List of message dicts with `role` and `content`
- `response_schema`: Either a Pydantic model class or a JSON schema dict
- `max_tokens`: Maximum tokens in response (default: 1000)
- `temperature`: Sampling temperature (default: 0.7)
- `use_responses_api`: If True, uses xAI Responses API; otherwise uses Chat Completions

**Returns:**
- If `response_schema` is a Pydantic model: returns an instance of that model
- If `response_schema` is a dict: returns a parsed dict matching the schema

### Pre-defined Schemas

The module includes pre-defined schemas for common trading operations:

```python
from tradingagents.dataflows.llm_client import TradeReviewSchema, PortfolioSuggestionSchema

# Use for trade review analysis
result = structured_output(
    client, model, messages,
    response_schema=TradeReviewSchema.SCHEMA,
    use_responses_api=uses_responses
)

# Use for portfolio suggestions
result = structured_output(
    client, model, messages,
    response_schema=PortfolioSuggestionSchema.SCHEMA,
    use_responses_api=uses_responses
)
```

## Structured Outputs

Structured Outputs is a powerful feature that **guarantees** the LLM response matches your specified JSON schema. This eliminates:

- Manual JSON parsing errors
- Schema validation failures
- Unexpected response formats
- Need for retry logic on malformed responses

### Supported Data Types

| Type | Description |
|------|-------------|
| `string` | Text values |
| `number` | Integer or float values |
| `boolean` | True/false values |
| `object` | Nested objects with properties |
| `array` | Lists of items |
| `enum` | Restricted set of allowed values |
| `anyOf` | Union types (one of several schemas) |

### Limitations

- `allOf` is not currently supported
- `minLength`/`maxLength` for strings not enforced
- `minItems`/`maxItems` for arrays not enforced

### Example: Trade Analysis with Pydantic

```python
from pydantic import BaseModel
from typing import Optional
from tradingagents.dataflows.llm_client import get_llm_client, structured_output

class TradeAnalysis(BaseModel):
    recommendation: str  # "HOLD", "CLOSE", or "ADJUST"
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    risk_level: str  # "Low", "Medium", "High"
    reasoning: str

client, model, uses_responses = get_llm_client()

# Result is a TradeAnalysis instance with typed attributes
result = structured_output(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "Analyze trades and respond with structured data."},
        {"role": "user", "content": f"Position: XAUUSD BUY @ 2650, Current: 2680, P/L: +1.1%"},
    ],
    response_schema=TradeAnalysis,
    use_responses_api=uses_responses,
)

print(f"Recommendation: {result.recommendation}")
print(f"Risk Level: {result.risk_level}")
print(f"Reasoning: {result.reasoning}")
if result.suggested_sl:
    print(f"Move SL to: {result.suggested_sl}")
```

### Example: Portfolio Suggestions with JSON Schema

```python
from tradingagents.dataflows.llm_client import get_llm_client, structured_output

schema = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["symbol", "reason", "priority"]
            }
        },
        "portfolio_analysis": {"type": "string"}
    },
    "required": ["suggestions", "portfolio_analysis"]
}

client, model, uses_responses = get_llm_client()

result = structured_output(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "You are a portfolio manager."},
        {"role": "user", "content": "Current holdings: XAUUSD, XAGUSD. Suggest diversification."},
    ],
    response_schema=schema,
    use_responses_api=uses_responses,
)

for suggestion in result["suggestions"]:
    print(f"Add {suggestion['symbol']}: {suggestion['reason']} (Priority: {suggestion['priority']})")
```

## xAI Responses API vs Chat Completions

| Aspect | Chat Completions (deprecated) | Responses API (current) |
|--------|-------------------------------|-------------------------|
| Endpoint | `/v1/chat/completions` | `/v1/responses` |
| Input parameter | `messages` | `input` |
| Max tokens parameter | `max_tokens` | `max_output_tokens` |
| Response structure | `response.choices[0].message.content` | `response.output[...].content[...].text` |
| Server-side storage | No | Yes (can disable with `store=False`) |
| Stateful conversations | Manual (send all history) | Automatic (use response ID) |

## Supported Models

### xAI/Grok Models
- `grok-4-1-fast-reasoning` - Best for tool-calling with 2M context, reasoning enabled
- `grok-4-1-fast-non-reasoning` - Fast responses without reasoning, 2M context
- `grok-4-fast-reasoning` - 2M context with reasoning
- `grok-4-fast-non-reasoning` - 2M context without reasoning
- `grok-4-0709` - 256K context, powerful reasoning
- `grok-3-mini-fast` - Compact and efficient
- `grok-code-fast-1` - Optimized for agentic coding, 256K context

### OpenAI Models (fallback)
- `gpt-4o-mini` - Fast and efficient
- `gpt-4o` - Standard model
- `o4-mini`, `o3-mini`, `o3`, `o1` - Reasoning models

## Environment Variables

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | xAI/Grok API key (preferred) |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |

The module automatically selects the appropriate API based on which key is available, with xAI taking precedence.

## Usage Examples

### Basic Usage

```python
from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

# Get client (auto-detects xAI or OpenAI)
client, model, uses_responses = get_llm_client()

# Make a call
response = chat_completion(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "You are a trading expert."},
        {"role": "user", "content": "Analyze XAUUSD current trend."},
    ],
    max_tokens=500,
    temperature=0.5,
    use_responses_api=uses_responses,
)

print(response)
```

### Error Handling

```python
from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

try:
    client, model, uses_responses = get_llm_client()
except ValueError as e:
    print(f"No API key available: {e}")
    # Handle missing API key

try:
    response = chat_completion(
        client=client,
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        use_responses_api=uses_responses,
    )
except Exception as e:
    print(f"API call failed: {e}")
    # Handle API errors
```

### With Specific Provider

```python
from openai import OpenAI
from tradingagents.dataflows.llm_client import chat_completion

# Force xAI
client = OpenAI(
    api_key="your-xai-key",
    base_url="https://api.x.ai/v1"
)

response = chat_completion(
    client=client,
    model="grok-3-mini-fast",
    messages=[{"role": "user", "content": "Hello"}],
    use_responses_api=True,  # Use Responses API for xAI
)

# Force OpenAI
client = OpenAI(api_key="your-openai-key")

response = chat_completion(
    client=client,
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    use_responses_api=False,  # Use Chat Completions for OpenAI
)
```

## Migration Guide

If you have existing code using `client.chat.completions.create()`, migrate as follows:

### Before (deprecated for xAI)

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

response = client.chat.completions.create(
    model="grok-3-mini-fast",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=500,
    temperature=0.7,
)
content = response.choices[0].message.content
```

### After (using Responses API)

```python
from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

client, model, uses_responses = get_llm_client()

content = chat_completion(
    client=client,
    model=model,
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=500,
    temperature=0.7,
    use_responses_api=uses_responses,
)
```

## References

- [xAI Responses API Documentation](https://docs.x.ai/docs/guides/chat)
- [xAI Structured Outputs Guide](https://docs.x.ai/docs/guides/structured-outputs)
- [xAI API Tutorial](https://docs.x.ai/docs/tutorial)
- [xAI Release Notes](https://docs.x.ai/docs/release-notes)
