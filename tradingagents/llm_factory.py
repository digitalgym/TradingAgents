"""
Centralised LLM factory with singleton caching.

All LLM instances should be obtained through get_llm() or get_llm_pair()
so that a single client is reused across the application.
"""

import os
import logging
from typing import Dict, Tuple, Optional

from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Singleton cache keyed by (provider, model, base_url)
_llm_cache: Dict[tuple, object] = {}


def _create_llm(provider: str, model: str, config: dict):
    """Create a new LLM instance for the given provider and model."""
    if provider in ("xai", "grok"):
        from langchain_xai import ChatXAI

        api_key = os.environ.get("XAI_API_KEY") or ""
        if not api_key:
            raise ValueError("XAI_API_KEY not found for xAI provider")
        return ChatXAI(model=model, xai_api_key=api_key)

    elif provider in ("openai", "ollama", "openrouter"):
        from langchain_openai import ChatOpenAI
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY not found for provider {provider}")
        sync_client = OpenAI(api_key=api_key, base_url=config["backend_url"])
        return ChatOpenAI(
            model=model,
            base_url=config["backend_url"],
            api_key=api_key,
            root_client=sync_client,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, base_url=config.get("backend_url", "")
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_llm(config: Optional[dict] = None, tier: str = "deep"):
    """Get or create a singleton LLM instance.

    Args:
        config: Config dict with llm_provider, deep_think_llm, etc.
                Defaults to DEFAULT_CONFIG.
        tier: "deep" or "quick" to select model name.

    Returns:
        Cached LLM instance for this provider+model combo.
    """
    config = config or DEFAULT_CONFIG
    provider = config["llm_provider"].lower()
    model = config["deep_think_llm"] if tier == "deep" else config["quick_think_llm"]
    base_url = config.get("backend_url", "")
    key = (provider, model, base_url)

    if key not in _llm_cache:
        logger.info(f"Creating LLM singleton: provider={provider}, model={model}")
        _llm_cache[key] = _create_llm(provider, model, config)

    return _llm_cache[key]


def get_llm_pair(config: Optional[dict] = None) -> Tuple:
    """Get (deep_thinking_llm, quick_thinking_llm) singleton pair."""
    return get_llm(config, "deep"), get_llm(config, "quick")
