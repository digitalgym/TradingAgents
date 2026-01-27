import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings - use xAI/Grok by default (OpenAI as fallback)
    "llm_provider": "xai",
    "deep_think_llm": "grok-4-1-fast-reasoning",
    "quick_think_llm": "grok-4-1-fast-non-reasoning",
    "backend_url": "https://api.x.ai/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Memory settings
    "use_memory": True,
    # Embedding provider: "auto", "local", "fastembed", "openai", "ollama"
    # "auto" = fastembed for xAI/grok (lightweight), OpenAI for others
    # "local" = sentence-transformers (requires PyTorch, but more accurate)
    # "fastembed" = lightweight local embeddings (no PyTorch needed)
    "embedding_provider": "local",  # Use sentence-transformers (works in .venv)
    "local_embedding_model": "all-MiniLM-L6-v2",  # Model for local embeddings
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local, mt5
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",        # Options: openai, alpha_vantage, google, local
    },
    # Asset type: stock, commodity, forex (affects which analysts are used)
    "asset_type": "auto",  # Options: auto, stock, commodity, forex
    # Commodity symbol mappings for convenience
    "commodity_symbols": {
        "gold": "XAUUSD",
        "silver": "XAGUSD",
        "platinum": "XPTUSD",
        "copper": "COPPER-C",
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
}
