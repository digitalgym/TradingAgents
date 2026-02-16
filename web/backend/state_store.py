"""
Persistent state store for TradingAgents backend.

Uses SQLite to persist automation and learning cycle state across backend restarts.
This allows the system to:
1. Track if services SHOULD be running (vs just if process is alive)
2. Auto-recover services after backend restart
3. Maintain configuration state
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

# Database file location
DB_PATH = Path(__file__).parent.parent.parent / "data" / "state.db"

# Track if DB has been initialized
_db_initialized = False


def _init_db():
    """Initialize database tables. Called once on first connection."""
    global _db_initialized
    if _db_initialized:
        return

    # Ensure directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create state table for key-value storage
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)

    # Create history table for tracking state changes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            value TEXT,
            changed_at TEXT,
            action TEXT
        )
    """)

    conn.commit()
    conn.close()
    _db_initialized = True


@contextmanager
def get_connection():
    """Get a database connection with proper cleanup."""
    _init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def set_state(key: str, value: Any, record_history: bool = True) -> None:
    """Set a state value."""
    now = datetime.now().isoformat()
    value_json = json.dumps(value)

    with get_connection() as conn:
        cursor = conn.cursor()

        # Upsert the state value
        cursor.execute("""
            INSERT INTO state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
        """, (key, value_json, now, value_json, now))

        # Record history if requested
        if record_history:
            cursor.execute("""
                INSERT INTO state_history (key, value, changed_at, action)
                VALUES (?, ?, ?, 'set')
            """, (key, value_json, now))

        conn.commit()


def get_state(key: str, default: Any = None) -> Any:
    """Get a state value."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            return json.loads(row["value"])
        return default


def delete_state(key: str) -> None:
    """Delete a state value."""
    now = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()

        # Get current value for history
        cursor.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row:
            cursor.execute("""
                INSERT INTO state_history (key, value, changed_at, action)
                VALUES (?, ?, ?, 'delete')
            """, (key, row["value"], now))

        cursor.execute("DELETE FROM state WHERE key = ?", (key,))
        conn.commit()


def get_all_state() -> Dict[str, Any]:
    """Get all state values."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM state")
        rows = cursor.fetchall()

        return {row["key"]: json.loads(row["value"]) for row in rows}


# ----- Automation State Helpers -----

class AutomationState:
    """State management for portfolio automation."""

    KEY_ENABLED = "automation.enabled"
    KEY_CONFIG = "automation.config"
    KEY_LAST_START = "automation.last_start"
    KEY_LAST_STOP = "automation.last_stop"
    KEY_STOP_REASON = "automation.stop_reason"

    @classmethod
    def set_enabled(cls, enabled: bool, config: Optional[Dict] = None) -> None:
        """Set automation enabled state."""
        set_state(cls.KEY_ENABLED, enabled)
        if enabled:
            set_state(cls.KEY_LAST_START, datetime.now().isoformat())
            if config:
                set_state(cls.KEY_CONFIG, config)
        else:
            set_state(cls.KEY_LAST_STOP, datetime.now().isoformat())

    @classmethod
    def set_stop_reason(cls, reason: str) -> None:
        """Record why automation was stopped."""
        set_state(cls.KEY_STOP_REASON, reason)

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if automation should be running."""
        return get_state(cls.KEY_ENABLED, False)

    @classmethod
    def get_config(cls) -> Optional[Dict]:
        """Get saved automation config."""
        return get_state(cls.KEY_CONFIG)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get full automation state."""
        return {
            "enabled": get_state(cls.KEY_ENABLED, False),
            "config": get_state(cls.KEY_CONFIG),
            "last_start": get_state(cls.KEY_LAST_START),
            "last_stop": get_state(cls.KEY_LAST_STOP),
            "stop_reason": get_state(cls.KEY_STOP_REASON),
        }


class LearningCycleState:
    """State management for daily learning cycle."""

    KEY_ENABLED = "learning.enabled"
    KEY_CONFIG = "learning.config"
    KEY_SYMBOLS = "learning.symbols"
    KEY_LAST_START = "learning.last_start"
    KEY_LAST_STOP = "learning.last_stop"
    KEY_STOP_REASON = "learning.stop_reason"
    KEY_RUN_AT = "learning.run_at"

    @classmethod
    def set_enabled(cls, enabled: bool, symbols: Optional[list] = None,
                    run_at: Optional[int] = None) -> None:
        """Set learning cycle enabled state."""
        set_state(cls.KEY_ENABLED, enabled)
        if enabled:
            set_state(cls.KEY_LAST_START, datetime.now().isoformat())
            if symbols:
                set_state(cls.KEY_SYMBOLS, symbols)
            if run_at is not None:
                set_state(cls.KEY_RUN_AT, run_at)
        else:
            set_state(cls.KEY_LAST_STOP, datetime.now().isoformat())

    @classmethod
    def set_stop_reason(cls, reason: str) -> None:
        """Record why learning was stopped."""
        set_state(cls.KEY_STOP_REASON, reason)

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if learning cycle should be running."""
        return get_state(cls.KEY_ENABLED, False)

    @classmethod
    def get_symbols(cls) -> list:
        """Get symbols being tracked."""
        return get_state(cls.KEY_SYMBOLS, [])

    @classmethod
    def set_selected_symbols(cls, symbols: list) -> None:
        """Save selected symbols without starting the cycle.

        This allows persisting the user's symbol selection even before
        the learning cycle is started, so it survives page refreshes.
        """
        set_state(cls.KEY_SYMBOLS, symbols)

    @classmethod
    def get_run_at(cls) -> Optional[int]:
        """Get scheduled run hour."""
        return get_state(cls.KEY_RUN_AT)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get full learning cycle state."""
        return {
            "enabled": get_state(cls.KEY_ENABLED, False),
            "symbols": get_state(cls.KEY_SYMBOLS, []),
            "run_at": get_state(cls.KEY_RUN_AT),
            "last_start": get_state(cls.KEY_LAST_START),
            "last_stop": get_state(cls.KEY_LAST_STOP),
            "stop_reason": get_state(cls.KEY_STOP_REASON),
        }


class AnalysisCache:
    """Cache for storing recent analysis results per symbol."""

    KEY_PREFIX = "analysis_cache"

    @classmethod
    def _get_key(cls, symbol: str, timeframe: str = "H1") -> str:
        """Get the cache key for a symbol/timeframe combination."""
        return f"{cls.KEY_PREFIX}.{symbol}.{timeframe}"

    @classmethod
    def store(cls, symbol: str, timeframe: str, result: Dict[str, Any]) -> None:
        """Store an analysis result."""
        cache_entry = {
            "symbol": symbol,
            "timeframe": timeframe,
            "result": result,
            "cached_at": datetime.now().isoformat(),
        }
        set_state(cls._get_key(symbol, timeframe), cache_entry, record_history=False)

    @classmethod
    def get(cls, symbol: str, timeframe: str = "H1") -> Optional[Dict[str, Any]]:
        """Get cached analysis for a symbol.

        Returns dict with:
            - symbol: str
            - timeframe: str
            - result: the analysis result
            - cached_at: ISO timestamp when it was cached
        Returns None if no cache exists.
        """
        return get_state(cls._get_key(symbol, timeframe))

    @classmethod
    def get_age_hours(cls, symbol: str, timeframe: str = "H1") -> Optional[float]:
        """Get the age of cached analysis in hours."""
        cache = cls.get(symbol, timeframe)
        if not cache:
            return None
        cached_at = datetime.fromisoformat(cache["cached_at"])
        age = datetime.now() - cached_at
        return age.total_seconds() / 3600

    @classmethod
    def is_fresh(cls, symbol: str, timeframe: str = "H1", max_age_hours: float = 4.0) -> bool:
        """Check if cached analysis is fresh enough to use."""
        age = cls.get_age_hours(symbol, timeframe)
        if age is None:
            return False
        return age < max_age_hours

    @classmethod
    def clear(cls, symbol: str, timeframe: str = "H1") -> None:
        """Clear cached analysis for a symbol."""
        delete_state(cls._get_key(symbol, timeframe))

    @classmethod
    def list_cached(cls) -> list:
        """List all cached analyses with their ages."""
        all_state = get_all_state()
        cached = []
        for key, value in all_state.items():
            if key.startswith(cls.KEY_PREFIX + "."):
                age = cls.get_age_hours(value["symbol"], value["timeframe"])
                cached.append({
                    "symbol": value["symbol"],
                    "timeframe": value["timeframe"],
                    "cached_at": value["cached_at"],
                    "age_hours": round(age, 2) if age else None,
                    "signal": value.get("result", {}).get("signal"),
                })
        return cached


class TrailingStopState:
    """Manage trailing stop state for positions.

    Trailing stops are tracked per ticket and include:
    - atr_multiplier: Distance multiplier (e.g., 1.5x ATR)
    - trail_distance: Calculated distance to trail
    - highest_price (for BUY) or lowest_price (for SELL): Best price seen
    - enabled_at: When trailing was enabled
    """

    KEY_PREFIX = "trailing_stop"

    @classmethod
    def _get_key(cls, ticket: int) -> str:
        return f"{cls.KEY_PREFIX}.{ticket}"

    @classmethod
    def enable(cls, ticket: int, symbol: str, direction: str,
               trail_distance: float, atr_multiplier: float,
               current_price: float) -> None:
        """Enable trailing stop for a position."""
        data = {
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,  # BUY or SELL
            "trail_distance": trail_distance,
            "atr_multiplier": atr_multiplier,
            "best_price": current_price,  # Tracks highest (BUY) or lowest (SELL)
            "enabled_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        set_state(cls._get_key(ticket), data, record_history=False)

    @classmethod
    def disable(cls, ticket: int) -> None:
        """Disable trailing stop for a position."""
        delete_state(cls._get_key(ticket))

    @classmethod
    def get(cls, ticket: int) -> Optional[Dict[str, Any]]:
        """Get trailing stop config for a position."""
        return get_state(cls._get_key(ticket))

    @classmethod
    def update_best_price(cls, ticket: int, new_best_price: float) -> None:
        """Update the best price seen for a position."""
        data = cls.get(ticket)
        if data:
            data["best_price"] = new_best_price
            data["last_updated"] = datetime.now().isoformat()
            set_state(cls._get_key(ticket), data, record_history=False)

    @classmethod
    def get_all(cls) -> Dict[int, Dict[str, Any]]:
        """Get all active trailing stops."""
        all_state = get_all_state()
        trailing = {}
        for key, value in all_state.items():
            if key.startswith(cls.KEY_PREFIX + "."):
                trailing[value["ticket"]] = value
        return trailing

    @classmethod
    def list_tickets(cls) -> list:
        """List all tickets with active trailing stops."""
        return list(cls.get_all().keys())


class AgentOutputCache:
    """Cache agent outputs to avoid re-running slow agents.

    Cacheable agents:
    - social_analyst: 4 hours (sentiment doesn't change rapidly)
    - news_analyst: 2 hours (news breaks slowly)
    - fundamentals_analyst: 24 hours (quarterly reports)

    Non-cacheable agents (need live data):
    - market_analyst: needs current prices/technicals
    - trader: needs current broker price for entry/SL/TP
    - risk_manager: validates against live prices

    Derived agents (synthesize other outputs, must always run):
    - bull_researcher, bear_researcher, research_manager
    - risky_analyst, safe_analyst, neutral_analyst
    """

    KEY_PREFIX = "agent_output_cache"

    # Default TTL in hours for each cacheable agent
    AGENT_TTL = {
        "social_analyst": 4.0,
        "news_analyst": 2.0,
        "fundamentals_analyst": 24.0,
        "sentiment_report": 4.0,  # Alternative name
        "news_report": 2.0,  # Alternative name
    }

    @classmethod
    def _get_key(cls, symbol: str, agent: str) -> str:
        """Get cache key for symbol/agent combination."""
        return f"{cls.KEY_PREFIX}.{symbol}.{agent}"

    @classmethod
    def get(cls, symbol: str, agent: str) -> Optional[Dict[str, Any]]:
        """Get cached agent output if valid.

        Returns dict with:
            - output: The cached agent output/report
            - cached_at: ISO timestamp when cached
            - age_hours: How old the cache is
            - expired: Whether the cache has expired
        Returns None if no cache exists.
        """
        cache = get_state(cls._get_key(symbol, agent))
        if not cache:
            return None

        # Calculate age
        cached_at = datetime.fromisoformat(cache["cached_at"])
        age = datetime.now() - cached_at
        age_hours = age.total_seconds() / 3600

        # Check expiration
        ttl = cache.get("ttl_hours", cls.AGENT_TTL.get(agent, 4.0))
        expired = age_hours >= ttl

        return {
            "output": cache["output"],
            "cached_at": cache["cached_at"],
            "age_hours": round(age_hours, 2),
            "expired": expired,
            "ttl_hours": ttl,
        }

    @classmethod
    def store(cls, symbol: str, agent: str, output: str, ttl_hours: float = None) -> None:
        """Store agent output in cache.

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            agent: Agent name (e.g., "social_analyst")
            output: The agent's output/report string
            ttl_hours: Custom TTL, or uses default for agent type
        """
        if ttl_hours is None:
            ttl_hours = cls.AGENT_TTL.get(agent, 4.0)

        cache_entry = {
            "symbol": symbol,
            "agent": agent,
            "output": output,
            "cached_at": datetime.now().isoformat(),
            "ttl_hours": ttl_hours,
        }
        set_state(cls._get_key(symbol, agent), cache_entry, record_history=False)

    @classmethod
    def is_valid(cls, symbol: str, agent: str) -> bool:
        """Check if there's a valid (non-expired) cache for this agent."""
        cache = cls.get(symbol, agent)
        return cache is not None and not cache["expired"]

    @classmethod
    def clear(cls, symbol: str, agent: str = None) -> None:
        """Clear cache for a symbol/agent.

        Args:
            symbol: Trading symbol
            agent: Specific agent to clear, or None to clear all agents for symbol
        """
        if agent:
            delete_state(cls._get_key(symbol, agent))
        else:
            # Clear all agents for this symbol
            all_state = get_all_state()
            prefix = f"{cls.KEY_PREFIX}.{symbol}."
            for key in list(all_state.keys()):
                if key.startswith(prefix):
                    delete_state(key)

    @classmethod
    def clear_all(cls) -> None:
        """Clear all agent output caches."""
        all_state = get_all_state()
        for key in list(all_state.keys()):
            if key.startswith(cls.KEY_PREFIX + "."):
                delete_state(key)

    @classmethod
    def get_cache_status(cls, symbol: str) -> Dict[str, Any]:
        """Get cache status for all agents for a symbol.

        Returns dict mapping agent names to their cache status.
        """
        all_state = get_all_state()
        prefix = f"{cls.KEY_PREFIX}.{symbol}."
        status = {}

        for key, value in all_state.items():
            if key.startswith(prefix):
                agent = value["agent"]
                cache_info = cls.get(symbol, agent)
                if cache_info:
                    status[agent] = {
                        "cached": True,
                        "age_hours": cache_info["age_hours"],
                        "expired": cache_info["expired"],
                        "ttl_hours": cache_info["ttl_hours"],
                    }

        return status

    @classmethod
    def list_all_cached(cls) -> list:
        """List all cached agent outputs."""
        all_state = get_all_state()
        cached = []
        for key, value in all_state.items():
            if key.startswith(cls.KEY_PREFIX + "."):
                cache_info = cls.get(value["symbol"], value["agent"])
                if cache_info:
                    cached.append({
                        "symbol": value["symbol"],
                        "agent": value["agent"],
                        "cached_at": cache_info["cached_at"],
                        "age_hours": cache_info["age_hours"],
                        "expired": cache_info["expired"],
                    })
        return cached
