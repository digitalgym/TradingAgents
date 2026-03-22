"""
PostgreSQL storage implementation for Vercel Postgres / Neon.

Requires:
    pip install asyncpg

Environment variables:
    POSTGRES_URL: Connection string (postgresql://user:pass@host:5432/db?sslmode=require)
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base import DecisionStore, AutomationStateStore


def _get_connection_string() -> str:
    """Get Postgres connection string from environment."""
    url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "POSTGRES_URL environment variable not set. "
            "Get this from your Vercel Postgres dashboard."
        )
    return url


# Global event loop for sync wrappers
_loop = None
_loop_thread = None


def _get_or_create_loop():
    """Get or create a persistent event loop for sync operations."""
    global _loop, _loop_thread
    import threading

    if _loop is not None and _loop.is_running():
        return _loop

    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()

    return _loop


def _run_async(coro):
    """Run async code from sync context using a persistent loop."""
    import concurrent.futures

    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # We're in async context - just await directly won't work from sync
        # Fall through to use our background loop
    except RuntimeError:
        pass  # No running loop, that's fine

    # Use persistent background loop
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)  # 2 minute timeout


class PostgresDecisionStore(DecisionStore):
    """
    PostgreSQL-backed decision storage.

    Schema:
        decisions: Main decision data as JSONB
        decision_contexts: Large context blobs stored separately
        decision_events: Event log for audit trail

    Uses connection pooling for efficiency in serverless environments.
    """

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=10,
                command_timeout=60,
                statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
            )
        return self._pool

    async def _ensure_tables(self):
        """Create tables if they don't exist."""
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    mt5_ticket BIGINT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    exit_date TIMESTAMPTZ,
                    data JSONB NOT NULL,

                    -- Indexes for common queries
                    CONSTRAINT valid_status CHECK (status IN ('active', 'closed', 'failed', 'cancelled', 'retried'))
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
                CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
                CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(mt5_ticket) WHERE mt5_ticket IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_decisions_exit ON decisions(exit_date DESC) WHERE exit_date IS NOT NULL;
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_contexts (
                    decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id) ON DELETE CASCADE,
                    context BYTEA NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_events (
                    id SERIAL PRIMARY KEY,
                    decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    source TEXT,
                    details JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_events_decision ON decision_events(decision_id);
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_state (
                    instance_name TEXT PRIMARY KEY,
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_guardrails (
                    instance_name TEXT PRIMARY KEY,
                    guardrails JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

        self._initialized = True

    def _sync_ensure_tables(self):
        """Sync wrapper for table creation."""
        _run_async(self._ensure_tables())

    # --- Sync wrappers for DecisionStore interface ---

    def store(self, decision: Dict[str, Any]) -> str:
        return _run_async(self._store_async(decision))

    async def _store_async(self, decision: Dict[str, Any]) -> str:
        await self._ensure_tables()
        pool = await self._get_pool()

        decision_id = decision["decision_id"]
        symbol = decision["symbol"]
        status = decision.get("status", "active")
        mt5_ticket = decision.get("mt5_ticket")
        created_at = decision.get("created_at", datetime.now().isoformat())

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decisions (decision_id, symbol, status, mt5_ticket, created_at, data)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (decision_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    mt5_ticket = EXCLUDED.mt5_ticket,
                    data = EXCLUDED.data
                """,
                decision_id,
                symbol,
                status,
                mt5_ticket,
                created_at,
                json.dumps(decision, default=str),
            )

        return decision_id

    def load(self, decision_id: str) -> Dict[str, Any]:
        return _run_async(self._load_async(decision_id))

    async def _load_async(self, decision_id: str) -> Dict[str, Any]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM decisions WHERE decision_id = $1", decision_id
            )

        if not row:
            raise KeyError(f"Decision not found: {decision_id}")

        return json.loads(row["data"])

    def update(self, decision_id: str, updates: Dict[str, Any]) -> None:
        _run_async(self._update_async(decision_id, updates))

    async def _update_async(self, decision_id: str, updates: Dict[str, Any]) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        # Load current, merge updates
        decision = await self._load_async(decision_id)
        decision.update(updates)

        status = decision.get("status", "active")
        mt5_ticket = decision.get("mt5_ticket")
        exit_date = decision.get("exit_date")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE decisions
                SET status = $2, mt5_ticket = $3, exit_date = $4, data = $5
                WHERE decision_id = $1
                """,
                decision_id,
                status,
                mt5_ticket,
                exit_date,
                json.dumps(decision, default=str),
            )

    def list_active(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return _run_async(self._list_active_async(symbol))

    async def _list_active_async(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'active' AND symbol = $1
                    ORDER BY created_at DESC
                    """,
                    symbol,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    """
                )

        return [json.loads(row["data"]) for row in rows]

    def list_closed(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return _run_async(self._list_closed_async(symbol, limit))

    async def _list_closed_async(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'closed' AND symbol = $1
                    ORDER BY exit_date DESC
                    LIMIT $2
                    """,
                    symbol,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'closed'
                    ORDER BY exit_date DESC
                    LIMIT $1
                    """,
                    limit,
                )

        return [json.loads(row["data"]) for row in rows]

    def find_by_ticket(self, mt5_ticket: int) -> Optional[Dict[str, Any]]:
        return _run_async(self._find_by_ticket_async(mt5_ticket))

    async def _find_by_ticket_async(
        self, mt5_ticket: int
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM decisions WHERE mt5_ticket = $1", mt5_ticket
            )

        if not row:
            return None

        return json.loads(row["data"])

    def store_context(self, decision_id: str, context: Dict[str, Any]) -> None:
        _run_async(self._store_context_async(decision_id, context))

    async def _store_context_async(
        self, decision_id: str, context: Dict[str, Any]
    ) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        # Serialize context with pickle for complex objects
        context_bytes = pickle.dumps(context)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decision_contexts (decision_id, context)
                VALUES ($1, $2)
                ON CONFLICT (decision_id) DO UPDATE SET
                    context = EXCLUDED.context
                """,
                decision_id,
                context_bytes,
            )

            # Update has_context flag
            await conn.execute(
                """
                UPDATE decisions
                SET data = jsonb_set(data, '{has_context}', 'true')
                WHERE decision_id = $1
                """,
                decision_id,
            )

    def load_context(self, decision_id: str) -> Optional[Dict[str, Any]]:
        return _run_async(self._load_context_async(decision_id))

    async def _load_context_async(
        self, decision_id: str
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT context FROM decision_contexts WHERE decision_id = $1",
                decision_id,
            )

        if not row:
            return None

        return pickle.loads(row["context"])

    def add_event(
        self,
        decision_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        _run_async(self._add_event_async(decision_id, event_type, details, source))

    async def _add_event_async(
        self,
        decision_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decision_events (decision_id, event_type, source, details)
                VALUES ($1, $2, $3, $4)
                """,
                decision_id,
                event_type,
                source,
                json.dumps(details) if details else None,
            )

            # Also update the events array in the main decision JSON
            # This keeps backward compatibility with code that reads events from decision
            await conn.execute(
                """
                UPDATE decisions
                SET data = jsonb_set(
                    data,
                    '{events}',
                    COALESCE(data->'events', '[]'::jsonb) || $2::jsonb
                )
                WHERE decision_id = $1
                """,
                decision_id,
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "type": event_type,
                        "source": source,
                        **(details or {}),
                    }
                ),
            )


class PostgresAutomationStateStore(AutomationStateStore):
    """PostgreSQL-backed automation state storage."""

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=5,
                command_timeout=30,
                statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
            )
        return self._pool

    async def _ensure_tables(self):
        """Tables are created by PostgresDecisionStore, just mark initialized."""
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_state (
                    instance_name TEXT PRIMARY KEY,
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_guardrails (
                    instance_name TEXT PRIMARY KEY,
                    guardrails JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

        self._initialized = True

    def save_state(self, instance_name: str, state: Dict[str, Any]) -> None:
        _run_async(self._save_state_async(instance_name, state))

    async def _save_state_async(
        self, instance_name: str, state: Dict[str, Any]
    ) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO automation_state (instance_name, state, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (instance_name) DO UPDATE SET
                    state = EXCLUDED.state,
                    updated_at = NOW()
                """,
                instance_name,
                json.dumps(state, default=str),
            )

    def load_state(self, instance_name: str) -> Optional[Dict[str, Any]]:
        return _run_async(self._load_state_async(instance_name))

    async def _load_state_async(
        self, instance_name: str
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM automation_state WHERE instance_name = $1",
                instance_name,
            )

        if not row:
            return None

        return json.loads(row["state"])

    def save_guardrails(self, instance_name: str, guardrails: Dict[str, Any]) -> None:
        _run_async(self._save_guardrails_async(instance_name, guardrails))

    async def _save_guardrails_async(
        self, instance_name: str, guardrails: Dict[str, Any]
    ) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO automation_guardrails (instance_name, guardrails, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (instance_name) DO UPDATE SET
                    guardrails = EXCLUDED.guardrails,
                    updated_at = NOW()
                """,
                instance_name,
                json.dumps(guardrails, default=str),
            )

    def load_guardrails(self, instance_name: str) -> Optional[Dict[str, Any]]:
        return _run_async(self._load_guardrails_async(instance_name))

    async def _load_guardrails_async(
        self, instance_name: str
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT guardrails FROM automation_guardrails WHERE instance_name = $1",
                instance_name,
            )

        if not row:
            return None

        return json.loads(row["guardrails"])
