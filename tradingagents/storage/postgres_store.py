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
                    CONSTRAINT valid_status CHECK (status IN ('active', 'closed', 'failed', 'cancelled', 'retried', 'order_unfilled'))
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

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_weights (
                    id TEXT PRIMARY KEY DEFAULT 'default',
                    weights JSONB NOT NULL,
                    weight_history JSONB DEFAULT '[]',
                    learning_rate FLOAT DEFAULT 0.1,
                    momentum FLOAT DEFAULT 0.9,
                    last_update TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_history (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    pipeline TEXT NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tuning_symbol_pipeline
                ON tuning_history(symbol, pipeline);
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id TEXT PRIMARY KEY DEFAULT 'default',
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    trade_date TEXT,
                    entry_price FLOAT,
                    stop_loss FLOAT,
                    take_profit FLOAT,
                    recommended_size FLOAT,
                    rationale TEXT,
                    key_factors JSONB DEFAULT '[]',
                    smc_analysis JSONB,
                    pipeline TEXT,
                    source TEXT,
                    executed BOOLEAN DEFAULT FALSE,
                    decision_id TEXT,
                    analysis_duration_seconds FLOAT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed);
            """)

            # --- Trade Management Agent tables ---
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS position_management_policies (
                    ticket BIGINT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    policy JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS management_actions (
                    id SERIAL PRIMARY KEY,
                    ticket BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    old_value DOUBLE PRECISION,
                    new_value DOUBLE PRECISION,
                    reason TEXT,
                    success BOOLEAN NOT NULL,
                    error TEXT,
                    decision_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_ticket ON management_actions(ticket);
                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_created ON management_actions(created_at DESC);
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_risk_alerts_created ON risk_alerts(created_at DESC);
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
        created_at_raw = decision.get("created_at")
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        else:
            created_at = datetime.now()

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
        exit_date_raw = decision.get("exit_date")
        if isinstance(exit_date_raw, str):
            exit_date = datetime.fromisoformat(exit_date_raw.replace("Z", "+00:00"))
        elif isinstance(exit_date_raw, datetime):
            exit_date = exit_date_raw
        else:
            exit_date = None

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

    def list_all(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return _run_async(self._list_all_async(symbol, status, source, limit))

    async def _list_all_async(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        # Build query with filters
        conditions = []
        params = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1
        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1
        if source:
            conditions.append(f"data->>'source' = ${param_idx}")
            params.append(source)
            param_idx += 1

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT data FROM decisions
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [json.loads(row["data"]) for row in rows]

    def list_failed(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return _run_async(self._list_failed_async(symbol, limit))

    async def _list_failed_async(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'failed' AND symbol = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    symbol,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT data FROM decisions
                    WHERE status = 'failed'
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

        return [json.loads(row["data"]) for row in rows]

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        return _run_async(self._get_stats_async(symbol))

    async def _get_stats_async(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if symbol:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'active') as active,
                        COUNT(*) FILTER (WHERE status = 'closed') as closed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
                        COUNT(*) FILTER (WHERE data->>'was_correct' = 'true') as wins,
                        COUNT(*) FILTER (WHERE data->>'was_correct' = 'false') as losses,
                        AVG((data->>'pnl')::float) FILTER (WHERE status = 'closed') as avg_pnl,
                        SUM((data->>'pnl')::float) FILTER (WHERE status = 'closed') as total_pnl
                    FROM decisions
                    WHERE symbol = $1
                    """,
                    symbol,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'active') as active,
                        COUNT(*) FILTER (WHERE status = 'closed') as closed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
                        COUNT(*) FILTER (WHERE data->>'was_correct' = 'true') as wins,
                        COUNT(*) FILTER (WHERE data->>'was_correct' = 'false') as losses,
                        AVG((data->>'pnl')::float) FILTER (WHERE status = 'closed') as avg_pnl,
                        SUM((data->>'pnl')::float) FILTER (WHERE status = 'closed') as total_pnl
                    FROM decisions
                    """
                )

        return {
            "total": row["total"] or 0,
            "active": row["active"] or 0,
            "closed": row["closed"] or 0,
            "failed": row["failed"] or 0,
            "cancelled": row["cancelled"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "avg_pnl": float(row["avg_pnl"]) if row["avg_pnl"] else 0.0,
            "total_pnl": float(row["total_pnl"]) if row["total_pnl"] else 0.0,
            "win_rate": (
                row["wins"] / (row["wins"] + row["losses"])
                if (row["wins"] or 0) + (row["losses"] or 0) > 0
                else 0.0
            ),
        }

    def link_ticket(self, decision_id: str, mt5_ticket: int) -> None:
        _run_async(self._link_ticket_async(decision_id, mt5_ticket))

    async def _link_ticket_async(self, decision_id: str, mt5_ticket: int) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE decisions
                SET mt5_ticket = $2,
                    data = jsonb_set(data, '{mt5_ticket}', $2::text::jsonb)
                WHERE decision_id = $1
                """,
                decision_id,
                mt5_ticket,
            )

    def mark_reviewed(self, decision_id: str) -> None:
        _run_async(self._mark_reviewed_async(decision_id))

    async def _mark_reviewed_async(self, decision_id: str) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE decisions
                SET data = jsonb_set(data, '{reviewed_at}', to_jsonb(NOW()::text))
                WHERE decision_id = $1
                """,
                decision_id,
            )

    def cancel(self, decision_id: str, reason: str = "") -> None:
        _run_async(self._cancel_async(decision_id, reason))

    async def _cancel_async(self, decision_id: str, reason: str = "") -> None:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE decisions
                SET status = 'cancelled',
                    data = jsonb_set(
                        jsonb_set(data, '{status}', '"cancelled"'),
                        '{cancel_reason}',
                        $2::jsonb
                    )
                WHERE decision_id = $1
                """,
                decision_id,
                json.dumps(reason),
            )

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

            # --- Trade Management Agent tables ---
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS position_management_policies (
                    ticket BIGINT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    policy JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS management_actions (
                    id SERIAL PRIMARY KEY,
                    ticket BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    old_value DOUBLE PRECISION,
                    new_value DOUBLE PRECISION,
                    reason TEXT,
                    success BOOLEAN NOT NULL,
                    error TEXT,
                    decision_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_ticket ON management_actions(ticket);
                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_created ON management_actions(created_at DESC);
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_risk_alerts_created ON risk_alerts(created_at DESC);
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


class PostgresConfigStore:
    """PostgreSQL-backed configuration storage."""

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=3,
                command_timeout=30,
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        self._initialized = True

    def get(self, key: str) -> Optional[Any]:
        return _run_async(self._get_async(key))

    async def _get_async(self, key: str) -> Optional[Any]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM configs WHERE key = $1", key
            )
        if not row:
            return None
        return json.loads(row["value"])

    def set(self, key: str, value: Any) -> None:
        _run_async(self._set_async(key, value))

    async def _set_async(self, key: str, value: Any) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO configs (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
                """,
                key,
                json.dumps(value, default=str),
            )

    def delete(self, key: str) -> None:
        _run_async(self._delete_async(key))

    async def _delete_async(self, key: str) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM configs WHERE key = $1", key)

    def get_all(self, prefix: str = "") -> Dict[str, Any]:
        return _run_async(self._get_all_async(prefix))

    async def _get_all_async(self, prefix: str = "") -> Dict[str, Any]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if prefix:
                rows = await conn.fetch(
                    "SELECT key, value FROM configs WHERE key LIKE $1",
                    f"{prefix}%",
                )
            else:
                rows = await conn.fetch("SELECT key, value FROM configs")
        return {row["key"]: json.loads(row["value"]) for row in rows}


class PostgresAgentWeightsStore:
    """PostgreSQL-backed agent weights storage for online RL."""

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=3,
                command_timeout=30,
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_weights (
                    id TEXT PRIMARY KEY DEFAULT 'default',
                    weights JSONB NOT NULL,
                    weight_history JSONB DEFAULT '[]',
                    learning_rate FLOAT DEFAULT 0.1,
                    momentum FLOAT DEFAULT 0.9,
                    last_update TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        self._initialized = True

    def load(self, weights_id: str = "default") -> Optional[Dict[str, Any]]:
        return _run_async(self._load_async(weights_id))

    async def _load_async(self, weights_id: str = "default") -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT weights, weight_history, learning_rate, momentum, last_update
                FROM agent_weights WHERE id = $1
                """,
                weights_id,
            )
        if not row:
            return None
        return {
            "weights": json.loads(row["weights"]),
            "weight_history": json.loads(row["weight_history"]) if row["weight_history"] else [],
            "learning_rate": row["learning_rate"],
            "momentum": row["momentum"],
            "last_update": row["last_update"].isoformat() if row["last_update"] else None,
        }

    def save(
        self,
        weights: Dict[str, float],
        weight_history: List[Dict] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weights_id: str = "default",
    ) -> None:
        _run_async(
            self._save_async(weights, weight_history, learning_rate, momentum, weights_id)
        )

    async def _save_async(
        self,
        weights: Dict[str, float],
        weight_history: List[Dict] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weights_id: str = "default",
    ) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_weights (id, weights, weight_history, learning_rate, momentum, last_update, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    weights = EXCLUDED.weights,
                    weight_history = EXCLUDED.weight_history,
                    learning_rate = EXCLUDED.learning_rate,
                    momentum = EXCLUDED.momentum,
                    last_update = NOW(),
                    updated_at = NOW()
                """,
                weights_id,
                json.dumps(weights),
                json.dumps(weight_history or []),
                learning_rate,
                momentum,
            )


class PostgresPortfolioStateStore:
    """PostgreSQL-backed portfolio state storage."""

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=3,
                command_timeout=30,
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id TEXT PRIMARY KEY DEFAULT 'default',
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
        self._initialized = True

    def load(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        return _run_async(self._load_async(portfolio_id))

    async def _load_async(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM portfolio_state WHERE id = $1",
                portfolio_id,
            )
        if not row:
            return None
        return json.loads(row["state"])

    def save(self, state: Dict[str, Any], portfolio_id: str = "default") -> None:
        _run_async(self._save_async(state, portfolio_id))

    async def _save_async(self, state: Dict[str, Any], portfolio_id: str = "default") -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_state (id, state, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    state = EXCLUDED.state,
                    updated_at = NOW()
                """,
                portfolio_id,
                json.dumps(state, default=str),
            )


class PostgresTuningHistoryStore:
    """PostgreSQL-backed tuning history storage."""

    def __init__(self):
        self._pool = None
        self._initialized = False

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                _get_connection_string(),
                min_size=1,
                max_size=3,
                command_timeout=30,
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_history (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    pipeline TEXT NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tuning_symbol_pipeline
                ON tuning_history(symbol, pipeline);
            """)
        self._initialized = True

    def add(self, symbol: str, pipeline: str, result: Dict[str, Any]) -> int:
        return _run_async(self._add_async(symbol, pipeline, result))

    async def _add_async(self, symbol: str, pipeline: str, result: Dict[str, Any]) -> int:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tuning_history (symbol, pipeline, result)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                symbol,
                pipeline,
                json.dumps(result, default=str),
            )
        return row["id"]

    def get_history(
        self, symbol: str = None, pipeline: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return _run_async(self._get_history_async(symbol, pipeline, limit))

    async def _get_history_async(
        self, symbol: str = None, pipeline: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if symbol and pipeline:
                rows = await conn.fetch(
                    """
                    SELECT id, symbol, pipeline, result, created_at
                    FROM tuning_history
                    WHERE symbol = $1 AND pipeline = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    symbol,
                    pipeline,
                    limit,
                )
            elif symbol:
                rows = await conn.fetch(
                    """
                    SELECT id, symbol, pipeline, result, created_at
                    FROM tuning_history
                    WHERE symbol = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    symbol,
                    limit,
                )
            elif pipeline:
                rows = await conn.fetch(
                    """
                    SELECT id, symbol, pipeline, result, created_at
                    FROM tuning_history
                    WHERE pipeline = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    pipeline,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, symbol, pipeline, result, created_at
                    FROM tuning_history
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
        return [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "pipeline": row["pipeline"],
                "result": json.loads(row["result"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def get_latest(self, symbol: str, pipeline: str) -> Optional[Dict[str, Any]]:
        return _run_async(self._get_latest_async(symbol, pipeline))

    async def _get_latest_async(
        self, symbol: str, pipeline: str
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, symbol, pipeline, result, created_at
                FROM tuning_history
                WHERE symbol = $1 AND pipeline = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                symbol,
                pipeline,
            )
        if not row:
            return None
        return {
            "id": row["id"],
            "symbol": row["symbol"],
            "pipeline": row["pipeline"],
            "result": json.loads(row["result"]),
            "created_at": row["created_at"].isoformat(),
        }


class PostgresSignalStore:
    """PostgreSQL-backed signal storage for tracking all generated signals."""

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
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    trade_date TEXT,
                    entry_price FLOAT,
                    stop_loss FLOAT,
                    take_profit FLOAT,
                    recommended_size FLOAT,
                    rationale TEXT,
                    key_factors JSONB DEFAULT '[]',
                    smc_analysis JSONB,
                    pipeline TEXT,
                    source TEXT,
                    executed BOOLEAN DEFAULT FALSE,
                    decision_id TEXT,
                    analysis_duration_seconds FLOAT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed);
            """)
        self._initialized = True

    def store(self, signal: Dict[str, Any]) -> str:
        return _run_async(self._store_async(signal))

    async def _store_async(self, signal: Dict[str, Any]) -> str:
        await self._ensure_tables()
        pool = await self._get_pool()

        # Generate signal_id if not provided
        signal_id = signal.get("signal_id")
        if not signal_id:
            from datetime import datetime
            symbol = signal.get("symbol", "UNKNOWN")
            signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO signals (
                    signal_id, symbol, signal, confidence, trade_date,
                    entry_price, stop_loss, take_profit, recommended_size,
                    rationale, key_factors, smc_analysis, pipeline, source,
                    executed, decision_id, analysis_duration_seconds
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (signal_id) DO UPDATE SET
                    executed = EXCLUDED.executed,
                    decision_id = EXCLUDED.decision_id
                """,
                signal_id,
                signal.get("symbol"),
                signal.get("signal"),
                signal.get("confidence", 0.0),
                signal.get("trade_date"),
                signal.get("entry_price"),
                signal.get("stop_loss"),
                signal.get("take_profit"),
                signal.get("recommended_size"),
                signal.get("rationale"),
                json.dumps(signal.get("key_factors", [])),
                json.dumps(signal.get("smc_analysis")) if signal.get("smc_analysis") else None,
                signal.get("pipeline"),
                signal.get("source"),
                signal.get("executed", False),
                signal.get("decision_id"),
                signal.get("analysis_duration_seconds"),
            )

        return signal_id

    def mark_executed(self, signal_id: str, decision_id: str) -> None:
        _run_async(self._mark_executed_async(signal_id, decision_id))

    async def _mark_executed_async(self, signal_id: str, decision_id: str) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE signals
                SET executed = TRUE, decision_id = $2
                WHERE signal_id = $1
                """,
                signal_id,
                decision_id,
            )

    def list_signals(
        self,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
        pipeline: Optional[str] = None,
        signal: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return _run_async(self._list_signals_async(symbol, executed, pipeline, signal, source, limit))

    async def _list_signals_async(
        self,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
        pipeline: Optional[str] = None,
        signal: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        pool = await self._get_pool()

        conditions = []
        params = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1
        if executed is not None:
            conditions.append(f"executed = ${param_idx}")
            params.append(executed)
            param_idx += 1
        if pipeline:
            conditions.append(f"pipeline = ${param_idx}")
            params.append(pipeline)
            param_idx += 1
        if signal:
            conditions.append(f"UPPER(signal) = ${param_idx}")
            params.append(signal.upper())
            param_idx += 1
        if source:
            conditions.append(f"source = ${param_idx}")
            params.append(source)
            param_idx += 1

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT * FROM signals
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "signal_id": row["signal_id"],
                "symbol": row["symbol"],
                "signal": row["signal"],
                "confidence": row["confidence"],
                "trade_date": row["trade_date"],
                "entry_price": row["entry_price"],
                "stop_loss": row["stop_loss"],
                "take_profit": row["take_profit"],
                "recommended_size": row["recommended_size"],
                "rationale": row["rationale"],
                "key_factors": json.loads(row["key_factors"]) if row["key_factors"] else [],
                "smc_analysis": json.loads(row["smc_analysis"]) if row["smc_analysis"] else None,
                "pipeline": row["pipeline"],
                "source": row["source"],
                "executed": row["executed"],
                "decision_id": row["decision_id"],
                "analysis_duration_seconds": row["analysis_duration_seconds"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        return _run_async(self._get_stats_async(symbol))

    async def _get_stats_async(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if symbol:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE signal = 'BUY') as buy_signals,
                        COUNT(*) FILTER (WHERE signal = 'SELL') as sell_signals,
                        COUNT(*) FILTER (WHERE signal = 'HOLD') as hold_signals,
                        COUNT(*) FILTER (WHERE executed = TRUE) as executed,
                        COUNT(*) FILTER (WHERE executed = FALSE) as not_executed,
                        AVG(confidence) as avg_confidence
                    FROM signals
                    WHERE symbol = $1
                    """,
                    symbol,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE signal = 'BUY') as buy_signals,
                        COUNT(*) FILTER (WHERE signal = 'SELL') as sell_signals,
                        COUNT(*) FILTER (WHERE signal = 'HOLD') as hold_signals,
                        COUNT(*) FILTER (WHERE executed = TRUE) as executed,
                        COUNT(*) FILTER (WHERE executed = FALSE) as not_executed,
                        AVG(confidence) as avg_confidence
                    FROM signals
                    """
                )

        return {
            "total": row["total"] or 0,
            "buy_signals": row["buy_signals"] or 0,
            "sell_signals": row["sell_signals"] or 0,
            "hold_signals": row["hold_signals"] or 0,
            "executed": row["executed"] or 0,
            "not_executed": row["not_executed"] or 0,
            "execution_rate": (
                row["executed"] / row["total"] if row["total"] > 0 else 0.0
            ),
            "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0.0,
        }


class PostgresManagementStore:
    """PostgreSQL-backed storage for Trade Management Agent data."""

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
                statement_cache_size=0,
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS position_management_policies (
                    ticket BIGINT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    policy JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS management_actions (
                    id SERIAL PRIMARY KEY,
                    ticket BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    old_value DOUBLE PRECISION,
                    new_value DOUBLE PRECISION,
                    reason TEXT,
                    success BOOLEAN NOT NULL,
                    error TEXT,
                    decision_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_ticket ON management_actions(ticket);
                CREATE INDEX IF NOT EXISTS idx_mgmt_actions_created ON management_actions(created_at DESC);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_risk_alerts_created ON risk_alerts(created_at DESC);
            """)
        self._initialized = True

    # --- Policy methods ---

    def save_management_policy(self, ticket: int, symbol: str, policy_dict: dict) -> None:
        _run_async(self._save_management_policy_async(ticket, symbol, policy_dict))

    async def _save_management_policy_async(self, ticket: int, symbol: str, policy_dict: dict) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO position_management_policies (ticket, symbol, policy, created_at, updated_at)
                VALUES ($1, $2, $3, NOW(), NOW())
                ON CONFLICT (ticket) DO UPDATE SET
                    symbol = EXCLUDED.symbol,
                    policy = EXCLUDED.policy,
                    updated_at = NOW()
                """,
                ticket,
                symbol,
                json.dumps(policy_dict, default=str),
            )

    def load_management_policy(self, ticket: int) -> Optional[dict]:
        return _run_async(self._load_management_policy_async(ticket))

    async def _load_management_policy_async(self, ticket: int) -> Optional[dict]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT policy FROM position_management_policies WHERE ticket = $1",
                ticket,
            )
        if not row:
            return None
        return json.loads(row["policy"])

    def load_all_management_policies(self) -> Dict[int, dict]:
        return _run_async(self._load_all_management_policies_async())

    async def _load_all_management_policies_async(self) -> Dict[int, dict]:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ticket, policy FROM position_management_policies"
            )
        result = {}
        for row in rows:
            policy = json.loads(row["policy"])
            policy["ticket"] = row["ticket"]
            result[row["ticket"]] = policy
        return result

    def delete_management_policy(self, ticket: int) -> None:
        _run_async(self._delete_management_policy_async(ticket))

    async def _delete_management_policy_async(self, ticket: int) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM position_management_policies WHERE ticket = $1",
                ticket,
            )

    # --- Action methods ---

    def save_management_action(self, action_dict: dict) -> None:
        _run_async(self._save_management_action_async(action_dict))

    async def _save_management_action_async(self, action_dict: dict) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO management_actions
                    (ticket, symbol, action_type, old_value, new_value, reason, success, error, decision_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                action_dict.get("ticket"),
                action_dict.get("symbol", ""),
                action_dict.get("action_type", ""),
                action_dict.get("old_value"),
                action_dict.get("new_value"),
                action_dict.get("reason", ""),
                action_dict.get("success", True),
                action_dict.get("error"),
                action_dict.get("decision_id"),
            )

    def get_management_actions(self, ticket: int = None, limit: int = 50) -> list:
        return _run_async(self._get_management_actions_async(ticket, limit))

    async def _get_management_actions_async(self, ticket: int = None, limit: int = 50) -> list:
        await self._ensure_tables()
        pool = await self._get_pool()

        if ticket:
            query = """
                SELECT * FROM management_actions
                WHERE ticket = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            params = [ticket, limit]
        else:
            query = """
                SELECT * FROM management_actions
                ORDER BY created_at DESC
                LIMIT $1
            """
            params = [limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "ticket": row["ticket"],
                "symbol": row["symbol"],
                "action_type": row["action_type"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "reason": row["reason"],
                "success": row["success"],
                "error": row["error"],
                "decision_id": row["decision_id"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    # --- Risk alert methods ---

    def save_risk_alert(self, alert_type: str, severity: str, message: str,
                        details: Optional[dict] = None) -> None:
        _run_async(self._save_risk_alert_async(alert_type, severity, message, details))

    async def _save_risk_alert_async(self, alert_type: str, severity: str,
                                      message: str, details: Optional[dict] = None) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO risk_alerts (alert_type, severity, message, details)
                VALUES ($1, $2, $3, $4)
                """,
                alert_type,
                severity,
                message,
                json.dumps(details, default=str) if details else None,
            )

    def get_risk_alerts(self, limit: int = 20, unacknowledged_only: bool = False) -> list:
        return _run_async(self._get_risk_alerts_async(limit, unacknowledged_only))

    async def _get_risk_alerts_async(self, limit: int = 20,
                                      unacknowledged_only: bool = False) -> list:
        await self._ensure_tables()
        pool = await self._get_pool()

        if unacknowledged_only:
            query = """
                SELECT * FROM risk_alerts
                WHERE acknowledged = FALSE
                ORDER BY created_at DESC
                LIMIT $1
            """
        else:
            query = """
                SELECT * FROM risk_alerts
                ORDER BY created_at DESC
                LIMIT $1
            """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, limit)

        return [
            {
                "id": row["id"],
                "alert_type": row["alert_type"],
                "severity": row["severity"],
                "message": row["message"],
                "details": json.loads(row["details"]) if row["details"] else None,
                "acknowledged": row["acknowledged"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def acknowledge_risk_alert(self, alert_id: int) -> None:
        _run_async(self._acknowledge_risk_alert_async(alert_id))

    async def _acknowledge_risk_alert_async(self, alert_id: int) -> None:
        await self._ensure_tables()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE risk_alerts SET acknowledged = TRUE WHERE id = $1",
                alert_id,
            )


# Singleton instances
_decision_store: Optional[PostgresDecisionStore] = None
_state_store: Optional[PostgresAutomationStateStore] = None
_config_store: Optional[PostgresConfigStore] = None
_signal_store: Optional[PostgresSignalStore] = None
_weights_store: Optional[PostgresAgentWeightsStore] = None
_portfolio_store: Optional[PostgresPortfolioStateStore] = None
_tuning_store: Optional[PostgresTuningHistoryStore] = None
_management_store: Optional[PostgresManagementStore] = None


def get_decision_store() -> PostgresDecisionStore:
    global _decision_store
    if _decision_store is None:
        _decision_store = PostgresDecisionStore()
    return _decision_store


def get_state_store() -> PostgresAutomationStateStore:
    global _state_store
    if _state_store is None:
        _state_store = PostgresAutomationStateStore()
    return _state_store


def get_config_store() -> PostgresConfigStore:
    global _config_store
    if _config_store is None:
        _config_store = PostgresConfigStore()
    return _config_store


def get_weights_store() -> PostgresAgentWeightsStore:
    global _weights_store
    if _weights_store is None:
        _weights_store = PostgresAgentWeightsStore()
    return _weights_store


def get_portfolio_store() -> PostgresPortfolioStateStore:
    global _portfolio_store
    if _portfolio_store is None:
        _portfolio_store = PostgresPortfolioStateStore()
    return _portfolio_store


def get_tuning_store() -> PostgresTuningHistoryStore:
    global _tuning_store
    if _tuning_store is None:
        _tuning_store = PostgresTuningHistoryStore()
    return _tuning_store


def get_signal_store() -> PostgresSignalStore:
    global _signal_store
    if _signal_store is None:
        _signal_store = PostgresSignalStore()
    return _signal_store


def get_management_store() -> PostgresManagementStore:
    global _management_store
    if _management_store is None:
        _management_store = PostgresManagementStore()
    return _management_store
