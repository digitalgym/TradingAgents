"""
Command Queue for Multi-Worker Architecture

Provides a PostgreSQL-backed command queue that supports multiple workers.
Workers atomically claim messages using FOR UPDATE SKIP LOCKED to prevent
duplicate processing.

Command Types:
- start_automation: Start an automation instance
- stop_automation: Stop an automation instance
- execute_trade: Execute a trade signal
- modify_position: Modify SL/TP on a position
- close_position: Close a position
- run_analysis: Run a single analysis cycle

Features:
- Atomic claim with SKIP LOCKED (no duplicate processing)
- Worker heartbeats (detect dead workers)
- Stale message recovery (reassign stuck messages)
- Priority support (urgent commands processed first)
"""

import asyncio
import json
import os
import socket
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from enum import Enum


class CommandType(str, Enum):
    START_AUTOMATION = "start_automation"
    STOP_AUTOMATION = "stop_automation"
    EXECUTE_TRADE = "execute_trade"
    MODIFY_POSITION = "modify_position"
    CLOSE_POSITION = "close_position"
    RUN_ANALYSIS = "run_analysis"
    SYNC_STATE = "sync_state"  # Force state sync from DB


class CommandStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"  # Timed out without completion


class CommandPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3  # Processed before all others


@dataclass
class Command:
    """A command to be processed by a worker."""
    id: str
    command_type: str
    payload: Dict[str, Any]
    status: str = "pending"
    priority: int = 1
    source: str = "api"
    target_worker: Optional[str] = None  # If set, only this worker can claim
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkerInfo:
    """Information about a registered worker."""
    worker_id: str
    hostname: str
    capabilities: List[str]  # Command types this worker can handle
    status: str = "active"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    current_command: Optional[str] = None
    commands_processed: int = 0
    started_at: datetime = field(default_factory=datetime.now)


class CommandQueue:
    """
    PostgreSQL-backed command queue with multi-worker support.

    Uses FOR UPDATE SKIP LOCKED for atomic claiming - multiple workers
    can poll simultaneously without blocking or duplicate processing.
    """

    def __init__(self):
        self._pool = None
        self._initialized = False

    def _get_connection_string(self) -> str:
        url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("POSTGRES_URL not set")
        return url

    async def _get_pool(self):
        if self._pool is None or self._pool._closed:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._get_connection_string(),
                min_size=1,
                max_size=5,
                command_timeout=30,
                statement_cache_size=0,
            )
            self._initialized = False
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Command queue table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS command_queue (
                    id TEXT PRIMARY KEY,
                    command_type TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 1,
                    source TEXT DEFAULT 'api',
                    target_worker TEXT,
                    claimed_by TEXT,
                    claimed_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                );
            """)

            # Indexes for efficient querying
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cmdq_pending
                ON command_queue(priority DESC, created_at ASC)
                WHERE status = 'pending';
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cmdq_claimed
                ON command_queue(claimed_by, claimed_at)
                WHERE status = 'claimed';
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cmdq_type
                ON command_queue(command_type);
            """)

            # Worker registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    capabilities JSONB NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    current_command TEXT,
                    commands_processed INTEGER DEFAULT 0,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workers_status
                ON workers(status, last_heartbeat);
            """)

        self._initialized = True

    # === Command Publishing ===

    async def publish(
        self,
        command_type: str,
        payload: Dict[str, Any],
        priority: int = CommandPriority.NORMAL,
        source: str = "api",
        target_worker: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> str:
        """
        Publish a command to the queue.

        Args:
            command_type: Type of command (from CommandType enum)
            payload: Command-specific data
            priority: Processing priority (higher = sooner)
            source: Who created this command
            target_worker: If set, only this worker can claim it
            expires_in_seconds: Auto-expire if not completed in time

        Returns:
            Command ID
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        command_id = f"{command_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO command_queue
                    (id, command_type, payload, priority, source, target_worker, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                command_id,
                command_type,
                json.dumps(payload),
                priority,
                source,
                target_worker,
                expires_at,
            )

        return command_id

    # === Worker Operations ===

    async def register_worker(
        self,
        worker_id: str,
        capabilities: List[str] = None,
    ) -> WorkerInfo:
        """
        Register a worker with the queue.

        Args:
            worker_id: Unique worker identifier
            capabilities: List of command types this worker can handle
                         (None = all types)
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        hostname = socket.gethostname()
        caps = capabilities or [t.value for t in CommandType]

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workers (worker_id, hostname, capabilities, status, last_heartbeat, started_at)
                VALUES ($1, $2, $3, 'active', NOW(), NOW())
                ON CONFLICT (worker_id) DO UPDATE SET
                    hostname = EXCLUDED.hostname,
                    capabilities = EXCLUDED.capabilities,
                    status = 'active',
                    last_heartbeat = NOW()
                """,
                worker_id,
                hostname,
                json.dumps(caps),
            )

        return WorkerInfo(
            worker_id=worker_id,
            hostname=hostname,
            capabilities=caps,
        )

    async def heartbeat(self, worker_id: str, current_command: Optional[str] = None) -> bool:
        """
        Send worker heartbeat. Should be called every 30 seconds.

        Returns False if worker is not registered or marked inactive.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE workers
                SET last_heartbeat = NOW(), current_command = $2
                WHERE worker_id = $1 AND status = 'active'
                """,
                worker_id,
                current_command,
            )

        return "UPDATE 1" in result

    async def deregister_worker(self, worker_id: str) -> None:
        """Mark worker as inactive and release any claimed commands."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Release claimed commands back to pending
            await conn.execute(
                """
                UPDATE command_queue
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL
                WHERE claimed_by = $1 AND status = 'claimed'
                """,
                worker_id,
            )

            # Mark worker inactive
            await conn.execute(
                """
                UPDATE workers
                SET status = 'inactive', current_command = NULL
                WHERE worker_id = $1
                """,
                worker_id,
            )

    async def claim_next(
        self,
        worker_id: str,
        command_types: List[str] = None,
    ) -> Optional[Command]:
        """
        Atomically claim the next available command.

        Uses FOR UPDATE SKIP LOCKED - multiple workers can call this
        simultaneously without blocking or getting the same command.

        Args:
            worker_id: Worker claiming the command
            command_types: Types to claim (None = any type worker can handle)

        Returns:
            Claimed command or None if queue is empty
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        # Build type filter
        types_filter = ""
        if command_types:
            types_json = json.dumps(command_types)
            types_filter = f"AND command_type = ANY(ARRAY(SELECT jsonb_array_elements_text('{types_json}'::jsonb)))"

        async with pool.acquire() as conn:
            # Atomic claim with SKIP LOCKED
            row = await conn.fetchrow(
                f"""
                UPDATE command_queue
                SET status = 'claimed', claimed_by = $1, claimed_at = NOW()
                WHERE id = (
                    SELECT id FROM command_queue
                    WHERE status = 'pending'
                      AND (target_worker IS NULL OR target_worker = $1)
                      AND (expires_at IS NULL OR expires_at > NOW())
                      {types_filter}
                    ORDER BY priority DESC, created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
                """,
                worker_id,
            )

        if not row:
            return None

        return Command(
            id=row["id"],
            command_type=row["command_type"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            status=row["status"],
            priority=row["priority"],
            source=row["source"],
            target_worker=row["target_worker"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            expires_at=row["expires_at"],
            created_at=row["created_at"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
        )

    async def complete(
        self,
        command_id: str,
        result: Dict[str, Any] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Mark command as completed.

        Args:
            command_id: Command to complete
            result: Result data
            worker_id: If set, only complete if claimed by this worker
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if worker_id:
                res = await conn.execute(
                    """
                    UPDATE command_queue
                    SET status = 'completed', result = $2, completed_at = NOW()
                    WHERE id = $1 AND claimed_by = $3
                    """,
                    command_id,
                    json.dumps(result) if result else None,
                    worker_id,
                )
            else:
                res = await conn.execute(
                    """
                    UPDATE command_queue
                    SET status = 'completed', result = $2, completed_at = NOW()
                    WHERE id = $1
                    """,
                    command_id,
                    json.dumps(result) if result else None,
                )

            # Increment worker's processed count
            if worker_id:
                await conn.execute(
                    """
                    UPDATE workers
                    SET commands_processed = commands_processed + 1,
                        current_command = NULL
                    WHERE worker_id = $1
                    """,
                    worker_id,
                )

        return "UPDATE 1" in res

    async def fail(
        self,
        command_id: str,
        error: str,
        worker_id: Optional[str] = None,
        retry: bool = True,
    ) -> bool:
        """
        Mark command as failed.

        If retry=True and retries remaining, moves back to pending.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Get current retry count
            row = await conn.fetchrow(
                "SELECT retry_count, max_retries FROM command_queue WHERE id = $1",
                command_id,
            )

            if not row:
                return False

            retry_count = row["retry_count"]
            max_retries = row["max_retries"]

            if retry and retry_count < max_retries:
                # Retry - move back to pending
                await conn.execute(
                    """
                    UPDATE command_queue
                    SET status = 'pending',
                        claimed_by = NULL,
                        claimed_at = NULL,
                        retry_count = retry_count + 1,
                        error = $2
                    WHERE id = $1
                    """,
                    command_id,
                    error,
                )
            else:
                # Final failure
                await conn.execute(
                    """
                    UPDATE command_queue
                    SET status = 'failed', error = $2, completed_at = NOW()
                    WHERE id = $1
                    """,
                    command_id,
                    error,
                )

            # Clear worker's current command
            if worker_id:
                await conn.execute(
                    "UPDATE workers SET current_command = NULL WHERE worker_id = $1",
                    worker_id,
                )

        return True

    # === Maintenance Operations ===

    async def recover_stale_commands(self, stale_seconds: int = 300) -> int:
        """
        Recover commands claimed by workers that haven't sent heartbeat.

        Called periodically by a maintenance task. Commands claimed more than
        stale_seconds ago by inactive workers are moved back to pending.

        Returns number of recovered commands.
        """
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Find workers with stale heartbeats
            stale_cutoff = datetime.now() - timedelta(seconds=stale_seconds)

            result = await conn.execute(
                """
                UPDATE command_queue
                SET status = 'pending',
                    claimed_by = NULL,
                    claimed_at = NULL,
                    retry_count = retry_count + 1
                WHERE status = 'claimed'
                  AND claimed_at < $1
                  AND claimed_by IN (
                      SELECT worker_id FROM workers
                      WHERE last_heartbeat < $1
                  )
                """,
                stale_cutoff,
            )

        # Extract count from "UPDATE N"
        try:
            return int(result.split()[1])
        except (IndexError, ValueError):
            return 0

    async def expire_old_commands(self) -> int:
        """Mark expired commands. Returns count of expired commands."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE command_queue
                SET status = 'expired'
                WHERE status = 'pending'
                  AND expires_at IS NOT NULL
                  AND expires_at < NOW()
                """
            )

        try:
            return int(result.split()[1])
        except (IndexError, ValueError):
            return 0

    async def cleanup_old_commands(self, days: int = 7) -> int:
        """Delete completed/failed commands older than N days."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM command_queue
                WHERE status IN ('completed', 'failed', 'expired')
                  AND created_at < NOW() - INTERVAL '%s days'
                """,
                days,
            )

        try:
            return int(result.split()[1])
        except (IndexError, ValueError):
            return 0

    # === Query Operations ===

    async def get_command(self, command_id: str) -> Optional[Command]:
        """Get a command by ID."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM command_queue WHERE id = $1",
                command_id,
            )

        if not row:
            return None

        return self._row_to_command(row)

    async def get_pending_count(self, command_type: str = None) -> int:
        """Get count of pending commands."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if command_type:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM command_queue WHERE status = 'pending' AND command_type = $1",
                    command_type,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM command_queue WHERE status = 'pending'"
                )

        return row[0] if row else 0

    async def list_commands(
        self,
        status: str = None,
        command_type: str = None,
        limit: int = 50,
    ) -> List[Command]:
        """List commands with optional filters."""
        await self._ensure_tables()
        pool = await self._get_pool()

        conditions = []
        params = []
        param_idx = 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if command_type:
            conditions.append(f"command_type = ${param_idx}")
            params.append(command_type)
            param_idx += 1

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT * FROM command_queue
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_command(row) for row in rows]

    async def list_workers(self, active_only: bool = True) -> List[WorkerInfo]:
        """List registered workers."""
        await self._ensure_tables()
        pool = await self._get_pool()

        query = "SELECT * FROM workers"
        if active_only:
            query += " WHERE status = 'active'"
        query += " ORDER BY last_heartbeat DESC"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query)

        return [
            WorkerInfo(
                worker_id=row["worker_id"],
                hostname=row["hostname"],
                capabilities=json.loads(row["capabilities"]) if row["capabilities"] else [],
                status=row["status"],
                last_heartbeat=row["last_heartbeat"],
                current_command=row["current_command"],
                commands_processed=row["commands_processed"],
                started_at=row["started_at"],
            )
            for row in rows
        ]

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'claimed') as claimed,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COUNT(*) FILTER (WHERE status = 'expired') as expired
                FROM command_queue
            """)

            workers_row = await conn.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'active') as active_workers,
                    COUNT(*) FILTER (WHERE status = 'inactive') as inactive_workers
                FROM workers
            """)

        return {
            "pending": row["pending"] or 0,
            "claimed": row["claimed"] or 0,
            "completed": row["completed"] or 0,
            "failed": row["failed"] or 0,
            "expired": row["expired"] or 0,
            "active_workers": workers_row["active_workers"] or 0,
            "inactive_workers": workers_row["inactive_workers"] or 0,
        }

    def _row_to_command(self, row) -> Command:
        return Command(
            id=row["id"],
            command_type=row["command_type"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            status=row["status"],
            priority=row["priority"],
            source=row["source"],
            target_worker=row["target_worker"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            completed_at=row["completed_at"],
            expires_at=row["expires_at"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            created_at=row["created_at"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
        )


# Singleton
_command_queue: Optional[CommandQueue] = None


def get_command_queue() -> CommandQueue:
    global _command_queue
    if _command_queue is None:
        _command_queue = CommandQueue()
    return _command_queue
