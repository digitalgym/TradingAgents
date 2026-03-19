"""
Remote Trade Execution Queue

Allows web UI to queue trade commands that the home MT5 machine executes.

This is OPTIONAL - only needed if you want to trigger trades from a remote UI.
The home machine's quant_automation handles normal automated trading regardless.

Commands:
- execute: Place a new trade
- modify_sl: Modify stop loss
- modify_tp: Modify take profit
- close: Close a position

Usage:
- The existing quant_automation can integrate queue polling
- Or run queue_worker.py separately alongside automation
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, asdict
from enum import Enum

CommandType = Literal["execute", "modify_sl", "modify_tp", "close", "cancel_order"]
CommandStatus = Literal["pending", "processing", "completed", "failed"]


@dataclass
class TradeCommand:
    """A trade command to be executed by the home MT5 machine."""
    command_id: str
    command_type: CommandType
    symbol: str
    payload: Dict[str, Any]  # Command-specific data
    status: CommandStatus = "pending"
    created_at: str = ""
    processed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: str = "web_ui"  # Who created this command

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.command_id:
            self.command_id = f"{self.symbol}_{self.command_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


class TradeQueueStore:
    """
    PostgreSQL-backed trade command queue.

    Web UI writes commands, home MT5 machine reads and executes them.
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
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._get_connection_string(),
                min_size=1,
                max_size=5,
                command_timeout=30,
                statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
            )
        return self._pool

    async def _ensure_tables(self):
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_queue (
                    command_id TEXT PRIMARY KEY,
                    command_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    processed_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT,
                    source TEXT DEFAULT 'web_ui'
                );

                CREATE INDEX IF NOT EXISTS idx_queue_status ON trade_queue(status);
                CREATE INDEX IF NOT EXISTS idx_queue_created ON trade_queue(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_queue_pending ON trade_queue(status, created_at)
                    WHERE status = 'pending';
            """)

        self._initialized = True

    # === Write methods (called by Web UI) ===

    async def queue_execute(
        self,
        symbol: str,
        direction: str,  # BUY or SELL
        volume: float,
        entry_price: Optional[float] = None,  # None = market order
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = "market",  # market, limit, stop
        source: str = "web_ui",
        decision_id: Optional[str] = None,
    ) -> str:
        """Queue a trade execution command."""
        await self._ensure_tables()

        command = TradeCommand(
            command_id="",
            command_type="execute",
            symbol=symbol,
            payload={
                "direction": direction.upper(),
                "volume": volume,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_type": order_type,
                "decision_id": decision_id,
            },
            source=source,
        )

        return await self._insert_command(command)

    async def queue_modify_sl(
        self,
        ticket: int,
        new_sl: float,
        source: str = "web_ui",
    ) -> str:
        """Queue a stop loss modification."""
        await self._ensure_tables()

        command = TradeCommand(
            command_id="",
            command_type="modify_sl",
            symbol="",  # Will be filled by worker
            payload={
                "ticket": ticket,
                "new_sl": new_sl,
            },
            source=source,
        )

        return await self._insert_command(command)

    async def queue_modify_tp(
        self,
        ticket: int,
        new_tp: float,
        source: str = "web_ui",
    ) -> str:
        """Queue a take profit modification."""
        await self._ensure_tables()

        command = TradeCommand(
            command_id="",
            command_type="modify_tp",
            symbol="",
            payload={
                "ticket": ticket,
                "new_tp": new_tp,
            },
            source=source,
        )

        return await self._insert_command(command)

    async def queue_close(
        self,
        ticket: int,
        volume: Optional[float] = None,  # None = close full position
        source: str = "web_ui",
    ) -> str:
        """Queue a position close command."""
        await self._ensure_tables()

        command = TradeCommand(
            command_id="",
            command_type="close",
            symbol="",
            payload={
                "ticket": ticket,
                "volume": volume,
            },
            source=source,
        )

        return await self._insert_command(command)

    async def _insert_command(self, command: TradeCommand) -> str:
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trade_queue
                    (command_id, command_type, symbol, payload, status, created_at, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                command.command_id,
                command.command_type,
                command.symbol,
                json.dumps(command.payload),
                command.status,
                datetime.fromisoformat(command.created_at),
                command.source,
            )

        return command.command_id

    # === Read methods (called by MT5 worker) ===

    async def get_pending_commands(self, limit: int = 10) -> List[TradeCommand]:
        """Get pending commands for execution."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT command_id, command_type, symbol, payload, status,
                       created_at, processed_at, result, error, source
                FROM trade_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT $1
                """,
                limit,
            )

        commands = []
        for row in rows:
            commands.append(TradeCommand(
                command_id=row["command_id"],
                command_type=row["command_type"],
                symbol=row["symbol"],
                payload=json.loads(row["payload"]) if row["payload"] else {},
                status=row["status"],
                created_at=row["created_at"].isoformat() if row["created_at"] else "",
                processed_at=row["processed_at"].isoformat() if row["processed_at"] else None,
                result=json.loads(row["result"]) if row["result"] else None,
                error=row["error"],
                source=row["source"] or "web_ui",
            ))

        return commands

    async def claim_command(self, command_id: str) -> bool:
        """Mark a command as processing (prevents double-execution)."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE trade_queue
                SET status = 'processing', processed_at = NOW()
                WHERE command_id = $1 AND status = 'pending'
                """,
                command_id,
            )

        return "UPDATE 1" in result

    async def complete_command(
        self,
        command_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Mark a command as completed with result."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE trade_queue
                SET status = 'completed', result = $2, processed_at = NOW()
                WHERE command_id = $1
                """,
                command_id,
                json.dumps(result),
            )

    async def fail_command(
        self,
        command_id: str,
        error: str,
    ) -> None:
        """Mark a command as failed with error."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE trade_queue
                SET status = 'failed', error = $2, processed_at = NOW()
                WHERE command_id = $1
                """,
                command_id,
                error,
            )

    # === Status methods (called by Web UI to check status) ===

    async def get_command_status(self, command_id: str) -> Optional[TradeCommand]:
        """Get current status of a command."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trade_queue WHERE command_id = $1",
                command_id,
            )

        if not row:
            return None

        return TradeCommand(
            command_id=row["command_id"],
            command_type=row["command_type"],
            symbol=row["symbol"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            status=row["status"],
            created_at=row["created_at"].isoformat() if row["created_at"] else "",
            processed_at=row["processed_at"].isoformat() if row["processed_at"] else None,
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            source=row["source"] or "web_ui",
        )

    async def get_recent_commands(self, limit: int = 20) -> List[TradeCommand]:
        """Get recent commands (all statuses)."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM trade_queue
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        commands = []
        for row in rows:
            commands.append(TradeCommand(
                command_id=row["command_id"],
                command_type=row["command_type"],
                symbol=row["symbol"],
                payload=json.loads(row["payload"]) if row["payload"] else {},
                status=row["status"],
                created_at=row["created_at"].isoformat() if row["created_at"] else "",
                processed_at=row["processed_at"].isoformat() if row["processed_at"] else None,
                result=json.loads(row["result"]) if row["result"] else None,
                error=row["error"],
                source=row["source"] or "web_ui",
            ))

        return commands


# Singleton instance
_queue_store = None


def get_trade_queue() -> TradeQueueStore:
    global _queue_store
    if _queue_store is None:
        _queue_store = TradeQueueStore()
    return _queue_store
