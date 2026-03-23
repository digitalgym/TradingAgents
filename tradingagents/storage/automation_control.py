"""
Remote Automation Control

Allows web UI to control automation instances running on the home machine:
- Start/stop/pause/resume automation
- Update configuration (symbols, intervals, etc.)
- Get real-time status

The home machine polls for control commands and applies them.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, asdict

ControlAction = Literal["start", "stop", "pause", "resume", "update_config", "restart"]


@dataclass
class AutomationStatus:
    """Current status of an automation instance."""
    instance_name: str
    status: str  # running, stopped, paused, error
    pipeline: str
    symbols: List[str]
    auto_execute: bool
    last_analysis: Optional[str] = None
    last_trade: Optional[str] = None
    active_positions: int = 0
    error_message: Optional[str] = None
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


@dataclass
class ControlCommand:
    """A control command for an automation instance."""
    command_id: str
    instance_name: str  # Which automation to control
    action: ControlAction
    payload: Dict[str, Any]  # Action-specific data (e.g., new config)
    status: str = "pending"  # pending, applied, failed
    created_at: str = ""
    applied_at: Optional[str] = None
    error: Optional[str] = None
    source: str = "web_ui"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.command_id:
            self.command_id = f"{self.instance_name}_{self.action}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


class AutomationControlStore:
    """
    PostgreSQL-backed automation control.

    Stores:
    - automation_status: Current status of each automation instance
    - automation_control: Pending control commands
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
                max_size=3,
                command_timeout=30,
                statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
            )
            self._initialized = False  # Re-check tables on reconnect
        return self._pool

    async def _reset_pool(self):
        """Reset the connection pool after a server-side disconnect."""
        if self._pool and not self._pool._closed:
            try:
                await self._pool.close()
            except Exception:
                pass
        self._pool = None
        self._initialized = False

    async def _ensure_tables(self):
        if self._initialized:
            return

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Status table - each automation reports its status here
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_status (
                    instance_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'stopped',
                    pipeline TEXT,
                    symbols JSONB,
                    auto_execute BOOLEAN DEFAULT FALSE,
                    last_analysis TIMESTAMPTZ,
                    last_trade TIMESTAMPTZ,
                    active_positions INT DEFAULT 0,
                    error_message TEXT,
                    config JSONB,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # Control commands table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS automation_control (
                    command_id TEXT PRIMARY KEY,
                    instance_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload JSONB,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    applied_at TIMESTAMPTZ,
                    error TEXT,
                    source TEXT DEFAULT 'web_ui'
                );

                CREATE INDEX IF NOT EXISTS idx_control_instance
                    ON automation_control(instance_name, status);
                CREATE INDEX IF NOT EXISTS idx_control_pending
                    ON automation_control(instance_name, created_at)
                    WHERE status = 'pending';
            """)

        self._initialized = True

    # === Status methods (written by home machine, read by web UI) ===

    async def update_status(
        self,
        instance_name: str,
        status: str,
        pipeline: str = None,
        symbols: List[str] = None,
        auto_execute: bool = None,
        last_analysis: datetime = None,
        last_trade: datetime = None,
        active_positions: int = None,
        error_message: str = None,
        config: Dict[str, Any] = None,
    ) -> None:
        """Update automation status (called by home machine)."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO automation_status
                    (instance_name, status, pipeline, symbols, auto_execute,
                     last_analysis, last_trade, active_positions, error_message, config, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                ON CONFLICT (instance_name) DO UPDATE SET
                    status = COALESCE($2, automation_status.status),
                    pipeline = COALESCE($3, automation_status.pipeline),
                    symbols = COALESCE($4, automation_status.symbols),
                    auto_execute = COALESCE($5, automation_status.auto_execute),
                    last_analysis = COALESCE($6, automation_status.last_analysis),
                    last_trade = COALESCE($7, automation_status.last_trade),
                    active_positions = COALESCE($8, automation_status.active_positions),
                    error_message = $9,
                    config = COALESCE($10, automation_status.config),
                    updated_at = NOW()
                """,
                instance_name,
                status,
                pipeline,
                json.dumps(symbols) if symbols else None,
                auto_execute,
                last_analysis,
                last_trade,
                active_positions,
                error_message,
                json.dumps(config) if config else None,
            )

    async def get_status(self, instance_name: str) -> Optional[Dict[str, Any]]:
        """Get status of an automation instance."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM automation_status WHERE instance_name = $1",
                instance_name,
            )

        if not row:
            return None

        return {
            "instance_name": row["instance_name"],
            "status": row["status"],
            "pipeline": row["pipeline"],
            "symbols": json.loads(row["symbols"]) if row["symbols"] else [],
            "auto_execute": row["auto_execute"],
            "last_analysis": row["last_analysis"].isoformat() if row["last_analysis"] else None,
            "last_trade": row["last_trade"].isoformat() if row["last_trade"] else None,
            "active_positions": row["active_positions"],
            "error_message": row["error_message"],
            "config": json.loads(row["config"]) if row["config"] else {},
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all automation instances."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM automation_status ORDER BY instance_name"
            )

        statuses = []
        for row in rows:
            statuses.append({
                "instance_name": row["instance_name"],
                "status": row["status"],
                "pipeline": row["pipeline"],
                "symbols": json.loads(row["symbols"]) if row["symbols"] else [],
                "auto_execute": row["auto_execute"],
                "last_analysis": row["last_analysis"].isoformat() if row["last_analysis"] else None,
                "last_trade": row["last_trade"].isoformat() if row["last_trade"] else None,
                "active_positions": row["active_positions"],
                "error_message": row["error_message"],
                "config": json.loads(row["config"]) if row["config"] else {},
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })

        return statuses

    async def delete_status(self, instance_name: str) -> bool:
        """Delete an automation status record."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM automation_status WHERE instance_name = $1",
                instance_name,
            )
            # Also clean up any pending commands for this instance
            await conn.execute(
                "DELETE FROM automation_control WHERE instance_name = $1",
                instance_name,
            )

        return "DELETE 1" in result

    # === Control methods (written by web UI, read by home machine) ===

    async def send_command(
        self,
        instance_name: str,
        action: ControlAction,
        payload: Dict[str, Any] = None,
        source: str = "web_ui",
    ) -> str:
        """Send a control command (called by web UI)."""
        await self._ensure_tables()

        command = ControlCommand(
            command_id="",
            instance_name=instance_name,
            action=action,
            payload=payload or {},
            source=source,
        )

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO automation_control
                    (command_id, instance_name, action, payload, status, created_at, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                command.command_id,
                command.instance_name,
                command.action,
                json.dumps(command.payload),
                command.status,
                datetime.fromisoformat(command.created_at),
                command.source,
            )

        return command.command_id

    async def get_pending_commands(self, instance_name: str) -> List[ControlCommand]:
        """Get pending control commands (called by home machine)."""
        await self._ensure_tables()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM automation_control
                WHERE instance_name = $1 AND status = 'pending'
                ORDER BY created_at ASC
                """,
                instance_name,
            )

        commands = []
        for row in rows:
            commands.append(ControlCommand(
                command_id=row["command_id"],
                instance_name=row["instance_name"],
                action=row["action"],
                payload=json.loads(row["payload"]) if row["payload"] else {},
                status=row["status"],
                created_at=row["created_at"].isoformat() if row["created_at"] else "",
                applied_at=row["applied_at"].isoformat() if row["applied_at"] else None,
                error=row["error"],
                source=row["source"] or "web_ui",
            ))

        return commands

    async def mark_applied(self, command_id: str) -> None:
        """Mark a command as applied."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE automation_control
                SET status = 'applied', applied_at = NOW()
                WHERE command_id = $1
                """,
                command_id,
            )

    async def mark_failed(self, command_id: str, error: str) -> None:
        """Mark a command as failed."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE automation_control
                SET status = 'failed', error = $2, applied_at = NOW()
                WHERE command_id = $1
                """,
                command_id,
                error,
            )

    # === Convenience methods for web UI ===

    async def stop_automation(self, instance_name: str, source: str = "web_ui") -> str:
        """Send stop command."""
        return await self.send_command(instance_name, "stop", source=source)

    async def start_automation(self, instance_name: str, source: str = "web_ui") -> str:
        """Send start command."""
        return await self.send_command(instance_name, "start", source=source)

    async def pause_automation(self, instance_name: str, source: str = "web_ui") -> str:
        """Send pause command (stops new trades but continues monitoring)."""
        return await self.send_command(instance_name, "pause", source=source)

    async def resume_automation(self, instance_name: str, source: str = "web_ui") -> str:
        """Send resume command."""
        return await self.send_command(instance_name, "resume", source=source)

    async def update_config(
        self,
        instance_name: str,
        config_updates: Dict[str, Any],
        source: str = "web_ui",
    ) -> str:
        """Send config update command."""
        return await self.send_command(
            instance_name, "update_config", payload=config_updates, source=source
        )


# Singleton
_control_store = None


def get_automation_control() -> AutomationControlStore:
    global _control_store
    if _control_store is None:
        _control_store = AutomationControlStore()
    return _control_store
