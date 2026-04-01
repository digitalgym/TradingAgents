"""
TMA Worker - Dedicated process for the Trade Management Agent.

Separate from mt5_worker.py (which handles quant automations).
This eliminates the routing bug where TMA was started as a quant.

Usage:
    python tma_worker.py start     # Start TMA worker
    python tma_worker.py stop      # Send stop command via DB
"""
import asyncio
import json
import os
import sys
import uuid
import socket
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import asyncpg
import MetaTrader5 as mt5

DATABASE_URL = os.environ.get("POSTGRES_URL")
INSTANCE_NAME = "trade_manager"


class TMAWorker:
    """Dedicated worker process for the Trade Management Agent."""

    def __init__(self):
        self.pool = None
        self.running = False
        self.agent = None
        self.agent_task = None
        self.worker_id = f"tma_{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:6]}"

    async def start(self):
        """Initialize and start the TMA worker."""
        print(f"[TMA Worker] Starting worker ID: {self.worker_id}")

        # Connect to Postgres
        self.pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=3,
            statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
        )
        print(f"[TMA Worker] Connected to Postgres")

        # Initialize MT5
        if not mt5.initialize():
            print(f"[TMA Worker] Failed to initialize MT5: {mt5.last_error()}")
            return
        print(f"[TMA Worker] MT5 initialized: {mt5.terminal_info().name}")

        self.running = True

        # Register worker in DB
        await self._update_worker_status("running")

        # Auto-recover TMA if it was running before shutdown
        await self._recover()

        # Run control + status loops
        await asyncio.gather(
            self._control_loop(),
            self._status_update_loop(),
        )

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        # Stop TMA agent if running
        if self.agent:
            try:
                self.agent.stop()
                if self.agent_task and not self.agent_task.done():
                    self.agent_task.cancel()
                    try:
                        await self.agent_task
                    except (asyncio.CancelledError, Exception):
                        pass
            except Exception as e:
                print(f"[TMA Worker] Error stopping TMA: {e}")

        # Mark TMA as pending_start for recovery on next startup
        if self.pool:
            try:
                async with self.pool.acquire() as conn:
                    if self.agent:
                        await conn.execute("""
                            UPDATE automation_status
                            SET status = 'pending_start', updated_at = NOW()
                            WHERE instance_name = $1
                        """, INSTANCE_NAME)
                        print(f"[TMA Worker] Marked {INSTANCE_NAME} as pending_start for recovery")

                    await conn.execute("""
                        UPDATE automation_status
                        SET status = 'stopped', updated_at = NOW()
                        WHERE instance_name = 'tma_worker'
                    """)
            except Exception as e:
                print(f"[TMA Worker] Failed to update DB on shutdown: {e}")

        if self.pool:
            await self.pool.close()
        mt5.shutdown()
        print(f"[TMA Worker] Shutdown complete")

    async def _recover(self):
        """Auto-recover TMA if it was running before last shutdown."""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT status, config, pipeline
                    FROM automation_status
                    WHERE instance_name = $1
                      AND status IN ('pending_start', 'running')
                """, INSTANCE_NAME)

                if row:
                    raw_config = row["config"]
                    if isinstance(raw_config, str):
                        config = json.loads(raw_config)
                    elif isinstance(raw_config, dict):
                        config = raw_config
                    else:
                        config = {}

                    print(f"[TMA Worker] Recovering TMA (was {row['status']})...")
                    await self._start_tma(config)
                else:
                    print(f"[TMA Worker] No TMA to recover")
        except Exception as e:
            print(f"[TMA Worker] Recovery failed: {e}")

    async def _start_tma(self, config: dict):
        """Start the Trade Management Agent."""
        if self.agent and self.agent._running:
            print(f"[TMA Worker] TMA already running, skipping")
            return

        print(f"[TMA Worker] Starting Trade Management Agent...")

        try:
            from tradingagents.automation.trade_management_agent import (
                TradeManagementAgent,
                TradeManagementConfig,
            )

            config["instance_name"] = INSTANCE_NAME
            tma_config = TradeManagementConfig.from_dict(config)
            self.agent = TradeManagementAgent(tma_config)

            worker_ref = self

            def done_callback(task):
                try:
                    exc = task.exception()
                    if exc:
                        print(f"[TMA Worker] ERROR: TMA failed: {exc}")
                        import traceback
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                except asyncio.CancelledError:
                    pass

            self.agent_task = asyncio.create_task(self.agent.start())
            self.agent_task.add_done_callback(done_callback)

            # Update status
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO automation_status
                        (instance_name, status, pipeline, config, updated_at)
                    VALUES ($1, 'running', 'trade_management', $2::jsonb, NOW())
                    ON CONFLICT (instance_name) DO UPDATE SET
                        status = 'running',
                        pipeline = 'trade_management',
                        config = $2::jsonb,
                        updated_at = NOW()
                """, INSTANCE_NAME, json.dumps(config))

            print(f"[TMA Worker] Trade Management Agent started")

        except Exception as e:
            print(f"[TMA Worker] Failed to start TMA: {e}")
            import traceback
            traceback.print_exc()

    async def _stop_tma(self):
        """Stop the Trade Management Agent."""
        if self.agent:
            self.agent.stop()
            if self.agent_task and not self.agent_task.done():
                try:
                    await asyncio.wait_for(self.agent_task, timeout=10)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self.agent_task.cancel()
            self.agent = None
            self.agent_task = None

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE automation_status
                    SET status = 'stopped', updated_at = NOW()
                    WHERE instance_name = $1
                """, INSTANCE_NAME)

            print(f"[TMA Worker] TMA stopped")

    async def _control_loop(self):
        """Poll for start/stop commands for the TMA."""
        print(f"[TMA Worker] Control loop started (polling every 3s)")
        while self.running:
            try:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT command_id, instance_name, action, payload
                        FROM automation_control
                        WHERE instance_name = $1
                          AND status = 'pending'
                        ORDER BY created_at ASC
                        FOR UPDATE SKIP LOCKED
                    """, INSTANCE_NAME)

                    for row in rows:
                        cmd_id = row["command_id"]
                        action = row["action"]
                        raw_payload = row["payload"]

                        if isinstance(raw_payload, str):
                            config = json.loads(raw_payload)
                        elif isinstance(raw_payload, dict):
                            config = raw_payload
                        else:
                            config = {}

                        print(f"[TMA Worker] Command: {action}")

                        try:
                            if action == "start":
                                await self._start_tma(config)
                            elif action == "stop":
                                await self._stop_tma()
                            elif action == "restart":
                                await self._stop_tma()
                                await self._start_tma(config)
                            elif action == "update_config":
                                # Restart with new config
                                await self._stop_tma()
                                await self._start_tma(config)
                            else:
                                print(f"[TMA Worker] Unknown action: {action}")

                            await conn.execute("""
                                UPDATE automation_control
                                SET status = 'completed', applied_at = NOW()
                                WHERE command_id = $1
                            """, cmd_id)

                        except Exception as e:
                            print(f"[TMA Worker] Command {cmd_id} failed: {e}")
                            await conn.execute("""
                                UPDATE automation_control
                                SET status = 'failed', error = $2, applied_at = NOW()
                                WHERE command_id = $1
                            """, cmd_id, str(e))

            except Exception as e:
                print(f"[TMA Worker] Control loop error: {e}")

            await asyncio.sleep(3)

    async def _status_update_loop(self):
        """Periodically update worker status and check for stop requests."""
        print(f"[TMA Worker] Status update loop started")
        while self.running:
            try:
                async with self.pool.acquire() as conn:
                    # Check for stop request on the worker itself
                    row = await conn.fetchrow("""
                        SELECT status FROM automation_status
                        WHERE instance_name = 'tma_worker'
                    """)
                    if row and row["status"] == "stop_requested":
                        print(f"[TMA Worker] Stop requested, shutting down...")
                        self.running = False
                        break

                    # Update worker status
                    tma_running = self.agent._running if self.agent else False
                    metadata = {
                        "worker_id": self.worker_id,
                        "tma_running": tma_running,
                        "tma_status": self.agent._status if self.agent else "not_started",
                    }
                    await conn.execute("""
                        INSERT INTO automation_status
                            (instance_name, status, config, updated_at)
                        VALUES ('tma_worker', 'running', $1::jsonb, NOW())
                        ON CONFLICT (instance_name) DO UPDATE SET
                            status = 'running', config = $1::jsonb, updated_at = NOW()
                    """, json.dumps(metadata))

            except Exception as e:
                print(f"[TMA Worker] Status update error: {e}")

            await asyncio.sleep(10)

    async def _update_worker_status(self, status: str):
        """Update worker status in DB."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO automation_status
                        (instance_name, status, pipeline, updated_at)
                    VALUES ('tma_worker', $1, 'trade_management', NOW())
                    ON CONFLICT (instance_name) DO UPDATE SET
                        status = $1, pipeline = 'trade_management', updated_at = NOW()
                """, status)
        except Exception as e:
            print(f"[TMA Worker] Failed to update worker status: {e}")


async def main():
    worker = TMAWorker()
    try:
        await worker.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[TMA Worker] Shutting down...")
    finally:
        try:
            await worker.stop()
        except Exception as e:
            print(f"[TMA Worker] Error during shutdown: {e}")


async def stop_worker():
    """Send stop command to running TMA worker via database."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE automation_status
            SET status = 'stop_requested', updated_at = NOW()
            WHERE instance_name = 'tma_worker'
        """)
    await pool.close()
    print("[TMA Worker] Stop request sent")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        asyncio.run(stop_worker())
    else:
        asyncio.run(main())
