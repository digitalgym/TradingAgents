"""
MT5 Worker - Runs on home machine only.
Polls trade_queue and automation_control tables, executes via MT5.
"""
import asyncio
import json
import os
import sys
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

import uuid
import socket
import aiohttp

# Backend API URL for broadcasting status via WebSocket
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

class MT5Worker:
    def __init__(self):
        self.pool = None
        self.running = False
        self.poll_interval = 3  # seconds
        # Track running automation instances
        self._automation_instances: dict = {}
        self._automation_tasks: dict = {}
        # Unique worker ID for this instance
        self.worker_id = f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:6]}"

    async def start(self):
        """Initialize connections and start polling."""
        print(f"[MT5 Worker] Starting worker ID: {self.worker_id}")

        # Connect to Postgres
        self.pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=5,
            statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
        )
        print(f"[MT5 Worker] Connected to Postgres")

        # Initialize MT5
        if not mt5.initialize():
            print(f"[MT5 Worker] Failed to initialize MT5: {mt5.last_error()}")
            return
        print(f"[MT5 Worker] MT5 initialized: {mt5.terminal_info().name}")

        self.running = True

        # Recover stale instances (pending_start/running with no active worker)
        await self._recover_stale_instances()

        # Run all loops concurrently
        await asyncio.gather(
            self._trade_queue_loop(),
            self._control_loop(),
            self._status_update_loop(),
            self._reconciliation_loop(),
        )

    async def stop(self):
        self.running = False

        # Capture instance names before stopping (stop removes them from dict)
        instance_names = list(self._automation_instances.keys())

        # Stop all running automations
        for name in instance_names:
            try:
                await self._stop_automation(name)
            except Exception as e:
                print(f"[MT5 Worker] Error stopping {name}: {e}")

        # Mark instances for auto-recovery on next worker startup.
        # Previously this set status='stopped', which prevented recovery.
        # Now we set 'pending_start' so _recover_stale_instances picks them up.
        if self.pool:
            try:
                async with self.pool.acquire() as conn:
                    for name in instance_names:
                        try:
                            await conn.execute("""
                                UPDATE automation_status
                                SET status = 'pending_start', updated_at = NOW()
                                WHERE instance_name = $1
                            """, name)
                            print(f"[MT5 Worker] Marked {name} as pending_start for recovery")
                        except Exception as e:
                            print(f"[MT5 Worker] Failed to update DB status for {name}: {e}")

                    # Mark mt5_worker itself as stopped (it's the worker, not an automation)
                    await conn.execute("""
                        UPDATE automation_status
                        SET status = 'stopped', updated_at = NOW()
                        WHERE instance_name = 'mt5_worker'
                    """)
            except Exception as e:
                print(f"[MT5 Worker] Failed to update DB statuses on shutdown: {e}")

        if self.pool:
            await self.pool.close()
        mt5.shutdown()
        print("[MT5 Worker] Stopped")

    async def _broadcast_status(self, instance_name: str, status: str, **extra):
        """Notify the backend API to broadcast status change via WebSocket."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"instance_name": instance_name, "status": status, **extra}
                async with session.post(
                    f"{BACKEND_URL}/api/automation/quant/broadcast-status",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        print(f"[MT5 Worker] Broadcast failed: {resp.status}")
        except Exception as e:
            print(f"[MT5 Worker] Broadcast error: {e}")

    async def _recover_stale_instances(self):
        """On startup, find instances stuck in pending_start/running and re-start them."""
        try:
            async with self.pool.acquire() as conn:
                # Find instances that were pending_start or running (from a previous worker)
                rows = await conn.fetch("""
                    SELECT instance_name, status, config, pipeline
                    FROM automation_status
                    WHERE status IN ('pending_start', 'running')
                      AND instance_name != 'mt5_worker'
                """)

                if not rows:
                    print("[MT5 Worker] No stale instances to recover")
                    return

                print(f"[MT5 Worker] Recovering {len(rows)} stale instance(s)...")

                for row in rows:
                    name = row["instance_name"]
                    raw_config = row["config"]
                    if isinstance(raw_config, str):
                        config = json.loads(raw_config)
                    elif isinstance(raw_config, dict):
                        config = raw_config
                    else:
                        config = {}

                    # Inject pipeline from DB column into config (recovery may lose it)
                    db_pipeline = row.get("pipeline")
                    config_pipeline_before = config.get("pipeline", "MISSING")
                    if db_pipeline:
                        config.setdefault("pipeline", db_pipeline)
                    print(f"[MT5 Worker] Recovery {name}: db_pipeline={db_pipeline}, config_pipeline_before={config_pipeline_before}, config_pipeline_after={config.get('pipeline', 'MISSING')}")

                    if not config:
                        print(f"[MT5 Worker] Skipping {name}: no config saved")
                        await conn.execute("""
                            UPDATE automation_status
                            SET status = 'stopped', updated_at = NOW()
                            WHERE instance_name = $1
                        """, name)
                        continue

                    try:
                        print(f"[MT5 Worker] Recovering {name} (was {row['status']})...")
                        await self._start_automation(name, config)
                    except Exception as e:
                        print(f"[MT5 Worker] Failed to recover {name}: {e}")
                        await conn.execute("""
                            UPDATE automation_status
                            SET status = 'stopped', error_message = $2, updated_at = NOW()
                            WHERE instance_name = $1
                        """, name, f"Recovery failed: {e}")

        except Exception as e:
            print(f"[MT5 Worker] Recovery check failed: {e}")

    async def _trade_queue_loop(self):
        """Poll trade_queue for pending commands."""
        print("[MT5 Worker] Trade queue loop started")

        while self.running:
            try:
                async with self.pool.acquire() as conn:
                    # Claim a pending command
                    # DB schema: command_id (text PK), command_type, symbol, payload (jsonb),
                    #            status, created_at, processed_at, result (jsonb), error, source
                    row = await conn.fetchrow("""
                        UPDATE trade_queue
                        SET status = 'processing', processed_at = NOW()
                        WHERE command_id = (
                            SELECT command_id FROM trade_queue
                            WHERE status = 'pending'
                            ORDER BY created_at
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                    """)

                    if row:
                        await self._execute_trade_command(conn, row)

            except Exception as e:
                print(f"[MT5 Worker] Trade queue error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _execute_trade_command(self, conn, row):
        """Execute a trade command via MT5."""
        cmd_id = row["command_id"]
        command = row["command_type"]
        source = row.get("source", "trade_queue")
        # payload is jsonb — asyncpg returns it as a dict/str depending on driver
        raw_payload = row["payload"]
        if isinstance(raw_payload, str):
            payload = json.loads(raw_payload)
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            payload = {}

        print(f"[MT5 Worker] Executing: {command} - {payload}")

        try:
            result = None
            error = None

            if command == "execute":
                result, error = self._mt5_execute(
                    payload["symbol"],
                    payload["direction"],
                    payload["volume"],
                    payload.get("sl"),
                    payload.get("tp"),
                )
                # Create decision for successful trades
                if result and not error:
                    try:
                        from tradingagents.trade_decisions import store_decision
                        decision_id = store_decision(
                            symbol=payload["symbol"],
                            decision_type="OPEN",
                            action=payload["direction"],
                            rationale=payload.get("rationale", f"Trade queue: {cmd_id}"),
                            source=source,
                            entry_price=result.get("price"),
                            stop_loss=payload.get("sl"),
                            take_profit=payload.get("tp"),
                            volume=result.get("volume"),
                            mt5_ticket=result.get("ticket"),
                            confidence=payload.get("confidence"),
                            pipeline=payload.get("pipeline"),
                        )
                        result["decision_id"] = decision_id
                        print(f"[MT5 Worker] Decision created: {decision_id}")
                    except Exception as e:
                        print(f"[MT5 Worker] Failed to create decision: {e}")

            elif command == "close":
                result, error = self._mt5_close(payload["ticket"])
            elif command == "modify_sl":
                result, error = self._mt5_modify_sl(payload["ticket"], payload.get("new_sl"))
            elif command == "modify_tp":
                result, error = self._mt5_modify_tp(payload["ticket"], payload.get("new_tp"))
            else:
                error = f"Unknown command: {command}"

            if error:
                await conn.execute("""
                    UPDATE trade_queue
                    SET status = 'failed', error = $2, processed_at = NOW()
                    WHERE command_id = $1
                """, cmd_id, str(error))
                print(f"[MT5 Worker] Command {cmd_id} failed: {error}")
            else:
                result_json = json.dumps(result) if result else None
                await conn.execute("""
                    UPDATE trade_queue
                    SET status = 'completed', result = $2::jsonb, processed_at = NOW()
                    WHERE command_id = $1
                """, cmd_id, result_json)
                print(f"[MT5 Worker] Command {cmd_id} completed")

        except Exception as e:
            await conn.execute("""
                UPDATE trade_queue
                SET status = 'failed', error = $2, processed_at = NOW()
                WHERE command_id = $1
            """, cmd_id, str(e))
            print(f"[MT5 Worker] Command {cmd_id} exception: {e}")

    def _mt5_execute(self, symbol, direction, volume, sl=None, tp=None):
        """Execute a trade on MT5."""
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            return None, f"Symbol {symbol} not found"

        if not info.visible:
            mt5.symbol_select(symbol, True)

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None, f"No tick data for {symbol}"

        order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction.upper() == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "TradingAgents Remote",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl:
            request["sl"] = float(sl)
        if tp:
            request["tp"] = float(tp)

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return None, f"Order failed: {result.retcode} - {result.comment}"

        return {"ticket": result.order, "price": result.price, "volume": result.volume}, None

    def _mt5_close(self, ticket):
        """Close a position by ticket."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return None, f"Position {ticket} not found"

        pos = position[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "TradingAgents Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return None, f"Close failed: {result.retcode} - {result.comment}"

        return {"closed": True, "price": result.price}, None

    def _mt5_modify_sl(self, ticket, new_sl):
        """Modify stop loss."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return None, f"Position {ticket} not found"

        pos = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": float(new_sl) if new_sl else 0,
            "tp": pos.tp,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return None, f"Modify SL failed: {result.retcode} - {result.comment}"

        return {"modified": True, "new_sl": new_sl}, None

    def _mt5_modify_tp(self, ticket, new_tp):
        """Modify take profit."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return None, f"Position {ticket} not found"

        pos = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": pos.sl,
            "tp": float(new_tp) if new_tp else 0,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return None, f"Modify TP failed: {result.retcode} - {result.comment}"

        return {"modified": True, "new_tp": new_tp}, None

    async def _control_loop(self):
        """Poll automation_control for pending commands."""
        print("[MT5 Worker] Control loop started")

        while self.running:
            try:
                async with self.pool.acquire() as conn:
                    # DB schema: command_id (text PK), instance_name, action,
                    #            payload (jsonb), status, created_at, applied_at, error, source
                    rows = await conn.fetch("""
                        UPDATE automation_control
                        SET status = 'processing', applied_at = NOW()
                        WHERE command_id IN (
                            SELECT command_id FROM automation_control
                            WHERE status = 'pending'
                            ORDER BY created_at
                            LIMIT 10
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                    """)

                    for row in rows:
                        await self._execute_control_command(conn, row)

            except Exception as e:
                print(f"[MT5 Worker] Control loop error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _execute_control_command(self, conn, row):
        """Execute automation control command."""
        cmd_id = row["command_id"]
        instance = row["instance_name"]
        action = row["action"]
        raw_payload = row["payload"]
        if isinstance(raw_payload, str):
            config = json.loads(raw_payload)
        elif isinstance(raw_payload, dict):
            config = raw_payload
        else:
            config = {}

        print(f"[MT5 Worker] Control: {action} for {instance}, payload pipeline={config.get('pipeline', 'MISSING')}")

        try:
            if action == "start":
                await self._start_automation(instance, config)
            elif action == "stop":
                await self._stop_automation(instance)
            elif action == "pause":
                await self._pause_automation(instance)
            elif action == "resume":
                await self._resume_automation(instance)
            elif action == "update_config":
                await self._update_automation_config(instance, config)
            elif action == "restart":
                await self._stop_automation(instance)
                await self._start_automation(instance, config)
            elif action == "batch_train":
                await self._run_batch_train(instance, config, conn, cmd_id)
                return  # batch_train manages its own completion status
            elif action == "optimize":
                await self._run_pair_optimization(instance, config, conn, cmd_id)
                return
            else:
                print(f"[MT5 Worker] Unknown action: {action}")

            # Mark as completed
            await conn.execute("""
                UPDATE automation_control
                SET status = 'completed', applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id)
            print(f"[MT5 Worker] Control {cmd_id} completed")

        except Exception as e:
            print(f"[MT5 Worker] Control {cmd_id} failed: {e}")
            await conn.execute("""
                UPDATE automation_control
                SET status = 'failed', error = $2, applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id, str(e))

    async def _start_automation(self, instance_name: str, config: dict):
        """Start an automation instance."""
        print(f"[MT5 Worker] _start_automation called for {instance_name}, config pipeline={config.get('pipeline', 'MISSING')}")

        if instance_name in self._automation_instances:
            automation = self._automation_instances[instance_name]
            inst_type = type(automation).__name__
            print(f"[MT5 Worker] Found existing instance type={inst_type}, _running={getattr(automation, '_running', 'N/A')}")
            if automation._running:
                print(f"[MT5 Worker] {instance_name} is already running as {inst_type}, skipping")
                return
            else:
                # Stale stopped instance — remove so we can start fresh
                print(f"[MT5 Worker] Removing stale stopped {inst_type} instance {instance_name}")
                del self._automation_instances[instance_name]

        try:
            # Hardcode: trade_manager instance ALWAYS uses TMA, regardless of config
            if instance_name == "trade_manager":
                old_pipeline = config.get("pipeline", "MISSING")
                config["pipeline"] = "trade_management"
                print(f"[MT5 Worker] Forced pipeline: {old_pipeline} -> trade_management for {instance_name}")

            pipeline_name = config.get("pipeline", "smc_quant_basic")
            print(f"[MT5 Worker] ROUTING DECISION: {instance_name} -> pipeline={pipeline_name}")

            # Trade Management Agent — separate path
            if pipeline_name == "trade_management":
                print(f"[MT5 Worker] Starting Trade Management Agent (TradeManagementAgent class)...")
                from tradingagents.automation.trade_management_agent import (
                    TradeManagementAgent,
                    TradeManagementConfig,
                )
                config["instance_name"] = instance_name
                tma_config = TradeManagementConfig.from_dict(config)
                agent = TradeManagementAgent(tma_config)
                self._automation_instances[instance_name] = agent

                worker_ref = self

                def tma_done_callback(task):
                    try:
                        exc = task.exception()
                        if exc:
                            print(f"[MT5 Worker] ERROR: TMA failed: {exc}")
                            import traceback
                            traceback.print_exception(type(exc), exc, exc.__traceback__)
                            asyncio.ensure_future(worker_ref._broadcast_status(
                                instance_name, "error", error=str(exc)
                            ))
                    except asyncio.CancelledError:
                        pass

                task = asyncio.create_task(agent.start())
                task.add_done_callback(tma_done_callback)
                self._automation_tasks[instance_name] = task
                print(f"[MT5 Worker] Trade Management Agent started (TradeManagementAgent)")
                await self._broadcast_status(instance_name, "running")
                return

            # If we reach here, it's a quant automation — warn if instance is trade_manager
            if instance_name == "trade_manager":
                print(f"[MT5 Worker] WARNING: trade_manager fell through to QuantAutomation! pipeline={pipeline_name}")

            from tradingagents.automation.quant_automation import (
                QuantAutomation,
                QuantAutomationConfig,
            )

            # Ensure instance_name and state_file are set
            config["instance_name"] = instance_name
            config.setdefault("state_file", f"quant_automation_state_{instance_name}.json")

            # Backward compat
            if pipeline_name == "quant":
                config["pipeline"] = "smc_quant_basic"

            print(f"[MT5 Worker] Building QuantAutomationConfig via from_dict()...")
            auto_config = QuantAutomationConfig.from_dict(config)
            print(f"[MT5 Worker] Config built: symbols={auto_config.symbols}, pipeline={auto_config.pipeline}, delegate={auto_config.delegate_position_management}")

            print(f"[MT5 Worker] Creating QuantAutomation instance...")
            automation = QuantAutomation(auto_config)
            self._automation_instances[instance_name] = automation
            print(f"[MT5 Worker] QuantAutomation created")

            # Start in background with error callback
            worker_ref = self

            def task_done_callback(task):
                try:
                    exc = task.exception()
                    if exc:
                        print(f"[MT5 Worker] ERROR: {instance_name} task failed: {exc}")
                        import traceback
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                        # Broadcast error status
                        asyncio.ensure_future(worker_ref._broadcast_status(
                            instance_name, "error", error=str(exc)
                        ))
                    else:
                        # Task completed normally (automation stopped)
                        asyncio.ensure_future(worker_ref._broadcast_status(
                            instance_name, "stopped"
                        ))
                except asyncio.CancelledError:
                    print(f"[MT5 Worker] {instance_name} task was cancelled")
                    asyncio.ensure_future(worker_ref._broadcast_status(
                        instance_name, "stopped"
                    ))
                except asyncio.InvalidStateError:
                    pass  # Task not done yet

            print(f"[MT5 Worker] Creating background task for {instance_name}...")
            task = asyncio.create_task(automation.start())
            task.add_done_callback(task_done_callback)
            self._automation_tasks[instance_name] = task

            # Broadcast that it's now running
            await self._broadcast_status(instance_name, "running")
            print(f"[MT5 Worker] Started automation: {instance_name}")

        except Exception as e:
            print(f"[MT5 Worker] ERROR starting {instance_name}: {e}")
            import traceback
            traceback.print_exc()
            await self._broadcast_status(instance_name, "error", error=str(e))
            raise

    async def _stop_automation(self, instance_name: str):
        """Stop an automation instance."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found in memory, updating DB status to stopped")
            # Instance not in memory (worker restarted, or never started) — still update DB + broadcast
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE automation_status
                        SET status = 'stopped', updated_at = NOW()
                        WHERE instance_name = $1
                    """, instance_name)
            except Exception as e:
                print(f"[MT5 Worker] Failed to update DB for {instance_name}: {e}")
            await self._broadcast_status(instance_name, "stopped")
            return

        automation.stop()

        task = self._automation_tasks.get(instance_name)
        if task:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
            except Exception:
                pass
            self._automation_tasks.pop(instance_name, None)

        self._automation_instances.pop(instance_name, None)
        await self._broadcast_status(instance_name, "stopped")
        print(f"[MT5 Worker] Stopped automation: {instance_name}")

    async def _pause_automation(self, instance_name: str):
        """Pause an automation instance (disable auto-execute)."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found")
            return

        automation.pause()
        await self._broadcast_status(instance_name, "paused")
        print(f"[MT5 Worker] Paused automation: {instance_name}")

    async def _resume_automation(self, instance_name: str):
        """Resume an automation instance (enable auto-execute)."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found")
            return

        automation.resume()
        await self._broadcast_status(instance_name, "running")
        print(f"[MT5 Worker] Resumed automation: {instance_name}")

    async def _update_automation_config(self, instance_name: str, updates: dict):
        """Update config for a running automation instance."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found for config update")
            return

        await automation.update_config(updates)
        print(f"[MT5 Worker] Updated config for: {instance_name}")

    async def _run_batch_train(self, instance_name: str, config: dict, conn, cmd_id: str):
        """Run batch training in the worker (has MT5 data access)."""
        print(f"[MT5 Worker] Starting batch training: {config}")

        await self._broadcast_status(instance_name, "running", message="Batch training starting...")

        try:
            from tradingagents.xgb_quant.batch_trainer import BatchTrainer
            from dataclasses import asdict

            trainer = BatchTrainer()
            worker_ref = self
            loop = asyncio.get_running_loop()

            def progress(current, total, msg):
                # Schedule broadcast on the main event loop from this background thread
                loop.call_soon_threadsafe(
                    loop.create_task,
                    worker_ref._broadcast_status(
                        instance_name, "running",
                        current=current, total=total, message=msg,
                    ),
                )

            trainer.on_progress(progress)

            result = await asyncio.to_thread(
                trainer.run,
                symbols=config.get("symbols") or None,
                timeframes=config.get("timeframes", ["D1", "H4"]),
                strategies=config.get("strategies") or None,
                bars=config.get("bars", 2000),
                skip_fresh_days=config.get("skip_fresh_days", 0),
            )

            result_dict = asdict(result)

            await self._broadcast_status(
                instance_name, "done",
                message=(
                    f"Done: {result.completed} trained, {result.skipped} skipped, "
                    f"{result.failed} failed, {len(result.blacklist)} blacklisted "
                    f"in {result.duration_seconds:.0f}s"
                ),
                result=result_dict,
            )

            await conn.execute("""
                UPDATE automation_control
                SET status = 'completed', applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id)
            print(f"[MT5 Worker] Batch training completed: {result.completed} trained")

        except Exception as e:
            import traceback
            print(f"[MT5 Worker] Batch training failed: {e}")
            traceback.print_exc()
            await self._broadcast_status(instance_name, "error", error=str(e))
            await conn.execute("""
                UPDATE automation_control
                SET status = 'failed', error = $2, applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id, str(e))

    async def _run_pair_optimization(self, instance_name: str, config: dict, conn, cmd_id: str):
        """Run per-pair optimization in the worker."""
        print(f"[MT5 Worker] Starting pair optimization: {config}")
        await self._broadcast_status(instance_name, "running", message="Pair optimization starting...")

        try:
            from tradingagents.xgb_quant.pair_optimizer import PairOptimizer
            from tradingagents.xgb_quant.config import OptimizationConfig
            from dataclasses import asdict

            optimizer = PairOptimizer(
                config=OptimizationConfig(),
                max_hours=config.get("max_hours", 6.0),
            )
            loop = asyncio.get_running_loop()

            def progress(pair_idx, total_pairs, msg):
                loop.call_soon_threadsafe(
                    loop.create_task,
                    self._broadcast_status(
                        instance_name, "running",
                        current=pair_idx, total=total_pairs, message=msg,
                    ),
                )

            optimizer.on_progress(progress)

            result = await asyncio.to_thread(
                optimizer.run,
                symbols=config.get("symbols") or None,
                timeframes=config.get("timeframes", ["D1", "H4"]),
                strategies=config.get("strategies") or None,
                bars=config.get("bars", 2000),
            )

            result_dict = asdict(result)
            await self._broadcast_status(
                instance_name, "done",
                message=(
                    f"Done: {result.tier_b_improved} improved, "
                    f"{result.tier_a_count} already good, "
                    f"{result.tier_c_count} non-viable, "
                    f"{result.overfit_count} overfit "
                    f"in {result.duration_seconds:.0f}s"
                ),
                result=result_dict,
            )

            await conn.execute("""
                UPDATE automation_control
                SET status = 'completed', applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id)
            print(f"[MT5 Worker] Pair optimization completed")

        except Exception as e:
            import traceback
            print(f"[MT5 Worker] Pair optimization failed: {e}")
            traceback.print_exc()
            await self._broadcast_status(instance_name, "error", error=str(e))
            await conn.execute("""
                UPDATE automation_control
                SET status = 'failed', error = $2, applied_at = NOW()
                WHERE command_id = $1
            """, cmd_id, str(e))

    async def _status_update_loop(self):
        """Periodically update automation status in DB."""
        print("[MT5 Worker] Status update loop started")

        while self.running:
            try:
                # Get account info
                account = mt5.account_info()
                if account:
                    positions = mt5.positions_get() or []
                    metadata = {
                        "worker_id": self.worker_id,
                        "balance": account.balance,
                        "equity": account.equity,
                        "margin_free": account.margin_free,
                        "positions": len(positions),
                        "running_automations": list(self._automation_instances.keys()),
                    }
                    async with self.pool.acquire() as conn:
                        # Check for stop request for the worker itself
                        row = await conn.fetchrow("""
                            SELECT status FROM automation_status
                            WHERE instance_name = 'mt5_worker'
                        """)
                        if row and row["status"] == "stop_requested":
                            print("[MT5 Worker] Stop requested, shutting down...")
                            self.running = False
                            break

                        # Update mt5_worker status
                        await conn.execute("""
                            INSERT INTO automation_status (instance_name, status, active_positions, config, updated_at)
                            VALUES ('mt5_worker', 'running', $1, $2::jsonb, NOW())
                            ON CONFLICT (instance_name)
                            DO UPDATE SET status = 'running', active_positions = $1, config = $2::jsonb, updated_at = NOW()
                        """, len(positions), json.dumps(metadata))

                        # Update status for each running automation instance
                        for name, automation in self._automation_instances.items():
                            try:
                                status = automation.get_status()
                                config = automation.config.to_dict() if hasattr(automation.config, 'to_dict') else {}
                                await conn.execute("""
                                    INSERT INTO automation_status
                                        (instance_name, status, pipeline, symbols, auto_execute,
                                         active_positions, config, updated_at)
                                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb, NOW())
                                    ON CONFLICT (instance_name) DO UPDATE SET
                                        status = $2,
                                        pipeline = COALESCE($3, automation_status.pipeline),
                                        symbols = COALESCE($4::jsonb, automation_status.symbols),
                                        auto_execute = $5,
                                        active_positions = $6,
                                        config = COALESCE($7::jsonb, automation_status.config),
                                        updated_at = NOW()
                                """,
                                    name,
                                    "running" if status.get("running") else "stopped",
                                    config.get("pipeline"),
                                    json.dumps(config.get("symbols", [])),
                                    config.get("auto_execute", False),
                                    status.get("active_positions", 0),
                                    json.dumps(config),
                                )
                            except Exception as e:
                                print(f"[MT5 Worker] Failed to update status for {name}: {e}")

            except Exception as e:
                print(f"[MT5 Worker] Status update error: {e}")

            await asyncio.sleep(10)  # Check every 10 seconds


    async def _reconciliation_loop(self):
        """Periodically reconcile active decisions against MT5 closed positions.

        Catches orphaned decisions (still 'active' in DB but position already
        closed in MT5) regardless of whether TMA or quant automation is running.
        """
        print("[MT5 Worker] Reconciliation loop started (every 5 min)")
        await asyncio.sleep(60)  # Initial delay — let other loops start first

        while self.running:
            try:
                from tradingagents.trade_decisions import reconcile_decisions
                reconciled = reconcile_decisions(days_back=14)
                if reconciled:
                    print(f"[MT5 Worker] Reconciled {len(reconciled)} orphaned decision(s):")
                    for d in reconciled:
                        print(f"  {d.get('decision_id')}: {d.get('symbol')} {d.get('action')} -> closed")
            except Exception as e:
                print(f"[MT5 Worker] Reconciliation error: {e}")

            await asyncio.sleep(300)  # Every 5 minutes


async def main():
    worker = MT5Worker()

    try:
        await worker.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[MT5 Worker] Shutting down...")
    finally:
        try:
            await worker.stop()
        except Exception as e:
            print(f"[MT5 Worker] Error during shutdown: {e}")


async def stop_worker():
    """Send stop command to running worker via database."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE automation_status
            SET status = 'stop_requested'
            WHERE instance_name = 'mt5_worker'
        """)
    await pool.close()
    print("[MT5 Worker] Stop signal sent")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MT5 Worker - Trade queue executor")
    parser.add_argument("command", nargs="?", default="start", choices=["start", "stop"],
                        help="start or stop the worker")
    args = parser.parse_args()

    if args.command == "stop":
        asyncio.run(stop_worker())
    else:
        asyncio.run(main())
