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

class MT5Worker:
    def __init__(self):
        self.pool = None
        self.running = False
        self.poll_interval = 3  # seconds
        # Track running automation instances
        self._automation_instances: dict = {}
        self._automation_tasks: dict = {}

    async def start(self):
        """Initialize connections and start polling."""
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

        # Run both loops concurrently
        await asyncio.gather(
            self._trade_queue_loop(),
            self._control_loop(),
            self._status_update_loop(),
        )

    async def stop(self):
        self.running = False

        # Stop all running automations
        for name in list(self._automation_instances.keys()):
            try:
                await self._stop_automation(name)
            except Exception as e:
                print(f"[MT5 Worker] Error stopping {name}: {e}")

        if self.pool:
            await self.pool.close()
        mt5.shutdown()
        print("[MT5 Worker] Stopped")

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

        print(f"[MT5 Worker] Control: {action} for {instance}")

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
        if instance_name in self._automation_instances:
            automation = self._automation_instances[instance_name]
            if automation._running:
                print(f"[MT5 Worker] {instance_name} is already running")
                return

        from tradingagents.automation.quant_automation import (
            QuantAutomation,
            QuantAutomationConfig,
            PipelineType,
        )

        # Build config
        pipeline_name = config.get("pipeline", "smc_quant_basic")
        if pipeline_name == "quant":
            pipeline_name = "smc_quant_basic"

        instance_state_file = config.get("state_file", f"quant_automation_state_{instance_name}.json")

        auto_config = QuantAutomationConfig(
            instance_name=instance_name,
            pipeline=PipelineType(pipeline_name),
            symbols=config.get("symbols", []),
            timeframe=config.get("timeframe", "H1"),
            analysis_interval_seconds=config.get("analysis_interval_seconds", 180),
            position_check_interval_seconds=config.get("position_check_interval_seconds", 60),
            auto_execute=config.get("auto_execute", False),
            min_confidence=config.get("min_confidence", 0.65),
            max_positions_per_symbol=config.get("max_positions_per_symbol", 1),
            enable_trailing_stop=config.get("enable_trailing_stop", True),
            trailing_stop_atr_multiplier=config.get("trailing_stop_atr_multiplier", 1.5),
            enable_breakeven_stop=config.get("enable_breakeven_stop", True),
            move_to_breakeven_atr_mult=config.get("move_to_breakeven_atr_mult", 1.5),
            enable_reversal_close=config.get("enable_reversal_close", True),
            max_risk_per_trade_pct=config.get("max_risk_per_trade_pct", 1.0),
            default_lot_size=config.get("default_lot_size", 0.01),
            daily_loss_limit_pct=config.get("daily_loss_limit_pct", 3.0),
            max_consecutive_losses=config.get("max_consecutive_losses", 3),
            assumption_review_interval_seconds=config.get("assumption_review_interval_seconds", 3600),
            assumption_review_auto_apply=config.get("assumption_review_auto_apply", False),
            enable_trade_queue=config.get("enable_trade_queue", True),
            trade_queue_poll_seconds=config.get("trade_queue_poll_seconds", 5),
            enable_remote_control=config.get("enable_remote_control", True),
            control_poll_seconds=config.get("control_poll_seconds", 3),
            state_file=instance_state_file,
            logs_dir=config.get("logs_dir", "logs/quant_automation"),
        )

        automation = QuantAutomation(auto_config)
        self._automation_instances[instance_name] = automation

        # Start in background
        task = asyncio.create_task(automation.start())
        self._automation_tasks[instance_name] = task

        print(f"[MT5 Worker] Started automation: {instance_name}")

    async def _stop_automation(self, instance_name: str):
        """Stop an automation instance."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found")
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
        print(f"[MT5 Worker] Stopped automation: {instance_name}")

    async def _pause_automation(self, instance_name: str):
        """Pause an automation instance (disable auto-execute)."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found")
            return

        automation.pause()
        print(f"[MT5 Worker] Paused automation: {instance_name}")

    async def _resume_automation(self, instance_name: str):
        """Resume an automation instance (enable auto-execute)."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found")
            return

        automation.resume()
        print(f"[MT5 Worker] Resumed automation: {instance_name}")

    async def _update_automation_config(self, instance_name: str, updates: dict):
        """Update config for a running automation instance."""
        automation = self._automation_instances.get(instance_name)
        if automation is None:
            print(f"[MT5 Worker] {instance_name} not found for config update")
            return

        await automation.update_config(updates)
        print(f"[MT5 Worker] Updated config for: {instance_name}")

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
                        "balance": account.balance,
                        "equity": account.equity,
                        "margin_free": account.margin_free,
                        "positions": len(positions),
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


async def main():
    worker = MT5Worker()

    try:
        await worker.start()
    except KeyboardInterrupt:
        print("\n[MT5 Worker] Shutting down...")
    finally:
        await worker.stop()


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
