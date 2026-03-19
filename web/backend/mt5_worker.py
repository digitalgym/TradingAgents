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

        # For now, just mark as completed
        # The actual automation control would integrate with quant_automation
        await conn.execute("""
            UPDATE automation_control
            SET status = 'completed', applied_at = NOW()
            WHERE command_id = $1
        """, cmd_id)

        print(f"[MT5 Worker] Control {cmd_id} completed")

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
                        # DB schema: instance_name (text PK), status, pipeline, symbols (jsonb),
                        #            auto_execute (bool), last_analysis, last_trade,
                        #            active_positions (int), error_message, config (jsonb), updated_at
                        await conn.execute("""
                            INSERT INTO automation_status (instance_name, status, active_positions, config, updated_at)
                            VALUES ('mt5_worker', 'running', $1, $2::jsonb, NOW())
                            ON CONFLICT (instance_name)
                            DO UPDATE SET status = 'running', active_positions = $1, config = $2::jsonb, updated_at = NOW()
                        """, len(positions), json.dumps(metadata))

            except Exception as e:
                print(f"[MT5 Worker] Status update error: {e}")

            await asyncio.sleep(30)  # Update every 30 seconds


async def main():
    worker = MT5Worker()

    try:
        await worker.start()
    except KeyboardInterrupt:
        print("\n[MT5 Worker] Shutting down...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
