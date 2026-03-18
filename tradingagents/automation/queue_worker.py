"""
Trade Queue Worker

Runs on the home machine with MT5 connection.
Polls the Postgres trade_queue for pending commands and executes them.

Usage:
    python -m tradingagents.automation.queue_worker

Or run alongside quant_automation - they can run together.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("QueueWorker")


class TradeQueueWorker:
    """
    Worker that polls Postgres for trade commands and executes via MT5.
    """

    def __init__(
        self,
        poll_interval: float = 5.0,  # Check every 5 seconds
        batch_size: int = 5,
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._running = False
        self._queue = None

    async def _get_queue(self):
        if self._queue is None:
            from tradingagents.storage.trade_queue import get_trade_queue
            self._queue = get_trade_queue()
        return self._queue

    async def start(self):
        """Start the worker loop."""
        logger.info("=" * 50)
        logger.info("Trade Queue Worker Starting")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info("=" * 50)

        # Verify MT5 connection
        if not await self._verify_mt5():
            logger.error("MT5 not connected. Worker cannot start.")
            return

        self._running = True

        while self._running:
            try:
                await self._process_pending_commands()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the worker."""
        logger.info("Stopping queue worker...")
        self._running = False

    async def _verify_mt5(self) -> bool:
        """Verify MT5 is connected."""
        try:
            from tradingagents.dataflows.mt5_data import get_mt5_account_info
            info = get_mt5_account_info()
            if info:
                logger.info(f"MT5 connected: {info.get('name', 'Unknown')} (#{info.get('login', '?')})")
                return True
            return False
        except Exception as e:
            logger.error(f"MT5 verification failed: {e}")
            return False

    async def _process_pending_commands(self):
        """Process all pending commands."""
        queue = await self._get_queue()

        # Get pending commands
        commands = await queue.get_pending_commands(limit=self.batch_size)

        if not commands:
            return

        logger.info(f"Found {len(commands)} pending commands")

        for cmd in commands:
            # Try to claim the command (prevents double-execution)
            claimed = await queue.claim_command(cmd.command_id)
            if not claimed:
                logger.warning(f"Could not claim {cmd.command_id} - already processing?")
                continue

            logger.info(f"Processing: {cmd.command_type} {cmd.symbol} [{cmd.command_id}]")

            try:
                result = await self._execute_command(cmd)
                await queue.complete_command(cmd.command_id, result)
                logger.info(f"Completed: {cmd.command_id} -> {result.get('status', 'ok')}")

            except Exception as e:
                error_msg = str(e)
                await queue.fail_command(cmd.command_id, error_msg)
                logger.error(f"Failed: {cmd.command_id} -> {error_msg}")

    async def _execute_command(self, cmd) -> dict:
        """Execute a single command via MT5."""
        from tradingagents.dataflows.mt5_data import (
            execute_trade_signal,
            modify_position,
            close_position,
            get_mt5_symbol_info,
        )

        if cmd.command_type == "execute":
            return await self._execute_trade(cmd)

        elif cmd.command_type == "modify_sl":
            ticket = cmd.payload.get("ticket")
            new_sl = cmd.payload.get("new_sl")
            result = modify_position(ticket, sl=new_sl)
            return {
                "status": "success" if result.get("success") else "error",
                "ticket": ticket,
                "new_sl": new_sl,
                "mt5_result": result,
            }

        elif cmd.command_type == "modify_tp":
            ticket = cmd.payload.get("ticket")
            new_tp = cmd.payload.get("new_tp")
            result = modify_position(ticket, tp=new_tp)
            return {
                "status": "success" if result.get("success") else "error",
                "ticket": ticket,
                "new_tp": new_tp,
                "mt5_result": result,
            }

        elif cmd.command_type == "close":
            ticket = cmd.payload.get("ticket")
            volume = cmd.payload.get("volume")  # None = close all
            result = close_position(ticket, volume=volume)
            return {
                "status": "success" if result.get("success") else "error",
                "ticket": ticket,
                "mt5_result": result,
            }

        else:
            raise ValueError(f"Unknown command type: {cmd.command_type}")

    async def _execute_trade(self, cmd) -> dict:
        """Execute a trade command."""
        from tradingagents.dataflows.mt5_data import (
            execute_trade_signal,
            get_mt5_current_price,
        )
        from tradingagents.trade_decisions import store_decision, link_decision_to_ticket

        payload = cmd.payload
        symbol = cmd.symbol
        direction = payload.get("direction", "").upper()
        volume = payload.get("volume", 0.01)
        entry_price = payload.get("entry_price")
        stop_loss = payload.get("stop_loss")
        take_profit = payload.get("take_profit")
        order_type = payload.get("order_type", "market")
        decision_id = payload.get("decision_id")

        # Get current price if market order
        if order_type == "market" or entry_price is None:
            price_info = get_mt5_current_price(symbol)
            if not price_info:
                raise ValueError(f"Could not get price for {symbol}")
            entry_price = price_info["ask"] if direction == "BUY" else price_info["bid"]

        # Execute via MT5
        result = execute_trade_signal(
            symbol=symbol,
            signal=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=volume,
        )

        if not result.get("success"):
            raise ValueError(result.get("error", "Trade execution failed"))

        ticket = result.get("ticket")

        # Store decision if not already provided
        if not decision_id:
            decision_id = store_decision(
                symbol=symbol,
                decision_type="OPEN",
                action=direction,
                rationale=f"Remote execution via queue [{cmd.command_id}]",
                source=cmd.source,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                mt5_ticket=ticket,
            )
        else:
            # Link existing decision to ticket
            link_decision_to_ticket(decision_id, ticket)

        return {
            "status": "success",
            "ticket": ticket,
            "decision_id": decision_id,
            "entry_price": entry_price,
            "mt5_result": result,
        }


async def main():
    """Run the queue worker."""
    worker = TradeQueueWorker(
        poll_interval=5.0,  # Check every 5 seconds
    )

    try:
        await worker.start()
    except KeyboardInterrupt:
        worker.stop()
        logger.info("Worker stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
