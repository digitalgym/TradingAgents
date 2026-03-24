"""
MT5 Worker - Processes trading commands via MetaTrader 5

Handles:
- execute_trade: Place new trades
- modify_position: Modify SL/TP
- close_position: Close positions
- start_automation: Start a quant automation instance
- stop_automation: Stop an automation instance
- run_analysis: Run a single analysis cycle

Run this worker on the machine with MT5 installed.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from tradingagents.worker.base import BaseWorker
from tradingagents.storage.command_queue import Command, CommandType


class MT5Worker(BaseWorker):
    """
    Worker that processes MT5 trading commands.

    Runs on the machine with MetaTrader 5 installed.
    Processes commands from the queue and executes them via MT5.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
    ):
        super().__init__(
            worker_id=worker_id,
            poll_interval=poll_interval,
        )

        # Track running automation instances
        self._automations: Dict[str, Any] = {}
        self._automation_tasks: Dict[str, asyncio.Task] = {}

    def get_capabilities(self) -> List[str]:
        """This worker handles all MT5-related commands."""
        return [
            CommandType.EXECUTE_TRADE.value,
            CommandType.MODIFY_POSITION.value,
            CommandType.CLOSE_POSITION.value,
            CommandType.START_AUTOMATION.value,
            CommandType.STOP_AUTOMATION.value,
            CommandType.RUN_ANALYSIS.value,
        ]

    async def on_startup(self) -> None:
        """Initialize MT5 connection on startup."""
        self.logger.info("Initializing MT5 connection...")

        # Check MT5 connection
        try:
            from tradingagents.dataflows.mt5_data import check_mt5_autotrading
            status = check_mt5_autotrading()
            self.logger.info(f"MT5 status: {status}")

            if not status.get("connected"):
                self.logger.warning("MT5 not connected - some commands may fail")
            elif not status.get("autotrading_enabled"):
                self.logger.warning("MT5 autotrading disabled - execute commands will fail")

        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")

        # Restore any running automations from DB
        await self._restore_automations()

    async def on_shutdown(self) -> None:
        """Stop all automations on shutdown."""
        self.logger.info("Stopping all automations...")

        for instance_name in list(self._automations.keys()):
            await self._stop_automation_instance(instance_name)

    async def handle_command(self, command: Command) -> Dict[str, Any]:
        """Process a single command."""
        cmd_type = command.command_type
        payload = command.payload

        self.logger.info(f"Handling {cmd_type}: {payload}")

        if cmd_type == CommandType.EXECUTE_TRADE.value:
            return await self._handle_execute_trade(payload)

        elif cmd_type == CommandType.MODIFY_POSITION.value:
            return await self._handle_modify_position(payload)

        elif cmd_type == CommandType.CLOSE_POSITION.value:
            return await self._handle_close_position(payload)

        elif cmd_type == CommandType.START_AUTOMATION.value:
            return await self._handle_start_automation(payload)

        elif cmd_type == CommandType.STOP_AUTOMATION.value:
            return await self._handle_stop_automation(payload)

        elif cmd_type == CommandType.RUN_ANALYSIS.value:
            return await self._handle_run_analysis(payload)

        else:
            raise ValueError(f"Unknown command type: {cmd_type}")

    # === Trade Execution ===

    async def _handle_execute_trade(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade via MT5."""
        from tradingagents.dataflows.mt5_data import execute_trade_signal

        symbol = payload["symbol"]
        direction = payload["direction"]
        volume = payload["volume"]
        entry_price = payload.get("entry_price")
        stop_loss = payload.get("stop_loss")
        take_profit = payload.get("take_profit")

        result = execute_trade_signal(
            symbol=symbol,
            signal=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=volume,
            comment=f"Worker {self.worker_id}",
        )

        if result.get("success"):
            self.logger.info(
                f"Trade executed: {symbol} {direction} {volume} lots, "
                f"ticket={result.get('order_id')}"
            )
        else:
            self.logger.error(f"Trade failed: {result.get('error')}")

        return result

    async def _handle_modify_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Modify position SL/TP."""
        from tradingagents.dataflows.mt5_data import modify_position

        ticket = payload["ticket"]
        new_sl = payload.get("new_sl")
        new_tp = payload.get("new_tp")

        result = modify_position(
            ticket=ticket,
            new_sl=new_sl,
            new_tp=new_tp,
        )

        if result.get("success"):
            self.logger.info(f"Position #{ticket} modified: sl={new_sl}, tp={new_tp}")
        else:
            self.logger.error(f"Modify failed: {result.get('error')}")

        return result

    async def _handle_close_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Close a position."""
        from tradingagents.dataflows.mt5_data import close_position

        ticket = payload["ticket"]
        volume = payload.get("volume")  # None = close full

        result = close_position(ticket=ticket, volume=volume)

        if result.get("success"):
            self.logger.info(f"Position #{ticket} closed")
        else:
            self.logger.error(f"Close failed: {result.get('error')}")

        return result

    # === Automation Control ===

    async def _handle_start_automation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Start an automation instance."""
        instance_name = payload["instance_name"]
        config = payload.get("config", {})

        if instance_name in self._automations:
            return {
                "success": False,
                "error": f"Automation {instance_name} already running",
            }

        try:
            from tradingagents.automation.quant_automation import (
                QuantAutomation,
                QuantAutomationConfig,
            )

            # Create config
            auto_config = QuantAutomationConfig.from_dict(config)

            # Create automation instance
            automation = QuantAutomation(
                config=auto_config,
                instance_name=instance_name,
            )

            # Start in background task
            task = asyncio.create_task(automation.start())
            self._automations[instance_name] = automation
            self._automation_tasks[instance_name] = task

            # Update status in DB
            await self._update_automation_status(instance_name, "running")

            self.logger.info(f"Started automation: {instance_name}")

            return {
                "success": True,
                "instance_name": instance_name,
                "status": "running",
            }

        except Exception as e:
            self.logger.error(f"Failed to start automation: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _handle_stop_automation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Stop an automation instance."""
        instance_name = payload["instance_name"]

        if instance_name not in self._automations:
            return {
                "success": False,
                "error": f"Automation {instance_name} not running",
            }

        try:
            await self._stop_automation_instance(instance_name)

            return {
                "success": True,
                "instance_name": instance_name,
                "status": "stopped",
            }

        except Exception as e:
            self.logger.error(f"Failed to stop automation: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _stop_automation_instance(self, instance_name: str) -> None:
        """Stop a specific automation instance."""
        if instance_name not in self._automations:
            return

        automation = self._automations[instance_name]
        task = self._automation_tasks.get(instance_name)

        # Stop the automation
        try:
            await automation.stop()
        except Exception as e:
            self.logger.warning(f"Error stopping automation {instance_name}: {e}")

        # Cancel the task
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Remove from tracking
        del self._automations[instance_name]
        self._automation_tasks.pop(instance_name, None)

        # Update status in DB
        await self._update_automation_status(instance_name, "stopped")

        self.logger.info(f"Stopped automation: {instance_name}")

    async def _handle_run_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single analysis cycle."""
        instance_name = payload.get("instance_name")
        symbol = payload.get("symbol")

        if instance_name and instance_name in self._automations:
            automation = self._automations[instance_name]
            result = await automation.run_single_analysis(symbol)

            return {
                "success": True,
                "signal": result.signal,
                "confidence": result.confidence,
                "rationale": result.rationale[:500],
                "executed": result.executed,
            }

        else:
            return {
                "success": False,
                "error": f"Automation {instance_name} not running",
            }

    async def _restore_automations(self) -> None:
        """Restore automations that were running before worker restart."""
        try:
            from tradingagents.storage.automation_control import get_automation_control
            control = get_automation_control()

            # Find automations that should be running
            statuses = await control.list_statuses()

            for status in statuses:
                if status.get("status") == "running" and status.get("worker_id") == self.worker_id:
                    instance_name = status.get("instance_name")
                    config = status.get("config", {})

                    self.logger.info(f"Restoring automation: {instance_name}")

                    try:
                        await self._handle_start_automation({
                            "instance_name": instance_name,
                            "config": config,
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to restore {instance_name}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to restore automations: {e}")

    async def _update_automation_status(self, instance_name: str, status: str) -> None:
        """Update automation status in DB."""
        try:
            from tradingagents.storage.automation_control import get_automation_control
            control = get_automation_control()

            await control.update_status(
                instance_name=instance_name,
                status=status,
                worker_id=self.worker_id,
            )
        except Exception as e:
            self.logger.warning(f"Failed to update automation status: {e}")


def run_mt5_worker(worker_id: Optional[str] = None) -> None:
    """Run the MT5 worker as a standalone process."""
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reconfigure stdout for Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    worker = MT5Worker(worker_id=worker_id)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        print("\nWorker stopped by user")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MT5 Worker")
    parser.add_argument("--worker-id", type=str, help="Unique worker ID")
    args = parser.parse_args()

    run_mt5_worker(args.worker_id)
