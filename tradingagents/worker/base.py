"""
Base Worker Class for Queue-Based Processing

Provides the core worker loop that:
- Registers with the command queue
- Sends periodic heartbeats
- Claims and processes commands
- Handles graceful shutdown

Subclass and implement handle_command() for specific worker types.
"""

import asyncio
import logging
import os
import signal
import socket
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional

from tradingagents.storage.command_queue import (
    CommandQueue,
    Command,
    CommandType,
    get_command_queue,
)


class BaseWorker(ABC):
    """
    Base class for queue-based workers.

    Subclass and implement:
    - handle_command(command): Process a single command
    - get_capabilities(): Return list of command types this worker handles
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        heartbeat_interval: float = 30.0,
        stale_recovery_interval: float = 60.0,
    ):
        """
        Initialize the worker.

        Args:
            worker_id: Unique worker ID (auto-generated if not provided)
            poll_interval: Seconds between queue polls
            heartbeat_interval: Seconds between heartbeats
            stale_recovery_interval: Seconds between stale command recovery checks
        """
        self.worker_id = worker_id or self._generate_worker_id()
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.stale_recovery_interval = stale_recovery_interval

        self.queue: CommandQueue = get_command_queue()
        self.logger = logging.getLogger(f"Worker.{self.worker_id}")

        self._running = False
        self._shutdown_event = asyncio.Event()
        self._current_command: Optional[Command] = None
        self._commands_processed = 0
        self._started_at: Optional[datetime] = None

    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID."""
        hostname = socket.gethostname()
        pid = os.getpid()
        short_uuid = uuid.uuid4().hex[:6]
        return f"{hostname}-{pid}-{short_uuid}"

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of command types this worker can handle.

        Override in subclass. Return empty list to handle all types.
        """
        pass

    @abstractmethod
    async def handle_command(self, command: Command) -> Dict[str, Any]:
        """
        Process a single command.

        Args:
            command: The command to process

        Returns:
            Result dict to store with the completed command

        Raises:
            Exception: If command processing fails
        """
        pass

    async def on_startup(self) -> None:
        """Called when worker starts. Override for initialization."""
        pass

    async def on_shutdown(self) -> None:
        """Called when worker stops. Override for cleanup."""
        pass

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._started_at = datetime.now()
        self._shutdown_event.clear()

        # Setup signal handlers
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._handle_signal)

        self.logger.info("=" * 60)
        self.logger.info(f"WORKER STARTING: {self.worker_id}")
        self.logger.info(f"Capabilities: {self.get_capabilities() or 'ALL'}")
        self.logger.info("=" * 60)

        try:
            # Register with queue
            await self.queue.register_worker(
                self.worker_id,
                self.get_capabilities() or None,
            )
            self.logger.info("Registered with command queue")

            # Run startup hook
            await self.on_startup()

            # Start background tasks
            await asyncio.gather(
                self._process_loop(),
                self._heartbeat_loop(),
                self._maintenance_loop(),
            )

        except asyncio.CancelledError:
            self.logger.info("Worker cancelled")
        except Exception as e:
            self.logger.error(f"Worker error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self.logger.info("Stopping worker...")
        self._running = False
        self._shutdown_event.set()

    def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        self.logger.info("Received shutdown signal")
        asyncio.create_task(self.stop())

    async def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        try:
            await self.on_shutdown()
            await self.queue.deregister_worker(self.worker_id)
            self.logger.info(f"Worker deregistered. Processed {self._commands_processed} commands.")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

        self.logger.info("=" * 60)
        self.logger.info("WORKER STOPPED")
        self.logger.info("=" * 60)

    async def _process_loop(self) -> None:
        """Main processing loop - claim and process commands."""
        self.logger.info("Process loop started")

        while self._running:
            try:
                # Try to claim next command
                command = await self.queue.claim_next(
                    self.worker_id,
                    self.get_capabilities() or None,
                )

                if command:
                    self._current_command = command
                    self.logger.info(
                        f"Claimed command: {command.id} "
                        f"(type={command.command_type}, priority={command.priority})"
                    )

                    try:
                        # Process the command
                        result = await self.handle_command(command)

                        # Mark as completed
                        await self.queue.complete(
                            command.id,
                            result,
                            self.worker_id,
                        )
                        self._commands_processed += 1
                        self.logger.info(f"Completed command: {command.id}")

                    except Exception as e:
                        error_msg = str(e)
                        self.logger.error(f"Command {command.id} failed: {error_msg}")

                        # Mark as failed (will retry if retries remaining)
                        await self.queue.fail(
                            command.id,
                            error_msg,
                            self.worker_id,
                            retry=True,
                        )

                    finally:
                        self._current_command = None

                else:
                    # No commands available, wait before polling again
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.poll_interval,
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue polling

            except Exception as e:
                self.logger.error(f"Process loop error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # Back off on errors

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                current_cmd = self._current_command.id if self._current_command else None
                success = await self.queue.heartbeat(self.worker_id, current_cmd)

                if not success:
                    self.logger.warning("Heartbeat failed - worker may be marked inactive")

            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.heartbeat_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _maintenance_loop(self) -> None:
        """Periodic maintenance tasks (stale recovery, etc.)."""
        while self._running:
            try:
                # Recover stale commands from dead workers
                recovered = await self.queue.recover_stale_commands()
                if recovered > 0:
                    self.logger.info(f"Recovered {recovered} stale commands")

                # Expire old commands
                expired = await self.queue.expire_old_commands()
                if expired > 0:
                    self.logger.info(f"Expired {expired} commands")

            except Exception as e:
                self.logger.error(f"Maintenance error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.stale_recovery_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "commands_processed": self._commands_processed,
            "current_command": self._current_command.id if self._current_command else None,
            "capabilities": self.get_capabilities(),
        }
