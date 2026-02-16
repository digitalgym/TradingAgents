"""
Daily Scheduler

Background daemon that runs daily workflow cycles at configured times:
- Morning Analysis (default 8:00 UTC)
- Midday Review (default 13:00 UTC)
- Evening Reflect (default 20:00 UTC)
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any
import json

try:
    import pytz
except ImportError:
    pytz = None

from .portfolio_config import ScheduleConfig
from .portfolio_automation import PortfolioAutomation
from .reporting import DailyAnalysisReport, PositionReviewReport, ReflectionReport


class DailyScheduler:
    """
    Schedules daily workflow execution.

    Supports:
    - Configurable times for each cycle
    - Timezone-aware scheduling
    - Manual trigger override
    - Graceful shutdown
    - Retry logic for failures
    """

    def __init__(
        self,
        portfolio_automation: PortfolioAutomation,
        schedule_config: Optional[ScheduleConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            portfolio_automation: The automation orchestrator
            schedule_config: Schedule configuration (uses default if not provided)
        """
        self.automation = portfolio_automation
        self.config = schedule_config or portfolio_automation.config.schedule

        # Timezone
        if pytz and self.config.timezone:
            self.tz = pytz.timezone(self.config.timezone)
        else:
            self.tz = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_run: dict = {
            "morning": None,
            "midday": None,
            "evening": None,
        }

        # Logging
        self.logger = logging.getLogger("DailyScheduler")

        # PID file for daemon management
        self._pid_file = Path("portfolio_scheduler.pid")

        # Load previous state if available
        self._load_last_run()

    def _get_current_time(self) -> datetime:
        """Get current time in configured timezone."""
        now = datetime.now()
        if self.tz:
            now = datetime.now(self.tz)
        return now

    def _should_run_cycle(self, cycle: str) -> bool:
        """Check if a cycle should run now."""
        now = self._get_current_time()
        today = now.date().isoformat()

        # Get cycle time
        if cycle == "morning":
            hour = self.config.morning_analysis_hour
            minute = self.config.morning_analysis_minute
        elif cycle == "midday":
            hour = self.config.midday_review_hour
            minute = self.config.midday_review_minute
        elif cycle == "evening":
            hour = self.config.evening_reflect_hour
            minute = self.config.evening_reflect_minute
        else:
            return False

        # Check if we're within the window (within 5 minutes of scheduled time)
        scheduled_time = dt_time(hour, minute)
        current_time = now.time()

        # Convert to minutes for easier comparison
        scheduled_minutes = hour * 60 + minute
        current_minutes = current_time.hour * 60 + current_time.minute

        # Within 5 minute window
        if abs(current_minutes - scheduled_minutes) > 5:
            return False

        # Check if already run today
        if self._last_run[cycle] == today:
            return False

        return True

    async def _run_cycle(self, cycle: str) -> Any:
        """Run a specific cycle."""
        now = self._get_current_time()
        today = now.date().isoformat()

        self.logger.info(f"Running {cycle} cycle...")

        try:
            if cycle == "morning":
                report = await self.automation.run_morning_analysis()
            elif cycle == "midday":
                report = await self.automation.run_midday_review()
            elif cycle == "evening":
                report = await self.automation.run_evening_reflect()
            else:
                raise ValueError(f"Unknown cycle: {cycle}")

            # Mark as run
            self._last_run[cycle] = today
            self._save_last_run()

            return report

        except Exception as e:
            self.logger.error(f"{cycle} cycle failed: {e}")
            raise

    async def _run_with_retry(
        self,
        cycle: str,
        max_retries: int = 3,
        retry_delay: int = 60,
    ) -> Optional[Any]:
        """Run a cycle with retry logic."""
        for attempt in range(max_retries):
            try:
                return await self._run_cycle(cycle)
            except Exception as e:
                self.logger.error(f"{cycle} attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"{cycle} failed after {max_retries} attempts")
                    return None

    async def start(self):
        """Start the scheduler loop."""
        self._running = True
        self._write_pid_file()

        # Setup signal handlers for graceful shutdown
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self._handle_shutdown)

        self.logger.info("=" * 50)
        self.logger.info("PORTFOLIO SCHEDULER STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Timezone: {self.config.timezone}")
        self.logger.info(f"Morning Analysis: {self.config.morning_analysis_hour:02d}:{self.config.morning_analysis_minute:02d}")
        self.logger.info(f"Midday Review: {self.config.midday_review_hour:02d}:{self.config.midday_review_minute:02d}")
        self.logger.info(f"Evening Reflect: {self.config.evening_reflect_hour:02d}:{self.config.evening_reflect_minute:02d}")
        self.logger.info("=" * 50)

        try:
            while self._running:
                # Check each cycle
                for cycle in ["morning", "midday", "evening"]:
                    if self._should_run_cycle(cycle):
                        try:
                            result = await self._run_with_retry(cycle)
                            if result is None:
                                self.logger.warning(f"{cycle} cycle returned None (all retries failed)")
                            else:
                                self.logger.info(f"{cycle} cycle completed successfully")
                        except Exception as e:
                            # Catch any exception that bubbles up from retry logic
                            self.logger.error(f"Unhandled exception in {cycle} cycle: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            # Continue to next cycle instead of crashing

                # Sleep for 1 minute before checking again
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60,
                    )
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass

        except Exception as e:
            self.logger.error(f"Critical error in scheduler main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self._cleanup()

    def _handle_shutdown(self):
        """Handle shutdown signal."""
        self.logger.info("Shutdown signal received...")
        self._running = False
        self._shutdown_event.set()

    def stop(self):
        """Stop the scheduler."""
        self.logger.info("Stopping scheduler...")
        self._running = False
        self._shutdown_event.set()

    def _cleanup(self):
        """Cleanup on shutdown."""
        self._remove_pid_file()
        self.logger.info("Scheduler stopped")

    def _write_pid_file(self):
        """Write PID file for daemon management."""
        try:
            with open(self._pid_file, "w") as f:
                f.write(str(os.getpid()))
        except Exception as e:
            self.logger.warning(f"Could not write PID file: {e}")

    def _remove_pid_file(self):
        """Remove PID file."""
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
        except Exception as e:
            self.logger.warning(f"Could not remove PID file: {e}")

    def _save_last_run(self):
        """Save last run times."""
        state_file = Path("scheduler_state.json")
        try:
            with open(state_file, "w") as f:
                json.dump(self._last_run, f)
            self.logger.debug(f"Saved scheduler state: {self._last_run}")
        except Exception as e:
            self.logger.warning(f"Failed to save scheduler state: {e}")

    def _load_last_run(self):
        """Load last run times."""
        state_file = Path("scheduler_state.json")
        try:
            if state_file.exists():
                with open(state_file, "r") as f:
                    self._last_run = json.load(f)
                self.logger.debug(f"Loaded scheduler state: {self._last_run}")
            else:
                self.logger.debug("No scheduler state file found, starting fresh")
        except Exception as e:
            self.logger.warning(f"Failed to load scheduler state: {e}")

    async def trigger_manual(self, cycle: str) -> Any:
        """
        Manually trigger a specific cycle.

        Args:
            cycle: Cycle to trigger ("morning", "midday", "evening")

        Returns:
            Cycle report
        """
        if cycle not in ["morning", "midday", "evening"]:
            raise ValueError(f"Unknown cycle: {cycle}. Must be morning, midday, or evening.")

        self.logger.info(f"Manual trigger: {cycle}")
        return await self._run_cycle(cycle)

    def get_next_run_times(self) -> dict:
        """Get estimated next run times for each cycle."""
        now = self._get_current_time()
        today = now.date()

        result = {}
        for cycle, (hour, minute) in [
            ("morning", (self.config.morning_analysis_hour, self.config.morning_analysis_minute)),
            ("midday", (self.config.midday_review_hour, self.config.midday_review_minute)),
            ("evening", (self.config.evening_reflect_hour, self.config.evening_reflect_minute)),
        ]:
            scheduled = datetime.combine(today, dt_time(hour, minute))
            if self.tz:
                scheduled = self.tz.localize(scheduled)

            # If already past, next run is tomorrow
            if now.time() > dt_time(hour, minute) or self._last_run[cycle] == today.isoformat():
                from datetime import timedelta
                scheduled = scheduled + timedelta(days=1)

            result[cycle] = scheduled.isoformat()

        return result

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "timezone": self.config.timezone,
            "last_run": self._last_run,
            "next_run": self.get_next_run_times(),
            "schedule": {
                "morning": f"{self.config.morning_analysis_hour:02d}:{self.config.morning_analysis_minute:02d}",
                "midday": f"{self.config.midday_review_hour:02d}:{self.config.midday_review_minute:02d}",
                "evening": f"{self.config.evening_reflect_hour:02d}:{self.config.evening_reflect_minute:02d}",
            },
        }


def run_scheduler_daemon(config_path: Optional[str] = None):
    """
    Run the scheduler as a daemon process.

    Args:
        config_path: Path to portfolio configuration file
    """
    from .portfolio_config import load_portfolio_config, get_default_config
    import traceback

    # Setup logging for daemon
    logger = logging.getLogger("SchedulerDaemon")

    logger.info("=" * 60)
    logger.info("STARTING PORTFOLIO SCHEDULER DAEMON")
    logger.info("=" * 60)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path or 'default'}")
        if config_path:
            config = load_portfolio_config(config_path)
        else:
            config = get_default_config()
        logger.info(f"Configuration loaded successfully. Execution mode: {config.execution_mode}")
        logger.info(f"Symbols configured: {[s.symbol for s in config.symbols]}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.error("Daemon cannot start without valid configuration")
        return
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        logger.error("Daemon cannot start with invalid configuration")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        # Initialize automation
        logger.info("Initializing PortfolioAutomation...")
        automation = PortfolioAutomation(config)
        logger.info("PortfolioAutomation initialized successfully")

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to initialize PortfolioAutomation: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        # Initialize scheduler
        logger.info("Initializing DailyScheduler...")
        scheduler = DailyScheduler(automation)
        logger.info("DailyScheduler initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize DailyScheduler: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        # Run the scheduler
        logger.info("Starting scheduler event loop...")
        asyncio.run(scheduler.start())

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Scheduler crashed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("=" * 60)
        logger.info("PORTFOLIO SCHEDULER DAEMON STOPPED")
        logger.info("=" * 60)
