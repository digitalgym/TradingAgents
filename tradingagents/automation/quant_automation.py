"""
Quant Automation System

Frequent-interval automation for quant and multi-agent pipelines.
Supports:
- Configurable update intervals (e.g., 2-3 minutes)
- Pipeline selection (quant vs multi-agent)
- Automatic trade execution
- Position monitoring and management (modify SL/TP, close)
"""

import asyncio
import logging
import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

# MT5 imports
from tradingagents.dataflows.mt5_data import (
    get_mt5_current_price,
    get_open_positions,
    get_pending_orders,
    modify_position,
    close_position,
    execute_trade_signal,
    get_mt5_symbol_info,
    check_mt5_autotrading,
    is_market_open,
)

# Risk management
from tradingagents.risk import (
    RiskGuardrails,
    DynamicStopLoss,
    PositionSizer,
    get_atr_for_symbol,
)

# Trade decision tracking
from tradingagents.trade_decisions import (
    store_decision,
    list_active_decisions,
    close_decision,
    find_decision_by_ticket,
    add_trade_event,
    DECISIONS_DIR,
)

from tradingagents.dataflows.mt5_data import get_closed_deal_by_ticket


# Project root for resolving state/config files regardless of CWD
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def normalize_signal(signal) -> str:
    """
    Normalize a signal value from various formats to BUY/SELL/HOLD.

    Handles:
    - Dict format: {"value": "buy_to_enter"}
    - String format: "buy_to_enter", "BUY_TO_ENTER", "buy", "BUY"

    Returns: "BUY", "SELL", or "HOLD"
    """
    if signal is None:
        return "HOLD"
    if isinstance(signal, dict):
        signal = signal.get("value", "HOLD")
    signal = str(signal).upper()

    if signal in ["BUY_TO_ENTER", "BUY"]:
        return "BUY"
    elif signal in ["SELL_TO_ENTER", "SELL"]:
        return "SELL"
    return "HOLD"

# Global symbol limits file (shared across all automations)
_SYMBOL_LIMITS_FILE = _PROJECT_ROOT / "automation_symbol_limits.json"


def _get_global_position_limits() -> Dict[str, int]:
    """Load per-symbol max positions from automation_symbol_limits.json.

    Returns dict like {"XAUUSD": 3, "XAGUSD": 1}.
    Symbols not in the file default to 3.
    """
    limits: Dict[str, int] = {}
    try:
        with open(_SYMBOL_LIMITS_FILE) as f:
            data = json.load(f)
        for sym, cfg in data.items():
            if isinstance(cfg, dict) and "max_positions" in cfg:
                limits[sym] = cfg["max_positions"]
    except Exception:
        pass
    return limits


class PipelineType(str, Enum):
    """Available analysis pipelines."""
    SMC_QUANT_BASIC = "smc_quant_basic"
    SMC_QUANT = "smc_quant"
    SMC_MTF = "smc_mtf"
    BREAKOUT_QUANT = "breakout_quant"
    RANGE_QUANT = "range_quant"
    VOLUME_PROFILE = "volume_profile"
    RULE_BASED = "rule_based"
    MULTI_AGENT = "multi_agent"
    XGBOOST = "xgboost"
    XGBOOST_ENSEMBLE = "xgboost_ensemble"


class AutomationStatus(str, Enum):
    """Automation status states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class QuantAutomationConfig:
    """Configuration for quant automation."""
    # Instance identity
    instance_name: str = ""  # Set by backend, used as trade decision source

    # Pipeline settings
    pipeline: PipelineType = PipelineType.SMC_QUANT_BASIC
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    timeframe: str = "H1"

    # Interval settings
    analysis_interval_seconds: int = 180  # 3 minutes default
    position_check_interval_seconds: int = 60  # 1 minute for position monitoring

    # Execution settings
    auto_execute: bool = False  # Require explicit enable for auto-trading
    min_confidence: float = 0.65  # Minimum confidence to execute trades

    # Position management (per-automation limit)
    max_positions_per_symbol: int = 1  # Max positions THIS automation can open per symbol
    enable_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    enable_breakeven_stop: bool = True  # Move SL to breakeven after profit >= N * ATR
    move_to_breakeven_atr_mult: float = 1.5  # Move SL to breakeven after profit >= 1.5x ATR
    enable_reversal_close: bool = True  # Close position on reversal signal from analysis

    # Risk settings
    max_risk_per_trade_pct: float = 1.0  # 1% of account per trade
    default_lot_size: float = 0.01
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 3

    # Position assumption review settings
    assumption_review_interval_seconds: int = 3600  # 1 hour default
    assumption_review_auto_apply: bool = False  # Auto-apply SL/TP adjustments from review

    # Remote trade queue settings
    enable_trade_queue: bool = True  # Process remote trade commands from web UI
    trade_queue_poll_seconds: int = 5  # Check queue every 5 seconds

    # Remote control settings
    enable_remote_control: bool = True  # Allow start/stop/config from web UI
    control_poll_seconds: int = 3  # Check for control commands every 3 seconds

    # State persistence
    state_file: str = "quant_automation_state.json"
    logs_dir: str = "logs/quant_automation"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["pipeline"] = self.pipeline.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantAutomationConfig":
        """Create from dictionary."""
        if "pipeline" in data and isinstance(data["pipeline"], str):
            data["pipeline"] = PipelineType(data["pipeline"])
        # Backward compat: migrate old move_to_breakeven_pct to move_to_breakeven_atr_mult
        if "move_to_breakeven_pct" in data and "move_to_breakeven_atr_mult" not in data:
            # Old configs used %, new uses ATR multiplier. Default to 1.5x ATR.
            data["move_to_breakeven_atr_mult"] = 1.5
        # Strip unknown keys (e.g. removed fields from older configs)
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AnalysisCycleResult:
    """Result from an analysis cycle."""
    timestamp: datetime
    symbol: str
    pipeline: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: str = ""
    trailing_stop_atr_multiplier: Optional[float] = None
    executed: bool = False
    execution_ticket: Optional[int] = None
    execution_error: Optional[str] = None
    duration_seconds: float = 0


@dataclass
class PositionManagementResult:
    """Result from position management cycle."""
    timestamp: datetime
    ticket: int
    symbol: str
    action: str  # "adjusted_sl", "adjusted_tp", "closed", "no_change"
    old_sl: Optional[float] = None
    new_sl: Optional[float] = None
    old_tp: Optional[float] = None
    new_tp: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None


class QuantAutomation:
    """
    Quant Automation System.

    Runs analysis at configurable intervals and manages positions.
    """

    def __init__(self, config: Optional[QuantAutomationConfig] = None):
        """Initialize quant automation."""
        self.config = config or QuantAutomationConfig()

        # Source identifier for trade decisions
        self._source = (
            self.config.instance_name
            if self.config.instance_name
            else f"quant_automation_{self.config.pipeline.value}"
        )

        # State
        self._status = AutomationStatus.STOPPED
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_analysis_time: Dict[str, datetime] = {}
        self._last_position_check: Optional[datetime] = None
        self._last_assumption_review: Optional[datetime] = None
        self._analysis_results: List[AnalysisCycleResult] = []
        self._position_results: List[PositionManagementResult] = []
        self._assumption_review_results: List[Dict[str, Any]] = []
        self._error_message: Optional[str] = None
        self._missing_ticket_counts: Dict[int, int] = {}  # ticket -> consecutive missing count
        self._market_closed_until: Optional[datetime] = None  # suppress modifications when market is closed

        # Lock for state file writes (multiple async loops share this)
        self._state_lock = asyncio.Lock()

        # Shared aiohttp session for API calls (created lazily)
        self._http_session: Optional["aiohttp.ClientSession"] = None

        # Postgres pool for state persistence (created lazily)
        self._pg_pool: Optional[Any] = None
        self._pg_url: Optional[str] = os.environ.get("POSTGRES_URL")

        # Risk management
        self.guardrails = RiskGuardrails(
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            max_consecutive_losses=self.config.max_consecutive_losses,
        )
        self.stop_loss_manager = DynamicStopLoss(
            atr_multiplier=2.0,
            trailing_multiplier=self.config.trailing_stop_atr_multiplier,
        )

        # Logging
        self._setup_logging()

        # Load state
        self._load_state()

    def _setup_logging(self):
        """Setup logging."""
        logs_dir = _PROJECT_ROOT / self.config.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"QuantAutomation.{self._source}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # File handler
        log_file = logs_dir / f"quant_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        self.logger.addHandler(console_handler)

    async def _get_http_session(self) -> "aiohttp.ClientSession":
        """Get or create shared aiohttp session."""
        import aiohttp

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _close_http_session(self) -> None:
        """Close the shared HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def _get_pg_pool(self):
        """Get or create asyncpg connection pool for state persistence."""
        if self._pg_pool is None and self._pg_url:
            try:
                import asyncpg
                self._pg_pool = await asyncpg.create_pool(
                    self._pg_url, min_size=1, max_size=2,
                    statement_cache_size=0,  # Required for Neon pooler (PgBouncer)
                )
                # Ensure state column exists
                async with self._pg_pool.acquire() as conn:
                    await conn.execute("""
                        ALTER TABLE automation_status
                        ADD COLUMN IF NOT EXISTS state jsonb
                    """)
                self.logger.info("Postgres state persistence enabled")
            except Exception as e:
                self.logger.warning(f"Postgres state persistence unavailable: {e}")
                self._pg_pool = None
        return self._pg_pool

    def _load_state(self):
        """Load automation state from file (sync, used in __init__)."""
        state_file = _PROJECT_ROOT / self.config.state_file
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self._apply_state(state)
            except Exception as e:
                self.logger.warning(f"Could not load state from file: {e}")

    async def _load_state_from_db(self):
        """Load state from Postgres if available (async, called after startup)."""
        pool = await self._get_pg_pool()
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT state FROM automation_status WHERE instance_name = $1",
                    self._source,
                )
                if row and row["state"]:
                    state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
                    # DB state is newer if it has a later last_updated
                    db_updated = state.get("last_updated")
                    file_updated = None
                    state_file = _PROJECT_ROOT / self.config.state_file
                    if state_file.exists():
                        try:
                            with open(state_file, "r") as f:
                                file_state = json.load(f)
                            file_updated = file_state.get("last_updated")
                        except Exception:
                            pass
                    if db_updated and (not file_updated or db_updated > file_updated):
                        self._apply_state(state)
                        self.logger.info(f"Loaded state from Postgres (updated {db_updated})")
        except Exception as e:
            self.logger.warning(f"Could not load state from Postgres: {e}")

    def _apply_state(self, state: dict):
        """Apply a state dict to instance attributes."""
        self._last_analysis_time = {
            k: datetime.fromisoformat(v)
            for k, v in state.get("last_analysis_time", {}).items()
        }
        # Restore persisted results history
        self._analysis_results.clear()
        for item in state.get("analysis_results", []):
            try:
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                self._analysis_results.append(AnalysisCycleResult(**item))
            except Exception:
                continue
        self._position_results.clear()
        for item in state.get("position_results", []):
            try:
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                self._position_results.append(PositionManagementResult(**item))
            except Exception:
                continue
        # Restore assumption review state
        last_review = state.get("last_assumption_review")
        if last_review:
            self._last_assumption_review = datetime.fromisoformat(last_review)
        self._assumption_review_results = state.get("assumption_review_results", [])

    async def _save_state(self):
        """Save automation state to Postgres (primary) and JSON file (fallback).

        Postgres eliminates Windows file-locking issues with os.replace().
        JSON file kept as fallback when POSTGRES_URL is not set.
        """
        def _serialize_result(r):
            d = asdict(r)
            d["timestamp"] = r.timestamp.isoformat()
            return d

        state = {
            "last_analysis_time": {
                k: v.isoformat() for k, v in self._last_analysis_time.items()
            },
            "last_updated": datetime.now().isoformat(),
            "status": self._status.value,
            "analysis_results": [_serialize_result(r) for r in self._analysis_results[-100:]],
            "position_results": [_serialize_result(r) for r in self._position_results[-100:]],
            "last_assumption_review": self._last_assumption_review.isoformat() if self._last_assumption_review else None,
            "assumption_review_results": self._assumption_review_results,
        }

        saved_to_db = False

        # Primary: write to Postgres
        pool = await self._get_pg_pool()
        if pool:
            try:
                state_json = json.dumps(state)
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO automation_status (instance_name, status, state, updated_at)
                        VALUES ($1, $2, $3::jsonb, NOW())
                        ON CONFLICT (instance_name)
                        DO UPDATE SET status = $2, state = $3::jsonb, updated_at = NOW()
                    """, self._source, self._status.value, state_json)
                saved_to_db = True
            except Exception as e:
                self.logger.warning(f"Failed to save state to Postgres: {e}")
                # Reset pool on connection errors to force reconnect next time
                if "closed" in str(e).lower() or "release" in str(e).lower():
                    self._pg_pool = None

        # Fallback: write to JSON file (always, for local tooling / backend reads)
        state_file = _PROJECT_ROOT / self.config.state_file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = state_file.with_suffix(f".{os.getpid()}.tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(state, f, indent=2)
            last_err = None
            for attempt in range(4):
                try:
                    os.replace(str(tmp_file), str(state_file))
                    last_err = None
                    break
                except PermissionError as e:
                    last_err = e
                    if attempt < 3:
                        await asyncio.sleep(0.1 * (attempt + 1))
            if last_err:
                if saved_to_db:
                    self.logger.debug(f"File save failed but state saved to Postgres: {last_err}")
                else:
                    self.logger.warning(f"Failed to save state (file + DB both failed): {last_err}")
        except Exception as e:
            if not saved_to_db:
                self.logger.warning(f"Failed to save state: {e}")
        finally:
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _get_account_balance(self) -> float:
        """Get current account balance from MT5."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                self.logger.warning("MT5 not initialized, using default balance 10000.0")
                return 10000.0
            account_info = mt5.account_info()
            if account_info:
                self.logger.debug(f"Account balance: {account_info.balance}")
                return account_info.balance
            self.logger.warning("MT5 account_info returned None, using default balance 10000.0")
            return 10000.0
        except Exception as e:
            self.logger.warning(f"Failed to get account balance: {e}, using default 10000.0")
            return 10000.0

    async def _run_quant_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run quant pipeline analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

            # Call the quant analysis API endpoint
            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/quant",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Quant API error: {error_text}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="quant",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=time.time() - start_time,
                        )

                    result = await response.json()

            # Log full API response for debugging
            self.logger.info(f"API response status: {result.get('status')}")
            self.logger.info(f"API response keys: {list(result.keys())}")

            # Log the prompt that was sent to the LLM
            prompt_sent = result.get("prompt_sent")
            if prompt_sent:
                self.logger.info(f"\n{'='*80}\nPROMPT SENT TO LLM for {symbol}\n{'='*80}\n{prompt_sent}\n{'='*80}")
            else:
                self.logger.warning(f"No prompt_sent field in API response for {symbol}")

            # Extract decision from API response
            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in API response for {symbol}. Full response: {json.dumps(result, default=str)[:500]}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="quant",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from analysis",
                    duration_seconds=time.time() - start_time,
                )

            # Log the decision received
            self.logger.info(
                f"Decision received for {symbol}: signal={decision.get('signal')}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )
            if decision.get("rationale"):
                self.logger.info(f"Rationale: {str(decision.get('rationale', ''))[:200]}")

            signal = normalize_signal(decision.get("signal"))
            self.logger.info(f"Mapped signal for {symbol}: {signal}")

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="quant",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Quant analysis error for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=time.time() - start_time,
            )

    async def _run_smc_quant_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run dedicated SMC quant pipeline analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"SMC Quant: calling API for {symbol} (timeout=120s)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/smc-quant",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"SMC Quant API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="smc_quant",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            llm_duration = result.get("llm_duration_seconds")
            timing_str = f"total={api_duration:.1f}s"
            if llm_duration is not None:
                timing_str += f", llm={llm_duration:.1f}s, overhead={api_duration - llm_duration:.1f}s"
            self.logger.info(f"SMC Quant API response: status={result.get('status')} [{timing_str}]")
            if api_duration > 60:
                self.logger.warning(f"SMC Quant API call was SLOW ({api_duration:.1f}s) for {symbol}")

            # Log error details if the API returned an error
            if result.get("status") == "error":
                self.logger.error(f"SMC Quant API error for {symbol}: {result.get('error', 'unknown')}")
                if result.get("traceback"):
                    self.logger.error(f"Traceback: {result['traceback'][:500]}")

            prompt_sent = result.get("prompt_sent")
            if prompt_sent:
                self.logger.info(f"\n{'='*80}\nSMC QUANT PROMPT SENT for {symbol}\n{'='*80}\n{prompt_sent}\n{'='*80}")

            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in SMC Quant API response for {symbol}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="smc_quant",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from analysis",
                    duration_seconds=time.time() - start_time,
                )

            self.logger.info(
                f"SMC Quant decision for {symbol}: signal={decision.get('signal')}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )

            signal = normalize_signal(decision.get("signal"))

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_quant",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                trailing_stop_atr_multiplier=decision.get("trailing_stop_atr_multiplier"),
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"SMC Quant TIMEOUT for {symbol} after {elapsed:.1f}s - LLM likely hanging or overloaded")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"SMC Quant analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_breakout_quant_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run dedicated Breakout quant pipeline analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"Breakout Quant: calling API for {symbol} (timeout=120s)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/breakout-quant",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Breakout Quant API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="breakout_quant",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            llm_duration = result.get("llm_duration_seconds")
            timing_str = f"total={api_duration:.1f}s"
            if llm_duration is not None:
                timing_str += f", llm={llm_duration:.1f}s, overhead={api_duration - llm_duration:.1f}s"
            self.logger.info(f"Breakout Quant API response: status={result.get('status')} [{timing_str}]")
            if api_duration > 60:
                self.logger.warning(f"Breakout Quant API call was SLOW ({api_duration:.1f}s) for {symbol}")

            # Log error details if the API returned an error
            if result.get("status") == "error":
                self.logger.error(f"Breakout Quant API error for {symbol}: {result.get('error', 'unknown')}")
                if result.get("traceback"):
                    self.logger.error(f"Traceback: {result['traceback'][:500]}")

            # Log consolidation info
            consolidation = result.get("consolidation", {})
            if consolidation:
                self.logger.info(
                    f"Breakout Quant consolidation for {symbol}: "
                    f"is_consolidating={consolidation.get('is_consolidating')}, "
                    f"squeeze_strength={consolidation.get('squeeze_strength', 0):.1f}%, "
                    f"structure_bias={consolidation.get('structure_bias')}, "
                    f"breakout_ready={consolidation.get('breakout_ready')}"
                )

            prompt_sent = result.get("prompt_sent")
            if prompt_sent:
                self.logger.info(f"\n{'='*80}\nBREAKOUT QUANT PROMPT SENT for {symbol}\n{'='*80}\n{prompt_sent}\n{'='*80}")

            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in Breakout Quant API response for {symbol}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="breakout_quant",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from analysis",
                    duration_seconds=time.time() - start_time,
                )

            self.logger.info(
                f"Breakout Quant decision for {symbol}: signal={decision.get('signal')}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )

            signal = normalize_signal(decision.get("signal"))

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="breakout_quant",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                trailing_stop_atr_multiplier=decision.get("trailing_stop_atr_multiplier"),
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"Breakout Quant TIMEOUT for {symbol} after {elapsed:.1f}s - LLM likely hanging or overloaded")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="breakout_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Breakout Quant analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="breakout_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_range_quant_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run dedicated Range quant pipeline analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"Range Quant: calling API for {symbol} (timeout=120s)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/range-quant",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Range Quant API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="range_quant",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            llm_duration = result.get("llm_duration_seconds")
            timing_str = f"total={api_duration:.1f}s"
            if llm_duration is not None:
                timing_str += f", llm={llm_duration:.1f}s, overhead={api_duration - llm_duration:.1f}s"
            self.logger.info(f"Range Quant API response: status={result.get('status')} [{timing_str}]")
            if api_duration > 60:
                self.logger.warning(f"Range Quant API call was SLOW ({api_duration:.1f}s) for {symbol}")

            if result.get("status") == "error":
                self.logger.error(f"Range Quant API error for {symbol}: {result.get('error', 'unknown')}")
                if result.get("traceback"):
                    self.logger.error(f"Traceback: {result['traceback'][:500]}")

            # Log range info
            range_info = result.get("range_analysis", {})
            if range_info:
                self.logger.info(
                    f"Range Quant range for {symbol}: "
                    f"is_ranging={range_info.get('is_ranging')}, "
                    f"mr_score={range_info.get('mean_reversion_score', 0):.1f}, "
                    f"price_position={range_info.get('price_position')}"
                )

            prompt_sent = result.get("prompt_sent")
            if prompt_sent:
                self.logger.info(f"\n{'='*80}\nRANGE QUANT PROMPT SENT for {symbol}\n{'='*80}\n{prompt_sent}\n{'='*80}")

            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in Range Quant API response for {symbol}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="range_quant",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from analysis",
                    duration_seconds=time.time() - start_time,
                )

            self.logger.info(
                f"Range Quant decision for {symbol}: signal={decision.get('signal')}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )

            signal = normalize_signal(decision.get("signal"))

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="range_quant",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                trailing_stop_atr_multiplier=decision.get("trailing_stop_atr_multiplier"),
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"Range Quant TIMEOUT for {symbol} after {elapsed:.1f}s - LLM likely hanging or overloaded")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="range_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Range Quant analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="range_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_vp_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run volume profile quant analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"VP Quant: calling API for {symbol} (timeout=120s)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/vp-quant",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"VP quant API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="volume_profile",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            llm_duration = result.get("llm_duration_seconds")
            timing_str = f"total={api_duration:.1f}s"
            if llm_duration is not None:
                timing_str += f", llm={llm_duration:.1f}s, overhead={api_duration - llm_duration:.1f}s"
            self.logger.info(f"[VP] API response: status={result.get('status')} [{timing_str}]")
            if api_duration > 60:
                self.logger.warning(f"VP Quant API call was SLOW ({api_duration:.1f}s) for {symbol}")

            if result.get("status") == "error":
                self.logger.warning(f"[VP] API returned error for {symbol}: {result.get('error')}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="volume_profile",
                    signal="HOLD",
                    confidence=0.0,
                    rationale=f"API error: {result.get('error', 'unknown')}",
                    duration_seconds=time.time() - start_time,
                )

            # VP endpoint returns flat structure (signal, entry_price, etc. at top level)
            self.logger.info(
                f"[VP] Decision for {symbol}: signal={result.get('signal')}, "
                f"confidence={result.get('confidence')}, "
                f"entry={result.get('entry_price')}, sl={result.get('stop_loss')}, tp={result.get('take_profit')}"
            )
            if result.get("justification"):
                self.logger.info(f"[VP] Justification: {str(result.get('justification', ''))[:200]}")

            signal = normalize_signal(result.get("signal"))
            justification = result.get("justification", "")
            invalidation = result.get("invalidation", "")
            rationale = f"{justification}\n\n**Invalidation**: {invalidation}" if invalidation else justification

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="volume_profile",
                signal=signal,
                confidence=result.get("confidence", 0.5),
                entry_price=result.get("entry_price"),
                stop_loss=result.get("stop_loss"),
                take_profit=result.get("take_profit"),
                rationale=rationale,
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"VP Quant TIMEOUT for {symbol} after {elapsed:.1f}s - LLM likely hanging or overloaded")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="volume_profile",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"VP analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="volume_profile",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_rule_based_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run rule-based SMC analysis (no LLM, instant & free)."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"Rule-Based: calling API for {symbol} (no LLM)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/rule-based",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Rule-Based API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="rule_based",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            self.logger.info(f"Rule-Based API response: status={result.get('status', 'ok')} [total={api_duration:.1f}s]")

            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in Rule-Based API response for {symbol}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="rule_based",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from rule-based analysis",
                    duration_seconds=api_duration,
                )

            signal = normalize_signal(decision.get("signal"))

            self.logger.info(
                f"Rule-Based decision for {symbol}: signal={signal}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="rule_based",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                duration_seconds=api_duration,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"Rule-Based TIMEOUT for {symbol} after {elapsed:.1f}s")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="rule_based",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Rule-Based analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="rule_based",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_smc_mtf_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run SMC Multi-Timeframe analysis (OTE + Channel, no LLM)."""
        start_time = time.time()

        try:
            import aiohttp

            self.logger.info(f"SMC MTF: calling API for {symbol} (no LLM)...")

            session = await self._get_http_session()
            payload = {
                "symbol": symbol,
                "timeframe": self.config.timeframe,
            }
            async with session.post(
                "http://localhost:8000/api/analysis/smc-mtf",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                    api_duration = time.time() - start_time
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"SMC MTF API error (HTTP {response.status}) after {api_duration:.1f}s: {error_text[:200]}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="smc_mtf",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=api_duration,
                        )

                    result = await response.json()

            api_duration = time.time() - start_time
            self.logger.info(f"SMC MTF API response: status={result.get('status', 'ok')} [total={api_duration:.1f}s]")

            # Log MTF alignment details
            mtf_details = result.get("mtf_details", {})
            self.logger.info(
                f"SMC MTF details: score={mtf_details.get('alignment_score')}, "
                f"bias={mtf_details.get('trade_bias')}, htf={mtf_details.get('htf_bias')}, "
                f"ltf={mtf_details.get('ltf_bias')}, ote={mtf_details.get('price_in_ote')}, "
                f"confirmation={mtf_details.get('has_entry_confirmation')}"
            )

            decision = result.get("decision", {})

            if not decision:
                self.logger.warning(f"No decision in SMC MTF API response for {symbol}")
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="smc_mtf",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from MTF analysis",
                    duration_seconds=api_duration,
                )

            signal = normalize_signal(decision.get("signal"))

            self.logger.info(
                f"SMC MTF decision for {symbol}: signal={signal}, "
                f"confidence={decision.get('confidence')}, "
                f"entry={decision.get('entry_price')}, "
                f"sl={decision.get('stop_loss')}, tp={decision.get('take_profit')}"
            )

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_mtf",
                signal=signal,
                confidence=decision.get("confidence", 0.5),
                entry_price=decision.get("entry_price"),
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                rationale=decision.get("rationale", ""),
                duration_seconds=api_duration,
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"SMC MTF TIMEOUT for {symbol} after {elapsed:.1f}s")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_mtf",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Timeout after {elapsed:.0f}s",
                duration_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"SMC MTF analysis error for {symbol} after {elapsed:.1f}s: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_mtf",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=elapsed,
            )

    async def _run_multi_agent_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run multi-agent pipeline analysis for a symbol."""
        start_time = time.time()

        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.default_config import DEFAULT_CONFIG

            # Initialize trading graph
            config = DEFAULT_CONFIG.copy()
            ta = TradingAgentsGraph(debug=False, config=config)

            # Run analysis
            trade_date = datetime.now().strftime("%Y-%m-%d")
            final_state, decision = ta.propagate(symbol, trade_date)

            if final_state is None:
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="multi_agent",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="Analysis returned no state",
                    duration_seconds=time.time() - start_time,
                )

            # Extract signal and confidence
            final_decision = final_state.get("final_trade_decision", "")
            signal = "HOLD"
            if "BUY" in final_decision.upper():
                signal = "BUY"
            elif "SELL" in final_decision.upper():
                signal = "SELL"

            # Extract confidence
            confidence = 0.5
            if "high confidence" in final_decision.lower() or "strong" in final_decision.lower():
                confidence = 0.8
            elif "medium" in final_decision.lower() or "moderate" in final_decision.lower():
                confidence = 0.6

            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="multi_agent",
                signal=signal,
                confidence=confidence,
                entry_price=decision.get("entry_price") if decision else None,
                stop_loss=decision.get("stop_loss") if decision else None,
                take_profit=decision.get("take_profit") if decision else None,
                rationale=final_decision[:500] if final_decision else "",
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"Multi-agent analysis error for {symbol}: {e}")
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="multi_agent",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=time.time() - start_time,
            )

    async def _execute_trade(self, result: AnalysisCycleResult) -> AnalysisCycleResult:
        """Execute a trade based on analysis result."""
        self.logger.info(
            f"--- TRADE EXECUTION CHECK for {result.symbol} ---"
            f" signal={result.signal}, confidence={result.confidence:.2f}"
        )

        if not self.config.auto_execute:
            self.logger.info(f"Auto-execute disabled, skipping trade for {result.symbol}")
            return result

        if result.signal == "HOLD":
            self.logger.info(f"Signal is HOLD, no trade for {result.symbol}")
            return result

        if result.confidence < self.config.min_confidence:
            self.logger.info(
                f"Confidence {result.confidence:.2f} below threshold "
                f"{self.config.min_confidence} for {result.symbol}, skipping"
            )
            return result

        # Check position limits
        # 1) Per-automation: count BOTH open positions AND recent active decisions
        #    (prevents rapid re-entry after SL hit)
        # 2) Global: per-symbol limit across ALL automations on the account
        # 3) Cooldown: min time between trades on same symbol per automation
        positions = get_open_positions()
        pending = get_pending_orders()

        # Build owned state for this automation
        my_source = self._source
        owned_tickets = set()
        my_recent_decision_count = 0  # Active decisions for this symbol (even if position closed)
        COOLDOWN_SECONDS = 1800  # 30 min cooldown after placing a trade
        latest_trade_time = None

        try:
            active_decisions = list_active_decisions()
            now = datetime.now()
            for dec in active_decisions:
                if dec.get("source") != my_source:
                    continue
                if dec.get("mt5_ticket"):
                    owned_tickets.add(dec["mt5_ticket"])
                # Count active decisions for this symbol (catches recently-closed positions)
                if dec.get("symbol") == result.symbol:
                    my_recent_decision_count += 1
                    # Track when the most recent trade was placed
                    created = dec.get("created_at", "")
                    if created:
                        try:
                            t = datetime.fromisoformat(created)
                            if latest_trade_time is None or t > latest_trade_time:
                                latest_trade_time = t
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

        # Per-automation count: open positions owned by this automation
        my_open_count = sum(
            1 for p in positions
            if p.get("symbol") == result.symbol and p.get("ticket") in owned_tickets
        )

        # Global counts (all positions + pending on the account for this symbol)
        global_symbol_count = sum(1 for p in positions if p.get("symbol") == result.symbol)
        global_symbol_count += sum(1 for o in pending if o.get("symbol") == result.symbol)

        # Load global per-symbol limits
        global_limits = _get_global_position_limits()
        global_max_symbol = global_limits.get(result.symbol, 3)

        self.logger.info(
            f"Position limits: {result.symbol} open={my_open_count}, "
            f"active_decisions={my_recent_decision_count}/{self.config.max_positions_per_symbol}, "
            f"global={global_symbol_count}/{global_max_symbol}"
        )

        # Block if we have an open position for this symbol
        if my_open_count >= self.config.max_positions_per_symbol:
            self.logger.info(f"Automation max open positions reached for {result.symbol} ({my_open_count}), skipping")
            return result

        # Block if we have too many active decisions (prevents rapid re-entry after SL)
        if my_recent_decision_count >= self.config.max_positions_per_symbol:
            self.logger.info(
                f"Automation has {my_recent_decision_count} active decision(s) for {result.symbol} "
                f"(max {self.config.max_positions_per_symbol}), skipping — stale decisions may need cleanup"
            )
            return result

        # Cooldown: don't re-enter too soon after a trade on same symbol
        if latest_trade_time:
            seconds_since = (datetime.now() - latest_trade_time).total_seconds()
            if seconds_since < COOLDOWN_SECONDS:
                remaining = int(COOLDOWN_SECONDS - seconds_since)
                self.logger.info(
                    f"Cooldown active for {result.symbol}: last trade {seconds_since:.0f}s ago, "
                    f"need {COOLDOWN_SECONDS}s ({remaining}s remaining), skipping"
                )
                return result

        if global_symbol_count >= global_max_symbol:
            self.logger.info(f"Global max for {result.symbol} reached ({global_symbol_count}/{global_max_symbol}), skipping")
            return result

        # Check guardrails
        balance = self._get_account_balance()
        can_trade, reason = self.guardrails.check_can_trade(balance)
        if not can_trade:
            self.logger.warning(f"Trading blocked by guardrails: {reason} (balance={balance})")
            return result

        self.logger.info(f"All pre-trade checks passed for {result.symbol} {result.signal}")

        try:
            # Get current price
            price_info = get_mt5_current_price(result.symbol)
            entry_price = price_info.get("ask") if result.signal == "BUY" else price_info.get("bid")
            self.logger.info(f"Current price for {result.symbol}: bid={price_info.get('bid')}, ask={price_info.get('ask')}, using={entry_price}")

            # Use provided SL/TP or calculate defaults
            stop_loss = result.stop_loss
            take_profit = result.take_profit

            if not stop_loss or not take_profit:
                # Calculate ATR-based levels
                atr = get_atr_for_symbol(result.symbol, period=14)
                self.logger.info(f"ATR for {result.symbol}: {atr}, calculating default SL/TP")
                if result.signal == "BUY":
                    stop_loss = stop_loss or (entry_price - 2 * atr)
                    take_profit = take_profit or (entry_price + 3 * atr)
                else:
                    stop_loss = stop_loss or (entry_price + 2 * atr)
                    take_profit = take_profit or (entry_price - 3 * atr)

            # Validate SL/TP are on correct side of entry
            if result.signal == "BUY":
                if stop_loss >= entry_price:
                    self.logger.warning(f"REJECTED: BUY SL ({stop_loss}) must be below entry ({entry_price})")
                    result.executed = False
                    result.rationale += f" | REJECTED: BUY SL {stop_loss} >= entry {entry_price}"
                    return result
                if take_profit <= entry_price:
                    self.logger.warning(f"REJECTED: BUY TP ({take_profit}) must be above entry ({entry_price})")
                    result.executed = False
                    result.rationale += f" | REJECTED: BUY TP {take_profit} <= entry {entry_price}"
                    return result
            else:
                if stop_loss <= entry_price:
                    self.logger.warning(f"REJECTED: SELL SL ({stop_loss}) must be above entry ({entry_price})")
                    result.executed = False
                    result.rationale += f" | REJECTED: SELL SL {stop_loss} <= entry {entry_price}"
                    return result
                if take_profit >= entry_price:
                    self.logger.warning(f"REJECTED: SELL TP ({take_profit}) must be below entry ({entry_price})")
                    result.executed = False
                    result.rationale += f" | REJECTED: SELL TP {take_profit} >= entry {entry_price}"
                    return result

            self.logger.info(
                f"Trade params: {result.symbol} {result.signal} "
                f"entry={entry_price}, sl={stop_loss}, tp={take_profit}"
            )

            # Validate minimum risk:reward ratio
            sl_distance = abs(entry_price - stop_loss)
            tp_distance = abs(take_profit - entry_price)
            min_rr = 0.5  # Minimum acceptable RR ratio
            if sl_distance > 0:
                rr_ratio = tp_distance / sl_distance
                self.logger.info(f"Risk:Reward ratio: {rr_ratio:.2f}:1")
                if rr_ratio < min_rr:
                    self.logger.warning(
                        f"REJECTED: R:R ratio {rr_ratio:.2f}:1 is below minimum {min_rr}:1 "
                        f"(SL dist={sl_distance:.5f}, TP dist={tp_distance:.5f})"
                    )
                    result.executed = False
                    result.rationale += f" | REJECTED: R:R {rr_ratio:.2f}:1 below minimum {min_rr}:1"
                    return result
            else:
                self.logger.warning(f"REJECTED: SL distance is zero (SL={stop_loss}, entry={entry_price})")
                result.executed = False
                result.rationale += " | REJECTED: SL distance is zero"
                return result

            # Calculate position size
            symbol_info = get_mt5_symbol_info(result.symbol)
            lot_size = self.config.default_lot_size
            self.logger.info(f"Lot size: {lot_size}")

            # Execute trade
            self.logger.info(f"Sending order to MT5...")
            trade_result = execute_trade_signal(
                symbol=result.symbol,
                signal=result.signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=lot_size,
                comment=f"QuantAuto {result.pipeline}",
            )
            self.logger.info(f"MT5 trade result: {trade_result}")

            if trade_result.get("success"):
                result.executed = True
                result.execution_ticket = trade_result.get("order_id")
                result.entry_price = trade_result.get("price") or trade_result.get("entry_price") or entry_price
                result.stop_loss = stop_loss
                result.take_profit = take_profit

                # Store decision for tracking
                decision_id = store_decision(
                    symbol=result.symbol,
                    decision_type="OPEN",
                    action=result.signal,
                    rationale=result.rationale[:500],
                    source=self._source,
                    entry_price=result.entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=lot_size,
                    mt5_ticket=result.execution_ticket,
                    confidence=result.confidence,
                    pipeline=result.pipeline,
                    trailing_stop_atr_multiplier=result.trailing_stop_atr_multiplier,
                )

                # Post signal to X/Twitter (non-blocking, failures are swallowed)
                try:
                    from tradingagents.automation.x_signal_poster import post_trade_signal
                    post_trade_signal(
                        symbol=result.symbol,
                        signal=result.signal,
                        entry_price=result.entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        rationale=result.rationale,
                    )
                except Exception as e:
                    self.logger.debug(f"X post skipped: {e}")

                self.logger.info(
                    f"TRADE EXECUTED: {result.symbol} {result.signal} "
                    f"{lot_size} lots @ {result.entry_price}, "
                    f"ticket={result.execution_ticket}, sl={stop_loss}, tp={take_profit}"
                )

                # Log execution event
                if decision_id:
                    add_trade_event(decision_id, "executed", {
                        "ticket": result.execution_ticket,
                        "price": result.entry_price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "volume": lot_size,
                    }, source=self._source)
            else:
                result.execution_error = trade_result.get("error", "Unknown error")
                self.logger.error(
                    f"TRADE FAILED: {result.symbol} {result.signal} - {result.execution_error}. "
                    f"Full result: {trade_result}"
                )

                # Store failed decision for manual retry
                error_detail = f"{result.execution_error} (retcode: {trade_result.get('retcode', 'N/A')})"
                try:
                    store_decision(
                        symbol=result.symbol,
                        decision_type="OPEN",
                        action=result.signal,
                        rationale=result.rationale[:500],
                        source=self._source,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=lot_size,
                        status="failed",
                        execution_error=error_detail,
                        confidence=result.confidence,
                        pipeline=result.pipeline,
                    )
                    self.logger.info(f"Failed trade saved as decision for manual retry")
                except Exception as store_err:
                    self.logger.error(f"Could not store failed decision: {store_err}")

        except Exception as e:
            result.execution_error = str(e)
            import traceback
            self.logger.error(f"TRADE EXECUTION ERROR: {e}\n{traceback.format_exc()}")

        return result

    def _infer_exit_reason(self, exit_price: float, entry: float, sl: float, tp: float,
                           direction: str, deal_comment: str = "") -> str:
        """Infer how a trade exited based on MT5 deal comment and exit price vs SL/TP levels."""
        # MT5 deal comment is most reliable — broker tags "sl", "tp", etc.
        if deal_comment:
            comment_lower = deal_comment.lower().strip()
            # Exact match
            if comment_lower in ("sl", "stop loss"):
                return "sl_hit"
            if comment_lower in ("tp", "take profit"):
                return "tp_hit"
            # Bracket format: "[sl 4988.70]", "[tp 5130.89]"
            if "[sl" in comment_lower:
                return "sl_hit"
            if "[tp" in comment_lower:
                return "tp_hit"
            if "trailing" in comment_lower:
                return "trailing_stop"

        if not exit_price:
            return "unknown"
        tolerance = abs(entry * 0.0005) if entry else 0.5  # 0.05% tolerance
        if tp and abs(exit_price - tp) <= tolerance:
            return "tp_hit"
        if sl and abs(exit_price - sl) <= tolerance:
            return "sl_hit"

        # Check if exit is near entry (breakeven stop was hit)
        if entry and abs(exit_price - entry) <= tolerance:
            return "breakeven_stop"

        return "manual_close"

    def _close_decision_for_ticket(self, ticket: int, exit_price: float, profit: float,
                                     exit_reason: str, notes: str) -> None:
        """Find and close the trade decision linked to an MT5 ticket, then reflect."""
        try:
            decision = find_decision_by_ticket(ticket)
            if not decision or decision.get("status") != "active":
                return

            decision_id = decision["decision_id"]

            # Backfill entry_price if it was stored as 0
            if not decision.get("entry_price"):
                # Try to get entry price from open position or deal history
                positions = get_open_positions(decision.get("symbol"))
                pos_entry = None
                for p in positions:
                    if p.get("ticket") == ticket:
                        pos_entry = p.get("price_open")
                        break
                if pos_entry:
                    decision["entry_price"] = pos_entry
                    # Persist fix to file
                    dec_file = os.path.join(DECISIONS_DIR, f"{decision_id}.json")
                    if os.path.exists(dec_file):
                        with open(dec_file, "r") as f:
                            data = json.load(f)
                        data["entry_price"] = pos_entry
                        with open(dec_file, "w") as f:
                            json.dump(data, f, indent=2, default=str)
                    self.logger.info(f"  Backfilled entry_price={pos_entry} for decision {decision_id}")

            closed = close_decision(
                decision_id,
                exit_price=exit_price,
                outcome_notes=f"{notes}. MT5 profit: {profit:.2f}",
                exit_reason=exit_reason,
            )
            self.logger.info(f"  Decision {decision_id} closed: {exit_reason}, pnl={profit:.2f}")

            # Log close event with outcome summary
            add_trade_event(decision_id, "closed", {
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "mt5_profit": profit,
                "pnl_percent": closed.get("pnl_percent") if closed else None,
                "was_correct": closed.get("was_correct") if closed else None,
                "result": closed.get("structured_outcome", {}).get("result") if closed else None,
            }, source=self._source)

            # Record in guardrails for circuit breaker
            # Skip stale cleanups and unfilled orders — they aren't real losses
            closed_exit_reason = closed.get("exit_reason", "") if closed else exit_reason
            is_real_trade = closed_exit_reason not in ("stale_cleanup", "order_unfilled")
            pnl_pct = closed.get("pnl_percent", 0) if closed else 0
            was_win = closed.get("was_correct", False) if closed else profit > 0
            try:
                if is_real_trade:
                    self.guardrails.record_trade_result(
                        was_win=was_win,
                        pnl_pct=pnl_pct,
                        account_balance=self._get_account_balance(),
                    )
                    self.logger.info(f"  Guardrails updated: {'win' if was_win else 'loss'} {pnl_pct:+.2f}%")
                else:
                    self.logger.info(f"  Guardrails skipped for {closed_exit_reason} (not a real trade)")
            except Exception as e:
                self.logger.warning(f"  Failed to update guardrails: {e}")

            # SMC pattern reflection (works without full agent state)
            self._reflect_on_closed_trade(decision_id)

        except Exception as e:
            self.logger.error(f"  Failed to close decision for #{ticket}: {e}")

    def _reflect_on_closed_trade(self, decision_id: str) -> None:
        """Run SMC pattern reflection on a closed trade decision.

        This creates memories from the trade outcome without requiring
        full multi-agent state. It uses the decision's rationale and
        SMC context to generate pattern-specific lessons.
        """
        try:
            from tradingagents.trade_decisions import load_decision
            closed_decision = load_decision(decision_id)
            if not closed_decision:
                return

            # Check if we have SMC context or a rationale to learn from
            smc_context = closed_decision.get("smc_context", {})
            setup_type = smc_context.get("setup_type") or closed_decision.get("setup_type")
            rationale = closed_decision.get("rationale", "")

            # Try to infer setup_type from rationale if not explicitly set
            if not setup_type and rationale:
                rationale_lower = rationale.lower()
                if "fvg" in rationale_lower or "fair value gap" in rationale_lower:
                    setup_type = "fvg_entry"
                elif "order block" in rationale_lower or "ob " in rationale_lower:
                    setup_type = "ob_entry"
                elif "liquidity" in rationale_lower and "sweep" in rationale_lower:
                    setup_type = "liquidity_sweep"
                elif "breakout" in rationale_lower or "bos" in rationale_lower:
                    setup_type = "bos"
                elif "choch" in rationale_lower or "change of character" in rationale_lower:
                    setup_type = "choch"
                elif "volume" in rationale_lower and ("poc" in rationale_lower or "val" in rationale_lower or "vah" in rationale_lower):
                    setup_type = "volume_profile"
                elif "mean reversion" in rationale_lower:
                    setup_type = "mean_reversion"

                if setup_type:
                    # Persist inferred setup_type
                    closed_decision["setup_type"] = setup_type
                    if "smc_context" not in closed_decision:
                        closed_decision["smc_context"] = {}
                    closed_decision["smc_context"]["setup_type"] = setup_type

            if not setup_type:
                self.logger.debug(f"  No setup_type for {decision_id}, skipping SMC reflection")
                return

            # Use the Reflector directly for SMC pattern memory
            from tradingagents.graph.reflection import Reflector
            from tradingagents.agents.utils.memory import SMCPatternMemory
            from tradingagents.default_config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG.copy()
            smc_memory = SMCPatternMemory(config)

            # We don't need an LLM for SMC pattern reflection —
            # it uses rule-based lesson generation
            reflector = Reflector.__new__(Reflector)
            reflector.reflect_smc_pattern(
                decision=closed_decision,
                smc_memory=smc_memory,
            )
            self.logger.info(f"  SMC pattern memory stored for {decision_id} (setup: {setup_type})")

        except Exception as e:
            self.logger.warning(f"  SMC reflection failed for {decision_id}: {e}")

    async def _manage_positions(self) -> List[PositionManagementResult]:
        """Manage existing positions - trailing stops, breakeven, close signals."""
        results = []
        self.logger.debug("--- Position management cycle start ---")

        # Skip if market was recently detected as closed
        if self._market_closed_until and datetime.now() < self._market_closed_until:
            self.logger.debug(f"Market closed — skipping position management until {self._market_closed_until.strftime('%H:%M')}")
            return results

        try:
            # Verify MT5 is connected before proceeding
            import MetaTrader5 as _mt5
            if not _mt5.terminal_info():
                if not _mt5.initialize():
                    err = _mt5.last_error()
                    self.logger.warning(f"MT5 not connected (skipping position check): {err}")
                    return results

            positions = get_open_positions()

            # Build set of tickets owned by THIS automation (matched by source + symbol)
            # Also cache per-ticket trailing stop multiplier from LLM
            owned_tickets = set()
            ticket_trail_mult = {}  # ticket -> LLM-suggested trailing ATR multiplier
            try:
                active_decisions = list_active_decisions()
                my_source = self._source
                self.logger.info(
                    f"Ownership check: my_source='{my_source}', "
                    f"active_decisions={len(active_decisions)}, "
                    f"sources={[d.get('source') for d in active_decisions]}, "
                    f"tickets={[d.get('mt5_ticket') for d in active_decisions]}"
                )
                for dec in active_decisions:
                    if (dec.get("source") == my_source
                            and dec.get("symbol") in self.config.symbols
                            and dec.get("mt5_ticket")):
                        owned_tickets.add(dec["mt5_ticket"])
                        if dec.get("trailing_stop_atr_multiplier"):
                            ticket_trail_mult[dec["mt5_ticket"]] = dec["trailing_stop_atr_multiplier"]
            except Exception as e:
                self.logger.error(f"Failed to load owned tickets, skipping position management: {e}")
                return results

            managed_positions = [
                p for p in positions
                if p.get("symbol") in self.config.symbols and p.get("ticket") in owned_tickets
            ]
            self.logger.info(
                f"Position check: {len(positions)} total open, "
                f"{len(managed_positions)} owned by '{self._source}' ({', '.join(self.config.symbols)})"
            )

            for position in managed_positions:
                symbol = position.get("symbol")
                ticket = position.get("ticket")

                pos_type = position.get("type", "")
                direction = "BUY" if "BUY" in pos_type.upper() else "SELL"

                entry_price = position.get("price_open", 0)
                current_price = position.get("price_current", 0)
                current_sl = position.get("sl", 0)
                current_tp = position.get("tp", 0)
                profit = position.get("profit", 0)

                # Calculate P&L percentage
                if entry_price > 0 and current_price > 0:
                    if direction == "BUY":
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                result = PositionManagementResult(
                    timestamp=datetime.now(),
                    ticket=ticket,
                    symbol=symbol,
                    action="no_change",
                    old_sl=current_sl,
                    old_tp=current_tp,
                )

                # Structured log tracking for summary
                be_status = "disabled"
                trail_status = "disabled"
                reversal_status = "disabled"

                try:
                    if current_price <= 0:
                        be_status = trail_status = reversal_status = "skip:invalid_price"
                        raise ValueError(f"Invalid current_price={current_price}")

                    # Get ATR for stop calculations
                    atr = get_atr_for_symbol(symbol, period=14)

                    # --- Breakeven Stop ---
                    if self.config.enable_breakeven_stop:
                        breakeven_threshold_distance = self.config.move_to_breakeven_atr_mult * atr
                        if direction == "BUY":
                            profit_distance = current_price - entry_price
                        else:
                            profit_distance = entry_price - current_price

                        breakeven_eligible = profit_distance >= breakeven_threshold_distance and current_sl != 0

                        if breakeven_eligible:
                            breakeven_sl, is_eligible = self.stop_loss_manager.calculate_breakeven_stop(
                                entry_price=entry_price,
                                current_price=current_price,
                                direction=direction,
                                atr=atr,
                            )
                            if is_eligible and breakeven_sl:
                                is_better = (
                                    (direction == "BUY" and breakeven_sl > current_sl) or
                                    (direction == "SELL" and breakeven_sl < current_sl)
                                )
                                if is_better and self.config.auto_execute:
                                    modify_result = modify_position(ticket, sl=breakeven_sl)
                                    if modify_result.get("success"):
                                        result.action = "adjusted_sl"
                                        result.new_sl = breakeven_sl
                                        be_status = f"SET {current_sl:.2f}->{breakeven_sl:.2f}"
                                        dec = find_decision_by_ticket(ticket)
                                        if dec:
                                            add_trade_event(dec["decision_id"], "breakeven_set", {
                                                "old_sl": current_sl, "new_sl": breakeven_sl,
                                                "price": current_price, "pnl_pct": round(pnl_pct, 3),
                                            }, source=self._source)
                                    else:
                                        be_error = modify_result.get("error", "unknown")
                                        be_status = f"FAILED: {be_error}"
                                        if modify_result.get("market_closed") or "Market closed" in be_error:
                                            self._market_closed_until = datetime.now() + timedelta(minutes=30)
                                            self.logger.info(f"Market closed — suppressing modifications until {self._market_closed_until.strftime('%H:%M')}")
                                elif not self.config.auto_execute:
                                    be_status = "eligible:auto_execute=off"
                                else:
                                    be_status = "not_better"
                            else:
                                be_status = "not_eligible"
                        else:
                            be_status = f"not_reached({profit_distance:.2f}/{breakeven_threshold_distance:.2f})"

                    # --- Trailing Stop ---
                    if self.config.enable_trailing_stop and pnl_pct > 0:
                        trail_mult = ticket_trail_mult.get(ticket, self.config.trailing_stop_atr_multiplier)
                        trail_distance = trail_mult * atr
                        if direction == "BUY":
                            candidate_sl = round(current_price - trail_distance, 5)
                            should_trail = candidate_sl > current_sl
                        else:
                            candidate_sl = round(current_price + trail_distance, 5)
                            should_trail = candidate_sl < current_sl

                        if should_trail and self.config.auto_execute:
                            modify_result = modify_position(ticket, sl=candidate_sl)
                            if modify_result.get("success"):
                                result.action = "adjusted_sl"
                                result.new_sl = candidate_sl
                                trail_status = f"MOVED {current_sl:.2f}->{candidate_sl:.2f} ({trail_mult}x)"
                                dec = find_decision_by_ticket(ticket)
                                if dec:
                                    add_trade_event(dec["decision_id"], "trailing_stop", {
                                        "old_sl": current_sl, "new_sl": candidate_sl,
                                        "price": current_price, "pnl_pct": round(pnl_pct, 3),
                                        "atr_mult": trail_mult,
                                    }, source=self._source)
                            else:
                                error_msg = modify_result.get("error", "unknown")
                                trail_status = f"FAILED: {error_msg}"
                                if modify_result.get("market_closed") or "Market closed" in error_msg:
                                    self._market_closed_until = datetime.now() + timedelta(minutes=30)
                                    self.logger.info(f"Market closed — suppressing modifications until {self._market_closed_until.strftime('%H:%M')}")
                        elif should_trail and not self.config.auto_execute:
                            trail_status = "eligible:auto_execute=off"
                        else:
                            trail_status = "no_move"
                    elif self.config.enable_trailing_stop:
                        trail_status = f"no_profit(pnl={pnl_pct:.2f}%)"

                    # --- Reversal Signal Close ---
                    if self.config.enable_reversal_close:
                        if self.config.pipeline in (PipelineType.SMC_QUANT_BASIC, PipelineType.SMC_QUANT, PipelineType.BREAKOUT_QUANT, PipelineType.RANGE_QUANT, PipelineType.RULE_BASED, PipelineType.SMC_MTF, PipelineType.XGBOOST, PipelineType.XGBOOST_ENSEMBLE):
                            if self.config.pipeline == PipelineType.SMC_QUANT:
                                analysis = await self._run_smc_quant_analysis(symbol)
                            elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
                                analysis = await self._run_breakout_quant_analysis(symbol)
                            elif self.config.pipeline == PipelineType.RANGE_QUANT:
                                analysis = await self._run_range_quant_analysis(symbol)
                            elif self.config.pipeline == PipelineType.RULE_BASED:
                                analysis = await self._run_rule_based_analysis(symbol)
                            elif self.config.pipeline == PipelineType.SMC_MTF:
                                analysis = await self._run_smc_mtf_analysis(symbol)
                            elif self.config.pipeline in (PipelineType.XGBOOST, PipelineType.XGBOOST_ENSEMBLE):
                                analysis = await self._run_xgboost_analysis(symbol)
                            else:
                                analysis = await self._run_quant_analysis(symbol)

                            is_reversal = analysis.signal == "CLOSE" or (
                                analysis.signal != "HOLD" and
                                analysis.signal != direction and
                                analysis.confidence > 0.7
                            )

                            if is_reversal:
                                if self.config.auto_execute:
                                    close_result = close_position(ticket)
                                    if close_result.get("success"):
                                        result.action = "closed"
                                        result.close_reason = f"Reversal signal: {analysis.signal}"
                                        result.pnl = profit
                                        reversal_status = f"CLOSED sig={analysis.signal} conf={analysis.confidence:.0%}"
                                        dec = find_decision_by_ticket(ticket)
                                        if dec:
                                            add_trade_event(dec["decision_id"], "reversal_signal", {
                                                "new_signal": analysis.signal, "position_direction": direction,
                                                "confidence": round(analysis.confidence, 2),
                                                "price": current_price, "profit": profit,
                                            }, source=self._source)
                                        deal_info = get_closed_deal_by_ticket(ticket, days_back=7)
                                        exit_px = deal_info["price"] if deal_info else current_price
                                        self._close_decision_for_ticket(
                                            ticket, exit_px, profit, "reversal_signal",
                                            f"Reversal signal: {analysis.signal}"
                                        )
                                    else:
                                        reversal_status = f"CLOSE_FAILED: {close_result.get('error', 'unknown')}"
                                else:
                                    reversal_status = f"detected:auto_execute=off (sig={analysis.signal})"
                            else:
                                reversal_status = f"no_signal(sig={analysis.signal} conf={analysis.confidence:.0%})"
                        else:
                            reversal_status = "unsupported_pipeline"

                except ValueError:
                    pass  # Already set status above
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error managing position #{ticket}: {e}\n{traceback.format_exc()}")
                    be_status = trail_status = reversal_status = f"ERROR: {e}"

                # Structured summary log
                self.logger.info(
                    f"POS_MGMT #{ticket} {symbol} {direction}: "
                    f"entry={entry_price} cur={current_price} pnl={pnl_pct:+.2f}% "
                    f"| BE: {be_status} | TRAIL: {trail_status} | REV: {reversal_status}"
                )

                results.append(result)

            # Detect positions that disappeared (SL/TP hit externally)
            # Include both filled positions AND pending orders — limit orders
            # may not fill immediately, so the ticket exists as a pending order
            # before it becomes a position.
            open_tickets = {p.get("ticket") for p in managed_positions}
            try:
                import MetaTrader5 as mt5
                pending_orders = mt5.orders_get()
                if pending_orders:
                    for order in pending_orders:
                        open_tickets.add(order.ticket)
            except Exception:
                pass  # If orders_get fails, proceed with positions only
            try:
                my_source = self._source
                active_decisions = list_active_decisions()
                # Collect all sources from active decisions to identify orphans
                all_sources = {d.get("source") for d in active_decisions if d.get("source")}
                for dec in active_decisions:
                    dec_ticket = dec.get("mt5_ticket")
                    dec_symbol = dec.get("symbol", "")
                    dec_source = dec.get("source", "")
                    # Skip decisions for symbols this instance doesn't handle
                    if dec_symbol not in self.config.symbols:
                        continue
                    # Process decisions owned by this instance directly
                    # Also process orphan decisions (e.g. web_ui, manual) — sources
                    # that don't start with any symbol prefix are orphans
                    is_mine = dec_source == my_source
                    is_orphan = not any(
                        dec_source.startswith(sym.lower()) for sym in self.config.symbols
                    ) and dec_source != my_source
                    if not is_mine and not is_orphan:
                        continue
                    if not dec_ticket or dec_ticket in open_tickets:
                        continue
                    # This decision's position is gone - close it
                    deal_info = get_closed_deal_by_ticket(dec_ticket, days_back=14)
                    if deal_info:
                        exit_price = deal_info["price"]
                        mt5_profit = deal_info.get("profit", 0)
                        exit_reason = self._infer_exit_reason(
                            exit_price, dec.get("entry_price", 0),
                            dec.get("stop_loss"), dec.get("take_profit"),
                            dec.get("action", "BUY"),
                            deal_comment=deal_info.get("comment", ""),
                        )
                        # Log the external close with MT5 deal details
                        add_trade_event(dec["decision_id"], "external_close", {
                            "exit_price": exit_price,
                            "mt5_profit": mt5_profit,
                            "inferred_reason": exit_reason,
                            "deal_type": deal_info.get("type"),
                            "deal_comment": deal_info.get("comment", ""),
                            "entry_price": dec.get("entry_price"),
                            "sl": dec.get("stop_loss"),
                            "tp": dec.get("take_profit"),
                        }, source=self._source)
                        self._close_decision_for_ticket(
                            dec_ticket, exit_price, mt5_profit, exit_reason,
                            f"Position closed externally ({exit_reason})"
                        )
                        results.append(PositionManagementResult(
                            timestamp=datetime.now(), ticket=dec_ticket, symbol=dec_symbol,
                            action="closed", close_reason=exit_reason, pnl=mt5_profit,
                        ))
                    else:
                        # Track consecutive checks where ticket is gone but no deal found
                        self._missing_ticket_counts[dec_ticket] = self._missing_ticket_counts.get(dec_ticket, 0) + 1
                        miss_count = self._missing_ticket_counts[dec_ticket]
                        self.logger.info(
                            f"  Decision {dec['decision_id']} ticket #{dec_ticket} gone but no deal history found "
                            f"(check {miss_count}/5)"
                        )
                        if miss_count >= 5:
                            # Position confirmed gone — determine if order never filled
                            # or if a filled position disappeared
                            entry_price = dec.get("entry_price", 0)
                            # Check events: if we only have an "executed" event with deal_id=0
                            # and no POS_MGMT ever ran, the order was placed but never filled
                            events = dec.get("events", [])
                            executed_evt = next((e for e in events if e.get("type") == "executed"), None)
                            had_fill = any(e.get("type") in ("position_update", "external_close") for e in events)
                            # deal_id=0 from MT5 means limit order placed, not filled
                            order_never_filled = (
                                executed_evt
                                and not had_fill
                                and executed_evt.get("deal_id", -1) == 0
                            )
                            if not order_never_filled:
                                # Also check: if no deal history and pnl=0, likely unfilled
                                order_never_filled = not had_fill

                            if order_never_filled:
                                exit_reason = "order_unfilled"
                                event_type = "order_unfilled"
                                note = f"Limit order #{dec_ticket} was placed but never filled — expired or cancelled (insufficient margin, price moved away, etc.)"
                                close_reason = "order_unfilled"
                            else:
                                exit_reason = "stale_cleanup"
                                event_type = "stale_cleanup"
                                note = f"Ticket #{dec_ticket} missing from MT5 for {miss_count} checks, no deal history found"
                                close_reason = "stale_cleanup"

                            self.logger.warning(
                                f"  Closing {'unfilled order' if order_never_filled else 'stale'} decision "
                                f"{dec['decision_id']} ticket #{dec_ticket} — "
                                f"gone for {miss_count} consecutive checks, no MT5 deal found"
                            )
                            add_trade_event(dec["decision_id"], event_type, {
                                "exit_price": entry_price,
                                "mt5_profit": 0,
                                "inferred_reason": exit_reason,
                                "note": note,
                                "entry_price": entry_price,
                                "sl": dec.get("stop_loss"),
                                "tp": dec.get("take_profit"),
                            }, source=self._source)
                            self._close_decision_for_ticket(
                                dec_ticket, entry_price, 0.0, exit_reason,
                                note
                            )
                            del self._missing_ticket_counts[dec_ticket]
                            results.append(PositionManagementResult(
                                timestamp=datetime.now(), ticket=dec_ticket, symbol=dec_symbol,
                                action="closed", close_reason=close_reason, pnl=0.0,
                            ))
                # Clear missing counts for tickets that reappeared (e.g. race condition resolved)
                for ticket in list(self._missing_ticket_counts.keys()):
                    if ticket in open_tickets:
                        del self._missing_ticket_counts[ticket]
            except Exception as e:
                self.logger.error(f"Error detecting closed positions: {e}")

        except ConnectionError as e:
            self.logger.warning(f"MT5 connection lost during position management: {e}")
        except Exception as e:
            import traceback
            self.logger.error(f"Position management error: {e}\n{traceback.format_exc()}")

        self.logger.debug(f"--- Position management cycle end ({len(results)} positions checked) ---")
        return results

    async def _analysis_loop(self):
        """Main analysis loop."""
        await asyncio.sleep(0)  # Yield to let other coroutines start
        while self._running:
            try:
                now = datetime.now()

                for symbol in self.config.symbols:
                    # Check if enough time has passed since last analysis
                    last_time = self._last_analysis_time.get(symbol)
                    if last_time:
                        elapsed = (now - last_time).total_seconds()
                        if elapsed < self.config.analysis_interval_seconds:
                            continue

                    # Check if market is open before running analysis
                    try:
                        market_status = is_market_open(symbol)
                        if not market_status["open"]:
                            self.logger.info(
                                f"Market closed for {symbol}: {market_status['reason']}, skipping analysis"
                            )
                            continue
                    except Exception as mkt_err:
                        self.logger.warning(f"Could not check market status for {symbol}: {mkt_err}")

                    self.logger.info(f"Running {self.config.pipeline.value} analysis for {symbol}...")

                    # Run analysis based on pipeline
                    if self.config.pipeline == PipelineType.SMC_QUANT_BASIC:
                        result = await self._run_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.SMC_QUANT:
                        result = await self._run_smc_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.SMC_MTF:
                        result = await self._run_smc_mtf_analysis(symbol)
                    elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
                        result = await self._run_breakout_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.RANGE_QUANT:
                        result = await self._run_range_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
                        result = await self._run_vp_analysis(symbol)
                    elif self.config.pipeline == PipelineType.RULE_BASED:
                        result = await self._run_rule_based_analysis(symbol)
                    elif self.config.pipeline in (PipelineType.XGBOOST, PipelineType.XGBOOST_ENSEMBLE):
                        result = await self._run_xgboost_analysis(symbol)
                    else:
                        result = await self._run_multi_agent_analysis(symbol)

                    self.logger.info(
                        f"{symbol}: {result.signal} (confidence: {result.confidence:.2f})"
                    )

                    # Execute trade if conditions met
                    if result.signal in ["BUY", "SELL"]:
                        result = await self._execute_trade(result)

                    # Store result
                    self._analysis_results.append(result)
                    if len(self._analysis_results) > 100:
                        self._analysis_results = self._analysis_results[-100:]

                    # Update last analysis time
                    self._last_analysis_time[symbol] = now

                await self._save_state()

            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

            # Wait for next cycle or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=min(60, self.config.analysis_interval_seconds),
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

    async def _position_loop(self):
        """Position management loop."""
        await asyncio.sleep(0)  # Yield to let other coroutines start
        self.logger.info("Position management loop started")
        while self._running:
            try:
                results = await self._manage_positions()

                # Store results
                self._position_results.extend(results)
                if len(self._position_results) > 100:
                    self._position_results = self._position_results[-100:]

                self._last_position_check = datetime.now()

                # Log summary
                actions = [r for r in results if r.action != "no_change"]
                if actions:
                    for a in actions:
                        self.logger.info(f"Position action: #{a.ticket} {a.symbol} -> {a.action}")
                else:
                    self.logger.debug(f"Position check complete: {len(results)} positions, no changes")

            except Exception as e:
                import traceback
                self.logger.error(f"Position loop error: {e}\n{traceback.format_exc()}")

            # Wait for next cycle or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.position_check_interval_seconds,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop

    async def _assumption_review_loop(self):
        """Daily assumption review loop — checks open positions against current SMC structure."""
        self.logger.info(
            f"Assumption review loop started "
            f"(interval: {self.config.assumption_review_interval_seconds}s, "
            f"auto_apply: {self.config.assumption_review_auto_apply})"
        )
        while self._running:
            try:
                now = datetime.now()

                # Check if enough time has passed since last review
                should_run = True
                if self._last_assumption_review:
                    elapsed = (now - self._last_assumption_review).total_seconds()
                    if elapsed < self.config.assumption_review_interval_seconds:
                        should_run = False
                        self.logger.debug(f"Assumption review: {elapsed:.0f}s since last, need {self.config.assumption_review_interval_seconds}s")

                self.logger.info(f"Assumption review check: should_run={should_run}, last_review={self._last_assumption_review}")

                if should_run:
                    self.logger.info("=" * 50)
                    self.logger.info("RUNNING POSITION ASSUMPTION REVIEW")
                    self.logger.info(f"  source_filter='{self._source}', symbols={self.config.symbols}")
                    self.logger.info("=" * 50)

                    from tradingagents.automation.position_assumption_review import (
                        review_all_positions,
                        format_review_summary,
                    )

                    import time as _time
                    _t0 = _time.time()
                    # Run synchronous review in thread to avoid blocking the event loop
                    reports = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: review_all_positions(
                            source_filter=self._source,
                            symbols=self.config.symbols,
                            timeframe=self.config.timeframe,
                            auto_apply=self.config.assumption_review_auto_apply,
                            use_llm=True,
                        ),
                    )
                    self.logger.info(f"Assumption review completed: {len(reports)} reports in {_time.time()-_t0:.1f}s")

                    self._last_assumption_review = now

                    # Store summary for status endpoint
                    self._assumption_review_results = [
                        {
                            "decision_id": r.decision_id,
                            "symbol": r.symbol,
                            "direction": r.direction,
                            "ticket": r.ticket,
                            "pnl_pct": r.pnl_pct,
                            "recommended_action": r.recommended_action,
                            "suggested_sl": r.suggested_sl,
                            "suggested_tp": r.suggested_tp,
                            "findings_count": len(r.findings),
                            "has_critical": r.has_critical,
                            "findings": [
                                {
                                    "category": f.category,
                                    "severity": f.severity,
                                    "message": f.message,
                                    "suggested_action": f.suggested_action,
                                    "suggested_value": f.suggested_value,
                                }
                                for f in r.findings
                            ],
                            "llm_assessment": r.llm_assessment,
                            "error": r.error,
                            "timestamp": r.review_timestamp.isoformat(),
                        }
                        for r in reports
                    ]

                    summary = format_review_summary(reports)
                    self.logger.info(summary)

                    await self._save_state()

            except Exception as e:
                import traceback
                self.logger.error(f"Assumption review error: {e}\n{traceback.format_exc()}")

            # Wait for next cycle or shutdown — check every 5 minutes
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=300,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _trade_queue_loop(self):
        """
        Process remote trade commands from the Postgres queue.

        Allows the web UI to trigger trades that this automation executes.
        """
        from tradingagents.storage.trade_queue import get_trade_queue

        self.logger.info("Trade queue loop started")
        queue = get_trade_queue()

        while self._running:
            try:
                # Get pending commands
                commands = await queue.get_pending_commands(limit=5)

                for cmd in commands:
                    # Claim command to prevent double-execution
                    claimed = await queue.claim_command(cmd.command_id)
                    if not claimed:
                        continue

                    self.logger.info(f"[QUEUE] Processing: {cmd.command_type} {cmd.symbol}")

                    try:
                        result = await self._execute_queue_command(cmd)
                        await queue.complete_command(cmd.command_id, result)
                        self.logger.info(f"[QUEUE] Completed: {cmd.command_id}")

                    except Exception as e:
                        await queue.fail_command(cmd.command_id, str(e))
                        self.logger.error(f"[QUEUE] Failed: {cmd.command_id} - {e}")

            except Exception as e:
                self.logger.error(f"Trade queue error: {e!r}")
                # Reset pool on connection errors to force reconnect
                if "closed" in str(e).lower() or "release" in str(e).lower() or "connect" in str(e).lower():
                    try:
                        queue = get_trade_queue()
                        await queue._reset_pool()
                        self.logger.info("Trade queue pool reset after connection error")
                    except Exception:
                        pass

            # Wait for next poll or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.trade_queue_poll_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _execute_queue_command(self, cmd) -> dict:
        """Execute a queued trade command."""
        payload = cmd.payload

        if cmd.command_type == "execute":
            # Execute trade
            symbol = cmd.symbol
            direction = payload.get("direction", "").upper()
            volume = payload.get("volume", self.config.default_lot_size)
            stop_loss = payload.get("stop_loss")
            take_profit = payload.get("take_profit")
            decision_id = payload.get("decision_id")

            # Get current price
            price_info = get_mt5_current_price(symbol)
            if not price_info:
                raise ValueError(f"Could not get price for {symbol}")
            entry_price = price_info["ask"] if direction == "BUY" else price_info["bid"]

            # Execute
            result = execute_trade_signal(
                symbol=symbol,
                signal=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
            )

            if not result.get("success"):
                raise ValueError(result.get("error", "Trade failed"))

            ticket = result.get("ticket")

            # Store decision
            if not decision_id:
                decision_id = store_decision(
                    symbol=symbol,
                    decision_type="OPEN",
                    action=direction,
                    rationale=f"Remote queue: {cmd.command_id}",
                    source=cmd.source,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    mt5_ticket=ticket,
                )

            return {
                "status": "success",
                "ticket": ticket,
                "decision_id": decision_id,
                "entry_price": entry_price,
            }

        elif cmd.command_type == "modify_sl":
            ticket = payload.get("ticket")
            new_sl = payload.get("new_sl")
            result = modify_position(ticket, sl=new_sl)
            return {"status": "success" if result.get("success") else "error", **result}

        elif cmd.command_type == "modify_tp":
            ticket = payload.get("ticket")
            new_tp = payload.get("new_tp")
            result = modify_position(ticket, tp=new_tp)
            return {"status": "success" if result.get("success") else "error", **result}

        elif cmd.command_type == "close":
            ticket = payload.get("ticket")
            volume = payload.get("volume")
            result = close_position(ticket, volume=volume)
            return {"status": "success" if result.get("success") else "error", **result}

        else:
            raise ValueError(f"Unknown command: {cmd.command_type}")

    async def _update_remote_status(self):
        """Update automation status in Postgres for remote monitoring."""
        try:
            from tradingagents.storage.automation_control import get_automation_control

            control = get_automation_control()
            active_decisions = list_active_decisions()
            active_for_symbols = [
                d for d in active_decisions
                if d.get("symbol") in self.config.symbols
            ]

            await control.update_status(
                instance_name=self._source,
                status=self._status.value,
                pipeline=self.config.pipeline.value,
                symbols=self.config.symbols,
                auto_execute=self.config.auto_execute,
                last_analysis=self._last_analysis_time.get(self.config.symbols[0]) if self.config.symbols else None,
                active_positions=len(active_for_symbols),
                error_message=self._error_message,
                config=self.config.to_dict(),
            )
        except Exception as e:
            self.logger.warning(f"Failed to update remote status: {e}")

    async def _control_loop(self):
        """
        Poll for remote control commands (start/stop/pause/config updates).

        Allows the web UI to control this automation instance.
        """
        from tradingagents.storage.automation_control import get_automation_control

        self.logger.info("Control loop started")
        control = get_automation_control()

        while self._running:
            try:
                # Get pending commands for this instance
                commands = await control.get_pending_commands(self._source)

                for cmd in commands:
                    self.logger.info(f"[CONTROL] Received: {cmd.action} from {cmd.source}")

                    try:
                        await self._apply_control_command(cmd)
                        await control.mark_applied(cmd.command_id)
                        self.logger.info(f"[CONTROL] Applied: {cmd.action}")

                    except Exception as e:
                        await control.mark_failed(cmd.command_id, str(e))
                        self.logger.error(f"[CONTROL] Failed: {cmd.action} - {e}")

                # Update status periodically
                await self._update_remote_status()

            except Exception as e:
                self.logger.error(f"Control loop error: {e!r}")
                # Reset pool on connection errors to force reconnect
                if "closed" in str(e).lower() or "release" in str(e).lower() or "connect" in str(e).lower():
                    try:
                        await control._reset_pool()
                        self.logger.info("Control pool reset after connection error")
                    except Exception:
                        pass

            # Wait for next poll
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.control_poll_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

        # Final status update on shutdown
        await self._update_remote_status()

    async def _apply_control_command(self, cmd):
        """Apply a control command."""
        if cmd.action == "stop":
            self.stop()

        elif cmd.action == "pause":
            self.pause()

        elif cmd.action == "resume":
            self.resume()

        elif cmd.action == "update_config":
            # Update configuration
            await self.update_config(cmd.payload)
            self.logger.info(f"[CONTROL] Config updated: {cmd.payload}")

        elif cmd.action == "restart":
            # Stop and let external supervisor restart
            self.logger.info("[CONTROL] Restart requested - stopping...")
            self.stop()

        elif cmd.action == "start":
            # Already running, just acknowledge
            self.logger.info("[CONTROL] Start command received - already running")

        else:
            raise ValueError(f"Unknown control action: {cmd.action}")

    async def start(self):
        """Start the automation."""
        if self._running:
            self.logger.warning("Automation already running")
            return

        # Check MT5 connection
        mt5_status = check_mt5_autotrading()
        if not mt5_status.get("connected"):
            self._status = AutomationStatus.ERROR
            self._error_message = "MT5 not connected"
            raise RuntimeError("MT5 not connected")

        # Try loading state from Postgres (may be newer than local file)
        await self._load_state_from_db()

        self._running = True
        self._status = AutomationStatus.RUNNING
        self._shutdown_event.clear()
        self._error_message = None

        self.logger.info("=" * 50)
        self.logger.info("QUANT AUTOMATION STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Pipeline: {self.config.pipeline.value}")
        self.logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        self.logger.info(f"Analysis interval: {self.config.analysis_interval_seconds}s")
        self.logger.info(f"Assumption review interval: {self.config.assumption_review_interval_seconds}s")
        self.logger.info(f"Auto-execute: {self.config.auto_execute}")
        self.logger.info(f"Trade queue: {'enabled' if self.config.enable_trade_queue else 'disabled'}")
        self.logger.info(f"Remote control: {'enabled' if self.config.enable_remote_control else 'disabled'}")
        self.logger.info("=" * 50)

        # Report initial status to Postgres
        if self.config.enable_remote_control:
            await self._update_remote_status()

        # Build list of loops to run
        loops = [
            self._analysis_loop(),
            self._position_loop(),
            self._assumption_review_loop(),
        ]

        # Add trade queue loop if enabled
        if self.config.enable_trade_queue:
            loops.append(self._trade_queue_loop())

        # Add control loop if enabled
        if self.config.enable_remote_control:
            loops.append(self._control_loop())

        # Run all loops concurrently
        try:
            await asyncio.gather(*loops)
        except Exception as e:
            self._status = AutomationStatus.ERROR
            self._error_message = str(e)
            self.logger.error(f"Automation error: {e}")
            raise
        finally:
            self._running = False
            self._status = AutomationStatus.STOPPED
            self.logger.info("Quant automation stopped")

    def stop(self):
        """Stop the automation."""
        self.logger.info("Stopping quant automation...")
        self._running = False
        self._shutdown_event.set()
        # Schedule HTTP session and DB pool cleanup
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._close_http_session())
            if self._pg_pool:
                loop.create_task(self._pg_pool.close())
                self._pg_pool = None
        except RuntimeError:
            pass  # No running loop, resources will be garbage collected

    def pause(self):
        """Pause the automation (stop new trades but continue monitoring)."""
        self.config.auto_execute = False
        self._status = AutomationStatus.PAUSED
        self.logger.info("Automation paused - monitoring only")

    def resume(self):
        """Resume automation with auto-execution."""
        self.config.auto_execute = True
        self._status = AutomationStatus.RUNNING
        self.logger.info("Automation resumed - auto-execute enabled")

    async def update_config(self, updates: Dict[str, Any]):
        """Update configuration dynamically."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                if key == "pipeline" and isinstance(value, str):
                    value = PipelineType(value)
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {old_value} -> {value}")
            else:
                self.logger.warning(f"Config update ignored: unknown key '{key}'")
        await self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        positions = []
        try:
            all_positions = get_open_positions()
            positions = [p for p in all_positions if p.get("symbol") in self.config.symbols]
        except Exception as e:
            self.logger.error(f"Failed to get positions for status: {e}")

        return {
            "status": self._status.value,
            "running": self._running,
            "error": self._error_message,
            "config": self.config.to_dict(),
            "positions": {
                "managed": len(positions),
                "max_per_symbol": self.config.max_positions_per_symbol,
            },
            "last_analysis": {
                symbol: t.isoformat() if t else None
                for symbol, t in self._last_analysis_time.items()
            },
            "last_position_check": self._last_position_check.isoformat() if self._last_position_check else None,
            "last_assumption_review": self._last_assumption_review.isoformat() if self._last_assumption_review else None,
            "recent_results": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "symbol": r.symbol,
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "executed": r.executed,
                }
                for r in self._analysis_results[-10:]
            ],
            "assumption_review": self._assumption_review_results,
            "guardrails": self.guardrails.get_status(),
        }

    async def run_single_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run a single analysis cycle for testing."""
        if self.config.pipeline == PipelineType.SMC_QUANT_BASIC:
            return await self._run_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.SMC_QUANT:
            return await self._run_smc_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.SMC_MTF:
            return await self._run_smc_mtf_analysis(symbol)
        elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
            return await self._run_breakout_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.RANGE_QUANT:
            return await self._run_range_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
            return await self._run_vp_analysis(symbol)
        elif self.config.pipeline == PipelineType.RULE_BASED:
            return await self._run_rule_based_analysis(symbol)
        elif self.config.pipeline in (PipelineType.XGBOOST, PipelineType.XGBOOST_ENSEMBLE):
            return await self._run_xgboost_analysis(symbol)
        else:
            return await self._run_multi_agent_analysis(symbol)

    async def _run_xgboost_analysis(self, symbol: str) -> "AnalysisCycleResult":
        """Run XGBoost strategy analysis — local inference, no LLM call."""
        import time as _time
        start = _time.time()

        try:
            from tradingagents.automation.auto_tuner import load_mt5_data, _compute_atr
            from tradingagents.xgb_quant.predictor import LivePredictor
            from tradingagents.xgb_quant.config import REGIME_SUITABILITY

            df = load_mt5_data(symbol, self.config.timeframe, bars=500)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            close = df["close"].values.astype(float)
            atr = _compute_atr(high, low, close)
            current_atr = float(atr[-1]) if not np.isnan(atr[-1]) else 1.0
            current_price = float(close[-1])

            predictor = LivePredictor()

            if self.config.pipeline == PipelineType.XGBOOST_ENSEMBLE:
                available = predictor.get_available_models(symbol, self.config.timeframe)
                if len(available) < 2:
                    return AnalysisCycleResult(
                        timestamp=datetime.now(), symbol=symbol,
                        pipeline=self.config.pipeline.value,
                        signal="HOLD", confidence=0.0,
                        rationale=f"Only {len(available)} models available, need 2+",
                        duration_seconds=_time.time() - start,
                    )
                signal = predictor.predict_ensemble(
                    available, symbol, self.config.timeframe,
                    df, current_price, current_atr,
                )
            else:
                # Single strategy — use strategy selector to pick best
                from tradingagents.xgb_quant.strategy_selector import StrategySelector
                selector = StrategySelector()
                selection = selector.select(symbol)
                strategy_name = selection.recommended_strategy

                # Use the timeframe that performed best in training
                timeframe = selection.recommended_timeframe or self.config.timeframe
                if timeframe != self.config.timeframe:
                    self.logger.info(
                        f"XGBoost: using best timeframe {timeframe} "
                        f"(trained) instead of {self.config.timeframe} (configured) "
                        f"for {strategy_name} on {symbol}"
                    )
                    df = load_mt5_data(symbol, timeframe, bars=500)
                    high = df["high"].values.astype(float)
                    low = df["low"].values.astype(float)
                    close = df["close"].values.astype(float)
                    atr = _compute_atr(high, low, close)
                    current_atr = float(atr[-1]) if not np.isnan(atr[-1]) else 1.0
                    current_price = float(close[-1])

                signal = predictor.predict_single(
                    strategy_name, symbol, timeframe,
                    df, current_price, current_atr,
                )

            return AnalysisCycleResult(
                timestamp=datetime.now(), symbol=symbol,
                pipeline=self.config.pipeline.value,
                signal=signal.direction,
                confidence=signal.confidence,
                entry_price=signal.entry if signal.entry else current_price,
                stop_loss=signal.stop_loss if signal.stop_loss else None,
                take_profit=signal.take_profit if signal.take_profit else None,
                rationale=signal.rationale,
                duration_seconds=_time.time() - start,
            )

        except Exception as e:
            self.logger.error(f"XGBoost analysis failed for {symbol}: {e}")
            return AnalysisCycleResult(
                timestamp=datetime.now(), symbol=symbol,
                pipeline=self.config.pipeline.value,
                signal="HOLD", confidence=0.0,
                rationale=f"XGBoost error: {e}",
                duration_seconds=_time.time() - start,
            )


# Global instance for API access
_automation_instance: Optional[QuantAutomation] = None
_automation_task: Optional[asyncio.Task] = None


def get_automation_instance() -> Optional[QuantAutomation]:
    """Get the global automation instance."""
    return _automation_instance


async def start_automation(config: Optional[Dict[str, Any]] = None) -> QuantAutomation:
    """Start the global automation instance."""
    global _automation_instance, _automation_task

    logger = logging.getLogger("QuantAutomation")

    if _automation_instance and _automation_instance._running:
        logger.warning("start_automation called but automation already running")
        raise RuntimeError("Automation already running")

    # Create config from dict if provided
    if config:
        logger.info(f"Starting automation with config: {config}")
        auto_config = QuantAutomationConfig.from_dict(config)
    else:
        logger.info("Starting automation with default config")
        auto_config = QuantAutomationConfig()

    _automation_instance = QuantAutomation(auto_config)

    # Start in background task
    _automation_task = asyncio.create_task(_automation_instance.start())

    # Give it a moment to start
    await asyncio.sleep(0.5)

    logger.info(f"Automation started, status={_automation_instance._status.value}")
    return _automation_instance


async def stop_automation():
    """Stop the global automation instance."""
    global _automation_instance, _automation_task

    logger = logging.getLogger("QuantAutomation")
    logger.info("stop_automation called")

    if _automation_instance:
        _automation_instance.stop()
    else:
        logger.warning("stop_automation called but no instance exists")

    if _automation_task:
        try:
            await asyncio.wait_for(_automation_task, timeout=5.0)
            logger.info("Automation task completed cleanly")
        except asyncio.TimeoutError:
            logger.warning("Automation task did not stop in 5s, cancelling")
            _automation_task.cancel()
        _automation_task = None
