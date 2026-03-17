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
from datetime import datetime
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
    DECISIONS_DIR,
)

from tradingagents.dataflows.mt5_data import get_closed_deal_by_ticket


# Global symbol limits file (shared across all automations)
_SYMBOL_LIMITS_FILE = Path(__file__).parent.parent.parent / "automation_symbol_limits.json"


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
    BREAKOUT_QUANT = "breakout_quant"
    RANGE_QUANT = "range_quant"
    VOLUME_PROFILE = "volume_profile"
    RULE_BASED = "rule_based"
    MULTI_AGENT = "multi_agent"


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
    move_to_breakeven_atr_mult: float = 1.5  # Move SL to breakeven after profit >= 1.5x ATR

    # Risk settings
    max_risk_per_trade_pct: float = 1.0  # 1% of account per trade
    default_lot_size: float = 0.01
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 3

    # Position assumption review settings
    assumption_review_interval_seconds: int = 3600  # 1 hour default
    assumption_review_auto_apply: bool = False  # Auto-apply SL/TP adjustments from review

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

        # Lock for state file writes (multiple async loops share this)
        self._state_lock = asyncio.Lock()

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
        logs_dir = Path(self.config.logs_dir)
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

    def _load_state(self):
        """Load automation state from file."""
        state_file = Path(self.config.state_file)
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self._last_analysis_time = {
                    k: datetime.fromisoformat(v)
                    for k, v in state.get("last_analysis_time", {}).items()
                }
                # Restore persisted results history
                for item in state.get("analysis_results", []):
                    try:
                        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                        self._analysis_results.append(AnalysisCycleResult(**item))
                    except Exception:
                        continue
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
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save automation state to file.

        Uses atomic write (write to temp, then rename) to avoid corruption
        when multiple async loops call this concurrently.
        """
        state_file = Path(self.config.state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)

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

        # Atomic write: write to temp file, then rename
        tmp_file = state_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(state, f, indent=2)
            # On Windows, os.replace is atomic and overwrites the target
            os.replace(str(tmp_file), str(state_file))
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")
            # Clean up temp file on failure
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
            async with aiohttp.ClientSession() as session:
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

            # Extract signal - handle both string and dict formats
            signal = decision.get("signal", "HOLD")
            if isinstance(signal, dict):
                signal = signal.get("value", "HOLD")
            signal = signal.upper()

            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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

            async with aiohttp.ClientSession() as session:
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

            signal = decision.get("signal", "HOLD")
            if isinstance(signal, dict):
                signal = signal.get("value", "HOLD")
            signal = signal.upper()

            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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

            async with aiohttp.ClientSession() as session:
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

            signal = decision.get("signal", "HOLD")
            if isinstance(signal, dict):
                signal = signal.get("value", "HOLD")
            signal = signal.upper()

            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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

            async with aiohttp.ClientSession() as session:
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

            signal = decision.get("signal", "HOLD")
            if isinstance(signal, dict):
                signal = signal.get("value", "HOLD")
            signal = signal.upper()

            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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

            async with aiohttp.ClientSession() as session:
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

            signal = result.get("signal", "HOLD")
            if isinstance(signal, dict):
                signal = signal.get("value", "HOLD")
            signal = signal.upper()

            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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

            async with aiohttp.ClientSession() as session:
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

            signal = decision.get("signal", "HOLD").upper()
            if signal in ["BUY_TO_ENTER", "BUY"]:
                signal = "BUY"
            elif signal in ["SELL_TO_ENTER", "SELL"]:
                signal = "SELL"
            else:
                signal = "HOLD"

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
        # 1) Per-automation: only count positions THIS automation owns
        # 2) Global: per-symbol and total limits from portfolio_config.yaml
        positions = get_open_positions()
        pending = get_pending_orders()

        # Build owned tickets for this automation
        my_source = self._source
        owned_tickets = set()
        try:
            active_decisions = list_active_decisions()
            for dec in active_decisions:
                if (dec.get("source") == my_source and dec.get("mt5_ticket")):
                    owned_tickets.add(dec["mt5_ticket"])
        except Exception:
            pass

        # Per-automation count (owned only)
        my_symbol_count = sum(
            1 for p in positions
            if p.get("symbol") == result.symbol and p.get("ticket") in owned_tickets
        )

        # Global counts (all positions on the account)
        global_symbol_count = sum(1 for p in positions if p.get("symbol") == result.symbol)
        global_symbol_count += sum(1 for o in pending if o.get("symbol") == result.symbol)
        global_total = len(positions) + len(pending)

        # Load global per-symbol limits
        global_limits = _get_global_position_limits()
        global_max_symbol = global_limits.get(result.symbol, 3)  # default 3 if symbol not in file

        self.logger.info(
            f"Position limits: {result.symbol} owned={my_symbol_count}/{self.config.max_positions_per_symbol}, "
            f"global={global_symbol_count}/{global_max_symbol}"
        )

        if my_symbol_count >= self.config.max_positions_per_symbol:
            self.logger.info(f"Automation max positions reached for {result.symbol} ({my_symbol_count} owned), skipping")
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

            self.logger.info(
                f"Trade params: {result.symbol} {result.signal} "
                f"entry={entry_price}, sl={stop_loss}, tp={take_profit}"
            )

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
                )

                # Store LLM-suggested trailing stop multiplier on the decision
                if result.trailing_stop_atr_multiplier and decision_id:
                    try:
                        from tradingagents.trade_decisions import load_decision, DECISIONS_DIR as _DEC_DIR
                        import json as _json
                        dec = load_decision(decision_id)
                        dec["trailing_stop_atr_multiplier"] = result.trailing_stop_atr_multiplier
                        dec_file = os.path.join(_DEC_DIR, f"{decision_id}.json")
                        with open(dec_file, "w") as _f:
                            _json.dump(dec, _f, indent=2, default=str)
                        self.logger.info(
                            f"  LLM trailing stop multiplier: {result.trailing_stop_atr_multiplier}x ATR"
                        )
                    except Exception as e:
                        self.logger.debug(f"  Could not save trailing multiplier: {e}")

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
                    )
                    self.logger.info(f"Failed trade saved as decision for manual retry")
                except Exception as store_err:
                    self.logger.error(f"Could not store failed decision: {store_err}")

        except Exception as e:
            result.execution_error = str(e)
            import traceback
            self.logger.error(f"TRADE EXECUTION ERROR: {e}\n{traceback.format_exc()}")

        return result

    def _infer_exit_reason(self, exit_price: float, entry: float, sl: float, tp: float, direction: str) -> str:
        """Infer how a trade exited based on exit price vs SL/TP levels."""
        if not exit_price:
            return "unknown"
        tolerance = abs(entry * 0.0005) if entry else 0.5  # 0.05% tolerance
        if tp and abs(exit_price - tp) <= tolerance:
            return "tp_hit"
        if sl and abs(exit_price - sl) <= tolerance:
            return "sl_hit"
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

            # Record in guardrails for circuit breaker
            pnl_pct = closed.get("pnl_percent", 0) if closed else 0
            was_win = closed.get("was_correct", False) if closed else profit > 0
            try:
                self.guardrails.record_trade_result(
                    was_win=was_win,
                    pnl_pct=pnl_pct,
                    account_balance=self._get_account_balance(),
                )
                self.logger.info(f"  Guardrails updated: {'win' if was_win else 'loss'} {pnl_pct:+.2f}%")
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

                self.logger.info(
                    f"Managing position #{ticket} {symbol} {direction}: "
                    f"entry={entry_price}, current={current_price}, "
                    f"sl={current_sl}, tp={current_tp}, profit={profit:.2f}"
                )

                # Calculate P&L percentage
                if entry_price > 0 and current_price > 0:
                    if direction == "BUY":
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                else:
                    pnl_pct = 0
                    if current_price <= 0:
                        self.logger.warning(
                            f"  Invalid current_price={current_price} for #{ticket}, "
                            f"skipping SL adjustments"
                        )

                result = PositionManagementResult(
                    timestamp=datetime.now(),
                    ticket=ticket,
                    symbol=symbol,
                    action="no_change",
                    old_sl=current_sl,
                    old_tp=current_tp,
                )

                try:
                    # Get ATR for stop calculations
                    atr = get_atr_for_symbol(symbol, period=14)
                    self.logger.debug(f"  ATR={atr:.5f}, pnl_pct={pnl_pct:.2f}%")

                    # Check breakeven condition (ATR-based: profit must exceed N * ATR)
                    breakeven_threshold_distance = self.config.move_to_breakeven_atr_mult * atr
                    if direction == "BUY":
                        profit_distance = current_price - entry_price
                    else:
                        profit_distance = entry_price - current_price

                    breakeven_eligible = profit_distance >= breakeven_threshold_distance and current_sl != 0

                    if breakeven_eligible:
                        self.logger.info(
                            f"  Breakeven check: profit_distance {profit_distance:.5f} >= "
                            f"threshold {breakeven_threshold_distance:.5f} ({self.config.move_to_breakeven_atr_mult}x ATR)"
                        )
                        breakeven_sl, is_eligible = self.stop_loss_manager.calculate_breakeven_stop(
                            entry_price=entry_price,
                            current_price=current_price,
                            direction=direction,
                            atr=atr,
                        )
                        self.logger.info(f"  Breakeven SL={breakeven_sl}, eligible={is_eligible}")

                        if is_eligible and breakeven_sl:
                            is_better = (
                                (direction == "BUY" and breakeven_sl > current_sl) or
                                (direction == "SELL" and breakeven_sl < current_sl)
                            )

                            if is_better and self.config.auto_execute:
                                self.logger.info(f"  Modifying SL to breakeven: {current_sl:.5f} -> {breakeven_sl:.5f}")
                                modify_result = modify_position(ticket, sl=breakeven_sl)
                                self.logger.info(f"  Modify result: {modify_result}")
                                if modify_result.get("success"):
                                    result.action = "adjusted_sl"
                                    result.new_sl = breakeven_sl
                                    self.logger.info(
                                        f"  BREAKEVEN SET for {symbol} #{ticket}: {current_sl:.5f} -> {breakeven_sl:.5f}"
                                    )
                                else:
                                    self.logger.error(f"  Failed to set breakeven for #{ticket}: {modify_result}")
                            elif not self.config.auto_execute:
                                self.logger.info(f"  Breakeven eligible but auto_execute=False, skipping")
                            else:
                                self.logger.debug(f"  Breakeven not better: current_sl={current_sl}, breakeven_sl={breakeven_sl}")

                    # Check trailing stop condition
                    elif self.config.enable_trailing_stop and pnl_pct > 0:
                        # Use per-trade LLM-suggested multiplier if available, else config default
                        trail_mult = ticket_trail_mult.get(ticket, self.config.trailing_stop_atr_multiplier)
                        trail_distance = trail_mult * atr
                        if direction == "BUY":
                            candidate_sl = round(current_price - trail_distance, 5)
                            should_trail = candidate_sl > current_sl
                        else:
                            candidate_sl = round(current_price + trail_distance, 5)
                            should_trail = candidate_sl < current_sl
                        new_sl = candidate_sl if should_trail else current_sl
                        self.logger.debug(f"  Trailing stop check: mult={trail_mult}x ATR, new_sl={new_sl}, should_trail={should_trail}")

                        if should_trail and new_sl and self.config.auto_execute:
                            self.logger.info(f"  Trailing SL: {current_sl:.5f} -> {new_sl:.5f}")
                            modify_result = modify_position(ticket, sl=new_sl)
                            self.logger.info(f"  Modify result: {modify_result}")
                            if modify_result.get("success"):
                                result.action = "adjusted_sl"
                                result.new_sl = new_sl
                                self.logger.info(
                                    f"  TRAILING STOP for {symbol} #{ticket}: {current_sl:.5f} -> {new_sl:.5f} ({trail_mult}x ATR)"
                                )
                            else:
                                self.logger.error(f"  Failed to trail stop for #{ticket}: {modify_result}")
                        elif should_trail and not self.config.auto_execute:
                            self.logger.info(f"  Trailing stop eligible but auto_execute=False, skipping")
                    else:
                        self.logger.debug(f"  No SL adjustment needed: pnl_pct={pnl_pct:.2f}%, trailing={self.config.enable_trailing_stop}")

                    # Run quick analysis to check for close signal
                    if self.config.pipeline in (PipelineType.SMC_QUANT_BASIC, PipelineType.SMC_QUANT, PipelineType.BREAKOUT_QUANT, PipelineType.RANGE_QUANT, PipelineType.RULE_BASED):
                        self.logger.info(f"  Running reversal check analysis for #{ticket} {symbol}...")
                        if self.config.pipeline == PipelineType.SMC_QUANT:
                            analysis = await self._run_smc_quant_analysis(symbol)
                        elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
                            analysis = await self._run_breakout_quant_analysis(symbol)
                        elif self.config.pipeline == PipelineType.RANGE_QUANT:
                            analysis = await self._run_range_quant_analysis(symbol)
                        elif self.config.pipeline == PipelineType.RULE_BASED:
                            analysis = await self._run_rule_based_analysis(symbol)
                        else:
                            analysis = await self._run_quant_analysis(symbol)
                        self.logger.info(f"  Reversal check result: signal={analysis.signal}, confidence={analysis.confidence:.2f}")

                        if analysis.signal == "CLOSE" or (
                            analysis.signal != "HOLD" and
                            analysis.signal != direction and
                            analysis.confidence > 0.7
                        ):
                            self.logger.warning(
                                f"  REVERSAL SIGNAL for #{ticket} {symbol}: "
                                f"signal={analysis.signal} (position={direction}), confidence={analysis.confidence:.2f}"
                            )
                            # Strong reversal signal
                            if self.config.auto_execute:
                                close_result = close_position(ticket)
                                self.logger.info(f"  Close result: {close_result}")
                                if close_result.get("success"):
                                    result.action = "closed"
                                    result.close_reason = f"Reversal signal: {analysis.signal}"
                                    result.pnl = profit
                                    self.logger.info(
                                        f"  POSITION CLOSED #{ticket} {symbol}: {result.close_reason}, pnl={profit:.2f}"
                                    )
                                    # Close the linked trade decision
                                    deal_info = get_closed_deal_by_ticket(ticket, days_back=7)
                                    exit_px = deal_info["price"] if deal_info else current_price
                                    self._close_decision_for_ticket(
                                        ticket, exit_px, profit, "reversal_signal",
                                        f"Reversal signal: {analysis.signal}"
                                    )
                                else:
                                    self.logger.error(f"  Failed to close position #{ticket}: {close_result}")
                            else:
                                self.logger.info(f"  Reversal detected but auto_execute=False, skipping close")

                except Exception as e:
                    import traceback
                    self.logger.error(f"Error managing position #{ticket}: {e}\n{traceback.format_exc()}")

                results.append(result)

            # Detect positions that disappeared (SL/TP hit externally)
            open_tickets = {p.get("ticket") for p in managed_positions}
            try:
                my_source = self._source
                active_decisions = list_active_decisions()
                for dec in active_decisions:
                    dec_ticket = dec.get("mt5_ticket")
                    dec_symbol = dec.get("symbol", "")
                    # Only check decisions owned by this automation
                    if dec.get("source") != my_source:
                        continue
                    if not dec_ticket or dec_ticket in open_tickets or dec_symbol not in self.config.symbols:
                        continue
                    # This decision's position is gone - close it
                    deal_info = get_closed_deal_by_ticket(dec_ticket, days_back=14)
                    if deal_info:
                        exit_price = deal_info["price"]
                        mt5_profit = deal_info.get("profit", 0)
                        exit_reason = self._infer_exit_reason(
                            exit_price, dec.get("entry_price", 0),
                            dec.get("stop_loss"), dec.get("take_profit"),
                            dec.get("action", "BUY")
                        )
                        self._close_decision_for_ticket(
                            dec_ticket, exit_price, mt5_profit, exit_reason,
                            f"Position closed externally ({exit_reason})"
                        )
                        results.append(PositionManagementResult(
                            timestamp=datetime.now(), ticket=dec_ticket, symbol=dec_symbol,
                            action="closed", close_reason=exit_reason, pnl=mt5_profit,
                        ))
                    else:
                        self.logger.debug(f"  Decision {dec['decision_id']} ticket #{dec_ticket} gone but no deal history found")
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
                    elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
                        result = await self._run_breakout_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.RANGE_QUANT:
                        result = await self._run_range_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
                        result = await self._run_vp_analysis(symbol)
                    elif self.config.pipeline == PipelineType.RULE_BASED:
                        result = await self._run_rule_based_analysis(symbol)
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

                self._save_state()

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

                    self._save_state()

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
        self.logger.info("=" * 50)

        # Run all loops concurrently
        try:
            await asyncio.gather(
                self._analysis_loop(),
                self._position_loop(),
                self._assumption_review_loop(),
            )
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

    def update_config(self, updates: Dict[str, Any]):
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
        self._save_state()

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
        elif self.config.pipeline == PipelineType.BREAKOUT_QUANT:
            return await self._run_breakout_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.RANGE_QUANT:
            return await self._run_range_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
            return await self._run_vp_analysis(symbol)
        elif self.config.pipeline == PipelineType.RULE_BASED:
            return await self._run_rule_based_analysis(symbol)
        else:
            return await self._run_multi_agent_analysis(symbol)


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
