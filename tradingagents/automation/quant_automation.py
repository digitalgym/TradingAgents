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
from dataclasses import dataclass, field, asdict
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


class PipelineType(str, Enum):
    """Available analysis pipelines."""
    QUANT = "quant"
    SMC_QUANT = "smc_quant"
    VOLUME_PROFILE = "volume_profile"
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
    # Pipeline settings
    pipeline: PipelineType = PipelineType.QUANT
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    timeframe: str = "H1"

    # Interval settings
    analysis_interval_seconds: int = 180  # 3 minutes default
    position_check_interval_seconds: int = 60  # 1 minute for position monitoring

    # Execution settings
    auto_execute: bool = False  # Require explicit enable for auto-trading
    min_confidence: float = 0.65  # Minimum confidence to execute trades

    # Position management
    max_positions_per_symbol: int = 1
    max_total_positions: int = 3
    enable_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    move_to_breakeven_pct: float = 1.0  # Move SL to breakeven at 1% profit

    # Risk settings
    max_risk_per_trade_pct: float = 1.0  # 1% of account per trade
    default_lot_size: float = 0.01
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 3

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
        return cls(**data)


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

        # State
        self._status = AutomationStatus.STOPPED
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_analysis_time: Dict[str, datetime] = {}
        self._last_position_check: Optional[datetime] = None
        self._analysis_results: List[AnalysisCycleResult] = []
        self._position_results: List[PositionManagementResult] = []
        self._error_message: Optional[str] = None

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

        self.logger = logging.getLogger("QuantAutomation")
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
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save automation state to file."""
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
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

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
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"SMC Quant API error: {error_text}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="smc_quant",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=time.time() - start_time,
                        )

                    result = await response.json()

            self.logger.info(f"SMC Quant API response status: {result.get('status')}")

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
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            self.logger.error(f"SMC Quant analysis error for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="smc_quant",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=time.time() - start_time,
            )

    async def _run_vp_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run volume profile quant analysis for a symbol."""
        start_time = time.time()

        try:
            import aiohttp

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
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"VP quant API error: {error_text}")
                        return AnalysisCycleResult(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            pipeline="volume_profile",
                            signal="HOLD",
                            confidence=0.0,
                            rationale=f"API Error: {response.status}",
                            duration_seconds=time.time() - start_time,
                        )

                    result = await response.json()

            self.logger.info(f"[VP] API response status: {result.get('status')}")

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

        except Exception as e:
            self.logger.error(f"VP analysis error for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return AnalysisCycleResult(
                timestamp=datetime.now(),
                symbol=symbol,
                pipeline="volume_profile",
                signal="HOLD",
                confidence=0.0,
                rationale=f"Error: {str(e)}",
                duration_seconds=time.time() - start_time,
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

        # Check position limits (count both open positions AND pending orders)
        positions = get_open_positions()
        pending = get_pending_orders()
        symbol_positions = [p for p in positions if p.get("symbol") == result.symbol]
        symbol_pending = [o for o in pending if o.get("symbol") == result.symbol]
        symbol_total = len(symbol_positions) + len(symbol_pending)
        all_total = len(positions) + len(pending)
        self.logger.info(
            f"Position check: {symbol_total}/{self.config.max_positions_per_symbol} "
            f"for {result.symbol} ({len(symbol_positions)} open + {len(symbol_pending)} pending), "
            f"{all_total}/{self.config.max_total_positions} total"
        )

        if symbol_total >= self.config.max_positions_per_symbol:
            self.logger.info(f"Max positions reached for {result.symbol} ({len(symbol_positions)} open + {len(symbol_pending)} pending), skipping")
            return result

        if all_total >= self.config.max_total_positions:
            self.logger.info(f"Max total positions reached ({all_total}: {len(positions)} open + {len(pending)} pending), skipping")
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
                store_decision(
                    symbol=result.symbol,
                    decision_type="OPEN",
                    action=result.signal,
                    rationale=result.rationale[:500],
                    source=f"quant_automation_{result.pipeline}",
                    entry_price=result.entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=lot_size,
                    mt5_ticket=result.execution_ticket,
                )

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
                        source=f"quant_automation_{result.pipeline}",
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
        """Find and close the trade decision linked to an MT5 ticket."""
        try:
            decision = find_decision_by_ticket(ticket)
            if not decision or decision.get("status") != "active":
                return

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
                    dec_file = os.path.join(DECISIONS_DIR, f"{decision['decision_id']}.json")
                    if os.path.exists(dec_file):
                        with open(dec_file, "r") as f:
                            data = json.load(f)
                        data["entry_price"] = pos_entry
                        with open(dec_file, "w") as f:
                            json.dump(data, f, indent=2, default=str)
                    self.logger.info(f"  Backfilled entry_price={pos_entry} for decision {decision['decision_id']}")

            close_decision(
                decision["decision_id"],
                exit_price=exit_price,
                outcome_notes=f"{notes}. MT5 profit: {profit:.2f}",
                exit_reason=exit_reason,
            )
            self.logger.info(f"  Decision {decision['decision_id']} closed: {exit_reason}, pnl={profit:.2f}")
        except Exception as e:
            self.logger.error(f"  Failed to close decision for #{ticket}: {e}")

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
            managed_positions = [p for p in positions if p.get("symbol") in self.config.symbols]
            self.logger.info(
                f"Position check: {len(positions)} total open, "
                f"{len(managed_positions)} managed ({', '.join(self.config.symbols)})"
            )

            for position in positions:
                symbol = position.get("symbol")
                ticket = position.get("ticket")

                # Skip if not in our managed symbols
                if symbol not in self.config.symbols:
                    continue

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
                if entry_price > 0:
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

                try:
                    # Get ATR for stop calculations
                    atr = get_atr_for_symbol(symbol, period=14)
                    self.logger.debug(f"  ATR={atr:.5f}, pnl_pct={pnl_pct:.2f}%")

                    # Check breakeven condition
                    if pnl_pct >= self.config.move_to_breakeven_pct and current_sl != 0:
                        self.logger.info(f"  Breakeven check: pnl {pnl_pct:.2f}% >= threshold {self.config.move_to_breakeven_pct}%")
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
                        new_sl, should_trail = self.stop_loss_manager.calculate_trailing_stop(
                            current_price=current_price,
                            current_sl=current_sl,
                            atr=atr,
                            direction=direction,
                        )
                        self.logger.debug(f"  Trailing stop check: new_sl={new_sl}, should_trail={should_trail}")

                        if should_trail and new_sl and self.config.auto_execute:
                            self.logger.info(f"  Trailing SL: {current_sl:.5f} -> {new_sl:.5f}")
                            modify_result = modify_position(ticket, sl=new_sl)
                            self.logger.info(f"  Modify result: {modify_result}")
                            if modify_result.get("success"):
                                result.action = "adjusted_sl"
                                result.new_sl = new_sl
                                self.logger.info(
                                    f"  TRAILING STOP for {symbol} #{ticket}: {current_sl:.5f} -> {new_sl:.5f}"
                                )
                            else:
                                self.logger.error(f"  Failed to trail stop for #{ticket}: {modify_result}")
                        elif should_trail and not self.config.auto_execute:
                            self.logger.info(f"  Trailing stop eligible but auto_execute=False, skipping")
                    else:
                        self.logger.debug(f"  No SL adjustment needed: pnl_pct={pnl_pct:.2f}%, trailing={self.config.enable_trailing_stop}")

                    # Run quick analysis to check for close signal
                    if self.config.pipeline in (PipelineType.QUANT, PipelineType.SMC_QUANT):
                        self.logger.info(f"  Running reversal check analysis for #{ticket} {symbol}...")
                        if self.config.pipeline == PipelineType.SMC_QUANT:
                            analysis = await self._run_smc_quant_analysis(symbol)
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
                active_decisions = list_active_decisions()
                for dec in active_decisions:
                    dec_ticket = dec.get("mt5_ticket")
                    dec_symbol = dec.get("symbol", "")
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
                    if self.config.pipeline == PipelineType.QUANT:
                        result = await self._run_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.SMC_QUANT:
                        result = await self._run_smc_quant_analysis(symbol)
                    elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
                        result = await self._run_vp_analysis(symbol)
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
        self.logger.info(f"Auto-execute: {self.config.auto_execute}")
        self.logger.info("=" * 50)

        # Run both loops concurrently
        try:
            await asyncio.gather(
                self._analysis_loop(),
                self._position_loop(),
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
                "max_total": self.config.max_total_positions,
            },
            "last_analysis": {
                symbol: t.isoformat() if t else None
                for symbol, t in self._last_analysis_time.items()
            },
            "last_position_check": self._last_position_check.isoformat() if self._last_position_check else None,
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
            "guardrails": self.guardrails.get_status(),
        }

    async def run_single_analysis(self, symbol: str) -> AnalysisCycleResult:
        """Run a single analysis cycle for testing."""
        if self.config.pipeline == PipelineType.QUANT:
            return await self._run_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.SMC_QUANT:
            return await self._run_smc_quant_analysis(symbol)
        elif self.config.pipeline == PipelineType.VOLUME_PROFILE:
            return await self._run_vp_analysis(symbol)
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
