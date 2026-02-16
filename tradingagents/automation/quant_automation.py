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
    modify_position,
    close_position,
    execute_trade_signal,
    get_mt5_symbol_info,
    check_mt5_autotrading,
    get_atr_for_symbol,
)

# Risk management
from tradingagents.risk import (
    RiskGuardrails,
    DynamicStopLoss,
    PositionSizer,
)

# Trade decision tracking
from tradingagents.trade_decisions import (
    store_decision,
    list_active_decisions,
    close_decision,
)


class PipelineType(str, Enum):
    """Available analysis pipelines."""
    QUANT = "quant"
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
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save automation state to file."""
        state_file = Path(self.config.state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "last_analysis_time": {
                k: v.isoformat() for k, v in self._last_analysis_time.items()
            },
            "last_updated": datetime.now().isoformat(),
            "status": self._status.value,
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _get_account_balance(self) -> float:
        """Get current account balance from MT5."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return 10000.0
            account_info = mt5.account_info()
            if account_info:
                return account_info.balance
            return 10000.0
        except Exception:
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

            # Extract decision from API response
            decision = result.get("decision", {})

            if not decision:
                return AnalysisCycleResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    pipeline="quant",
                    signal="HOLD",
                    confidence=0.0,
                    rationale="No decision from analysis",
                    duration_seconds=time.time() - start_time,
                )

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
        if not self.config.auto_execute:
            self.logger.info(f"Auto-execute disabled, skipping trade for {result.symbol}")
            return result

        if result.signal == "HOLD":
            return result

        if result.confidence < self.config.min_confidence:
            self.logger.info(
                f"Confidence {result.confidence:.2f} below threshold "
                f"{self.config.min_confidence} for {result.symbol}"
            )
            return result

        # Check position limits
        positions = get_open_positions()
        symbol_positions = [p for p in positions if p.get("symbol") == result.symbol]

        if len(symbol_positions) >= self.config.max_positions_per_symbol:
            self.logger.info(f"Max positions reached for {result.symbol}")
            return result

        if len(positions) >= self.config.max_total_positions:
            self.logger.info(f"Max total positions reached")
            return result

        # Check guardrails
        can_trade, reason = self.guardrails.check_can_trade(self._get_account_balance())
        if not can_trade:
            self.logger.warning(f"Trading blocked: {reason}")
            return result

        try:
            # Get current price
            price_info = get_mt5_current_price(result.symbol)
            entry_price = price_info.get("ask") if result.signal == "BUY" else price_info.get("bid")

            # Use provided SL/TP or calculate defaults
            stop_loss = result.stop_loss
            take_profit = result.take_profit

            if not stop_loss or not take_profit:
                # Calculate ATR-based levels
                atr = get_atr_for_symbol(result.symbol, period=14)
                if result.signal == "BUY":
                    stop_loss = stop_loss or (entry_price - 2 * atr)
                    take_profit = take_profit or (entry_price + 3 * atr)
                else:
                    stop_loss = stop_loss or (entry_price + 2 * atr)
                    take_profit = take_profit or (entry_price - 3 * atr)

            # Calculate position size
            symbol_info = get_mt5_symbol_info(result.symbol)
            lot_size = self.config.default_lot_size

            # Execute trade
            trade_result = execute_trade_signal(
                symbol=result.symbol,
                signal=result.signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=lot_size,
                comment=f"QuantAuto {result.pipeline}",
            )

            if trade_result.get("success"):
                result.executed = True
                result.execution_ticket = trade_result.get("order_id")
                result.entry_price = trade_result.get("price", entry_price)
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
                    f"Trade executed: {result.symbol} {result.signal} "
                    f"{lot_size} lots @ {result.entry_price}"
                )
            else:
                result.execution_error = trade_result.get("error", "Unknown error")
                self.logger.error(f"Trade failed: {result.execution_error}")

        except Exception as e:
            result.execution_error = str(e)
            self.logger.error(f"Trade execution error: {e}")

        return result

    async def _manage_positions(self) -> List[PositionManagementResult]:
        """Manage existing positions - trailing stops, breakeven, close signals."""
        results = []

        try:
            positions = get_open_positions()

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

                    # Check breakeven condition
                    if pnl_pct >= self.config.move_to_breakeven_pct and current_sl != 0:
                        breakeven_sl = self.stop_loss_manager.calculate_breakeven_stop(
                            entry_price=entry_price,
                            current_price=current_price,
                            current_sl=current_sl,
                            direction=direction,
                        )

                        if breakeven_sl:
                            is_better = (
                                (direction == "BUY" and breakeven_sl > current_sl) or
                                (direction == "SELL" and breakeven_sl < current_sl)
                            )

                            if is_better and self.config.auto_execute:
                                modify_result = modify_position(ticket, sl=breakeven_sl)
                                if modify_result.get("success"):
                                    result.action = "adjusted_sl"
                                    result.new_sl = breakeven_sl
                                    self.logger.info(
                                        f"Breakeven set for {symbol}: {current_sl:.5f} -> {breakeven_sl:.5f}"
                                    )

                    # Check trailing stop condition
                    elif self.config.enable_trailing_stop and pnl_pct > 0:
                        new_sl, should_trail = self.stop_loss_manager.calculate_trailing_stop(
                            current_price=current_price,
                            current_sl=current_sl,
                            atr=atr,
                            direction=direction,
                        )

                        if should_trail and new_sl and self.config.auto_execute:
                            modify_result = modify_position(ticket, sl=new_sl)
                            if modify_result.get("success"):
                                result.action = "adjusted_sl"
                                result.new_sl = new_sl
                                self.logger.info(
                                    f"Trailing stop for {symbol}: {current_sl:.5f} -> {new_sl:.5f}"
                                )

                    # Run quick analysis to check for close signal
                    if self.config.pipeline == PipelineType.QUANT:
                        analysis = await self._run_quant_analysis(symbol)
                        if analysis.signal == "CLOSE" or (
                            analysis.signal != "HOLD" and
                            analysis.signal != direction and
                            analysis.confidence > 0.7
                        ):
                            # Strong reversal signal
                            if self.config.auto_execute:
                                close_result = close_position(ticket)
                                if close_result.get("success"):
                                    result.action = "closed"
                                    result.close_reason = f"Reversal signal: {analysis.signal}"
                                    result.pnl = profit
                                    self.logger.info(
                                        f"Position closed for {symbol}: {result.close_reason}"
                                    )

                except Exception as e:
                    self.logger.error(f"Error managing position {ticket}: {e}")

                results.append(result)

        except Exception as e:
            self.logger.error(f"Position management error: {e}")

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

                    self.logger.info(f"Running {self.config.pipeline.value} analysis for {symbol}...")

                    # Run analysis based on pipeline
                    if self.config.pipeline == PipelineType.QUANT:
                        result = await self._run_quant_analysis(symbol)
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
        while self._running:
            try:
                results = await self._manage_positions()

                # Store results
                self._position_results.extend(results)
                if len(self._position_results) > 100:
                    self._position_results = self._position_results[-100:]

                self._last_position_check = datetime.now()

            except Exception as e:
                self.logger.error(f"Position loop error: {e}")

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
                if key == "pipeline" and isinstance(value, str):
                    value = PipelineType(value)
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {value}")
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        positions = []
        try:
            all_positions = get_open_positions()
            positions = [p for p in all_positions if p.get("symbol") in self.config.symbols]
        except Exception:
            pass

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

    if _automation_instance and _automation_instance._running:
        raise RuntimeError("Automation already running")

    # Create config from dict if provided
    if config:
        auto_config = QuantAutomationConfig.from_dict(config)
    else:
        auto_config = QuantAutomationConfig()

    _automation_instance = QuantAutomation(auto_config)

    # Start in background task
    _automation_task = asyncio.create_task(_automation_instance.start())

    # Give it a moment to start
    await asyncio.sleep(0.5)

    return _automation_instance


async def stop_automation():
    """Stop the global automation instance."""
    global _automation_instance, _automation_task

    if _automation_instance:
        _automation_instance.stop()

    if _automation_task:
        try:
            await asyncio.wait_for(_automation_task, timeout=5.0)
        except asyncio.TimeoutError:
            _automation_task.cancel()
        _automation_task = None
