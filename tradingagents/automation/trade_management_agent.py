"""
Trade Management Agent (TMA)

Centralized position management for all MT5 positions, replacing per-automation
trailing/breakeven/close logic with a single dedicated agent.

Features:
- Trailing stops (ATR-based)
- Breakeven stops
- Partial take-profit
- Time-based position limits
- Account-level risk monitoring (exposure, correlation)
- Per-position policy overrides via Postgres
- Closed position detection and decision closure
- Audit trail of all management actions
"""

import asyncio
import logging
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, fields, asdict


class SafeFileHandler(logging.FileHandler):
    """FileHandler that recovers from stale file handles on Windows (sleep/wake cycles)."""

    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            try:
                self.close()
                self.stream = self._open()
                super().emit(record)
            except Exception:
                pass


# MT5 imports
from tradingagents.dataflows.mt5_data import (
    get_mt5_current_price,
    get_open_positions,
    modify_position,
    close_position,
    get_mt5_symbol_info,
    check_mt5_autotrading,
    is_market_open,
)

# Risk management
from tradingagents.risk import (
    DynamicStopLoss,
    get_atr_for_symbol,
)

# Trade decision tracking
from tradingagents.trade_decisions import (
    list_active_decisions,
    close_decision,
    find_decision_by_ticket,
    add_trade_event,
    DECISIONS_DIR,
)

from tradingagents.dataflows.mt5_data import get_closed_deal_by_ticket


_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TradeManagementConfig:
    """Configuration for the Trade Management Agent."""
    # Instance identity
    instance_name: str = "trade_manager"

    # Position management intervals
    management_interval_seconds: float = 900.0  # How often to check positions (15 min)
    risk_check_interval_seconds: float = 900.0  # Account-level risk checks (same as management)
    control_poll_seconds: float = 3.0          # Postgres command polling

    # Trailing stop defaults
    enable_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5

    # Breakeven stop defaults
    enable_breakeven_stop: bool = True
    breakeven_atr_multiplier: float = 1.5  # Move SL to breakeven after profit >= N * ATR

    # Per-symbol overrides (symbol -> settings dict)
    # Keys: trailing_stop_atr_multiplier, breakeven_atr_multiplier, enable_trailing_stop, etc.
    symbol_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Partial take-profit defaults
    enable_partial_tp: bool = False
    partial_tp_percent: float = 50.0     # Close this % of volume at partial TP
    partial_tp_rr_ratio: float = 1.0     # Trigger at 1:1 risk-reward

    # Assumption review (SMC structure check on open positions)
    enable_assumption_review: bool = True
    assumption_review_use_llm: bool = False  # LLM assessment adds nuance but costs tokens
    assumption_review_auto_apply: bool = False  # Auto-apply SL/TP adjustments

    # Time-based limits
    enable_time_limit: bool = False
    max_position_hours: float = 48.0     # Close position after N hours

    # Account risk thresholds
    max_account_exposure_pct: float = 10.0   # Max total exposure as % of equity
    max_correlated_positions: int = 3        # Max positions in correlated pairs
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "gold": ["XAUUSD", "XAGUSD"],
        "usd_majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        "crypto": ["BTCUSD", "ETHUSD"],
    })

    # ATR cache TTL
    atr_cache_ttl_seconds: float = 60.0

    # Policy cache TTL
    policy_cache_ttl_seconds: float = 30.0

    # Logging
    logs_dir: str = "logs/trade_management"

    # Scalp mode for volume profile trades
    # Backtested optimal params: 1.5x ATR TP, 0.5x ATR BE, 6h max
    # Turns VP from -4.15% to +1.79% (80% WR on 5 trades, 4 blocked by trend filter)
    enable_scalp_mode: bool = True  # Auto-detect VP trades and apply scalp management
    scalp_tp_atr_multiplier: float = 1.5  # Take profit at 1.5x ATR from entry
    scalp_be_atr_multiplier: float = 0.5  # Move to breakeven after 0.5x ATR profit
    scalp_max_hours: float = 6.0  # Close after 6 hours if still open
    scalp_pipelines: List[str] = field(default_factory=lambda: ["volume_profile"])

    # Opposing position conflict resolution
    enable_opposing_check: bool = True
    auto_resolve_opposing: bool = True  # Auto-close the weaker opposing position
    opposing_score_threshold: float = 20.0  # Minimum score difference to recommend closure

    # Remote control
    enable_remote_control: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeManagementConfig":
        """Create from dictionary."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Per-position policy override
# ---------------------------------------------------------------------------

@dataclass
class PositionManagementPolicy:
    """Per-position override for management behavior."""
    ticket: int
    symbol: str

    # Overrides (None = use global defaults)
    enable_trailing_stop: Optional[bool] = None
    trailing_stop_atr_multiplier: Optional[float] = None
    enable_breakeven_stop: Optional[bool] = None
    breakeven_atr_multiplier: Optional[float] = None
    enable_partial_tp: Optional[bool] = None
    partial_tp_percent: Optional[float] = None
    partial_tp_rr_ratio: Optional[float] = None
    enable_time_limit: Optional[bool] = None
    max_position_hours: Optional[float] = None

    # Manual overrides
    manual_sl: Optional[float] = None   # Manually set SL (TMA won't move it)
    manual_tp: Optional[float] = None   # Manually set TP
    frozen: bool = False                 # Freeze all management for this position

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionManagementPolicy":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@dataclass
class ManagementAction:
    """Record of a management action for audit trail."""
    ticket: int
    symbol: str
    action_type: str       # trailing_stop, breakeven, partial_tp, time_close, risk_close, closed_detected
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    reason: str = ""
    success: bool = True
    error: Optional[str] = None
    decision_id: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trade Management Agent
# ---------------------------------------------------------------------------

class TradeManagementAgent:
    """
    Centralized position management agent.

    Manages all MT5 positions with trailing stops, breakeven,
    partial TP, time limits, and account-level risk checks.
    """

    def __init__(self, config: Optional[TradeManagementConfig] = None):
        self.config = config or TradeManagementConfig()

        # State
        self._running = False
        self._status = "stopped"
        self._shutdown_event = asyncio.Event()
        self._error_message: Optional[str] = None
        self._market_closed_until: Optional[datetime] = None
        self._missing_ticket_counts: Dict[int, int] = {}

        # ATR cache: symbol -> (atr_value, timestamp)
        self._atr_cache: Dict[str, Tuple[float, float]] = {}

        # Price extremes tracking for excursion analysis: ticket -> {high, low, direction}
        self._price_extremes: Dict[int, Dict[str, Any]] = {}

        # Policy cache: ticket -> (policy, timestamp)
        self._policy_cache: Dict[int, Tuple[PositionManagementPolicy, float]] = {}
        self._policies_loaded_at: float = 0.0

        # Postgres pool (lazy)
        self._pg_pool: Optional[Any] = None

        # Stop loss calculator
        self.stop_loss_manager = DynamicStopLoss(
            atr_multiplier=2.0,
            trailing_multiplier=self.config.trailing_stop_atr_multiplier,
        )

        # Counters for status reporting
        self._positions_managed: int = 0
        self._actions_taken: int = 0
        self._last_management_cycle: Optional[datetime] = None
        self._last_risk_check: Optional[datetime] = None

        # Logging
        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self):
        """Setup file + console logging."""
        logs_dir = _PROJECT_ROOT / self.config.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"TMA.{self.config.instance_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        log_file = logs_dir / f"tma_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = SafeFileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - [TMA] %(message)s")
        )
        self.logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Postgres helpers
    # ------------------------------------------------------------------

    async def _get_pg_pool(self):
        """Get or create asyncpg pool."""
        if self._pg_pool is None:
            import asyncpg
            url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
            if url:
                self._pg_pool = await asyncpg.create_pool(
                    url,
                    min_size=1,
                    max_size=5,
                    command_timeout=30,
                    statement_cache_size=0,
                )
        return self._pg_pool

    def _get_postgres_store(self):
        """Get PostgresStateStore for management tables."""
        try:
            from tradingagents.storage.postgres_store import get_management_store
            return get_management_store()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # ATR caching
    # ------------------------------------------------------------------

    def _get_cached_atr(self, symbol: str) -> float:
        """Get ATR with caching (TTL from config)."""
        now = time.time()
        cached = self._atr_cache.get(symbol)
        if cached and (now - cached[1]) < self.config.atr_cache_ttl_seconds:
            return cached[0]

        atr = get_atr_for_symbol(symbol, period=14)
        if atr > 0:
            self._atr_cache[symbol] = (atr, now)
        return atr

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    async def _load_policies(self) -> Dict[int, PositionManagementPolicy]:
        """Load all policies from Postgres with caching."""
        now = time.time()
        if (now - self._policies_loaded_at) < self.config.policy_cache_ttl_seconds:
            return {t: p for t, (p, _) in self._policy_cache.items()}

        store = self._get_postgres_store()
        if store:
            try:
                raw_policies = store.load_all_management_policies()
                self._policy_cache.clear()
                for ticket, policy_dict in raw_policies.items():
                    policy = PositionManagementPolicy.from_dict(policy_dict)
                    self._policy_cache[ticket] = (policy, now)
                self._policies_loaded_at = now
            except Exception as e:
                self.logger.warning(f"Failed to load policies from DB: {e}")

        return {t: p for t, (p, _) in self._policy_cache.items()}

    def _get_symbol_setting(self, symbol: str, key: str, default: Any) -> Any:
        """Get per-symbol setting, falling back to default."""
        sym_settings = self.config.symbol_settings.get(symbol, {})
        return sym_settings.get(key, default)

    def _get_effective_setting(self, policy: Optional[PositionManagementPolicy],
                                attr: str, symbol: str = "") -> Any:
        """Get effective setting: policy override > symbol override > config default."""
        if policy is not None:
            val = getattr(policy, attr, None)
            if val is not None:
                return val
        if symbol:
            sym_val = self.config.symbol_settings.get(symbol, {}).get(attr)
            if sym_val is not None:
                return sym_val
        return getattr(self.config, attr)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def _log_action(self, action: ManagementAction) -> None:
        """Log a management action to Postgres and local log."""
        level = "INFO" if action.success else "WARNING"
        self.logger.log(
            logging.INFO if action.success else logging.WARNING,
            f"ACTION #{action.ticket} {action.symbol} {action.action_type}: "
            f"{action.old_value} -> {action.new_value} | {action.reason} "
            f"{'OK' if action.success else f'FAILED: {action.error}'}"
        )

        store = self._get_postgres_store()
        if store:
            try:
                store.save_management_action(action.to_dict())
            except Exception as e:
                self.logger.warning(f"Failed to save action to DB: {e}")

        self._actions_taken += 1

    def _log_risk_alert(self, alert_type: str, severity: str, message: str,
                        details: Optional[Dict] = None) -> None:
        """Log a risk alert."""
        self.logger.warning(f"RISK ALERT [{severity}] {alert_type}: {message}")

        store = self._get_postgres_store()
        if store:
            try:
                store.save_risk_alert(alert_type, severity, message, details)
            except Exception as e:
                self.logger.warning(f"Failed to save risk alert to DB: {e}")

    # ------------------------------------------------------------------
    # Exit reason inference (mirrors quant_automation pattern)
    # ------------------------------------------------------------------

    def _infer_exit_reason(self, exit_price: float, entry: float,
                           sl: float, tp: float, direction: str,
                           deal_comment: str = "") -> str:
        """Infer how a trade exited based on deal comment and price levels."""
        if deal_comment:
            comment_lower = deal_comment.lower().strip()
            if comment_lower in ("sl", "stop loss"):
                return "sl_hit"
            if comment_lower in ("tp", "take profit"):
                return "tp_hit"
            if "[sl" in comment_lower:
                return "sl_hit"
            if "[tp" in comment_lower:
                return "tp_hit"
            if "trailing" in comment_lower:
                return "trailing_stop"

        if not exit_price:
            return "unknown"
        tolerance = abs(entry * 0.0005) if entry else 0.5
        if tp and abs(exit_price - tp) <= tolerance:
            return "tp_hit"
        if sl and abs(exit_price - sl) <= tolerance:
            return "sl_hit"
        if entry and abs(exit_price - entry) <= tolerance:
            return "breakeven_stop"
        return "manual_close"

    # ------------------------------------------------------------------
    # Decision closure (mirrors quant_automation pattern)
    # ------------------------------------------------------------------

    def _close_decision_for_ticket(self, ticket: int, exit_price: float,
                                    profit: float, exit_reason: str,
                                    notes: str) -> None:
        """Find and close the trade decision linked to an MT5 ticket."""
        try:
            decision = find_decision_by_ticket(ticket)
            if not decision or decision.get("status") != "active":
                return

            decision_id = decision["decision_id"]

            # Backfill entry_price if stored as 0
            if not decision.get("entry_price"):
                positions = get_open_positions(decision.get("symbol"))
                for p in positions:
                    if p.get("ticket") == ticket:
                        decision["entry_price"] = p.get("price_open")
                        break

            # Get tracked price extremes for excursion analysis
            max_favorable_price = None
            max_adverse_price = None
            if ticket in self._price_extremes:
                extremes = self._price_extremes[ticket]
                direction = extremes.get("direction", decision.get("action", "BUY"))
                if direction == "BUY":
                    max_favorable_price = extremes["high"]
                    max_adverse_price = extremes["low"]
                else:
                    max_favorable_price = extremes["low"]
                    max_adverse_price = extremes["high"]
                # Clean up tracked extremes
                del self._price_extremes[ticket]

            closed = close_decision(
                decision_id,
                exit_price=exit_price,
                outcome_notes=f"{notes}. MT5 profit: {profit:.2f}",
                exit_reason=exit_reason,
                max_favorable_price=max_favorable_price,
                max_adverse_price=max_adverse_price,
            )
            self.logger.info(f"  Decision {decision_id} closed: {exit_reason}, pnl={profit:.2f}")

            add_trade_event(decision_id, "closed", {
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "mt5_profit": profit,
                "pnl_percent": closed.get("pnl_percent") if closed else None,
                "was_correct": closed.get("was_correct") if closed else None,
                "result": closed.get("structured_outcome", {}).get("result") if closed else None,
                "managed_by": "trade_management_agent",
            }, source=self.config.instance_name)

            # SMC pattern reflection
            self._reflect_on_closed_trade(decision_id)

        except Exception as e:
            self.logger.error(f"  Failed to close decision for #{ticket}: {e}")

    def _reflect_on_closed_trade(self, decision_id: str) -> None:
        """Run SMC pattern reflection on a closed trade decision."""
        try:
            from tradingagents.trade_decisions import load_decision
            closed_decision = load_decision(decision_id)
            if not closed_decision:
                return

            smc_context = closed_decision.get("smc_context", {})
            setup_type = smc_context.get("setup_type") or closed_decision.get("setup_type")
            rationale = closed_decision.get("rationale", "")

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
                    closed_decision["setup_type"] = setup_type
                    if "smc_context" not in closed_decision:
                        closed_decision["smc_context"] = {}
                    closed_decision["smc_context"]["setup_type"] = setup_type

            if not setup_type:
                self.logger.debug(f"  No setup_type for {decision_id}, skipping SMC reflection")
                return

            from tradingagents.graph.reflection import Reflector
            from tradingagents.agents.utils.memory import SMCPatternMemory
            from tradingagents.default_config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG.copy()
            smc_memory = SMCPatternMemory(config)

            reflector = Reflector.__new__(Reflector)
            reflector.reflect_smc_pattern(
                decision=closed_decision,
                smc_memory=smc_memory,
            )
            self.logger.info(f"  SMC pattern memory stored for {decision_id} (setup: {setup_type})")

        except Exception as e:
            self.logger.warning(f"  SMC reflection failed for {decision_id}: {e}")

    # ------------------------------------------------------------------
    # Partial close (MT5 direct)
    # ------------------------------------------------------------------

    def _partial_close(self, ticket: int, symbol: str, direction: str,
                       volume: float) -> Dict[str, Any]:
        """Partially close a position by sending a counter-deal with reduced volume."""
        try:
            import MetaTrader5 as mt5

            close_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": "Could not get price"}

            price = tick.bid if direction == "BUY" else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "TMA partial TP",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                return {"success": False, "error": f"Partial close failed: {error}"}
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Partial close failed: {result.comment}"}
            return {"success": True, "ticket": ticket, "volume_closed": volume}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Core position management
    # ------------------------------------------------------------------

    async def _manage_positions(self) -> None:
        """Main position management cycle. Runs every management_interval_seconds."""
        self.logger.debug("--- Position management cycle start ---")

        # Check market closed cooldown
        if self._market_closed_until and datetime.now() < self._market_closed_until:
            import MetaTrader5 as _mt5
            if _mt5.terminal_info():
                # Verify market is still closed
                positions = get_open_positions()
                if positions:
                    # If we can see positions, market may be open for some symbols
                    self._market_closed_until = None
            if self._market_closed_until:
                self.logger.debug(f"Market closed — skipping until {self._market_closed_until.strftime('%H:%M')}")
                return

        try:
            import MetaTrader5 as _mt5
            if not _mt5.terminal_info():
                if not _mt5.initialize():
                    self.logger.warning("MT5 not connected, skipping position management")
                    return

            positions = get_open_positions()
            if not positions:
                # Still need to detect closed positions
                await self._detect_closed_positions(set())
                return

            # Load policies
            policies = await self._load_policies()

            # All positions with active decisions
            active_decisions = list_active_decisions()
            decision_by_ticket: Dict[int, Dict] = {}
            for dec in active_decisions:
                ticket = dec.get("mt5_ticket")
                if ticket:
                    decision_by_ticket[ticket] = dec

            managed_tickets = set()

            for position in positions:
                symbol = position.get("symbol")
                ticket = position.get("ticket")
                if not ticket or not symbol:
                    continue

                managed_tickets.add(ticket)

                pos_type = position.get("type", "")
                direction = "BUY" if "BUY" in pos_type.upper() else "SELL"
                entry_price = position.get("price_open", 0)
                current_price = position.get("price_current", 0)
                current_sl = position.get("sl", 0)
                current_tp = position.get("tp", 0)
                profit = position.get("profit", 0)
                volume = position.get("volume", 0)
                open_time = position.get("time", 0)

                if current_price <= 0:
                    continue

                # Track price extremes for excursion analysis (max favorable/adverse)
                if ticket not in self._price_extremes:
                    self._price_extremes[ticket] = {
                        "high": current_price,
                        "low": current_price,
                        "direction": direction,
                        "entry": entry_price,
                    }
                else:
                    self._price_extremes[ticket]["high"] = max(
                        self._price_extremes[ticket]["high"], current_price
                    )
                    self._price_extremes[ticket]["low"] = min(
                        self._price_extremes[ticket]["low"], current_price
                    )

                # Calculate P&L percentage
                if entry_price > 0:
                    if direction == "BUY":
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                # Get policy for this position
                policy = policies.get(ticket)
                if policy and policy.frozen:
                    self.logger.debug(f"  #{ticket} {symbol}: FROZEN, skipping")
                    continue

                # Get ATR
                atr = self._get_cached_atr(symbol)
                if atr <= 0:
                    self.logger.debug(f"  #{ticket} {symbol}: No ATR, skipping")
                    continue

                # Get linked decision for trail mult override
                decision = decision_by_ticket.get(ticket)
                trail_mult_override = None
                if decision and decision.get("trailing_stop_atr_multiplier"):
                    trail_mult_override = decision["trailing_stop_atr_multiplier"]

                # Status tracking
                be_status = "disabled"
                trail_status = "disabled"
                partial_status = "disabled"
                time_status = "disabled"
                scalp_status = "disabled"

                # --- Scalp Mode (Volume Profile trades) ---
                is_scalp = False
                if self.config.enable_scalp_mode and decision:
                    dec_pipeline = (decision.get("pipeline") or "").lower()
                    if dec_pipeline in [p.lower() for p in self.config.scalp_pipelines]:
                        is_scalp = True
                        scalp_tp_dist = self.config.scalp_tp_atr_multiplier * atr
                        scalp_be_dist = self.config.scalp_be_atr_multiplier * atr

                        if direction == "BUY":
                            profit_distance = current_price - entry_price
                        else:
                            profit_distance = entry_price - current_price

                        # Scalp TP: close at 0.75x ATR profit
                        if profit_distance >= scalp_tp_dist:
                            result = close_position(ticket)
                            action = ManagementAction(
                                ticket=ticket, symbol=symbol,
                                action_type="scalp_tp",
                                old_value=entry_price, new_value=current_price,
                                reason=f"Scalp TP hit: profit {profit_distance:.2f} >= {scalp_tp_dist:.2f} ({self.config.scalp_tp_atr_multiplier}x ATR)",
                                success=result.get("success", False),
                                decision_id=decision.get("decision_id"),
                            )
                            self._log_action(action)
                            if action.success:
                                scalp_status = f"TP_HIT +{profit_distance:.2f}"
                                deal_info = get_closed_deal_by_ticket(ticket, days_back=7)
                                exit_px = deal_info["price"] if deal_info else current_price
                                self._close_decision_for_ticket(
                                    ticket, exit_px, profit, "scalp_tp",
                                    f"Scalp mode: VP trade closed at {self.config.scalp_tp_atr_multiplier}x ATR profit"
                                )
                            else:
                                scalp_status = f"TP_CLOSE_FAILED"

                        # Scalp BE: tighten SL to breakeven quickly
                        elif profit_distance >= scalp_be_dist and current_sl != 0:
                            if direction == "BUY":
                                scalp_be_sl = entry_price + atr * 0.05  # Tiny buffer above entry
                                should_move = scalp_be_sl > current_sl
                            else:
                                scalp_be_sl = entry_price - atr * 0.05
                                should_move = scalp_be_sl < current_sl

                            if should_move:
                                result = modify_position(ticket, sl=round(scalp_be_sl, 5))
                                if result.get("success"):
                                    scalp_status = f"BE_SET {current_sl:.2f}->{scalp_be_sl:.2f}"
                                    action = ManagementAction(
                                        ticket=ticket, symbol=symbol,
                                        action_type="scalp_breakeven",
                                        old_value=current_sl, new_value=scalp_be_sl,
                                        reason=f"Scalp BE: profit {profit_distance:.2f} >= {scalp_be_dist:.2f} ({self.config.scalp_be_atr_multiplier}x ATR)",
                                        success=True,
                                        decision_id=decision.get("decision_id"),
                                    )
                                    self._log_action(action)
                                else:
                                    scalp_status = f"BE_FAILED"
                            else:
                                scalp_status = f"BE_already_set"

                        # Scalp time limit: close after max hours
                        elif open_time:
                            hours_open = (datetime.now().timestamp() - open_time) / 3600 if isinstance(open_time, (int, float)) else 0
                            if hours_open >= self.config.scalp_max_hours:
                                result = close_position(ticket)
                                action = ManagementAction(
                                    ticket=ticket, symbol=symbol,
                                    action_type="scalp_time_close",
                                    old_value=hours_open, new_value=0,
                                    reason=f"Scalp time limit: {hours_open:.1f}h >= {self.config.scalp_max_hours}h",
                                    success=result.get("success", False),
                                    decision_id=decision.get("decision_id"),
                                )
                                self._log_action(action)
                                if action.success:
                                    scalp_status = f"TIME_CLOSED {hours_open:.1f}h"
                                    deal_info = get_closed_deal_by_ticket(ticket, days_back=7)
                                    exit_px = deal_info["price"] if deal_info else current_price
                                    self._close_decision_for_ticket(
                                        ticket, exit_px, profit, "scalp_time_limit",
                                        f"Scalp mode: VP trade closed after {hours_open:.1f}h (limit {self.config.scalp_max_hours}h)"
                                    )
                                else:
                                    scalp_status = f"TIME_CLOSE_FAILED"
                            else:
                                scalp_status = f"waiting({profit_distance:.2f}/{scalp_tp_dist:.2f}, {hours_open:.1f}h/{self.config.scalp_max_hours}h)"
                        else:
                            scalp_status = f"waiting({profit_distance:.2f}/{scalp_tp_dist:.2f})"

                        # Log and skip default management for scalp trades
                        self.logger.info(
                            f"POS #{ticket} {symbol} {direction} [SCALP]: "
                            f"entry={entry_price} cur={current_price} pnl={pnl_pct:+.2f}% "
                            f"| SCALP: {scalp_status}"
                        )
                        continue  # Skip breakeven/trailing/partial — scalp mode handles it

                # --- Breakeven Stop ---
                enable_be = self._get_effective_setting(policy, "enable_breakeven_stop", symbol)
                be_mult = self._get_effective_setting(policy, "breakeven_atr_multiplier", symbol)
                if enable_be:
                    breakeven_threshold = be_mult * atr
                    if direction == "BUY":
                        profit_distance = current_price - entry_price
                    else:
                        profit_distance = entry_price - current_price

                    if profit_distance >= breakeven_threshold and current_sl != 0:
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
                            if is_better:
                                result = modify_position(ticket, sl=breakeven_sl)
                                action = ManagementAction(
                                    ticket=ticket, symbol=symbol,
                                    action_type="breakeven",
                                    old_value=current_sl, new_value=breakeven_sl,
                                    reason=f"Profit {profit_distance:.2f} >= {breakeven_threshold:.2f} (BE threshold)",
                                    success=result.get("success", False),
                                    error=result.get("error") if not result.get("success") else None,
                                    decision_id=decision.get("decision_id") if decision else None,
                                )
                                self._log_action(action)

                                if result.get("success"):
                                    current_sl = breakeven_sl
                                    be_status = f"SET {action.old_value:.2f}->{breakeven_sl:.2f}"
                                    if decision:
                                        add_trade_event(decision["decision_id"], "breakeven_set", {
                                            "old_sl": action.old_value, "new_sl": breakeven_sl,
                                            "price": current_price, "pnl_pct": round(pnl_pct, 3),
                                            "managed_by": "trade_management_agent",
                                        }, source=self.config.instance_name)
                                else:
                                    be_status = f"FAILED: {result.get('error', 'unknown')}"
                                    if result.get("market_closed") or "Market closed" in str(result.get("error", "")):
                                        self._market_closed_until = datetime.now() + timedelta(minutes=30)
                            else:
                                be_status = "not_better"
                        else:
                            be_status = "not_eligible"
                    else:
                        be_status = f"not_reached({profit_distance:.2f}/{breakeven_threshold:.2f})"

                # --- Trailing Stop ---
                enable_trail = self._get_effective_setting(policy, "enable_trailing_stop", symbol)
                if enable_trail and pnl_pct > 0:
                    if trail_mult_override:
                        trail_mult = trail_mult_override
                    elif policy and policy.trailing_stop_atr_multiplier is not None:
                        trail_mult = policy.trailing_stop_atr_multiplier
                    else:
                        trail_mult = self._get_symbol_setting(symbol, "trailing_stop_atr_multiplier", self.config.trailing_stop_atr_multiplier)

                    trail_distance = trail_mult * atr
                    if direction == "BUY":
                        candidate_sl = round(current_price - trail_distance, 5)
                        should_trail = candidate_sl > current_sl
                    else:
                        candidate_sl = round(current_price + trail_distance, 5)
                        should_trail = candidate_sl < current_sl

                    if should_trail:
                        result = modify_position(ticket, sl=candidate_sl)
                        action = ManagementAction(
                            ticket=ticket, symbol=symbol,
                            action_type="trailing_stop",
                            old_value=current_sl, new_value=candidate_sl,
                            reason=f"Trail {trail_mult}x ATR, pnl={pnl_pct:+.2f}%",
                            success=result.get("success", False),
                            error=result.get("error") if not result.get("success") else None,
                            decision_id=decision.get("decision_id") if decision else None,
                        )
                        self._log_action(action)

                        if result.get("success"):
                            self._market_closed_until = None
                            trail_status = f"MOVED {current_sl:.2f}->{candidate_sl:.2f} ({trail_mult}x)"
                            if decision:
                                add_trade_event(decision["decision_id"], "trailing_stop", {
                                    "old_sl": current_sl, "new_sl": candidate_sl,
                                    "price": current_price, "pnl_pct": round(pnl_pct, 3),
                                    "atr_mult": trail_mult,
                                    "managed_by": "trade_management_agent",
                                }, source=self.config.instance_name)
                        else:
                            error_msg = result.get("error", "unknown")
                            trail_status = f"FAILED: {error_msg}"
                            if result.get("market_closed") or "Market closed" in error_msg:
                                self._market_closed_until = datetime.now() + timedelta(minutes=30)
                    else:
                        trail_status = "no_move"
                elif enable_trail:
                    trail_status = f"no_profit(pnl={pnl_pct:.2f}%)"

                # --- Partial Take Profit ---
                enable_ptp = self._get_effective_setting(policy, "enable_partial_tp")
                if enable_ptp and current_sl != 0 and entry_price > 0:
                    ptp_pct = self._get_effective_setting(policy, "partial_tp_percent")
                    ptp_rr = self._get_effective_setting(policy, "partial_tp_rr_ratio")

                    sl_distance = abs(entry_price - current_sl)
                    target_distance = sl_distance * ptp_rr

                    if direction == "BUY":
                        profit_distance = current_price - entry_price
                    else:
                        profit_distance = entry_price - current_price

                    if profit_distance >= target_distance and target_distance > 0:
                        partial_volume = round(volume * (ptp_pct / 100), 2)
                        # Ensure minimum lot size
                        sym_info = get_mt5_symbol_info(symbol)
                        vol_min = sym_info.get("volume_min", 0.01) if sym_info else 0.01
                        vol_step = sym_info.get("volume_step", 0.01) if sym_info else 0.01
                        partial_volume = max(vol_min, round(partial_volume / vol_step) * vol_step)

                        if partial_volume < volume:
                            result = self._partial_close(ticket, symbol, direction, partial_volume)
                            action = ManagementAction(
                                ticket=ticket, symbol=symbol,
                                action_type="partial_tp",
                                old_value=volume, new_value=volume - partial_volume,
                                reason=f"Partial TP {ptp_pct}% at {ptp_rr}R, closed {partial_volume} lots",
                                success=result.get("success", False),
                                error=result.get("error") if not result.get("success") else None,
                                decision_id=decision.get("decision_id") if decision else None,
                            )
                            self._log_action(action)
                            partial_status = "TAKEN" if action.success else f"FAILED: {action.error}"

                            # Log partial close event for MTF analysis
                            if action.success and decision:
                                add_trade_event(decision["decision_id"], "partial_close", {
                                    "volume_closed": partial_volume,
                                    "volume_remaining": volume - partial_volume,
                                    "close_price": current_price,
                                    "pnl_pct_at_partial": round(pnl_pct, 4),
                                    "reason": f"rr_target_{ptp_rr}R",
                                    "managed_by": "trade_management_agent",
                                }, source=self.config.instance_name)
                        else:
                            partial_status = "vol_too_small"
                    else:
                        partial_status = f"not_reached({profit_distance:.2f}/{target_distance:.2f})"

                # --- Time Limit ---
                enable_time = self._get_effective_setting(policy, "enable_time_limit")
                if enable_time and open_time:
                    max_hours = self._get_effective_setting(policy, "max_position_hours")
                    if isinstance(open_time, (int, float)):
                        open_dt = datetime.fromtimestamp(open_time)
                    else:
                        open_dt = open_time
                    hours_open = (datetime.now() - open_dt).total_seconds() / 3600

                    if hours_open >= max_hours:
                        result = close_position(ticket)
                        action = ManagementAction(
                            ticket=ticket, symbol=symbol,
                            action_type="time_close",
                            old_value=hours_open, new_value=0,
                            reason=f"Position open {hours_open:.1f}h >= {max_hours}h limit",
                            success=result.get("success", False),
                            error=result.get("error") if not result.get("success") else None,
                            decision_id=decision.get("decision_id") if decision else None,
                        )
                        self._log_action(action)

                        if action.success:
                            time_status = f"CLOSED after {hours_open:.1f}h"
                            deal_info = get_closed_deal_by_ticket(ticket, days_back=7)
                            exit_px = deal_info["price"] if deal_info else current_price
                            self._close_decision_for_ticket(
                                ticket, exit_px, profit, "time_limit",
                                f"Position closed: exceeded {max_hours}h time limit"
                            )
                        else:
                            time_status = f"CLOSE_FAILED: {action.error}"
                    else:
                        time_status = f"{hours_open:.1f}h/{max_hours}h"

                # Log summary
                self.logger.info(
                    f"POS #{ticket} {symbol} {direction}: "
                    f"entry={entry_price} cur={current_price} pnl={pnl_pct:+.2f}% "
                    f"| BE: {be_status} | TRAIL: {trail_status} | PTP: {partial_status} | TIME: {time_status}"
                )

            self._positions_managed = len(managed_tickets)

            # Check for opposing positions on same symbol
            if self.config.enable_opposing_check:
                await self._check_opposing_positions(positions)

            # Detect closed positions
            await self._detect_closed_positions(managed_tickets)

        except Exception as e:
            import traceback
            self.logger.error(f"Position management error: {e}\n{traceback.format_exc()}")

        self._last_management_cycle = datetime.now()
        self.logger.debug("--- Position management cycle end ---")

    # ------------------------------------------------------------------
    # Opposing position conflict resolution
    # ------------------------------------------------------------------

    async def _check_opposing_positions(self, positions: list) -> None:
        """Detect and analyze opposing positions (BUY + SELL) on the same symbol."""
        from tradingagents.automation.strategy_reviewers import (
            _fetch_ohlcv, _compute_directional_indicators, _compute_atr,
        )
        from tradingagents.trade_decisions import find_decision_by_ticket

        # Group positions by symbol
        by_symbol: Dict[str, list] = {}
        for pos in positions:
            sym = pos.get("symbol", "")
            by_symbol.setdefault(sym, []).append(pos)

        for symbol, sym_positions in by_symbol.items():
            buys = [p for p in sym_positions if p.get("type") == "BUY"]
            sells = [p for p in sym_positions if p.get("type") == "SELL"]

            if not buys or not sells:
                continue  # No opposing positions

            # Analyze each BUY vs SELL pair (take the most significant pair)
            # Use the largest volume position from each side
            buy = max(buys, key=lambda p: p.get("volume", 0))
            sell = max(sells, key=lambda p: p.get("volume", 0))

            buy_ticket = buy["ticket"]
            sell_ticket = sell["ticket"]
            buy_entry = buy.get("price_open", 0)
            sell_entry = sell.get("price_open", 0)
            current_price = buy.get("price_current", 0)

            # P/L percentages
            buy_pnl = ((current_price - buy_entry) / buy_entry * 100) if buy_entry else 0
            sell_pnl = ((sell_entry - current_price) / sell_entry * 100) if sell_entry else 0

            # TP capture percentages
            buy_tp = buy.get("tp", 0)
            sell_tp = sell.get("tp", 0)
            buy_tp_capture = ((current_price - buy_entry) / (buy_tp - buy_entry) * 100) if buy_tp and buy_tp != buy_entry else 0
            sell_tp_capture = ((sell_entry - current_price) / (sell_entry - sell_tp) * 100) if sell_tp and sell_entry != sell_tp else 0

            # Age in hours
            now_ts = datetime.now().timestamp()
            buy_hours = (now_ts - buy.get("time", now_ts)) / 3600
            sell_hours = (now_ts - sell.get("time", now_ts)) / 3600

            # Fetch market indicators
            df = _fetch_ohlcv(symbol, "H1", bars=250)
            indicators = _compute_directional_indicators(df) if df is not None else None

            # --- Score each position (0-100, higher = stronger) ---
            buy_score = 0.0
            sell_score = 0.0

            # Dimension 1: Momentum alignment (30%)
            if indicators:
                adx = indicators["adx"]
                momentum = indicators["momentum"]  # "bullish" or "bearish"

                if adx > 25:
                    # Strong trend — heavily favor aligned position
                    if momentum == "bullish":
                        buy_score += 85
                        sell_score += 15
                    else:
                        buy_score += 15
                        sell_score += 85
                elif adx > 20:
                    # Moderate — slight favor
                    if momentum == "bullish":
                        buy_score += 65
                        sell_score += 35
                    else:
                        buy_score += 35
                        sell_score += 65
                else:
                    # Weak/no trend — neutral
                    buy_score += 50
                    sell_score += 50

                # EMA alignment bonus
                if indicators["ema20"] > indicators["ema50"]:
                    buy_score += 10
                else:
                    sell_score += 10
            else:
                buy_score += 50
                sell_score += 50

            # Dimension 2: Age/exhaustion (20%)
            buy_age_score = max(0, 100 - (buy_hours / 24) * 15 - max(0, buy_tp_capture) * 0.5)
            sell_age_score = max(0, 100 - (sell_hours / 24) * 15 - max(0, sell_tp_capture) * 0.5)
            buy_score += buy_age_score
            sell_score += sell_age_score

            # Dimension 3: Profit asymmetry (25%)
            buy_profit_score = max(0, min(100, 50 + buy_pnl * 5))
            sell_profit_score = max(0, min(100, 50 + sell_pnl * 5))
            # Bonus for remaining upside
            buy_remaining = abs(buy_tp - current_price) / current_price * 100 if buy_tp else 0
            sell_remaining = abs(current_price - sell_tp) / current_price * 100 if sell_tp else 0
            if buy_remaining > 2 * abs(buy_pnl) and buy_remaining > 1:
                buy_profit_score += 15
            if sell_remaining > 2 * abs(sell_pnl) and sell_remaining > 1:
                sell_profit_score += 15
            buy_score += buy_profit_score
            sell_score += sell_profit_score

            # Dimension 4: Hedge cost (10%)
            buy_swap = buy.get("swap", 0)
            sell_swap = sell.get("swap", 0)
            buy_score += max(0, min(100, 50 + buy_swap * 10))
            sell_score += max(0, min(100, 50 + sell_swap * 10))

            # Dimension 5: Regime alignment (15%)
            buy_dec = find_decision_by_ticket(buy_ticket)
            sell_dec = find_decision_by_ticket(sell_ticket)
            buy_strategy = (buy_dec.get("setup_type", "") or "") if buy_dec else ""
            sell_strategy = (sell_dec.get("setup_type", "") or "") if sell_dec else ""

            if indicators and indicators["trend_strength"] == "weak":
                # Low ADX: mean reversion favored
                if "mean_reversion" in buy_strategy:
                    buy_score += 80
                else:
                    buy_score += 40
                if "mean_reversion" in sell_strategy:
                    sell_score += 80
                else:
                    sell_score += 40
            elif indicators and indicators["trend_strength"] == "strong":
                # High ADX: trend following favored
                if "trend" in buy_strategy or "breakout" in buy_strategy:
                    buy_score += 80
                else:
                    buy_score += 50
                if "trend" in sell_strategy or "breakout" in sell_strategy:
                    sell_score += 80
                else:
                    sell_score += 50
            else:
                buy_score += 50
                sell_score += 50

            # --- Normalize to weighted average ---
            # Weights: momentum=30, age=20, profit=25, hedge=10, regime=15 = 100
            # Each dimension contributed raw 0-100 per position, sum is ~500 max
            # Normalize to 0-100
            buy_total = buy_score / 5
            sell_total = sell_score / 5

            score_diff = abs(buy_total - sell_total)
            weaker = "BUY" if buy_total < sell_total else "SELL"
            weaker_ticket = buy_ticket if buy_total < sell_total else sell_ticket
            stronger_ticket = sell_ticket if buy_total < sell_total else buy_ticket

            # Build reason string
            ind_str = ""
            if indicators:
                ind_str = (f"ADX={indicators['adx']:.1f} "
                           f"+DI={indicators['plus_di']:.1f} -DI={indicators['minus_di']:.1f} "
                           f"momentum={indicators['momentum']}")

            reason = (
                f"OPPOSING {symbol}: BUY#{buy_ticket} score={buy_total:.1f} vs "
                f"SELL#{sell_ticket} score={sell_total:.1f} (diff={score_diff:.1f}). "
                f"BUY: pnl={buy_pnl:+.2f}% age={buy_hours:.0f}h tp_capture={buy_tp_capture:.0f}%. "
                f"SELL: pnl={sell_pnl:+.2f}% age={sell_hours:.0f}h tp_capture={sell_tp_capture:.0f}%. "
                f"{ind_str}"
            )

            if score_diff >= self.config.opposing_score_threshold:
                self.logger.warning(
                    f"  {reason} -> RECOMMEND closing {weaker} #{weaker_ticket}"
                )

                action = ManagementAction(
                    ticket=weaker_ticket,
                    symbol=symbol,
                    action_type="opposing_analysis",
                    old_value=None,
                    new_value=None,
                    reason=f"close_hedge: {reason}",
                    success=True,
                )
                self._log_action(action)

                if self.config.auto_resolve_opposing:
                    from tradingagents.dataflows.mt5_data import close_position
                    from tradingagents.trade_decisions import close_decision

                    result = close_position(weaker_ticket)
                    if result.get("success"):
                        self.logger.info(
                            f"  AUTO-CLOSED {weaker} #{weaker_ticket} "
                            f"(profit={result.get('profit', 0):.2f})"
                        )
                        # Close the decision record with rationale
                        dec = find_decision_by_ticket(weaker_ticket)
                        if dec:
                            try:
                                exit_price = current_price
                                close_decision(
                                    dec["decision_id"],
                                    exit_price=exit_price,
                                    outcome_notes=f"Auto-closed by TMA opposing position resolution. {reason}",
                                    exit_reason="opposing_hedge_close",
                                )
                            except Exception as e:
                                self.logger.warning(f"  Failed to close decision: {e}")

                        close_action = ManagementAction(
                            ticket=weaker_ticket,
                            symbol=symbol,
                            action_type="close_hedge",
                            old_value=None,
                            new_value=result.get("profit", 0),
                            reason=reason,
                            success=True,
                        )
                        self._log_action(close_action)
                    else:
                        self.logger.error(f"  Failed to close #{weaker_ticket}: {result.get('error')}")
                else:
                    self.logger.info(
                        f"  Set auto_resolve_opposing=True to auto-close"
                    )
            else:
                self.logger.info(
                    f"  OPPOSING {symbol}: BUY#{buy_ticket}={buy_total:.1f} vs "
                    f"SELL#{sell_ticket}={sell_total:.1f} (diff={score_diff:.1f} < "
                    f"{self.config.opposing_score_threshold}) -> HOLD both"
                )

    # ------------------------------------------------------------------
    # Closed position detection
    # ------------------------------------------------------------------

    async def _detect_closed_positions(self, open_tickets: set) -> None:
        """Detect positions that disappeared (SL/TP hit externally)."""
        try:
            # Include pending orders in open set
            import MetaTrader5 as mt5
            pending_orders = mt5.orders_get()
            if pending_orders:
                for order in pending_orders:
                    open_tickets.add(order.ticket)
        except Exception:
            pass

        try:
            active_decisions = list_active_decisions()
            for dec in active_decisions:
                dec_ticket = dec.get("mt5_ticket")
                if not dec_ticket or dec_ticket in open_tickets:
                    # Clear missing count if ticket reappeared
                    self._missing_ticket_counts.pop(dec_ticket, None)
                    continue

                # Ticket is gone
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
                    add_trade_event(dec["decision_id"], "external_close", {
                        "exit_price": exit_price,
                        "mt5_profit": mt5_profit,
                        "inferred_reason": exit_reason,
                        "deal_type": deal_info.get("type"),
                        "deal_comment": deal_info.get("comment", ""),
                        "managed_by": "trade_management_agent",
                    }, source=self.config.instance_name)
                    self._close_decision_for_ticket(
                        dec_ticket, exit_price, mt5_profit, exit_reason,
                        f"Position closed externally ({exit_reason})"
                    )
                    self._missing_ticket_counts.pop(dec_ticket, None)

                    action = ManagementAction(
                        ticket=dec_ticket, symbol=dec.get("symbol", ""),
                        action_type="closed_detected",
                        reason=f"External close detected: {exit_reason}, pnl={mt5_profit:.2f}",
                        decision_id=dec.get("decision_id"),
                    )
                    self._log_action(action)
                else:
                    self._missing_ticket_counts[dec_ticket] = self._missing_ticket_counts.get(dec_ticket, 0) + 1
                    miss_count = self._missing_ticket_counts[dec_ticket]
                    self.logger.info(
                        f"  Decision {dec['decision_id']} ticket #{dec_ticket} gone "
                        f"but no deal found (check {miss_count}/5)"
                    )
                    if miss_count >= 5:
                        entry_price = dec.get("entry_price", 0)
                        events = dec.get("events", [])
                        executed_evt = next((e for e in events if e.get("type") == "executed"), None)
                        had_fill = any(e.get("type") in ("position_update", "external_close") for e in events)
                        order_never_filled = (
                            executed_evt
                            and not had_fill
                            and executed_evt.get("deal_id", -1) == 0
                        )
                        if not order_never_filled:
                            order_never_filled = not had_fill

                        if order_never_filled:
                            exit_reason = "order_unfilled"
                            note = f"Limit order #{dec_ticket} never filled"
                        else:
                            exit_reason = "stale_cleanup"
                            note = f"Ticket #{dec_ticket} missing for {miss_count} checks"

                        self.logger.warning(f"  Closing stale decision {dec['decision_id']}: {note}")
                        add_trade_event(dec["decision_id"], exit_reason, {
                            "exit_price": entry_price,
                            "mt5_profit": 0,
                            "note": note,
                            "managed_by": "trade_management_agent",
                        }, source=self.config.instance_name)
                        self._close_decision_for_ticket(
                            dec_ticket, entry_price, 0.0, exit_reason, note
                        )
                        del self._missing_ticket_counts[dec_ticket]

        except Exception as e:
            self.logger.error(f"Error detecting closed positions: {e}")

    # ------------------------------------------------------------------
    # Account risk monitoring
    # ------------------------------------------------------------------

    async def _check_account_risk(self) -> None:
        """Account-level risk checks: exposure, correlation."""
        try:
            import MetaTrader5 as mt5
            account = mt5.account_info()
            if not account:
                return

            equity = account.equity
            if equity <= 0:
                return

            positions = get_open_positions()
            if not positions:
                return

            # --- Total exposure check ---
            total_exposure = 0.0
            symbol_exposure: Dict[str, float] = {}
            for pos in positions:
                sym = pos.get("symbol", "")
                vol = pos.get("volume", 0)
                price = pos.get("price_current", 0)

                # Approximate exposure in account currency
                sym_info = get_mt5_symbol_info(sym)
                contract_size = sym_info.get("trade_contract_size", 100000) if sym_info else 100000
                exposure = vol * contract_size * price
                total_exposure += exposure
                symbol_exposure[sym] = symbol_exposure.get(sym, 0) + exposure

            exposure_pct = (total_exposure / equity) * 100 if equity > 0 else 0
            if exposure_pct > self.config.max_account_exposure_pct:
                self._log_risk_alert(
                    "high_exposure", "warning",
                    f"Total exposure {exposure_pct:.1f}% exceeds limit of {self.config.max_account_exposure_pct}%",
                    {"exposure_pct": exposure_pct, "equity": equity, "total_exposure": total_exposure},
                )

            # --- Correlation check ---
            for group_name, group_symbols in self.config.correlation_groups.items():
                correlated_count = 0
                for pos in positions:
                    if pos.get("symbol") in group_symbols:
                        correlated_count += 1

                if correlated_count > self.config.max_correlated_positions:
                    self._log_risk_alert(
                        "correlated_positions", "warning",
                        f"{correlated_count} positions in '{group_name}' group exceeds limit of {self.config.max_correlated_positions}",
                        {"group": group_name, "count": correlated_count, "symbols": group_symbols},
                    )

            self.logger.debug(
                f"Risk check: equity={equity:.2f}, exposure={exposure_pct:.1f}%, "
                f"positions={len(positions)}"
            )

        except Exception as e:
            self.logger.error(f"Account risk check error: {e}")

        self._last_risk_check = datetime.now()

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def _run_assumption_review(self) -> None:
        """Run SMC assumption review on all open positions with linked decisions."""
        if not self.config.enable_assumption_review:
            return

        try:
            from tradingagents.automation.position_assumption_review import (
                review_all_positions,
            )
        except ImportError:
            self.logger.warning("position_assumption_review not available, skipping")
            return

        self.logger.info("Running assumption review on open positions...")

        try:
            reports = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: review_all_positions(
                    source_filter=None,  # Review ALL positions, not just one source
                    symbols=None,
                    timeframe="H1",
                    auto_apply=self.config.assumption_review_auto_apply,
                    use_llm=self.config.assumption_review_use_llm,
                ),
            )

            for r in reports:
                if r.error:
                    self.logger.warning(f"  Review #{r.ticket} {r.symbol}: ERROR - {r.error}")
                    continue

                findings_str = ", ".join(
                    f"[{f.severity}] {f.category}" for f in r.findings
                ) if r.findings else "no issues"

                self.logger.info(
                    f"  Review #{r.ticket} {r.symbol} {r.direction}: "
                    f"pnl={r.pnl_pct:+.2f}% action={r.recommended_action} "
                    f"| {findings_str}"
                )

                # Log critical/warning findings as management actions
                if r.has_critical or r.has_warnings:
                    action = ManagementAction(
                        ticket=r.ticket,
                        symbol=r.symbol,
                        action_type="assumption_review",
                        old_value=r.current_sl if r.suggested_sl else None,
                        new_value=r.suggested_sl or r.suggested_tp,
                        reason=f"{r.recommended_action}: {findings_str}",
                        success=True,
                        decision_id=r.decision_id,
                    )
                    self._log_action(action)

                # Auto-apply SL/TP adjustments if enabled
                if self.config.assumption_review_auto_apply and r.recommended_action != "hold":
                    entry = getattr(r, 'entry_price', None) or 0
                    current = getattr(r, 'current_price', None) or 0
                    direction = getattr(r, 'direction', 'BUY')

                    if r.recommended_action == "adjust_sl" and r.suggested_sl:
                        # SAFETY: SL must be on correct side of CURRENT PRICE
                        # (not entry — a profitable SELL can have SL below entry)
                        sl_valid = (
                            (direction == "BUY" and r.suggested_sl < current) or
                            (direction == "SELL" and r.suggested_sl > current) or
                            current == 0  # Can't validate without price
                        )
                        if not sl_valid:
                            self.logger.warning(
                                f"  BLOCKED SL adjust #{r.ticket}: {r.suggested_sl:.5f} is on WRONG SIDE "
                                f"of current price {current:.5f} for {direction}"
                            )
                        # Skip if no actual change or change is too small (< 0.05%)
                        elif r.current_sl and abs(r.suggested_sl - r.current_sl) / (r.suggested_sl + 1e-10) * 100 < 0.05:
                            pass  # Silent skip — no-op
                        else:
                            result = modify_position(r.ticket, sl=r.suggested_sl)
                            applied = result.get("success", False)
                            self.logger.info(
                                f"  Auto-applied SL adjustment #{r.ticket}: "
                                f"{r.current_sl:.5f} -> {r.suggested_sl:.5f} "
                                f"({'OK' if applied else 'FAILED'})"
                            )
                    elif r.recommended_action == "adjust_tp" and r.suggested_tp:
                        # SAFETY: TP must be on correct side of entry
                        tp_valid = (
                            (direction == "BUY" and r.suggested_tp > entry) or
                            (direction == "SELL" and r.suggested_tp < entry) or
                            entry == 0  # Can't validate without entry
                        )
                        if not tp_valid:
                            self.logger.warning(
                                f"  BLOCKED TP adjust #{r.ticket}: {r.suggested_tp:.5f} is on WRONG SIDE "
                                f"of entry {entry:.5f} for {direction} — would cause instant loss"
                            )
                        # Skip if no actual change or change is too small (< 0.05%)
                        elif r.current_tp and abs(r.suggested_tp - r.current_tp) / (r.suggested_tp + 1e-10) * 100 < 0.05:
                            pass  # Silent skip — no-op
                        else:
                            result = modify_position(r.ticket, tp=r.suggested_tp)
                            applied = result.get("success", False)
                            self.logger.info(
                                f"  Auto-applied TP adjustment #{r.ticket}: "
                                f"{r.current_tp:.5f} -> {r.suggested_tp:.5f} "
                                f"({'OK' if applied else 'FAILED'})"
                            )
                    elif r.recommended_action == "close":
                        self.logger.warning(
                            f"  CLOSE recommended for #{r.ticket} {r.symbol} — "
                            f"not auto-closing (requires manual confirmation)"
                        )

            self.logger.info(f"Assumption review complete: {len(reports)} positions reviewed")

        except Exception as e:
            self.logger.error(f"Assumption review failed: {e}")

    async def _management_loop(self):
        """Position management loop - runs every management_interval_seconds."""
        self.logger.info("Management loop started")
        while self._running:
            try:
                await self._manage_positions()
                await self._run_assumption_review()
            except Exception as e:
                self.logger.error(f"Management loop error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.management_interval_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _account_risk_loop(self):
        """Account risk monitoring loop - runs every risk_check_interval_seconds."""
        self.logger.info("Account risk loop started")
        while self._running:
            try:
                await self._check_account_risk()
            except Exception as e:
                self.logger.error(f"Risk loop error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.risk_check_interval_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _control_loop(self):
        """Poll Postgres for control commands."""
        from tradingagents.storage.automation_control import get_automation_control

        self.logger.info("Control loop started")
        control = get_automation_control()

        while self._running:
            try:
                commands = await control.get_pending_commands(self.config.instance_name)

                for cmd in commands:
                    self.logger.info(f"[CONTROL] Received: {cmd.action} from {cmd.source}")

                    try:
                        await self._apply_control_command(cmd)
                        await control.mark_applied(cmd.command_id)
                        self.logger.info(f"[CONTROL] Applied: {cmd.action}")
                    except Exception as e:
                        await control.mark_failed(cmd.command_id, str(e))
                        self.logger.error(f"[CONTROL] Failed: {cmd.action} - {e}")

                # Update status
                await self._update_remote_status()

            except Exception as e:
                self.logger.error(f"Control loop error: {e!r}")
                if "closed" in str(e).lower() or "connect" in str(e).lower():
                    try:
                        await control._reset_pool()
                        self.logger.info("Control pool reset after connection error")
                    except Exception:
                        pass

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.control_poll_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

        await self._update_remote_status()

    async def _apply_control_command(self, cmd) -> None:
        """Apply a control command."""
        if cmd.action == "stop":
            self.stop()
        elif cmd.action == "pause":
            self._status = "paused"
            self.logger.info("TMA paused — management continues but read-only")
        elif cmd.action == "resume":
            self._status = "running"
            self.logger.info("TMA resumed")
        elif cmd.action == "update_config":
            for key, value in cmd.payload.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    self.logger.info(f"Config updated: {key} = {old_value} -> {value}")
        elif cmd.action == "start":
            self.logger.info("[CONTROL] Start command received — already running")
        elif cmd.action == "restart":
            self.logger.info("[CONTROL] Restart requested — stopping...")
            self.stop()
        else:
            raise ValueError(f"Unknown control action: {cmd.action}")

    async def _update_remote_status(self) -> None:
        """Update status in Postgres for remote monitoring."""
        try:
            from tradingagents.storage.automation_control import get_automation_control

            control = get_automation_control()
            await control.update_status(
                instance_name=self.config.instance_name,
                status=self._status,
                pipeline="trade_management",
                symbols=[],  # TMA manages all symbols
                auto_execute=True,
                active_positions=self._positions_managed,
                error_message=self._error_message,
                config=self.config.to_dict(),
            )
        except Exception as e:
            self.logger.warning(f"Failed to update remote status: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the Trade Management Agent."""
        if self._running:
            self.logger.warning("TMA already running")
            return

        # Check MT5
        mt5_status = check_mt5_autotrading()
        if not mt5_status.get("connected"):
            self._status = "error"
            self._error_message = "MT5 not connected"
            raise RuntimeError("MT5 not connected")

        self._running = True
        self._status = "running"
        self._shutdown_event.clear()
        self._error_message = None

        self.logger.info("=" * 50)
        self.logger.info("TRADE MANAGEMENT AGENT STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Management interval: {self.config.management_interval_seconds}s")
        self.logger.info(f"Risk check interval: {self.config.risk_check_interval_seconds}s")
        self.logger.info(f"Trailing stop: {'enabled' if self.config.enable_trailing_stop else 'disabled'} ({self.config.trailing_stop_atr_multiplier}x ATR)")
        self.logger.info(f"Breakeven: {'enabled' if self.config.enable_breakeven_stop else 'disabled'} ({self.config.breakeven_atr_multiplier}x ATR)")
        self.logger.info(f"Partial TP: {'enabled' if self.config.enable_partial_tp else 'disabled'}")
        self.logger.info(f"Assumption review: {'enabled' if self.config.enable_assumption_review else 'disabled'} (auto_apply={self.config.assumption_review_auto_apply}, llm={self.config.assumption_review_use_llm})")
        self.logger.info(f"Time limit: {'enabled' if self.config.enable_time_limit else 'disabled'} ({self.config.max_position_hours}h)")
        self.logger.info(f"Scalp mode: {'enabled' if self.config.enable_scalp_mode else 'disabled'} (TP={self.config.scalp_tp_atr_multiplier}x ATR, BE={self.config.scalp_be_atr_multiplier}x ATR, max={self.config.scalp_max_hours}h, pipelines={self.config.scalp_pipelines})")
        self.logger.info(f"Opposing check: {'enabled' if self.config.enable_opposing_check else 'disabled'} (auto_resolve={self.config.auto_resolve_opposing}, threshold={self.config.opposing_score_threshold})")
        self.logger.info("=" * 50)

        # Report initial status
        if self.config.enable_remote_control:
            await self._update_remote_status()

        loops = [
            self._management_loop(),
            self._account_risk_loop(),
        ]

        if self.config.enable_remote_control:
            loops.append(self._control_loop())

        try:
            await asyncio.gather(*loops)
        except Exception as e:
            self._status = "error"
            self._error_message = str(e)
            self.logger.error(f"TMA error: {e}")
            raise
        finally:
            self._running = False
            self._status = "stopped"
            if self.config.enable_remote_control:
                await self._update_remote_status()
            self.logger.info("Trade Management Agent stopped")

    def stop(self):
        """Stop the TMA."""
        self.logger.info("Stopping Trade Management Agent...")
        self._running = False
        self._shutdown_event.set()
        try:
            loop = asyncio.get_running_loop()
            if self._pg_pool:
                loop.create_task(self._pg_pool.close())
                self._pg_pool = None
        except RuntimeError:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Get current TMA status."""
        return {
            "status": self._status,
            "running": self._running,
            "error": self._error_message,
            "config": self.config.to_dict(),
            "positions_managed": self._positions_managed,
            "actions_taken": self._actions_taken,
            "last_management_cycle": self._last_management_cycle.isoformat() if self._last_management_cycle else None,
            "last_risk_check": self._last_risk_check.isoformat() if self._last_risk_check else None,
        }
