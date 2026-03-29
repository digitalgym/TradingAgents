"""
FastAPI Backend for TradingAgents Web UI
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Any, Dict
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import traceback

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add backend directory to path for local imports (state_store)
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# Portfolio config path (in project root)
PORTFOLIO_CONFIG_FILE = PROJECT_ROOT / "portfolio_config.yaml"
PORTFOLIO_SCHEDULER_PID_FILE = PROJECT_ROOT / "portfolio_scheduler.pid"
SCHEDULER_STATE_FILE = PROJECT_ROOT / "scheduler_state.json"

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents import trade_decisions
from state_store import AutomationState, LearningCycleState, AnalysisCache, AgentOutputCache
from tradingagents.schemas import (
    TradeAnalysisResult,
    PositionReview,
    QuickPositionReview,
    PortfolioSuggestion,
    SignalType,
    RiskLevel,
    Recommendation,
    generate_json_schemas,
)

# Setup logger for the backend
logger = logging.getLogger("tradingagents.backend")

# Helper to safely get attribute from dict or dataclass object
def safe_attr(obj, attr, default=None):
    """Safely get attribute from dict or dataclass object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# Global state for WebSocket connections
websocket_connections: List[WebSocket] = []
analysis_tasks: dict = {}


# Old trailing stop monitor removed — replaced by Trade Management Agent (TMA)


async def check_and_recover_services():
    """Check if services should be running and recover them if needed."""
    import psutil

    # Check Learning Cycle
    learning_state = LearningCycleState.get_status()
    if learning_state.get("enabled"):
        pid_file = PROJECT_ROOT / "daily_cycle.pid"
        process_alive = False

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                process_alive = psutil.pid_exists(pid)
            except Exception:
                pass

        if not process_alive:
            print("[Auto-Recovery] Learning cycle was enabled but process died. Attempting restart...")
            symbols = learning_state.get("symbols", [])
            run_at = learning_state.get("run_at", 9)
            if symbols:
                try:
                    import subprocess
                    project_root = Path(__file__).parent.parent.parent
                    cmd = [
                        sys.executable,
                        str(project_root / "examples" / "daily_cycle.py"),
                        "--symbols", *symbols,
                        "--run-at", str(run_at),
                    ]
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(project_root),
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                    pid_file.write_text(str(process.pid))
                    print(f"[Auto-Recovery] Learning cycle restarted with PID {process.pid}")
                except Exception as e:
                    print(f"[Auto-Recovery] Failed to restart learning cycle: {e}")

    # Check Portfolio Automation
    automation_state = AutomationState.get_status()
    if automation_state.get("enabled"):
        pid_file = PORTFOLIO_SCHEDULER_PID_FILE
        process_alive = False

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                process_alive = psutil.pid_exists(pid)
            except Exception:
                pass

        if not process_alive:
            print("[Auto-Recovery] Portfolio automation was enabled but process died. Attempting restart...")
            try:
                import subprocess
                process = subprocess.Popen(
                    [sys.executable, "-m", "cli.main", "portfolio", "start"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(Path(__file__).parent.parent.parent)
                )
                print(f"[Auto-Recovery] Portfolio automation restart initiated")
            except Exception as e:
                print(f"[Auto-Recovery] Failed to restart portfolio automation: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("TradingAgents API starting...")

    # Check and recover services that should be running
    await check_and_recover_services()

    yield

    print("TradingAgents API shutting down...")


app = FastAPI(
    title="TradingAgents API",
    description="Web API for TradingAgents multi-agent trading system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler to catch all errors and show full traceback
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_trace = traceback.format_exc()
    print(f"Global exception: {error_trace}", file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_trace}
    )


# ============= Pydantic Models =============

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"
    use_smc: bool = True
    save_decision: bool = True
    force_fresh: bool = False  # Bypass agent output cache when True


class PositionModifyRequest(BaseModel):
    ticket: int
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None


class PositionCloseRequest(BaseModel):
    ticket: int


class PortfolioConfigUpdate(BaseModel):
    # Trading Control
    execution_mode: Optional[str] = None
    max_total_positions: Optional[int] = None
    max_daily_trades: Optional[int] = None
    total_risk_budget_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    # Fine-tuning
    use_atr_stops: Optional[bool] = None
    atr_stop_multiplier: Optional[float] = None
    atr_trailing_multiplier: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    # Schedule
    schedule: Optional[dict] = None
    # Legacy
    symbols: Optional[List[dict]] = None


class MemoryQueryRequest(BaseModel):
    collection: str
    query: str
    n_results: int = 5


class BatchReviewRequest(BaseModel):
    tickets: Optional[List[int]] = None  # If None, review all
    review_mode: str = "llm"  # "llm" or "atr"


class BatchCloseRequest(BaseModel):
    tickets: List[int]


class PositionReviewRequest(BaseModel):
    ticket: int
    review_mode: str = "llm"  # "llm" or "atr"


class TradingPlanRequest(BaseModel):
    plan_text: str
    dry_run: bool = True


class MarketOrderRequest(BaseModel):
    symbol: str
    direction: str  # "BUY" or "SELL"
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "TradingAgents"
    magic: int = 123456


class LimitOrderRequest(BaseModel):
    symbol: str
    direction: str  # "BUY" or "SELL"
    volume: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "TradingAgents"
    magic: int = 123456


class PositionSizeRequest(BaseModel):
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None  # For actual R:R calculation
    risk_amount: Optional[float] = None  # Fixed dollar amount
    risk_percent: Optional[float] = None  # Percentage of account


class SaveDecisionRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    entry_type: str  # "market" or "limit"
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float
    mt5_ticket: Optional[int] = None
    rationale: Optional[str] = None
    risk_percent: Optional[float] = None
    confidence: Optional[float] = None
    analysis_context: Optional[dict] = None


class PositionDeepAnalysisRequest(BaseModel):
    """Request for deep multi-agent analysis of an existing position"""
    ticket: int
    timeframe: str = "H1"
    use_smc: bool = True


# ============= Helper Functions =============

async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    for ws in websocket_connections:
        try:
            await ws.send_json(message)
        except:
            pass


async def broadcast_automation_status(instance_name: str, status: str, **extra):
    """Broadcast automation status change to all connected WebSocket clients.

    This notifies all open browser tabs so loading spinners resolve in real-time.
    """
    message = {
        "type": "automation_status",
        "instance": instance_name,
        "status": status,  # running, stopped, paused, pending_start, error
        **extra,
    }
    for ws in websocket_connections[:]:
        try:
            await ws.send_json(message)
        except Exception:
            pass


def get_mt5_status() -> dict:
    """Get MT5 connection status"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {"connected": False, "error": "Failed to initialize MT5"}

        account_info = mt5.account_info()
        if account_info is None:
            return {"connected": False, "error": "No account info"}

        terminal_info = mt5.terminal_info()

        return {
            "connected": True,
            "account": account_info.login,
            "server": account_info.server,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "free_margin": account_info.margin_free,
            "profit": account_info.profit,
            "auto_trading": terminal_info.trade_allowed if terminal_info else False
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def get_open_positions() -> List[dict]:
    """Get all open MT5 positions"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "swap": pos.swap,
                "open_time": datetime.fromtimestamp(pos.time).isoformat(),
                "magic": pos.magic,
                "comment": pos.comment
            })
        return result
    except Exception as e:
        return []


def get_pending_orders() -> List[dict]:
    """Get all pending MT5 orders"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return []

        orders = mt5.orders_get()
        if orders is None:
            return []

        order_types = {
            2: "BUY_LIMIT",
            3: "SELL_LIMIT",
            4: "BUY_STOP",
            5: "SELL_STOP",
            6: "BUY_STOP_LIMIT",
            7: "SELL_STOP_LIMIT"
        }

        result = []
        for order in orders:
            result.append({
                "ticket": order.ticket,
                "symbol": order.symbol,
                "type": order_types.get(order.type, f"TYPE_{order.type}"),
                "volume": order.volume_current,
                "price": order.price_open,
                "sl": order.sl,
                "tp": order.tp,
                "time_setup": datetime.fromtimestamp(order.time_setup).isoformat(),
                "magic": order.magic,
                "comment": order.comment
            })
        return result
    except Exception as e:
        return []


# ============= API Routes =============

# ----- Status & Overview -----

@app.get("/api/status")
async def get_status():
    """Get overall system status"""
    import psutil

    mt5_status = get_mt5_status()

    # Check portfolio automation status
    automation_status = {"running": False}
    pid_file = PORTFOLIO_SCHEDULER_PID_FILE
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if psutil.pid_exists(pid):
                automation_status = {"running": True, "pid": pid}
        except:
            pass

    # Add persistent state info for automation
    automation_persistent = AutomationState.get_status()
    automation_status["enabled"] = automation_persistent.get("enabled", False)
    automation_status["last_start"] = automation_persistent.get("last_start")
    automation_status["last_stop"] = automation_persistent.get("last_stop")
    automation_status["stop_reason"] = automation_persistent.get("stop_reason")

    # Check daily cycle status
    daily_cycle_status = {"running": False}
    daily_cycle_pid_file = DAILY_CYCLE_PID_FILE
    if daily_cycle_pid_file.exists():
        try:
            pid = int(daily_cycle_pid_file.read_text().strip())
            if psutil.pid_exists(pid):
                daily_cycle_status = {"running": True, "pid": pid}
        except:
            pass

    # Add persistent state info for daily cycle
    learning_persistent = LearningCycleState.get_status()
    daily_cycle_status["enabled"] = learning_persistent.get("enabled", False)
    daily_cycle_status["last_start"] = learning_persistent.get("last_start")
    daily_cycle_status["last_stop"] = learning_persistent.get("last_stop")
    daily_cycle_status["stop_reason"] = learning_persistent.get("stop_reason")

    # Get decision store stats
    active_decisions = trade_decisions.list_active_decisions()
    closed_decisions = trade_decisions.list_closed_decisions(limit=100)

    # Get guardrails status for header alert
    guardrails_status = None
    try:
        from tradingagents.risk.guardrails import RiskGuardrails
        guardrails = RiskGuardrails()
        g_status = guardrails.get_status()
        guardrails_status = {
            "can_trade": g_status.get("can_trade", True),
            "blocked": g_status.get("blocked", False),
            "in_cooldown": g_status.get("in_cooldown", False),
            "cooldown_enabled": g_status.get("cooldown_enabled", True),
            "cooldown_until": g_status.get("cooldown_until"),
            "reason": g_status.get("reason"),
        }
    except Exception:
        pass

    return {
        "mt5": mt5_status,
        "automation": automation_status,
        "daily_cycle": daily_cycle_status,
        "guardrails": guardrails_status,
        "decisions": {
            "total": len(active_decisions) + len(closed_decisions),
            "open": len(active_decisions)
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard overview data"""
    mt5_status = get_mt5_status()
    positions = get_open_positions()
    pending = get_pending_orders()

    # Calculate totals
    total_profit = sum(p["profit"] for p in positions)
    total_swap = sum(p["swap"] for p in positions)

    # Get recent decisions (combine active and closed, sort by date)
    active = trade_decisions.list_active_decisions()
    closed = trade_decisions.list_closed_decisions(limit=5)
    all_decisions = active + closed
    # Sort by created_at and take the 5 most recent
    recent_decisions = sorted(all_decisions, key=lambda d: d.get("created_at", ""), reverse=True)[:5]
    # Transform to match frontend expected format
    recent_decisions = [
        {
            "id": d.get("decision_id"),
            "symbol": d.get("symbol"),
            "signal": d.get("action"),
            "confidence": 0.8,  # Not stored in current schema
            "timestamp": d.get("created_at"),
            "outcome": {"status": d.get("status", "active")}
        }
        for d in recent_decisions
    ]

    return {
        "account": mt5_status,
        "positions": {
            "count": len(positions),
            "total_profit": total_profit,
            "total_swap": total_swap,
            "items": positions[:5]  # Last 5
        },
        "pending_orders": {
            "count": len(pending),
            "items": pending[:5]
        },
        "recent_decisions": recent_decisions,
        "timestamp": datetime.now().isoformat()
    }


# ----- Positions -----

@app.get("/api/positions")
async def list_positions():
    """Get all open positions with trailing stop status and automation source"""
    positions = get_open_positions()

    # Build ticket -> source lookup from active decisions
    ticket_source: dict[int, str] = {}
    try:
        from tradingagents.trade_decisions import list_active_decisions
        for dec in list_active_decisions():
            t = dec.get("mt5_ticket")
            s = dec.get("source")
            if t and s:
                ticket_source[t] = s
    except Exception:
        pass

    # Get ATR per symbol (cached across positions sharing a symbol)
    from tradingagents.risk.stop_loss import get_atr_for_symbol
    atr_cache: dict[str, float | None] = {}

    # Add trailing stop status from TMA policies, ATR, and automation source
    tma_policies = {}
    try:
        store = _get_management_store()
        tma_policies = store.load_all_management_policies()
    except Exception:
        pass

    for pos in positions:
        ticket = pos["ticket"]
        policy = tma_policies.get(ticket, {})
        pos["trailing_active"] = not policy.get("frozen", False) and policy.get("enable_trailing_stop", True) if policy else False
        pos["trailing_distance"] = policy.get("trail_distance") if policy else None
        pos["trailing_best_price"] = policy.get("best_price") if policy else None

        # Automation source tag
        pos["source"] = ticket_source.get(pos["ticket"])

        # ATR for this symbol
        sym = pos.get("symbol")
        if sym and sym not in atr_cache:
            try:
                atr_cache[sym] = get_atr_for_symbol(sym, period=14)
            except Exception:
                atr_cache[sym] = None
        pos["atr"] = atr_cache.get(sym)

    return {"positions": positions, "count": len(positions)}


@app.get("/api/positions/atr")
async def get_positions_atr():
    """Get current ATR for all symbols with open positions"""
    from tradingagents.risk.stop_loss import get_atr_for_symbol
    positions = get_open_positions()
    symbols = list({p["symbol"] for p in positions})
    result = {}
    for sym in symbols:
        try:
            atr = get_atr_for_symbol(sym, period=14)
            result[sym] = round(atr, 5) if atr else None
        except Exception:
            result[sym] = None
    return {"atr": result}


@app.get("/api/signals")
async def list_signals(
    symbol: Optional[str] = None,
    pipeline: Optional[str] = None,
    signal: Optional[str] = None,
    source: Optional[str] = None,
    executed: Optional[bool] = None,
    limit: int = 200,
):
    """Get signals from the signals table with optional filters."""
    try:
        from tradingagents.storage.postgres_store import get_signal_store
        store = get_signal_store()
        signals = store.list_signals(
            symbol=symbol,
            executed=executed,
            pipeline=pipeline,
            signal=signal,
            source=source,
            limit=min(limit, 1000),
        )
        return {"signals": signals, "count": len(signals)}
    except Exception as e:
        return {"signals": [], "count": 0, "error": str(e)}


@app.get("/api/signals/stats")
async def get_signal_stats(symbol: Optional[str] = None):
    """Get signal statistics."""
    try:
        from tradingagents.storage.postgres_store import get_signal_store
        store = get_signal_store()
        stats = store.get_stats(symbol=symbol)
        return stats
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/orders")
async def list_orders():
    """Get all pending orders"""
    orders = get_pending_orders()
    return {"orders": orders, "count": len(orders)}


@app.post("/api/positions/modify")
async def modify_position(request: PositionModifyRequest):
    """Modify position SL/TP"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        position = mt5.positions_get(ticket=request.ticket)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = position[0]
        old_sl = pos.sl
        old_tp = pos.tp
        new_sl = request.new_sl if request.new_sl else pos.sl
        new_tp = request.new_tp if request.new_tp else pos.tp

        # Skip if nothing actually changed
        if new_sl == old_sl and new_tp == old_tp:
            return {"success": True, "message": "No changes needed", "ticket": request.ticket}

        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": request.ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(modify_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Modify failed: {result.comment} (retcode={result.retcode})")

        # Log the modification for reflection/learning
        changes = []
        if old_sl != new_sl:
            changes.append(f"SL: {old_sl} -> {new_sl}")
        if old_tp != new_tp:
            changes.append(f"TP: {old_tp} -> {new_tp}")

        if changes:
            try:
                from tradingagents.trade_decisions import store_decision
                store_decision(
                    symbol=pos.symbol,
                    decision_type="ADJUST",
                    action="MODIFY_SLTP",
                    rationale=f"Manual adjustment via Web UI: {', '.join(changes)}",
                    source="web_ui",
                    stop_loss=new_sl,
                    take_profit=new_tp,
                    mt5_ticket=request.ticket,
                    analysis_context={
                        "old_sl": old_sl,
                        "old_tp": old_tp,
                        "new_sl": new_sl,
                        "new_tp": new_tp,
                        "entry_price": pos.price_open,
                        "current_price": pos.price_current,
                        "profit_at_modify": pos.profit,
                    }
                )
            except Exception as log_err:
                # Don't fail the modification if logging fails
                print(f"Warning: Failed to log modification: {log_err}")

        return {"success": True, "message": "Position modified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/close")
async def close_position(request: PositionCloseRequest):
    """Close a position"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        position = mt5.positions_get(ticket=request.ticket)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = position[0]

        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": request.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "Closed via Web UI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Close failed: {result.comment}")

        return {"success": True, "message": "Position closed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/breakeven/{ticket}")
async def set_position_breakeven(ticket: int):
    """Set stop loss to entry price (breakeven) for a position"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = position[0]
        entry_price = pos.price_open
        direction = "BUY" if pos.type == 0 else "SELL"

        # Get current price to check if breakeven is valid
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            raise HTTPException(status_code=500, detail=f"Could not get tick data for {pos.symbol}")

        current_price = tick.bid if pos.type == 1 else tick.ask

        # Validate breakeven makes sense (position should be in profit)
        if direction == "BUY" and current_price <= entry_price:
            raise HTTPException(status_code=400, detail="Position not in profit - cannot set breakeven")
        if direction == "SELL" and current_price >= entry_price:
            raise HTTPException(status_code=400, detail="Position not in profit - cannot set breakeven")

        # Get symbol info for price formatting
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=500, detail=f"Could not get symbol info for {pos.symbol}")

        # Round entry price to symbol's digits
        breakeven_sl = round(entry_price, symbol_info.digits)

        # Modify position
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": breakeven_sl,
            "tp": pos.tp,  # Keep existing TP
        }

        result = mt5.order_send(modify_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Modify failed: {result.comment}")

        return {
            "success": True,
            "message": f"Stop loss set to breakeven ({breakeven_sl})",
            "new_sl": breakeven_sl
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrailingStopRequest(BaseModel):
    atr_multiplier: float = 1.5  # Default 1.5x ATR for trailing


@app.post("/api/positions/trailing/{ticket}")
async def set_position_trailing(ticket: int, request: TrailingStopRequest = None):
    """Enable a proper trailing stop for a position.

    This sets an initial SL based on ATR and enables automatic trailing
    that will follow price as it moves favorably.
    """
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = position[0]
        entry_price = pos.price_open
        direction = "BUY" if pos.type == 0 else "SELL"

        # Get current price
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            raise HTTPException(status_code=500, detail=f"Could not get tick data for {pos.symbol}")

        current_price = tick.bid if pos.type == 1 else tick.ask

        # Get ATR for trailing distance
        from tradingagents.risk.stop_loss import get_atr_for_symbol
        atr = get_atr_for_symbol(pos.symbol, period=14)
        if atr is None or atr <= 0:
            raise HTTPException(status_code=400, detail="Could not calculate ATR for trailing stop")

        # Get symbol info for price formatting
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=500, detail=f"Could not get symbol info for {pos.symbol}")

        # Calculate trailing stop
        multiplier = request.atr_multiplier if request else 1.5
        trail_distance = atr * multiplier

        if direction == "BUY":
            # For BUY, trailing SL is below current price
            trailing_sl = current_price - trail_distance
            # Don't move SL below entry if not in profit
            if trailing_sl < entry_price:
                trailing_sl = entry_price
        else:
            # For SELL, trailing SL is above current price
            trailing_sl = current_price + trail_distance
            # Don't move SL above entry if not in profit
            if trailing_sl > entry_price:
                trailing_sl = entry_price

        # Round to symbol's digits
        trailing_sl = round(trailing_sl, symbol_info.digits)

        # Only apply if new SL is better than existing
        if pos.sl > 0:
            if direction == "BUY" and trailing_sl <= pos.sl:
                raise HTTPException(
                    status_code=400,
                    detail=f"New trailing SL ({trailing_sl}) is not better than current SL ({pos.sl})"
                )
            if direction == "SELL" and trailing_sl >= pos.sl:
                raise HTTPException(
                    status_code=400,
                    detail=f"New trailing SL ({trailing_sl}) is not better than current SL ({pos.sl})"
                )

        # Modify position to set initial SL
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": trailing_sl,
            "tp": pos.tp,  # Keep existing TP
        }

        result = mt5.order_send(modify_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Modify failed: {result.comment}")

        # Store as TMA policy so the Trade Management Agent picks it up
        try:
            store = _get_management_store()
            store.save_management_policy(ticket, pos.symbol, {
                "ticket": ticket,
                "symbol": pos.symbol,
                "trailing_stop_atr_multiplier": multiplier,
                "enable_trailing_stop": True,
                "trail_distance": trail_distance,
                "best_price": current_price,
                "direction": direction,
            })
        except Exception as e:
            logger.warning(f"Failed to save TMA policy: {e}")

        return {
            "success": True,
            "message": f"Trailing stop enabled at {trail_distance:.2f} ({multiplier}x ATR). SL will follow price automatically.",
            "new_sl": trailing_sl,
            "trailing_active": True,
            "atr": atr,
            "trail_distance": trail_distance
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/positions/trailing/{ticket}")
async def disable_position_trailing(ticket: int):
    """Disable automatic trailing for a position (keeps current SL)."""
    try:
        store = _get_management_store()
        policy = store.load_management_policy(ticket)
        if not policy:
            raise HTTPException(status_code=404, detail="No active trailing stop for this position")

        store.delete_management_policy(ticket)

        return {
            "success": True,
            "message": f"Trailing stop disabled for position {ticket}. Current SL remains in place."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/trailing")
async def get_active_trailing_stops():
    """Get all positions with active trailing stops (from TMA policies)."""
    try:
        store = _get_management_store()
        policies = store.load_all_management_policies()
        return {
            "trailing_stops": [
                {
                    "ticket": int(ticket),
                    "symbol": policy.get("symbol"),
                    "direction": policy.get("direction"),
                    "trail_distance": policy.get("trail_distance"),
                    "atr_multiplier": policy.get("trailing_stop_atr_multiplier"),
                    "best_price": policy.get("best_price"),
                }
                for ticket, policy in policies.items()
                if policy.get("enable_trailing_stop", True) and not policy.get("frozen", False)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/batch-close")
async def batch_close_positions(request: BatchCloseRequest):
    """Close multiple positions at once"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        results = []
        for ticket in request.tickets:
            try:
                position = mt5.positions_get(ticket=ticket)
                if not position:
                    results.append({"ticket": ticket, "success": False, "error": "Not found"})
                    continue

                pos = position[0]
                close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask

                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "price": price,
                    "deviation": 20,
                    "magic": pos.magic,
                    "comment": "Batch closed via Web UI",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    results.append({"ticket": ticket, "success": True})
                else:
                    results.append({"ticket": ticket, "success": False, "error": result.comment})
            except Exception as e:
                results.append({"ticket": ticket, "success": False, "error": str(e)})

        success_count = sum(1 for r in results if r["success"])
        return {
            "success": success_count == len(request.tickets),
            "results": results,
            "closed": success_count,
            "failed": len(request.tickets) - success_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/review/{ticket}")
async def review_position(ticket: int, request: PositionReviewRequest):
    """Review a single position with LLM analysis and get SL/TP suggestions (runs in background)"""
    import asyncio

    task_id = f"review_{ticket}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize task state
    analysis_tasks[task_id] = {
        "status": "running",
        "ticket": ticket,
        "review_mode": request.review_mode,
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "message": f"Starting position review for ticket {ticket}..."
    }

    def run_review_sync():
        """Run position review synchronously in background thread"""
        import traceback
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                analysis_tasks[task_id].update({
                    "status": "error",
                    "error": "MT5 not initialized"
                })
                return

            analysis_tasks[task_id]["message"] = "Fetching position data..."
            analysis_tasks[task_id]["progress"] = 10

            position = mt5.positions_get(ticket=ticket)
            if not position:
                analysis_tasks[task_id].update({
                    "status": "error",
                    "error": "Position not found"
                })
                return

            pos = position[0]

            # Get current price
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                analysis_tasks[task_id].update({
                    "status": "error",
                    "error": f"Could not get tick data for {pos.symbol}"
                })
                return
            current_price = tick.bid if pos.type == 1 else tick.ask

            analysis_tasks[task_id]["message"] = "Calculating ATR analysis..."
            analysis_tasks[task_id]["progress"] = 20

            # Get ATR analysis
            from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
            atr = get_atr_for_symbol(pos.symbol, period=14)
            if atr is None:
                atr = 0.0

            # Position details
            entry = pos.price_open
            sl = pos.sl
            tp = pos.tp
            direction = "BUY" if pos.type == 0 else "SELL"

            if direction == "BUY":
                pnl_pct = ((current_price - entry) / entry) * 100
                sl_distance = entry - sl if sl > 0 else 0
                tp_distance = tp - entry if tp > 0 else 0
            else:
                pnl_pct = ((entry - current_price) / entry) * 100
                sl_distance = sl - entry if sl > 0 else 0
                tp_distance = entry - tp if tp > 0 else 0

            risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0

            analysis_tasks[task_id]["progress"] = 30

            # ATR-based suggestions
            suggestions = {}
            if atr and atr > 0:
                dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)
                result_suggestions = dsl.suggest_stop_adjustment(
                    entry_price=entry,
                    current_price=current_price,
                    current_sl=sl,
                    current_tp=tp,
                    atr=atr,
                    direction=direction,
                )
                # Ensure suggestions is always a dict (suggest_stop_adjustment can return None)
                if result_suggestions is not None:
                    suggestions = result_suggestions

            analysis_tasks[task_id]["progress"] = 40

            # Calculate suggested TP targets based on R:R ratios
            sl_risk = 0
            tp_2r = None
            tp_3r = None
            if atr and atr > 0:
                if direction == "BUY":
                    sl_risk = entry - sl if sl > 0 else atr * 2
                    tp_2r = entry + (sl_risk * 2)  # 2:1 R:R
                    tp_3r = entry + (sl_risk * 3)  # 3:1 R:R
                else:
                    sl_risk = sl - entry if sl > 0 else atr * 2
                    tp_2r = entry - (sl_risk * 2)  # 2:1 R:R
                    tp_3r = entry - (sl_risk * 3)  # 3:1 R:R

            result = {
                "position": {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": direction,
                    "volume": pos.volume,
                    "entry": entry,
                    "current_price": current_price,
                    "sl": sl,
                    "tp": tp,
                    "profit": pos.profit,
                    "pnl_pct": pnl_pct,
                    "risk_reward": risk_reward,
                },
                "atr_analysis": {
                    "atr": atr,
                    "recommendation": suggestions.get("recommendation", "N/A"),
                    "breakeven_sl": suggestions.get("breakeven", {}).get("new_sl") if suggestions.get("breakeven") else None,
                    "trailing_sl": suggestions.get("trailing", {}).get("new_sl") if suggestions.get("trailing") else None,
                    "atr_stop_distance": atr * 2 if atr > 0 else 0,
                    "atr_trail_distance": atr * 1.5 if atr > 0 else 0,
                    "risk_reward": risk_reward,
                    "tp_2r": tp_2r,  # TP for 2:1 R:R
                    "tp_3r": tp_3r,  # TP for 3:1 R:R
                }
            }

            # LLM analysis if requested
            if request.review_mode == "llm":
                analysis_tasks[task_id]["message"] = "Running LLM analysis..."
                analysis_tasks[task_id]["progress"] = 50

                from tradingagents.dataflows.llm_client import get_llm_client, structured_output
                from tradingagents.schemas import QuickPositionReview

                try:
                    client, model, uses_responses = get_llm_client()
                except ValueError as e:
                    result["llm_analysis"] = {"error": str(e)}
                    analysis_tasks[task_id].update({
                        "status": "completed",
                        "progress": 100,
                        "result": result
                    })
                    return

                analysis_tasks[task_id]["message"] = "Fetching SMC context..."
                analysis_tasks[task_id]["progress"] = 60

                # Fetch SMC data for market structure context using reusable function
                from tradingagents.dataflows.smc_utils import get_smc_position_review_context

                smc_review = get_smc_position_review_context(
                    symbol=pos.symbol,
                    direction=direction,
                    entry_price=entry,
                    current_price=current_price,
                    sl=sl,
                    tp=tp,
                    timeframe='H1',
                    lookback=100
                )

                smc_context = smc_review.get('smc_context', '')

                # Store SMC data in result for frontend
                if 'error' not in smc_review:
                    result["smc_context"] = {
                        "bias": smc_review.get('bias'),
                        "bias_aligns": smc_review.get('bias_aligns'),
                        "structure_shift": smc_review.get('structure_shift'),
                        "unmitigated_obs": smc_review.get('unmitigated_obs', 0),
                        "unmitigated_fvgs": smc_review.get('unmitigated_fvgs', 0),
                        "support_count": len(smc_review.get('support_levels', [])),
                        "resistance_count": len(smc_review.get('resistance_levels', [])),
                        "sl_at_risk": smc_review.get('sl_at_risk', False),
                        "sl_risk_reason": smc_review.get('sl_risk_reason'),
                        "suggested_sl": smc_review.get('suggested_sl'),
                        "suggested_tp": smc_review.get('suggested_tp'),
                        "suggested_tp_source": smc_review.get('suggested_tp_source'),
                        "trailing_sl": smc_review.get('trailing_sl'),
                        "trailing_sl_source": smc_review.get('trailing_sl_source'),
                    }

                analysis_tasks[task_id]["message"] = "Generating LLM recommendations..."
                analysis_tasks[task_id]["progress"] = 80

                prompt = f"""You are reviewing an OPEN position. Provide a fresh market assessment based on current structure.

POSITION: {pos.symbol} {direction}
Entry: {entry}, Current: {current_price}
P/L: {pnl_pct:+.2f}% (${pos.profit:.2f})
Current SL: {sl if sl > 0 else 'None'}, Current TP: {tp if tp > 0 else 'None'}
Current R:R: {risk_reward:.2f}:1
Volume: {pos.volume} lots
{smc_context}

TASK: Assess whether current market structure still supports this {direction} position.

Guidelines for your recommendation:
- HOLD if position is well-placed and market structure supports it
- ADJUST if SL/TP should be updated (provide new suggested_sl and/or suggested_tp values)
- CLOSE only if bias has clearly shifted AGAINST position WITH a structure break (CHOCH)
- Place SL below/above nearest SMC support/resistance (Order Blocks, FVGs)
- Set TP at next SMC resistance/support level
- Set suggested_sl/suggested_tp to null ONLY if current levels are well-placed"""

                try:
                    # Use structured output for guaranteed JSON schema compliance
                    parsed = structured_output(
                        client=client,
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert trade manager. Analyze the position and provide your recommendation."},
                            {"role": "user", "content": prompt},
                        ],
                        response_schema=QuickPositionReview,
                        max_tokens=400,
                        temperature=0.3,
                        use_responses_api=uses_responses,
                    )

                    # structured_output returns a Pydantic model instance
                    result["llm_analysis"] = {
                        "recommendation": parsed.recommendation.value,
                        "suggested_sl": parsed.suggested_sl,
                        "suggested_tp": parsed.suggested_tp,
                        "risk_level": parsed.risk_level.value,
                        "reasoning": parsed.reasoning,
                        "model": model,
                        "structured_output": True,
                    }
                except Exception as e:
                    # Fallback if structured output fails
                    import traceback as tb
                    result["llm_analysis"] = {
                        "error": str(e),
                        "trace": tb.format_exc(),
                        "model": model,
                    }

            # Complete task with result
            analysis_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Review completed",
                "result": result
            })

        except Exception as e:
            import sys
            error_trace = traceback.format_exc()
            print(f"Review error: {error_trace}", file=sys.stderr)
            analysis_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "trace": error_trace
            })

    async def run_in_background():
        await asyncio.to_thread(run_review_sync)

    asyncio.create_task(run_in_background())

    return {"task_id": task_id, "status": "started", "message": "Position review started in background"}


@app.post("/api/positions/batch-review")
async def batch_review_positions(request: BatchReviewRequest):
    """Get review analysis for multiple positions (runs in background)"""
    import asyncio

    task_id = f"batch_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize task state
    analysis_tasks[task_id] = {
        "status": "running",
        "tickets": request.tickets,
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "message": "Starting batch position review..."
    }

    def run_batch_review_sync():
        """Run batch review synchronously in background thread"""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                analysis_tasks[task_id].update({
                    "status": "error",
                    "error": "MT5 not initialized"
                })
                return

            analysis_tasks[task_id]["message"] = "Fetching positions..."
            analysis_tasks[task_id]["progress"] = 10

            # Get positions to review
            if request.tickets:
                positions = []
                for ticket in request.tickets:
                    pos = mt5.positions_get(ticket=ticket)
                    if pos:
                        positions.append(pos[0])
            else:
                positions = mt5.positions_get()

            if not positions:
                analysis_tasks[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "result": {"positions": [], "count": 0}
                })
                return

            from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol

            results = []
            total_positions = len(positions)

            for idx, pos in enumerate(positions):
                progress = 10 + int((idx / total_positions) * 80)
                analysis_tasks[task_id]["message"] = f"Analyzing position {idx + 1}/{total_positions}..."
                analysis_tasks[task_id]["progress"] = progress

                tick = mt5.symbol_info_tick(pos.symbol)
                current_price = tick.bid if pos.type == 1 else tick.ask

                entry = pos.price_open
                direction = "BUY" if pos.type == 0 else "SELL"

                if direction == "BUY":
                    pnl_pct = ((current_price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100

                # ATR analysis
                atr = get_atr_for_symbol(pos.symbol, period=14)
                suggestions = {}
                if atr and atr > 0:
                    dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)
                    result_suggestions = dsl.suggest_stop_adjustment(
                        entry_price=entry,
                        current_price=current_price,
                        current_sl=pos.sl,
                        current_tp=pos.tp,
                        atr=atr,
                        direction=direction,
                    )
                    if result_suggestions is not None:
                        suggestions = result_suggestions

                # Safely extract breakeven/trailing (values can be None even if key exists)
                breakeven_data = suggestions.get("breakeven") or {}
                trailing_data = suggestions.get("trailing") or {}

                results.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": direction,
                    "volume": pos.volume,
                    "entry": entry,
                    "current_price": current_price,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "pnl_pct": pnl_pct,
                    "atr": atr,
                    "recommendation": suggestions.get("recommendation", "N/A"),
                    "breakeven_sl": breakeven_data.get("new_sl"),
                    "trailing_sl": trailing_data.get("new_sl"),
                })

            analysis_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": f"Reviewed {len(results)} positions",
                "result": {"positions": results, "count": len(results)}
            })

        except Exception as e:
            import traceback
            analysis_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc()
            })

    async def run_in_background():
        await asyncio.to_thread(run_batch_review_sync)

    asyncio.create_task(run_in_background())

    return {"task_id": task_id, "status": "started", "message": "Batch review started in background"}


@app.delete("/api/orders/{ticket}")
async def cancel_order(ticket: int):
    """Cancel a pending order"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        cancel_request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }

        result = mt5.order_send(cancel_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Cancel failed: {result.comment}")

        return {"success": True, "message": "Order cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orders/batch-cancel")
async def batch_cancel_orders(request: BatchCloseRequest):
    """Cancel multiple pending orders"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        results = []
        for ticket in request.tickets:
            try:
                cancel_request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": ticket,
                }
                result = mt5.order_send(cancel_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    results.append({"ticket": ticket, "success": True})
                else:
                    results.append({"ticket": ticket, "success": False, "error": result.comment})
            except Exception as e:
                results.append({"ticket": ticket, "success": False, "error": str(e)})

        success_count = sum(1 for r in results if r["success"])
        return {
            "success": success_count == len(request.tickets),
            "results": results,
            "cancelled": success_count,
            "failed": len(request.tickets) - success_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Position Deep Analysis (Multi-Agent) -----

@app.post("/api/positions/deep-analysis/{ticket}")
async def start_position_deep_analysis(ticket: int, request: PositionDeepAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start a deep multi-agent analysis for an existing position.

    This runs the same analysis pipeline as new trade analysis but in the context
    of reviewing an existing position. Returns recommendations to HOLD, ADJUST, or CLOSE.
    """
    import asyncio
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise HTTPException(status_code=500, detail="MT5 not initialized")

    # Get the position
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise HTTPException(status_code=404, detail=f"Position {ticket} not found")

    pos = positions[0]
    symbol = pos.symbol
    direction = "BUY" if pos.type == 0 else "SELL"
    entry_price = pos.price_open
    current_sl = pos.sl
    current_tp = pos.tp
    volume = pos.volume

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        raise HTTPException(status_code=500, detail=f"Could not get tick for {symbol}")
    current_price = tick.bid if pos.type == 1 else tick.ask

    # Calculate P/L
    if direction == "BUY":
        pnl_pips = current_price - entry_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pips = entry_price - current_price
        pnl_pct = ((entry_price - current_price) / entry_price) * 100

    task_id = f"position_analysis_{ticket}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Agent steps for UI display
    agent_step_list = [
        {"id": "market_analyst", "title": "Market Analyst", "description": "Analyzing current price action..."},
        {"id": "social_analyst", "title": "Social Sentiment", "description": "Checking social media sentiment..."},
        {"id": "news_analyst", "title": "News Analyst", "description": "Analyzing market news..."},
        {"id": "bull_researcher", "title": "Bull Researcher", "description": "Building bullish case..."},
        {"id": "bear_researcher", "title": "Bear Researcher", "description": "Building bearish case..."},
        {"id": "research_manager", "title": "Research Manager", "description": "Synthesizing research..."},
        {"id": "trader", "title": "Trader Agent", "description": "Reviewing position and formulating adjustment plan..."},
        {"id": "risk_manager", "title": "Risk Manager", "description": "Making final position assessment..."},
    ]

    # Initialize task state
    analysis_tasks[task_id] = {
        "status": "running",
        "progress": 5,
        "current_step": "market_analyst",
        "current_step_title": "Market Analyst",
        "current_step_description": "Analyzing current price action...",
        "steps_completed": [],
        "in_progress_agents": ["market_analyst"],
        "agent_outputs": {},
        "agent_steps": agent_step_list,
        "symbol": symbol,
        "timeframe": request.timeframe,
        "position_context": {
            "ticket": ticket,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": current_price,
            "current_sl": current_sl,
            "current_tp": current_tp,
            "volume": volume,
            "pnl_pct": pnl_pct,
            "pnl_pips": pnl_pips,
            "profit": pos.profit,
        }
    }

    def run_position_analysis_sync():
        """Run position analysis synchronously"""
        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.dataflows.mt5_data import get_asset_type
            from tradingagents.dataflows.smc_utils import get_smc_position_review_context

            asset_type = get_asset_type(symbol)

            config = DEFAULT_CONFIG.copy()
            config["asset_type"] = asset_type

            # Select analysts based on asset type
            if asset_type in ["commodity", "forex"]:
                selected_analysts = ["market", "social", "news"]
            else:
                selected_analysts = ["market", "social", "news", "fundamentals"]

            ta = TradingAgentsGraph(debug=False, config=config, selected_analysts=selected_analysts)

            # Get SMC context for the position
            smc_review = get_smc_position_review_context(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                sl=current_sl,
                tp=current_tp,
                timeframe=request.timeframe
            )

            # Build position context for the analysis
            position_context = f"""
REVIEWING EXISTING POSITION:
- Ticket: #{ticket}
- Direction: {direction}
- Entry Price: {entry_price:.5f}
- Current Price: {current_price:.5f}
- Current SL: {current_sl if current_sl else 'NONE SET - RISK EXPOSURE'}
- Current TP: {current_tp if current_tp else 'NONE SET'}
- Volume: {volume} lots
- P/L: {pnl_pct:+.2f}% (${pos.profit:.2f})

{smc_review.get('smc_context', '')}

TASK: Review this position and recommend one of:
1. HOLD - Position is fine, keep current SL/TP
2. ADJUST - Suggest new SL and/or TP values with reasoning
3. CLOSE - Position should be closed (reversal signals, structure break against position, etc.)

If recommending ADJUST, provide specific price levels for new SL and/or TP.
If recommending CLOSE, explain the urgent reason (reversal signals, overextended, structure break, etc.)
"""

            # Progress callback
            def progress_callback(step_name, step_title, step_description, step_output, progress, completed_steps, agent_outputs=None, in_progress_agents=None):
                analysis_tasks[task_id].update({
                    "progress": progress,
                    "current_step": step_name,
                    "current_step_title": step_title,
                    "current_step_description": step_description,
                    "steps_completed": completed_steps,
                    "in_progress_agents": in_progress_agents or [],
                })
                if agent_outputs:
                    if "agent_outputs" not in analysis_tasks[task_id]:
                        analysis_tasks[task_id]["agent_outputs"] = {}
                    analysis_tasks[task_id]["agent_outputs"].update(agent_outputs)
                elif step_output:
                    if "agent_outputs" not in analysis_tasks[task_id]:
                        analysis_tasks[task_id]["agent_outputs"] = {}
                    analysis_tasks[task_id]["agent_outputs"][step_name] = {
                        "title": step_title,
                        "output": step_output[:1000]
                    }

            # Run analysis with position context
            final_state, decision = ta.propagate_with_progress(
                symbol,
                request.timeframe,
                smc_context=position_context,
                progress_callback=progress_callback
            )

            if final_state is None:
                final_state = {}
            if decision is None:
                decision = {}

            # Extract trading plans (for display, not decision-making)
            trader_plan = final_state.get("trader_investment_plan", "")
            risk_decision = final_state.get("final_trade_decision", "")

            # Get market analysis and research summary for the position manager
            market_analysis = final_state.get("market_report", "")
            research_summary = final_state.get("final_report", "")

            # Update progress: Running Position Manager
            analysis_tasks[task_id].update({
                "progress": 90,
                "current_step": "position_manager",
                "current_step_title": "Position Manager making final decision...",
                "in_progress_agents": ["position_manager"],
            })

            # Use the dedicated Position Manager Agent for position-specific decisions
            # This agent is specifically designed for managing existing positions,
            # unlike the trader agent which is designed for new trade entries.
            from tradingagents.agents.position_manager import create_position_manager_agent
            from tradingagents.dataflows.llm_client import get_llm_client

            llm_client, model, uses_responses = get_llm_client()

            position_manager = create_position_manager_agent(
                llm_client=llm_client,
                model=model,
                use_responses_api=uses_responses,
            )

            # Build position context for the manager
            position_context = {
                'direction': direction,
                'entry_price': entry_price,
                'current_price': current_price,
                'current_sl': current_sl,
                'current_tp': current_tp,
                'pnl_pct': pnl_pct,
                'profit': pos.profit,
                'volume': volume,
            }

            # Build SMC context with support/resistance levels
            smc_context_for_manager = {
                'bias': smc_review.get('bias', 'neutral'),
                'bias_aligns': smc_review.get('bias_aligns', True),
                'structure_shift': smc_review.get('structure_shift', False),
                'sl_at_risk': smc_review.get('sl_at_risk', False),
                'sl_risk_reason': smc_review.get('sl_risk_reason', ''),
                'suggested_sl': smc_review.get('suggested_sl'),
                'suggested_tp': smc_review.get('suggested_tp'),
                'trailing_sl': smc_review.get('trailing_sl'),
                'support_levels': smc_review.get('support_levels', []),
                'resistance_levels': smc_review.get('resistance_levels', []),
            }

            # Get position management decision from dedicated agent
            pm_decision = position_manager(
                position_context=position_context,
                smc_context=smc_context_for_manager,
                market_analysis=market_analysis,
                research_summary=research_summary,
            )

            # Extract decision values
            recommendation = pm_decision.action.value  # HOLD, ADJUST, or CLOSE
            suggested_sl = pm_decision.suggested_sl or pm_decision.trail_sl_to or smc_review.get('suggested_sl')
            suggested_tp = pm_decision.suggested_tp or smc_review.get('suggested_tp')
            close_reason = pm_decision.close_reason

            # Get the trading signal from the original decision (for display only)
            trading_signal = decision.get('signal', 'HOLD').upper() if decision else 'HOLD'

            # Add position manager output for frontend display
            if "agent_outputs" not in analysis_tasks[task_id]:
                analysis_tasks[task_id]["agent_outputs"] = {}
            analysis_tasks[task_id]["agent_outputs"]["position_manager"] = {
                "title": "Position Manager Decision",
                "output": f"**Recommendation:** {recommendation}\n\n**Risk Level:** {pm_decision.risk_assessment}\n\n**Key Factors:**\n" + "\n".join([f"- {f}" for f in pm_decision.key_factors[:3]]) + f"\n\n**Reasoning:**\n{pm_decision.reasoning[:800]}"
            }

            analysis_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "current_step": "complete",
                "current_step_title": "Position Analysis Complete",
                "in_progress_agents": [],  # Clear in-progress
                "decision": decision,
                "position_review": {
                    "recommendation": recommendation,
                    "trading_signal": trading_signal,  # What the trader would do for a NEW trade (informational only)
                    "suggested_sl": suggested_sl,
                    "suggested_tp": suggested_tp,
                    "suggested_trailing_sl": pm_decision.trail_sl_to or smc_review.get('trailing_sl'),
                    "trailing_sl_source": smc_review.get('trailing_sl_source'),
                    "close_reason": close_reason,
                    "bias": smc_review.get('bias'),
                    "bias_aligns": smc_review.get('bias_aligns'),
                    "structure_shift": smc_review.get('structure_shift'),
                    "sl_at_risk": smc_review.get('sl_at_risk'),
                    "sl_risk_reason": smc_review.get('sl_risk_reason'),
                    # Additional fields from position manager
                    "urgency": pm_decision.urgency.value,
                    "risk_assessment": pm_decision.risk_assessment,
                    "key_factors": pm_decision.key_factors,
                    "pm_reasoning": pm_decision.reasoning,
                },
                "trading_plan": {
                    "trader_plan": trader_plan,
                    "risk_decision": risk_decision,
                },
                "smc_context": smc_review,
            })

        except Exception as e:
            import traceback as tb
            analysis_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "traceback": tb.format_exc()
            })

    async def run_in_background():
        await asyncio.to_thread(run_position_analysis_sync)

    asyncio.create_task(run_in_background())

    return {"task_id": task_id, "status": "started", "ticket": ticket}


@app.get("/api/positions/deep-analysis/status/{task_id}")
async def get_position_deep_analysis_status(task_id: str):
    """Get position deep analysis task status"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return analysis_tasks[task_id]


# ----- Trade Execution -----

@app.post("/api/trade/market")
async def place_market_order(request: MarketOrderRequest):
    """Place a market order (immediate execution)"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info
        symbol_info = mt5.symbol_info(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        if not symbol_info.visible:
            if not mt5.symbol_select(request.symbol, True):
                raise HTTPException(status_code=400, detail=f"Failed to select symbol {request.symbol}")

        # Determine order type and price
        if request.direction.upper() == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(request.symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(request.symbol).bid

        # Build order request
        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": request.magic,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if request.stop_loss:
            order_request["sl"] = request.stop_loss
        if request.take_profit:
            order_request["tp"] = request.take_profit

        # Send order
        result = mt5.order_send(order_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(
                status_code=400,
                detail=f"Order failed: {result.comment} (code: {result.retcode})"
            )

        return {
            "success": True,
            "order_ticket": result.order,
            "deal_ticket": result.deal,
            "volume": result.volume,
            "price": result.price,
            "message": f"Market {request.direction} order executed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/limit")
async def place_limit_order(request: LimitOrderRequest):
    """Place a limit order (pullback entry)"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info
        symbol_info = mt5.symbol_info(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        if not symbol_info.visible:
            if not mt5.symbol_select(request.symbol, True):
                raise HTTPException(status_code=400, detail=f"Failed to select symbol {request.symbol}")

        # Get current price to determine order type
        tick = mt5.symbol_info_tick(request.symbol)
        current_price = tick.ask if request.direction.upper() == "BUY" else tick.bid

        # Determine order type based on entry price vs current price
        if request.direction.upper() == "BUY":
            if request.entry_price < current_price:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT  # Buy below current price
            else:
                order_type = mt5.ORDER_TYPE_BUY_STOP  # Buy above current price
        else:
            if request.entry_price > current_price:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT  # Sell above current price
            else:
                order_type = mt5.ORDER_TYPE_SELL_STOP  # Sell below current price

        # Build order request
        order_request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": order_type,
            "price": request.entry_price,
            "deviation": 20,
            "magic": request.magic,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        if request.stop_loss:
            order_request["sl"] = request.stop_loss
        if request.take_profit:
            order_request["tp"] = request.take_profit

        # Send order
        result = mt5.order_send(order_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(
                status_code=400,
                detail=f"Order failed: {result.comment} (code: {result.retcode})"
            )

        order_type_names = {
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
        }

        return {
            "success": True,
            "order_ticket": result.order,
            "order_type": order_type_names.get(order_type, "PENDING"),
            "volume": request.volume,
            "entry_price": request.entry_price,
            "current_price": current_price,
            "message": f"Pending {order_type_names.get(order_type, 'LIMIT')} order placed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/calculate-size")
async def calculate_position_size(request: PositionSizeRequest):
    """Calculate optimal position size based on risk parameters"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get account info
        account = mt5.account_info()
        if account is None:
            raise HTTPException(status_code=500, detail="Failed to get account info")

        # Get symbol info
        symbol_info = mt5.symbol_info(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        # Calculate stop loss distance in price
        sl_distance = abs(request.entry_price - request.stop_loss)

        if sl_distance == 0:
            raise HTTPException(status_code=400, detail="Stop loss cannot be equal to entry price")

        # Determine risk amount
        if request.risk_amount:
            risk_amount = request.risk_amount
        elif request.risk_percent:
            risk_amount = account.balance * (request.risk_percent / 100)
        else:
            # Default to 1% risk
            risk_amount = account.balance * 0.01

        # Calculate position size
        # For forex/commodities: pip_value = (pip_size / price) * lot_size * contract_size
        contract_size = symbol_info.trade_contract_size
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value

        if tick_value > 0 and tick_size > 0:
            # Calculate ticks in SL distance
            ticks_in_sl = sl_distance / tick_size
            # Value per tick per lot
            value_per_tick = tick_value
            # Position size = risk_amount / (ticks * value_per_tick)
            lots = risk_amount / (ticks_in_sl * value_per_tick)
        else:
            # Fallback calculation
            lots = risk_amount / (sl_distance * contract_size)

        # Round to symbol's lot step
        lot_step = symbol_info.volume_step
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max

        lots = max(min_lot, min(max_lot, round(lots / lot_step) * lot_step))

        # Calculate ACTUAL risk/reward based on TP if provided
        actual_rr = None
        actual_profit = None
        rr_warning = None

        if request.take_profit:
            tp_distance = abs(request.take_profit - request.entry_price)
            actual_rr = round(tp_distance / sl_distance, 2) if sl_distance > 0 else 0
            actual_profit = (tp_distance / tick_size) * tick_value * lots if tick_size > 0 else 0

            # Validate R:R - warn if below 1.5
            if actual_rr < 1.0:
                rr_warning = f"POOR R:R ({actual_rr}:1) - You risk more than you can gain. Consider moving TP further or SL closer."
            elif actual_rr < 1.5:
                rr_warning = f"SUBOPTIMAL R:R ({actual_rr}:1) - Below recommended 1.5:1 minimum."

        # Calculate theoretical 2R profit for comparison
        potential_reward_2r = sl_distance * 2
        potential_profit_2r = (potential_reward_2r / tick_size) * tick_value * lots if tick_size > 0 else 0

        return {
            "position_size": round(lots, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": round((risk_amount / account.balance) * 100, 2),
            "sl_distance": round(sl_distance, symbol_info.digits),
            "sl_ticks": round(sl_distance / tick_size) if tick_size > 0 else 0,
            "potential_loss": round(risk_amount, 2),
            "potential_profit_2r": round(potential_profit_2r, 2),  # Theoretical 2R
            "actual_rr": actual_rr,  # Actual R:R ratio
            "actual_profit": round(actual_profit, 2) if actual_profit else None,  # Actual profit at TP
            "rr_warning": rr_warning,  # Warning if R:R is bad
            "min_lot": min_lot,
            "max_lot": max_lot,
            "lot_step": lot_step,
            "account_balance": account.balance,
            "account_equity": account.equity
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade/market-watch")
async def get_market_watch_symbols():
    """Get symbols currently in MT5 Market Watch"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbols = mt5.symbols_get()
        if symbols is None:
            return {"symbols": [], "count": 0}

        market_watch = []
        seen_symbols = set()
        for s in symbols:
            if s.visible and s.name not in seen_symbols:  # visible=True means it's in Market Watch
                seen_symbols.add(s.name)
                tick = mt5.symbol_info_tick(s.name)
                market_watch.append({
                    "symbol": s.name,
                    "description": s.description,
                    "bid": tick.bid if tick else None,
                    "ask": tick.ask if tick else None,
                    "spread": s.spread,
                    "digits": s.digits,
                    "currency_base": s.currency_base,
                    "currency_profit": s.currency_profit,
                    "contract_size": s.trade_contract_size,
                    "volume_min": s.volume_min,
                })

        return {"symbols": market_watch, "count": len(market_watch)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/market-watch/add")
async def add_to_market_watch(request: dict):
    """Add a symbol to MT5 Market Watch"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbol = request.get("symbol", "").upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol required")

        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            # Try to find similar symbols
            all_symbols = mt5.symbols_get()
            similar = [s.name for s in all_symbols if symbol in s.name.upper()][:5]
            raise HTTPException(
                status_code=400,
                detail=f"Symbol '{symbol}' not found. Similar: {similar}"
            )

        # Add to Market Watch
        if not mt5.symbol_select(symbol, True):
            raise HTTPException(status_code=500, detail=f"Failed to add {symbol} to Market Watch")

        return {"success": True, "symbol": symbol, "message": f"{symbol} added to Market Watch"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/market-watch/remove")
async def remove_from_market_watch(request: dict):
    """Remove a symbol from MT5 Market Watch"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbol = request.get("symbol", "").upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol required")

        # Check if there are open positions for this symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot remove {symbol} - has {len(positions)} open position(s)"
            )

        # Check if there are pending orders for this symbol
        orders = mt5.orders_get(symbol=symbol)
        if orders and len(orders) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot remove {symbol} - has {len(orders)} pending order(s)"
            )

        # Remove from Market Watch
        if not mt5.symbol_select(symbol, False):
            raise HTTPException(status_code=500, detail=f"Failed to remove {symbol} from Market Watch")

        return {"success": True, "symbol": symbol, "message": f"{symbol} removed from Market Watch"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade/symbols/search")
async def search_symbols(q: str = ""):
    """Search available MT5 symbols"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbols = mt5.symbols_get()
        if symbols is None:
            return {"symbols": []}

        query = q.upper()
        results = []
        for s in symbols:
            if query in s.name.upper() or query in s.description.upper():
                results.append({
                    "symbol": s.name,
                    "description": s.description,
                    "visible": s.visible,
                })
                if len(results) >= 20:  # Limit results
                    break

        return {"symbols": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade/symbol-info/{symbol}")
async def get_symbol_info(symbol: str):
    """Get trading parameters for a symbol"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found")

        tick = mt5.symbol_info_tick(symbol)

        return {
            "symbol": symbol,
            "bid": tick.bid if tick else None,
            "ask": tick.ask if tick else None,
            "spread": round((tick.ask - tick.bid) / symbol_info.point) if tick else None,
            "digits": symbol_info.digits,
            "point": symbol_info.point,
            "tick_size": symbol_info.trade_tick_size,
            "tick_value": symbol_info.trade_tick_value,
            "contract_size": symbol_info.trade_contract_size,
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step,
            "trade_mode": symbol_info.trade_mode,
            "currency_profit": symbol_info.currency_profit,
            "currency_base": symbol_info.currency_base,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade/market-status/{symbol}")
async def get_market_status(symbol: str):
    """Check if market is open for a symbol"""
    try:
        from tradingagents.dataflows.mt5_data import is_market_open
        status = is_market_open(symbol)

        # Add session info
        session = _get_trading_session()

        return {
            "symbol": symbol,
            "open": status["open"],
            "reason": status["reason"],
            "trade_mode": status["trade_mode"],
            "session": session,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "open": False,
            "reason": f"Error checking market status: {str(e)}",
            "trade_mode": -1,
            "session": "unknown",
        }


@app.get("/api/trade/market-status")
async def get_market_status_multi(symbols: str = "XAUUSD"):
    """Check market status for multiple symbols (comma-separated)"""
    from tradingagents.dataflows.mt5_data import is_market_open
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results = {}
    for sym in symbol_list:
        try:
            status = is_market_open(sym)
            results[sym] = {"open": status["open"], "reason": status["reason"]}
        except Exception as e:
            results[sym] = {"open": False, "reason": str(e)}

    return {"symbols": results, "session": _get_trading_session()}


@app.get("/api/trade/swing-levels/{symbol}")
async def get_swing_levels(symbol: str, direction: str = "BUY", timeframe: str = "H1", lookback: int = 100):
    """
    Get recent swing lows/highs for limit entry suggestions.

    For BUY: returns recent swing lows (support levels below current price)
    For SELL: returns recent swing highs (resistance levels above current price)
    """
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get timeframe constant
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get price data
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, lookback)
        if rates is None or len(rates) < 20:
            raise HTTPException(status_code=400, detail="Insufficient price data")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Get current price and symbol info
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        if tick is None or symbol_info is None:
            raise HTTPException(status_code=400, detail="Cannot get symbol info")

        current_price = tick.ask if direction.upper() == "BUY" else tick.bid
        digits = symbol_info.digits

        # Find swing points using a simple algorithm
        # A swing low: low is lower than N bars before and after
        # A swing high: high is higher than N bars before and after
        swing_window = 3  # bars on each side to confirm swing

        swing_lows = []
        swing_highs = []

        for i in range(swing_window, len(df) - swing_window):
            # Check for swing low
            is_swing_low = True
            for j in range(1, swing_window + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append({
                    'price': float(df['low'].iloc[i]),
                    'time': df['time'].iloc[i].isoformat(),
                    'index': i,
                    'type': 'swing_low'
                })

            # Check for swing high
            is_swing_high = True
            for j in range(1, swing_window + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append({
                    'price': float(df['high'].iloc[i]),
                    'time': df['time'].iloc[i].isoformat(),
                    'index': i,
                    'type': 'swing_high'
                })

        # Filter and sort based on direction
        if direction.upper() == "BUY":
            # For BUY, we want swing lows BELOW current price (support levels)
            relevant_levels = [sl for sl in swing_lows if sl['price'] < current_price]
            # Sort by distance from current price (nearest first), but also consider recency
            relevant_levels.sort(key=lambda x: (current_price - x['price'], -x['index']))
        else:
            # For SELL, we want swing highs ABOVE current price (resistance levels)
            relevant_levels = [sh for sh in swing_highs if sh['price'] > current_price]
            # Sort by distance from current price (nearest first)
            relevant_levels.sort(key=lambda x: (x['price'] - current_price, -x['index']))

        # Get top 3 levels with descriptions
        top_levels = []
        for i, level in enumerate(relevant_levels[:5]):
            # Calculate distance from current price
            distance = abs(current_price - level['price'])
            distance_pct = (distance / current_price) * 100

            # Determine how recent (bars ago)
            bars_ago = len(df) - level['index'] - 1

            top_levels.append({
                'price': round(level['price'], digits),
                'type': level['type'],
                'time': level['time'],
                'bars_ago': bars_ago,
                'distance_pct': round(distance_pct, 2),
                'label': f"Swing {'Low' if level['type'] == 'swing_low' else 'High'} ({bars_ago} bars ago)"
            })

        # Also add recent day's low/high and week's low/high
        # Get daily data for PDL/PDH
        daily_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 5)
        if daily_rates is not None and len(daily_rates) >= 2:
            prev_day = daily_rates[-2]  # Yesterday
            pdl = float(prev_day['low'])
            pdh = float(prev_day['high'])

            if direction.upper() == "BUY" and pdl < current_price:
                top_levels.append({
                    'price': round(pdl, digits),
                    'type': 'pdl',
                    'label': 'Previous Day Low (PDL)',
                    'distance_pct': round(((current_price - pdl) / current_price) * 100, 2),
                    'bars_ago': 0
                })
            elif direction.upper() == "SELL" and pdh > current_price:
                top_levels.append({
                    'price': round(pdh, digits),
                    'type': 'pdh',
                    'label': 'Previous Day High (PDH)',
                    'distance_pct': round(((pdh - current_price) / current_price) * 100, 2),
                    'bars_ago': 0
                })

        # Sort all levels by distance and take top 3
        if direction.upper() == "BUY":
            top_levels.sort(key=lambda x: x['distance_pct'])
        else:
            top_levels.sort(key=lambda x: x['distance_pct'])

        return {
            'symbol': symbol,
            'direction': direction.upper(),
            'current_price': round(current_price, digits),
            'timeframe': timeframe,
            'levels': top_levels[:3],  # Return top 3
            'all_levels': top_levels[:6],  # Return up to 6 for more options
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chart/candles/{symbol}")
async def get_chart_candles(symbol: str, timeframe: str = "H1", bars: int = 100):
    """
    Get OHLC candle data for charting.

    Returns candle data with open, high, low, close, time.
    """
    try:
        import MetaTrader5 as mt5
        import pandas as pd

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get timeframe constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }
        tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        # Limit bars to reasonable amount
        bars = min(bars, 500)

        # Get price data
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=400, detail="No price data available")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Get current tick for live price
        tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits if symbol_info else 5

        candles = []
        for _, row in df.iterrows():
            candles.append({
                'time': row['time'].isoformat(),
                'open': round(float(row['open']), digits),
                'high': round(float(row['high']), digits),
                'low': round(float(row['low']), digits),
                'close': round(float(row['close']), digits),
                'volume': int(row['tick_volume']) if 'tick_volume' in row else 0,
            })

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles,
            'current_price': round(tick.bid, digits) if tick else None,
            'digits': digits,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/save-decision")
async def save_trade_decision(request: SaveDecisionRequest):
    """Save a trade decision for tracking and learning"""
    try:
        from tradingagents.trade_decisions import store_decision, link_decision_to_ticket

        # Determine entry type description
        entry_desc = "Market Entry" if request.entry_type == "market" else "Limit/Pullback Entry"

        # Build rationale with entry type info
        rationale = request.rationale or ""
        if rationale:
            rationale = f"[{entry_desc}] {rationale}"
        else:
            rationale = f"[{entry_desc}] Trade executed via Web UI"

        # Store the decision
        decision_id = store_decision(
            symbol=request.symbol,
            decision_type="OPEN",
            action=request.action,
            rationale=rationale,
            source="web_ui",
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            volume=request.volume,
            mt5_ticket=request.mt5_ticket,
            confidence=request.confidence,
            analysis_context=request.analysis_context,
            position_sizing={
                "risk_percent": request.risk_percent,
                "volume": request.volume,
            } if request.risk_percent else None,
        )

        # If we have an MT5 ticket, link it
        if request.mt5_ticket:
            link_decision_to_ticket(decision_id, request.mt5_ticket)

        return {
            "success": True,
            "decision_id": decision_id,
            "message": f"Decision saved for learning: {decision_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Decisions -----

@app.get("/api/decisions")
async def list_decisions(
    limit: int = 50,
    status: Optional[str] = None,
    symbol: Optional[str] = None
):
    """List trade decisions"""
    # Get both active and closed decisions
    if status == "open" or status == "active":
        decisions = trade_decisions.list_active_decisions(symbol=symbol)
    elif status == "closed":
        decisions = trade_decisions.list_closed_decisions(symbol=symbol, limit=limit)
    elif status == "failed":
        decisions = trade_decisions.list_failed_decisions(symbol=symbol, limit=limit)
    else:
        # Get all
        active = trade_decisions.list_active_decisions(symbol=symbol)
        closed = trade_decisions.list_closed_decisions(symbol=symbol, limit=limit)
        failed = trade_decisions.list_failed_decisions(symbol=symbol, limit=limit)
        decisions = active + closed + failed
        decisions = sorted(decisions, key=lambda d: d.get("created_at", ""), reverse=True)[:limit]

    # Transform to match frontend expected format
    formatted = []
    for d in decisions:
        formatted.append({
            "id": d.get("decision_id"),
            "symbol": d.get("symbol"),
            "signal": d.get("action"),
            "confidence": d.get("confidence") if d.get("confidence") is not None else (d.get("confluence_score", 5) / 10 if d.get("confluence_score") else None),
            "timestamp": d.get("created_at"),
            "entry_price": d.get("entry_price"),
            "stop_loss": d.get("stop_loss"),
            "take_profit": d.get("take_profit"),
            "rationale": d.get("rationale"),
            "setup_type": d.get("setup_type"),
            "higher_tf_bias": d.get("higher_tf_bias"),
            "confluence_score": d.get("confluence_score"),
            "volatility_regime": d.get("volatility_regime"),
            "market_regime": d.get("market_regime"),
            "key_factors": d.get("confluence_factors", []),
            "execution_error": d.get("execution_error"),
            "outcome": {
                "status": d.get("status", "active"),
                "pnl": d.get("pnl"),
                "pnl_percent": d.get("pnl_percent"),
                "was_correct": d.get("was_correct"),
                "exit_price": d.get("exit_price"),
                "exit_reason": d.get("exit_reason")
            }
        })

    return {"decisions": formatted, "count": len(formatted)}


@app.get("/api/decisions/performance")
async def get_performance_stats(
    symbol: Optional[str] = None,
    days: Optional[int] = None,
):
    """Get comprehensive performance statistics for trade evaluation."""
    try:
        closed = trade_decisions.list_closed_decisions(symbol=symbol, limit=1000)
        active = trade_decisions.list_active_decisions(symbol=symbol)

        # Filter by time period if specified
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            closed = [d for d in closed if d.get("exit_date", "") >= cutoff]

        if not closed:
            return {
                "total_closed": 0, "active": len(active),
                "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl": 0, "avg_pnl": 0,
                "avg_win": 0, "avg_loss": 0,
                "best_trade": None, "worst_trade": None,
                "by_symbol": {}, "by_exit_reason": {},
                "equity_curve": [],
                "streaks": {"current_streak": 0, "max_win_streak": 0, "max_loss_streak": 0},
                "quality": {"sl_placement": {}, "tp_placement": {}},
            }

        wins = [d for d in closed if d.get("was_correct")]
        losses = [d for d in closed if not d.get("was_correct")]
        total_pnl = sum(d.get("pnl", 0) for d in closed)

        # By symbol breakdown
        by_symbol = {}
        for d in closed:
            sym = d.get("symbol", "UNKNOWN")
            if sym not in by_symbol:
                by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0}
            by_symbol[sym]["trades"] += 1
            if d.get("was_correct"):
                by_symbol[sym]["wins"] += 1
            by_symbol[sym]["pnl"] += d.get("pnl", 0)
        for sym in by_symbol:
            s = by_symbol[sym]
            s["win_rate"] = s["wins"] / s["trades"] * 100 if s["trades"] else 0

        # By exit reason
        by_exit_reason = {}
        for d in closed:
            reason = d.get("exit_reason") or "unknown"
            if reason not in by_exit_reason:
                by_exit_reason[reason] = {"count": 0, "pnl": 0}
            by_exit_reason[reason]["count"] += 1
            by_exit_reason[reason]["pnl"] += d.get("pnl", 0)

        # Equity curve (cumulative P/L over time)
        equity_curve = []
        cum_pnl = 0
        for d in sorted(closed, key=lambda x: x.get("exit_date", "")):
            cum_pnl += d.get("pnl", 0)
            equity_curve.append({
                "date": d.get("exit_date", "")[:10],
                "pnl": round(cum_pnl, 2),
                "symbol": d.get("symbol"),
                "trade_pnl": round(d.get("pnl", 0), 2),
            })

        # Win/loss streaks
        sorted_decs = sorted(closed, key=lambda d: d.get("exit_date", ""))
        current_streak = 0
        max_win = 0
        max_loss = 0
        temp_streak = 0
        for d in sorted_decs:
            if d.get("was_correct"):
                temp_streak = temp_streak + 1 if temp_streak > 0 else 1
                max_win = max(max_win, temp_streak)
            else:
                temp_streak = temp_streak - 1 if temp_streak < 0 else -1
                max_loss = max(max_loss, abs(temp_streak))
            current_streak = temp_streak

        # Trade quality from structured outcome
        sl_placement = {"too_tight": 0, "appropriate": 0, "too_wide": 0}
        tp_placement = {"too_ambitious": 0, "appropriate": 0, "too_conservative": 0}
        for d in closed:
            outcome = d.get("structured_outcome", {}) or {}
            sl_p = outcome.get("sl_placement")
            tp_p = outcome.get("tp_placement")
            if sl_p in sl_placement:
                sl_placement[sl_p] += 1
            if tp_p in tp_placement:
                tp_placement[tp_p] += 1

        def _trade_summary(d):
            return {
                "decision_id": d.get("decision_id"),
                "symbol": d.get("symbol"),
                "action": d.get("action"),
                "pnl": d.get("pnl"),
                "pnl_percent": d.get("pnl_percent"),
                "exit_reason": d.get("exit_reason"),
                "exit_date": d.get("exit_date"),
            }

        return {
            "total_closed": len(closed),
            "active": len(active),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(closed), 2),
            "avg_win": round(sum(d.get("pnl", 0) for d in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(d.get("pnl", 0) for d in losses) / len(losses), 2) if losses else 0,
            "best_trade": _trade_summary(max(closed, key=lambda d: d.get("pnl", 0))),
            "worst_trade": _trade_summary(min(closed, key=lambda d: d.get("pnl", 0))),
            "by_symbol": by_symbol,
            "by_exit_reason": by_exit_reason,
            "equity_curve": equity_curve,
            "streaks": {
                "current_streak": current_streak,
                "max_win_streak": max_win,
                "max_loss_streak": max_loss,
            },
            "quality": {
                "sl_placement": sl_placement,
                "tp_placement": tp_placement,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/stats")
async def get_decision_stats():
    """Get decision statistics"""
    try:
        all_decisions = trade_decisions.list_decisions(limit=1000)

        total = len(all_decisions)
        closed = [d for d in all_decisions if d.get("status") == "closed" or d.get("outcome")]
        open_decisions = [d for d in all_decisions if d.get("status") == "active" or not d.get("outcome")]

        wins = sum(1 for d in closed if d.get("was_correct") or (d.get("outcome") and d.get("outcome", {}).get("was_correct")))
        losses = len(closed) - wins

        by_symbol = {}
        for d in all_decisions:
            sym = d.get("symbol", "UNKNOWN")
            if sym not in by_symbol:
                by_symbol[sym] = {"total": 0, "wins": 0, "pnl": 0}
            by_symbol[sym]["total"] += 1
            if d.get("was_correct") or (d.get("outcome") and d.get("outcome", {}).get("was_correct")):
                by_symbol[sym]["wins"] += 1
            pnl = d.get("pnl") or (d.get("outcome", {}).get("pnl") if d.get("outcome") else 0) or 0
            by_symbol[sym]["pnl"] += pnl

        total_pnl = sum(d.get("pnl") or (d.get("outcome", {}).get("pnl") if d.get("outcome") else 0) or 0 for d in closed)

        return {
            "total_decisions": total,
            "open_decisions": len(open_decisions),
            "closed_decisions": len(closed),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(closed) * 100) if closed else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0,
            "by_symbol": by_symbol,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/decisions/mtf-analysis-status")
async def get_mtf_analysis_status():
    """
    Check status of MTF data collection for partial close analysis.

    We're collecting data to answer: "Do H1 levels cause pullbacks on D1 trades?"
    Review when ~50 trades have closed with MTF data.
    """
    try:
        all_decisions = trade_decisions.list_decisions(limit=1000)

        # Count trades with MTF data
        with_timeframe = [d for d in all_decisions if d.get("timeframe")]
        with_mtf_context = [d for d in all_decisions if d.get("mtf_context")]

        closed_with_mtf = [
            d for d in with_timeframe
            if d.get("status") == "closed"
        ]

        # Count MTF conflict events
        mtf_conflicts = 0
        opposing_events = 0
        for d in all_decisions:
            events = d.get("events", [])
            for e in events:
                if e.get("type") == "mtf_conflict":
                    mtf_conflicts += 1
                elif e.get("type") == "opposing_position_opened":
                    opposing_events += 1

        # Count trades with excursion data
        with_excursion = [
            d for d in closed_with_mtf
            if d.get("structured_outcome", {}).get("max_favorable_pips") is not None
        ]

        ready_for_analysis = len(closed_with_mtf) >= 50

        return {
            "status": "ready_for_analysis" if ready_for_analysis else "collecting_data",
            "trades_with_timeframe": len(with_timeframe),
            "trades_with_mtf_context": len(with_mtf_context),
            "closed_with_mtf_data": len(closed_with_mtf),
            "with_excursion_data": len(with_excursion),
            "mtf_conflict_events": mtf_conflicts,
            "opposing_position_events": opposing_events,
            "target_for_analysis": 50,
            "progress_pct": min(100, round(len(closed_with_mtf) / 50 * 100, 1)),
            "message": (
                f"Ready for analysis! {len(closed_with_mtf)} trades with MTF data."
                if ready_for_analysis else
                f"Collecting data: {len(closed_with_mtf)}/50 closed trades with MTF data. "
                f"{mtf_conflicts} MTF conflicts logged, {opposing_events} opposing position events."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/decisions/reconcile")
async def reconcile_decisions_endpoint():
    """Reconcile active decisions against MT5 closed positions."""
    try:
        from tradingagents.trade_decisions import reconcile_decisions
        reconciled = reconcile_decisions(days_back=14)
        return {
            "reconciled_count": len(reconciled),
            "reconciled": [
                {
                    "decision_id": d.get("decision_id"),
                    "symbol": d.get("symbol"),
                    "pnl": d.get("pnl"),
                    "exit_reason": d.get("exit_reason"),
                }
                for d in reconciled
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}")
async def get_decision(decision_id: str):
    """Get a specific decision"""
    try:
        decision = trade_decisions.load_decision(decision_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Decision not found")

    # Transform to match frontend expected format
    return {
        "id": decision.get("decision_id"),
        "symbol": decision.get("symbol"),
        "signal": decision.get("action"),
        "confidence": decision.get("confidence") if decision.get("confidence") is not None else (decision.get("confluence_score", 5) / 10 if decision.get("confluence_score") else None),
        "timestamp": decision.get("created_at"),
        "entry_price": decision.get("entry_price"),
        "stop_loss": decision.get("stop_loss"),
        "take_profit": decision.get("take_profit"),
        "rationale": decision.get("rationale"),
        "setup_type": decision.get("setup_type"),
        "higher_tf_bias": decision.get("higher_tf_bias"),
        "confluence_score": decision.get("confluence_score"),
        "confluence_factors": decision.get("confluence_factors", []),
        "volatility_regime": decision.get("volatility_regime"),
        "market_regime": decision.get("market_regime"),
        "session": decision.get("session"),
        "key_factors": decision.get("confluence_factors", []),
        "execution_error": decision.get("execution_error"),
        "outcome": {
            "status": decision.get("status", "active"),
            "pnl": decision.get("pnl"),
            "pnl_percent": decision.get("pnl_percent"),
            "was_correct": decision.get("was_correct"),
            "exit_price": decision.get("exit_price"),
            "exit_date": decision.get("exit_date"),
            "exit_reason": decision.get("exit_reason"),
            "rr_planned": decision.get("rr_planned"),
            "rr_realized": decision.get("rr_realized")
        },
        "learning": {
            "reward_signal": decision.get("reward_signal"),
            "sharpe_contribution": decision.get("sharpe_contribution"),
            "drawdown_impact": decision.get("drawdown_impact"),
            "pattern_tags": decision.get("pattern_tags", [])
        }
    }


class CloseDecisionRequest(BaseModel):
    exit_price: float
    outcome: str  # "win", "loss", "breakeven"
    notes: Optional[str] = None


@app.post("/api/decisions/{decision_id}/close")
async def close_decision(decision_id: str, request: CloseDecisionRequest):
    """Close/reflect a decision with outcome"""
    try:
        decision = trade_decisions.load_decision(decision_id)

        # Calculate P/L
        entry = decision.get("entry_price", 0)
        direction = decision.get("action", "BUY")
        if direction == "BUY":
            pnl_pct = ((request.exit_price - entry) / entry * 100) if entry else 0
        else:
            pnl_pct = ((entry - request.exit_price) / entry * 100) if entry else 0

        was_correct = request.outcome == "win" or (request.outcome == "breakeven" and pnl_pct >= 0)

        # Close the decision
        trade_decisions.close_decision(
            decision_id,
            exit_price=request.exit_price,
            outcome_notes=request.notes or f"Manually closed via Web UI: {request.outcome}"
        )

        return {
            "success": True,
            "decision_id": decision_id,
            "pnl_percent": pnl_pct,
            "was_correct": was_correct,
            "message": f"Decision closed as {request.outcome}"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Decision not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}/retry-info")
async def get_retry_info(decision_id: str):
    """Get failed decision info + current market price for retry"""
    try:
        decision = trade_decisions.load_decision(decision_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Decision not found")

    if decision.get("status") not in ("failed", "retried"):
        raise HTTPException(status_code=400, detail="Only failed decisions can be retried")

    symbol = decision.get("symbol", "")
    current_price = None
    try:
        from tradingagents.dataflows.mt5_data import get_mt5_current_price
        price_info = get_mt5_current_price(symbol)
        current_price = {
            "bid": price_info.get("bid"),
            "ask": price_info.get("ask"),
        }
    except Exception:
        pass

    return {
        "decision_id": decision_id,
        "symbol": symbol,
        "signal": decision.get("action"),
        "original_entry": decision.get("entry_price"),
        "stop_loss": decision.get("stop_loss"),
        "take_profit": decision.get("take_profit"),
        "volume": decision.get("volume"),
        "rationale": decision.get("rationale"),
        "execution_error": decision.get("execution_error"),
        "failed_at": decision.get("created_at"),
        "current_price": current_price,
    }


@app.post("/api/decisions/{decision_id}/mark-retried")
async def mark_decision_retried(decision_id: str):
    """Mark a failed decision as retried after successful re-execution"""
    try:
        decision = trade_decisions.load_decision(decision_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Decision not found")

    if decision.get("status") != "failed":
        raise HTTPException(status_code=400, detail="Only failed decisions can be marked as retried")

    decision["status"] = "retried"
    decision["retried_at"] = datetime.now().isoformat()

    decision_file = Path(trade_decisions.DECISIONS_DIR) / f"{decision_id}.json"
    with open(decision_file, "w") as f:
        json.dump(decision, f, indent=2, default=str)

    return {"success": True, "message": f"Decision {decision_id} marked as retried"}




# ----- Analysis -----

@app.post("/api/analysis/run")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis (runs in background using asyncio.create_task with to_thread)"""
    import asyncio

    task_id = f"analysis_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Define the agent steps for UI display
    agent_step_list = [
        {"id": "market_analyst", "title": "Market Analyst", "description": "Analyzing price action and technical indicators..."},
        {"id": "social_analyst", "title": "Social Sentiment", "description": "Analyzing social media sentiment..."},
        {"id": "news_analyst", "title": "News Analyst", "description": "Analyzing market news..."},
        {"id": "bull_researcher", "title": "Bull Researcher", "description": "Building bullish case..."},
        {"id": "bear_researcher", "title": "Bear Researcher", "description": "Building bearish case..."},
        {"id": "research_manager", "title": "Research Manager", "description": "Synthesizing research debate..."},
        {"id": "trader", "title": "Trader Agent", "description": "Formulating trading plan..."},
        {"id": "risk_manager", "title": "Risk Manager", "description": "Making final risk assessment..."},
    ]

    # Initialize task state immediately - show first agent as running
    analysis_tasks[task_id] = {
        "status": "running",
        "progress": 5,
        "current_step": "market_analyst",
        "current_step_title": "Market Analyst",
        "current_step_description": "Analyzing price action and technical indicators...",
        "steps_completed": [],
        "in_progress_agents": ["market_analyst"],  # First agent starts immediately
        "agent_outputs": {},
        "agent_steps": agent_step_list,
        "symbol": request.symbol,
        "timeframe": request.timeframe
    }

    def run_analysis_sync():
        """Run analysis synchronously (called via asyncio.to_thread for proper isolation)"""
        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.dataflows.mt5_data import get_asset_type
            from tradingagents.dataflows.smc_utils import analyze_multi_timeframe_smc, format_smc_for_prompt

            # Determine asset type for the symbol to select correct analysts
            asset_type = get_asset_type(request.symbol)

            config = DEFAULT_CONFIG.copy()
            config["asset_type"] = asset_type  # Set explicit asset type

            # Select analysts based on asset type (exclude fundamentals for commodities/forex)
            if asset_type in ["commodity", "forex"]:
                selected_analysts = ["market", "social", "news"]
            else:
                selected_analysts = ["market", "social", "news", "fundamentals"]

            ta = TradingAgentsGraph(debug=False, config=config, selected_analysts=selected_analysts)

            # Run SMC analysis if requested
            smc_context_str = None
            smc_raw_data = None
            if request.use_smc:
                try:
                    # Map timeframe to multi-timeframe set
                    tf_upper = request.timeframe.upper()
                    if tf_upper in ['M15', 'M30', 'H1', '1H']:
                        timeframes = ['1H', '4H', 'D1']
                    elif tf_upper in ['H4', '4H']:
                        timeframes = ['4H', 'D1', 'W1']
                    else:
                        timeframes = ['D1', 'W1']

                    smc_raw_data = analyze_multi_timeframe_smc(request.symbol, timeframes)
                    if smc_raw_data:
                        smc_context_str = format_smc_for_prompt(smc_raw_data, request.symbol)
                except Exception as e:
                    # Log but don't fail if SMC analysis fails
                    print(f"SMC analysis failed: {e}")

            # Progress callback to update task state with real-time agent progress
            def progress_callback(step_name, step_title, step_description, step_output, progress, completed_steps, agent_outputs=None, in_progress_agents=None):
                analysis_tasks[task_id].update({
                    "progress": progress,
                    "current_step": step_name,
                    "current_step_title": step_title,
                    "current_step_description": step_description,
                    "steps_completed": completed_steps,
                    "in_progress_agents": in_progress_agents or [],
                })
                if agent_outputs:
                    # Merge with existing outputs (don't overwrite)
                    if "agent_outputs" not in analysis_tasks[task_id]:
                        analysis_tasks[task_id]["agent_outputs"] = {}
                    analysis_tasks[task_id]["agent_outputs"].update(agent_outputs)
                elif step_output:
                    if "agent_outputs" not in analysis_tasks[task_id]:
                        analysis_tasks[task_id]["agent_outputs"] = {}
                    analysis_tasks[task_id]["agent_outputs"][step_name] = {
                        "title": step_title,
                        "output": step_output[:1000]
                    }

            # Run analysis with progress tracking and SMC context
            final_state, decision = ta.propagate_with_progress(
                request.symbol,
                request.timeframe,
                smc_context=smc_context_str,
                progress_callback=progress_callback,
                force_fresh=request.force_fresh
            )

            # Handle case where final_state is None (analysis failed or returned no state)
            if final_state is None:
                final_state = {}

            # Handle case where decision is None
            if decision is None:
                decision = {}

            # Extract the full trading plans for display
            trader_plan = final_state.get("trader_investment_plan", "")
            risk_decision = final_state.get("final_trade_decision", "")
            investment_plan = final_state.get("investment_plan", "")

            # Extract SMC levels for pullback entry suggestions
            # Use smc_raw_data (the structured dict) instead of final_state["smc_context"] (which is a string)
            smc_levels = []

            # smc_raw_data is the structured dict from analyze_multi_timeframe_smc()
            # final_state["smc_context"] is a formatted string for the LLM prompt - don't use it
            smc_data = smc_raw_data if smc_raw_data and isinstance(smc_raw_data, dict) else {}

            # Safely extract order blocks (handle None values)
            order_blocks = smc_data.get("order_blocks") if isinstance(smc_data, dict) else {}
            order_blocks = order_blocks or {}
            for ob in (order_blocks.get("bullish") if isinstance(order_blocks, dict) else []):
                if ob:
                    ob_top = safe_attr(ob, "top", 0)
                    ob_bottom = safe_attr(ob, "bottom", 0)
                    smc_levels.append({
                        "type": "order_block",
                        "price": (ob_top + ob_bottom) / 2,
                        "direction": "bullish",
                        "strength": safe_attr(ob, "strength", 0.5),
                        "description": f"Bullish OB at {ob_bottom:.5f}-{ob_top:.5f}"
                    })
            for ob in (order_blocks.get("bearish") if isinstance(order_blocks, dict) else []):
                if ob:
                    ob_top = safe_attr(ob, "top", 0)
                    ob_bottom = safe_attr(ob, "bottom", 0)
                    smc_levels.append({
                        "type": "order_block",
                        "price": (ob_top + ob_bottom) / 2,
                        "direction": "bearish",
                        "strength": safe_attr(ob, "strength", 0.5),
                        "description": f"Bearish OB at {ob_bottom:.5f}-{ob_top:.5f}"
                    })

            # Safely extract FVGs (handle None values)
            fair_value_gaps = smc_data.get("fair_value_gaps") if isinstance(smc_data, dict) else {}
            fair_value_gaps = fair_value_gaps or {}
            for fvg in (fair_value_gaps.get("bullish") if isinstance(fair_value_gaps, dict) else []):
                if fvg:
                    fvg_top = safe_attr(fvg, "top", 0)
                    fvg_bottom = safe_attr(fvg, "bottom", 0)
                    smc_levels.append({
                        "type": "fvg",
                        "price": (fvg_top + fvg_bottom) / 2,
                        "direction": "bullish",
                        "strength": 0.7,
                        "description": f"Bullish FVG at {fvg_bottom:.5f}-{fvg_top:.5f}"
                    })
            for fvg in (fair_value_gaps.get("bearish") if isinstance(fair_value_gaps, dict) else []):
                if fvg:
                    fvg_top = safe_attr(fvg, "top", 0)
                    fvg_bottom = safe_attr(fvg, "bottom", 0)
                    smc_levels.append({
                        "type": "fvg",
                        "price": (fvg_top + fvg_bottom) / 2,
                        "direction": "bearish",
                        "strength": 0.7,
                        "description": f"Bearish FVG at {fvg_bottom:.5f}-{fvg_top:.5f}"
                    })

            # Safely extract liquidity zones (handle None values)
            liquidity_zones = smc_data.get("liquidity_zones") if isinstance(smc_data, dict) else []
            for zone in (liquidity_zones or []):
                if zone:
                    zone_price = safe_attr(zone, "price", 0)
                    zone_type = safe_attr(zone, "type", "")
                    smc_levels.append({
                        "type": "liquidity",
                        "price": zone_price,
                        "direction": "bullish" if zone_type == "support" else "bearish",
                        "strength": safe_attr(zone, "strength", 0.5),
                        "description": f"Liquidity zone at {zone_price:.5f}"
                    })

            # Add nearest support/resistance as pullback entry levels
            if smc_raw_data and isinstance(smc_raw_data, dict):
                # Get levels from primary timeframe (1H typically)
                for tf_name, tf_data in smc_raw_data.items():
                    if not isinstance(tf_data, dict):
                        continue

                    # Add nearest support (good for BUY pullback)
                    nearest_support = tf_data.get("nearest_support")
                    if nearest_support:
                        support_price = safe_attr(nearest_support, "top", 0) or safe_attr(nearest_support, "bottom", 0)
                        if support_price > 0:
                            smc_levels.append({
                                "type": "support",
                                "price": support_price,
                                "direction": "bullish",
                                "strength": safe_attr(nearest_support, "strength", 0.7),
                                "description": f"Nearest Support ({tf_name}): {support_price:.5f}"
                            })

                    # Add nearest resistance (good for SELL pullback)
                    nearest_resistance = tf_data.get("nearest_resistance")
                    if nearest_resistance:
                        resistance_price = safe_attr(nearest_resistance, "bottom", 0) or safe_attr(nearest_resistance, "top", 0)
                        if resistance_price > 0:
                            smc_levels.append({
                                "type": "resistance",
                                "price": resistance_price,
                                "direction": "bearish",
                                "strength": safe_attr(nearest_resistance, "strength", 0.7),
                                "description": f"Nearest Resistance ({tf_name}): {resistance_price:.5f}"
                            })

                    # Only process first timeframe for now to avoid duplicates
                    break

            # Add PDL/PDH from MT5 D1 data for pullback entries
            try:
                rates = mt5.copy_rates_from_pos(request.symbol, mt5.TIMEFRAME_D1, 0, 2)
                if rates is not None and len(rates) >= 2:
                    prev_day = rates[-2]  # Yesterday's candle
                    pdl = prev_day['low']
                    pdh = prev_day['high']

                    # PDL - good for BUY pullback entry
                    smc_levels.append({
                        "type": "pdl",
                        "price": pdl,
                        "direction": "bullish",
                        "strength": 0.8,
                        "description": f"Previous Day Low: {pdl:.5f}"
                    })

                    # PDH - good for SELL pullback entry
                    smc_levels.append({
                        "type": "pdh",
                        "price": pdh,
                        "direction": "bearish",
                        "strength": 0.8,
                        "description": f"Previous Day High: {pdh:.5f}"
                    })
            except Exception as e:
                print(f"Failed to get PDL/PDH: {e}")

            # Add SMC levels to decision
            if decision:
                decision["smc_levels"] = smc_levels

            # Override LLM-extracted prices with actual market data and SMC levels
            # The SignalProcessor often hallucinates prices - use real data instead
            try:
                from tradingagents.dataflows.mt5_data import get_mt5_current_price

                # Get current market price
                price_data = get_mt5_current_price(request.symbol)
                if price_data:
                    current_price = price_data.get("bid", 0)
                    ask_price = price_data.get("ask", 0)

                    # Always use market price for entry (not LLM guess)
                    if decision.get("signal") == "BUY":
                        decision["entry_price"] = ask_price
                    elif decision.get("signal") == "SELL":
                        decision["entry_price"] = current_price

                    # Get SL/TP from SMC data if available
                    if smc_raw_data:
                        # Use primary timeframe data (1H typically)
                        primary_tf = list(smc_raw_data.keys())[0] if smc_raw_data else None
                        if primary_tf:
                            tf_data = smc_raw_data[primary_tf]
                            nearest_support = tf_data.get("nearest_support")
                            nearest_resistance = tf_data.get("nearest_resistance")

                            if decision.get("signal") == "BUY":
                                # SL below support, TP at resistance
                                support_bottom = safe_attr(nearest_support, "bottom", 0)
                                resist_bottom = safe_attr(nearest_resistance, "bottom", 0)
                                if nearest_support and support_bottom:
                                    decision["stop_loss"] = support_bottom * 0.998
                                if nearest_resistance and resist_bottom:
                                    decision["take_profit"] = resist_bottom
                            elif decision.get("signal") == "SELL":
                                # SL above resistance, TP at support
                                resist_top = safe_attr(nearest_resistance, "top", 0)
                                support_top = safe_attr(nearest_support, "top", 0)
                                if nearest_resistance and resist_top:
                                    decision["stop_loss"] = resist_top * 1.002
                                if nearest_support and support_top:
                                    decision["take_profit"] = support_top

                    # Validate: if SL/TP are still None or obviously wrong, clear them
                    for field in ["stop_loss", "take_profit"]:
                        val = decision.get(field)
                        if val and current_price:
                            deviation = abs(val - current_price) / current_price
                            if deviation > 0.5:  # More than 50% away = wrong
                                decision[field] = None
            except Exception as e:
                print(f"Error getting market prices: {e}")

            # Format SMC analysis data for response
            smc_response_data = None
            if smc_raw_data:
                smc_response_data = {}
                for tf, tf_data in smc_raw_data.items():
                    smc_response_data[tf] = {
                        "bias": tf_data.get("bias", "neutral"),
                        "current_price": tf_data.get("current_price"),
                        "nearest_support": tf_data.get("nearest_support"),
                        "nearest_resistance": tf_data.get("nearest_resistance"),
                        "order_blocks": {
                            "unmitigated": tf_data.get("order_blocks", {}).get("unmitigated", 0),
                            "bullish_count": len([ob for ob in tf_data.get("order_blocks", {}).get("all", []) if hasattr(ob, 'type') and ob.type == 'bullish']),
                            "bearish_count": len([ob for ob in tf_data.get("order_blocks", {}).get("all", []) if hasattr(ob, 'type') and ob.type == 'bearish']),
                        },
                        "fair_value_gaps": {
                            "unmitigated": tf_data.get("fair_value_gaps", {}).get("unmitigated", 0),
                        },
                        "structure": {
                            "recent_bos": len(tf_data.get("structure", {}).get("recent_bos", [])),
                            "recent_choc": len(tf_data.get("structure", {}).get("recent_choc", [])),
                        }
                    }

            # Build the result object
            result = {
                "status": "completed",
                "progress": 100,
                "current_step": "complete",
                "current_step_title": "Analysis Complete",
                "decision": decision,
                # Include full trading plans for actionable display
                "trading_plan": {
                    "trader_plan": trader_plan,  # Full plan from Trader agent
                    "risk_decision": risk_decision,  # Final decision from Risk Manager
                    "investment_plan": investment_plan,  # Investment plan summary
                },
                # Include SMC analysis data
                "smc_analysis": smc_response_data,
                # Include full analysis context for reflection/learning
                # This is the final_state from the graph, needed for creating memories
                "analysis_context": {
                    "final_state": final_state,  # Full graph state for reflection
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                },
            }

            analysis_tasks[task_id].update(result)

            # Cache the analysis result for later retrieval
            try:
                cache_result = {
                    "decision": decision,
                    "trading_plan": result["trading_plan"],
                    "smc_analysis": smc_response_data,
                    "agent_outputs": analysis_tasks[task_id].get("agent_outputs", {}),
                }
                AnalysisCache.store(request.symbol, request.timeframe, cache_result)
            except Exception as cache_err:
                print(f"Failed to cache analysis result: {cache_err}")

        except Exception as e:
            import traceback as tb
            analysis_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "traceback": tb.format_exc()
            })

    async def run_in_background():
        """Wrapper to run sync analysis in a thread pool"""
        await asyncio.to_thread(run_analysis_sync)

    # Create background task using asyncio (proper async handling)
    asyncio.create_task(run_in_background())

    return {"task_id": task_id, "status": "started"}


@app.get("/api/analysis/status/{task_id}")
async def get_analysis_status(task_id: str):
    """Get analysis task status"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return analysis_tasks[task_id]


@app.get("/api/analysis/cached/{symbol}")
async def get_cached_analysis(symbol: str, timeframe: str = "H1"):
    """Get cached analysis for a symbol/timeframe combination"""
    cache = AnalysisCache.get(symbol, timeframe)
    if not cache:
        return {"cached": False, "symbol": symbol, "timeframe": timeframe}

    age_hours = AnalysisCache.get_age_hours(symbol, timeframe)
    return {
        "cached": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "cached_at": cache.get("cached_at"),
        "age_hours": round(age_hours, 2) if age_hours else None,
        "is_fresh": AnalysisCache.is_fresh(symbol, timeframe),
        "result": cache.get("result"),
    }


@app.get("/api/analysis/cached")
async def list_cached_analyses():
    """List all cached analyses with their ages"""
    cached = AnalysisCache.list_cached()
    return {"cached_analyses": cached}


@app.delete("/api/analysis/cached/{symbol}")
async def clear_cached_analysis(symbol: str, timeframe: str = "H1"):
    """Clear cached analysis for a symbol/timeframe combination"""
    AnalysisCache.clear(symbol, timeframe)
    return {"status": "cleared", "symbol": symbol, "timeframe": timeframe}


# ----- Agent Output Cache -----
# Caches individual agent outputs (social, news, fundamentals) to avoid re-running slow agents

@app.get("/api/analysis/agent-cache/{symbol}")
async def get_agent_cache_status(symbol: str):
    """Get cache status for all agents for a symbol.

    Returns which agents have cached outputs, their age, and expiration status.
    Used by UI to show cache indicators next to agent names.
    """
    cache_status = AgentOutputCache.get_cache_status(symbol)
    return {
        "symbol": symbol,
        "agents": cache_status,
    }


@app.delete("/api/analysis/agent-cache/{symbol}")
async def clear_agent_cache(symbol: str, agent: str = None):
    """Clear agent output cache for a symbol.

    Args:
        symbol: The trading symbol
        agent: Specific agent to clear (e.g., "social_analyst"), or None to clear all
    """
    AgentOutputCache.clear(symbol, agent)
    return {
        "status": "cleared",
        "symbol": symbol,
        "agent": agent or "all",
    }


@app.delete("/api/analysis/agent-cache")
async def clear_all_agent_cache():
    """Clear all agent output caches."""
    AgentOutputCache.clear_all()
    return {"status": "cleared", "message": "All agent caches cleared"}


# ----- Risk Metrics -----

@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get risk metrics"""
    try:
        from tradingagents.risk.metrics import calculate_risk_metrics
        from tradingagents.risk.portfolio import Portfolio

        portfolio = Portfolio()
        metrics = calculate_risk_metrics(portfolio)

        return {
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "sortino_ratio": metrics.get("sortino_ratio"),
            "calmar_ratio": metrics.get("calmar_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "var_95": metrics.get("var_95"),
            "return_pct": metrics.get("return_pct"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor")
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk/guardrails")
async def get_risk_guardrails():
    """Get risk guardrails status"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()
        status = guardrails.get_status()

        return status
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk/circuit-breaker")
async def get_circuit_breaker():
    """Get circuit breaker status"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()
        status = guardrails.get_status()

        return {
            "active": status.get("blocked", False) or status.get("in_cooldown", False),
            "reason": status.get("block_reason", None) or status.get("reason", None),
            "daily_loss_used": status.get("daily_loss_used", 0) or status.get("daily_loss_pct", 0),
            "daily_loss_limit": status.get("daily_loss_limit", 5),
            "consecutive_losses": status.get("consecutive_losses", 0),
            "max_consecutive_losses": status.get("max_consecutive_losses", 3),
            "in_cooldown": status.get("in_cooldown", False),
            "cooldown_until": status.get("cooldown_until"),
            "cooldown_enabled": status.get("cooldown_enabled", True),
            "blocked": status.get("blocked", False)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/risk/breach-history")
async def get_breach_history(limit: int = 20):
    """Get risk breach history"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()

        # Get breach history if available
        history = []
        if hasattr(guardrails, 'get_breach_history'):
            history = guardrails.get_breach_history(limit=limit)
        elif hasattr(guardrails, 'breach_history'):
            history = guardrails.breach_history[-limit:] if guardrails.breach_history else []

        return {"breaches": history, "count": len(history)}
    except Exception as e:
        return {"error": str(e), "breaches": [], "count": 0}


@app.post("/api/risk/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Reset circuit breaker — clears cooldown, daily loss, and consecutive loss counters"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()

        if hasattr(guardrails, 'reset_all'):
            guardrails.reset_all()
        elif hasattr(guardrails, 'reset'):
            guardrails.reset()
        elif hasattr(guardrails, 'reset_cooldown'):
            guardrails.reset_cooldown()

        return {"success": True, "message": "Circuit breaker fully reset"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/risk/cooldown/toggle")
async def toggle_cooldown(enabled: bool = True):
    """Enable or disable cooldown periods"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()
        guardrails.set_cooldown_enabled(enabled)

        return {
            "success": True,
            "cooldown_enabled": enabled,
            "message": f"Cooldown {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/risk/position-size")
async def calculate_position_size(
    symbol: str,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = 1.0
):
    """Calculate optimal position size"""
    try:
        from tradingagents.risk.position_sizing import calculate_position_size

        result = calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=risk_percent
        )

        return result
    except Exception as e:
        return {"error": str(e)}


# ----- Learning System -----

@app.get("/api/learning/status")
async def get_learning_status():
    """Get learning system status"""
    try:
        from tradingagents.learning.online_rl import OnlineRLUpdater

        updater = OnlineRLUpdater()
        weights = updater.get_current_weights()
        history = updater.get_weight_history(n=1)

        last_update = None
        if history:
            last_update = history[-1].get("timestamp")

        return {
            "agent_weights": weights,
            "last_update": last_update,
            "total_updates": len(updater.weight_history),
            "weight_history": updater.get_weight_history(n=10)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/learning/patterns")
async def get_patterns(symbol: Optional[str] = None, limit: int = 10):
    """Get identified trading patterns"""
    try:
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=3)

        # Transform patterns for frontend
        raw_patterns = analysis.get("patterns", [])[:limit]
        patterns = []
        for p in raw_patterns:
            patterns.append({
                "type": p.get("pattern_type", "unknown"),
                "win_rate": p.get("win_rate", 0),
                "description": p.get("pattern_value", ""),
                "occurrences": p.get("sample_size", 0),
                "avg_rr": p.get("avg_rr", 0),
                "quality": p.get("quality", "neutral"),
                "impact": p.get("impact", 0),
            })

        return {
            "patterns": patterns,
            "recommendations": analysis.get("recommendations", []),
            "statistics": analysis.get("statistics", {})
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/learning/update-patterns")
async def update_patterns():
    """Run pattern analysis and update agent weights"""
    try:
        from tradingagents.learning.pattern_analyzer import PatternAnalyzer
        from tradingagents.learning.online_rl import OnlineRLUpdater

        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=3)
        patterns = analysis.get("patterns", [])

        updater = OnlineRLUpdater()

        # Calculate agent performances and update weights if we have data
        performances = updater.calculate_agent_performances(lookback_days=30)
        if any(p.get("sample_size", 0) > 0 for p in performances.values()):
            update_result = updater.update_weights(performances)
            weights = update_result["new_weights"]
        else:
            weights = updater.get_current_weights()

        return {
            "patterns_found": len(patterns),
            "agent_weights": weights,
            "recommendations": analysis.get("recommendations", []),
            "message": "Pattern analysis complete"
        }
    except Exception as e:
        return {"error": str(e)}


class SimilarTradesRequest(BaseModel):
    symbol: str
    direction: str
    conditions: Optional[list] = None


@app.post("/api/learning/similar-trades")
async def find_similar_trades(request: SimilarTradesRequest):
    """Find similar historical trades for comparison"""
    try:
        from tradingagents.trade_decisions import list_decisions

        # Get all closed decisions
        all_decisions = list_decisions(limit=500)

        # Filter for similar trades
        similar = []
        for dec in all_decisions:
            if dec.get("symbol") == request.symbol:
                if dec.get("action") == request.direction or dec.get("signal") == request.direction:
                    if dec.get("outcome"):  # Only closed trades
                        similar.append({
                            "id": dec.get("id"),
                            "timestamp": dec.get("timestamp"),
                            "symbol": dec.get("symbol"),
                            "signal": dec.get("signal") or dec.get("action"),
                            "entry_price": dec.get("entry_price"),
                            "exit_price": dec.get("outcome", {}).get("exit_price"),
                            "pnl": dec.get("outcome", {}).get("pnl"),
                            "was_correct": dec.get("outcome", {}).get("was_correct"),
                            "rationale": dec.get("rationale", "")[:200] + "..." if dec.get("rationale", "") else "",
                            "key_factors": dec.get("key_factors", [])
                        })

        # Sort by recency and limit
        similar.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        similar = similar[:20]

        # Calculate stats
        wins = sum(1 for s in similar if s.get("was_correct"))
        total = len(similar)

        return {
            "similar_trades": similar,
            "stats": {
                "total_found": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": (wins / total * 100) if total > 0 else 0,
                "avg_pnl": sum(s.get("pnl", 0) for s in similar) / total if total > 0 else 0
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ----- Memory System -----

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory database statistics"""
    try:
        import chromadb
        from pathlib import Path

        # Use absolute path to memory_db in project root
        memory_db_path = Path(__file__).parent.parent.parent / "memory_db"
        client = chromadb.PersistentClient(path=str(memory_db_path))
        collections = client.list_collections()

        stats = []
        for coll in collections:
            stats.append({
                "name": coll.name,
                "count": coll.count()
            })

        return {"collections": stats}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/memory/query")
async def query_memory(request: MemoryQueryRequest):
    """Query memory collection"""
    try:
        import chromadb
        from pathlib import Path

        # Use absolute path to memory_db in project root
        memory_db_path = Path(__file__).parent.parent.parent / "memory_db"
        client = chromadb.PersistentClient(path=str(memory_db_path))
        collection = client.get_collection(request.collection)

        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results
        )

        return {
            "results": results["documents"][0] if results["documents"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/memory/lessons")
async def get_memory_lessons(
    collection: str = None,
    tier: str = None,
    limit: int = 20
):
    """
    Get recent lessons/reflections from memory with full metadata.

    Args:
        collection: Optional collection name filter (default: all collections)
        tier: Optional tier filter ("short", "mid", "long")
        limit: Maximum number of lessons to return
    """
    try:
        import chromadb
        from pathlib import Path

        memory_db_path = Path(__file__).parent.parent.parent / "memory_db"
        client = chromadb.PersistentClient(path=str(memory_db_path))

        lessons = []
        collections_to_query = []

        if collection:
            try:
                coll = client.get_collection(collection)
                collections_to_query.append((collection, coll))
            except Exception:
                return {"error": f"Collection '{collection}' not found"}
        else:
            # Query all collections
            for coll in client.list_collections():
                collections_to_query.append((coll.name, coll))

        for coll_name, coll in collections_to_query:
            try:
                # Get all items with metadata
                count = coll.count()
                if count == 0:
                    continue

                results = coll.get(
                    include=["metadatas", "documents"],
                    limit=min(count, 100)  # Cap at 100 per collection
                )

                if not results["documents"]:
                    continue

                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}

                    # Apply tier filter
                    if tier and metadata.get("tier") != tier:
                        continue

                    lesson = {
                        "id": results["ids"][i] if results.get("ids") else str(i),
                        "collection": coll_name,
                        "situation": doc[:500] if doc else "",  # Truncate for display
                        "recommendation": metadata.get("recommendation", "")[:1000],  # Truncate
                        "tier": metadata.get("tier", "short"),
                        "confidence": metadata.get("confidence", 0.5),
                        "outcome_quality": metadata.get("outcome_quality", 0.5),
                        "prediction_correct": metadata.get("prediction_correct", "unknown"),
                        "timestamp": metadata.get("timestamp"),
                        "reference_count": metadata.get("reference_count", 0),
                        "market_regime": metadata.get("market_regime"),
                        "volatility_regime": metadata.get("volatility_regime"),
                    }
                    lessons.append(lesson)
            except Exception as e:
                # Skip collections that fail
                continue

        # Sort by timestamp (newest first) and confidence
        lessons.sort(key=lambda x: (
            x.get("timestamp") or "1970-01-01",
            x.get("confidence", 0)
        ), reverse=True)

        # Apply limit
        lessons = lessons[:limit]

        # Calculate summary stats
        total = len(lessons)
        correct = sum(1 for l in lessons if l["prediction_correct"] == "True")
        incorrect = sum(1 for l in lessons if l["prediction_correct"] == "False")
        avg_confidence = sum(l["confidence"] for l in lessons) / total if total > 0 else 0

        tier_counts = {}
        for l in lessons:
            t = l["tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1

        return {
            "lessons": lessons,
            "summary": {
                "total": total,
                "correct_predictions": correct,
                "incorrect_predictions": incorrect,
                "unknown": total - correct - incorrect,
                "avg_confidence": round(avg_confidence, 2),
                "by_tier": tier_counts
            }
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Track reflection tasks
reflection_tasks: Dict[str, Dict] = {}


@app.post("/api/memory/reflect")
async def trigger_reflection():
    """
    Manually trigger reflection on closed trades (runs in background).

    This runs the evening reflection cycle which:
    - Finds closed trades that haven't been reflected on
    - Analyzes the outcomes and generates learnings
    - Stores reflections in memory for future reference

    Returns a task_id immediately - poll /api/memory/reflect/status/{task_id} for results.
    """
    import asyncio

    task_id = f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize task state
    reflection_tasks[task_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }

    async def run_reflection():
        try:
            from tradingagents.automation.portfolio_automation import PortfolioAutomation

            automation = PortfolioAutomation()
            result = await automation.run_evening_reflect()

            reflection_tasks[task_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "success": True,
                "trades_processed": result.trades_processed,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "total_pnl": result.total_pnl,
                "reflections_created": result.reflections_created,
                "memories_stored": result.memories_stored,
                "errors": result.errors if result.errors else {},
                "duration_seconds": result.total_duration_seconds
            })
        except Exception as e:
            import traceback
            reflection_tasks[task_id].update({
                "status": "error",
                "completed_at": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    # Run in background
    asyncio.create_task(run_reflection())

    return {"task_id": task_id, "status": "started"}


@app.get("/api/memory/reflect/status/{task_id}")
async def get_reflection_status(task_id: str):
    """Get reflection task status"""
    if task_id not in reflection_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return reflection_tasks[task_id]


@app.delete("/api/memory/{collection}/{memory_id}")
async def delete_memory(collection: str, memory_id: str):
    """Delete a specific memory from a collection"""
    try:
        import chromadb
        from pathlib import Path

        memory_db_path = Path(__file__).parent.parent.parent / "memory_db"
        client = chromadb.PersistentClient(path=str(memory_db_path))

        try:
            coll = client.get_collection(collection)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

        # Verify the memory exists
        existing = coll.get(ids=[memory_id])
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found in collection '{collection}'")

        # Delete the memory
        coll.delete(ids=[memory_id])

        return {"success": True, "message": f"Memory '{memory_id}' deleted from '{collection}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Portfolio Automation -----

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get portfolio automation status"""
    status = {"running": False}

    pid_file = PORTFOLIO_SCHEDULER_PID_FILE
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                status = {"running": True, "pid": pid}
        except:
            pass

    # Load state file if exists
    state_file = SCHEDULER_STATE_FILE
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            status["last_run"] = state.get("last_run")
            status["next_run"] = state.get("next_run")
        except:
            pass

    # Include persistent state info
    persistent_state = AutomationState.get_status()
    status["enabled"] = persistent_state.get("enabled", False)
    status["last_start"] = persistent_state.get("last_start")
    status["last_stop"] = persistent_state.get("last_stop")
    status["stop_reason"] = persistent_state.get("stop_reason")

    return status


@app.get("/api/portfolio/config")
async def get_portfolio_config():
    """Get portfolio configuration"""
    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/config/add-symbol")
async def add_portfolio_symbol(request: dict):
    """Add a symbol to portfolio config"""
    import yaml

    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    symbol = request.get("symbol", "").upper()
    if not symbol:
        return {"error": "Symbol required"}

    try:
        # Read current config
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Check if symbol already exists
        existing_symbols = [s.get("symbol") for s in config.get("symbols", [])]
        if symbol in existing_symbols:
            return {"error": f"{symbol} already in portfolio config"}

        # Determine correlation group based on symbol
        correlation_group = "forex"
        if any(metal in symbol.upper() for metal in ["XAU", "XAG", "XPT"]):
            correlation_group = "metals"
        elif "COPPER" in symbol.upper() or "XCU" in symbol.upper():
            correlation_group = "industrial_metals"
        elif any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "LTC"]):
            correlation_group = "crypto"

        # Add new symbol with default config
        new_symbol = {
            "symbol": symbol,
            "max_positions": 1,
            "risk_budget_pct": 1.5,
            "correlation_group": correlation_group,
            "timeframes": ["1H", "4H", "D1"],
            "enabled": True,
            "min_confidence": 0.6
        }

        if "symbols" not in config:
            config["symbols"] = []
        config["symbols"].append(new_symbol)

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {"success": True, "symbol": symbol, "config": new_symbol}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/config/remove-symbol")
async def remove_portfolio_symbol(request: dict):
    """Remove a symbol from portfolio config"""
    import yaml

    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    symbol = request.get("symbol", "").upper()
    if not symbol:
        return {"error": "Symbol required"}

    try:
        # Read current config
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Find and remove the symbol
        symbols = config.get("symbols", [])
        original_count = len(symbols)
        config["symbols"] = [s for s in symbols if s.get("symbol") != symbol]

        if len(config["symbols"]) == original_count:
            return {"error": f"{symbol} not found in portfolio config"}

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {"success": True, "symbol": symbol, "message": f"{symbol} removed from portfolio config"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/config/update-symbol")
async def update_portfolio_symbol(request: dict):
    """Update a symbol's configuration in portfolio config"""
    import yaml

    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    symbol = request.get("symbol", "").upper()
    if not symbol:
        return {"error": "Symbol required"}

    try:
        # Read current config
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Find and update the symbol
        found = False
        for s in config.get("symbols", []):
            if s.get("symbol") == symbol:
                # Update fields if provided
                if "enabled" in request:
                    s["enabled"] = request["enabled"]
                if "risk_budget_pct" in request:
                    s["risk_budget_pct"] = request["risk_budget_pct"]
                if "max_positions" in request:
                    s["max_positions"] = request["max_positions"]
                if "min_confidence" in request:
                    s["min_confidence"] = request["min_confidence"]
                if "timeframes" in request:
                    s["timeframes"] = request["timeframes"]
                if "correlation_group" in request:
                    s["correlation_group"] = request["correlation_group"]
                found = True
                break

        if not found:
            return {"error": f"{symbol} not found in portfolio config"}

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {"success": True, "symbol": symbol}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/config/update")
async def update_portfolio_config(request: PortfolioConfigUpdate):
    """Update portfolio-level configuration settings"""
    import yaml

    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        updated_fields = []

        # Execution mode validation
        if request.execution_mode is not None:
            valid_modes = ["FULL_AUTO", "SEMI_AUTO", "PAPER", "full_auto", "semi_auto", "paper"]
            if request.execution_mode not in valid_modes:
                return {"error": "Invalid execution_mode. Must be one of: FULL_AUTO, SEMI_AUTO, PAPER"}
            config["execution_mode"] = request.execution_mode.lower()
            updated_fields.append("execution_mode")

        # Integer validations
        if request.max_total_positions is not None:
            if request.max_total_positions < 1 or request.max_total_positions > 20:
                return {"error": "max_total_positions must be between 1 and 20"}
            config["max_total_positions"] = request.max_total_positions
            updated_fields.append("max_total_positions")

        if request.max_daily_trades is not None:
            if request.max_daily_trades < 1 or request.max_daily_trades > 50:
                return {"error": "max_daily_trades must be between 1 and 50"}
            config["max_daily_trades"] = request.max_daily_trades
            updated_fields.append("max_daily_trades")

        # Float validations for percentages
        if request.total_risk_budget_pct is not None:
            if request.total_risk_budget_pct < 0.5 or request.total_risk_budget_pct > 20.0:
                return {"error": "total_risk_budget_pct must be between 0.5% and 20%"}
            config["total_risk_budget_pct"] = request.total_risk_budget_pct
            updated_fields.append("total_risk_budget_pct")

        if request.daily_loss_limit_pct is not None:
            if request.daily_loss_limit_pct < 0.5 or request.daily_loss_limit_pct > 10.0:
                return {"error": "daily_loss_limit_pct must be between 0.5% and 10%"}
            config["daily_loss_limit_pct"] = request.daily_loss_limit_pct
            updated_fields.append("daily_loss_limit_pct")

        # ATR settings
        if request.use_atr_stops is not None:
            config["use_atr_stops"] = request.use_atr_stops
            updated_fields.append("use_atr_stops")

        if request.atr_stop_multiplier is not None:
            if request.atr_stop_multiplier < 0.5 or request.atr_stop_multiplier > 5.0:
                return {"error": "atr_stop_multiplier must be between 0.5 and 5.0"}
            config["atr_stop_multiplier"] = request.atr_stop_multiplier
            updated_fields.append("atr_stop_multiplier")

        if request.atr_trailing_multiplier is not None:
            if request.atr_trailing_multiplier < 0.5 or request.atr_trailing_multiplier > 5.0:
                return {"error": "atr_trailing_multiplier must be between 0.5 and 5.0"}
            config["atr_trailing_multiplier"] = request.atr_trailing_multiplier
            updated_fields.append("atr_trailing_multiplier")

        if request.risk_reward_ratio is not None:
            if request.risk_reward_ratio < 0.5 or request.risk_reward_ratio > 10.0:
                return {"error": "risk_reward_ratio must be between 0.5 and 10.0"}
            config["risk_reward_ratio"] = request.risk_reward_ratio
            updated_fields.append("risk_reward_ratio")

        # Schedule updates
        if request.schedule is not None:
            if "schedule" not in config:
                config["schedule"] = {}
            for key in ["morning_analysis_hour", "midday_review_hour", "evening_reflect_hour"]:
                if key in request.schedule:
                    hour = request.schedule[key]
                    if not isinstance(hour, int) or hour < 0 or hour > 23:
                        return {"error": f"{key} must be an integer between 0 and 23"}
                    config["schedule"][key] = hour
                    updated_fields.append(f"schedule.{key}")
            if "timezone" in request.schedule:
                config["schedule"]["timezone"] = request.schedule["timezone"]
                updated_fields.append("schedule.timezone")

        if not updated_fields:
            return {"error": "No fields provided to update"}

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {
            "success": True,
            "updated_fields": updated_fields,
            "message": f"Updated {len(updated_fields)} field(s)"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/portfolio/config/suggestions")
async def get_portfolio_suggestions():
    """Get LLM-powered suggestions for portfolio balance"""
    import yaml
    from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

    config_file = PORTFOLIO_CONFIG_FILE
    if not config_file.exists():
        return {"error": "Config file not found"}

    try:
        # Read current portfolio config
        with open(config_file) as f:
            config = yaml.safe_load(f)

        current_symbols = config.get("symbols", [])
        if not current_symbols:
            current_portfolio = "Empty portfolio"
        else:
            current_portfolio = "\n".join([
                f"- {s['symbol']} ({s.get('correlation_group', 'unknown')}, risk: {s.get('risk_budget_pct', 0)}%)"
                for s in current_symbols
            ])

        # Get available symbols from MT5 Market Watch
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return {"error": "MT5 not initialized"}

        symbols = mt5.symbols_get()
        if symbols is None:
            return {"error": "Could not get MT5 symbols"}

        # Get Market Watch symbols (visible ones)
        market_watch = []
        current_symbol_names = {s["symbol"] for s in current_symbols}
        for s in symbols:
            if s.visible and s.name not in current_symbol_names:
                market_watch.append({
                    "symbol": s.name,
                    "description": s.description,
                })

        if not market_watch:
            return {"error": "No additional symbols available in Market Watch"}

        available_symbols = "\n".join([
            f"- {s['symbol']}: {s['description']}"
            for s in market_watch[:30]  # Limit to first 30
        ])

        # Setup LLM client
        try:
            client, model, uses_responses = get_llm_client()
        except ValueError as e:
            return {"error": str(e)}

        prompt = f"""Analyze this trading portfolio and suggest symbols to add for better diversification and balance.

CURRENT PORTFOLIO:
{current_portfolio}

AVAILABLE SYMBOLS IN MARKET WATCH (not in portfolio):
{available_symbols}

Consider:
1. Correlation - avoid symbols that move together (e.g., XAUUSD and XAGUSD are highly correlated)
2. Asset class diversity - mix of metals, forex pairs, indices, etc.
3. Risk distribution - suggest symbols that could hedge or balance existing positions
4. Volatility balance - mix of high and low volatility instruments

Respond in this exact JSON format:
{{
  "suggestions": [
    {{
      "symbol": "SYMBOL_NAME",
      "reason": "Brief reason for adding this symbol",
      "correlation_group": "suggested group (metals/forex/crypto/indices/energy)",
      "priority": "high/medium/low"
    }}
  ],
  "portfolio_analysis": "Brief analysis of current portfolio balance",
  "risk_notes": "Any risk considerations"
}}

Suggest 3-5 symbols. Only suggest symbols from the AVAILABLE list."""

        content = chat_completion(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert portfolio manager specializing in diversification and risk management. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
            use_responses_api=uses_responses,
        )

        if content:
            # Try to parse as JSON
            import json
            try:
                # Clean up the response - remove markdown code blocks if present
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                suggestions_data = json.loads(content)
                return {
                    "success": True,
                    "suggestions": suggestions_data.get("suggestions", []),
                    "portfolio_analysis": suggestions_data.get("portfolio_analysis", ""),
                    "risk_notes": suggestions_data.get("risk_notes", ""),
                    "current_symbols": [s["symbol"] for s in current_symbols],
                    "available_count": len(market_watch),
                }
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {
                    "success": True,
                    "raw_response": content,
                    "current_symbols": [s["symbol"] for s in current_symbols],
                }
        else:
            return {"error": "Empty response from LLM"}

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/start")
async def start_portfolio_automation():
    """Start portfolio automation"""
    try:
        import subprocess
        import time

        project_root = Path(__file__).parent.parent.parent

        # Check if already running
        pid_file = project_root / "portfolio_scheduler.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if psutil.pid_exists(pid):
                    return {"success": True, "message": "Portfolio automation already running", "pid": pid}
                else:
                    # Stale PID file, remove it
                    pid_file.unlink()
            except:
                pid_file.unlink()

        log_dir = project_root / "logs" / "scheduler"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file for subprocess output
        log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Start the daemon with output to log file
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [sys.executable, "-m", "cli.main", "portfolio", "start"],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(project_root)
            )

        # Wait briefly to see if it crashes immediately
        time.sleep(2)

        # Check if process is still running
        if process.poll() is not None:
            # Process exited - read log for error
            try:
                error_output = log_file.read_text()[-2000:]  # Last 2000 chars
            except:
                error_output = "Unknown error"

            AutomationState.set_enabled(False)
            AutomationState.set_stop_reason(f"startup_failed: {error_output[:500]}")
            return {"error": f"Automation failed to start. Check {log_file}. Last output: {error_output[:500]}"}

        # Check if PID file was created
        pid_file = project_root / "portfolio_scheduler.pid"
        if not pid_file.exists():
            # Give it another moment
            time.sleep(1)

        if pid_file.exists():
            pid = pid_file.read_text().strip()
            print(f"[Portfolio Start] Automation started with PID {pid}, log: {log_file}")
        else:
            print(f"[Portfolio Start] Warning: PID file not created, but process appears running")

        # Persist enabled state
        AutomationState.set_enabled(True)

        return {"success": True, "message": "Portfolio automation started", "log_file": str(log_file)}
    except Exception as e:
        import traceback
        print(f"[Portfolio Start] Exception: {traceback.format_exc()}")
        return {"error": str(e)}


@app.get("/api/portfolio/diagnose")
async def diagnose_portfolio_automation():
    """Diagnose portfolio automation - test imports and config loading"""
    diagnostics = {
        "imports_ok": False,
        "config_ok": False,
        "scheduler_ok": False,
        "errors": []
    }

    # Test imports
    try:
        from tradingagents.automation import (
            PortfolioAutomation,
            DailyScheduler,
            load_portfolio_config,
            get_default_config,
        )
        diagnostics["imports_ok"] = True
    except Exception as e:
        diagnostics["errors"].append(f"Import error: {str(e)}")
        return diagnostics

    # Test config loading
    try:
        config_file = PORTFOLIO_CONFIG_FILE
        if config_file.exists():
            config = load_portfolio_config(str(config_file))
            diagnostics["config_file"] = str(config_file)
        else:
            config = get_default_config()
            diagnostics["config_file"] = "default"

        errors = config.validate()
        if errors:
            diagnostics["errors"].extend([f"Config error: {e}" for e in errors])
        else:
            diagnostics["config_ok"] = True
            diagnostics["execution_mode"] = config.execution_mode.value
            diagnostics["symbols"] = [s.symbol for s in config.get_enabled_symbols()]
    except Exception as e:
        diagnostics["errors"].append(f"Config load error: {str(e)}")
        return diagnostics

    # Test scheduler init
    try:
        automation = PortfolioAutomation(config)
        scheduler = DailyScheduler(automation)
        diagnostics["scheduler_ok"] = True
    except Exception as e:
        import traceback
        diagnostics["errors"].append(f"Scheduler init error: {str(e)}")
        diagnostics["traceback"] = traceback.format_exc()

    return diagnostics


@app.post("/api/portfolio/stop")
async def stop_portfolio_automation():
    """Stop portfolio automation"""
    pid_file = PORTFOLIO_SCHEDULER_PID_FILE
    if not pid_file.exists():
        # Still update state even if PID file missing
        AutomationState.set_enabled(False)
        AutomationState.set_stop_reason("user_stopped")
        return {"error": "Not running"}

    try:
        pid = int(pid_file.read_text().strip())
        import psutil

        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=5)

        pid_file.unlink()

        # Persist disabled state
        AutomationState.set_enabled(False)
        AutomationState.set_stop_reason("user_stopped")

        return {"success": True, "message": "Portfolio automation stopped"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/trigger")
async def trigger_daily_cycle(cycle_type: str = "morning"):
    """Manually trigger a daily cycle (runs in background - safe to switch tabs)"""
    import asyncio

    task_id = f"trigger_{cycle_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize task state
    analysis_tasks[task_id] = {
        "status": "running",
        "cycle_type": cycle_type,
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "message": f"Starting {cycle_type} cycle..."
    }

    def run_trigger_sync():
        """Run trigger synchronously in background thread"""
        try:
            import asyncio
            from tradingagents.automation.portfolio_automation import PortfolioAutomation

            analysis_tasks[task_id]["message"] = f"Initializing {cycle_type} analysis..."
            analysis_tasks[task_id]["progress"] = 10

            automation = PortfolioAutomation()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                analysis_tasks[task_id]["message"] = f"Running {cycle_type} cycle..."
                analysis_tasks[task_id]["progress"] = 30

                if cycle_type == "morning":
                    result = loop.run_until_complete(automation.run_morning_analysis())
                elif cycle_type == "midday":
                    result = loop.run_until_complete(automation.run_midday_review())
                elif cycle_type == "evening":
                    result = loop.run_until_complete(automation.run_evening_reflect())
                else:
                    analysis_tasks[task_id].update({
                        "status": "error",
                        "error": f"Unknown cycle type: {cycle_type}"
                    })
                    return
            finally:
                loop.close()

            # Convert result to dict for JSON serialization
            result_dict = None
            if result:
                try:
                    result_dict = result.__dict__ if hasattr(result, '__dict__') else str(result)
                except:
                    result_dict = str(result)

            analysis_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "result": result_dict,
                "message": f"{cycle_type.capitalize()} cycle completed",
                "completed_at": datetime.now().isoformat()
            })

        except Exception as e:
            import traceback
            analysis_tasks[task_id].update({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    async def run_in_background():
        """Wrapper to run sync trigger in a thread pool"""
        await asyncio.to_thread(run_trigger_sync)

    # Create background task - returns immediately
    asyncio.create_task(run_in_background())

    return {"task_id": task_id, "status": "started", "message": f"{cycle_type.capitalize()} cycle started in background"}


# ----- SMC Analysis -----

@app.get("/api/smc/analysis")
async def run_smc_analysis(
    symbol: str,
    timeframe: str = "H1",
    fvg_min_size: float = 0.1,  # Lower threshold = more sensitive FVG detection
    lookback: int = 50,
    debug: bool = False
):
    """Run SMC analysis

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (M15, M30, H1, H4, D1)
        fvg_min_size: Minimum FVG size as ATR multiplier (default 0.1 = 10% of ATR)
        lookback: How many bars to analyze (default 50)
        debug: Include all FVGs including mitigated ones in response
    """
    try:
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
        import MetaTrader5 as mt5

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get OHLCV data
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 0, 500)
        if rates is None:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        import pandas as pd
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calculate PDH/PDL (Previous Day High/Low)
        # Get daily data to find previous day's high and low
        pdh = None
        pdl = None
        try:
            # Get the last 5 daily bars to find previous day
            daily_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 5)
            if daily_rates is not None and len(daily_rates) >= 2:
                # Previous day is index -2 (index -1 is current incomplete day)
                prev_day = daily_rates[-2]
                pdh = float(prev_day['high'])
                pdl = float(prev_day['low'])
        except Exception as e:
            logger.warning(f"Failed to calculate PDH/PDL: {e}")

        # Use configurable thresholds
        analyzer = SmartMoneyAnalyzer(fvg_min_size_atr=fvg_min_size)

        # Run extended analysis with new features (equal levels, breakers, OTE, confluence)
        current_price = df.iloc[-1]['close']
        extended_result = analyzer.analyze_full_smc(df, current_price=current_price)

        # Extract components from extended analysis
        order_blocks_raw = (
            extended_result.get('order_blocks', {}).get('bullish', []) +
            extended_result.get('order_blocks', {}).get('bearish', [])
        )
        fvgs_raw = (
            extended_result.get('fair_value_gaps', {}).get('bullish', []) +
            extended_result.get('fair_value_gaps', {}).get('bearish', [])
        )
        swing_points = extended_result.get('swing_points', [])
        structure_breaks = extended_result.get('structure', {})
        zones = extended_result.get('zones', {'support': [], 'resistance': []})

        # NEW: Extract new SMC features (handle nested structures)
        equal_levels_data = extended_result.get('equal_levels', {})
        if isinstance(equal_levels_data, dict):
            # Combine equal_highs and equal_lows into single list
            equal_levels = (
                equal_levels_data.get('equal_highs', []) +
                equal_levels_data.get('equal_lows', [])
            )
        else:
            equal_levels = equal_levels_data if isinstance(equal_levels_data, list) else []

        breaker_blocks_data = extended_result.get('breaker_blocks', {})
        if isinstance(breaker_blocks_data, dict):
            # Combine bullish and bearish breakers into single list
            breaker_blocks = (
                breaker_blocks_data.get('bullish', []) +
                breaker_blocks_data.get('bearish', [])
            )
        else:
            breaker_blocks = breaker_blocks_data if isinstance(breaker_blocks_data, list) else []

        ote_zones = extended_result.get('ote_zones', [])
        if not isinstance(ote_zones, list):
            ote_zones = []

        premium_discount = extended_result.get('premium_discount', None)
        # Confluence score is at current_confluence key
        confluence_score = extended_result.get('current_confluence', None)

        # NEW: Extract advanced SMC patterns (liquidity sweeps, inducements, rejection blocks, turtle soup, alerts)
        liquidity_sweeps_data = extended_result.get('liquidity_sweeps', {})
        inducements_data = extended_result.get('inducements', {})
        rejection_blocks_data = extended_result.get('rejection_blocks', {})
        turtle_soup_data = extended_result.get('turtle_soup', {})
        pattern_alerts = extended_result.get('alerts', [])

        result = {
            'current_price': current_price,
            'order_blocks': extended_result.get('order_blocks', {}),
            'fair_value_gaps': extended_result.get('fair_value_gaps', {}),
            'structure': {
                'swing_points': len(swing_points),
                'bos_count': len(structure_breaks.get('bos', [])),
                'choc_count': len(structure_breaks.get('choc', [])),
                'recent_bos': structure_breaks.get('recent_bos', []),
                'recent_choc': structure_breaks.get('recent_choc', [])
            },
            'zones': zones,
            'nearest_support': zones['support'][0] if zones['support'] else None,
            'nearest_resistance': zones['resistance'][0] if zones['resistance'] else None,
            'bias': extended_result.get('bias', 'neutral'),
            # NEW: Include new SMC features in result
            'equal_levels': equal_levels,
            'breaker_blocks': breaker_blocks,
            'ote_zones': ote_zones,
            'premium_discount': premium_discount,
            'confluence_score': confluence_score
        }

        # Extract order blocks as list with serializable format
        ob_data = result.get("order_blocks", {})
        order_blocks = []
        for ob in ob_data.get("bullish", []) + ob_data.get("bearish", []):
            order_blocks.append({
                "type": ob.type,
                "top": ob.top,
                "bottom": ob.bottom,
                "strength": ob.strength,
                "mitigated": ob.mitigated,
                "timestamp": ob.timestamp
            })

        # Extract FVGs as list with serializable format
        fvg_data = result.get("fair_value_gaps", {})
        fair_value_gaps = []
        for fvg in fvg_data.get("bullish", []) + fvg_data.get("bearish", []):
            fair_value_gaps.append({
                "type": fvg.type,
                "top": fvg.top,
                "bottom": fvg.bottom,
                "size": fvg.size,
                "mitigated": fvg.mitigated,
                "timestamp": fvg.timestamp
            })

        # Extract liquidity zones (stop loss clusters at swing highs/lows)
        raw_liquidity_zones = result.get("liquidity_zones", [])
        liquidity_zones = []
        for lz in raw_liquidity_zones:
            # Handle both LiquidityZone dataclass objects and dicts
            if hasattr(lz, 'type'):
                # It's a dataclass object
                liquidity_zones.append({
                    "type": lz.type,  # "buy-side" or "sell-side"
                    "price": lz.price,
                    "strength": lz.strength,  # 0-100
                    "touched": lz.touched,
                    "timestamp": lz.timestamp
                })
            else:
                # It's already a dict
                liquidity_zones.append({
                    "type": lz.get("type", "unknown"),
                    "price": lz.get("price", 0),
                    "strength": lz.get("strength", 0),
                    "touched": lz.get("touched", False),
                    "timestamp": lz.get("timestamp", "")
                })

        # Build key levels from nearest support/resistance and unmitigated zones
        current_price = result.get("current_price", df.iloc[-1]['close'])
        key_levels = []

        # Add nearest support
        nearest_support = result.get("nearest_support")
        if nearest_support:
            source_type = safe_attr(nearest_support, "type", "zone")
            source_label = "FVG" if source_type == "fvg" else "OB" if source_type == "order_block" else str(source_type).upper()
            support_top = safe_attr(nearest_support, "top", 0)
            support_bottom = safe_attr(nearest_support, "bottom", 0)
            key_levels.append({
                "type": "Support Zone",
                "price": round((support_top + support_bottom) / 2, 2) if support_top or support_bottom else 0,
                "source": source_label,
                "direction": "bullish"
            })

        # Add nearest resistance
        nearest_resistance = result.get("nearest_resistance")
        if nearest_resistance:
            source_type = safe_attr(nearest_resistance, "type", "zone")
            source_label = "FVG" if source_type == "fvg" else "OB" if source_type == "order_block" else str(source_type).upper()
            resist_top = safe_attr(nearest_resistance, "top", 0)
            resist_bottom = safe_attr(nearest_resistance, "bottom", 0)
            key_levels.append({
                "type": "Resistance Zone",
                "price": round((resist_top + resist_bottom) / 2, 2) if resist_top or resist_bottom else 0,
                "source": source_label,
                "direction": "bearish"
            })

        # Add unmitigated OBs as key levels
        for ob in order_blocks:
            if not safe_attr(ob, "mitigated", True):
                ob_type = safe_attr(ob, "type", "")
                is_bullish = ob_type == "bullish" or str(ob_type).lower().startswith("bull")
                level_type = "Demand Zone" if is_bullish else "Supply Zone"
                ob_top = safe_attr(ob, "top", 0)
                ob_bottom = safe_attr(ob, "bottom", 0)
                key_levels.append({
                    "type": level_type,
                    "price": round((ob_top + ob_bottom) / 2, 2) if ob_top or ob_bottom else 0,
                    "source": "OB",
                    "direction": "bullish" if is_bullish else "bearish"
                })

        # Add unmitigated FVGs as key levels
        for fvg in fair_value_gaps:
            if not safe_attr(fvg, "mitigated", True):
                fvg_type = safe_attr(fvg, "type", "")
                is_bullish = fvg_type == "bullish" or str(fvg_type).lower().startswith("bull")
                level_type = "Bullish FVG" if is_bullish else "Bearish FVG"
                fvg_top = safe_attr(fvg, "top", 0)
                fvg_bottom = safe_attr(fvg, "bottom", 0)
                key_levels.append({
                    "type": level_type,
                    "price": round((fvg_top + fvg_bottom) / 2, 2) if fvg_top or fvg_bottom else 0,
                    "source": "FVG",
                    "direction": "bullish" if is_bullish else "bearish"
                })

        # Sort by distance from current price and take closest 8
        key_levels.sort(key=lambda x: abs(x["price"] - current_price))
        key_levels = key_levels[:8]

        # Build bias explanation
        bias = result.get("bias", "neutral")
        structure = result.get("structure", {})
        recent_bos = structure.get("recent_bos", [])
        recent_choc = structure.get("recent_choc", [])

        # Count bullish vs bearish FVGs
        bullish_fvgs = len([f for f in fair_value_gaps if f["type"] == "bullish" and not f["mitigated"]])
        bearish_fvgs = len([f for f in fair_value_gaps if f["type"] == "bearish" and not f["mitigated"]])

        # Count bullish vs bearish OBs
        bullish_obs = len([o for o in order_blocks if o["type"] == "bullish" and not o["mitigated"]])
        bearish_obs = len([o for o in order_blocks if o["type"] == "bearish" and not o["mitigated"]])

        # Build explanation
        bias_factors = []

        if recent_choc:
            last_choc = recent_choc[-1] if hasattr(recent_choc[-1], 'type') else recent_choc[-1]
            choc_type = last_choc.type if hasattr(last_choc, 'type') else last_choc.get('type', '')
            if choc_type == 'high':
                bias_factors.append("Recent Change of Character (CHOC) broke a swing high, indicating bearish momentum shift")
            else:
                bias_factors.append("Recent Change of Character (CHOC) broke a swing low, indicating bullish momentum shift")

        if recent_bos:
            last_bos = recent_bos[-1] if hasattr(recent_bos[-1], 'type') else recent_bos[-1]
            bos_type = last_bos.type if hasattr(last_bos, 'type') else last_bos.get('type', '')
            if bos_type == 'high':
                bias_factors.append("Break of Structure (BOS) above swing high confirms bullish continuation")
            else:
                bias_factors.append("Break of Structure (BOS) below swing low confirms bearish continuation")

        if bullish_fvgs > bearish_fvgs:
            bias_factors.append(f"{bullish_fvgs} unmitigated bullish FVGs vs {bearish_fvgs} bearish - buyers left imbalances")
        elif bearish_fvgs > bullish_fvgs:
            bias_factors.append(f"{bearish_fvgs} unmitigated bearish FVGs vs {bullish_fvgs} bullish - sellers left imbalances")

        if bullish_obs > bearish_obs:
            bias_factors.append(f"{bullish_obs} active demand zones (bullish OBs) indicate institutional buying interest")
        elif bearish_obs > bullish_obs:
            bias_factors.append(f"{bearish_obs} active supply zones (bearish OBs) indicate institutional selling interest")

        # Position relative to key levels
        if nearest_support and nearest_resistance:
            support_dist = current_price - safe_attr(nearest_support, "bottom", 0)
            resist_dist = safe_attr(nearest_resistance, "top", 0) - current_price
            if support_dist < resist_dist:
                bias_factors.append(f"Price is closer to support ({support_dist:.2f} pts) than resistance ({resist_dist:.2f} pts)")
            else:
                bias_factors.append(f"Price is closer to resistance ({resist_dist:.2f} pts) than support ({support_dist:.2f} pts)")

        if not bias_factors:
            bias_factors.append("No strong directional signals detected - market structure is neutral")

        bias_summary = f"The market bias is **{bias.upper()}** based on Smart Money Concepts analysis."
        bias_explanation = " ".join([f"• {factor}" for factor in bias_factors])

        # Serialize new SMC features with proper attribute mapping
        equal_levels_serialized = []
        for el in equal_levels:
            if hasattr(el, 'type'):
                # Map "equal_highs" to "eqh" and "equal_lows" to "eql"
                el_type = "eqh" if el.type == "equal_highs" else "eql" if el.type == "equal_lows" else el.type
                equal_levels_serialized.append({
                    "type": el_type,
                    "price": el.price,
                    "touches": el.touches,
                    "strength": 0.5 + (el.touches * 0.1) if el.touches else 0.5,  # Strength based on touches
                    "swept": getattr(el, 'swept', False)
                })

        breaker_blocks_serialized = []
        for bb in breaker_blocks:
            if hasattr(bb, 'current_type'):
                breaker_blocks_serialized.append({
                    "type": bb.current_type,  # "bullish" or "bearish"
                    "top": bb.top,
                    "bottom": bb.bottom,
                    "original_ob_type": getattr(bb, 'original_type', None),
                    "break_index": getattr(bb, 'break_index', None),
                    "strength": getattr(bb, 'strength', 0.5),
                    "mitigated": getattr(bb, 'mitigated', False)
                })

        ote_zones_serialized = []
        for ote in ote_zones:
            if hasattr(ote, 'direction'):
                # Calculate Fibonacci levels from swing points
                swing_high = getattr(ote, 'swing_high', 0)
                swing_low = getattr(ote, 'swing_low', 0)
                range_size = swing_high - swing_low
                if ote.direction == "bullish":
                    # Bullish OTE: retracement from high to low
                    fib_618 = swing_high - (range_size * 0.618)
                    fib_705 = swing_high - (range_size * 0.705)
                    fib_79 = swing_high - (range_size * 0.79)
                else:
                    # Bearish OTE: retracement from low to high
                    fib_618 = swing_low + (range_size * 0.618)
                    fib_705 = swing_low + (range_size * 0.705)
                    fib_79 = swing_low + (range_size * 0.79)

                ote_zones_serialized.append({
                    "type": ote.direction,  # "bullish" or "bearish"
                    "fib_618": fib_618,
                    "fib_705": fib_705,
                    "fib_79": fib_79,
                    "swing_high": swing_high,
                    "swing_low": swing_low
                })

        premium_discount_serialized = None
        if premium_discount and hasattr(premium_discount, 'zone'):
            range_high = getattr(premium_discount, 'range_high', 0)
            range_low = getattr(premium_discount, 'range_low', 0)
            equilibrium = getattr(premium_discount, 'equilibrium', (range_high + range_low) / 2)
            premium_discount_serialized = {
                "current_zone": premium_discount.zone,  # "premium", "discount", or "equilibrium"
                "equilibrium": equilibrium,
                "premium_start": equilibrium,  # Premium starts at 50%
                "discount_end": equilibrium,  # Discount ends at 50%
                "range_high": range_high,
                "range_low": range_low,
                "position_percent": getattr(premium_discount, 'position_pct', 50)
            }

        confluence_score_serialized = None
        if confluence_score and hasattr(confluence_score, 'total_score'):
            # Map factors to categories
            factors = getattr(confluence_score, 'factors', [])
            bullish_factors = getattr(confluence_score, 'bullish_factors', 0)
            bearish_factors = getattr(confluence_score, 'bearish_factors', 0)

            # Determine recommendation based on score and factors
            total_score = confluence_score.total_score
            if total_score >= 70:
                recommendation = "bullish" if bullish_factors > bearish_factors else "bearish" if bearish_factors > bullish_factors else "neutral"
            elif total_score >= 50:
                recommendation = "cautious_" + ("bullish" if bullish_factors > bearish_factors else "bearish" if bearish_factors > bullish_factors else "neutral")
            else:
                recommendation = "wait"

            confluence_score_serialized = {
                "total_score": total_score / 10,  # Convert 0-100 to 0-10 scale for UI
                "bias_alignment": 1 if bullish_factors > 0 or bearish_factors > 0 else 0,
                "zone_proximity": len([f for f in factors if "zone" in f.lower() or "ob" in f.lower() or "fvg" in f.lower()]),
                "structure_confirmation": len([f for f in factors if "bos" in f.lower() or "choc" in f.lower()]),
                "liquidity_target": len([f for f in factors if "liquidity" in f.lower()]),
                "mtf_alignment": len([f for f in factors if "timeframe" in f.lower() or "htf" in f.lower()]),
                "session_factor": len([f for f in factors if "session" in f.lower() or "london" in f.lower() or "ny" in f.lower()]),
                "recommendation": recommendation
            }

        # Serialize Liquidity Sweeps
        liquidity_sweeps_serialized = []
        sweeps_list = (
            liquidity_sweeps_data.get('recent', []) if isinstance(liquidity_sweeps_data, dict)
            else liquidity_sweeps_data if isinstance(liquidity_sweeps_data, list) else []
        )
        for sweep in sweeps_list:
            if hasattr(sweep, 'type'):
                liquidity_sweeps_serialized.append({
                    "type": sweep.type,  # "bullish" or "bearish"
                    "sweep_level": sweep.sweep_level,
                    "sweep_low": getattr(sweep, 'sweep_low', 0),
                    "sweep_high": getattr(sweep, 'sweep_high', 0),
                    "close_price": getattr(sweep, 'close_price', 0),
                    "rejection_strength": getattr(sweep, 'rejection_strength', 0),
                    "atr_penetration": getattr(sweep, 'atr_penetration', 0),
                    "timestamp": sweep.timestamp,
                    "session": getattr(sweep, 'session', None)
                })

        # Serialize Inducements
        inducements_serialized = []
        inducements_list = (
            inducements_data.get('recent', []) if isinstance(inducements_data, dict)
            else inducements_data if isinstance(inducements_data, list) else []
        )
        for ind in inducements_list:
            if hasattr(ind, 'type'):
                inducements_serialized.append({
                    "type": ind.type,  # "bullish" or "bearish"
                    "inducement_level": ind.inducement_level,
                    "inducement_index": getattr(ind, 'inducement_index', 0),
                    "break_index": getattr(ind, 'break_index', 0),
                    "reversal_index": getattr(ind, 'reversal_index', 0),
                    "target_liquidity": getattr(ind, 'target_liquidity', None),
                    "timestamp": ind.timestamp,
                    "trapped_direction": getattr(ind, 'trapped_direction', '')
                })

        # Serialize Rejection Blocks
        rejection_blocks_serialized = []
        rejection_list = (
            rejection_blocks_data.get('unmitigated', []) if isinstance(rejection_blocks_data, dict)
            else rejection_blocks_data if isinstance(rejection_blocks_data, list) else []
        )
        for rb in rejection_list:
            if hasattr(rb, 'type'):
                rejection_blocks_serialized.append({
                    "type": rb.type,  # "bullish" or "bearish"
                    "rejection_price": rb.rejection_price,
                    "body_top": getattr(rb, 'body_top', 0),
                    "body_bottom": getattr(rb, 'body_bottom', 0),
                    "wick_size": getattr(rb, 'wick_size', 0),
                    "wick_atr_ratio": getattr(rb, 'wick_atr_ratio', 0),
                    "timestamp": rb.timestamp,
                    "session": getattr(rb, 'session', None),
                    "held": getattr(rb, 'held', True),
                    "mitigated": getattr(rb, 'mitigated', False)
                })

        # Serialize Turtle Soup patterns
        turtle_soup_serialized = []
        turtle_list = (
            turtle_soup_data.get('recent', []) if isinstance(turtle_soup_data, dict)
            else turtle_soup_data if isinstance(turtle_soup_data, list) else []
        )
        for ts in turtle_list:
            if hasattr(ts, 'type'):
                turtle_soup_serialized.append({
                    "type": ts.type,  # "bullish" or "bearish"
                    "level": ts.level,
                    "penetration": getattr(ts, 'penetration', 0),
                    "penetration_atr": getattr(ts, 'penetration_atr', 0),
                    "timestamp": ts.timestamp,
                    "swing_index": getattr(ts, 'swing_index', 0)
                })

        # Serialize Pattern Alerts
        alerts_serialized = []
        for alert in (pattern_alerts if isinstance(pattern_alerts, list) else []):
            if hasattr(alert, 'priority'):
                alerts_serialized.append({
                    "priority": alert.priority,  # "HIGH", "MEDIUM", "LOW"
                    "pattern_type": alert.pattern_type,
                    "direction": alert.direction,
                    "message": alert.message,
                    "price_level": getattr(alert, 'price_level', None),
                    "timestamp": alert.timestamp,
                    "action_suggestion": getattr(alert, 'action_suggestion', None)
                })
            elif isinstance(alert, dict):
                alerts_serialized.append(alert)

        # Serialize structure breaks (BOS/CHoCH) for chart rendering
        def _serialize_structure_point(sp):
            if hasattr(sp, 'type'):
                return {
                    "type": sp.type,
                    "price": float(sp.price),
                    "break_type": getattr(sp, 'break_type', 'BOS'),
                    "break_index": getattr(sp, 'break_index', None),
                    "timestamp": getattr(sp, 'timestamp', ''),
                }
            elif isinstance(sp, dict):
                return {
                    "type": sp.get("type", ""),
                    "price": float(sp.get("price", 0)),
                    "break_type": sp.get("break_type", "BOS"),
                    "break_index": sp.get("break_index"),
                    "timestamp": sp.get("timestamp", ""),
                }
            return None

        structure_serialized = {"recent_bos": [], "recent_choc": [], "all_bos": [], "all_choc": []}

        # Serialize all structure breaks
        for bos_point in structure.get("all_bos", []):
            serialized = _serialize_structure_point(bos_point)
            if serialized:
                structure_serialized["all_bos"].append(serialized)
        for choc_point in structure.get("all_choc", []):
            serialized = _serialize_structure_point(choc_point)
            if serialized:
                structure_serialized["all_choc"].append(serialized)

        # Serialize recent breaks using same helper
        for bos_point in structure.get("recent_bos", []):
            serialized = _serialize_structure_point(bos_point)
            if serialized:
                structure_serialized["recent_bos"].append(serialized)
        for choc_point in structure.get("recent_choc", []):
            serialized = _serialize_structure_point(choc_point)
            if serialized:
                structure_serialized["recent_choc"].append(serialized)

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "liquidity_zones": liquidity_zones,
            "pdh": pdh,  # Previous Day High
            "pdl": pdl,  # Previous Day Low
            "bias": bias,
            "bias_summary": bias_summary,
            "bias_factors": bias_factors,
            "key_levels": key_levels,
            "current_price": round(current_price, 2),
            # Structure breaks for chart labels
            "structure": structure_serialized,
            # NEW: Include new SMC features
            "equal_levels": equal_levels_serialized,
            "breaker_blocks": breaker_blocks_serialized,
            "ote_zones": ote_zones_serialized,
            "premium_discount": premium_discount_serialized,
            "confluence_score": confluence_score_serialized,
            # Advanced SMC patterns
            "liquidity_sweeps": liquidity_sweeps_serialized,
            "inducements": inducements_serialized,
            "rejection_blocks": rejection_blocks_serialized,
            "turtle_soup": turtle_soup_serialized,
            "alerts": alerts_serialized,
            "summary": {
                "total_obs": ob_data.get("total", 0),
                "unmitigated_obs": ob_data.get("unmitigated", 0),
                "total_fvgs": fvg_data.get("total", 0),
                "unmitigated_fvgs": fvg_data.get("unmitigated", 0),
                "bullish_fvgs": bullish_fvgs,
                "bearish_fvgs": bearish_fvgs,
                "bullish_obs": bullish_obs,
                "bearish_obs": bearish_obs,
                # NEW: Summary of new features
                "equal_levels_count": len(equal_levels_serialized),
                "breaker_blocks_count": len(breaker_blocks_serialized),
                "ote_zones_count": len(ote_zones_serialized),
                # Advanced pattern counts
                "liquidity_sweeps_count": len(liquidity_sweeps_serialized),
                "inducements_count": len(inducements_serialized),
                "rejection_blocks_count": len(rejection_blocks_serialized),
                "turtle_soup_count": len(turtle_soup_serialized),
                "active_alerts_count": len(alerts_serialized)
            }
        }

        # Add debug info if requested
        if debug:
            response["debug"] = {
                "lookback_bars": lookback,
                "fvg_min_size_atr": fvg_min_size,
                "all_fvgs_detected": len(fair_value_gaps),
                "mitigated_fvgs": len([f for f in fair_value_gaps if safe_attr(f, "mitigated", False)]),
                "data_bars": len(df),
                "fvg_details": fair_value_gaps,  # Full details of all FVGs
                "ob_details": order_blocks       # Full details of all OBs
            }

        return response
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ----- Market Regime -----

@app.get("/api/regime/{symbol}")
async def get_market_regime(symbol: str, timeframe: str = "H1"):
    """Detect current market regime"""
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 0, 100)
        if rates is None:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        df = pd.DataFrame(rates)

        # Calculate ATR for volatility
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        avg_atr = high_low.rolling(50).mean().iloc[-1]

        # Calculate ADX for trend strength
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = high_low.copy()
        atr_14 = tr.rolling(14).mean()

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]

        # Determine regime
        volatility = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

        if adx > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                trend = "bullish_trending"
            else:
                trend = "bearish_trending"
        else:
            trend = "ranging"

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": trend,
            "volatility": volatility,
            "adx": round(adx, 2) if not np.isnan(adx) else None,
            "atr": round(atr, 5) if not np.isnan(atr) else None,
            "atr_percentile": round((atr / avg_atr) * 100, 1) if avg_atr > 0 else None
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


# ----- LLM Health Check -----

@app.get("/api/llm/status")
async def check_llm_status():
    """
    Check if the LLM API is available and has credits.
    Makes a minimal test call to verify the API is working.
    """
    import os
    import time

    start_time = time.time()
    provider = None
    model = None
    status = "unknown"
    message = ""
    error_type = None

    try:
        # Determine which provider is configured
        if os.getenv("XAI_API_KEY"):
            provider = "xAI"
            api_key = os.getenv("XAI_API_KEY")
            base_url = "https://api.x.ai/v1"
            model = "grok-3-mini-fast"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "OpenAI"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
            model = "gpt-4o-mini"
        else:
            return {
                "status": "not_configured",
                "provider": None,
                "model": None,
                "message": "No LLM API key configured. Set XAI_API_KEY or OPENAI_API_KEY.",
                "response_time_ms": 0,
                "recommendation": "rule-based"
            }

        # Make a minimal test call
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Minimal prompt to test API - just ask for a single word
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only the word 'OK'"}],
            max_tokens=5,
            temperature=0
        )

        response_time = (time.time() - start_time) * 1000

        if response and response.choices:
            status = "available"
            message = "LLM API is working and has credits"
            return {
                "status": status,
                "provider": provider,
                "model": model,
                "message": message,
                "response_time_ms": round(response_time, 0),
                "recommendation": "ai" if response_time < 5000 else "rule-based"
            }

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        error_str = str(e).lower()

        # Detect credit/quota errors
        if any(term in error_str for term in ["credit", "quota", "billing", "insufficient", "exceeded", "limit"]):
            status = "no_credits"
            error_type = "credits_exhausted"
            message = "LLM credits exhausted or billing issue"
        elif any(term in error_str for term in ["rate", "too many"]):
            status = "rate_limited"
            error_type = "rate_limit"
            message = "Rate limited - too many requests"
        elif any(term in error_str for term in ["auth", "key", "invalid", "unauthorized"]):
            status = "auth_error"
            error_type = "authentication"
            message = "API key is invalid or unauthorized"
        elif any(term in error_str for term in ["timeout", "timed out"]):
            status = "timeout"
            error_type = "timeout"
            message = "API request timed out"
        elif any(term in error_str for term in ["connect", "network", "unreachable"]):
            status = "network_error"
            error_type = "network"
            message = "Cannot connect to LLM API"
        else:
            status = "error"
            error_type = "unknown"
            message = f"LLM error: {str(e)[:100]}"

        return {
            "status": status,
            "provider": provider,
            "model": model,
            "message": message,
            "error_type": error_type,
            "error_detail": str(e)[:200],
            "response_time_ms": round(response_time, 0),
            "recommendation": "rule-based"
        }


# ----- Rule-Based Analysis (No LLM) -----

class RuleBasedAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/rule-based")
async def run_rule_based_analysis(request: RuleBasedAnalysisRequest):
    """
    Run analysis using pure SMC rules without any LLM calls.
    This is a fallback when LLM credits are exhausted or for fast/free analysis.
    """
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2

        # Map timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get price data
        rates = mt5.copy_rates_from_pos(request.symbol, tf, 0, 200)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Market Regime ===
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        avg_atr = high_low.rolling(50).mean().iloc[-1]

        # ADX calculation
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = high_low.copy()
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean().iloc[-1]

        # Determine regime
        volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

        if adx > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                market_regime = "trending-up"
            else:
                market_regime = "trending-down"
        else:
            market_regime = "ranging"

        # === HTF Bias Check ===
        htf_info = _calculate_htf_bias(request.symbol, request.timeframe)

        # === STEP 2: Run SMC Analysis ===
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer

        analyzer = SmartMoneyAnalyzer()
        smc_result = analyzer.analyze_full_smc(
            df,
            current_price=current_price,
            use_structural_obs=True
        )

        # === STEP 3: Generate Rule-Based Trade Plan ===
        from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator

        generator = SMCTradePlanGenerator(
            min_quality_score=50.0,  # Slightly lower threshold for rule-based
            min_rr_ratio=1.5,
            sl_buffer_atr=0.5,
            entry_zone_percent=0.5
        )

        # Flatten SMC analysis for the generator
        flat_smc = {
            "order_blocks": smc_result.get("order_blocks", {}),
            "fair_value_gaps": smc_result.get("fair_value_gaps", {}),
            "liquidity_zones": smc_result.get("liquidity_zones", {}),
            "market_structure": smc_result.get("market_structure", {}),
            "equal_levels": smc_result.get("equal_levels", {}),
            "atr": atr if not np.isnan(atr) else current_price * 0.01,
        }

        base_plan = generator.generate_plan(
            smc_analysis=flat_smc,
            current_price=current_price,
            atr=atr if not np.isnan(atr) else current_price * 0.01,
            market_regime=market_regime,
            session=None
        )

        # === STEP 4: Build Response ===
        if base_plan:
            # Calculate R:R
            entry = base_plan.entry_price
            sl = base_plan.stop_loss
            tp = base_plan.take_profit
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr_ratio = reward / risk if risk > 0 else 0

            # Build checklist summary
            checklist_items = []
            if base_plan.checklist.htf_trend_aligned:
                checklist_items.append("HTF trend aligned")
            if base_plan.checklist.zone_unmitigated:
                checklist_items.append("Zone unmitigated")
            if base_plan.checklist.has_confluence:
                checklist_items.append("Has confluence")
            if base_plan.checklist.liquidity_target_exists:
                checklist_items.append("Liquidity target exists")
            if base_plan.checklist.structure_confirmed:
                checklist_items.append("Structure confirmed")
            if base_plan.checklist.in_discount_premium:
                checklist_items.append("Premium/Discount correct")
            if base_plan.checklist.session_favorable:
                checklist_items.append("Session favorable")

            # Build rationale
            rationale = f"""## Rule-Based SMC Analysis

### Market Conditions
- **Regime**: {market_regime}
- **Volatility**: {volatility_regime}
- **ADX**: {adx:.1f} ({"Strong trend" if adx > 25 else "Weak/No trend"})

### Trade Setup
- **Signal**: {base_plan.signal}
- **Setup Type**: {base_plan.setup_type.value}
- **Zone Quality**: {base_plan.zone_quality_score:.0f}/100
- **Risk:Reward**: 1:{rr_ratio:.2f}

### Entry Checklist ({base_plan.checklist.passed_count}/{base_plan.checklist.total_count} passed)
{chr(10).join(f"- ✅ {item}" for item in checklist_items)}

### Confluence Factors
{chr(10).join(f"- {factor}" for factor in base_plan.confluence_factors)}

### Recommendation
**{base_plan.recommendation}**{f" - {base_plan.skip_reason}" if base_plan.skip_reason else ""}

---
*This analysis was generated using systematic SMC rules without AI interpretation. For more nuanced analysis with market context, run the full multi-agent analysis.*
"""

            # Respect the trade plan's recommendation — SKIP means don't trade
            effective_signal = base_plan.signal if base_plan.recommendation == "TAKE" else "HOLD"

            decision = {
                "signal": effective_signal,
                # Map quality 50-100 → confidence 0.65-1.0 (50 is the min threshold, so any signal should be executable)
                "confidence": (0.65 + (base_plan.zone_quality_score - 50) * 0.35 / 50) if effective_signal != "HOLD" else 0.0,
                "entry_price": round(entry, 5),
                "stop_loss": round(sl, 5),
                "take_profit": round(tp, 5),
                "rationale": rationale,
                "setup_type": base_plan.setup_type.value,
                "key_factors": base_plan.confluence_factors,
                "analysis_mode": "rule-based",
                "zone_quality": base_plan.zone_quality_score,
                "rr_ratio": round(rr_ratio, 2),
                "checklist": {
                    "passed": base_plan.checklist.passed_count,
                    "total": base_plan.checklist.total_count,
                    "items": checklist_items
                }
            }
        else:
            # No valid setup found
            decision = {
                "signal": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "rationale": f"""## Rule-Based SMC Analysis

### Market Conditions
- **Regime**: {market_regime}
- **Volatility**: {volatility_regime}
- **ADX**: {adx:.1f}

### No Valid Setup Found

The systematic SMC rules did not identify a high-probability trade setup at this time.

**Possible reasons:**
- No unmitigated zones near current price
- Zone quality below threshold (50)
- Risk:Reward ratio below minimum (1.5:1)
- Trend not aligned with potential entries

**Recommendation**: Wait for price to approach a quality SMC zone or run full multi-agent analysis for deeper context.

---
*This analysis was generated using systematic SMC rules without AI interpretation.*
""",
                "setup_type": None,
                "key_factors": [],
                "analysis_mode": "rule-based",
                "zone_quality": 0,
                "rr_ratio": 0,
                "checklist": None
            }

        # Include SMC levels for chart display
        smc_levels = []

        # Helper to safely get attribute from dataclass or dict
        def safe_attr(obj, attr, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        # Order blocks
        obs = smc_result.get("order_blocks", {})
        for ob in (obs.get("bullish") or []):
            if ob and not safe_attr(ob, "mitigated", False):
                smc_levels.append({
                    "type": "order_block",
                    "price": (safe_attr(ob, "top", 0) + safe_attr(ob, "bottom", 0)) / 2,
                    "direction": "bullish",
                    "strength": safe_attr(ob, "strength", 0.5)
                })
        for ob in (obs.get("bearish") or []):
            if ob and not safe_attr(ob, "mitigated", False):
                smc_levels.append({
                    "type": "order_block",
                    "price": (safe_attr(ob, "top", 0) + safe_attr(ob, "bottom", 0)) / 2,
                    "direction": "bearish",
                    "strength": safe_attr(ob, "strength", 0.5)
                })

        # FVGs
        fvgs = smc_result.get("fair_value_gaps", {})
        for fvg in (fvgs.get("bullish") or []):
            if fvg and not safe_attr(fvg, "mitigated", False):
                smc_levels.append({
                    "type": "fvg",
                    "price": (safe_attr(fvg, "top", 0) + safe_attr(fvg, "bottom", 0)) / 2,
                    "direction": "bullish"
                })
        for fvg in (fvgs.get("bearish") or []):
            if fvg and not safe_attr(fvg, "mitigated", False):
                smc_levels.append({
                    "type": "fvg",
                    "price": (safe_attr(fvg, "top", 0) + safe_attr(fvg, "bottom", 0)) / 2,
                    "direction": "bearish"
                })

        # Liquidity - liquidity_zones is a list, not a dict
        liq_zones = smc_result.get("liquidity_zones", [])
        # Handle both list and dict formats
        if isinstance(liq_zones, dict):
            # Dict format with buy_side/sell_side keys
            for zone in (liq_zones.get("buy_side") or []):
                if zone:
                    smc_levels.append({
                        "type": "liquidity",
                        "price": safe_attr(zone, "price", 0),
                        "direction": "bullish"
                    })
            for zone in (liq_zones.get("sell_side") or []):
                if zone:
                    smc_levels.append({
                        "type": "liquidity",
                        "price": safe_attr(zone, "price", 0),
                        "direction": "bearish"
                    })
        else:
            # List format - each zone has a 'type' field
            for zone in liq_zones:
                if zone:
                    zone_type = safe_attr(zone, "type", "")
                    direction = "bullish" if zone_type == "buy-side" else "bearish"
                    smc_levels.append({
                        "type": "liquidity",
                        "price": safe_attr(zone, "price", 0),
                        "direction": direction
                    })

        return {
            "status": "completed",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": round(current_price, 5),
            "decision": decision,
            "smc_levels": smc_levels,
            "regime": {
                "market_regime": market_regime,
                "volatility": volatility_regime,
                "adx": round(adx, 2) if not np.isnan(adx) else None,
                "atr": round(atr, 5) if not np.isnan(atr) else None
            },
            "htf_bias": htf_info,
            "analysis_mode": "rule-based",
            "llm_used": False
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ----- XGBoost Strategy Analysis (No LLM, local inference) -----

class XGBoostAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "D1"
    strategy: str = ""  # Empty = auto-select best strategy
    ensemble: bool = False  # If True, use ensemble voting


@app.post("/api/analysis/xgboost")
async def run_xgboost_analysis(request: XGBoostAnalysisRequest):
    """
    Run XGBoost strategy analysis. No LLM call — pure ML inference.
    Sub-100ms response time.
    """
    import time as _time
    start = _time.time()

    try:
        from tradingagents.automation.auto_tuner import load_mt5_data, _compute_atr
        from tradingagents.xgb_quant.predictor import LivePredictor
        import numpy as np_local

        df = load_mt5_data(request.symbol, request.timeframe, bars=500)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        atr = _compute_atr(high, low, close)
        current_atr = float(atr[-1]) if not np_local.isnan(atr[-1]) else 1.0
        current_price = float(close[-1])

        predictor = LivePredictor()

        if request.ensemble:
            available = predictor.get_available_models(request.symbol, request.timeframe)
            if len(available) < 2:
                return {
                    "status": "success",
                    "decision": {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "rationale": f"Only {len(available)} models available for ensemble, need 2+",
                    },
                    "available_models": available,
                    "duration_seconds": _time.time() - start,
                }
            signal = predictor.predict_ensemble(
                available, request.symbol, request.timeframe,
                df, current_price, current_atr,
            )
        else:
            strategy_name = request.strategy
            if not strategy_name:
                from tradingagents.xgb_quant.strategy_selector import StrategySelector
                selector = StrategySelector()
                selection = selector.select(request.symbol)
                strategy_name = selection.recommended_strategy

            signal = predictor.predict_single(
                strategy_name, request.symbol, request.timeframe,
                df, current_price, current_atr,
            )

        return {
            "status": "success",
            "decision": {
                "signal": signal.direction,
                "confidence": signal.confidence,
                "entry_price": signal.entry,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "rationale": signal.rationale,
                "strategies_agreed": signal.strategies_agreed,
            },
            "available_models": predictor.get_available_models(request.symbol, request.timeframe),
            "duration_seconds": _time.time() - start,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


class XGBoostTrainRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "D1"
    timeframes: list = []  # Empty = just use timeframe field; non-empty = train across all listed TFs
    strategies: list = []  # Empty = all strategies
    bars: int = 2000
    use_trade_labels: bool = False
    skip_existing: bool = True  # Skip strategies that already have a trained model for this symbol+TF


@app.post("/api/xgboost/train")
async def train_xgboost_models(request: XGBoostTrainRequest):
    """Train XGBoost models for a symbol. Runs walk-forward backtest and saves models."""
    import time as _time
    start = _time.time()

    try:
        from tradingagents.automation.auto_tuner import load_mt5_data
        from tradingagents.xgb_quant.trainer import WalkForwardTrainer
        from tradingagents.xgb_quant.strategies.trend_following import TrendFollowingStrategy
        from tradingagents.xgb_quant.strategies.mean_reversion import MeanReversionStrategy
        from tradingagents.xgb_quant.strategies.breakout import BreakoutStrategy
        from tradingagents.xgb_quant.strategies.smc_zones import SMCZonesStrategy
        from tradingagents.xgb_quant.strategies.volume_profile_strat import VolumeProfileStrategy

        all_strategies = {
            "trend_following": TrendFollowingStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "smc_zones": SMCZonesStrategy,
            "volume_profile_strat": VolumeProfileStrategy,
        }

        # Filter strategies if specified
        if request.strategies:
            selected = {k: v for k, v in all_strategies.items() if k in request.strategies}
        else:
            selected = all_strategies

        if not selected:
            return {"status": "error", "error": f"No valid strategies. Available: {list(all_strategies.keys())}"}

        # Determine timeframes to train across
        timeframes = request.timeframes if request.timeframes else [request.timeframe]

        from tradingagents.xgb_quant.config import MODELS_DIR

        trainer = WalkForwardTrainer()
        results = []

        for tf in timeframes:
            # Load data for this timeframe
            df = await asyncio.to_thread(load_mt5_data, request.symbol, tf, request.bars)

            for name, strategy_cls in selected.items():
                # Skip if model already exists
                if request.skip_existing:
                    model_path = MODELS_DIR / name / f"{request.symbol}_{tf}.json"
                    if model_path.exists():
                        results.append({
                            "strategy": name,
                            "timeframe": tf,
                            "status": "skipped",
                            "reason": "Model already exists",
                        })
                        continue

                try:
                    strategy = strategy_cls()
                    result = await asyncio.to_thread(
                        trainer.train_and_evaluate,
                        strategy=strategy,
                        df=df,
                        symbol=request.symbol,
                        timeframe=tf,
                        use_trade_labels=request.use_trade_labels,
                    )
                    results.append({
                        "strategy": name,
                        "timeframe": tf,
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                        "sharpe": result.sharpe,
                        "max_drawdown_pct": result.max_drawdown_pct,
                        "status": "success",
                    })
                except Exception as e:
                    results.append({
                        "strategy": name,
                        "timeframe": tf,
                        "status": "error",
                        "error": str(e),
                    })

        # For skipped models, load their existing results so we can compare
        from tradingagents.xgb_quant.config import RESULTS_DIR
        import json as _json
        for r in results:
            if r["status"] == "skipped":
                result_file = RESULTS_DIR / request.symbol / f"{r['strategy']}_{r['timeframe']}.json"
                if result_file.exists():
                    try:
                        data = _json.loads(result_file.read_text())
                        r["total_trades"] = data.get("total_trades", 0)
                        r["win_rate"] = data.get("win_rate", 0)
                        r["profit_factor"] = data.get("profit_factor", 0)
                        r["sharpe"] = data.get("sharpe", 0)
                        r["max_drawdown_pct"] = data.get("max_drawdown_pct", 0)
                    except Exception:
                        pass

        # Find best combo across trained + skipped
        candidates = [r for r in results if r.get("total_trades", 0) > 0 and r["status"] in ("success", "skipped")]
        best = max(candidates, key=lambda r: r.get("sharpe", 0)) if candidates else None

        return {
            "status": "success",
            "symbol": request.symbol,
            "timeframes": timeframes,
            "results": results,
            "best": {
                "strategy": best["strategy"],
                "timeframe": best["timeframe"],
                "sharpe": best["sharpe"],
                "profit_factor": best["profit_factor"],
                "win_rate": best["win_rate"],
            } if best else None,
            "duration_seconds": round(_time.time() - start, 1),
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/xgboost/models")
async def get_xgboost_models():
    """List all available trained XGBoost models."""
    try:
        from tradingagents.xgb_quant.config import MODELS_DIR
        models = {}
        if MODELS_DIR.exists():
            for strategy_dir in MODELS_DIR.iterdir():
                if strategy_dir.is_dir():
                    strategy_name = strategy_dir.name
                    models[strategy_name] = []
                    for model_file in strategy_dir.glob("*.json"):
                        parts = model_file.stem.split("_")
                        if len(parts) >= 2:
                            models[strategy_name].append({
                                "symbol": parts[0],
                                "timeframe": parts[1],
                                "file": model_file.name,
                            })
        return {"status": "success", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/xgboost/performance-matrix")
async def get_xgboost_performance_matrix():
    """Get performance matrix: symbol × strategy → metrics."""
    try:
        from tradingagents.xgb_quant.strategy_selector import StrategySelector
        selector = StrategySelector()
        matrix = selector.get_performance_matrix()
        return {"status": "success", "matrix": matrix}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class PairScanRequest(BaseModel):
    min_score: int = 40
    max_candidates: int = 10
    timeframe: str = "H4"


@app.post("/api/xgboost/scan")
async def run_pair_scan(request: PairScanRequest = PairScanRequest()):
    """Scan watchlist for pairs on the move."""
    try:
        from tradingagents.xgb_quant.scanner import PairScanner
        from tradingagents.xgb_quant.config import ScannerConfig
        from dataclasses import asdict

        cfg = ScannerConfig(
            scan_timeframe=request.timeframe,
            min_momentum_score=request.min_score,
        )
        scanner = PairScanner(config=cfg)

        # Get existing positions to mark them
        existing = []
        try:
            from tradingagents.mt5_interface import get_open_positions
            positions = get_open_positions()
            if positions:
                existing = list({p.get("symbol", "") for p in positions})
        except Exception:
            pass

        result = scanner.scan(existing_positions=existing)

        shortlist = result.shortlist[:request.max_candidates]

        return {
            "status": "success",
            "timestamp": result.timestamp,
            "watchlist_size": result.watchlist_size,
            "shortlist": [asdict(s) for s in shortlist],
            "disqualified": [asdict(s) for s in result.disqualified],
            "disqualified_count": len(result.disqualified),
            "best_candidate": asdict(result.best_candidate) if result.best_candidate else None,
        }
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/xgboost/scanner-status")
async def get_scanner_status(instance: str = ""):
    """Get scanner status from a running automation instance."""
    try:
        # Check local instances first
        if instance and instance in _automation_instances:
            inst = _automation_instances[instance]
            if hasattr(inst, "automation") and inst.automation:
                status = inst.automation.get_scanner_status()
                return {"status": "success", "scanner": status}

        # Check all local instances for any scanner data
        for name, inst in _automation_instances.items():
            if hasattr(inst, "automation") and inst.automation:
                status = inst.automation.get_scanner_status()
                if status:
                    return {"status": "success", "instance": name, "scanner": status}

        return {"status": "success", "scanner": None, "message": "No active scanner found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ----- XGBoost Batch Training (Weekly Job) -----
# Runs in the MT5 worker via command queue (has MT5 data access).
# Status updates arrive via WebSocket broadcast from the worker.

_batch_train_task: Dict[str, Any] = {}  # Updated by WebSocket broadcasts from worker


class BatchTrainRequest(BaseModel):
    symbols: List[str] = []  # Empty = full watchlist
    timeframes: List[str] = ["D1", "H4"]
    strategies: List[str] = []  # Empty = all 5
    bars: int = 2000
    skip_fresh_days: int = 0  # 0 = retrain all, 7 = skip models < 7 days old


@app.post("/api/xgboost/batch-train")
async def start_batch_training(request: BatchTrainRequest = BatchTrainRequest()):
    """Send batch training command to the MT5 worker."""
    if _batch_train_task.get("status") == "running":
        return {"status": "error", "error": "Batch training already running", "task": _batch_train_task}

    try:
        control = _get_automation_control()

        payload = {
            "symbols": request.symbols,
            "timeframes": request.timeframes,
            "strategies": request.strategies,
            "bars": request.bars,
            "skip_fresh_days": request.skip_fresh_days,
        }

        command_id = await control.send_command(
            instance_name="batch_trainer",
            action="batch_train",
            payload=payload,
        )

        _batch_train_task.clear()
        _batch_train_task.update({
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "current": 0,
            "total": 0,
            "message": "Sent to MT5 worker, waiting for start...",
            "command_id": command_id,
            "result": None,
        })

        return {"status": "started", "message": "Batch training sent to MT5 worker", "command_id": command_id}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/xgboost/batch-train/status")
async def get_batch_training_status():
    """Get current batch training progress (updated via WebSocket from worker)."""
    if not _batch_train_task:
        return {"status": "idle", "message": "No batch training has been run"}
    return _batch_train_task


@app.post("/api/xgboost/batch-train/cancel")
async def cancel_batch_training():
    """Send cancel command to the MT5 worker."""
    if _batch_train_task.get("status") != "running":
        return {"status": "error", "error": "No running batch training to cancel"}
    try:
        control = _get_automation_control()
        await control.send_command(
            instance_name="batch_trainer",
            action="stop",
            payload={},
        )
        _batch_train_task["status"] = "cancelling"
        _batch_train_task["message"] = "Cancel sent to worker..."
        return {"status": "cancelling"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ----- XGBoost Per-Pair Optimization -----

_optimize_task: Dict[str, Any] = {}


class OptimizeRequest(BaseModel):
    symbols: List[str] = []
    timeframes: List[str] = ["D1", "H4"]
    strategies: List[str] = []
    bars: int = 2000
    max_hours: float = 6.0


@app.post("/api/xgboost/optimize")
async def start_pair_optimization(request: OptimizeRequest = OptimizeRequest()):
    """Send per-pair optimization command to the MT5 worker."""
    if _optimize_task.get("status") == "running":
        return {"status": "error", "error": "Optimization already running", "task": _optimize_task}

    try:
        control = _get_automation_control()
        payload = {
            "symbols": request.symbols,
            "timeframes": request.timeframes,
            "strategies": request.strategies,
            "bars": request.bars,
            "max_hours": request.max_hours,
        }
        command_id = await control.send_command(
            instance_name="pair_optimizer",
            action="optimize",
            payload=payload,
        )

        _optimize_task.clear()
        _optimize_task.update({
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "current": 0,
            "total": 0,
            "message": "Sent to MT5 worker, waiting for start...",
            "command_id": command_id,
            "result": None,
        })
        return {"status": "started", "command_id": command_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/xgboost/optimize/status")
async def get_optimization_status():
    """Get per-pair optimization progress."""
    if not _optimize_task:
        return {"status": "idle", "message": "No optimization has been run"}
    return _optimize_task


@app.post("/api/xgboost/optimize/cancel")
async def cancel_optimization():
    """Cancel running optimization."""
    if _optimize_task.get("status") != "running":
        return {"status": "error", "error": "No running optimization to cancel"}
    try:
        control = _get_automation_control()
        await control.send_command(instance_name="pair_optimizer", action="stop", payload={})
        _optimize_task["status"] = "cancelling"
        _optimize_task["message"] = "Cancel sent to worker..."
        return {"status": "cancelling"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ----- Gold/Silver Pullback Backtest -----

_gs_backtest_task = {"status": "idle"}

class GoldSilverBacktestRequest(BaseModel):
    bars: int = 800
    min_trades: int = 3
    timeframes: list = ["D1", "H4"]


@app.post("/api/backtest/gold-silver-pullback")
async def start_gold_silver_backtest(request: GoldSilverBacktestRequest = GoldSilverBacktestRequest()):
    """Run gold/silver pullback strategy backtest via auto-tuner."""
    if _gs_backtest_task.get("status") == "running":
        return {"status": "already_running"}

    _gs_backtest_task.update({
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "progress": {"phase": "starting", "current": 0, "total": 0, "message": "Starting backtest...", "steps": []},
        "result": None,
        "error": None,
    })

    async def _run_gs_backtest():
        loop = asyncio.get_running_loop()

        def progress_callback(phase, current, total, message, steps=None):
            _gs_backtest_task["progress"] = {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "steps": steps or _gs_backtest_task["progress"].get("steps", []),
            }

        try:
            from tradingagents.automation.auto_tuner import run_tune
            result = await run_tune(
                symbol="XAUUSD",
                pipeline="gold_silver_pullback",
                timeframes=request.timeframes,
                bars=request.bars,
                min_trades=request.min_trades,
                progress_callback=progress_callback,
            )
            if result.get("error"):
                _gs_backtest_task["status"] = "error"
                _gs_backtest_task["error"] = result["error"]
                _gs_backtest_task["result"] = result
            else:
                _gs_backtest_task["status"] = "done"
                _gs_backtest_task["result"] = result
        except Exception as e:
            import traceback
            traceback.print_exc()
            _gs_backtest_task["status"] = "error"
            _gs_backtest_task["error"] = str(e)

    asyncio.create_task(_run_gs_backtest())
    return {"status": "started"}


@app.get("/api/backtest/gold-silver-pullback/status")
async def get_gold_silver_backtest_status():
    """Get current gold/silver backtest progress."""
    return _gs_backtest_task


# ----- SMC MTF Analysis (Multi-Timeframe OTE & Channel, No LLM) -----

class SmcMtfAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "D1"  # Higher timeframe (lower TF derived automatically)


@app.post("/api/analysis/smc-mtf")
async def run_smc_mtf_analysis(request: SmcMtfAnalysisRequest):
    """
    Run SMC Multi-Timeframe analysis (OTE + Channel + Weekend Gap).
    No LLM calls — pure math like rule-based but with dual-timeframe alignment.
    """
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2

        # Map timeframes
        tf_map = {
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        }

        # Higher TF from request, derive lower TF
        htf_name = request.timeframe.upper()
        ltf_map = {"D1": "H4", "H4": "H1", "H1": "M15", "W1": "D1"}
        ltf_name = ltf_map.get(htf_name, "H1")

        htf = tf_map.get(htf_name, mt5.TIMEFRAME_D1)
        ltf = tf_map.get(ltf_name, mt5.TIMEFRAME_H1)

        # Load data for both timeframes
        htf_rates = mt5.copy_rates_from_pos(request.symbol, htf, 0, 200)
        ltf_rates = mt5.copy_rates_from_pos(request.symbol, ltf, 0, 400)

        if htf_rates is None or len(htf_rates) == 0:
            raise HTTPException(status_code=400, detail=f"Failed to get {htf_name} data")
        if ltf_rates is None or len(ltf_rates) == 0:
            raise HTTPException(status_code=400, detail=f"Failed to get {ltf_name} data")

        htf_df = pd.DataFrame(htf_rates)
        htf_df['time'] = pd.to_datetime(htf_df['time'], unit='s')
        ltf_df = pd.DataFrame(ltf_rates)
        ltf_df['time'] = pd.to_datetime(ltf_df['time'], unit='s')

        # Run MTF analysis
        from tradingagents.agents.analysts.smc_mtf_quant import (
            run_mtf_analysis, calculate_atr, _format_mtf_analysis_for_prompt
        )

        analysis = run_mtf_analysis(
            higher_tf_df=htf_df,
            lower_tf_df=ltf_df,
            current_price=current_price,
        )

        atr = calculate_atr(
            ltf_df['high'].values, ltf_df['low'].values, ltf_df['close'].values
        )
        if np.isnan(atr) or atr <= 0:
            atr = current_price * 0.01

        # Convert MTF analysis to trade decision
        signal = "HOLD"
        confidence = 0.0
        entry_price = None
        stop_loss = None
        take_profit = None

        bias = analysis.trade_bias
        score = analysis.alignment_score

        if score >= 50 and bias in ("bullish", "bearish"):
            signal = "BUY" if bias == "bullish" else "SELL"
            # Map score 50-100 → confidence 0.55-0.95
            confidence = 0.55 + (score - 50) * 0.40 / 50

            # Entry, SL, TP from OTE zone + ATR
            if analysis.ote_zone:
                ote = analysis.ote_zone
                if signal == "BUY":
                    entry_price = round(ote.fib_618, 5)  # OTE level
                    stop_loss = round(ote.swing_low - atr * 0.5, 5)
                    take_profit = round(ote.fib_ext_1272, 5)
                else:
                    entry_price = round(ote.fib_618, 5)
                    stop_loss = round(ote.swing_high + atr * 0.5, 5)
                    take_profit = round(ote.fib_ext_1272, 5)
            else:
                # Fallback: use current price + ATR-based levels
                if signal == "BUY":
                    entry_price = round(current_price, 5)
                    stop_loss = round(current_price - atr * 2, 5)
                    take_profit = round(current_price + atr * 3, 5)
                else:
                    entry_price = round(current_price, 5)
                    stop_loss = round(current_price + atr * 2, 5)
                    take_profit = round(current_price - atr * 3, 5)

            # Require entry confirmation for high-confidence signals
            if not analysis.has_entry_confirmation and confidence > 0.7:
                confidence = 0.65  # Downgrade without confirmation

        # Build rationale
        rationale = _format_mtf_analysis_for_prompt(analysis, current_price, atr)
        rationale += f"\n\n### Alignment Score: {score}/100"
        rationale += f"\n- Trade Bias: **{bias.upper()}**"
        rationale += f"\n- Entry Confirmation: {'Yes (' + analysis.confirmation_type + ')' if analysis.has_entry_confirmation else 'No'}"
        rationale += f"\n- Price in OTE: {'Yes' if analysis.price_in_ote else 'No'}"
        rationale += f"\n- Channel Support: {'Yes' if analysis.price_in_or_touches_channel else 'No'}"
        rationale += "\n\n---\n*SMC Multi-Timeframe analysis (no LLM). HTF={}, LTF={}*".format(htf_name, ltf_name)

        decision = {
            "signal": signal,
            "confidence": round(confidence, 3),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rationale": rationale,
            "analysis_mode": "smc_mtf",
        }

        return {
            "status": "completed",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "higher_tf": htf_name,
            "lower_tf": ltf_name,
            "current_price": round(current_price, 5),
            "decision": decision,
            "mtf_details": {
                "alignment_score": score,
                "trade_bias": bias,
                "htf_bias": analysis.higher_tf_bias,
                "ltf_bias": analysis.lower_tf_bias,
                "price_in_ote": analysis.price_in_ote,
                "has_entry_confirmation": analysis.has_entry_confirmation,
                "confirmation_type": analysis.confirmation_type,
                "price_in_channel": analysis.price_in_channel,
            },
            "analysis_mode": "smc_mtf",
            "llm_used": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ----- Quant Analysis (SMC + Indicators, Single LLM Call) -----

class QuantAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/quant")
async def run_quant_analysis(request: QuantAnalysisRequest):
    """
    Run quantitative SMC analysis using a single LLM call.

    This is a focused quant approach that:
    - Uses only chart data (SMC levels, RSI, MACD, Bollinger Bands, etc.)
    - No news, no sentiment, no fundamentals
    - Single systematic LLM prompt for trade decision
    - Faster than full multi-agent pipeline

    Returns structured trade decision compatible with trade execution modal.
    """
    import time as _time
    _endpoint_start = _time.time()
    logger.info(f"[QUANT API] Request: symbol={request.symbol}, timeframe={request.timeframe}")
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2
        bid = symbol_info.bid
        ask = symbol_info.ask

        # Map timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get price data
        rates = mt5.copy_rates_from_pos(request.symbol, tf, 0, 200)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Technical Indicators ===

        # ATR
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        macd_value = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_histogram = macd_hist.iloc[-1]

        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        bb_upper_val = bb_upper.iloc[-1]
        bb_lower_val = bb_lower.iloc[-1]
        bb_middle_val = sma20.iloc[-1]

        # EMA
        ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

        # ADX for trend strength
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = high_low.copy()
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean().iloc[-1]

        # Market regime
        avg_atr = high_low.rolling(50).mean().iloc[-1]
        volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

        if adx > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                market_regime = "trending-up"
            else:
                market_regime = "trending-down"
        else:
            market_regime = "ranging"

        # === Volume Analysis ===
        volume_col = 'tick_volume' if 'tick_volume' in df.columns else 'real_volume'
        if volume_col in df.columns:
            current_volume = df[volume_col].iloc[-1]
            avg_volume_20 = df[volume_col].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

            # Volume trend (last 5 bars)
            recent_volumes = df[volume_col].iloc[-5:]
            volume_trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"

            # Volume spike detection (>1.5x average)
            volume_spike = current_volume > avg_volume_20 * 1.5

            # Volume profile description
            if volume_ratio > 2.0:
                volume_profile = "Very High (spike)"
            elif volume_ratio > 1.5:
                volume_profile = "High"
            elif volume_ratio > 0.8:
                volume_profile = "Normal"
            elif volume_ratio > 0.5:
                volume_profile = "Low"
            else:
                volume_profile = "Very Low"
        else:
            current_volume = 0
            avg_volume_20 = 0
            volume_ratio = 1.0
            volume_trend = "unknown"
            volume_spike = False
            volume_profile = "N/A"

        # === Key Price Levels (PDH/PDL, PWH/PWL, Previous H4) ===
        key_levels = {}
        level_timeframes = {
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
        }
        for tf_name, tf_val in level_timeframes.items():
            try:
                tf_rates = mt5.copy_rates_from_pos(request.symbol, tf_val, 0, 2)
                if tf_rates is not None and len(tf_rates) >= 2:
                    prev_bar = tf_rates[0]  # Previous completed candle
                    key_levels[tf_name] = {
                        "high": float(prev_bar['high']),
                        "low": float(prev_bar['low']),
                        "open": float(prev_bar['open']),
                        "close": float(prev_bar['close']),
                    }
            except Exception as e:
                logger.warning(f"[QUANT API] Failed to get {tf_name} key levels: {e}")

        # === STEP 2: Run SMC Analysis (Extended) ===
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer

        analyzer = SmartMoneyAnalyzer()
        smc_result = analyzer.analyze_full_smc(
            df,
            current_price=current_price,
            use_structural_obs=True,
            include_equal_levels=True,
            include_breakers=True,
            include_ote=True,
            include_sweeps=True,
            include_inducements=False,  # Can be noisy
            include_rejections=False,   # Can be noisy
            include_turtle_soup=False   # Can be noisy
        )

        # === STEP 2b: Calculate Volume Profile ===
        from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer

        vp_analyzer = VolumeProfileAnalyzer()
        vp_result = vp_analyzer.calculate_volume_profile(df, num_bins=50, lookback=100)
        volume_profile_context = vp_analyzer.format_for_prompt(vp_result, current_price)

        # Format SMC context for LLM
        smc_context = _format_smc_for_quant(smc_result, current_price, atr, timeframe=request.timeframe)

        # === STEP 3: Build Indicator Context ===
        indicators_context = f"""## Technical Indicators

### Momentum
- **RSI(14)**: {rsi_value:.1f} {"(Overbought)" if rsi_value > 70 else "(Oversold)" if rsi_value < 30 else "(Neutral)"}
- **MACD**: {macd_value:.5f} | Signal: {macd_signal:.5f} | Histogram: {macd_histogram:.5f}
  - {"Bullish (MACD > Signal)" if macd_value > macd_signal else "Bearish (MACD < Signal)"}

### Trend
- **EMA20**: {ema20:.5f} {"(Price above)" if current_price > ema20 else "(Price below)"}
- **EMA50**: {ema50:.5f} {"(Price above)" if current_price > ema50 else "(Price below)"}
- **ADX**: {adx:.1f} {"(Strong trend)" if adx > 25 else "(Weak/No trend)"}
- **Regime**: {market_regime}

### Volatility
- **ATR(14)**: {atr:.5f}
- **Volatility**: {volatility_regime}
- **Bollinger Bands**: Upper={bb_upper_val:.5f} | Middle={bb_middle_val:.5f} | Lower={bb_lower_val:.5f}
  - {"Price near upper band" if current_price > bb_upper_val * 0.99 else "Price near lower band" if current_price < bb_lower_val * 1.01 else "Price within bands"}

### Volume
- **Current Bar Volume**: {current_volume:,.0f} ticks
- **20-bar Avg Volume**: {avg_volume_20:,.0f} ticks
- **Volume Ratio**: {volume_ratio:.2f}x average {"⚠️ SPIKE" if volume_spike else ""}
- **Volume Profile**: {volume_profile}
- **Volume Trend (5 bars)**: {volume_trend}
"""

        # Append key price levels to indicators context
        if key_levels:
            level_lines = ["\n## Key Price Levels\n"]
            tf_labels = {
                "H4": ("Previous H4", "PH4H", "PH4L"),
                "D1": ("Previous Day", "PDH", "PDL"),
                "W1": ("Previous Week", "PWH", "PWL"),
            }
            for tf_name in ["D1", "W1", "H4"]:  # Most important first
                if tf_name not in key_levels:
                    continue
                levels = key_levels[tf_name]
                label, high_abbr, low_abbr = tf_labels[tf_name]
                candle_range = levels["high"] - levels["low"]
                dist_high = current_price - levels["high"]
                dist_low = current_price - levels["low"]
                level_lines.append(f"### {label}")
                level_lines.append(f"- **{high_abbr}** (High): {levels['high']:.5f} (price is {dist_high:+.5f} from it)")
                level_lines.append(f"- **{low_abbr}** (Low): {levels['low']:.5f} (price is {dist_low:+.5f} from it)")
                level_lines.append(f"- **Open**: {levels['open']:.5f} | **Close**: {levels['close']:.5f}")
                level_lines.append(f"- **Range**: {candle_range:.5f}")
                # Context: is price above/below/inside the previous range?
                if current_price > levels["high"]:
                    level_lines.append(f"- Price is **ABOVE** {label} range")
                elif current_price < levels["low"]:
                    level_lines.append(f"- Price is **BELOW** {label} range")
                else:
                    level_lines.append(f"- Price is **INSIDE** {label} range")
                level_lines.append("")
            indicators_context += "\n".join(level_lines)

        # Add Volume Profile analysis
        indicators_context += "\n" + volume_profile_context

        # === STEP 4: Call Quant Analyst LLM ===
        from tradingagents.agents.analysts.quant_analyst import create_quant_analyst
        from tradingagents.llm_factory import get_llm

        llm = get_llm(tier="deep")

        # Fetch trade memories for this symbol (lessons from past trades)
        from tradingagents.trade_decisions import get_trade_memories
        trade_memories = get_trade_memories(request.symbol)
        if trade_memories:
            logger.info(f"[QUANT API] Injecting trade memories for {request.symbol} ({len(trade_memories)} chars)")

        # Create mock state with all the data
        quant_state = {
            "company_of_interest": request.symbol,
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "smc_context": smc_context,
            "smc_analysis": smc_result,
            "market_report": indicators_context,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trading_session": _get_trading_session(),
            "trade_memories": trade_memories,
        }

        # Build the full prompt for logging/debugging
        from tradingagents.agents.analysts.quant_analyst import _build_data_context, _build_quant_prompt
        debug_data_context = _build_data_context(
            ticker=request.symbol,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_result,
            market_report=indicators_context,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=_get_trading_session(),
            current_date=quant_state["trade_date"],
        )
        full_prompt = _build_quant_prompt(debug_data_context, trade_memories=trade_memories)

        # Run quant analyst
        logger.info(f"[QUANT API] Running quant analyst for {request.symbol}...")
        quant_analyst = create_quant_analyst(llm, use_structured_output=True)
        result = quant_analyst(quant_state)

        quant_report = result.get("quant_report", "")
        quant_decision = result.get("quant_decision")
        logger.info(f"[QUANT API] Analyst returned: quant_decision={'present' if quant_decision else 'None'}")

        # === STEP 5: Build Response ===
        if quant_decision:
            # Map signal
            signal_map = {
                "buy_to_enter": "BUY",
                "sell_to_enter": "SELL",
                "hold": "HOLD",
                "close": "HOLD"
            }
            signal = quant_decision.get("signal", "hold")
            if isinstance(signal, dict):
                signal = signal.get("value", "hold")

            mapped_signal = signal_map.get(signal, "HOLD")

            decision = {
                "signal": mapped_signal,
                "confidence": quant_decision.get("confidence", 0.5),
                "entry_price": quant_decision.get("entry_price"),
                "stop_loss": quant_decision.get("stop_loss"),
                "take_profit": quant_decision.get("profit_target"),
                "rationale": f"{quant_decision.get('justification', '')}\n\n**Invalidation**: {quant_decision.get('invalidation_condition', 'N/A')}",
                "analysis_mode": "smc_quant_basic",
                "leverage": quant_decision.get("leverage"),
                "risk_usd": quant_decision.get("risk_usd"),
                "risk_level": quant_decision.get("risk_level"),
                "risk_reward_ratio": quant_decision.get("risk_reward_ratio"),
                "full_report": quant_report
            }
        else:
            # Fallback if structured output failed
            decision = {
                "signal": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "rationale": quant_report or "Quant analysis could not generate a structured decision.",
                "analysis_mode": "smc_quant_basic",
                "full_report": quant_report
            }

        # Include SMC levels for chart display (same as rule-based)
        smc_levels = _extract_smc_levels_for_chart(smc_result)

        _endpoint_duration = _time.time() - _endpoint_start
        logger.info(
            f"[QUANT API] Response: {request.symbol} signal={decision.get('signal')} "
            f"confidence={decision.get('confidence')} "
            f"entry={decision.get('entry_price')} sl={decision.get('stop_loss')} tp={decision.get('take_profit')} "
            f"[took {_endpoint_duration:.1f}s]"
        )

        return {
            "status": "completed",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": round(current_price, 5),
            "bid": round(bid, 5),
            "ask": round(ask, 5),
            "decision": decision,
            "smc_levels": smc_levels,
            "indicators": {
                "rsi": round(rsi_value, 2) if not np.isnan(rsi_value) else None,
                "macd": round(macd_value, 5) if not np.isnan(macd_value) else None,
                "macd_signal": round(macd_signal, 5) if not np.isnan(macd_signal) else None,
                "macd_histogram": round(macd_histogram, 5) if not np.isnan(macd_histogram) else None,
                "ema20": round(ema20, 5) if not np.isnan(ema20) else None,
                "ema50": round(ema50, 5) if not np.isnan(ema50) else None,
                "atr": round(atr, 5) if not np.isnan(atr) else None,
                "adx": round(adx, 2) if not np.isnan(adx) else None,
                "bb_upper": round(bb_upper_val, 5) if not np.isnan(bb_upper_val) else None,
                "bb_middle": round(bb_middle_val, 5) if not np.isnan(bb_middle_val) else None,
                "bb_lower": round(bb_lower_val, 5) if not np.isnan(bb_lower_val) else None,
            },
            "regime": {
                "market_regime": market_regime,
                "volatility": volatility_regime,
                "adx": round(adx, 2) if not np.isnan(adx) else None,
                "atr": round(atr, 5) if not np.isnan(atr) else None
            },
            "analysis_mode": "smc_quant_basic",
            "llm_used": True,
            "prompt_sent": full_prompt,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        _endpoint_duration = _time.time() - _endpoint_start
        logger.error(f"[QUANT API] ERROR for {request.symbol} after {_endpoint_duration:.1f}s: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


class VPQuantAnalysisRequest(BaseModel):
    """Request model for Volume Profile quant analysis."""
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/vp-quant")
async def run_vp_quant_analysis(request: VPQuantAnalysisRequest):
    """
    Run quantitative Volume Profile analysis using a single LLM call.

    This quant focuses on Volume Profile concepts:
    - POC (Point of Control) as price magnet
    - Value Area for fair value determination
    - HVN (High Volume Nodes) for support/resistance
    - LVN (Low Volume Nodes) for fast-move zones

    Returns structured trade decision compatible with trade execution modal.
    """
    import time as _time
    _endpoint_start = _time.time()
    logger.info(f"[VP QUANT API] Request: symbol={request.symbol}, timeframe={request.timeframe}")
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2
        bid = symbol_info.bid
        ask = symbol_info.ask

        # Map timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get price data (more for VP analysis)
        rates = mt5.copy_rates_from_pos(request.symbol, tf, 0, 300)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Volume Profile ===
        from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer
        from tradingagents.agents.analysts.volume_profile_quant import (
            create_volume_profile_quant,
            analyze_volume_profile_for_quant,
        )

        vp_data = analyze_volume_profile_for_quant(df, current_price, num_bins=50, lookback=150)
        volume_profile = vp_data["volume_profile"]
        volume_profile_context = vp_data["volume_profile_context"]

        # === STEP 2: Calculate Technical Indicators ===
        # ATR
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_value = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_histogram = (macd_line - signal_line).iloc[-1]

        # EMA
        ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = high_low.copy()
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean().iloc[-1]

        # Market regime
        avg_atr = high_low.rolling(50).mean().iloc[-1]
        if not np.isnan(atr) and not np.isnan(avg_atr) and avg_atr > 0:
            vol_ratio = atr / avg_atr
            if vol_ratio > 1.5:
                volatility_regime = "high"
            elif vol_ratio < 0.7:
                volatility_regime = "low"
            else:
                volatility_regime = "normal"
        else:
            volatility_regime = "normal"

        if adx > 25:
            if ema20 > ema50:
                market_regime = "trending-up"
            else:
                market_regime = "trending-down"
        else:
            market_regime = "ranging"

        # Volume stats
        current_volume = df['tick_volume'].iloc[-1] if 'tick_volume' in df.columns else df.get('volume', pd.Series([0])).iloc[-1]
        avg_volume_20 = df['tick_volume'].rolling(20).mean().iloc[-1] if 'tick_volume' in df.columns else 0
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        # Build indicator context
        indicators_context = f"""### Momentum
- **RSI(14)**: {rsi_value:.1f} {"(Overbought)" if rsi_value > 70 else "(Oversold)" if rsi_value < 30 else "(Neutral)"}
- **MACD**: {macd_value:.5f} | Signal: {macd_signal:.5f}
  - {"Bullish (MACD > Signal)" if macd_value > macd_signal else "Bearish (MACD < Signal)"}

### Trend
- **EMA20**: {ema20:.5f} {"(Price above)" if current_price > ema20 else "(Price below)"}
- **EMA50**: {ema50:.5f} {"(Price above)" if current_price > ema50 else "(Price below)"}
- **ADX**: {adx:.1f} {"(Strong trend)" if adx > 25 else "(Weak/No trend)"}
- **Regime**: {market_regime}

### Volatility
- **ATR(14)**: {atr:.5f}
- **Volatility**: {volatility_regime}

### Volume
- **Current Volume**: {current_volume:,.0f}
- **Avg Volume (20)**: {avg_volume_20:,.0f}
- **Volume Ratio**: {volume_ratio:.2f}x average
"""

        # === STEP 3: Call VP Quant Analyst LLM ===
        from tradingagents.llm_factory import get_llm

        llm = get_llm(tier="deep")

        # Fetch trade memories for this symbol
        from tradingagents.trade_decisions import get_trade_memories as get_vp_trade_memories
        vp_trade_memories = get_vp_trade_memories(request.symbol)
        if vp_trade_memories:
            logger.info(f"[VP QUANT API] Injecting trade memories for {request.symbol} ({len(vp_trade_memories)} chars)")

        # Create state for VP quant
        vp_quant_state = {
            "company_of_interest": request.symbol,
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "volume_profile": volume_profile,
            "volume_profile_context": volume_profile_context,
            "market_report": indicators_context,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trading_session": _get_trading_session(),
            "trade_memories": vp_trade_memories,
        }

        # Build the full prompt for logging/debugging
        from tradingagents.agents.analysts.volume_profile_quant import _build_vp_data_context, _build_vp_quant_prompt
        debug_data_context = _build_vp_data_context(
            ticker=request.symbol,
            current_price=current_price,
            volume_profile=volume_profile,
            volume_profile_context=volume_profile_context,
            market_report=indicators_context,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=_get_trading_session(),
            current_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        full_vp_prompt = _build_vp_quant_prompt(debug_data_context, trade_memories=vp_trade_memories)

        # Run VP quant analyst
        logger.info(f"[VP QUANT API] Running VP quant analyst for {request.symbol}...")
        _llm_start = _time.time()
        vp_quant_analyst = create_volume_profile_quant(llm, use_structured_output=True)
        result = vp_quant_analyst(vp_quant_state)
        _llm_duration = _time.time() - _llm_start

        vp_quant_report = result.get("vp_quant_report", "")
        vp_quant_decision = result.get("vp_quant_decision")
        logger.info(f"[VP QUANT API] Analyst returned: vp_quant_decision={'present' if vp_quant_decision else 'None'} [LLM took {_llm_duration:.1f}s]")

        # Build response
        _endpoint_duration = _time.time() - _endpoint_start
        logger.info(f"[VP QUANT API] Completed for {request.symbol} in {_endpoint_duration:.1f}s")

        if vp_quant_decision:
            signal_map = {
                "buy_to_enter": "BUY",
                "sell_to_enter": "SELL",
                "hold": "HOLD",
                "close": "HOLD",
            }
            raw_signal = vp_quant_decision.get("signal", "hold")
            if isinstance(raw_signal, dict):
                raw_signal = raw_signal.get("value", "hold")

            return {
                "status": "success",
                "symbol": request.symbol,
                "signal": signal_map.get(raw_signal, "HOLD"),
                "raw_signal": raw_signal,
                "entry_price": vp_quant_decision.get("entry_price"),
                "stop_loss": vp_quant_decision.get("stop_loss"),
                "take_profit": vp_quant_decision.get("profit_target"),
                "confidence": vp_quant_decision.get("confidence", 0.5),
                "justification": vp_quant_decision.get("justification", ""),
                "invalidation": vp_quant_decision.get("invalidation_condition", ""),
                "risk_level": vp_quant_decision.get("risk_level", "Medium"),
                "risk_reward_ratio": vp_quant_decision.get("risk_reward_ratio"),
                "report": vp_quant_report,
                "current_price": current_price,
                "bid": bid,
                "ask": ask,
                "volume_profile": {
                    "poc": volume_profile.poc,
                    "poc_volume_pct": volume_profile.poc_volume_pct,
                    "value_area_high": volume_profile.value_area_high,
                    "value_area_low": volume_profile.value_area_low,
                    "hvn_count": len(volume_profile.high_volume_nodes),
                    "lvn_count": len(volume_profile.low_volume_nodes),
                },
                "market_context": {
                    "regime": market_regime,
                    "volatility": volatility_regime,
                    "adx": round(adx, 2) if not np.isnan(adx) else None,
                    "atr": round(atr, 5) if not np.isnan(atr) else None
                },
                "analysis_mode": "vp_quant",
                "llm_used": True,
                "llm_duration_seconds": round(_llm_duration, 2),
                "endpoint_duration_seconds": round(_endpoint_duration, 2),
                "prompt_sent": full_vp_prompt,
            }
        else:
            return {
                "status": "success",
                "symbol": request.symbol,
                "signal": "HOLD",
                "raw_signal": "hold",
                "confidence": 0.5,
                "justification": "VP quant analysis did not return a structured decision",
                "report": vp_quant_report,
                "current_price": current_price,
                "analysis_mode": "vp_quant",
                "llm_used": True,
                "llm_duration_seconds": round(_llm_duration, 2),
                "endpoint_duration_seconds": round(_endpoint_duration, 2),
                "prompt_sent": full_vp_prompt,
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        _endpoint_duration = _time.time() - _endpoint_start
        logger.error(f"[VP QUANT API] ERROR for {request.symbol} after {_endpoint_duration:.1f}s: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ----- SMC Quant Analysis (Dedicated SMC-focused, Single LLM Call) -----

class SmcQuantAnalysisRequest(BaseModel):
    """Request model for dedicated SMC quant analysis."""
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/smc-quant")
async def run_smc_quant_analysis(request: SmcQuantAnalysisRequest):
    """
    Run dedicated Smart Money Concepts quant analysis using a single LLM call.

    This is a deep SMC-focused quant that specializes in:
    - Order Blocks as institutional entry zones
    - Fair Value Gaps for price imbalance entries
    - BOS/CHoCH for trend confirmation
    - Liquidity pools as take profit targets
    - Premium/Discount zones for entry timing

    Uses the dedicated smc_quant agent with specialized SMC prompt.
    Returns structured trade decision compatible with trade execution modal.
    """
    import time as _time
    _endpoint_start = _time.time()
    logger.info(f"[SMC QUANT API] Request: symbol={request.symbol}, timeframe={request.timeframe}")
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2
        bid = symbol_info.bid
        ask = symbol_info.ask

        # Map timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get price data
        rates = mt5.copy_rates_from_pos(request.symbol, tf, 0, 200)
        if rates is None or len(rates) == 0:
            raise HTTPException(status_code=400, detail="Failed to get price data")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Technical Indicators ===

        # ATR
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        macd_value = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_histogram = macd_hist.iloc[-1]

        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)
        bb_upper_val = bb_upper.iloc[-1]
        bb_lower_val = bb_lower.iloc[-1]
        bb_middle_val = sma20.iloc[-1]

        # EMA
        ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

        # ADX for trend strength
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = high_low.copy()
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean().iloc[-1]

        # Market regime
        avg_atr = high_low.rolling(50).mean().iloc[-1]
        volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

        if adx > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                market_regime = "trending-up"
            else:
                market_regime = "trending-down"
        else:
            market_regime = "ranging"

        # === Volume Analysis ===
        volume_col = 'tick_volume' if 'tick_volume' in df.columns else 'real_volume'
        if volume_col in df.columns:
            current_volume = df[volume_col].iloc[-1]
            avg_volume_20 = df[volume_col].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            recent_volumes = df[volume_col].iloc[-5:]
            volume_trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"
            volume_spike = current_volume > avg_volume_20 * 1.5
            if volume_ratio > 2.0:
                volume_profile = "Very High (spike)"
            elif volume_ratio > 1.5:
                volume_profile = "High"
            elif volume_ratio > 0.8:
                volume_profile = "Normal"
            elif volume_ratio > 0.5:
                volume_profile = "Low"
            else:
                volume_profile = "Very Low"
        else:
            current_volume = 0
            avg_volume_20 = 0
            volume_ratio = 1.0
            volume_trend = "unknown"
            volume_spike = False
            volume_profile = "N/A"

        # === STEP 2: Run dedicated SMC Analysis ===
        from tradingagents.agents.analysts.smc_quant import (
            create_smc_quant,
            analyze_smc_for_quant,
            _build_smc_data_context,
            _build_smc_quant_prompt,
        )

        smc_data = analyze_smc_for_quant(df, current_price)
        smc_analysis = smc_data["smc_analysis"]
        smc_context = smc_data["smc_context"]

        # === STEP 3: Build Indicator Context ===
        indicators_context = f"""## Technical Indicators

### Momentum
- **RSI(14)**: {rsi_value:.1f} {"(Overbought)" if rsi_value > 70 else "(Oversold)" if rsi_value < 30 else "(Neutral)"}
- **MACD**: {macd_value:.5f} | Signal: {macd_signal:.5f} | Histogram: {macd_histogram:.5f}
  - {"Bullish (MACD > Signal)" if macd_value > macd_signal else "Bearish (MACD < Signal)"}

### Trend
- **EMA20**: {ema20:.5f} {"(Price above)" if current_price > ema20 else "(Price below)"}
- **EMA50**: {ema50:.5f} {"(Price above)" if current_price > ema50 else "(Price below)"}
- **ADX**: {adx:.1f} {"(Strong trend)" if adx > 25 else "(Weak/No trend)"}
- **Regime**: {market_regime}

### Volatility
- **ATR(14)**: {atr:.5f}
- **Volatility**: {volatility_regime}
- **Bollinger Bands**: Upper={bb_upper_val:.5f} | Middle={bb_middle_val:.5f} | Lower={bb_lower_val:.5f}
  - {"Price near upper band" if current_price > bb_upper_val * 0.99 else "Price near lower band" if current_price < bb_lower_val * 1.01 else "Price within bands"}

### Volume
- **Current Bar Volume**: {current_volume:,.0f} ticks
- **20-bar Avg Volume**: {avg_volume_20:,.0f} ticks
- **Volume Ratio**: {volume_ratio:.2f}x average {"⚠️ SPIKE" if volume_spike else ""}
- **Volume Profile**: {volume_profile}
- **Volume Trend (5 bars)**: {volume_trend}
"""

        # === STEP 4: Call SMC Quant Analyst LLM ===
        from tradingagents.llm_factory import get_llm

        llm = get_llm(tier="deep")

        # Fetch trade memories for this symbol
        from tradingagents.trade_decisions import get_trade_memories
        trade_memories = get_trade_memories(request.symbol)
        if trade_memories:
            logger.info(f"[SMC QUANT API] Injecting trade memories for {request.symbol} ({len(trade_memories)} chars)")

        # Create state with all the data
        smc_quant_state = {
            "company_of_interest": request.symbol,
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "smc_context": smc_context,
            "smc_analysis": smc_analysis,
            "market_report": indicators_context,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trading_session": _get_trading_session(),
            "trade_memories": trade_memories,
        }

        # Build full prompt for logging/debugging
        debug_data_context = _build_smc_data_context(
            ticker=request.symbol,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=indicators_context,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=_get_trading_session(),
            current_date=smc_quant_state["trade_date"],
        )
        full_prompt = _build_smc_quant_prompt(debug_data_context, trade_memories=trade_memories)

        # Run SMC quant analyst
        import time as _time
        logger.info(f"[SMC QUANT API] Running SMC quant analyst for {request.symbol}...")
        _llm_start = _time.time()
        smc_quant_analyst = create_smc_quant(llm, use_structured_output=True)
        result = smc_quant_analyst(smc_quant_state)
        _llm_duration = _time.time() - _llm_start

        smc_quant_report = result.get("smc_quant_report", "")
        smc_quant_decision = result.get("smc_quant_decision")
        logger.info(f"[SMC QUANT API] Analyst returned: smc_quant_decision={'present' if smc_quant_decision else 'None'} [LLM took {_llm_duration:.1f}s]")

        # === STEP 5: Build Response ===
        if smc_quant_decision:
            signal_map = {
                "buy_to_enter": "BUY",
                "sell_to_enter": "SELL",
                "hold": "HOLD",
                "close": "HOLD"
            }
            signal = smc_quant_decision.get("signal", "hold")
            if isinstance(signal, dict):
                signal = signal.get("value", "hold")

            mapped_signal = signal_map.get(signal, "HOLD")

            decision = {
                "signal": mapped_signal,
                "confidence": smc_quant_decision.get("confidence", 0.5),
                "entry_price": smc_quant_decision.get("entry_price"),
                "stop_loss": smc_quant_decision.get("stop_loss"),
                "take_profit": smc_quant_decision.get("profit_target"),
                "rationale": f"{smc_quant_decision.get('justification', '')}\n\n**Invalidation**: {smc_quant_decision.get('invalidation_condition', 'N/A')}",
                "analysis_mode": "smc_quant",
                "leverage": smc_quant_decision.get("leverage"),
                "risk_usd": smc_quant_decision.get("risk_usd"),
                "risk_level": smc_quant_decision.get("risk_level"),
                "risk_reward_ratio": smc_quant_decision.get("risk_reward_ratio"),
                "trailing_stop_atr_multiplier": smc_quant_decision.get("trailing_stop_atr_multiplier"),
                "full_report": smc_quant_report
            }
        else:
            decision = {
                "signal": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "rationale": smc_quant_report or "SMC quant analysis could not generate a structured decision.",
                "analysis_mode": "smc_quant",
                "full_report": smc_quant_report
            }

        # === HTF Bias Check ===
        htf_info = _calculate_htf_bias(request.symbol, request.timeframe)
        logger.info(f"[SMC QUANT API] HTF bias for {request.symbol}: {htf_info['htf_bias']} ({htf_info['htf_timeframe']})")

        # Extract SMC levels for chart display
        from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
        # Use the extended analysis for chart levels
        analyzer = SmartMoneyAnalyzer()
        smc_extended = analyzer.analyze_full_smc(
            df,
            current_price=current_price,
            use_structural_obs=True,
            include_equal_levels=True,
            include_breakers=True,
            include_ote=True,
            include_sweeps=True,
        )
        smc_levels = _extract_smc_levels_for_chart(smc_extended)

        _endpoint_duration = _time.time() - _endpoint_start
        logger.info(
            f"[SMC QUANT API] Response: {request.symbol} signal={decision.get('signal')} "
            f"confidence={decision.get('confidence')} "
            f"entry={decision.get('entry_price')} sl={decision.get('stop_loss')} tp={decision.get('take_profit')} "
            f"[took {_endpoint_duration:.1f}s]"
        )

        return {
            "status": "completed",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": round(current_price, 5),
            "bid": round(bid, 5),
            "ask": round(ask, 5),
            "decision": decision,
            "smc_levels": smc_levels,
            "indicators": {
                "rsi": round(rsi_value, 2) if not np.isnan(rsi_value) else None,
                "macd": round(macd_value, 5) if not np.isnan(macd_value) else None,
                "macd_signal": round(macd_signal, 5) if not np.isnan(macd_signal) else None,
                "macd_histogram": round(macd_histogram, 5) if not np.isnan(macd_histogram) else None,
                "ema20": round(ema20, 5) if not np.isnan(ema20) else None,
                "ema50": round(ema50, 5) if not np.isnan(ema50) else None,
                "atr": round(atr, 5) if not np.isnan(atr) else None,
                "adx": round(adx, 2) if not np.isnan(adx) else None,
                "bb_upper": round(bb_upper_val, 5) if not np.isnan(bb_upper_val) else None,
                "bb_middle": round(bb_middle_val, 5) if not np.isnan(bb_middle_val) else None,
                "bb_lower": round(bb_lower_val, 5) if not np.isnan(bb_lower_val) else None,
            },
            "regime": {
                "market_regime": market_regime,
                "volatility": volatility_regime,
                "adx": round(adx, 2) if not np.isnan(adx) else None,
                "atr": round(atr, 5) if not np.isnan(atr) else None
            },
            "htf_bias": htf_info,
            "analysis_mode": "smc_quant",
            "llm_used": True,
            "llm_duration_seconds": round(_llm_duration, 2),
            "endpoint_duration_seconds": round(_endpoint_duration, 2),
            "prompt_sent": full_prompt,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        _endpoint_duration = _time.time() - _endpoint_start
        logger.error(f"[SMC QUANT API] ERROR for {request.symbol} after {_endpoint_duration:.1f}s: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _format_smc_for_quant(smc_result: dict, current_price: float, atr: float, timeframe: str = "H1") -> str:
    """
    Format SMC analysis for quant analyst prompt.

    Improvements:
    - Shows BOTH bullish AND bearish levels regardless of bias
    - Includes BOS/CHoCH with direction context
    - Shows mitigation status (fill percentage for FVGs)
    - Includes breaker blocks, premium/discount zones
    - Adds liquidity sweep alerts
    - Provides actionable context for each level
    """
    lines = ["## Smart Money Concepts Analysis\n"]

    # Helper to safely get attribute
    def safe_attr(obj, attr, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    # === MARKET STRUCTURE & BIAS ===
    bias = smc_result.get("bias", "neutral")
    structure = smc_result.get("structure", {})  # Fixed: was "market_structure"
    recent_bos = structure.get("recent_bos", [])
    recent_choc = structure.get("recent_choc", [])

    # Build structure context
    structure_context = []
    if recent_choc:
        last_choc = recent_choc[-1]
        choc_type = safe_attr(last_choc, "type", "")
        choc_price = safe_attr(last_choc, "price", "")
        choc_ts = safe_attr(last_choc, "timestamp", "")
        if choc_type == "high":
            label = f"CHoCH (bearish reversal signal) on {timeframe}"
        else:
            label = f"CHoCH (bullish reversal signal) on {timeframe}"
        if choc_ts:
            label += f" at {choc_ts}"
        if choc_price:
            label += f" @ {choc_price}"
        structure_context.append(label)
    if recent_bos:
        last_bos = recent_bos[-1]
        bos_type = safe_attr(last_bos, "type", "")
        bos_price = safe_attr(last_bos, "price", "")
        bos_ts = safe_attr(last_bos, "timestamp", "")
        if bos_type == "high":
            label = f"BOS (bullish continuation) on {timeframe}"
        else:
            label = f"BOS (bearish continuation) on {timeframe}"
        if bos_ts:
            label += f" at {bos_ts}"
        if bos_price:
            label += f" @ {bos_price}"
        structure_context.append(label)

    lines.append(f"### Market Structure")
    lines.append(f"**Timeframe**: {timeframe}")
    lines.append(f"**Bias**: {bias.upper()}")
    if structure_context:
        lines.append(f"**Recent Structure**: {', '.join(structure_context)}")
    else:
        lines.append(f"**Recent Structure**: No recent BOS/CHoCH on {timeframe}")
    lines.append("")

    # === PREMIUM/DISCOUNT ZONE ===
    premium_discount = smc_result.get("premium_discount")
    if premium_discount:
        # Handle both dict and dataclass (PremiumDiscountZone)
        zone = safe_attr(premium_discount, "zone", "equilibrium")
        position_pct = safe_attr(premium_discount, "position_pct", 50)
        equilibrium = safe_attr(premium_discount, "equilibrium", 0)

        zone_advice = ""
        if zone == "premium":
            zone_advice = "(favor SELLS - price is expensive)"
        elif zone == "discount":
            zone_advice = "(favor BUYS - price is cheap)"
        else:
            zone_advice = "(neutral - wait for extremes)"

        lines.append(f"### Price Position")
        lines.append(f"**Zone**: {zone.upper()} ({position_pct:.0f}% of range) {zone_advice}")
        if equilibrium and equilibrium > 0:
            lines.append(f"**Equilibrium**: {equilibrium:.5f}")
        lines.append("")

    # === ORDER BLOCKS (Both bullish AND bearish) ===
    obs = smc_result.get("order_blocks", {})
    bullish_obs = obs.get("bullish") or []
    bearish_obs = obs.get("bearish") or []

    # Filter to unmitigated only
    bullish_obs_active = [ob for ob in bullish_obs if ob and not safe_attr(ob, "mitigated", False)]
    bearish_obs_active = [ob for ob in bearish_obs if ob and not safe_attr(ob, "mitigated", False)]

    if bullish_obs_active or bearish_obs_active:
        lines.append("### Order Blocks (Unmitigated)")

        # Sort by proximity to current price
        def ob_distance(ob):
            mid = (safe_attr(ob, "top", 0) + safe_attr(ob, "bottom", 0)) / 2
            return abs(mid - current_price)

        # Show bullish OBs (support zones - below price for buys)
        for ob in sorted(bullish_obs_active, key=ob_distance)[:3]:
            top = safe_attr(ob, "top", 0)
            bottom = safe_attr(ob, "bottom", 0)
            strength = safe_attr(ob, "strength", 0.5)
            method = safe_attr(ob, "detection_method", "candle")
            dist_pct = ((current_price - top) / current_price * 100) if current_price > top else 0

            method_tag = "[STRUCTURAL]" if method == "structural" else ""
            if dist_pct > 0:
                lines.append(f"- BULLISH OB {method_tag}: {bottom:.5f} - {top:.5f} (strength: {strength:.0%}) | -{dist_pct:.2f}% from price")
            else:
                lines.append(f"- BULLISH OB {method_tag}: {bottom:.5f} - {top:.5f} (strength: {strength:.0%}) | AT PRICE")

        # Show bearish OBs (resistance zones - above price for sells)
        for ob in sorted(bearish_obs_active, key=ob_distance)[:3]:
            top = safe_attr(ob, "top", 0)
            bottom = safe_attr(ob, "bottom", 0)
            strength = safe_attr(ob, "strength", 0.5)
            method = safe_attr(ob, "detection_method", "candle")
            dist_pct = ((bottom - current_price) / current_price * 100) if current_price < bottom else 0

            method_tag = "[STRUCTURAL]" if method == "structural" else ""
            if dist_pct > 0:
                lines.append(f"- BEARISH OB {method_tag}: {bottom:.5f} - {top:.5f} (strength: {strength:.0%}) | +{dist_pct:.2f}% from price")
            else:
                lines.append(f"- BEARISH OB {method_tag}: {bottom:.5f} - {top:.5f} (strength: {strength:.0%}) | AT PRICE")

        if not bullish_obs_active:
            lines.append("- No unmitigated BULLISH OBs (no demand zones)")
        if not bearish_obs_active:
            lines.append("- No unmitigated BEARISH OBs (no supply zones)")
        lines.append("")

    # === FAIR VALUE GAPS (with fill percentage) ===
    fvgs = smc_result.get("fair_value_gaps", {})
    bullish_fvgs = fvgs.get("bullish") or []
    bearish_fvgs = fvgs.get("bearish") or []

    # Include partially filled FVGs (not fully mitigated)
    bullish_fvgs_active = [fvg for fvg in bullish_fvgs if fvg and not safe_attr(fvg, "mitigated", False)]
    bearish_fvgs_active = [fvg for fvg in bearish_fvgs if fvg and not safe_attr(fvg, "mitigated", False)]

    if bullish_fvgs_active or bearish_fvgs_active:
        lines.append("### Fair Value Gaps")

        for fvg in bullish_fvgs_active[:3]:
            top = safe_attr(fvg, "top", 0)
            bottom = safe_attr(fvg, "bottom", 0)
            fill_pct = safe_attr(fvg, "fill_percentage", 0)
            dist_pct = ((current_price - top) / current_price * 100) if current_price > top else 0

            fill_status = f"({fill_pct:.0f}% filled)" if fill_pct > 0 else "(unfilled)"
            if dist_pct > 0:
                lines.append(f"- BULLISH FVG: {bottom:.5f} - {top:.5f} {fill_status} | -{dist_pct:.2f}% from price")
            else:
                lines.append(f"- BULLISH FVG: {bottom:.5f} - {top:.5f} {fill_status} | AT PRICE")

        for fvg in bearish_fvgs_active[:3]:
            top = safe_attr(fvg, "top", 0)
            bottom = safe_attr(fvg, "bottom", 0)
            fill_pct = safe_attr(fvg, "fill_percentage", 0)
            dist_pct = ((bottom - current_price) / current_price * 100) if current_price < bottom else 0

            fill_status = f"({fill_pct:.0f}% filled)" if fill_pct > 0 else "(unfilled)"
            if dist_pct > 0:
                lines.append(f"- BEARISH FVG: {bottom:.5f} - {top:.5f} {fill_status} | +{dist_pct:.2f}% from price")
            else:
                lines.append(f"- BEARISH FVG: {bottom:.5f} - {top:.5f} {fill_status} | AT PRICE")

        lines.append("")

    # === BREAKER BLOCKS ===
    breakers = smc_result.get("breaker_blocks", {})
    bullish_breakers = breakers.get("bullish") or []
    bearish_breakers = breakers.get("bearish") or []

    # Filter to unmitigated
    active_breakers = []
    for bb in bullish_breakers:
        if bb and not safe_attr(bb, "mitigated", False):
            active_breakers.append(("BULLISH", bb))
    for bb in bearish_breakers:
        if bb and not safe_attr(bb, "mitigated", False):
            active_breakers.append(("BEARISH", bb))

    if active_breakers:
        lines.append("### Breaker Blocks (Failed OBs - Strong Reversal Zones)")
        for bb_type, bb in active_breakers[:3]:
            top = safe_attr(bb, "top", 0)
            bottom = safe_attr(bb, "bottom", 0)
            original = safe_attr(bb, "original_type", "unknown")
            lines.append(f"- {bb_type} BREAKER: {bottom:.5f} - {top:.5f} (was {original} OB)")
        lines.append("")

    # === LIQUIDITY ZONES ===
    liq_zones = smc_result.get("liquidity_zones", [])
    if liq_zones and isinstance(liq_zones, list):
        lines.append("### Liquidity Zones (Stop Loss Clusters)")

        buy_side = [z for z in liq_zones if safe_attr(z, "type", "") == "buy-side"]
        sell_side = [z for z in liq_zones if safe_attr(z, "type", "") == "sell-side"]

        for zone in buy_side[:2]:
            price = safe_attr(zone, "price", 0)
            strength = safe_attr(zone, "strength", 50)
            dist_pct = ((price - current_price) / current_price * 100) if price > current_price else 0
            lines.append(f"- BUY-SIDE (above): {price:.5f} (strength: {strength:.1f}) | +{dist_pct:.2f}% - target for shorts")

        for zone in sell_side[:2]:
            price = safe_attr(zone, "price", 0)
            strength = safe_attr(zone, "strength", 50)
            dist_pct = ((current_price - price) / current_price * 100) if price < current_price else 0
            lines.append(f"- SELL-SIDE (below): {price:.5f} (strength: {strength:.1f}) | -{dist_pct:.2f}% - target for longs")

        lines.append("")

    # === LIQUIDITY SWEEPS (Recent) ===
    sweeps = smc_result.get("liquidity_sweeps", {})
    recent_sweeps = sweeps.get("recent", [])
    if recent_sweeps:
        lines.append("### Recent Liquidity Sweeps (Last 10 candles)")
        for sweep in recent_sweeps[:2]:
            sweep_type = safe_attr(sweep, "type", "unknown")
            level_price = safe_attr(sweep, "level_price", 0)
            is_strong = safe_attr(sweep, "is_strong", False)
            strong_tag = "[STRONG]" if is_strong else ""

            if sweep_type == "bullish":
                lines.append(f"- BULLISH SWEEP {strong_tag}: Swept lows at {level_price:.5f} - look for reversal UP")
            else:
                lines.append(f"- BEARISH SWEEP {strong_tag}: Swept highs at {level_price:.5f} - look for reversal DOWN")
        lines.append("")

    # === OTE ZONES ===
    ote_zones = smc_result.get("ote_zones", {})
    bullish_ote = ote_zones.get("bullish", []) if isinstance(ote_zones, dict) else []
    bearish_ote = ote_zones.get("bearish", []) if isinstance(ote_zones, dict) else []

    if bullish_ote or bearish_ote:
        lines.append("### OTE Zones (Optimal Trade Entry - Fib 61.8%-79%)")
        for ote in bullish_ote[:1]:
            fib_618 = safe_attr(ote, "fib_618", 0)
            fib_79 = safe_attr(ote, "fib_79", 0)
            if fib_618 and fib_79:
                lines.append(f"- BULLISH OTE: {fib_79:.5f} - {fib_618:.5f} (buy zone)")
        for ote in bearish_ote[:1]:
            fib_618 = safe_attr(ote, "fib_618", 0)
            fib_79 = safe_attr(ote, "fib_79", 0)
            if fib_618 and fib_79:
                lines.append(f"- BEARISH OTE: {fib_618:.5f} - {fib_79:.5f} (sell zone)")
        lines.append("")

    # === SUMMARY LINE ===
    lines.append(f"**Current Price**: {current_price:.5f}")
    lines.append(f"**ATR**: {atr:.5f}")

    return "\n".join(lines)


def _extract_smc_levels_for_chart(smc_result: dict) -> list:
    """Extract SMC levels for chart display."""
    smc_levels = []

    def safe_attr(obj, attr, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    # Order blocks
    obs = smc_result.get("order_blocks", {})
    for ob in (obs.get("bullish") or []):
        if ob and not safe_attr(ob, "mitigated", False):
            smc_levels.append({
                "type": "order_block",
                "price": (safe_attr(ob, "top", 0) + safe_attr(ob, "bottom", 0)) / 2,
                "direction": "bullish",
                "strength": safe_attr(ob, "strength", 0.5)
            })
    for ob in (obs.get("bearish") or []):
        if ob and not safe_attr(ob, "mitigated", False):
            smc_levels.append({
                "type": "order_block",
                "price": (safe_attr(ob, "top", 0) + safe_attr(ob, "bottom", 0)) / 2,
                "direction": "bearish",
                "strength": safe_attr(ob, "strength", 0.5)
            })

    # FVGs
    fvgs = smc_result.get("fair_value_gaps", {})
    for fvg in (fvgs.get("bullish") or []):
        if fvg and not safe_attr(fvg, "mitigated", False):
            smc_levels.append({
                "type": "fvg",
                "price": (safe_attr(fvg, "top", 0) + safe_attr(fvg, "bottom", 0)) / 2,
                "direction": "bullish"
            })
    for fvg in (fvgs.get("bearish") or []):
        if fvg and not safe_attr(fvg, "mitigated", False):
            smc_levels.append({
                "type": "fvg",
                "price": (safe_attr(fvg, "top", 0) + safe_attr(fvg, "bottom", 0)) / 2,
                "direction": "bearish"
            })

    # Liquidity
    liq_zones = smc_result.get("liquidity_zones", [])
    if isinstance(liq_zones, list):
        for zone in liq_zones:
            if zone:
                zone_type = safe_attr(zone, "type", "")
                direction = "bullish" if zone_type == "buy-side" else "bearish"
                smc_levels.append({
                    "type": "liquidity",
                    "price": safe_attr(zone, "price", 0),
                    "direction": direction
                })

    return smc_levels


def _calculate_htf_bias(symbol: str, analysis_timeframe: str) -> dict:
    """
    Calculate higher-timeframe bias by loading the next TF up and checking
    swing structure (higher highs/lows vs lower highs/lows) + EMA alignment.

    Returns dict with:
        htf_bias: "bullish" | "bearish" | "neutral"
        htf_timeframe: e.g. "H4"
        htf_details: human-readable summary
    """
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np

    # Map analysis TF → higher TF
    htf_map = {
        "M15": ("H1", mt5.TIMEFRAME_H1),
        "M30": ("H1", mt5.TIMEFRAME_H1),
        "H1": ("H4", mt5.TIMEFRAME_H4),
        "H4": ("D1", mt5.TIMEFRAME_D1),
        "D1": ("W1", mt5.TIMEFRAME_W1),
    }
    htf_name, htf_tf = htf_map.get(analysis_timeframe.upper(), ("H4", mt5.TIMEFRAME_H4))

    try:
        rates = mt5.copy_rates_from_pos(symbol, htf_tf, 0, 100)
        if rates is None or len(rates) < 20:
            return {"htf_bias": "neutral", "htf_timeframe": htf_name, "htf_details": "Insufficient HTF data"}

        df = pd.DataFrame(rates)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # --- Swing detection (simple pivot highs/lows, lookback=3) ---
        lookback = 3
        swing_highs = []
        swing_lows = []
        for i in range(lookback, len(high) - lookback):
            if all(high[i] >= high[i - j] for j in range(1, lookback + 1)) and \
               all(high[i] >= high[i + j] for j in range(1, lookback + 1)):
                swing_highs.append((i, high[i]))
            if all(low[i] <= low[i - j] for j in range(1, lookback + 1)) and \
               all(low[i] <= low[i + j] for j in range(1, lookback + 1)):
                swing_lows.append((i, low[i]))

        # Need at least 2 swing highs and 2 swing lows to determine structure
        details_parts = []
        structure_score = 0  # positive = bullish, negative = bearish

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check last 2 swing highs
            sh1, sh2 = swing_highs[-2][1], swing_highs[-1][1]
            # Check last 2 swing lows
            sl1, sl2 = swing_lows[-2][1], swing_lows[-1][1]

            if sh2 > sh1:
                structure_score += 1
                details_parts.append("Higher High")
            elif sh2 < sh1:
                structure_score -= 1
                details_parts.append("Lower High")

            if sl2 > sl1:
                structure_score += 1
                details_parts.append("Higher Low")
            elif sl2 < sl1:
                structure_score -= 1
                details_parts.append("Lower Low")

        # --- EMA alignment ---
        ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1]
        current = close[-1]

        if current > ema20 > ema50:
            structure_score += 1
            details_parts.append("Price > EMA20 > EMA50")
        elif current < ema20 < ema50:
            structure_score -= 1
            details_parts.append("Price < EMA20 < EMA50")
        else:
            details_parts.append("EMAs mixed")

        # --- ADX trend strength on HTF ---
        high_low = df['high'] - df['low']
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        atr_14 = high_low.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx_val = dx.rolling(14).mean().iloc[-1]

        if not np.isnan(adx_val) and adx_val > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                structure_score += 1
                details_parts.append(f"ADX {adx_val:.0f} bullish")
            else:
                structure_score -= 1
                details_parts.append(f"ADX {adx_val:.0f} bearish")
        else:
            adx_display = f"{adx_val:.0f}" if not np.isnan(adx_val) else "N/A"
            details_parts.append(f"ADX {adx_display} weak trend")

        # Determine bias
        if structure_score >= 2:
            bias = "bullish"
        elif structure_score <= -2:
            bias = "bearish"
        else:
            bias = "neutral"

        details = f"{htf_name} bias: {bias} (score {structure_score}). {', '.join(details_parts)}"
        return {"htf_bias": bias, "htf_timeframe": htf_name, "htf_details": details}

    except Exception as e:
        logger.warning(f"HTF bias calculation failed for {symbol}/{htf_name}: {e}")
        return {"htf_bias": "neutral", "htf_timeframe": htf_name, "htf_details": f"Error: {e}"}


def _get_trading_session() -> str:
    """Get current trading session based on UTC time."""
    from datetime import datetime
    hour = datetime.utcnow().hour

    if 0 <= hour < 7:
        return "asian"
    elif 7 <= hour < 12:
        return "london"
    elif 12 <= hour < 16:
        return "london_ny_overlap"
    elif 16 <= hour < 21:
        return "new_york"
    else:
        return "asian"


# ----- Breakout Quant Analysis (Consolidation Detection, Single LLM Call) -----

class BreakoutQuantAnalysisRequest(BaseModel):
    """Request model for breakout quant analysis."""
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/breakout-quant")
async def run_breakout_quant_analysis(request: BreakoutQuantAnalysisRequest):
    """
    Run dedicated Breakout Quant analysis using a single LLM call.

    This is a consolidation/breakout-focused quant that specializes in:
    - BB Squeeze detection (Bollinger Band width contraction)
    - Range boundary identification (consolidation high/low)
    - Structure bias analysis (higher lows = bullish, lower highs = bearish)
    - Breakout anticipation and confirmation entries

    Uses the dedicated breakout_quant agent with specialized consolidation prompt.
    Returns structured trade decision compatible with trade execution modal.
    """
    import time as _time
    _endpoint_start = _time.time()
    logger.info(f"[BREAKOUT QUANT API] Request: symbol={request.symbol}, timeframe={request.timeframe}")
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2
        bid = symbol_info.bid
        ask = symbol_info.ask

        # Map timeframe
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = timeframe_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get OHLC data
        bars = mt5.copy_rates_from_pos(request.symbol, tf, 0, 200)
        if bars is None or len(bars) < 50:
            raise HTTPException(status_code=400, detail=f"Not enough data for {request.symbol}")

        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Indicators ===
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # EMAs
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        smoothed_tr = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / smoothed_tr)
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / smoothed_tr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.ewm(span=14, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

        # Get latest values
        latest = df.iloc[-1]
        atr = float(latest['atr']) if pd.notna(latest['atr']) else 0
        adx = float(latest['adx']) if pd.notna(latest['adx']) else 0

        # === STEP 2: Determine Regime ===
        from tradingagents.indicators.regime import RegimeDetector
        detector = RegimeDetector()

        high_arr = df['high'].values
        low_arr = df['low'].values
        close_arr = df['close'].values

        regime = detector.get_full_regime(high_arr, low_arr, close_arr)
        market_regime = regime.get("market_regime", "unknown")
        volatility_regime = regime.get("volatility_regime", "normal")
        expansion_regime = regime.get("expansion_regime", "neutral")

        # === STEP 3: Consolidation Analysis ===
        from tradingagents.agents.analysts.breakout_quant import (
            analyze_consolidation,
            create_breakout_quant,
            _build_breakout_data_context,
            _build_breakout_prompt,
        )

        consolidation = analyze_consolidation(high_arr, low_arr, close_arr, lookback=20)

        # === STEP 4: Get SMC Context (for additional confluence) ===
        smc_context = ""
        smc_analysis = {}
        try:
            from tradingagents.agents.analysts.smc_quant import analyze_smc_for_quant
            smc_data = analyze_smc_for_quant(df, current_price)
            smc_analysis = smc_data.get("smc_analysis", {})
            smc_context = smc_data.get("smc_context", "")
        except Exception as smc_err:
            logger.warning(f"[BREAKOUT QUANT API] SMC analysis failed: {smc_err}")

        # === STEP 5: Build Indicator Context ===
        indicators_context = f"""## Technical Indicators

### Momentum
- RSI(14): {latest['rsi']:.1f} {'(overbought)' if latest['rsi'] > 70 else '(oversold)' if latest['rsi'] < 30 else '(neutral)'}
- MACD: {latest['macd']:.5f}
- MACD Signal: {latest['macd_signal']:.5f}
- MACD Histogram: {latest['macd_histogram']:.5f} {'(bullish)' if latest['macd_histogram'] > 0 else '(bearish)'}

### Trend
- EMA20: {latest['ema20']:.5f} (Price {'above' if current_price > latest['ema20'] else 'below'})
- EMA50: {latest['ema50']:.5f} (Price {'above' if current_price > latest['ema50'] else 'below'})
- ADX: {adx:.1f} ({'Strong trend' if adx > 25 else 'Weak/No trend'})

### Volatility
- ATR(14): {atr:.5f}
- BB Upper: {latest['bb_upper']:.5f}
- BB Middle: {latest['bb_middle']:.5f}
- BB Lower: {latest['bb_lower']:.5f}
- BB Width: {latest['bb_width']:.2f}%
"""

        # === STEP 6: Get Trade Memories ===
        from tradingagents.trade_decisions import get_trade_memories
        trade_memories = get_trade_memories(request.symbol)
        if trade_memories:
            logger.info(f"[BREAKOUT QUANT API] Injecting trade memories for {request.symbol} ({len(trade_memories)} chars)")

        # === STEP 7: Initialize LLM ===
        from tradingagents.llm_factory import get_llm

        llm = get_llm(tier="deep")

        # === STEP 8: Build State and Run Breakout Quant ===
        breakout_state = {
            "company_of_interest": request.symbol,
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "smc_context": smc_context,
            "smc_analysis": smc_analysis,
            "market_report": indicators_context,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "expansion_regime": expansion_regime,
            "trading_session": _get_trading_session(),
            "trade_memories": trade_memories,
            "price_data": {
                "high": high_arr.tolist(),
                "low": low_arr.tolist(),
                "close": close_arr.tolist(),
            }
        }

        # For debugging: log the prompt
        debug_data_context = _build_breakout_data_context(
            ticker=request.symbol,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=indicators_context,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            expansion_regime=expansion_regime,
            trading_session=_get_trading_session(),
            current_date=breakout_state["trade_date"],
            consolidation=consolidation,
        )
        full_prompt = _build_breakout_prompt(debug_data_context, trade_memories=trade_memories)

        # Run breakout quant analyst
        logger.info(f"[BREAKOUT QUANT API] Running breakout quant analyst for {request.symbol}...")
        _llm_start = _time.time()
        breakout_quant_analyst = create_breakout_quant(llm, use_structured_output=True)
        result = breakout_quant_analyst(breakout_state)
        _llm_duration = _time.time() - _llm_start

        breakout_report = result.get("breakout_quant_report", "")
        breakout_decision = result.get("breakout_quant_decision")
        consolidation_result = result.get("consolidation_analysis", consolidation)
        logger.info(f"[BREAKOUT QUANT API] Analyst returned: breakout_quant_decision={'present' if breakout_decision else 'None'} [LLM took {_llm_duration:.1f}s]")

        # === STEP 9: Build Response ===
        if breakout_decision:
            signal_map = {
                "buy_to_enter": "BUY",
                "sell_to_enter": "SELL",
                "hold": "HOLD",
                "close": "HOLD"
            }
            signal = breakout_decision.get("signal", "hold")
            if isinstance(signal, dict):
                signal = signal.get("value", "hold")

            mapped_signal = signal_map.get(signal, "HOLD")

            decision = {
                "signal": mapped_signal,
                "confidence": breakout_decision.get("confidence", 0.5),
                "entry_price": breakout_decision.get("entry_price"),
                "stop_loss": breakout_decision.get("stop_loss"),
                "take_profit": breakout_decision.get("profit_target"),
                "rationale": f"{breakout_decision.get('justification', '')}\n\n**Invalidation**: {breakout_decision.get('invalidation_condition', 'N/A')}",
                "analysis_mode": "breakout_quant",
                "leverage": breakout_decision.get("leverage"),
                "risk_usd": breakout_decision.get("risk_usd"),
                "risk_level": breakout_decision.get("risk_level"),
                "risk_reward_ratio": breakout_decision.get("risk_reward_ratio"),
                "trailing_stop_atr_multiplier": breakout_decision.get("trailing_stop_atr_multiplier"),
                "full_report": breakout_report
            }
        else:
            decision = {
                "signal": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "rationale": breakout_report or "Breakout quant analysis could not generate a structured decision.",
                "analysis_mode": "breakout_quant",
                "full_report": breakout_report
            }

        # Convert numpy types in consolidation for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        consolidation_serializable = make_serializable(consolidation_result)

        _endpoint_duration = _time.time() - _endpoint_start
        logger.info(f"[BREAKOUT QUANT API] Completed for {request.symbol} in {_endpoint_duration:.1f}s - Signal: {decision['signal']}")

        return {
            "status": "success",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": current_price,
            "bid": bid,
            "ask": ask,
            "decision": decision,
            "consolidation": consolidation_serializable,
            "regime": {
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
                "expansion_regime": expansion_regime,
                "adx": adx,
                "atr": atr,
            },
            "indicators": {
                "rsi": float(latest['rsi']) if pd.notna(latest['rsi']) else None,
                "macd": float(latest['macd']) if pd.notna(latest['macd']) else None,
                "macd_signal": float(latest['macd_signal']) if pd.notna(latest['macd_signal']) else None,
                "macd_histogram": float(latest['macd_histogram']) if pd.notna(latest['macd_histogram']) else None,
                "ema20": float(latest['ema20']) if pd.notna(latest['ema20']) else None,
                "ema50": float(latest['ema50']) if pd.notna(latest['ema50']) else None,
                "atr": atr,
                "adx": adx,
                "bb_upper": float(latest['bb_upper']) if pd.notna(latest['bb_upper']) else None,
                "bb_middle": float(latest['bb_middle']) if pd.notna(latest['bb_middle']) else None,
                "bb_lower": float(latest['bb_lower']) if pd.notna(latest['bb_lower']) else None,
                "bb_width": float(latest['bb_width']) if pd.notna(latest['bb_width']) else None,
            },
            "analysis_mode": "breakout_quant",
            "llm_used": True,
            "llm_duration_seconds": round(_llm_duration, 2),
            "endpoint_duration_seconds": round(_endpoint_duration, 2),
            "prompt_sent": full_prompt,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        _endpoint_duration = _time.time() - _endpoint_start
        logger.error(f"[BREAKOUT QUANT API] ERROR for {request.symbol} after {_endpoint_duration:.1f}s: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


class RangeQuantAnalysisRequest(BaseModel):
    """Request model for range quant analysis."""
    symbol: str
    timeframe: str = "H1"


@app.post("/api/analysis/range-quant")
async def run_range_quant_analysis(request: RangeQuantAnalysisRequest):
    """
    Run dedicated Range-Bound Market quant analysis using SMC levels.

    This is a range-focused quant that specializes in:
    - Identifying ranging/sideways market conditions
    - Mapping SMC zones (OBs, FVGs, equal levels) as range boundaries
    - Mean reversion entries at range extremes with SMC confluence
    - Liquidity sweep reversals at range boundaries

    Uses the dedicated range_quant agent with specialized ranging market prompt.
    Returns structured trade decision compatible with trade execution modal.
    """
    import time as _time
    _endpoint_start = _time.time()
    logger.info(f"[RANGE QUANT API] Request: symbol={request.symbol}, timeframe={request.timeframe}")
    try:
        import MetaTrader5 as mt5
        import pandas as pd
        import numpy as np

        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        # Get symbol info for current price
        symbol_info = mt5.symbol_info_tick(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not found")

        current_price = (symbol_info.bid + symbol_info.ask) / 2
        bid = symbol_info.bid
        ask = symbol_info.ask

        # Map timeframe
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = timeframe_map.get(request.timeframe.upper(), mt5.TIMEFRAME_H1)

        # Get OHLC data
        bars = mt5.copy_rates_from_pos(request.symbol, tf, 0, 200)
        if bars is None or len(bars) < 50:
            raise HTTPException(status_code=400, detail=f"Not enough data for {request.symbol}")

        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # === STEP 1: Calculate Indicators ===
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # EMAs
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        smoothed_tr = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / smoothed_tr)
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / smoothed_tr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.ewm(span=14, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

        # Get latest values
        latest = df.iloc[-1]
        atr = float(latest['atr']) if pd.notna(latest['atr']) else 0
        adx = float(latest['adx']) if pd.notna(latest['adx']) else 0

        # === STEP 2: Determine Regime ===
        high_arr = df['high'].values
        low_arr = df['low'].values
        close_arr = df['close'].values

        if adx > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                market_regime = "trending-up"
            else:
                market_regime = "trending-down"
        else:
            market_regime = "ranging"

        avg_atr = high_low.rolling(50).mean().iloc[-1]
        volatility_regime = "high" if atr > avg_atr * 1.5 else "normal" if atr > avg_atr * 0.7 else "low"

        # === STEP 3: Range Analysis ===
        from tradingagents.agents.analysts.range_quant import (
            analyze_range,
            create_range_quant,
            analyze_smc_for_range,
            _build_range_data_context,
            _build_range_prompt,
        )

        range_analysis = analyze_range(high_arr, low_arr, close_arr, lookback=25)

        # === REGIME GATE: Skip LLM call if market is NOT ranging ===
        # This saves the expensive LLM call when the market is clearly trending.
        # The LLM should only make decisions inside confirmed ranges.
        if not range_analysis.get("is_ranging", False):
            _gate_duration = _time.time() - _endpoint_start
            gate_reason = (
                f"Market not ranging for {request.symbol}: "
                f"trend_strength={range_analysis.get('trend_strength', 0):.2f}, "
                f"adx_proxy={range_analysis.get('adx_proxy', 0):.1f}, "
                f"mr_score={range_analysis.get('mean_reversion_score', 0):.0f}"
            )
            logger.info(f"[RANGE QUANT API] REGIME GATE blocked: {gate_reason}")
            return {
                "status": "ok",
                "symbol": request.symbol,
                "decision": {
                    "signal": "hold",
                    "confidence": 0.0,
                    "symbol": request.symbol,
                    "justification": f"Regime gate: {gate_reason}",
                    "invalidation_condition": "Market enters ranging conditions",
                    "entry_price": None,
                    "stop_loss": None,
                    "profit_target": None,
                    "risk_reward_ratio": None,
                    "leverage": None,
                    "risk_usd": None,
                    "risk_level": "low",
                    "order_type": None,
                    "trailing_stop_atr_multiplier": None,
                },
                "report": f"Range quant blocked by regime gate — {gate_reason}",
                "range_analysis": range_analysis,
                "regime": {
                    "market_regime": market_regime,
                    "volatility_regime": volatility_regime,
                },
                "analysis_mode": "range_quant",
                "prompt_sent": None,
                "llm_duration_seconds": 0,
            }

        # === STEP 4: Get SMC Context ===
        smc_context = ""
        smc_analysis = {}
        try:
            smc_data = analyze_smc_for_range(df, current_price)
            smc_analysis = smc_data.get("smc_analysis", {})
            smc_context = smc_data.get("smc_context", "")
        except Exception as smc_err:
            logger.warning(f"[RANGE QUANT API] SMC analysis failed: {smc_err}")

        # === STEP 5: Build Indicator Context ===
        indicators_context = f"""## Technical Indicators

### Momentum
- RSI(14): {latest['rsi']:.1f} {'(overbought)' if latest['rsi'] > 70 else '(oversold)' if latest['rsi'] < 30 else '(neutral)'}
- MACD: {latest['macd']:.5f}
- MACD Signal: {latest['macd_signal']:.5f}
- MACD Histogram: {latest['macd_histogram']:.5f} {'(bullish)' if latest['macd_histogram'] > 0 else '(bearish)'}

### Trend
- EMA20: {latest['ema20']:.5f} (Price {'above' if current_price > latest['ema20'] else 'below'})
- EMA50: {latest['ema50']:.5f} (Price {'above' if current_price > latest['ema50'] else 'below'})
- ADX: {adx:.1f} ({'Strong trend - CAUTION for range trading' if adx > 25 else 'Weak trend - favorable for range trading'})

### Volatility
- ATR(14): {atr:.5f}
- BB Upper: {latest['bb_upper']:.5f}
- BB Middle: {latest['bb_middle']:.5f}
- BB Lower: {latest['bb_lower']:.5f}
- BB Width: {latest['bb_width']:.2f}%
"""

        # === STEP 6: Get Trade Memories ===
        from tradingagents.trade_decisions import get_trade_memories
        trade_memories = get_trade_memories(request.symbol)
        if trade_memories:
            logger.info(f"[RANGE QUANT API] Injecting trade memories for {request.symbol} ({len(trade_memories)} chars)")

        # === STEP 7: Initialize LLM ===
        from tradingagents.llm_factory import get_llm

        llm = get_llm(tier="deep")

        # === STEP 8: Build State and Run Range Quant ===
        range_state = {
            "company_of_interest": request.symbol,
            "trade_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "smc_context": smc_context,
            "smc_analysis": smc_analysis,
            "market_report": indicators_context,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trading_session": _get_trading_session(),
            "trade_memories": trade_memories,
            "price_data": {
                "high": high_arr.tolist(),
                "low": low_arr.tolist(),
                "close": close_arr.tolist(),
            }
        }

        # For debugging: log the prompt
        debug_data_context = _build_range_data_context(
            ticker=request.symbol,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=indicators_context,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=_get_trading_session(),
            current_date=range_state["trade_date"],
            range_analysis=range_analysis,
        )
        full_prompt = _build_range_prompt(debug_data_context, trade_memories=trade_memories)

        # Run range quant analyst
        logger.info(f"[RANGE QUANT API] Running range quant analyst for {request.symbol}...")
        _llm_start = _time.time()
        range_quant_analyst = create_range_quant(llm, use_structured_output=True)
        result = range_quant_analyst(range_state)
        _llm_duration = _time.time() - _llm_start

        range_report = result.get("range_quant_report", "")
        range_decision = result.get("range_quant_decision")
        range_result = result.get("range_analysis", range_analysis)
        logger.info(f"[RANGE QUANT API] Analyst returned: range_quant_decision={'present' if range_decision else 'None'} [LLM took {_llm_duration:.1f}s]")

        # === STEP 9: Build Response ===
        if range_decision:
            signal_map = {
                "buy_to_enter": "BUY",
                "sell_to_enter": "SELL",
                "hold": "HOLD",
                "close": "HOLD"
            }
            signal = range_decision.get("signal", "hold")
            if isinstance(signal, dict):
                signal = signal.get("value", "hold")

            mapped_signal = signal_map.get(signal, "HOLD")

            decision = {
                "signal": mapped_signal,
                "confidence": range_decision.get("confidence", 0.5),
                "entry_price": range_decision.get("entry_price"),
                "stop_loss": range_decision.get("stop_loss"),
                "take_profit": range_decision.get("profit_target"),
                "rationale": f"{range_decision.get('justification', '')}\n\n**Invalidation**: {range_decision.get('invalidation_condition', 'N/A')}",
                "analysis_mode": "range_quant",
                "leverage": range_decision.get("leverage"),
                "risk_usd": range_decision.get("risk_usd"),
                "risk_level": range_decision.get("risk_level"),
                "risk_reward_ratio": range_decision.get("risk_reward_ratio"),
                "trailing_stop_atr_multiplier": range_decision.get("trailing_stop_atr_multiplier"),
                "full_report": range_report
            }
        else:
            decision = {
                "signal": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "rationale": range_report or "Range quant analysis could not generate a structured decision.",
                "analysis_mode": "range_quant",
                "full_report": range_report
            }

        # Convert numpy types for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        range_serializable = make_serializable(range_result)

        _endpoint_duration = _time.time() - _endpoint_start
        logger.info(f"[RANGE QUANT API] Completed for {request.symbol} in {_endpoint_duration:.1f}s - Signal: {decision['signal']}")

        return {
            "status": "success",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": current_price,
            "bid": bid,
            "ask": ask,
            "decision": decision,
            "range_analysis": range_serializable,
            "regime": {
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
                "adx": adx,
                "atr": atr,
            },
            "indicators": {
                "rsi": float(latest['rsi']) if pd.notna(latest['rsi']) else None,
                "macd": float(latest['macd']) if pd.notna(latest['macd']) else None,
                "macd_signal": float(latest['macd_signal']) if pd.notna(latest['macd_signal']) else None,
                "macd_histogram": float(latest['macd_histogram']) if pd.notna(latest['macd_histogram']) else None,
                "ema20": float(latest['ema20']) if pd.notna(latest['ema20']) else None,
                "ema50": float(latest['ema50']) if pd.notna(latest['ema50']) else None,
                "atr": atr,
                "adx": adx,
                "bb_upper": float(latest['bb_upper']) if pd.notna(latest['bb_upper']) else None,
                "bb_middle": float(latest['bb_middle']) if pd.notna(latest['bb_middle']) else None,
                "bb_lower": float(latest['bb_lower']) if pd.notna(latest['bb_lower']) else None,
                "bb_width": float(latest['bb_width']) if pd.notna(latest['bb_width']) else None,
            },
            "analysis_mode": "range_quant",
            "llm_used": True,
            "llm_duration_seconds": round(_llm_duration, 2),
            "endpoint_duration_seconds": round(_endpoint_duration, 2),
            "prompt_sent": full_prompt,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        _endpoint_duration = _time.time() - _endpoint_start
        logger.error(f"[RANGE QUANT API] ERROR for {request.symbol} after {_endpoint_duration:.1f}s: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ----- WebSocket for Real-time Updates -----

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Handle subscription requests
            try:
                msg = json.loads(data)
                if msg.get("type") == "subscribe":
                    # Handle subscription
                    pass
            except:
                pass

    except WebSocketDisconnect:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


@app.post("/api/automation/quant/broadcast-status")
async def broadcast_quant_status(payload: Dict[str, Any]):
    """Internal endpoint for the MT5 worker to broadcast status changes via WebSocket.

    Called by the worker whenever an automation instance changes state (started, stopped, etc).
    """
    instance = payload.get("instance_name")
    status = payload.get("status")

    # Intercept pair_optimizer broadcasts
    if instance == "pair_optimizer" and status:
        _optimize_task["status"] = status
        for key in ("message", "current", "total", "result", "error"):
            if key in payload:
                _optimize_task[key] = payload[key]

    # Intercept batch_trainer broadcasts to update the in-memory task state
    if instance == "batch_trainer" and status:
        _batch_train_task["status"] = status
        if "message" in payload:
            _batch_train_task["message"] = payload["message"]
        if "current" in payload:
            _batch_train_task["current"] = payload["current"]
        if "total" in payload:
            _batch_train_task["total"] = payload["total"]
        if "result" in payload:
            _batch_train_task["result"] = payload["result"]
        if "error" in payload:
            _batch_train_task["error"] = payload["error"]

    if instance and status:
        await broadcast_automation_status(instance, status, **{
            k: v for k, v in payload.items() if k not in ("instance_name", "status")
        })
    return {"ok": True}


# ----- Schemas -----

@app.get("/api/schemas")
async def get_schemas():
    """
    Get all JSON schemas for frontend type generation.

    Returns all Pydantic model schemas used for structured outputs,
    useful for generating TypeScript types in the frontend.
    """
    return generate_json_schemas()


@app.get("/api/schemas/{schema_name}")
async def get_schema(schema_name: str):
    """Get a specific schema by name"""
    schemas = generate_json_schemas()
    if schema_name not in schemas:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' not found")
    return schemas[schema_name]


# ----- Daily Cycle (Prediction Tracking) -----

DAILY_CYCLE_PID_FILE = PROJECT_ROOT / "daily_cycle.pid"
DAILY_CYCLE_STATE_FILE = PROJECT_ROOT / "examples" / ".last_run_state.json"
DAILY_CYCLE_PREDICTIONS_DIR = PROJECT_ROOT / "examples" / "pending_predictions"
DAILY_CYCLE_CONFIG_FILE = PROJECT_ROOT / "examples" / ".daily_cycle_config.json"


@app.get("/api/daily-cycle/status")
async def get_daily_cycle_status():
    """Get daily cycle status - shows if prediction tracking is running"""
    import psutil

    status = {
        "running": False,
        "pid": None,
        "last_run": None,
        "pending_predictions": 0,
        "symbols": [],
        "run_at": None,
        "started_at": None
    }

    # Check if process is running
    if DAILY_CYCLE_PID_FILE.exists():
        try:
            pid = int(DAILY_CYCLE_PID_FILE.read_text().strip())
            if psutil.pid_exists(pid):
                status["running"] = True
                status["pid"] = pid
        except Exception:
            pass

    # Load config with symbols being tracked
    if DAILY_CYCLE_CONFIG_FILE.exists():
        try:
            config = json.loads(DAILY_CYCLE_CONFIG_FILE.read_text())
            status["symbols"] = config.get("symbols", [])
            status["run_at"] = config.get("run_at")
            status["started_at"] = config.get("started_at")
            logging.debug(f"Daily cycle config loaded: symbols={status['symbols']}")
        except Exception as e:
            logging.error(f"Failed to load daily cycle config: {e}")
    else:
        logging.debug(f"Daily cycle config file not found: {DAILY_CYCLE_CONFIG_FILE}")

    # Load last run state
    if DAILY_CYCLE_STATE_FILE.exists():
        try:
            state = json.loads(DAILY_CYCLE_STATE_FILE.read_text())
            # Get most recent last_run across all symbols
            latest_run = None
            for symbol, info in state.items():
                run_time = info.get("last_run")
                if run_time and (latest_run is None or run_time > latest_run):
                    latest_run = run_time
            status["last_run"] = latest_run
        except Exception:
            pass

    # Count pending predictions
    if DAILY_CYCLE_PREDICTIONS_DIR.exists():
        try:
            status["pending_predictions"] = len(list(DAILY_CYCLE_PREDICTIONS_DIR.glob("*.pkl")))
        except Exception:
            pass

    # Include persistent state info
    persistent_state = LearningCycleState.get_status()
    status["enabled"] = persistent_state.get("enabled", False)
    status["last_start"] = persistent_state.get("last_start")
    status["last_stop"] = persistent_state.get("last_stop")
    status["stop_reason"] = persistent_state.get("stop_reason")

    # Symbol selection logic:
    # - When cycle is RUNNING: use config file symbols (what it's actually tracking)
    # - When cycle is STOPPED: use SQLite-persisted symbols (user's saved selection)
    # This ensures the user's selection survives page refreshes when not running
    if status["running"]:
        # Keep config file symbols (already loaded above)
        # Fall back to persisted if config is empty
        if not status["symbols"] and persistent_state.get("symbols"):
            status["symbols"] = persistent_state["symbols"]
    else:
        # When stopped, always prefer persisted user selection
        if persistent_state.get("symbols"):
            status["symbols"] = persistent_state["symbols"]

    # Also use persisted run_at if not in config
    if status["run_at"] is None and persistent_state.get("run_at") is not None:
        status["run_at"] = persistent_state["run_at"]

    # Debug info
    status["_debug"] = {
        "config_file_exists": DAILY_CYCLE_CONFIG_FILE.exists(),
        "config_file_path": str(DAILY_CYCLE_CONFIG_FILE),
        "persistent_symbols": persistent_state.get("symbols", []),
        "config_file_symbols": status.get("symbols", []) if status["running"] else [],
    }

    return status


class SaveSelectedSymbolsRequest(BaseModel):
    symbols: List[str]


@app.post("/api/daily-cycle/save-symbols")
async def save_daily_cycle_symbols(request: SaveSelectedSymbolsRequest):
    """Save selected symbols for daily cycle without starting it.

    This allows persisting the user's symbol selection so it survives
    page refreshes even before the cycle is started.
    """
    try:
        LearningCycleState.set_selected_symbols(request.symbols)
        logging.info(f"Saved {len(request.symbols)} selected symbols for daily cycle")
        return {
            "success": True,
            "symbols_saved": len(request.symbols),
            "symbols": request.symbols
        }
    except Exception as e:
        logging.error(f"Failed to save selected symbols: {e}")
        return {"success": False, "error": str(e)}


class DailyCycleStartRequest(BaseModel):
    symbols: Optional[List[str]] = None  # List of symbols to track
    use_market_watch: bool = False  # If true, use MT5 market watch symbols
    run_at: int = 9  # Hour of day to run (0-23)
    stagger_minutes: int = 5  # Minutes between each symbol analysis


@app.post("/api/daily-cycle/start")
async def start_daily_cycle(request: DailyCycleStartRequest = None):
    """Start daily cycle for symbols

    Args:
        symbols: List of trading symbols to track
        use_market_watch: If true, use symbols from MT5 market watch
        run_at: Hour of day to run analysis (0-23, default: 9)
        stagger_minutes: Minutes between each symbol analysis (default: 5)
    """
    import subprocess

    # Handle both query params (old style) and request body (new style)
    if request is None:
        request = DailyCycleStartRequest()

    # Check if already running
    if DAILY_CYCLE_PID_FILE.exists():
        try:
            pid = int(DAILY_CYCLE_PID_FILE.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                return {"error": "Daily cycle already running", "pid": pid}
        except Exception:
            pass

    # Determine which symbols to use
    symbols = []

    if request.use_market_watch:
        # Get symbols from MT5 market watch
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                all_symbols = mt5.symbols_get()
                if all_symbols:
                    symbols = [s.name for s in all_symbols if s.visible]
        except Exception as e:
            return {"error": f"Failed to get market watch symbols: {e}"}

        if not symbols:
            return {"error": "No symbols found in MT5 market watch"}
    elif request.symbols and len(request.symbols) > 0:
        symbols = request.symbols
    else:
        # Default to market watch symbols if none provided
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                all_symbols = mt5.symbols_get()
                if all_symbols:
                    symbols = [s.name for s in all_symbols if s.visible]
        except Exception:
            pass

        if not symbols:
            return {"error": "No symbols provided and unable to get market watch symbols"}

    try:
        # Start the daily cycle as a subprocess
        project_root = Path(__file__).parent.parent.parent

        # Create log directory and files for subprocess output
        log_dir = project_root / "logs" / "daily_cycle"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"subprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Build command with multiple symbols
        cmd = [
            sys.executable,
            str(project_root / "examples" / "daily_cycle.py"),
            "--symbols", *symbols,
            "--run-at", str(request.run_at),
            "--stagger-minutes", str(request.stagger_minutes)
        ]

        # Open log file for subprocess output
        log_handle = open(log_file, "w", encoding="utf-8")

        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout (both go to log file)
            cwd=str(project_root),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )

        # Save PID and symbols info
        DAILY_CYCLE_PID_FILE.write_text(str(process.pid))

        # Save symbols being tracked
        DAILY_CYCLE_CONFIG_FILE.write_text(json.dumps({
            "symbols": symbols,
            "run_at": request.run_at,
            "started_at": datetime.now().isoformat()
        }))

        # Persist enabled state
        LearningCycleState.set_enabled(True, symbols=symbols, run_at=request.run_at)

        return {
            "success": True,
            "message": f"Daily cycle started for {len(symbols)} symbol(s)",
            "pid": process.pid,
            "symbols": symbols,
            "run_at": request.run_at,
            "log_file": str(log_file)
        }
    except Exception as e:
        import traceback
        logging.error(f"Failed to start daily cycle: {e}")
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@app.post("/api/daily-cycle/stop")
async def stop_daily_cycle():
    """Stop the daily cycle"""
    import psutil

    if not DAILY_CYCLE_PID_FILE.exists():
        # Still update state even if PID file missing
        LearningCycleState.set_enabled(False)
        LearningCycleState.set_stop_reason("user_stopped")
        return {"error": "Daily cycle not running"}

    try:
        pid = int(DAILY_CYCLE_PID_FILE.read_text().strip())

        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()

        DAILY_CYCLE_PID_FILE.unlink(missing_ok=True)

        # Persist disabled state
        LearningCycleState.set_enabled(False)
        LearningCycleState.set_stop_reason("user_stopped")

        return {"success": True, "message": "Daily cycle stopped"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/daily-cycle/logs")
async def get_daily_cycle_logs(lines: int = 100, file: str = None):
    """Get recent daily cycle logs for debugging

    Args:
        lines: Number of lines to return (default 100, max 1000)
        file: Specific log file name to read (if not provided, uses most recent)
    """
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs" / "daily_cycle"

    if not log_dir.exists():
        return {"logs": [], "message": "No logs directory found", "available_files": []}

    # Get available log files
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    available_files = [f.name for f in log_files[:10]]  # Show last 10 files

    if not log_files:
        return {"logs": [], "message": "No log files found", "available_files": []}

    # Select which file to read
    if file:
        log_file = log_dir / file
        if not log_file.exists():
            return {"error": f"Log file '{file}' not found", "available_files": available_files}
    else:
        log_file = log_files[0]  # Most recent

    # Read the file
    try:
        lines = min(lines, 1000)  # Cap at 1000 lines
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        # Return last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "file": log_file.name,
            "total_lines": len(all_lines),
            "returned_lines": len(log_lines),
            "logs": [line.rstrip() for line in log_lines],
            "available_files": available_files
        }
    except Exception as e:
        return {"error": f"Failed to read log file: {e}", "available_files": available_files}


@app.get("/api/daily-cycle/predictions")
async def get_pending_predictions(symbol: str = None):
    """Get list of pending predictions awaiting evaluation"""
    import pickle

    if not DAILY_CYCLE_PREDICTIONS_DIR.exists():
        return {"predictions": []}

    predictions = []
    for f in DAILY_CYCLE_PREDICTIONS_DIR.glob("*.pkl"):
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)

            if data.get("status") != "pending":
                continue

            if symbol and data.get("symbol") != symbol:
                continue

            pred = data.get("prediction", {})
            predictions.append({
                "symbol": data.get("symbol"),
                "signal": pred.get("signal"),
                "expected_direction": pred.get("expected_direction"),
                "price_at_analysis": pred.get("price_at_analysis"),
                "analysis_timestamp": pred.get("analysis_timestamp"),
                "evaluation_due": pred.get("evaluation_due"),
                "filename": f.name
            })
        except Exception:
            continue

    # Sort by evaluation_due
    predictions.sort(key=lambda x: x.get("evaluation_due", ""), reverse=True)

    return {"predictions": predictions}


# ----- Quant Automation -----

# Global state: keyed by instance_name (e.g. "quant", "volume_profile")
# NOTE: These are only used when running automations locally (same machine as backend)
# For remote automations, status comes from Postgres and commands go via automation_control table
_automation_instances: Dict[str, Any] = {}
_automation_tasks: Dict[str, Any] = {}

# Legacy file path - configs now stored in Postgres
_AUTOMATION_CONFIGS_FILE = PROJECT_ROOT / "automation_configs.json"

# Remote automation control via Postgres
_automation_control = None

def _get_automation_control():
    """Get the automation control store singleton."""
    global _automation_control
    if _automation_control is None:
        from tradingagents.storage.automation_control import get_automation_control
        _automation_control = get_automation_control()
    return _automation_control


async def _load_saved_configs() -> Dict[str, dict]:
    """Load saved automation configs from Postgres.

    Falls back to local file for migration, then syncs to Postgres.
    """
    try:
        control = _get_automation_control()
        statuses = await control.get_all_statuses()

        configs = {}
        for status in statuses:
            name = status.get("instance_name")
            config = status.get("config") or {}
            if name and config:
                configs[name] = config

        # Migration: if we have local configs not in Postgres, sync them
        if _AUTOMATION_CONFIGS_FILE.exists():
            try:
                with open(_AUTOMATION_CONFIGS_FILE, "r") as f:
                    local_configs = json.load(f)
                for name, cfg in local_configs.items():
                    if name not in configs:
                        # Migrate to Postgres
                        await control.update_status(
                            instance_name=name,
                            status="stopped",
                            pipeline=cfg.get("pipeline"),
                            symbols=cfg.get("symbols"),
                            auto_execute=cfg.get("auto_execute", False),
                            config=cfg,
                        )
                        configs[name] = cfg
                        logger.info(f"Migrated config '{name}' from local file to Postgres")
            except Exception as e:
                logger.warning(f"Failed to migrate local configs: {e}")

        return configs
    except Exception as e:
        logger.warning(f"Failed to load configs from Postgres: {e}")
        # Fallback to local file
        try:
            if _AUTOMATION_CONFIGS_FILE.exists():
                with open(_AUTOMATION_CONFIGS_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}


def _load_saved_configs_sync() -> Dict[str, dict]:
    """Synchronous wrapper for loading configs (for non-async contexts)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _load_saved_configs())
                return future.result(timeout=5)
        else:
            return loop.run_until_complete(_load_saved_configs())
    except Exception as e:
        logger.warning(f"Failed to load configs sync: {e}")
        # Fallback to local file
        try:
            if _AUTOMATION_CONFIGS_FILE.exists():
                with open(_AUTOMATION_CONFIGS_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}


async def _save_config(instance_name: str, config: dict):
    """Save an automation config to Postgres."""
    try:
        control = _get_automation_control()
        await control.update_status(
            instance_name=instance_name,
            status="stopped",  # Will be updated when automation starts
            pipeline=config.get("pipeline"),
            symbols=config.get("symbols"),
            auto_execute=config.get("auto_execute", False),
            config=config,
        )
        logger.info(f"Saved automation config for '{instance_name}' to Postgres")
    except Exception as e:
        logger.error(f"Failed to save config to Postgres: {e}")
        # Fallback: save locally
        try:
            configs = {}
            if _AUTOMATION_CONFIGS_FILE.exists():
                with open(_AUTOMATION_CONFIGS_FILE, "r") as f:
                    configs = json.load(f)
            configs[instance_name] = config
            with open(_AUTOMATION_CONFIGS_FILE, "w") as f:
                json.dump(configs, f, indent=2)
            logger.info(f"Saved automation config for '{instance_name}' to local file (fallback)")
        except Exception as e2:
            logger.error(f"Failed to save config locally: {e2}")
            raise


# Global symbol position limits (shared across all automations)
_SYMBOL_LIMITS_FILE = PROJECT_ROOT / "automation_symbol_limits.json"


def _load_symbol_limits() -> Dict[str, dict]:
    """Load global per-symbol position limits.

    Returns dict like {"XAUUSD": {"max_positions": 3}, ...}
    """
    try:
        if _SYMBOL_LIMITS_FILE.exists():
            with open(_SYMBOL_LIMITS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load symbol limits: {e}")
    return {}


def _save_symbol_limits(limits: Dict[str, dict]):
    with open(_SYMBOL_LIMITS_FILE, "w") as f:
        json.dump(limits, f, indent=2)


class QuantAutomationConfigRequest(BaseModel):
    """Request model for quant automation configuration.

    Accepts extra fields via model_config so new config options
    (e.g. delegate_position_management, cooldown_enabled) pass through
    to Postgres and the worker without needing to be added here.
    """
    model_config = ConfigDict(extra="allow")

    instance_name: str = "smc_quant_basic"
    pipeline: str = "smc_quant_basic"
    symbols: List[str] = ["XAUUSD"]
    timeframe: str = "H1"
    analysis_interval_seconds: int = 180
    position_check_interval_seconds: int = 60
    auto_execute: bool = False
    min_confidence: float = 0.65
    max_positions_per_symbol: int = 1
    enable_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    enable_breakeven_stop: bool = True
    move_to_breakeven_atr_mult: float = 1.5
    enable_reversal_close: bool = True
    delegate_position_management: bool = False
    cooldown_enabled: bool = True
    max_risk_per_trade_pct: float = 1.0
    default_lot_size: float = 0.01
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 3


def _get_automation_instance(instance_name: str):
    """Get an automation instance by name, or None if not found."""
    return _automation_instances.get(instance_name)


@app.get("/api/automation/quant/status")
async def get_quant_automation_status(instance: str = "quant"):
    """Get quant automation status for a specific instance.

    Reads status from Postgres (shared across all web UI instances).
    Falls back to local instance if running on same machine.
    """
    # First check Postgres for remote status
    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance)
        if remote_status:
            # Check if status is stale (no update in 60 seconds = likely stopped)
            updated_at = remote_status.get("updated_at")
            if updated_at:
                from datetime import datetime, timezone
                try:
                    last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - last_update).total_seconds() > 60:
                        remote_status["status"] = "stopped"
                        remote_status["running"] = False
                        remote_status["stale"] = True
                    else:
                        remote_status["running"] = remote_status.get("status") == "running"
                        remote_status["stale"] = False
                except Exception:
                    remote_status["running"] = remote_status.get("status") == "running"
            return remote_status
    except Exception as e:
        logger.warning(f"Failed to get remote status for {instance}: {e}")

    # Fallback to local instance
    automation = _get_automation_instance(instance)
    if automation is not None:
        status = automation.get_status()
        status["instance_name"] = instance
        return status

    # Return saved config with stopped status
    saved_configs = await _load_saved_configs()
    saved_config = saved_configs.get(instance)
    return {
        "status": "stopped",
        "running": False,
        "error": None,
        "config": saved_config,
        "instance_name": instance,
    }


@app.get("/api/automation/quant/instances")
async def list_automation_instances():
    """List all automation instances (from Postgres + local).

    Reads status from Postgres for cross-machine visibility.
    """
    result = {}

    # Get all statuses from Postgres (primary source)
    try:
        control = _get_automation_control()
        remote_statuses = await control.get_all_statuses()

        from datetime import datetime, timezone

        for status in remote_statuses:
            name = status.get("instance_name")
            if not name:
                continue

            # Check if status is stale (no update in 60 seconds = likely stopped)
            updated_at = status.get("updated_at")
            if updated_at:
                try:
                    last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - last_update).total_seconds() > 60:
                        status["status"] = "stopped"
                        status["running"] = False
                        status["stale"] = True
                    else:
                        status["running"] = status.get("status") == "running"
                        status["stale"] = False
                except Exception:
                    status["running"] = status.get("status") == "running"
            else:
                status["running"] = status.get("status") == "running"

            result[name] = status

    except Exception as e:
        logger.warning(f"Failed to get remote statuses: {e}")
        # Fallback to saved configs
        saved_configs = await _load_saved_configs()
        for name, cfg in saved_configs.items():
            result[name] = {
                "status": "stopped",
                "running": False,
                "error": None,
                "config": cfg,
                "instance_name": name,
            }

    # Include local running instances not in Postgres (edge case)
    for name, instance in _automation_instances.items():
        if name not in result:
            status = instance.get_status()
            status["instance_name"] = name
            result[name] = status

    return {"instances": result}


@app.get("/api/automation/symbol-limits")
async def get_symbol_limits():
    """Get global per-symbol position limits.

    Auto-populates with all symbols used across saved automation configs.
    """
    limits = _load_symbol_limits()

    # Collect all symbols from saved automation configs
    configs = await _load_saved_configs()
    all_symbols: set = set()
    for cfg in configs.values():
        for sym in cfg.get("symbols", []):
            all_symbols.add(sym)

    # Ensure every active symbol has an entry (default max_positions=3)
    for sym in sorted(all_symbols):
        if sym not in limits:
            limits[sym] = {"max_positions": 3}

    return {"limits": limits}


@app.post("/api/automation/symbol-limits")
async def update_symbol_limits(request: dict):
    """Update global per-symbol position limits.

    Body: {"limits": {"XAUUSD": {"max_positions": 3}, ...}}
    """
    new_limits = request.get("limits", {})
    if not isinstance(new_limits, dict):
        return {"error": "limits must be a dict"}

    # Validate
    for sym, cfg in new_limits.items():
        mp = cfg.get("max_positions")
        if mp is not None and (not isinstance(mp, int) or mp < 1 or mp > 20):
            return {"error": f"max_positions for {sym} must be 1-20"}

    _save_symbol_limits(new_limits)
    return {"success": True, "limits": new_limits}


@app.post("/api/automation/symbol-limits/{symbol}")
async def update_single_symbol_limit(symbol: str, request: dict):
    """Update limit for a single symbol.

    Body: {"max_positions": 3}
    """
    mp = request.get("max_positions")
    if mp is None or not isinstance(mp, int) or mp < 1 or mp > 20:
        return {"error": "max_positions must be 1-20"}

    limits = _load_symbol_limits()
    limits[symbol.upper()] = {"max_positions": mp}
    _save_symbol_limits(limits)
    return {"success": True, "symbol": symbol.upper(), "max_positions": mp}


@app.delete("/api/automation/quant/config/{instance_name}")
async def delete_automation_config(instance_name: str):
    """Delete a saved automation config. Must be stopped first."""
    # Check if running locally
    automation = _automation_instances.get(instance_name)
    if automation and automation._running:
        raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is still running. Stop it first.")

    # Check if running remotely
    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance_name)
        if remote_status and remote_status.get("status") == "running":
            # Check if stale
            from datetime import datetime, timezone
            updated_at = remote_status.get("updated_at")
            if updated_at:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() <= 60:
                    raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is still running remotely. Stop it first.")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to check remote status: {e}")

    # Remove from Postgres
    try:
        control = _get_automation_control()
        await control.delete_status(instance_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete config: {e}")

    # Clean up any stale local references
    _automation_instances.pop(instance_name, None)
    _automation_tasks.pop(instance_name, None)

    return {"status": "deleted", "instance_name": instance_name}


@app.put("/api/automation/quant/config/{instance_name}/rename")
async def rename_automation_config(instance_name: str, new_name: str = Query(...)):
    """Rename a saved automation config. Must be stopped first."""
    if not new_name or not new_name.strip():
        raise HTTPException(status_code=400, detail="New name cannot be empty.")

    new_name = new_name.strip()

    # Cannot rename a running instance (local)
    automation = _automation_instances.get(instance_name)
    if automation and automation._running:
        raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is still running. Stop it first.")

    # Cannot rename a running instance (remote)
    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance_name)
        if remote_status and remote_status.get("status") == "running":
            from datetime import datetime, timezone
            updated_at = remote_status.get("updated_at")
            if updated_at:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() <= 60:
                    raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is still running remotely. Stop it first.")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to check remote status: {e}")

    configs = await _load_saved_configs()
    if instance_name not in configs:
        # Instance might exist only in memory (never saved) - grab its config
        mem_instance = _automation_instances.get(instance_name)
        if mem_instance:
            configs[instance_name] = mem_instance.config.to_dict()
        else:
            raise HTTPException(status_code=404, detail=f"Instance '{instance_name}' not found.")
    if new_name in configs:
        raise HTTPException(status_code=409, detail=f"Instance '{new_name}' already exists.")

    # Move config under new name in Postgres
    config = configs[instance_name]
    config["instance_name"] = new_name
    try:
        control = _get_automation_control()
        # Save new config
        await control.update_status(
            instance_name=new_name,
            status="stopped",
            pipeline=config.get("pipeline"),
            symbols=config.get("symbols"),
            auto_execute=config.get("auto_execute", False),
            config=config,
        )
        # Delete old config
        await control.delete_status(instance_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename config: {e}")

    # Clean up stale local references under old name
    _automation_instances.pop(instance_name, None)
    _automation_tasks.pop(instance_name, None)

    logger.info(f"Renamed automation instance '{instance_name}' -> '{new_name}'")
    return {"status": "renamed", "old_name": instance_name, "new_name": new_name}


@app.post("/api/automation/quant/start")
async def start_quant_automation(config: QuantAutomationConfigRequest):
    """Start quant automation with given configuration.

    Saves config to Postgres and sends a 'start' command for the remote worker.
    The home machine's worker will pick up the command and start the automation.
    """
    instance_name = config.instance_name

    # Check if already running (remote)
    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance_name)
        if remote_status and remote_status.get("status") == "running":
            from datetime import datetime, timezone
            updated_at = remote_status.get("updated_at")
            if updated_at:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() <= 60:
                    raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is already running")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to check remote status: {e}")

    # Check if already running (local)
    existing = _get_automation_instance(instance_name)
    if existing and existing._running:
        raise HTTPException(status_code=400, detail=f"Instance '{instance_name}' is already running locally")

    try:
        # Backward compat: rename old "quant" pipeline to "smc_quant_basic"
        pipeline_name = config.pipeline
        if pipeline_name == "quant":
            pipeline_name = "smc_quant_basic"

        # Build config dict from all request fields (typed + extra)
        config_dict = config.model_dump()
        config_dict["instance_name"] = instance_name
        config_dict["pipeline"] = pipeline_name
        config_dict["state_file"] = f"quant_automation_state_{instance_name}.json"
        config_dict["enable_remote_control"] = True
        config_dict["control_poll_seconds"] = 3

        # Save config to Postgres
        control = _get_automation_control()
        await control.update_status(
            instance_name=instance_name,
            status="pending_start",
            pipeline=pipeline_name,
            symbols=config.symbols,
            auto_execute=config.auto_execute,
            config=config_dict,
        )

        # Broadcast pending_start to all clients
        await broadcast_automation_status(instance_name, "pending_start")

        # Send start command for the remote worker to pick up
        command_id = await control.send_command(
            instance_name=instance_name,
            action="start",
            payload=config_dict,
        )

        return {
            "status": "start_requested",
            "instance_name": instance_name,
            "config": config_dict,
            "command_id": command_id,
            "message": "Start command sent. Worker will pick it up shortly.",
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/automation/quant/start-all")
async def start_all_quant_automations():
    """Start all configured automation instances.

    Sends start commands via Postgres for remote worker to pick up.
    """
    from datetime import datetime, timezone

    # Get all configs from Postgres
    all_configs = await _load_saved_configs()
    if not all_configs:
        raise HTTPException(status_code=404, detail="No automation configs found")

    results = {}
    control = _get_automation_control()

    for instance_name, cfg in all_configs.items():
        # Check if already running
        try:
            remote_status = await control.get_status(instance_name)
            if remote_status and remote_status.get("status") == "running":
                updated_at = remote_status.get("updated_at")
                if updated_at:
                    last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - last_update).total_seconds() <= 60:
                        results[instance_name] = {"status": "already_running"}
                        continue
        except Exception as e:
            logger.warning(f"Failed to check status for {instance_name}: {e}")

        try:
            # Send start command
            command_id = await control.send_command(
                instance_name=instance_name,
                action="start",
                payload=cfg,
            )
            # Update status to pending
            await control.update_status(
                instance_name=instance_name,
                status="pending_start",
                pipeline=cfg.get("pipeline"),
                symbols=cfg.get("symbols"),
                auto_execute=cfg.get("auto_execute", False),
                config=cfg,
            )
            await broadcast_automation_status(instance_name, "pending_start")
            results[instance_name] = {"status": "start_requested", "command_id": command_id}
        except Exception as e:
            results[instance_name] = {"status": "error", "error": str(e)}

    return {"results": results}


@app.post("/api/automation/quant/stop-all")
async def stop_all_quant_automations():
    """Stop all running automation instances.

    Sends stop commands via Postgres for remote worker to pick up.
    """
    results = {}
    control = _get_automation_control()

    # Get all statuses from Postgres
    try:
        statuses = await control.get_all_statuses()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statuses: {e}")

    for status in statuses:
        name = status.get("instance_name")
        if not name:
            continue

        if status.get("status") in ("running", "pending_start"):
            try:
                command_id = await control.send_command(
                    instance_name=name,
                    action="stop",
                )
                await broadcast_automation_status(name, "stopping")
                results[name] = {"status": "stop_requested", "command_id": command_id}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        else:
            results[name] = {"status": "already_stopped"}

    # Also stop any local instances
    for name, automation in list(_automation_instances.items()):
        if automation._running:
            try:
                automation.stop()
                results[name] = {"status": "stopped_locally"}
            except Exception as e:
                if name not in results:
                    results[name] = {"status": "error", "error": str(e)}
        _automation_instances.pop(name, None)
        _automation_tasks.pop(name, None)

    return {"results": results}


@app.post("/api/automation/quant/stop")
async def stop_quant_automation(instance: str = Query(default="quant")):
    """Stop a quant automation instance.

    Sends a stop command via Postgres for remote worker to pick up.
    """
    # Send stop command via Postgres
    try:
        control = _get_automation_control()
        command_id = await control.send_command(
            instance_name=instance,
            action="stop",
        )
        # Broadcast stopping state to all clients (spinner stays until worker confirms stopped)
        await broadcast_automation_status(instance, "stopping")
        return {
            "status": "stop_requested",
            "instance_name": instance,
            "command_id": command_id,
            "message": "Stop command sent. Worker will pick it up shortly.",
        }
    except Exception as e:
        logger.error(f"Failed to send stop command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/automation/quant/pause")
async def pause_quant_automation(instance: str = Query(default="quant")):
    """Pause automation (disable auto-execute but continue monitoring).

    Sends a pause command via Postgres for remote worker to pick up.
    """
    try:
        control = _get_automation_control()
        command_id = await control.send_command(
            instance_name=instance,
            action="pause",
        )
        await broadcast_automation_status(instance, "pausing")
        return {
            "status": "pause_requested",
            "instance_name": instance,
            "command_id": command_id,
            "message": "Pause command sent. Worker will pick it up shortly.",
        }
    except Exception as e:
        logger.error(f"Failed to send pause command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/automation/quant/resume")
async def resume_quant_automation(instance: str = Query(default="quant")):
    """Resume automation with auto-execute enabled.

    Sends a resume command via Postgres for remote worker to pick up.
    """
    try:
        control = _get_automation_control()
        command_id = await control.send_command(
            instance_name=instance,
            action="resume",
        )
        await broadcast_automation_status(instance, "resuming")
        return {
            "status": "resume_requested",
            "instance_name": instance,
            "command_id": command_id,
            "message": "Resume command sent. Worker will pick it up shortly.",
        }
    except Exception as e:
        logger.error(f"Failed to send resume command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/automation/quant/config")
async def update_quant_automation_config(updates: Dict[str, Any], instance: str = Query(default="quant")):
    """Update automation configuration dynamically. Works for running or stopped instances.

    Sends config update to Postgres. If running remotely, sends an update_config command.
    """
    control = _get_automation_control()

    # Get current config from Postgres
    saved_configs = await _load_saved_configs()
    config_dict = saved_configs.get(instance, {})
    config_dict.update(updates)

    # Save to Postgres
    try:
        await control.update_status(
            instance_name=instance,
            status=None,  # Don't change status
            pipeline=config_dict.get("pipeline"),
            symbols=config_dict.get("symbols"),
            auto_execute=config_dict.get("auto_execute"),
            config=config_dict,
        )
    except Exception as e:
        logger.error(f"Failed to save config for '{instance}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # If instance is running, send update_config command
    try:
        remote_status = await control.get_status(instance)
        if remote_status and remote_status.get("status") == "running":
            await control.send_command(
                instance_name=instance,
                action="update_config",
                payload=updates,
            )
    except Exception as e:
        logger.warning(f"Failed to send update_config command: {e}")

    return {
        "status": "updated",
        "instance_name": instance,
        "config": config_dict,
    }


@app.post("/api/automation/quant/test-analysis/{symbol}")
async def test_quant_analysis(symbol: str, pipeline: str = Query(default="smc_quant_basic")):
    """Run a single analysis cycle for testing (without execution)."""
    try:
        from tradingagents.automation.quant_automation import (
            QuantAutomation,
            QuantAutomationConfig,
            PipelineType,
        )

        # Backward compat
        if pipeline == "quant":
            pipeline = "smc_quant_basic"

        config = QuantAutomationConfig(
            pipeline=PipelineType(pipeline),
            symbols=[symbol],
            auto_execute=False,
        )
        automation = QuantAutomation(config)

        result = await automation.run_single_analysis(symbol)

        return {
            "symbol": result.symbol,
            "pipeline": result.pipeline,
            "signal": result.signal,
            "confidence": result.confidence,
            "entry_price": result.entry_price,
            "stop_loss": result.stop_loss,
            "take_profit": result.take_profit,
            "rationale": result.rationale,
            "duration_seconds": result.duration_seconds,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decisions/cleanup-stale")
async def cleanup_stale_decisions_endpoint(dry_run: bool = Query(default=True)):
    """Find and close active decisions whose MT5 positions no longer exist.

    Args:
        dry_run: If True (default), only report stale decisions. If False, close them.
    """
    import asyncio
    try:
        from tradingagents.trade_decisions import cleanup_stale_decisions

        stale = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: cleanup_stale_decisions(dry_run=dry_run),
        )

        return {
            "dry_run": dry_run,
            "stale_count": len(stale),
            "stale_decisions": [
                {
                    "decision_id": d.get("decision_id"),
                    "symbol": d.get("symbol"),
                    "action": d.get("action"),
                    "source": d.get("source"),
                    "mt5_ticket": d.get("mt5_ticket"),
                    "entry_price": d.get("entry_price"),
                    "created_at": d.get("created_at"),
                }
                for d in stale
            ],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/automation/quant/assumption-review")
async def trigger_assumption_review(
    instance: str = Query(default="btc_smc_basic"),
    use_llm: bool = Query(default=True),
):
    """Manually trigger a position assumption review for an automation instance."""
    import asyncio

    try:
        from tradingagents.automation.position_assumption_review import (
            review_all_positions,
            format_review_summary,
        )

        # Get source and symbols from running instance or saved config
        automation = _get_automation_instance(instance)
        if automation:
            source = automation._source
            symbols = automation.config.symbols
            timeframe = automation.config.timeframe
        else:
            saved_configs = _load_saved_configs()
            cfg = saved_configs.get(instance, {})
            source = instance
            symbols = cfg.get("symbols", [])
            timeframe = cfg.get("timeframe", "H1")

        if not symbols:
            raise HTTPException(status_code=400, detail=f"No symbols configured for instance '{instance}'")

        # Run in thread to avoid blocking
        reports = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: review_all_positions(
                source_filter=source,
                symbols=symbols,
                timeframe=timeframe,
                auto_apply=False,
                use_llm=use_llm,
            ),
        )

        results = []
        for r in reports:
            results.append({
                "decision_id": r.decision_id,
                "symbol": r.symbol,
                "direction": r.direction,
                "ticket": r.ticket,
                "entry_price": r.entry_price,
                "current_price": r.current_price,
                "current_sl": r.current_sl,
                "current_tp": r.current_tp,
                "pnl_pct": r.pnl_pct,
                "recommended_action": r.recommended_action,
                "suggested_sl": r.suggested_sl,
                "suggested_tp": r.suggested_tp,
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
            })

        # Update running instance state if available
        if automation:
            automation._last_assumption_review = datetime.now()
            automation._assumption_review_results = results

        return {
            "instance": instance,
            "source": source,
            "symbols": symbols,
            "reports": results,
            "summary": format_review_summary(reports),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ----- Auto-Tuner -----

_tune_tasks: Dict[str, Dict[str, Any]] = {}
_TUNE_HISTORY_FILE = PROJECT_ROOT / "tune_history.json"


def _load_tune_history() -> Dict[str, list]:
    """Load tune history from disk. Format: {symbol_pipeline: [tune_records]}."""
    try:
        if _TUNE_HISTORY_FILE.exists():
            with open(_TUNE_HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load tune history: {e}")
    return {}


def _tune_history_key(symbol: str, pipeline: str) -> str:
    """Consistent key for tune history: SYMBOL_pipeline."""
    return f"{symbol}_{pipeline}"


def _get_instance_symbol_pipeline(instance_name: str) -> tuple:
    """Resolve symbol and pipeline from an instance name."""
    automation = _get_automation_instance(instance_name)
    if automation:
        symbol = automation.config.symbols[0] if automation.config.symbols else None
        pipeline = automation.config.pipeline
    else:
        saved_configs = _load_saved_configs()
        cfg = saved_configs.get(instance_name, {})
        symbol = cfg.get("symbols", [None])[0] if cfg.get("symbols") else None
        pipeline = cfg.get("pipeline", "")
    return symbol, pipeline


def _save_tune_record(symbol: str, pipeline: str, record: dict):
    """Append a tune record to history, keyed by symbol+pipeline (shared across instances)."""
    history = _load_tune_history()
    key = _tune_history_key(symbol, pipeline)
    if key not in history:
        history[key] = []
    history[key].append(record)
    # Keep last 20 records per symbol+pipeline
    history[key] = history[key][-20:]
    with open(_TUNE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Saved tune record for '{key}' ({len(history[key])} total)")


@app.post("/api/automation/tune/{instance_name}")
async def start_tune(
    instance_name: str,
    bars: int = Query(default=800),
    min_trades: int = Query(default=5),
    auto_apply: bool = Query(default=False),
):
    """Start a background parameter tuning task for an automation instance."""
    try:
        # Check if tune is already running for this instance
        existing = _tune_tasks.get(instance_name)
        if existing and existing.get("status") == "running":
            return {"task_id": instance_name, "status": "already_running"}

        # Look up instance config to get symbol + pipeline
        automation = _get_automation_instance(instance_name)
        if automation:
            symbol = automation.config.symbols[0] if automation.config.symbols else None
            pipeline = automation.config.pipeline
        else:
            saved_configs = _load_saved_configs()
            cfg = saved_configs.get(instance_name, {})
            symbol = cfg.get("symbols", [None])[0] if cfg.get("symbols") else None
            pipeline = cfg.get("pipeline", "")

        if not symbol:
            raise HTTPException(status_code=400, detail=f"No symbol configured for instance '{instance_name}'")
        if not pipeline:
            raise HTTPException(status_code=400, detail=f"No pipeline configured for instance '{instance_name}'")
        if pipeline == "multi_agent":
            raise HTTPException(status_code=400, detail="multi_agent pipeline requires LLM and cannot be auto-tuned")

        # Initialize task state
        _tune_tasks[instance_name] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "symbol": symbol,
            "pipeline": pipeline,
            "bars": bars,
            "min_trades": min_trades,
            "auto_apply": auto_apply,
            "progress": {"phase": "starting", "current": 0, "total": 0, "message": "Starting tune...", "steps": []},
            "result": None,
            "error": None,
        }

        # Spawn background task
        async def _run_tune_task():
            from tradingagents.automation.auto_tuner import run_tune, best_params_to_config_updates

            # Capture the event loop so we can schedule coroutines from worker threads
            loop = asyncio.get_running_loop()

            def progress_callback(phase, current, total, message, steps=None):
                task_state = _tune_tasks.get(instance_name)
                if task_state:
                    task_state["progress"] = {
                        "phase": phase,
                        "current": current,
                        "total": total,
                        "message": message,
                        "steps": steps or task_state["progress"].get("steps", []),
                    }
                    # Broadcast via WebSocket (thread-safe)
                    try:
                        loop.call_soon_threadsafe(
                            asyncio.ensure_future,
                            _broadcast_tune_progress(instance_name, task_state["progress"]),
                        )
                    except RuntimeError:
                        pass  # Loop closed

            try:
                result = await run_tune(
                    symbol=symbol,
                    pipeline=pipeline,
                    bars=bars,
                    min_trades=min_trades,
                    progress_callback=progress_callback,
                )

                task_state = _tune_tasks.get(instance_name)
                if not task_state:
                    return

                if result.get("error"):
                    task_state["status"] = "error"
                    task_state["error"] = result["error"]
                    task_state["result"] = result
                else:
                    task_state["status"] = "done"
                    task_state["result"] = result

                    # Save tune record to history (keyed by symbol+pipeline)
                    _save_tune_record(symbol, pipeline, {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "pipeline": pipeline,
                        "bars": bars,
                        "duration_seconds": result.get("duration_seconds"),
                        "best": result.get("best"),
                        "top_5": result.get("top_5", [])[:5],
                        "config_updates": result.get("config_updates", {}),
                        "applied": False,
                        "applied_at": None,
                        "config_before_apply": None,
                    })

                    # Auto-apply if requested
                    if auto_apply and result.get("config_updates"):
                        try:
                            _apply_tune_config(instance_name, result["config_updates"])
                            task_state["applied"] = True
                        except Exception as e:
                            task_state["apply_error"] = str(e)
                            task_state["applied"] = False

                # Broadcast completion
                asyncio.ensure_future(_broadcast_tune_complete(instance_name, task_state))

            except Exception as e:
                import traceback
                traceback.print_exc()
                task_state = _tune_tasks.get(instance_name)
                if task_state:
                    task_state["status"] = "error"
                    task_state["error"] = str(e)

        asyncio.create_task(_run_tune_task())

        return {"task_id": instance_name, "status": "started", "symbol": symbol, "pipeline": pipeline}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/automation/tune/{instance_name}/status")
async def get_tune_status(instance_name: str):
    """Get the status/result of a tuning task."""
    task_state = _tune_tasks.get(instance_name)
    if not task_state:
        raise HTTPException(status_code=404, detail=f"No tune task found for '{instance_name}'")
    return task_state


@app.post("/api/automation/tune/{instance_name}/apply")
async def apply_tune_result(instance_name: str):
    """Apply the best config from a completed tune task."""
    task_state = _tune_tasks.get(instance_name)
    if not task_state:
        raise HTTPException(status_code=404, detail=f"No tune task found for '{instance_name}'")
    if task_state.get("status") != "done":
        raise HTTPException(status_code=400, detail="Tune task is not complete")

    result = task_state.get("result", {})
    config_updates = result.get("config_updates", {})
    if not config_updates:
        raise HTTPException(status_code=400, detail="No config updates available from tune result")

    try:
        _apply_tune_config(instance_name, config_updates)
        task_state["applied"] = True
        return {"applied": True, "config_updates": config_updates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _apply_tune_config(instance_name: str, config_updates: Dict[str, Any]):
    """Apply tuned config updates to a running or saved automation instance.

    Snapshots the current config before applying so we can revert later.
    """
    # Snapshot current config values (only the keys we're about to change)
    saved_configs = _load_saved_configs()
    current_cfg = saved_configs.get(instance_name, {})
    config_before = {k: current_cfg.get(k) for k in config_updates}

    # Update running instance if available
    automation = _get_automation_instance(instance_name)
    if automation:
        for key, value in config_updates.items():
            if hasattr(automation.config, key):
                setattr(automation.config, key, value)
        logger.info(f"Applied tune config to running instance '{instance_name}': {config_updates}")

    # Always update saved config
    if instance_name in saved_configs:
        saved_configs[instance_name].update(config_updates)
        with open(_AUTOMATION_CONFIGS_FILE, "w") as f:
            json.dump(saved_configs, f, indent=2)
        logger.info(f"Applied tune config to saved config '{instance_name}': {config_updates}")

    # Update the latest tune history record with apply info
    symbol, pipeline = _get_instance_symbol_pipeline(instance_name)
    history = _load_tune_history()
    key = _tune_history_key(symbol, pipeline) if symbol and pipeline else instance_name
    records = history.get(key, [])
    if records:
        latest = records[-1]
        latest["applied"] = True
        latest["applied_at"] = datetime.now().isoformat()
        latest["config_before_apply"] = config_before
        with open(_TUNE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)


@app.get("/api/automation/tune/{instance_name}/history")
async def get_tune_history(instance_name: str):
    """Get tune history for an instance (shared by symbol+pipeline)."""
    symbol, pipeline = _get_instance_symbol_pipeline(instance_name)
    history = _load_tune_history()
    key = _tune_history_key(symbol, pipeline) if symbol and pipeline else instance_name
    records = history.get(key, [])
    # Fallback: check old instance-keyed records and migrate
    if not records and instance_name in history:
        records = history[instance_name]
        if symbol and pipeline:
            history[key] = records
            del history[instance_name]
            with open(_TUNE_HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2, default=str)
    return {"instance": instance_name, "key": key, "symbol": symbol, "pipeline": pipeline, "records": records}


@app.post("/api/automation/tune/{instance_name}/revert/{record_index}")
async def revert_tune(instance_name: str, record_index: int):
    """Revert to the config that was in place before a specific tune was applied."""
    symbol, pipeline = _get_instance_symbol_pipeline(instance_name)
    history = _load_tune_history()
    key = _tune_history_key(symbol, pipeline) if symbol and pipeline else instance_name
    records = history.get(key, [])
    if record_index < 0 or record_index >= len(records):
        raise HTTPException(status_code=404, detail=f"Tune record #{record_index} not found")

    record = records[record_index]
    config_before = record.get("config_before_apply")
    if not config_before:
        raise HTTPException(status_code=400, detail="No pre-apply config snapshot available for this record")

    # Apply the old config back
    _apply_tune_config(instance_name, config_before)

    # Mark the record as reverted
    record["reverted"] = True
    record["reverted_at"] = datetime.now().isoformat()
    with open(_TUNE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)

    return {"reverted": True, "config_restored": config_before}


async def _broadcast_tune_progress(instance_name: str, progress: dict):
    """Broadcast tune progress via WebSocket."""
    message = json.dumps({
        "type": "tune_progress",
        "instance": instance_name,
        "progress": progress,
    })
    for ws in websocket_connections[:]:
        try:
            await ws.send_text(message)
        except Exception:
            pass


async def _broadcast_tune_complete(instance_name: str, task_state: dict):
    """Broadcast tune completion via WebSocket."""
    message = json.dumps({
        "type": "tune_complete",
        "instance": instance_name,
        "status": task_state.get("status"),
        "result": task_state.get("result"),
        "error": task_state.get("error"),
    }, default=str)
    for ws in websocket_connections[:]:
        try:
            await ws.send_text(message)
        except Exception:
            pass


@app.get("/api/automation/quant/history")
async def get_quant_automation_history(instance: str = Query(default="quant")):
    """Get recent analysis history for an instance from the signals table."""
    try:
        from tradingagents.storage.postgres_store import get_signal_store
        store = get_signal_store()
        signals = store.list_signals(source=instance, limit=100)

        return {
            "instance_name": instance,
            "analysis_results": [
                {
                    "timestamp": s["created_at"],
                    "symbol": s["symbol"],
                    "pipeline": s["pipeline"],
                    "signal": s["signal"],
                    "confidence": s["confidence"],
                    "entry_price": s["entry_price"],
                    "stop_loss": s["stop_loss"],
                    "take_profit": s["take_profit"],
                    "rationale": s["rationale"],
                    "executed": s["executed"],
                    "execution_ticket": s.get("execution_ticket") or s.get("decision_id"),
                    "execution_error": None,
                    "duration_seconds": s["analysis_duration_seconds"],
                }
                for s in signals
            ],
            "position_results": [],
        }
    except Exception as e:
        logger.warning(f"Failed to load signals for '{instance}': {e}")
        return {"instance_name": instance, "analysis_results": [], "position_results": []}


# ----- Trade Management Agent API -----


def _get_management_store():
    """Get management store singleton."""
    from tradingagents.storage.postgres_store import get_management_store
    return get_management_store()


@app.post("/api/trade-manager/start")
async def start_trade_manager(config: Dict[str, Any] = {}):
    """Start the Trade Management Agent.

    Sends a start command via Postgres for the home machine worker.
    """
    instance_name = config.get("instance_name", "trade_manager")

    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance_name)
        if remote_status and remote_status.get("status") == "running":
            from datetime import timezone
            updated_at = remote_status.get("updated_at")
            if updated_at:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() <= 60:
                    raise HTTPException(status_code=400, detail="Trade Manager is already running")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to check TMA remote status: {e}")

    try:
        logger.info(f"[TMA START] Step 1: update_status(pipeline='trade_management', config keys={list(config.keys())})")
        await control.update_status(
            instance_name=instance_name,
            status="pending_start",
            pipeline="trade_management",
            symbols=[],
            auto_execute=True,
            config=config,
        )
        await broadcast_automation_status(instance_name, "pending_start")

        # Ensure pipeline is always set so the worker routes to TMA, not quant
        config["pipeline"] = "trade_management"
        config["instance_name"] = instance_name
        logger.info(f"[TMA START] Step 2: config['pipeline']={config['pipeline']}, sending command")

        command_id = await control.send_command(
            instance_name=instance_name,
            action="start",
            payload=config,
        )
        logger.info(f"[TMA START] Step 3: command sent, id={command_id}, payload pipeline={config.get('pipeline')}")

        return {
            "status": "start_requested",
            "instance_name": instance_name,
            "command_id": command_id,
            "message": "Start command sent. Worker will pick it up shortly.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade-manager/stop")
async def stop_trade_manager(instance: str = Query(default="trade_manager")):
    """Stop the Trade Management Agent."""
    try:
        control = _get_automation_control()
        command_id = await control.send_command(
            instance_name=instance,
            action="stop",
        )
        await broadcast_automation_status(instance, "stopping")
        return {
            "status": "stop_requested",
            "instance_name": instance,
            "command_id": command_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/status")
async def get_trade_manager_status(instance: str = Query(default="trade_manager")):
    """Get Trade Management Agent status."""
    try:
        control = _get_automation_control()
        remote_status = await control.get_status(instance)
        if remote_status:
            updated_at = remote_status.get("updated_at")
            if updated_at:
                from datetime import timezone
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - last_update).total_seconds() > 60:
                    remote_status["status"] = "stopped"
                    remote_status["running"] = False
                    remote_status["stale"] = True
                else:
                    remote_status["running"] = remote_status.get("status") == "running"
            return remote_status
        return {"status": "not_found", "instance_name": instance, "running": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/actions")
async def get_trade_manager_actions(
    ticket: Optional[int] = Query(default=None),
    limit: int = Query(default=50, le=200),
):
    """Get management action history."""
    try:
        store = _get_management_store()
        actions = store.get_management_actions(ticket=ticket, limit=limit)
        return {"actions": actions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/config")
async def get_trade_manager_config(instance: str = Query(default="trade_manager")):
    """Get current TMA configuration."""
    try:
        control = _get_automation_control()
        status = await control.get_status(instance)
        if status and status.get("config"):
            return {"config": status["config"]}
        return {"config": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/trade-manager/config")
async def update_trade_manager_config(
    updates: Dict[str, Any],
    instance: str = Query(default="trade_manager"),
):
    """Update TMA configuration. Persists to DB and sends update command to running instance."""
    try:
        control = _get_automation_control()

        # Persist: merge updates into stored config
        current = await control.get_status(instance)
        config = current.get("config", {}) if current else {}
        if isinstance(config, str):
            config = json.loads(config)
        config.update(updates)

        await control.update_status(
            instance_name=instance,
            status=current.get("status", "stopped") if current else "stopped",
            config=config,
        )

        # Also send command so running instance picks it up
        command_id = None
        try:
            command_id = await control.send_command(
                instance_name=instance,
                action="update_config",
                payload=updates,
            )
        except Exception:
            pass

        return {"status": "config_saved", "command_id": command_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/policies")
async def get_trade_manager_policies():
    """Get all position management policies."""
    try:
        store = _get_management_store()
        policies = store.load_all_management_policies()
        return {"policies": {str(k): v for k, v in policies.items()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/policies/{ticket}")
async def get_trade_manager_policy(ticket: int):
    """Get policy for a specific position."""
    try:
        store = _get_management_store()
        policy = store.load_management_policy(ticket)
        if policy is None:
            raise HTTPException(status_code=404, detail=f"No policy for ticket {ticket}")
        return {"ticket": ticket, "policy": policy}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/trade-manager/policies/{ticket}")
async def update_trade_manager_policy(ticket: int, policy: Dict[str, Any]):
    """Create or update a position management policy."""
    try:
        symbol = policy.get("symbol", "")
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol is required in policy")

        store = _get_management_store()
        store.save_management_policy(ticket, symbol, policy)
        return {"status": "saved", "ticket": ticket}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/trade-manager/policies/{ticket}")
async def delete_trade_manager_policy(ticket: int):
    """Delete a position management policy."""
    try:
        store = _get_management_store()
        store.delete_management_policy(ticket)
        return {"status": "deleted", "ticket": ticket}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-manager/alerts")
async def get_trade_manager_alerts(
    limit: int = Query(default=20, le=100),
    unacknowledged: bool = Query(default=False),
):
    """Get risk alerts."""
    try:
        store = _get_management_store()
        alerts = store.get_risk_alerts(limit=limit, unacknowledged_only=unacknowledged)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade-manager/alerts/{alert_id}/acknowledge")
async def acknowledge_trade_manager_alert(alert_id: int):
    """Acknowledge a risk alert."""
    try:
        store = _get_management_store()
        store.acknowledge_risk_alert(alert_id)
        return {"status": "acknowledged", "alert_id": alert_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----- Health Check -----

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ----- Command Queue API -----
# Multi-worker command queue for distributed processing

class PublishCommandRequest(BaseModel):
    command_type: str
    payload: Dict[str, Any] = {}
    priority: int = 1
    target_worker: Optional[str] = None
    expires_in_seconds: Optional[int] = None


def _get_command_queue():
    """Get command queue singleton."""
    from tradingagents.storage.command_queue import get_command_queue
    return get_command_queue()


@app.post("/api/queue/publish")
async def publish_command(request: PublishCommandRequest):
    """Publish a command to the queue for workers to process."""
    try:
        queue = _get_command_queue()
        command_id = await queue.publish(
            command_type=request.command_type,
            payload=request.payload,
            priority=request.priority,
            source="api",
            target_worker=request.target_worker,
            expires_in_seconds=request.expires_in_seconds,
        )
        return {"command_id": command_id, "status": "published"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queue/command/{command_id}")
async def get_command(command_id: str):
    """Get status of a specific command."""
    try:
        queue = _get_command_queue()
        command = await queue.get_command(command_id)
        if not command:
            raise HTTPException(status_code=404, detail="Command not found")
        return {
            "id": command.id,
            "command_type": command.command_type,
            "payload": command.payload,
            "status": command.status,
            "priority": command.priority,
            "claimed_by": command.claimed_by,
            "claimed_at": command.claimed_at.isoformat() if command.claimed_at else None,
            "completed_at": command.completed_at.isoformat() if command.completed_at else None,
            "result": command.result,
            "error": command.error,
            "created_at": command.created_at.isoformat() if command.created_at else None,
            "retry_count": command.retry_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queue/commands")
async def list_commands(
    status: Optional[str] = None,
    command_type: Optional[str] = None,
    limit: int = 50,
):
    """List commands with optional filters."""
    try:
        queue = _get_command_queue()
        commands = await queue.list_commands(status=status, command_type=command_type, limit=limit)
        return {
            "commands": [
                {
                    "id": c.id,
                    "command_type": c.command_type,
                    "status": c.status,
                    "priority": c.priority,
                    "claimed_by": c.claimed_by,
                    "error": c.error,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in commands
            ],
            "count": len(commands),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queue/stats")
async def queue_stats():
    """Get queue statistics."""
    try:
        queue = _get_command_queue()
        stats = await queue.get_queue_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/queue/workers")
async def list_workers(active_only: bool = True):
    """List registered workers."""
    try:
        queue = _get_command_queue()
        workers = await queue.list_workers(active_only=active_only)
        return {
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "hostname": w.hostname,
                    "status": w.status,
                    "capabilities": w.capabilities,
                    "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
                    "current_command": w.current_command,
                    "commands_processed": w.commands_processed,
                    "started_at": w.started_at.isoformat() if w.started_at else None,
                }
                for w in workers
            ],
            "count": len(workers),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/cleanup")
async def cleanup_queue(days: int = 7):
    """Clean up old completed/failed commands."""
    try:
        queue = _get_command_queue()
        deleted = await queue.cleanup_old_commands(days=days)
        return {"deleted": deleted, "message": f"Deleted {deleted} old commands"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Convenience endpoints for common commands

@app.post("/api/queue/start-automation")
async def queue_start_automation(
    instance_name: str,
    config: Dict[str, Any] = {},
):
    """Queue a start automation command."""
    try:
        queue = _get_command_queue()
        command_id = await queue.publish(
            command_type="start_automation",
            payload={"instance_name": instance_name, "config": config},
            priority=2,  # High priority
            source="api",
        )
        return {"command_id": command_id, "status": "queued", "instance_name": instance_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/stop-automation")
async def queue_stop_automation(instance_name: str):
    """Queue a stop automation command."""
    try:
        queue = _get_command_queue()
        command_id = await queue.publish(
            command_type="stop_automation",
            payload={"instance_name": instance_name},
            priority=3,  # Urgent
            source="api",
        )
        return {"command_id": command_id, "status": "queued", "instance_name": instance_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/execute-trade")
async def queue_execute_trade(
    symbol: str,
    direction: str,
    volume: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    entry_price: Optional[float] = None,
):
    """Queue a trade execution command."""
    try:
        queue = _get_command_queue()
        command_id = await queue.publish(
            command_type="execute_trade",
            payload={
                "symbol": symbol,
                "direction": direction.upper(),
                "volume": volume,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_price": entry_price,
            },
            priority=3,  # Urgent
            source="api",
        )
        return {"command_id": command_id, "status": "queued", "symbol": symbol, "direction": direction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/queue/close-position")
async def queue_close_position(ticket: int, volume: Optional[float] = None):
    """Queue a position close command."""
    try:
        queue = _get_command_queue()
        command_id = await queue.publish(
            command_type="close_position",
            payload={"ticket": ticket, "volume": volume},
            priority=3,  # Urgent
            source="api",
        )
        return {"command_id": command_id, "status": "queued", "ticket": ticket}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
