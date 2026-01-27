"""
FastAPI Backend for TradingAgents Web UI
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents import trade_decisions
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

# Global state for WebSocket connections
websocket_connections: List[WebSocket] = []
analysis_tasks: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("TradingAgents API starting...")
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


class PositionModifyRequest(BaseModel):
    ticket: int
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None


class PositionCloseRequest(BaseModel):
    ticket: int


class PortfolioConfigUpdate(BaseModel):
    execution_mode: Optional[str] = None
    symbols: Optional[List[dict]] = None
    schedule: Optional[dict] = None


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
    mt5_status = get_mt5_status()

    # Check portfolio automation status
    automation_status = {"running": False}
    pid_file = Path("portfolio_scheduler.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                automation_status = {"running": True, "pid": pid}
        except:
            pass

    # Get decision store stats
    active_decisions = trade_decisions.list_active_decisions()
    closed_decisions = trade_decisions.list_closed_decisions(limit=100)

    return {
        "mt5": mt5_status,
        "automation": automation_status,
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
    """Get all open positions"""
    positions = get_open_positions()
    return {"positions": positions, "count": len(positions)}


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

        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": request.ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(modify_request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Modify failed: {result.comment}")

        # Log the modification for reflection/learning
        changes = []
        if old_sl != new_sl:
            changes.append(f"SL: {old_sl} → {new_sl}")
        if old_tp != new_tp:
            changes.append(f"TP: {old_tp} → {new_tp}")

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
    """Review a single position with LLM analysis and get SL/TP suggestions"""
    import traceback
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        pos = position[0]

        # Get current price
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            raise HTTPException(status_code=500, detail=f"Could not get tick data for {pos.symbol}")
        current_price = tick.bid if pos.type == 1 else tick.ask

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
            from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

            try:
                client, model, uses_responses = get_llm_client()
            except ValueError as e:
                result["llm_analysis"] = {"error": str(e)}
                return result

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

            prompt = f"""You are reviewing an OPEN position. Provide a fresh market assessment based on current structure.

POSITION: {pos.symbol} {direction}
Entry: {entry}, Current: {current_price}
P/L: {pnl_pct:+.2f}% (${pos.profit:.2f})
Current SL: {sl if sl > 0 else 'None'}, Current TP: {tp if tp > 0 else 'None'}
Current R:R: {risk_reward:.2f}:1
Volume: {pos.volume} lots
{smc_context}

TASK: Assess whether current market structure still supports this {direction} position.

Respond with ONLY valid JSON:
{{
  "recommendation": "HOLD" | "CLOSE" | "ADJUST",
  "suggested_sl": <number based on SMC levels or null>,
  "suggested_tp": <number based on SMC levels or null>,
  "risk_level": "Low" | "Medium" | "High",
  "reasoning": "<2-3 sentences explaining if market structure supports the position, key levels to watch, and any shift in bias>"
}}

Guidelines:
- If bias has shifted AGAINST position direction, recommend CLOSE or tight SL
- Place SL below/above nearest SMC support/resistance (Order Blocks, FVGs)
- Set TP at next SMC resistance/support level
- Consider if price is approaching liquidity zones or structure breaks
- Set to null ONLY if current levels align well with SMC structure"""

            try:
                analysis_content = chat_completion(
                    client=client,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert trade manager. Always respond with valid JSON only, no markdown or extra text."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                    use_responses_api=uses_responses,
                )
                if not analysis_content:
                    analysis_content = "{}"

                # Parse the JSON response
                import json
                try:
                    # Clean up response - remove markdown code blocks if present
                    clean_content = analysis_content.strip()
                    if clean_content.startswith("```"):
                        clean_content = clean_content.split("```")[1]
                        if clean_content.startswith("json"):
                            clean_content = clean_content[4:]
                        clean_content = clean_content.strip()

                    parsed = json.loads(clean_content)
                    result["llm_analysis"] = {
                        "recommendation": parsed.get("recommendation", "HOLD"),
                        "suggested_sl": parsed.get("suggested_sl"),
                        "suggested_tp": parsed.get("suggested_tp"),
                        "risk_level": parsed.get("risk_level", "Medium"),
                        "reasoning": parsed.get("reasoning", ""),
                        "model": model,
                        "raw": analysis_content
                    }
                except json.JSONDecodeError:
                    # Fallback to raw text if JSON parsing fails
                    result["llm_analysis"] = {
                        "analysis": analysis_content,
                        "model": model,
                        "parse_error": True
                    }
            except Exception as e:
                import traceback as tb
                result["llm_analysis"] = {"error": str(e), "trace": tb.format_exc()}

        return result
    except HTTPException:
        raise
    except Exception as e:
        import sys
        error_trace = traceback.format_exc()
        print(f"Review error: {error_trace}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{error_trace}")


@app.post("/api/positions/batch-review")
async def batch_review_positions(request: BatchReviewRequest):
    """Get review analysis for multiple positions"""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")

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
            return {"positions": [], "count": 0}

        from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol

        results = []
        for pos in positions:
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

        return {"positions": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

            # Extract trading plans
            trader_plan = final_state.get("trader_investment_plan", "")
            risk_decision = final_state.get("final_trade_decision", "")

            # Parse recommendation from the decision
            # Primary signal: Trading plan (BUY/SELL) vs position direction
            # Secondary: SMC data (structure shifts, bias alignment)
            recommendation = "HOLD"  # Default
            suggested_sl = None
            suggested_tp = None
            close_reason = None

            # FIRST: Check if trading signal conflicts with position direction
            # This is the most important check - if plan says SELL and we're in a BUY, close it
            trading_signal = decision.get('signal', 'HOLD').upper()

            if trading_signal == "SELL" and direction == "BUY":
                recommendation = "CLOSE"
                close_reason = f"Trading plan recommends SELL - conflicts with current BUY position"
            elif trading_signal == "BUY" and direction == "SELL":
                recommendation = "CLOSE"
                close_reason = f"Trading plan recommends BUY - conflicts with current SELL position"
            else:
                # Signal aligns with position or is HOLD - use SMC data for finer adjustments
                bias_aligns = smc_review.get('bias_aligns', True)
                structure_shift = smc_review.get('structure_shift', False)
                sl_at_risk = smc_review.get('sl_at_risk', False)

                if structure_shift and not bias_aligns:
                    # Clear bearish signal against position - recommend CLOSE
                    recommendation = "CLOSE"
                    close_reason = "Structure shift (CHOCH) detected against position with misaligned bias"
                elif structure_shift:
                    # Structure shift but bias might still be ok - recommend ADJUST
                    recommendation = "ADJUST"
                    close_reason = "Structure shift detected - consider tightening stops"
                elif not bias_aligns:
                    # Bias against position but no structure shift yet - ADJUST
                    recommendation = "ADJUST"
                    close_reason = "Market bias against position direction"
                elif sl_at_risk and smc_review.get('trailing_sl'):
                    # SL is weak but we have a better level - ADJUST
                    recommendation = "ADJUST"

            # Only override with text parsing if we find VERY explicit close signals
            # Look for patterns like "CLOSE IMMEDIATELY" or "EXIT NOW" not just the words
            combined_text = f"{risk_decision} {trader_plan}".upper()
            close_patterns = [
                "CLOSE IMMEDIATELY", "CLOSE NOW", "EXIT IMMEDIATELY", "EXIT NOW",
                "RECOMMEND CLOSING", "SHOULD CLOSE", "MUST CLOSE",
                "CLOSE THIS POSITION", "EXIT THIS POSITION"
            ]
            if any(pattern in combined_text for pattern in close_patterns):
                recommendation = "CLOSE"
                if not close_reason:
                    close_reason = "Agent recommends closing based on analysis"

            # Use SMC-suggested values if available
            if smc_review.get('suggested_sl'):
                suggested_sl = smc_review['suggested_sl']
            if smc_review.get('suggested_tp'):
                suggested_tp = smc_review['suggested_tp']

            # If decision has specific values, use those
            if decision.get('stop_loss'):
                suggested_sl = decision['stop_loss']
            if decision.get('take_profit'):
                suggested_tp = decision['take_profit']

            analysis_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "current_step": "complete",
                "current_step_title": "Position Analysis Complete",
                "decision": decision,
                "position_review": {
                    "recommendation": recommendation,
                    "trading_signal": trading_signal,  # What the plan recommends (BUY/SELL/HOLD)
                    "suggested_sl": suggested_sl,
                    "suggested_tp": suggested_tp,
                    "suggested_trailing_sl": smc_review.get('trailing_sl'),
                    "trailing_sl_source": smc_review.get('trailing_sl_source'),
                    "close_reason": close_reason,
                    "bias": smc_review.get('bias'),
                    "bias_aligns": smc_review.get('bias_aligns'),
                    "structure_shift": smc_review.get('structure_shift'),
                    "sl_at_risk": smc_review.get('sl_at_risk'),
                    "sl_risk_reason": smc_review.get('sl_risk_reason'),
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

        # Calculate risk/reward if we had a standard 2:1 ratio
        potential_reward = sl_distance * 2
        potential_profit = (potential_reward / tick_size) * tick_value * lots if tick_size > 0 else 0

        return {
            "position_size": round(lots, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": round((risk_amount / account.balance) * 100, 2),
            "sl_distance": round(sl_distance, symbol_info.digits),
            "sl_ticks": round(sl_distance / tick_size) if tick_size > 0 else 0,
            "potential_loss": round(risk_amount, 2),
            "potential_profit_2r": round(potential_profit, 2),
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
        for s in symbols:
            if s.visible:  # visible=True means it's in Market Watch
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
    else:
        # Get all
        active = trade_decisions.list_active_decisions(symbol=symbol)
        closed = trade_decisions.list_closed_decisions(symbol=symbol, limit=limit)
        decisions = active + closed
        decisions = sorted(decisions, key=lambda d: d.get("created_at", ""), reverse=True)[:limit]

    # Transform to match frontend expected format
    formatted = []
    for d in decisions:
        formatted.append({
            "id": d.get("decision_id"),
            "symbol": d.get("symbol"),
            "signal": d.get("action"),
            "confidence": d.get("confluence_score", 5) / 10 if d.get("confluence_score") else 0.7,
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
        "confidence": decision.get("confluence_score", 5) / 10 if decision.get("confluence_score") else 0.7,
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


@app.get("/api/decisions/stats")
async def get_decision_stats():
    """Get decision statistics"""
    try:
        all_decisions = trade_decisions.list_decisions(limit=1000)

        # Calculate stats
        total = len(all_decisions)
        closed = [d for d in all_decisions if d.get("status") == "closed" or d.get("outcome")]
        open_decisions = [d for d in all_decisions if d.get("status") == "active" or not d.get("outcome")]

        wins = sum(1 for d in closed if d.get("was_correct") or (d.get("outcome") and d.get("outcome", {}).get("was_correct")))
        losses = len(closed) - wins

        # Calculate by symbol
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
            "by_symbol": by_symbol
        }
    except Exception as e:
        return {"error": str(e)}


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
                progress_callback=progress_callback
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
            for ob in (order_blocks.get("bullish") or []):
                if ob and isinstance(ob, dict):
                    smc_levels.append({
                        "type": "order_block",
                        "price": (ob.get("top", 0) + ob.get("bottom", 0)) / 2,
                        "direction": "bullish",
                        "strength": ob.get("strength", 0.5),
                        "description": f"Bullish OB at {ob.get('bottom', 0):.5f}-{ob.get('top', 0):.5f}"
                    })
            for ob in (order_blocks.get("bearish") or []):
                if ob and isinstance(ob, dict):
                    smc_levels.append({
                        "type": "order_block",
                        "price": (ob.get("top", 0) + ob.get("bottom", 0)) / 2,
                        "direction": "bearish",
                        "strength": ob.get("strength", 0.5),
                        "description": f"Bearish OB at {ob.get('bottom', 0):.5f}-{ob.get('top', 0):.5f}"
                    })

            # Safely extract FVGs (handle None values)
            fair_value_gaps = smc_data.get("fair_value_gaps") if isinstance(smc_data, dict) else {}
            fair_value_gaps = fair_value_gaps or {}
            for fvg in (fair_value_gaps.get("bullish") or []):
                if fvg and isinstance(fvg, dict):
                    smc_levels.append({
                        "type": "fvg",
                        "price": (fvg.get("top", 0) + fvg.get("bottom", 0)) / 2,
                        "direction": "bullish",
                        "strength": 0.7,
                        "description": f"Bullish FVG at {fvg.get('bottom', 0):.5f}-{fvg.get('top', 0):.5f}"
                    })
            for fvg in (fair_value_gaps.get("bearish") or []):
                if fvg and isinstance(fvg, dict):
                    smc_levels.append({
                        "type": "fvg",
                        "price": (fvg.get("top", 0) + fvg.get("bottom", 0)) / 2,
                        "direction": "bearish",
                        "strength": 0.7,
                        "description": f"Bearish FVG at {fvg.get('bottom', 0):.5f}-{fvg.get('top', 0):.5f}"
                    })

            # Safely extract liquidity zones (handle None values)
            liquidity_zones = smc_data.get("liquidity_zones") if isinstance(smc_data, dict) else []
            for zone in (liquidity_zones or []):
                if zone and isinstance(zone, dict):
                    smc_levels.append({
                        "type": "liquidity",
                        "price": zone.get("price", 0),
                        "direction": "bullish" if zone.get("type") == "support" else "bearish",
                        "strength": zone.get("strength", 0.5),
                        "description": f"Liquidity zone at {zone.get('price', 0):.5f}"
                    })

            # Add nearest support/resistance as pullback entry levels
            if smc_raw_data and isinstance(smc_raw_data, dict):
                # Get levels from primary timeframe (1H typically)
                for tf_name, tf_data in smc_raw_data.items():
                    if not isinstance(tf_data, dict):
                        continue

                    # Add nearest support (good for BUY pullback)
                    nearest_support = tf_data.get("nearest_support")
                    if nearest_support and isinstance(nearest_support, dict):
                        support_price = nearest_support.get("top", nearest_support.get("bottom", 0))
                        if support_price > 0:
                            smc_levels.append({
                                "type": "support",
                                "price": support_price,
                                "direction": "bullish",
                                "strength": nearest_support.get("strength", 0.7),
                                "description": f"Nearest Support ({tf_name}): {support_price:.5f}"
                            })

                    # Add nearest resistance (good for SELL pullback)
                    nearest_resistance = tf_data.get("nearest_resistance")
                    if nearest_resistance and isinstance(nearest_resistance, dict):
                        resistance_price = nearest_resistance.get("bottom", nearest_resistance.get("top", 0))
                        if resistance_price > 0:
                            smc_levels.append({
                                "type": "resistance",
                                "price": resistance_price,
                                "direction": "bearish",
                                "strength": nearest_resistance.get("strength", 0.7),
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
                                if nearest_support and nearest_support.get("bottom"):
                                    decision["stop_loss"] = nearest_support["bottom"] * 0.998
                                if nearest_resistance and nearest_resistance.get("bottom"):
                                    decision["take_profit"] = nearest_resistance["bottom"]
                            elif decision.get("signal") == "SELL":
                                # SL above resistance, TP at support
                                if nearest_resistance and nearest_resistance.get("top"):
                                    decision["stop_loss"] = nearest_resistance["top"] * 1.002
                                if nearest_support and nearest_support.get("top"):
                                    decision["take_profit"] = nearest_support["top"]

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

            analysis_tasks[task_id].update({
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
            })

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
            "reason": status.get("block_reason", None),
            "daily_loss_used": status.get("daily_loss_used", 0),
            "daily_loss_limit": status.get("daily_loss_limit", 5),
            "consecutive_losses": status.get("consecutive_losses", 0),
            "max_consecutive_losses": status.get("max_consecutive_losses", 3),
            "in_cooldown": status.get("in_cooldown", False),
            "cooldown_until": status.get("cooldown_until"),
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
    """Reset circuit breaker (use with caution)"""
    try:
        from tradingagents.risk.guardrails import RiskGuardrails

        guardrails = RiskGuardrails()

        if hasattr(guardrails, 'reset'):
            guardrails.reset()
        elif hasattr(guardrails, 'reset_cooldown'):
            guardrails.reset_cooldown()

        return {"success": True, "message": "Circuit breaker reset"}
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


@app.post("/api/memory/reflect")
async def trigger_reflection():
    """
    Manually trigger reflection on closed trades.

    This runs the evening reflection cycle which:
    - Finds closed trades that haven't been reflected on
    - Analyzes the outcomes and generates learnings
    - Stores reflections in memory for future reference
    """
    try:
        from tradingagents.automation.portfolio_automation import PortfolioAutomation

        automation = PortfolioAutomation()
        # run_evening_reflect is an async method, await it directly
        result = await automation.run_evening_reflect()

        # Convert dataclass to dict for JSON response
        return {
            "success": True,
            "trades_processed": result.trades_processed,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "total_pnl": result.total_pnl,
            "reflections_created": result.reflections_created,
            "memories_stored": result.memories_stored,
            "errors": result.errors if result.errors else {},
            "duration_seconds": result.total_duration_seconds
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ----- Portfolio Automation -----

@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get portfolio automation status"""
    status = {"running": False}

    pid_file = Path("portfolio_scheduler.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            import psutil
            if psutil.pid_exists(pid):
                status = {"running": True, "pid": pid}
        except:
            pass

    # Load state file if exists
    state_file = Path("scheduler_state.json")
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            status["last_run"] = state.get("last_run")
            status["next_run"] = state.get("next_run")
        except:
            pass

    return status


@app.get("/api/portfolio/config")
async def get_portfolio_config():
    """Get portfolio configuration"""
    config_file = Path("portfolio_config.yaml")
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

    config_file = Path("portfolio_config.yaml")
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

    config_file = Path("portfolio_config.yaml")
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

    config_file = Path("portfolio_config.yaml")
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


@app.get("/api/portfolio/config/suggestions")
async def get_portfolio_suggestions():
    """Get LLM-powered suggestions for portfolio balance"""
    import yaml
    from tradingagents.dataflows.llm_client import get_llm_client, chat_completion

    config_file = Path("portfolio_config.yaml")
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

        # Start the daemon
        process = subprocess.Popen(
            [sys.executable, "-m", "cli.main", "portfolio", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).parent.parent.parent)
        )

        return {"success": True, "message": "Portfolio automation started"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/stop")
async def stop_portfolio_automation():
    """Stop portfolio automation"""
    pid_file = Path("portfolio_scheduler.pid")
    if not pid_file.exists():
        return {"error": "Not running"}

    try:
        pid = int(pid_file.read_text().strip())
        import psutil

        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=5)

        pid_file.unlink()
        return {"success": True, "message": "Portfolio automation stopped"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/portfolio/trigger")
async def trigger_daily_cycle(cycle_type: str = "morning"):
    """Manually trigger a daily cycle"""
    try:
        from tradingagents.automation.portfolio_automation import PortfolioAutomation

        automation = PortfolioAutomation()

        if cycle_type == "morning":
            result = await asyncio.to_thread(automation.run_morning_analysis)
        elif cycle_type == "midday":
            result = await asyncio.to_thread(automation.run_midday_review)
        elif cycle_type == "evening":
            result = await asyncio.to_thread(automation.run_evening_reflect)
        else:
            return {"error": f"Unknown cycle type: {cycle_type}"}

        return {"success": True, "result": result}
    except Exception as e:
        return {"error": str(e)}


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

        # Use configurable thresholds
        analyzer = SmartMoneyAnalyzer(fvg_min_size_atr=fvg_min_size)

        # Run analysis with custom lookback
        order_blocks_raw = analyzer.detect_order_blocks(df, lookback=lookback)
        fvgs_raw = analyzer.detect_fair_value_gaps(df, lookback=lookback)
        swing_points = analyzer.detect_swing_points(df, lookback=lookback)
        structure_breaks = analyzer.detect_structure_breaks(df, swing_points)
        current_price = df.iloc[-1]['close']
        zones = analyzer.get_unmitigated_zones(order_blocks_raw, fvgs_raw, current_price)

        result = {
            'current_price': current_price,
            'order_blocks': {
                'total': len(order_blocks_raw),
                'unmitigated': len([ob for ob in order_blocks_raw if not ob.mitigated]),
                'bullish': [ob for ob in order_blocks_raw if ob.type == 'bullish'],
                'bearish': [ob for ob in order_blocks_raw if ob.type == 'bearish']
            },
            'fair_value_gaps': {
                'total': len(fvgs_raw),
                'unmitigated': len([fvg for fvg in fvgs_raw if not fvg.mitigated]),
                'bullish': [fvg for fvg in fvgs_raw if fvg.type == 'bullish'],
                'bearish': [fvg for fvg in fvgs_raw if fvg.type == 'bearish']
            },
            'structure': {
                'swing_points': len(swing_points),
                'bos_count': len(structure_breaks['bos']),
                'choc_count': len(structure_breaks['choc']),
                'recent_bos': [sp for sp in structure_breaks['bos'] if sp.break_index and sp.break_index >= len(df) - 20],
                'recent_choc': [sp for sp in structure_breaks['choc'] if sp.break_index and sp.break_index >= len(df) - 20]
            },
            'zones': zones,
            'nearest_support': zones['support'][0] if zones['support'] else None,
            'nearest_resistance': zones['resistance'][0] if zones['resistance'] else None,
            'bias': analyzer._determine_bias(
                structure_breaks.get('recent_bos', []),
                structure_breaks.get('recent_choc', []),
                zones,
                current_price
            )
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

        # Extract unmitigated zones as liquidity zones
        zones = result.get("zones", {})
        liquidity_zones = []
        for z in zones.get("support", []) + zones.get("resistance", []):
            liquidity_zones.append({
                "type": z.get("type", "zone"),
                "price": (z.get("top", 0) + z.get("bottom", 0)) / 2,
                "top": z.get("top"),
                "bottom": z.get("bottom"),
                "strength": z.get("strength", 0)
            })

        # Build key levels from nearest support/resistance and unmitigated zones
        current_price = result.get("current_price", df.iloc[-1]['close'])
        key_levels = []

        # Add nearest support
        nearest_support = result.get("nearest_support")
        if nearest_support:
            source_type = nearest_support.get("type", "zone")
            source_label = "FVG" if source_type == "fvg" else "OB" if source_type == "order_block" else source_type.upper()
            key_levels.append({
                "type": "Support Zone",
                "price": round((nearest_support.get("top", 0) + nearest_support.get("bottom", 0)) / 2, 2),
                "source": source_label,
                "direction": "bullish"
            })

        # Add nearest resistance
        nearest_resistance = result.get("nearest_resistance")
        if nearest_resistance:
            source_type = nearest_resistance.get("type", "zone")
            source_label = "FVG" if source_type == "fvg" else "OB" if source_type == "order_block" else source_type.upper()
            key_levels.append({
                "type": "Resistance Zone",
                "price": round((nearest_resistance.get("top", 0) + nearest_resistance.get("bottom", 0)) / 2, 2),
                "source": source_label,
                "direction": "bearish"
            })

        # Add unmitigated OBs as key levels
        for ob in order_blocks:
            if not ob.get("mitigated", True):
                ob_type = ob.get("type", "")
                is_bullish = ob_type == "bullish" or str(ob_type).lower().startswith("bull")
                level_type = "Demand Zone" if is_bullish else "Supply Zone"
                key_levels.append({
                    "type": level_type,
                    "price": round((ob["top"] + ob["bottom"]) / 2, 2),
                    "source": "OB",
                    "direction": "bullish" if is_bullish else "bearish"
                })

        # Add unmitigated FVGs as key levels
        for fvg in fair_value_gaps:
            if not fvg.get("mitigated", True):
                fvg_type = fvg.get("type", "")
                is_bullish = fvg_type == "bullish" or str(fvg_type).lower().startswith("bull")
                level_type = "Bullish FVG" if is_bullish else "Bearish FVG"
                key_levels.append({
                    "type": level_type,
                    "price": round((fvg["top"] + fvg["bottom"]) / 2, 2),
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
            support_dist = current_price - nearest_support.get("bottom", 0)
            resist_dist = nearest_resistance.get("top", 0) - current_price
            if support_dist < resist_dist:
                bias_factors.append(f"Price is closer to support ({support_dist:.2f} pts) than resistance ({resist_dist:.2f} pts)")
            else:
                bias_factors.append(f"Price is closer to resistance ({resist_dist:.2f} pts) than support ({support_dist:.2f} pts)")

        if not bias_factors:
            bias_factors.append("No strong directional signals detected - market structure is neutral")

        bias_summary = f"The market bias is **{bias.upper()}** based on Smart Money Concepts analysis."
        bias_explanation = " ".join([f"• {factor}" for factor in bias_factors])

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "liquidity_zones": liquidity_zones,
            "bias": bias,
            "bias_summary": bias_summary,
            "bias_factors": bias_factors,
            "key_levels": key_levels,
            "current_price": round(current_price, 2),
            "summary": {
                "total_obs": ob_data.get("total", 0),
                "unmitigated_obs": ob_data.get("unmitigated", 0),
                "total_fvgs": fvg_data.get("total", 0),
                "unmitigated_fvgs": fvg_data.get("unmitigated", 0),
                "bullish_fvgs": bullish_fvgs,
                "bearish_fvgs": bearish_fvgs,
                "bullish_obs": bullish_obs,
                "bearish_obs": bearish_obs
            }
        }

        # Add debug info if requested
        if debug:
            response["debug"] = {
                "lookback_bars": lookback,
                "fvg_min_size_atr": fvg_min_size,
                "all_fvgs_detected": len(fair_value_gaps),
                "mitigated_fvgs": len([f for f in fair_value_gaps if f.get("mitigated")]),
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
        websocket_connections.remove(websocket)


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


# ----- Health Check -----

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
