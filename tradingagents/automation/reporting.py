"""
Reporting System

Dataclasses for daily analysis, position review, and reflection reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class AnalysisResult:
    """Result of analyzing a single symbol."""
    symbol: str
    signal: str                             # BUY, SELL, HOLD
    confidence: float                       # 0.0 - 1.0
    trade_date: str

    # Analysis context
    final_state: Optional[Dict[str, Any]] = None
    smc_analysis: Optional[Dict[str, Any]] = None

    # Recommended trade parameters
    recommended_size: float = 0.0           # Lots
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Reasoning
    rationale: str = ""
    key_factors: List[str] = field(default_factory=list)

    # Timing
    analysis_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes large final_state)."""
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "trade_date": self.trade_date,
            "recommended_size": self.recommended_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "rationale": self.rationale,
            "key_factors": self.key_factors,
            "analysis_duration_seconds": self.analysis_duration_seconds,
        }


@dataclass
class PositionAdjustment:
    """A position adjustment recommendation or action."""
    ticket: int
    symbol: str
    adjustment_type: str                    # trailing_stop, breakeven, tp_update, close
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    reason: str = ""

    # Execution status
    applied: bool = False
    pending_approval: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "adjustment_type": self.adjustment_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "applied": self.applied,
            "pending_approval": self.pending_approval,
            "error": self.error,
        }


@dataclass
class TradeExecution:
    """Record of a trade execution."""
    symbol: str
    signal: str                             # BUY or SELL
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float

    # MT5 result
    success: bool
    ticket: Optional[int] = None
    error: Optional[str] = None

    # Decision tracking
    decision_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "volume": self.volume,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "success": self.success,
            "ticket": self.ticket,
            "error": self.error,
            "decision_id": self.decision_id,
        }


@dataclass
class ClosedTrade:
    """Record of a closed trade."""
    decision_id: str
    symbol: str
    signal: str                             # Original BUY or SELL
    entry_price: float
    exit_price: float
    volume: float
    pnl: float                              # Profit/loss in account currency
    pnl_percent: float                      # Percentage return
    was_profitable: bool
    exit_reason: str                        # tp_hit, sl_hit, manual, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "signal": self.signal,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "volume": self.volume,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "was_profitable": self.was_profitable,
            "exit_reason": self.exit_reason,
        }


@dataclass
class DailyAnalysisReport:
    """Morning analysis cycle report."""
    timestamp: datetime
    cycle_name: str = "morning_analysis"

    # Status
    blocked: bool = False
    blocked_reason: str = ""

    # Portfolio state
    current_positions: List[Dict] = field(default_factory=list)
    account_balance: float = 0.0
    account_equity: float = 0.0

    # Analysis results
    analysis_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    opportunities: List[AnalysisResult] = field(default_factory=list)

    # Skipped symbols and reasons
    skipped_symbols: List[tuple] = field(default_factory=list)

    # Errors during analysis
    errors: Dict[str, str] = field(default_factory=dict)

    # Trades executed (FULL_AUTO mode)
    trades_executed: List[TradeExecution] = field(default_factory=list)

    # Timing
    total_duration_seconds: float = 0.0

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines = [
            "=" * 50,
            "MORNING ANALYSIS REPORT",
            "=" * 50,
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if self.blocked:
            lines.append(f"[BLOCKED] {self.blocked_reason}")
            lines.append("")
            return "\n".join(lines)

        # Portfolio status
        lines.append("Portfolio Status:")
        lines.append(f"  Balance: ${self.account_balance:,.2f}")
        lines.append(f"  Equity: ${self.account_equity:,.2f}")
        lines.append(f"  Open Positions: {len(self.current_positions)}")
        lines.append("")

        # Analysis summary
        lines.append("Analysis Summary:")
        lines.append(f"  Symbols Analyzed: {len(self.analysis_results)}")
        lines.append(f"  Opportunities Found: {len(self.opportunities)}")
        lines.append(f"  Skipped: {len(self.skipped_symbols)}")
        lines.append(f"  Errors: {len(self.errors)}")
        lines.append("")

        # Opportunities
        if self.opportunities:
            lines.append("Opportunities (ranked by confidence):")
            for opp in self.opportunities[:5]:
                lines.append(
                    f"  {opp.symbol}: {opp.signal} "
                    f"(confidence: {opp.confidence:.2f}, "
                    f"size: {opp.recommended_size:.2f} lots)"
                )
            lines.append("")

        # Trades executed
        if self.trades_executed:
            lines.append("Trades Executed:")
            for trade in self.trades_executed:
                status = "OK" if trade.success else f"FAILED: {trade.error}"
                lines.append(
                    f"  {trade.symbol} {trade.signal} "
                    f"{trade.volume} lots @ {trade.entry_price:.2f} - {status}"
                )
            lines.append("")

        # Skipped symbols
        if self.skipped_symbols:
            lines.append("Skipped Symbols:")
            for symbol, reason in self.skipped_symbols:
                lines.append(f"  {symbol}: {reason}")
            lines.append("")

        # Errors
        if self.errors:
            lines.append("Errors:")
            for symbol, error in self.errors.items():
                lines.append(f"  {symbol}: {error}")
            lines.append("")

        lines.append(f"Duration: {self.total_duration_seconds:.1f}s")
        lines.append("=" * 50)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle_name": self.cycle_name,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "current_positions_count": len(self.current_positions),
            "account_balance": self.account_balance,
            "account_equity": self.account_equity,
            "analysis_results": {
                k: v.to_dict() for k, v in self.analysis_results.items()
            },
            "opportunities": [o.to_dict() for o in self.opportunities],
            "skipped_symbols": self.skipped_symbols,
            "errors": self.errors,
            "trades_executed": [t.to_dict() for t in self.trades_executed],
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass
class PositionReviewReport:
    """Midday position review report."""
    timestamp: datetime
    cycle_name: str = "midday_review"

    # Positions reviewed
    positions_reviewed: int = 0

    # Adjustments
    adjustments: List[PositionAdjustment] = field(default_factory=list)
    adjustments_applied: int = 0
    adjustments_pending: int = 0

    # Errors
    errors: Dict[str, str] = field(default_factory=dict)

    # Timing
    total_duration_seconds: float = 0.0

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines = [
            "=" * 50,
            "POSITION REVIEW REPORT",
            "=" * 50,
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Positions Reviewed: {self.positions_reviewed}",
            f"Adjustments Identified: {len(self.adjustments)}",
            f"Adjustments Applied: {self.adjustments_applied}",
            f"Adjustments Pending: {self.adjustments_pending}",
            "",
        ]

        if self.adjustments:
            lines.append("Adjustments:")
            for adj in self.adjustments:
                status = "Applied" if adj.applied else ("Pending" if adj.pending_approval else "Skipped")
                lines.append(
                    f"  [{status}] {adj.symbol} (#{adj.ticket}): "
                    f"{adj.adjustment_type} - {adj.reason}"
                )
                if adj.old_value and adj.new_value:
                    lines.append(f"           {adj.old_value:.2f} -> {adj.new_value:.2f}")
            lines.append("")

        if self.errors:
            lines.append("Errors:")
            for symbol, error in self.errors.items():
                lines.append(f"  {symbol}: {error}")
            lines.append("")

        lines.append(f"Duration: {self.total_duration_seconds:.1f}s")
        lines.append("=" * 50)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle_name": self.cycle_name,
            "positions_reviewed": self.positions_reviewed,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "adjustments_applied": self.adjustments_applied,
            "adjustments_pending": self.adjustments_pending,
            "errors": self.errors,
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass
class ReflectionReport:
    """Evening reflection report."""
    timestamp: datetime
    cycle_name: str = "evening_reflect"

    # Trades processed
    trades_processed: int = 0
    closed_trades: List[ClosedTrade] = field(default_factory=list)

    # Learning
    reflections_created: int = 0
    memories_stored: int = 0

    # P&L summary
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0

    # Errors
    errors: Dict[str, str] = field(default_factory=dict)

    # Timing
    total_duration_seconds: float = 0.0

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines = [
            "=" * 50,
            "REFLECTION REPORT",
            "=" * 50,
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Trades Processed: {self.trades_processed}",
            f"Winning Trades: {self.winning_trades}",
            f"Losing Trades: {self.losing_trades}",
            f"Total P&L: ${self.total_pnl:,.2f}",
            "",
            f"Reflections Created: {self.reflections_created}",
            f"Memories Stored: {self.memories_stored}",
            "",
        ]

        if self.closed_trades:
            lines.append("Closed Trades:")
            for trade in self.closed_trades:
                pnl_sign = "+" if trade.pnl >= 0 else ""
                lines.append(
                    f"  {trade.symbol} {trade.signal}: "
                    f"{pnl_sign}${trade.pnl:.2f} ({pnl_sign}{trade.pnl_percent:.2f}%) "
                    f"- {trade.exit_reason}"
                )
            lines.append("")

        if self.errors:
            lines.append("Errors:")
            for decision_id, error in self.errors.items():
                lines.append(f"  {decision_id}: {error}")
            lines.append("")

        lines.append(f"Duration: {self.total_duration_seconds:.1f}s")
        lines.append("=" * 50)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle_name": self.cycle_name,
            "trades_processed": self.trades_processed,
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "reflections_created": self.reflections_created,
            "memories_stored": self.memories_stored,
            "total_pnl": self.total_pnl,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "errors": self.errors,
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass
class DailySummaryReport:
    """End-of-day summary combining all cycles."""
    date: str
    morning_report: Optional[DailyAnalysisReport] = None
    midday_report: Optional[PositionReviewReport] = None
    evening_report: Optional[ReflectionReport] = None

    # Aggregate stats
    trades_opened: int = 0
    trades_closed: int = 0
    adjustments_made: int = 0
    total_pnl: float = 0.0

    def format_summary(self) -> str:
        """Format as human-readable daily summary."""
        lines = [
            "=" * 60,
            f"DAILY SUMMARY - {self.date}",
            "=" * 60,
            "",
            f"Trades Opened: {self.trades_opened}",
            f"Trades Closed: {self.trades_closed}",
            f"Adjustments Made: {self.adjustments_made}",
            f"Total P&L: ${self.total_pnl:,.2f}",
            "",
        ]

        if self.morning_report:
            lines.append(f"Morning: {len(self.morning_report.opportunities)} opportunities, "
                        f"{len(self.morning_report.trades_executed)} trades")

        if self.midday_report:
            lines.append(f"Midday: {self.midday_report.positions_reviewed} positions reviewed, "
                        f"{self.midday_report.adjustments_applied} adjustments")

        if self.evening_report:
            lines.append(f"Evening: {self.evening_report.trades_processed} trades processed, "
                        f"{self.evening_report.reflections_created} reflections")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
