"""
Portfolio Automation Orchestrator

Main orchestrator for automated portfolio management with three daily cycles:
1. Morning Analysis - Analyze symbols and execute trades
2. Midday Review - Review positions and adjust stops
3. Evening Reflect - Process closed trades and learn
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from .portfolio_config import PortfolioConfig, SymbolConfig, ExecutionMode, get_default_config
from .correlation_manager import CorrelationManager
from .reporting import (
    AnalysisResult,
    PositionAdjustment,
    TradeExecution,
    ClosedTrade,
    DailyAnalysisReport,
    PositionReviewReport,
    ReflectionReport,
)

# Imports from existing TradingAgents modules
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.risk import (
    RiskGuardrails,
    DynamicStopLoss,
    get_atr_for_symbol,
    PositionSizer,
)
from tradingagents.trade_decisions import (
    store_decision,
    load_decision,
    load_decision_context,
    close_decision,
    list_active_decisions,
    get_decision_stats,
    link_decision_to_ticket,
)
from tradingagents.dataflows.mt5_data import (
    get_mt5_current_price,
    get_open_positions,
    modify_position,
    close_position,
    execute_trade_signal,
    get_closed_deal_by_ticket,
    get_mt5_symbol_info,
    check_mt5_autotrading,
    get_asset_type,
)
from tradingagents.dataflows.smc_utils import get_smc_position_review_context
from tradingagents.default_config import DEFAULT_CONFIG


def run_deep_position_analysis(
    symbol: str,
    direction: str,
    entry_price: float,
    current_price: float,
    sl: float = 0,
    tp: float = 0,
    volume: float = 0,
    profit: float = 0,
    timeframe: str = 'H1',
) -> Dict[str, Any]:
    """
    Run deep multi-agent analysis for an existing position.

    This is the same analysis pipeline used by the web API and can be
    called directly by the automation scheduler.

    Args:
        symbol: Trading symbol
        direction: 'BUY' or 'SELL'
        entry_price: Position entry price
        current_price: Current market price
        sl: Current stop loss (0 if none)
        tp: Current take profit (0 if none)
        volume: Position volume in lots
        profit: Current P/L in account currency
        timeframe: Primary timeframe for analysis

    Returns:
        dict with:
        - recommendation: 'HOLD', 'ADJUST', or 'CLOSE'
        - suggested_sl: Suggested stop loss (if ADJUST)
        - suggested_tp: Suggested take profit (if ADJUST)
        - suggested_trailing_sl: Trailing stop suggestion
        - close_reason: Reason for close (if CLOSE)
        - trading_plan: Full plan from agents
        - smc_context: SMC analysis context
        - error: Error message if analysis failed
    """
    try:
        # Calculate P/L percentage
        if entry_price > 0:
            if direction.upper() == 'BUY':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            pnl_pct = 0

        # Get asset type
        asset_type = get_asset_type(symbol)

        # Select analysts based on asset type
        if asset_type in ["commodity", "forex"]:
            selected_analysts = ["market", "social", "news"]
        else:
            selected_analysts = ["market", "social", "news", "fundamentals"]

        # Get SMC context
        smc_review = get_smc_position_review_context(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            sl=sl,
            tp=tp,
            timeframe=timeframe
        )

        # Build position context for analysis
        position_context = f"""
REVIEWING EXISTING POSITION:
- Direction: {direction}
- Entry Price: {entry_price:.5f}
- Current Price: {current_price:.5f}
- Current SL: {sl if sl else 'NONE SET - RISK EXPOSURE'}
- Current TP: {tp if tp else 'NONE SET'}
- Volume: {volume} lots
- P/L: {pnl_pct:+.2f}% (${profit:.2f})

{smc_review.get('smc_context', '')}

TASK: Review this position and recommend one of:
1. HOLD - Position is fine, keep current SL/TP
2. ADJUST - Suggest new SL and/or TP values with reasoning
3. CLOSE - Position should be closed (reversal signals, structure break against position, etc.)

If recommending ADJUST, provide specific price levels for new SL and/or TP.
If recommending CLOSE, explain the urgent reason (reversal signals, overextended, structure break, etc.)
"""

        # Initialize trading agents graph
        config = DEFAULT_CONFIG.copy()
        config["asset_type"] = asset_type
        ta = TradingAgentsGraph(debug=False, config=config, selected_analysts=selected_analysts)

        # Run analysis with position context
        final_state, decision = ta.propagate(
            symbol,
            timeframe,
            smc_context=position_context
        )

        if final_state is None:
            final_state = {}
        if decision is None:
            decision = {}

        # Extract trading plans
        trader_plan = final_state.get("trader_investment_plan", "")
        risk_decision = final_state.get("final_trade_decision", "")

        # Parse recommendation from the decision
        recommendation = "HOLD"  # Default
        suggested_sl = None
        suggested_tp = None
        close_reason = None

        combined_text = f"{risk_decision} {trader_plan}".upper()
        if "CLOSE" in combined_text and ("IMMEDIATELY" in combined_text or "REVERSAL" in combined_text or "EXIT" in combined_text):
            recommendation = "CLOSE"
            if "reversal" in risk_decision.lower():
                close_reason = "Reversal signals detected"
            elif "structure" in risk_decision.lower() and "break" in risk_decision.lower():
                close_reason = "Structure break against position"
            else:
                close_reason = "Market conditions unfavorable"
        elif "ADJUST" in combined_text or "MOVE" in combined_text or "TRAIL" in combined_text:
            recommendation = "ADJUST"

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

        return {
            "recommendation": recommendation,
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
            "trading_plan": {
                "trader_plan": trader_plan,
                "risk_decision": risk_decision,
            },
            "smc_context": smc_review,
            "decision": decision,
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "recommendation": "HOLD",  # Default to HOLD on error
        }


class PortfolioAutomation:
    """
    Main orchestrator for automated portfolio management.

    Coordinates:
    - Daily analysis cycles across multiple symbols
    - Trade execution with portfolio-level risk controls
    - Position review and adjustment
    - Closed trade reflection and learning
    """

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
    ):
        """
        Initialize portfolio automation.

        Args:
            config: Portfolio configuration. Uses default if not provided.
        """
        self.config = config or get_default_config()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Initialize components
        self.guardrails = RiskGuardrails(
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            max_consecutive_losses=self.config.max_consecutive_losses,
            cooldown_hours=self.config.cooldown_hours,
        )

        self.correlation_manager = CorrelationManager(self.config)

        # Dynamic stop loss manager
        self.stop_loss_manager = DynamicStopLoss(
            atr_multiplier=self.config.atr_stop_multiplier,
            trailing_multiplier=self.config.atr_trailing_multiplier,
            risk_reward_ratio=self.config.risk_reward_ratio,
        )

        # Lazy-initialized graph
        self._graph: Optional[TradingAgentsGraph] = None

        # Daily trade counter
        self._trades_today: List[str] = []
        self._last_trade_date: Optional[str] = None

        # Logging
        self._setup_logging()

        # State persistence
        self._state_file = Path(self.config.state_file)
        self._load_state()

    def _setup_logging(self):
        """Setup logging for portfolio automation."""
        logs_dir = Path(self.config.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("PortfolioAutomation")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = logs_dir / f"portfolio_{datetime.now().strftime('%Y%m%d')}.log"
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
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                self._trades_today = state.get("trades_today", [])
                self._last_trade_date = state.get("last_trade_date")
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")
                self._trades_today = []
                self._last_trade_date = None

    def _save_state(self):
        """Save automation state to file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "trades_today": self._trades_today,
            "last_trade_date": self._last_trade_date,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _get_graph(self) -> TradingAgentsGraph:
        """Lazy-initialize the trading graph."""
        if self._graph is None:
            # Build config for TradingAgentsGraph
            graph_config = DEFAULT_CONFIG.copy()
            graph_config.update({
                "llm_provider": self.config.llm_provider,
                "deep_think_llm": self.config.deep_think_llm,
                "quick_think_llm": self.config.quick_think_llm,
                "max_debate_rounds": self.config.max_debate_rounds,
                "asset_type": self.config.asset_type,
                "data_vendors": {
                    "core_stock_apis": "mt5",
                    "technical_indicators": "mt5",
                    "news_data": "xai",
                },
                "tool_vendors": {
                    "get_insider_sentiment": "xai",
                },
            })
            self._graph = TradingAgentsGraph(config=graph_config, debug=False)
        return self._graph

    def _get_account_balance(self) -> float:
        """Get current account balance from MT5."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return 10000.0  # Default fallback
            account_info = mt5.account_info()
            if account_info:
                return account_info.balance
            return 10000.0
        except Exception:
            return 10000.0

    def _get_account_equity(self) -> float:
        """Get current account equity from MT5."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return 10000.0
            account_info = mt5.account_info()
            if account_info:
                return account_info.equity
            return 10000.0
        except Exception:
            return 10000.0

    def _count_trades_today(self) -> int:
        """Count trades executed today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_trade_date != today:
            self._trades_today = []
            self._last_trade_date = today
            self._save_state()
        return len(self._trades_today)

    def _extract_confidence(self, final_state: Dict[str, Any]) -> float:
        """Extract confidence score from final state."""
        # Try to extract from final_trade_decision
        decision = final_state.get("final_trade_decision", "")

        # Default confidence based on signal strength indicators
        confidence = 0.5

        # Look for confidence indicators in the decision text
        decision_lower = decision.lower()

        if "strong" in decision_lower or "high confidence" in decision_lower:
            confidence = 0.8
        elif "moderate" in decision_lower or "medium confidence" in decision_lower:
            confidence = 0.6
        elif "weak" in decision_lower or "low confidence" in decision_lower:
            confidence = 0.4

        # Boost confidence if multiple analysts agree
        market_report = final_state.get("market_report", "")
        news_report = final_state.get("news_report", "")
        sentiment_report = final_state.get("sentiment_report", "")

        bullish_count = sum(1 for r in [market_report, news_report, sentiment_report]
                          if "bullish" in r.lower() or "buy" in r.lower())
        bearish_count = sum(1 for r in [market_report, news_report, sentiment_report]
                          if "bearish" in r.lower() or "sell" in r.lower())

        if bullish_count >= 2 or bearish_count >= 2:
            confidence = min(confidence + 0.1, 0.95)

        return confidence

    def _extract_signal(self, final_state: Dict[str, Any]) -> str:
        """Extract trade signal from final state."""
        decision = final_state.get("final_trade_decision", "")
        decision_upper = decision.upper()

        if "BUY" in decision_upper:
            return "BUY"
        elif "SELL" in decision_upper:
            return "SELL"
        else:
            return "HOLD"

    # =========================================================================
    # MORNING ANALYSIS CYCLE
    # =========================================================================

    async def run_morning_analysis(self) -> DailyAnalysisReport:
        """
        Morning analysis cycle - analyze all symbols and execute trades.

        Steps:
        1. Check guardrails - can we trade today?
        2. Get MT5 open positions for current state
        3. For each enabled symbol:
           - Skip if already at max positions
           - Skip if correlation group at limit
           - Run analysis via TradingAgentsGraph.propagate()
           - Store signal and context
        4. Rank opportunities by confidence
        5. Execute top opportunities (if FULL_AUTO mode)
        """
        start_time = time.time()
        report = DailyAnalysisReport(timestamp=datetime.now())
        self.logger.info("=" * 50)
        self.logger.info("STARTING MORNING ANALYSIS CYCLE")
        self.logger.info("=" * 50)

        try:
            return await self._run_morning_analysis_impl(report, start_time)
        except Exception as e:
            import traceback
            self.logger.error(f"CRITICAL ERROR in morning analysis: {e}")
            self.logger.error(traceback.format_exc())
            report.blocked = True
            report.blocked_reason = f"Cycle crashed: {str(e)}"
            report.errors["_cycle"] = str(e)
            report.total_duration_seconds = time.time() - start_time
            return report

    async def _run_morning_analysis_impl(self, report: DailyAnalysisReport, start_time: float) -> DailyAnalysisReport:
        """Internal implementation of morning analysis."""
        # Check MT5 connection
        mt5_status = check_mt5_autotrading()
        if not mt5_status.get("connected"):
            report.blocked = True
            report.blocked_reason = "MT5 not connected"
            self.logger.error("MT5 not connected")
            return report

        if not mt5_status.get("autotrading_enabled"):
            self.logger.warning("MT5 AutoTrading is disabled")

        # Check guardrails
        account_balance = self._get_account_balance()
        report.account_balance = account_balance
        report.account_equity = self._get_account_equity()

        can_trade, reason = self.guardrails.check_can_trade(account_balance)
        if not can_trade:
            report.blocked = True
            report.blocked_reason = reason
            self.logger.warning(f"Trading blocked: {reason}")
            return report

        # Get current portfolio state from MT5
        current_positions = get_open_positions()
        report.current_positions = current_positions
        self.logger.info(f"Current positions: {len(current_positions)}")

        # Check portfolio-level position limit
        if len(current_positions) >= self.config.max_total_positions:
            report.blocked = True
            report.blocked_reason = f"Max total positions reached ({len(current_positions)}/{self.config.max_total_positions})"
            self.logger.warning(report.blocked_reason)
            return report

        # Track trades placed today
        trades_today = self._count_trades_today()
        self.logger.info(f"Trades today: {trades_today}/{self.config.max_daily_trades}")

        # Analyze each enabled symbol
        for symbol_config in self.config.get_enabled_symbols():
            symbol = symbol_config.symbol
            self.logger.info(f"Analyzing {symbol}...")

            # Check symbol-level limits
            symbol_positions = [p for p in current_positions if p.get("symbol") == symbol]
            if len(symbol_positions) >= symbol_config.max_positions:
                report.skipped_symbols.append((symbol, "max_positions_reached"))
                self.logger.info(f"  Skipped: max positions for {symbol}")
                continue

            # Check correlation group limits
            if not self.correlation_manager.can_open_position(symbol, current_positions):
                report.skipped_symbols.append((symbol, "correlation_limit"))
                self.logger.info(f"  Skipped: correlation group limit")
                continue

            # Check daily trade limit
            if trades_today >= self.config.max_daily_trades:
                report.skipped_symbols.append((symbol, "daily_trade_limit"))
                self.logger.info(f"  Skipped: daily trade limit reached")
                continue

            # Run analysis
            try:
                analysis_result = await self._analyze_symbol(symbol, symbol_config)
                report.analysis_results[symbol] = analysis_result

                if analysis_result.signal in ["BUY", "SELL"]:
                    if analysis_result.confidence >= symbol_config.min_confidence:
                        report.opportunities.append(analysis_result)
                        self.logger.info(
                            f"  Opportunity: {analysis_result.signal} "
                            f"(confidence: {analysis_result.confidence:.2f})"
                        )
                    else:
                        self.logger.info(
                            f"  Signal {analysis_result.signal} below confidence threshold "
                            f"({analysis_result.confidence:.2f} < {symbol_config.min_confidence})"
                        )
                else:
                    self.logger.info(f"  Signal: HOLD")

            except Exception as e:
                self.logger.error(f"  Analysis failed: {e}")
                report.errors[symbol] = str(e)

        # Rank opportunities by confidence
        report.opportunities.sort(key=lambda x: x.confidence, reverse=True)

        # Execute trades if FULL_AUTO mode
        if self.config.execution_mode == ExecutionMode.FULL_AUTO and report.opportunities:
            max_trades = self.config.max_daily_trades - trades_today
            opportunities_to_execute = report.opportunities[:max_trades]

            self.logger.info(f"Executing {len(opportunities_to_execute)} trades...")
            executions = await self._execute_opportunities(opportunities_to_execute)
            report.trades_executed = executions

        elif self.config.execution_mode == ExecutionMode.SEMI_AUTO and report.opportunities:
            self.logger.info(
                f"SEMI_AUTO mode: {len(report.opportunities)} opportunities identified, "
                "manual execution required"
            )

        elif self.config.execution_mode == ExecutionMode.PAPER and report.opportunities:
            self.logger.info(
                f"PAPER mode: {len(report.opportunities)} opportunities identified (not executed)"
            )

        report.total_duration_seconds = time.time() - start_time
        self._save_state()

        # Log report summary
        self.logger.info(report.format_summary())

        return report

    async def _analyze_symbol(
        self,
        symbol: str,
        symbol_config: SymbolConfig,
    ) -> AnalysisResult:
        """Analyze a single symbol."""
        start_time = time.time()
        trade_date = datetime.now().strftime("%Y-%m-%d")

        # Run SMC analysis first (for commodities)
        smc_context = None
        smc_analysis = None
        try:
            from tradingagents.dataflows.smc_utils import (
                analyze_multi_timeframe_smc,
                format_smc_for_prompt,
            )
            smc_analysis = analyze_multi_timeframe_smc(
                symbol=symbol,
                timeframes=symbol_config.timeframes,
            )
            if smc_analysis:
                smc_context = format_smc_for_prompt(smc_analysis, symbol)
        except Exception as e:
            self.logger.debug(f"SMC analysis skipped for {symbol}: {e}")

        # Run main analysis
        graph = self._get_graph()
        final_state, signal = graph.propagate(symbol, trade_date, smc_context=smc_context)

        # Extract confidence
        confidence = self._extract_confidence(final_state)

        # Get current price and calculate entry levels
        price_info = get_mt5_current_price(symbol)
        current_price = price_info.get("ask") if signal == "BUY" else price_info.get("bid")

        # Calculate ATR-based stops
        entry_price = current_price
        stop_loss = None
        take_profit = None

        if self.config.use_atr_stops:
            try:
                atr = get_atr_for_symbol(symbol, period=14)
                levels = self.stop_loss_manager.calculate_levels(
                    entry_price=entry_price,
                    atr=atr,
                    direction=signal if signal in ["BUY", "SELL"] else "BUY",
                )
                stop_loss = levels.stop_loss
                take_profit = levels.take_profit
            except Exception as e:
                self.logger.warning(f"Could not calculate ATR stops: {e}")

        # Calculate position size
        recommended_size = self.config.default_lot_size
        try:
            recommended_size = self._calculate_position_size(
                symbol, signal, symbol_config, entry_price, stop_loss
            )
        except Exception as e:
            self.logger.warning(f"Could not calculate position size: {e}")

        return AnalysisResult(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            trade_date=trade_date,
            final_state=final_state,
            smc_analysis=smc_analysis,
            recommended_size=recommended_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=final_state.get("final_trade_decision", ""),
            analysis_duration_seconds=time.time() - start_time,
        )

    def _calculate_position_size(
        self,
        symbol: str,
        signal: str,
        symbol_config: SymbolConfig,
        entry_price: float,
        stop_loss: Optional[float],
    ) -> float:
        """Calculate position size respecting all limits."""
        account_balance = self._get_account_balance()

        # Get historical trade returns for Kelly
        stats = get_decision_stats(symbol)

        # Get symbol info for lot calculations
        symbol_info = get_mt5_symbol_info(symbol)
        contract_size = symbol_info.get("trade_contract_size", 100)
        min_lot = symbol_info.get("volume_min", 0.01)
        lot_step = symbol_info.get("volume_step", 0.01)

        sizer = PositionSizer(
            account_balance=account_balance,
            max_risk_per_trade=symbol_config.risk_budget_pct / 100,
        )

        # Calculate stop loss if not provided
        if not stop_loss and entry_price:
            # Default 2% stop
            if signal == "BUY":
                stop_loss = entry_price * 0.98
            else:
                stop_loss = entry_price * 1.02

        # Use Kelly if enough history, else fixed fractional
        if stats.get("total_decisions", 0) >= 10:
            win_rate = stats.get("correct_rate", 0.5)
            avg_pnl = stats.get("avg_pnl_percent", 1.0)
            result = sizer.kelly_size(
                win_rate=win_rate,
                avg_win=abs(avg_pnl) / 100 if avg_pnl > 0 else 0.02,
                avg_loss=abs(avg_pnl) / 100 if avg_pnl < 0 else 0.01,
                entry_price=entry_price,
                stop_loss=stop_loss,
            )
        else:
            result = sizer.fixed_fractional_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
            )

        # Convert to lots
        lots = sizer.calculate_lots(
            result.recommended_size,
            contract_size=contract_size,
            min_lot=min_lot,
            lot_step=lot_step,
        )

        return lots

    async def _execute_opportunities(
        self,
        opportunities: List[AnalysisResult],
    ) -> List[TradeExecution]:
        """Execute trades for approved opportunities."""
        executions = []

        for opp in opportunities:
            if opp.signal == "HOLD":
                continue

            execution = TradeExecution(
                symbol=opp.symbol,
                signal=opp.signal,
                volume=opp.recommended_size,
                entry_price=opp.entry_price or 0,
                stop_loss=opp.stop_loss or 0,
                take_profit=opp.take_profit or 0,
                success=False,
            )

            try:
                # Execute trade
                result = execute_trade_signal(
                    symbol=opp.symbol,
                    signal=opp.signal,
                    entry_price=opp.entry_price,
                    stop_loss=opp.stop_loss,
                    take_profit=opp.take_profit,
                    volume=opp.recommended_size,
                    comment=f"PortfolioAuto {opp.signal}",
                )

                if result.get("success"):
                    execution.success = True
                    execution.ticket = result.get("order_id")
                    execution.entry_price = result.get("price", opp.entry_price)

                    # Store decision for tracking
                    decision_id = store_decision(
                        symbol=opp.symbol,
                        decision_type="OPEN",
                        action=opp.signal,
                        rationale=opp.rationale,
                        source="portfolio_automation",
                        entry_price=execution.entry_price,
                        stop_loss=opp.stop_loss,
                        take_profit=opp.take_profit,
                        volume=opp.recommended_size,
                        mt5_ticket=execution.ticket,
                        analysis_context={"final_state": opp.final_state},
                    )
                    execution.decision_id = decision_id

                    # Track trade
                    self._trades_today.append(decision_id)
                    self._save_state()

                    self.logger.info(
                        f"Trade executed: {opp.symbol} {opp.signal} "
                        f"{opp.recommended_size} lots @ {execution.entry_price}"
                    )
                else:
                    execution.error = result.get("error", "Unknown error")
                    self.logger.error(f"Execution failed for {opp.symbol}: {execution.error}")

            except Exception as e:
                execution.error = str(e)
                self.logger.error(f"Execution failed for {opp.symbol}: {e}")

            executions.append(execution)

        return executions

    # =========================================================================
    # MIDDAY REVIEW CYCLE
    # =========================================================================

    async def run_midday_review(self) -> PositionReviewReport:
        """
        Midday position review - adjust stops, review exits.

        Steps:
        1. Get all open MT5 positions
        2. For each position:
           - Fetch current price and ATR
           - Check trailing stop conditions
           - Apply adjustments (breakeven, trailing)
        3. Generate review report
        """
        start_time = time.time()
        report = PositionReviewReport(timestamp=datetime.now())
        self.logger.info("=" * 50)
        self.logger.info("STARTING MIDDAY REVIEW CYCLE")
        self.logger.info("=" * 50)

        try:
            positions = get_open_positions()
        except Exception as e:
            import traceback
            self.logger.error(f"CRITICAL ERROR getting positions: {e}")
            self.logger.error(traceback.format_exc())
            report.errors["_cycle"] = f"Failed to get positions: {str(e)}"
            report.total_duration_seconds = time.time() - start_time
            return report

        report.positions_reviewed = len(positions)
        self.logger.info(f"Reviewing {len(positions)} positions...")

        for position in positions:
            symbol = position.get("symbol")
            ticket = position.get("ticket")
            pos_type = position.get("type", "")
            direction = "BUY" if "BUY" in pos_type.upper() else "SELL"

            try:
                # Get ATR for dynamic stop calculations
                atr = get_atr_for_symbol(symbol, period=14)

                current_price = position.get("price_current", 0)
                entry_price = position.get("price_open", 0)
                current_sl = position.get("sl", 0)
                current_tp = position.get("tp", 0)

                # Calculate P&L percentage
                if entry_price > 0:
                    if direction == "BUY":
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                else:
                    pnl_pct = 0

                self.logger.info(
                    f"{symbol} {direction} (#{ticket}): "
                    f"P&L: {pnl_pct:+.2f}%, ATR: {atr:.4f}"
                )

                # Check for trailing stop update
                new_sl, should_trail = self.stop_loss_manager.calculate_trailing_stop(
                    current_price=current_price,
                    current_sl=current_sl,
                    atr=atr,
                    direction=direction,
                )

                if should_trail and new_sl:
                    adjustment = PositionAdjustment(
                        ticket=ticket,
                        symbol=symbol,
                        adjustment_type="trailing_stop",
                        old_value=current_sl,
                        new_value=new_sl,
                        reason=f"ATR trailing stop (1.5x ATR from current price)",
                    )

                    # Apply adjustment based on execution mode
                    if self.config.execution_mode == ExecutionMode.FULL_AUTO:
                        result = modify_position(ticket, sl=new_sl)
                        adjustment.applied = result.get("success", False)
                        if adjustment.applied:
                            report.adjustments_applied += 1
                            self.logger.info(f"  Trailing SL updated: {current_sl:.2f} -> {new_sl:.2f}")
                        else:
                            adjustment.error = result.get("error")
                    else:
                        adjustment.pending_approval = True
                        report.adjustments_pending += 1
                        self.logger.info(f"  Trailing SL suggested: {current_sl:.2f} -> {new_sl:.2f}")

                    report.adjustments.append(adjustment)

                # Check breakeven conditions (if in profit and SL not yet at breakeven)
                elif pnl_pct >= 1.0:  # 1% in profit
                    breakeven_sl = self.stop_loss_manager.calculate_breakeven_stop(
                        entry_price=entry_price,
                        current_price=current_price,
                        current_sl=current_sl,
                        direction=direction,
                    )

                    if breakeven_sl and breakeven_sl != current_sl:
                        # Check if breakeven is better than current SL
                        is_better = (
                            (direction == "BUY" and breakeven_sl > current_sl) or
                            (direction == "SELL" and breakeven_sl < current_sl)
                        )

                        if is_better:
                            adjustment = PositionAdjustment(
                                ticket=ticket,
                                symbol=symbol,
                                adjustment_type="breakeven",
                                old_value=current_sl,
                                new_value=breakeven_sl,
                                reason=f"Move to breakeven (+0.1% buffer)",
                            )

                            if self.config.execution_mode == ExecutionMode.FULL_AUTO:
                                result = modify_position(ticket, sl=breakeven_sl)
                                adjustment.applied = result.get("success", False)
                                if adjustment.applied:
                                    report.adjustments_applied += 1
                                    self.logger.info(
                                        f"  Breakeven SL set: {current_sl:.2f} -> {breakeven_sl:.2f}"
                                    )
                                else:
                                    adjustment.error = result.get("error")
                            else:
                                adjustment.pending_approval = True
                                report.adjustments_pending += 1

                            report.adjustments.append(adjustment)

            except Exception as e:
                self.logger.error(f"Error reviewing {symbol}: {e}")
                report.errors[symbol] = str(e)

        report.total_duration_seconds = time.time() - start_time
        self.logger.info(report.format_summary())

        return report

    # =========================================================================
    # EVENING REFLECT CYCLE
    # =========================================================================

    async def run_evening_reflect(self) -> ReflectionReport:
        """
        Evening reflection - process closed trades and update learning.

        Steps:
        1. Get all active decisions from trade_decisions
        2. Check MT5 history for closed positions
        3. For each closed trade:
           - Auto-fetch exit price from MT5
           - Calculate returns
           - Close the decision record
           - Run reflect_and_remember() for learning
        4. Generate reflection report
        """
        start_time = time.time()
        report = ReflectionReport(timestamp=datetime.now())
        self.logger.info("=" * 50)
        self.logger.info("STARTING EVENING REFLECTION CYCLE")
        self.logger.info("=" * 50)

        # Get active decisions
        try:
            active_decisions = list_active_decisions()
        except Exception as e:
            import traceback
            self.logger.error(f"CRITICAL ERROR getting active decisions: {e}")
            self.logger.error(traceback.format_exc())
            report.errors["_cycle"] = f"Failed to get decisions: {str(e)}"
            report.total_duration_seconds = time.time() - start_time
            return report

        self.logger.info(f"Active decisions: {len(active_decisions)}")

        for decision in active_decisions:
            ticket = decision.get("mt5_ticket")
            if not ticket:
                continue

            decision_id = decision.get("decision_id")

            # Check if position is still open
            try:
                positions = get_open_positions()
            except Exception as e:
                self.logger.error(f"Error getting positions for {decision_id}: {e}")
                report.errors[decision_id] = f"Failed to check position: {str(e)}"
                continue

            if any(p.get("ticket") == ticket for p in positions):
                self.logger.debug(f"{decision_id}: Position still open")
                continue  # Still open

            # Position closed - get from history
            try:
                closed_deal = get_closed_deal_by_ticket(ticket, days_back=7)
                if not closed_deal:
                    self.logger.debug(f"{decision_id}: No closed deal found")
                    continue

                exit_price = closed_deal.get("price", 0)
                profit = closed_deal.get("profit", 0)
                exit_reason = closed_deal.get("reason", "unknown")

                self.logger.info(
                    f"Processing closed trade: {decision_id} "
                    f"(exit: {exit_price}, profit: ${profit:.2f})"
                )

                # Close the decision
                closed = close_decision(
                    decision_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                )

                pnl_percent = closed.get("pnl_percent", 0)
                was_profitable = closed.get("was_correct", False)

                # Record in guardrails
                self.guardrails.record_trade_result(
                    was_win=was_profitable,
                    pnl_pct=pnl_percent,
                    account_balance=self._get_account_balance(),
                )

                # Create closed trade record
                closed_trade = ClosedTrade(
                    decision_id=decision_id,
                    symbol=decision.get("symbol"),
                    signal=decision.get("action"),
                    entry_price=decision.get("entry_price", 0),
                    exit_price=exit_price,
                    volume=decision.get("volume", 0),
                    pnl=profit,
                    pnl_percent=pnl_percent,
                    was_profitable=was_profitable,
                    exit_reason=exit_reason,
                )
                report.closed_trades.append(closed_trade)
                report.trades_processed += 1
                report.total_pnl += profit

                if was_profitable:
                    report.winning_trades += 1
                else:
                    report.losing_trades += 1

                # Run reflection for learning
                if decision.get("has_context"):
                    try:
                        context = load_decision_context(decision_id)
                        if context and context.get("final_state"):
                            graph = self._get_graph()
                            graph.curr_state = context["final_state"]
                            # Reload decision to get updated outcome fields for SMC learning
                            closed_decision = load_decision(decision_id)
                            result = graph.reflect_and_remember(pnl_percent, decision=closed_decision)
                            report.reflections_created += result.get("reflections_created", 1)
                            report.memories_stored += result.get("memories_stored", 1)
                            self.logger.info(f"  Reflection created for {decision_id}")
                    except Exception as e:
                        self.logger.warning(f"  Could not create reflection: {e}")

            except Exception as e:
                self.logger.error(f"Error processing {decision_id}: {e}")
                report.errors[decision_id] = str(e)

        report.total_duration_seconds = time.time() - start_time
        self.logger.info(report.format_summary())

        return report

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        positions = get_open_positions()
        guardrails_status = self.guardrails.get_status()

        return {
            "config": {
                "execution_mode": self.config.execution_mode.value,
                "symbols": [s.symbol for s in self.config.get_enabled_symbols()],
                "max_positions": self.config.max_total_positions,
                "max_daily_trades": self.config.max_daily_trades,
            },
            "portfolio": {
                "open_positions": len(positions),
                "account_balance": self._get_account_balance(),
                "account_equity": self._get_account_equity(),
            },
            "guardrails": guardrails_status,
            "today": {
                "trades_executed": len(self._trades_today),
                "max_trades": self.config.max_daily_trades,
            },
            "correlation_warnings": self.correlation_manager.get_correlation_warnings(positions),
        }

    def format_status_report(self) -> str:
        """Format status as human-readable report."""
        status = self.get_status()

        lines = [
            "=" * 60,
            "PORTFOLIO AUTOMATION STATUS",
            "=" * 60,
            "",
            f"Execution Mode: {status['config']['execution_mode']}",
            f"Symbols: {', '.join(status['config']['symbols'])}",
            "",
            "Portfolio:",
            f"  Open Positions: {status['portfolio']['open_positions']}/{status['config']['max_positions']}",
            f"  Balance: ${status['portfolio']['account_balance']:,.2f}",
            f"  Equity: ${status['portfolio']['account_equity']:,.2f}",
            "",
            "Today:",
            f"  Trades: {status['today']['trades_executed']}/{status['today']['max_trades']}",
            "",
            "Guardrails:",
            f"  Can Trade: {'Yes' if status['guardrails']['can_trade'] else 'No'}",
            f"  Status: {status['guardrails']['status_summary']}",
        ]

        if status["correlation_warnings"]:
            lines.append("")
            lines.append("Correlation Warnings:")
            for warning in status["correlation_warnings"]:
                lines.append(f"  - {warning}")

        lines.append("=" * 60)

        return "\n".join(lines)
