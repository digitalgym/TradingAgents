"""
LLM Trade Refiner

Adds contextual intelligence to rule-based SMC trade plans.
The LLM refines decisions by considering:
- Historical trade performance
- Current market context
- Lessons learned from past trades
- Risk factors the rules can't capture

This is the "intelligence layer" that sits on top of systematic rules.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI

from tradingagents.dataflows.smc_trade_plan import SMCTradePlan


@dataclass
class RefinedTradePlan:
    """A trade plan refined by LLM intelligence."""
    # Original plan
    base_plan: SMCTradePlan

    # LLM decision
    action: str  # "TAKE", "SKIP", "MODIFY"
    confidence: float  # 0-1

    # Adjusted levels (None = use base plan)
    adjusted_entry: Optional[float] = None
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None
    partial_tp_levels: List[float] = field(default_factory=list)

    # Position sizing
    size_multiplier: float = 1.0  # 0.5 = half size, 1.5 = 1.5x size

    # LLM reasoning
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Historical context that influenced decision
    historical_win_rate: Optional[float] = None
    similar_trades_count: int = 0

    @property
    def final_entry(self) -> float:
        return self.adjusted_entry if self.adjusted_entry else self.base_plan.entry_price

    @property
    def final_sl(self) -> float:
        return self.adjusted_sl if self.adjusted_sl else self.base_plan.stop_loss

    @property
    def final_tp(self) -> float:
        return self.adjusted_tp if self.adjusted_tp else self.base_plan.take_profit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "signal": self.base_plan.signal,
            "entry": self.final_entry,
            "stop_loss": self.final_sl,
            "take_profit": self.final_tp,
            "partial_tp_levels": self.partial_tp_levels,
            "size_multiplier": self.size_multiplier,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "warnings": self.warnings,
            "historical_win_rate": self.historical_win_rate,
            "similar_trades_count": self.similar_trades_count,
            "base_plan": self.base_plan.to_dict(),
            "adjustments": {
                "entry_adjusted": self.adjusted_entry is not None,
                "sl_adjusted": self.adjusted_sl is not None,
                "tp_adjusted": self.adjusted_tp is not None,
            }
        }


class LLMTradeRefiner:
    """
    Refines trade plans using LLM intelligence.

    The LLM's role:
    1. Evaluate context the rules can't capture (news, session, sentiment)
    2. Learn from historical trade outcomes
    3. Adjust entry/SL/TP based on micro-structure
    4. Decide position sizing based on conviction
    5. Provide clear reasoning for decisions
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        """
        Initialize the trade refiner.

        Args:
            llm: LangChain LLM instance (will create one if not provided)
            model_name: Model to use if creating LLM
            temperature: LLM temperature (lower = more consistent)
        """
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def refine_plan(
        self,
        base_plan: SMCTradePlan,
        historical_context: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        smc_memory_insights: Optional[str] = None,
    ) -> RefinedTradePlan:
        """
        Refine a trade plan using LLM intelligence.

        Args:
            base_plan: The rule-based trade plan to refine
            historical_context: Historical trade data for this setup type
            market_context: Current market conditions (news, session, etc.)
            smc_memory_insights: Insights from SMC pattern memory

        Returns:
            RefinedTradePlan with LLM adjustments and reasoning
        """
        # Build the prompt
        prompt = self._build_refinement_prompt(
            base_plan, historical_context, market_context, smc_memory_insights
        )

        # Get LLM response
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_response(response.content, base_plan)
        except Exception as e:
            # Fallback: return base plan with no modifications
            result = RefinedTradePlan(
                base_plan=base_plan,
                action=base_plan.recommendation,
                confidence=0.5,
                reasoning=f"LLM refinement failed: {str(e)}. Using base plan.",
                warnings=["LLM refinement unavailable"]
            )

        return result

    def _build_refinement_prompt(
        self,
        base_plan: SMCTradePlan,
        historical_context: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
        smc_memory_insights: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build the prompt for LLM refinement."""

        # Format base plan
        plan_text = f"""
SYSTEMATIC TRADE PLAN (Rule-Based):
=====================================
Signal: {base_plan.signal}
Entry Price: {base_plan.entry_price:.5f}
Stop Loss: {base_plan.stop_loss:.5f}
Take Profit: {base_plan.take_profit:.5f}

Zone Quality Score: {base_plan.zone_quality_score:.0f}/100
Setup Type: {base_plan.setup_type.value}
Risk:Reward Ratio: {base_plan.risk_reward_ratio:.2f}

Entry Checklist ({base_plan.checklist.passed_count}/{base_plan.checklist.total_count} passed):
- HTF Trend Aligned: {"✓" if base_plan.checklist.htf_trend_aligned else "✗"}
- Zone Unmitigated: {"✓" if base_plan.checklist.zone_unmitigated else "✗"}
- Has Confluence: {"✓" if base_plan.checklist.has_confluence else "✗"}
- Liquidity Target: {"✓" if base_plan.checklist.liquidity_target_exists else "✗"}
- Structure Confirmed: {"✓" if base_plan.checklist.structure_confirmed else "✗"}
- Premium/Discount: {"✓" if base_plan.checklist.in_discount_premium else "✗"}
- Session Favorable: {"✓" if base_plan.checklist.session_favorable else "✗"}

Confluence Factors: {', '.join(base_plan.confluence_factors) if base_plan.confluence_factors else 'None'}

Rule-Based Recommendation: {base_plan.recommendation}
{f"Skip Reason: {base_plan.skip_reason}" if base_plan.skip_reason else ""}
"""

        # Format historical context
        history_text = ""
        if historical_context:
            history_text = f"""
HISTORICAL PERFORMANCE:
=======================
Setup Type: {historical_context.get('setup_type', 'Unknown')}
Total Trades: {historical_context.get('total_trades', 0)}
Win Rate: {historical_context.get('win_rate', 0):.1%}
Average R:R Achieved: {historical_context.get('avg_rr', 0):.2f}

Similar Recent Trades:
"""
            for trade in historical_context.get('similar_trades', [])[:5]:
                outcome = "WIN" if trade.get('was_win') else "LOSS"
                history_text += f"- {outcome}: {trade.get('returns_pct', 0):+.2f}%"
                if trade.get('lesson'):
                    history_text += f" | Lesson: {trade['lesson'][:80]}"
                history_text += "\n"

            if historical_context.get('key_lessons'):
                history_text += f"\nKey Lessons Learned:\n"
                for lesson in historical_context.get('key_lessons', [])[:3]:
                    history_text += f"- {lesson}\n"

        # Format market context
        market_text = ""
        if market_context:
            market_text = f"""
CURRENT MARKET CONTEXT:
=======================
Session: {market_context.get('session', 'Unknown')}
Volatility: {market_context.get('volatility', 'Unknown')}
Market Regime: {market_context.get('market_regime', 'Unknown')}
"""
            if market_context.get('upcoming_news'):
                market_text += f"Upcoming News: {market_context['upcoming_news']}\n"
            if market_context.get('daily_pnl_pct') is not None:
                market_text += f"Daily P/L: {market_context['daily_pnl_pct']:+.2f}%\n"
            if market_context.get('existing_positions'):
                market_text += f"Existing Positions: {market_context['existing_positions']}\n"
            if market_context.get('correlation_warning'):
                market_text += f"Correlation Warning: {market_context['correlation_warning']}\n"

        # Format SMC memory insights
        memory_text = ""
        if smc_memory_insights:
            memory_text = f"""
SMC PATTERN INSIGHTS:
=====================
{smc_memory_insights}
"""

        # Build system prompt
        system_prompt = """You are an expert trade refiner that adds intelligence to systematic SMC trade plans.

YOUR ROLE:
- The systematic rules have identified a potential setup
- You add contextual intelligence that rules cannot capture
- You learn from historical performance to improve decisions
- You provide clear reasoning for all adjustments

CAPABILITIES:
1. TAKE/SKIP/MODIFY decision based on context
2. Adjust entry/SL/TP for micro-structure or risk
3. Set position size multiplier (0.5x to 1.5x)
4. Add partial take profit levels
5. Provide confidence level and reasoning

RULES YOU MUST FOLLOW:
- For BUY: SL must be BELOW entry, TP must be ABOVE entry
- For SELL: SL must be ABOVE entry, TP must be BELOW entry
- Never violate these directional rules
- Base adjustments on evidence, not hunches
- If historical win rate < 40%, require strong justification to TAKE
- If rules say SKIP but you see opportunity, explain why clearly

OUTPUT FORMAT (JSON):
{
    "action": "TAKE" | "SKIP" | "MODIFY",
    "confidence": 0.0-1.0,
    "adjusted_entry": null or price,
    "adjusted_sl": null or price,
    "adjusted_tp": null or price,
    "partial_tp_levels": [],
    "size_multiplier": 0.5-1.5,
    "reasoning": "Clear explanation...",
    "key_factors": ["factor1", "factor2"],
    "warnings": ["warning1"] or []
}"""

        user_prompt = f"""{plan_text}
{history_text}
{market_text}
{memory_text}

Based on the systematic plan and context above, provide your refined decision.
Remember to output valid JSON only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _parse_response(
        self,
        response: str,
        base_plan: SMCTradePlan,
    ) -> RefinedTradePlan:
        """Parse LLM response into RefinedTradePlan."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            # Validate and extract fields
            action = data.get("action", "SKIP").upper()
            if action not in ["TAKE", "SKIP", "MODIFY"]:
                action = "SKIP"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # Get adjusted levels
            adjusted_entry = data.get("adjusted_entry")
            adjusted_sl = data.get("adjusted_sl")
            adjusted_tp = data.get("adjusted_tp")

            # Validate adjustments don't violate directional rules
            entry = adjusted_entry if adjusted_entry else base_plan.entry_price
            sl = adjusted_sl if adjusted_sl else base_plan.stop_loss
            tp = adjusted_tp if adjusted_tp else base_plan.take_profit

            if base_plan.signal == "BUY":
                if sl >= entry:
                    adjusted_sl = None  # Invalid, use base
                if tp <= entry:
                    adjusted_tp = None  # Invalid, use base
            else:  # SELL
                if sl <= entry:
                    adjusted_sl = None
                if tp >= entry:
                    adjusted_tp = None

            # Get partial TPs
            partial_tps = data.get("partial_tp_levels", [])
            if not isinstance(partial_tps, list):
                partial_tps = []

            # Get size multiplier
            size_mult = float(data.get("size_multiplier", 1.0))
            size_mult = max(0.5, min(1.5, size_mult))

            return RefinedTradePlan(
                base_plan=base_plan,
                action=action,
                confidence=confidence,
                adjusted_entry=adjusted_entry,
                adjusted_sl=adjusted_sl,
                adjusted_tp=adjusted_tp,
                partial_tp_levels=partial_tps,
                size_multiplier=size_mult,
                reasoning=data.get("reasoning", ""),
                key_factors=data.get("key_factors", []),
                warnings=data.get("warnings", []),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Parse failed - return base plan with warning
            return RefinedTradePlan(
                base_plan=base_plan,
                action=base_plan.recommendation,
                confidence=0.5,
                reasoning=f"Could not parse LLM response: {str(e)}. Using base plan.",
                warnings=["Response parsing failed"]
            )

    def build_historical_context(
        self,
        setup_type: str,
        symbol: str,
        smc_memory: Any = None,
        trade_similarity: Any = None,
    ) -> Dict[str, Any]:
        """
        Build historical context for refinement.

        Args:
            setup_type: The SMC setup type (e.g., "ob_fvg_confluence")
            symbol: Trading symbol
            smc_memory: SMCPatternMemory instance
            trade_similarity: TradeSimilaritySearch instance

        Returns:
            Historical context dict
        """
        context = {
            "setup_type": setup_type,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "similar_trades": [],
            "key_lessons": [],
        }

        # Query SMC pattern memory
        if smc_memory:
            try:
                # Get best setups for this type
                best = smc_memory.get_best_setups(
                    symbol=symbol,
                    setup_type=setup_type,
                    min_samples=2
                )
                if best:
                    for setup in best[:3]:
                        context["key_lessons"].append(
                            setup.get("recommendation", "")
                        )

                # Get similar patterns
                similar = smc_memory.get_similar_patterns(
                    f"{symbol} {setup_type}",
                    symbol=symbol,
                    n_matches=5
                )
                for p in similar:
                    context["similar_trades"].append({
                        "was_win": p.get("was_win", False),
                        "returns_pct": p.get("returns_pct", 0),
                        "lesson": p.get("lesson", ""),
                    })
            except Exception:
                pass

        # Query trade similarity search
        if trade_similarity:
            try:
                result = trade_similarity.find_similar_trades(
                    {"symbol": symbol, "setup_type": setup_type},
                    n_results=10
                )
                stats = result.get("statistics", {})
                context["total_trades"] = stats.get("total_similar", 0)
                context["win_rate"] = stats.get("win_rate", 0)
                context["avg_rr"] = stats.get("avg_rr_achieved", 0)

                for trade in result.get("similar_trades", [])[:5]:
                    if trade not in context["similar_trades"]:
                        context["similar_trades"].append({
                            "was_win": trade.get("outcome") == "win",
                            "returns_pct": trade.get("returns_pct", 0),
                            "lesson": trade.get("lesson", ""),
                        })
            except Exception:
                pass

        return context

    def build_market_context(
        self,
        session: Optional[str] = None,
        volatility: Optional[str] = None,
        market_regime: Optional[str] = None,
        upcoming_news: Optional[str] = None,
        daily_pnl_pct: Optional[float] = None,
        existing_positions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build market context for refinement.

        Args:
            session: Current trading session
            volatility: Current volatility level
            market_regime: Current market regime
            upcoming_news: Any upcoming high-impact news
            daily_pnl_pct: Current daily P/L percentage
            existing_positions: List of existing position descriptions

        Returns:
            Market context dict
        """
        context = {
            "session": session or "Unknown",
            "volatility": volatility or "Unknown",
            "market_regime": market_regime or "Unknown",
        }

        if upcoming_news:
            context["upcoming_news"] = upcoming_news

        if daily_pnl_pct is not None:
            context["daily_pnl_pct"] = daily_pnl_pct

        if existing_positions:
            context["existing_positions"] = ", ".join(existing_positions)

            # Check for correlation warnings
            # This is simplified - real implementation would check actual correlations
            symbols_in_positions = [p.split()[0] for p in existing_positions]
            if len(symbols_in_positions) > 1:
                # Crude correlation check for precious metals
                metals = {"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"}
                metals_in_pos = [s for s in symbols_in_positions if s in metals]
                if len(metals_in_pos) > 1:
                    context["correlation_warning"] = f"Multiple correlated metals positions: {metals_in_pos}"

        return context


def create_hybrid_trade_decision(
    smc_analysis: Dict[str, Any],
    current_price: float,
    atr: float,
    llm: Optional[ChatOpenAI] = None,
    market_regime: Optional[str] = None,
    session: Optional[str] = None,
    smc_memory: Any = None,
    trade_similarity: Any = None,
    market_context: Optional[Dict[str, Any]] = None,
) -> RefinedTradePlan:
    """
    Convenience function to create a hybrid trade decision.

    Combines rule-based SMC plan generation with LLM refinement.

    Args:
        smc_analysis: Full SMC analysis
        current_price: Current market price
        atr: Current ATR value
        llm: Optional LLM instance
        market_regime: Current market regime
        session: Current trading session
        smc_memory: SMCPatternMemory instance for historical patterns
        trade_similarity: TradeSimilaritySearch instance
        market_context: Additional market context

    Returns:
        RefinedTradePlan with both systematic levels and LLM refinements
    """
    from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator

    # Step 1: Generate rule-based plan
    generator = SMCTradePlanGenerator()
    base_plan = generator.generate_plan(
        smc_analysis=smc_analysis,
        current_price=current_price,
        atr=atr,
        market_regime=market_regime,
        session=session,
    )

    if not base_plan:
        # No valid setup found by rules
        return None

    # Step 2: Refine with LLM
    refiner = LLMTradeRefiner(llm=llm)

    # Build contexts
    historical_context = refiner.build_historical_context(
        setup_type=base_plan.setup_type.value,
        symbol=smc_analysis.get("symbol", "UNKNOWN"),
        smc_memory=smc_memory,
        trade_similarity=trade_similarity,
    )

    if not market_context:
        market_context = refiner.build_market_context(
            session=session,
            market_regime=market_regime,
        )

    # Get SMC memory insights
    smc_insights = None
    if smc_memory:
        try:
            best = smc_memory.get_best_setups(
                symbol=smc_analysis.get("symbol"),
                min_samples=2,
                min_win_rate=0.5
            )
            if best:
                smc_insights = "\n".join([
                    f"- {s.get('recommendation', '')}" for s in best[:3]
                ])
        except Exception:
            pass

    # Refine
    refined = refiner.refine_plan(
        base_plan=base_plan,
        historical_context=historical_context,
        market_context=market_context,
        smc_memory_insights=smc_insights,
    )

    return refined
