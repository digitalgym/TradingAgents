# TradingAgents/graph/reflection.py

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI

from tradingagents.agents.utils.memory import TIER_SHORT, TIER_MID, TIER_LONG, SMCPatternMemory


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis. 
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning:
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Market intelligence.
     - Technical indicators.
     - Technical signals.
     - Price movement analysis.
     - Overall market data analysis 
     - News analysis.
     - Social media and sentiment analysis.
     - Fundamental data analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement:
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.

4. Query:
   - Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
   - Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def _determine_tier_from_returns(self, returns_losses: float) -> str:
        """
        Determine the appropriate memory tier based on returns magnitude.
        
        High-impact trades (large gains or losses) go to higher tiers.
        """
        abs_returns = abs(returns_losses)
        if abs_returns >= 5.0:  # 5%+ moves are significant
            return TIER_LONG
        elif abs_returns >= 2.0:  # 2-5% moves are notable
            return TIER_MID
        else:
            return TIER_SHORT
    
    def _was_prediction_correct(self, returns_losses: float, component_type: str, report: str) -> bool:
        """
        Determine if the prediction was correct based on returns.
        
        For BULL: positive returns = correct
        For BEAR: negative returns = correct (they predicted decline)
        For others: positive returns = correct
        """
        if component_type == "BEAR":
            # Bear researcher is correct if price went down (negative returns on long)
            # But if we're measuring from a short perspective, positive P&L = correct
            return returns_losses > 0
        else:
            return returns_losses > 0

    def reflect_bull_researcher(
        self, 
        current_state: Dict[str, Any], 
        returns_losses: float, 
        bull_memory,
        prediction_correct: Optional[bool] = None
    ):
        """
        Reflect on bull researcher's analysis and update memory with confidence scoring.
        
        Args:
            current_state: The current trading state
            returns_losses: The returns/losses from the trade
            bull_memory: The bull researcher's memory instance
            prediction_correct: Override for whether prediction was correct (optional)
        """
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        
        # Determine if prediction was correct
        if prediction_correct is None:
            prediction_correct = self._was_prediction_correct(returns_losses, "BULL", bull_debate_history)
        
        # Determine tier based on impact
        tier = self._determine_tier_from_returns(returns_losses)
        
        # Add with confidence scoring
        bull_memory.add_situations(
            [(situation, result)],
            tier=tier,
            returns=returns_losses,
            prediction_correct=prediction_correct
        )

    def reflect_bear_researcher(
        self, 
        current_state: Dict[str, Any], 
        returns_losses: float, 
        bear_memory,
        prediction_correct: Optional[bool] = None
    ):
        """
        Reflect on bear researcher's analysis and update memory with confidence scoring.
        
        Args:
            current_state: The current trading state
            returns_losses: The returns/losses from the trade
            bear_memory: The bear researcher's memory instance
            prediction_correct: Override for whether prediction was correct (optional)
        """
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        
        # For bear researcher, correct prediction means price went down
        if prediction_correct is None:
            prediction_correct = self._was_prediction_correct(returns_losses, "BEAR", bear_debate_history)
        
        tier = self._determine_tier_from_returns(returns_losses)
        
        bear_memory.add_situations(
            [(situation, result)],
            tier=tier,
            returns=returns_losses,
            prediction_correct=prediction_correct
        )

    def reflect_trader(
        self, 
        current_state: Dict[str, Any], 
        returns_losses: float, 
        trader_memory,
        prediction_correct: Optional[bool] = None
    ):
        """
        Reflect on trader's decision and update memory with confidence scoring.
        
        Args:
            current_state: The current trading state
            returns_losses: The returns/losses from the trade
            trader_memory: The trader's memory instance
            prediction_correct: Override for whether prediction was correct (optional)
        """
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        
        if prediction_correct is None:
            prediction_correct = returns_losses > 0
        
        tier = self._determine_tier_from_returns(returns_losses)
        
        trader_memory.add_situations(
            [(situation, result)],
            tier=tier,
            returns=returns_losses,
            prediction_correct=prediction_correct
        )

    def reflect_invest_judge(
        self, 
        current_state: Dict[str, Any], 
        returns_losses: float, 
        invest_judge_memory,
        prediction_correct: Optional[bool] = None
    ):
        """
        Reflect on investment judge's decision and update memory with confidence scoring.
        
        Args:
            current_state: The current trading state
            returns_losses: The returns/losses from the trade
            invest_judge_memory: The investment judge's memory instance
            prediction_correct: Override for whether prediction was correct (optional)
        """
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        
        if prediction_correct is None:
            prediction_correct = returns_losses > 0
        
        tier = self._determine_tier_from_returns(returns_losses)
        
        invest_judge_memory.add_situations(
            [(situation, result)],
            tier=tier,
            returns=returns_losses,
            prediction_correct=prediction_correct
        )

    def reflect_risk_manager(
        self, 
        current_state: Dict[str, Any], 
        returns_losses: float, 
        risk_manager_memory,
        prediction_correct: Optional[bool] = None
    ):
        """
        Reflect on risk manager's decision and update memory with confidence scoring.
        
        Args:
            current_state: The current trading state
            returns_losses: The returns/losses from the trade
            risk_manager_memory: The risk manager's memory instance
            prediction_correct: Override for whether prediction was correct (optional)
        """
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        
        if prediction_correct is None:
            prediction_correct = returns_losses > 0
        
        tier = self._determine_tier_from_returns(returns_losses)

        risk_manager_memory.add_situations(
            [(situation, result)],
            tier=tier,
            returns=returns_losses,
            prediction_correct=prediction_correct
        )

    def reflect_smc_pattern(
        self,
        decision: Dict[str, Any],
        smc_memory: SMCPatternMemory,
        current_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Reflect on an SMC pattern trade and store for pattern learning.

        This extracts SMC-specific lessons to learn which setups work best.

        Args:
            decision: The closed trade decision with outcome
            smc_memory: The SMC pattern memory instance
            current_state: Optional full state for situation extraction
        """
        # Check if decision has SMC context
        smc_context = decision.get("smc_context", {})
        setup_type = smc_context.get("setup_type") or decision.get("setup_type")

        if not setup_type:
            # No SMC setup type - skip SMC pattern storage
            return

        # Check for structured outcome
        structured_outcome = decision.get("structured_outcome", {})
        if not structured_outcome:
            # Build basic outcome from decision fields
            structured_outcome = {
                "result": "win" if decision.get("was_correct") else "loss",
                "returns_pct": decision.get("pnl_percent", 0),
                "direction_correct": decision.get("direction_correct", decision.get("was_correct")),
                "sl_placement": decision.get("sl_placement", "unknown"),
                "tp_placement": decision.get("tp_placement", "unknown"),
                "exit_type": decision.get("exit_reason", "unknown"),
            }

        # Build situation text
        situation_parts = []
        if current_state:
            situation = self._extract_current_situation(current_state)
            situation_parts.append(situation)
        else:
            # Build from decision rationale
            if decision.get("rationale"):
                situation_parts.append(decision["rationale"])

        situation_text = "\n".join(situation_parts) if situation_parts else f"{decision['symbol']} {setup_type} trade"

        # Generate SMC-specific lesson
        lesson = self._generate_smc_lesson(setup_type, smc_context, structured_outcome)

        # Store the pattern
        smc_memory.store_pattern(
            decision_id=decision.get("decision_id", "unknown"),
            symbol=decision.get("symbol", "unknown"),
            setup_type=setup_type,
            direction=decision.get("action", "unknown"),
            smc_context=smc_context,
            situation_text=situation_text[:2000],  # Truncate for embedding
            outcome=structured_outcome,
            lesson=lesson,
        )

    def _generate_smc_lesson(
        self,
        setup_type: str,
        smc_context: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> str:
        """Generate a lesson from the SMC pattern outcome."""
        lessons = []

        result = outcome.get("result", "unknown")
        direction_correct = outcome.get("direction_correct", True)
        sl_placement = outcome.get("sl_placement", "unknown")
        tp_placement = outcome.get("tp_placement", "unknown")

        # Entry zone quality
        entry_zone = smc_context.get("entry_zone", "unknown")
        zone_strength = smc_context.get("entry_zone_strength", 0.5)
        with_trend = smc_context.get("with_trend")
        confluences = smc_context.get("confluences", [])

        if result == "win":
            if with_trend:
                lessons.append(f"{setup_type} worked well WITH trend.")
            if zone_strength and zone_strength >= 0.7:
                lessons.append(f"High strength zone ({zone_strength:.0%}) provided good entry.")
            if confluences:
                lessons.append(f"Confluences ({', '.join(confluences[:2])}) supported the trade.")

        elif result == "loss":
            if not direction_correct:
                if not with_trend:
                    lessons.append(f"{setup_type} failed - was counter-trend. Consider only with-trend entries.")
                else:
                    lessons.append(f"{setup_type} direction was wrong despite with-trend. Review signal quality.")

            elif sl_placement == "too_tight":
                lessons.append(f"{setup_type} direction correct but SL too tight. Use wider stop for this setup.")
                if zone_strength and zone_strength < 0.6:
                    lessons.append(f"Zone strength was only {zone_strength:.0%} - weak zones need wider stops.")

            if tp_placement == "too_ambitious":
                lessons.append(f"TP was too ambitious for {setup_type}. Use more conservative targets.")

        # Confluence insights
        if result == "win" and confluences:
            lessons.append(f"Winning confluences: {', '.join(confluences)}.")
        elif result == "loss" and not confluences:
            lessons.append(f"{setup_type} without confluences failed. Require confirmations.")

        return " ".join(lessons) if lessons else f"{setup_type} {result}."

    def reflect_with_smc(
        self,
        current_state: Dict[str, Any],
        returns_losses: float,
        memories: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
        smc_memory: Optional[SMCPatternMemory] = None,
    ) -> Dict[str, int]:
        """
        Full reflection including SMC pattern storage.

        This is the main entry point for reflection that handles both
        standard agent memories and SMC pattern learning.

        Args:
            current_state: The full trading state
            returns_losses: The returns/losses from the trade
            memories: Dict of agent memories (bull_memory, bear_memory, etc.)
            decision: Optional closed decision for SMC pattern extraction
            smc_memory: Optional SMC pattern memory instance

        Returns:
            Dict with counts of reflections and memories stored
        """
        reflections_created = 0
        memories_stored = 0

        # Standard agent reflections
        if "bull_memory" in memories:
            self.reflect_bull_researcher(current_state, returns_losses, memories["bull_memory"])
            reflections_created += 1
            memories_stored += 1

        if "bear_memory" in memories:
            self.reflect_bear_researcher(current_state, returns_losses, memories["bear_memory"])
            reflections_created += 1
            memories_stored += 1

        if "trader_memory" in memories:
            self.reflect_trader(current_state, returns_losses, memories["trader_memory"])
            reflections_created += 1
            memories_stored += 1

        if "invest_judge_memory" in memories:
            self.reflect_invest_judge(current_state, returns_losses, memories["invest_judge_memory"])
            reflections_created += 1
            memories_stored += 1

        if "risk_manager_memory" in memories:
            self.reflect_risk_manager(current_state, returns_losses, memories["risk_manager_memory"])
            reflections_created += 1
            memories_stored += 1

        # SMC pattern reflection
        if decision and smc_memory:
            setup_type = decision.get("smc_context", {}).get("setup_type") or decision.get("setup_type")
            if setup_type:
                self.reflect_smc_pattern(decision, smc_memory, current_state)
                reflections_created += 1
                memories_stored += 1

        return {
            "reflections_created": reflections_created,
            "memories_stored": memories_stored,
        }
