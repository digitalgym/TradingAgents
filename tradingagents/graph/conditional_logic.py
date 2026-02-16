# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def _has_tool_calls(self, message) -> bool:
        """Safely check if a message has tool_calls."""
        tool_calls = getattr(message, "tool_calls", None)
        return bool(tool_calls)

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if self._has_tool_calls(last_message):
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if self._has_tool_calls(last_message):
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if self._has_tool_calls(last_message):
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if self._has_tool_calls(last_message):
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_quant(self, state: AgentState):
        """Determine if quant analysis should continue.

        The quant analyst is a single-prompt agent without tools,
        so it always proceeds directly to the next step.
        """
        # Quant analyst has no tool calls - always proceed
        return "Msg Clear Quant"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"
