# TradingAgents/graph/setup.py

from typing import Dict, Any, Callable, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic

# Import caching - will be available when running via backend
# Silently skip if not available (e.g., running standalone)
try:
    import sys
    from pathlib import Path
    # Add backend to path if not already there
    backend_path = Path(__file__).parent.parent.parent / "web" / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from state_store import AgentOutputCache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    AgentOutputCache = None


def create_cached_analyst_node(
    original_node: Callable,
    agent_name: str,
    report_key: str,
    cache_ttl_hours: float
) -> Callable:
    """Wrap an analyst node with caching logic.

    Args:
        original_node: The original analyst node function
        agent_name: Name used for cache key (e.g., "social_analyst")
        report_key: State key for the report (e.g., "sentiment_report")
        cache_ttl_hours: How long to cache the output

    Returns:
        Wrapped node function with caching
    """
    def cached_node(state):
        # Check if caching is available and not bypassed
        if not CACHING_AVAILABLE or not AgentOutputCache:
            return original_node(state)

        symbol = state.get("company_of_interest", "")
        force_fresh = state.get("force_fresh", False)

        # Check cache (unless force_fresh is set)
        if not force_fresh:
            cached = AgentOutputCache.get(symbol, agent_name)
            if cached and not cached.get("expired"):
                print(f"[Cache] Using cached {agent_name} output for {symbol} (age: {cached['age_hours']:.1f}h)")
                # Return cached report - empty messages means no tool calls needed
                return {
                    "messages": [],
                    report_key: cached["output"],
                }

        # Run original node
        result = original_node(state)

        # Cache the report if it was generated (non-empty report means final output)
        report = result.get(report_key, "")
        if report and len(report.strip()) > 0:
            print(f"[Cache] Storing {agent_name} output for {symbol} (TTL: {cache_ttl_hours}h)")
            AgentOutputCache.store(symbol, agent_name, report, cache_ttl_hours)

        return result

    return cached_node


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        smc_pattern_memory=None,
        meta_pattern_learning=None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.smc_pattern_memory = smc_pattern_memory
        self.meta_pattern_learning = meta_pattern_learning

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst (uses tools for indicators)
                - "social": Social media analyst (uses tools for news)
                - "news": News analyst (uses tools for news)
                - "fundamentals": Fundamentals analyst (uses tools for financial data)
                - "quant": Quant analyst (single-prompt, no tools, uses state data)
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            # Wrap social analyst with caching (4 hour TTL)
            original_social = create_social_media_analyst(self.quick_thinking_llm)
            analyst_nodes["social"] = create_cached_analyst_node(
                original_social, "social_analyst", "sentiment_report", 4.0
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            # Wrap news analyst with caching (2 hour TTL)
            original_news = create_news_analyst(self.quick_thinking_llm)
            analyst_nodes["news"] = create_cached_analyst_node(
                original_news, "news_analyst", "news_report", 2.0
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            # Wrap fundamentals analyst with caching (24 hour TTL)
            original_fundamentals = create_fundamentals_analyst(self.quick_thinking_llm)
            analyst_nodes["fundamentals"] = create_cached_analyst_node(
                original_fundamentals, "fundamentals_analyst", "fundamentals_report", 24.0
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "quant" in selected_analysts:
            # Quant analyst is a single-prompt agent that uses structured output
            # It receives all data via state (SMC, indicators, etc.) - no tools needed
            analyst_nodes["quant"] = create_quant_analyst(
                self.deep_thinking_llm, use_structured_output=True
            )
            delete_nodes["quant"] = create_msg_delete()
            # No tool node for quant - it's a pure analysis agent

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm,
            self.invest_judge_memory,
            smc_memory=self.smc_pattern_memory,
            meta_pattern_learning=self.meta_pattern_learning,
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory, self.smc_pattern_memory)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            # Only add tool nodes for analysts that have them (quant doesn't)
            if analyst_type in tool_nodes:
                workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Check if this analyst has tools (quant doesn't)
            has_tools = analyst_type in tool_nodes

            if has_tools:
                current_tools = f"tools_{analyst_type}"
                # Add conditional edges for current analyst (tool-using analysts)
                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)
            else:
                # No-tool analysts (like quant) go directly to message clear
                workflow.add_edge(current_analyst, current_clear)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )

        workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
