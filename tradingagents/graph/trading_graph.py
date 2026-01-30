# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory, SMCPatternMemory, MemoryUsageTracker, MetaPatternLearning
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.mt5_data import get_asset_type, get_mt5_current_price

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=None,
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include. If None, auto-selects based on asset_type.
                             For commodities/forex: ["market", "social", "news"]
                             For stocks: ["market", "social", "news", "fundamentals"]
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        
        # Auto-select analysts based on asset type if not specified
        if selected_analysts is None:
            asset_type = self.config.get("asset_type", "auto")
            if asset_type in ["commodity", "forex"]:
                selected_analysts = ["market", "social", "news"]
            else:
                selected_analysts = ["market", "social", "news", "fundamentals"]
        
        self.selected_analysts = selected_analysts

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() in ["openai", "ollama", "openrouter", "xai", "grok"]:
            # Get API key as plain string (NOT callable) to avoid async issues
            if self.config["llm_provider"].lower() in ["xai", "grok"]:
                api_key = os.environ.get("XAI_API_KEY") or ""
            else:
                api_key = os.environ.get("OPENAI_API_KEY") or ""

            if not api_key:
                raise ValueError(f"API key not found for provider {self.config['llm_provider']}")

            # Create OpenAI client explicitly for sync operations
            from openai import OpenAI
            sync_client = OpenAI(
                api_key=api_key,
                base_url=self.config["backend_url"]
            )

            # Pass both api_key as string AND root_client for sync support
            self.deep_thinking_llm = ChatOpenAI(
                model=self.config["deep_think_llm"],
                base_url=self.config["backend_url"],
                api_key=api_key,
                root_client=sync_client
            )
            self.quick_thinking_llm = ChatOpenAI(
                model=self.config["quick_think_llm"],
                base_url=self.config["backend_url"],
                api_key=api_key,
                root_client=sync_client
            )
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        # Initialize memories (requires OpenAI API for embeddings)
        if self.config.get("use_memory", True):
            self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
            self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
            self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
            self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
            self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)
            # SMC pattern memory for setup-specific learning
            self.smc_pattern_memory = SMCPatternMemory(self.config)
            # Memory usage tracker for feedback loop validation
            self.memory_usage_tracker = MemoryUsageTracker(self.config)
            # Meta-pattern learning for cross-agent insights
            self.meta_pattern_learning = MetaPatternLearning(self.config)
        else:
            self.bull_memory = None
            self.bear_memory = None
            self.trader_memory = None
            self.invest_judge_memory = None
            self.risk_manager_memory = None
            self.smc_pattern_memory = None
            self.memory_usage_tracker = None
            self.meta_pattern_learning = None

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            self.smc_pattern_memory,
            self.meta_pattern_learning,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(self.selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date, smc_context: str = None):
        """Run the trading agents graph for a company on a specific date.

        Args:
            company_name: Ticker symbol or company name
            trade_date: Date to analyze
            smc_context: Optional formatted SMC analysis string to inject into agent prompts
        """

        self.ticker = company_name

        # Get current broker price FIRST - this is critical for accurate analysis
        # Some brokers use different price formats (e.g., silver at 108.xx instead of 31.xx)
        current_price = self._get_current_price(company_name)

        # Initialize state with current price
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, current_price=current_price
        )

        # Add SMC context to state if provided
        if smc_context:
            init_agent_state["smc_context"] = smc_context

        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Get current price for signal validation
        current_price = self._get_current_price(company_name)

        # Get structured decision if available (from Risk Manager's structured output)
        structured_decision = final_state.get("final_trade_decision_structured")

        # Return decision and processed signal
        return final_state, self.process_signal(
            final_state["final_trade_decision"],
            current_price,
            structured_decision
        )

    def propagate_with_progress(self, company_name, trade_date, smc_context: str = None, progress_callback=None, force_fresh: bool = False):
        """Run the trading agents graph with progress callbacks.

        Args:
            company_name: Ticker symbol or company name
            trade_date: Date to analyze
            smc_context: Optional formatted SMC analysis string
            progress_callback: Callable for progress updates
            force_fresh: If True, bypass agent output cache and re-run all agents

        Uses graph.stream() for real-time progress updates as each agent completes.
        """
        self.ticker = company_name

        # Map graph node names to UI agent IDs and display info
        # The graph uses capitalized names with spaces (see setup.py)
        node_to_agent = {
            "Market Analyst": ("market_analyst", "Market Analyst", "Analyzing price action and technical indicators..."),
            "Social Analyst": ("social_analyst", "Social Sentiment", "Analyzing social media sentiment..."),
            "News Analyst": ("news_analyst", "News Analyst", "Analyzing market news..."),
            "Fundamentals Analyst": ("fundamentals_analyst", "Fundamentals Analyst", "Analyzing company fundamentals..."),
            "Bull Researcher": ("bull_researcher", "Bull Researcher", "Building bullish case..."),
            "Bear Researcher": ("bear_researcher", "Bear Researcher", "Building bearish case..."),
            "Research Manager": ("research_manager", "Research Manager", "Synthesizing research debate..."),
            "Trader": ("trader", "Trader Agent", "Formulating trading plan..."),
            "Risky Analyst": ("risky_analyst", "Aggressive Risk", "Analyzing high-risk scenarios..."),
            "Safe Analyst": ("safe_analyst", "Conservative Risk", "Analyzing low-risk scenarios..."),
            "Neutral Analyst": ("neutral_analyst", "Balanced Risk", "Analyzing balanced approach..."),
            "Risk Judge": ("risk_manager", "Risk Manager", "Making final risk assessment..."),
        }

        # Execution order of agents (based on setup.py graph structure - sequential)
        # This helps predict which agent runs next
        execution_order = [
            "market_analyst", "social_analyst", "news_analyst",
            "bull_researcher", "bear_researcher", "research_manager",
            "trader", "risky_analyst", "safe_analyst", "neutral_analyst", "risk_manager"
        ]

        # Progress weights for each agent (total = 100)
        agent_progress = {
            "market_analyst": 15,
            "social_analyst": 10,
            "news_analyst": 10,
            "fundamentals_analyst": 5,
            "bull_researcher": 10,
            "bear_researcher": 10,
            "research_manager": 10,
            "trader": 10,
            "risky_analyst": 5,
            "safe_analyst": 5,
            "neutral_analyst": 5,
            "risk_manager": 5,
        }

        # Get current broker price FIRST - critical for accurate analysis
        current_price = self._get_current_price(company_name)

        # Initialize state with current price
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, current_price=current_price
        )
        if smc_context:
            init_agent_state["smc_context"] = smc_context

        # Pass force_fresh flag to agent nodes for cache bypass
        init_agent_state["force_fresh"] = force_fresh

        # Use "updates" mode to get individual node outputs for progress tracking
        args = self.propagator.get_graph_args(stream_mode="updates")

        # Track completed agents and their outputs
        completed_agents = set()
        agent_outputs = {}
        current_progress = 5

        # Start with first agent as in-progress
        in_progress_agents = ["market_analyst"]

        # Emit starting progress - show first agent as running
        if progress_callback:
            progress_callback(
                step_name="starting",
                step_title="Market Analyst",
                step_description="Analyzing price action and technical indicators...",
                step_output="",
                progress=5,
                completed_steps=[],
                agent_outputs={},
                in_progress_agents=["market_analyst"]
            )

        # Map tool nodes to their parent agents for intermediate progress
        tool_to_agent = {
            "tools_market": "market_analyst",
            "tools_social": "social_analyst",
            "tools_news": "news_analyst",
            "tools_fundamentals": "fundamentals_analyst",
        }

        # Use streaming to get real-time updates
        # Initialize with a copy of init_agent_state so we can accumulate updates
        final_state = init_agent_state.copy()

        for chunk in self.graph.stream(init_agent_state, **args):
            # chunk is a dict with node name as key and output as value
            for node_name, node_output in chunk.items():
                # Skip internal nodes
                if node_name.startswith("_") or node_name == "__end__":
                    continue

                # Handle tool node completions - shows the agent is actively working
                if node_name in tool_to_agent:
                    parent_agent = tool_to_agent[node_name]
                    if progress_callback and parent_agent in in_progress_agents:
                        # Small progress bump to show activity
                        current_progress = min(current_progress + 1, 95)
                        # Get agent title from the agent_progress keys
                        agent_titles = {
                            "market_analyst": "Market Analyst",
                            "social_analyst": "Social Sentiment",
                            "news_analyst": "News Analyst",
                            "fundamentals_analyst": "Fundamentals Analyst",
                        }
                        progress_callback(
                            step_name=parent_agent,
                            step_title=agent_titles.get(parent_agent, parent_agent),
                            step_description="Fetching market data...",
                            step_output="",
                            progress=current_progress,
                            completed_steps=list(completed_agents),
                            agent_outputs=dict(agent_outputs),  # Pass copy to avoid reference issues
                            in_progress_agents=list(in_progress_agents)
                        )
                    continue

                # Map node name to agent ID
                if node_name in node_to_agent:
                    agent_id, agent_title, agent_desc = node_to_agent[node_name]

                    # Extract output based on node type
                    output_content = self._extract_node_output(node_name, node_output)

                    # Only mark as completed when we have actual output
                    # (Analysts run multiple times - once for tool calls, once for final report)
                    has_real_output = bool(output_content and output_content.strip())

                    if has_real_output:
                        # Mark as completed only when we have real output
                        completed_agents.add(agent_id)

                        # Remove from in-progress
                        if agent_id in in_progress_agents:
                            in_progress_agents.remove(agent_id)

                        # Determine next agent to mark as in-progress
                        try:
                            current_idx = execution_order.index(agent_id)
                            if current_idx + 1 < len(execution_order):
                                next_agent = execution_order[current_idx + 1]
                                if next_agent not in completed_agents and next_agent not in in_progress_agents:
                                    in_progress_agents.append(next_agent)
                        except ValueError:
                            pass  # Agent not in execution order

                        # Store the output
                        agent_outputs[agent_id] = {
                            "title": agent_title,
                            "output": output_content[:1000]
                        }

                        # Update progress only when agent actually completes with output
                        current_progress += agent_progress.get(agent_id, 5)
                        current_progress = min(95, current_progress)  # Cap at 95 until fully complete

                        # Emit progress callback with a COPY of agent_outputs
                        # to ensure current state is captured immediately
                        if progress_callback:
                            progress_callback(
                                step_name=agent_id,
                                step_title=agent_title,
                                step_description=f"{agent_title} completed",
                                step_output=output_content[:500],
                                progress=current_progress,
                                completed_steps=list(completed_agents),
                                agent_outputs=dict(agent_outputs),  # Pass copy to capture current state
                                in_progress_agents=list(in_progress_agents)
                            )

                # Track the final state by accumulating updates
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if value is not None:  # Only update non-None values
                            final_state[key] = value

        # Ensure required state keys exist
        for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report",
                    "investment_debate_state", "risk_debate_state", "trader_investment_plan",
                    "final_trade_decision", "final_trade_decision_structured", "investment_plan"]:
            if key not in final_state:
                if key == "final_trade_decision_structured":
                    final_state[key] = None
                else:
                    final_state[key] = init_agent_state.get(key, "" if "report" in key or "plan" in key or "decision" in key else {})

        # Store current state
        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        # Finalize agent outputs from final state (merge with streaming outputs)
        final_agent_outputs = self._collect_final_outputs(final_state)
        agent_outputs.update(final_agent_outputs)

        # Emit final completion
        if progress_callback:
            progress_callback(
                step_name="complete",
                step_title="Analysis Complete",
                step_description="All agents have completed their analysis.",
                step_output=final_state.get("final_trade_decision", ""),
                progress=100,
                completed_steps=list(agent_outputs.keys()),
                agent_outputs=dict(agent_outputs),  # Pass copy for consistency
                in_progress_agents=[]
            )

        # Get current price for signal validation
        current_price = self._get_current_price(company_name)

        # Get structured decision if available
        structured_decision = final_state.get("final_trade_decision_structured")

        return final_state, self.process_signal(
            final_state.get("final_trade_decision", ""),
            current_price,
            structured_decision
        )

    def _extract_node_output(self, node_name: str, node_output: dict) -> str:
        """Extract readable output from a node's output dictionary."""
        if not isinstance(node_output, dict):
            return str(node_output)[:1000] if node_output else ""

        # Map graph node names to their output keys
        output_keys = {
            "Market Analyst": "market_report",
            "Social Analyst": "sentiment_report",
            "News Analyst": "news_report",
            "Fundamentals Analyst": "fundamentals_report",
            "Trader": "trader_investment_plan",
        }

        # Try direct output key
        if node_name in output_keys:
            key = output_keys[node_name]
            if key in node_output:
                return node_output[key][:1000] if node_output[key] else ""

        # For debate nodes, extract from debate state
        if node_name in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
            debate_state = node_output.get("investment_debate_state", {})
            if node_name == "Bull Researcher" and debate_state.get("bull_history"):
                content = debate_state["bull_history"]
                return ("..." + content[-997:]) if len(content) > 1000 else content.strip()
            elif node_name == "Bear Researcher" and debate_state.get("bear_history"):
                content = debate_state["bear_history"]
                return ("..." + content[-997:]) if len(content) > 1000 else content.strip()
            elif node_name == "Research Manager" and debate_state.get("judge_decision"):
                return debate_state["judge_decision"][:1000]

        # For risk nodes
        if node_name in ["Risky Analyst", "Safe Analyst", "Neutral Analyst", "Risk Judge"]:
            risk_state = node_output.get("risk_debate_state", {})
            history_keys = {
                "Risky Analyst": "risky_history",
                "Safe Analyst": "safe_history",
                "Neutral Analyst": "neutral_history",
                "Risk Judge": "judge_decision"
            }
            key = history_keys.get(node_name)
            if key and risk_state.get(key):
                content = risk_state[key]
                if key == "judge_decision":
                    return content[:1000]
                return ("..." + content[-997:]) if len(content) > 1000 else content.strip()

        return ""

    def _collect_final_outputs(self, final_state: dict) -> dict:
        """Collect all agent outputs from the final state."""
        agent_outputs = {}

        if final_state.get("market_report"):
            agent_outputs["market_analyst"] = {
                "title": "Market Analyst",
                "output": final_state["market_report"][:1000]
            }

        if final_state.get("sentiment_report"):
            agent_outputs["social_analyst"] = {
                "title": "Social Sentiment",
                "output": final_state["sentiment_report"][:1000]
            }

        if final_state.get("news_report"):
            agent_outputs["news_analyst"] = {
                "title": "News Analyst",
                "output": final_state["news_report"][:1000]
            }

        if final_state.get("fundamentals_report"):
            agent_outputs["fundamentals_analyst"] = {
                "title": "Fundamentals Analyst",
                "output": final_state["fundamentals_report"][:1000]
            }

        debate_state = final_state.get("investment_debate_state", {})
        if debate_state:
            if debate_state.get("bull_history"):
                content = debate_state["bull_history"]
                agent_outputs["bull_researcher"] = {
                    "title": "Bull Researcher",
                    "output": ("..." + content[-997:]) if len(content) > 1000 else content.strip()
                }
            if debate_state.get("bear_history"):
                content = debate_state["bear_history"]
                agent_outputs["bear_researcher"] = {
                    "title": "Bear Researcher",
                    "output": ("..." + content[-997:]) if len(content) > 1000 else content.strip()
                }
            if debate_state.get("judge_decision"):
                agent_outputs["research_manager"] = {
                    "title": "Research Manager",
                    "output": debate_state["judge_decision"][:1000]
                }

        if final_state.get("trader_investment_plan"):
            agent_outputs["trader"] = {
                "title": "Trader Agent",
                "output": final_state["trader_investment_plan"][:1000]
            }

        risk_state = final_state.get("risk_debate_state", {})
        if risk_state:
            if risk_state.get("risky_history"):
                content = risk_state["risky_history"]
                agent_outputs["risky_analyst"] = {
                    "title": "Aggressive Risk",
                    "output": ("..." + content[-997:]) if len(content) > 1000 else content.strip()
                }

            if risk_state.get("safe_history"):
                content = risk_state["safe_history"]
                agent_outputs["safe_analyst"] = {
                    "title": "Conservative Risk",
                    "output": ("..." + content[-997:]) if len(content) > 1000 else content.strip()
                }

            if risk_state.get("neutral_history"):
                content = risk_state["neutral_history"]
                agent_outputs["neutral_analyst"] = {
                    "title": "Balanced Risk",
                    "output": ("..." + content[-997:]) if len(content) > 1000 else content.strip()
                }

            if risk_state.get("judge_decision"):
                agent_outputs["risk_manager"] = {
                    "title": "Risk Manager",
                    "output": risk_state["judge_decision"][:1000]
                }

        return agent_outputs

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "final_trade_decision_structured": final_state.get("final_trade_decision_structured"),
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses, decision: Optional[Dict[str, Any]] = None):
        """Reflect on decisions and update memory based on returns.

        Args:
            returns_losses: The returns/losses percentage from the trade
            decision: Optional closed decision dict for SMC pattern learning
        """
        memories = {
            "bull_memory": self.bull_memory,
            "bear_memory": self.bear_memory,
            "trader_memory": self.trader_memory,
            "invest_judge_memory": self.invest_judge_memory,
            "risk_manager_memory": self.risk_manager_memory,
        }

        # Use unified reflection method that handles both standard and SMC patterns
        result = self.reflector.reflect_with_smc(
            self.curr_state,
            returns_losses,
            memories,
            decision=decision,
            smc_memory=self.smc_pattern_memory,
        )

        # PHASE 4: Feedback loop validation
        # Update memory confidence based on trade outcomes
        if decision and self.memory_usage_tracker:
            decision_id = decision.get("decision_id")
            was_successful = decision.get("was_correct", returns_losses > 0)

            if decision_id:
                try:
                    validation_result = self.memory_usage_tracker.update_on_outcome(
                        trade_id=decision_id,
                        was_successful=was_successful,
                        returns_pct=returns_losses,
                        agent_memories=memories,
                    )
                    result["memories_validated"] = validation_result.get("updated", 0)
                except Exception as e:
                    print(f"Warning: Failed to update memory confidence: {e}")

        # PHASE 5: Meta-pattern learning
        # Record cross-agent patterns for higher-level learning
        if decision and self.meta_pattern_learning and self.curr_state:
            try:
                decision_id = decision.get("decision_id")
                was_successful = decision.get("was_correct", returns_losses > 0)

                # Extract bull/bear signals from state
                debate_state = self.curr_state.get("investment_debate_state", {})
                bull_history = debate_state.get("bull_history", "")
                bear_history = debate_state.get("bear_history", "")

                # Infer signals from history (simplified - look for keywords)
                bull_signal = "neutral"
                if "bullish" in bull_history.lower() or "buy" in bull_history.lower():
                    bull_signal = "bullish"
                elif "bearish" in bull_history.lower() or "sell" in bull_history.lower():
                    bull_signal = "bearish"

                bear_signal = "neutral"
                if "bullish" in bear_history.lower() or "buy" in bear_history.lower():
                    bear_signal = "bullish"
                elif "bearish" in bear_history.lower() or "sell" in bear_history.lower():
                    bear_signal = "bearish"

                self.meta_pattern_learning.record_trade_outcome(
                    decision_id=decision_id,
                    symbol=decision.get("symbol", "unknown"),
                    bull_signal=bull_signal,
                    bear_signal=bear_signal,
                    final_action=decision.get("action", "HOLD"),
                    was_successful=was_successful,
                    returns_pct=returns_losses,
                    market_regime=decision.get("market_regime"),
                    volatility_regime=decision.get("volatility_regime"),
                )
                result["meta_pattern_recorded"] = True
            except Exception as e:
                print(f"Warning: Failed to record meta-pattern: {e}")

        return result

    def track_memory_usage(self, decision_id: str, memory_ids: List[str], collection: str, agent: str):
        """Track which memories were used for a trade decision.

        This enables the feedback loop - when the trade closes,
        we can update memory confidence based on whether it helped or not.

        Args:
            decision_id: The trade/decision ID
            memory_ids: List of memory IDs that were retrieved
            collection: Which collection the memories came from
            agent: Which agent used the memories
        """
        if self.memory_usage_tracker and memory_ids:
            self.memory_usage_tracker.track_usage(
                trade_id=decision_id,
                memory_ids=memory_ids,
                memory_collection=collection,
                agent_name=agent,
            )

    def get_meta_insights(
        self,
        bull_signal: str = "neutral",
        bear_signal: str = "neutral",
        market_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get meta-pattern insights for decision-making.

        This provides higher-level insights from cross-agent analysis:
        - Bull/bear agreement patterns
        - Regime-specific performance

        Args:
            bull_signal: Bull researcher's signal (bullish/bearish/neutral)
            bear_signal: Bear researcher's signal (bullish/bearish/neutral)
            market_regime: Current market regime

        Returns:
            Dict with insights and recommendations
        """
        if not self.meta_pattern_learning:
            return {"message": "Meta-pattern learning not available"}

        return self.meta_pattern_learning.get_meta_insights_for_decision(
            bull_signal=bull_signal,
            bear_signal=bear_signal,
            market_regime=market_regime,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory systems.

        Useful for monitoring and debugging the learning system.
        """
        stats = {}

        if self.bull_memory:
            stats["bull_memory"] = self.bull_memory.get_memory_stats()
        if self.bear_memory:
            stats["bear_memory"] = self.bear_memory.get_memory_stats()
        if self.trader_memory:
            stats["trader_memory"] = self.trader_memory.get_memory_stats()
        if self.invest_judge_memory:
            stats["invest_judge_memory"] = self.invest_judge_memory.get_memory_stats()
        if self.risk_manager_memory:
            stats["risk_manager_memory"] = self.risk_manager_memory.get_memory_stats()
        if self.smc_pattern_memory:
            stats["smc_patterns"] = self.smc_pattern_memory.get_memory_stats()
        if self.memory_usage_tracker:
            stats["memory_usage"] = self.memory_usage_tracker.get_validation_stats()
        if self.meta_pattern_learning:
            stats["meta_patterns"] = self.meta_pattern_learning.get_stats()

        return stats

    def process_signal(self, full_signal, current_price: float = None, structured_decision: dict = None):
        """Process a signal to extract the core decision.

        Args:
            full_signal: Full text signal from the Risk Manager
            current_price: Current market price for validation
            structured_decision: Pre-structured decision dict from Risk Manager (if available)

        Returns:
            Processed signal dictionary with signal, confidence, prices, etc.
        """
        return self.signal_processor.process_signal(full_signal, current_price, structured_decision)

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol from MT5."""
        try:
            price_data = get_mt5_current_price(symbol)
            if price_data and "bid" in price_data:
                return price_data["bid"]
        except Exception:
            pass
        return None
