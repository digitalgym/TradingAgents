# TradingAgents/graph/propagation.py

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        current_price: float = None,
        market_regime: str = None,
        volatility_regime: str = None,
        regime_description: str = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        Args:
            company_name: Ticker symbol or company name
            trade_date: Date to analyze
            current_price: Current broker price for the symbol (important for broker-specific pricing)
            market_regime: Detected market regime (trending-up, trending-down, ranging)
            volatility_regime: Detected volatility regime (low, normal, high, extreme)
            regime_description: Human-readable description of current regime
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "current_price": current_price,  # Broker's actual price for this symbol
            "market_regime": market_regime,  # For regime-aware decisions
            "volatility_regime": volatility_regime,  # For position sizing
            "regime_description": regime_description,  # Human-readable regime context
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "quant_report": None,
            "quant_decision": None,
            "final_trade_decision": "",
            "final_trade_decision_structured": None,
            "smc_context": None,
            "smc_analysis": None,
        }

    def get_graph_args(self, stream_mode: str = "values") -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            stream_mode: Either "values" (full state) or "updates" (node outputs only)
        """
        return {
            "stream_mode": stream_mode,
            "config": {"recursion_limit": self.max_recur_limit},
        }
