"""
Base Quant Analyst

Provides shared infrastructure for all quant analyst agents:
- Logger setup (file-based, date-stamped)
- State extraction from graph
- LLM structured output handling
- Common decision building patterns

Subclasses only need to implement:
- LOGGER_NAME: str - name for the logger (e.g., "smc_quant")
- LOG_SUBDIR: str - subdirectory in logs/ (e.g., "smc_quant_prompts")
- build_prompt(self, state, context) -> str
- get_strategy_name() -> str
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any

from tradingagents.schemas import QuantAnalystDecision, QuantSignalType, RiskLevel
from tradingagents.dataflows.smc_trade_plan import safe_get


class BaseQuantAnalyst(ABC):
    """Base class for all quant analyst agents."""

    # Subclasses must define these
    LOGGER_NAME: str = "quant"
    LOG_SUBDIR: str = "quant_prompts"

    def __init__(self, llm, use_structured_output: bool = True):
        """
        Initialize the quant analyst.

        Args:
            llm: The language model to use
            use_structured_output: If True, use LLM structured output for JSON
        """
        self.llm = llm
        self.use_structured_output = use_structured_output
        self._logger = None
        self._structured_llm = None

        # Setup structured output if supported
        if use_structured_output:
            try:
                self._structured_llm = llm.with_structured_output(QuantAnalystDecision)
            except Exception as e:
                print(f"Warning: Structured output not supported for {self.LOGGER_NAME}, falling back to free-form: {e}")

    def get_logger(self) -> logging.Logger:
        """Get or create the logger for this quant analyst."""
        if self._logger is None:
            self._logger = logging.getLogger(self.LOGGER_NAME)
            self._logger.setLevel(logging.DEBUG)

            # Create logs directory
            log_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "logs", self.LOG_SUBDIR
            )
            os.makedirs(log_dir, exist_ok=True)

            # Date-stamped log file
            log_file = os.path.join(
                log_dir, f"{self.LOGGER_NAME}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter("%(asctime)s | %(message)s")
            file_handler.setFormatter(formatter)

            # Avoid duplicate handlers
            if not self._logger.handlers:
                self._logger.addHandler(file_handler)

        return self._logger

    def extract_state_context(self, state: dict) -> Dict[str, Any]:
        """
        Extract common state fields used by all quants.

        Returns dict with:
            ticker, current_date, current_price, market_report,
            smc_analysis, smc_context, market_regime, volatility_regime,
            trading_session, position_context
        """
        return {
            "ticker": state.get("company_of_interest"),
            "current_date": state.get("trade_date"),
            "current_price": state.get("current_price"),
            "market_report": state.get("market_report") or "",
            "smc_analysis": state.get("smc_analysis"),
            "smc_context": state.get("smc_context") or "",
            "market_regime": state.get("market_regime") or "unknown",
            "volatility_regime": state.get("volatility_regime") or "normal",
            "trading_session": state.get("trading_session") or "unknown",
            "position_context": state.get("position_context") or "",
        }

    def log_prompt_and_response(self, prompt: str, response: Any, ticker: str):
        """Log the prompt and response for debugging."""
        logger = self.get_logger()
        logger.debug(f"\n{'='*80}\n{ticker} PROMPT:\n{prompt}\n{'='*80}")
        logger.debug(f"\n{ticker} RESPONSE:\n{response}\n{'='*80}\n")

    def create_hold_decision(self, reason: str, symbol: str = "UNKNOWN") -> QuantAnalystDecision:
        """Create a HOLD decision with explanation."""
        return QuantAnalystDecision(
            symbol=symbol,
            signal=QuantSignalType.HOLD,
            confidence=0.5,
            entry_price=None,
            stop_loss=None,
            profit_target=None,
            risk_level=RiskLevel.MEDIUM,
            invalidation_condition="N/A",
            justification=reason,
        )

    @abstractmethod
    def build_prompt(self, state: dict, context: Dict[str, Any]) -> str:
        """
        Build the LLM prompt for this strategy.

        Args:
            state: Full graph state
            context: Extracted context from extract_state_context()

        Returns:
            The prompt string to send to the LLM
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name (e.g., 'SMC Quant', 'Breakout Quant')."""
        pass

    def invoke_llm(self, prompt: str) -> QuantAnalystDecision:
        """
        Invoke the LLM and return a QuantAnalystDecision.

        Uses structured output if available, otherwise parses free-form.
        """
        if self._structured_llm:
            return self._structured_llm.invoke(prompt)
        else:
            # Fall back to free-form parsing
            response = self.llm.invoke(prompt)
            # Subclasses can override this for custom parsing
            return self.parse_freeform_response(response)

    def parse_freeform_response(self, response) -> QuantAnalystDecision:
        """
        Parse a free-form LLM response into QuantAnalystDecision.
        Override in subclass for custom parsing logic.
        """
        # Default: return HOLD if parsing fails
        return self.create_hold_decision(
            f"Could not parse LLM response for {self.get_strategy_name()}"
        )

    def __call__(self, state: dict) -> dict:
        """
        Process state and return quant analysis.

        This is the main entry point, called by the graph.
        """
        context = self.extract_state_context(state)
        ticker = context["ticker"] or "UNKNOWN"

        try:
            prompt = self.build_prompt(state, context)
            decision = self.invoke_llm(prompt)
            self.log_prompt_and_response(prompt, decision, ticker)

            return {"quant_signal": decision}

        except Exception as e:
            self.get_logger().error(f"Error in {self.get_strategy_name()} for {ticker}: {e}")
            return {
                "quant_signal": self.create_hold_decision(
                    f"{self.get_strategy_name()} error: {str(e)}",
                    symbol=ticker,
                )
            }


# Re-export safe_get for backwards compatibility
__all__ = ["BaseQuantAnalyst", "safe_get"]
