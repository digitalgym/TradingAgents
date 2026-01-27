# TradingAgents/graph/signal_processing.py

import re
import json
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(
        self,
        full_signal: str,
        current_price: float = None,
        structured_decision: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Process a full trading signal to extract the core decision and details.

        If a structured_decision dict is provided (from Risk Manager's structured output),
        it is used directly, saving an LLM call. Otherwise, falls back to LLM parsing.

        Args:
            full_signal: Complete trading signal text (used for display and fallback)
            current_price: Optional current market price for validation
            structured_decision: Pre-structured decision from Risk Manager (if available)

        Returns:
            Dictionary with signal, confidence, prices, and rationale
        """
        # If we have a structured decision from the Risk Manager, use it directly
        if structured_decision is not None:
            return self._process_structured_decision(
                structured_decision, full_signal, current_price
            )

        # Fall back to LLM parsing for unstructured text
        return self._parse_with_llm(full_signal, current_price)

    def _process_structured_decision(
        self,
        structured: Dict[str, Any],
        full_signal: str,
        current_price: float = None
    ) -> dict:
        """
        Process a pre-structured decision from the Risk Manager.

        This skips the LLM parsing step since we already have guaranteed schema-compliant data.
        """
        result = {
            "signal": structured.get("signal", "HOLD"),
            "confidence": structured.get("confidence"),
            "entry_price": structured.get("entry_price"),
            "stop_loss": structured.get("stop_loss"),
            "take_profit": structured.get("take_profit"),
            "take_profit_2": structured.get("take_profit_2"),
            "risk_level": structured.get("risk_level"),
            "risk_reward_ratio": structured.get("risk_reward_ratio"),
            "rationale": structured.get("rationale"),
            "key_risks": structured.get("key_risks"),
            "key_catalysts": structured.get("key_catalysts"),
            "position_size_recommendation": structured.get("position_size_recommendation"),
            "full_response": full_signal[:2000] if full_signal else "",
            "from_structured_output": True,  # Flag indicating no LLM parsing was needed
        }

        # Validate prices against current price if available
        if current_price:
            result = self._validate_prices(result, current_price)

        return result

    def _parse_with_llm(self, full_signal: str, current_price: float = None) -> dict:
        """
        Parse an unstructured signal using LLM (fallback method).

        This is used when the Risk Manager didn't produce structured output.
        """
        # Build context with current price if available
        price_context = ""
        if current_price:
            price_context = f"\n\nIMPORTANT: The current market price is {current_price:.5f}. Entry, stop loss, and take profit prices MUST be within a reasonable range of this price (typically within 5-10%). If specific prices are not explicitly mentioned in the report, use null rather than guessing."

        # Extract structured data using LLM
        messages = [
            (
                "system",
                f"""You are an assistant that extracts trading decisions from analyst reports.
Extract the following information and return as JSON:
{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": 0.0 to 1.0 (your confidence in this signal),
  "entry_price": number or null,
  "stop_loss": number or null,
  "take_profit": number or null,
  "rationale": "brief 1-2 sentence summary of the main reasoning"
}}

CRITICAL: Only extract prices that are EXPLICITLY mentioned in the report.
If specific prices are not clearly stated, you MUST use null - do NOT guess or make up prices.{price_context}
Respond ONLY with the JSON, no other text.""",
            ),
            ("human", full_signal),
        ]

        try:
            response = self.quick_thinking_llm.invoke(messages).content

            # Try to parse JSON from response
            # Handle potential markdown code blocks
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: try simple signal extraction
                result = self._extract_basic_signal(full_signal)

            # Ensure required fields exist
            result.setdefault("signal", "HOLD")
            result.setdefault("confidence", None)
            result.setdefault("entry_price", None)
            result.setdefault("stop_loss", None)
            result.setdefault("take_profit", None)
            result.setdefault("rationale", None)
            result["from_structured_output"] = False

            # Validate prices against current price if available
            if current_price:
                result = self._validate_prices(result, current_price)

            # Include full response for display
            result["full_response"] = full_signal[:2000] if full_signal else ""

            return result

        except Exception as e:
            # Fallback on any error
            return self._extract_basic_signal(full_signal)

    def _validate_prices(self, result: dict, current_price: float) -> dict:
        """
        Validate extracted prices against current market price.
        Prices that are wildly different from current price are likely hallucinated.

        Args:
            result: Extracted trading decision
            current_price: Current market price

        Returns:
            Result with invalid prices set to None
        """
        # Maximum deviation from current price (50% - very generous to allow for volatile markets)
        max_deviation = 0.5

        for price_field in ["entry_price", "stop_loss", "take_profit"]:
            price = result.get(price_field)
            if price is not None and price > 0:
                deviation = abs(price - current_price) / current_price
                if deviation > max_deviation:
                    # Price is too far from current market price - likely hallucinated
                    result[price_field] = None

        return result

    def _extract_basic_signal(self, full_signal: str) -> dict:
        """Extract basic signal when JSON parsing fails."""
        signal = "HOLD"
        full_upper = full_signal.upper() if full_signal else ""

        if "BUY" in full_upper and "SELL" not in full_upper:
            signal = "BUY"
        elif "SELL" in full_upper and "BUY" not in full_upper:
            signal = "SELL"
        elif "STRONG BUY" in full_upper:
            signal = "BUY"
        elif "STRONG SELL" in full_upper:
            signal = "SELL"

        return {
            "signal": signal,
            "confidence": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "rationale": None,
            "full_response": full_signal[:2000] if full_signal else ""
        }
