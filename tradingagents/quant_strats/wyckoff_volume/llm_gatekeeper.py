"""
Wyckoff LLM Gatekeeper — core module.

Sits between XGBoost signal generation and order execution. Receives a
structured market context snapshot, calls an LLM for Wyckoff interpretation,
and returns a structured verdict: APPROVE, REJECT, or HOLD.
"""

import asyncio
import logging
import time
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from tradingagents.llm_factory import get_llm
from tradingagents.quant_strats.features.wyckoff import WyckoffFeatures
from tradingagents.quant_strats.features.technical import TechnicalFeatures
from tradingagents.quant_strats.wyckoff_volume.gatekeeper_prompts import (
    WYCKOFF_SYSTEM_PROMPT,
    build_context_snapshot,
)
from tradingagents.quant_strats.wyckoff_volume.gatekeeper_logger import GatekeeperLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Verdict schema
# --------------------------------------------------------------------------

class WyckoffVerdict(BaseModel):
    """Structured LLM verdict from the Wyckoff gatekeeper."""

    verdict: Literal["APPROVE", "REJECT", "HOLD"] = Field(
        description="APPROVE to proceed, REJECT to skip, HOLD to wait for next bar"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the verdict (0-1)"
    )
    wyckoff_phase: str = Field(
        description="Identified Wyckoff phase: accumulation, markup, distribution, markdown, or uncertain"
    )
    phase_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the phase identification (0-1)"
    )
    key_signals_supporting: List[str] = Field(
        default_factory=list,
        description="Wyckoff signals that support the trade"
    )
    key_signals_against: List[str] = Field(
        default_factory=list,
        description="Wyckoff signals that contradict the trade"
    )
    effort_result_assessment: Literal["confirming", "diverging", "neutral"] = Field(
        description="Whether volume effort confirms the price result"
    )
    bar_quality: Literal["strong", "moderate", "weak"] = Field(
        description="Quality of the current bar for the signal direction"
    )
    reasoning: str = Field(
        description="2-3 sentence Wyckoff rationale for the verdict"
    )
    hold_condition: Optional[str] = Field(
        default=None,
        description="If HOLD, what to look for on the next bar"
    )


# --------------------------------------------------------------------------
# Default safe verdict (used on failure)
# --------------------------------------------------------------------------

def _safe_reject(reason: str) -> WyckoffVerdict:
    """Return a safe REJECT verdict — never trade on failure."""
    return WyckoffVerdict(
        verdict="REJECT",
        confidence=0.0,
        wyckoff_phase="uncertain",
        phase_confidence=0.0,
        key_signals_supporting=[],
        key_signals_against=[],
        effort_result_assessment="neutral",
        bar_quality="weak",
        reasoning=reason,
        hold_condition=None,
    )


# --------------------------------------------------------------------------
# Gatekeeper class
# --------------------------------------------------------------------------

class WyckoffGatekeeper:
    """
    LLM-based Wyckoff gatekeeper for trade signal filtering.

    Computes Wyckoff features, builds a context snapshot, sends to an LLM,
    and returns a structured verdict.
    """

    # Minimum LLM confidence to allow APPROVE (below this -> HOLD)
    # Backtest showed LLM consistently returns 0.85+ confidence, so 0.70
    # is a safety net for edge cases without interfering with normal operation.
    MIN_APPROVE_CONFIDENCE = 0.70

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise the gatekeeper.

        Args:
            config: Optional config dict. If provided, should contain
                    'llm_provider' and 'deep_think_llm' keys. Defaults
                    to the project's default LLM (xAI Grok).
        """
        from tradingagents.default_config import DEFAULT_CONFIG
        gatekeeper_config = {
            "llm_provider": DEFAULT_CONFIG["llm_provider"],
            "deep_think_llm": DEFAULT_CONFIG["quick_think_llm"],
            "quick_think_llm": DEFAULT_CONFIG["quick_think_llm"],
            "backend_url": DEFAULT_CONFIG.get("backend_url", ""),
        }
        if config:
            gatekeeper_config.update(config)

        llm = get_llm(gatekeeper_config, tier="deep")
        # Bind temperature=0 for deterministic output
        self._llm = llm.bind(temperature=0)
        self._structured_llm = self._llm.with_structured_output(WyckoffVerdict)

        self._wyckoff_features = WyckoffFeatures()
        self._technical_features = TechnicalFeatures()
        self._gatekeeper_logger = GatekeeperLogger()

    def evaluate(
        self,
        df: pd.DataFrame,
        xgb_prob: float,
        direction: str,
        symbol: str = "XAUUSD",
        timeframe: str = "D1",
        htf_bias: Optional[str] = None,
    ) -> WyckoffVerdict:
        """
        Evaluate a signal through the Wyckoff gatekeeper.

        Args:
            df: OHLCV DataFrame (needs 60+ bars for feature warmup)
            xgb_prob: XGBoost probability score (0-1)
            direction: "BUY" or "SELL"
            symbol: Trading symbol
            timeframe: Chart timeframe
            htf_bias: Higher timeframe bias string

        Returns:
            WyckoffVerdict with APPROVE, REJECT, or HOLD verdict
        """
        start_time = time.time()

        try:
            # Compute features
            wyckoff_df = self._wyckoff_features.compute(df)
            technical_df = self._technical_features.compute(df)

            wyckoff_row = wyckoff_df.iloc[-1]
            technical_row = technical_df.iloc[-1]

            # Build context snapshot
            context = build_context_snapshot(
                wyckoff_row=wyckoff_row,
                technical_row=technical_row,
                xgb_prob=xgb_prob,
                direction=direction,
                symbol=symbol,
                timeframe=timeframe,
                htf_bias=htf_bias,
            )

            # Call LLM
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=WYCKOFF_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]

            verdict = self._structured_llm.invoke(messages)

            # Enforce minimum confidence for APPROVE
            if (
                verdict.verdict == "APPROVE"
                and verdict.confidence < self.MIN_APPROVE_CONFIDENCE
            ):
                verdict = verdict.model_copy(update={
                    "verdict": "HOLD",
                    "hold_condition": (
                        f"LLM confidence {verdict.confidence:.2f} below "
                        f"threshold ({self.MIN_APPROVE_CONFIDENCE})"
                    ),
                })

            latency_ms = (time.time() - start_time) * 1000

            # Log verdict
            self._gatekeeper_logger.log_verdict(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                xgb_prob=xgb_prob,
                verdict=verdict.model_dump(),
                context_snapshot=context,
                latency_ms=latency_ms,
            )

            return verdict

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Wyckoff gatekeeper error: {e}", exc_info=True)

            reject = _safe_reject(f"Gatekeeper error: {e}")

            self._gatekeeper_logger.log_verdict(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                xgb_prob=xgb_prob,
                verdict=reject.model_dump(),
                context_snapshot="<error before context build>",
                latency_ms=latency_ms,
            )

            return reject

    async def async_evaluate(
        self,
        df: pd.DataFrame,
        xgb_prob: float,
        direction: str,
        symbol: str = "XAUUSD",
        timeframe: str = "D1",
        htf_bias: Optional[str] = None,
    ) -> WyckoffVerdict:
        """Async wrapper for evaluate() — runs LLM call in a thread."""
        return await asyncio.to_thread(
            self.evaluate, df, xgb_prob, direction, symbol, timeframe, htf_bias
        )

    def run_gatekeeper(
        self,
        features: dict,
        xgb_prob: float,
        direction: str,
        symbol: str = "XAUUSD",
        timeframe: str = "D1",
    ) -> dict:
        """
        Convenience wrapper matching the user's original API.

        Accepts a features dict, builds a minimal DataFrame, and returns
        the verdict as a plain dict.
        """
        # Build a minimal single-row DataFrame from features dict
        # The features dict should contain OHLCV keys at minimum
        required = ["open", "high", "low", "close", "volume"]
        missing = [k for k in required if k not in features]
        if missing:
            return _safe_reject(
                f"Missing required OHLCV keys: {missing}"
            ).model_dump()

        df = pd.DataFrame([features])
        # Single bar is not enough for feature warmup — caller should
        # pass full OHLCV DataFrame via evaluate() instead
        if len(df) < 60:
            logger.warning(
                "run_gatekeeper: single-row dict passed, features will have NaN. "
                "Use evaluate() with full OHLCV DataFrame for production."
            )

        verdict = self.evaluate(
            df=df,
            xgb_prob=xgb_prob,
            direction=direction,
            symbol=symbol,
            timeframe=timeframe,
        )
        return verdict.model_dump()
