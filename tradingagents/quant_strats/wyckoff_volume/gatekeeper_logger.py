"""
Wyckoff Gatekeeper Logger — file-based + JSONL structured logging.

Every LLM verdict is logged twice:
1. Human-readable summary → file logger (date-stamped .log files)
2. Structured JSON record → JSONL file (one object per line, for backtest)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tradingagents.agents.analysts.quant_utils import create_quant_logger


VERDICTS_DIR = Path(
    os.path.dirname(__file__), "..", "..", "..", "logs", "wyckoff_gatekeeper", "verdicts"
)


class GatekeeperLogger:
    """Logs Wyckoff gatekeeper verdicts for audit and backtest replay."""

    def __init__(self):
        self._logger = create_quant_logger(
            "wyckoff_gatekeeper", "wyckoff_gatekeeper"
        )
        VERDICTS_DIR.mkdir(parents=True, exist_ok=True)

    def log_verdict(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        xgb_prob: float,
        verdict: dict,
        context_snapshot: str,
        latency_ms: float,
    ) -> None:
        """
        Log a gatekeeper verdict.

        Args:
            symbol: Trading symbol (e.g. "XAUUSD")
            timeframe: Chart timeframe (e.g. "D1")
            direction: Signal direction ("BUY" or "SELL")
            xgb_prob: XGBoost probability score
            verdict: Parsed verdict dict (from WyckoffVerdict model)
            context_snapshot: Full context string sent to LLM
            latency_ms: LLM call latency in milliseconds
        """
        now = datetime.now(timezone.utc)
        verdict_str = verdict.get("verdict", "UNKNOWN")
        confidence = verdict.get("confidence", 0.0)
        phase = verdict.get("wyckoff_phase", "unknown")
        reasoning = verdict.get("reasoning", "")

        # --- Human-readable log ---
        self._logger.info(
            f"\n{'='*70}\n"
            f"WYCKOFF GATEKEEPER — {symbol} {timeframe}\n"
            f"{'='*70}\n"
            f"Direction: {direction} | Prob: {xgb_prob:.2%}\n"
            f"Verdict: {verdict_str} | Confidence: {confidence:.2f}\n"
            f"Phase: {phase}\n"
            f"Reasoning: {reasoning}\n"
            f"Latency: {latency_ms:.0f}ms\n"
            f"{'='*70}"
        )

        # Log full context at DEBUG level
        self._logger.debug(
            f"\n--- CONTEXT SNAPSHOT ---\n{context_snapshot}\n"
            f"--- END CONTEXT ---"
        )

        # --- Structured JSONL ---
        record = {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "xgb_prob": round(xgb_prob, 4),
            "verdict": verdict_str,
            "confidence": verdict.get("confidence"),
            "wyckoff_phase": phase,
            "phase_confidence": verdict.get("phase_confidence"),
            "effort_result_assessment": verdict.get("effort_result_assessment"),
            "bar_quality": verdict.get("bar_quality"),
            "key_signals_supporting": verdict.get("key_signals_supporting", []),
            "key_signals_against": verdict.get("key_signals_against", []),
            "reasoning": reasoning,
            "hold_condition": verdict.get("hold_condition"),
            "latency_ms": round(latency_ms, 1),
        }

        jsonl_path = VERDICTS_DIR / f"{now.strftime('%Y%m%d')}.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            self._logger.error(f"Failed to write JSONL verdict: {e}")
