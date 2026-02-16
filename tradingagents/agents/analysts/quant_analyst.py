"""
Quant Analyst Agent

A systematic quant trader agent with strict risk discipline.
Receives all available technical data (SMC, RSI, MACD, indicators, price action)
and outputs structured trade decisions compatible with the trade execution modal.

This agent:
- Ignores narratives and trades only price, volume, and technical indicators
- Enforces strict risk management (1-2% risk per trade, max 3 positions)
- Never pyramids or averages down
- Always pre-defines profit target, stop loss, and invalidation condition
"""

import logging
import os
from datetime import datetime
from typing import Optional
from tradingagents.schemas import QuantAnalystDecision, RiskLevel


def _safe_get(obj, attr, default=None):
    """Safely get attribute from dict or dataclass object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# Set up quant prompt logger
_quant_logger = None

def _get_quant_logger():
    """Get or create the quant prompt logger."""
    global _quant_logger
    if _quant_logger is None:
        _quant_logger = logging.getLogger("quant_prompts")
        _quant_logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "quant_prompts")
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler with date-based filename
        log_file = os.path.join(log_dir, f"quant_prompts_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler (avoid duplicates)
        if not _quant_logger.handlers:
            _quant_logger.addHandler(file_handler)

    return _quant_logger


def create_quant_analyst(llm, use_structured_output: bool = True):
    """
    Create a quant analyst node that produces systematic trade decisions.

    Args:
        llm: The language model to use for analysis
        use_structured_output: If True, uses LLM structured output for guaranteed JSON

    Returns:
        A function that processes state and returns the quant analysis
    """

    # Create structured output LLM wrapper if supported
    structured_llm = None
    if use_structured_output:
        try:
            structured_llm = llm.with_structured_output(QuantAnalystDecision)
        except Exception as e:
            print(f"Warning: Structured output not supported for quant analyst, falling back to free-form: {e}")
            structured_llm = None

    def quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Gather all available technical data
        smc_context = state.get("smc_context") or ""
        smc_analysis = state.get("smc_analysis") or {}
        market_report = state.get("market_report") or ""

        # Extract regime information
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        trading_session = state.get("trading_session") or "unknown"

        # Build the comprehensive data context
        data_context = _build_data_context(
            ticker=ticker,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=market_report,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=trading_session,
            current_date=current_date,
        )

        # Build the quant analyst prompt
        system_prompt = _build_quant_prompt(data_context)

        # Log the prompt being sent to LLM
        logger = _get_quant_logger()
        logger.info(f"\n{'='*80}\nQUANT ANALYSIS REQUEST - {ticker}\n{'='*80}")
        logger.info(f"Symbol: {ticker} | Date: {current_date} | Price: {current_price}")
        logger.info(f"\n--- DATA CONTEXT ---\n{data_context}")
        logger.info(f"\n--- FULL PROMPT ---\n{system_prompt}")
        logger.info(f"\n{'='*80}\n")

        if structured_llm is not None:
            try:
                decision: QuantAnalystDecision = structured_llm.invoke(system_prompt)

                # Log the structured response
                logger.info(f"--- LLM RESPONSE (Structured) ---")
                logger.info(f"Signal: {decision.signal}")
                logger.info(f"Confidence: {decision.confidence}")
                logger.info(f"Entry: {decision.entry_price} | SL: {decision.stop_loss} | TP: {decision.profit_target}")
                logger.info(f"Justification: {decision.justification}")
                logger.info(f"Invalidation: {decision.invalidation_condition}")
                logger.info(f"\n{'='*80}\n")

                # Generate human-readable report
                report = _format_quant_report(decision)

                return {
                    "quant_report": report,
                    "quant_decision": decision.model_dump(),
                }
            except Exception as e:
                logger.error(f"Structured output failed: {e}")
                print(f"Structured output failed for quant analyst: {e}")

        # Fallback to unstructured output
        response = llm.invoke(system_prompt)
        report = response.content if hasattr(response, 'content') else str(response)

        # Log unstructured response
        logger.info(f"--- LLM RESPONSE (Unstructured) ---")
        logger.info(f"{report[:2000]}...")  # Truncate if very long
        logger.info(f"\n{'='*80}\n")

        return {
            "quant_report": report,
            "quant_decision": None,
        }

    return quant_analyst_node


def _build_data_context(
    ticker: str,
    current_price: Optional[float],
    smc_context: str,
    smc_analysis: dict,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    trading_session: str,
    current_date: str,
) -> str:
    """Build comprehensive data context for the quant analyst."""

    sections = []

    # Current price information
    if current_price:
        sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price (Broker)**: {current_price:.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
""")
    else:
        sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
""")

    # SMC Analysis
    if smc_context:
        sections.append(f"""## SMART MONEY CONCEPTS (SMC) ANALYSIS
{smc_context}
""")

    # Extract key SMC levels if available
    if smc_analysis:
        levels_text = _extract_smc_levels(smc_analysis, current_price)
        if levels_text:
            sections.append(f"""## KEY SMC LEVELS
{levels_text}
""")

    # Technical indicators from market report
    if market_report:
        sections.append(f"""## TECHNICAL ANALYSIS (Indicators)
{market_report}
""")

    return "\n".join(sections)


def _extract_smc_levels(smc_analysis: dict, current_price: Optional[float]) -> str:
    """Extract key SMC levels from analysis for quick reference."""
    lines = []

    # Order blocks - handle both dict and list formats
    obs = smc_analysis.get("order_blocks", {})
    if isinstance(obs, dict):
        for ob_type in ["bullish", "bearish"]:
            for ob in obs.get(ob_type, [])[:3]:  # Top 3 of each type
                top = _safe_get(ob, "top", 0)
                bottom = _safe_get(ob, "bottom", 0)
                strength = _safe_get(ob, "strength", 0.5)
                mitigated = _safe_get(ob, "mitigated", False)
                if not mitigated and top and bottom:
                    lines.append(f"- {ob_type.upper()} OB: {bottom:.5f} - {top:.5f} (strength: {strength:.0%})")
    elif isinstance(obs, list):
        for ob in obs[:6]:  # Top 6 total
            ob_type = _safe_get(ob, "type", "unknown")
            top = _safe_get(ob, "top", 0)
            bottom = _safe_get(ob, "bottom", 0)
            strength = _safe_get(ob, "strength", 0.5)
            mitigated = _safe_get(ob, "mitigated", False)
            if not mitigated and top and bottom:
                lines.append(f"- {ob_type.upper()} OB: {bottom:.5f} - {top:.5f} (strength: {strength:.0%})")

    # FVGs - handle both dict and list formats
    fvgs = smc_analysis.get("fair_value_gaps", {})
    if isinstance(fvgs, dict):
        for fvg_type in ["bullish", "bearish"]:
            for fvg in fvgs.get(fvg_type, [])[:2]:  # Top 2 of each type
                top = _safe_get(fvg, "top", 0)
                bottom = _safe_get(fvg, "bottom", 0)
                if top and bottom:
                    lines.append(f"- {fvg_type.upper()} FVG: {bottom:.5f} - {top:.5f}")
    elif isinstance(fvgs, list):
        for fvg in fvgs[:4]:  # Top 4 total
            fvg_type = _safe_get(fvg, "type", "unknown")
            top = _safe_get(fvg, "top", 0)
            bottom = _safe_get(fvg, "bottom", 0)
            if top and bottom:
                lines.append(f"- {fvg_type.upper()} FVG: {bottom:.5f} - {top:.5f}")

    # Liquidity zones
    liquidity = smc_analysis.get("liquidity_zones", [])
    for lz in liquidity[:4]:
        lz_price = _safe_get(lz, "price", 0)
        lz_type = _safe_get(lz, "type", "unknown")
        lz_strength = _safe_get(lz, "strength", 50)
        if lz_price:
            lines.append(f"- {lz_type.upper()} liquidity: {lz_price:.5f} (strength: {lz_strength})")

    # ATR if available
    atr = smc_analysis.get("atr")
    if atr:
        lines.append(f"- ATR: {atr:.5f}")

    # Bias
    bias = smc_analysis.get("bias")
    if bias:
        lines.append(f"- Overall Bias: {bias.upper()}")

    return "\n".join(lines) if lines else ""


def _build_quant_prompt(data_context: str) -> str:
    """Build the complete quant analyst prompt."""

    return f"""You are a systematic quant trader with strict risk discipline. Your only goal is to maximize long-term PnL while surviving drawdowns.

## RULES YOU MUST NEVER BREAK

1. **Risk no more than 1-2% of account value per trade** (risk_usd = position size × distance to stop-loss)
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and one clear invalidation condition before entry**
5. **Hold only when current plan remains valid; never flip-flop without new high-conviction signal**
6. **Ignore narratives; trade only price, volume, and the provided technical indicators**
7. **Fees are 0.025% maker / 0.05% taker + funding - size accordingly**
8. **Leverage is a tool, not a goal - default 5-20x, higher only with extreme conviction**

## STOP LOSS VALIDATION (CRITICAL)
- For BUY orders: Stop loss MUST be BELOW entry price
- For SELL orders: Stop loss MUST be ABOVE entry price
- If you cannot identify a valid stop loss placement, output "hold"

## CURRENT MARKET DATA
{data_context}

## YOUR TASK

Analyze the provided technical data and make a systematic trading decision.

Think step-by-step:
1. What is the overall market structure and bias?
2. Are there high-probability entry zones (OB, FVG) near current price?
3. Where would stop loss be placed? Is the R:R acceptable (>1.5:1)?
4. What would invalidate this setup?
5. What is your confidence level based on confluence?

Then provide your decision. For "hold" or "close", explain why in the justification.

Remember:
- Only enter trades with clear edge
- Wait for price to come to your levels
- No FOMO entries
- Discipline > prediction"""


def _format_quant_report(decision: QuantAnalystDecision) -> str:
    """Format the quant decision into a human-readable report."""
    lines = [
        f"## QUANT ANALYST DECISION: **{decision.signal.value.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
    ]

    if decision.signal.value in ["buy_to_enter", "sell_to_enter"]:
        lines.append("### Trade Parameters")
        if decision.entry_price:
            lines.append(f"- **Entry Price**: {decision.entry_price}")
        if decision.stop_loss:
            lines.append(f"- **Stop Loss**: {decision.stop_loss}")
        if decision.profit_target:
            lines.append(f"- **Take Profit**: {decision.profit_target}")
        if decision.risk_reward_ratio:
            lines.append(f"- **Risk/Reward**: {decision.risk_reward_ratio:.2f}")
        if decision.leverage:
            lines.append(f"- **Leverage**: {decision.leverage}x")
        if decision.risk_usd:
            lines.append(f"- **Risk (USD)**: ${decision.risk_usd:.2f}")
        if decision.quantity:
            lines.append(f"- **Quantity**: {decision.quantity}")
        lines.append("")

    lines.extend([
        "### Justification",
        decision.justification,
        "",
        "### Invalidation Condition",
        decision.invalidation_condition,
        "",
    ])

    if decision.risk_level:
        lines.append(f"**Risk Level**: {decision.risk_level}")

    return "\n".join(lines)


def get_quant_decision_for_modal(quant_decision: dict) -> dict:
    """
    Convert a quant decision dict to trade modal format.

    Args:
        quant_decision: The quant_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not quant_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = quant_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    return {
        "symbol": quant_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "suggestedEntry": quant_decision.get("entry_price"),
        "suggestedStopLoss": quant_decision.get("stop_loss"),
        "suggestedTakeProfit": quant_decision.get("profit_target"),
        "rationale": f"{quant_decision.get('justification', '')}. Invalidation: {quant_decision.get('invalidation_condition', '')}",
        "confidence": quant_decision.get("confidence", 0.5),
    }
