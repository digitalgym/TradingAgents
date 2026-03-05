"""
Volume Profile Quant Analyst Agent

A systematic quant trader agent focused on Volume Profile analysis.
Uses POC, Value Area, HVN/LVN to identify high-probability entry/exit zones.

Key Volume Profile Concepts:
- POC (Point of Control): Price level with highest volume - acts as magnet
- Value Area: Range containing 70% of volume - fair value zone
- HVN (High Volume Nodes): Strong support/resistance levels
- LVN (Low Volume Nodes): Fast-move zones where price accelerates

Trading Logic:
- Mean reversion when price is outside value area
- Breakout trades when price escapes value area with volume
- POC as magnet for targets
- HVN as support/resistance for stops
- LVN as zones where price moves quickly (use for entries after sweep)
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from tradingagents.schemas import QuantAnalystDecision, RiskLevel
from tradingagents.indicators.volume_profile import (
    VolumeProfileAnalyzer,
    VolumeProfile,
    VolumeNode,
)


def _safe_get(obj, attr, default=None):
    """Safely get attribute from dict or dataclass object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# Set up VP quant prompt logger
_vp_quant_logger = None


def _get_vp_quant_logger():
    """Get or create the VP quant prompt logger."""
    global _vp_quant_logger
    if _vp_quant_logger is None:
        _vp_quant_logger = logging.getLogger("vp_quant_prompts")
        _vp_quant_logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "logs", "vp_quant_prompts"
        )
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler with date-based filename
        log_file = os.path.join(
            log_dir, f"vp_quant_prompts_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler (avoid duplicates)
        if not _vp_quant_logger.handlers:
            _vp_quant_logger.addHandler(file_handler)

    return _vp_quant_logger


def create_volume_profile_quant(llm, use_structured_output: bool = True):
    """
    Create a Volume Profile quant analyst node.

    This quant focuses on:
    - POC as price magnet and mean reversion target
    - Value Area for fair value determination
    - HVN for support/resistance levels
    - LVN for fast-move zones and breakout opportunities

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
            print(
                f"Warning: Structured output not supported for VP quant analyst, falling back to free-form: {e}"
            )
            structured_llm = None

    def vp_quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Gather technical data
        market_report = state.get("market_report") or ""
        volume_profile = state.get("volume_profile")  # VolumeProfile dataclass
        volume_profile_context = state.get("volume_profile_context") or ""

        # Extract regime information
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        trading_session = state.get("trading_session") or "unknown"

        # Build the comprehensive data context
        data_context = _build_vp_data_context(
            ticker=ticker,
            current_price=current_price,
            volume_profile=volume_profile,
            volume_profile_context=volume_profile_context,
            market_report=market_report,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=trading_session,
            current_date=current_date,
        )

        # Build the VP quant analyst prompt
        system_prompt = _build_vp_quant_prompt(data_context)

        # Log the prompt being sent to LLM
        logger = _get_vp_quant_logger()
        logger.info(
            f"\n{'='*80}\nVOLUME PROFILE QUANT ANALYSIS - {ticker}\n{'='*80}"
        )
        logger.info(f"Symbol: {ticker} | Date: {current_date} | Price: {current_price}")
        logger.info(f"\n--- DATA CONTEXT ---\n{data_context}")
        logger.info(f"\n--- FULL PROMPT ---\n{system_prompt}")
        logger.info(f"\n{'='*80}\n")

        if structured_llm is not None:
            try:
                import time as _time

                _llm_start = _time.time()
                decision: QuantAnalystDecision = structured_llm.invoke(system_prompt)
                _llm_duration = _time.time() - _llm_start

                # Log the structured response
                logger.info(
                    f"--- LLM RESPONSE (Structured) [took {_llm_duration:.1f}s] ---"
                )
                logger.info(f"Signal: {decision.signal}")
                logger.info(f"Confidence: {decision.confidence}")
                logger.info(
                    f"Entry: {decision.entry_price} | SL: {decision.stop_loss} | TP: {decision.profit_target}"
                )
                logger.info(f"Justification: {decision.justification}")
                logger.info(f"Invalidation: {decision.invalidation_condition}")

                # Log raw decision dict for debugging
                decision_dict = decision.model_dump()
                logger.info(f"Raw decision dict: {decision_dict}")
                logger.info(f"\n{'='*80}\n")

                # Generate human-readable report
                report = _format_vp_quant_report(decision)

                logger.info(
                    f"Structured output SUCCESS - returning vp_quant_decision with signal={decision.signal}"
                )
                return {
                    "vp_quant_report": report,
                    "vp_quant_decision": decision_dict,
                }
            except Exception as e:
                import traceback as _tb

                logger.error(f"Structured output failed: {e}")
                logger.error(f"Traceback:\n{_tb.format_exc()}")
                print(f"Structured output failed for VP quant analyst: {e}")

        # Fallback to unstructured output
        logger.info(f"--- Falling back to unstructured LLM call ---")
        import time as _time

        _llm_start = _time.time()
        response = llm.invoke(system_prompt)
        _llm_duration = _time.time() - _llm_start
        report = response.content if hasattr(response, "content") else str(response)

        # Log unstructured response
        logger.info(
            f"--- LLM RESPONSE (Unstructured) [took {_llm_duration:.1f}s] ---"
        )
        logger.info(f"{report[:2000]}...")  # Truncate if very long
        logger.info(f"\n{'='*80}\n")

        logger.warning(f"Returning vp_quant_decision=None (unstructured fallback)")
        return {
            "vp_quant_report": report,
            "vp_quant_decision": None,
        }

    return vp_quant_analyst_node


def _build_vp_data_context(
    ticker: str,
    current_price: Optional[float],
    volume_profile: Optional[VolumeProfile],
    volume_profile_context: str,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    trading_session: str,
    current_date: str,
) -> str:
    """Build comprehensive data context for the VP quant analyst."""

    sections = []

    # Current price information
    if current_price:
        sections.append(
            f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price**: {current_price:.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
"""
        )
    else:
        sections.append(
            f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
"""
        )

    # Volume Profile Analysis
    if volume_profile_context:
        sections.append(f"## VOLUME PROFILE ANALYSIS\n{volume_profile_context}")
    elif volume_profile:
        # Generate context from VolumeProfile dataclass
        sections.append(_format_volume_profile_for_prompt(volume_profile, current_price))

    # Technical indicators (market_report already has its own header)
    if market_report:
        sections.append(f"## TECHNICAL INDICATORS\n{market_report}")

    return "\n".join(sections)


def _format_volume_profile_for_prompt(
    profile: VolumeProfile, current_price: Optional[float]
) -> str:
    """Format VolumeProfile dataclass for prompt."""
    if not profile or profile.total_volume <= 0:
        return "## VOLUME PROFILE ANALYSIS\nInsufficient volume data.\n"

    lines = ["## VOLUME PROFILE ANALYSIS"]
    cp = current_price or profile.poc

    # POC
    poc_distance = ((cp - profile.poc) / cp * 100) if cp else 0
    if poc_distance > 0:
        poc_position = f"Price is +{poc_distance:.2f}% above POC"
    else:
        poc_position = f"Price is {poc_distance:.2f}% below POC"

    lines.append(
        f"**POC (Point of Control)**: {profile.poc:.5f} ({profile.poc_volume_pct:.1f}% of volume)"
    )
    lines.append(f"  - {poc_position} - POC acts as magnet")

    # Value Area
    if cp >= profile.value_area_low and cp <= profile.value_area_high:
        va_status = "INSIDE value area (fair value zone)"
    elif cp > profile.value_area_high:
        va_status = "ABOVE value area (premium - expect mean reversion DOWN)"
    else:
        va_status = "BELOW value area (discount - expect mean reversion UP)"

    lines.append(
        f"**Value Area**: {profile.value_area_low:.5f} - {profile.value_area_high:.5f} ({profile.value_area_pct:.0f}% of volume)"
    )
    lines.append(f"  - {va_status}")

    # HVN
    if profile.high_volume_nodes:
        lines.append("**High Volume Nodes** (support/resistance):")
        for hvn in profile.high_volume_nodes[:3]:
            dist = ((cp - hvn.price) / cp * 100) if cp else 0
            if dist > 0.1:
                lines.append(
                    f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | -{abs(dist):.2f}% below"
                )
            elif dist < -0.1:
                lines.append(
                    f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | +{abs(dist):.2f}% above"
                )
            else:
                lines.append(
                    f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | AT PRICE"
                )

    # LVN
    if profile.low_volume_nodes:
        lines.append("**Low Volume Nodes** (fast-move zones):")
        for lvn in profile.low_volume_nodes[:3]:
            dist = ((cp - lvn.price) / cp * 100) if cp else 0
            if dist > 0:
                lines.append(
                    f"  - LVN at {lvn.price:.5f} | -{abs(dist):.2f}% - price moves fast here"
                )
            else:
                lines.append(
                    f"  - LVN at {lvn.price:.5f} | +{abs(dist):.2f}% - price moves fast here"
                )

    lines.append("")
    return "\n".join(lines)


def _build_vp_quant_prompt(data_context: str) -> str:
    """Build the complete Volume Profile quant analyst prompt."""

    return f"""You are a systematic Volume Profile trader with strict risk discipline. You trade based on volume-at-price analysis to identify high-probability entries.

## VOLUME PROFILE TRADING RULES

### Core Concepts
1. **POC (Point of Control)**: The price level where most volume traded. Acts as a MAGNET - price tends to return here. Use as profit target in ranging markets.

2. **Value Area (VA)**: The price range containing 70% of volume. This is "fair value."
   - Price ABOVE VA = Premium, look for SHORTS (mean reversion down)
   - Price BELOW VA = Discount, look for LONGS (mean reversion up)
   - Price INSIDE VA = Fair value, wait for extremes or breakouts

3. **Value Area High (VAH)**: Upper boundary of value area. Acts as RESISTANCE.
4. **Value Area Low (VAL)**: Lower boundary of value area. Acts as SUPPORT.

5. **High Volume Nodes (HVN)**: Price levels with high volume.
   - Act as strong support/resistance
   - Good for stop loss placement (behind HVN)
   - Price tends to consolidate at HVN

6. **Low Volume Nodes (LVN)**: Price levels with low volume.
   - Price moves FAST through these zones
   - Avoid placing entries in LVN (will get run through)
   - Can be used to identify breakout acceleration zones

### Trading Setups

**Setup 1: Mean Reversion to POC**
- Price is significantly above/below POC
- Enter in direction of POC
- Target: POC level
- Stop: Beyond the extreme

**Setup 2: Value Area Edge Trade**
- Price touches VAH/VAL from inside
- Fade the move (short at VAH, long at VAL)
- Target: POC
- Stop: Beyond VAH/VAL by 1 ATR

**Setup 3: Value Area Breakout**
- Price closes outside VA with increasing volume
- Trade the breakout direction
- Target: Next HVN or previous day's POC
- Stop: Back inside VA

**Setup 4: HVN Bounce**
- Price approaches significant HVN
- Enter on rejection at HVN
- Target: Next HVN or POC
- Stop: Beyond the HVN

## RISK MANAGEMENT RULES (NEVER BREAK)

1. **Risk no more than 1-2% of account value per trade**
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and invalidation before entry**
5. **Place stops behind HVN levels when possible** (avoid LVN for stops)
6. **Fees are 0.025% maker / 0.05% taker + funding - size accordingly**
7. **Leverage is a tool, not a goal - default 5-20x**

## STOP LOSS VALIDATION (CRITICAL)
- For BUY orders: Stop loss MUST be BELOW entry price
- For SELL orders: Stop loss MUST be ABOVE entry price
- Prefer placing stops behind HVN (high volume = strong support/resistance)
- If you cannot identify a valid stop loss placement, output "hold"

{data_context}

## YOUR TASK

Analyze the Volume Profile data and make a systematic trading decision.

Think step-by-step:
1. Where is price relative to POC? (above/below/at)
2. Where is price relative to Value Area? (inside/above/below)
3. Are there nearby HVN levels for support/resistance?
4. Are there LVN zones that could accelerate price movement?
5. What Volume Profile setup applies? (mean reversion, VA edge, breakout, HVN bounce)
6. Where should stop loss be placed? (behind HVN preferred)
7. What's the risk:reward ratio? (must be >1.5:1)

## SIGNAL OPTIONS (you MUST pick one)
- **buy_to_enter** - Open a long position. Use when price is at VAL, below POC in discount, or bouncing off bullish HVN. MUST provide entry_price, stop_loss, and profit_target.
- **sell_to_enter** - Open a short position. Use when price is at VAH, above POC in premium, or rejecting at bearish HVN. MUST provide entry_price, stop_loss, and profit_target.
- **hold** - No action. Use when price is at fair value (inside VA near POC), no clear setup, or in LVN territory.
- **close** - Close existing position. Use when original thesis is invalidated.

Remember:
- Volume Profile shows WHERE institutional volume traded
- POC is a magnet - price tends to return
- Value Area edges are decision points
- HVN = consolidation/reversal zones
- LVN = acceleration zones
- Trade mean reversion inside VA, breakouts outside VA"""


def _format_vp_quant_report(decision: QuantAnalystDecision) -> str:
    """Format the VP quant decision into a human-readable report."""
    signal_str = (
        decision.signal
        if isinstance(decision.signal, str)
        else decision.signal.value
    )
    lines = [
        f"## VOLUME PROFILE QUANT DECISION: **{signal_str.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
    ]

    if signal_str in ["buy_to_enter", "sell_to_enter"]:
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

    lines.extend(
        [
            "### Justification",
            decision.justification,
            "",
            "### Invalidation Condition",
            decision.invalidation_condition,
            "",
        ]
    )

    if decision.risk_level:
        lines.append(f"**Risk Level**: {decision.risk_level}")

    return "\n".join(lines)


def get_vp_quant_decision_for_modal(vp_quant_decision: dict) -> dict:
    """
    Convert a VP quant decision dict to trade modal format.

    Args:
        vp_quant_decision: The vp_quant_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not vp_quant_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = vp_quant_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    return {
        "symbol": vp_quant_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "suggestedEntry": vp_quant_decision.get("entry_price"),
        "suggestedStopLoss": vp_quant_decision.get("stop_loss"),
        "suggestedTakeProfit": vp_quant_decision.get("profit_target"),
        "rationale": f"{vp_quant_decision.get('justification', '')}. Invalidation: {vp_quant_decision.get('invalidation_condition', '')}",
        "confidence": vp_quant_decision.get("confidence", 0.5),
    }


def analyze_volume_profile_for_quant(
    df,
    current_price: float,
    num_bins: int = 50,
    lookback: int = 100,
) -> Dict[str, Any]:
    """
    Run Volume Profile analysis and return formatted context for quant.

    Args:
        df: DataFrame with OHLCV data
        current_price: Current market price
        num_bins: Number of price bins for profile
        lookback: Number of candles to analyze

    Returns:
        Dict with 'volume_profile' (dataclass) and 'volume_profile_context' (string)
    """
    analyzer = VolumeProfileAnalyzer()
    profile = analyzer.calculate_volume_profile(df, num_bins=num_bins, lookback=lookback)
    context = analyzer.format_for_prompt(profile, current_price)

    return {
        "volume_profile": profile,
        "volume_profile_context": context,
    }
