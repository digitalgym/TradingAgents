"""
Smart Money Concepts (SMC) Quant Analyst Agent

A systematic quant trader agent focused on Smart Money Concepts analysis.
Uses Order Blocks, FVGs, BOS/CHoCH, liquidity zones, and premium/discount
to identify high-probability institutional entry zones.

Key SMC Concepts:
- Order Blocks (OB): Last candle before strong move - institutional entry zones
- Fair Value Gaps (FVG): Price imbalance zones where price tends to return
- Break of Structure (BOS): Trend continuation confirmation
- Change of Character (CHoCH): Potential trend reversal signal
- Liquidity: EQH/EQL where stop losses cluster - targets for smart money
- Premium/Discount: Buy in discount (below 50% of range), sell in premium

Trading Logic:
- Trade with trend confirmed by BOS
- Enter at unmitigated OBs with FVG confluence
- Target liquidity pools for take profit
- Place stops beyond OB zones
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from tradingagents.schemas import QuantAnalystDecision, RiskLevel
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator, safe_get


# Set up SMC quant prompt logger
_smc_quant_logger = None


def _get_smc_quant_logger():
    """Get or create the SMC quant prompt logger."""
    global _smc_quant_logger
    if _smc_quant_logger is None:
        _smc_quant_logger = logging.getLogger("smc_quant_prompts")
        _smc_quant_logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "logs", "smc_quant_prompts"
        )
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler with date-based filename
        log_file = os.path.join(
            log_dir, f"smc_quant_prompts_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler (avoid duplicates)
        if not _smc_quant_logger.handlers:
            _smc_quant_logger.addHandler(file_handler)

    return _smc_quant_logger


def create_smc_quant(llm, use_structured_output: bool = True):
    """
    Create an SMC quant analyst node.

    This quant focuses on:
    - Order Blocks as institutional entry zones
    - FVGs for price imbalance entries
    - BOS/CHoCH for trend confirmation
    - Liquidity pools as take profit targets
    - Premium/Discount zones for entry timing

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
                f"Warning: Structured output not supported for SMC quant analyst, falling back to free-form: {e}"
            )
            structured_llm = None

    def smc_quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Gather SMC data
        market_report = state.get("market_report") or ""
        smc_analysis = state.get("smc_analysis")  # Dict from SmartMoneyAnalyzer
        smc_context = state.get("smc_context") or ""

        # Extract regime information
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        trading_session = state.get("trading_session") or "unknown"

        # Build the comprehensive data context
        data_context = _build_smc_data_context(
            ticker=ticker,
            current_price=current_price,
            smc_analysis=smc_analysis,
            smc_context=smc_context,
            market_report=market_report,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=trading_session,
            current_date=current_date,
        )

        # Build the SMC quant analyst prompt
        system_prompt = _build_smc_quant_prompt(data_context)

        # Log the prompt being sent to LLM
        logger = _get_smc_quant_logger()
        logger.info(
            f"\n{'='*80}\nSMC QUANT ANALYSIS - {ticker}\n{'='*80}"
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
                report = _format_smc_quant_report(decision)

                logger.info(
                    f"Structured output SUCCESS - returning smc_quant_decision with signal={decision.signal}"
                )
                return {
                    "smc_quant_report": report,
                    "smc_quant_decision": decision_dict,
                }
            except Exception as e:
                import traceback as _tb

                logger.error(f"Structured output failed: {e}")
                logger.error(f"Traceback:\n{_tb.format_exc()}")
                print(f"Structured output failed for SMC quant analyst: {e}")

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

        logger.warning(f"Returning smc_quant_decision=None (unstructured fallback)")
        return {
            "smc_quant_report": report,
            "smc_quant_decision": None,
        }

    return smc_quant_analyst_node


def _build_smc_data_context(
    ticker: str,
    current_price: Optional[float],
    smc_analysis: Optional[Dict[str, Any]],
    smc_context: str,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    trading_session: str,
    current_date: str,
) -> str:
    """Build comprehensive data context for the SMC quant analyst."""

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

    # SMC Analysis
    if smc_context:
        sections.append(f"## SMART MONEY CONCEPTS ANALYSIS\n{smc_context}")
    elif smc_analysis:
        # Generate context from SMC analysis dict
        sections.append(_format_smc_analysis_for_prompt(smc_analysis, current_price))

    # Technical indicators (market_report already has its own header)
    if market_report:
        sections.append(f"## TECHNICAL INDICATORS\n{market_report}")

    return "\n".join(sections)


def _format_smc_analysis_for_prompt(
    analysis: Dict[str, Any], current_price: Optional[float]
) -> str:
    """Format SMC analysis dict for prompt."""
    if not analysis:
        return "## SMART MONEY CONCEPTS ANALYSIS\nInsufficient SMC data.\n"

    lines = ["## SMART MONEY CONCEPTS ANALYSIS"]
    cp = current_price or 0

    # Market Bias
    bias = analysis.get("bias", "unknown")
    lines.append(f"**Market Bias**: {bias.upper()}")

    # Premium/Discount Zone
    pd_zone = analysis.get("premium_discount")
    if pd_zone:
        zone_name = safe_get(pd_zone, 'zone', 'equilibrium')
        zone_pct = safe_get(pd_zone, 'position_pct', 50)
        if zone_name == "discount":
            lines.append(f"**Premium/Discount**: DISCOUNT zone ({zone_pct:.0f}%) - favorable for LONGS")
        elif zone_name == "premium":
            lines.append(f"**Premium/Discount**: PREMIUM zone ({zone_pct:.0f}%) - favorable for SHORTS")
        else:
            lines.append(f"**Premium/Discount**: EQUILIBRIUM ({zone_pct:.0f}%)")

    # Structure (BOS/CHoCH)
    structure = analysis.get("structure", {})
    recent_bos = structure.get("recent_bos", [])
    recent_choc = structure.get("recent_choc", [])

    if recent_bos or recent_choc:
        lines.append("\n**Market Structure**:")
        for bos in recent_bos[-3:]:  # Last 3
            bos_type = safe_get(bos, 'type', 'unknown')
            bos_price = safe_get(bos, 'price', 0)
            lines.append(f"  - BOS {bos_type.upper()} at {bos_price:.5f}")
        for choc in recent_choc[-2:]:  # Last 2
            choc_type = safe_get(choc, 'type', 'unknown')
            choc_price = safe_get(choc, 'price', 0)
            lines.append(f"  - CHoCH {choc_type.upper()} at {choc_price:.5f} (potential reversal)")

    # Order Blocks
    obs = analysis.get("order_blocks", {})
    bullish_obs = obs.get("bullish", [])
    bearish_obs = obs.get("bearish", [])

    if bullish_obs or bearish_obs:
        lines.append("\n**Order Blocks** (institutional entry zones):")
        for ob in bullish_obs[:3]:
            top = safe_get(ob, 'top', 0)
            bottom = safe_get(ob, 'bottom', 0)
            strength = safe_get(ob, 'strength', 0)
            mitigated = safe_get(ob, 'mitigated', False)
            status = "MITIGATED" if mitigated else "UNMITIGATED"
            dist = ((cp - bottom) / cp * 100) if cp else 0
            lines.append(
                f"  - BULLISH OB: {bottom:.5f}-{top:.5f} | Strength: {strength*100:.0f}% | {status} | {dist:.2f}% from price"
            )
        for ob in bearish_obs[:3]:
            top = safe_get(ob, 'top', 0)
            bottom = safe_get(ob, 'bottom', 0)
            strength = safe_get(ob, 'strength', 0)
            mitigated = safe_get(ob, 'mitigated', False)
            status = "MITIGATED" if mitigated else "UNMITIGATED"
            dist = ((top - cp) / cp * 100) if cp else 0
            lines.append(
                f"  - BEARISH OB: {bottom:.5f}-{top:.5f} | Strength: {strength*100:.0f}% | {status} | {dist:.2f}% from price"
            )

    # Fair Value Gaps
    fvgs = analysis.get("fair_value_gaps", {})
    bullish_fvgs = fvgs.get("bullish", [])
    bearish_fvgs = fvgs.get("bearish", [])

    if bullish_fvgs or bearish_fvgs:
        lines.append("\n**Fair Value Gaps** (price imbalance zones):")
        for fvg in bullish_fvgs[:3]:
            top = safe_get(fvg, 'top', 0)
            bottom = safe_get(fvg, 'bottom', 0)
            mitigated = safe_get(fvg, 'mitigated', False)
            status = "FILLED" if mitigated else "UNFILLED"
            dist = ((cp - bottom) / cp * 100) if cp else 0
            lines.append(
                f"  - BULLISH FVG: {bottom:.5f}-{top:.5f} | {status} | {dist:.2f}% below price (expect bounce UP)"
            )
        for fvg in bearish_fvgs[:3]:
            top = safe_get(fvg, 'top', 0)
            bottom = safe_get(fvg, 'bottom', 0)
            mitigated = safe_get(fvg, 'mitigated', False)
            status = "FILLED" if mitigated else "UNFILLED"
            dist = ((top - cp) / cp * 100) if cp else 0
            lines.append(
                f"  - BEARISH FVG: {bottom:.5f}-{top:.5f} | {status} | {dist:.2f}% above price (expect rejection DOWN)"
            )

    # Liquidity Zones
    liquidity = analysis.get("liquidity_zones", [])
    if liquidity:
        lines.append("\n**Liquidity Zones** (stop loss clusters - smart money targets):")
        for lz in liquidity[:4]:
            lz_price = safe_get(lz, 'price', 0)
            lz_type = safe_get(lz, 'type', 'unknown')
            lz_strength = safe_get(lz, 'strength', 50)
            swept = safe_get(lz, 'swept', False)
            status = "SWEPT" if swept else "UNSWEPT"
            dist = ((lz_price - cp) / cp * 100) if cp else 0
            direction = "above" if dist > 0 else "below"
            lines.append(
                f"  - {lz_type.upper()} at {lz_price:.5f} | Strength: {lz_strength:.0f}% | {status} | {abs(dist):.2f}% {direction}"
            )

    # OTE Zone
    ote = analysis.get("ote_zone")
    if ote:
        ote_high = safe_get(ote, 'high', 0)
        ote_low = safe_get(ote, 'low', 0)
        ote_type = safe_get(ote, 'type', 'bullish')
        lines.append(f"\n**Optimal Trade Entry (OTE)**: {ote_low:.5f}-{ote_high:.5f} ({ote_type})")
        lines.append("  - OTE is the 61.8%-78.6% Fibonacci retracement zone - high probability entries")

    lines.append("")
    return "\n".join(lines)


def _build_smc_quant_prompt(data_context: str) -> str:
    """Build the complete SMC quant analyst prompt."""

    return f"""You are a systematic Smart Money Concepts (SMC) trader with strict risk discipline. You trade based on institutional order flow analysis to identify high-probability entries.

## SMART MONEY CONCEPTS TRADING RULES

### Core Concepts
1. **Order Blocks (OB)**: The last opposing candle before a strong impulse move. This marks where institutions entered.
   - Bullish OB: Last bearish candle before bullish impulse - expect price to bounce UP
   - Bearish OB: Last bullish candle before bearish impulse - expect price to reject DOWN
   - Only trade UNMITIGATED OBs (price hasn't returned yet)

2. **Fair Value Gaps (FVG)**: Three-candle imbalance where price moved so fast it left a gap.
   - Price tends to return to "fill" FVGs
   - Bullish FVG: Gap below price - expect bounce UP when price returns
   - Bearish FVG: Gap above price - expect rejection DOWN when price returns
   - UNFILLED FVGs are active; FILLED FVGs are spent

3. **Break of Structure (BOS)**: Price breaks previous swing high/low IN the direction of trend.
   - Bullish BOS: Higher high formed - trend continuation UP
   - Bearish BOS: Lower low formed - trend continuation DOWN
   - BOS confirms you can trade with the trend

4. **Change of Character (CHoCH)**: Price breaks structure AGAINST the trend.
   - First warning of potential reversal
   - Look for new OBs forming in the reversal direction
   - Wait for confirmation before trading reversal

5. **Liquidity**: Areas where stop losses cluster (equal highs/lows, round numbers).
   - Smart money hunts liquidity before reversing
   - EQH (Equal Highs): Buy-side liquidity above - price may spike up to sweep then drop
   - EQL (Equal Lows): Sell-side liquidity below - price may spike down to sweep then rise
   - After a liquidity sweep, look for reversal entries

6. **Premium/Discount**: The current range divided into zones.
   - Premium (>50% of range): Expensive - favor SHORTS
   - Discount (<50% of range): Cheap - favor LONGS
   - Equilibrium (50%): Wait for price to move to extreme

### Trading Setups

**Setup 1: OB Entry with Trend**
- BOS confirms trend direction
- Price retraces to unmitigated OB
- Enter at OB in trend direction
- SL: Beyond OB
- TP: Next liquidity pool or opposing OB

**Setup 2: OB + FVG Confluence**
- Unmitigated OB overlaps with unfilled FVG
- Highest probability setup
- Enter at the overlap zone
- Tighter SL possible due to confluence

**Setup 3: Liquidity Sweep Reversal**
- Price sweeps liquidity (EQH/EQL)
- Look for CHoCH or BOS in opposite direction
- Enter at the new OB formed after sweep
- Target: Opposite liquidity pool

**Setup 4: OTE Entry (Optimal Trade Entry)**
- After impulse move, wait for retracement
- Enter at 61.8%-78.6% Fibonacci zone
- This is where institutions typically re-enter
- Combine with OB/FVG for confluence

## RISK MANAGEMENT RULES (NEVER BREAK)

1. **Risk no more than 1-2% of account value per trade**
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and invalidation before entry**
5. **Place stops beyond OB/FVG zones** (allow for liquidity sweep)
6. **Fees are 0.025% maker / 0.05% taker + funding - size accordingly**
7. **Leverage is a tool, not a goal - default 5-20x**

## STOP LOSS VALIDATION (CRITICAL)
- For BUY orders: Stop loss MUST be BELOW entry price
- For SELL orders: Stop loss MUST be ABOVE entry price
- Place stops beyond the OB/FVG zone that forms your entry basis
- If you cannot identify a valid stop loss placement, output "hold"

{data_context}

## YOUR TASK

Analyze the SMC data and make a systematic trading decision.

Think step-by-step:
1. What is the market bias? (bullish/bearish based on BOS)
2. Are we in premium or discount? (favor shorts in premium, longs in discount)
3. Are there unmitigated OBs or unfilled FVGs near current price?
4. Is there OB+FVG confluence?
5. Where are the liquidity targets? (EQH/EQL for TP)
6. Has there been a recent liquidity sweep that signals reversal?
7. Where should stop loss be placed? (beyond OB/FVG)
8. What's the risk:reward ratio? (must be >1.5:1)

## SIGNAL OPTIONS (you MUST pick one)
- **buy_to_enter** - Open a long position. Use when: bullish BOS, price at bullish OB/FVG in discount, or after sell-side liquidity sweep. MUST provide entry_price, stop_loss, and profit_target.
- **sell_to_enter** - Open a short position. Use when: bearish BOS, price at bearish OB/FVG in premium, or after buy-side liquidity sweep. MUST provide entry_price, stop_loss, and profit_target.
- **hold** - No action. Use when: no clear structure, price at equilibrium, no unmitigated zones nearby, or conflicting signals.
- **close** - Close existing position. Use when: price reaches liquidity target, OB/FVG invalidated, or CHoCH against position.

## ORDER TYPE (you MUST pick one for buy/sell signals)
- **market** - Execute immediately at current market price. Use when:
  - Price is ALREADY AT the OB/FVG zone (within 0.1%)
  - Liquidity sweep just happened and you want immediate entry
  - Setup is confirmed and you don't want to miss it
- **limit** - Place pending order at entry_price. Use when:
  - Price is NOT YET at the zone (waiting for retracement)
  - You want better entry at the OB/FVG level
  - Zone is nearby but price hasn't reached it yet

Remember:
- Smart money leaves footprints (OBs, FVGs)
- Structure (BOS/CHoCH) tells you the trend
- Liquidity shows you where price is likely to go
- Trade with trend, enter at discount (longs) or premium (shorts)
- Confluence = higher probability"""


def _format_smc_quant_report(decision: QuantAnalystDecision) -> str:
    """Format the SMC quant decision into a human-readable report."""
    signal_str = (
        decision.signal
        if isinstance(decision.signal, str)
        else decision.signal.value
    )
    lines = [
        f"## SMC QUANT DECISION: **{signal_str.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
    ]

    if signal_str in ["buy_to_enter", "sell_to_enter"]:
        lines.append("### Trade Parameters")
        if decision.order_type:
            order_type_str = decision.order_type if isinstance(decision.order_type, str) else decision.order_type.value
            lines.append(f"- **Order Type**: {order_type_str.upper()}")
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


def get_smc_quant_decision_for_modal(smc_quant_decision: dict) -> dict:
    """
    Convert an SMC quant decision dict to trade modal format.

    Args:
        smc_quant_decision: The smc_quant_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not smc_quant_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = smc_quant_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    # Extract order_type
    order_type = smc_quant_decision.get("order_type", "market")
    if isinstance(order_type, dict):
        order_type = order_type.get("value", "market")

    return {
        "symbol": smc_quant_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "orderType": order_type,  # "market" or "limit"
        "suggestedEntry": smc_quant_decision.get("entry_price"),
        "suggestedStopLoss": smc_quant_decision.get("stop_loss"),
        "suggestedTakeProfit": smc_quant_decision.get("profit_target"),
        "rationale": f"{smc_quant_decision.get('justification', '')}. Invalidation: {smc_quant_decision.get('invalidation_condition', '')}",
        "confidence": smc_quant_decision.get("confidence", 0.5),
    }


def analyze_smc_for_quant(
    df,
    current_price: float,
    swing_lookback: int = 5,
    ob_strength_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run SMC analysis and return formatted context for quant.

    Args:
        df: DataFrame with OHLCV data
        current_price: Current market price
        swing_lookback: Periods to look back for swing points
        ob_strength_threshold: Minimum OB strength (0-1)

    Returns:
        Dict with 'smc_analysis' (dict) and 'smc_context' (string)
    """
    analyzer = SmartMoneyAnalyzer(
        swing_lookback=swing_lookback,
        ob_strength_threshold=ob_strength_threshold,
    )
    analysis = analyzer.analyze(df, include_zones=True)
    context = _format_smc_analysis_for_prompt(analysis, current_price)

    return {
        "smc_analysis": analysis,
        "smc_context": context,
    }
