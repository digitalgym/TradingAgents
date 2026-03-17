"""
Breakout Quant Analyst

A specialized quant analyst focused on consolidation detection and breakout trading.
Identifies when markets are in tight ranges (squeezes) and predicts breakout opportunities.

Key concepts:
- Consolidation: Period of low volatility where price trades in a tight range
- Squeeze: Bollinger Bands contract (low BB width percentile)
- Breakout: Price breaks out of consolidation range with momentum
- Direction: Uses structure (higher lows = bullish, lower highs = bearish)

Trading approach:
- Wait for consolidation to form (contraction regime)
- Identify range boundaries (high/low of consolidation)
- Trade breakout with confirmation OR anticipate direction from structure
- Stop loss inside the range (below breakout level)
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from tradingagents.schemas import QuantAnalystDecision, RiskLevel


def _safe_get(obj, attr, default=None):
    """Safely get attribute from dict or dataclass object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# Set up breakout quant logger
_breakout_logger = None


def _get_breakout_logger():
    """Get or create the breakout quant prompt logger."""
    global _breakout_logger
    if _breakout_logger is None:
        _breakout_logger = logging.getLogger("breakout_quant_prompts")
        _breakout_logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "quant_prompts")
        os.makedirs(log_dir, exist_ok=True)

        # Create file handler with date-based filename
        log_file = os.path.join(log_dir, f"breakout_quant_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler (avoid duplicates)
        if not _breakout_logger.handlers:
            _breakout_logger.addHandler(file_handler)

    return _breakout_logger


def analyze_consolidation(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray = None,
    lookback: int = 20
) -> Dict[str, Any]:
    """
    Analyze price data for consolidation patterns.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data (optional but recommended for breakout confirmation)
        lookback: Number of candles to analyze for consolidation

    Returns:
        Dict with consolidation analysis:
        - is_consolidating: bool
        - range_high: Upper boundary
        - range_low: Lower boundary
        - range_midpoint: Middle of range
        - range_percent: Range as % of price
        - bb_squeeze: Whether BB width is contracting
        - structure_bias: "bullish" (higher lows), "bearish" (lower highs), or "neutral"
        - squeeze_strength: 0-100 (higher = tighter squeeze)
        - breakout_ready: bool - True if consolidating with clear bias
        - volume_contracting: bool - Volume drying up inside consolidation
        - volume_contraction_pct: How much volume contracted vs prior period
        - breakout_detected: bool - Price closed outside range
        - breakout_direction: "up", "down", or None
        - breakout_volume_surge: bool - Breakout candle has volume surge
        - breakout_confirmed: bool - Close outside range WITH volume surge
    """
    if len(close) < lookback:
        return {
            "is_consolidating": False,
            "range_high": None,
            "range_low": None,
            "range_midpoint": None,
            "range_percent": None,
            "bb_squeeze": False,
            "structure_bias": "neutral",
            "squeeze_strength": 0,
            "breakout_ready": False,
            "volume_contracting": False,
            "volume_contraction_pct": 0.0,
            "breakout_detected": False,
            "breakout_direction": None,
            "breakout_volume_surge": False,
            "breakout_confirmed": False,
        }

    # Get recent data
    recent_high = high[-lookback:]
    recent_low = low[-lookback:]
    recent_close = close[-lookback:]

    # Calculate range boundaries EXCLUDING the current candle (for breakout detection)
    # The range is defined by all candles except the last one
    if len(recent_high) > 1:
        range_high = float(np.max(recent_high[:-1]))
        range_low = float(np.min(recent_low[:-1]))
    else:
        range_high = float(np.max(recent_high))
        range_low = float(np.min(recent_low))
    range_midpoint = (range_high + range_low) / 2
    range_percent = ((range_high - range_low) / range_midpoint) * 100

    # Calculate Bollinger Band width for squeeze detection
    sma = np.mean(recent_close)
    std = np.std(recent_close, ddof=1)
    bb_width = (4 * std / sma) * 100 if sma > 0 else 0

    # Calculate historical BB width for comparison
    if len(close) >= lookback * 2:
        historical_widths = []
        for i in range(lookback, len(close)):
            hist_close = close[i-lookback:i]
            hist_sma = np.mean(hist_close)
            hist_std = np.std(hist_close, ddof=1)
            hist_width = (4 * hist_std / hist_sma) * 100 if hist_sma > 0 else 0
            historical_widths.append(hist_width)

        if historical_widths:
            percentile = (np.sum(np.array(historical_widths) > bb_width) / len(historical_widths)) * 100
            squeeze_strength = percentile  # Higher = current width is lower than historical
        else:
            squeeze_strength = 50
    else:
        squeeze_strength = 50

    bb_squeeze = squeeze_strength > 70  # In squeeze if current width is lower than 70% of history

    # Determine structure bias (higher lows = bullish, lower highs = bearish)
    half = lookback // 2
    first_half_low = np.min(recent_low[:half])
    second_half_low = np.min(recent_low[half:])
    first_half_high = np.max(recent_high[:half])
    second_half_high = np.max(recent_high[half:])

    higher_lows = second_half_low > first_half_low
    lower_highs = second_half_high < first_half_high

    if higher_lows and not lower_highs:
        structure_bias = "bullish"
    elif lower_highs and not higher_lows:
        structure_bias = "bearish"
    else:
        structure_bias = "neutral"

    # Is consolidating: tight range + low volatility
    is_consolidating = range_percent < 3.0 and squeeze_strength > 60

    # Breakout ready: consolidating with clear directional bias
    breakout_ready = is_consolidating and structure_bias != "neutral"

    # === VOLUME ANALYSIS (key for breakout confirmation) ===
    volume_contracting = False
    volume_contraction_pct = 0.0
    breakout_detected = False
    breakout_direction = None
    breakout_volume_surge = False
    breakout_confirmed = False

    if volume is not None and len(volume) >= lookback:
        recent_volume = volume[-lookback:]
        current_candle_volume = volume[-1]

        # Average volume inside consolidation (excluding last candle)
        avg_volume_in_range = float(np.mean(recent_volume[:-1])) if len(recent_volume) > 1 else float(np.mean(recent_volume))

        # Compare to volume before consolidation (prior period)
        if len(volume) >= lookback * 2:
            prior_volume = volume[-lookback * 2:-lookback]
            avg_volume_before = float(np.mean(prior_volume))

            # Volume contracting = current period avg < 80% of prior period
            if avg_volume_before > 0:
                volume_contraction_pct = ((avg_volume_before - avg_volume_in_range) / avg_volume_before) * 100
                volume_contracting = avg_volume_in_range < avg_volume_before * 0.8
        else:
            # Not enough history, just check if recent volume is low relative to range
            volume_contraction_pct = 0.0

        # === BREAKOUT DETECTION (close-based, not wick) ===
        current_close = close[-1]

        # Breakout UP: close above range high
        if current_close > range_high:
            breakout_detected = True
            breakout_direction = "up"

        # Breakout DOWN: close below range low
        elif current_close < range_low:
            breakout_detected = True
            breakout_direction = "down"

        # === VOLUME SURGE ON BREAKOUT ===
        # Surge = current candle volume > 1.5x average volume in range
        if avg_volume_in_range > 0:
            volume_surge_ratio = current_candle_volume / avg_volume_in_range
            breakout_volume_surge = volume_surge_ratio >= 1.5

        # === CONFIRMED BREAKOUT = Close outside range + Volume surge ===
        breakout_confirmed = breakout_detected and breakout_volume_surge

    return {
        "is_consolidating": is_consolidating,
        "range_high": range_high,
        "range_low": range_low,
        "range_midpoint": range_midpoint,
        "range_percent": range_percent,
        "bb_width": bb_width,
        "bb_squeeze": bb_squeeze,
        "breakout_ready": breakout_ready,
        "structure_bias": structure_bias,
        "squeeze_strength": squeeze_strength,
        # Volume analysis
        "volume_contracting": volume_contracting,
        "volume_contraction_pct": volume_contraction_pct,
        # Breakout detection
        "breakout_detected": breakout_detected,
        "breakout_direction": breakout_direction,
        "breakout_volume_surge": breakout_volume_surge,
        "breakout_confirmed": breakout_confirmed,
    }


def create_breakout_quant(llm, use_structured_output: bool = True):
    """
    Create a breakout quant analyst node that specializes in consolidation and breakout trading.

    This analyst:
    - Identifies when markets are consolidating (tight ranges, low volatility)
    - Detects BB squeeze conditions
    - Analyzes structure for breakout direction bias
    - Provides breakout trade setups with clear invalidation

    Args:
        llm: The language model to use for analysis
        use_structured_output: If True, uses LLM structured output for guaranteed JSON

    Returns:
        A function that processes state and returns the breakout analysis
    """

    # Create structured output LLM wrapper if supported
    structured_llm = None
    if use_structured_output:
        try:
            structured_llm = llm.with_structured_output(QuantAnalystDecision)
        except Exception as e:
            print(f"Warning: Structured output not supported for breakout quant, falling back to free-form: {e}")
            structured_llm = None

    def breakout_quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Gather technical data
        smc_context = state.get("smc_context") or ""
        smc_analysis = state.get("smc_analysis") or {}
        market_report = state.get("market_report") or ""

        # Extract regime information - critical for breakout analysis
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        expansion_regime = state.get("expansion_regime") or "neutral"
        trading_session = state.get("trading_session") or "unknown"

        # Get price data for consolidation analysis
        price_data = state.get("price_data") or {}
        high = price_data.get("high")
        low = price_data.get("low")
        close = price_data.get("close")
        volume = price_data.get("volume")  # Volume data for breakout confirmation

        # Perform consolidation analysis if price data available
        consolidation = None
        if high is not None and low is not None and close is not None:
            try:
                high_arr = np.array(high) if not isinstance(high, np.ndarray) else high
                low_arr = np.array(low) if not isinstance(low, np.ndarray) else low
                close_arr = np.array(close) if not isinstance(close, np.ndarray) else close
                volume_arr = np.array(volume) if volume is not None and not isinstance(volume, np.ndarray) else volume
                consolidation = analyze_consolidation(high_arr, low_arr, close_arr, volume_arr)
            except Exception as e:
                print(f"Consolidation analysis error: {e}")
                consolidation = None

        # Build consolidation-specific data context
        data_context = _build_breakout_data_context(
            ticker=ticker,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=market_report,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            expansion_regime=expansion_regime,
            trading_session=trading_session,
            current_date=current_date,
            consolidation=consolidation,
        )

        # Build the breakout quant prompt
        trade_memories = state.get("trade_memories") or ""
        system_prompt = _build_breakout_prompt(data_context, trade_memories=trade_memories)

        # Log the prompt
        logger = _get_breakout_logger()
        logger.info(f"\n{'='*80}\nBREAKOUT QUANT ANALYSIS - {ticker}\n{'='*80}")
        logger.info(f"Symbol: {ticker} | Date: {current_date} | Price: {current_price}")
        logger.info(f"Expansion Regime: {expansion_regime} | Volatility: {volatility_regime}")
        if consolidation:
            logger.info(f"Consolidation: {consolidation}")
        logger.info(f"\n--- DATA CONTEXT ---\n{data_context}")
        logger.info(f"\n--- FULL PROMPT ---\n{system_prompt}")
        logger.info(f"\n{'='*80}\n")

        if structured_llm is not None:
            try:
                import time as _time
                _llm_start = _time.time()
                decision: QuantAnalystDecision = structured_llm.invoke(system_prompt)
                _llm_duration = _time.time() - _llm_start

                # Log response
                logger.info(f"--- LLM RESPONSE (Structured) [took {_llm_duration:.1f}s] ---")
                logger.info(f"Signal: {decision.signal}")
                logger.info(f"Confidence: {decision.confidence}")
                logger.info(f"Entry: {decision.entry_price} | SL: {decision.stop_loss} | TP: {decision.profit_target}")
                logger.info(f"Justification: {decision.justification}")
                logger.info(f"Invalidation: {decision.invalidation_condition}")

                decision_dict = decision.model_dump()
                logger.info(f"Raw decision dict: {decision_dict}")
                logger.info(f"\n{'='*80}\n")

                # Generate report
                report = _format_breakout_report(decision, consolidation)

                logger.info(f"Structured output SUCCESS - returning breakout_quant_decision with signal={decision.signal}")
                return {
                    "breakout_quant_report": report,
                    "breakout_quant_decision": decision_dict,
                    "consolidation_analysis": consolidation,
                }
            except Exception as e:
                import traceback as _tb
                logger.error(f"Structured output failed: {e}")
                logger.error(f"Traceback:\n{_tb.format_exc()}")
                print(f"Structured output failed for breakout quant: {e}")

        # Fallback to unstructured
        logger.info(f"--- Falling back to unstructured LLM call ---")
        import time as _time
        _llm_start = _time.time()
        response = llm.invoke(system_prompt)
        _llm_duration = _time.time() - _llm_start
        report = response.content if hasattr(response, 'content') else str(response)

        logger.info(f"--- LLM RESPONSE (Unstructured) [took {_llm_duration:.1f}s] ---")
        logger.info(f"{report[:2000]}...")
        logger.info(f"\n{'='*80}\n")

        logger.warning(f"Returning breakout_quant_decision=None (unstructured fallback)")
        return {
            "breakout_quant_report": report,
            "breakout_quant_decision": None,
            "consolidation_analysis": consolidation,
        }

    return breakout_quant_analyst_node


def _build_breakout_data_context(
    ticker: str,
    current_price: Optional[float],
    smc_context: str,
    smc_analysis: dict,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    expansion_regime: str,
    trading_session: str,
    current_date: str,
    consolidation: Optional[Dict[str, Any]],
) -> str:
    """Build comprehensive data context for breakout analysis."""

    sections = []

    # Current market data
    if current_price:
        sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price**: {current_price:.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
""")
    else:
        sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
""")

    # Regime information - critical for breakout strategy
    sections.append(f"""## MARKET REGIME (Critical for Breakout Strategy)
- **Trend Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
- **Expansion Regime**: {expansion_regime}

### Regime Interpretation for Breakouts:
- If expansion_regime = "contraction": Market is in SQUEEZE - prime breakout setup
- If expansion_regime = "expansion": Breakout may already be in progress
- If volatility_regime = "low": Ideal for breakout anticipation
- If volatility_regime = "high/extreme": Breakout likely already happened, wait for consolidation
""")

    # Consolidation analysis
    if consolidation:
        squeeze_status = "YES - BB SQUEEZE ACTIVE" if consolidation.get("bb_squeeze") else "No squeeze"
        consol_status = "YES - IN CONSOLIDATION" if consolidation.get("is_consolidating") else "Not consolidating"

        # Format range values safely
        range_high = consolidation.get('range_high')
        range_low = consolidation.get('range_low')
        range_mid = consolidation.get('range_midpoint')
        range_high_str = f"{range_high:.5f}" if range_high is not None else "N/A"
        range_low_str = f"{range_low:.5f}" if range_low is not None else "N/A"
        range_mid_str = f"{range_mid:.5f}" if range_mid is not None else "N/A"

        # Volume analysis
        vol_contracting = consolidation.get('volume_contracting', False)
        vol_contraction_pct = consolidation.get('volume_contraction_pct', 0)
        vol_status = f"YES - Volume dried up ({vol_contraction_pct:.0f}% lower than prior period)" if vol_contracting else "No significant contraction"

        # Breakout detection
        breakout_detected = consolidation.get('breakout_detected', False)
        breakout_dir = consolidation.get('breakout_direction')
        breakout_vol_surge = consolidation.get('breakout_volume_surge', False)
        breakout_confirmed = consolidation.get('breakout_confirmed', False)

        sections.append(f"""## CONSOLIDATION ANALYSIS
- **Is Consolidating**: {consol_status}
- **BB Squeeze**: {squeeze_status}
- **Squeeze Strength**: {consolidation.get('squeeze_strength', 0):.0f}% (higher = tighter squeeze)
- **Range High (Resistance)**: {range_high_str}
- **Range Low (Support)**: {range_low_str}
- **Range Midpoint**: {range_mid_str}
- **Range Width**: {consolidation.get('range_percent', 0):.2f}% of price
- **Structure Bias**: {consolidation.get('structure_bias', 'neutral').upper()}

### Volume Analysis (Critical for Breakout Confirmation):
- **Volume Contracting Inside Range**: {vol_status}
- **Breakout Detected**: {"YES - " + breakout_dir.upper() if breakout_detected else "No breakout yet"}
- **Volume Surge on Breakout**: {"YES - 1.5x+ average volume" if breakout_vol_surge else "No surge"}
- **BREAKOUT CONFIRMED**: {"YES - Close outside range WITH volume surge" if breakout_confirmed else "NOT CONFIRMED"}

### Breakout Confirmation Rules:
1. Price must CLOSE outside range (not just wick)
2. Breakout candle must have volume surge (>1.5x average)
3. Volume should be contracting BEFORE breakout (coiled spring)
4. If breakout_confirmed = YES, this is a valid entry trigger

### Structure Interpretation:
- BULLISH structure (higher lows): Buyers accumulating, expect upside breakout
- BEARISH structure (lower highs): Sellers distributing, expect downside breakout
- NEUTRAL: No clear bias, wait for breakout direction confirmation
""")
    else:
        sections.append("""## CONSOLIDATION ANALYSIS
- Price data not available for consolidation analysis
- Use regime indicators and technical data to assess consolidation
""")

    # SMC context
    if smc_context:
        sections.append(smc_context)

    # Technical indicators
    if market_report:
        sections.append(market_report)

    return "\n".join(sections)


def _build_breakout_prompt(data_context: str, trade_memories: str = None) -> str:
    """Build the complete breakout quant analyst prompt."""

    memories_section = ""
    if trade_memories:
        memories_section = f"""

{trade_memories}

IMPORTANT: The above lessons are from YOUR past breakout trades. Study what went wrong
and apply corrections. Pay special attention to false breakout lessons.

"""

    return f"""You are a breakout trading specialist with strict risk discipline. You specialize in identifying consolidation patterns and trading breakouts.

## YOUR TRADING EDGE: CONSOLIDATION TO BREAKOUT

Markets alternate between consolidation (range) and expansion (trend). Your edge is:
1. Identify when price is consolidating (tight range, low volatility, BB squeeze)
2. Look for VOLUME CONTRACTION inside the range (drying up = coiled spring)
3. Determine likely breakout direction from structure (higher lows = up, lower highs = down)
4. Enter on CONFIRMED breakout: close outside range + volume surge (1.5x+ average)

**THE KEY FILTER**: Volume dry-up inside range + volume surge on breakout = real breakout
Without volume confirmation, most breakouts fail and trap traders.

## BREAKOUT TRADING RULES

### 1. IDENTIFYING CONSOLIDATION
- **BB Squeeze**: When Bollinger Band width is in lowest 30% of recent history
- **Low Volatility**: ATR is low relative to recent history
- **Tight Range**: Price contained within narrow high/low boundaries
- **Time**: Good consolidations last 10-30 candles

### 2. DETERMINING BREAKOUT DIRECTION
- **Higher Lows** within range = Bullish accumulation = Expect UPSIDE breakout
- **Lower Highs** within range = Bearish distribution = Expect DOWNSIDE breakout
- **Neutral structure** = Wait for breakout confirmation before entering

### 3. ENTRY STRATEGIES
- **Anticipation Entry**: Enter before breakout when structure is clear + volume contracting
  - BUY near range low when structure is bullish AND volume is drying up
  - SELL near range high when structure is bearish AND volume is drying up
  - SL: Opposite side of range
  - Requires: volume_contracting = True

- **Breakout Entry**: Enter on CONFIRMED breakout only
  - BUY when price CLOSES above range high WITH volume surge (1.5x+)
  - SELL when price CLOSES below range low WITH volume surge (1.5x+)
  - SL: Beyond the opposite side of the breakout level (at least 1x ATR buffer)
  - **CRITICAL**: breakout_confirmed must be TRUE (close + volume surge)
  - Do NOT enter on wick-only breaks or low-volume breakouts

### 4. STOP LOSS PLACEMENT (CRITICAL)
- For BUY: Stop loss MUST be BELOW entry price
  - Anticipation buy: SL below range low minus 1x ATR buffer
  - Breakout buy: SL below the breakout level minus 1x ATR buffer (allow for retest)

- For SELL: Stop loss MUST be ABOVE entry price
  - Anticipation sell: SL above range high plus 1x ATR buffer
  - Breakdown sell: SL above the breakdown level plus 1x ATR buffer (allow for retest)

- **MINIMUM SL DISTANCE**: Stop loss must be at least 1x ATR from entry. SL within 0.5% of entry is TOO TIGHT and will get stopped out by normal price noise. For gold (XAUUSD), minimum SL should be $15-30+ from entry.
- If you cannot place SL with adequate distance while maintaining R:R >= 1.5:1, output "hold"

### 5. TAKE PROFIT TARGETS
- Minimum target: 1.5x the range width
- Preferred target: 2x the range width
- Extended target: Previous swing high/low, major SMC level, or proven equal level (3+ retests = strong liquidity magnet)

### 6. WHEN NOT TO TRADE
- **No consolidation**: Price is already trending (expansion regime)
- **Extreme volatility**: Too choppy, wait for consolidation to form
- **Unclear structure**: No higher lows or lower highs, wait for clarity
- **News imminent**: Major catalyst could invalidate technical setup

## RULES YOU MUST NEVER BREAK

1. **Risk no more than 1-2% of account per trade**
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and invalidation**
5. **If no clear consolidation or direction, output HOLD**
6. **Stop loss MUST be on correct side of entry**

{data_context}
{memories_section}
## YOUR TASK

Analyze the consolidation and regime data to make a breakout trading decision.

Think step-by-step:
1. Is the market in consolidation? (Check expansion_regime, BB squeeze, range %)
2. Is volume contracting inside the range? (volume_contracting = coiled spring)
3. What is the structure bias? (Higher lows = bullish, lower highs = bearish)
4. Where are the range boundaries? (Resistance high, Support low)
5. Has a breakout occurred? Check breakout_confirmed (close outside + volume surge)
6. Is price at a good entry zone? (Near range extreme or confirmed breakout)
7. Where is the stop loss and is the R:R acceptable (>1.5:1)?
8. What would invalidate this setup?

## SIGNAL OPTIONS (you MUST pick one)
- **buy_to_enter** - Long position. Use when:
  - Consolidation with BULLISH structure + volume contracting, price near range low (anticipation), OR
  - breakout_confirmed = TRUE with direction = "up" (confirmed breakout)
  - MUST provide entry_price (at range low or above breakout level), stop_loss (BELOW entry), profit_target

- **sell_to_enter** - Short position. Use when:
  - Consolidation with BEARISH structure + volume contracting, price near range high (anticipation), OR
  - breakout_confirmed = TRUE with direction = "down" (confirmed breakdown)
  - MUST provide entry_price (at range high or below breakdown level), stop_loss (ABOVE entry), profit_target

- **hold** - No action. Use when:
  - No consolidation (trending market, wait for range to form)
  - Neutral structure (no higher lows or lower highs)
  - Breakout detected but NOT confirmed (no volume surge) - HIGH FALSE BREAKOUT RISK
  - Price is mid-range (not at good entry zone)
  - Unclear setup or poor R:R

- **close** - Exit existing position. Use when:
  - Breakout failed (price returned inside range)
  - Invalidation condition met

## ORDER TYPE
- **limit** - Preferred for anticipation entries at range extremes
- **market** - Use for confirmed breakouts when price has already broken out

## TRAILING STOP DISTANCE (trailing_stop_atr_multiplier)
For buy/sell signals, suggest a trailing stop distance as an ATR multiplier based on current volatility:
- **Low volatility** (tight squeeze, small ATR): 1.5-2.0x ATR
- **Normal volatility**: 2.0-3.0x ATR
- **High volatility** (post-breakout expansion, large ATR): 3.0-5.0x ATR
- **XAUUSD/Gold**: Typically 2.5-4.0x ATR
- **Forex majors**: Typically 1.5-2.5x ATR
- If unsure, default to 2.5x ATR

Remember: The best breakout trades come from CLEAR consolidation with CLEAR structure bias. When in doubt, HOLD."""


def _format_breakout_report(decision: QuantAnalystDecision, consolidation: Optional[Dict] = None) -> str:
    """Format the breakout decision into a human-readable report."""
    signal_str = decision.signal if isinstance(decision.signal, str) else decision.signal.value

    lines = [
        f"## BREAKOUT QUANT DECISION: **{signal_str.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
    ]

    # Add consolidation context
    if consolidation:
        squeeze_status = "ACTIVE" if consolidation.get("bb_squeeze") else "Inactive"
        vol_status = "CONTRACTING" if consolidation.get("volume_contracting") else "Normal"
        breakout_status = "CONFIRMED" if consolidation.get("breakout_confirmed") else (
            "Detected (no vol surge)" if consolidation.get("breakout_detected") else "None"
        )
        lines.extend([
            "### Consolidation Status",
            f"- **Squeeze**: {squeeze_status} ({consolidation.get('squeeze_strength', 0):.0f}%)",
            f"- **Range**: {consolidation.get('range_low', 0):.5f} - {consolidation.get('range_high', 0):.5f}",
            f"- **Structure Bias**: {consolidation.get('structure_bias', 'neutral').upper()}",
            f"- **Volume**: {vol_status}",
            f"- **Breakout**: {breakout_status}",
            "",
        ])

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


def get_breakout_decision_for_modal(breakout_decision: dict) -> dict:
    """
    Convert a breakout quant decision dict to trade modal format.

    Args:
        breakout_decision: The breakout_quant_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not breakout_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = breakout_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    order_type = breakout_decision.get("order_type", "limit")
    if isinstance(order_type, dict):
        order_type = order_type.get("value", "limit")

    return {
        "symbol": breakout_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "orderType": order_type,
        "suggestedEntry": breakout_decision.get("entry_price"),
        "suggestedStopLoss": breakout_decision.get("stop_loss"),
        "suggestedTakeProfit": breakout_decision.get("profit_target"),
        "rationale": f"BREAKOUT: {breakout_decision.get('justification', '')}. Invalidation: {breakout_decision.get('invalidation_condition', '')}",
        "confidence": breakout_decision.get("confidence", 0.5),
    }
