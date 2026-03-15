"""
Range-Bound Market Quant Analyst (SMC Levels)

A specialized quant analyst focused on trading ranging/sideways markets using
Smart Money Concepts levels as entry and exit zones.

Key concepts:
- Ranging market: ADX < 25, price oscillating between support and resistance
- SMC zones as boundaries: Order Blocks and FVGs define range extremes
- Mean reversion: In ranges, price tends to revert from extremes toward equilibrium
- Liquidity sweeps: False breakouts of range extremes create high-probability reversal entries

Trading approach:
- Identify ranging conditions (low ADX, no clear BOS trend)
- Map unmitigated OBs and FVGs as range boundaries
- Buy at bullish OBs/FVGs near range lows (discount zone)
- Sell at bearish OBs/FVGs near range highs (premium zone)
- Target the opposite side of the range or equilibrium
- Use liquidity sweeps as entry confirmation (sweep then reverse)
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
from tradingagents.schemas import QuantAnalystDecision, RiskLevel
from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
from tradingagents.dataflows.smc_trade_plan import safe_get


# Set up range quant logger
_range_logger = None


def _get_range_logger():
    """Get or create the range quant prompt logger."""
    global _range_logger
    if _range_logger is None:
        _range_logger = logging.getLogger("range_quant_prompts")
        _range_logger.setLevel(logging.DEBUG)

        log_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "logs", "range_quant_prompts"
        )
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f"range_quant_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)

        if not _range_logger.handlers:
            _range_logger.addHandler(file_handler)

    return _range_logger


def analyze_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 25,
) -> Dict[str, Any]:
    """
    Analyze price data for ranging market conditions.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        lookback: Number of candles to analyze

    Returns:
        Dict with range analysis:
        - is_ranging: bool - True if market is in a range
        - range_high: Upper boundary
        - range_low: Lower boundary
        - range_midpoint: Middle of range (equilibrium)
        - range_percent: Range width as % of price
        - mean_reversion_score: 0-100 (higher = stronger mean reversion)
        - price_position: "premium", "discount", or "equilibrium"
        - touches_high: Number of times price tested range high
        - touches_low: Number of times price tested range low
    """
    if len(close) < lookback:
        return {
            "is_ranging": False,
            "range_high": None,
            "range_low": None,
            "range_midpoint": None,
            "range_percent": None,
            "mean_reversion_score": 0.0,
            "price_position": "unknown",
            "touches_high": 0,
            "touches_low": 0,
            "trend_strength": 0.0,
            "position_pct": 50.0,
            "adx_proxy": 0.0,
        }

    recent_high = high[-lookback:]
    recent_low = low[-lookback:]
    recent_close = close[-lookback:]

    range_high = float(np.max(recent_high))
    range_low = float(np.min(recent_low))
    range_midpoint = (range_high + range_low) / 2
    range_size = range_high - range_low
    range_percent = float((range_size / range_midpoint) * 100) if range_midpoint > 0 else 0.0

    # Count touches of range extremes (within 15% of range edges)
    touch_threshold = range_size * 0.15
    touches_high = int(np.sum(recent_high >= (range_high - touch_threshold)))
    touches_low = int(np.sum(recent_low <= (range_low + touch_threshold)))

    # --- Trend strength: combine slope analysis with directional movement ---
    # 1. Linear regression slope (net displacement relative to range)
    x = np.arange(lookback, dtype=float)
    slope = float(np.polyfit(x, recent_close, 1)[0])
    net_displacement = abs(slope * lookback)
    trend_slope = net_displacement / range_size if range_size > 0 else 0.0

    # 2. ADX-like proxy using directional movement ratio
    #    Measures how much of the total movement is directional vs oscillating
    abs_return = abs(float(recent_close[-1]) - float(recent_close[0]))
    total_path = float(np.sum(np.abs(np.diff(recent_close))))
    # Efficiency ratio: 1.0 = pure trend, 0.0 = pure noise/range
    efficiency_ratio = abs_return / total_path if total_path > 0 else 0.0

    # Combined trend strength (weighted)
    trend_strength = float(trend_slope * 0.4 + efficiency_ratio * 0.6)

    # ADX proxy: scale efficiency ratio to 0-50 range (ADX-like)
    adx_proxy = float(efficiency_ratio * 50)

    # --- Mean reversion score ---
    # 1. Midpoint crosses: how often price crosses the range midpoint
    crosses_mid = 0
    for i in range(1, len(recent_close)):
        if (recent_close[i - 1] < range_midpoint and recent_close[i] >= range_midpoint) or \
           (recent_close[i - 1] > range_midpoint and recent_close[i] <= range_midpoint):
            crosses_mid += 1

    max_possible_crosses = lookback // 2
    cross_ratio = min(crosses_mid / max(max_possible_crosses, 1), 1.0)

    # 2. Touch ratio: how often price visits the range extremes
    touch_ratio = min((touches_high + touches_low) / max(lookback * 0.25, 1), 1.0)

    # 3. Low trend penalty: stronger trend = lower MR score
    trend_penalty = max(0.0, 1.0 - trend_strength * 2)

    # 4. Reversal count: how many times direction flips (up->down or down->up)
    diffs = np.diff(recent_close)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    max_sign_changes = lookback - 2
    reversal_ratio = sign_changes / max_sign_changes if max_sign_changes > 0 else 0

    mean_reversion_score = float(
        (cross_ratio * 0.30 + touch_ratio * 0.25 + trend_penalty * 0.25 + reversal_ratio * 0.20) * 100
    )
    mean_reversion_score = max(0.0, min(100.0, mean_reversion_score))

    # --- Is ranging ---
    # Require: low trend strength + touches at both extremes + decent MR score
    is_ranging = bool(
        trend_strength < 0.35
        and (touches_high >= 2 and touches_low >= 2)
        and mean_reversion_score > 45
        and adx_proxy < 20
    )

    # Current price position within range
    current_price = float(close[-1])
    if range_size > 0:
        position_pct = float((current_price - range_low) / range_size * 100)
    else:
        position_pct = 50.0

    if position_pct > 70:
        price_position = "premium"
    elif position_pct < 30:
        price_position = "discount"
    else:
        price_position = "equilibrium"

    # --- Higher-timeframe structural bias ---
    # Look at 3x lookback to detect if the range sits within a broader trend.
    # If the asset is structurally bullish, range sells are lower probability.
    htf_len = min(lookback * 3, len(close))
    htf_close = close[-htf_len:]
    htf_slope = float(np.polyfit(np.arange(htf_len, dtype=float), htf_close, 1)[0])
    htf_abs_return = abs(float(htf_close[-1]) - float(htf_close[0]))
    htf_path = float(np.sum(np.abs(np.diff(htf_close))))
    htf_efficiency = htf_abs_return / htf_path if htf_path > 0 else 0.0

    if htf_slope > 0 and htf_efficiency > 0.15:
        structural_bias = "bullish"
    elif htf_slope < 0 and htf_efficiency > 0.15:
        structural_bias = "bearish"
    else:
        structural_bias = "neutral"

    return {
        "is_ranging": is_ranging,
        "range_high": range_high,
        "range_low": range_low,
        "range_midpoint": range_midpoint,
        "range_percent": range_percent,
        "mean_reversion_score": mean_reversion_score,
        "price_position": price_position,
        "position_pct": position_pct,
        "touches_high": touches_high,
        "touches_low": touches_low,
        "trend_strength": float(trend_strength),
        "adx_proxy": adx_proxy,
        "structural_bias": structural_bias,
    }


def create_range_quant(llm, use_structured_output: bool = True):
    """
    Create a range-bound market quant analyst node using SMC levels.

    This analyst:
    - Identifies ranging market conditions (low ADX, oscillating price)
    - Maps SMC zones (OBs, FVGs) as range boundaries
    - Enters at range extremes with SMC confluence
    - Uses liquidity sweeps as entry confirmation
    - Targets opposite range extreme or equilibrium

    Args:
        llm: The language model to use for analysis
        use_structured_output: If True, uses LLM structured output for guaranteed JSON

    Returns:
        A function that processes state and returns the range analysis
    """

    structured_llm = None
    if use_structured_output:
        try:
            structured_llm = llm.with_structured_output(QuantAnalystDecision)
        except Exception as e:
            print(
                f"Warning: Structured output not supported for range quant, falling back to free-form: {e}"
            )
            structured_llm = None

    def range_quant_analyst_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Gather SMC data
        smc_context = state.get("smc_context") or ""
        smc_analysis = state.get("smc_analysis") or {}
        market_report = state.get("market_report") or ""

        # Regime info
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        trading_session = state.get("trading_session") or "unknown"

        # Price data for range analysis
        price_data = state.get("price_data") or {}
        high = price_data.get("high")
        low = price_data.get("low")
        close = price_data.get("close")

        # Perform range analysis
        range_analysis = None
        if high is not None and low is not None and close is not None:
            try:
                high_arr = np.array(high) if not isinstance(high, np.ndarray) else high
                low_arr = np.array(low) if not isinstance(low, np.ndarray) else low
                close_arr = np.array(close) if not isinstance(close, np.ndarray) else close
                range_analysis = analyze_range(high_arr, low_arr, close_arr)
            except Exception as e:
                print(f"Range analysis error: {e}")
                range_analysis = None

        # Build data context
        data_context = _build_range_data_context(
            ticker=ticker,
            current_price=current_price,
            smc_context=smc_context,
            smc_analysis=smc_analysis,
            market_report=market_report,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=trading_session,
            current_date=current_date,
            range_analysis=range_analysis,
        )

        # Build prompt
        trade_memories = state.get("trade_memories") or ""
        system_prompt = _build_range_prompt(data_context, trade_memories=trade_memories)

        # Log
        logger = _get_range_logger()
        logger.info(f"\n{'='*80}\nRANGE QUANT ANALYSIS - {ticker}\n{'='*80}")
        logger.info(f"Symbol: {ticker} | Date: {current_date} | Price: {current_price}")
        logger.info(f"Market Regime: {market_regime} | Volatility: {volatility_regime}")
        if range_analysis:
            logger.info(f"Range Analysis: {range_analysis}")
        logger.info(f"\n--- DATA CONTEXT ---\n{data_context}")
        logger.info(f"\n--- FULL PROMPT ---\n{system_prompt}")
        logger.info(f"\n{'='*80}\n")

        if structured_llm is not None:
            try:
                import time as _time

                _llm_start = _time.time()
                decision: QuantAnalystDecision = structured_llm.invoke(system_prompt)
                _llm_duration = _time.time() - _llm_start

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

                decision_dict = decision.model_dump()
                logger.info(f"Raw decision dict: {decision_dict}")
                logger.info(f"\n{'='*80}\n")

                report = _format_range_report(decision, range_analysis)

                logger.info(
                    f"Structured output SUCCESS - returning range_quant_decision with signal={decision.signal}"
                )
                return {
                    "range_quant_report": report,
                    "range_quant_decision": decision_dict,
                    "range_analysis": range_analysis,
                }
            except Exception as e:
                import traceback as _tb

                logger.error(f"Structured output failed: {e}")
                logger.error(f"Traceback:\n{_tb.format_exc()}")
                print(f"Structured output failed for range quant: {e}")

        # Fallback to unstructured
        logger.info(f"--- Falling back to unstructured LLM call ---")
        import time as _time

        _llm_start = _time.time()
        response = llm.invoke(system_prompt)
        _llm_duration = _time.time() - _llm_start
        report = response.content if hasattr(response, "content") else str(response)

        logger.info(
            f"--- LLM RESPONSE (Unstructured) [took {_llm_duration:.1f}s] ---"
        )
        logger.info(f"{report[:2000]}...")
        logger.info(f"\n{'='*80}\n")

        logger.warning(f"Returning range_quant_decision=None (unstructured fallback)")
        return {
            "range_quant_report": report,
            "range_quant_decision": None,
            "range_analysis": range_analysis,
        }

    return range_quant_analyst_node


def _build_range_data_context(
    ticker: str,
    current_price: Optional[float],
    smc_context: str,
    smc_analysis: dict,
    market_report: str,
    market_regime: str,
    volatility_regime: str,
    trading_session: str,
    current_date: str,
    range_analysis: Optional[Dict[str, Any]],
) -> str:
    """Build comprehensive data context for range-bound analysis."""

    sections = []

    # Current market data
    if current_price:
        sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price**: {current_price:.5f}
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

    # Range analysis
    if range_analysis:
        ranging_status = "YES - RANGE DETECTED" if range_analysis.get("is_ranging") else "No clear range"
        price_pos = range_analysis.get("price_position", "unknown").upper()

        range_high = range_analysis.get("range_high")
        range_low = range_analysis.get("range_low")
        range_mid = range_analysis.get("range_midpoint")
        range_high_str = f"{range_high:.5f}" if range_high is not None else "N/A"
        range_low_str = f"{range_low:.5f}" if range_low is not None else "N/A"
        range_mid_str = f"{range_mid:.5f}" if range_mid is not None else "N/A"

        sections.append(f"""## RANGE ANALYSIS (Critical for Range Trading)
- **Is Ranging**: {ranging_status}
- **Range High (Resistance)**: {range_high_str}
- **Range Low (Support)**: {range_low_str}
- **Range Midpoint (Equilibrium)**: {range_mid_str}
- **Range Width**: {range_analysis.get('range_percent', 0):.2f}% of price
- **Price Position**: {price_pos} ({range_analysis.get('position_pct', 50):.0f}% within range)
- **Mean Reversion Score**: {range_analysis.get('mean_reversion_score', 0):.0f}/100 (higher = stronger range behavior)
- **Touches of Range High**: {range_analysis.get('touches_high', 0)}
- **Touches of Range Low**: {range_analysis.get('touches_low', 0)}
- **Trend Strength**: {range_analysis.get('trend_strength', 0):.2f} (lower = more range-like)
- **Structural Bias (Higher Timeframe)**: {range_analysis.get('structural_bias', 'neutral').upper()}

### Range Interpretation:
- PREMIUM (>70%): Price near range top — favor SELL entries at bearish OBs/FVGs
- DISCOUNT (<30%): Price near range bottom — favor BUY entries at bullish OBs/FVGs
- EQUILIBRIUM (30-70%): Mid-range — wait for price to reach an extreme
- Multiple touches = proven boundary. More touches = stronger level.
- Mean Reversion Score >60 = strong range. >80 = very strong range.

### Structural Bias Filter (IMPORTANT):
- **BULLISH structural bias**: The broader trend is UP. Range BUYS at discount are HIGH PROBABILITY. Range SELLS at premium are LOW PROBABILITY — the range is more likely to break upward. Prefer buy-only or reduce sell confidence.
- **BEARISH structural bias**: The broader trend is DOWN. Range SELLS at premium are HIGH PROBABILITY. Range BUYS at discount are LOW PROBABILITY — the range is more likely to break downward. Prefer sell-only or reduce buy confidence.
- **NEUTRAL structural bias**: No dominant trend. Both buy and sell at range extremes are valid.
""")
    else:
        sections.append("""## RANGE ANALYSIS
- Price data not available for range analysis
- Use regime indicators and SMC levels to assess range boundaries
""")

    # SMC context — crucial for range trading with SMC levels
    if smc_context:
        sections.append(smc_context)

    # Technical indicators
    if market_report:
        sections.append(market_report)

    return "\n".join(sections)


def _build_range_prompt(data_context: str, trade_memories: str = None) -> str:
    """Build the complete range quant analyst prompt."""

    memories_section = ""
    if trade_memories:
        memories_section = f"""

{trade_memories}

IMPORTANT: The above lessons are from YOUR past range trades. Study what went wrong
and apply corrections. Pay special attention to fakeout and premature entry lessons.

"""

    return f"""You are a range-bound market specialist using Smart Money Concepts (SMC) levels. You excel at identifying sideways markets and trading reversals at range extremes using institutional zones.

## YOUR TRADING EDGE: RANGE-BOUND MARKETS WITH SMC LEVELS

Most markets spend 60-70% of time in ranges. Your edge is:
1. Identify when price is ranging (low ADX, oscillating between S/R)
2. Map SMC zones (OBs, FVGs, equal levels) as high-probability range boundaries
3. Enter at range extremes where SMC zones provide confluence
4. Use liquidity sweeps as entry confirmation (smart money grabs stops then reverses)
5. Target the opposite range extreme or equilibrium

## RANGE TRADING RULES WITH SMC

### 1. CONFIRMING A RANGE
- **ADX < 25**: Weak or no trend = range conditions
- **No dominant BOS**: No clear series of higher highs/higher lows (or lower lows/lower highs)
- **CHoCH present**: Multiple changes of character = choppy/ranging
- **Multiple touches**: Range high and low tested 2+ times each
- **Mean Reversion Score > 50**: Price regularly returns to midpoint

### 2. MAPPING RANGE BOUNDARIES WITH SMC
- **Range Resistance**: Bearish OBs and bearish FVGs near range high
- **Range Support**: Bullish OBs and bullish FVGs near range low
- **Equal Highs/Lows**: Proven S/R levels with 3+ touches are strongest boundaries
- **Liquidity pools**: Above range highs (buy-side) and below range lows (sell-side)
- Unmitigated zones are active; mitigated zones are spent

### 3. ENTRY STRATEGIES

**Setup 1: Buy at Range Low + Bullish OB/FVG**
- Price drops to range low (discount zone, <30% of range)
- Bullish OB or FVG overlaps with range support
- Enter LONG at the OB/FVG zone
- SL: Below the OB/FVG zone (beyond the range low + ATR buffer)
- TP: Range midpoint (conservative) or range high (aggressive)

**Setup 2: Sell at Range High + Bearish OB/FVG**
- Price rises to range high (premium zone, >70% of range)
- Bearish OB or FVG overlaps with range resistance
- Enter SHORT at the OB/FVG zone
- SL: Above the OB/FVG zone (beyond the range high + ATR buffer)
- TP: Range midpoint (conservative) or range low (aggressive)

**Setup 3: Liquidity Sweep Reversal (Highest Probability)**
- Price sweeps beyond range high (takes buy-side liquidity) then reverses
  - Enter SHORT after sweep with CHoCH confirmation
- Price sweeps beyond range low (takes sell-side liquidity) then reverses
  - Enter LONG after sweep with CHoCH confirmation
- SL: Beyond the sweep high/low + ATR buffer
- TP: Opposite range extreme

**Setup 4: Equal Level Bounce**
- Price approaches proven equal highs (3+ touches) or equal lows (3+ touches)
- These are liquidity magnets — expect price to reach them
- After reaching the level, look for rejection candles and enter reversal
- Works best when confluent with OB or FVG at the same zone

### 4. STOP LOSS PLACEMENT (CRITICAL)
- For BUY: Stop loss MUST be BELOW entry price
  - Place below the OB/FVG zone + at least 1x ATR buffer
  - Must be beyond the range low to survive a liquidity sweep
- For SELL: Stop loss MUST be ABOVE entry price
  - Place above the OB/FVG zone + at least 1x ATR buffer
  - Must be beyond the range high to survive a liquidity sweep
- **MINIMUM SL DISTANCE**: At least 1x ATR from entry. SL within 0.5% is TOO TIGHT.
- For gold (XAUUSD): minimum SL should be $15-30+ from entry
- If you cannot place SL with adequate distance while maintaining R:R >= 1.5:1, output "hold"

### 5. TAKE PROFIT TARGETS
- **Conservative**: Range midpoint (equilibrium) — safest target in ranges
- **Standard**: Opposite range extreme — full range move
- **With confluence**: Proven equal level or strong OB on the opposite side
- Minimum R:R: 1.5:1

### 6. STRUCTURAL BIAS FILTER (CRITICAL FOR PROFITABILITY)
Backtesting shows that trading WITH the structural bias is far more profitable than fading it:
- **BULLISH structural bias** (e.g., XAUUSD in an uptrend): BUY at range discount = ~70% win rate. SELL at range premium = ~35% win rate. **Strongly prefer BUY trades. Reduce confidence on SELL trades or skip them.**
- **BEARISH structural bias**: SELL at range premium is the higher-probability trade. Skip or reduce confidence on BUY trades.
- **NEUTRAL**: Both sides equally valid.
- Check the "Structural Bias (Higher Timeframe)" field in your data and factor it into every decision.

### 7. WHEN NOT TO TRADE
- **Trending market**: ADX > 25 with clear BOS series — DO NOT fade the trend
- **Breakout in progress**: Price breaking range with momentum + volume — wait for new range
- **Range too narrow**: If range width < 1x ATR, not enough room for a trade
- **No SMC confluence**: Range extremes with no OB/FVG support — skip
- **Mid-range**: Price at equilibrium with no clear direction — wait for extreme

### 8. CRITICAL: DO NOT FADE A BREAKOUT
- If price breaks the range with a strong BOS + high volume, the range is OVER
- Do not try to sell a breakout above range high or buy a breakdown below range low
- Instead output "hold" and wait for a new range to form
- A confirmed breakout = transition from range to trend

## RULES YOU MUST NEVER BREAK

1. **Risk no more than 1-2% of account per trade**
2. **Never hold >3 positions at once**
3. **Never pyramid or average down**
4. **Always pre-define profit target, stop loss, and invalidation**
5. **Never fade a confirmed breakout — if ADX > 25 with BOS, DO NOT counter-trade**
6. **Stop loss MUST be on correct side of entry**
7. **Only trade at range extremes with SMC confluence, never mid-range**

{data_context}
{memories_section}
## YOUR TASK

Analyze the range and SMC data to make a range-trading decision.

Think step-by-step:
1. Is the market ranging? (Check ADX, BOS pattern, mean reversion score)
2. Where are the range boundaries? (Map OBs, FVGs, equal levels at extremes)
3. Is price at a range extreme? (Premium/Discount zone, not equilibrium)
4. Is there SMC confluence at this extreme? (OB + FVG overlap, unmitigated zones)
5. Has there been a liquidity sweep? (Sweep beyond range = highest probability entry)
6. Where should SL be placed? (Beyond OB/FVG + ATR buffer, beyond range extreme)
7. What is the TP target? (Opposite extreme or midpoint)
8. Is R:R acceptable? (Must be >= 1.5:1)

## SIGNAL OPTIONS (you MUST pick one)
- **buy_to_enter** - Long position. Use when:
  - Market is ranging AND price is in discount zone (<30% of range)
  - Bullish OB or FVG confluence at range support
  - OR: Sell-side liquidity sweep just occurred (price swept below range then reversed)
  - MUST provide entry_price, stop_loss (BELOW entry), profit_target

- **sell_to_enter** - Short position. Use when:
  - Market is ranging AND price is in premium zone (>70% of range)
  - Bearish OB or FVG confluence at range resistance
  - OR: Buy-side liquidity sweep just occurred (price swept above range then reversed)
  - MUST provide entry_price, stop_loss (ABOVE entry), profit_target

- **hold** - No action. Use when:
  - Market is NOT ranging (trending with clear BOS)
  - Price is at equilibrium (mid-range, no edge)
  - No SMC confluence at current price level
  - Breakout is in progress (do not fade it)
  - Range is too narrow for a trade

- **close** - Exit existing position. Use when:
  - Price reached opposite range extreme (TP zone)
  - Range is breaking (confirmed BOS beyond range)
  - Setup invalidated

## ORDER TYPE
- **limit** - Preferred for entries at OB/FVG zones (waiting for price to reach zone)
- **market** - Use when price is already at the zone and showing rejection, or after a liquidity sweep

## TRAILING STOP DISTANCE (trailing_stop_atr_multiplier)
Range trades typically use tighter trailing stops since the move is capped:
- **Tight range** (small ATR, narrow range): 1.5-2.0x ATR
- **Normal range**: 2.0-2.5x ATR
- **Wide range** (large ATR): 2.5-3.5x ATR
- **XAUUSD/Gold in range**: 2.0-3.0x ATR
- If unsure, default to 2.0x ATR (tighter than trend-following)

Remember:
- Ranges are your edge — most traders lose money fighting ranges by trying to catch breakouts
- SMC zones define WHERE to enter within the range
- Liquidity sweeps beyond range = smart money entry signal
- Always trade FROM the extreme, never from the middle
- When the range breaks, STOP trading the range"""


def _format_range_report(
    decision: QuantAnalystDecision, range_analysis: Optional[Dict] = None
) -> str:
    """Format the range quant decision into a human-readable report."""
    signal_str = (
        decision.signal
        if isinstance(decision.signal, str)
        else decision.signal.value
    )

    lines = [
        f"## RANGE QUANT DECISION: **{signal_str.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
    ]

    # Add range context
    if range_analysis:
        ranging_status = "ACTIVE" if range_analysis.get("is_ranging") else "Not confirmed"
        lines.extend(
            [
                "### Range Status",
                f"- **Ranging**: {ranging_status} (MR Score: {range_analysis.get('mean_reversion_score', 0):.0f}/100)",
                f"- **Range**: {range_analysis.get('range_low', 0):.5f} - {range_analysis.get('range_high', 0):.5f}",
                f"- **Price Position**: {range_analysis.get('price_position', 'unknown').upper()} ({range_analysis.get('position_pct', 50):.0f}%)",
                f"- **Touches**: High={range_analysis.get('touches_high', 0)}, Low={range_analysis.get('touches_low', 0)}",
                "",
            ]
        )

    if signal_str in ["buy_to_enter", "sell_to_enter"]:
        lines.append("### Trade Parameters")
        if decision.order_type:
            order_type_str = (
                decision.order_type
                if isinstance(decision.order_type, str)
                else decision.order_type.value
            )
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


def get_range_quant_decision_for_modal(range_decision: dict) -> dict:
    """
    Convert a range quant decision dict to trade modal format.

    Args:
        range_decision: The range_quant_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not range_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = range_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    order_type = range_decision.get("order_type", "limit")
    if isinstance(order_type, dict):
        order_type = order_type.get("value", "limit")

    return {
        "symbol": range_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "orderType": order_type,
        "suggestedEntry": range_decision.get("entry_price"),
        "suggestedStopLoss": range_decision.get("stop_loss"),
        "suggestedTakeProfit": range_decision.get("profit_target"),
        "rationale": f"RANGE: {range_decision.get('justification', '')}. Invalidation: {range_decision.get('invalidation_condition', '')}",
        "confidence": range_decision.get("confidence", 0.5),
    }


def analyze_smc_for_range(
    df,
    current_price: float,
    swing_lookback: int = 5,
    ob_strength_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run SMC analysis optimized for range trading context.

    Args:
        df: DataFrame with OHLCV data
        current_price: Current market price
        swing_lookback: Periods to look back for swing points
        ob_strength_threshold: Minimum OB strength (0-1)

    Returns:
        Dict with 'smc_analysis' (dict) and 'smc_context' (string)
    """
    from tradingagents.agents.analysts.smc_quant import _format_smc_analysis_for_prompt

    analyzer = SmartMoneyAnalyzer(
        swing_lookback=swing_lookback,
        ob_strength_threshold=ob_strength_threshold,
    )
    analysis = analyzer.analyze_full_smc(df, current_price=current_price)
    context = _format_smc_analysis_for_prompt(analysis, current_price)

    return {
        "smc_analysis": analysis,
        "smc_context": context,
    }
