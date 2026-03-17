"""
SMC Multi-Timeframe OTE & Channel Quant Analyst
Strategy Name: SMCICT_Video_Strategy

A systematic quant trader combining Smart Money Concepts with:
- Multi-Timeframe Analysis: Higher TF (D1/H4) for bias, Lower TF (H1/M15) for entries
- OTE (Optimal Trade Entry): Fibonacci 0.5, 0.618, 0.786 retracement zones
- Parallel Distribution Channels: Linear regression on swing points (20-30 swings)
- Weekend Gap Analysis: Monday open vs Friday close for bias confirmation
- Protected High/Low Flips: Structure level violations and polarity changes

EXECUTION RULES (from video strategy):
1. Higher TF Bias: HH+HL+bullish BOS = Bullish | LH+LL+bearish BOS = Bearish
2. Weekend gap must confirm bias (optional but strong filter)
3. Only trade in direction of HTF bias on LTF
4. Wait for pullback to 0.618 Fib (OTE) of most recent swing leg
5. Pullback must occur inside or touch the parallel distribution channel
6. Confirmation REQUIRED: weekend gap in bias direction OR Protected Low/High flip on LTF
7. Entry on close of confirmation candle at OTE level
8. SL: Below swing low (bullish) or above swing high (bearish)
9. TP: Fib extension (1.272 or 1.618) OR opposite channel edge
10. Risk 1% of account per trade

Based on ICT (Inner Circle Trader) methodology.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from tradingagents.schemas import QuantAnalystDecision, RiskLevel
from tradingagents.dataflows.smc_trade_plan import safe_get


# Logger setup
_mtf_logger = None


def _get_mtf_logger():
    """Get or create the MTF quant prompt logger."""
    global _mtf_logger
    if _mtf_logger is None:
        _mtf_logger = logging.getLogger("smc_mtf_quant_prompts")
        _mtf_logger.setLevel(logging.DEBUG)

        log_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "logs", "smc_mtf_quant_prompts"
        )
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f"smc_mtf_quant_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler.setFormatter(formatter)

        if not _mtf_logger.handlers:
            _mtf_logger.addHandler(file_handler)

    return _mtf_logger


# =============================================================================
# Data Structures
# =============================================================================

class BiasDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SwingPoint:
    """A swing high or low point."""
    type: str  # "high" or "low"
    price: float
    index: int
    timestamp: Optional[str] = None
    broken: bool = False
    break_index: Optional[int] = None


@dataclass
class StructureBreak:
    """BOS or CHoCH event."""
    type: str  # "bos" or "choch"
    direction: str  # "bullish" or "bearish"
    price: float
    index: int
    swing_type: str  # "high" or "low" that was broken
    choch_reason: Optional[str] = None  # "failed_extension" or "retraced_swing" for CHoCH


# Configurable parameters (per video strategy)
CHOCH_FAIL_BARS = 10  # N bars to fail making new extreme = CHoCH
CHANNEL_SWING_LOOKBACK = 25  # Fit channel on 20-30 swings
PROTECTED_LEVEL_MIN_BARS = 10  # Minimum bars for level to be "protected"
RISK_PER_TRADE_PCT = 1.0  # Risk 1% of account per trade


@dataclass
class OTEZone:
    """Optimal Trade Entry zone based on Fibonacci retracement."""
    direction: str  # "bullish" or "bearish"
    swing_high: float
    swing_low: float
    fib_50: float
    fib_618: float
    fib_786: float
    ote_top: float  # Upper bound of OTE (0.5)
    ote_bottom: float  # Lower bound of OTE (0.786)
    # Fib extensions for take profit
    fib_ext_1272: float  # 127.2% extension
    fib_ext_1618: float  # 161.8% extension


@dataclass
class RegressionChannel:
    """Linear regression channel fitted on swing points (not raw prices)."""
    slope: float
    intercept: float
    upper_band: float  # Current upper channel value
    lower_band: float  # Current lower channel value
    middle: float  # Current regression line value
    std_dev: float
    r_squared: float  # Fit quality (0-1)
    direction: str  # "up", "down", "flat"
    num_swings: int  # Number of swings used for fitting


@dataclass
class WeekendGap:
    """Weekend gap analysis."""
    exists: bool
    direction: str  # "up", "down", "none"
    gap_size: float  # Absolute gap size
    gap_percent: float  # Gap as % of Friday close
    friday_close: float
    monday_open: float


@dataclass
class ProtectedLevel:
    """A protected high or low that hasn't been violated."""
    type: str  # "high" or "low"
    price: float
    index: int
    bars_protected: int  # How many bars it's been unbroken
    flipped: bool = False  # True if price closed beyond and held
    flip_index: Optional[int] = None


@dataclass
class MTFAnalysis:
    """Complete multi-timeframe analysis result."""
    # Higher TF bias
    higher_tf_bias: str  # "bullish", "bearish", "neutral"
    higher_tf_structure: List[StructureBreak]
    higher_tf_swing_sequence: str  # "HH-HL", "LH-LL", "mixed"

    # Lower TF analysis
    lower_tf_bias: str
    lower_tf_structure: List[StructureBreak]

    # OTE zones
    ote_zone: Optional[OTEZone]
    price_in_ote: bool

    # Channel analysis
    channel: Optional[RegressionChannel]
    price_in_channel: str  # "upper", "lower", "middle", "outside"

    # Weekend gap
    weekend_gap: Optional[WeekendGap]

    # Protected levels
    protected_highs: List[ProtectedLevel]
    protected_lows: List[ProtectedLevel]
    recent_flip: Optional[ProtectedLevel]

    # Confluence
    alignment_score: int  # 0-100
    trade_bias: str  # Final recommendation

    # Entry confirmation (per video strategy: REQUIRED)
    # Must have weekend gap confirmation OR protected level flip
    has_entry_confirmation: bool
    confirmation_type: Optional[str]  # "weekend_gap", "protected_flip", or None

    # Price touches/inside channel (per video strategy: REQUIRED)
    price_in_or_touches_channel: bool


# =============================================================================
# Indicator Calculations
# =============================================================================

def detect_swing_points(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 5
) -> List[SwingPoint]:
    """
    Detect swing highs and lows using peak/trough detection.

    Similar to scipy.signal.find_peaks but manual implementation
    for better control over parameters.

    Args:
        high: Array of high prices
        low: Array of low prices
        lookback: Minimum distance between swings (like find_peaks distance param)

    Returns:
        List of SwingPoint objects sorted by index
    """
    swings = []
    n = len(high)

    if n < lookback * 2 + 1:
        return swings

    for i in range(lookback, n - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if high[i - j] >= high[i] or high[i + j] >= high[i]:
                is_swing_high = False
                break

        if is_swing_high:
            swings.append(SwingPoint(
                type="high",
                price=float(high[i]),
                index=i
            ))

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if low[i - j] <= low[i] or low[i + j] <= low[i]:
                is_swing_low = False
                break

        if is_swing_low:
            swings.append(SwingPoint(
                type="low",
                price=float(low[i]),
                index=i
            ))

    return sorted(swings, key=lambda x: x.index)


def detect_structure_breaks(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    swings: List[SwingPoint],
    choch_fail_bars: int = CHOCH_FAIL_BARS
) -> Tuple[List[StructureBreak], str]:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).

    BOS: Close above previous swing high (bullish) or below swing low (bearish)
    CHoCH: After BOS, if price:
        1. Fails to make new extreme in N bars, OR
        2. Retraces past the last swing point

    Args:
        close: Array of close prices
        high: Array of high prices
        low: Array of low prices
        swings: List of swing points
        choch_fail_bars: Number of bars to fail making new extreme = CHoCH

    Returns:
        Tuple of (list of structure breaks, swing sequence string like "HH-HL")
    """
    breaks = []

    if len(swings) < 3:
        return breaks, "insufficient"

    # Track swing sequence for bias
    swing_highs = [s for s in swings if s.type == "high"]
    swing_lows = [s for s in swings if s.type == "low"]

    # Determine swing sequence (HH-HL or LH-LL)
    sequence = []
    if len(swing_highs) >= 2:
        if swing_highs[-1].price > swing_highs[-2].price:
            sequence.append("HH")
        else:
            sequence.append("LH")

    if len(swing_lows) >= 2:
        if swing_lows[-1].price > swing_lows[-2].price:
            sequence.append("HL")
        else:
            sequence.append("LL")

    swing_sequence = "-".join(sequence) if sequence else "unknown"

    # Track current bias and last extreme for CHoCH detection
    current_bias = "neutral"
    last_extreme_price = 0.0
    last_bos_break_index = 0  # Track where last BOS occurred

    for i, swing in enumerate(swings[:-1]):
        # Look for close that breaks this swing
        for j in range(swing.index + 1, len(close)):
            if swing.type == "high" and close[j] > swing.price:
                # Broke a high
                if current_bias == "bearish":
                    # CHoCH - broke high in downtrend (retraced past last swing)
                    breaks.append(StructureBreak(
                        type="choch",
                        direction="bullish",
                        price=swing.price,
                        index=j,
                        swing_type="high",
                        choch_reason="retraced_swing"
                    ))
                    current_bias = "bullish"
                else:
                    # BOS - broke high in uptrend or neutral
                    breaks.append(StructureBreak(
                        type="bos",
                        direction="bullish",
                        price=swing.price,
                        index=j,
                        swing_type="high"
                    ))
                    current_bias = "bullish"
                    last_bos_break_index = j
                    last_extreme_price = high[j]
                swing.broken = True
                swing.break_index = j
                break

            elif swing.type == "low" and close[j] < swing.price:
                # Broke a low
                if current_bias == "bullish":
                    # CHoCH - broke low in uptrend (retraced past last swing)
                    breaks.append(StructureBreak(
                        type="choch",
                        direction="bearish",
                        price=swing.price,
                        index=j,
                        swing_type="low",
                        choch_reason="retraced_swing"
                    ))
                    current_bias = "bearish"
                else:
                    # BOS - broke low in downtrend or neutral
                    breaks.append(StructureBreak(
                        type="bos",
                        direction="bearish",
                        price=swing.price,
                        index=j,
                        swing_type="low"
                    ))
                    current_bias = "bearish"
                    last_bos_break_index = j
                    last_extreme_price = low[j]
                swing.broken = True
                swing.break_index = j
                break

    # Check for CHoCH via "failed extension" (no new extreme in N bars after BOS)
    if breaks and current_bias != "neutral" and last_bos_break_index > 0:
        last_break = breaks[-1]
        if last_break.type == "bos":
            bars_since = len(close) - last_bos_break_index - 1

            if bars_since >= choch_fail_bars:
                # Check if we made a new extreme since the BOS
                if current_bias == "bullish":
                    # Should have made higher high
                    max_since_bos = np.max(high[last_break.index:])
                    if max_since_bos <= last_extreme_price:
                        # Failed to extend - this is a CHoCH signal
                        breaks.append(StructureBreak(
                            type="choch",
                            direction="bearish",
                            price=float(low[last_break.index + choch_fail_bars]),
                            index=last_break.index + choch_fail_bars,
                            swing_type="failed_high",
                            choch_reason="failed_extension"
                        ))
                else:
                    # Should have made lower low
                    min_since_bos = np.min(low[last_break.index:])
                    if min_since_bos >= last_extreme_price:
                        # Failed to extend - this is a CHoCH signal
                        breaks.append(StructureBreak(
                            type="choch",
                            direction="bullish",
                            price=float(high[last_break.index + choch_fail_bars]),
                            index=last_break.index + choch_fail_bars,
                            swing_type="failed_low",
                            choch_reason="failed_extension"
                        ))

    return breaks, swing_sequence


def calculate_ote_zone(
    swings: List[SwingPoint],
    current_price: float,
    bias: str
) -> Optional[OTEZone]:
    """
    Calculate Optimal Trade Entry zone using Fibonacci retracement.

    For bullish bias: OTE is 50-78.6% retracement of recent swing low to high
    For bearish bias: OTE is 50-78.6% retracement of recent swing high to low

    Also calculates Fib extensions (1.272, 1.618) for take profit targets.

    Args:
        swings: List of swing points
        current_price: Current market price
        bias: "bullish" or "bearish"

    Returns:
        OTEZone if valid swing leg found, None otherwise
    """
    highs = [s for s in swings if s.type == "high"]
    lows = [s for s in swings if s.type == "low"]

    if not highs or not lows:
        return None

    # Get most recent swing high and low
    recent_high = max(highs, key=lambda x: x.index)
    recent_low = max(lows, key=lambda x: x.index)

    swing_high = recent_high.price
    swing_low = recent_low.price
    range_size = swing_high - swing_low

    if range_size <= 0:
        return None

    # Calculate Fibonacci retracement levels
    fib_50 = swing_low + range_size * 0.5
    fib_618 = swing_low + range_size * 0.618
    fib_786 = swing_low + range_size * 0.786

    if bias == "bullish":
        # For bullish, OTE is discount zone (retracement from high)
        # Entry zone: 50%-78.6% retracement from high
        ote_top = swing_high - range_size * 0.5  # 50% retracement
        ote_bottom = swing_high - range_size * 0.786  # 78.6% retracement

        # Fib extensions for TP (project from swing low through swing high)
        fib_ext_1272 = swing_low + range_size * 1.272
        fib_ext_1618 = swing_low + range_size * 1.618

        return OTEZone(
            direction="bullish",
            swing_high=swing_high,
            swing_low=swing_low,
            fib_50=fib_50,
            fib_618=fib_618,
            fib_786=fib_786,
            ote_top=ote_top,
            ote_bottom=ote_bottom,
            fib_ext_1272=fib_ext_1272,
            fib_ext_1618=fib_ext_1618
        )
    else:
        # For bearish, OTE is premium zone (retracement from low)
        # Entry zone: 50%-78.6% retracement from low
        ote_bottom = swing_low + range_size * 0.5  # 50% retracement
        ote_top = swing_low + range_size * 0.786  # 78.6% retracement

        # Fib extensions for TP (project from swing high through swing low)
        fib_ext_1272 = swing_high - range_size * 1.272
        fib_ext_1618 = swing_high - range_size * 1.618

        return OTEZone(
            direction="bearish",
            swing_high=swing_high,
            swing_low=swing_low,
            fib_50=fib_50,
            fib_618=fib_618,
            fib_786=fib_786,
            ote_top=ote_top,
            ote_bottom=ote_bottom,
            fib_ext_1272=fib_ext_1272,
            fib_ext_1618=fib_ext_1618
        )


def calculate_regression_channel(
    close: np.ndarray,
    swings: Optional[List[SwingPoint]] = None,
    swing_lookback: int = CHANNEL_SWING_LOOKBACK,
    fallback_bar_lookback: int = 50
) -> Optional[RegressionChannel]:
    """
    Calculate linear regression channel fitted on swing points.

    Per video strategy: fit channel on the last 20-30 swings (not raw prices).
    Price should mean-revert to the opposite channel line.

    Falls back to close prices if insufficient swings.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        swings: List of swing points (preferred)
        swing_lookback: Number of swings to use for channel (20-30 recommended)
        fallback_bar_lookback: Bars to use if swings unavailable

    Returns:
        RegressionChannel with slope, bands, and fit quality
    """
    # Try to fit on swing points first (per video strategy)
    if swings and len(swings) >= 5:
        # Use last N swings
        recent_swings = swings[-swing_lookback:] if len(swings) > swing_lookback else swings

        # Extract swing prices and their x-positions (indices)
        swing_indices = np.array([s.index for s in recent_swings], dtype=float)
        swing_prices = np.array([s.price for s in recent_swings], dtype=float)

        if len(swing_indices) >= 3:
            # Normalize x to start from 0
            x = swing_indices - swing_indices[0]

            # Linear regression on swing points
            coeffs = np.polyfit(x, swing_prices, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])

            # Calculate fitted values and residuals
            fitted = slope * x + intercept
            residuals = swing_prices - fitted
            std_dev = float(np.std(residuals))

            # R-squared (fit quality)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((swing_prices - np.mean(swing_prices)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Project channel to current bar
            current_x = len(close) - 1 - swing_indices[0]
            current_middle = slope * current_x + intercept

            # Use 2 standard deviations for channel width
            upper_band = float(current_middle + 2 * std_dev)
            lower_band = float(current_middle - 2 * std_dev)

            # Determine direction
            if slope > 0.0001 * abs(current_middle):
                direction = "up"
            elif slope < -0.0001 * abs(current_middle):
                direction = "down"
            else:
                direction = "flat"

            return RegressionChannel(
                slope=slope,
                intercept=intercept,
                upper_band=upper_band,
                lower_band=lower_band,
                middle=float(current_middle),
                std_dev=std_dev,
                r_squared=r_squared,
                direction=direction,
                num_swings=len(recent_swings)
            )

    # Fallback: fit on close prices if insufficient swings
    if len(close) < fallback_bar_lookback:
        return None

    recent_close = close[-fallback_bar_lookback:]
    x = np.arange(fallback_bar_lookback, dtype=float)

    # Linear regression
    coeffs = np.polyfit(x, recent_close, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # Calculate fitted values and residuals
    fitted = slope * x + intercept
    residuals = recent_close - fitted
    std_dev = float(np.std(residuals))

    # R-squared (fit quality)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((recent_close - np.mean(recent_close)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Current channel values (at last bar)
    current_middle = slope * (fallback_bar_lookback - 1) + intercept

    # Use 2 standard deviations for channel width
    upper_band = float(current_middle + 2 * std_dev)
    lower_band = float(current_middle - 2 * std_dev)

    # Determine direction
    if slope > 0.0001 * abs(current_middle):
        direction = "up"
    elif slope < -0.0001 * abs(current_middle):
        direction = "down"
    else:
        direction = "flat"

    return RegressionChannel(
        slope=slope,
        intercept=intercept,
        upper_band=upper_band,
        lower_band=lower_band,
        middle=float(current_middle),
        std_dev=std_dev,
        r_squared=r_squared,
        direction=direction,
        num_swings=0  # Fallback mode, no swings used
    )


def detect_weekend_gap(
    df: pd.DataFrame
) -> Optional[WeekendGap]:
    """
    Detect weekend gap between Friday close and Monday open.

    Looks for the most recent Friday-Monday transition in the data.

    Args:
        df: DataFrame with datetime index and OHLC data

    Returns:
        WeekendGap analysis or None if no weekend found
    """
    if len(df) < 5:
        return None

    # Try to find Friday and Monday bars
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Get day of week (Monday=0, Friday=4)
        days = df.index.dayofweek

        # Find most recent Friday
        friday_mask = days == 4
        if not friday_mask.any():
            return WeekendGap(
                exists=False,
                direction="none",
                gap_size=0.0,
                gap_percent=0.0,
                friday_close=0.0,
                monday_open=0.0
            )

        friday_idx = df.index[friday_mask][-1]
        friday_close = float(df.loc[friday_idx, 'close'])

        # Find Monday after this Friday
        monday_mask = (days == 0) & (df.index > friday_idx)
        if not monday_mask.any():
            return WeekendGap(
                exists=False,
                direction="none",
                gap_size=0.0,
                gap_percent=0.0,
                friday_close=friday_close,
                monday_open=0.0
            )

        monday_idx = df.index[monday_mask][0]
        monday_open = float(df.loc[monday_idx, 'open'])

        gap_size = monday_open - friday_close
        gap_percent = (gap_size / friday_close) * 100 if friday_close > 0 else 0.0

        if gap_size > 0:
            direction = "up"
        elif gap_size < 0:
            direction = "down"
        else:
            direction = "none"

        return WeekendGap(
            exists=abs(gap_percent) > 0.05,  # At least 0.05% gap
            direction=direction,
            gap_size=abs(gap_size),
            gap_percent=gap_percent,
            friday_close=friday_close,
            monday_open=monday_open
        )

    except Exception:
        return None


def detect_protected_levels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swings: List[SwingPoint],
    min_bars_protected: int = 10
) -> Tuple[List[ProtectedLevel], List[ProtectedLevel], Optional[ProtectedLevel]]:
    """
    Detect protected highs and lows (unbroken swing points).

    A protected level is a swing high/low that hasn't been violated
    for a minimum number of bars. When price finally closes beyond
    a protected level, it's a "flip" indicating structure change.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        swings: List of swing points
        min_bars_protected: Minimum bars for level to be "protected"

    Returns:
        Tuple of (protected_highs, protected_lows, most_recent_flip)
    """
    protected_highs = []
    protected_lows = []
    recent_flip = None
    n = len(close)

    for swing in swings:
        bars_since = n - swing.index - 1

        if swing.type == "high":
            # Check if high has been violated
            violated = False
            flip_idx = None
            for i in range(swing.index + 1, n):
                if close[i] > swing.price:
                    violated = True
                    flip_idx = i
                    break

            if not violated and bars_since >= min_bars_protected:
                protected_highs.append(ProtectedLevel(
                    type="high",
                    price=swing.price,
                    index=swing.index,
                    bars_protected=bars_since
                ))
            elif violated:
                # This was a protected level that just flipped
                if bars_since < 5:  # Recent flip
                    flip = ProtectedLevel(
                        type="high",
                        price=swing.price,
                        index=swing.index,
                        bars_protected=flip_idx - swing.index if flip_idx else 0,
                        flipped=True,
                        flip_index=flip_idx
                    )
                    if recent_flip is None or (flip_idx and flip_idx > (recent_flip.flip_index or 0)):
                        recent_flip = flip

        else:  # low
            # Check if low has been violated
            violated = False
            flip_idx = None
            for i in range(swing.index + 1, n):
                if close[i] < swing.price:
                    violated = True
                    flip_idx = i
                    break

            if not violated and bars_since >= min_bars_protected:
                protected_lows.append(ProtectedLevel(
                    type="low",
                    price=swing.price,
                    index=swing.index,
                    bars_protected=bars_since
                ))
            elif violated:
                if bars_since < 5:  # Recent flip
                    flip = ProtectedLevel(
                        type="low",
                        price=swing.price,
                        index=swing.index,
                        bars_protected=flip_idx - swing.index if flip_idx else 0,
                        flipped=True,
                        flip_index=flip_idx
                    )
                    if recent_flip is None or (flip_idx and flip_idx > (recent_flip.flip_index or 0)):
                        recent_flip = flip

    return protected_highs, protected_lows, recent_flip


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> float:
    """Calculate Average True Range."""
    if len(close) < period + 1:
        return float(np.mean(high - low))

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    return float(np.mean(tr[-period:]))


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = RISK_PER_TRADE_PCT,
    contract_size: float = 1.0
) -> float:
    """
    Calculate position size based on fixed percentage risk.

    Per video strategy: Risk 1% of account per trade.

    Args:
        account_balance: Total account equity
        entry_price: Planned entry price
        stop_loss: Stop loss price
        risk_percent: Percentage of account to risk (default 1%)
        contract_size: Contract/lot size multiplier

    Returns:
        Position size (lots/contracts)
    """
    risk_amount = account_balance * (risk_percent / 100.0)
    sl_distance = abs(entry_price - stop_loss)

    if sl_distance <= 0:
        return 0.0

    # Position size = Risk Amount / (SL Distance * Contract Size)
    position_size = risk_amount / (sl_distance * contract_size)

    return position_size


# =============================================================================
# Multi-Timeframe Analysis
# =============================================================================

def analyze_timeframe(
    df: pd.DataFrame,
    swing_lookback: int = 5
) -> Dict[str, Any]:
    """
    Analyze a single timeframe for structure and bias.

    Args:
        df: DataFrame with OHLCV data
        swing_lookback: Lookback for swing detection

    Returns:
        Dict with bias, structure breaks, and swing sequence
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    swings = detect_swing_points(high, low, swing_lookback)
    breaks, sequence = detect_structure_breaks(close, high, low, swings)

    # Determine bias from recent structure
    recent_breaks = [b for b in breaks if b.index >= len(close) - 20]

    bullish_count = sum(1 for b in recent_breaks if b.direction == "bullish")
    bearish_count = sum(1 for b in recent_breaks if b.direction == "bearish")

    if bullish_count > bearish_count:
        bias = "bullish"
    elif bearish_count > bullish_count:
        bias = "bearish"
    else:
        # Check swing sequence
        if "HH" in sequence and "HL" in sequence:
            bias = "bullish"
        elif "LH" in sequence and "LL" in sequence:
            bias = "bearish"
        else:
            bias = "neutral"

    return {
        "bias": bias,
        "structure_breaks": breaks,
        "swing_sequence": sequence,
        "swings": swings
    }


def run_mtf_analysis(
    higher_tf_df: pd.DataFrame,
    lower_tf_df: pd.DataFrame,
    current_price: float,
    swing_lookback: int = 5,
    channel_lookback: int = 50
) -> MTFAnalysis:
    """
    Run complete multi-timeframe analysis.

    Args:
        higher_tf_df: Higher timeframe data (e.g., D1, H4)
        lower_tf_df: Lower timeframe data (e.g., H1, M15)
        current_price: Current market price
        swing_lookback: Lookback for swing detection
        channel_lookback: Lookback for regression channel

    Returns:
        MTFAnalysis with complete multi-timeframe context
    """
    # Analyze higher timeframe for bias
    htf = analyze_timeframe(higher_tf_df, swing_lookback)

    # Analyze lower timeframe for entries
    ltf = analyze_timeframe(lower_tf_df, swing_lookback)

    # Get arrays for lower TF calculations
    high = lower_tf_df['high'].values
    low = lower_tf_df['low'].values
    close = lower_tf_df['close'].values

    # Calculate OTE zone based on higher TF bias
    ote_zone = calculate_ote_zone(ltf["swings"], current_price, htf["bias"])

    # Check if price is in OTE
    price_in_ote = False
    if ote_zone:
        price_in_ote = ote_zone.ote_bottom <= current_price <= ote_zone.ote_top

    # Calculate regression channel (fit on swings per video strategy)
    channel = calculate_regression_channel(close, ltf["swings"], channel_lookback)

    # Determine price position in channel
    if channel:
        if current_price >= channel.upper_band:
            price_in_channel = "upper"
        elif current_price <= channel.lower_band:
            price_in_channel = "lower"
        elif current_price >= channel.middle:
            price_in_channel = "upper_half"
        else:
            price_in_channel = "lower_half"
    else:
        price_in_channel = "unknown"

    # Weekend gap analysis
    weekend_gap = detect_weekend_gap(lower_tf_df)

    # Protected levels
    protected_highs, protected_lows, recent_flip = detect_protected_levels(
        high, low, close, ltf["swings"]
    )

    # Calculate alignment score (0-100)
    alignment_score = 0

    # Higher TF bias clear (+25)
    if htf["bias"] in ["bullish", "bearish"]:
        alignment_score += 25

    # Lower TF aligns with higher TF (+20)
    if ltf["bias"] == htf["bias"]:
        alignment_score += 20

    # Price in OTE zone (+20)
    if price_in_ote:
        alignment_score += 20

    # Channel supports direction (+15)
    if channel:
        if htf["bias"] == "bullish" and price_in_channel in ["lower", "lower_half"]:
            alignment_score += 15
        elif htf["bias"] == "bearish" and price_in_channel in ["upper", "upper_half"]:
            alignment_score += 15

    # Entry confirmation check (per video strategy: REQUIRED)
    # Must have: weekend gap in bias direction OR protected level flip
    has_gap_confirmation = False
    has_flip_confirmation = False
    confirmation_type = None

    if weekend_gap and weekend_gap.exists:
        if htf["bias"] == "bullish" and weekend_gap.direction == "up":
            has_gap_confirmation = True
            alignment_score += 10
        elif htf["bias"] == "bearish" and weekend_gap.direction == "down":
            has_gap_confirmation = True
            alignment_score += 10

    if recent_flip:
        if htf["bias"] == "bullish" and recent_flip.type == "high":
            has_flip_confirmation = True
            alignment_score += 10  # Bullish flip (broke above protected high)
        elif htf["bias"] == "bearish" and recent_flip.type == "low":
            has_flip_confirmation = True
            alignment_score += 10  # Bearish flip (broke below protected low)

    has_entry_confirmation = has_gap_confirmation or has_flip_confirmation
    if has_gap_confirmation and has_flip_confirmation:
        confirmation_type = "both"
    elif has_gap_confirmation:
        confirmation_type = "weekend_gap"
    elif has_flip_confirmation:
        confirmation_type = "protected_flip"

    # Check if price is in or touches channel (per video strategy: REQUIRED)
    price_in_or_touches_channel = price_in_channel in ["upper", "lower", "upper_half", "lower_half"]

    # Determine final trade bias
    if alignment_score >= 60:
        trade_bias = htf["bias"]
    elif alignment_score >= 40:
        trade_bias = f"weak_{htf['bias']}"
    else:
        trade_bias = "neutral"

    return MTFAnalysis(
        higher_tf_bias=htf["bias"],
        higher_tf_structure=htf["structure_breaks"],
        higher_tf_swing_sequence=htf["swing_sequence"],
        lower_tf_bias=ltf["bias"],
        lower_tf_structure=ltf["structure_breaks"],
        ote_zone=ote_zone,
        price_in_ote=price_in_ote,
        channel=channel,
        price_in_channel=price_in_channel,
        weekend_gap=weekend_gap,
        protected_highs=protected_highs,
        protected_lows=protected_lows,
        recent_flip=recent_flip,
        alignment_score=alignment_score,
        trade_bias=trade_bias,
        has_entry_confirmation=has_entry_confirmation,
        confirmation_type=confirmation_type,
        price_in_or_touches_channel=price_in_or_touches_channel
    )


# =============================================================================
# Prompt Building
# =============================================================================

def _format_mtf_analysis_for_prompt(
    analysis: MTFAnalysis,
    current_price: float,
    atr: float
) -> str:
    """Format MTF analysis for LLM prompt."""
    lines = ["## MULTI-TIMEFRAME ANALYSIS"]

    # Higher TF Bias
    lines.append(f"\n### Higher Timeframe Bias: **{analysis.higher_tf_bias.upper()}**")
    lines.append(f"- Swing Sequence: {analysis.higher_tf_swing_sequence}")
    if analysis.higher_tf_structure:
        recent_htf = analysis.higher_tf_structure[-3:]
        for brk in recent_htf:
            lines.append(f"  - {brk.type.upper()} {brk.direction} at {brk.price:.5f}")

    # Lower TF
    lines.append(f"\n### Lower Timeframe Bias: **{analysis.lower_tf_bias.upper()}**")
    if analysis.lower_tf_structure:
        recent_ltf = analysis.lower_tf_structure[-3:]
        for brk in recent_ltf:
            lines.append(f"  - {brk.type.upper()} {brk.direction} at {brk.price:.5f}")

    # Alignment
    lines.append(f"\n### Timeframe Alignment")
    if analysis.higher_tf_bias == analysis.lower_tf_bias:
        lines.append(f"- **ALIGNED**: Both timeframes are {analysis.higher_tf_bias}")
    else:
        lines.append(f"- **DIVERGENT**: HTF={analysis.higher_tf_bias}, LTF={analysis.lower_tf_bias}")
        lines.append("  - Consider waiting for alignment or trade with HTF bias only")

    # OTE Zone
    lines.append(f"\n### OTE Zone (Optimal Trade Entry)")
    if analysis.ote_zone:
        ote = analysis.ote_zone
        lines.append(f"- Direction: {ote.direction}")
        lines.append(f"- **Retracement Levels (Entry Zone)**:")
        lines.append(f"  - Fib 50%: {ote.fib_50:.5f}")
        lines.append(f"  - Fib 61.8% (OTE): {ote.fib_618:.5f}")
        lines.append(f"  - Fib 78.6%: {ote.fib_786:.5f}")
        lines.append(f"- OTE Zone: {ote.ote_bottom:.5f} - {ote.ote_top:.5f}")
        lines.append(f"- **Extension Levels (Take Profit)**:")
        lines.append(f"  - Fib 127.2%: {ote.fib_ext_1272:.5f}")
        lines.append(f"  - Fib 161.8%: {ote.fib_ext_1618:.5f}")

        if analysis.price_in_ote:
            lines.append(f"- **PRICE IN OTE** at {current_price:.5f} - High probability entry zone!")
        else:
            dist_to_ote = min(
                abs(current_price - ote.ote_top),
                abs(current_price - ote.ote_bottom)
            )
            lines.append(f"- Price is {dist_to_ote:.5f} ({dist_to_ote/atr:.1f}x ATR) from OTE zone")
    else:
        lines.append("- No valid OTE zone identified")

    # Regression Channel (fitted on swings per video strategy)
    lines.append(f"\n### Regression Channel (Parallel Distribution)")
    if analysis.channel:
        ch = analysis.channel
        swing_info = f"fitted on {ch.num_swings} swings" if ch.num_swings > 0 else "fitted on closes (fallback)"
        lines.append(f"- Method: {swing_info}")
        lines.append(f"- Direction: {ch.direction.upper()}")
        lines.append(f"- R-squared: {ch.r_squared:.2f} (fit quality, >0.7 = good)")
        lines.append(f"- Upper Band (+2σ): {ch.upper_band:.5f}")
        lines.append(f"- Middle (Regression): {ch.middle:.5f}")
        lines.append(f"- Lower Band (-2σ): {ch.lower_band:.5f}")
        lines.append(f"- Price Position: **{analysis.price_in_channel.upper()}**")

        # Mean reversion guidance (per video strategy)
        if analysis.price_in_channel == "upper":
            lines.append("  - Price at upper channel - expect mean reversion to opposite (lower) band")
        elif analysis.price_in_channel == "lower":
            lines.append("  - Price at lower channel - expect mean reversion to opposite (upper) band")
    else:
        lines.append("- Insufficient data for channel calculation")

    # Weekend Gap
    lines.append(f"\n### Weekend Gap Analysis")
    if analysis.weekend_gap and analysis.weekend_gap.exists:
        gap = analysis.weekend_gap
        lines.append(f"- Gap Direction: {gap.direction.upper()}")
        lines.append(f"- Gap Size: {gap.gap_size:.5f} ({gap.gap_percent:.2f}%)")
        lines.append(f"- Friday Close: {gap.friday_close:.5f}")
        lines.append(f"- Monday Open: {gap.monday_open:.5f}")

        # Gap interpretation
        if gap.direction == "up":
            lines.append("  - Bullish gap = institutional buying over weekend")
        elif gap.direction == "down":
            lines.append("  - Bearish gap = institutional selling over weekend")
    else:
        lines.append("- No significant weekend gap detected")

    # Protected Levels
    lines.append(f"\n### Protected Levels")
    if analysis.protected_highs:
        lines.append("**Protected Highs** (unbroken resistance - liquidity above):")
        for ph in analysis.protected_highs[:3]:
            lines.append(f"  - {ph.price:.5f} (protected for {ph.bars_protected} bars)")

    if analysis.protected_lows:
        lines.append("**Protected Lows** (unbroken support - liquidity below):")
        for pl in analysis.protected_lows[:3]:
            lines.append(f"  - {pl.price:.5f} (protected for {pl.bars_protected} bars)")

    if analysis.recent_flip:
        flip = analysis.recent_flip
        flip_dir = "BULLISH" if flip.type == "high" else "BEARISH"
        lines.append(f"\n**RECENT FLIP**: {flip.type.upper()} at {flip.price:.5f}")
        lines.append(f"  - Internal structure shift {flip_dir}")

    # Entry Confirmation Status (REQUIRED per video strategy)
    lines.append(f"\n### Entry Confirmation Status")
    if analysis.has_entry_confirmation:
        lines.append(f"- **CONFIRMED** via: {analysis.confirmation_type}")
        if analysis.confirmation_type == "both":
            lines.append("  - Both weekend gap AND protected flip confirm - STRONGEST signal")
    else:
        lines.append("- **NOT CONFIRMED** - Missing required confirmation!")
        lines.append("  - Need: weekend gap in bias direction OR protected level flip")
        lines.append("  - Per video strategy: DO NOT ENTER without confirmation")

    # Channel touch requirement
    lines.append(f"\n### Channel Touch Requirement")
    if analysis.price_in_or_touches_channel:
        lines.append(f"- **MET** - Price is in channel ({analysis.price_in_channel})")
    else:
        lines.append("- **NOT MET** - Price outside channel")
        lines.append("  - Per video strategy: pullback must touch channel")

    # Confluence Score
    lines.append(f"\n### Confluence Score: **{analysis.alignment_score}/100**")
    lines.append(f"### Trade Bias: **{analysis.trade_bias.upper()}**")

    if analysis.alignment_score >= 70:
        lines.append("- HIGH CONFLUENCE: Multiple factors aligned, high probability setup")
    elif analysis.alignment_score >= 50:
        lines.append("- MODERATE CONFLUENCE: Some alignment, proceed with normal sizing")
    else:
        lines.append("- LOW CONFLUENCE: Factors not aligned, consider waiting or reduced size")

    # Final entry checklist
    lines.append(f"\n### Entry Checklist")
    htf_ok = analysis.higher_tf_bias in ["bullish", "bearish"]
    ltf_aligned = analysis.lower_tf_bias == analysis.higher_tf_bias
    lines.append(f"- [{'x' if htf_ok else ' '}] Higher TF bias clear")
    lines.append(f"- [{'x' if ltf_aligned else ' '}] Lower TF aligns with HTF")
    lines.append(f"- [{'x' if analysis.price_in_ote else ' '}] Price in OTE zone")
    lines.append(f"- [{'x' if analysis.price_in_or_touches_channel else ' '}] Price in/touches channel")
    lines.append(f"- [{'x' if analysis.has_entry_confirmation else ' '}] Entry confirmation (gap/flip)")

    all_checks = htf_ok and ltf_aligned and analysis.price_in_ote and analysis.price_in_or_touches_channel and analysis.has_entry_confirmation
    if all_checks:
        lines.append("\n**ALL CHECKS PASSED - VALID ENTRY SETUP**")
    else:
        lines.append("\n**MISSING REQUIREMENTS - WAIT FOR BETTER SETUP**")

    return "\n".join(lines)


def _build_mtf_quant_prompt(
    data_context: str,
    trade_memories: str = ""
) -> str:
    """Build the complete SMC MTF quant analyst prompt."""

    memories_section = ""
    if trade_memories:
        memories_section = f"""

{trade_memories}

IMPORTANT: Apply lessons from past trades. Do NOT repeat the same mistakes.

"""

    return f"""You are a systematic Smart Money Concepts (SMC) trader specializing in Multi-Timeframe Analysis, OTE entries, and Regression Channel trading.

## CORE METHODOLOGY

### Multi-Timeframe Analysis (MTF)
1. **Higher Timeframe (D1/H4)**: Determines overall BIAS
   - HH-HL sequence = BULLISH bias (trade longs only)
   - LH-LL sequence = BEARISH bias (trade shorts only)
   - BOS confirms trend continuation
   - CHoCH warns of potential reversal

2. **Lower Timeframe (H1/M15)**: Determines ENTRY timing
   - Wait for pullback into OTE zone
   - Enter when lower TF aligns with higher TF bias
   - Use structure breaks for entry confirmation

### Optimal Trade Entry (OTE)
The OTE zone is the 50%-78.6% Fibonacci retracement of the most recent swing:
- **0.5 (50%)**: Start of OTE zone - early entry
- **0.618 (61.8%)**: Golden ratio - optimal entry point
- **0.786 (78.6%)**: Deep retracement - maximum discount

For BULLISH trades: Enter when price retraces DOWN into OTE (buy at discount)
For BEARISH trades: Enter when price retraces UP into OTE (sell at premium)

### Regression Channel Trading
Linear regression channels identify the mean and standard deviation bands:
- **Upper Band (+2σ)**: Overbought - expect mean reversion DOWN
- **Middle (Regression Line)**: Fair value - expect price to gravitate here
- **Lower Band (-2σ)**: Oversold - expect mean reversion UP

Channel direction confirms or conflicts with bias:
- Upward sloping channel + bullish bias = HIGH CONFLUENCE
- Downward sloping channel + bearish bias = HIGH CONFLUENCE
- Conflicting direction = REDUCED CONFLUENCE

### Weekend Gap Analysis
Gaps between Friday close and Monday open indicate institutional positioning:
- **Gap UP**: Bullish institutional flow - confirms bullish bias
- **Gap DOWN**: Bearish institutional flow - confirms bearish bias
- Gaps often get "filled" but direction indicates sentiment

### Protected High/Low Flips
Levels that have been protected (unbroken) for many bars are significant:
- When a protected HIGH is broken = BULLISH internal structure shift
- When a protected LOW is broken = BEARISH internal structure shift
- Recent flips are strong confirmation signals

## TRADE SETUP REQUIREMENTS

**For BUY entry:**
1. Higher TF bias is BULLISH (HH-HL structure)
2. Price has pulled back to OTE zone (50-78.6% retracement)
3. Lower TF shows BOS bullish or CHoCH bullish
4. Preferably at lower regression channel band
5. Weekend gap UP confirms (if applicable)
6. Protected low held OR protected high just broke

**For SELL entry:**
1. Higher TF bias is BEARISH (LH-LL structure)
2. Price has rallied to OTE zone (50-78.6% retracement)
3. Lower TF shows BOS bearish or CHoCH bearish
4. Preferably at upper regression channel band
5. Weekend gap DOWN confirms (if applicable)
6. Protected high held OR protected low just broke

## RISK MANAGEMENT

- Stop Loss: Beyond the recent swing high/low that forms the OTE zone
- Take Profit: Next Fibonacci extension (1.272, 1.618) or opposing channel band
- Position Size: Based on ATR-adjusted stop distance
- **MINIMUM SL DISTANCE**: At least 1x ATR from entry
- **MINIMUM R:R**: At least 1.5:1

{data_context}
{memories_section}
## YOUR TASK

Analyze the multi-timeframe data and make a systematic trading decision.

Think step-by-step:
1. What is the HIGHER TF bias? (bullish/bearish/neutral)
2. Does LOWER TF align with higher TF?
3. Is price currently in the OTE zone?
4. What does the regression channel show?
5. Is there a weekend gap that confirms or conflicts?
6. Are there protected levels or recent flips?
7. What is the overall confluence score?
8. Where should entry, SL, and TP be placed?

## SIGNAL OPTIONS
- **buy_to_enter**: Higher TF bullish + price in OTE + confluence >= 50
- **sell_to_enter**: Higher TF bearish + price in OTE + confluence >= 50
- **hold**: No clear setup, low confluence, or conflicting signals
- **close**: Existing position invalidated

## ORDER TYPE
- **market**: Price is IN the OTE zone now
- **limit**: Price approaching OTE, place pending order

Remember:
- Trade WITH the higher timeframe bias
- Enter at OTE for optimal risk:reward
- Use regression channel for mean reversion confirmation
- Weekend gaps and protected level flips add confluence
- Minimum 50/100 confluence score for entry"""


def _build_data_context(
    ticker: str,
    current_price: float,
    mtf_analysis: MTFAnalysis,
    atr: float,
    market_regime: str = "unknown",
    volatility_regime: str = "normal",
    trading_session: str = "unknown",
    current_date: str = "",
    additional_context: str = ""
) -> str:
    """Build complete data context for prompt."""
    sections = []

    # Current market data
    sections.append(f"""## CURRENT MARKET DATA
- **Symbol**: {ticker}
- **Current Price**: {current_price:.5f}
- **Date**: {current_date}
- **Trading Session**: {trading_session}
- **Market Regime**: {market_regime}
- **Volatility Regime**: {volatility_regime}
- **ATR (14)**: {atr:.5f}
""")

    # MTF Analysis
    sections.append(_format_mtf_analysis_for_prompt(mtf_analysis, current_price, atr))

    # Additional context (market report, etc.)
    if additional_context:
        sections.append(f"\n## ADDITIONAL CONTEXT\n{additional_context}")

    return "\n".join(sections)


# =============================================================================
# Quant Agent Factory
# =============================================================================

def create_smc_mtf_quant(llm, use_structured_output: bool = True):
    """
    Create an SMC Multi-Timeframe OTE & Channel quant analyst node.

    This quant focuses on:
    - Multi-Timeframe bias determination (Higher TF for direction)
    - OTE zone entries (Fibonacci 0.5-0.786 retracement)
    - Regression channel mean reversion
    - Weekend gap confirmation
    - Protected level flip signals

    Args:
        llm: The language model to use for analysis
        use_structured_output: If True, uses LLM structured output for guaranteed JSON

    Returns:
        A function that processes state and returns the quant analysis
    """

    # Create structured output LLM wrapper
    structured_llm = None
    if use_structured_output:
        try:
            structured_llm = llm.with_structured_output(QuantAnalystDecision)
        except Exception as e:
            print(f"Warning: Structured output not supported for SMC MTF quant: {e}")
            structured_llm = None

    def smc_mtf_quant_node(state) -> dict:
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        current_price = state.get("current_price")

        # Get dataframes for MTF analysis
        higher_tf_df = state.get("higher_tf_data")  # D1 or H4
        lower_tf_df = state.get("lower_tf_data")  # H1 or M15

        # Fallback to single timeframe if MTF not available
        if higher_tf_df is None or lower_tf_df is None:
            df = state.get("price_data") or state.get("ohlcv_data")
            if df is not None:
                higher_tf_df = df
                lower_tf_df = df

        if higher_tf_df is None or lower_tf_df is None or current_price is None:
            return {
                "smc_mtf_report": "Insufficient data for MTF analysis",
                "smc_mtf_decision": None,
            }

        # Run MTF analysis
        mtf_analysis = run_mtf_analysis(
            higher_tf_df=higher_tf_df,
            lower_tf_df=lower_tf_df,
            current_price=current_price,
            swing_lookback=5,
            channel_lookback=50
        )

        # Calculate ATR
        atr = calculate_atr(
            lower_tf_df['high'].values,
            lower_tf_df['low'].values,
            lower_tf_df['close'].values
        )

        # Extract context
        market_regime = state.get("market_regime") or "unknown"
        volatility_regime = state.get("volatility_regime") or "normal"
        trading_session = state.get("trading_session") or "unknown"
        market_report = state.get("market_report") or ""

        # Build prompt
        data_context = _build_data_context(
            ticker=ticker,
            current_price=current_price,
            mtf_analysis=mtf_analysis,
            atr=atr,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trading_session=trading_session,
            current_date=current_date,
            additional_context=market_report
        )

        trade_memories = state.get("trade_memories") or ""
        system_prompt = _build_mtf_quant_prompt(data_context, trade_memories)

        # Log
        logger = _get_mtf_logger()
        logger.info(f"\n{'='*80}\nSMC MTF ANALYSIS - {ticker}\n{'='*80}")
        logger.info(f"Symbol: {ticker} | Date: {current_date} | Price: {current_price}")
        logger.info(f"HTF Bias: {mtf_analysis.higher_tf_bias} | LTF Bias: {mtf_analysis.lower_tf_bias}")
        logger.info(f"Alignment Score: {mtf_analysis.alignment_score} | Trade Bias: {mtf_analysis.trade_bias}")
        logger.info(f"\n--- FULL PROMPT ---\n{system_prompt[:2000]}...")

        if structured_llm is not None:
            try:
                import time as _time
                _llm_start = _time.time()
                decision: QuantAnalystDecision = structured_llm.invoke(system_prompt)
                _llm_duration = _time.time() - _llm_start

                logger.info(f"--- LLM RESPONSE [took {_llm_duration:.1f}s] ---")
                logger.info(f"Signal: {decision.signal} | Confidence: {decision.confidence}")

                decision_dict = decision.model_dump()
                report = _format_mtf_report(decision, mtf_analysis)

                return {
                    "smc_mtf_report": report,
                    "smc_mtf_decision": decision_dict,
                }
            except Exception as e:
                logger.error(f"Structured output failed: {e}")
                print(f"Structured output failed for SMC MTF quant: {e}")

        # Fallback
        response = llm.invoke(system_prompt)
        report = response.content if hasattr(response, "content") else str(response)

        return {
            "smc_mtf_report": report,
            "smc_mtf_decision": None,
        }

    return smc_mtf_quant_node


def _format_mtf_report(decision: QuantAnalystDecision, analysis: MTFAnalysis) -> str:
    """Format the decision into a human-readable report."""
    signal_str = (
        decision.signal
        if isinstance(decision.signal, str)
        else decision.signal.value
    )

    lines = [
        f"## SMC MTF QUANT DECISION: **{signal_str.upper()}**",
        f"**Symbol**: {decision.symbol}",
        f"**Confidence**: {decision.confidence:.0%}",
        "",
        f"### Multi-Timeframe Context",
        f"- Higher TF Bias: {analysis.higher_tf_bias}",
        f"- Lower TF Bias: {analysis.lower_tf_bias}",
        f"- Alignment Score: {analysis.alignment_score}/100",
        f"- Price in OTE: {'Yes' if analysis.price_in_ote else 'No'}",
        f"- Channel Position: {analysis.price_in_channel}",
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
        lines.append("")

    lines.extend([
        "### Justification",
        decision.justification,
        "",
        "### Invalidation Condition",
        decision.invalidation_condition,
    ])

    return "\n".join(lines)


def get_mtf_quant_decision_for_modal(mtf_decision: dict) -> dict:
    """
    Convert MTF quant decision dict to trade modal format.

    Args:
        mtf_decision: The smc_mtf_decision dict from agent state

    Returns:
        Dict formatted for TradeExecutionWizard props
    """
    if not mtf_decision:
        return {}

    signal_map = {
        "buy_to_enter": "BUY",
        "sell_to_enter": "SELL",
        "hold": "HOLD",
        "close": "HOLD",
    }

    signal = mtf_decision.get("signal", "hold")
    if isinstance(signal, dict):
        signal = signal.get("value", "hold")

    order_type = mtf_decision.get("order_type", "market")
    if isinstance(order_type, dict):
        order_type = order_type.get("value", "market")

    return {
        "symbol": mtf_decision.get("symbol", ""),
        "signal": signal_map.get(signal, "HOLD"),
        "orderType": order_type,
        "suggestedEntry": mtf_decision.get("entry_price"),
        "suggestedStopLoss": mtf_decision.get("stop_loss"),
        "suggestedTakeProfit": mtf_decision.get("profit_target"),
        "rationale": f"{mtf_decision.get('justification', '')}. Invalidation: {mtf_decision.get('invalidation_condition', '')}",
        "confidence": mtf_decision.get("confidence", 0.5),
    }


def analyze_mtf_for_quant(
    higher_tf_df: pd.DataFrame,
    lower_tf_df: pd.DataFrame,
    current_price: float,
    swing_lookback: int = 5,
    channel_lookback: int = 50
) -> Dict[str, Any]:
    """
    Run MTF analysis and return formatted context for quant.

    Convenience function for external use.

    Args:
        higher_tf_df: Higher timeframe DataFrame
        lower_tf_df: Lower timeframe DataFrame
        current_price: Current market price
        swing_lookback: Lookback for swing detection
        channel_lookback: Lookback for regression channel

    Returns:
        Dict with 'mtf_analysis' (MTFAnalysis object) and 'mtf_context' (string)
    """
    analysis = run_mtf_analysis(
        higher_tf_df=higher_tf_df,
        lower_tf_df=lower_tf_df,
        current_price=current_price,
        swing_lookback=swing_lookback,
        channel_lookback=channel_lookback
    )

    atr = calculate_atr(
        lower_tf_df['high'].values,
        lower_tf_df['low'].values,
        lower_tf_df['close'].values
    )

    context = _format_mtf_analysis_for_prompt(analysis, current_price, atr)

    return {
        "mtf_analysis": analysis,
        "mtf_context": context,
    }
