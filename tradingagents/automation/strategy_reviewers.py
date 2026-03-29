"""
Strategy-specific position assumption reviewers.

Each reviewer checks whether the original trade thesis still holds,
using logic appropriate to the strategy that placed the trade.

All reviewers share the same signature:
    def review_xxx(decision, position, report, timeframe) -> PositionAssumptionReport

The report is pre-populated with basic fields (symbol, direction, ticket, entry,
current price, P/L). Each reviewer adds findings and sets recommended_action.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from tradingagents.automation.position_assumption_review import (
    AssumptionFinding,
    PositionAssumptionReport,
)

logger = logging.getLogger("StrategyReviewers")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_ohlcv(symbol: str, timeframe: str, bars: int = 250) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from MT5 as a DataFrame."""
    try:
        import MetaTrader5 as mt5

        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    except Exception as e:
        logger.debug(f"_fetch_ohlcv failed: {e}")
        return None


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR for the last bar."""
    if df is None or len(df) < period + 1:
        return 0.0
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


def _compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute ADX for the last bar. Returns None if insufficient data."""
    if df is None or len(df) < period * 3:
        return None
    try:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        # Zero out whichever is smaller
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))

        # Smoothed averages
        atr = pd.Series(tr).rolling(period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / (atr + 1e-10)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(period).mean().values
        val = adx[-1]
        return float(val) if not np.isnan(val) else None
    except Exception:
        return None


def _bb_width_percentile(close: np.ndarray, lookback: int = 100, bb_period: int = 20) -> float:
    """Bollinger Band width percentile over lookback bars."""
    if len(close) < lookback:
        return 0.5
    sma = pd.Series(close).rolling(bb_period).mean().values
    std = pd.Series(close).rolling(bb_period).std().values
    widths = (std / (sma + 1e-10))[-lookback:]
    widths = widths[~np.isnan(widths)]
    if len(widths) < 10:
        return 0.5
    current = widths[-1]
    return float(np.sum(widths < current) / len(widths))


def _add_pnl_checks(report: PositionAssumptionReport):
    """Shared P/L extreme and time-in-trade checks."""
    if report.pnl_pct < -3.0:
        report.findings.append(AssumptionFinding(
            category="pnl_extreme",
            severity="warning",
            message=f"Position at {report.pnl_pct:+.2f}% — monitor for exit.",
            suggested_action="monitor",
        ))
    elif report.pnl_pct > 8.0:
        report.findings.append(AssumptionFinding(
            category="pnl_extreme",
            severity="info",
            message=f"Position at {report.pnl_pct:+.2f}% — consider taking profit.",
            suggested_action="monitor",
        ))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PIPELINE_REVIEWERS = {}  # Populated after function definitions


def route_to_reviewer(decision: Dict[str, Any]):
    """Pick the correct reviewer function based on pipeline and setup_type."""
    pipeline = decision.get("pipeline", "") or ""
    setup_type = (decision.get("setup_type", "") or
                  decision.get("smc_context", {}).get("setup_type", "") or "")

    # XGBoost uses sub-strategies
    if pipeline in ("xgboost", "xgboost_ensemble"):
        st = setup_type.lower()
        if "mean_reversion" in st:
            return review_mean_reversion
        if "breakout" in st:
            return review_breakout
        if "volume_profile" in st:
            return review_volume_profile
        if "trend" in st:
            return review_rule_based_trend
        if "smc" in st:
            return review_smc_position
        return review_safe_fallback  # Unknown XGB strategy

    # Direct pipeline mapping
    pl = pipeline.lower()
    if pl in ("smc_quant", "smc_quant_basic", "smc_mtf"):
        return review_smc_position
    if pl in ("breakout_quant",):
        return review_breakout
    if pl in ("range_quant",):
        return review_mean_reversion
    if pl in ("volume_profile",):
        return review_volume_profile
    if pl in ("rule_based",):
        return review_rule_based_trend

    # Fallback: check setup_type
    st = setup_type.lower()
    if st in ("fvg_bounce", "ob_bounce", "liquidity_sweep", "choch", "bos"):
        return review_smc_position
    if "mean_reversion" in st:
        return review_mean_reversion
    if "breakout" in st:
        return review_breakout
    if "volume_profile" in st:
        return review_volume_profile
    if "trend" in st:
        return review_rule_based_trend

    # Unknown strategy — safe fallback, NOT SMC
    return review_safe_fallback


# ---------------------------------------------------------------------------
# 1. Safe Fallback — monitors P/L only, never adjusts SL/TP
# ---------------------------------------------------------------------------

def review_safe_fallback(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Safe reviewer for unknown/unhandled strategies.
    Only checks P/L extremes. Never suggests SL/TP changes.
    """
    report.review_strategy = "safe_fallback"
    _add_pnl_checks(report)
    report.recommended_action = "hold"
    return report


# ---------------------------------------------------------------------------
# 2. SMC Position Review — existing logic extracted
# ---------------------------------------------------------------------------

def review_smc_position(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    SMC-based review: bias alignment, structure breaks, zone mitigation,
    TP blocking zones, trailing stop opportunities.
    This is the EXISTING logic from position_assumption_review.py extracted here.
    """
    from tradingagents.dataflows.smc_utils import get_smc_position_review_context

    report.review_strategy = "smc"

    try:
        smc = get_smc_position_review_context(
            symbol=report.symbol,
            direction=report.direction,
            entry_price=report.entry_price,
            current_price=report.current_price,
            sl=report.current_sl,
            tp=report.current_tp,
            timeframe=timeframe,
        )
    except Exception as e:
        report.error = f"SMC analysis failed: {e}"
        return report

    if smc.get("error"):
        report.error = f"SMC analysis error: {smc['error']}"
        return report

    direction = report.direction
    current_sl = report.current_sl
    current_tp = report.current_tp
    current_price = report.current_price
    pnl_pct = report.pnl_pct

    # CHECK 1: Bias alignment
    bias = smc.get("bias", "neutral")
    bias_aligns = smc.get("bias_aligns", True)
    if not bias_aligns:
        report.findings.append(AssumptionFinding(
            category="bias_shift",
            severity="critical",
            message=f"Market bias is now {bias.upper()} — against your {direction} position. "
                    f"Original trade assumed {'bullish' if direction == 'BUY' else 'bearish'} conditions.",
            suggested_action="close" if pnl_pct < 0 else "monitor",
        ))

    # CHECK 2: Structure break (CHOCH) against position
    if smc.get("structure_shift"):
        report.findings.append(AssumptionFinding(
            category="structure_break",
            severity="critical",
            message=f"Change of Character (CHOCH) detected AGAINST your {direction} position. "
                    f"Market structure has shifted.",
            suggested_action="close",
        ))

    # CHECK 3: SL placement risk
    if smc.get("sl_at_risk") and current_sl > 0:
        report.findings.append(AssumptionFinding(
            category="sl_risk",
            severity="warning",
            message=smc.get("sl_risk_reason", "SL is in a vulnerable location"),
            suggested_action="adjust_sl",
            suggested_value=smc.get("suggested_sl"),
        ))

    if current_sl > 0:
        support_levels = smc.get("support_levels", [])
        resistance_levels = smc.get("resistance_levels", [])

        if direction == "BUY" and support_levels:
            nearest_support_bottom = support_levels[0].get("bottom", 0)
            if nearest_support_bottom > 0 and current_sl > nearest_support_bottom:
                report.findings.append(AssumptionFinding(
                    category="sl_risk",
                    severity="warning",
                    message=f"SL at {current_sl:.5f} is above nearest support zone at {nearest_support_bottom:.5f}.",
                    suggested_action="adjust_sl",
                    suggested_value=nearest_support_bottom * 0.998,
                ))
        elif direction == "SELL" and resistance_levels:
            nearest_resistance_top = resistance_levels[0].get("top", 0)
            if nearest_resistance_top > 0 and current_sl < nearest_resistance_top:
                report.findings.append(AssumptionFinding(
                    category="sl_risk",
                    severity="warning",
                    message=f"SL at {current_sl:.5f} is below nearest resistance zone at {nearest_resistance_top:.5f}.",
                    suggested_action="adjust_sl",
                    suggested_value=nearest_resistance_top * 1.002,
                ))

    # CHECK 4: TP still realistic?
    if current_tp > 0:
        support_levels = smc.get("support_levels", [])
        resistance_levels = smc.get("resistance_levels", [])

        if direction == "BUY" and resistance_levels:
            for zone in resistance_levels:
                zone_price = zone.get("price", 0)
                zone_strength = zone.get("strength", 0)
                if 0 < zone_price < current_tp and zone_price > current_price:
                    if zone_strength >= 0.6:
                        report.findings.append(AssumptionFinding(
                            category="tp_blocked",
                            severity="warning",
                            message=f"Resistance zone at {zone_price:.5f} (strength {zone_strength:.0%}) "
                                    f"sits between current price and TP at {current_tp:.5f}.",
                            suggested_action="adjust_tp",
                            suggested_value=zone.get("bottom", zone_price),
                        ))
                        break
        elif direction == "SELL" and support_levels:
            for zone in support_levels:
                zone_price = zone.get("price", 0)
                zone_strength = zone.get("strength", 0)
                if 0 < zone_price > current_tp and zone_price < current_price:
                    if zone_strength >= 0.6:
                        report.findings.append(AssumptionFinding(
                            category="tp_blocked",
                            severity="warning",
                            message=f"Support zone at {zone_price:.5f} (strength {zone_strength:.0%}) "
                                    f"sits between current price and TP at {current_tp:.5f}.",
                            suggested_action="adjust_tp",
                            suggested_value=zone.get("top", zone_price),
                        ))
                        break

    # CHECK 5: Original entry zone status
    if direction == "BUY":
        if not smc.get("support_levels", []):
            report.findings.append(AssumptionFinding(
                category="zone_mitigated",
                severity="warning",
                message="No unmitigated support zones remain below current price.",
                suggested_action="monitor",
            ))
    else:
        if not smc.get("resistance_levels", []):
            report.findings.append(AssumptionFinding(
                category="zone_mitigated",
                severity="warning",
                message="No unmitigated resistance zones remain above current price.",
                suggested_action="monitor",
            ))

    # CHECK 6: New zones emerged (trailing opportunity)
    if smc.get("trailing_sl") and current_sl > 0:
        trailing_sl = smc["trailing_sl"]
        trailing_source = smc.get("trailing_sl_source", "SMC zone")
        is_better = (
            (direction == "BUY" and trailing_sl > current_sl) or
            (direction == "SELL" and trailing_sl < current_sl)
        )
        if is_better:
            report.findings.append(AssumptionFinding(
                category="zone_emerged",
                severity="info",
                message=f"New {trailing_source} has formed. "
                        f"Trailing stop can be tightened from {current_sl:.5f} to {trailing_sl:.5f}.",
                suggested_action="adjust_sl",
                suggested_value=trailing_sl,
            ))

    # Determine recommendation (same logic as original)
    _determine_smc_recommendation(report, smc)
    return report


def _determine_smc_recommendation(report: PositionAssumptionReport, smc: Dict[str, Any]):
    """Set recommendation based on SMC findings."""
    has_bias_shift = any(f.category == "bias_shift" for f in report.findings)
    has_structure_break = any(f.category == "structure_break" for f in report.findings)
    has_sl_risk = any(f.category == "sl_risk" for f in report.findings)
    has_tp_blocked = any(f.category == "tp_blocked" for f in report.findings)
    has_trailing = any(f.category == "zone_emerged" and f.suggested_action == "adjust_sl"
                       for f in report.findings)

    if has_bias_shift and has_structure_break:
        report.recommended_action = "close"
        return
    if has_structure_break and report.pnl_pct < 0:
        report.recommended_action = "close"
        return
    if has_sl_risk:
        sl_findings = [f for f in report.findings if f.category == "sl_risk" and f.suggested_value]
        if sl_findings:
            report.recommended_action = "adjust_sl"
            report.suggested_sl = sl_findings[0].suggested_value
    if has_tp_blocked:
        tp_findings = [f for f in report.findings if f.category == "tp_blocked" and f.suggested_value]
        if tp_findings:
            if report.recommended_action == "hold":
                report.recommended_action = "adjust_tp"
            report.suggested_tp = tp_findings[0].suggested_value
    if has_trailing and report.recommended_action == "hold":
        trail_findings = [f for f in report.findings if f.category == "zone_emerged" and f.suggested_value]
        if trail_findings:
            report.recommended_action = "adjust_sl"
            report.suggested_sl = trail_findings[0].suggested_value
    if not report.findings:
        report.recommended_action = "hold"


# ---------------------------------------------------------------------------
# 3. Mean Reversion Review
# ---------------------------------------------------------------------------

def review_mean_reversion(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Mean reversion review:
    1. Z-score check — is price still at an extreme or reverting to mean?
    2. Volatility regime change — calm->volatile invalidates MR
    3. XGBoost re-prediction — has P(direction) changed?
    4. Range integrity — has price broken out of expected range?
    5. P/L safety checks
    """
    report.review_strategy = "mean_reversion"
    symbol = report.symbol
    direction = report.direction

    df = _fetch_ohlcv(symbol, timeframe, bars=250)

    if df is not None and len(df) >= 100:
        close = df["close"].values.astype(float)
        current = close[-1]

        # --- Z-score check ---
        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        std20 = pd.Series(close).rolling(20).std().iloc[-1]
        zscore = (current - sma20) / (std20 + 1e-10)

        # Mean reversion BUY entered at negative extreme.
        # If z-score returned to ~0, trade thesis is COMPLETING (good).
        # If z-score went even MORE negative, range may be breaking.
        if direction == "BUY" and zscore < -3.0:
            report.findings.append(AssumptionFinding(
                category="range_break",
                severity="warning",
                message=f"Z-score at {zscore:.1f} — price extending further from mean. "
                        f"Range may be breaking down.",
                suggested_action="monitor",
            ))
        elif direction == "SELL" and zscore > 3.0:
            report.findings.append(AssumptionFinding(
                category="range_break",
                severity="warning",
                message=f"Z-score at {zscore:.1f} — price extending further from mean. "
                        f"Range may be breaking up.",
                suggested_action="monitor",
            ))

        # --- Volatility regime check ---
        original_regime = (decision.get("volatility_regime") or "normal").lower()
        bb_pctile = _bb_width_percentile(close)
        current_regime = "high" if bb_pctile > 0.8 else ("low" if bb_pctile < 0.2 else "normal")

        if original_regime in ("low", "normal") and current_regime == "high":
            report.findings.append(AssumptionFinding(
                category="regime_change",
                severity="critical",
                message=f"Volatility regime changed from {original_regime} to {current_regime}. "
                        f"Mean reversion unreliable in high-volatility regimes.",
                suggested_action="close" if report.pnl_pct < 0 else "monitor",
            ))

        # --- Bollinger Band check: has price returned to mean? ---
        if direction == "BUY" and zscore > 0.5:
            report.findings.append(AssumptionFinding(
                category="thesis_completing",
                severity="info",
                message=f"Z-score at {zscore:.1f} — price has reverted past the mean. "
                        f"Mean reversion thesis may be completing.",
                suggested_action="monitor",
            ))
        elif direction == "SELL" and zscore < -0.5:
            report.findings.append(AssumptionFinding(
                category="thesis_completing",
                severity="info",
                message=f"Z-score at {zscore:.1f} — price has reverted past the mean. "
                        f"Mean reversion thesis may be completing.",
                suggested_action="monitor",
            ))

        # --- Donchian channel break (range broken) ---
        high_20 = pd.Series(df["high"].values.astype(float)).rolling(20).max().iloc[-1]
        low_20 = pd.Series(df["low"].values.astype(float)).rolling(20).min().iloc[-1]

        if direction == "BUY" and current < low_20:
            report.findings.append(AssumptionFinding(
                category="range_break",
                severity="critical",
                message=f"Price ({current:.2f}) broke below 20-bar low ({low_20:.2f}). "
                        f"Range may be breaking — mean reversion thesis invalidated.",
                suggested_action="close" if report.pnl_pct < -1.0 else "monitor",
            ))
        elif direction == "SELL" and current > high_20:
            report.findings.append(AssumptionFinding(
                category="range_break",
                severity="critical",
                message=f"Price ({current:.2f}) broke above 20-bar high ({high_20:.2f}). "
                        f"Range may be breaking — mean reversion thesis invalidated.",
                suggested_action="close" if report.pnl_pct < -1.0 else "monitor",
            ))

    # --- XGBoost re-prediction ---
    try:
        from tradingagents.xgb_quant.predictor import LivePredictor
        predictor = LivePredictor()
        atr = _compute_atr(df) if df is not None else 0

        if df is not None and atr > 0:
            signal = predictor.predict_single(
                "mean_reversion", symbol, timeframe, df,
                float(df["close"].iloc[-1]), atr,
            )
            original_confidence = decision.get("confidence", 0.5)

            # Model now disagrees with trade direction
            if (signal.direction != direction and
                    signal.direction != "HOLD" and
                    signal.confidence > 0.6):
                report.findings.append(AssumptionFinding(
                    category="model_reversal",
                    severity="critical",
                    message=f"XGBoost mean_reversion now signals {signal.direction} "
                            f"(conf={signal.confidence:.0%}) vs your {direction} position.",
                    suggested_action="close" if report.pnl_pct < 0 else "monitor",
                ))
            elif signal.confidence < 0.4 and original_confidence > 0.6:
                report.findings.append(AssumptionFinding(
                    category="model_weakened",
                    severity="warning",
                    message=f"XGBoost confidence dropped from {original_confidence:.0%} to "
                            f"{signal.confidence:.0%}.",
                    suggested_action="monitor",
                ))
    except Exception as e:
        logger.debug(f"XGBoost re-prediction failed: {e}")

    _add_pnl_checks(report)
    _determine_mr_recommendation(report)
    return report


def _determine_mr_recommendation(report: PositionAssumptionReport):
    """Set recommendation for mean reversion trades."""
    has_range_break = any(f.category == "range_break" and f.severity == "critical"
                         for f in report.findings)
    has_regime_change = any(f.category == "regime_change" for f in report.findings)
    has_model_reversal = any(f.category == "model_reversal" for f in report.findings)

    # Critical: range broke + losing = close
    if has_range_break and report.pnl_pct < -1.0:
        report.recommended_action = "close"
        return
    # Critical: model reversed + losing = close
    if has_model_reversal and report.pnl_pct < 0:
        report.recommended_action = "close"
        return
    # Critical: regime shift + losing = close
    if has_regime_change and report.pnl_pct < 0:
        report.recommended_action = "close"
        return

    # Mean reversion: NEVER adjust SL/TP based on review — only hold or close
    # The original SL/TP were calculated from the range; adjusting them with
    # structural analysis makes no sense for MR trades.
    report.recommended_action = "hold"


# ---------------------------------------------------------------------------
# 4. Breakout Review
# ---------------------------------------------------------------------------

def review_breakout(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Breakout review:
    1. Is momentum still strong? (ADX check)
    2. Has ATR collapsed? (momentum dying)
    3. Has price returned inside the breakout range? (breakout failed)
    4. P/L safety checks
    """
    report.review_strategy = "breakout"
    symbol = report.symbol
    direction = report.direction
    entry_price = report.entry_price

    df = _fetch_ohlcv(symbol, timeframe, bars=250)

    if df is not None and len(df) >= 50:
        close = df["close"].values.astype(float)
        current = close[-1]

        # --- ADX check ---
        adx = _compute_adx(df, period=14)
        if adx is not None and adx < 20:
            report.findings.append(AssumptionFinding(
                category="momentum_loss",
                severity="warning",
                message=f"ADX at {adx:.1f} — momentum fading. Breakout may be failing.",
                suggested_action="monitor",
            ))

        # --- ATR compression ---
        atr = _compute_atr(df)
        atr_series = []
        for i in range(max(0, len(df) - 20), len(df)):
            subset = df.iloc[max(0, i - 14):i + 1]
            if len(subset) >= 2:
                h = subset["high"].values.astype(float)
                l = subset["low"].values.astype(float)
                c = subset["close"].values.astype(float)
                tr_vals = np.maximum(h[1:] - l[1:],
                                     np.maximum(np.abs(h[1:] - c[:-1]),
                                                np.abs(l[1:] - c[:-1])))
                atr_series.append(float(np.mean(tr_vals[-14:])) if len(tr_vals) >= 14 else float(np.mean(tr_vals)))

        if len(atr_series) >= 10 and atr > 0:
            atr_sma = np.mean(atr_series)
            atr_ratio = atr / (atr_sma + 1e-10)
            if atr_ratio < 0.7:
                report.findings.append(AssumptionFinding(
                    category="momentum_loss",
                    severity="info",
                    message=f"ATR contracting (ratio={atr_ratio:.2f}). Breakout expansion may be over.",
                    suggested_action="monitor",
                ))

        # --- Breakout failure: price returned past entry ---
        if direction == "BUY" and current < entry_price * 0.995:
            report.findings.append(AssumptionFinding(
                category="breakout_failed",
                severity="critical",
                message=f"Price ({current:.2f}) returned below entry ({entry_price:.2f}). "
                        f"Breakout may have failed.",
                suggested_action="close" if report.pnl_pct < -1.0 else "monitor",
            ))
        elif direction == "SELL" and current > entry_price * 1.005:
            report.findings.append(AssumptionFinding(
                category="breakout_failed",
                severity="critical",
                message=f"Price ({current:.2f}) returned above entry ({entry_price:.2f}). "
                        f"Breakout may have failed.",
                suggested_action="close" if report.pnl_pct < -1.0 else "monitor",
            ))

    _add_pnl_checks(report)
    _determine_breakout_recommendation(report)
    return report


def _determine_breakout_recommendation(report: PositionAssumptionReport):
    """Set recommendation for breakout trades."""
    has_failed = any(f.category == "breakout_failed" and f.severity == "critical"
                     for f in report.findings)
    has_momentum_loss = any(f.category == "momentum_loss" and f.severity == "warning"
                           for f in report.findings)

    if has_failed and report.pnl_pct < -1.0:
        report.recommended_action = "close"
        return
    if has_failed and has_momentum_loss and report.pnl_pct < 0:
        report.recommended_action = "close"
        return

    # Breakout: don't adjust SL/TP from review — hold or close only
    report.recommended_action = "hold"


# ---------------------------------------------------------------------------
# 5. Volume Profile Review
# ---------------------------------------------------------------------------

def review_volume_profile(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Volume profile review:
    1. Has POC shifted significantly?
    2. Is price still respecting VAL/VAH?
    3. Has the value area shifted?
    4. P/L safety checks
    """
    report.review_strategy = "volume_profile"
    symbol = report.symbol
    direction = report.direction

    df = _fetch_ohlcv(symbol, timeframe, bars=150)

    if df is not None and len(df) >= 50:
        try:
            from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer
            analyzer = VolumeProfileAnalyzer()
            profile = analyzer.calculate_volume_profile(df)
            current = float(df["close"].iloc[-1])
            atr = _compute_atr(df)

            # POC shift: is current POC far from entry?
            if profile and profile.poc > 0 and atr > 0:
                poc_dist = abs(current - profile.poc) / atr

                # Price is far from POC in wrong direction
                if direction == "BUY" and current < profile.poc and poc_dist > 2.0:
                    report.findings.append(AssumptionFinding(
                        category="poc_shift",
                        severity="warning",
                        message=f"Price ({current:.2f}) is {poc_dist:.1f}x ATR below POC ({profile.poc:.2f}). "
                                f"Value area may have shifted against your BUY.",
                        suggested_action="monitor",
                    ))
                elif direction == "SELL" and current > profile.poc and poc_dist > 2.0:
                    report.findings.append(AssumptionFinding(
                        category="poc_shift",
                        severity="warning",
                        message=f"Price ({current:.2f}) is {poc_dist:.1f}x ATR above POC ({profile.poc:.2f}). "
                                f"Value area may have shifted against your SELL.",
                        suggested_action="monitor",
                    ))

                # Price outside value area
                if current > profile.value_area_high:
                    report.findings.append(AssumptionFinding(
                        category="value_area_break",
                        severity="info" if direction == "BUY" else "warning",
                        message=f"Price ({current:.2f}) is above Value Area High ({profile.value_area_high:.2f}).",
                        suggested_action="monitor",
                    ))
                elif current < profile.value_area_low:
                    report.findings.append(AssumptionFinding(
                        category="value_area_break",
                        severity="info" if direction == "SELL" else "warning",
                        message=f"Price ({current:.2f}) is below Value Area Low ({profile.value_area_low:.2f}).",
                        suggested_action="monitor",
                    ))

        except Exception as e:
            logger.debug(f"Volume profile review failed: {e}")

    _add_pnl_checks(report)
    # VP: hold or close only, no SL/TP adjustments from review
    report.recommended_action = "hold"
    return report


# ---------------------------------------------------------------------------
# 6. Rule-Based Trend Review
# ---------------------------------------------------------------------------

def review_rule_based_trend(
    decision: Dict[str, Any],
    position: Dict[str, Any],
    report: PositionAssumptionReport,
    timeframe: str = "H1",
) -> PositionAssumptionReport:
    """
    Rule-based trend review:
    1. EMA alignment still intact?
    2. ADX still strong?
    3. Price crossed below/above key EMA?
    4. P/L safety checks
    """
    report.review_strategy = "rule_based_trend"
    symbol = report.symbol
    direction = report.direction

    df = _fetch_ohlcv(symbol, timeframe, bars=250)

    if df is not None and len(df) >= 60:
        close = pd.Series(df["close"].values.astype(float))
        current = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])

        # EMA alignment check
        if direction == "BUY":
            if current < ema50:
                report.findings.append(AssumptionFinding(
                    category="trend_broken",
                    severity="warning",
                    message=f"Price ({current:.2f}) below EMA50 ({ema50:.2f}). Uptrend may be reversing.",
                    suggested_action="monitor",
                ))
            elif ema20 < ema50:
                report.findings.append(AssumptionFinding(
                    category="trend_weakening",
                    severity="info",
                    message=f"EMA20 ({ema20:.2f}) crossed below EMA50 ({ema50:.2f}). Trend momentum fading.",
                    suggested_action="monitor",
                ))
        else:  # SELL
            if current > ema50:
                report.findings.append(AssumptionFinding(
                    category="trend_broken",
                    severity="warning",
                    message=f"Price ({current:.2f}) above EMA50 ({ema50:.2f}). Downtrend may be reversing.",
                    suggested_action="monitor",
                ))
            elif ema20 > ema50:
                report.findings.append(AssumptionFinding(
                    category="trend_weakening",
                    severity="info",
                    message=f"EMA20 ({ema20:.2f}) crossed above EMA50 ({ema50:.2f}). Trend momentum fading.",
                    suggested_action="monitor",
                ))

        # ADX check
        adx = _compute_adx(df, period=14)
        if adx is not None:
            if adx < 15:
                report.findings.append(AssumptionFinding(
                    category="trend_broken",
                    severity="critical",
                    message=f"ADX at {adx:.1f} — no trend. Rule-based trend thesis invalidated.",
                    suggested_action="close" if report.pnl_pct < 0 else "monitor",
                ))
            elif adx < 20:
                report.findings.append(AssumptionFinding(
                    category="trend_weakening",
                    severity="warning",
                    message=f"ADX at {adx:.1f} — trend weakening.",
                    suggested_action="monitor",
                ))

    _add_pnl_checks(report)
    _determine_trend_recommendation(report)
    return report


def _determine_trend_recommendation(report: PositionAssumptionReport):
    """Set recommendation for trend trades."""
    has_broken = any(f.category == "trend_broken" and f.severity == "critical"
                     for f in report.findings)

    if has_broken and report.pnl_pct < 0:
        report.recommended_action = "close"
        return

    # Trend: hold or close only
    report.recommended_action = "hold"
