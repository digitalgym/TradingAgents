"""
Pair Scanner — finds forex pairs on the move.

Scans a watchlist for momentum, ATR expansion, trend strength,
and structure breaks. Pure price data — no LLM, no ML.
Returns a ranked shortlist with direction bias.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from tradingagents.quant_strats.config import ScannerConfig, DEFAULT_WATCHLIST

logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Momentum score for a single pair."""
    symbol: str
    direction: str              # "LONG" or "SHORT"
    momentum_score: int         # 0-100 composite

    # Components
    atr_expansion: float = 0.0
    adx_strength: float = 0.0
    directional_move_pct: float = 0.0
    structure_break: bool = False
    ema_alignment: bool = False
    volume_confirmation: bool = False
    spread_cost_ratio: float = 0.0

    # Regime detection
    regime: str = ""            # "trending", "ranging", "squeeze", "volatile_trend", etc.
    recommended_pipeline: str = ""  # Pipeline best suited for this regime
    recommended_timeframe: str = ""  # Best analysis timeframe for this regime

    # Filters
    is_choppy: bool = False
    spread_too_wide: bool = False
    already_has_position: bool = False
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class ScanResult:
    """Result of scanning the watchlist."""
    timestamp: str = ""
    watchlist_size: int = 0
    shortlist: List[PairScore] = field(default_factory=list)
    disqualified: List[PairScore] = field(default_factory=list)
    best_candidate: Optional[PairScore] = None


class PairScanner:
    """Scans forex pairs for momentum and movement."""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self._blacklist = self._load_blacklist()

    @staticmethod
    def _load_blacklist() -> set:
        """Load blacklisted symbol+pipeline combos from batch training results."""
        try:
            from tradingagents.quant_strats.batch_trainer import BatchTrainer
            entries = BatchTrainer.load_blacklist()
            # Build a set of "SYMBOL:pipeline" for fast lookup
            # Map strategy names back to pipeline names where they differ
            _STRATEGY_TO_PIPELINE = {
                "trend_following": "xgboost",
                "mean_reversion": "xgboost",
                "breakout": "xgboost",
                "smc_zones": "xgboost",
                "volume_profile_strat": "xgboost",
            }
            bl = set()
            for e in entries:
                # Blacklist the specific strategy for xgboost routing
                bl.add(f"{e['symbol']}:{e['strategy']}:{e.get('timeframe', '')}")
            return bl
        except Exception:
            return set()

    def scan(
        self,
        watchlist: Optional[List[str]] = None,
        existing_positions: Optional[List[str]] = None,
        fetch_data_fn=None,
    ) -> ScanResult:
        """
        Scan watchlist and return ranked pairs.

        Args:
            watchlist: List of symbols to scan (default: DEFAULT_WATCHLIST)
            existing_positions: Symbols we already hold (will be filtered out)
            fetch_data_fn: Callable(symbol, timeframe, bars) -> pd.DataFrame
                           If None, uses MT5 data loading.
        """
        from datetime import datetime

        watchlist = watchlist or DEFAULT_WATCHLIST
        existing = set(existing_positions or [])

        if fetch_data_fn is None:
            fetch_data_fn = self._fetch_mt5_data

        result = ScanResult(
            timestamp=datetime.utcnow().isoformat(),
            watchlist_size=len(watchlist),
        )

        for symbol in watchlist:
            try:
                df = fetch_data_fn(symbol, self.config.scan_timeframe, self.config.lookback_bars)
                if df is None or len(df) < 50:
                    continue

                score = self._score_pair(df, symbol)

                # Apply filters
                if symbol in existing:
                    score.already_has_position = True
                    score.disqualified = True
                    score.disqualify_reason = "Already positioned"

                if score.is_choppy:
                    score.disqualified = True
                    score.disqualify_reason = "Choppy (ADX < 20)"

                if score.spread_too_wide:
                    score.disqualified = True
                    score.disqualify_reason = f"Spread too wide ({score.spread_cost_ratio:.2f})"

                if score.momentum_score < self.config.min_momentum_score:
                    score.disqualified = True
                    score.disqualify_reason = f"Low momentum ({score.momentum_score})"

                if score.disqualified:
                    result.disqualified.append(score)
                else:
                    result.shortlist.append(score)

            except Exception as e:
                logger.warning(f"Failed to scan {symbol}: {e}")
                continue

        # Sort by momentum score
        result.shortlist.sort(key=lambda s: s.momentum_score, reverse=True)

        if result.shortlist:
            result.best_candidate = result.shortlist[0]

        logger.info(
            f"Scan complete: {len(result.shortlist)} candidates from "
            f"{result.watchlist_size} pairs. Best: "
            f"{result.best_candidate.symbol if result.best_candidate else 'none'}"
        )

        return result

    def _score_pair(self, df: pd.DataFrame, symbol: str) -> PairScore:
        """Calculate momentum score for a single pair."""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df else np.ones(len(close))

        # --- ATR expansion ---
        atr = self._compute_atr(high, low, close)
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
        avg_atr = np.nanmean(atr[-20:])
        atr_expansion = current_atr / avg_atr if avg_atr > 0 else 1.0

        # --- ADX ---
        adx, plus_di, minus_di = self._compute_adx(high, low, close)
        adx_val = adx[-1] if not np.isnan(adx[-1]) else 0

        # --- Direction ---
        ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
        ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values

        # Directional move over last 10 bars
        if close[-11] != 0:
            dir_move = (close[-1] - close[-11]) / close[-11] * 100
        else:
            dir_move = 0

        # EMA alignment
        ema_aligned = ema20[-1] > ema50[-1]

        # Determine direction
        if dir_move > 0 and ema_aligned:
            direction = "LONG"
        elif dir_move < 0 and not ema_aligned:
            direction = "SHORT"
        elif abs(dir_move) > 0.5:
            direction = "LONG" if dir_move > 0 else "SHORT"
        else:
            direction = "LONG" if ema_aligned else "SHORT"

        # --- Structure break ---
        recent_high = np.max(high[-21:-1]) if len(high) > 21 else high[-1]
        recent_low = np.min(low[-21:-1]) if len(low) > 21 else low[-1]
        structure_break = close[-1] > recent_high or close[-1] < recent_low

        # --- Volume confirmation ---
        vol_avg = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_confirm = volume[-1] > vol_avg * 1.2

        # --- Spread cost (estimated from recent bars) ---
        avg_range = np.mean(high[-20:] - low[-20:])
        spread_est = avg_range * 0.01  # Rough estimate, 1% of range
        spread_ratio = spread_est / current_atr if current_atr > 0 else 1.0

        # --- Bollinger Band squeeze detection ---
        bb_period = 20
        bb_std_mult = 2.0
        sma = pd.Series(close).rolling(bb_period).mean().values
        std = pd.Series(close).rolling(bb_period).std().values
        bb_width = (std[-1] * bb_std_mult * 2) / sma[-1] if sma[-1] > 0 else 0
        avg_bb_width = np.nanmean(
            (std[-40:] * bb_std_mult * 2) / np.where(sma[-40:] > 0, sma[-40:], 1)
        )
        bb_squeeze = bb_width < avg_bb_width * 0.6 if avg_bb_width > 0 else False

        # BB recently tight but now expanding = fresh breakout signature.
        # Check if BB was squeezed 3-5 bars ago but current width is above squeeze threshold.
        bb_width_prev = np.nanmean(
            (std[-6:-1] * bb_std_mult * 2) / np.where(sma[-6:-1] > 0, sma[-6:-1], 1)
        ) if len(std) > 6 else bb_width
        bb_was_squeezed = bb_width_prev < avg_bb_width * 0.6 if avg_bb_width > 0 else False
        bb_expanding = not bb_squeeze and bb_was_squeezed  # Was tight, now opening up

        # --- ADX slope (is trend strengthening or fading?) ---
        # Compare current ADX to ADX 5 bars ago
        adx_prev = adx[-6] if len(adx) > 6 and not np.isnan(adx[-6]) else adx_val
        adx_slope = adx_val - adx_prev  # positive = strengthening, negative = fading

        # --- RSI for exhaustion divergence ---
        rsi_val = self._compute_rsi(close)

        # --- Regime classification ---
        regime, recommended_pipeline, recommended_tf = self._classify_regime(
            adx_val, atr_expansion, bb_squeeze, structure_break, ema_aligned, dir_move,
            adx_slope, rsi_val, bb_expanding, vol_confirm,
        )

        # --- Score calculation ---
        score = 0

        # ATR expansion (0-25)
        if atr_expansion > 1.5:
            score += 25
        elif atr_expansion > 1.2:
            score += 15
        elif atr_expansion > 1.0:
            score += 5

        # ADX (0-25)
        if adx_val > 40:
            score += 25
        elif adx_val > 30:
            score += 20
        elif adx_val > 25:
            score += 15
        elif adx_val > 20:
            score += 5

        # Directional move (0-20)
        abs_move = abs(dir_move)
        if abs_move > 1.5:
            score += 20
        elif abs_move > 1.0:
            score += 15
        elif abs_move > 0.5:
            score += 10

        # Structure break (0-15)
        if structure_break:
            score += 15

        # EMA alignment (0-10)
        if ema_aligned and direction == "LONG":
            score += 10
        elif not ema_aligned and direction == "SHORT":
            score += 10

        # Volume (0-5)
        if vol_confirm:
            score += 5

        # Validate assignment against backtest blacklist
        # If the xgboost strategy for this pair is blacklisted, the selector
        # will route to a better one automatically. But if a non-xgboost pipeline
        # (e.g. range_quant) has no backtest data, we trust the regime mapping.
        recommended_pipeline, recommended_tf = self._validate_assignment(
            symbol, recommended_pipeline, recommended_tf, regime,
        )

        return PairScore(
            symbol=symbol,
            direction=direction,
            momentum_score=min(score, 100),
            atr_expansion=atr_expansion,
            adx_strength=adx_val,
            directional_move_pct=dir_move,
            structure_break=structure_break,
            ema_alignment=ema_aligned,
            volume_confirmation=vol_confirm,
            spread_cost_ratio=spread_ratio,
            regime=regime,
            recommended_pipeline=recommended_pipeline,
            recommended_timeframe=recommended_tf,
            is_choppy=adx_val < self.config.min_adx,
            spread_too_wide=spread_ratio > self.config.max_spread_atr_ratio,
        )

    # Regime → (pipeline, timeframe) mapping.
    # Timeframe rationale: trends develop on D1, breakouts on H4, ranging on H4.
    REGIME_PIPELINE_MAP = {
        "strong_trend":     ("rule_based",     "D1"),  # 75% WR on D1, trends need daily perspective
        "moderate_trend":   ("xgboost",        "D1"),  # ML picks pattern, D1 best in backtests
        "ranging":          ("range_quant",    "H4"),  # Mean-reversion needs faster granularity
        "squeeze":          ("breakout_quant", "H4"),  # Breakouts need H4 to catch the move
        "breakout_fresh":   ("breakout_quant", "H4"),  # Continuation entry on H4
        "volatile_trend":   ("smc_mtf",        "D1"),  # MTF uses D1 as higher TF
        "trend_exhaustion": ("smc_quant_basic","D1"),  # LLM needs D1 context to judge exhaustion
    }

    @staticmethod
    def _classify_regime(
        adx: float,
        atr_expansion: float,
        bb_squeeze: bool,
        structure_break: bool,
        ema_aligned: bool,
        dir_move: float,
        adx_slope: float = 0.0,
        rsi: float = 50.0,
        bb_expanding: bool = False,
        volume_confirmed: bool = False,
    ) -> tuple:
        """
        Classify market regime from scanner indicators.

        Returns (regime_name, recommended_pipeline, recommended_timeframe).
        """
        abs_move = abs(dir_move)

        def _map(regime: str):
            pipeline, tf = PairScanner.REGIME_PIPELINE_MAP[regime]
            return regime, pipeline, tf

        # Squeeze: tight BB + low ADX + no structure break → about to move
        if bb_squeeze and adx < 25 and not structure_break:
            return _map("squeeze")

        # Fresh breakout: BB was squeezed but just expanded + structure break.
        if bb_expanding and structure_break:
            return _map("breakout_fresh")

        # Trend exhaustion: ADX was high but is now falling + RSI extreme.
        if adx > 25 and adx_slope < -3 and (rsi > 70 or rsi < 30):
            return _map("trend_exhaustion")

        # Strong trend: high ADX + EMA aligned + big move
        if adx > 35 and ema_aligned and abs_move > 0.8:
            if atr_expansion > 1.5:
                return _map("volatile_trend")
            return _map("strong_trend")

        # Moderate trend: decent ADX or clear EMA alignment
        if adx > 22 and (ema_aligned or abs_move > 0.5):
            if atr_expansion > 1.4:
                return _map("volatile_trend")
            return _map("moderate_trend")

        # Ranging: low ADX, small move, no alignment
        if adx < 22 and abs_move < 0.5:
            return _map("ranging")

        # Default: moderate trend (safe fallback)
        return _map("moderate_trend")

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute current RSI value."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _validate_assignment(
        self,
        symbol: str,
        pipeline: str,
        timeframe: str,
        regime: str,
    ) -> tuple:
        """
        Check if the assigned pipeline has viable backtest results for this pair.

        If xgboost is assigned but no good model exists for this pair, fall back
        to rule_based. Other pipelines (range_quant, breakout_quant, etc.) are
        trusted since they don't depend on trained models.
        """
        # Only xgboost depends on trained models
        if pipeline != "xgboost":
            return pipeline, timeframe

        # Check if any viable model exists for this pair
        try:
            from tradingagents.quant_strats.strategy_selector import StrategySelector
            selector = StrategySelector()
            selection = selector.select(symbol, regime=regime)

            # If best strategy has decent backtest, keep xgboost
            if selection.sharpe > 0 and selection.win_rate >= 40:
                # Use the selector's recommended timeframe if available
                return pipeline, selection.recommended_timeframe or timeframe

            # Bad backtest — fall back to rule_based (free, no model needed)
            logger.info(
                f"SCANNER: {symbol} xgboost has poor backtest "
                f"(Sharpe={selection.sharpe:.2f}, WR={selection.win_rate:.0f}%), "
                f"falling back to rule_based"
            )
            return "rule_based", "D1"

        except Exception:
            # Can't check — trust the regime mapping
            return pipeline, timeframe

    def _fetch_mt5_data(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch data from MT5."""
        try:
            from tradingagents.automation.auto_tuner import load_mt5_data
            return load_mt5_data(symbol, timeframe, bars)
        except Exception as e:
            logger.warning(f"MT5 fetch failed for {symbol}: {e}")
            return None

    @staticmethod
    def _compute_atr(high, low, close, period: int = 14) -> np.ndarray:
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _compute_adx(high, low, close, period: int = 14):
        n = len(close)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)
        for i in range(1, n):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(period).mean().values
        return adx, plus_di, minus_di
