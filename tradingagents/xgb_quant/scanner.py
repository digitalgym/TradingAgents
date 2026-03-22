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

from tradingagents.xgb_quant.config import ScannerConfig, DEFAULT_WATCHLIST

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
            is_choppy=adx_val < self.config.min_adx,
            spread_too_wide=spread_ratio > self.config.max_spread_atr_ratio,
        )

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
