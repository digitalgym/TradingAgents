"""
Vectorised Technical Features for XGBoost.

Computes all standard technical indicators as full arrays (one value per bar).
Normalised relative to price/ATR so features are comparable across symbols.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from tradingagents.xgb_quant.features.base import BaseFeatureSet
from tradingagents.xgb_quant.config import FeatureWindows


class TechnicalFeatures(BaseFeatureSet):
    """Core technical indicator features — shared across all strategies."""

    @property
    def feature_names(self) -> list:
        return [
            # ATR / Volatility
            "atr_14", "atr_pct",
            # RSI
            "rsi_14",
            # MACD
            "macd_hist", "macd_hist_norm",
            # Bollinger Bands
            "bb_pct", "bb_width",
            # EMA
            "ema_short_dist", "ema_long_dist", "ema_cross",
            # ADX
            "adx_14", "plus_di", "minus_di", "di_diff",
            # Volume
            "volume_ratio", "volume_sma_ratio",
            # Returns / Momentum
            "returns_1", "returns_5", "returns_20",
            "roc_10",
            # Price position
            "high_low_range_pct",
            "close_position",  # Where close is within high-low range
            # Time
            "hour", "day_of_week",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_ = df["open"].values.astype(float)
        volume = df["volume"].values.astype(float)
        w = self.windows

        features = pd.DataFrame(index=df.index)

        # --- ATR ---
        atr = self._compute_atr(high, low, close, w.atr_period)
        features["atr_14"] = atr
        features["atr_pct"] = self._safe_divide(atr, close)

        # --- RSI ---
        features["rsi_14"] = self._compute_rsi(close, w.rsi_period)

        # --- MACD ---
        ema_fast = self._ema(close, w.macd_fast)
        ema_slow = self._ema(close, w.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, w.macd_signal)
        macd_hist = macd_line - signal_line
        features["macd_hist"] = macd_hist
        features["macd_hist_norm"] = self._safe_divide(macd_hist, atr)

        # --- Bollinger Bands ---
        sma = pd.Series(close).rolling(w.bb_period).mean().values
        std = pd.Series(close).rolling(w.bb_period).std().values
        bb_upper = sma + w.bb_std * std
        bb_lower = sma - w.bb_std * std
        bb_range = bb_upper - bb_lower
        features["bb_pct"] = self._safe_divide(close - bb_lower, bb_range)
        features["bb_width"] = self._safe_divide(bb_range, sma)

        # --- EMA distances (normalised to ATR) ---
        ema_short = self._ema(close, w.ema_short)
        ema_long = self._ema(close, w.ema_long)
        features["ema_short_dist"] = self._safe_divide(close - ema_short, atr)
        features["ema_long_dist"] = self._safe_divide(close - ema_long, atr)
        features["ema_cross"] = (ema_short > ema_long).astype(float)

        # --- ADX ---
        adx, plus_di, minus_di = self._compute_adx(high, low, close, w.adx_period)
        features["adx_14"] = adx
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di

        # --- Volume ---
        vol_sma = pd.Series(volume).rolling(w.volume_avg_period).mean().values
        features["volume_ratio"] = self._safe_divide(volume, vol_sma)
        features["volume_sma_ratio"] = self._safe_divide(
            pd.Series(volume).rolling(5).mean().values, vol_sma
        )

        # --- Returns ---
        cs = pd.Series(close)
        features["returns_1"] = cs.pct_change(1).values
        features["returns_5"] = cs.pct_change(5).values
        features["returns_20"] = cs.pct_change(20).values
        features["roc_10"] = cs.pct_change(10).values

        # --- Price structure ---
        hl_range = high - low
        features["high_low_range_pct"] = self._safe_divide(hl_range, close)
        features["close_position"] = self._safe_divide(close - low, hl_range)

        # --- Time features ---
        if hasattr(df, "index") and hasattr(df.index, "hour"):
            features["hour"] = df.index.hour.astype(float)
            features["day_of_week"] = df.index.dayofweek.astype(float)
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"])
            features["hour"] = dt.dt.hour.astype(float)
            features["day_of_week"] = dt.dt.dayofweek.astype(float)
        else:
            features["hour"] = 0.0
            features["day_of_week"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Indicator computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def _compute_atr(high, low, close, period: int = 14) -> np.ndarray:
        """Wilder's ATR — same as auto_tuner._compute_atr."""
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI as full array."""
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).values

    @staticmethod
    def _compute_adx(high, low, close, period: int = 14):
        """ADX with +DI and -DI as full arrays."""
        n = len(close)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

        atr = pd.Series(tr).rolling(period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / (atr + 1e-10)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(period).mean().values

        return adx, plus_di, minus_di
