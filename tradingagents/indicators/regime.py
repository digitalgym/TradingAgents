"""
Market Regime Detection Module

Detects and classifies market regimes using multiple indicators:
1. Trend Regime: ADX-based (trending-up, trending-down, ranging)
2. Volatility Regime: ATR percentile-based (low, normal, high, extreme)
3. Expansion Regime: Bollinger Band width (expansion, contraction, neutral)

Used for:
- Context-aware decision making
- Regime-filtered memory retrieval
- Pattern analysis by regime
- Adaptive strategy selection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RegimeDetector:
    """Detect market regime using multiple indicators"""
    
    def __init__(
        self,
        adx_threshold_trending: float = 25.0,
        atr_percentile_high: float = 75.0,
        atr_percentile_extreme: float = 90.0,
        bb_width_percentile_expansion: float = 70.0,
        lookback_period: int = 100
    ):
        """
        Initialize regime detector with configurable thresholds.
        
        Args:
            adx_threshold_trending: ADX value above which market is trending (default 25)
            atr_percentile_high: ATR percentile for high volatility (default 75)
            atr_percentile_extreme: ATR percentile for extreme volatility (default 90)
            bb_width_percentile_expansion: BB width percentile for expansion (default 70)
            lookback_period: Historical period for percentile calculations (default 100)
        """
        self.adx_threshold_trending = adx_threshold_trending
        self.atr_percentile_high = atr_percentile_high
        self.atr_percentile_extreme = atr_percentile_extreme
        self.bb_width_percentile_expansion = bb_width_percentile_expansion
        self.lookback_period = lookback_period
    
    def detect_trend_regime(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> str:
        """
        Detect trend regime using ADX (Average Directional Index).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX calculation period (default 14)
        
        Returns:
            "trending-up", "trending-down", or "ranging"
        """
        if len(close) < period + 1:
            return "ranging"
        
        adx = self._calculate_adx(high, low, close, period)
        
        if adx > self.adx_threshold_trending:
            # Strong trend - check direction using price action
            if close[-1] > close[-period]:
                return "trending-up"
            else:
                return "trending-down"
        else:
            return "ranging"
    
    def detect_volatility_regime(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> str:
        """
        Detect volatility regime using ATR percentile.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR calculation period (default 14)
        
        Returns:
            "low", "normal", "high", or "extreme"
        """
        if len(close) < self.lookback_period:
            return "normal"
        
        # Calculate current ATR
        current_atr = self._calculate_atr(high[-period:], low[-period:], close[-period:], period)
        
        # Calculate historical ATR values for percentile comparison
        atr_history = []
        for i in range(period, min(len(close), self.lookback_period)):
            atr = self._calculate_atr(
                high[i-period:i+1],
                low[i-period:i+1],
                close[i-period:i+1],
                period
            )
            atr_history.append(atr)
        
        if len(atr_history) < 10:
            return "normal"
        
        # Calculate percentiles
        percentiles = np.percentile(atr_history, [25, 50, self.atr_percentile_high, self.atr_percentile_extreme])
        
        if current_atr < percentiles[0]:
            return "low"
        elif current_atr < percentiles[1]:
            return "normal"
        elif current_atr < percentiles[2]:
            return "high"
        else:
            return "extreme"
    
    def detect_expansion_regime(
        self,
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> str:
        """
        Detect expansion/contraction using Bollinger Band width.
        
        Args:
            close: Close prices
            period: BB calculation period (default 20)
            std_dev: Standard deviations for bands (default 2.0)
        
        Returns:
            "expansion", "contraction", or "neutral"
        """
        if len(close) < self.lookback_period:
            return "neutral"
        
        # Calculate current BB width
        current_width = self._calculate_bb_width(close[-period:], period, std_dev)
        
        # Calculate historical BB widths
        width_history = []
        for i in range(period, min(len(close), self.lookback_period)):
            width = self._calculate_bb_width(close[i-period:i+1], period, std_dev)
            width_history.append(width)
        
        if len(width_history) < 10:
            return "neutral"
        
        # Calculate percentiles
        percentile_70 = np.percentile(width_history, self.bb_width_percentile_expansion)
        percentile_30 = np.percentile(width_history, 30)
        
        if current_width > percentile_70:
            return "expansion"
        elif current_width < percentile_30:
            return "contraction"
        else:
            return "neutral"
    
    def get_full_regime(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Get complete regime classification.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamp: Optional timestamp for the regime
        
        Returns:
            {
                "market_regime": "trending-up",
                "volatility_regime": "high",
                "expansion_regime": "expansion",
                "timestamp": "2026-01-11T14:30:00"
            }
        """
        market_regime = self.detect_trend_regime(high, low, close)
        volatility_regime = self.detect_volatility_regime(high, low, close)
        expansion_regime = self.detect_expansion_regime(close)
        
        return {
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "expansion_regime": expansion_regime,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
    
    def _calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (0-100):
        - 0-25: Weak or no trend (ranging)
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """
        if len(close) < period + 1:
            return 0.0
        
        # Calculate True Range
        tr = self._calculate_true_range(high, low, close)
        
        # Calculate Directional Movement
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        # Smooth the values
        atr = self._smooth(tr, period)
        plus_di = 100 * self._smooth(plus_dm, period) / atr
        minus_di = 100 * self._smooth(minus_dm, period) / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._smooth(dx, period)
        
        return float(adx[-1]) if len(adx) > 0 else 0.0
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Average True Range (ATR)."""
        if len(close) < 2:
            return 0.0
        
        tr = self._calculate_true_range(high, low, close)
        atr = self._smooth(tr, period)
        
        return float(atr[-1]) if len(atr) > 0 else 0.0
    
    def _calculate_true_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """Calculate True Range."""
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        tr[0] = high[0] - low[0]
        return tr
    
    def _calculate_bb_width(
        self,
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> float:
        """Calculate Bollinger Band width as percentage."""
        if len(close) < period:
            return 0.0
        
        sma = np.mean(close[-period:])
        std = np.std(close[-period:], ddof=1)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Width as percentage of middle band
        width = ((upper_band - lower_band) / sma) * 100 if sma > 0 else 0.0
        
        return float(width)
    
    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Smooth data using Wilder's smoothing (EMA-like)."""
        if len(data) < period:
            return data
        
        smoothed = np.zeros(len(data))
        smoothed[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period
        
        return smoothed
    
    def get_regime_description(self, regime: Dict[str, str]) -> str:
        """
        Get human-readable description of regime.
        
        Args:
            regime: Regime dict from get_full_regime()
        
        Returns:
            Descriptive string like "Trending up in high volatility with expansion"
        """
        market = regime.get("market_regime", "unknown")
        volatility = regime.get("volatility_regime", "unknown")
        expansion = regime.get("expansion_regime", "unknown")
        
        # Market description
        market_desc = {
            "trending-up": "Trending upward",
            "trending-down": "Trending downward",
            "ranging": "Ranging"
        }.get(market, "Unknown trend")
        
        # Volatility description
        vol_desc = {
            "low": "low volatility",
            "normal": "normal volatility",
            "high": "high volatility",
            "extreme": "extreme volatility"
        }.get(volatility, "unknown volatility")
        
        # Expansion description
        exp_desc = {
            "expansion": "with expansion",
            "contraction": "with contraction",
            "neutral": "with neutral range"
        }.get(expansion, "")
        
        return f"{market_desc} in {vol_desc} {exp_desc}".strip()
    
    def is_favorable_for_trend_trading(self, regime: Dict[str, str]) -> bool:
        """
        Check if regime is favorable for trend trading.
        
        Favorable conditions:
        - Trending market (up or down)
        - Normal to high volatility (not extreme)
        - Expansion or neutral
        """
        market = regime.get("market_regime", "")
        volatility = regime.get("volatility_regime", "")
        expansion = regime.get("expansion_regime", "")
        
        is_trending = market in ["trending-up", "trending-down"]
        good_volatility = volatility in ["normal", "high"]
        good_expansion = expansion in ["expansion", "neutral"]
        
        return is_trending and good_volatility and good_expansion
    
    def is_favorable_for_range_trading(self, regime: Dict[str, str]) -> bool:
        """
        Check if regime is favorable for range trading.
        
        Favorable conditions:
        - Ranging market
        - Low to normal volatility
        - Contraction or neutral
        """
        market = regime.get("market_regime", "")
        volatility = regime.get("volatility_regime", "")
        expansion = regime.get("expansion_regime", "")
        
        is_ranging = market == "ranging"
        low_volatility = volatility in ["low", "normal"]
        contracting = expansion in ["contraction", "neutral"]
        
        return is_ranging and low_volatility and contracting
    
    def get_risk_adjustment_factor(self, regime: Dict[str, str]) -> float:
        """
        Get position size adjustment factor based on regime.
        
        Returns:
            Multiplier for position size (0.5 to 1.5):
            - Extreme volatility: 0.5x (reduce risk)
            - High volatility: 0.75x
            - Normal: 1.0x
            - Low volatility: 1.25x (can increase slightly)
        """
        volatility = regime.get("volatility_regime", "normal")
        
        adjustments = {
            "extreme": 0.5,
            "high": 0.75,
            "normal": 1.0,
            "low": 1.25
        }
        
        return adjustments.get(volatility, 1.0)
