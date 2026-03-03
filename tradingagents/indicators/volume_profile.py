"""
Volume Profile Indicator

Calculates Volume Profile metrics from OHLCV data:
- POC (Point of Control): Price level with highest volume
- Value Area High/Low: Range containing 70% of volume
- HVN (High Volume Nodes): Price levels with above-average volume
- LVN (Low Volume Nodes): Price levels with below-average volume

Usage:
    from tradingagents.indicators.volume_profile import VolumeProfileAnalyzer

    analyzer = VolumeProfileAnalyzer()
    profile = analyzer.calculate_volume_profile(df, num_bins=50)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class VolumeNode:
    """A volume node at a specific price level."""
    price: float
    volume: float
    volume_pct: float  # Percentage of total volume
    node_type: str  # "hvn", "lvn", or "normal"


@dataclass
class VolumeProfile:
    """Complete volume profile analysis."""
    poc: float  # Point of Control - price with highest volume
    poc_volume: float  # Volume at POC
    poc_volume_pct: float  # POC volume as % of total
    value_area_high: float  # Upper bound of value area (70% of volume)
    value_area_low: float  # Lower bound of value area
    value_area_pct: float  # Actual percentage in value area
    high_volume_nodes: List[VolumeNode]  # HVN levels
    low_volume_nodes: List[VolumeNode]  # LVN levels (potential fast-move zones)
    profile_high: float  # Highest price in profile
    profile_low: float  # Lowest price in profile
    total_volume: float
    developing: bool  # True if this is a developing profile (current session)


class VolumeProfileAnalyzer:
    """
    Volume Profile analyzer for price-volume analysis.

    Volume Profile shows where volume traded at each price level,
    helping identify:
    - POC: Price where most trading occurred (magnet level)
    - Value Area: Fair value range where 70% of volume traded
    - HVN: High volume nodes act as support/resistance
    - LVN: Low volume nodes where price moves fast (less interest)
    """

    def __init__(
        self,
        value_area_pct: float = 0.70,
        hvn_threshold: float = 1.5,  # Volume > 1.5x average = HVN
        lvn_threshold: float = 0.5,  # Volume < 0.5x average = LVN
    ):
        """
        Initialize Volume Profile analyzer.

        Args:
            value_area_pct: Percentage of volume for value area (default 70%)
            hvn_threshold: Multiple of average volume to be HVN
            lvn_threshold: Multiple of average volume to be LVN
        """
        self.value_area_pct = value_area_pct
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold

    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        num_bins: int = 50,
        lookback: Optional[int] = None,
    ) -> VolumeProfile:
        """
        Calculate volume profile from OHLCV data.

        Uses TPO (Time Price Opportunity) style calculation where
        volume is distributed across the candle's price range.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            num_bins: Number of price bins for the profile
            lookback: Number of candles to analyze (None = all)

        Returns:
            VolumeProfile with all metrics
        """
        if lookback:
            data = df.tail(lookback).copy()
        else:
            data = df.copy()

        if len(data) < 5:
            # Not enough data
            return self._empty_profile()

        # Get price range
        profile_high = data['high'].max()
        profile_low = data['low'].min()
        price_range = profile_high - profile_low

        if price_range <= 0:
            return self._empty_profile()

        # Create price bins
        bin_size = price_range / num_bins
        bins = np.linspace(profile_low, profile_high, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Initialize volume array
        volume_at_price = np.zeros(num_bins)

        # Distribute each candle's volume across its price range
        for _, row in data.iterrows():
            candle_high = row['high']
            candle_low = row['low']
            candle_volume = row.get('volume', row.get('tick_volume', 1))

            if pd.isna(candle_volume) or candle_volume <= 0:
                candle_volume = 1

            # Find bins this candle covers
            for i in range(num_bins):
                bin_low = bins[i]
                bin_high = bins[i + 1]

                # Check overlap
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)

                if overlap_high > overlap_low:
                    # Calculate proportion of candle in this bin
                    candle_range = candle_high - candle_low
                    if candle_range > 0:
                        overlap_pct = (overlap_high - overlap_low) / candle_range
                    else:
                        overlap_pct = 1.0 if bin_low <= candle_low <= bin_high else 0.0

                    volume_at_price[i] += candle_volume * overlap_pct

        total_volume = volume_at_price.sum()
        if total_volume <= 0:
            return self._empty_profile()

        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]
        poc_volume = volume_at_price[poc_idx]
        poc_volume_pct = (poc_volume / total_volume) * 100

        # Calculate Value Area (70% of volume centered on POC)
        va_high, va_low, va_pct = self._calculate_value_area(
            bins, volume_at_price, poc_idx, total_volume
        )

        # Identify HVN and LVN
        avg_volume = total_volume / num_bins
        hvn_list = []
        lvn_list = []

        for i in range(num_bins):
            vol = volume_at_price[i]
            vol_pct = (vol / total_volume) * 100

            if vol > avg_volume * self.hvn_threshold:
                node_type = "hvn"
                hvn_list.append(VolumeNode(
                    price=bin_centers[i],
                    volume=vol,
                    volume_pct=vol_pct,
                    node_type=node_type
                ))
            elif vol < avg_volume * self.lvn_threshold and vol > 0:
                node_type = "lvn"
                lvn_list.append(VolumeNode(
                    price=bin_centers[i],
                    volume=vol,
                    volume_pct=vol_pct,
                    node_type=node_type
                ))

        # Sort HVN by volume (highest first), LVN by price (for context)
        hvn_list.sort(key=lambda x: x.volume, reverse=True)
        lvn_list.sort(key=lambda x: x.price)

        return VolumeProfile(
            poc=poc,
            poc_volume=poc_volume,
            poc_volume_pct=poc_volume_pct,
            value_area_high=va_high,
            value_area_low=va_low,
            value_area_pct=va_pct,
            high_volume_nodes=hvn_list[:5],  # Top 5 HVN
            low_volume_nodes=lvn_list[:5],   # Top 5 LVN
            profile_high=profile_high,
            profile_low=profile_low,
            total_volume=total_volume,
            developing=False
        )

    def _calculate_value_area(
        self,
        bins: np.ndarray,
        volume_at_price: np.ndarray,
        poc_idx: int,
        total_volume: float
    ) -> tuple:
        """Calculate value area containing target % of volume."""
        num_bins = len(volume_at_price)
        target_volume = total_volume * self.value_area_pct

        # Start at POC and expand outward
        va_volume = volume_at_price[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while va_volume < target_volume:
            # Check which direction to expand
            can_go_lower = low_idx > 0
            can_go_higher = high_idx < num_bins - 1

            if not can_go_lower and not can_go_higher:
                break

            # Calculate volume gain in each direction
            low_gain = volume_at_price[low_idx - 1] if can_go_lower else 0
            high_gain = volume_at_price[high_idx + 1] if can_go_higher else 0

            # Expand toward higher volume
            if low_gain >= high_gain and can_go_lower:
                low_idx -= 1
                va_volume += volume_at_price[low_idx]
            elif can_go_higher:
                high_idx += 1
                va_volume += volume_at_price[high_idx]
            elif can_go_lower:
                low_idx -= 1
                va_volume += volume_at_price[low_idx]
            else:
                break

        va_high = bins[high_idx + 1]  # Upper edge of high bin
        va_low = bins[low_idx]        # Lower edge of low bin
        va_pct = (va_volume / total_volume) * 100

        return va_high, va_low, va_pct

    def _empty_profile(self) -> VolumeProfile:
        """Return empty profile when insufficient data."""
        return VolumeProfile(
            poc=0,
            poc_volume=0,
            poc_volume_pct=0,
            value_area_high=0,
            value_area_low=0,
            value_area_pct=0,
            high_volume_nodes=[],
            low_volume_nodes=[],
            profile_high=0,
            profile_low=0,
            total_volume=0,
            developing=False
        )

    def format_for_prompt(
        self,
        profile: VolumeProfile,
        current_price: float
    ) -> str:
        """
        Format volume profile for LLM prompt consumption.

        Args:
            profile: Calculated volume profile
            current_price: Current market price

        Returns:
            Formatted string for prompt
        """
        if profile.total_volume <= 0:
            return "### Volume Profile\nInsufficient volume data.\n"

        lines = ["### Volume Profile"]

        # POC
        poc_distance = ((current_price - profile.poc) / current_price * 100)
        if poc_distance > 0:
            poc_position = f"+{poc_distance:.2f}% above POC"
        else:
            poc_position = f"{poc_distance:.2f}% below POC"

        lines.append(f"**POC (Point of Control)**: {profile.poc:.5f} ({profile.poc_volume_pct:.1f}% of volume)")
        lines.append(f"  - Price is {poc_position} - POC acts as magnet")

        # Value Area
        if current_price >= profile.value_area_low and current_price <= profile.value_area_high:
            va_status = "INSIDE value area (fair value zone)"
        elif current_price > profile.value_area_high:
            va_status = "ABOVE value area (premium - expect mean reversion)"
        else:
            va_status = "BELOW value area (discount - expect mean reversion)"

        lines.append(f"**Value Area**: {profile.value_area_low:.5f} - {profile.value_area_high:.5f} ({profile.value_area_pct:.0f}% of volume)")
        lines.append(f"  - Price is {va_status}")

        # HVN (support/resistance)
        if profile.high_volume_nodes:
            lines.append("**High Volume Nodes** (support/resistance):")
            for hvn in profile.high_volume_nodes[:3]:
                dist = ((current_price - hvn.price) / current_price * 100)
                if dist > 0.1:
                    lines.append(f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | -{abs(dist):.2f}% below")
                elif dist < -0.1:
                    lines.append(f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | +{abs(dist):.2f}% above")
                else:
                    lines.append(f"  - HVN at {hvn.price:.5f} ({hvn.volume_pct:.1f}% vol) | AT PRICE")

        # LVN (fast-move zones)
        if profile.low_volume_nodes:
            lines.append("**Low Volume Nodes** (fast-move zones):")
            for lvn in profile.low_volume_nodes[:3]:
                dist = ((current_price - lvn.price) / current_price * 100)
                if dist > 0:
                    lines.append(f"  - LVN at {lvn.price:.5f} | -{abs(dist):.2f}% - price may move fast through this level")
                else:
                    lines.append(f"  - LVN at {lvn.price:.5f} | +{abs(dist):.2f}% - price may move fast through this level")

        lines.append("")
        return "\n".join(lines)


def calculate_session_profile(
    df: pd.DataFrame,
    session_bars: int = 24,  # Typical session length
) -> VolumeProfile:
    """
    Calculate volume profile for current session (developing profile).

    Args:
        df: DataFrame with OHLCV data
        session_bars: Number of bars in a session

    Returns:
        VolumeProfile marked as developing
    """
    analyzer = VolumeProfileAnalyzer()
    profile = analyzer.calculate_volume_profile(df, lookback=session_bars)
    profile.developing = True
    return profile
