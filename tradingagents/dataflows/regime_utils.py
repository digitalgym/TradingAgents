"""
Regime detection utilities for market data

Provides helper functions to detect and attach regime information to market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from tradingagents.indicators.regime import RegimeDetector


def add_regime_to_dataframe(
    df: pd.DataFrame,
    detector: Optional[RegimeDetector] = None
) -> pd.DataFrame:
    """
    Add regime classification columns to a price DataFrame.
    
    Args:
        df: DataFrame with OHLC data (columns: High, Low, Close)
        detector: Optional RegimeDetector instance (creates default if None)
    
    Returns:
        DataFrame with added columns: market_regime, volatility_regime, expansion_regime
    """
    if detector is None:
        detector = RegimeDetector()
    
    # Ensure we have required columns
    required = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must contain columns: {required}")
    
    if len(df) < 20:
        # Not enough data for regime detection
        df['market_regime'] = 'ranging'
        df['volatility_regime'] = 'normal'
        df['expansion_regime'] = 'neutral'
        return df
    
    # Convert to numpy arrays
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # Detect regime for each row (using all data up to that point)
    market_regimes = []
    volatility_regimes = []
    expansion_regimes = []
    
    for i in range(len(df)):
        if i < 20:
            # Not enough data yet
            market_regimes.append('ranging')
            volatility_regimes.append('normal')
            expansion_regimes.append('neutral')
        else:
            # Use data up to current point
            regime = detector.get_full_regime(
                high[:i+1],
                low[:i+1],
                close[:i+1]
            )
            market_regimes.append(regime['market_regime'])
            volatility_regimes.append(regime['volatility_regime'])
            expansion_regimes.append(regime['expansion_regime'])
    
    df['market_regime'] = market_regimes
    df['volatility_regime'] = volatility_regimes
    df['expansion_regime'] = expansion_regimes
    
    return df


def get_current_regime_from_prices(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    detector: Optional[RegimeDetector] = None
) -> Dict[str, str]:
    """
    Get current regime from price arrays.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        detector: Optional RegimeDetector instance
    
    Returns:
        Regime dict with market_regime, volatility_regime, expansion_regime
    """
    if detector is None:
        detector = RegimeDetector()
    
    return detector.get_full_regime(high, low, close)


def format_regime_for_prompt(regime: Dict[str, str]) -> str:
    """
    Format regime information for inclusion in LLM prompts.
    
    Args:
        regime: Regime dict from RegimeDetector
    
    Returns:
        Formatted string for prompt inclusion
    """
    detector = RegimeDetector()
    description = detector.get_regime_description(regime)
    
    market = regime.get('market_regime', 'unknown')
    volatility = regime.get('volatility_regime', 'unknown')
    expansion = regime.get('expansion_regime', 'unknown')
    
    # Get trading recommendations
    trend_favorable = detector.is_favorable_for_trend_trading(regime)
    range_favorable = detector.is_favorable_for_range_trading(regime)
    risk_adjustment = detector.get_risk_adjustment_factor(regime)
    
    output = f"""MARKET REGIME:
- Trend: {market}
- Volatility: {volatility}
- Expansion: {expansion}
- Description: {description}

REGIME IMPLICATIONS:
- Trend trading favorable: {'YES' if trend_favorable else 'NO'}
- Range trading favorable: {'YES' if range_favorable else 'NO'}
- Position size adjustment: {risk_adjustment:.2f}x (based on volatility)
"""
    
    return output


def get_regime_summary(regime: Dict[str, str]) -> str:
    """
    Get a concise one-line regime summary.
    
    Args:
        regime: Regime dict from RegimeDetector
    
    Returns:
        One-line summary string
    """
    market = regime.get('market_regime', 'unknown')
    volatility = regime.get('volatility_regime', 'unknown')
    
    return f"{market} / {volatility} volatility"
