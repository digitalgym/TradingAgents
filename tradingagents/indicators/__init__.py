"""
Technical indicators module

Components:
- regime: Market regime detection (trend, volatility, expansion)
"""

from .regime import RegimeDetector

__all__ = [
    "RegimeDetector",
]
