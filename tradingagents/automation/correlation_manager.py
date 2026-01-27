"""
Correlation Manager

Manages correlation-aware position limits to prevent over-allocation
to correlated assets (e.g., Gold and Silver both in "metals" group).
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .portfolio_config import PortfolioConfig, SymbolConfig


@dataclass
class CorrelationGroupStatus:
    """Status of a correlation group."""
    group_name: str
    symbols: List[str]
    current_positions: int
    max_positions: int
    can_open_new: bool
    symbols_with_positions: List[str]


class CorrelationManager:
    """
    Manages correlation-aware position limits.

    Prevents over-allocation to correlated assets:
    - Metals (XAUUSD, XAGUSD, XPTUSD) typically correlate
    - Industrial metals (COPPER-C) may correlate differently
    - USD pairs often move together

    Example:
        If max_correlation_group_positions=2 and we have positions in
        both XAUUSD and XAGUSD (both in "metals" group), we cannot
        open another metals position.
    """

    # Default correlation groups for common assets
    DEFAULT_GROUPS = {
        "metals": ["XAUUSD", "XAGUSD", "XPTUSD"],
        "industrial_metals": ["COPPER-C", "ALUMINUM", "ZINC"],
        "energy": ["USOIL", "UKOIL", "NATGAS"],
        "usd_majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        "risk_on": ["AUDUSD", "NZDUSD"],
        "crypto": ["BTCUSD", "ETHUSD"],
    }

    def __init__(self, config: PortfolioConfig):
        """
        Initialize correlation manager.

        Args:
            config: Portfolio configuration with symbol configs
        """
        self.config = config
        self.max_group_positions = config.max_correlation_group_positions

        # Build symbol-to-group mapping from config
        self._groups: Dict[str, List[str]] = {}
        self._symbol_to_group: Dict[str, str] = {}

        self._build_groups()

    def _build_groups(self) -> None:
        """Build symbol-to-group mapping from config."""
        for symbol_config in self.config.symbols:
            group = symbol_config.correlation_group
            symbol = symbol_config.symbol

            if group not in self._groups:
                self._groups[group] = []

            self._groups[group].append(symbol)
            self._symbol_to_group[symbol] = group

    def get_group_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get correlation group for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Correlation group name or None if not configured
        """
        return self._symbol_to_group.get(symbol)

    def get_symbols_in_group(self, group: str) -> List[str]:
        """
        Get all symbols in a correlation group.

        Args:
            group: Correlation group name

        Returns:
            List of symbols in the group
        """
        return self._groups.get(group, [])

    def can_open_position(
        self,
        symbol: str,
        current_positions: List[Dict],
    ) -> bool:
        """
        Check if a new position can be opened for this symbol.

        Considers:
        - Correlation group limits
        - Total portfolio position limits

        Args:
            symbol: Symbol to open position for
            current_positions: List of current MT5 positions (with 'symbol' key)

        Returns:
            True if position can be opened
        """
        # Get correlation group for symbol
        group = self._symbol_to_group.get(symbol)

        if not group:
            # Symbol not in any configured group - allow
            return True

        # Count positions in same correlation group
        group_symbols = set(self._groups.get(group, []))
        group_positions = sum(
            1 for p in current_positions
            if p.get("symbol") in group_symbols
        )

        return group_positions < self.max_group_positions

    def get_group_status(
        self,
        group: str,
        current_positions: List[Dict],
    ) -> CorrelationGroupStatus:
        """
        Get detailed status of a correlation group.

        Args:
            group: Correlation group name
            current_positions: List of current MT5 positions

        Returns:
            CorrelationGroupStatus with details
        """
        group_symbols = self._groups.get(group, [])
        group_symbol_set = set(group_symbols)

        # Find which symbols have positions
        symbols_with_positions = [
            p.get("symbol") for p in current_positions
            if p.get("symbol") in group_symbol_set
        ]

        current_count = len(symbols_with_positions)
        can_open = current_count < self.max_group_positions

        return CorrelationGroupStatus(
            group_name=group,
            symbols=group_symbols,
            current_positions=current_count,
            max_positions=self.max_group_positions,
            can_open_new=can_open,
            symbols_with_positions=symbols_with_positions,
        )

    def get_all_group_statuses(
        self,
        current_positions: List[Dict],
    ) -> Dict[str, CorrelationGroupStatus]:
        """
        Get status of all correlation groups.

        Args:
            current_positions: List of current MT5 positions

        Returns:
            Dict mapping group name to status
        """
        return {
            group: self.get_group_status(group, current_positions)
            for group in self._groups
        }

    def get_available_symbols(
        self,
        current_positions: List[Dict],
    ) -> List[str]:
        """
        Get list of symbols that can accept new positions.

        Args:
            current_positions: List of current MT5 positions

        Returns:
            List of symbols that can be traded
        """
        available = []

        for symbol_config in self.config.symbols:
            if not symbol_config.enabled:
                continue

            symbol = symbol_config.symbol

            # Check symbol-level limit
            symbol_positions = sum(
                1 for p in current_positions
                if p.get("symbol") == symbol
            )
            if symbol_positions >= symbol_config.max_positions:
                continue

            # Check correlation group limit
            if not self.can_open_position(symbol, current_positions):
                continue

            available.append(symbol)

        return available

    def get_group_exposure(
        self,
        current_positions: List[Dict],
    ) -> Dict[str, float]:
        """
        Get total exposure (value) per correlation group.

        Args:
            current_positions: List of current MT5 positions
                              (with 'symbol', 'volume', 'price_current' keys)

        Returns:
            Dict mapping group name to total exposure value
        """
        exposure = {}

        for group, symbols in self._groups.items():
            symbol_set = set(symbols)
            group_value = sum(
                p.get("volume", 0) * p.get("price_current", 0)
                for p in current_positions
                if p.get("symbol") in symbol_set
            )
            exposure[group] = group_value

        return exposure

    def get_correlation_warnings(
        self,
        current_positions: List[Dict],
    ) -> List[str]:
        """
        Get warnings about correlation group exposure.

        Args:
            current_positions: List of current MT5 positions

        Returns:
            List of warning messages
        """
        warnings = []

        for group, status in self.get_all_group_statuses(current_positions).items():
            if status.current_positions >= status.max_positions:
                warnings.append(
                    f"Correlation group '{group}' at max positions "
                    f"({status.current_positions}/{status.max_positions}): "
                    f"{', '.join(status.symbols_with_positions)}"
                )
            elif status.current_positions == status.max_positions - 1:
                warnings.append(
                    f"Correlation group '{group}' near limit "
                    f"({status.current_positions}/{status.max_positions})"
                )

        return warnings

    def suggest_diversification(
        self,
        current_positions: List[Dict],
    ) -> List[str]:
        """
        Suggest symbols for better diversification.

        Returns symbols from groups with no current positions.

        Args:
            current_positions: List of current MT5 positions

        Returns:
            List of suggested symbols
        """
        suggestions = []

        for group, symbols in self._groups.items():
            symbol_set = set(symbols)
            has_position = any(
                p.get("symbol") in symbol_set
                for p in current_positions
            )

            if not has_position:
                # No position in this group - suggest enabled symbols
                for sym in symbols:
                    sym_config = self.config.get_symbol_config(sym)
                    if sym_config and sym_config.enabled:
                        suggestions.append(sym)
                        break  # One suggestion per group

        return suggestions
