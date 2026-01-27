"""
Portfolio Configuration Schema

Defines configuration for automated portfolio management including:
- Symbol-level settings (max positions, risk budget, correlation groups)
- Portfolio-level limits
- Execution mode (full_auto, semi_auto, paper)
- Scheduling settings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path
import yaml
import json


class ExecutionMode(Enum):
    """Trade execution mode."""
    FULL_AUTO = "full_auto"      # Execute trades without confirmation
    SEMI_AUTO = "semi_auto"      # Generate signals, require confirmation
    PAPER = "paper"              # Log decisions without executing


@dataclass
class SymbolConfig:
    """Configuration for a single trading symbol."""
    symbol: str                             # e.g., "XAUUSD"
    max_positions: int = 1                  # Max concurrent positions for this symbol
    risk_budget_pct: float = 2.0            # Max risk % per trade for this symbol
    correlation_group: str = "default"      # For correlation-aware limits
    timeframes: List[str] = field(default_factory=lambda: ["1H", "4H", "D1"])
    enabled: bool = True                    # Can be temporarily disabled

    # Optional symbol-specific overrides
    min_confidence: float = 0.6             # Minimum confidence to trade
    max_spread_pct: float = 0.1             # Max spread as % of price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "max_positions": self.max_positions,
            "risk_budget_pct": self.risk_budget_pct,
            "correlation_group": self.correlation_group,
            "timeframes": self.timeframes,
            "enabled": self.enabled,
            "min_confidence": self.min_confidence,
            "max_spread_pct": self.max_spread_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolConfig":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            max_positions=data.get("max_positions", 1),
            risk_budget_pct=data.get("risk_budget_pct", 2.0),
            correlation_group=data.get("correlation_group", "default"),
            timeframes=data.get("timeframes", ["1H", "4H", "D1"]),
            enabled=data.get("enabled", True),
            min_confidence=data.get("min_confidence", 0.6),
            max_spread_pct=data.get("max_spread_pct", 0.1),
        )


@dataclass
class ScheduleConfig:
    """Daily workflow schedule configuration."""
    morning_analysis_hour: int = 8          # Hour to run morning analysis (24h format)
    midday_review_hour: int = 13            # Hour to review positions
    evening_reflect_hour: int = 20          # Hour to process closed trades
    timezone: str = "UTC"                   # Timezone for scheduling

    # Optional: specific minutes (default: 0)
    morning_analysis_minute: int = 0
    midday_review_minute: int = 0
    evening_reflect_minute: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "morning_analysis_hour": self.morning_analysis_hour,
            "midday_review_hour": self.midday_review_hour,
            "evening_reflect_hour": self.evening_reflect_hour,
            "timezone": self.timezone,
            "morning_analysis_minute": self.morning_analysis_minute,
            "midday_review_minute": self.midday_review_minute,
            "evening_reflect_minute": self.evening_reflect_minute,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleConfig":
        """Create from dictionary."""
        return cls(
            morning_analysis_hour=data.get("morning_analysis_hour", 8),
            midday_review_hour=data.get("midday_review_hour", 13),
            evening_reflect_hour=data.get("evening_reflect_hour", 20),
            timezone=data.get("timezone", "UTC"),
            morning_analysis_minute=data.get("morning_analysis_minute", 0),
            midday_review_minute=data.get("midday_review_minute", 0),
            evening_reflect_minute=data.get("evening_reflect_minute", 0),
        )


@dataclass
class PortfolioConfig:
    """Master portfolio configuration."""

    # Symbols to trade
    symbols: List[SymbolConfig] = field(default_factory=list)

    # Portfolio-level limits
    max_total_positions: int = 5            # Max positions across all symbols
    max_daily_trades: int = 3               # Max new trades per day
    max_correlation_group_positions: int = 2 # Max positions in same correlation group

    # Risk settings
    total_risk_budget_pct: float = 6.0      # Max total portfolio risk %
    daily_loss_limit_pct: float = 3.0       # Daily loss circuit breaker
    max_consecutive_losses: int = 2         # Consecutive loss circuit breaker
    cooldown_hours: int = 24                # Cooldown after circuit breaker trip

    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.FULL_AUTO
    use_staged_entry: bool = False          # Multi-tranche entries (not yet implemented)
    default_lot_size: float = 0.01          # Default position size
    use_atr_stops: bool = True              # Use ATR-based stops
    atr_stop_multiplier: float = 2.0        # ATR multiplier for stops
    atr_trailing_multiplier: float = 1.5    # ATR multiplier for trailing stops
    risk_reward_ratio: float = 2.0          # Default R:R ratio

    # Schedule
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    # LLM/Analysis settings (overrides for DEFAULT_CONFIG)
    llm_provider: str = "xai"
    deep_think_llm: str = "grok-3-fast"
    quick_think_llm: str = "grok-3-fast"
    max_debate_rounds: int = 1

    # Data settings
    asset_type: str = "commodity"

    # Persistence
    state_file: str = "portfolio_state.json"
    decisions_dir: str = "examples/trade_decisions"
    logs_dir: str = "logs/portfolio"

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if not self.symbols:
            errors.append("At least one symbol must be configured")

        if self.max_total_positions < 1:
            errors.append("max_total_positions must be >= 1")

        if self.max_daily_trades < 1:
            errors.append("max_daily_trades must be >= 1")

        if self.max_correlation_group_positions < 1:
            errors.append("max_correlation_group_positions must be >= 1")

        if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 100:
            errors.append("daily_loss_limit_pct must be between 0 and 100")

        if self.total_risk_budget_pct <= 0 or self.total_risk_budget_pct > 100:
            errors.append("total_risk_budget_pct must be between 0 and 100")

        # Check symbol configs
        symbols_seen = set()
        for sym_config in self.symbols:
            if sym_config.symbol in symbols_seen:
                errors.append(f"Duplicate symbol: {sym_config.symbol}")
            symbols_seen.add(sym_config.symbol)

            if sym_config.risk_budget_pct <= 0:
                errors.append(f"{sym_config.symbol}: risk_budget_pct must be > 0")

            if sym_config.max_positions < 1:
                errors.append(f"{sym_config.symbol}: max_positions must be >= 1")

        # Check total symbol risk vs portfolio limit
        total_symbol_risk = sum(s.risk_budget_pct for s in self.symbols if s.enabled)
        if total_symbol_risk > self.total_risk_budget_pct * 2:
            errors.append(
                f"Sum of symbol risks ({total_symbol_risk}%) significantly exceeds "
                f"portfolio limit ({self.total_risk_budget_pct}%)"
            )

        # Validate schedule
        if not 0 <= self.schedule.morning_analysis_hour <= 23:
            errors.append("morning_analysis_hour must be between 0 and 23")
        if not 0 <= self.schedule.midday_review_hour <= 23:
            errors.append("midday_review_hour must be between 0 and 23")
        if not 0 <= self.schedule.evening_reflect_hour <= 23:
            errors.append("evening_reflect_hour must be between 0 and 23")

        return errors

    def get_enabled_symbols(self) -> List[SymbolConfig]:
        """Get list of enabled symbols."""
        return [s for s in self.symbols if s.enabled]

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get config for a specific symbol."""
        for s in self.symbols:
            if s.symbol == symbol:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbols": [s.to_dict() for s in self.symbols],
            "max_total_positions": self.max_total_positions,
            "max_daily_trades": self.max_daily_trades,
            "max_correlation_group_positions": self.max_correlation_group_positions,
            "total_risk_budget_pct": self.total_risk_budget_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "cooldown_hours": self.cooldown_hours,
            "execution_mode": self.execution_mode.value,
            "use_staged_entry": self.use_staged_entry,
            "default_lot_size": self.default_lot_size,
            "use_atr_stops": self.use_atr_stops,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "atr_trailing_multiplier": self.atr_trailing_multiplier,
            "risk_reward_ratio": self.risk_reward_ratio,
            "schedule": self.schedule.to_dict(),
            "llm_provider": self.llm_provider,
            "deep_think_llm": self.deep_think_llm,
            "quick_think_llm": self.quick_think_llm,
            "max_debate_rounds": self.max_debate_rounds,
            "asset_type": self.asset_type,
            "state_file": self.state_file,
            "decisions_dir": self.decisions_dir,
            "logs_dir": self.logs_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConfig":
        """Create from dictionary."""
        symbols = [SymbolConfig.from_dict(s) for s in data.get("symbols", [])]
        schedule = ScheduleConfig.from_dict(data.get("schedule", {}))

        execution_mode_str = data.get("execution_mode", "full_auto")
        execution_mode = ExecutionMode(execution_mode_str)

        return cls(
            symbols=symbols,
            max_total_positions=data.get("max_total_positions", 5),
            max_daily_trades=data.get("max_daily_trades", 3),
            max_correlation_group_positions=data.get("max_correlation_group_positions", 2),
            total_risk_budget_pct=data.get("total_risk_budget_pct", 6.0),
            daily_loss_limit_pct=data.get("daily_loss_limit_pct", 3.0),
            max_consecutive_losses=data.get("max_consecutive_losses", 2),
            cooldown_hours=data.get("cooldown_hours", 24),
            execution_mode=execution_mode,
            use_staged_entry=data.get("use_staged_entry", False),
            default_lot_size=data.get("default_lot_size", 0.01),
            use_atr_stops=data.get("use_atr_stops", True),
            atr_stop_multiplier=data.get("atr_stop_multiplier", 2.0),
            atr_trailing_multiplier=data.get("atr_trailing_multiplier", 1.5),
            risk_reward_ratio=data.get("risk_reward_ratio", 2.0),
            schedule=schedule,
            llm_provider=data.get("llm_provider", "xai"),
            deep_think_llm=data.get("deep_think_llm", "grok-3-fast"),
            quick_think_llm=data.get("quick_think_llm", "grok-3-fast"),
            max_debate_rounds=data.get("max_debate_rounds", 1),
            asset_type=data.get("asset_type", "commodity"),
            state_file=data.get("state_file", "portfolio_state.json"),
            decisions_dir=data.get("decisions_dir", "examples/trade_decisions"),
            logs_dir=data.get("logs_dir", "logs/portfolio"),
        )


def get_default_config() -> PortfolioConfig:
    """Get default portfolio configuration with XAUUSD, XAGUSD, COPPER-C."""
    return PortfolioConfig(
        symbols=[
            SymbolConfig(
                symbol="XAUUSD",
                max_positions=1,
                risk_budget_pct=2.0,
                correlation_group="metals",
                timeframes=["1H", "4H", "D1"],
            ),
            SymbolConfig(
                symbol="XAGUSD",
                max_positions=1,
                risk_budget_pct=1.5,
                correlation_group="metals",
                timeframes=["1H", "4H", "D1"],
            ),
            SymbolConfig(
                symbol="COPPER-C",
                max_positions=1,
                risk_budget_pct=1.5,
                correlation_group="industrial_metals",
                timeframes=["1H", "4H", "D1"],
            ),
        ],
        max_total_positions=4,
        max_daily_trades=3,
        max_correlation_group_positions=2,
        execution_mode=ExecutionMode.FULL_AUTO,
    )


def load_portfolio_config(config_path: str) -> PortfolioConfig:
    """
    Load portfolio configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        PortfolioConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    return PortfolioConfig.from_dict(data)


def save_portfolio_config(config: PortfolioConfig, config_path: str) -> None:
    """
    Save portfolio configuration to YAML or JSON file.

    Args:
        config: PortfolioConfig instance
        config_path: Path to save configuration
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(path, "w") as f:
        if path.suffix in [".yaml", ".yml"]:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2)
