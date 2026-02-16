"""
TradingAgents Automation Package

Provides automated portfolio management with:
- Multi-symbol analysis cycles
- Automatic trade execution
- Position review and adjustment
- Learning from outcomes
"""

from .portfolio_config import (
    PortfolioConfig,
    SymbolConfig,
    ScheduleConfig,
    ExecutionMode,
    load_portfolio_config,
    save_portfolio_config,
    get_default_config,
)
from .correlation_manager import CorrelationManager
from .reporting import (
    AnalysisResult,
    PositionAdjustment,
    DailyAnalysisReport,
    PositionReviewReport,
    ReflectionReport,
)
from .portfolio_automation import PortfolioAutomation, run_deep_position_analysis
from .daily_scheduler import DailyScheduler
from .quant_automation import (
    QuantAutomation,
    QuantAutomationConfig,
    PipelineType,
    AutomationStatus,
    AnalysisCycleResult,
    PositionManagementResult,
    get_automation_instance,
    start_automation,
    stop_automation,
)

__all__ = [
    # Config
    "PortfolioConfig",
    "SymbolConfig",
    "ScheduleConfig",
    "ExecutionMode",
    "load_portfolio_config",
    "save_portfolio_config",
    "get_default_config",
    # Managers
    "CorrelationManager",
    "PortfolioAutomation",
    "DailyScheduler",
    # Reporting
    "AnalysisResult",
    "PositionAdjustment",
    "DailyAnalysisReport",
    "PositionReviewReport",
    "ReflectionReport",
    # Analysis functions
    "run_deep_position_analysis",
    # Quant Automation
    "QuantAutomation",
    "QuantAutomationConfig",
    "PipelineType",
    "AutomationStatus",
    "AnalysisCycleResult",
    "PositionManagementResult",
    "get_automation_instance",
    "start_automation",
    "stop_automation",
]
