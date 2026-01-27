"""
CLI output formatting utilities.

Supports multiple output formats:
- rich: Rich formatted terminal output (default)
- json: Machine-readable JSON output
- table: Tabular format for data
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


class OutputFormat(str, Enum):
    """Output format options."""

    RICH = "rich"
    JSON = "json"
    TABLE = "table"


def output_json(data: Union[Dict, List, Any], indent: int = 2) -> None:
    """Output data as formatted JSON."""
    if hasattr(data, "model_dump"):
        # Pydantic model
        json_str = data.model_dump_json(indent=indent)
    elif hasattr(data, "dict"):
        # Pydantic v1 compatibility
        json_str = json.dumps(data.dict(), indent=indent, default=str)
    else:
        json_str = json.dumps(data, indent=indent, default=str)

    console.print_json(json_str)


def output_table(
    data: Union[Dict, List[Dict]],
    title: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> None:
    """Output data as a rich table."""
    table = Table(box=box.ROUNDED, title=title)

    if isinstance(data, dict):
        # Single dict - display as key-value pairs
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for key, value in data.items():
            table.add_row(str(key), str(value))
    elif isinstance(data, list) and len(data) > 0:
        # List of dicts - display as rows
        if columns:
            for col in columns:
                table.add_column(col)
        else:
            # Auto-detect columns from first item
            first = data[0]
            if isinstance(first, dict):
                for col in first.keys():
                    table.add_column(str(col))

        for item in data:
            if isinstance(item, dict):
                row = [str(item.get(col, "")) for col in (columns or item.keys())]
                table.add_row(*row)
    else:
        console.print("[yellow]No data to display[/yellow]")
        return

    console.print(table)


def format_position_review(review: Dict, output_format: OutputFormat = OutputFormat.RICH) -> None:
    """Format a position review result."""
    if output_format == OutputFormat.JSON:
        output_json(review)
        return

    if output_format == OutputFormat.TABLE:
        output_table(review, title="Position Review")
        return

    # Rich format
    from rich.panel import Panel

    recommendation = review.get("recommendation", "N/A")
    risk_level = review.get("risk_level", "N/A")
    reasoning = review.get("reasoning", "N/A")

    rec_color = {
        "HOLD": "yellow",
        "CLOSE": "red",
        "ADJUST": "cyan",
    }.get(recommendation, "white")

    risk_color = {
        "Low": "green",
        "Medium": "yellow",
        "High": "red",
        "Extreme": "bold red",
    }.get(risk_level, "white")

    content = f"""[bold]Recommendation:[/bold] [{rec_color}]{recommendation}[/{rec_color}]
[bold]Risk Level:[/bold] [{risk_color}]{risk_level}[/{risk_color}]
"""

    if review.get("suggested_sl"):
        content += f"[bold]Suggested SL:[/bold] {review['suggested_sl']}\n"
    if review.get("suggested_tp"):
        content += f"[bold]Suggested TP:[/bold] {review['suggested_tp']}\n"

    content += f"\n[bold]Reasoning:[/bold]\n{reasoning}"

    console.print(Panel(content, title="Position Review", border_style="cyan"))


def format_trade_analysis(analysis: Dict, output_format: OutputFormat = OutputFormat.RICH) -> None:
    """Format a trade analysis result."""
    if output_format == OutputFormat.JSON:
        output_json(analysis)
        return

    if output_format == OutputFormat.TABLE:
        # Flatten nested structures for table display
        flat = {
            "symbol": analysis.get("symbol"),
            "signal": analysis.get("signal"),
            "confidence": f"{analysis.get('confidence', 0) * 100:.1f}%",
            "entry_price": analysis.get("entry_price"),
            "stop_loss": analysis.get("stop_loss"),
            "take_profit": analysis.get("take_profit"),
            "risk_level": analysis.get("risk_level"),
            "risk_reward": analysis.get("risk_reward_ratio"),
        }
        output_table(flat, title=f"Trade Analysis: {analysis.get('symbol', 'N/A')}")
        return

    # Rich format
    from rich.panel import Panel
    from rich.markdown import Markdown

    signal = analysis.get("signal", "N/A")
    signal_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(signal, "white")

    content = f"""## Signal: [{signal_color}]{signal}[/{signal_color}]
**Confidence:** {analysis.get('confidence', 0) * 100:.1f}%
**Risk Level:** {analysis.get('risk_level', 'N/A')}

### Trade Parameters
- Entry: {analysis.get('entry_price', 'N/A')}
- Stop Loss: {analysis.get('stop_loss', 'N/A')}
- Take Profit: {analysis.get('take_profit', 'N/A')}
- Risk/Reward: {analysis.get('risk_reward_ratio', 'N/A')}

### Rationale
{analysis.get('rationale', 'No rationale provided')}
"""

    console.print(Panel(Markdown(content), title=f"Analysis: {analysis.get('symbol', 'N/A')}", border_style="cyan"))


def format_portfolio_suggestion(suggestion: Dict, output_format: OutputFormat = OutputFormat.RICH) -> None:
    """Format a portfolio suggestion result."""
    if output_format == OutputFormat.JSON:
        output_json(suggestion)
        return

    if output_format == OutputFormat.TABLE:
        suggestions = suggestion.get("suggestions", [])
        if suggestions:
            output_table(
                suggestions,
                title="Portfolio Suggestions",
                columns=["symbol", "reason", "correlation_group", "priority"],
            )
        return

    # Rich format
    from rich.panel import Panel

    content = f"**Portfolio Analysis:**\n{suggestion.get('portfolio_analysis', 'N/A')}\n\n"

    suggestions = suggestion.get("suggestions", [])
    if suggestions:
        content += "**Suggestions:**\n"
        for s in suggestions:
            priority_color = {"high": "red", "medium": "yellow", "low": "green"}.get(
                s.get("priority", ""), "white"
            )
            content += f"- [{priority_color}][{s.get('priority', '?').upper()}][/{priority_color}] **{s.get('symbol', '?')}**: {s.get('reason', 'N/A')} ({s.get('correlation_group', 'N/A')})\n"

    content += f"\n**Risk Notes:**\n{suggestion.get('risk_notes', 'N/A')}"

    console.print(Panel(content, title="Portfolio Suggestions", border_style="cyan"))
