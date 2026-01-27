"""
Portfolio Automation CLI Commands

Commands for managing the automated portfolio system:
- portfolio start: Start the automation daemon
- portfolio status: View current status
- portfolio trigger: Manually trigger a cycle
- portfolio stop: Stop the daemon
- portfolio config: View/edit configuration
"""

import asyncio
import os
import sys
import signal
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Create the portfolio sub-app
portfolio_app = typer.Typer(
    name="portfolio",
    help="Automated portfolio management commands",
    no_args_is_help=True,
)

console = Console()


def _get_automation():
    """Get portfolio automation instance."""
    from tradingagents.automation import (
        PortfolioAutomation,
        PortfolioConfig,
        load_portfolio_config,
        get_default_config,
    )

    config_file = Path("portfolio_config.yaml")
    if config_file.exists():
        config = load_portfolio_config(str(config_file))
    else:
        config = get_default_config()

    return PortfolioAutomation(config)


@portfolio_app.command("start")
def start_portfolio(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to portfolio configuration file"
    ),
    daemon: bool = typer.Option(
        False, "--daemon", "-d",
        help="Run in background as daemon"
    ),
):
    """Start the automated portfolio manager."""
    from tradingagents.automation import (
        PortfolioAutomation,
        DailyScheduler,
        load_portfolio_config,
        get_default_config,
    )

    console.print("[bold green]Starting Portfolio Automation...[/bold green]")

    # Load configuration
    if config_file and Path(config_file).exists():
        config = load_portfolio_config(config_file)
        console.print(f"Loaded config from: {config_file}")
    elif Path("portfolio_config.yaml").exists():
        config = load_portfolio_config("portfolio_config.yaml")
        console.print("Loaded config from: portfolio_config.yaml")
    else:
        config = get_default_config()
        console.print("Using default configuration")

    # Validate configuration
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)

    # Display configuration summary
    table = Table(title="Portfolio Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Execution Mode", config.execution_mode.value)
    table.add_row("Symbols", ", ".join(s.symbol for s in config.get_enabled_symbols()))
    table.add_row("Max Positions", str(config.max_total_positions))
    table.add_row("Max Daily Trades", str(config.max_daily_trades))
    table.add_row("Daily Loss Limit", f"{config.daily_loss_limit_pct}%")
    table.add_row("Morning Analysis", f"{config.schedule.morning_analysis_hour:02d}:00")
    table.add_row("Midday Review", f"{config.schedule.midday_review_hour:02d}:00")
    table.add_row("Evening Reflect", f"{config.schedule.evening_reflect_hour:02d}:00")

    console.print(table)

    # Initialize automation
    try:
        automation = PortfolioAutomation(config)
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        raise typer.Exit(1)

    # Initialize scheduler
    scheduler = DailyScheduler(automation)

    if daemon:
        console.print("[yellow]Daemon mode not yet implemented on Windows.[/yellow]")
        console.print("Running in foreground...")

    console.print("")
    console.print("[bold]Press Ctrl+C to stop[/bold]")
    console.print("")

    # Run scheduler
    try:
        asyncio.run(scheduler.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.stop()


@portfolio_app.command("status")
def portfolio_status():
    """View current portfolio automation status."""
    try:
        automation = _get_automation()
        status = automation.get_status()

        # Portfolio panel
        portfolio_info = [
            f"Open Positions: {status['portfolio']['open_positions']}/{status['config']['max_positions']}",
            f"Balance: ${status['portfolio']['account_balance']:,.2f}",
            f"Equity: ${status['portfolio']['account_equity']:,.2f}",
        ]
        console.print(Panel("\n".join(portfolio_info), title="Portfolio Status"))

        # Config panel
        config_info = [
            f"Execution Mode: {status['config']['execution_mode']}",
            f"Symbols: {', '.join(status['config']['symbols'])}",
            f"Trades Today: {status['today']['trades_executed']}/{status['today']['max_trades']}",
        ]
        console.print(Panel("\n".join(config_info), title="Configuration"))

        # Guardrails panel
        guardrails = status['guardrails']
        can_trade_str = "[green]Yes[/green]" if guardrails['can_trade'] else "[red]No[/red]"
        # Clean up status summary - remove emojis for Windows console compatibility
        status_summary = guardrails.get('status_summary', 'OK')
        # Replace common emojis with text equivalents
        status_summary = status_summary.replace('\u2705', '[OK]').replace('\u26d4', '[STOP]')
        status_summary = status_summary.replace('\u23f0', '[CLOCK]').replace('\ud83d\udcc9', '[DOWN]')
        status_summary = status_summary.replace('\ud83d\udcca', '[CHART]')
        guardrails_info = f"Can Trade: {can_trade_str}\nStatus: {status_summary}"
        console.print(Panel(guardrails_info, title="Risk Guardrails"))

        # Correlation warnings
        if status['correlation_warnings']:
            console.print("[yellow]Correlation Warnings:[/yellow]")
            for warning in status['correlation_warnings']:
                console.print(f"  - {warning}")

    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")
        raise typer.Exit(1)


@portfolio_app.command("trigger")
def trigger_cycle(
    cycle: str = typer.Argument(
        ...,
        help="Cycle to trigger: morning, midday, or evening"
    ),
):
    """Manually trigger a workflow cycle."""
    if cycle not in ["morning", "midday", "evening"]:
        console.print(f"[red]Invalid cycle: {cycle}[/red]")
        console.print("Valid options: morning, midday, evening")
        raise typer.Exit(1)

    console.print(f"[bold]Triggering {cycle} cycle...[/bold]")

    try:
        from tradingagents.automation import DailyScheduler

        automation = _get_automation()
        scheduler = DailyScheduler(automation)

        # Run the cycle
        report = asyncio.run(scheduler.trigger_manual(cycle))

        # Display report
        console.print("")
        console.print(report.format_summary())

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@portfolio_app.command("stop")
def stop_daemon():
    """Stop the portfolio automation daemon."""
    pid_file = Path("portfolio_scheduler.pid")

    if not pid_file.exists():
        console.print("[yellow]No running daemon found[/yellow]")
        return

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        console.print(f"Stopping daemon (PID: {pid})...")

        if sys.platform == "win32":
            # Windows
            import subprocess
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            # Unix
            os.kill(pid, signal.SIGTERM)

        # Clean up PID file
        pid_file.unlink()
        console.print("[green]Daemon stopped[/green]")

    except ProcessLookupError:
        console.print("[yellow]Process not running, cleaning up PID file[/yellow]")
        pid_file.unlink()
    except Exception as e:
        console.print(f"[red]Error stopping daemon: {e}[/red]")


@portfolio_app.command("config")
def show_config(
    create: bool = typer.Option(
        False, "--create",
        help="Create a default configuration file"
    ),
):
    """Display or create portfolio configuration."""
    from tradingagents.automation import (
        get_default_config,
        save_portfolio_config,
        load_portfolio_config,
    )

    config_file = Path("portfolio_config.yaml")

    if create:
        if config_file.exists():
            if not typer.confirm("Configuration file exists. Overwrite?"):
                raise typer.Exit(0)

        config = get_default_config()
        save_portfolio_config(config, str(config_file))
        console.print(f"[green]Created: {config_file}[/green]")
        console.print("Edit this file to customize your portfolio settings.")
        return

    # Display current configuration
    if config_file.exists():
        config = load_portfolio_config(str(config_file))
        console.print(f"[cyan]Configuration from: {config_file}[/cyan]")
    else:
        config = get_default_config()
        console.print("[yellow]Using default configuration[/yellow]")
        console.print("Run 'portfolio config --create' to create a config file.")

    # Display as table
    table = Table(title="Portfolio Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Execution Mode", config.execution_mode.value)
    table.add_row("Max Total Positions", str(config.max_total_positions))
    table.add_row("Max Daily Trades", str(config.max_daily_trades))
    table.add_row("Max Correlation Group", str(config.max_correlation_group_positions))
    table.add_row("Daily Loss Limit", f"{config.daily_loss_limit_pct}%")
    table.add_row("Max Consecutive Losses", str(config.max_consecutive_losses))
    table.add_row("ATR Stop Multiplier", str(config.atr_stop_multiplier))
    table.add_row("Risk Reward Ratio", str(config.risk_reward_ratio))

    console.print(table)

    # Symbols table
    symbols_table = Table(title="Configured Symbols")
    symbols_table.add_column("Symbol", style="cyan")
    symbols_table.add_column("Max Pos", style="green")
    symbols_table.add_column("Risk %", style="green")
    symbols_table.add_column("Correlation Group", style="yellow")
    symbols_table.add_column("Enabled", style="green")

    for s in config.symbols:
        symbols_table.add_row(
            s.symbol,
            str(s.max_positions),
            f"{s.risk_budget_pct}%",
            s.correlation_group,
            "Yes" if s.enabled else "No",
        )

    console.print(symbols_table)

    # Schedule table
    schedule_table = Table(title="Schedule")
    schedule_table.add_column("Cycle", style="cyan")
    schedule_table.add_column("Time", style="green")

    schedule_table.add_row(
        "Morning Analysis",
        f"{config.schedule.morning_analysis_hour:02d}:{config.schedule.morning_analysis_minute:02d}"
    )
    schedule_table.add_row(
        "Midday Review",
        f"{config.schedule.midday_review_hour:02d}:{config.schedule.midday_review_minute:02d}"
    )
    schedule_table.add_row(
        "Evening Reflect",
        f"{config.schedule.evening_reflect_hour:02d}:{config.schedule.evening_reflect_minute:02d}"
    )
    schedule_table.add_row("Timezone", config.schedule.timezone)

    console.print(schedule_table)


@portfolio_app.command("run-morning")
def run_morning():
    """Run morning analysis cycle now."""
    console.print("[bold]Running Morning Analysis...[/bold]")

    try:
        automation = _get_automation()
        report = asyncio.run(automation.run_morning_analysis())
        console.print("")
        console.print(report.format_summary())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@portfolio_app.command("run-review")
def run_review():
    """Run midday position review now."""
    console.print("[bold]Running Midday Review...[/bold]")

    try:
        automation = _get_automation()
        report = asyncio.run(automation.run_midday_review())
        console.print("")
        console.print(report.format_summary())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@portfolio_app.command("run-reflect")
def run_reflect():
    """Run evening reflection cycle now."""
    console.print("[bold]Running Evening Reflection...[/bold]")

    try:
        automation = _get_automation()
        report = asyncio.run(automation.run_evening_reflect())
        console.print("")
        console.print(report.format_summary())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Function to get the app for registration in main.py
def get_portfolio_app():
    """Return the portfolio typer app for registration."""
    return portfolio_app
