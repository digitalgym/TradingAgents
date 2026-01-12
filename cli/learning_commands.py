"""
CLI commands for continuous learning system

New commands:
- learning-status: View system status
- update-patterns: Run pattern analysis and update weights
- risk-status: View risk guardrails status
- regime: Detect current market regime
- similar-trades: Find similar historical trades
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional
import numpy as np

console = Console()


def learning_status_command():
    """Display continuous learning system status."""
    from tradingagents.risk import RiskGuardrails
    from tradingagents.learning import OnlineRLUpdater, PatternAnalyzer
    from tradingagents.learning.portfolio_state import PortfolioStateTracker
    
    console.print("\n[bold cyan]═══ CONTINUOUS LEARNING SYSTEM STATUS ═══[/bold cyan]\n")
    
    # ========================================================================
    # Risk Guardrails
    # ========================================================================
    console.print("[bold yellow]Risk Guardrails[/bold yellow]")
    
    guardrails = RiskGuardrails()
    status = guardrails.get_status()
    
    risk_table = Table(box=box.ROUNDED)
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", justify="right")
    risk_table.add_column("Limit", justify="right")
    risk_table.add_column("Status")
    
    # Trading allowed
    can_trade_icon = "✅" if status['can_trade'] else "⛔"
    risk_table.add_row(
        "Trading Allowed",
        can_trade_icon,
        "-",
        status['reason']
    )
    
    # Consecutive losses
    consec = status['consecutive_losses']
    consec_limit = guardrails.max_consecutive_losses
    consec_status = "🔥" if consec >= consec_limit else "✓" if consec == 0 else "⚠️"
    risk_table.add_row(
        "Consecutive Losses",
        str(consec),
        str(consec_limit),
        consec_status
    )
    
    # Daily loss
    daily = status['daily_loss_pct']
    daily_limit = guardrails.daily_loss_limit_pct
    daily_status = "🔥" if daily >= daily_limit else "✓" if daily == 0 else "⚠️"
    risk_table.add_row(
        "Daily Loss",
        f"{daily:.2f}%",
        f"{daily_limit:.2f}%",
        daily_status
    )
    
    # Total breaches
    risk_table.add_row(
        "Total Breaches",
        str(status['total_breaches']),
        "-",
        "📊"
    )
    
    console.print(risk_table)
    
    # ========================================================================
    # Agent Weights
    # ========================================================================
    console.print("\n[bold yellow]Agent Weights (Adaptive)[/bold yellow]")
    
    updater = OnlineRLUpdater()
    weights = updater.get_current_weights()
    
    weight_table = Table(box=box.ROUNDED)
    weight_table.add_column("Agent", style="cyan")
    weight_table.add_column("Weight", justify="right")
    weight_table.add_column("Status")
    weight_table.add_column("Influence")
    
    for agent, weight in weights.items():
        # Status indicator
        if weight > 0.4:
            status_icon = "🔥"
            influence = "High"
        elif weight < 0.25:
            status_icon = "❄️"
            influence = "Low"
        else:
            status_icon = "→"
            influence = "Normal"
        
        weight_table.add_row(
            agent.capitalize(),
            f"{weight:.3f}",
            status_icon,
            influence
        )
    
    console.print(weight_table)
    
    # ========================================================================
    # Pattern Analysis Status
    # ========================================================================
    console.print("\n[bold yellow]Pattern Analysis[/bold yellow]")
    
    should_update, trades_since = updater.should_update()
    
    pattern_table = Table(box=box.ROUNDED)
    pattern_table.add_column("Metric", style="cyan")
    pattern_table.add_column("Value", justify="right")
    
    pattern_table.add_row("Trades Since Update", f"{trades_since}/30")
    pattern_table.add_row("Update Needed", "Yes ⚠️" if should_update else "No ✓")
    
    console.print(pattern_table)
    
    if should_update:
        console.print("\n[yellow]⚠️  Pattern analysis due - run: tradingagents update-patterns[/yellow]")
    
    # ========================================================================
    # Portfolio State
    # ========================================================================
    console.print("\n[bold yellow]Portfolio Performance[/bold yellow]")
    
    try:
        portfolio = PortfolioStateTracker.load_state()
        stats = portfolio.get_statistics()
        
        perf_table = Table(box=box.ROUNDED)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", justify="right")
        
        perf_table.add_row("Total Trades", str(stats['total_trades']))
        
        wr = stats['win_rate'] * 100
        wr_color = "green" if wr >= 50 else "red"
        perf_table.add_row("Win Rate", f"[{wr_color}]{wr:.1f}%[/{wr_color}]")
        
        perf_table.add_row("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
        
        dd = stats['max_drawdown'] * 100
        dd_color = "green" if dd < 10 else "yellow" if dd < 20 else "red"
        perf_table.add_row("Max Drawdown", f"[{dd_color}]{dd:.2f}%[/{dd_color}]")
        
        perf_table.add_row("Current Equity", f"${stats['current_equity']:.2f}")
        
        console.print(perf_table)
    except Exception as e:
        console.print(f"[dim]Portfolio data not available: {e}[/dim]")
    
    console.print()


def update_patterns_command():
    """Run pattern analysis and update agent weights."""
    from tradingagents.learning import PatternAnalyzer, OnlineRLUpdater
    
    console.print("\n[bold cyan]═══ PATTERN ANALYSIS & WEIGHT UPDATE ═══[/bold cyan]\n")
    
    # Check if update needed
    updater = OnlineRLUpdater()
    should_update, trades_since = updater.should_update()
    
    if not should_update:
        console.print(f"[yellow]Pattern analysis not due yet ({trades_since}/30 trades)[/yellow]")
        console.print("[dim]Run anyway? This will update weights based on current data.[/dim]")
        
        import questionary
        if not questionary.confirm("Continue?").ask():
            return
    
    # ========================================================================
    # Pattern Analysis
    # ========================================================================
    console.print("[bold yellow]Analyzing Trade Patterns...[/bold yellow]\n")
    
    with console.status("[cyan]Analyzing patterns..."):
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_patterns(lookback_days=30, min_cluster_size=2)
    
    stats = analysis['statistics']
    
    # Overall stats
    console.print(f"[cyan]Total Trades Analyzed: {stats['total_trades']}[/cyan]")
    console.print(f"[cyan]Overall Win Rate: {stats['overall_win_rate']*100:.1f}%[/cyan]")
    console.print(f"[cyan]Overall Avg RR: {stats['overall_avg_rr']:.2f}[/cyan]")
    console.print(f"[cyan]Patterns Found: {stats['patterns_found']}[/cyan]")
    
    # Pattern quality distribution
    console.print(f"\n[bold]Pattern Quality:[/bold]")
    console.print(f"  🌟 Excellent: {stats['excellent_patterns']}")
    console.print(f"  ✓ Good: {stats['good_patterns']}")
    console.print(f"  → Neutral: {stats['neutral_patterns']}")
    console.print(f"  ✗ Poor: {stats['poor_patterns']}")
    
    # Top patterns
    if analysis['patterns']:
        console.print(f"\n[bold yellow]Top 5 Patterns:[/bold yellow]")
        
        pattern_table = Table(box=box.ROUNDED)
        pattern_table.add_column("Pattern", style="cyan")
        pattern_table.add_column("Type")
        pattern_table.add_column("Win Rate", justify="right")
        pattern_table.add_column("Avg RR", justify="right")
        pattern_table.add_column("Sample", justify="right")
        pattern_table.add_column("Quality")
        
        for pattern in analysis['patterns'][:5]:
            quality_emoji = {
                "excellent": "🌟",
                "good": "✓",
                "neutral": "→",
                "poor": "✗"
            }.get(pattern['quality'], "?")
            
            pattern_table.add_row(
                pattern['pattern_value'],
                pattern['pattern_type'],
                f"{pattern['win_rate']*100:.1f}%",
                f"{pattern['avg_rr']:.2f}",
                str(pattern['sample_size']),
                quality_emoji
            )
        
        console.print(pattern_table)
    
    # Recommendations
    if analysis['recommendations']:
        console.print(f"\n[bold yellow]Recommendations:[/bold yellow]")
        for rec in analysis['recommendations']:
            if "INCREASE" in rec or "✓" in rec:
                console.print(f"  [green]{rec}[/green]")
            elif "AVOID" in rec or "REDUCE" in rec or "✗" in rec:
                console.print(f"  [red]{rec}[/red]")
            else:
                console.print(f"  [yellow]{rec}[/yellow]")
    
    # ========================================================================
    # Update Agent Weights
    # ========================================================================
    console.print(f"\n[bold yellow]Updating Agent Weights...[/bold yellow]\n")
    
    with console.status("[cyan]Calculating agent performances..."):
        performances = updater.calculate_agent_performances(lookback_days=30)
        result = updater.update_weights(performances)
    
    # Show weight changes
    weight_table = Table(box=box.ROUNDED, title="Agent Weight Changes")
    weight_table.add_column("Agent", style="cyan")
    weight_table.add_column("Old Weight", justify="right")
    weight_table.add_column("New Weight", justify="right")
    weight_table.add_column("Change", justify="right")
    weight_table.add_column("Direction")
    
    for agent in result['old_weights'].keys():
        old = result['old_weights'][agent]
        new = result['new_weights'][agent]
        change = result['changes'][agent]
        
        if abs(change) < 0.01:
            arrow = "→"
            color = "dim"
        elif change > 0:
            arrow = "↑"
            color = "green"
        else:
            arrow = "↓"
            color = "red"
        
        weight_table.add_row(
            agent.capitalize(),
            f"{old:.3f}",
            f"{new:.3f}",
            f"[{color}]{change:+.3f}[/{color}]",
            arrow
        )
    
    console.print(weight_table)
    
    # Reasoning
    if result['reasoning']:
        console.print(f"\n[bold]Reasoning:[/bold]")
        for line in result['reasoning'].split('\n'):
            if line.strip():
                console.print(f"  {line}")
    
    console.print(f"\n[green]✅ Learning system updated![/green]\n")


def risk_status_command():
    """View detailed risk guardrails status."""
    from tradingagents.risk import RiskGuardrails
    
    guardrails = RiskGuardrails()
    report = guardrails.format_report()
    
    console.print(Panel(report, title="[bold cyan]Risk Guardrails Status[/bold cyan]", border_style="cyan"))


def regime_command(
    symbol: str = typer.Option("XAUUSD", "--symbol", "-s", help="Symbol to analyze"),
    days: int = typer.Option(100, "--days", "-d", help="Days of historical data")
):
    """Detect current market regime for a symbol."""
    from tradingagents.indicators import RegimeDetector
    from tradingagents.dataflows.mt5_data import get_mt5_data
    from datetime import datetime, timedelta
    import pandas as pd
    from io import StringIO
    
    console.print(f"\n[bold cyan]═══ REGIME DETECTION: {symbol} ═══[/bold cyan]\n")
    
    # Get price data
    console.print("[dim]Fetching price data from MT5...[/dim]")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    try:
        price_data = get_mt5_data(symbol, start_date, end_date, timeframe="D1")
        df = pd.read_csv(StringIO(price_data), comment='#')
        
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        console.print(f"[dim]Loaded {len(close)} bars[/dim]\n")
        
        # Detect regime
        detector = RegimeDetector()
        regime = detector.get_full_regime(high, low, close)
        
        # Display results
        regime_table = Table(box=box.ROUNDED, title=f"Current Regime: {symbol}")
        regime_table.add_column("Component", style="cyan")
        regime_table.add_column("Value", justify="right")
        regime_table.add_column("Description")
        
        regime_table.add_row(
            "Market Trend",
            regime['market_regime'],
            "📈" if regime['market_regime'] == 'trending-up' else "📉" if regime['market_regime'] == 'trending-down' else "↔️"
        )
        
        regime_table.add_row(
            "Volatility",
            regime['volatility_regime'],
            "🔥" if regime['volatility_regime'] == 'extreme' else "⚡" if regime['volatility_regime'] == 'high' else "→" if regime['volatility_regime'] == 'normal' else "❄️"
        )
        
        regime_table.add_row(
            "Expansion",
            regime['expansion_regime'],
            "📊"
        )
        
        console.print(regime_table)
        
        # Description
        description = detector.get_regime_description(regime)
        console.print(f"\n[bold]Description:[/bold] {description}")
        
        # Trading implications
        console.print(f"\n[bold yellow]Trading Implications:[/bold yellow]")
        
        trend_favorable = detector.is_favorable_for_trend_trading(regime)
        range_favorable = detector.is_favorable_for_range_trading(regime)
        risk_adj = detector.get_risk_adjustment_factor(regime)
        
        console.print(f"  Trend Trading: {'✅ Favorable' if trend_favorable else '❌ Not Favorable'}")
        console.print(f"  Range Trading: {'✅ Favorable' if range_favorable else '❌ Not Favorable'}")
        console.print(f"  Position Size Adjustment: {risk_adj:.2f}x")
        
        if risk_adj < 1.0:
            console.print(f"    [yellow]→ Reduce size to {risk_adj*100:.0f}% due to volatility[/yellow]")
        elif risk_adj > 1.0:
            console.print(f"    [green]→ Can increase size to {risk_adj*100:.0f}%[/green]")
        
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def similar_trades_command(
    symbol: str = typer.Option("XAUUSD", "--symbol", "-s", help="Symbol"),
    direction: str = typer.Option("BUY", "--direction", "-d", help="Direction (BUY/SELL)"),
    setup: Optional[str] = typer.Option(None, "--setup", help="Setup type"),
    regime: Optional[str] = typer.Option(None, "--regime", help="Market regime"),
    n: int = typer.Option(5, "--limit", "-n", help="Number of results")
):
    """Find similar historical trades."""
    from tradingagents.learning import TradeSimilaritySearch
    
    console.print(f"\n[bold cyan]═══ SIMILAR TRADES: {symbol} {direction} ═══[/bold cyan]\n")
    
    # Build current setup
    current_setup = {
        "symbol": symbol,
        "direction": direction.upper()
    }
    
    if setup:
        current_setup["setup_type"] = setup
    if regime:
        current_setup["market_regime"] = regime
    
    console.print("[dim]Searching historical trades...[/dim]\n")
    
    # Search
    searcher = TradeSimilaritySearch()
    result = searcher.find_similar_trades(current_setup, n_results=n)
    
    stats = result['statistics']
    
    # Statistics
    if stats['sample_size'] == 0:
        console.print("[yellow]No similar trades found.[/yellow]")
        console.print("[dim]Try broadening your search criteria.[/dim]\n")
        return
    
    stats_table = Table(box=box.ROUNDED, title="Historical Performance")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right")
    
    stats_table.add_row("Similar Trades", str(stats['sample_size']))
    
    wr = stats['win_rate'] * 100
    wr_color = "green" if wr >= 60 else "yellow" if wr >= 45 else "red"
    stats_table.add_row("Win Rate", f"[{wr_color}]{wr:.1f}%[/{wr_color}]")
    
    stats_table.add_row("Avg RR", f"{stats['avg_rr']:.2f}")
    stats_table.add_row("Best Trade", f"{stats['best_rr']:+.2f}R")
    stats_table.add_row("Worst Trade", f"{stats['worst_rr']:+.2f}R")
    
    adj = stats['confidence_adjustment']
    adj_color = "green" if adj > 0 else "red" if adj < 0 else "dim"
    stats_table.add_row("Confidence Adj", f"[{adj_color}]{adj:+.2f}[/{adj_color}]")
    
    console.print(stats_table)
    
    # Similar trades
    if result['similar_trades']:
        console.print(f"\n[bold yellow]Top {len(result['similar_trades'])} Similar Trades:[/bold yellow]")
        
        trades_table = Table(box=box.ROUNDED)
        trades_table.add_column("#", justify="right")
        trades_table.add_column("Outcome")
        trades_table.add_column("RR", justify="right")
        trades_table.add_column("Setup")
        trades_table.add_column("Regime")
        trades_table.add_column("Similarity", justify="right")
        
        for i, (trade, score) in enumerate(zip(result['similar_trades'], result['similarity_scores']), 1):
            outcome = "✅ WIN" if trade.get('was_correct') else "❌ LOSS"
            outcome_color = "green" if trade.get('was_correct') else "red"
            
            rr = trade.get('rr_realized', 0)
            rr_color = "green" if rr > 0 else "red"
            
            setup_type = trade.get('setup_type', 'unknown')
            regime_str = f"{trade.get('market_regime', '?')}/{trade.get('volatility_regime', '?')}"
            
            trades_table.add_row(
                str(i),
                f"[{outcome_color}]{outcome}[/{outcome_color}]",
                f"[{rr_color}]{rr:+.2f}R[/{rr_color}]",
                setup_type,
                regime_str,
                f"{score:.2f}"
            )
        
        console.print(trades_table)
    
    # Recommendation
    console.print(f"\n[bold]Recommendation:[/bold]")
    console.print(f"  {result['recommendation']}")
    
    console.print()
