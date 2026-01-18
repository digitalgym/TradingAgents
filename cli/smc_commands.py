"""
CLI commands for Smart Money Concepts analysis
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional

console = Console()


def smc_analyze_command(
    symbol: str,
    timeframes: Optional[str] = "1H,4H,D1"
):
    """
    Analyze smart money concepts for a symbol.
    
    Args:
        symbol: Trading symbol
        timeframes: Comma-separated timeframes (e.g., "1H,4H,D1")
    """
    from tradingagents.dataflows.smc_utils import (
        analyze_multi_timeframe_smc,
        get_htf_bias_alignment
    )
    from tradingagents.indicators.smart_money import SmartMoneyAnalyzer
    
    console.print(f"\n[bold cyan]═══ SMC ANALYSIS: {symbol} ═══[/bold cyan]\n")
    
    # Parse timeframes
    tf_list = [tf.strip() for tf in timeframes.split(',')]
    
    # Run analysis
    console.print("[dim]Analyzing smart money concepts...[/dim]")
    mtf_analysis = analyze_multi_timeframe_smc(symbol, tf_list)
    
    if not mtf_analysis:
        console.print("[red]❌ Could not fetch data. Ensure MT5 is running and logged in.[/red]\n")
        return
    
    # Display each timeframe
    for tf, analysis in mtf_analysis.items():
        console.print(f"\n[bold yellow]{tf} TIMEFRAME[/bold yellow]")
        
        # Create table
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        
        table.add_row("Current Price", f"${analysis['current_price']:.2f}")
        
        bias_color = "green" if analysis['bias'] == 'bullish' else "red" if analysis['bias'] == 'bearish' else "yellow"
        table.add_row("Market Bias", f"[{bias_color}]{analysis['bias'].upper()}[/{bias_color}]")
        
        table.add_row("Order Blocks (Unmitigated)", str(analysis['order_blocks']['unmitigated']))
        table.add_row("Fair Value Gaps (Unmitigated)", str(analysis['fair_value_gaps']['unmitigated']))
        
        if analysis['structure']['recent_bos']:
            table.add_row("Recent BOS", f"✓ {len(analysis['structure']['recent_bos'])}")
        
        if analysis['structure']['recent_choc']:
            table.add_row("Recent CHOC", f"⚠️  {len(analysis['structure']['recent_choc'])}")
        
        console.print(table)
        
        # Show key zones
        if analysis['nearest_support']:
            s = analysis['nearest_support']
            dist = ((analysis['current_price'] - s['top']) / analysis['current_price'] * 100)
            console.print(f"  [green]Support:[/green] ${s['bottom']:.2f}-${s['top']:.2f} ({s['type']}) | -{dist:.2f}%")
        
        if analysis['nearest_resistance']:
            r = analysis['nearest_resistance']
            dist = ((r['bottom'] - analysis['current_price']) / analysis['current_price'] * 100)
            console.print(f"  [red]Resistance:[/red] ${r['bottom']:.2f}-${r['top']:.2f} ({r['type']}) | +{dist:.2f}%")
    
    # HTF alignment
    console.print(f"\n[bold cyan]═══ HIGHER TIMEFRAME ALIGNMENT ═══[/bold cyan]\n")
    
    alignment = get_htf_bias_alignment(mtf_analysis)
    
    align_color = "green" if alignment['aligned'] else "yellow"
    console.print(f"[{align_color}]{alignment['message']}[/{align_color}]")
    console.print(f"Bias: [bold]{alignment['bias'].upper()}[/bold]")
    console.print(f"Strength: {alignment['strength'].upper()}")
    
    console.print()


def smc_levels_command(
    symbol: str,
    direction: str = "BUY",
    entry: Optional[float] = None
):
    """
    Get SMC-based stop loss and take profit suggestions.
    
    Args:
        symbol: Trading symbol
        direction: Trade direction (BUY/SELL)
        entry: Entry price (uses current if not provided)
    """
    from tradingagents.dataflows.smc_utils import (
        analyze_multi_timeframe_smc,
        suggest_smc_stop_loss,
        suggest_smc_take_profits
    )
    
    console.print(f"\n[bold cyan]═══ SMC LEVELS: {symbol} {direction} ═══[/bold cyan]\n")
    
    # Get D1 analysis for levels
    mtf_analysis = analyze_multi_timeframe_smc(symbol, ['D1'])
    
    if not mtf_analysis or 'D1' not in mtf_analysis:
        console.print("[red]❌ Could not fetch D1 data[/red]\n")
        return
    
    d1_analysis = mtf_analysis['D1']
    
    if entry is None:
        entry = d1_analysis['current_price']
    
    console.print(f"Entry Price: ${entry:.2f}")
    console.print(f"Direction: {direction}\n")
    
    # Stop loss suggestion
    console.print("[bold yellow]STOP LOSS SUGGESTION:[/bold yellow]\n")
    
    stop_suggestion = suggest_smc_stop_loss(
        smc_analysis=d1_analysis,
        direction=direction,
        entry_price=entry,
        max_distance_pct=3.0
    )
    
    if stop_suggestion:
        console.print(f"  Price: [red]${stop_suggestion['price']:.2f}[/red]")
        console.print(f"  Zone: ${stop_suggestion['zone_bottom']:.2f}-${stop_suggestion['zone_top']:.2f}")
        console.print(f"  Source: {stop_suggestion['source']}")
        console.print(f"  Strength: {stop_suggestion['strength']:.0%}")
        console.print(f"  Distance: {stop_suggestion['distance_pct']:.2f}%")
        console.print(f"  Reason: {stop_suggestion['reason']}")
    else:
        console.print("  [yellow]No suitable SMC stop found within 3% distance[/yellow]")
    
    # Take profit suggestions
    console.print(f"\n[bold yellow]TAKE PROFIT SUGGESTIONS:[/bold yellow]\n")
    
    tp_suggestions = suggest_smc_take_profits(
        smc_analysis=d1_analysis,
        direction=direction,
        entry_price=entry,
        num_targets=3
    )
    
    if tp_suggestions:
        tp_table = Table(box=box.ROUNDED)
        tp_table.add_column("Target", style="cyan")
        tp_table.add_column("Price", style="green")
        tp_table.add_column("Zone")
        tp_table.add_column("Distance")
        tp_table.add_column("Source")
        
        for tp in tp_suggestions:
            tp_table.add_row(
                f"TP{tp['number']}",
                f"${tp['price']:.2f}",
                f"${tp['zone_bottom']:.2f}-${tp['zone_top']:.2f}",
                f"+{tp['distance_pct']:.1f}%",
                tp['source']
            )
        
        console.print(tp_table)
    else:
        console.print("  [yellow]No suitable SMC targets found[/yellow]")
    
    console.print()


def smc_validate_command(
    symbol: str,
    direction: str,
    entry: float,
    stop: float,
    target: float
):
    """
    Validate a trade plan against SMC levels.
    
    Args:
        symbol: Trading symbol
        direction: Trade direction (BUY/SELL)
        entry: Entry price
        stop: Stop loss price
        target: Take profit price
    """
    from tradingagents.dataflows.smc_utils import (
        analyze_multi_timeframe_smc,
        validate_trade_against_smc
    )
    
    console.print(f"\n[bold cyan]═══ SMC VALIDATION: {symbol} ═══[/bold cyan]\n")
    
    # Get D1 analysis
    mtf_analysis = analyze_multi_timeframe_smc(symbol, ['D1'])
    
    if not mtf_analysis or 'D1' not in mtf_analysis:
        console.print("[red]❌ Could not fetch D1 data[/red]\n")
        return
    
    d1_analysis = mtf_analysis['D1']
    
    # Display trade plan
    console.print("[bold]Trade Plan:[/bold]")
    console.print(f"  Direction: {direction}")
    console.print(f"  Entry: ${entry:.2f}")
    console.print(f"  Stop Loss: ${stop:.2f}")
    console.print(f"  Take Profit: ${target:.2f}\n")
    
    # Validate
    validation = validate_trade_against_smc(
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        smc_analysis=d1_analysis
    )
    
    # Display results
    score_color = "green" if validation['score'] >= 80 else "yellow" if validation['score'] >= 60 else "red"
    console.print(f"[{score_color}]Validation Score: {validation['score']}/100[/{score_color}]")
    console.print(f"Valid: {'✓ YES' if validation['valid'] else '✗ NO'}\n")
    
    if validation['issues']:
        console.print("[bold red]Issues:[/bold red]")
        for issue in validation['issues']:
            console.print(f"  {issue}")
        console.print()
    
    if validation['suggestions']:
        console.print("[bold green]Suggestions:[/bold green]")
        for suggestion in validation['suggestions']:
            console.print(f"  {suggestion}")
        console.print()
