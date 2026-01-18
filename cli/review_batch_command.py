"""
Batch review command for iterating through unreviewed trades grouped by symbol.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional
import questionary

console = Console()


def batch_review_command():
    """
    Iterate through all unreviewed trades grouped by symbol.
    
    Prevents accidentally re-reviewing trades by:
    - Only showing unreviewed trades
    - Marking trades as reviewed after processing
    - Grouping by symbol for organized review
    """
    from tradingagents.trade_decisions import (
        list_unreviewed_decisions,
        group_decisions_by_symbol,
        close_decision,
        load_decision_context,
        mark_decision_reviewed
    )
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    
    console.print("\n[bold cyan]═══ BATCH TRADE REVIEW ═══[/bold cyan]\n")
    
    # Get all unreviewed closed trades
    unreviewed = list_unreviewed_decisions()
    
    if not unreviewed:
        console.print("[green]✓ No unreviewed trades found. All caught up![/green]\n")
        return
    
    # Group by symbol
    grouped = group_decisions_by_symbol(unreviewed)
    
    console.print(f"[cyan]Found {len(unreviewed)} unreviewed trade(s) across {len(grouped)} symbol(s)[/cyan]\n")
    
    # Show summary
    summary_table = Table(box=box.ROUNDED, title="Unreviewed Trades by Symbol")
    summary_table.add_column("Symbol", style="cyan")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Oldest", style="dim")
    
    for symbol, trades in sorted(grouped.items()):
        oldest = min(trades, key=lambda d: d.get("exit_date", ""))
        summary_table.add_row(
            symbol,
            str(len(trades)),
            oldest.get("exit_date", "N/A")[:10]
        )
    
    console.print(summary_table)
    console.print()
    
    # Ask which symbol to review
    symbols = list(grouped.keys())
    
    if len(symbols) == 1:
        selected_symbol = symbols[0]
        console.print(f"[cyan]Reviewing {selected_symbol}...[/cyan]\n")
    else:
        symbol_choice = questionary.select(
            "Select symbol to review:",
            choices=symbols + ["[Review All]", "[Cancel]"]
        ).ask()
        
        if symbol_choice == "[Cancel]":
            return
        elif symbol_choice == "[Review All]":
            selected_symbol = None
        else:
            selected_symbol = symbol_choice
    
    # Review trades for selected symbol(s)
    if selected_symbol:
        trades_to_review = grouped[selected_symbol]
        console.print(f"\n[bold]Reviewing {len(trades_to_review)} trade(s) for {selected_symbol}[/bold]\n")
    else:
        trades_to_review = unreviewed
        console.print(f"\n[bold]Reviewing all {len(trades_to_review)} trade(s)[/bold]\n")
    
    # Sort by exit date (oldest first)
    trades_to_review = sorted(trades_to_review, key=lambda d: d.get("exit_date", ""))
    
    reviewed_count = 0
    skipped_count = 0
    
    for i, decision in enumerate(trades_to_review, 1):
        console.print(f"[bold cyan]═══ Trade {i}/{len(trades_to_review)} ═══[/bold cyan]")
        
        # Display trade details
        trade_table = Table(box=box.SIMPLE)
        trade_table.add_column("Field", style="cyan")
        trade_table.add_column("Value")
        
        trade_table.add_row("Symbol", decision["symbol"])
        trade_table.add_row("Action", decision["action"])
        trade_table.add_row("Entry", f"${decision.get('entry_price', 'N/A'):.2f}")
        trade_table.add_row("Exit", f"${decision.get('exit_price', 'N/A'):.2f}")
        
        pnl_pct = decision.get("pnl_percent", 0)
        pnl_color = "green" if pnl_pct >= 0 else "red"
        trade_table.add_row("P&L", f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]")
        
        if decision.get("rr_realized"):
            rr_color = "green" if decision["rr_realized"] > 0 else "red"
            trade_table.add_row("Risk-Reward", f"[{rr_color}]{decision['rr_realized']:+.2f}R[/{rr_color}]")
        
        trade_table.add_row("Opened", decision.get("created_at", "N/A")[:19])
        trade_table.add_row("Closed", decision.get("exit_date", "N/A")[:19])
        trade_table.add_row("Exit Reason", decision.get("exit_reason", "N/A"))
        
        console.print(trade_table)
        
        # Show rationale if available
        if decision.get("rationale"):
            console.print(f"\n[dim]Rationale: {decision['rationale'][:150]}...[/dim]")
        
        console.print()
        
        # Ask what to do
        action = questionary.select(
            "Action:",
            choices=[
                "Review & Create Memory",
                "Mark as Reviewed (Skip Memory)",
                "Skip (Keep Unreviewed)",
                "Quit Batch Review"
            ]
        ).ask()
        
        if action == "Quit Batch Review":
            console.print(f"\n[yellow]Batch review stopped. Reviewed: {reviewed_count}, Skipped: {skipped_count}[/yellow]\n")
            return
        
        elif action == "Skip (Keep Unreviewed)":
            console.print("[dim]Skipping...[/dim]\n")
            skipped_count += 1
            continue
        
        elif action == "Mark as Reviewed (Skip Memory)":
            mark_decision_reviewed(decision["decision_id"])
            console.print("[green]✓ Marked as reviewed[/green]\n")
            reviewed_count += 1
            continue
        
        elif action == "Review & Create Memory":
            # Try to load context for memory creation
            context = load_decision_context(decision["decision_id"])
            
            if context:
                # Create memory
                config = DEFAULT_CONFIG.copy()
                config['use_memory'] = True
                config['embedding_provider'] = 'local'
                
                console.print("[dim]Creating memory from trade...[/dim]")
                
                try:
                    graph = TradingAgentsGraph(config=config, debug=False)
                    graph.curr_state = context
                    
                    returns = decision.get("pnl_percent", 0)
                    graph.reflect_and_remember(returns)
                    
                    console.print("[green]✓ Memory created[/green]")
                    
                except Exception as e:
                    console.print(f"[red]✗ Memory creation failed: {e}[/red]")
                    console.print("[yellow]Marking as reviewed anyway...[/yellow]")
            else:
                console.print("[yellow]No context available for memory creation[/yellow]")
                console.print("[dim]Marking as reviewed for statistics...[/dim]")
            
            # Mark as reviewed
            mark_decision_reviewed(decision["decision_id"])
            console.print("[green]✓ Trade reviewed[/green]\n")
            reviewed_count += 1
    
    # Summary
    console.print(f"\n[bold green]═══ BATCH REVIEW COMPLETE ═══[/bold green]")
    console.print(f"  Reviewed: {reviewed_count}")
    console.print(f"  Skipped: {skipped_count}")
    console.print(f"  Total: {len(trades_to_review)}\n")
    
    # Check if more unreviewed trades remain
    remaining = list_unreviewed_decisions()
    if remaining:
        console.print(f"[yellow]⚠️  {len(remaining)} unreviewed trade(s) remaining[/yellow]")
        
        continue_review = questionary.confirm(
            "Continue reviewing?",
            default=False
        ).ask()
        
        if continue_review:
            batch_review_command()  # Recursive call
    else:
        console.print("[green]✓ All trades reviewed![/green]\n")
