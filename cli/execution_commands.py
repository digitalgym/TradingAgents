"""
CLI commands for intelligent trade execution

Commands:
- execute-plan: Parse and execute a trading plan
- monitor-plan: Monitor active staged entries
- review-plan: Review and get adjustment recommendations
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional
import os
import json

console = Console()


def execute_plan_command(
    plan_file: str,
    symbol: Optional[str] = None,
    dry_run: bool = True
):
    """
    Parse and execute a trading plan.
    
    Args:
        plan_file: Path to file containing trading plan text
        symbol: Symbol override
        dry_run: If True, simulate execution without placing real orders
    """
    from tradingagents.execution import (
        TradingPlanParser,
        OrderExecutor,
        StagedEntryManager
    )
    
    console.print("\n[bold cyan]═══ EXECUTE TRADING PLAN ═══[/bold cyan]\n")
    
    # Read plan file
    if not os.path.exists(plan_file):
        console.print(f"[red]Plan file not found: {plan_file}[/red]")
        return
    
    with open(plan_file, 'r') as f:
        plan_text = f.read()
    
    # Parse plan
    console.print("[dim]Parsing trading plan...[/dim]")
    parser = TradingPlanParser()
    plan = parser.parse_plan(plan_text, symbol=symbol)
    
    console.print(parser.format_plan_summary(plan))
    
    # Get current price
    import MetaTrader5 as mt5
    if not mt5.initialize():
        console.print("[red]MT5 initialization failed[/red]")
        return
    
    tick = mt5.symbol_info_tick(plan.symbol)
    if tick is None:
        console.print(f"[red]Could not get price for {plan.symbol}[/red]")
        return
    
    current_price = tick.ask if plan.direction == "BUY" else tick.bid
    console.print(f"\n[cyan]Current Price: ${current_price:.2f}[/cyan]\n")
    
    # Decide order types
    console.print("[bold yellow]Order Type Decisions:[/bold yellow]\n")
    
    executor = OrderExecutor(plan.symbol)
    
    order_table = Table(box=box.ROUNDED)
    order_table.add_column("Tranche", style="cyan")
    order_table.add_column("Size")
    order_table.add_column("Target Price")
    order_table.add_column("Order Type")
    order_table.add_column("Urgency")
    
    decisions = []
    for tranche in plan.entry_tranches:
        decision = executor.decide_order_type(
            direction=plan.direction,
            target_price=tranche.price_level,
            price_range=tranche.price_range,
            conditions=tranche.conditions
        )
        decisions.append(decision)
        
        order_table.add_row(
            str(tranche.tranche_number),
            f"{tranche.size_pct:.1f}%",
            f"${tranche.price_level:.2f}",
            decision.order_type.value.upper(),
            decision.urgency
        )
    
    console.print(order_table)
    
    # Ask for confirmation
    if not dry_run:
        import questionary
        confirm = questionary.confirm(
            "Execute these orders in MT5?",
            default=False
        ).ask()
        
        if not confirm:
            console.print("\n[yellow]Execution cancelled[/yellow]")
            return
    else:
        console.print("\n[dim]DRY RUN MODE - No orders will be placed[/dim]")
    
    # Initialize staged entry manager
    from datetime import datetime
    plan_id = f"{plan.symbol}_{plan.direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    staged_manager = StagedEntryManager(plan_id)
    staged_manager.initialize_tranches(plan.entry_tranches)
    
    # Execute tranches
    console.print(f"\n[bold yellow]Executing Tranches:[/bold yellow]\n")
    
    for i, (tranche, decision) in enumerate(zip(plan.entry_tranches, decisions)):
        console.print(f"Tranche {tranche.tranche_number}: {tranche.size_pct:.1f}% at ${tranche.price_level:.2f}")
        
        if dry_run:
            console.print(f"  [dim]Would place {decision.order_type.value} order at ${decision.price:.2f}[/dim]")
            # Simulate
            staged_manager.mark_tranche_active(tranche.tranche_number, 10000 + i)
        else:
            # Real execution
            account_info = mt5.account_info()
            if account_info is None:
                console.print(f"  [red]Could not get account info[/red]")
                continue
            
            account_balance = account_info.balance
            volume = (tranche.size_pct / 100) * account_balance / decision.price
            volume = round(volume, 2)  # Round to 2 decimals
            
            result = executor.execute_order(
                direction=plan.direction,
                volume=volume,
                order_decision=decision,
                stop_loss=plan.stop_loss.initial_price,
                comment=f"Tranche {tranche.tranche_number}"
            )
            
            if result['success']:
                console.print(f"  [green]✓ Order placed: #{result['ticket']}[/green]")
                staged_manager.mark_tranche_active(tranche.tranche_number, result['ticket'])
            else:
                console.print(f"  [red]✗ Order failed: {result.get('error')}[/red]")
    
    # Save plan metadata
    plan_meta = {
        "plan_id": plan_id,
        "symbol": plan.symbol,
        "direction": plan.direction,
        "created": datetime.now().isoformat(),
        "dry_run": dry_run,
        "plan_file": plan_file
    }
    
    meta_dir = "examples/execution_plans"
    os.makedirs(meta_dir, exist_ok=True)
    
    with open(f"{meta_dir}/{plan_id}.json", 'w') as f:
        json.dump(plan_meta, f, indent=2)
    
    console.print(f"\n[green]✓ Plan execution initiated: {plan_id}[/green]")
    console.print(f"[dim]Monitor with: tradingagents monitor-plan {plan_id}[/dim]\n")


def monitor_plan_command(plan_id: str):
    """
    Monitor active staged entry plan.
    
    Args:
        plan_id: Plan ID to monitor
    """
    from tradingagents.execution import StagedEntryManager, DynamicStopManager
    
    console.print(f"\n[bold cyan]═══ MONITORING PLAN: {plan_id} ═══[/bold cyan]\n")
    
    # Load staged entry manager
    try:
        staged_manager = StagedEntryManager.load_state(plan_id)
    except FileNotFoundError:
        console.print(f"[red]Plan not found: {plan_id}[/red]")
        return
    
    # Display status
    console.print(staged_manager.format_status_report())
    
    # Load plan metadata
    meta_file = f"examples/execution_plans/{plan_id}.json"
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        symbol = meta['symbol']
        direction = meta['direction']
        
        # Get current price
        import MetaTrader5 as mt5
        if mt5.initialize():
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                current_price = tick.ask if direction == "BUY" else tick.bid
                
                console.print(f"\n[cyan]Current Price: ${current_price:.2f}[/cyan]")
                
                # Check for adjustments
                adjustment = staged_manager.should_adjust_remaining(current_price)
                
                if adjustment['should_adjust']:
                    console.print(f"\n[yellow]⚠️  {adjustment['reason']}[/yellow]")
                    console.print(f"[yellow]   {adjustment['recommendation']}[/yellow]")
    
    console.print()


def review_plan_command(
    plan_id: str,
    current_price: Optional[float] = None
):
    """
    Review plan and get adjustment recommendations.
    
    Args:
        plan_id: Plan ID to review
        current_price: Current price (will fetch from MT5 if not provided)
    """
    from tradingagents.execution import (
        StagedEntryManager,
        PlanReviewer,
        TradingPlanParser
    )
    
    console.print(f"\n[bold cyan]═══ REVIEWING PLAN: {plan_id} ═══[/bold cyan]\n")
    
    # Load plan metadata
    meta_file = f"examples/execution_plans/{plan_id}.json"
    if not os.path.exists(meta_file):
        console.print(f"[red]Plan metadata not found: {plan_id}[/red]")
        return
    
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    symbol = meta['symbol']
    direction = meta['direction']
    plan_file = meta.get('plan_file')
    
    # Load original plan
    if not plan_file or not os.path.exists(plan_file):
        console.print(f"[yellow]Original plan file not found, review will be limited[/yellow]")
        return
    
    with open(plan_file, 'r') as f:
        plan_text = f.read()
    
    parser = TradingPlanParser()
    plan = parser.parse_plan(plan_text, symbol=symbol)
    
    # Load staged entry state
    try:
        staged_manager = StagedEntryManager.load_state(plan_id)
    except FileNotFoundError:
        console.print(f"[red]Staged entry state not found: {plan_id}[/red]")
        return
    
    # Get current price
    if current_price is None:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            console.print("[red]MT5 initialization failed[/red]")
            return
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            console.print(f"[red]Could not get price for {symbol}[/red]")
            return
        
        current_price = tick.ask if direction == "BUY" else tick.bid
    
    console.print(f"[cyan]Current Price: ${current_price:.2f}[/cyan]\n")
    
    # Calculate time in trade
    from datetime import datetime
    created = datetime.fromisoformat(meta['created'])
    days_in_trade = (datetime.now() - created).days
    
    # Review plan
    reviewer = PlanReviewer()
    
    review = reviewer.review_plan(
        plan=plan,
        current_price=current_price,
        entry_price=staged_manager.avg_entry_price if staged_manager.avg_entry_price > 0 else None,
        position_size=staged_manager.total_filled_pct,
        time_in_trade=days_in_trade
    )
    
    console.print(reviewer.format_review_report(review))
