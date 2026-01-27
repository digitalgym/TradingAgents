"""
Batch review command for iterating through unreviewed trades grouped by symbol.

Supports two modes:
1. Review closed trades for memory creation (original behavior)
2. Review open positions with SL/TP updates (like 'review' command)

Uses structured outputs for reliable LLM response parsing.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional
import questionary
import os
from datetime import datetime, timedelta

console = Console()


def batch_review_open_positions():
    """
    Batch review all open positions with LLM analysis and SL/TP updates.

    Similar to the 'review' command but iterates through all positions automatically.
    """
    from tradingagents.dataflows.llm_client import get_llm_client, structured_output
    from tradingagents.schemas import QuickPositionReview
    from tradingagents.dataflows.mt5_data import (
        get_open_positions,
        get_mt5_current_price,
        get_mt5_data,
        modify_position,
    )
    from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
    from tradingagents.trade_decisions import store_decision
    import csv
    import io

    console.print("\n[bold cyan]═══ BATCH POSITION REVIEW ═══[/bold cyan]\n")

    # Get open positions
    try:
        pos_list = get_open_positions()

        if not pos_list:
            console.print("[yellow]No open positions to review.[/yellow]")
            return

    except Exception as e:
        console.print(f"[red]Error getting positions: {e}[/red]")
        return

    console.print(f"[cyan]Found {len(pos_list)} open position(s)[/cyan]\n")

    # Show summary table
    summary_table = Table(box=box.ROUNDED, title="Open Positions")
    summary_table.add_column("#", style="dim")
    summary_table.add_column("Symbol", style="cyan")
    summary_table.add_column("Type")
    summary_table.add_column("Volume", justify="right")
    summary_table.add_column("Entry", justify="right")
    summary_table.add_column("P/L", justify="right")

    for i, p in enumerate(pos_list, 1):
        profit_color = "green" if p['profit'] >= 0 else "red"
        summary_table.add_row(
            str(i),
            p['symbol'],
            p['type'],
            f"{p['volume']}",
            f"{p['price_open']:.5f}",
            f"[{profit_color}]${p['profit']:.2f}[/{profit_color}]"
        )

    console.print(summary_table)
    console.print()

    # Ask how to proceed
    review_mode = questionary.select(
        "How would you like to review?",
        choices=[
            "Review all positions",
            "Select specific position",
            "Cancel"
        ]
    ).ask()

    if review_mode == "Cancel":
        return

    if review_mode == "Select specific position":
        pos_num = questionary.text(
            "Enter position number:",
            default="1",
        ).ask()
        try:
            idx = int(pos_num) - 1
            positions_to_review = [pos_list[idx]]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection.[/red]")
            return
    else:
        positions_to_review = pos_list

    # Setup LLM client (uses xAI Responses API or OpenAI Chat Completions)
    try:
        client, model, uses_responses = get_llm_client()
    except ValueError as e:
        console.print(f"[red]No API key available: {e}[/red]")
        return

    reviewed_count = 0
    modified_count = 0
    skipped_count = 0

    for i, pos in enumerate(positions_to_review, 1):
        console.print(f"\n[bold cyan]═══ Position {i}/{len(positions_to_review)}: {pos['symbol']} {pos['type']} ═══[/bold cyan]")

        # Get current market data
        try:
            price_info = get_mt5_current_price(pos['symbol'])
            current_price = price_info['bid'] if pos['type'] == 'SELL' else price_info['ask']

            # Get recent price history
            today = datetime.now()
            start_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
            history_csv = get_mt5_data(pos['symbol'], start_date, end_date)

            # Parse for high/low
            recent_high = current_price
            recent_low = current_price
            if history_csv and not history_csv.startswith("Error"):
                lines = [l for l in history_csv.split('\n') if l and not l.startswith('#')]
                if len(lines) > 1:
                    reader = csv.DictReader(io.StringIO('\n'.join(lines)))
                    rows = list(reader)
                    if rows:
                        recent_high = max(float(r.get('high', current_price)) for r in rows)
                        recent_low = min(float(r.get('low', current_price)) for r in rows)

            # Calculate metrics
            entry = pos['price_open']
            sl = pos['sl']
            tp = pos['tp']

            if pos['type'] == 'BUY':
                current_pnl_pct = ((current_price - entry) / entry) * 100
                sl_distance = entry - sl if sl > 0 else 0
                tp_distance = tp - entry if tp > 0 else 0
                distance_to_sl = current_price - sl if sl > 0 else 0
                distance_to_tp = tp - current_price if tp > 0 else 0
            else:  # SELL
                current_pnl_pct = ((entry - current_price) / entry) * 100
                sl_distance = sl - entry if sl > 0 else 0
                tp_distance = entry - tp if tp > 0 else 0
                distance_to_sl = sl - current_price if sl > 0 else 0
                distance_to_tp = current_price - tp if tp > 0 else 0

            risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0

        except Exception as e:
            console.print(f"[red]Error getting market data: {e}[/red]")
            skipped_count += 1
            continue

        # Display position details
        detail_table = Table(box=box.SIMPLE)
        detail_table.add_column("", style="dim")
        detail_table.add_column("")

        profit_color = "green" if pos['profit'] >= 0 else "red"
        detail_table.add_row("Entry", f"{entry:.5f}")
        detail_table.add_row("Current", f"{current_price:.5f}")
        detail_table.add_row("P/L", f"[{profit_color}]{current_pnl_pct:+.2f}% (${pos['profit']:.2f})[/{profit_color}]")
        detail_table.add_row("Stop Loss", f"{sl} (distance: {distance_to_sl:.5f})")
        detail_table.add_row("Take Profit", f"{tp} (distance: {distance_to_tp:.5f})")
        detail_table.add_row("Volume", f"{pos['volume']} lots")
        detail_table.add_row("Ticket", str(pos['ticket']))

        console.print(detail_table)

        # Get ATR and calculate suggestions
        atr = get_atr_for_symbol(pos['symbol'], period=14)
        dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)

        atr_suggestions = ""
        breakeven_sl = None
        trailing_sl = None

        if atr > 0:
            suggestions = dsl.suggest_stop_adjustment(
                entry_price=entry,
                current_price=current_price,
                current_sl=sl,
                current_tp=tp,
                atr=atr,
                direction=pos['type'],
            )

            console.print(f"\n[bold]ATR Analysis (ATR: {atr:.5f}):[/bold]")
            console.print(f"  Recommendation: {suggestions['recommendation']}")

            if suggestions.get('breakeven'):
                breakeven_sl = suggestions['breakeven']['new_sl']
                console.print(f"  [green]Breakeven SL: {breakeven_sl}[/green]")

            if suggestions.get('trailing'):
                trailing_sl = suggestions['trailing']['new_sl']
                console.print(f"  [yellow]Trailing SL: {trailing_sl}[/yellow]")

            atr_suggestions = f"""
ATR-BASED ANALYSIS:
- Current ATR (14-period): {atr:.5f}
- ATR-based Stop Distance: {atr * 2:.5f} (2x ATR)
- ATR-based Trailing Distance: {atr * 1.5:.5f} (1.5x ATR)
- Suggested Breakeven SL: {breakeven_sl if breakeven_sl else 'N/A'}
- Suggested Trailing SL: {trailing_sl if trailing_sl else 'N/A'}
- System Recommendation: {suggestions.get('recommendation', 'N/A')}
"""

        # Ask action for this position
        action = questionary.select(
            "Action:",
            choices=[
                "Analyze with LLM & update SL/TP",
                "Quick update (use ATR suggestions)",
                "Skip this position",
                "Quit batch review"
            ]
        ).ask()

        if action == "Quit batch review":
            console.print(f"\n[yellow]Batch review stopped. Reviewed: {reviewed_count}, Modified: {modified_count}, Skipped: {skipped_count}[/yellow]")
            return

        if action == "Skip this position":
            skipped_count += 1
            continue

        if action == "Quick update (use ATR suggestions)":
            # Apply ATR-based suggestions directly
            new_sl = sl
            new_tp = tp

            if breakeven_sl or trailing_sl:
                sl_choice = questionary.select(
                    "Select SL update:",
                    choices=[
                        f"Breakeven: {breakeven_sl}" if breakeven_sl else None,
                        f"Trailing: {trailing_sl}" if trailing_sl else None,
                        "Keep current",
                    ]
                ).ask()

                if sl_choice and "Breakeven" in sl_choice:
                    new_sl = breakeven_sl
                elif sl_choice and "Trailing" in sl_choice:
                    new_sl = trailing_sl

            if new_sl != sl:
                console.print(f"[bold]Updating SL: {sl} -> {new_sl}[/bold]")
                try:
                    result = modify_position(ticket=pos['ticket'], sl=new_sl, tp=new_tp)
                    if result and result.get("success"):
                        console.print("[green]Position modified[/green]")
                        modified_count += 1
                    else:
                        console.print(f"[red]Error: {result.get('error') if result else 'No result'}[/red]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            else:
                console.print("[dim]No changes made[/dim]")

            reviewed_count += 1
            continue

        # Full LLM analysis
        console.print("\n[dim]Analyzing with LLM...[/dim]")

        prompt = f"""Analyze this open trade and provide specific recommendations for managing it.

POSITION DETAILS:
- Symbol: {pos['symbol']}
- Direction: {pos['type']}
- Entry Price: {entry}
- Current Price: {current_price}
- Current P/L: {current_pnl_pct:+.2f}%
- Unrealized Profit: ${pos['profit']:.2f}
- Volume: {pos['volume']} lots

CURRENT RISK MANAGEMENT:
- Stop Loss: {sl} (distance from current: {distance_to_sl:.5f})
- Take Profit: {tp} (distance from current: {distance_to_tp:.5f})
- Risk/Reward Ratio: {risk_reward:.2f}

RECENT PRICE ACTION:
- 10-day High: {recent_high}
- 10-day Low: {recent_low}
- Current vs High: {((current_price - recent_high) / recent_high * 100):+.2f}%
- Current vs Low: {((current_price - recent_low) / recent_low * 100):+.2f}%
{atr_suggestions}
Analyze the position and provide your recommendation."""

        try:
            # Use structured output for guaranteed schema compliance
            review = structured_output(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert trade manager. Analyze open positions and provide recommendations with specific price levels for stop loss and take profit adjustments."},
                    {"role": "user", "content": prompt},
                ],
                response_schema=QuickPositionReview,
                max_tokens=600,
                temperature=0.7,
                use_responses_api=uses_responses,
            )

            if not review:
                console.print("[yellow]No analysis returned[/yellow]")
                skipped_count += 1
                continue

            # Extract values from structured response
            suggested_sl = review.suggested_sl
            suggested_tp = review.suggested_tp
            recommendation = review.recommendation
            risk_level = review.risk_level
            reasoning = review.reasoning

            # Display the analysis
            rec_color = {"HOLD": "yellow", "CLOSE": "red", "ADJUST": "cyan"}.get(recommendation, "white")
            risk_color = {"Low": "green", "Medium": "yellow", "High": "red"}.get(risk_level, "white")

            analysis_content = f"""[bold]Recommendation:[/bold] [{rec_color}]{recommendation}[/{rec_color}]
[bold]Risk Level:[/bold] [{risk_color}]{risk_level}[/{risk_color}]
"""
            if suggested_sl:
                analysis_content += f"[bold]Suggested SL:[/bold] {suggested_sl}\n"
            if suggested_tp:
                analysis_content += f"[bold]Suggested TP:[/bold] {suggested_tp}\n"
            analysis_content += f"\n[bold]Reasoning:[/bold]\n{reasoning}"

            console.print(Panel(analysis_content, title=f"Analysis: {pos['symbol']}", border_style="cyan"))

            # Ask how to apply
            apply_choice = questionary.select(
                "Apply changes?",
                choices=[
                    "Use suggested values" if (suggested_sl or suggested_tp) else "No suggested values found",
                    "Enter manual values",
                    "Skip changes"
                ]
            ).ask()

            new_sl = sl
            new_tp = tp

            if apply_choice == "Use suggested values" and (suggested_sl or suggested_tp):
                new_sl = suggested_sl if suggested_sl else sl
                new_tp = suggested_tp if suggested_tp else tp

            elif apply_choice == "Enter manual values":
                sl_input = questionary.text(f"Stop Loss (current: {sl}, blank to keep):").ask()
                if sl_input:
                    try:
                        new_sl = float(sl_input)
                    except ValueError:
                        console.print("[yellow]Invalid SL, keeping current[/yellow]")

                tp_input = questionary.text(f"Take Profit (current: {tp}, blank to keep):").ask()
                if tp_input:
                    try:
                        new_tp = float(tp_input)
                    except ValueError:
                        console.print("[yellow]Invalid TP, keeping current[/yellow]")

            # Apply to MT5
            if new_sl != sl or new_tp != tp:
                console.print(f"\n[bold]Changes:[/bold]")
                if new_sl != sl:
                    console.print(f"  SL: {sl} -> {new_sl}")
                if new_tp != tp:
                    console.print(f"  TP: {tp} -> {new_tp}")

                confirm = questionary.confirm("Apply to MT5?", default=True).ask()

                if confirm:
                    try:
                        result = modify_position(ticket=pos['ticket'], sl=new_sl, tp=new_tp)
                        if result and result.get("success"):
                            console.print("[green]Position modified[/green]")
                            modified_count += 1
                        else:
                            console.print(f"[red]Error: {result.get('error') if result else 'No result'}[/red]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")

            reviewed_count += 1

        except Exception as e:
            console.print(f"[red]Error during analysis: {e}[/red]")
            skipped_count += 1

    # Summary
    console.print(f"\n[bold green]═══ BATCH REVIEW COMPLETE ═══[/bold green]")
    console.print(f"  Reviewed: {reviewed_count}")
    console.print(f"  Modified: {modified_count}")
    console.print(f"  Skipped: {skipped_count}")
    console.print(f"  Total: {len(positions_to_review)}\n")


def batch_review_command(review_open: bool = False):
    """
    Batch review trades.

    Two modes:
    - review_open=False: Review closed unreviewed trades for memory creation (default)
    - review_open=True: Review open positions with SL/TP updates
    """
    if review_open:
        batch_review_open_positions()
        return

    # Original closed trade review logic
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
