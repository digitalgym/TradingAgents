#!/usr/bin/env python
"""
Migrate file-based storage to PostgreSQL database.

Run from project root:
    python scripts/migrate_to_db.py

Options:
    --dry-run    Show what would be migrated without making changes
    --force      Re-migrate all data, even if already in DB
"""

import sys
import os
import json
import pickle
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def migrate_decisions(dry_run: bool = False, force: bool = False):
    """Migrate trade decisions from JSON files to database."""
    from tradingagents.storage.postgres_store import get_decision_store

    decisions_dir = Path("examples/trade_decisions")
    if not decisions_dir.exists():
        print("No decisions directory found")
        return 0

    store = get_decision_store()

    # Get existing decision IDs from DB
    existing_ids = set()
    if not force:
        try:
            existing = store.list_all(limit=10000)
            existing_ids = {d["decision_id"] for d in existing}
        except Exception as e:
            print(f"Warning: Could not fetch existing decisions: {e}")

    migrated = 0
    skipped = 0
    errors = 0

    decision_files = sorted(decisions_dir.glob("*.json"))
    decision_files = [f for f in decision_files if not f.name.startswith("_")]

    print(f"Found {len(decision_files)} decision files")

    for f in decision_files:
        decision_id = f.stem

        if decision_id in existing_ids and not force:
            skipped += 1
            continue

        try:
            with open(f, "r") as fp:
                decision = json.load(fp)

            if dry_run:
                print(f"  Would migrate: {decision_id}")
            else:
                store.store(decision)

                # Check for context file
                context_file = decisions_dir / f"{decision_id}_context.pkl"
                if context_file.exists():
                    try:
                        with open(context_file, "rb") as cf:
                            context = pickle.load(cf)
                        store.store_context(decision_id, context)
                    except Exception as e:
                        print(f"  Warning: Could not migrate context for {decision_id}: {e}")

            migrated += 1
        except Exception as e:
            print(f"  Error migrating {decision_id}: {e}")
            errors += 1

    print(f"Decisions: {migrated} migrated, {skipped} skipped, {errors} errors")
    return migrated


async def migrate_configs(dry_run: bool = False, force: bool = False):
    """Migrate config files to database."""
    from tradingagents.storage.postgres_store import get_config_store

    store = get_config_store()
    migrated = 0

    # Symbol limits
    limits_file = Path("automation_symbol_limits.json")
    if limits_file.exists():
        try:
            with open(limits_file, "r") as f:
                data = json.load(f)

            existing = store.get("symbol_limits") if not force else None
            if existing is None:
                if dry_run:
                    print(f"  Would migrate: symbol_limits")
                else:
                    store.set("symbol_limits", data)
                    print(f"  Migrated: symbol_limits")
                migrated += 1
            else:
                print(f"  Skipped: symbol_limits (already exists)")
        except Exception as e:
            print(f"  Error migrating symbol_limits: {e}")

    # Scheduler state
    scheduler_file = Path("scheduler_state.json")
    if scheduler_file.exists():
        try:
            with open(scheduler_file, "r") as f:
                data = json.load(f)

            existing = store.get("scheduler_state") if not force else None
            if existing is None:
                if dry_run:
                    print(f"  Would migrate: scheduler_state")
                else:
                    store.set("scheduler_state", data)
                    print(f"  Migrated: scheduler_state")
                migrated += 1
            else:
                print(f"  Skipped: scheduler_state (already exists)")
        except Exception as e:
            print(f"  Error migrating scheduler_state: {e}")

    # Automation configs (these go to automation_status table via control store)
    configs_file = Path("automation_configs.json")
    if configs_file.exists():
        try:
            with open(configs_file, "r") as f:
                data = json.load(f)

            from tradingagents.storage.automation_control import get_automation_control
            control = get_automation_control()

            for name, cfg in data.items():
                try:
                    existing = await control.get_status(name) if not force else None
                    if existing is None or not existing.get("config"):
                        if dry_run:
                            print(f"  Would migrate: automation config '{name}'")
                        else:
                            await control.update_status(
                                instance_name=name,
                                status="stopped",
                                pipeline=cfg.get("pipeline"),
                                symbols=cfg.get("symbols"),
                                auto_execute=cfg.get("auto_execute", False),
                                config=cfg,
                            )
                            print(f"  Migrated: automation config '{name}'")
                        migrated += 1
                    else:
                        print(f"  Skipped: automation config '{name}' (already exists)")
                except Exception as e:
                    print(f"  Error migrating config '{name}': {e}")
        except Exception as e:
            print(f"  Error reading automation_configs.json: {e}")

    print(f"Configs: {migrated} migrated")
    return migrated


async def migrate_guardrails(dry_run: bool = False, force: bool = False):
    """Migrate risk guardrails state to database."""
    from tradingagents.storage.postgres_store import get_state_store

    store = get_state_store()
    migrated = 0

    risk_file = Path("examples/risk_state.pkl")
    if risk_file.exists():
        try:
            with open(risk_file, "rb") as f:
                state = pickle.load(f)

            # Migrate as "default" instance
            existing = store.load_guardrails("default") if not force else None
            if existing is None:
                if dry_run:
                    print(f"  Would migrate: guardrails (default)")
                else:
                    store.save_guardrails("default", state)
                    print(f"  Migrated: guardrails (default)")
                migrated += 1
            else:
                print(f"  Skipped: guardrails (already exists)")
        except Exception as e:
            print(f"  Error migrating guardrails: {e}")

    print(f"Guardrails: {migrated} migrated")
    return migrated


async def migrate_weights(dry_run: bool = False, force: bool = False):
    """Migrate agent weights to database."""
    from tradingagents.storage.postgres_store import get_weights_store

    store = get_weights_store()
    migrated = 0

    weights_file = Path("examples/agent_weights.pkl")
    if weights_file.exists():
        try:
            with open(weights_file, "rb") as f:
                data = pickle.load(f)

            existing = store.load() if not force else None
            if existing is None:
                if dry_run:
                    print(f"  Would migrate: agent_weights")
                else:
                    store.save(
                        weights=data.get("weights", {}),
                        weight_history=data.get("weight_history", []),
                        learning_rate=data.get("learning_rate", 0.1),
                        momentum=data.get("momentum", 0.9),
                    )
                    print(f"  Migrated: agent_weights")
                migrated += 1
            else:
                print(f"  Skipped: agent_weights (already exists)")
        except Exception as e:
            print(f"  Error migrating agent_weights: {e}")

    print(f"Agent weights: {migrated} migrated")
    return migrated


async def migrate_portfolio(dry_run: bool = False, force: bool = False):
    """Migrate portfolio state to database."""
    from tradingagents.learning.portfolio_state import PortfolioStateTracker

    migrated = 0

    portfolio_file = Path("examples/portfolio_state.pkl")
    if portfolio_file.exists():
        try:
            with open(portfolio_file, "rb") as f:
                tracker = pickle.load(f)

            if dry_run:
                print(f"  Would migrate: portfolio_state (trades={tracker.trade_count})")
            else:
                # Use the updated save_state which writes to DB
                tracker.save_state()
                print(f"  Migrated: portfolio_state (trades={tracker.trade_count}, equity={tracker.current_equity:.2f})")
            migrated += 1
        except Exception as e:
            print(f"  Error migrating portfolio_state: {e}")

    print(f"Portfolio state: {migrated} migrated")
    return migrated


async def main():
    parser = argparse.ArgumentParser(description="Migrate file-based storage to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--force", action="store_true", help="Re-migrate all data")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")

    print("Starting migration...\n")

    total = 0

    print("--- Trade Decisions ---")
    total += await migrate_decisions(args.dry_run, args.force)
    print()

    print("--- Configs ---")
    total += await migrate_configs(args.dry_run, args.force)
    print()

    print("--- Guardrails ---")
    total += await migrate_guardrails(args.dry_run, args.force)
    print()

    print("--- Agent Weights ---")
    total += await migrate_weights(args.dry_run, args.force)
    print()

    print("--- Portfolio State ---")
    total += await migrate_portfolio(args.dry_run, args.force)
    print()

    print(f"=== Migration complete: {total} items migrated ===")


if __name__ == "__main__":
    asyncio.run(main())
