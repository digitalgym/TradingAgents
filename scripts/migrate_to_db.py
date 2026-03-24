#!/usr/bin/env python
"""
Migrate file-based storage to PostgreSQL database.

Run from project root:
    python scripts/migrate_to_db.py

Options:
    --dry-run    Show what would be migrated without making changes
    --force      Re-migrate all data, even if already in DB
    --schema     Only create/update schema (no data migration)
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


async def ensure_schema():
    """Create all tables and indexes if they don't exist."""
    import asyncpg

    url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("POSTGRES_URL or DATABASE_URL environment variable not set")

    conn = await asyncpg.connect(url)
    try:
        # --- decisions (core) ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                mt5_ticket BIGINT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                exit_date TIMESTAMPTZ,
                data JSONB NOT NULL,
                CONSTRAINT valid_status CHECK (status IN (
                    'active', 'closed', 'failed', 'cancelled', 'retried', 'order_unfilled'
                ))
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
            CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
            CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(mt5_ticket) WHERE mt5_ticket IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_decisions_exit ON decisions(exit_date DESC) WHERE exit_date IS NOT NULL;
        """)

        # --- decision_contexts ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_contexts (
                decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id) ON DELETE CASCADE,
                context BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- decision_events ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_events (
                id SERIAL PRIMARY KEY,
                decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                source TEXT,
                details JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_events_decision ON decision_events(decision_id);
        """)

        # --- automation_state ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_state (
                instance_name TEXT PRIMARY KEY,
                state JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- automation_guardrails ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_guardrails (
                instance_name TEXT PRIMARY KEY,
                guardrails JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- automation_status ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_status (
                instance_name TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'stopped',
                pipeline TEXT,
                symbols JSONB,
                auto_execute BOOLEAN DEFAULT FALSE,
                last_analysis TIMESTAMPTZ,
                last_trade TIMESTAMPTZ,
                active_positions INT DEFAULT 0,
                error_message TEXT,
                config JSONB,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- automation_control ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_control (
                command_id TEXT PRIMARY KEY,
                instance_name TEXT NOT NULL,
                action TEXT NOT NULL,
                payload JSONB,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                applied_at TIMESTAMPTZ,
                error TEXT,
                source TEXT DEFAULT 'web_ui'
            );
            CREATE INDEX IF NOT EXISTS idx_control_instance ON automation_control(instance_name, status);
            CREATE INDEX IF NOT EXISTS idx_control_pending ON automation_control(instance_name, created_at)
                WHERE status = 'pending';
        """)

        # --- trade_queue ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_queue (
                command_id TEXT PRIMARY KEY,
                command_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                payload JSONB NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                processed_at TIMESTAMPTZ,
                result JSONB,
                error TEXT,
                source TEXT DEFAULT 'web_ui'
            );
            CREATE INDEX IF NOT EXISTS idx_queue_status ON trade_queue(status);
            CREATE INDEX IF NOT EXISTS idx_queue_created ON trade_queue(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_queue_pending ON trade_queue(status, created_at)
                WHERE status = 'pending';
        """)

        # --- agent_weights ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_weights (
                id TEXT PRIMARY KEY DEFAULT 'default',
                weights JSONB NOT NULL,
                weight_history JSONB DEFAULT '[]',
                learning_rate FLOAT DEFAULT 0.1,
                momentum FLOAT DEFAULT 0.9,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- configs ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- tuning_history ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tuning_history (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                pipeline TEXT NOT NULL,
                result JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_tuning_symbol ON tuning_history(symbol);
        """)

        # --- portfolio_state ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id TEXT PRIMARY KEY DEFAULT 'default',
                state JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # --- signals ---
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                data JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
        """)

        print("Schema: all 13 tables ensured")

    finally:
        await conn.close()


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
    parser.add_argument("--schema", action="store_true", help="Only create/update schema, no data migration")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")

    print("Starting migration...\n")

    # Always ensure schema first
    print("--- Schema ---")
    await ensure_schema()
    print()

    if args.schema:
        print("=== Schema-only mode, skipping data migration ===")
        return

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
