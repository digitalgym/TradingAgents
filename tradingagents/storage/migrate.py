"""
Migration tools for moving data between storage backends.

Usage:
    # Migrate file-based decisions to Postgres
    python -m tradingagents.storage.migrate files-to-postgres

    # Verify migration
    python -m tradingagents.storage.migrate verify

    # Export Postgres to files (backup)
    python -m tradingagents.storage.migrate postgres-to-files
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from .file_store import FileDecisionStore, DECISIONS_DIR


async def migrate_files_to_postgres(dry_run: bool = False) -> Dict[str, Any]:
    """
    Migrate all file-based decisions to PostgreSQL.

    Returns stats about the migration.
    """
    from .postgres_store import PostgresDecisionStore

    file_store = FileDecisionStore()
    pg_store = PostgresDecisionStore()

    stats = {
        "decisions_migrated": 0,
        "contexts_migrated": 0,
        "errors": [],
        "skipped": 0,
    }

    # Find all decision files
    decision_files = list(DECISIONS_DIR.glob("*.json"))
    decision_files = [f for f in decision_files if not f.name.startswith("_")]

    print(f"Found {len(decision_files)} decision files to migrate")

    for filepath in decision_files:
        decision_id = filepath.stem

        try:
            # Load from file
            decision = file_store.load(decision_id)

            if dry_run:
                print(f"  [DRY RUN] Would migrate: {decision_id}")
                stats["decisions_migrated"] += 1
                continue

            # Check if already in Postgres
            try:
                existing = await pg_store._load_async(decision_id)
                print(f"  [SKIP] Already exists: {decision_id}")
                stats["skipped"] += 1
                continue
            except KeyError:
                pass  # Not in Postgres, proceed with migration

            # Store in Postgres
            await pg_store._store_async(decision)
            stats["decisions_migrated"] += 1
            print(f"  [OK] Migrated: {decision_id}")

            # Migrate context if exists
            context = file_store.load_context(decision_id)
            if context:
                await pg_store._store_context_async(decision_id, context)
                stats["contexts_migrated"] += 1
                print(f"       + context")

        except Exception as e:
            stats["errors"].append({"decision_id": decision_id, "error": str(e)})
            print(f"  [ERROR] {decision_id}: {e}")

    return stats


async def verify_migration() -> Dict[str, Any]:
    """
    Verify that all file-based decisions exist in Postgres with matching data.
    """
    from .postgres_store import PostgresDecisionStore

    file_store = FileDecisionStore()
    pg_store = PostgresDecisionStore()

    stats = {
        "total_files": 0,
        "matched": 0,
        "missing": [],
        "mismatched": [],
    }

    decision_files = list(DECISIONS_DIR.glob("*.json"))
    decision_files = [f for f in decision_files if not f.name.startswith("_")]
    stats["total_files"] = len(decision_files)

    for filepath in decision_files:
        decision_id = filepath.stem

        try:
            file_decision = file_store.load(decision_id)
            pg_decision = await pg_store._load_async(decision_id)

            # Compare key fields
            mismatches = []
            for key in ["symbol", "status", "mt5_ticket", "entry_price", "exit_price", "pnl"]:
                file_val = file_decision.get(key)
                pg_val = pg_decision.get(key)
                if file_val != pg_val:
                    mismatches.append(f"{key}: {file_val} != {pg_val}")

            if mismatches:
                stats["mismatched"].append({
                    "decision_id": decision_id,
                    "mismatches": mismatches
                })
                print(f"  [MISMATCH] {decision_id}: {', '.join(mismatches)}")
            else:
                stats["matched"] += 1

        except KeyError:
            stats["missing"].append(decision_id)
            print(f"  [MISSING] {decision_id}")
        except Exception as e:
            print(f"  [ERROR] {decision_id}: {e}")

    print(f"\nVerification complete:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Matched: {stats['matched']}")
    print(f"  Missing: {len(stats['missing'])}")
    print(f"  Mismatched: {len(stats['mismatched'])}")

    return stats


async def export_postgres_to_files(output_dir: Path = None) -> Dict[str, Any]:
    """
    Export all Postgres decisions to JSON files (backup/restore).
    """
    from .postgres_store import PostgresDecisionStore

    pg_store = PostgresDecisionStore()
    output_dir = output_dir or (DECISIONS_DIR.parent / "trade_decisions_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "exported": 0,
        "contexts_exported": 0,
        "errors": [],
    }

    # Get all decisions (active + closed)
    await pg_store._ensure_tables()
    pool = await pg_store._get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT decision_id, data FROM decisions")

    print(f"Found {len(rows)} decisions to export")

    for row in rows:
        decision_id = row["decision_id"]
        try:
            decision = json.loads(row["data"])

            # Write to file
            filepath = output_dir / f"{decision_id}.json"
            with open(filepath, "w") as f:
                json.dump(decision, f, indent=2, default=str)

            stats["exported"] += 1

            # Export context if exists
            context = await pg_store._load_context_async(decision_id)
            if context:
                import pickle
                context_file = output_dir / f"{decision_id}_context.pkl"
                with open(context_file, "wb") as f:
                    pickle.dump(context, f)
                stats["contexts_exported"] += 1

            print(f"  [OK] Exported: {decision_id}")

        except Exception as e:
            stats["errors"].append({"decision_id": decision_id, "error": str(e)})
            print(f"  [ERROR] {decision_id}: {e}")

    print(f"\nExport complete to: {output_dir}")
    print(f"  Decisions: {stats['exported']}")
    print(f"  Contexts: {stats['contexts_exported']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate TradingAgents storage")
    parser.add_argument(
        "command",
        choices=["files-to-postgres", "verify", "postgres-to-files"],
        help="Migration command",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without doing it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for postgres-to-files export",
    )

    args = parser.parse_args()

    if args.command == "files-to-postgres":
        stats = asyncio.run(migrate_files_to_postgres(dry_run=args.dry_run))
        print(f"\nMigration complete:")
        print(f"  Decisions migrated: {stats['decisions_migrated']}")
        print(f"  Contexts migrated: {stats['contexts_migrated']}")
        print(f"  Skipped: {stats['skipped']}")
        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"    - {err['decision_id']}: {err['error']}")

    elif args.command == "verify":
        asyncio.run(verify_migration())

    elif args.command == "postgres-to-files":
        asyncio.run(export_postgres_to_files(args.output_dir))


if __name__ == "__main__":
    main()
