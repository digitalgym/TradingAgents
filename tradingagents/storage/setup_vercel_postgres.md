# Setting Up Vercel Postgres for TradingAgents

## 1. Create Vercel Postgres Database

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **Storage** tab
3. Click **Create Database** → Select **Postgres**
4. Name it (e.g., `tradingagents-db`)
5. Copy the connection string from the dashboard

## 2. Install Dependencies

```bash
pip install asyncpg
```

## 3. Configure Environment

Add to your `.env` file:

```bash
# Vercel Postgres connection
POSTGRES_URL=postgresql://user:password@host:5432/database?sslmode=require

# Enable Postgres storage
TRADING_AGENTS_STORAGE=postgres
```

## 4. Migrate Existing Data

If you have existing file-based decisions:

```bash
# Preview what will be migrated
python -m tradingagents.storage.migrate files-to-postgres --dry-run

# Run the migration
python -m tradingagents.storage.migrate files-to-postgres

# Verify migration
python -m tradingagents.storage.migrate verify
```

## 5. Backup (Optional)

To export Postgres data back to files:

```bash
python -m tradingagents.storage.migrate postgres-to-files --output-dir ./backup
```

## Free Tier Limits (Neon)

- Storage: ~0.5 GB
- Compute: 191.9 hours/month
- Connections: Limited (use pooling)

For typical trading with ~100 decisions/month, this is plenty.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Local Machine     │     │   Vercel Postgres   │
│   (MT5 Connection)  │────▶│   (Neon Backend)    │
│                     │     │                     │
│  - quant_automation │     │  - decisions table  │
│  - trade execution  │     │  - contexts table   │
│                     │     │  - events table     │
└─────────────────────┘     │  - state tables     │
                            └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │   Web UI (Vercel)   │
                            │   No MT5 needed!    │
                            │                     │
                            │  - View decisions   │
                            │  - See performance  │
                            │  - Review trades    │
                            └─────────────────────┘
```

## Switching Back to File Storage

Just remove or comment out the environment variable:

```bash
# TRADING_AGENTS_STORAGE=postgres
```

The system defaults to file-based storage for backward compatibility.
