# TradingAgents Web UI

A modern web interface for the TradingAgents multi-agent trading system built with Next.js and FastAPI.

## Features

- **Dashboard**: Real-time overview of account status, positions, and recent decisions
- **Analysis**: Run multi-agent analysis on any symbol with SMC support
- **Positions**: View and manage open positions and pending orders
- **Decisions**: Browse and analyze past trading decisions
- **SMC Analysis**: Smart Money Concepts analysis (Order Blocks, FVGs, Liquidity)
- **Risk Metrics**: Monitor Sharpe, Sortino, VaR, and calculate position sizes
- **Learning System**: View agent weights and discovered patterns
- **Memory Database**: Query the ChromaDB memory system
- **Portfolio Automation**: Control automated trading cycles

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- MetaTrader 5 (for live data)

### 1. Start the Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
cd web/frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`

## Architecture

```
web/
├── backend/
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── app/          # Next.js pages
    │   │   ├── page.tsx          # Dashboard
    │   │   ├── analysis/         # Analysis page
    │   │   ├── positions/        # Positions management
    │   │   ├── decisions/        # Decisions viewer
    │   │   ├── smc/              # SMC analysis
    │   │   ├── risk/             # Risk metrics
    │   │   ├── learning/         # Learning system
    │   │   ├── memory/           # Memory database
    │   │   ├── automation/       # Portfolio automation
    │   │   └── settings/         # Settings
    │   │
    │   ├── components/
    │   │   ├── ui/               # shadcn/ui components
    │   │   └── layout/           # Layout components
    │   │
    │   └── lib/
    │       ├── api.ts            # API client
    │       ├── utils.ts          # Utility functions
    │       └── websocket.ts      # WebSocket hooks
    │
    └── package.json
```

## API Endpoints

### Status & Dashboard
- `GET /api/status` - System status
- `GET /api/dashboard` - Dashboard data

### Positions
- `GET /api/positions` - List open positions
- `GET /api/orders` - List pending orders
- `POST /api/positions/modify` - Modify position SL/TP
- `POST /api/positions/close` - Close a position

### Decisions
- `GET /api/decisions` - List decisions
- `GET /api/decisions/{id}` - Get decision details

### Analysis
- `POST /api/analysis/run` - Start analysis
- `GET /api/analysis/status/{task_id}` - Get analysis status

### Risk
- `GET /api/risk/metrics` - Risk metrics
- `GET /api/risk/guardrails` - Risk guardrails status
- `POST /api/risk/position-size` - Calculate position size

### Learning
- `GET /api/learning/status` - Learning system status
- `GET /api/learning/patterns` - Identified patterns

### Memory
- `GET /api/memory/stats` - Memory statistics
- `POST /api/memory/query` - Query memory

### Portfolio Automation
- `GET /api/portfolio/status` - Automation status
- `GET /api/portfolio/config` - Configuration
- `POST /api/portfolio/start` - Start automation
- `POST /api/portfolio/stop` - Stop automation
- `POST /api/portfolio/trigger` - Trigger daily cycle

### SMC Analysis
- `GET /api/smc/analysis` - Run SMC analysis

### Market Regime
- `GET /api/regime/{symbol}` - Detect market regime

## WebSocket

Connect to `ws://localhost:8000/ws` for real-time updates:

```typescript
// Analysis updates
{ type: "analysis_started", task_id: "...", symbol: "XAUUSD" }
{ type: "analysis_completed", task_id: "...", decision: {...} }
{ type: "analysis_error", task_id: "...", error: "..." }
```

## Tech Stack

### Frontend
- **Next.js 14** - React framework
- **shadcn/ui** - UI components
- **Tailwind CSS** - Styling
- **Recharts** - Charts
- **Lucide React** - Icons

### Backend
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

## Development

### Frontend Development

```bash
cd web/frontend
npm run dev     # Start dev server
npm run build   # Build for production
npm run lint    # Run linter
```

### Backend Development

```bash
cd web/backend
uvicorn main:app --reload  # Start with hot reload
```

## Configuration

The frontend proxies API requests to the backend via Next.js rewrites (configured in `next.config.mjs`).

To change the backend URL, modify:
- `web/frontend/next.config.mjs` - API proxy
- `web/frontend/src/lib/api.ts` - API base URL
