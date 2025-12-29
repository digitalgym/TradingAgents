<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework 

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

<div align="center">
<a href="https://www.star-history.com/#TauricResearch/TradingAgents&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" />
   <img alt="TradingAgents Star History" src="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" style="width: 80%; height: auto;" />
 </picture>
</a>
</div>

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.13
conda activate tradingagents
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Required APIs

You will need the OpenAI API for all the agents, and [Alpha Vantage API](https://www.alphavantage.co/support/#api-key) for fundamental and news data (default configuration).

```bash
export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
export ALPHA_VANTAGE_API_KEY=$YOUR_ALPHA_VANTAGE_API_KEY
```

Alternatively, you can create a `.env` file in the project root with your API keys (see `.env.example` for reference):
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

**Note:** We are happy to partner with Alpha Vantage to provide robust API support for TradingAgents. You can get a free AlphaVantage API [here](https://www.alphavantage.co/support/#api-key), TradingAgents-sourced requests also have increased rate limits to 60 requests per minute with no daily limits. Typically the quota is sufficient for performing complex tasks with TradingAgents thanks to Alpha Vantage’s open-source support program. If you prefer to use OpenAI for these data sources instead, you can modify the data vendor settings in `tradingagents/default_config.py`.

### CLI Usage

You can also try out the CLI directly by running:
```bash
python -m cli.main
```
You will see a screen where you can select your desired tickers, date, LLMs, research depth, etc.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### CLI Commands

| Command | Description |
|---------|-------------|
| `python -m cli.main` | Run analysis (default) |
| `python -m cli.main analyze` | Run analysis |
| `python -m cli.main positions` | View/modify MT5 positions and orders |
| `python -m cli.main review` | Re-analyze open trades and get strategy updates |
| `python -m cli.main reflect` | Process closed trades and create memories |

### CLI Commodity Trading

The CLI supports commodity trading with MT5 and xAI Grok:

1. **Step 1: Asset Type** - Select "Commodity" to enable commodity mode
2. **Step 2: Ticker** - Defaults to XAUUSD (Gold). Other options: XAGUSD (Silver), XPTUSD (Platinum)
3. **Step 6: LLM Provider** - Select "xAI (Grok)" for real-time news and X sentiment
4. **Step 8: Data Sources** - Choose MT5 for price data, xAI for news
5. **Step 9: Sentiment** - Choose xAI for X (Twitter) sentiment analysis

When commodity mode is selected:
- Fundamentals Analyst is automatically excluded (commodities don't have company financials)
- Default ticker changes to XAUUSD (Gold)
- MT5 and xAI data vendors become available

**Requirements for commodity trading:**
- MetaTrader 5 terminal installed and logged in
- `pip install MetaTrader5`
- `export XAI_API_KEY=$YOUR_XAI_API_KEY` (for xAI Grok)

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. We utilize `o1-preview` and `gpt-4o` as our deep thinking and fast thinking LLMs for our experiments. However, for testing purposes, we recommend you use `o4-mini` and `gpt-4.1-mini` to save on costs as our framework makes **lots of** API calls.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["quick_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance and Alpha Vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

> The default configuration uses yfinance for stock price and technical data, and Alpha Vantage for fundamental and news data. For production use or if you encounter rate limits, consider upgrading to [Alpha Vantage Premium](https://www.alphavantage.co/premium/) for more stable and reliable data access. For offline experimentation, there's a local data vendor option that uses our **Tauric TradingDB**, a curated dataset for backtesting, though this is still in development. We're currently refining this dataset and plan to release it soon alongside our upcoming projects. Stay tuned!

You can view the full list of configurations in `tradingagents/default_config.py`.

---

## Alternative LLM Providers

### Using xAI Grok

TradingAgents supports xAI's Grok models as an alternative to OpenAI. This includes access to real-time news via web search and X (Twitter) sentiment analysis.

```bash
export XAI_API_KEY=$YOUR_XAI_API_KEY
```

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config.update({
    "llm_provider": "xai",
    "deep_think_llm": "grok-4-1-fast-reasoning",    # Best tool-calling with 2M context
    "quick_think_llm": "grok-4-fast-non-reasoning", # Fast 2M context without reasoning
    "backend_url": "https://api.x.ai/v1",
    # Use xAI for news (web search) and sentiment (X/Twitter)
    "data_vendors": {
        "news_data": "xai",  # Uses Grok's web_search tool
    },
    "tool_vendors": {
        "get_insider_sentiment": "xai",  # Uses Grok's x_search for Twitter sentiment
    },
})

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
```

**Available Grok Models:**
| Model | Context | Description |
|-------|---------|-------------|
| `grok-4-1-fast-reasoning` | 2M | Best tool-calling, reasoning enabled |
| `grok-4-1-fast-non-reasoning` | 2M | Fast without reasoning |
| `grok-4-fast-reasoning` | 2M | Reasoning enabled |
| `grok-4-fast-non-reasoning` | 2M | Fast without reasoning |
| `grok-4-0709` | 256K | Powerful reasoning |
| `grok-code-fast-1` | 256K | Optimized for coding |
| `grok-3-mini` | - | Compact and efficient |

---

## Commodity Trading with MT5

TradingAgents supports commodity trading (Gold, Silver, Platinum, Copper) using MetaTrader 5 for price data.

### Prerequisites
1. Install MetaTrader 5 terminal and login to your broker (e.g., Vantage)
2. Install MT5 Python package: `pip install MetaTrader5`

### Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config.update({
    "data_vendors": {
        "core_stock_apis": "mt5",  # Use MT5 for OHLCV data
    },
    "asset_type": "commodity",  # Auto-excludes fundamentals analyst
})

ta = TradingAgentsGraph(debug=True, config=config)

# Analyze Gold
_, decision = ta.propagate("XAUUSD", "2025-12-26")
```

See `examples/trade_commodities.py` for a complete example with xAI Grok and MT5.

---

## MT5 Trade Execution

TradingAgents can execute trades directly in your MetaTrader 5 account with automatic stop loss and take profit.

### CLI Trade Execution

After running analysis in commodity mode with MT5, you'll be prompted to execute the trade:

```bash
python -m cli.main
```

1. Select **Commodity** → **XAUUSD** → **MT5** for data
2. Complete the analysis
3. When the report shows a BUY/SELL signal, you'll see:

```
🚀 Trade Execution
──────────────────
Trade Signal: SELL

Would you like to execute this trade on MT5?
> Execute SELL order for XAUUSD? (y/N)
```

4. Enter trade parameters:
   - **Lot size**: e.g., 0.01 (micro lot)
   - **Entry price**: Current price or custom limit price
   - **Stop Loss**: Default 2% from entry
   - **Take Profit**: Default 4% from entry
   - **Order type**: Limit or Market

5. Confirm and the order is placed in MT5

### Reviewing Open Trades

Re-analyze your open positions with current market data to get strategy recommendations:

```bash
python -m cli.main review
```

This command:
1. Shows all open MT5 positions
2. Fetches current price and recent price history
3. Analyzes each position with LLM
4. Provides specific recommendations:
   - **HOLD / CLOSE / ADJUST** decision
   - **Stop Loss updates** with exact price levels
   - **Take Profit adjustments** based on market conditions
   - **Risk assessment** (Low/Medium/High)
   - **Key levels to watch**

Example output:
```
═══ Reviewing XAUUSD BUY ═══

Current Price: 2665.70
Entry: 2650.50 | P/L: +0.57%
Distance to SL: 35.70 | Distance to TP: 34.30

📊 Strategy Review: XAUUSD BUY
──────────────────────────────
1. **ADJUST** - Move stop loss to protect profits

2. **Stop Loss Update**: Move SL from 2630.00 to 2655.00 (breakeven + buffer)
   - Price has moved favorably, lock in gains
   
3. **Take Profit Update**: Consider extending TP to 2720.00
   - 5-day high at 2718 suggests room to run
   
4. **Risk Assessment**: LOW
   - Trade is in profit with clear trend continuation
   
5. **Key Levels**: 
   - Support: 2658 (recent swing low)
   - Resistance: 2680 (psychological level)
```

### Managing Open Positions

View and modify your MT5 positions:

```bash
python -m cli.main positions
```

**Available actions:**
- **Modify position SL/TP** - Adjust stop loss or take profit on open positions
- **Close position** - Close at market price
- **Modify pending order** - Change entry price, SL, or TP
- **Cancel pending order** - Remove pending orders

Example:
```
Open Positions:
  1. XAUUSD BUY 0.01 lots @ 2650.50
      SL: 2630.00 | TP: 2700.00 | P/L: +15.20
      Ticket: 123456

Actions:
> Modify position SL/TP

Current XAUUSD price: 2665.70
Current SL: 2630.00 | Current TP: 2700.00

New Stop Loss (blank to keep 2630.00): 2655.00
New Take Profit (blank to keep 2700.00): 2720.00

✅ Position modified!
```

### Python API for Trade Execution

```python
from tradingagents.dataflows.mt5_data import (
    execute_trade_signal,
    place_limit_order,
    modify_position,
    close_position,
    get_open_positions,
)

# Execute based on analysis signal
result = execute_trade_signal(
    symbol="XAUUSD",
    signal="BUY",
    entry_price=2650.00,
    stop_loss=2630.00,
    take_profit=2700.00,
    volume=0.01,
    use_limit_order=True,
)

# Modify an open position
modify_position(ticket=123456, sl=2655.00, tp=2720.00)

# Close a position
close_position(ticket=123456)

# View positions
positions = get_open_positions()
```

---

## Trade Reflection & Learning

When trades close, TradingAgents learns from the outcome to improve future decisions.

### CLI Reflection Command

```bash
python -m cli.main reflect
```

This command:
1. Lists all pending trades (saved when you executed via CLI)
2. You enter the exit price
3. Calculates returns (handles long/short correctly)
4. Creates memory for future learning
5. **Generates improvement suggestions** using LLM analysis

Example output:
```
Processing: XAGUSD SELL
Entry price: 30.50

Enter exit price: 29.80

Trade Result:
  Entry: 30.50
  Exit: 29.80
  Returns: +2.30%

✅ Memory created successfully!

💡 Trade Improvement Suggestions
─────────────────────────────────
1. **Trailing Stop**: A trailing stop of $0.30 would have locked in 
   +3.1% as price briefly hit $29.50 before recovering.

2. **Partial Profits**: Taking 50% off at $29.70 support would have 
   secured gains before the bounce.

3. **Entry Timing**: Waiting for RSI to cross below 30 would have 
   provided a better entry at $30.80 (+0.98% improvement).
```

### How Reflection Works

1. **Trade Execution**: State is saved to `pending_trades/` with full analysis context
2. **Trade Closes**: You run `reflect` and enter exit price
3. **Returns Calculated**: Automatically handles BUY (long) vs SELL (short)
4. **Memory Created**: Lessons stored via `reflect_and_remember()`
5. **Suggestions Generated**: LLM analyzes what could have been improved:
   - Stop loss placement (too tight/loose?)
   - Take profit targets (realistic?)
   - Trailing stop opportunities
   - Entry timing improvements
   - Position sizing recommendations

### Trade Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Analyze   │────▶│   Execute   │────▶│   Reflect   │
│  (report)   │     │  (MT5 order)│     │  (learning) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   BUY/SELL           Order placed        Memory stored
    signal            State saved         Suggestions shown
```

---

## Memory System & Learning from Past Trades

TradingAgents includes a memory system that learns from past trading decisions. When a trade completes, you can reflect on the outcome to store lessons for future analysis.

### Embedding Providers

The memory system uses embeddings to find similar past situations. Configure the embedding provider:

```python
config["use_memory"] = True
config["embedding_provider"] = "local"  # Options: auto, local, openai, ollama
config["local_embedding_model"] = "all-MiniLM-L6-v2"  # For local embeddings
```

- **`local`**: Uses `sentence-transformers` (no API needed, runs locally)
- **`openai`**: Uses OpenAI's embedding API (requires `OPENAI_API_KEY`)
- **`ollama`**: Uses Ollama's local embeddings
- **`auto`**: Uses local for xAI/Grok, OpenAI for others

### Trade Lifecycle with Reflection

The memory system learns by reflecting on completed trades:

```python
from examples.trade_commodities import (
    analyze_commodity, 
    complete_trade, 
    list_pending_trades
)

# STEP 1: Analyze and save state for later reflection
final_state, signal, trade_id = analyze_commodity(
    "XAUUSD", 
    "2025-12-26", 
    entry_price=2650.00,
    save_for_reflection=True
)
# Output: Trade ID: XAUUSD_2025-12-26_143052

# STEP 2: Execute trade based on signal...

# STEP 3: Days/weeks later, when trade closes
list_pending_trades()  # See pending trades

# STEP 4: Complete trade with actual exit price
complete_trade("XAUUSD_2025-12-26_143052", exit_price=2720.00)
# This calculates returns (+2.64%), runs reflection, stores lessons
```

### How Memory Works

1. **Analysis**: `final_state` contains market reports, news, sentiment, and decisions
2. **Execution**: Trade is executed based on signal (BUY/SELL/HOLD)
3. **Outcome**: When trade closes, you provide the actual returns/losses
4. **Reflection**: LLM analyzes what went right/wrong, generates lessons
5. **Storage**: Lessons are stored with situation embedding
6. **Retrieval**: Future similar situations retrieve past lessons via embedding similarity

The lessons are injected into agent prompts as "Learning from Past Mistakes" to improve future decisions.

---

## Data Vendors

TradingAgents supports multiple data vendors for different data types:

| Category | Vendors | Description |
|----------|---------|-------------|
| `core_stock_apis` | yfinance, alpha_vantage, local, **mt5** | OHLCV price data |
| `technical_indicators` | yfinance, alpha_vantage, local | Technical indicators (RSI, MACD, etc.) |
| `fundamental_data` | alpha_vantage, openai, local | Company financials |
| `news_data` | alpha_vantage, google, openai, **xai**, local | Market news |
| `get_insider_sentiment` | local, **xai** | Sentiment analysis (xAI uses X/Twitter) |

Configure in your config:
```python
config["data_vendors"] = {
    "core_stock_apis": "mt5",
    "news_data": "xai",
}
config["tool_vendors"] = {
    "get_insider_sentiment": "xai",
}
```

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better. If you are interested in this line of research, please consider joining our open-source financial AI research community [Tauric Research](https://tauric.ai/).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
