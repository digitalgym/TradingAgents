from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import (
    get_ticker, get_analysis_date, select_analysts, select_research_depth,
    select_shallow_thinking_agent, select_deep_thinking_agent, select_llm_provider,
    select_asset_type, select_data_vendor, select_sentiment_source
)

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback()
def main_callback():
    """Check MT5 connection and AutoTrading status on startup."""
    try:
        from tradingagents.dataflows.mt5_data import check_mt5_autotrading
        
        status = check_mt5_autotrading()
        
        if not status["connected"]:
            console.print(f"[yellow]{status['message']}[/yellow]")
        elif not status["autotrading_enabled"]:
            console.print(f"[bold yellow]{status['message']}[/bold yellow]")
        # Don't print anything if all is OK - keep output clean
        
    except Exception as e:
        # Don't block CLI if MT5 check fails
        pass


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        if any(
            self.report_sections[section]
            for section in [
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Asset Type (Stock or Commodity)
    console.print(
        create_question_box(
            "Step 1: Asset Type", "Select whether you're analyzing stocks or commodities"
        )
    )
    selected_asset_type = select_asset_type()
    is_commodity = selected_asset_type == "commodity"
    
    # Step 2: Ticker symbol
    if is_commodity:
        ticker_prompt = "Enter commodity symbol (XAUUSD=Gold, XAGUSD=Silver, XPTUSD=Platinum)"
        ticker_default = "XAUUSD"
    else:
        ticker_prompt = "Enter the ticker symbol to analyze"
        ticker_default = "SPY"
    
    console.print(
        create_question_box(
            "Step 2: Ticker Symbol", ticker_prompt, ticker_default
        )
    )
    selected_ticker = get_ticker(default=ticker_default)

    # Step 3: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 3: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 4: Select analysts (auto-exclude Fundamentals for commodities)
    console.print(
        create_question_box(
            "Step 4: Analysts Team", 
            "Select your LLM analyst agents for the analysis" + 
            (" (Fundamentals excluded for commodities)" if is_commodity else "")
        )
    )
    selected_analysts = select_analysts()
    
    # Auto-remove Fundamentals analyst for commodities
    if is_commodity:
        selected_analysts = [a for a in selected_analysts if a != AnalystType.FUNDAMENTALS]
        if not selected_analysts:
            # Ensure at least one analyst
            selected_analysts = [AnalystType.MARKET, AnalystType.NEWS]
    
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth
    console.print(
        create_question_box(
            "Step 5: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 6: LLM Provider
    console.print(
        create_question_box(
            "Step 6: LLM Provider", "Select which LLM service to use"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 7: Thinking agents
    console.print(
        create_question_box(
            "Step 7: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Data vendors (for commodities or if user wants to customize)
    data_vendors = {}
    tool_vendors = {}
    
    if is_commodity:
        console.print(
            create_question_box(
                "Step 8: Data Sources", "Select data sources for commodity analysis"
            )
        )
        # Price data vendor
        data_vendors["core_stock_apis"] = select_data_vendor("core_stock_apis", is_commodity=True)
        # News data vendor
        data_vendors["news_data"] = select_data_vendor("news_data", is_commodity=True)
        # Technical indicators - use yfinance for commodities
        data_vendors["technical_indicators"] = "yfinance"
        # Fundamentals not used for commodities
        data_vendors["fundamental_data"] = "openai"
        
        # Sentiment source
        console.print(
            create_question_box(
                "Step 9: Sentiment Source", "Select sentiment data source"
            )
        )
        sentiment_source = select_sentiment_source(selected_llm_provider)
        if sentiment_source == "xai":
            tool_vendors["get_insider_sentiment"] = "xai"

    # Normalize LLM provider name for the graph
    llm_provider_normalized = selected_llm_provider.lower()
    if "xai" in llm_provider_normalized or "grok" in llm_provider_normalized:
        llm_provider_normalized = "xai"
    
    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": llm_provider_normalized,
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "asset_type": selected_asset_type,
        "data_vendors": data_vendors,
        "tool_vendors": tool_vendors,
    }


def get_ticker(default="SPY"):
    """Get ticker symbol from user input."""
    return typer.prompt("", default=default)


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                Markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    Markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    Markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    Markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team Decision",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Portfolio Manager Decision
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        Markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Portfolio Manager Decision",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    
    # Apply asset type and data vendors for commodities
    if selections.get("asset_type") == "commodity":
        config["asset_type"] = "commodity"
        if selections.get("data_vendors"):
            config["data_vendors"] = selections["data_vendors"]
        if selections.get("tool_vendors"):
            config["tool_vendors"] = selections["tool_vendors"]
        # Enable local embeddings for memory (no API needed)
        config["use_memory"] = True
        config["embedding_provider"] = "local"

    # Initialize the graph
    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        # Reset agent statuses
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        # Reset report sections
        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        # Initialize state and get graph args
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # Add message to buffer
                message_buffer.add_message(msg_type, content)                

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    # Set next analyst to in_progress
                    if "fundamentals" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Fundamentals Analyst", "in_progress"
                        )

                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    # Set all research team members to in_progress
                    update_research_team_status("in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            # Update research report with bull's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            # Update research report with bear's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        # Keep all research team members in progress until final decision
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        # Update research report with final decision
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        # Mark all research team members as completed
                        update_research_team_status("completed")
                        # Set first risk analyst to in_progress
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final decision
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final decision only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Decision\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                # Update the display
                update_display(layout)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        # Final update before exiting live display
        update_display(layout)
    
    # Display the complete final report OUTSIDE the Live context to prevent flickering
    display_complete_report(final_state)
    
    # Prompt for trade execution if commodity mode with MT5
    if selections.get("asset_type") == "commodity" and selections.get("data_vendors", {}).get("core_stock_apis") == "mt5":
        prompt_trade_execution(selections["ticker"], decision, final_state)


def prompt_trade_execution(ticker: str, signal: str, final_state: dict):
    """Prompt user to execute the trade via MT5."""
    import questionary
    from tradingagents.dataflows.mt5_data import (
        execute_trade_signal,
        get_mt5_current_price,
        get_mt5_symbol_info,
        get_open_positions,
    )
    
    console.print("\n")
    console.print(Panel(
        f"[bold]Trade Signal: {signal}[/bold]\n\n"
        f"Would you like to execute this trade on MT5?",
        title="🚀 Trade Execution",
        border_style="green" if signal == "BUY" else "red" if signal == "SELL" else "yellow",
    ))
    
    if signal == "HOLD":
        console.print("[yellow]Signal is HOLD - no trade to execute.[/yellow]")
        return
    
    # IMPORTANT: Check for open positions when SELL signal detected
    if signal == "SELL":
        try:
            open_positions = get_open_positions()
            # Check for any BUY positions (longs) that might need covering
            long_positions = [p for p in open_positions if p['type'] == 'BUY']
            
            if long_positions:
                console.print(Panel(
                    "[bold yellow]⚠️ REMINDER: You have open LONG positions![/bold yellow]\n\n"
                    "Before entering a SHORT, consider covering your longs to protect profits.\n\n"
                    "[bold]Open Long Positions:[/bold]",
                    title="🔔 Cover Longs Reminder",
                    border_style="yellow",
                ))
                for p in long_positions:
                    profit_color = "green" if p['profit'] >= 0 else "red"
                    console.print(f"  [{profit_color}]{p['symbol']} BUY {p['volume']} lots @ {p['price_open']} | P/L: {p['profit']:.2f}[/{profit_color}]")
                console.print()
                
                cover_first = questionary.confirm(
                    "Would you like to close/reduce your longs first?",
                    default=True,
                ).ask()
                
                if cover_first:
                    console.print("[cyan]Run 'python -m cli.main positions' to manage your positions.[/cyan]")
                    return
        except Exception as e:
            pass  # Continue if we can't check positions
    
    # Ask user if they want to execute
    execute = questionary.confirm(
        f"Execute {signal} order for {ticker}?",
        default=False,
    ).ask()
    
    if not execute:
        console.print("[yellow]Trade execution skipped.[/yellow]")
        return
    
    try:
        # Get current price and symbol info
        price_info = get_mt5_current_price(ticker)
        symbol_info = get_mt5_symbol_info(ticker)
        
        current_price = price_info["ask"] if signal == "BUY" else price_info["bid"]
        
        console.print(f"\n[cyan]Current {ticker} price: {current_price}[/cyan]")
        console.print(f"[dim]Spread: {symbol_info['spread']} | Min lot: {symbol_info['volume_min']}[/dim]")
        
        # Get trade parameters from user
        console.print("\n[bold]Enter trade parameters:[/bold]")
        
        # Volume
        volume = questionary.text(
            "Lot size (e.g., 0.01, 0.1, 1.0):",
            default="0.01",
        ).ask()
        volume = float(volume)
        
        # Entry price
        entry_input = questionary.text(
            f"Entry price (leave blank for current: {current_price}):",
            default="",
        ).ask()
        entry_price = float(entry_input) if entry_input else current_price
        
        # Stop Loss - offer dynamic ATR-based option
        from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
        
        atr = get_atr_for_symbol(ticker, period=14)
        dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5, risk_reward_ratio=2.0)
        
        # Calculate ATR-based defaults
        if atr > 0:
            atr_levels = dsl.calculate_levels(entry_price, atr, signal)
            default_sl = round(atr_levels.stop_loss, symbol_info["digits"])
            default_tp = round(atr_levels.take_profit, symbol_info["digits"])
            console.print(f"\n[cyan]ATR (14): {atr:.4f} | ATR-based SL: {default_sl} | ATR-based TP: {default_tp}[/cyan]")
        else:
            # Fallback to percentage-based
            if signal == "BUY":
                default_sl = round(entry_price * 0.98, symbol_info["digits"])  # 2% below
                default_tp = round(entry_price * 1.04, symbol_info["digits"])  # 4% above
            else:
                default_sl = round(entry_price * 1.02, symbol_info["digits"])  # 2% above
                default_tp = round(entry_price * 0.96, symbol_info["digits"])  # 4% below
        
        # SL input method
        sl_method = questionary.select(
            "Stop Loss method:",
            choices=[
                f"ATR-based (2x ATR): {default_sl}" if atr > 0 else f"Default (2%): {default_sl}",
                "ATR with custom multiplier",
                "Enter manual price",
            ],
        ).ask()
        
        if "custom multiplier" in sl_method:
            atr_mult = questionary.text(
                "ATR multiplier for SL (e.g., 1.5, 2.0, 2.5):",
                default="2.0",
            ).ask()
            atr_mult = float(atr_mult)
            custom_dsl = DynamicStopLoss(atr_multiplier=atr_mult, risk_reward_ratio=2.0)
            custom_levels = custom_dsl.calculate_levels(entry_price, atr, signal)
            stop_loss = round(custom_levels.stop_loss, symbol_info["digits"])
            console.print(f"[cyan]SL set to {stop_loss} ({atr_mult}x ATR)[/cyan]")
        elif "manual" in sl_method.lower():
            sl_input = questionary.text(
                f"Stop Loss price:",
                default=str(default_sl),
            ).ask()
            stop_loss = float(sl_input) if sl_input else default_sl
        else:
            stop_loss = default_sl
        
        # TP input method
        tp_method = questionary.select(
            "Take Profit method:",
            choices=[
                f"ATR-based (2:1 R:R): {default_tp}" if atr > 0 else f"Default (4%): {default_tp}",
                "Custom risk:reward ratio",
                "Enter manual price",
            ],
        ).ask()
        
        if "custom risk:reward" in tp_method.lower():
            rr_ratio = questionary.text(
                "Risk:Reward ratio (e.g., 1.5, 2.0, 3.0):",
                default="2.0",
            ).ask()
            rr_ratio = float(rr_ratio)
            take_profit = round(dsl.calculate_initial_tp(entry_price, stop_loss, signal, rr_ratio), symbol_info["digits"])
            console.print(f"[cyan]TP set to {take_profit} ({rr_ratio}:1 R:R)[/cyan]")
        elif "manual" in tp_method.lower():
            tp_input = questionary.text(
                f"Take Profit price:",
                default=str(default_tp),
            ).ask()
            take_profit = float(tp_input) if tp_input else default_tp
        else:
            take_profit = default_tp
        
        # Order type
        use_limit = questionary.confirm(
            "Use limit order? (No = market order)",
            default=True,
        ).ask()
        
        # Confirm
        console.print("\n[bold]Order Summary:[/bold]")
        console.print(f"  Symbol: {ticker}")
        console.print(f"  Type: {signal} {'LIMIT' if use_limit else 'MARKET'}")
        console.print(f"  Volume: {volume} lots")
        console.print(f"  Entry: {entry_price}")
        console.print(f"  Stop Loss: {stop_loss}")
        console.print(f"  Take Profit: {take_profit}")
        
        confirm = questionary.confirm(
            "Confirm and place order?",
            default=False,
        ).ask()
        
        if not confirm:
            console.print("[yellow]Order cancelled.[/yellow]")
            return
        
        # Execute the trade
        result = execute_trade_signal(
            symbol=ticker,
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume=volume,
            use_limit_order=use_limit,
            comment=f"TradingAgents {signal}",
        )
        
        if result.get("success"):
            console.print(f"\n[bold green]✅ Order placed successfully![/bold green]")
            console.print(f"  Order ID: {result.get('order_id')}")
            console.print(f"  Price: {result.get('price')}")
            
            # Save trade state for later reflection
            save_trade_state(
                ticker=ticker,
                signal=signal,
                order_id=result.get('order_id'),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                final_state=final_state,
            )
        else:
            console.print(f"\n[bold red]❌ Order failed: {result.get('error')}[/bold red]")
            
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")


def save_trade_state(
    ticker: str,
    signal: str,
    order_id: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    volume: float,
    final_state: dict,
):
    """Save trade state for later reflection when trade closes."""
    import json
    from pathlib import Path
    from datetime import datetime
    from tradingagents.trade_decisions import store_decision, link_decision_to_ticket
    
    trades_dir = Path("pending_trades")
    trades_dir.mkdir(exist_ok=True)
    
    # Extract rationale from final decision
    rationale = final_state.get("final_trade_decision", "")[:500]
    if not rationale:
        rationale = f"{signal} signal from TradingAgents analysis"
    
    # Store decision for tracking
    decision_id = store_decision(
        symbol=ticker,
        decision_type="OPEN",
        action=signal,
        rationale=rationale,
        source="analysis",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volume=volume,
        mt5_ticket=order_id,
        analysis_context={
            "company_of_interest": ticker,
            "trade_date": final_state.get("trade_date", datetime.now().strftime("%Y-%m-%d")),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        },
    )
    
    trade_data = {
        "ticker": ticker,
        "signal": signal,
        "order_id": order_id,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "volume": volume,
        "opened_at": datetime.now().isoformat(),
        "status": "pending",
        "decision_id": decision_id,  # Link to decision tracking
        "final_state": {
            "company_of_interest": ticker,
            "trade_date": final_state.get("trade_date", datetime.now().strftime("%Y-%m-%d")),
            "curr_situation": final_state.get("curr_situation", ""),
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", "N/A"),
            "investment_debate_state": final_state.get("investment_debate_state", {}),
            "risk_debate_state": final_state.get("risk_debate_state", {}),
            "investment_plan": final_state.get("investment_plan", ""),
            "trader_investment_plan": final_state.get("trader_investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        },
    }
    
    # Save to file
    filename = trades_dir / f"trade_{ticker}_{order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(trade_data, f, indent=2, default=str)
    
    console.print(f"\n[dim]Trade state saved to {filename}[/dim]")
    console.print(f"[dim]Decision ID: {decision_id}[/dim]")
    console.print(f"[dim]Close with: python -m cli.main decisions close {decision_id} --exit <price>[/dim]")
    console.print(f"[dim]Run 'python -m cli.main reflect' when trade closes to create memory.[/dim]")


@app.command()
def analyze():
    """Run trading analysis."""
    run_analysis()


@app.command()
def reflect():
    """Process closed trades and create memories for learning."""
    import json
    import questionary
    from pathlib import Path
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.trade_decisions import (
        list_active_decisions,
        close_decision,
        load_decision_context,
    )
    
    # Check both old pending_trades and new decision tracking system
    trades_dir = Path("pending_trades")
    old_trade_files = list(trades_dir.glob("trade_*.json")) if trades_dir.exists() else []
    
    # Get active decisions from new system
    active_decisions = list_active_decisions()
    
    if not old_trade_files and not active_decisions:
        console.print("[yellow]No pending trades or active decisions found.[/yellow]")
        console.print("[dim]Trades are stored when you execute from analysis or act on review recommendations.[/dim]")
        return
    
    # Combine both sources
    all_trades = []
    
    # Add old-style trades
    for trade_file in old_trade_files:
        with open(trade_file, "r", encoding="utf-8") as f:
            trade_data = json.load(f)
        all_trades.append({
            "source": "pending_trades",
            "file": trade_file,
            "data": trade_data,
            "display": f"{trade_data['ticker']} {trade_data['signal']} @ {trade_data['entry_price']}",
            "entry": trade_data['entry_price'],
            "signal": trade_data['signal'],
            "ticker": trade_data['ticker'],
        })
    
    # Add new-style decisions
    for decision in active_decisions:
        all_trades.append({
            "source": "decisions",
            "decision_id": decision["decision_id"],
            "data": decision,
            "display": f"{decision['symbol']} {decision['action']} @ {decision.get('entry_price', 'N/A')}",
            "entry": decision.get("entry_price"),
            "signal": decision.get("action"),
            "ticker": decision["symbol"],
        })
    
    console.print(f"\n[bold]Found {len(all_trades)} trade(s) to reflect on:[/bold]\n")
    
    for i, trade in enumerate(all_trades):
        source_label = "[dim](legacy)[/dim]" if trade["source"] == "pending_trades" else "[cyan](decision)[/cyan]"
        console.print(f"[cyan]{i+1}.[/cyan] {trade['display']} {source_label}")
        if trade["source"] == "decisions":
            d = trade["data"]
            console.print(f"   SL: {d.get('stop_loss', 'N/A')} | TP: {d.get('take_profit', 'N/A')}")
            console.print(f"   Created: {d['created_at']}")
            console.print(f"   Rationale: [dim]{d.get('rationale', '')[:80]}...[/dim]")
        else:
            d = trade["data"]
            console.print(f"   SL: {d['stop_loss']} | TP: {d['take_profit']}")
            console.print(f"   Opened: {d['opened_at']}")
        console.print()
    
    # Ask which trade to process
    trade_num = questionary.text(
        "Enter trade number to process (or 'q' to quit):",
        default="1",
    ).ask()
    
    if trade_num.lower() == 'q':
        return
    
    try:
        idx = int(trade_num) - 1
        selected_trade = all_trades[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid trade number.[/red]")
        return
    
    console.print(f"\n[bold]Processing: {selected_trade['display']}[/bold]")
    
    entry = selected_trade["entry"]
    if entry is None:
        entry_str = questionary.text("Entry price not recorded. Enter entry price:").ask()
        try:
            entry = float(entry_str)
        except ValueError:
            console.print("[red]Invalid price.[/red]")
            return
    
    console.print(f"Entry price: {entry}")
    
    # Try to get exit price from MT5 if we have a ticket
    exit_price = None
    mt5_ticket = selected_trade["data"].get("mt5_ticket")
    
    if mt5_ticket:
        try:
            from tradingagents.dataflows.mt5_data import get_closed_deal_by_ticket
            
            console.print(f"[dim]Checking MT5 history for ticket {mt5_ticket}...[/dim]")
            closed_deal = get_closed_deal_by_ticket(mt5_ticket)
            
            if closed_deal:
                exit_price = closed_deal["price"]
                profit = closed_deal.get("profit", 0)
                close_time = closed_deal.get("time", "")
                
                console.print(f"[green]✓ Found closed deal in MT5:[/green]")
                console.print(f"  Exit Price: {exit_price}")
                console.print(f"  Profit: ${profit:.2f}")
                console.print(f"  Closed: {close_time}")
                
                use_mt5_price = questionary.confirm(
                    f"Use MT5 exit price ({exit_price})?",
                    default=True,
                ).ask()
                
                if not use_mt5_price:
                    exit_price = None  # Will prompt for manual entry
            else:
                console.print("[yellow]Position not found in MT5 closed history (may still be open)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Could not fetch from MT5: {e}[/yellow]")
    
    # Get exit price from user if not from MT5
    if exit_price is None:
        exit_price_str = questionary.text(
            "Enter exit price:",
        ).ask()
        
        try:
            exit_price = float(exit_price_str)
        except ValueError:
            console.print("[red]Invalid price.[/red]")
            return
    
    # Calculate returns
    signal = selected_trade["signal"]
    
    if signal and signal.upper() in ["BUY", "LONG"]:
        returns = ((exit_price - entry) / entry) * 100
    elif signal and signal.upper() in ["SELL", "SHORT"]:
        returns = ((entry - exit_price) / entry) * 100
    else:
        # For HOLD/ADJUST decisions, calculate based on position direction
        returns = ((exit_price - entry) / entry) * 100
    
    console.print(f"\n[bold]Trade Result:[/bold]")
    console.print(f"  Entry: {entry}")
    console.print(f"  Exit: {exit_price}")
    if returns >= 0:
        console.print(f"  Returns: [green]+{returns:.2f}%[/green]")
    else:
        console.print(f"  Returns: [red]{returns:.2f}%[/red]")
    
    # Confirm reflection
    confirm = questionary.confirm(
        "Create memory from this trade?",
        default=True,
    ).ask()
    
    if not confirm:
        console.print("[yellow]Reflection cancelled.[/yellow]")
        return
    
    # Close the decision in the tracking system
    if selected_trade["source"] == "decisions":
        close_decision(
            selected_trade["decision_id"],
            exit_price,
            outcome_notes=f"Reflected with {returns:+.2f}% returns",
        )
        console.print(f"[green]✓ Decision closed: {selected_trade['decision_id']}[/green]")
    
    # Check if we have context for memory creation
    trade_data = selected_trade["data"]
    has_context = False
    
    if selected_trade["source"] == "decisions":
        # Try to load context from decision
        context = load_decision_context(selected_trade["decision_id"])
        if context:
            has_context = True
            final_state = context
        else:
            final_state = {
                "company_of_interest": selected_trade["ticker"],
                "trade_date": trade_data.get("created_at", ""),
                "final_trade_decision": trade_data.get("rationale", ""),
            }
    else:
        # Legacy pending_trades format
        final_state = trade_data.get('final_state', {})
        has_context = bool(final_state)
    
    if has_context:
        # Create graph with memory enabled
        config = DEFAULT_CONFIG.copy()
        config['use_memory'] = True
        config['embedding_provider'] = 'local'
        
        console.print("\n[dim]Initializing memory system...[/dim]")
        
        try:
            graph = TradingAgentsGraph(config=config, debug=False)
            
            # Set the curr_state from saved trade data
            graph.curr_state = final_state
            
            console.print("[dim]Running reflection...[/dim]")
            
            # Run reflection
            graph.reflect_and_remember(returns)
            
            console.print(f"\n[bold green]✅ Memory created successfully![/bold green]")
            console.print(f"[dim]Lessons from this {selected_trade['ticker']} {signal} trade have been stored.[/dim]")
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Error during reflection: {e}[/bold red]")
    else:
        console.print("[yellow]No analysis context available for memory creation.[/yellow]")
        console.print("[dim]Decision outcome has been recorded for statistics.[/dim]")
    
    # Generate trade improvement suggestions
    console.print("\n[dim]Analyzing trade for improvement suggestions...[/dim]")
    suggestions = analyze_trade_improvements(trade_data, exit_price, returns)
    if suggestions:
        console.print(Panel(
            suggestions,
            title="💡 Trade Improvement Suggestions",
            border_style="cyan",
        ))
    
    # Handle legacy pending_trades file
    if selected_trade["source"] == "pending_trades":
        trade_file = selected_trade["file"]
        trade_data['status'] = 'closed'
        trade_data['exit_price'] = exit_price
        trade_data['returns'] = returns
        trade_data['reflected'] = True
        
        with open(trade_file, "w", encoding="utf-8") as f:
            json.dump(trade_data, f, indent=2, default=str)
        
        # Ask if user wants to archive the trade
        archive = questionary.confirm(
            "Archive this trade file?",
            default=True,
        ).ask()
        
        if archive:
            archive_dir = Path("archived_trades")
            archive_dir.mkdir(exist_ok=True)
            trade_file.rename(archive_dir / trade_file.name)
            console.print(f"[dim]Trade archived to {archive_dir / trade_file.name}[/dim]")


@app.command()
def auto_reflect():
    """Automatically process all closed MT5 orders that have active decisions."""
    from tradingagents.trade_decisions import (
        list_active_decisions,
        close_decision,
        load_decision_context,
    )
    from tradingagents.dataflows.mt5_data import get_closed_deal_by_ticket
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    
    console.print("\n[bold]🔄 Auto-Reflect: Processing closed trades...[/bold]\n")
    
    # Get all active decisions
    active_decisions = list_active_decisions()
    
    if not active_decisions:
        console.print("[yellow]No active decisions to process.[/yellow]")
        return
    
    console.print(f"Found {len(active_decisions)} active decision(s). Checking MT5 for closed trades...\n")
    
    closed_count = 0
    reflected_count = 0
    
    for decision in active_decisions:
        mt5_ticket = decision.get("mt5_ticket")
        decision_id = decision["decision_id"]
        symbol = decision["symbol"]
        entry = decision.get("entry_price")
        
        if not mt5_ticket:
            console.print(f"[dim]⏭ {decision_id}: No MT5 ticket, skipping[/dim]")
            continue
        
        # Check if this position is closed in MT5
        try:
            closed_deal = get_closed_deal_by_ticket(mt5_ticket)
            
            if not closed_deal:
                console.print(f"[dim]⏳ {decision_id}: Position still open[/dim]")
                continue
            
            # Found a closed deal!
            exit_price = closed_deal["price"]
            profit = closed_deal.get("profit", 0)
            close_time = closed_deal.get("time", "")
            
            console.print(f"\n[green]✓ {decision_id}[/green]")
            console.print(f"  Symbol: {symbol}")
            console.print(f"  Entry: {entry} → Exit: {exit_price}")
            console.print(f"  Profit: ${profit:.2f}")
            console.print(f"  Closed: {close_time}")
            
            # Calculate returns
            action = decision.get("action", "").upper()
            if entry and entry > 0:
                if action in ["BUY", "LONG"]:
                    returns = ((exit_price - entry) / entry) * 100
                elif action in ["SELL", "SHORT"]:
                    returns = ((entry - exit_price) / entry) * 100
                else:
                    # For HOLD/ADJUST, assume long position
                    returns = ((exit_price - entry) / entry) * 100
            else:
                returns = 0
            
            if returns >= 0:
                console.print(f"  Returns: [green]+{returns:.2f}%[/green]")
            else:
                console.print(f"  Returns: [red]{returns:.2f}%[/red]")
            
            # Close the decision
            close_decision(
                decision_id,
                exit_price,
                outcome_notes=f"Auto-reflected: {returns:+.2f}% returns, ${profit:.2f} profit",
            )
            closed_count += 1
            
            # Try to create memory if we have context
            context = load_decision_context(decision_id)
            if context:
                try:
                    config = DEFAULT_CONFIG.copy()
                    config['use_memory'] = True
                    config['embedding_provider'] = 'local'
                    
                    graph = TradingAgentsGraph(config=config, debug=False)
                    graph.curr_state = context
                    graph.reflect_and_remember(returns)
                    
                    console.print(f"  [green]✓ Memory created[/green]")
                    reflected_count += 1
                except Exception as e:
                    console.print(f"  [yellow]Memory creation failed: {e}[/yellow]")
            else:
                # Create basic memory from decision rationale
                try:
                    config = DEFAULT_CONFIG.copy()
                    config['use_memory'] = True
                    config['embedding_provider'] = 'local'
                    
                    graph = TradingAgentsGraph(config=config, debug=False)
                    graph.curr_state = {
                        "company_of_interest": symbol,
                        "trade_date": decision.get("created_at", ""),
                        "final_trade_decision": decision.get("rationale", ""),
                    }
                    graph.reflect_and_remember(returns)
                    
                    console.print(f"  [green]✓ Memory created (from rationale)[/green]")
                    reflected_count += 1
                except Exception as e:
                    console.print(f"  [dim]No memory created: {e}[/dim]")
            
        except Exception as e:
            console.print(f"[red]✗ {decision_id}: Error - {e}[/red]")
            continue
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Decisions processed: {closed_count}")
    console.print(f"  Memories created: {reflected_count}")
    console.print(f"  Still open: {len(active_decisions) - closed_count}")
    
    if closed_count > 0:
        console.print(f"\n[green]✓ Auto-reflection complete![/green]")
    else:
        console.print(f"\n[dim]No closed trades to process.[/dim]")


def analyze_trade_improvements(trade_data: dict, exit_price: float, returns: float) -> str:
    """Use LLM to analyze trade and suggest improvements."""
    import os
    from openai import OpenAI
    
    # Try xAI first, then OpenAI
    if os.getenv("XAI_API_KEY"):
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        model = "grok-3-mini-fast"
    elif os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = "gpt-4o-mini"
    else:
        return "[dim]No API key available for trade analysis.[/dim]"
    
    entry = trade_data['entry_price']
    sl = trade_data['stop_loss']
    tp = trade_data['take_profit']
    signal = trade_data['signal']
    ticker = trade_data['ticker']
    
    # Calculate key metrics
    sl_distance = abs(entry - sl)
    tp_distance = abs(tp - entry)
    risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
    
    # Determine what happened
    if signal == "BUY":
        hit_sl = exit_price <= sl
        hit_tp = exit_price >= tp
        max_favorable = tp - entry
        actual_move = exit_price - entry
    else:  # SELL
        hit_sl = exit_price >= sl
        hit_tp = exit_price <= tp
        max_favorable = entry - tp
        actual_move = entry - exit_price
    
    prompt = f"""Analyze this closed trade and provide specific, actionable improvement suggestions.

TRADE DETAILS:
- Symbol: {ticker}
- Direction: {signal}
- Entry Price: {entry}
- Stop Loss: {sl} (distance: {sl_distance:.2f})
- Take Profit: {tp} (distance: {tp_distance:.2f})
- Risk/Reward Ratio: {risk_reward:.2f}
- Exit Price: {exit_price}
- Returns: {returns:+.2f}%
- Hit Stop Loss: {hit_sl}
- Hit Take Profit: {hit_tp}

MARKET CONTEXT:
{trade_data['final_state'].get('market_report', 'N/A')[:500]}

ORIGINAL ANALYSIS:
{trade_data['final_state'].get('final_trade_decision', 'N/A')[:500]}

Based on this trade outcome, provide 3-5 specific improvement suggestions. Consider:
1. Stop Loss placement - was it too tight/loose? Should trailing stop have been used?
2. Take Profit placement - was target realistic? Should partial profits have been taken?
3. Position sizing and risk management
4. Entry timing - could a better entry have been achieved?
5. Trade management - when should the trade have been manually adjusted?

Be specific with numbers where possible (e.g., "A trailing stop of 50 pips would have locked in +2.1%").
Keep response concise and actionable."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert trading coach analyzing completed trades to help improve future performance. Be specific and actionable."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[dim]Could not generate suggestions: {e}[/dim]"


@app.command()
def positions():
    """Show open MT5 positions and pending orders with options to modify."""
    import questionary
    from tradingagents.dataflows.mt5_data import (
        get_open_positions,
        get_pending_orders,
        modify_position,
        modify_order,
        close_position,
        cancel_order,
        get_mt5_current_price,
    )
    
    try:
        console.print("\n[bold]Open Positions:[/bold]")
        pos_list = get_open_positions()
        if pos_list:
            for i, p in enumerate(pos_list):
                profit_color = "green" if p['profit'] >= 0 else "red"
                console.print(f"  [cyan]{i+1}.[/cyan] [{profit_color}]{p['symbol']} {p['type']} {p['volume']} lots @ {p['price_open']}[/{profit_color}]")
                console.print(f"      SL: {p['sl']} | TP: {p['tp']} | P/L: {p['profit']:.2f}")
                console.print(f"      Ticket: {p['ticket']}")
        else:
            console.print("  [dim]No open positions[/dim]")
        
        console.print("\n[bold]Pending Orders:[/bold]")
        order_list = get_pending_orders()
        if order_list:
            for i, o in enumerate(order_list):
                console.print(f"  [cyan]{i+1}.[/cyan] {o['symbol']} {o['type']} {o['volume']} lots @ {o['price']}")
                console.print(f"      SL: {o['sl']} | TP: {o['tp']}")
                console.print(f"      Ticket: {o['ticket']}")
        else:
            console.print("  [dim]No pending orders[/dim]")
        
        # Offer actions if there are positions or orders
        if not pos_list and not order_list:
            return
        
        console.print("\n[bold]Actions:[/bold]")
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Modify position SL/TP",
                "Close position",
                "Modify pending order",
                "Cancel pending order",
                "Exit",
            ],
        ).ask()
        
        if action == "Exit" or action is None:
            return
        
        if action == "Modify position SL/TP":
            if not pos_list:
                console.print("[yellow]No open positions to modify.[/yellow]")
                return
            
            ticket_str = questionary.text(
                "Enter position ticket number:",
            ).ask()
            
            try:
                ticket = int(ticket_str)
            except ValueError:
                console.print("[red]Invalid ticket number.[/red]")
                return
            
            # Find the position
            pos = next((p for p in pos_list if p['ticket'] == ticket), None)
            if not pos:
                console.print(f"[red]Position {ticket} not found.[/red]")
                return
            
            # Get current price for reference
            current_price = None
            try:
                price_info = get_mt5_current_price(pos['symbol'])
                current_price = price_info['bid'] if pos['type'] == 'SELL' else price_info['ask']
                console.print(f"\n[cyan]Current {pos['symbol']} price: {current_price}[/cyan]")
            except:
                pass
            
            console.print(f"Entry: {pos['price_open']} | Current SL: {pos['sl']} | Current TP: {pos['tp']}")
            
            # Get ATR for dynamic SL options
            from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
            
            atr = get_atr_for_symbol(pos['symbol'], period=14)
            dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5, risk_reward_ratio=2.0)
            
            new_sl = None
            new_tp = None
            
            # SL modification options
            sl_choices = ["Keep current", "Enter manual price"]
            
            if atr > 0 and current_price:
                # Calculate dynamic SL suggestions
                suggestions = dsl.suggest_stop_adjustment(
                    entry_price=pos['price_open'],
                    current_price=current_price,
                    current_sl=pos['sl'],
                    current_tp=pos['tp'],
                    atr=atr,
                    direction=pos['type'],
                )
                
                console.print(f"[cyan]ATR (14): {atr:.4f}[/cyan]")
                
                if suggestions.get('breakeven'):
                    be_sl = suggestions['breakeven']['new_sl']
                    sl_choices.insert(1, f"Breakeven: {be_sl}")
                
                if suggestions.get('trailing'):
                    trail_sl = suggestions['trailing']['new_sl']
                    sl_choices.insert(1, f"Trailing (1.5x ATR): {trail_sl}")
                
                # ATR-based options
                sl_choices.insert(1, "ATR with custom multiplier")
            
            sl_method = questionary.select(
                "Stop Loss update:",
                choices=sl_choices,
            ).ask()
            
            if sl_method == "Keep current":
                new_sl = None
            elif "Breakeven" in sl_method:
                new_sl = float(sl_method.split(": ")[1])
            elif "Trailing" in sl_method:
                new_sl = float(sl_method.split(": ")[1])
            elif "custom multiplier" in sl_method:
                atr_mult = questionary.text(
                    "ATR multiplier for SL (e.g., 1.0, 1.5, 2.0):",
                    default="1.5",
                ).ask()
                atr_mult = float(atr_mult)
                # Calculate trailing from current price
                if pos['type'] in ["BUY", "LONG"]:
                    new_sl = round(current_price - (atr * atr_mult), 5)
                else:
                    new_sl = round(current_price + (atr * atr_mult), 5)
                console.print(f"[cyan]SL set to {new_sl} ({atr_mult}x ATR from current price)[/cyan]")
            elif sl_method == "Enter manual price":
                sl_input = questionary.text(
                    f"New Stop Loss (blank to keep {pos['sl']}):",
                    default="",
                ).ask()
                new_sl = float(sl_input) if sl_input else None
            
            # TP modification
            tp_method = questionary.select(
                "Take Profit update:",
                choices=[
                    "Keep current",
                    "Custom risk:reward ratio" if new_sl else "Enter manual price",
                    "Enter manual price",
                ],
            ).ask()
            
            if tp_method == "Keep current":
                new_tp = None
            elif "risk:reward" in tp_method.lower() and new_sl:
                rr_ratio = questionary.text(
                    "Risk:Reward ratio (e.g., 1.5, 2.0, 3.0):",
                    default="2.0",
                ).ask()
                rr_ratio = float(rr_ratio)
                new_tp = round(dsl.calculate_initial_tp(pos['price_open'], new_sl, pos['type'], rr_ratio), 5)
                console.print(f"[cyan]TP set to {new_tp} ({rr_ratio}:1 R:R)[/cyan]")
            else:
                tp_input = questionary.text(
                    f"New Take Profit (blank to keep {pos['tp']}):",
                    default="",
                ).ask()
                new_tp = float(tp_input) if tp_input else None
            
            if new_sl is None and new_tp is None:
                console.print("[yellow]No changes made.[/yellow]")
                return
            
            result = modify_position(ticket, sl=new_sl, tp=new_tp)
            
            if result.get("success"):
                console.print(f"[bold green]✅ Position modified![/bold green]")
                console.print(f"  New SL: {result.get('new_sl')}")
                console.print(f"  New TP: {result.get('new_tp')}")
            else:
                console.print(f"[bold red]❌ Failed: {result.get('error')}[/bold red]")
        
        elif action == "Close position":
            if not pos_list:
                console.print("[yellow]No open positions to close.[/yellow]")
                return
            
            ticket_str = questionary.text(
                "Enter position ticket number to close:",
            ).ask()
            
            try:
                ticket = int(ticket_str)
            except ValueError:
                console.print("[red]Invalid ticket number.[/red]")
                return
            
            confirm = questionary.confirm(
                f"Close position {ticket}?",
                default=False,
            ).ask()
            
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return
            
            result = close_position(ticket)
            
            if result.get("success"):
                console.print(f"[bold green]✅ Position closed! P/L: {result.get('profit')}[/bold green]")
            else:
                console.print(f"[bold red]❌ Failed: {result.get('error')}[/bold red]")
        
        elif action == "Modify pending order":
            if not order_list:
                console.print("[yellow]No pending orders to modify.[/yellow]")
                return
            
            ticket_str = questionary.text(
                "Enter order ticket number:",
            ).ask()
            
            try:
                ticket = int(ticket_str)
            except ValueError:
                console.print("[red]Invalid ticket number.[/red]")
                return
            
            # Find the order
            order = next((o for o in order_list if o['ticket'] == ticket), None)
            if not order:
                console.print(f"[red]Order {ticket} not found.[/red]")
                return
            
            console.print(f"Current Price: {order['price']} | SL: {order['sl']} | TP: {order['tp']}")
            
            price_input = questionary.text(
                f"New entry price (blank to keep {order['price']}):",
                default="",
            ).ask()
            
            sl_input = questionary.text(
                f"New Stop Loss (blank to keep {order['sl']}):",
                default="",
            ).ask()
            
            tp_input = questionary.text(
                f"New Take Profit (blank to keep {order['tp']}):",
                default="",
            ).ask()
            
            new_price = float(price_input) if price_input else None
            new_sl = float(sl_input) if sl_input else None
            new_tp = float(tp_input) if tp_input else None
            
            if new_price is None and new_sl is None and new_tp is None:
                console.print("[yellow]No changes made.[/yellow]")
                return
            
            result = modify_order(ticket, price=new_price, sl=new_sl, tp=new_tp)
            
            if result.get("success"):
                console.print(f"[bold green]✅ Order modified![/bold green]")
                console.print(f"  New Price: {result.get('new_price')}")
                console.print(f"  New SL: {result.get('new_sl')}")
                console.print(f"  New TP: {result.get('new_tp')}")
            else:
                console.print(f"[bold red]❌ Failed: {result.get('error')}[/bold red]")
        
        elif action == "Cancel pending order":
            if not order_list:
                console.print("[yellow]No pending orders to cancel.[/yellow]")
                return
            
            ticket_str = questionary.text(
                "Enter order ticket number to cancel:",
            ).ask()
            
            try:
                ticket = int(ticket_str)
            except ValueError:
                console.print("[red]Invalid ticket number.[/red]")
                return
            
            confirm = questionary.confirm(
                f"Cancel order {ticket}?",
                default=False,
            ).ask()
            
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return
            
            result = cancel_order(ticket)
            
            if result.get("success"):
                console.print(f"[bold green]✅ Order cancelled![/bold green]")
            else:
                console.print(f"[bold red]❌ Failed: {result.get('error')}[/bold red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def review():
    """Re-analyze open trades and suggest strategy updates based on current market conditions."""
    import os
    import questionary
    from datetime import datetime
    from openai import OpenAI
    from tradingagents.dataflows.mt5_data import (
        get_open_positions,
        get_mt5_current_price,
        get_mt5_data,
    )
    from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
    
    try:
        pos_list = get_open_positions()
        
        if not pos_list:
            console.print("[yellow]No open positions to review.[/yellow]")
            return
        
        console.print(f"\n[bold]Open Positions ({len(pos_list)}):[/bold]\n")
        
        for i, p in enumerate(pos_list):
            profit_color = "green" if p['profit'] >= 0 else "red"
            console.print(f"  [cyan]{i+1}.[/cyan] [{profit_color}]{p['symbol']} {p['type']} {p['volume']} lots @ {p['price_open']}[/{profit_color}]")
            console.print(f"      SL: {p['sl']} | TP: {p['tp']} | P/L: {p['profit']:.2f}")
            console.print(f"      Ticket: {p['ticket']}")
        
        # Select position to review
        pos_num = questionary.text(
            "\nEnter position number to review (or 'all' for all, 'q' to quit):",
            default="1",
        ).ask()
        
        if pos_num.lower() == 'q':
            return
        
        if pos_num.lower() == 'all':
            positions_to_review = pos_list
        else:
            try:
                idx = int(pos_num) - 1
                positions_to_review = [pos_list[idx]]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection.[/red]")
                return
        
        # Setup LLM client
        if os.getenv("XAI_API_KEY"):
            client = OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
            model = "grok-3-mini-fast"
        elif os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = "gpt-4o-mini"
        else:
            console.print("[red]No API key available (XAI_API_KEY or OPENAI_API_KEY required).[/red]")
            return
        
        for pos in positions_to_review:
            console.print(f"\n[bold cyan]═══ Reviewing {pos['symbol']} {pos['type']} ═══[/bold cyan]")
            
            # Get current market data
            try:
                price_info = get_mt5_current_price(pos['symbol'])
                current_price = price_info['bid'] if pos['type'] == 'SELL' else price_info['ask']
                
                # Get recent price history for context
                from datetime import timedelta
                import csv
                import io
                today = datetime.now()
                start_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
                history_csv = get_mt5_data(pos['symbol'], start_date, end_date)
                
                # Parse CSV to get high/low
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
                
                # Calculate key metrics
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
                continue
            
            console.print(f"\n[dim]Current Price: {current_price}[/dim]")
            console.print(f"[dim]Entry: {entry} | P/L: {current_pnl_pct:+.2f}%[/dim]")
            console.print(f"[dim]Distance to SL: {distance_to_sl:.2f} | Distance to TP: {distance_to_tp:.2f}[/dim]")
            
            # Get ATR and calculate dynamic stop suggestions
            atr = get_atr_for_symbol(pos['symbol'], period=14)
            dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)
            
            atr_suggestions = ""
            if atr > 0:
                suggestions = dsl.suggest_stop_adjustment(
                    entry_price=entry,
                    current_price=current_price,
                    current_sl=sl,
                    current_tp=tp,
                    atr=atr,
                    direction=pos['type'],
                )
                
                console.print(f"\n[bold cyan]📈 ATR-Based Analysis (ATR: {atr:.4f}):[/bold cyan]")
                console.print(f"  [dim]Recommendation: {suggestions['recommendation']}[/dim]")
                
                if suggestions.get('breakeven'):
                    be = suggestions['breakeven']
                    console.print(f"  [green]Breakeven SL: {be['new_sl']}[/green]")
                
                if suggestions.get('trailing'):
                    trail = suggestions['trailing']
                    console.print(f"  [yellow]Trailing SL: {trail['new_sl']} (1.5x ATR from price)[/yellow]")
                
                # Build ATR context for LLM prompt
                atr_suggestions = f"""
ATR-BASED ANALYSIS:
- Current ATR (14-period): {atr:.4f}
- ATR-based Stop Distance: {atr * 2:.4f} (2x ATR)
- ATR-based Trailing Distance: {atr * 1.5:.4f} (1.5x ATR)
- Suggested Breakeven SL: {suggestions.get('breakeven', {}).get('new_sl', 'N/A')}
- Suggested Trailing SL: {suggestions.get('trailing', {}).get('new_sl', 'N/A')}
- System Recommendation: {suggestions['recommendation']}
"""
            
            console.print("\n[dim]Analyzing current market conditions...[/dim]")
            
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
- Stop Loss: {sl} (distance from current: {distance_to_sl:.2f})
- Take Profit: {tp} (distance from current: {distance_to_tp:.2f})
- Risk/Reward Ratio: {risk_reward:.2f}

RECENT PRICE ACTION:
- 5-day High: {recent_high}
- 5-day Low: {recent_low}
- Current vs High: {((current_price - recent_high) / recent_high * 100):+.2f}%
- Current vs Low: {((current_price - recent_low) / recent_low * 100):+.2f}%
{atr_suggestions}
Based on the current market position and price action, provide:

1. **HOLD / CLOSE / ADJUST** - Clear recommendation
2. **Stop Loss Update** - Should SL be moved? To what level and why?
3. **Take Profit Update** - Should TP be adjusted? To what level and why?
4. **Risk Assessment** - Current risk level (Low/Medium/High) and reasoning
5. **Key Levels to Watch** - Important price levels that would change the outlook

Be specific with exact price levels. Consider:
- Moving SL to breakeven if in profit
- Trailing stop opportunities
- Partial profit taking levels
- Key support/resistance from recent price action

Keep response concise and actionable."""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert trade manager analyzing open positions. Provide specific, actionable recommendations with exact price levels."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.7,
                )
                
                analysis = response.choices[0].message.content
                
                console.print(Panel(
                    analysis,
                    title=f"📊 Strategy Review: {pos['symbol']} {pos['type']}",
                    border_style="cyan",
                ))
                
                # Ask if user wants to act on this recommendation
                act_on_review = questionary.confirm(
                    "Act on this recommendation? (stores decision for tracking)",
                    default=False,
                ).ask()
                
                if act_on_review:
                    import re
                    from tradingagents.trade_decisions import store_decision
                    from tradingagents.dataflows.mt5_data import modify_position
                    
                    # Determine action type from analysis
                    analysis_upper = analysis.upper()
                    if "CLOSE" in analysis_upper[:100]:
                        action = "CLOSE"
                        decision_type = "CLOSE"
                    elif "ADJUST" in analysis_upper[:100]:
                        action = "ADJUST"
                        decision_type = "ADJUST"
                    else:
                        action = "HOLD"
                        decision_type = "HOLD"
                    
                    # Try to extract suggested SL/TP values from analysis
                    suggested_sl = None
                    suggested_tp = None
                    
                    # Look for SL patterns like "SL to 5.9315" or "Stop Loss at 5.9315"
                    sl_patterns = [
                        r'(?:SL|stop\s*loss|move\s*sl)\s*(?:to|at|:)?\s*\*?\*?(\d+\.?\d*)',
                        r'breakeven\s*(?:at|:)?\s*\*?\*?(\d+\.?\d*)',
                    ]
                    for pattern in sl_patterns:
                        match = re.search(pattern, analysis, re.IGNORECASE)
                        if match:
                            try:
                                suggested_sl = float(match.group(1))
                                break
                            except ValueError:
                                pass
                    
                    # Look for TP patterns like "TP at 6.0500" or "Take Profit at 6.0500"
                    tp_patterns = [
                        r'(?:TP|take\s*profit|target)\s*(?:to|at|:)?\s*\*?\*?(\d+\.?\d*)',
                    ]
                    for pattern in tp_patterns:
                        match = re.search(pattern, analysis, re.IGNORECASE)
                        if match:
                            try:
                                suggested_tp = float(match.group(1))
                                break
                            except ValueError:
                                pass
                    
                    # Show current and suggested values
                    console.print(f"\n[bold]Current Position:[/bold]")
                    console.print(f"  Entry: {entry} | Current SL: {sl} | Current TP: {tp}")
                    
                    if suggested_sl or suggested_tp:
                        console.print(f"\n[bold cyan]Suggested Values from Analysis:[/bold cyan]")
                        if suggested_sl:
                            console.print(f"  Stop Loss: {suggested_sl}")
                        if suggested_tp:
                            console.print(f"  Take Profit: {suggested_tp}")
                    
                    # Ask how to set values
                    value_choice = questionary.select(
                        "How would you like to set SL/TP?",
                        choices=[
                            "Use suggested values" if (suggested_sl or suggested_tp) else "Skip (no suggested values found)",
                            "Enter manual values",
                            "Mix: choose for each",
                            "Skip SL/TP update",
                        ],
                    ).ask()
                    
                    new_sl = sl
                    new_tp = tp
                    
                    if value_choice == "Use suggested values" and (suggested_sl or suggested_tp):
                        new_sl = suggested_sl if suggested_sl else sl
                        new_tp = suggested_tp if suggested_tp else tp
                    
                    elif value_choice == "Enter manual values":
                        sl_input = questionary.text(
                            f"Stop Loss (current: {sl}, blank to keep):",
                            default="",
                        ).ask()
                        if sl_input:
                            try:
                                new_sl = float(sl_input)
                            except ValueError:
                                console.print("[yellow]Invalid SL, keeping current[/yellow]")
                        
                        tp_input = questionary.text(
                            f"Take Profit (current: {tp}, blank to keep):",
                            default="",
                        ).ask()
                        if tp_input:
                            try:
                                new_tp = float(tp_input)
                            except ValueError:
                                console.print("[yellow]Invalid TP, keeping current[/yellow]")
                    
                    elif value_choice == "Mix: choose for each":
                        # Stop Loss
                        sl_choices = ["Keep current"]
                        if suggested_sl:
                            sl_choices.append(f"Use suggested: {suggested_sl}")
                        sl_choices.append("Enter manual")
                        
                        sl_choice = questionary.select(
                            f"Stop Loss (current: {sl}):",
                            choices=sl_choices,
                        ).ask()
                        
                        if "suggested" in sl_choice.lower():
                            new_sl = suggested_sl
                        elif sl_choice == "Enter manual":
                            sl_input = questionary.text("Enter Stop Loss:").ask()
                            try:
                                new_sl = float(sl_input)
                            except ValueError:
                                console.print("[yellow]Invalid, keeping current[/yellow]")
                        
                        # Take Profit
                        tp_choices = ["Keep current"]
                        if suggested_tp:
                            tp_choices.append(f"Use suggested: {suggested_tp}")
                        tp_choices.append("Enter manual")
                        
                        tp_choice = questionary.select(
                            f"Take Profit (current: {tp}):",
                            choices=tp_choices,
                        ).ask()
                        
                        if "suggested" in tp_choice.lower():
                            new_tp = suggested_tp
                        elif tp_choice == "Enter manual":
                            tp_input = questionary.text("Enter Take Profit:").ask()
                            try:
                                new_tp = float(tp_input)
                            except ValueError:
                                console.print("[yellow]Invalid, keeping current[/yellow]")
                    
                    # Apply changes to MT5 if SL/TP changed
                    if new_sl != sl or new_tp != tp:
                        console.print(f"\n[bold]New Values:[/bold]")
                        console.print(f"  SL: {sl} → {new_sl}")
                        console.print(f"  TP: {tp} → {new_tp}")
                        
                        apply_to_mt5 = questionary.confirm(
                            "Apply these changes to MT5 position?",
                            default=True,
                        ).ask()
                        
                        if apply_to_mt5:
                            try:
                                result = modify_position(
                                    ticket=pos['ticket'],
                                    sl=new_sl,
                                    tp=new_tp,
                                )
                                if result.get("success"):
                                    console.print(f"[green]✓ Position modified in MT5[/green]")
                                else:
                                    console.print(f"[red]✗ MT5 error: {result.get('error')}[/red]")
                            except Exception as e:
                                console.print(f"[red]✗ Error modifying position: {e}[/red]")
                    
                    # Store decision
                    decision_id = store_decision(
                        symbol=pos['symbol'],
                        decision_type=decision_type,
                        action=action,
                        rationale=analysis[:500],
                        source="review",
                        entry_price=entry,
                        stop_loss=new_sl,
                        take_profit=new_tp,
                        mt5_ticket=pos['ticket'],
                        volume=pos['volume'],
                    )
                    
                    console.print(f"\n[green]✓ Decision stored: {decision_id}[/green]")
                    console.print(f"[dim]Close with: python -m cli.main decisions close {decision_id} --exit <price>[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error analyzing position: {e}[/red]")
                continue
        
        console.print("\n[dim]Use 'python -m cli.main positions' to modify SL/TP in MT5.[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def memory_stats(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c",
        help="Specific collection to show stats for (default: all)"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d",
        help="Show detailed statistics including top memories"
    ),
    maintenance: bool = typer.Option(
        False, "--maintenance", "-m",
        help="Run maintenance tasks (prune, dedupe, promote)"
    ),
):
    """
    Display memory system statistics and optionally run maintenance.
    
    Shows tier distribution, confidence scores, and memory health metrics.
    """
    from tradingagents.agents.utils.memory import (
        FinancialSituationMemory, 
        TIER_SHORT, TIER_MID, TIER_LONG
    )
    from tradingagents.agents.utils.memory_maintenance import (
        MemoryMaintenance,
        run_maintenance_on_all_collections
    )
    from tradingagents.default_config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    config["embedding_provider"] = "local"
    
    collection_names = [
        "bull_memory",
        "bear_memory",
        "trader_memory",
        "invest_judge_memory",
        "risk_manager_memory",
        "prediction_accuracy",
        "technical_backtest",
    ]
    
    if collection:
        collection_names = [collection]
    
    if maintenance:
        console.print("\n[bold cyan]Running Memory Maintenance...[/bold cyan]\n")
        
        with console.status("[bold green]Processing..."):
            results = run_maintenance_on_all_collections(config)
        
        for name, result in results.items():
            if "error" in result:
                console.print(f"[red]✗ {name}: {result['error']}[/red]")
            else:
                promo = result.get("promotion", {})
                dedupe = result.get("deduplication", {})
                prune = result.get("pruning", {})
                
                console.print(f"\n[bold green]✓ {name}[/bold green]")
                console.print(f"  Promoted: {promo.get('promoted_to_mid', 0)} to mid, {promo.get('promoted_to_long', 0)} to long")
                console.print(f"  Deduplicated: {dedupe.get('merged', 0)} duplicates merged")
                console.print(f"  Pruned: {prune.get('pruned', 0)} low-quality memories")
        
        console.print("\n[bold]Maintenance complete![/bold]\n")
        return
    
    # Display stats
    console.print("\n[bold cyan]═══ Memory System Statistics ═══[/bold cyan]\n")
    
    total_memories = 0
    all_stats = {}
    
    for name in collection_names:
        try:
            memory = FinancialSituationMemory(name, config)
            maint = MemoryMaintenance(memory)
            
            if detailed:
                stats = maint.get_detailed_stats()
            else:
                stats = memory.get_memory_stats()
            
            all_stats[name] = stats
            total_memories += stats.get("total", 0)
            
        except Exception as e:
            all_stats[name] = {"error": str(e)}
    
    # Create summary table
    table = Table(title="Memory Collections", box=box.ROUNDED)
    table.add_column("Collection", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Short", justify="right", style="yellow")
    table.add_column("Mid", justify="right", style="blue")
    table.add_column("Long", justify="right", style="green")
    table.add_column("Avg Conf", justify="right")
    
    for name, stats in all_stats.items():
        if "error" in stats:
            table.add_row(name, "[red]Error[/red]", "-", "-", "-", "-")
        else:
            tiers = stats.get("by_tier", {})
            avg_conf = stats.get("avg_confidence", 0)
            table.add_row(
                name,
                str(stats.get("total", 0)),
                str(tiers.get(TIER_SHORT, 0)),
                str(tiers.get(TIER_MID, 0)),
                str(tiers.get(TIER_LONG, 0)),
                f"{avg_conf:.2f}" if avg_conf else "-"
            )
    
    console.print(table)
    console.print(f"\n[bold]Total memories across all collections: {total_memories}[/bold]\n")
    
    if detailed:
        # Show detailed stats for each collection
        for name, stats in all_stats.items():
            if "error" in stats or stats.get("total", 0) == 0:
                continue
            
            console.print(f"\n[bold cyan]─── {name} Details ───[/bold cyan]")
            
            # Confidence stats
            conf = stats.get("confidence", {})
            console.print(f"  Confidence: min={conf.get('min', 0):.2f}, max={conf.get('max', 0):.2f}, avg={conf.get('avg', 0):.2f}")
            
            # Outcome quality stats
            oq = stats.get("outcome_quality", {})
            console.print(f"  Outcome Quality: min={oq.get('min', 0):.2f}, max={oq.get('max', 0):.2f}, avg={oq.get('avg', 0):.2f}")
            
            # Age stats
            age = stats.get("age_days", {})
            console.print(f"  Age (days): min={age.get('min', 0)}, max={age.get('max', 0)}, avg={age.get('avg', 0):.1f}")
            
            # Reference counts
            refs = stats.get("reference_counts", {})
            console.print(f"  References: total={refs.get('total', 0)}, max={refs.get('max', 0)}, avg={refs.get('avg', 0):.1f}")
            
            # Prediction accuracy
            pred = stats.get("prediction_accuracy", {})
            correct = pred.get("correct", 0)
            incorrect = pred.get("incorrect", 0)
            total_pred = correct + incorrect
            if total_pred > 0:
                accuracy = (correct / total_pred) * 100
                console.print(f"  Prediction Accuracy: {accuracy:.1f}% ({correct}/{total_pred})")
            
            # Show top memories
            try:
                memory = FinancialSituationMemory(name, config)
                maint = MemoryMaintenance(memory)
                top = maint.get_top_memories(n=3, sort_by="reference_count")
                
                if top:
                    console.print(f"\n  [dim]Top 3 Most Referenced:[/dim]")
                    for i, mem in enumerate(top, 1):
                        console.print(f"    {i}. refs={mem['reference_count']}, conf={mem['confidence']:.2f}, tier={mem['tier']}")
                        console.print(f"       [dim]{mem['document'][:80]}...[/dim]")
            except Exception:
                pass
    
    console.print("\n[dim]Use --detailed for more info, --maintenance to run cleanup[/dim]")


@app.command()
def risk_metrics(
    backtest_file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="Specific backtest results file to analyze"
    ),
    latest: bool = typer.Option(
        True, "--latest/--all", "-l/-a",
        help="Show only latest backtest (default) or all"
    ),
    symbol: Optional[str] = typer.Option(
        None, "--symbol", "-s",
        help="Filter by symbol (e.g., XAUUSD)"
    ),
):
    """
    Display risk metrics from backtest results.
    
    Shows Sharpe ratio, Sortino ratio, max drawdown, VaR, and other
    quantitative risk metrics from historical backtesting.
    """
    import json
    from pathlib import Path
    from tradingagents.risk import RiskMetrics
    
    backtest_dir = Path(__file__).parent.parent / "examples" / "backtest_results"
    
    if not backtest_dir.exists():
        console.print("[yellow]No backtest results found.[/yellow]")
        console.print("[dim]Run 'python examples/backtest_training.py' to generate backtest data.[/dim]")
        return
    
    # Find backtest files
    if backtest_file:
        files = [Path(backtest_file)]
    else:
        pattern = f"{symbol}_*.json" if symbol else "*.json"
        files = sorted(backtest_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not files:
        console.print(f"[yellow]No backtest files found{f' for {symbol}' if symbol else ''}.[/yellow]")
        return
    
    if latest and len(files) > 1:
        files = [files[0]]
    
    console.print("\n[bold cyan]═══ Risk Metrics Analysis ═══[/bold cyan]\n")
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            symbol_name = data.get("symbol", "Unknown")
            timestamp = data.get("timestamp", "Unknown")
            stats = data.get("stats", {})
            risk = data.get("risk_metrics", {})
            
            console.print(f"[bold]{symbol_name}[/bold] - {timestamp}")
            console.print(f"[dim]{filepath.name}[/dim]\n")
            
            # Basic stats
            table = Table(box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            table.add_row("Total Trades", str(stats.get("total_predictions", 0)))
            table.add_row("Accuracy", f"{stats.get('accuracy', 0):.1f}%")
            table.add_row("Total P&L", f"{stats.get('total_hypothetical_pnl', 0):+.2f}%")
            table.add_row("Avg P&L/Trade", f"{stats.get('avg_pnl', 0):+.2f}%")
            
            if stats.get("final_equity"):
                table.add_row("Final Equity", f"${stats.get('final_equity', 100000):,.2f}")
            
            console.print(table)
            
            if risk:
                console.print("\n[bold]Risk Metrics:[/bold]")
                
                risk_table = Table(box=box.SIMPLE)
                risk_table.add_column("Metric", style="cyan")
                risk_table.add_column("Value", justify="right")
                risk_table.add_column("Rating", justify="center")
                
                # Sharpe Ratio
                sharpe = risk.get("sharpe_ratio", 0)
                sharpe_rating = "[green]Excellent[/green]" if sharpe > 2 else "[yellow]Good[/yellow]" if sharpe > 1 else "[red]Poor[/red]"
                risk_table.add_row("Sharpe Ratio", f"{sharpe:.3f}", sharpe_rating)
                
                # Sortino Ratio
                sortino = risk.get("sortino_ratio", 0)
                sortino_rating = "[green]Excellent[/green]" if sortino > 2 else "[yellow]Good[/yellow]" if sortino > 1 else "[red]Poor[/red]"
                risk_table.add_row("Sortino Ratio", f"{sortino:.3f}", sortino_rating)
                
                # Calmar Ratio
                calmar = risk.get("calmar_ratio", 0)
                calmar_rating = "[green]Excellent[/green]" if calmar > 3 else "[yellow]Good[/yellow]" if calmar > 1 else "[red]Poor[/red]"
                risk_table.add_row("Calmar Ratio", f"{calmar:.3f}", calmar_rating)
                
                # Max Drawdown
                max_dd = risk.get("max_drawdown_pct", 0)
                dd_rating = "[green]Low Risk[/green]" if abs(max_dd) < 10 else "[yellow]Moderate[/yellow]" if abs(max_dd) < 20 else "[red]High Risk[/red]"
                risk_table.add_row("Max Drawdown", f"{max_dd:.2f}%", dd_rating)
                
                # VaR
                var_95 = risk.get("var_95", 0) * 100
                var_rating = "[green]Low[/green]" if var_95 < 2 else "[yellow]Moderate[/yellow]" if var_95 < 5 else "[red]High[/red]"
                risk_table.add_row("VaR (95%)", f"{var_95:.2f}%", var_rating)
                
                # Win Rate
                win_rate = risk.get("win_rate", 0)
                win_rating = "[green]Strong[/green]" if win_rate > 60 else "[yellow]Average[/yellow]" if win_rate > 50 else "[red]Weak[/red]"
                risk_table.add_row("Win Rate", f"{win_rate:.1f}%", win_rating)
                
                # Profit Factor
                pf = risk.get("profit_factor", 0)
                pf_rating = "[green]Profitable[/green]" if pf > 1.5 else "[yellow]Marginal[/yellow]" if pf > 1 else "[red]Losing[/red]"
                risk_table.add_row("Profit Factor", f"{pf:.3f}", pf_rating)
                
                # Volatility
                vol = risk.get("volatility", 0) * 100
                risk_table.add_row("Volatility (Ann.)", f"{vol:.2f}%", "")
                
                console.print(risk_table)
            else:
                console.print("[dim]No risk metrics available (older backtest format)[/dim]")
            
            console.print("\n" + "─" * 50 + "\n")
            
        except Exception as e:
            console.print(f"[red]Error reading {filepath}: {e}[/red]")
    
    console.print("[dim]Run 'python examples/backtest_training.py' to generate new backtest data with risk metrics.[/dim]")


@app.command()
def position_size(
    symbol: str = typer.Option(
        "XAUUSD", "--symbol", "-s",
        help="Trading symbol"
    ),
    entry: float = typer.Option(
        ..., "--entry", "-e",
        help="Entry price"
    ),
    stop_loss: float = typer.Option(
        ..., "--stop", "-sl",
        help="Stop loss price"
    ),
    direction: str = typer.Option(
        "BUY", "--direction", "-d",
        help="Trade direction: BUY or SELL"
    ),
    balance: float = typer.Option(
        100000, "--balance", "-b",
        help="Account balance"
    ),
    risk_pct: float = typer.Option(
        0.02, "--risk", "-r",
        help="Max risk per trade (0.02 = 2%)"
    ),
    confidence: float = typer.Option(
        1.0, "--confidence", "-c",
        help="Trade confidence (0-1), reduces size if < 1"
    ),
    use_history: bool = typer.Option(
        True, "--history/--no-history",
        help="Use backtest history for Kelly sizing"
    ),
):
    """
    Calculate optimal position size for a trade.
    
    Uses Kelly criterion if backtest history available, otherwise fixed fractional.
    
    Example:
        python -m cli.main position-size -e 2650 -sl 2630 -d BUY
    """
    import json
    from pathlib import Path
    from tradingagents.risk import (
        PositionSizer, 
        calculate_kelly_from_history,
        recommend_position_size
    )
    
    # Validate direction
    direction = direction.upper()
    if direction not in ["BUY", "SELL"]:
        console.print("[red]Direction must be BUY or SELL[/red]")
        return
    
    # Validate stop loss
    if direction == "BUY" and stop_loss >= entry:
        console.print("[red]Stop loss must be below entry for BUY[/red]")
        return
    if direction == "SELL" and stop_loss <= entry:
        console.print("[red]Stop loss must be above entry for SELL[/red]")
        return
    
    console.print(f"\n[bold cyan]═══ Position Size Calculator ═══[/bold cyan]\n")
    
    # Try to load trade history from backtest results
    trade_history = None
    if use_history:
        backtest_dir = Path(__file__).parent.parent / "examples" / "backtest_results"
        if backtest_dir.exists():
            files = sorted(
                backtest_dir.glob(f"{symbol}_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if files:
                try:
                    with open(files[0], 'r') as f:
                        data = json.load(f)
                    # Extract trade returns from results
                    results = data.get("results", [])
                    if results:
                        trade_history = [r["hypothetical_pnl"] / 100 for r in results]
                        console.print(f"[dim]Using {len(trade_history)} trades from {files[0].name}[/dim]\n")
                except Exception:
                    pass
    
    # Calculate Kelly parameters if history available
    kelly_info = None
    if trade_history and len(trade_history) >= 10:
        win_rate, avg_win, avg_loss, kelly = calculate_kelly_from_history(trade_history)
        kelly_info = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "kelly_fraction": kelly
        }
    
    # Create sizer and calculate
    sizer = PositionSizer(
        account_balance=balance,
        max_risk_per_trade=risk_pct
    )
    
    if kelly_info and kelly_info["kelly_fraction"] > 0.05:
        # Use Kelly if we have a meaningful edge
        result = sizer.kelly_size(
            win_rate=kelly_info["win_rate"],
            avg_win=kelly_info["avg_win"],
            avg_loss=kelly_info["avg_loss"],
            entry_price=entry,
            stop_loss=stop_loss,
            confidence=confidence
        )
    else:
        # Fall back to fixed fractional
        if kelly_info and kelly_info["kelly_fraction"] <= 0.05:
            console.print("[yellow]Kelly suggests no edge - using fixed fractional instead[/yellow]\n")
        result = sizer.fixed_fractional_size(
            entry_price=entry,
            stop_loss=stop_loss,
            risk_percent=risk_pct,
            confidence=confidence
        )
    
    # Display input parameters
    table = Table(title="Trade Parameters", box=box.SIMPLE)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Symbol", symbol)
    table.add_row("Direction", f"[green]{direction}[/green]" if direction == "BUY" else f"[red]{direction}[/red]")
    table.add_row("Entry Price", f"${entry:,.2f}")
    table.add_row("Stop Loss", f"${stop_loss:,.2f}")
    table.add_row("Risk per Unit", f"${abs(entry - stop_loss):,.2f}")
    table.add_row("Account Balance", f"${balance:,.2f}")
    table.add_row("Max Risk %", f"{risk_pct*100:.1f}%")
    if confidence < 1.0:
        table.add_row("Confidence", f"{confidence*100:.0f}%")
    
    console.print(table)
    
    # Display Kelly info if available
    if kelly_info:
        console.print("\n[bold]Historical Performance (Kelly Input):[/bold]")
        kelly_table = Table(box=box.SIMPLE)
        kelly_table.add_column("Metric", style="cyan")
        kelly_table.add_column("Value", justify="right")
        
        kelly_table.add_row("Win Rate", f"{kelly_info['win_rate']*100:.1f}%")
        kelly_table.add_row("Avg Win", f"{kelly_info['avg_win']*100:.2f}%")
        kelly_table.add_row("Avg Loss", f"{kelly_info['avg_loss']*100:.2f}%")
        kelly_table.add_row("Raw Kelly %", f"{kelly_info['kelly_fraction']*100:.1f}%")
        kelly_table.add_row("Half-Kelly %", f"{kelly_info['kelly_fraction']*50:.1f}%")
        
        console.print(kelly_table)
    
    # Display recommendation
    console.print("\n[bold green]═══ RECOMMENDATION ═══[/bold green]\n")
    
    rec_table = Table(box=box.ROUNDED)
    rec_table.add_column("", style="bold")
    rec_table.add_column("Value", justify="right", style="green")
    
    rec_table.add_row("Method", result.method.replace("_", " ").title())
    rec_table.add_row("Position Size", f"{result.recommended_size:.4f} units")
    rec_table.add_row("Position Value", f"${result.position_value:,.2f}")
    rec_table.add_row("Risk Amount", f"${result.risk_amount:,.2f}")
    rec_table.add_row("Risk %", f"{result.risk_percent*100:.2f}%")
    
    # Calculate lots for MT5
    lots = sizer.calculate_lots(result.recommended_size, contract_size=100)
    rec_table.add_row("MT5 Lots", f"{lots:.2f}")
    
    # Calculate take profit (2:1 risk/reward)
    risk_distance = abs(entry - stop_loss)
    if direction == "BUY":
        tp_2r = entry + (risk_distance * 2)
        tp_3r = entry + (risk_distance * 3)
    else:
        tp_2r = entry - (risk_distance * 2)
        tp_3r = entry - (risk_distance * 3)
    
    rec_table.add_row("Take Profit (2R)", f"${tp_2r:,.2f}")
    rec_table.add_row("Take Profit (3R)", f"${tp_3r:,.2f}")
    
    console.print(rec_table)
    
    # Risk assessment
    console.print("\n[bold]Risk Assessment:[/bold]")
    if result.risk_percent <= 0.01:
        console.print("  [green]✓ Conservative position size (≤1% risk)[/green]")
    elif result.risk_percent <= 0.02:
        console.print("  [yellow]● Moderate position size (1-2% risk)[/yellow]")
    else:
        console.print("  [red]⚠ Aggressive position size (>2% risk)[/red]")
    
    if kelly_info and kelly_info["kelly_fraction"] < 0.1:
        console.print("  [yellow]⚠ Low Kelly fraction suggests marginal edge[/yellow]")
    
    if confidence < 0.7:
        console.print("  [yellow]● Low confidence - consider smaller size or skip[/yellow]")
    
    console.print("")


@app.command()
def trailing_stops():
    """Monitor and update trailing stops on open positions based on ATR."""
    import questionary
    from tradingagents.dataflows.mt5_data import (
        get_open_positions,
        get_mt5_current_price,
        modify_position,
    )
    from tradingagents.risk.stop_loss import DynamicStopLoss, get_atr_for_symbol
    
    try:
        pos_list = get_open_positions()
        
        if not pos_list:
            console.print("[yellow]No open positions to monitor.[/yellow]")
            return
        
        console.print(f"\n[bold]🔄 Trailing Stop Analysis ({len(pos_list)} positions)[/bold]\n")
        
        dsl = DynamicStopLoss(atr_multiplier=2.0, trailing_multiplier=1.5)
        updates_available = []
        
        for pos in pos_list:
            symbol = pos['symbol']
            entry = pos['price_open']
            current_sl = pos['sl']
            current_tp = pos['tp']
            direction = pos['type']
            ticket = pos['ticket']
            
            # Get current price
            try:
                price_info = get_mt5_current_price(symbol)
                current_price = price_info['bid'] if direction == 'SELL' else price_info['ask']
            except Exception as e:
                console.print(f"[red]Error getting price for {symbol}: {e}[/red]")
                continue
            
            # Get ATR
            atr = get_atr_for_symbol(symbol, period=14)
            if atr <= 0:
                console.print(f"[dim]{symbol}: Could not get ATR, skipping[/dim]")
                continue
            
            # Calculate suggestions
            suggestions = dsl.suggest_stop_adjustment(
                entry_price=entry,
                current_price=current_price,
                current_sl=current_sl,
                current_tp=current_tp,
                atr=atr,
                direction=direction,
            )
            
            pnl_pct = suggestions['pnl_percent']
            pnl_color = "green" if pnl_pct >= 0 else "red"
            
            console.print(f"[bold]{symbol} {direction}[/bold] (Ticket: {ticket})")
            console.print(f"  Entry: {entry} | Current: {current_price} | P/L: [{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]")
            console.print(f"  Current SL: {current_sl} | ATR: {atr:.4f}")
            
            # Check for updates
            new_sl = None
            update_reason = ""
            
            if suggestions.get('breakeven') and current_sl == 0:
                be = suggestions['breakeven']
                new_sl = be['new_sl']
                update_reason = "Move to breakeven"
                console.print(f"  [green]→ Breakeven SL available: {new_sl}[/green]")
            elif suggestions.get('trailing'):
                trail = suggestions['trailing']
                new_sl = trail['new_sl']
                update_reason = f"Trail SL (1.5x ATR)"
                console.print(f"  [yellow]→ Trailing SL available: {new_sl}[/yellow]")
            else:
                console.print(f"  [dim]→ No update needed[/dim]")
            
            if new_sl:
                updates_available.append({
                    'ticket': ticket,
                    'symbol': symbol,
                    'direction': direction,
                    'current_sl': current_sl,
                    'new_sl': new_sl,
                    'reason': update_reason,
                })
            
            console.print()
        
        # Offer to apply updates
        if updates_available:
            console.print(f"\n[bold cyan]═══ {len(updates_available)} Update(s) Available ═══[/bold cyan]\n")
            
            for i, upd in enumerate(updates_available):
                console.print(f"  {i+1}. {upd['symbol']} {upd['direction']}: SL {upd['current_sl']} → {upd['new_sl']} ({upd['reason']})")
            
            apply_choice = questionary.select(
                "\nApply updates?",
                choices=[
                    "Apply all",
                    "Select individually",
                    "Skip all",
                ],
            ).ask()
            
            if apply_choice == "Apply all":
                for upd in updates_available:
                    result = modify_position(upd['ticket'], sl=upd['new_sl'])
                    if result.get("success"):
                        console.print(f"[green]✓ {upd['symbol']}: SL updated to {upd['new_sl']}[/green]")
                    else:
                        console.print(f"[red]✗ {upd['symbol']}: {result.get('error')}[/red]")
            
            elif apply_choice == "Select individually":
                for upd in updates_available:
                    apply = questionary.confirm(
                        f"Update {upd['symbol']} SL to {upd['new_sl']}?",
                        default=True,
                    ).ask()
                    
                    if apply:
                        result = modify_position(upd['ticket'], sl=upd['new_sl'])
                        if result.get("success"):
                            console.print(f"[green]✓ Updated[/green]")
                        else:
                            console.print(f"[red]✗ {result.get('error')}[/red]")
            else:
                console.print("[dim]No updates applied.[/dim]")
        else:
            console.print("[dim]No trailing stop updates available.[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def decisions(
    action: str = typer.Argument("list", help="Action: list, close, stats, cancel"),
    decision_id: Optional[str] = typer.Argument(None, help="Decision ID for close/cancel"),
    exit_price: Optional[float] = typer.Option(None, "--exit", "-e", help="Exit price for closing"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
):
    """
    Manage trade decisions and track outcomes.
    
    Actions:
    - list: Show active decisions
    - close: Close a decision with exit price
    - stats: Show decision statistics
    - cancel: Cancel an active decision
    
    Examples:
        python -m cli.main decisions list
        python -m cli.main decisions close XAUUSD_20260106_120000 --exit 4500.00
        python -m cli.main decisions stats --symbol XAUUSD
    """
    from tradingagents.trade_decisions import (
        list_active_decisions,
        list_closed_decisions,
        close_decision,
        cancel_decision,
        get_decision_stats,
        load_decision,
    )
    
    if action == "list":
        active = list_active_decisions(symbol)
        
        if not active:
            console.print("[dim]No active decisions.[/dim]")
            return
        
        console.print(f"\n[bold]Active Decisions ({len(active)}):[/bold]\n")
        
        for d in active:
            console.print(f"[cyan]{d['decision_id']}[/cyan]")
            console.print(f"  {d['symbol']} {d['action']} @ {d.get('entry_price', 'market')}")
            console.print(f"  Type: {d['decision_type']} | Source: {d['source']}")
            console.print(f"  SL: {d.get('stop_loss', 'N/A')} | TP: {d.get('take_profit', 'N/A')}")
            console.print(f"  Created: {d['created_at']}")
            if d.get('mt5_ticket'):
                console.print(f"  MT5 Ticket: {d['mt5_ticket']}")
            console.print(f"  [dim]{d['rationale'][:100]}...[/dim]\n")
    
    elif action == "close":
        if not decision_id:
            console.print("[red]Please provide a decision ID to close.[/red]")
            return
        if exit_price is None:
            import questionary
            exit_str = questionary.text("Enter exit price:").ask()
            try:
                exit_price = float(exit_str)
            except ValueError:
                console.print("[red]Invalid exit price.[/red]")
                return
        
        try:
            result = close_decision(decision_id, exit_price)
            pnl_color = "green" if result.get("pnl_percent", 0) >= 0 else "red"
            console.print(f"\n[{pnl_color}]P&L: {result.get('pnl_percent', 0):+.2f}%[/{pnl_color}]")
        except FileNotFoundError:
            console.print(f"[red]Decision not found: {decision_id}[/red]")
    
    elif action == "cancel":
        if not decision_id:
            console.print("[red]Please provide a decision ID to cancel.[/red]")
            return
        
        import questionary
        reason = questionary.text("Reason for cancellation:").ask()
        cancel_decision(decision_id, reason or "")
    
    elif action == "stats":
        stats = get_decision_stats(symbol)
        
        console.print(f"\n[bold cyan]═══ Decision Statistics{f' ({symbol})' if symbol else ''} ═══[/bold cyan]\n")
        
        if stats["total_decisions"] == 0:
            console.print("[dim]No closed decisions yet.[/dim]")
            return
        
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Decisions", str(stats["total_decisions"]))
        table.add_row("Correct Decisions", str(stats["correct_decisions"]))
        
        rate = stats["correct_rate"] * 100
        rate_color = "green" if rate >= 50 else "red"
        table.add_row("Correct Rate", f"[{rate_color}]{rate:.1f}%[/{rate_color}]")
        
        avg_pnl = stats["avg_pnl_percent"]
        pnl_color = "green" if avg_pnl >= 0 else "red"
        table.add_row("Avg P&L %", f"[{pnl_color}]{avg_pnl:+.2f}%[/{pnl_color}]")
        
        total_pnl = stats["total_pnl"]
        total_color = "green" if total_pnl >= 0 else "red"
        table.add_row("Total P&L", f"[{total_color}]${total_pnl:+.2f}[/{total_color}]")
        
        console.print(table)
        
        if stats.get("best_decision"):
            best = stats["best_decision"]
            console.print(f"\n[green]Best: {best['decision_id']} ({best['pnl_percent']:+.2f}%)[/green]")
        
        if stats.get("worst_decision"):
            worst = stats["worst_decision"]
            console.print(f"[red]Worst: {worst['decision_id']} ({worst['pnl_percent']:+.2f}%)[/red]")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("[dim]Valid actions: list, close, stats, cancel[/dim]")


if __name__ == "__main__":
    app()
