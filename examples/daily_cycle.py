"""
Daily Analysis Cycle with 24-Hour Retrospective Evaluation

This script runs a continuous 24-hour analysis cycle that:
1. Evaluates yesterday's predictions against actual price movements
2. Runs new analysis and extracts predictions
3. Stores predictions for tomorrow's evaluation
4. Repeats every 24 hours

The goal is to refine analysis accuracy over time by learning from
prediction successes and failures.

Usage:
    python examples/daily_cycle.py --symbol XAUUSD
    python examples/daily_cycle.py --symbol XAUUSD --run-once  # Single cycle, no scheduling
    python examples/daily_cycle.py --symbol XAUUSD --run-at 18  # Run at 6 PM daily
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.memory import (
    FinancialSituationMemory,
    TIER_SHORT,
    TIER_MID,
    TIER_LONG
)


# =============================================================================
# CONFIGURATION
# =============================================================================

from tradingagents.default_config import DEFAULT_CONFIG

CYCLE_CONFIG = DEFAULT_CONFIG.copy()
CYCLE_CONFIG.update({
    "data_vendors": {
        "core_stock_apis": "mt5",
        "technical_indicators": "mt5",  # Use MT5 for commodities (yfinance doesn't have XAUUSD)
        "fundamental_data": "openai",
        "news_data": "xai",
    },
    "tool_vendors": {
        "get_insider_sentiment": "xai",
    },
    "asset_type": "commodity",
    "llm_provider": "xai",
    "deep_think_llm": "grok-3-fast",
    "quick_think_llm": "grok-3-fast",
    "backend_url": "https://api.x.ai/v1",
    "use_memory": True,
    "embedding_provider": "local",
})

# Directories for persistence
PREDICTIONS_DIR = Path(__file__).parent / "pending_predictions"
EVALUATIONS_DIR = Path(__file__).parent / "evaluation_history"
LAST_RUN_FILE = Path(__file__).parent / ".last_run_state.json"

# Minimum hours between analyses to prevent duplicates on restart
MIN_HOURS_BETWEEN_RUNS = 12


# =============================================================================
# PREDICTION EXTRACTION
# =============================================================================

def extract_prediction(
    final_state: Dict[str, Any], 
    signal: str, 
    current_price: float,
    evaluation_hours: int = 24,
    analysis_time: datetime = None
) -> Dict[str, Any]:
    """
    Extract structured prediction from analysis state.
    
    Parses the final_trade_decision to determine expected direction.
    
    Args:
        final_state: The analysis state from graph.propagate()
        signal: BUY/SELL/HOLD signal
        current_price: Price at time of analysis
        evaluation_hours: Hours until evaluation (default 24, can be 72 for swing trades)
        analysis_time: Override analysis timestamp (for backtesting)
    """
    decision = final_state.get("final_trade_decision", "")
    
    # Determine expected direction from signal
    if signal.upper() == "BUY":
        expected_direction = "up"
    elif signal.upper() == "SELL":
        expected_direction = "down"
    else:
        expected_direction = "sideways"
    
    # Extract key factors from reports (simplified - first 300 chars of each)
    key_factors = []
    if final_state.get("market_report"):
        key_factors.append(f"Market: {final_state['market_report'][:300]}")
    if final_state.get("news_report"):
        key_factors.append(f"News: {final_state['news_report'][:300]}")
    if final_state.get("sentiment_report"):
        key_factors.append(f"Sentiment: {final_state['sentiment_report'][:300]}")
    
    # Use provided analysis_time or current time
    ts = analysis_time or datetime.now()
    
    return {
        "signal": signal.upper(),
        "expected_direction": expected_direction,
        "key_factors": key_factors,
        "price_at_analysis": current_price,
        "analysis_timestamp": ts.isoformat(),
        "evaluation_due": (ts + timedelta(hours=evaluation_hours)).isoformat(),
        "evaluation_hours": evaluation_hours,
        "final_decision": decision[:1000],  # Truncate for storage
        "full_state_summary": {
            "market_report": final_state.get("market_report", "")[:500],
            "news_report": final_state.get("news_report", "")[:500],
            "sentiment_report": final_state.get("sentiment_report", "")[:500],
            "investment_plan": final_state.get("investment_plan", "")[:500],
        }
    }


def get_current_price(symbol: str) -> float:
    """Fetch current price from MT5."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get tick for {symbol}")
        
        return (tick.bid + tick.ask) / 2  # Mid price
    except Exception as e:
        print(f"Warning: Could not get live price: {e}")
        return 0.0


# =============================================================================
# PREDICTION STORAGE
# =============================================================================

def save_prediction(symbol: str, prediction: Dict[str, Any], final_state: Dict[str, Any]):
    """Save prediction for later evaluation."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timestamp}.pkl"
    filepath = PREDICTIONS_DIR / filename
    
    data = {
        "symbol": symbol,
        "prediction": prediction,
        "final_state": final_state,
        "status": "pending",
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Prediction saved: {filepath.name}")
    print(f"   Signal: {prediction['signal']}")
    print(f"   Expected: {prediction['expected_direction']}")
    print(f"   Price: ${prediction['price_at_analysis']:.2f}")
    print(f"   Evaluation due: {prediction['evaluation_due']}")
    
    return filepath


def load_pending_predictions(symbol: str = None) -> list:
    """Load all pending predictions, optionally filtered by symbol."""
    if not PREDICTIONS_DIR.exists():
        return []
    
    pending = []
    for f in PREDICTIONS_DIR.glob("*.pkl"):
        with open(f, "rb") as file:
            data = pickle.load(file)
        
        if data["status"] != "pending":
            continue
        
        if symbol and data["symbol"] != symbol:
            continue
        
        # Check if evaluation is due
        eval_due = datetime.fromisoformat(data["prediction"]["evaluation_due"])
        if datetime.now() >= eval_due:
            data["filepath"] = f
            pending.append(data)
    
    return pending


# =============================================================================
# COMPARATIVE EVALUATION
# =============================================================================

def evaluate_prediction(
    prediction_data: Dict[str, Any],
    current_price: float,
    graph: TradingAgentsGraph
) -> Dict[str, Any]:
    """
    Evaluate a prediction against actual price movement.
    
    Compares predicted direction vs actual direction and generates lessons.
    """
    prediction = prediction_data["prediction"]
    final_state = prediction_data["final_state"]
    symbol = prediction_data["symbol"]
    
    price_at_analysis = prediction["price_at_analysis"]
    expected_direction = prediction["expected_direction"]
    signal = prediction["signal"]
    
    # Calculate actual movement
    price_change = current_price - price_at_analysis
    pct_change = (price_change / price_at_analysis) * 100 if price_at_analysis > 0 else 0
    
    # Determine actual direction (0.1% threshold)
    if pct_change > 0.1:
        actual_direction = "up"
    elif pct_change < -0.1:
        actual_direction = "down"
    else:
        actual_direction = "sideways"
    
    # Was prediction correct?
    prediction_correct = (expected_direction == actual_direction)
    
    # Calculate hypothetical P&L
    if signal == "BUY":
        hypothetical_pnl = pct_change
    elif signal == "SELL":
        hypothetical_pnl = -pct_change  # Profit on shorts when price goes down
    else:
        hypothetical_pnl = 0  # HOLD = no position
    
    evaluation = {
        "symbol": symbol,
        "analysis_timestamp": prediction["analysis_timestamp"],
        "evaluation_timestamp": datetime.now().isoformat(),
        "price_at_analysis": price_at_analysis,
        "price_at_evaluation": current_price,
        "pct_change": pct_change,
        "expected_direction": expected_direction,
        "actual_direction": actual_direction,
        "signal": signal,
        "prediction_correct": prediction_correct,
        "hypothetical_pnl": hypothetical_pnl,
    }
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {symbol}")
    print(f"{'='*60}")
    print(f"Analysis time: {prediction['analysis_timestamp']}")
    print(f"Signal: {signal} (expected {expected_direction})")
    print(f"Price then: ${price_at_analysis:.2f}")
    print(f"Price now:  ${current_price:.2f}")
    print(f"Change: {pct_change:+.2f}% ({actual_direction})")
    print(f"Prediction: {'CORRECT' if prediction_correct else 'INCORRECT'}")
    print(f"Hypothetical P&L: {hypothetical_pnl:+.2f}%")
    
    # Generate comparative lesson using LLM
    lesson = generate_comparative_lesson(prediction, evaluation, graph)
    evaluation["lesson"] = lesson
    
    # Store lesson in memory
    store_lesson(prediction, evaluation, lesson, graph, final_state)
    
    return evaluation


def generate_comparative_lesson(
    prediction: Dict[str, Any],
    evaluation: Dict[str, Any],
    graph: TradingAgentsGraph
) -> str:
    """
    Use LLM to generate a comparative analysis lesson.
    
    Analyzes what the prediction got right/wrong and why.
    """
    prompt = f"""You are analyzing a trading prediction to extract lessons for future improvement.

PREDICTION MADE:
- Signal: {prediction['signal']}
- Expected Direction: {prediction['expected_direction']}
- Price at Analysis: ${prediction['price_at_analysis']:.2f}
- Key Reasoning: {prediction['final_decision'][:500]}

ACTUAL OUTCOME (24 hours later):
- Actual Direction: {evaluation['actual_direction']}
- Price Change: {evaluation['pct_change']:+.2f}%
- Current Price: ${evaluation['price_at_evaluation']:.2f}
- Prediction Correct: {evaluation['prediction_correct']}

KEY FACTORS FROM ANALYSIS:
{chr(10).join(prediction['key_factors'][:3])}

TASK:
1. Analyze whether the prediction was correct and why
2. Identify which factors in the analysis were predictive vs misleading
3. Provide a concise lesson (2-3 sentences) that can improve future predictions
4. Focus on actionable insights, not just observations

Format your response as:
ANALYSIS: [Your analysis of what happened]
PREDICTIVE FACTORS: [Factors that correctly predicted the outcome]
MISLEADING FACTORS: [Factors that led to incorrect conclusions]
LESSON: [Concise, actionable lesson for future predictions]
"""

    try:
        response = graph.quick_thinking_llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Warning: Failed to generate lesson: {e}")
        return f"Prediction was {'correct' if evaluation['prediction_correct'] else 'incorrect'}. Price moved {evaluation['pct_change']:+.2f}%."


def determine_tier_from_evaluation(evaluation: Dict[str, Any]) -> str:
    """
    Determine the appropriate memory tier based on the evaluation result.
    
    High-impact predictions (large moves, correct predictions) go to higher tiers.
    """
    abs_pnl = abs(evaluation.get('hypothetical_pnl', 0))
    prediction_correct = evaluation.get('prediction_correct', False)
    
    # High-impact trades go to higher tiers
    if abs_pnl >= 3.0 and prediction_correct:
        return TIER_LONG  # Significant correct predictions
    elif abs_pnl >= 1.5 or prediction_correct:
        return TIER_MID   # Notable results
    else:
        return TIER_SHORT  # Standard results


def store_lesson(
    prediction: Dict[str, Any],
    evaluation: Dict[str, Any],
    lesson: str,
    graph: TradingAgentsGraph,
    final_state: Dict[str, Any] = None
):
    """
    Store the lesson in memory with confidence scoring.
    
    Uses the tiered memory system to store lessons with appropriate
    confidence and tier based on the prediction outcome.
    """
    # Create situation string from the original analysis
    situation = "\n\n".join([
        prediction["full_state_summary"].get("market_report", ""),
        prediction["full_state_summary"].get("news_report", ""),
        prediction["full_state_summary"].get("sentiment_report", ""),
    ])
    
    # Create a prediction-specific memory if it doesn't exist
    if not hasattr(graph, 'prediction_memory'):
        graph.prediction_memory = FinancialSituationMemory(
            "prediction_accuracy", 
            graph.config
        )
    
    # Store the lesson with confidence scoring
    recommendation = f"""
PREDICTION EVALUATION ({evaluation['symbol']}):
Signal: {prediction['signal']} | Expected: {prediction['expected_direction']} | Actual: {evaluation['actual_direction']}
Result: {'CORRECT' if evaluation['prediction_correct'] else 'INCORRECT'} | P&L: {evaluation['hypothetical_pnl']:+.2f}%

{lesson}
"""
    
    # Determine tier based on evaluation quality
    tier = determine_tier_from_evaluation(evaluation)
    
    # Add with confidence scoring based on returns and correctness
    graph.prediction_memory.add_situations(
        [(situation, recommendation)],
        tier=tier,
        returns=evaluation['hypothetical_pnl'],
        prediction_correct=evaluation['prediction_correct']
    )
    print(f"Lesson stored in prediction_accuracy memory (tier: {tier})")
    
    # Also update the standard memories via reflection
    # This uses the existing reflection system with the hypothetical P&L
    # The reflection system now also uses confidence scoring
    if final_state is not None:
        try:
            graph.curr_state = final_state
            graph.reflect_and_remember(evaluation['hypothetical_pnl'])
        except Exception as e:
            print(f"Warning: Could not run standard reflection: {e}")


def save_evaluation(evaluation: Dict[str, Any], prediction_filepath: Path):
    """Save evaluation results and mark prediction as evaluated."""
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_file = EVALUATIONS_DIR / f"{evaluation['symbol']}_{timestamp}.json"
    
    with open(eval_file, "w") as f:
        json.dump(evaluation, f, indent=2)
    
    # Update prediction status
    with open(prediction_filepath, "rb") as f:
        data = pickle.load(f)
    
    data["status"] = "evaluated"
    data["evaluation"] = evaluation
    
    with open(prediction_filepath, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Evaluation saved: {eval_file.name}")


# =============================================================================
# MAIN CYCLE
# =============================================================================

class DailyAnalysisCycle:
    """Manages the analysis and evaluation cycle."""
    
    def __init__(self, symbol: str, config: Dict[str, Any] = None, evaluation_hours: int = 24):
        self.symbol = symbol
        self.config = config or CYCLE_CONFIG
        self.evaluation_hours = evaluation_hours
        self.graph = None
    
    def _ensure_graph(self):
        """Lazy initialization of the graph."""
        if self.graph is None:
            print(f"Initializing TradingAgentsGraph...")
            self.graph = TradingAgentsGraph(config=self.config, debug=False)
    
    def run_cycle(self) -> Tuple[Dict[str, Any], str]:
        """
        Run one complete analysis cycle:
        1. Evaluate any pending predictions that are due
        2. Run new analysis
        3. Store new prediction for later evaluation
        
        Returns:
            (final_state, signal) from the new analysis
        """
        self._ensure_graph()
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS CYCLE - {self.symbol}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Evaluation window: {self.evaluation_hours} hours")
        print(f"{'='*70}")
        
        # Step 1: Evaluate pending predictions
        self._evaluate_pending()
        
        # Step 2: Run new analysis
        final_state, signal = self._run_analysis()
        
        # Step 3: Store prediction for later evaluation
        self._store_prediction(final_state, signal)
        
        # Step 4: Save last run timestamp for recovery
        save_last_run_state(self.symbol)
        
        print(f"\n{'='*70}")
        print(f"CYCLE COMPLETE")
        print(f"Next evaluation in {self.evaluation_hours} hours")
        print(f"{'='*70}\n")
        
        return final_state, signal
    
    def _evaluate_pending(self):
        """Evaluate all pending predictions that are due."""
        pending = load_pending_predictions(self.symbol)
        
        if not pending:
            print(f"\nNo pending predictions to evaluate for {self.symbol}")
            return
        
        print(f"\nFound {len(pending)} pending prediction(s) to evaluate")
        
        current_price = get_current_price(self.symbol)
        if current_price <= 0:
            print("Warning: Could not get current price, skipping evaluation")
            return
        
        for pred_data in pending:
            try:
                evaluation = evaluate_prediction(pred_data, current_price, self.graph)
                save_evaluation(evaluation, pred_data["filepath"])
            except Exception as e:
                print(f"Failed to evaluate prediction: {e}")
                import traceback
                traceback.print_exc()
    
    def _run_analysis(self) -> Tuple[Dict[str, Any], str]:
        """Run the trading analysis."""
        trade_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\nRunning analysis for {self.symbol} on {trade_date}...")
        
        final_state, signal = self.graph.propagate(self.symbol, trade_date)
        
        print(f"\n--- Analysis Result ---")
        print(f"Signal: {signal}")
        print(f"Decision: {final_state.get('final_trade_decision', 'N/A')[:300]}...")
        
        return final_state, signal
    
    def _store_prediction(self, final_state: Dict[str, Any], signal: str):
        """Extract and store prediction for later evaluation."""
        current_price = get_current_price(self.symbol)
        
        if current_price <= 0:
            print("Warning: Could not get current price, prediction storage may be incomplete")
            current_price = 0.0
        
        prediction = extract_prediction(
            final_state, signal, current_price, 
            evaluation_hours=self.evaluation_hours
        )
        save_prediction(self.symbol, prediction, final_state)


def get_last_run_state(symbol: str) -> Optional[Dict[str, Any]]:
    """Get the last run state for a symbol."""
    if not LAST_RUN_FILE.exists():
        return None
    
    try:
        with open(LAST_RUN_FILE, "r") as f:
            states = json.load(f)
        return states.get(symbol)
    except Exception:
        return None


def save_last_run_state(symbol: str):
    """Save the current run timestamp for a symbol."""
    states = {}
    if LAST_RUN_FILE.exists():
        try:
            with open(LAST_RUN_FILE, "r") as f:
                states = json.load(f)
        except Exception:
            states = {}
    
    states[symbol] = {
        "last_run": datetime.now().isoformat(),
        "timestamp": datetime.now().timestamp()
    }
    
    with open(LAST_RUN_FILE, "w") as f:
        json.dump(states, f, indent=2)


def should_run_analysis(symbol: str, min_hours: float = MIN_HOURS_BETWEEN_RUNS) -> Tuple[bool, str]:
    """
    Check if enough time has passed since last analysis.
    
    Returns:
        (should_run, reason)
    """
    last_state = get_last_run_state(symbol)
    
    if last_state is None:
        return True, "No previous run found"
    
    last_run = datetime.fromisoformat(last_state["last_run"])
    hours_since = (datetime.now() - last_run).total_seconds() / 3600
    
    if hours_since < min_hours:
        return False, f"Last run was {hours_since:.1f} hours ago (minimum: {min_hours}h)"
    
    return True, f"Last run was {hours_since:.1f} hours ago"


def run_scheduler(symbol: str, run_at_hour: int = 9, evaluation_hours: int = 24):
    """
    Run the analysis cycle on a daily schedule.
    
    Args:
        symbol: Trading symbol to analyze
        run_at_hour: Hour of day to run (0-23, default 9 = 9 AM)
        evaluation_hours: Hours until prediction evaluation (default 24, use 72 for swing trades)
    """
    try:
        import schedule
        import time
    except ImportError:
        print("Error: Please install schedule: pip install schedule")
        return
    
    cycle = DailyAnalysisCycle(symbol, evaluation_hours=evaluation_hours)
    
    # Schedule the job
    schedule.every().day.at(f"{run_at_hour:02d}:00").do(cycle.run_cycle)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS SCHEDULER STARTED")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Scheduled time: {run_at_hour:02d}:00 daily")
    print(f"Evaluation window: {evaluation_hours} hours")
    print(f"Min hours between runs: {MIN_HOURS_BETWEEN_RUNS}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*70}\n")
    
    # Check if we should run immediately or skip
    should_run, reason = should_run_analysis(symbol)
    
    if should_run:
        print(f"Running initial cycle... ({reason})")
        cycle.run_cycle()
    else:
        print(f"Skipping initial cycle: {reason}")
        print(f"Will run at next scheduled time: {run_at_hour:02d}:00")
        # Still evaluate pending predictions
        cycle._ensure_graph()
        cycle._evaluate_pending()
    
    # Then run on schedule
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(
        description="Daily Analysis Cycle with 24-Hour Retrospective Evaluation"
    )
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="XAUUSD",
        help="Trading symbol to analyze (default: XAUUSD)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single cycle without scheduling"
    )
    parser.add_argument(
        "--run-at",
        type=int,
        default=9,
        help="Hour of day to run analysis (0-23, default: 9)"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate pending predictions, don't run new analysis"
    )
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="List all pending predictions"
    )
    parser.add_argument(
        "--evaluation-hours",
        type=int,
        default=24,
        help="Hours until prediction evaluation (default: 24, use 72 for swing trades)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("XAI_API_KEY"):
        print("Error: XAI_API_KEY not set in environment")
        return
    
    if args.list_pending:
        pending = load_pending_predictions(args.symbol)
        if not pending:
            print(f"No pending predictions for {args.symbol}")
        else:
            print(f"\nPending predictions for {args.symbol}:")
            for p in pending:
                pred = p["prediction"]
                print(f"  - {pred['signal']} @ ${pred['price_at_analysis']:.2f}")
                print(f"    Due: {pred['evaluation_due']}")
        return
    
    if args.evaluate_only:
        cycle = DailyAnalysisCycle(args.symbol)
        cycle._ensure_graph()
        cycle._evaluate_pending()
        return
    
    if args.run_once:
        cycle = DailyAnalysisCycle(args.symbol, evaluation_hours=args.evaluation_hours)
        cycle.run_cycle()
    else:
        run_scheduler(args.symbol, args.run_at, args.evaluation_hours)


if __name__ == "__main__":
    main()
