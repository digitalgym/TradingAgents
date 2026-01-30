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
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LOGGING SETUP
# =============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs" / "daily_cycle"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to both file and console
log_file = LOG_DIR / f"daily_cycle_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DailyCycle")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.memory import (
    FinancialSituationMemory,
    TIER_SHORT,
    TIER_MID,
    TIER_LONG
)
from tradingagents.risk import PositionSizer, calculate_kelly_from_history
from tradingagents.schemas import PredictionLesson
from tradingagents.dataflows.llm_client import structured_output
from openai import OpenAI


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
    signal,  # Can be str or dict with 'signal' key
    current_price: float,
    evaluation_hours: int = 24,
    analysis_time: datetime = None
) -> Dict[str, Any]:
    """
    Extract structured prediction from analysis state.

    Parses the final_trade_decision to determine expected direction.

    Args:
        final_state: The analysis state from graph.propagate()
        signal: BUY/SELL/HOLD signal - can be string or dict with 'signal' key
        current_price: Price at time of analysis
        evaluation_hours: Hours until evaluation (default 24, can be 72 for swing trades)
        analysis_time: Override analysis timestamp (for backtesting)
    """
    decision = final_state.get("final_trade_decision", "")

    # Handle signal as dict or string
    if isinstance(signal, dict):
        signal_str = signal.get("signal", "HOLD")
    else:
        signal_str = str(signal) if signal else "HOLD"

    # Determine expected direction from signal
    if signal_str.upper() == "BUY":
        expected_direction = "up"
    elif signal_str.upper() == "SELL":
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
        "signal": signal_str.upper(),
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

    logger.info(f"Prediction saved: {filepath.name}")
    logger.info(f"   Signal: {prediction['signal']}")
    logger.info(f"   Expected: {prediction['expected_direction']}")
    logger.info(f"   Price: ${prediction['price_at_analysis']:.2f}")
    logger.info(f"   Evaluation due: {prediction['evaluation_due']}")

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

    # Generate comparative lesson using LLM with structured output
    lesson_text, lesson_obj = generate_comparative_lesson(prediction, evaluation, graph)
    evaluation["lesson"] = lesson_text
    if lesson_obj:
        evaluation["lesson_structured"] = {
            "analysis": lesson_obj.analysis,
            "predictive_factors": lesson_obj.predictive_factors,
            "misleading_factors": lesson_obj.misleading_factors,
            "lesson": lesson_obj.lesson,
            "confidence_adjustment": lesson_obj.confidence_adjustment,
            "similar_pattern_advice": lesson_obj.similar_pattern_advice,
        }

    # Store lesson in memory with regime support
    store_lesson(prediction, evaluation, lesson_text, graph, final_state, lesson_obj)
    
    return evaluation


def generate_comparative_lesson(
    prediction: Dict[str, Any],
    evaluation: Dict[str, Any],
    graph: TradingAgentsGraph
) -> Tuple[str, Optional[PredictionLesson]]:
    """
    Use LLM with structured output to generate a comparative analysis lesson.

    Analyzes what the prediction got right/wrong and why.

    Returns:
        Tuple of (formatted lesson string, PredictionLesson object or None)
    """
    prompt = f"""You are analyzing a trading prediction to extract lessons for future improvement.

PREDICTION MADE:
- Signal: {prediction['signal']}
- Expected Direction: {prediction['expected_direction']}
- Price at Analysis: ${prediction['price_at_analysis']:.2f}
- Key Reasoning: {prediction['final_decision'][:500]}

ACTUAL OUTCOME ({prediction.get('evaluation_hours', 24)} hours later):
- Actual Direction: {evaluation['actual_direction']}
- Price Change: {evaluation['pct_change']:+.2f}%
- Current Price: ${evaluation['price_at_evaluation']:.2f}
- Prediction Correct: {evaluation['prediction_correct']}

KEY FACTORS FROM ANALYSIS:
{chr(10).join(prediction['key_factors'][:3])}

Analyze the prediction outcome and provide structured lessons for improvement.
Focus on actionable insights that can improve future predictions."""

    try:
        # Try structured output first
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

        messages = [
            {"role": "system", "content": "You are a trading performance analyst. Extract structured lessons from prediction outcomes."},
            {"role": "user", "content": prompt}
        ]

        lesson_obj = structured_output(
            client=client,
            model=graph.config.get("quick_think_llm", "grok-3-fast"),
            messages=messages,
            response_schema=PredictionLesson,
            max_tokens=1000,
            temperature=0.3,
            use_responses_api=True
        )

        # Format as string for backward compatibility
        formatted = f"""ANALYSIS: {lesson_obj.analysis}

PREDICTIVE FACTORS: {', '.join(lesson_obj.predictive_factors) if lesson_obj.predictive_factors else 'None identified'}

MISLEADING FACTORS: {', '.join(lesson_obj.misleading_factors) if lesson_obj.misleading_factors else 'None identified'}

LESSON: {lesson_obj.lesson}

CONFIDENCE ADJUSTMENT: {lesson_obj.confidence_adjustment:+.2f}"""

        if lesson_obj.similar_pattern_advice:
            formatted += f"\n\nPATTERN ADVICE: {lesson_obj.similar_pattern_advice}"

        return formatted, lesson_obj

    except Exception as e:
        print(f"Warning: Structured output failed, falling back to raw LLM: {e}")

        # Fallback to raw LLM output
        try:
            response = graph.quick_thinking_llm.invoke(prompt)
            return response.content, None
        except Exception as e2:
            print(f"Warning: Failed to generate lesson: {e2}")
            fallback = f"Prediction was {'correct' if evaluation['prediction_correct'] else 'incorrect'}. Price moved {evaluation['pct_change']:+.2f}%."
            return fallback, None


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


def extract_regime_from_state(final_state: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Extract market regime information from the analysis state.

    Returns regime dict compatible with memory system:
    - market_regime: trending_up, trending_down, ranging, volatile
    - volatility_regime: low, normal, high, extreme
    """
    regime = {}

    # Try to extract from SMC context
    smc_context = final_state.get("smc_context", "")
    market_report = final_state.get("market_report", "")

    # Simple heuristics - could be enhanced with LLM extraction
    combined_text = f"{smc_context} {market_report}".lower()

    # Detect market regime
    if "bullish" in combined_text or "uptrend" in combined_text:
        regime["market_regime"] = "trending_up"
    elif "bearish" in combined_text or "downtrend" in combined_text:
        regime["market_regime"] = "trending_down"
    elif "ranging" in combined_text or "consolidat" in combined_text:
        regime["market_regime"] = "ranging"
    elif "volatile" in combined_text or "choppy" in combined_text:
        regime["market_regime"] = "volatile"

    # Detect volatility regime
    if "high volatility" in combined_text or "very volatile" in combined_text:
        regime["volatility_regime"] = "high"
    elif "low volatility" in combined_text or "quiet" in combined_text:
        regime["volatility_regime"] = "low"
    elif "extreme" in combined_text:
        regime["volatility_regime"] = "extreme"

    return regime if regime else None


def store_lesson(
    prediction: Dict[str, Any],
    evaluation: Dict[str, Any],
    lesson: str,
    graph: TradingAgentsGraph,
    final_state: Dict[str, Any] = None,
    lesson_obj: Optional[PredictionLesson] = None
):
    """
    Store the lesson in memory with confidence scoring and regime support.

    Uses the tiered memory system to store lessons with appropriate
    confidence and tier based on the prediction outcome.

    Args:
        prediction: The original prediction data
        evaluation: The evaluation results
        lesson: Formatted lesson string
        graph: The TradingAgentsGraph instance
        final_state: The full analysis state (for regime extraction)
        lesson_obj: Structured PredictionLesson object (optional)
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

    # Build recommendation with structured lesson if available
    if lesson_obj:
        recommendation = f"""
PREDICTION EVALUATION ({evaluation['symbol']}):
Signal: {prediction['signal']} | Expected: {prediction['expected_direction']} | Actual: {evaluation['actual_direction']}
Result: {'CORRECT' if evaluation['prediction_correct'] else 'INCORRECT'} | P&L: {evaluation['hypothetical_pnl']:+.2f}%

ANALYSIS: {lesson_obj.analysis}

LESSON: {lesson_obj.lesson}

PREDICTIVE FACTORS: {', '.join(lesson_obj.predictive_factors) if lesson_obj.predictive_factors else 'None'}
MISLEADING FACTORS: {', '.join(lesson_obj.misleading_factors) if lesson_obj.misleading_factors else 'None'}

PATTERN ADVICE: {lesson_obj.similar_pattern_advice or 'N/A'}
"""
    else:
        recommendation = f"""
PREDICTION EVALUATION ({evaluation['symbol']}):
Signal: {prediction['signal']} | Expected: {prediction['expected_direction']} | Actual: {evaluation['actual_direction']}
Result: {'CORRECT' if evaluation['prediction_correct'] else 'INCORRECT'} | P&L: {evaluation['hypothetical_pnl']:+.2f}%

{lesson}
"""

    # Determine tier based on evaluation quality
    tier = determine_tier_from_evaluation(evaluation)

    # Extract regime from state for better memory retrieval later
    regime = None
    if final_state:
        regime = extract_regime_from_state(final_state)

    # Add with confidence scoring based on returns and correctness
    graph.prediction_memory.add_situations(
        [(situation, recommendation)],
        tier=tier,
        returns=evaluation['hypothetical_pnl'],
        prediction_correct=evaluation['prediction_correct'],
        regime=regime
    )
    print(f"Lesson stored in prediction_accuracy memory (tier: {tier}, regime: {regime})")
    
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
            logger.info(f"Initializing TradingAgentsGraph...")
            self.graph = TradingAgentsGraph(config=self.config, debug=False)
            logger.info(f"TradingAgentsGraph initialized successfully")
    
    def run_cycle(self) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Run one complete analysis cycle:
        1. Evaluate any pending predictions that are due
        2. Run new analysis with SMC integration
        3. Display comprehensive SMC trading plan with LLM enhancement
        4. Store new prediction for later evaluation

        Returns:
            (final_state, signal, smc_analysis) from the new analysis
        """
        logger.info(f"{'='*70}")
        logger.info(f"ANALYSIS CYCLE - {self.symbol}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Evaluation window: {self.evaluation_hours} hours")
        logger.info(f"{'='*70}")

        try:
            self._ensure_graph()
        except Exception as e:
            logger.error(f"Failed to initialize graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Step 1: Evaluate pending predictions
        self._evaluate_pending()

        # Step 2: Run new analysis with SMC
        final_state, signal, smc_analysis = self._run_analysis()

        # Step 3: Display SMC-based trading plan if available
        if smc_analysis and signal != "HOLD":
            trade_date = datetime.now().strftime("%Y-%m-%d")
            self._display_smc_plan(signal, smc_analysis, trade_date, final_state)

        # Step 4: Store prediction for later evaluation
        self._store_prediction(final_state, signal, smc_analysis)

        # Step 5: Save last run timestamp for recovery
        save_last_run_state(self.symbol)

        logger.info(f"{'='*70}")
        logger.info(f"CYCLE COMPLETE")
        logger.info(f"Next evaluation in {self.evaluation_hours} hours")
        logger.info(f"{'='*70}")

        return final_state, signal, smc_analysis
    
    def _evaluate_pending(self):
        """Evaluate all pending predictions that are due."""
        pending = load_pending_predictions(self.symbol)

        if not pending:
            logger.info(f"No pending predictions to evaluate for {self.symbol}")
            return

        logger.info(f"Found {len(pending)} pending prediction(s) to evaluate")

        current_price = get_current_price(self.symbol)
        if current_price <= 0:
            logger.warning("Could not get current price, skipping evaluation")
            return

        for pred_data in pending:
            try:
                evaluation = evaluate_prediction(pred_data, current_price, self.graph)
                save_evaluation(evaluation, pred_data["filepath"])
            except Exception as e:
                logger.error(f"Failed to evaluate prediction: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _run_analysis(self) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Run the trading analysis with SMC integration."""
        trade_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running analysis for {self.symbol} on {trade_date}...")

        # Run SMC analysis first
        smc_analysis = None
        try:
            from tradingagents.dataflows.smc_utils import (
                analyze_multi_timeframe_smc,
                format_smc_for_prompt,
                get_htf_bias_alignment
            )

            logger.info("Running Smart Money Concepts analysis...")
            smc_analysis = analyze_multi_timeframe_smc(
                symbol=self.symbol,
                timeframes=['1H', '4H', 'D1']
            )

            if smc_analysis:
                alignment = get_htf_bias_alignment(smc_analysis)
                logger.info(f"SMC Analysis: {alignment['message']}")

                # Format SMC context for LLM
                smc_context = format_smc_for_prompt(smc_analysis, self.symbol)

                # Inject SMC context into state before propagation
                # Initialize curr_state if it doesn't exist
                if self.graph.curr_state is None:
                    self.graph.curr_state = {}
                self.graph.curr_state['smc_context'] = smc_context
        except Exception as e:
            logger.warning(f"SMC analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # Run main analysis with SMC context
        logger.info("Running main analysis with graph.propagate()...")
        try:
            final_state, signal = self.graph.propagate(self.symbol, trade_date)
        except Exception as e:
            logger.error(f"Graph propagation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        logger.info(f"--- Analysis Result ---")
        logger.info(f"Signal: {signal}")
        logger.info(f"Decision: {final_state.get('final_trade_decision', 'N/A')[:300]}...")

        return final_state, signal, smc_analysis
    
    def _display_smc_plan(self, signal: str, smc_analysis: Dict[str, Any], trade_date: str, final_state: Dict[str, Any]):
        """Display comprehensive SMC trading plan with LLM enhancement."""
        try:
            from tradingagents.dataflows.smc_utils import generate_smc_trading_plan
            from tradingagents.dataflows.llm_smc_enhancer import enhance_plan_with_llm
            from tradingagents.dataflows.interface import route_to_vendor
            from langchain_openai import ChatOpenAI
            import os

            # Get current price
            current_price = None
            for tf in ['1H', '4H', 'H4', 'D1']:
                if tf in smc_analysis and 'current_price' in smc_analysis[tf]:
                    current_price = smc_analysis[tf]['current_price']
                    break

            if not current_price:
                print("Could not determine current price from SMC data")
                return

            # Get ATR
            atr_data = route_to_vendor("get_indicators", self.symbol, "atr", trade_date, 14)
            atr_value = None
            if "ATR(14):" in atr_data:
                try:
                    atr_value = float(atr_data.split("ATR(14):")[1].split()[0])
                except:
                    pass

            # Generate comprehensive trading plan
            plan = generate_smc_trading_plan(
                smc_analysis=smc_analysis,
                current_price=current_price,
                overall_bias=signal,
                primary_timeframe='1H',
                atr=atr_value
            )

            if 'error' in plan:
                print(f"\nError generating trading plan: {plan['error']}")
                return

            # Enhance with LLM
            xai_api_key = os.getenv("XAI_API_KEY")
            if xai_api_key:
                try:
                    print("\n[Enhancing plan with LLM contextual intelligence...]")
                    llm = ChatOpenAI(
                        model="grok-beta",
                        temperature=0.3,
                        api_key=xai_api_key,
                        base_url="https://api.x.ai/v1"
                    )

                    plan = enhance_plan_with_llm(
                        plan=plan,
                        smc_analysis=smc_analysis,
                        llm=llm,
                        atr=atr_value,
                        final_state=final_state
                    )
                    print("[LLM enhancement complete]")
                except Exception as e:
                    print(f"[LLM enhancement skipped: {e}]")

            # Display the plan (simplified version for daily cycle)
            print(f"\n{'='*70}")
            print("COMPREHENSIVE SMC TRADING PLAN")
            print(f"{'='*70}")

            pos = plan['position_analysis']
            print(f"\n[POSITION ANALYSIS]")
            print(f"Current Price: ${pos['current_price']:.2f}")

            # Determine position description
            if pos.get('at_resistance'):
                position_desc = f"At Resistance (${pos['nearest_resistance']['price_range'][0]:.2f} - ${pos['nearest_resistance']['price_range'][1]:.2f})"
            elif pos.get('at_support'):
                position_desc = f"At Support (${pos['nearest_support']['price_range'][0]:.2f} - ${pos['nearest_support']['price_range'][1]:.2f})"
            else:
                position_desc = "Between zones"

            print(f"Position: {position_desc}")

            # Display recommendation
            rec = plan['recommendation']
            print(f"\n[RECOMMENDATION: {rec['action']}] - Confidence: {rec['confidence']}")
            print(f"{rec['reason']}")

            # Display primary setup
            if plan.get('primary_setup'):
                setup = plan['primary_setup']
                print(f"\n[PRIMARY SETUP: {setup['direction']}]")
                print(f"Entry: ${setup['entry_price']:.2f} ({setup['entry_type']})")
                print(f"Stop Loss: ${setup['stop_loss']:.2f}")
                print(f"Take Profit 1: ${setup['take_profit_1']:.2f}")
                print(f"R:R Ratio: 1:{(setup['reward_pct_tp1']/setup['risk_pct']):.2f}")

                # Show LLM enhancement if available
                if 'llm_enhancement' in setup:
                    llm = setup['llm_enhancement']
                    print(f"\nLLM Confidence: {llm['confidence_level']}")
                    print(f"Adjusted Hold Probability: {llm['adjusted_hold_probability']:.0%}")
                    print(f"Recommended Action: {llm['recommended_action']}")

            print(f"\n{'='*70}\n")

        except Exception as e:
            print(f"Warning: Could not display SMC plan: {e}")
            import traceback
            traceback.print_exc()

    def _store_prediction(self, final_state: Dict[str, Any], signal, smc_analysis: Dict[str, Any] = None):
        """Extract and store prediction for later evaluation with SMC context."""
        # Handle signal as dict or string
        if isinstance(signal, dict):
            signal_str = signal.get("signal", "HOLD")
        else:
            signal_str = str(signal) if signal else "HOLD"

        current_price = get_current_price(self.symbol)

        if current_price <= 0:
            print("Warning: Could not get current price, prediction storage may be incomplete")
            current_price = 0.0

        prediction = extract_prediction(
            final_state, signal, current_price,
            evaluation_hours=self.evaluation_hours
        )

        # Add SMC context to prediction
        if smc_analysis:
            prediction["smc_context"] = {
                "has_smc_data": True,
                "timeframes": list(smc_analysis.keys()),
            }

        # Add position sizing recommendation if signal is BUY or SELL
        if signal_str.upper() in ["BUY", "SELL"] and current_price > 0:
            position_rec = self._calculate_position_size(signal, current_price, smc_analysis)
            if position_rec:
                prediction["position_sizing"] = position_rec
                print(f"\n--- Position Sizing ---")
                print(f"Recommended Size: {position_rec['recommended_size']:.4f} units")
                print(f"MT5 Lots: {position_rec['mt5_lots']:.2f}")
                print(f"Risk Amount: ${position_rec['risk_amount']:.2f}")
                print(f"Risk %: {position_rec['risk_percent']*100:.2f}%")

        save_prediction(self.symbol, prediction, final_state)
    
    def _calculate_position_size(
        self,
        signal,  # Can be str or dict with 'signal' key
        current_price: float,
        smc_analysis: Dict[str, Any] = None,
        account_balance: float = 100000,
        atr_multiplier: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate position size based on Kelly criterion and SMC-based stop loss.

        Uses SMC order blocks for stop loss placement when available,
        falls back to ATR-based stop loss otherwise.

        Uses backtest history for Kelly parameters if available.
        """
        # Handle signal as dict or string
        if isinstance(signal, dict):
            signal_str = signal.get("signal", "HOLD")
        else:
            signal_str = str(signal) if signal else "HOLD"

        try:
            # Get ATR for stop loss calculation
            from tradingagents.dataflows.mt5_data import get_mt5_data
            import pandas as pd
            import io

            # Fetch recent data for ATR
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            csv_data = get_mt5_data(self.symbol, start_date, end_date, timeframe="H4")

            if csv_data is None or isinstance(csv_data, str) and csv_data.startswith("Error"):
                print("Warning: Failed to get MT5 data for ATR calculation")
                return None

            # Parse CSV string to DataFrame
            if isinstance(csv_data, str):
                # Filter out comment lines starting with #
                lines = [l for l in csv_data.split('\n') if l and not l.startswith('#')]
                if len(lines) < 2:
                    print("Warning: Insufficient data for ATR calculation")
                    return None
                df = pd.read_csv(io.StringIO('\n'.join(lines)))
            else:
                df = csv_data

            if df is None or len(df) < 14:
                print("Warning: Insufficient data for ATR calculation")
                return None

            # Calculate ATR - handle column name variations
            high_col = 'high' if 'high' in df.columns else 'High'
            low_col = 'low' if 'low' in df.columns else 'Low'
            close_col = 'close' if 'close' in df.columns else 'Close'

            high = df[high_col].values
            low = df[low_col].values
            close = df[close_col].values

            tr = []
            for i in range(1, len(df)):
                tr.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                ))
            atr = sum(tr[-14:]) / 14  # 14-period ATR

            # Try to get SMC-based stop loss first
            stop_loss = None
            stop_loss_method = "ATR"

            if smc_analysis:
                try:
                    from tradingagents.dataflows.smc_utils import suggest_smc_stop_loss

                    smc_sl = suggest_smc_stop_loss(
                        smc_analysis=smc_analysis,
                        direction=signal,
                        entry_price=current_price,
                        atr=atr,
                        atr_multiplier=atr_multiplier,
                        primary_timeframe='1H'
                    )

                    if smc_sl and smc_sl.get('price'):
                        stop_loss = smc_sl['price']
                        stop_loss_method = f"SMC ({smc_sl.get('source', 'Order Block')})"
                        if smc_sl.get('confluence_score', 0) > 1.0:
                            stop_loss_method += f" Confluence: {smc_sl['confluence_score']:.1f}"
                except Exception as e:
                    print(f"Warning: SMC stop loss calculation failed: {e}")

            # Fallback to ATR-based stop loss
            if stop_loss is None:
                if signal_str.upper() == "BUY":
                    stop_loss = current_price - (atr * atr_multiplier)
                else:
                    stop_loss = current_price + (atr * atr_multiplier)
            
            # Load trade history for Kelly calculation
            trade_history = self._load_trade_history()
            
            # Create position sizer
            sizer = PositionSizer(
                account_balance=account_balance,
                max_risk_per_trade=0.02,
                kelly_fraction=0.5  # Half-Kelly for safety
            )
            
            # Calculate position size
            if trade_history and len(trade_history) >= 10:
                win_rate, avg_win, avg_loss, kelly = calculate_kelly_from_history(trade_history)
                
                if kelly > 0.05:  # Only use Kelly if meaningful edge
                    result = sizer.kelly_size(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        entry_price=current_price,
                        stop_loss=stop_loss
                    )
                else:
                    result = sizer.fixed_fractional_size(
                        entry_price=current_price,
                        stop_loss=stop_loss
                    )
            else:
                result = sizer.fixed_fractional_size(
                    entry_price=current_price,
                    stop_loss=stop_loss
                )
            
            # Calculate MT5 lots
            mt5_lots = sizer.calculate_lots(result.recommended_size, contract_size=100)
            
            # Calculate take profit levels
            risk_distance = abs(current_price - stop_loss)
            if signal_str.upper() == "BUY":
                tp_2r = current_price + (risk_distance * 2)
                tp_3r = current_price + (risk_distance * 3)
            else:
                tp_2r = current_price - (risk_distance * 2)
                tp_3r = current_price - (risk_distance * 3)
            
            return {
                "method": result.method,
                "recommended_size": result.recommended_size,
                "mt5_lots": mt5_lots,
                "position_value": result.position_value,
                "risk_amount": result.risk_amount,
                "risk_percent": result.risk_percent,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "stop_loss_method": stop_loss_method,
                "take_profit_2r": tp_2r,
                "take_profit_3r": tp_3r,
                "atr": atr,
                "kelly_fraction": result.kelly_fraction,
            }
            
        except Exception as e:
            print(f"Warning: Position sizing calculation failed: {e}")
            return None
    
    def _load_trade_history(self) -> Optional[list]:
        """Load trade history from backtest results for Kelly calculation."""
        try:
            backtest_dir = Path(__file__).parent / "backtest_results"
            if not backtest_dir.exists():
                return None
            
            files = sorted(
                backtest_dir.glob(f"{self.symbol}_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not files:
                return None
            
            import json
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            results = data.get("results", [])
            if results:
                return [r["hypothetical_pnl"] / 100 for r in results]
            
            return None
        except Exception:
            return None


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


def run_multi_symbol_scheduler(symbols: List[str], run_at_hour: int = 9, evaluation_hours: int = 24, stagger_minutes: int = 5):
    """
    Run analysis cycles for multiple symbols on a daily schedule.

    Staggers symbol analysis by a few minutes to avoid overwhelming the system.

    Args:
        symbols: List of trading symbols to analyze (e.g., ["XAUUSD", "XAGUSD", "XPTUSD", "COPPER-C"])
        run_at_hour: Hour of day to start (0-23, default 9 = 9 AM)
        evaluation_hours: Hours until prediction evaluation (default 24)
        stagger_minutes: Minutes between each symbol analysis (default 5)
    """
    logger.info(f"Starting multi-symbol scheduler with symbols: {symbols}")

    try:
        import schedule
        import time
    except ImportError:
        logger.error("Missing dependency: Please install schedule: pip install schedule")
        return

    # Create a cycle instance for each symbol
    cycles = {symbol: DailyAnalysisCycle(symbol, evaluation_hours=evaluation_hours) for symbol in symbols}

    logger.info(f"{'='*70}")
    logger.info(f"MULTI-SYMBOL ANALYSIS SCHEDULER STARTED")
    logger.info(f"{'='*70}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Start time: {run_at_hour:02d}:00 daily")
    logger.info(f"Stagger: {stagger_minutes} minutes between symbols")
    logger.info(f"Evaluation window: {evaluation_hours} hours")
    logger.info(f"Min hours between runs: {MIN_HOURS_BETWEEN_RUNS}")
    logger.info(f"Press Ctrl+C to stop")
    logger.info(f"{'='*70}")

    # Schedule each symbol at staggered times
    for i, symbol in enumerate(symbols):
        cycle = cycles[symbol]
        offset_minutes = i * stagger_minutes
        run_hour = run_at_hour
        run_minute = offset_minutes

        # Handle hour overflow
        if run_minute >= 60:
            run_hour = (run_hour + run_minute // 60) % 24
            run_minute = run_minute % 60

        schedule_time = f"{run_hour:02d}:{run_minute:02d}"
        schedule.every().day.at(schedule_time).do(cycle.run_cycle)

        logger.info(f"[{symbol}] Scheduled at {schedule_time}")

        # Check if we should run immediately
        should_run, reason = should_run_analysis(symbol)

        if should_run:
            logger.info(f"[{symbol}] Running initial cycle... ({reason})")
            try:
                cycle.run_cycle()
                # Brief pause between symbols to avoid overwhelming the system
                if i < len(symbols) - 1:
                    logger.info(f"Pausing {stagger_minutes//2} seconds before next symbol...")
                    time.sleep(stagger_minutes * 30)  # Half the stagger time
            except Exception as e:
                logger.error(f"[{symbol}] Error in initial cycle: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info(f"[{symbol}] Skipping initial cycle: {reason}")
            # Still evaluate pending predictions
            try:
                cycle._ensure_graph()
                cycle._evaluate_pending()
            except Exception as e:
                logger.error(f"[{symbol}] Error evaluating pending: {e}")
                import traceback
                logger.error(traceback.format_exc())

    logger.info(f"{'='*70}")
    logger.info("All symbols initialized. Running on schedule...")
    logger.info(f"{'='*70}")

    # Run on schedule
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Scheduler crashed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    logger.info("Daily Cycle starting...")

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
        "--symbols",
        type=str,
        nargs="+",
        help="Multiple symbols to analyze (e.g., --symbols XAUUSD XAGUSD XPTUSD)"
    )
    parser.add_argument(
        "--commodities",
        action="store_true",
        help="Analyze all major commodities: Gold (XAUUSD), Silver (XAGUSD), Platinum (XPTUSD), Copper (COPPER-C)"
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
        "--stagger-minutes",
        type=int,
        default=5,
        help="Minutes to stagger between symbols (default: 5)"
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
    logger.info(f"Arguments: symbols={args.symbols}, run_at={args.run_at}, run_once={args.run_once}")

    # Check API key
    if not os.getenv("XAI_API_KEY"):
        logger.error("XAI_API_KEY not set in environment")
        return

    # Determine which symbols to analyze
    symbols = []
    if args.commodities:
        symbols = ["XAUUSD", "XAGUSD", "XPTUSD", "COPPER-C"]
        logger.info(f"Analyzing all major commodities: {', '.join(symbols)}")
    elif args.symbols:
        symbols = args.symbols
        logger.info(f"Analyzing multiple symbols: {', '.join(symbols)}")
    else:
        symbols = [args.symbol]

    # Handle list-pending and evaluate-only for all symbols
    if args.list_pending:
        for symbol in symbols:
            pending = load_pending_predictions(symbol)
            if not pending:
                logger.info(f"No pending predictions for {symbol}")
            else:
                logger.info(f"Pending predictions for {symbol}:")
                for p in pending:
                    pred = p["prediction"]
                    logger.info(f"  - {pred['signal']} @ ${pred['price_at_analysis']:.2f}")
                    logger.info(f"    Due: {pred['evaluation_due']}")
        return

    if args.evaluate_only:
        for symbol in symbols:
            logger.info(f"{'='*70}")
            logger.info(f"Evaluating pending predictions for {symbol}")
            logger.info(f"{'='*70}")
            cycle = DailyAnalysisCycle(symbol)
            cycle._ensure_graph()
            cycle._evaluate_pending()
        return

    # Run analysis
    if len(symbols) == 1:
        # Single symbol mode
        if args.run_once:
            cycle = DailyAnalysisCycle(symbols[0], evaluation_hours=args.evaluation_hours)
            cycle.run_cycle()
        else:
            run_scheduler(symbols[0], args.run_at, args.evaluation_hours)
    else:
        # Multi-symbol mode
        if args.run_once:
            # Run all symbols once, staggered
            for i, symbol in enumerate(symbols):
                print(f"\n{'='*70}")
                print(f"Running cycle {i+1}/{len(symbols)}: {symbol}")
                print(f"{'='*70}")
                try:
                    cycle = DailyAnalysisCycle(symbol, evaluation_hours=args.evaluation_hours)
                    cycle.run_cycle()
                    # Brief pause between symbols
                    if i < len(symbols) - 1:
                        import time
                        logger.info(f"Pausing {args.stagger_minutes//2} seconds before next symbol...")
                        time.sleep(args.stagger_minutes * 30)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        else:
            # Multi-symbol scheduled mode
            run_multi_symbol_scheduler(symbols, args.run_at, args.evaluation_hours, args.stagger_minutes)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"FATAL ERROR: Daily cycle crashed: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
