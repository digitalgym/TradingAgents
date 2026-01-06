"""
Backtest Training - Build Initial Memory Corpus from Historical Data

This script runs historical backtesting using TECHNICAL ANALYSIS ONLY to build
an initial corpus of prediction->outcome lessons before starting live trading.

Why technical-only?
- Historical OHLCV data is available and accurate
- News/sentiment APIs return CURRENT data, not historical
- Technical patterns are the foundation that news/sentiment build upon
- Live daily_cycle.py will train the news/sentiment components going forward

Process:
1. Load historical OHLCV data for the symbol
2. For each historical date (e.g., every 3 days over 6 months):
   a. Calculate technical indicators (RSI, MACD, SMA, etc.) as of that date
   b. Generate signal based on technical analysis
   c. Get price N hours later (from historical data)
   d. Compare prediction vs actual direction
   e. Generate lesson and store in memory
3. Report accuracy statistics

Usage:
    python examples/backtest_training.py --symbol XAUUSD --months 6
    python examples/backtest_training.py --symbol XAUUSD --months 3 --interval-days 2
    python examples/backtest_training.py --symbol XAUUSD --evaluation-hours 72
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from tradingagents.agents.utils.memory import (
    FinancialSituationMemory, 
    TIER_SHORT, 
    TIER_MID, 
    TIER_LONG
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.risk import RiskMetrics, Portfolio


# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = DEFAULT_CONFIG.copy()
BACKTEST_CONFIG.update({
    "llm_provider": "xai",
    "deep_think_llm": "grok-3-fast",
    "quick_think_llm": "grok-3-fast",
    "backend_url": "https://api.x.ai/v1",
    "use_memory": True,
    "embedding_provider": "local",
})

# Directory for backtest results
BACKTEST_DIR = Path(__file__).parent / "backtest_results"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TechnicalState:
    """Technical analysis state for a specific date."""
    date: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    atr: float
    upper_band: float
    lower_band: float
    
    def to_market_report(self) -> str:
        """Generate a market report string from technical state."""
        # Trend analysis
        trend = "BULLISH" if self.sma_20 > self.sma_50 else "BEARISH"
        price_vs_sma = "above" if self.price > self.sma_20 else "below"
        
        # RSI analysis
        if self.rsi > 70:
            rsi_status = "OVERBOUGHT"
        elif self.rsi < 30:
            rsi_status = "OVERSOLD"
        else:
            rsi_status = "NEUTRAL"
        
        # MACD analysis
        macd_trend = "BULLISH" if self.macd > self.macd_signal else "BEARISH"
        macd_momentum = "strengthening" if self.macd_hist > 0 else "weakening"
        
        # Bollinger analysis
        if self.price > self.upper_band:
            bb_status = "above upper band (overbought)"
        elif self.price < self.lower_band:
            bb_status = "below lower band (oversold)"
        else:
            bb_status = "within bands"
        
        return f"""TECHNICAL ANALYSIS REPORT ({self.date})

PRICE ACTION:
- Current Price: ${self.price:.2f}
- Price is {price_vs_sma} SMA20 (${self.sma_20:.2f})
- SMA20 vs SMA50: {trend} (SMA20: ${self.sma_20:.2f}, SMA50: ${self.sma_50:.2f})

MOMENTUM INDICATORS:
- RSI(14): {self.rsi:.1f} - {rsi_status}
- MACD: {self.macd:.4f} (Signal: {self.macd_signal:.4f}) - {macd_trend}
- MACD Histogram: {self.macd_hist:.4f} - Momentum {macd_momentum}

VOLATILITY:
- ATR(14): {self.atr:.4f}
- Bollinger Bands: Upper ${self.upper_band:.2f}, Lower ${self.lower_band:.2f}
- Price is {bb_status}

OVERALL TREND: {trend} with {rsi_status} RSI and {macd_trend} MACD
"""

    def generate_signal(self) -> str:
        """Generate BUY/SELL/HOLD signal based on technical indicators."""
        score = 0
        
        # Trend following (SMA crossover)
        if self.sma_20 > self.sma_50:
            score += 1
        else:
            score -= 1
        
        # Price vs SMA
        if self.price > self.sma_20:
            score += 1
        else:
            score -= 1
        
        # RSI
        if self.rsi < 30:  # Oversold = buy signal
            score += 2
        elif self.rsi > 70:  # Overbought = sell signal
            score -= 2
        elif self.rsi < 45:
            score += 1
        elif self.rsi > 55:
            score -= 1
        
        # MACD
        if self.macd > self.macd_signal:
            score += 1
        else:
            score -= 1
        
        if self.macd_hist > 0:
            score += 1
        else:
            score -= 1
        
        # Bollinger Bands
        if self.price < self.lower_band:  # Oversold
            score += 1
        elif self.price > self.upper_band:  # Overbought
            score -= 1
        
        # Generate signal
        if score >= 3:
            return "BUY"
        elif score <= -3:
            return "SELL"
        else:
            return "HOLD"


@dataclass
class BacktestResult:
    """Result of a single backtest evaluation."""
    date: str
    signal: str
    expected_direction: str
    actual_direction: str
    price_at_analysis: float
    price_at_evaluation: float
    pct_change: float
    prediction_correct: bool
    hypothetical_pnl: float
    technical_state: TechnicalState = None
    lesson: str = ""


@dataclass
class BacktestStats:
    """Aggregate statistics from backtesting."""
    total_predictions: int = 0
    correct_predictions: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    total_hypothetical_pnl: float = 0.0
    results: List[BacktestResult] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=lambda: [100000.0])
    trade_returns: List[float] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100
    
    @property
    def avg_pnl(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_hypothetical_pnl / self.total_predictions
    
    def get_risk_metrics(self) -> Optional[Dict]:
        """Calculate risk metrics from backtest results."""
        if len(self.trade_returns) < 2:
            return None
        
        returns = np.array(self.trade_returns)
        equity = np.array(self.equity_curve)
        
        report = RiskMetrics.calculate_all(
            returns=returns,
            equity_curve=equity,
            trade_returns=returns,
            periods_per_year=120  # ~120 trades per year at 3-day intervals
        )
        return report.to_dict()


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# =============================================================================
# HISTORICAL DATA
# =============================================================================

def load_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Load historical OHLCV data from MT5.
    
    Returns DataFrame with columns: time, open, high, low, close, volume
    """
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        # Get hourly bars for granular analysis
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_H4,  # 4-hour bars for better indicator calculation
            start_date,
            end_date
        )
        
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data returned for {symbol}")
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Calculate all technical indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['upper_band'], _, df['lower_band'] = calculate_bollinger_bands(df['close'])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # Drop NaN rows (from indicator warmup)
        df.dropna(inplace=True)
        
        print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_technical_state_at_date(df: pd.DataFrame, target_date: datetime) -> Optional[TechnicalState]:
    """Get technical state at or near a specific date."""
    # Find closest date
    idx = df.index.get_indexer([target_date], method='nearest')[0]
    if idx < 0 or idx >= len(df):
        return None
    
    row = df.iloc[idx]
    actual_date = df.index[idx]
    
    return TechnicalState(
        date=actual_date.strftime("%Y-%m-%d %H:%M"),
        price=row['close'],
        rsi=row['rsi'],
        macd=row['macd'],
        macd_signal=row['macd_signal'],
        macd_hist=row['macd_hist'],
        sma_20=row['sma_20'],
        sma_50=row['sma_50'],
        ema_12=row['ema_12'],
        ema_26=row['ema_26'],
        atr=row['atr'],
        upper_band=row['upper_band'],
        lower_band=row['lower_band'],
    )


def get_price_at_date(df: pd.DataFrame, target_date: datetime) -> Optional[float]:
    """Get closing price at or near a specific date."""
    idx = df.index.get_indexer([target_date], method='nearest')[0]
    if idx < 0 or idx >= len(df):
        return None
    return df.iloc[idx]['close']


# =============================================================================
# EVALUATION AND LESSONS
# =============================================================================

def evaluate_prediction(
    signal: str,
    price_at_analysis: float,
    price_at_evaluation: float,
) -> Tuple[str, str, bool, float, float]:
    """
    Evaluate a prediction against actual price movement.
    
    Returns: (expected_direction, actual_direction, prediction_correct, hypothetical_pnl, pct_change)
    """
    # Expected direction from signal
    if signal.upper() == "BUY":
        expected_direction = "up"
    elif signal.upper() == "SELL":
        expected_direction = "down"
    else:
        expected_direction = "sideways"
    
    # Calculate actual movement
    pct_change = ((price_at_evaluation - price_at_analysis) / price_at_analysis) * 100
    
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
    if signal.upper() == "BUY":
        hypothetical_pnl = pct_change
    elif signal.upper() == "SELL":
        hypothetical_pnl = -pct_change
    else:
        hypothetical_pnl = 0
    
    return expected_direction, actual_direction, prediction_correct, hypothetical_pnl, pct_change


def generate_lesson(
    result: BacktestResult,
    llm: ChatOpenAI,
    evaluation_hours: int
) -> str:
    """Generate a lesson from the backtest result using LLM."""
    
    tech = result.technical_state
    market_report = tech.to_market_report() if tech else "No technical data"
    
    prompt = f"""You are analyzing a historical trading prediction based on TECHNICAL ANALYSIS to extract lessons.

PREDICTION:
- Signal: {result.signal}
- Expected Direction: {result.expected_direction}

ACTUAL OUTCOME ({evaluation_hours} hours later):
- Actual Direction: {result.actual_direction}
- Price Change: {result.pct_change:+.2f}%
- Prediction Correct: {result.prediction_correct}

TECHNICAL ANALYSIS AT TIME OF PREDICTION:
{market_report}

Generate a concise lesson (2-3 sentences) about what this outcome teaches us about technical analysis.
Focus on:
1. Which technical indicators were predictive vs misleading
2. Actionable insight for similar technical setups in the future

Format: LESSON: [your lesson]
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Prediction was {'correct' if result.prediction_correct else 'incorrect'}. Price moved {result.pct_change:+.2f}%."


def determine_tier_from_result(result: BacktestResult) -> str:
    """
    Determine the appropriate memory tier based on the backtest result.
    
    High-impact results (large moves, correct predictions) go to higher tiers.
    """
    abs_pnl = abs(result.hypothetical_pnl)
    
    # High-impact trades go to higher tiers
    if abs_pnl >= 3.0 and result.prediction_correct:
        return TIER_LONG  # Significant correct predictions
    elif abs_pnl >= 1.5 or result.prediction_correct:
        return TIER_MID   # Notable results
    else:
        return TIER_SHORT  # Standard results


def store_lesson_in_memory(
    result: BacktestResult,
    memory: FinancialSituationMemory
):
    """
    Store the lesson in memory with confidence scoring.
    
    Uses the tiered memory system to store lessons with appropriate
    confidence and tier based on the prediction outcome.
    """
    # Create situation string from technical state
    tech = result.technical_state
    if tech:
        situation = tech.to_market_report()
    else:
        situation = f"Price: {result.price_at_analysis}, Signal: {result.signal}"
    
    # Store lesson with confidence scoring
    recommendation = f"""
BACKTEST EVALUATION ({result.date}):
Signal: {result.signal} | Expected: {result.expected_direction} | Actual: {result.actual_direction}
Result: {'CORRECT' if result.prediction_correct else 'INCORRECT'} | P&L: {result.hypothetical_pnl:+.2f}%

{result.lesson}
"""
    
    # Determine tier based on result quality
    tier = determine_tier_from_result(result)
    
    # Add with confidence scoring based on returns and correctness
    memory.add_situations(
        [(situation, recommendation)],
        tier=tier,
        returns=result.hypothetical_pnl,
        prediction_correct=result.prediction_correct
    )


# =============================================================================
# MAIN BACKTEST
# =============================================================================

class TechnicalBacktester:
    """Runs historical backtesting using technical analysis only."""
    
    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any] = None,
        evaluation_hours: int = 72
    ):
        self.symbol = symbol
        self.config = config or BACKTEST_CONFIG
        self.evaluation_hours = evaluation_hours
        self.stats = BacktestStats()
        self.llm = None
        self.memory = None
    
    def _init_llm(self):
        """Initialize LLM for lesson generation."""
        if self.llm is None:
            api_key = os.getenv("XAI_API_KEY")
            self.llm = ChatOpenAI(
                model=self.config["quick_think_llm"],
                base_url=self.config["backend_url"],
                api_key=api_key
            )
    
    def _init_memory(self):
        """Initialize memory for storing lessons."""
        if self.memory is None:
            self.memory = FinancialSituationMemory(
                "technical_backtest",
                self.config
            )
    
    def run_backtest(
        self,
        months: int = 6,
        interval_days: int = 3,
        max_samples: int = 100
    ) -> BacktestStats:
        """
        Run historical backtest over specified period.
        
        Args:
            months: How many months of history to analyze
            interval_days: Days between each analysis
            max_samples: Maximum number of samples to generate
        """
        self._init_llm()
        self._init_memory()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # Need extra days for indicator warmup and evaluation window
        data_start = start_date - timedelta(days=60)  # Buffer for indicators
        data_end = end_date
        
        print(f"\n{'='*70}")
        print(f"TECHNICAL BACKTEST - {self.symbol}")
        print(f"{'='*70}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Interval: Every {interval_days} days")
        print(f"Evaluation window: {self.evaluation_hours} hours")
        print(f"Max samples: {max_samples}")
        print(f"Mode: TECHNICAL ANALYSIS ONLY (no news/sentiment)")
        print(f"{'='*70}\n")
        
        # Load historical data with indicators
        print("Loading historical price data and calculating indicators...")
        df = load_historical_data(self.symbol, data_start, data_end)
        
        if df is None or len(df) == 0:
            print("ERROR: Could not load historical data")
            return self.stats
        
        # Generate analysis dates
        analysis_dates = []
        current = start_date
        while current < end_date - timedelta(hours=self.evaluation_hours):
            analysis_dates.append(current)
            current += timedelta(days=interval_days)
        
        # Limit samples
        if len(analysis_dates) > max_samples:
            step = len(analysis_dates) // max_samples
            analysis_dates = analysis_dates[::step][:max_samples]
        
        print(f"Will analyze {len(analysis_dates)} historical dates\n")
        
        # Run backtest for each date
        for i, analysis_date in enumerate(analysis_dates):
            print(f"\n[{i+1}/{len(analysis_dates)}] Analyzing {analysis_date.date()}...")
            
            try:
                result = self._analyze_date(analysis_date, df)
                if result:
                    self.stats.results.append(result)
                    self._update_stats(result)
                    self._print_result(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Print final statistics
        self._print_final_stats()
        
        # Save results
        self._save_results()
        
        return self.stats
    
    def _analyze_date(self, analysis_date: datetime, df: pd.DataFrame) -> Optional[BacktestResult]:
        """Run technical analysis for a specific historical date."""
        
        # Get technical state at analysis time
        tech_state = get_technical_state_at_date(df, analysis_date)
        if tech_state is None:
            print(f"  No technical data for {analysis_date}")
            return None
        
        # Get price at evaluation time
        eval_time = analysis_date + timedelta(hours=self.evaluation_hours)
        price_at_evaluation = get_price_at_date(df, eval_time)
        if price_at_evaluation is None:
            print(f"  No price data for evaluation time {eval_time}")
            return None
        
        # Generate signal from technical indicators
        signal = tech_state.generate_signal()
        
        # Evaluate prediction
        expected_dir, actual_dir, correct, pnl, pct_change = evaluate_prediction(
            signal, tech_state.price, price_at_evaluation
        )
        
        # Create result
        result = BacktestResult(
            date=tech_state.date,
            signal=signal,
            expected_direction=expected_dir,
            actual_direction=actual_dir,
            price_at_analysis=tech_state.price,
            price_at_evaluation=price_at_evaluation,
            pct_change=pct_change,
            prediction_correct=correct,
            hypothetical_pnl=pnl,
            technical_state=tech_state,
        )
        
        # Generate lesson using LLM
        result.lesson = generate_lesson(result, self.llm, self.evaluation_hours)
        
        # Store in memory
        store_lesson_in_memory(result, self.memory)
        
        return result
    
    def _update_stats(self, result: BacktestResult):
        """Update aggregate statistics."""
        self.stats.total_predictions += 1
        if result.prediction_correct:
            self.stats.correct_predictions += 1
        
        if result.signal == "BUY":
            self.stats.buy_signals += 1
        elif result.signal == "SELL":
            self.stats.sell_signals += 1
        else:
            self.stats.hold_signals += 1
        
        self.stats.total_hypothetical_pnl += result.hypothetical_pnl
        
        # Track for risk metrics
        trade_return = result.hypothetical_pnl / 100  # Convert percentage to decimal
        self.stats.trade_returns.append(trade_return)
        
        # Update equity curve (compound returns)
        last_equity = self.stats.equity_curve[-1]
        new_equity = last_equity * (1 + trade_return)
        self.stats.equity_curve.append(new_equity)
    
    def _print_result(self, result: BacktestResult):
        """Print single result."""
        status = "CORRECT" if result.prediction_correct else "INCORRECT"
        tech = result.technical_state
        print(f"  Signal: {result.signal} (RSI: {tech.rsi:.1f}, MACD: {'+'if tech.macd_hist>0 else '-'})")
        print(f"  Price: ${result.price_at_analysis:.2f} -> ${result.price_at_evaluation:.2f} ({result.pct_change:+.2f}%)")
        print(f"  Result: {status} | P&L: {result.hypothetical_pnl:+.2f}%")
    
    def _print_final_stats(self):
        """Print final statistics."""
        print(f"\n{'='*70}")
        print(f"BACKTEST COMPLETE - {self.symbol}")
        print(f"{'='*70}")
        print(f"Total Predictions: {self.stats.total_predictions}")
        print(f"Correct Predictions: {self.stats.correct_predictions}")
        print(f"Accuracy: {self.stats.accuracy:.1f}%")
        print(f"")
        print(f"Signal Distribution:")
        print(f"  BUY:  {self.stats.buy_signals}")
        print(f"  SELL: {self.stats.sell_signals}")
        print(f"  HOLD: {self.stats.hold_signals}")
        print(f"")
        print(f"Hypothetical Performance:")
        print(f"  Total P&L: {self.stats.total_hypothetical_pnl:+.2f}%")
        print(f"  Avg P&L per trade: {self.stats.avg_pnl:+.2f}%")
        
        # Risk Metrics
        risk_metrics = self.stats.get_risk_metrics()
        if risk_metrics:
            print(f"")
            print(f"Risk Metrics:")
            print(f"  Sharpe Ratio:    {risk_metrics['sharpe_ratio']:>8.3f}")
            print(f"  Sortino Ratio:   {risk_metrics['sortino_ratio']:>8.3f}")
            print(f"  Calmar Ratio:    {risk_metrics['calmar_ratio']:>8.3f}")
            print(f"  Max Drawdown:    {risk_metrics['max_drawdown_pct']:>7.2f}%")
            print(f"  VaR (95%):       {risk_metrics['var_95']*100:>7.2f}%")
            print(f"  Win Rate:        {risk_metrics['win_rate']:>7.2f}%")
            print(f"  Profit Factor:   {risk_metrics['profit_factor']:>8.3f}")
            print(f"  Final Equity:    ${self.stats.equity_curve[-1]:>10,.2f}")
        
        print(f"{'='*70}\n")
    
    def _save_results(self):
        """Save backtest results to file."""
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_technical_backtest_{timestamp}.json"
        filepath = BACKTEST_DIR / filename
        
        risk_metrics = self.stats.get_risk_metrics()
        
        data = {
            "symbol": self.symbol,
            "evaluation_hours": self.evaluation_hours,
            "timestamp": timestamp,
            "mode": "technical_only",
            "stats": {
                "total_predictions": self.stats.total_predictions,
                "correct_predictions": self.stats.correct_predictions,
                "accuracy": self.stats.accuracy,
                "buy_signals": self.stats.buy_signals,
                "sell_signals": self.stats.sell_signals,
                "hold_signals": self.stats.hold_signals,
                "total_hypothetical_pnl": self.stats.total_hypothetical_pnl,
                "avg_pnl": self.stats.avg_pnl,
                "final_equity": self.stats.equity_curve[-1],
            },
            "risk_metrics": risk_metrics,
            "equity_curve": self.stats.equity_curve,
            "results": [
                {
                    "date": r.date,
                    "signal": r.signal,
                    "expected_direction": r.expected_direction,
                    "actual_direction": r.actual_direction,
                    "price_at_analysis": r.price_at_analysis,
                    "price_at_evaluation": r.price_at_evaluation,
                    "pct_change": r.pct_change,
                    "prediction_correct": r.prediction_correct,
                    "hypothetical_pnl": r.hypothetical_pnl,
                    "rsi": r.technical_state.rsi if r.technical_state else None,
                    "macd_hist": r.technical_state.macd_hist if r.technical_state else None,
                    "lesson": r.lesson[:500],
                }
                for r in self.stats.results
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        print(f"Lessons stored in memory collection: technical_backtest")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Training - Build Initial Memory Corpus using Technical Analysis"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol to backtest (default: XAUUSD)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Months of history to analyze (default: 6)"
    )
    parser.add_argument(
        "--interval-days",
        type=int,
        default=3,
        help="Days between each analysis (default: 3)"
    )
    parser.add_argument(
        "--evaluation-hours",
        type=int,
        default=72,
        help="Hours until prediction evaluation (default: 72)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to generate (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("XAI_API_KEY"):
        print("Error: XAI_API_KEY not set in environment")
        return
    
    # Run backtest
    backtester = TechnicalBacktester(
        symbol=args.symbol,
        evaluation_hours=args.evaluation_hours
    )
    
    stats = backtester.run_backtest(
        months=args.months,
        interval_days=args.interval_days,
        max_samples=args.max_samples
    )
    
    print(f"\nMemory corpus built with {stats.total_predictions} historical evaluations.")
    print(f"Technical analysis accuracy: {stats.accuracy:.1f}%")
    print(f"\nThe system is now ready for live trading.")
    print(f"Run daily_cycle.py to start learning from news/sentiment as well.")


if __name__ == "__main__":
    main()
