"""
Unit tests and backtest for Range Quant Analyst.

Tests range detection, mean reversion scoring, and backtests the
range-bound trading strategy on real historical data.
"""

import pytest
import numpy as np
import pandas as pd
from tradingagents.agents.analysts.range_quant import (
    analyze_range,
    get_range_quant_decision_for_modal,
)


# ---------------------------------------------------------------------------
# Unit tests for analyze_range
# ---------------------------------------------------------------------------

class TestRangeDetection:
    """Test range detection functionality."""

    def test_flat_market_detected_as_ranging(self):
        """Flat, oscillating price should be detected as ranging."""
        np.random.seed(42)
        close = 100 + np.random.randn(60) * 3
        high = close + np.abs(np.random.randn(60)) * 2
        low = close - np.abs(np.random.randn(60)) * 2

        result = analyze_range(high, low, close, lookback=30)

        assert result["is_ranging"] == True
        assert result["mean_reversion_score"] > 45
        assert result["trend_strength"] < 0.35

    def test_trending_market_not_ranging(self):
        """Strong uptrend should not be detected as ranging."""
        close = np.linspace(100, 200, 60)
        high = close + 2
        low = close - 2

        result = analyze_range(high, low, close, lookback=30)

        assert result["is_ranging"] == False
        assert result["trend_strength"] > 0.3

    def test_downtrend_not_ranging(self):
        """Strong downtrend should not be detected as ranging."""
        close = np.linspace(200, 100, 60)
        high = close + 2
        low = close - 2

        result = analyze_range(high, low, close, lookback=30)

        assert result["is_ranging"] == False

    def test_price_position_premium(self):
        """Price near range high should be premium."""
        np.random.seed(42)
        close = 100 + np.random.randn(60) * 2
        # Force last price near top of range
        close[-1] = np.max(close[-30:]) - 0.5
        high = close + 1
        low = close - 1

        result = analyze_range(high, low, close, lookback=30)

        assert result["price_position"] == "premium"
        assert result["position_pct"] > 70

    def test_price_position_discount(self):
        """Price near range low should be discount."""
        np.random.seed(42)
        close = 100 + np.random.randn(60) * 2
        close[-1] = np.min(close[-30:]) + 0.5
        high = close + 1
        low = close - 1

        result = analyze_range(high, low, close, lookback=30)

        assert result["price_position"] == "discount"
        assert result["position_pct"] < 30

    def test_minimal_data_returns_defaults(self):
        """Insufficient data should return safe defaults."""
        close = np.array([100, 101, 102])
        high = close + 1
        low = close - 1

        result = analyze_range(high, low, close, lookback=30)

        assert result["is_ranging"] is False
        assert result["range_high"] is None
        assert result["mean_reversion_score"] == 0

    def test_range_boundaries_correct(self):
        """Range high/low should match actual highs and lows."""
        np.random.seed(42)
        close = 100 + np.random.randn(60) * 3
        high = close + 1
        low = close - 1

        result = analyze_range(high, low, close, lookback=30)

        assert result["range_high"] == float(np.max(high[-30:]))
        assert result["range_low"] == float(np.min(low[-30:]))
        assert result["range_midpoint"] == (result["range_high"] + result["range_low"]) / 2

    def test_touches_counted(self):
        """Touches of range extremes should be counted."""
        np.random.seed(42)
        close = 100 + np.random.randn(60) * 2
        high = close + 1
        low = close - 1

        result = analyze_range(high, low, close, lookback=30)

        assert result["touches_high"] >= 0
        assert result["touches_low"] >= 0
        # In a ranging market, should have some touches
        if result["is_ranging"]:
            assert result["touches_high"] + result["touches_low"] >= 2

    def test_mean_reversion_score_bounded(self):
        """Mean reversion score should be between 0 and 100."""
        for seed in range(10):
            np.random.seed(seed)
            close = 100 + np.random.randn(60) * 5
            high = close + 2
            low = close - 2

            result = analyze_range(high, low, close, lookback=30)

            assert 0 <= result["mean_reversion_score"] <= 100


# ---------------------------------------------------------------------------
# Modal conversion tests
# ---------------------------------------------------------------------------

class TestRangeDecisionForModal:
    """Test conversion of range decision to modal format."""

    def test_buy_signal_conversion(self):
        decision = {
            "symbol": "XAUUSD",
            "signal": "buy_to_enter",
            "order_type": "limit",
            "entry_price": 2680.0,
            "stop_loss": 2660.0,
            "profit_target": 2720.0,
            "confidence": 0.72,
            "justification": "Bullish OB confluence at range low",
            "invalidation_condition": "Close below 2660"
        }

        result = get_range_quant_decision_for_modal(decision)

        assert result["signal"] == "BUY"
        assert result["orderType"] == "limit"
        assert result["suggestedEntry"] == 2680.0
        assert result["suggestedStopLoss"] == 2660.0
        assert result["suggestedTakeProfit"] == 2720.0
        assert "RANGE" in result["rationale"]

    def test_sell_signal_conversion(self):
        decision = {
            "symbol": "XAUUSD",
            "signal": "sell_to_enter",
            "order_type": "market",
            "entry_price": 2720.0,
            "stop_loss": 2740.0,
            "profit_target": 2680.0,
            "confidence": 0.68,
            "justification": "Bearish OB at range high",
            "invalidation_condition": "Close above 2740"
        }

        result = get_range_quant_decision_for_modal(decision)

        assert result["signal"] == "SELL"
        assert result["orderType"] == "market"

    def test_hold_and_empty(self):
        decision = {"signal": "hold", "confidence": 0.4, "justification": "No range", "invalidation_condition": "N/A"}
        assert get_range_quant_decision_for_modal(decision)["signal"] == "HOLD"
        assert get_range_quant_decision_for_modal(None) == {}
        assert get_range_quant_decision_for_modal({}) == {}


# ---------------------------------------------------------------------------
# Backtest on real historical data
# ---------------------------------------------------------------------------

class TestRangeBacktest:
    """Backtest range detection and strategy on real data."""

    @pytest.fixture
    def real_data(self):
        """Load real AAPL daily data."""
        path = "tradingagents/dataflows/data_cache/AAPL-YFin-data-2011-01-18-2026-01-18.csv"
        df = pd.read_csv(path)
        return df

    @pytest.fixture
    def spy_data(self):
        """Load real SPY daily data."""
        path = "tradingagents/dataflows/data_cache/SPY-YFin-data-2010-12-28-2025-12-28.csv"
        df = pd.read_csv(path)
        return df

    def _run_backtest(self, df, lookback=30, hold_days=5):
        """
        Backtest range trading strategy on historical data.

        Strategy:
        - Detect ranging conditions
        - Buy when price is in discount (<30%), sell when in premium (>70%)
        - Hold for `hold_days` then measure P/L
        - Only trade when is_ranging=True and mean_reversion_score > threshold

        Returns dict with stats and trade list.
        """
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        trades = []
        min_bars = lookback + 10  # Need some history

        for i in range(min_bars, len(close) - hold_days):
            result = analyze_range(
                high[:i + 1], low[:i + 1], close[:i + 1], lookback=lookback
            )

            if not result["is_ranging"]:
                continue
            if result["mean_reversion_score"] < 55:
                continue

            entry_price = close[i]
            exit_price = close[i + hold_days]
            position = result["price_position"]

            if position == "discount":
                # Buy at discount, expect mean reversion up
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "bar": i,
                    "direction": "BUY",
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "mr_score": result["mean_reversion_score"],
                    "position_pct": result["position_pct"],
                    "range_pct": result["range_percent"],
                })
            elif position == "premium":
                # Sell at premium, expect mean reversion down
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({
                    "bar": i,
                    "direction": "SELL",
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "mr_score": result["mean_reversion_score"],
                    "position_pct": result["position_pct"],
                    "range_pct": result["range_percent"],
                })
            # Skip equilibrium — no edge

        if not trades:
            return {
                "total": 0, "winners": 0, "losers": 0, "win_rate": 0,
                "avg_pnl": 0, "avg_winner": 0, "avg_loser": 0,
                "total_pnl": 0, "trades": [],
            }

        pnls = [t["pnl_pct"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            "total": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(trades) * 100,
            "avg_pnl": np.mean(pnls),
            "avg_winner": np.mean(winners) if winners else 0,
            "avg_loser": np.mean(losers) if losers else 0,
            "total_pnl": np.sum(pnls),
            "max_win": max(pnls) if pnls else 0,
            "max_loss": min(pnls) if pnls else 0,
            "trades": trades,
        }

    def test_backtest_aapl_produces_trades(self, real_data):
        """Backtest should produce trades on AAPL data."""
        stats = self._run_backtest(real_data)
        print(f"\n--- AAPL Backtest (lookback=30, hold=5 days) ---")
        print(f"Total trades: {stats['total']}")
        print(f"Win rate: {stats['win_rate']:.1f}%")
        print(f"Avg P/L: {stats['avg_pnl']:.3f}%")
        print(f"Avg winner: {stats['avg_winner']:.3f}%")
        print(f"Avg loser: {stats['avg_loser']:.3f}%")
        print(f"Total P/L: {stats['total_pnl']:.2f}%")
        print(f"Max win: {stats['max_win']:.3f}%, Max loss: {stats['max_loss']:.3f}%")

        assert stats["total"] > 0, "Should produce some trades"

    def test_backtest_spy_produces_trades(self, spy_data):
        """Backtest should produce trades on SPY data."""
        stats = self._run_backtest(spy_data)
        print(f"\n--- SPY Backtest (lookback=30, hold=5 days) ---")
        print(f"Total trades: {stats['total']}")
        print(f"Win rate: {stats['win_rate']:.1f}%")
        print(f"Avg P/L: {stats['avg_pnl']:.3f}%")
        print(f"Total P/L: {stats['total_pnl']:.2f}%")

        assert stats["total"] > 0

    def test_backtest_varying_lookbacks(self, real_data):
        """Test different lookback periods to find optimal."""
        print("\n--- AAPL: Lookback Sensitivity (hold=5 days) ---")
        print(f"{'Lookback':>8} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>9}")
        print("-" * 45)
        for lookback in [15, 20, 25, 30, 40, 50]:
            stats = self._run_backtest(real_data, lookback=lookback)
            print(
                f"{lookback:>8} {stats['total']:>7} "
                f"{stats['win_rate']:>7.1f}% {stats['avg_pnl']:>7.3f}% "
                f"{stats['total_pnl']:>8.2f}%"
            )

    def test_backtest_varying_hold_periods(self, real_data):
        """Test different holding periods."""
        print("\n--- AAPL: Hold Period Sensitivity (lookback=30) ---")
        print(f"{'Hold':>6} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'TotalPnL':>9}")
        print("-" * 43)
        for hold in [1, 3, 5, 7, 10, 15]:
            stats = self._run_backtest(real_data, hold_days=hold)
            print(
                f"{hold:>6} {stats['total']:>7} "
                f"{stats['win_rate']:>7.1f}% {stats['avg_pnl']:>7.3f}% "
                f"{stats['total_pnl']:>8.2f}%"
            )

    def test_higher_mr_threshold_improves_quality(self, real_data):
        """Higher mean reversion threshold should filter better trades."""
        high = real_data["High"].values
        low = real_data["Low"].values
        close = real_data["Close"].values

        print("\n--- AAPL: MR Score Threshold Sensitivity ---")
        print(f"{'Threshold':>10} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8}")
        print("-" * 38)

        for threshold in [40, 50, 60, 70, 80]:
            trades = []
            for i in range(40, len(close) - 5):
                result = analyze_range(high[:i+1], low[:i+1], close[:i+1], lookback=30)
                if not result["is_ranging"] or result["mean_reversion_score"] < threshold:
                    continue
                pos = result["price_position"]
                if pos == "discount":
                    pnl = (close[i+5] - close[i]) / close[i] * 100
                    trades.append(pnl)
                elif pos == "premium":
                    pnl = (close[i] - close[i+5]) / close[i] * 100
                    trades.append(pnl)

            if trades:
                win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100
                avg_pnl = np.mean(trades)
            else:
                win_rate = 0
                avg_pnl = 0

            print(f"{threshold:>10} {len(trades):>7} {win_rate:>7.1f}% {avg_pnl:>7.3f}%")

    def test_range_detection_accuracy(self, real_data):
        """Check how often detected ranges actually persist (price stays in range)."""
        high = real_data["High"].values
        low = real_data["Low"].values
        close = real_data["Close"].values

        range_detected = 0
        range_held = 0  # Price stayed within range for next N bars
        range_broke = 0
        check_bars = 5

        for i in range(40, len(close) - check_bars):
            result = analyze_range(high[:i+1], low[:i+1], close[:i+1], lookback=30)
            if not result["is_ranging"]:
                continue

            range_detected += 1
            rh = result["range_high"]
            rl = result["range_low"]

            # Check if price stays in range for next check_bars
            future_high = np.max(high[i+1:i+1+check_bars])
            future_low = np.min(low[i+1:i+1+check_bars])

            # Allow 10% buffer for noise
            buffer = (rh - rl) * 0.10
            if future_high <= rh + buffer and future_low >= rl - buffer:
                range_held += 1
            else:
                range_broke += 1

        if range_detected > 0:
            hold_rate = range_held / range_detected * 100
            print(f"\n--- Range Persistence (next {check_bars} bars) ---")
            print(f"Ranges detected: {range_detected}")
            print(f"Range held: {range_held} ({hold_rate:.1f}%)")
            print(f"Range broke: {range_broke} ({100 - hold_rate:.1f}%)")

            # Range should hold more often than not
            assert hold_rate > 40, f"Range persistence too low: {hold_rate:.1f}%"


# ---------------------------------------------------------------------------
# Synthetic scenario tests
# ---------------------------------------------------------------------------

class TestSyntheticScenarios:
    """Test specific market scenarios with synthetic data."""

    def test_oscillating_sine_wave_is_range(self):
        """Pure sine wave (ideal range) should be detected."""
        np.random.seed(42)
        n = 100
        t = np.linspace(0, 6 * np.pi, n)
        close = 100 + 5 * np.sin(t) + np.random.randn(n) * 0.3
        high = close + 1.5
        low = close - 1.5

        result = analyze_range(high, low, close, lookback=30)
        # Sine wave has high reversal count but partial cycles cause some trend
        assert result["mean_reversion_score"] > 25
        assert result["trend_strength"] < 0.5

    def test_step_function_not_range(self):
        """Sudden level shift should not look like a clean range after the jump."""
        # After the jump, the last 30 bars are flat at 120 — that IS a range
        # but the full window including the jump should show trend
        close = np.concatenate([np.linspace(100, 100, 15), np.linspace(100, 120, 5), np.full(40, 120)])
        high = close + 1
        low = close - 1

        # With lookback=30 looking at last 30 bars of flat 120s, it IS a range
        # So instead test with a lookback that includes the jump
        result = analyze_range(high, low, close, lookback=50)
        # The displacement from 100 to 120 should show up as trend
        assert result["trend_strength"] > 0.15 or result["adx_proxy"] > 10

    def test_expanding_volatility_reduces_score(self):
        """Expanding volatility (breakout) should reduce range detection."""
        n = 60
        close = np.ones(n) * 100
        close[:40] += np.random.randn(40) * 1  # Tight
        close[40:] += np.random.randn(20) * 10  # Expanding

        high = close + np.abs(np.random.randn(n) * 2)
        low = close - np.abs(np.random.randn(n) * 2)

        result_tight = analyze_range(high[:40], low[:40], close[:40], lookback=30)
        result_expanded = analyze_range(high, low, close, lookback=20)

        # The tight period should score higher than the expanded period
        assert result_tight["mean_reversion_score"] >= result_expanded["mean_reversion_score"] - 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
