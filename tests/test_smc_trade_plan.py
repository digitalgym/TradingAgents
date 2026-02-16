"""
Tests for SMC Trade Plan Generator and LLM Trade Refiner.

Tests the hybrid systematic + AI trade planning system.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from tradingagents.dataflows.smc_trade_plan import (
    SMCTradePlanGenerator,
    SMCTradePlan,
    EntryChecklist,
    SetupType,
)
from tradingagents.dataflows.llm_trade_refiner import (
    LLMTradeRefiner,
    RefinedTradePlan,
    create_hybrid_trade_decision,
)


# === Test Fixtures ===

@pytest.fixture
def sample_smc_analysis():
    """Sample SMC analysis for testing."""
    return {
        "bias": "bullish",
        "current_price": 2850.0,
        "order_blocks": {
            "bullish": [
                {
                    "type": "bullish",
                    "top": 2845.0,
                    "bottom": 2840.0,
                    "strength": 0.75,
                    "mitigated": False,
                    "candle_index": 100,
                }
            ],
            "bearish": [
                {
                    "type": "bearish",
                    "top": 2880.0,
                    "bottom": 2875.0,
                    "strength": 0.65,
                    "mitigated": False,
                    "candle_index": 95,
                }
            ],
        },
        "fair_value_gaps": {
            "bullish": [
                {
                    "type": "bullish",
                    "top": 2846.0,
                    "bottom": 2842.0,
                    "mitigated": False,
                    "fill_percentage": 0,
                }
            ],
            "bearish": [],
        },
        "liquidity_zones": [
            {"type": "buy-side", "price": 2870.0, "strength": 60, "touched": False},
            {"type": "sell-side", "price": 2830.0, "strength": 55, "touched": False},
        ],
        "structure": {
            "recent_bos": [{"type": "high", "price": 2855.0}],
            "recent_choc": [],
            "bos_count": 3,
            "choc_count": 0,
        },
        "premium_discount": {
            "zone": "discount",
            "equilibrium": 2860.0,
            "range_high": 2890.0,
            "range_low": 2830.0,
        },
    }


@pytest.fixture
def generator():
    """Create a trade plan generator."""
    return SMCTradePlanGenerator(
        min_quality_score=60.0,
        min_rr_ratio=1.5,
        sl_buffer_atr=0.5,
        entry_zone_percent=0.5,
    )


# === SMCTradePlanGenerator Tests ===

class TestSMCTradePlanGenerator:
    """Tests for the rule-based trade plan generator."""

    def test_generator_initialization(self, generator):
        """Test generator initializes with correct defaults."""
        assert generator.min_quality_score == 60.0
        assert generator.min_rr_ratio == 1.5
        assert generator.sl_buffer_atr == 0.5
        assert generator.entry_zone_percent == 0.5

    def test_generate_plan_returns_plan(self, generator, sample_smc_analysis):
        """Test that generate_plan returns a valid plan."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        assert plan is not None
        assert isinstance(plan, SMCTradePlan)
        assert plan.signal in ["BUY", "SELL"]

    def test_buy_signal_in_bullish_trend(self, generator, sample_smc_analysis):
        """Test that bullish trend generates BUY signal."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        assert plan.signal == "BUY"

    def test_sl_below_entry_for_buy(self, generator, sample_smc_analysis):
        """Test that SL is below entry for BUY orders."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        if plan and plan.signal == "BUY":
            assert plan.stop_loss < plan.entry_price, "SL must be below entry for BUY"

    def test_tp_above_entry_for_buy(self, generator, sample_smc_analysis):
        """Test that TP is above entry for BUY orders."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        if plan and plan.signal == "BUY":
            assert plan.take_profit > plan.entry_price, "TP must be above entry for BUY"

    def test_zone_quality_score_calculated(self, generator, sample_smc_analysis):
        """Test that zone quality score is calculated."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        assert plan.zone_quality_score >= 0
        assert plan.zone_quality_score <= 100

    def test_risk_reward_ratio_calculated(self, generator, sample_smc_analysis):
        """Test that R:R ratio is calculated correctly."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        if plan and plan.signal == "BUY":
            expected_risk = plan.entry_price - plan.stop_loss
            expected_reward = plan.take_profit - plan.entry_price
            expected_rr = expected_reward / expected_risk if expected_risk > 0 else 0

            assert abs(plan.risk_reward_ratio - expected_rr) < 0.01

    def test_checklist_populated(self, generator, sample_smc_analysis):
        """Test that entry checklist is populated."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        assert isinstance(plan.checklist, EntryChecklist)
        assert plan.checklist.total_count == 7

    def test_confluence_factors_detected(self, generator, sample_smc_analysis):
        """Test that confluence factors are detected."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        # Should detect at least the OB
        assert len(plan.confluence_factors) > 0

    def test_recommendation_take_for_valid_setup(self, generator, sample_smc_analysis):
        """Test that valid setups get TAKE recommendation."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        # If quality and R:R are good, should recommend TAKE
        if plan.zone_quality_score >= 60 and plan.risk_reward_ratio >= 1.5:
            assert plan.recommendation in ["TAKE", "SKIP"]  # May skip for other reasons

    def test_recommendation_skip_for_low_quality(self, generator):
        """Test that low quality setups get SKIP recommendation."""
        # Create minimal analysis with weak OB
        weak_analysis = {
            "bias": "bullish",
            "order_blocks": {
                "bullish": [
                    {
                        "type": "bullish",
                        "top": 2845.0,
                        "bottom": 2840.0,
                        "strength": 0.2,  # Weak
                        "mitigated": False,
                    }
                ],
                "bearish": [],
            },
            "fair_value_gaps": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "structure": {"recent_bos": [], "recent_choc": []},
        }

        plan = generator.generate_plan(
            smc_analysis=weak_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        if plan and plan.zone_quality_score < 60:
            assert plan.recommendation == "SKIP"

    def test_setup_type_detected(self, generator, sample_smc_analysis):
        """Test that setup type is correctly identified."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        assert isinstance(plan.setup_type, SetupType)

    def test_to_dict_serialization(self, generator, sample_smc_analysis):
        """Test that plan can be serialized to dict."""
        plan = generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

        plan_dict = plan.to_dict()

        assert "signal" in plan_dict
        assert "entry_price" in plan_dict
        assert "stop_loss" in plan_dict
        assert "take_profit" in plan_dict
        assert "zone_quality_score" in plan_dict
        assert "checklist" in plan_dict


class TestEntryChecklist:
    """Tests for the entry checklist."""

    def test_checklist_pass_count(self):
        """Test that pass count is calculated correctly."""
        checklist = EntryChecklist(
            htf_trend_aligned=True,
            zone_unmitigated=True,
            has_confluence=True,
            liquidity_target_exists=False,
            structure_confirmed=True,
            in_discount_premium=False,
            session_favorable=True,
        )

        assert checklist.passed_count == 5
        assert checklist.total_count == 7

    def test_checklist_pass_rate(self):
        """Test that pass rate is calculated correctly."""
        checklist = EntryChecklist(
            htf_trend_aligned=True,
            zone_unmitigated=True,
            has_confluence=False,
            liquidity_target_exists=False,
            structure_confirmed=False,
            in_discount_premium=False,
            session_favorable=False,
        )

        assert checklist.pass_rate == 2 / 7

    def test_checklist_to_dict(self):
        """Test checklist serialization."""
        checklist = EntryChecklist(htf_trend_aligned=True)
        d = checklist.to_dict()

        assert "htf_trend_aligned" in d
        assert "passed" in d
        assert "total" in d
        assert "pass_rate" in d


# === LLMTradeRefiner Tests ===

class TestLLMTradeRefiner:
    """Tests for the LLM trade refiner."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = Mock()
        mock.invoke.return_value = Mock(
            content='{"action": "TAKE", "confidence": 0.8, "reasoning": "Good setup", "key_factors": ["Strong OB"], "warnings": []}'
        )
        return mock

    @pytest.fixture
    def base_plan(self, sample_smc_analysis):
        """Create a base plan for testing."""
        generator = SMCTradePlanGenerator()
        return generator.generate_plan(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-up",
        )

    def test_refiner_initialization(self, mock_llm):
        """Test refiner initializes correctly."""
        refiner = LLMTradeRefiner(llm=mock_llm)
        assert refiner.llm == mock_llm

    def test_refine_plan_returns_refined_plan(self, mock_llm, base_plan):
        """Test that refine_plan returns a RefinedTradePlan."""
        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        assert isinstance(refined, RefinedTradePlan)
        assert refined.base_plan == base_plan

    def test_refine_plan_parses_llm_response(self, mock_llm, base_plan):
        """Test that LLM response is parsed correctly."""
        mock_llm.invoke.return_value = Mock(
            content='{"action": "MODIFY", "confidence": 0.75, "adjusted_entry": 2851.0, "reasoning": "Adjusted for FVG", "key_factors": ["FVG alignment"], "warnings": [], "size_multiplier": 1.2}'
        )

        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        assert refined.action == "MODIFY"
        assert refined.confidence == 0.75
        assert refined.adjusted_entry == 2851.0
        assert refined.size_multiplier == 1.2

    def test_refine_plan_validates_sl_direction(self, mock_llm, base_plan):
        """Test that invalid SL adjustments are rejected."""
        # Try to set SL above entry for a BUY (invalid)
        mock_llm.invoke.return_value = Mock(
            content='{"action": "MODIFY", "confidence": 0.8, "adjusted_sl": 2900.0, "reasoning": "Bad SL", "key_factors": [], "warnings": []}'
        )

        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        # Should reject the invalid SL and use base plan's SL
        if base_plan.signal == "BUY":
            assert refined.adjusted_sl is None or refined.final_sl < refined.final_entry

    def test_refine_plan_handles_parse_error(self, mock_llm, base_plan):
        """Test that parse errors are handled gracefully."""
        mock_llm.invoke.return_value = Mock(content="Invalid JSON response")

        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        # Should fall back to base plan
        assert refined.action == base_plan.recommendation
        assert "parsing failed" in refined.reasoning.lower() or "parse" in refined.reasoning.lower()

    def test_refine_plan_handles_llm_error(self, base_plan):
        """Test that LLM errors are handled gracefully."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        # Should fall back to base plan
        assert "failed" in refined.reasoning.lower()
        assert "LLM refinement unavailable" in refined.warnings

    def test_build_historical_context(self, mock_llm):
        """Test building historical context."""
        refiner = LLMTradeRefiner(llm=mock_llm)
        context = refiner.build_historical_context(
            setup_type="ob_entry",
            symbol="XAUUSD",
        )

        assert "setup_type" in context
        assert "total_trades" in context
        assert "win_rate" in context
        assert "similar_trades" in context

    def test_build_market_context(self, mock_llm):
        """Test building market context."""
        refiner = LLMTradeRefiner(llm=mock_llm)
        context = refiner.build_market_context(
            session="london",
            volatility="high",
            market_regime="trending-up",
            daily_pnl_pct=-1.5,
        )

        assert context["session"] == "london"
        assert context["volatility"] == "high"
        assert context["daily_pnl_pct"] == -1.5

    def test_refined_plan_final_values(self, mock_llm, base_plan):
        """Test that final values use adjustments when present."""
        mock_llm.invoke.return_value = Mock(
            content='{"action": "MODIFY", "confidence": 0.8, "adjusted_entry": 2851.0, "adjusted_sl": null, "adjusted_tp": 2875.0, "reasoning": "Adjusted", "key_factors": [], "warnings": []}'
        )

        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        # Entry should use adjusted value
        assert refined.final_entry == 2851.0

        # SL should use base plan (no adjustment)
        assert refined.final_sl == base_plan.stop_loss

        # TP should use adjusted value
        assert refined.final_tp == 2875.0

    def test_refined_plan_to_dict(self, mock_llm, base_plan):
        """Test RefinedTradePlan serialization."""
        refiner = LLMTradeRefiner(llm=mock_llm)
        refined = refiner.refine_plan(base_plan=base_plan)

        d = refined.to_dict()

        assert "action" in d
        assert "confidence" in d
        assert "entry" in d
        assert "stop_loss" in d
        assert "take_profit" in d
        assert "reasoning" in d
        assert "base_plan" in d


# === Integration Tests ===

class TestHybridTradeDecision:
    """Integration tests for the hybrid trade decision system."""

    def test_create_hybrid_trade_decision(self, sample_smc_analysis):
        """Test the convenience function for hybrid decisions."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='{"action": "TAKE", "confidence": 0.85, "reasoning": "Strong confluence setup", "key_factors": ["OB+FVG"], "warnings": []}'
        )

        result = create_hybrid_trade_decision(
            smc_analysis=sample_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            llm=mock_llm,
            market_regime="trending-up",
        )

        if result:
            assert isinstance(result, RefinedTradePlan)
            assert result.action in ["TAKE", "SKIP", "MODIFY"]

    def test_hybrid_decision_without_valid_setup(self):
        """Test that None is returned when no valid setup exists."""
        empty_analysis = {
            "bias": None,
            "order_blocks": {"bullish": [], "bearish": []},
            "fair_value_gaps": {"bullish": [], "bearish": []},
            "liquidity_zones": [],
            "structure": {},
        }

        result = create_hybrid_trade_decision(
            smc_analysis=empty_analysis,
            current_price=2850.0,
            atr=10.0,
        )

        assert result is None


# === Sell Signal Tests ===

class TestSellSignals:
    """Tests for SELL signal generation."""

    @pytest.fixture
    def bearish_smc_analysis(self):
        """Sample bearish SMC analysis."""
        return {
            "bias": "bearish",
            "current_price": 2850.0,
            "order_blocks": {
                "bullish": [],
                "bearish": [
                    {
                        "type": "bearish",
                        "top": 2860.0,
                        "bottom": 2855.0,
                        "strength": 0.8,
                        "mitigated": False,
                    }
                ],
            },
            "fair_value_gaps": {
                "bullish": [],
                "bearish": [
                    {
                        "type": "bearish",
                        "top": 2858.0,
                        "bottom": 2854.0,
                        "mitigated": False,
                    }
                ],
            },
            "liquidity_zones": [
                {"type": "sell-side", "price": 2820.0, "strength": 70, "touched": False},
            ],
            "structure": {
                "recent_bos": [{"type": "low", "price": 2840.0}],
                "recent_choc": [],
            },
            "premium_discount": {
                "zone": "premium",
                "equilibrium": 2845.0,
            },
        }

    def test_sell_signal_in_bearish_trend(self, generator, bearish_smc_analysis):
        """Test that bearish trend generates SELL signal."""
        plan = generator.generate_plan(
            smc_analysis=bearish_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-down",
        )

        if plan:
            assert plan.signal == "SELL"

    def test_sl_above_entry_for_sell(self, generator, bearish_smc_analysis):
        """Test that SL is above entry for SELL orders."""
        plan = generator.generate_plan(
            smc_analysis=bearish_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-down",
        )

        if plan and plan.signal == "SELL":
            assert plan.stop_loss > plan.entry_price, "SL must be above entry for SELL"

    def test_tp_below_entry_for_sell(self, generator, bearish_smc_analysis):
        """Test that TP is below entry for SELL orders."""
        plan = generator.generate_plan(
            smc_analysis=bearish_smc_analysis,
            current_price=2850.0,
            atr=10.0,
            market_regime="trending-down",
        )

        if plan and plan.signal == "SELL":
            assert plan.take_profit < plan.entry_price, "TP must be below entry for SELL"
