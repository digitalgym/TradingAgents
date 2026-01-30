"""
Tests for Memory System Improvements (Phases 1-5)

These tests verify the implementation of the memory improvement phases:
- Phase 1: Structured Trade Outcome Tracking
- Phase 2: SMC-Specific Learning
- Phase 3: Smarter Memory Retrieval
- Phase 4: Feedback Loop Validation
- Phase 5: Meta-Learning Across Agents

Note: conftest.py handles mocking of heavy dependencies (torch, transformers).
"""

import pytest
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import hashlib


def mock_embedding(text: str) -> List[float]:
    """Generate a deterministic mock embedding based on text hash."""
    h = hashlib.md5(text.encode()).hexdigest()
    embedding = []
    for i in range(0, min(len(h) * 2, 384), 2):
        idx = i % len(h)
        val = int(h[idx:idx+2], 16) / 255.0 - 0.5
        embedding.append(val)
    while len(embedding) < 384:
        embedding.append(0.0)
    return embedding[:384]


# ============================================================================
# PHASE 1: Structured Trade Outcome Tracking Tests
# ============================================================================

class TestPhase1StructuredOutcomes:
    """Test Phase 1: Structured Trade Outcome Tracking"""

    def test_outcome_schema_complete(self):
        """Verify all outcome fields are populated on trade analysis"""
        from tradingagents.trade_decisions import analyze_trade_outcome

        outcome = analyze_trade_outcome(
            entry_price=100.0,
            exit_price=102.0,
            stop_loss=98.0,
            take_profit=105.0,
            direction="BUY",
            max_favorable_price=103.0,
            max_adverse_price=99.5,
        )

        # Check all expected fields exist
        assert outcome["result"] in ["win", "loss", "breakeven"]
        assert outcome["exit_type"] in ["tp_hit", "sl_hit", "manual_close", "trailing_stop", "time_exit", "unknown"]
        assert outcome["direction_correct"] is not None
        assert outcome["sl_placement"] in ["too_tight", "appropriate", "too_wide"]
        assert outcome["tp_placement"] in ["too_ambitious", "appropriate", "too_conservative"]
        assert outcome["entry_quality"] in ["good", "poor", "neutral"]
        assert "lessons" in outcome

    def test_direction_correct_calculation(self):
        """Direction correct when price moved favorable even if SL hit"""
        from tradingagents.trade_decisions import analyze_trade_outcome

        # Trade that hit SL but direction was initially correct
        outcome = analyze_trade_outcome(
            entry_price=100.0,
            exit_price=98.0,  # Hit SL
            stop_loss=98.0,
            take_profit=110.0,
            direction="BUY",
            max_favorable_price=105.0,  # Price DID go up
            max_adverse_price=98.0,
            exit_reason="sl_hit",
        )

        assert outcome["direction_correct"] == True  # Price went up initially
        assert outcome["sl_placement"] == "too_tight"  # But SL was too tight
        assert outcome["result"] == "loss"

    def test_winning_trade_analysis(self):
        """Test analysis of a winning trade"""
        from tradingagents.trade_decisions import analyze_trade_outcome

        outcome = analyze_trade_outcome(
            entry_price=100.0,
            exit_price=105.0,
            stop_loss=98.0,
            take_profit=105.0,
            direction="BUY",
            exit_reason="tp_hit",
        )

        assert outcome["result"] == "win"
        assert outcome["returns_pct"] > 0
        assert outcome["exit_type"] == "tp_hit"

    def test_losing_trade_with_wrong_direction(self):
        """Test analysis when direction prediction was wrong"""
        from tradingagents.trade_decisions import analyze_trade_outcome

        outcome = analyze_trade_outcome(
            entry_price=100.0,
            exit_price=95.0,  # Price went down
            stop_loss=95.0,
            take_profit=110.0,
            direction="BUY",  # Predicted up
            max_favorable_price=100.0,  # Never went favorable at all (same as entry)
            max_adverse_price=95.0,
            exit_reason="sl_hit",
        )

        assert outcome["result"] == "loss"
        assert outcome["direction_correct"] == False  # Never went favorable
        # SL placement should be appropriate - it saved us from worse loss
        assert outcome["sl_placement"] == "appropriate"

    def test_lesson_generation(self):
        """Test that lessons are generated for outcomes"""
        from tradingagents.trade_decisions import analyze_trade_outcome

        # Trade with tight SL but correct direction
        outcome = analyze_trade_outcome(
            entry_price=100.0,
            exit_price=98.0,
            stop_loss=98.0,
            take_profit=110.0,
            direction="BUY",
            max_favorable_price=105.0,
            exit_reason="sl_hit",
        )

        lessons = outcome.get("lessons", [])
        assert len(lessons) > 0
        assert any("tight" in lesson.lower() or "stop" in lesson.lower() for lesson in lessons)


# ============================================================================
# PHASE 2: SMC-Specific Learning Tests
# ============================================================================

class TestPhase2SMCLearning:
    """Test Phase 2: SMC-Specific Learning"""

    @pytest.fixture
    def smc_memory(self, tmp_path, mock_chroma_client):
        """Create a temporary SMC pattern memory with mocked chromadb"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import SMCPatternMemory

            config = {"memory_db_path": str(tmp_path / "memory_db")}
            memory = SMCPatternMemory(config)
            memory.get_embedding = mock_embedding
            return memory

    def test_store_pattern(self, smc_memory):
        """Test storing an SMC pattern"""
        smc_memory.store_pattern(
            decision_id="TEST_001",
            symbol="XAUUSD",
            setup_type="fvg_bounce",
            direction="BUY",
            smc_context={
                "entry_zone": "fvg",
                "entry_zone_strength": 0.85,
                "with_trend": True,
                "confluences": ["ob_fvg_overlap", "pdh_pdl"],
            },
            situation_text="Gold bullish FVG bounce at 5500",
            outcome={"result": "win", "returns_pct": 2.5, "direction_correct": True},
            lesson="FVG bounce with OB overlap works well",
        )

        patterns = smc_memory.get_patterns_by_setup("fvg_bounce")
        assert len(patterns) == 1
        assert patterns[0]["symbol"] == "XAUUSD"
        assert patterns[0]["was_win"] == True

    def test_get_setup_stats(self, smc_memory):
        """Test getting statistics by setup type"""
        for i in range(5):
            smc_memory.store_pattern(
                decision_id=f"TEST_{i:03d}",
                symbol="XAUUSD",
                setup_type="fvg_bounce",
                direction="BUY",
                smc_context={"entry_zone": "fvg", "with_trend": True, "confluences": []},
                situation_text=f"Test pattern {i}",
                outcome={"result": "win" if i < 3 else "loss", "returns_pct": 2.0 if i < 3 else -1.0, "direction_correct": True},
            )

        stats = smc_memory.get_setup_stats(min_samples=3)
        assert "fvg_bounce" in stats
        assert stats["fvg_bounce"]["win_rate"] == 0.6  # 3/5
        assert stats["fvg_bounce"]["sample_size"] == 5

    def test_similar_patterns_retrieval(self, smc_memory):
        """Test retrieving similar patterns"""
        smc_memory.store_pattern(
            decision_id="TEST_SIMILAR",
            symbol="XAUUSD",
            setup_type="fvg_bounce",
            direction="BUY",
            smc_context={"entry_zone": "fvg", "with_trend": True, "confluences": []},
            situation_text="Gold bullish structure with FVG at 5500 support",
            outcome={"result": "win", "returns_pct": 2.5, "direction_correct": True},
        )

        similar = smc_memory.get_similar_patterns(
            situation="Gold bullish structure at 5480 support",
            symbol="XAUUSD",
            n_matches=1,
        )

        assert len(similar) == 1
        assert similar[0]["similarity"] > 0.5


# ============================================================================
# PHASE 3: Smarter Memory Retrieval Tests
# ============================================================================

class TestPhase3SmarterRetrieval:
    """Test Phase 3: Smarter Memory Retrieval"""

    def test_agent_memory_config_exists(self, mock_chroma_client):
        """Test that agent-specific configs are defined"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import AGENT_MEMORY_CONFIG

            assert "trader" in AGENT_MEMORY_CONFIG
            assert "risk_manager" in AGENT_MEMORY_CONFIG
            assert "bull_researcher" in AGENT_MEMORY_CONFIG
            assert "bear_researcher" in AGENT_MEMORY_CONFIG

    def test_role_specific_retrieval_config(self, mock_chroma_client):
        """Test that different agents get different configs"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import get_agent_memory_config

            trader_config = get_agent_memory_config("trader")
            risk_config = get_agent_memory_config("risk_manager")

            assert trader_config["n_matches"] > risk_config["n_matches"]
            assert risk_config["min_confidence"] > trader_config.get("min_confidence", 0)

    def test_risk_manager_focuses_on_losses(self, mock_chroma_client):
        """Risk manager config should focus on losses"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import get_agent_memory_config

            config = get_agent_memory_config("risk_manager")
            assert config.get("focus") == "losses"

    @pytest.fixture
    def memory(self, tmp_path, mock_chroma_client):
        """Create a temporary memory with mocked chromadb"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import FinancialSituationMemory, TIER_SHORT

            config = {"memory_db_path": str(tmp_path / "memory_db")}
            mem = FinancialSituationMemory("test_memory", config)
            mem.get_embedding = mock_embedding
            mem._tier_short = TIER_SHORT
            return mem

    def test_get_memories_for_agent(self, memory, mock_chroma_client):
        """Test agent-specific memory retrieval"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import TIER_SHORT, get_agent_memory_config

            memory.add_situations(
                [("Market is bullish with strong momentum", "Consider buying")],
                tier=TIER_SHORT,
                confidence=0.7,
                prediction_correct=True,
            )

            memories = memory.get_memories_for_agent(
                agent_name="trader",
                current_situation="Market showing bullish signs",
            )

            trader_config = get_agent_memory_config("trader")
            assert len(memories) <= trader_config["n_matches"]


# ============================================================================
# PHASE 4: Feedback Loop Validation Tests
# ============================================================================

class TestPhase4FeedbackLoop:
    """Test Phase 4: Feedback Loop Validation"""

    @pytest.fixture
    def usage_tracker(self, tmp_path, mock_chroma_client):
        """Create a temporary memory usage tracker with mocked chromadb"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import MemoryUsageTracker

            config = {"memory_db_path": str(tmp_path / "memory_db")}
            return MemoryUsageTracker(config)

    def test_track_usage(self, usage_tracker):
        """Test tracking memory usage"""
        usage_tracker.track_usage(
            trade_id="TRADE_001",
            memory_ids=["mem_1", "mem_2"],
            memory_collection="trader_memory",
            agent_name="trader",
        )

        usages = usage_tracker.get_memories_used_for_trade("TRADE_001")
        assert len(usages) == 2
        # Check that each usage has the expected memory_id
        memory_ids = [u.get("memory_id") for u in usages]
        assert "mem_1" in memory_ids
        assert "mem_2" in memory_ids

    def test_validation_stats(self, usage_tracker):
        """Test getting validation statistics"""
        usage_tracker.track_usage(
            trade_id="TRADE_001",
            memory_ids=["mem_1"],
            memory_collection="trader_memory",
            agent_name="trader",
        )

        stats = usage_tracker.get_validation_stats()
        assert stats["total_usages"] == 1
        assert stats["pending"] == 1


# ============================================================================
# PHASE 5: Meta-Learning Across Agents Tests
# ============================================================================

class TestPhase5MetaLearning:
    """Test Phase 5: Meta-Learning Across Agents"""

    @pytest.fixture
    def meta_learning(self, tmp_path, mock_chroma_client):
        """Create a temporary meta-pattern learning instance with mocked chromadb"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.agents.utils.memory import MetaPatternLearning

            config = {"memory_db_path": str(tmp_path / "memory_db")}
            return MetaPatternLearning(config)

    def test_record_trade_outcome(self, meta_learning):
        """Test recording trade outcomes for meta-pattern learning"""
        meta_learning.record_trade_outcome(
            decision_id="TEST_001",
            symbol="XAUUSD",
            bull_signal="bullish",
            bear_signal="bullish",
            final_action="BUY",
            was_successful=True,
            returns_pct=2.5,
            market_regime="trending_bullish",
        )

        stats = meta_learning.get_stats()
        assert stats["total_trades"] == 1

    def test_agreement_stats(self, meta_learning):
        """Test bull/bear agreement statistics"""
        for i in range(6):
            meta_learning.record_trade_outcome(
                decision_id=f"TEST_{i:03d}",
                symbol="XAUUSD",
                bull_signal="bullish",
                bear_signal="bullish",
                final_action="BUY",
                was_successful=i < 4,
                returns_pct=2.0 if i < 4 else -1.0,
            )

        stats = meta_learning.get_agreement_stats(min_samples=5)
        assert "both_bullish" in stats
        assert stats["both_bullish"]["win_rate"] == pytest.approx(4/6, 0.01)

    def test_disagreement_detection(self, meta_learning):
        """Test when bull and bear disagree"""
        for i in range(6):
            meta_learning.record_trade_outcome(
                decision_id=f"DISAGREE_{i:03d}",
                symbol="XAUUSD",
                bull_signal="bullish",
                bear_signal="bearish",
                final_action="BUY" if i < 3 else "SELL",
                was_successful=i % 2 == 0,
                returns_pct=2.0 if i % 2 == 0 else -1.0,
            )

        disagree_stats = meta_learning.get_disagreement_stats(min_samples=5)
        assert "total_disagreements" in disagree_stats
        assert disagree_stats["total_disagreements"] == 6

    def test_meta_insights_for_decision(self, meta_learning):
        """Test getting meta-insights for trading decisions"""
        for i in range(10):
            meta_learning.record_trade_outcome(
                decision_id=f"INSIGHT_{i:03d}",
                symbol="XAUUSD",
                bull_signal="bullish",
                bear_signal="bullish",
                final_action="BUY",
                was_successful=i < 7,
                returns_pct=2.0 if i < 7 else -1.0,
                market_regime="trending_bullish",
            )

        insights = meta_learning.get_meta_insights_for_decision(
            bull_signal="bullish",
            bear_signal="bullish",
            market_regime="trending_bullish",
        )

        assert insights.get("agreement_insight") is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full learning cycle"""

    def test_outcome_to_smc_pattern_flow(self, tmp_path, mock_chroma_client):
        """Test flow from trade outcome to SMC pattern storage"""
        with patch('chromadb.PersistentClient', return_value=mock_chroma_client):
            from tradingagents.trade_decisions import analyze_trade_outcome
            from tradingagents.agents.utils.memory import SMCPatternMemory

            config = {"memory_db_path": str(tmp_path / "memory_db")}
            smc_memory = SMCPatternMemory(config)
            smc_memory.get_embedding = mock_embedding

            outcome = analyze_trade_outcome(
                entry_price=5500.0,
                exit_price=5550.0,
                stop_loss=5470.0,
                take_profit=5550.0,
                direction="BUY",
                exit_reason="tp_hit",
            )

            smc_memory.store_pattern(
                decision_id="INTEGRATION_TEST",
                symbol="XAUUSD",
                setup_type="fvg_bounce",
                direction="BUY",
                smc_context={
                    "entry_zone": "fvg",
                    "entry_zone_strength": 0.8,
                    "with_trend": True,
                    "confluences": ["pdh_pdl"],
                },
                situation_text="Gold FVG bounce integration test",
                outcome=outcome,
                lesson=" ".join(outcome.get("lessons", [])),
            )

            patterns = smc_memory.get_patterns_by_setup("fvg_bounce")
            assert len(patterns) == 1
            assert patterns[0]["was_win"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
