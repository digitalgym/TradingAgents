import functools
import time
import json
from typing import Optional

from tradingagents.agents.utils.memory import SMCPatternMemory
from tradingagents.agents.utils.agent_utils import record_memory_usage, extract_memory_ids_from_results
from tradingagents.learning.trade_similarity import TradeSimilaritySearch


def create_trader(llm, memory, smc_memory: Optional[SMCPatternMemory] = None, use_hybrid_smc: bool = True):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get current broker price - CRITICAL for accurate price levels
        current_price = state.get("current_price")

        # Get SMC context if available
        smc_context = state.get("smc_context") or ""
        smc_analysis = state.get("smc_analysis")

        # Get market regime info
        market_regime = state.get("market_regime")
        volatility_regime = state.get("volatility_regime")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"

        # === HYBRID SMC TRADE PLAN ===
        # Generate systematic plan + LLM refinement if SMC analysis is available
        hybrid_plan_context = ""
        refined_plan = None

        if use_hybrid_smc and smc_analysis and current_price:
            try:
                from tradingagents.dataflows.smc_trade_plan import SMCTradePlanGenerator
                from tradingagents.dataflows.llm_trade_refiner import LLMTradeRefiner

                # Get ATR from SMC analysis
                atr = None
                for tf_key in ['1H', '4H', 'D1']:
                    tf_data = smc_analysis.get(tf_key, {})
                    if tf_data.get('atr'):
                        atr = tf_data['atr']
                        break

                if not atr:
                    # Estimate ATR as 1% of price if not available
                    atr = current_price * 0.01

                # Flatten SMC analysis for the generator (use primary timeframe)
                flat_smc = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1') or {}

                # Step 1: Generate rule-based plan
                generator = SMCTradePlanGenerator()
                base_plan = generator.generate_plan(
                    smc_analysis=flat_smc,
                    current_price=current_price,
                    atr=atr,
                    market_regime=market_regime,
                    session=None,  # Could add session detection
                )

                if base_plan:
                    # Step 2: Refine with LLM
                    refiner = LLMTradeRefiner(llm=llm)

                    # Build historical context
                    historical_context = refiner.build_historical_context(
                        setup_type=base_plan.setup_type.value,
                        symbol=company_name,
                        smc_memory=smc_memory,
                        trade_similarity=TradeSimilaritySearch() if TradeSimilaritySearch else None,
                    )

                    # Build market context
                    market_context = refiner.build_market_context(
                        market_regime=market_regime,
                        volatility=volatility_regime,
                    )

                    # Refine
                    refined_plan = refiner.refine_plan(
                        base_plan=base_plan,
                        historical_context=historical_context,
                        market_context=market_context,
                    )

                    # Format for prompt
                    hybrid_plan_context = f"""
=== SYSTEMATIC SMC TRADE PLAN ===
The following trade plan was generated using systematic SMC rules and refined by AI analysis.

RULE-BASED PLAN:
Signal: {base_plan.signal}
Entry: {base_plan.entry_price:.5f}
Stop Loss: {base_plan.stop_loss:.5f}
Take Profit: {base_plan.take_profit:.5f}
Zone Quality: {base_plan.zone_quality_score:.0f}/100
Setup Type: {base_plan.setup_type.value}
R:R Ratio: {base_plan.risk_reward_ratio:.2f}
Checklist: {base_plan.checklist.passed_count}/{base_plan.checklist.total_count} passed
Confluence: {', '.join(base_plan.confluence_factors)}
Rule Recommendation: {base_plan.recommendation}
{f"Skip Reason: {base_plan.skip_reason}" if base_plan.skip_reason else ""}

AI REFINEMENT:
Action: {refined_plan.action}
Confidence: {refined_plan.confidence:.0%}
Final Entry: {refined_plan.final_entry:.5f}
Final SL: {refined_plan.final_sl:.5f}
Final TP: {refined_plan.final_tp:.5f}
Position Size Multiplier: {refined_plan.size_multiplier:.1f}x
{f"Partial TPs: {refined_plan.partial_tp_levels}" if refined_plan.partial_tp_levels else ""}

AI Reasoning: {refined_plan.reasoning}
Key Factors: {', '.join(refined_plan.key_factors) if refined_plan.key_factors else 'None'}
{f"Warnings: {', '.join(refined_plan.warnings)}" if refined_plan.warnings else ""}

Historical Context:
- Similar trades analyzed: {historical_context.get('total_trades', 0)}
- Historical win rate: {historical_context.get('win_rate', 0):.1%}
================================

IMPORTANT: Use the above systematic plan as your primary reference. You may adjust based on qualitative factors from the analyst reports, but explain any deviations from the systematic plan.
"""
            except ImportError as e:
                hybrid_plan_context = f"\n[Hybrid SMC plan unavailable: {e}]\n"
            except Exception as e:
                hybrid_plan_context = f"\n[Hybrid SMC plan generation failed: {e}]\n"

        # Track memory IDs for feedback loop
        memory_ids_used = []

        past_memory_str = "No past memories found."
        if memory is not None:
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            if past_memories:
                past_memory_str = ""
                for i, rec in enumerate(past_memories, 1):
                    past_memory_str += rec["recommendation"] + "\n\n"
                # Track which memories were used
                memory_ids_used = extract_memory_ids_from_results(past_memories)

        # Query SMC pattern memory for setup-specific insights
        smc_pattern_insights = ""
        if smc_memory is not None:
            try:
                # Get best performing setups for this symbol
                best_setups = smc_memory.get_best_setups(symbol=company_name, min_samples=3, min_win_rate=0.5)
                if best_setups:
                    smc_pattern_insights = "\n\nSMC PATTERN INSIGHTS FROM PAST TRADES:\n"
                    for setup in best_setups[:3]:  # Top 3 setups
                        smc_pattern_insights += f"- {setup['recommendation']}\n"

                # Get similar pattern outcomes
                similar_patterns = smc_memory.get_similar_patterns(
                    curr_situation, symbol=company_name, n_matches=2
                )
                if similar_patterns:
                    smc_pattern_insights += "\nSIMILAR PAST TRADES:\n"
                    for p in similar_patterns:
                        outcome = "WON" if p["was_win"] else "LOST"
                        smc_pattern_insights += f"- {p['setup_type']} trade {outcome} ({p['returns_pct']:+.2f}%). "
                        if p.get("lesson"):
                            smc_pattern_insights += f"Lesson: {p['lesson'][:100]}\n"
                        else:
                            smc_pattern_insights += "\n"
            except Exception as e:
                pass  # Silently ignore SMC memory errors

        # Query TradeSimilaritySearch for RAG-based historical context
        historical_trade_context = ""
        try:
            similarity_search = TradeSimilaritySearch()

            # Build current setup from available state
            current_setup = {
                "symbol": company_name,
                "market_regime": state.get("market_regime"),
                "volatility_regime": state.get("volatility_regime"),
            }

            # Query similar historical trades
            similar_trades_result = similarity_search.find_similar_trades(
                current_setup,
                n_results=5,
                min_confidence=0.3
            )

            # Format for prompt if we have results
            if similar_trades_result.get("similar_trades"):
                historical_trade_context = similarity_search.format_for_prompt(
                    similar_trades_result["similar_trades"],
                    similar_trades_result["statistics"],
                    max_trades=3
                )
        except Exception as e:
            pass  # Silently ignore trade similarity errors

        # Build broker price context - CRITICAL
        price_context = ""
        if current_price:
            price_context = f"""
CRITICAL - BROKER PRICE:
The broker quotes {company_name} at {current_price:.5f}. This is the ACTUAL trading price.
NEWS SOURCES MAY REPORT DIFFERENT PRICES (e.g., spot vs CFD). Always use {current_price:.5f} as your reference when suggesting:
- Entry price (should be near {current_price:.5f})
- Stop loss (must be a realistic distance from {current_price:.5f})
- Take profit (must be a realistic distance from {current_price:.5f})
Do NOT use prices from news articles - they may use different quote conventions.

"""

        context = {
            "role": "user",
            "content": f"{price_context}Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        # Build system prompt with SMC context if available
        system_prompt = f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}"""

        # Add SMC pattern insights if available
        if smc_pattern_insights:
            system_prompt += smc_pattern_insights
        
        # Add hybrid SMC plan if available (preferred over raw SMC context)
        if hybrid_plan_context:
            system_prompt += hybrid_plan_context
        elif smc_context:
            # Fallback to raw SMC context if hybrid plan not available
            system_prompt += f"\n\n{smc_context}\n\nIMPORTANT: When suggesting entry, stop loss, and take profit levels, align them with the Smart Money Concepts zones shown above. Place stops BELOW support zones (for buys) or ABOVE resistance zones (for sells) to avoid premature stop-outs. Target take profits AT resistance zones (for buys) or support zones (for sells) where institutional orders are likely to be filled."

        # Add historical trade context from RAG search
        if historical_trade_context:
            system_prompt += f"\n\n{historical_trade_context}\n\nIMPORTANT: Consider this historical performance when making your decision. If the historical win rate is low (<50%), provide strong justification for why THIS setup is different. Adjust your confidence based on the historical data."

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            context,
        ]

        result = llm.invoke(messages)

        # Build return dict with memory tracking
        return_dict = {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

        # Include refined plan if available (for downstream structured processing)
        if refined_plan:
            return_dict["smc_trade_plan"] = refined_plan.to_dict()

        # Include memory tracking if memories were used
        if memory_ids_used:
            return_dict.update(record_memory_usage(state, "trader", memory_ids_used))

        return return_dict

    return functools.partial(trader_node, name="Trader")
