"""
LLM-Enhanced SMC Trading Plan Analysis

Adds contextual intelligence to rule-based SMC trading plans using LLM reasoning.
Provides:
- Context-aware confidence adjustments
- Historical pattern recognition
- Market regime adaptation
- Psychological insights
- Risk assessment beyond numerical heuristics
"""

import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI


def calculate_volatility_regime(smc_analysis: Dict, atr: Optional[float]) -> Dict[str, Any]:
    """
    Calculate current volatility regime.

    Returns:
    {
        'level': 'low', 'normal', or 'high',
        'atr_value': float,
        'atr_change_pct': float,  # vs 20-day average
        'description': str
    }
    """
    if not atr:
        return {
            'level': 'unknown',
            'atr_value': None,
            'atr_change_pct': None,
            'description': 'ATR data not available'
        }

    # Get 1H data for volatility context
    tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
    if not tf_data:
        return {
            'level': 'unknown',
            'atr_value': atr,
            'atr_change_pct': None,
            'description': f'ATR: {atr:.2f}'
        }

    current_price = tf_data.get('current_price', 100)

    # Calculate ATR as percentage of price
    atr_pct = (atr / current_price) * 100

    # Classify volatility (rough heuristics for commodities)
    if atr_pct < 1.0:
        level = 'low'
        description = f'Low volatility (ATR: {atr_pct:.2f}% of price)'
    elif atr_pct < 2.5:
        level = 'normal'
        description = f'Normal volatility (ATR: {atr_pct:.2f}% of price)'
    else:
        level = 'high'
        description = f'High volatility (ATR: {atr_pct:.2f}% of price)'

    return {
        'level': level,
        'atr_value': atr,
        'atr_pct': atr_pct,
        'atr_change_pct': None,  # Would need historical ATR data
        'description': description
    }


def determine_trend(smc_analysis: Dict) -> Dict[str, Any]:
    """
    Determine current trend direction from market structure.

    Returns:
    {
        'direction': 'bullish', 'bearish', or 'ranging',
        'strength': 'weak', 'moderate', or 'strong',
        'description': str
    }
    """
    # Get primary timeframe data
    tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
    if not tf_data:
        return {
            'direction': 'unknown',
            'strength': 'unknown',
            'description': 'Insufficient data for trend analysis'
        }

    # Analyze market structure from SMC data
    bullish_obs = len(tf_data.get('order_blocks', {}).get('bullish', []))
    bearish_obs = len(tf_data.get('order_blocks', {}).get('bearish', []))

    # Simple heuristic: more order blocks = dominant direction
    if bullish_obs > bearish_obs * 1.5:
        direction = 'bullish'
        strength = 'strong' if bullish_obs > bearish_obs * 2 else 'moderate'
    elif bearish_obs > bullish_obs * 1.5:
        direction = 'bearish'
        strength = 'strong' if bearish_obs > bullish_obs * 2 else 'moderate'
    else:
        direction = 'ranging'
        strength = 'weak'

    description = f"{strength.capitalize()} {direction} trend (Bullish OBs: {bullish_obs}, Bearish OBs: {bearish_obs})"

    return {
        'direction': direction,
        'strength': strength,
        'description': description
    }


def analyze_market_structure(smc_analysis: Dict) -> Dict[str, Any]:
    """
    Analyze overall market structure from SMC zones.

    Returns:
    {
        'structure_type': str,
        'key_levels': List[float],
        'description': str
    }
    """
    tf_data = smc_analysis.get('1H') or smc_analysis.get('4H') or smc_analysis.get('D1')
    if not tf_data:
        return {
            'structure_type': 'unknown',
            'key_levels': [],
            'description': 'Insufficient data'
        }

    # Count zones
    bullish_obs = tf_data.get('order_blocks', {}).get('bullish', [])
    bearish_obs = tf_data.get('order_blocks', {}).get('bearish', [])

    total_obs = len(bullish_obs) + len(bearish_obs)

    if total_obs > 10:
        structure_type = 'complex'
        description = f'Complex structure with {total_obs} order blocks'
    elif total_obs > 5:
        structure_type = 'moderate'
        description = f'Moderate structure with {total_obs} order blocks'
    else:
        structure_type = 'simple'
        description = f'Simple structure with {total_obs} order blocks'

    return {
        'structure_type': structure_type,
        'total_order_blocks': total_obs,
        'description': description
    }


def format_smc_context_for_llm(
    order_block: Any,
    ob_assessment: Dict,
    market_context: Dict,
    smc_analysis: Dict
) -> str:
    """Format SMC context into LLM prompt."""

    # Order block type
    ob_type = "Resistance (Bearish)" if hasattr(order_block, 'type') and 'bearish' in str(order_block.type).lower() else "Support (Bullish)"

    # Format the context
    context = f"""# ORDER BLOCK DATA
Type: {ob_type}
Price Range: ${order_block.bottom:.2f} - ${order_block.top:.2f}
Retests: {ob_assessment['retests']}
Volume Profile: {ob_assessment['volume_profile']}
Multi-timeframe Confluence: {ob_assessment['confluence_score']} across {', '.join(ob_assessment['aligned_timeframes'])}

# RULE-BASED ASSESSMENT (Baseline)
Strength Score: {ob_assessment['strength_score']}/10 ({ob_assessment['strength_category']})
Hold Probability: {ob_assessment['hold_probability']:.0%}
Breakout Probability: {ob_assessment['breakout_probability']:.0%}
Assessment: {ob_assessment['assessment']}

# MARKET CONTEXT
Volatility: {market_context['volatility']['description']}
Trend: {market_context['trend']['description']}
Market Structure: {market_context['structure']['description']}
"""

    # Add news/sentiment if available
    if market_context.get('news_sentiment'):
        context += f"News Sentiment: {market_context['news_sentiment']}\n"
    if market_context.get('social_sentiment'):
        context += f"Social Sentiment: {market_context['social_sentiment']}\n"

    return context


def evaluate_order_block_with_llm(
    order_block: Any,
    ob_assessment: Dict,
    smc_analysis: Dict,
    market_context: Dict,
    llm: ChatOpenAI,
    similar_trades: Optional[List] = None
) -> Dict[str, Any]:
    """
    Use LLM to holistically evaluate order block strength with context.

    Args:
        order_block: Order block object
        ob_assessment: Rule-based assessment from assess_order_block_strength()
        smc_analysis: Full SMC analysis
        market_context: Market regime, volatility, sentiment
        llm: LangChain ChatOpenAI instance
        similar_trades: List of similar past trades from memory

    Returns:
        dict with LLM assessment including adjusted probabilities, reasoning, risks
    """

    # Format context for LLM
    smc_context = format_smc_context_for_llm(order_block, ob_assessment, market_context, smc_analysis)

    # Add historical context if available
    historical_context = ""
    if similar_trades and len(similar_trades) > 0:
        historical_context = f"\n# HISTORICAL REFERENCE\n"
        historical_context += f"Similar order blocks analyzed in past: {len(similar_trades)} trades\n"
        historical_context += "Past outcomes provide pattern recognition context.\n"

    # Build prompt
    prompt = f"""You are a professional Smart Money Concepts (SMC) trading analyst with deep experience in institutional order flow analysis.

{smc_context}
{historical_context}

# YOUR TASK
Evaluate this order block holistically considering:

1. **Contextual Adjustment**: Does current market context make this OB more/less reliable than the rule-based score suggests?
   - In high volatility, OBs need stronger confluence to be reliable
   - In ranging markets, OBs hold better than in strong trends
   - Overbought/oversold conditions affect reliability

2. **Volatility Impact**: How does current volatility affect this specific setup?
   - High volatility increases breakout probability
   - Low volatility increases hold probability

3. **Trend Alignment**: Does the trend support or oppose this OB holding?
   - Resistance OBs in uptrends break more often
   - Support OBs in downtrends break more often
   - OBs aligned with trend hold better

4. **Retest Analysis**: What do the retests tell us about institutional behavior?
   - Multiple retests can indicate accumulation or distribution
   - Decreasing volume on retests = accumulation complete
   - Increasing volume on retests = zone weakening

5. **Risk Assessment**: What could invalidate this analysis?
   - Volatility expansion risk
   - Trend strength risk
   - Structural break risk

# RESPONSE FORMAT
Respond ONLY with valid JSON (no markdown, no code blocks):
{{
  "adjusted_hold_probability": 0.0-1.0,
  "confidence_in_assessment": 0.0-1.0,
  "key_reasoning": "2-3 sentence explanation focusing on what differs from rule-based assessment",
  "contextual_factors": [
    "Key factor 1 that influenced your assessment",
    "Key factor 2"
  ],
  "adjustment_vs_rules": "STRONGER/WEAKER/SIMILAR because...",
  "top_risks": [
    "Specific risk 1",
    "Specific risk 2",
    "Specific risk 3"
  ],
  "recommended_action": "SHORT_NOW/LONG_NOW/WAIT_CONFIRMATION/REDUCE_SIZE",
  "confidence_level": "HIGH/MEDIUM/LOW"
}}

Be specific and actionable. Focus on factors that DIFFER from the rule-based assessment.
"""

    try:
        # Call LLM
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            # Find the actual JSON content
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith('{'):
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if line.strip().endswith('}') and in_json:
                    break
            response_text = '\n'.join(json_lines)

        # Parse JSON response
        llm_assessment = json.loads(response_text)

        # Validate and add metadata
        llm_assessment['rule_based_hold_prob'] = ob_assessment['hold_probability']
        llm_assessment['probability_adjustment'] = llm_assessment['adjusted_hold_probability'] - ob_assessment['hold_probability']
        llm_assessment['adjusted_breakout_probability'] = 1.0 - llm_assessment['adjusted_hold_probability']

        return llm_assessment

    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        return {
            'adjusted_hold_probability': ob_assessment['hold_probability'],
            'confidence_in_assessment': 0.5,
            'key_reasoning': f"LLM response parsing failed: {str(e)}. Using rule-based assessment.",
            'contextual_factors': ['LLM parsing error - using rule-based only'],
            'adjustment_vs_rules': 'SIMILAR (fallback)',
            'top_risks': ['LLM assessment unavailable'],
            'recommended_action': 'USE_RULE_BASED',
            'confidence_level': 'LOW',
            'rule_based_hold_prob': ob_assessment['hold_probability'],
            'probability_adjustment': 0.0,
            'adjusted_breakout_probability': ob_assessment['breakout_probability'],
            'error': str(e),
            'raw_response': response_text if 'response_text' in locals() else 'No response'
        }
    except Exception as e:
        # General error fallback
        return {
            'adjusted_hold_probability': ob_assessment['hold_probability'],
            'confidence_in_assessment': 0.5,
            'key_reasoning': f"LLM call failed: {str(e)}. Using rule-based assessment.",
            'contextual_factors': ['LLM error - using rule-based only'],
            'adjustment_vs_rules': 'SIMILAR (error fallback)',
            'top_risks': ['LLM assessment unavailable'],
            'recommended_action': 'USE_RULE_BASED',
            'confidence_level': 'LOW',
            'rule_based_hold_prob': ob_assessment['hold_probability'],
            'probability_adjustment': 0.0,
            'adjusted_breakout_probability': ob_assessment['breakout_probability'],
            'error': str(e)
        }


def enhance_plan_with_llm(
    plan: Dict[str, Any],
    smc_analysis: Dict,
    llm: ChatOpenAI,
    atr: Optional[float] = None,
    final_state: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Enhance a rule-based SMC trading plan with LLM reasoning.

    Args:
        plan: Rule-based plan from generate_smc_trading_plan()
        smc_analysis: Full multi-timeframe SMC analysis
        llm: LangChain ChatOpenAI instance
        atr: ATR value for volatility analysis
        final_state: Optional final state with news/sentiment

    Returns:
        Enhanced plan with LLM insights added to setups and recommendation
    """

    # Build market context
    market_context = {
        'volatility': calculate_volatility_regime(smc_analysis, atr),
        'trend': determine_trend(smc_analysis),
        'structure': analyze_market_structure(smc_analysis),
        'news_sentiment': final_state.get('news_report', '')[:200] if final_state else None,
        'social_sentiment': final_state.get('sentiment_report', '')[:200] if final_state else None
    }

    # Enhance PRIMARY setup if exists
    if plan.get('primary_setup') and plan['primary_setup'].get('ob_strength'):
        pos_analysis = plan['position_analysis']

        # Get the actual order block object (we need to reconstruct or pass it)
        # For now, create a mock object with the data we have
        class MockOrderBlock:
            def __init__(self, zone_data, assessment):
                if zone_data:
                    self.bottom = zone_data[0]
                    self.top = zone_data[1]
                else:
                    self.bottom = 0
                    self.top = 0
                self.retests = assessment.get('retests', 0)

        # Get order block from position analysis
        if plan['primary_setup']['direction'] == 'SELL' and pos_analysis.get('nearest_resistance'):
            ob_zone = pos_analysis['nearest_resistance']['price_range']
            ob_assessment = pos_analysis['nearest_resistance']['assessment']
        elif plan['primary_setup']['direction'] == 'BUY' and pos_analysis.get('nearest_support'):
            ob_zone = pos_analysis['nearest_support']['price_range']
            ob_assessment = pos_analysis['nearest_support']['assessment']
        else:
            ob_zone = plan['primary_setup'].get('entry_zone')
            ob_assessment = plan['primary_setup']['ob_strength']

        if ob_zone and ob_assessment:
            mock_ob = MockOrderBlock(ob_zone, ob_assessment)

            # Get LLM assessment
            llm_assessment = evaluate_order_block_with_llm(
                order_block=mock_ob,
                ob_assessment=ob_assessment,
                smc_analysis=smc_analysis,
                market_context=market_context,
                llm=llm,
                similar_trades=None  # Could integrate memory here
            )

            # Add to plan
            plan['primary_setup']['llm_enhancement'] = llm_assessment
            plan['primary_setup']['final_confidence'] = llm_assessment['confidence_level']

            # Update recommendation if LLM significantly disagrees
            if abs(llm_assessment['probability_adjustment']) > 0.15:  # 15% threshold
                plan['recommendation']['llm_adjustment'] = {
                    'original_action': plan['recommendation']['action'],
                    'llm_recommended_action': llm_assessment['recommended_action'],
                    'reason': llm_assessment['key_reasoning'],
                    'probability_adjustment': f"{llm_assessment['probability_adjustment']:+.0%}"
                }

    # Enhance ALTERNATIVE setup if exists
    if plan.get('alternative_setup') and plan['alternative_setup'].get('ob_strength'):
        # Similar process for alternative setup
        alt_ob_assessment = plan['alternative_setup']['ob_strength']
        alt_zone = plan['alternative_setup'].get('entry_zone')

        if alt_zone:
            mock_alt_ob = MockOrderBlock(alt_zone, alt_ob_assessment)

            llm_alt_assessment = evaluate_order_block_with_llm(
                order_block=mock_alt_ob,
                ob_assessment=alt_ob_assessment,
                smc_analysis=smc_analysis,
                market_context=market_context,
                llm=llm,
                similar_trades=None
            )

            plan['alternative_setup']['llm_enhancement'] = llm_alt_assessment
            plan['alternative_setup']['final_confidence'] = llm_alt_assessment['confidence_level']

    # Add market context to plan
    plan['market_context'] = market_context

    return plan
