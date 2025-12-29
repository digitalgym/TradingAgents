"""
Create a memory/reflection for the XAGUSD SELL trade.
Trade: SELL at $78, price dropped to $75 then recovered.
Result: +3.85% profit on short position.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from examples.trade_commodities import COMMODITY_CONFIG

# Create graph with memory enabled
config = COMMODITY_CONFIG.copy()
config['use_memory'] = True
# Use local embeddings (sentence-transformers)
config['embedding_provider'] = 'local'
print("Using local embeddings")

graph = TradingAgentsGraph(config=config, debug=False)

# Create a final_state representing the XAGUSD SELL recommendation
graph.curr_state = {
    'company_of_interest': 'XAGUSD',
    'trade_date': '2025-12-26',
    'curr_situation': '''XAGUSD Silver Analysis - Dec 26, 2025
    
Market Situation:
- Silver at $75-80 range after 15% WoW spike
- RSI overbought at 80-84
- 90-95% retail bullishness (euphoria trap)
- Thin holiday volumes inflating prices
- Support at $71-72 (Dec 26 low at $71.97)
''',
    'market_report': 'Silver 15% WoW spike to $78, RSI 80-84 overbought, thin holiday volumes, support at $71-72',
    'sentiment_report': '90-95% retail bullish - classic exhaustion/euphoria trap mirroring 2011 signals',
    'news_report': 'Holiday trading with low liquidity. Structural bull case (117M oz deficit, ETF inflows) but near-term exhaustion signals.',
    'fundamentals_report': 'N/A - Commodity',
    'investment_debate_state': {
        'bull_history': ['Silver structural deficit supports long-term bull case'],
        'bear_history': ['Overbought RSI, euphoria trap, thin volumes - correction imminent'],
        'history': ['Bear case wins on near-term technicals'],
        'judge_decision': 'SELL - de-risk amid correction risks'
    },
    'risk_debate_state': {
        'risky_history': ['Aggressive SELL/short to capitalize on euphoria trap, target $65-60'],
        'safe_history': ['Exit longs but avoid shorts due to squeeze risk'],
        'neutral_history': ['Partial SELL (exit 50% longs), buy dips at $71-72'],
        'history': ['Consensus: SELL with moderate position sizing'],
        'judge_decision': 'SELL recommendation with $71-72 support as re-entry'
    },
    'investment_plan': 'SELL XAGUSD at $78, target $71-72 support, stop loss at $82',
    'trader_investment_plan': '''SELL XAGUSD at $78
- Position: Short silver
- Entry: $78.00
- Target: $71-72 (support cluster)
- Stop Loss: $82.00
- Risk/Reward: 1:2
- Rationale: Overbought RSI 80-84, 90-95% retail bullish (euphoria trap), thin holiday volumes
''',
    'final_trade_decision': '''SELL Recommendation for XAGUSD

Rationale: Euphoria trap with 90-95% retail bullishness, overbought RSI 80-84, 
thin holiday volumes. Short-term correction risk outweighs continuation odds.
Entry: $78, Target: $71-72, Stop: $82
'''
}

# The trade outcome: SELL at $78, price dropped to $75 then recovered
# Short position profit: (78-75)/78 = 3.85%
returns_losses = 3.85

print('Trade: SELL XAGUSD (Silver)')
print('Entry: $78.00')
print('Exit: $75.00 (covered before recovery)')
print(f'Returns: +{returns_losses:.2f}% (short position profit)')
print()
print('Running reflection to store lessons...')

graph.reflect_and_remember(returns_losses)

print()
print('✅ Memory created! Lessons stored for future XAGUSD analysis:')
print('- Overbought RSI + retail euphoria = valid SELL signal')
print('- $71-72 support held as predicted')
print('- Short-term correction played out before recovery')
