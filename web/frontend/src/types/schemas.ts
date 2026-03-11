// Auto-generated TypeScript types from TradingAgents Pydantic schemas
// Do not edit manually - regenerate with:
//   python -m tradingagents.schemas.typescript_generator > web/frontend/src/types/schemas.ts

export type SignalType = "BUY" | "SELL" | "HOLD";

export type RiskLevel = "Low" | "Medium" | "High" | "Extreme";

export type Recommendation = "HOLD" | "CLOSE" | "ADJUST";

export type Priority = "high" | "medium" | "low";

export type MarketRegime = "trending-up" | "trending-down" | "ranging" | "expansion";

export type VolatilityRegime = "low" | "normal" | "high" | "extreme";

export type Urgency = "immediate" | "high" | "normal" | "low";

/** A price level with optional rationale. */
export interface PriceLevel {
  /** Price level */
  price: number;
  /** Why this level was chosen */
  rationale?: string | null;
}

/** A key factor influencing a trading decision. */
export interface KeyFactor {
  /** The factor description */
  factor: string;
  /** Factor weight 0-1 */
  weight?: number | null;
  /** Bias direction this factor suggests */
  direction?: SignalType | null;
}

/** Smart Money Concept zone information. */
export interface SMCZone {
  /** Zone type: FVG, OB, BOS, CHOCH, liquidity */
  zone_type: string;
  /** Bullish or bearish zone */
  direction: SignalType;
  /** Zone start price */
  price_start: number;
  /** Zone end price */
  price_end: number;
  /** Zone strength 0-100% */
  strength?: number | null;
  /** Whether zone has been mitigated */
  mitigated?: boolean;
  /** Timeframe this zone was identified on */
  timeframe?: string | null;
}

/** Smart Money Concepts analysis summary. */
export interface SMCAnalysis {
  /** Overall SMC bias */
  bias: SignalType;
  /** Key SMC zones */
  key_zones?: SMCZone[];
  /** Current market structure description */
  structure?: string | null;
  /** Key liquidity levels */
  liquidity_levels?: number[];
}

/** Complete trade analysis result from the multi-agent system.

This schema captures the output of a full analysis run including
signal, confidence, entry/SL/TP levels, and supporting rationale. */
export interface TradeAnalysisResult {
  /** Trading symbol analyzed */
  symbol: string;
  /** Trading signal: BUY, SELL, or HOLD */
  signal: SignalType;
  /** Signal confidence 0-1 */
  confidence: number;
  /** Recommended entry price */
  entry_price?: number | null;
  /** Recommended stop loss */
  stop_loss?: number | null;
  /** Recommended take profit */
  take_profit?: number | null;
  /** Second take profit target */
  take_profit_2?: number | null;
  /** Third take profit target */
  take_profit_3?: number | null;
  /** Recommended position size in lots */
  recommended_volume?: number | null;
  /** Risk as percentage of account */
  risk_percent?: number | null;
  /** Overall risk assessment */
  risk_level: RiskLevel;
  /** Risk-to-reward ratio */
  risk_reward_ratio?: number | null;
  /** Current market regime */
  market_regime?: MarketRegime | null;
  /** Current volatility regime */
  volatility_regime?: VolatilityRegime | null;
  /** Average True Range value */
  atr?: number | null;
  /** Smart Money Concepts analysis */
  smc_analysis?: SMCAnalysis | null;
  /** Summary rationale for the recommendation */
  rationale: string;
  /** Key factors influencing the decision */
  key_factors?: KeyFactor[];
  /** Market analyst summary */
  market_summary?: string | null;
  /** News analyst summary */
  news_summary?: string | null;
  /** Sentiment analyst summary */
  sentiment_summary?: string | null;
  /** Date of analysis (YYYY-MM-DD) */
  analysis_date: string;
  /** Primary timeframe analyzed */
  timeframe?: string;
  /** Time taken for analysis */
  analysis_duration_seconds?: number | null;
}

/** Simplified trade analysis schema for quick LLM analysis.

Use this for rapid signal extraction without full analysis context. */
export interface QuickTradeAnalysis {
  /** Trading signal: BUY, SELL, or HOLD */
  signal: SignalType;
  /** Signal confidence 0-1 */
  confidence: number;
  /** Recommended entry price */
  entry_price?: number | null;
  /** Recommended stop loss */
  stop_loss?: number | null;
  /** Recommended take profit */
  take_profit?: number | null;
  /** Brief explanation of the signal */
  rationale: string;
}

/** Recommended stop loss adjustment. */
export interface StopLossAdjustment {
  /** Current stop loss */
  current_sl?: number | null;
  /** Suggested new stop loss */
  suggested_sl: number;
  /** Type: breakeven, trailing, or manual */
  adjustment_type: string;
  /** Why this adjustment is recommended */
  rationale: string;
}

/** Recommended take profit adjustment. */
export interface TakeProfitAdjustment {
  /** Current take profit */
  current_tp?: number | null;
  /** Suggested new take profit */
  suggested_tp: number;
  /** Percentage to close at this level */
  partial_close_pct?: number | null;
  /** Why this adjustment is recommended */
  rationale: string;
}

/** Position review result with HOLD/CLOSE/ADJUST recommendation.

This schema is used for LLM-based position reviews that analyze
open positions and provide management recommendations. */
export interface PositionReview {
  /** Trading symbol */
  symbol: string;
  /** MT5 ticket number */
  ticket?: number | null;
  /** Position direction: BUY or SELL */
  direction: SignalType;
  /** HOLD, CLOSE, or ADJUST */
  recommendation: Recommendation;
  /** Urgency: immediate, high, normal, low */
  urgency?: Urgency;
  /** Position entry price */
  entry_price: number;
  /** Current market price */
  current_price: number;
  /** Current P&L as percentage */
  current_pnl_percent: number;
  /** Current P&L in account currency */
  current_pnl_amount?: number | null;
  /** Current risk assessment */
  risk_level: RiskLevel;
  /** Distance to SL as percentage */
  distance_to_sl_pct?: number | null;
  /** Distance to TP as percentage */
  distance_to_tp_pct?: number | null;
  /** Suggested stop loss, null if no change */
  suggested_sl?: number | null;
  /** Suggested take profit, null if no change */
  suggested_tp?: number | null;
  /** Detailed SL adjustment */
  sl_adjustment?: StopLossAdjustment | null;
  /** Detailed TP adjustment */
  tp_adjustment?: TakeProfitAdjustment | null;
  /** Reason for closing if CLOSE recommended */
  close_reason?: string | null;
  /** Detailed reasoning for the recommendation */
  reasoning: string;
  /** Key market observations */
  key_observations?: string[];
  /** SMC-based analysis context */
  smc_context?: string | null;
}

/** Simplified position review schema for quick LLM analysis.

Compatible with existing TradeReviewSchema in llm_client.py. */
export interface QuickPositionReview {
  /** HOLD, CLOSE, or ADJUST */
  recommendation: Recommendation;
  /** Suggested stop loss, null if no change */
  suggested_sl?: number | null;
  /** Suggested take profit, null if no change */
  suggested_tp?: number | null;
  /** Current risk assessment */
  risk_level: RiskLevel;
  /** Brief explanation of the recommendation */
  reasoning: string;
}

/** Result of reviewing multiple positions in batch. */
export interface BatchPositionReview {
  /** Individual position reviews */
  reviews: QuickPositionReview[];
  /** Overall portfolio assessment */
  portfolio_summary?: string | null;
  /** Total number of positions reviewed */
  total_positions: number;
  /** Number of positions needing adjustment */
  positions_to_adjust?: number;
  /** Number of positions recommended to close */
  positions_to_close?: number;
}

/** Correlation between two symbols. */
export interface CorrelationPair {
  /** First symbol */
  symbol_a: string;
  /** Second symbol */
  symbol_b: string;
  /** Correlation coefficient -1 to 1 */
  correlation: number;
  /** Risk implication of this correlation */
  risk_note?: string | null;
}

/** A single symbol suggestion for portfolio diversification. */
export interface SymbolSuggestion {
  /** Symbol to consider adding */
  symbol: string;
  /** Why this symbol is suggested */
  reason: string;
  /** Correlation group: precious_metals, currencies, indices, etc. */
  correlation_group: string;
  /** Suggestion priority */
  priority: Priority;
  /** Expected trade direction */
  expected_direction?: SignalType | null;
  /** Correlation with current portfolio */
  correlation_with_portfolio?: number | null;
}

/** Portfolio diversification suggestions with correlation analysis.

This schema provides suggestions for portfolio diversification
based on current holdings and correlation analysis. */
export interface PortfolioSuggestion {
  /** Suggested symbols to add */
  suggestions: SymbolSuggestion[];
  /** Analysis of current portfolio balance */
  portfolio_analysis: string;
  /** Assessment of concentration risk */
  concentration_risk?: string | null;
  /** Highly correlated pairs in portfolio */
  high_correlations?: CorrelationPair[];
  /** Portfolio diversification score 0-100 */
  diversification_score?: number | null;
  /** Overall risk considerations */
  risk_notes: string;
  /** Specific recommended actions */
  recommended_actions?: string[];
}

/** Simplified portfolio suggestion schema.

Compatible with existing PortfolioSuggestionSchema in llm_client.py. */
export interface QuickPortfolioSuggestion {
  /** Suggested symbols */
  suggestions: SymbolSuggestion[];
  /** Analysis of current portfolio balance */
  portfolio_analysis: string;
  /** Risk considerations */
  risk_notes: string;
}

/** Overall portfolio risk assessment. */
export interface PortfolioRiskAssessment {
  /** Total portfolio exposure in lots */
  total_exposure: number;
  /** Margin used as percentage */
  margin_used_percent?: number | null;
  /** Maximum drawdown as percentage */
  max_drawdown_percent?: number | null;
  /** Assessment of correlation risk: low, medium, high */
  correlation_risk: string;
  /** Assessment of concentration risk: low, medium, high */
  concentration_risk: string;
  /** Overall portfolio risk level: low, medium, high, extreme */
  overall_risk_level: string;
  /** Risk mitigation recommendations */
  recommendations?: string[];
}

