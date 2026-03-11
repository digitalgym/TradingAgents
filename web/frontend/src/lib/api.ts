const API_BASE = '/api'

export interface ApiResponse<T> {
  data?: T
  error?: string
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      return { error: error.detail || `HTTP ${response.status}` }
    }

    const data = await response.json()
    return { data }
  } catch (error) {
    return { error: error instanceof Error ? error.message : 'Network error' }
  }
}

// Status & Dashboard
export const getStatus = () => fetchApi<any>('/status')
export const getDashboard = () => fetchApi<any>('/dashboard')

// Positions
export const getPositions = () => fetchApi<any>('/positions')
export const getOrders = () => fetchApi<any>('/orders')
export const modifyPosition = (ticket: number, newSl?: number, newTp?: number) =>
  fetchApi<any>('/positions/modify', {
    method: 'POST',
    body: JSON.stringify({ ticket, new_sl: newSl, new_tp: newTp }),
  })
export const closePosition = (ticket: number) =>
  fetchApi<any>('/positions/close', {
    method: 'POST',
    body: JSON.stringify({ ticket }),
  })

export const batchClosePositions = (tickets: number[]) =>
  fetchApi<any>('/positions/batch-close', {
    method: 'POST',
    body: JSON.stringify({ tickets }),
  })

export const reviewPosition = (ticket: number, reviewMode: 'llm' | 'atr' = 'llm') =>
  fetchApi<{ task_id: string; status: string; message: string }>(`/positions/review/${ticket}`, {
    method: 'POST',
    body: JSON.stringify({ ticket, review_mode: reviewMode }),
  })

export const getReviewStatus = (taskId: string) =>
  fetchApi<any>(`/analysis/status/${taskId}`)

export const batchReviewPositions = (tickets?: number[]) =>
  fetchApi<{ task_id: string; status: string; message: string }>('/positions/batch-review', {
    method: 'POST',
    body: JSON.stringify({ tickets }),
  })

export const getBatchReviewStatus = (taskId: string) =>
  fetchApi<any>(`/analysis/status/${taskId}`)

// Quick Position Actions
export const setPositionBreakeven = (ticket: number) =>
  fetchApi<{ success: boolean; message: string; new_sl: number }>(`/positions/breakeven/${ticket}`, {
    method: 'POST',
  })

export const setPositionTrailing = (ticket: number, atrMultiplier: number = 1.5) =>
  fetchApi<{ success: boolean; message: string; new_sl: number; atr: number; trail_distance: number; trailing_active: boolean }>(
    `/positions/trailing/${ticket}`,
    {
      method: 'POST',
      body: JSON.stringify({ atr_multiplier: atrMultiplier }),
    }
  )

export const disablePositionTrailing = (ticket: number) =>
  fetchApi<{ success: boolean; message: string }>(`/positions/trailing/${ticket}`, {
    method: 'DELETE',
  })

export const getActiveTrailingStops = () =>
  fetchApi<{ trailing_stops: Array<{ ticket: number; symbol: string; direction: string; trail_distance: number; best_price: number }> }>(
    '/positions/trailing'
  )

// Position Deep Analysis (Multi-Agent)
export const startPositionDeepAnalysis = (ticket: number, timeframe?: string, useSmc?: boolean) =>
  fetchApi<any>(`/positions/deep-analysis/${ticket}`, {
    method: 'POST',
    body: JSON.stringify({
      ticket,
      timeframe: timeframe || 'H1',
      use_smc: useSmc !== false,
    }),
  })

export const getPositionDeepAnalysisStatus = (taskId: string) =>
  fetchApi<any>(`/positions/deep-analysis/status/${taskId}`)

export const cancelOrder = (ticket: number) =>
  fetchApi<any>(`/orders/${ticket}`, {
    method: 'DELETE',
  })

export const batchCancelOrders = (tickets: number[]) =>
  fetchApi<any>('/orders/batch-cancel', {
    method: 'POST',
    body: JSON.stringify({ tickets }),
  })

// Decisions
export const getDecisions = (limit?: number, status?: string, symbol?: string) => {
  const params = new URLSearchParams()
  if (limit) params.set('limit', limit.toString())
  if (status) params.set('status', status)
  if (symbol) params.set('symbol', symbol)
  return fetchApi<any>(`/decisions?${params}`)
}
export const getDecision = (id: string) => fetchApi<any>(`/decisions/${id}`)
export const getDecisionStats = () => fetchApi<any>('/decisions/stats')
export const closeDecision = (id: string, exitPrice: number, outcome: string, notes?: string) =>
  fetchApi<any>(`/decisions/${id}/close`, {
    method: 'POST',
    body: JSON.stringify({ exit_price: exitPrice, outcome, notes }),
  })
export const getRetryInfo = (id: string) => fetchApi<any>(`/decisions/${id}/retry-info`)
export const markDecisionRetried = (id: string) =>
  fetchApi<any>(`/decisions/${id}/mark-retried`, { method: 'POST' })

// Analysis
export const runAnalysis = (symbol: string, timeframe?: string, useSmc?: boolean, forceFresh?: boolean) =>
  fetchApi<any>('/analysis/run', {
    method: 'POST',
    body: JSON.stringify({
      symbol,
      timeframe: timeframe || 'H1',
      use_smc: useSmc !== false,
      save_decision: true,
      force_fresh: forceFresh || false,
    }),
  })
export const getAnalysisStatus = (taskId: string) => fetchApi<any>(`/analysis/status/${taskId}`)

// Rule-based analysis (no LLM - works when credits exhausted or for fast/free analysis)
export interface RuleBasedAnalysisResult {
  status: string
  symbol: string
  timeframe: string
  current_price: number
  error?: string  // Present when status is "error"
  traceback?: string
  decision: {
    signal: "BUY" | "SELL" | "HOLD"
    confidence: number
    entry_price: number | null
    stop_loss: number | null
    take_profit: number | null
    rationale: string
    setup_type: string | null
    key_factors: string[]
    analysis_mode: "rule-based"
    zone_quality: number
    rr_ratio: number
    checklist: {
      passed: number
      total: number
      items: string[]
    } | null
  }
  smc_levels: any[]
  regime: {
    market_regime: string
    volatility: string
    adx: number | null
    atr: number | null
  }
  analysis_mode: "rule-based"
  llm_used: false
}

export const runRuleBasedAnalysis = (symbol: string, timeframe?: string) =>
  fetchApi<RuleBasedAnalysisResult>('/analysis/rule-based', {
    method: 'POST',
    body: JSON.stringify({
      symbol,
      timeframe: timeframe || 'H1',
    }),
  })

// Quant Analysis (SMC + Indicators with single LLM call - no news/sentiment)
export interface QuantAnalysisResult {
  status: string
  symbol: string
  timeframe: string
  current_price: number
  bid: number
  ask: number
  error?: string
  traceback?: string
  decision: {
    signal: "BUY" | "SELL" | "HOLD"
    confidence: number
    entry_price: number | null
    stop_loss: number | null
    take_profit: number | null
    rationale: string
    analysis_mode: "quant"
    leverage?: number
    risk_usd?: number
    risk_level?: string
    risk_reward_ratio?: number
    full_report?: string
  }
  smc_levels: any[]
  indicators: {
    rsi: number | null
    macd: number | null
    macd_signal: number | null
    macd_histogram: number | null
    ema20: number | null
    ema50: number | null
    atr: number | null
    adx: number | null
    bb_upper: number | null
    bb_middle: number | null
    bb_lower: number | null
  }
  regime: {
    market_regime: string
    volatility: string
    adx: number | null
    atr: number | null
  }
  analysis_mode: "quant"
  llm_used: true
}

export const runQuantAnalysis = (symbol: string, timeframe?: string) =>
  fetchApi<QuantAnalysisResult>('/analysis/quant', {
    method: 'POST',
    body: JSON.stringify({
      symbol,
      timeframe: timeframe || 'H1',
    }),
  })

// LLM Status Check
export interface LLMStatus {
  status: "available" | "no_credits" | "rate_limited" | "auth_error" | "timeout" | "network_error" | "not_configured" | "error" | "unknown"
  provider: string | null
  model: string | null
  message: string
  error_type?: string
  error_detail?: string
  response_time_ms: number
  recommendation: "ai" | "rule-based"
}

export const checkLLMStatus = () => fetchApi<LLMStatus>('/llm/status')

// Cached Analysis
export interface CachedAnalysis {
  cached: boolean
  symbol: string
  timeframe: string
  cached_at?: string
  age_hours?: number
  is_fresh?: boolean
  result?: {
    decision: any
    trading_plan: any
    smc_analysis: any
    agent_outputs: any
  }
}

export interface CachedAnalysisSummary {
  symbol: string
  timeframe: string
  cached_at: string
  age_hours: number | null
  signal: string | null
}

export const getCachedAnalysis = (symbol: string, timeframe: string = 'H1') =>
  fetchApi<CachedAnalysis>(`/analysis/cached/${symbol}?timeframe=${timeframe}`)

export const listCachedAnalyses = () =>
  fetchApi<{ cached_analyses: CachedAnalysisSummary[] }>('/analysis/cached')

export const clearCachedAnalysis = (symbol: string, timeframe: string = 'H1') =>
  fetchApi<any>(`/analysis/cached/${symbol}?timeframe=${timeframe}`, { method: 'DELETE' })

// Agent Output Cache (caches individual agent outputs for faster re-runs)
export interface AgentCacheInfo {
  cached: boolean
  age_hours?: number
  expired?: boolean
  ttl_hours?: number
}

export interface AgentCacheStatus {
  symbol: string
  agents: {
    [agentName: string]: AgentCacheInfo
  }
}

export const getAgentCacheStatus = (symbol: string) =>
  fetchApi<AgentCacheStatus>(`/analysis/agent-cache/${symbol}`)

export const clearAgentCache = (symbol: string, agent?: string) => {
  const params = agent ? `?agent=${agent}` : ''
  return fetchApi<any>(`/analysis/agent-cache/${symbol}${params}`, { method: 'DELETE' })
}

export const clearAllAgentCache = () =>
  fetchApi<any>('/analysis/agent-cache', { method: 'DELETE' })

// Risk
export const getRiskMetrics = () => fetchApi<any>('/risk/metrics')
export const getRiskGuardrails = () => fetchApi<any>('/risk/guardrails')
export const getCircuitBreaker = () => fetchApi<any>('/risk/circuit-breaker')
export const getBreachHistory = (limit?: number) => {
  const params = new URLSearchParams()
  if (limit) params.set('limit', limit.toString())
  return fetchApi<any>(`/risk/breach-history?${params}`)
}
export const resetCircuitBreaker = () => fetchApi<any>('/risk/circuit-breaker/reset', { method: 'POST' })
export const calculatePositionSize = (symbol: string, entryPrice: number, stopLoss: number, riskPercent?: number) => {
  const params = new URLSearchParams({
    symbol,
    entry_price: entryPrice.toString(),
    stop_loss: stopLoss.toString(),
    ...(riskPercent && { risk_percent: riskPercent.toString() }),
  })
  return fetchApi<any>(`/risk/position-size?${params}`, { method: 'POST' })
}

// Learning
export const getLearningStatus = () => fetchApi<any>('/learning/status')
export const getPatterns = (symbol?: string, limit?: number) => {
  const params = new URLSearchParams()
  if (symbol) params.set('symbol', symbol)
  if (limit) params.set('limit', limit.toString())
  return fetchApi<any>(`/learning/patterns?${params}`)
}
export const updatePatterns = () => fetchApi<any>('/learning/update-patterns', { method: 'POST' })
export const findSimilarTrades = (symbol: string, direction: string, conditions?: string[]) =>
  fetchApi<any>('/learning/similar-trades', {
    method: 'POST',
    body: JSON.stringify({ symbol, direction, conditions }),
  })

// Memory
export const getMemoryStats = () => fetchApi<any>('/memory/stats')
export const queryMemory = (collection: string, query: string, nResults?: number) =>
  fetchApi<any>('/memory/query', {
    method: 'POST',
    body: JSON.stringify({ collection, query, n_results: nResults || 5 }),
  })
export const triggerReflection = () =>
  fetchApi<{ task_id: string; status: string }>('/memory/reflect', { method: 'POST' })
export const getReflectionStatus = (taskId: string) =>
  fetchApi<any>(`/memory/reflect/status/${taskId}`)
export const getMemoryLessons = (collection?: string, tier?: string, limit?: number) => {
  const params = new URLSearchParams()
  if (collection) params.set('collection', collection)
  if (tier) params.set('tier', tier)
  if (limit) params.set('limit', limit.toString())
  return fetchApi<any>(`/memory/lessons?${params}`)
}
export const deleteMemory = (collection: string, memoryId: string) =>
  fetchApi<{ success: boolean; message: string }>(`/memory/${encodeURIComponent(collection)}/${encodeURIComponent(memoryId)}`, {
    method: 'DELETE',
  })

// Portfolio Automation
export const getPortfolioStatus = () => fetchApi<any>('/portfolio/status')
export const getPortfolioConfig = () => fetchApi<any>('/portfolio/config')
export const addPortfolioSymbol = (symbol: string) =>
  fetchApi<any>('/portfolio/config/add-symbol', {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  })
export const removePortfolioSymbol = (symbol: string) =>
  fetchApi<any>('/portfolio/config/remove-symbol', {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  })
export const updatePortfolioSymbol = (symbol: string, updates: Record<string, any>) =>
  fetchApi<any>('/portfolio/config/update-symbol', {
    method: 'POST',
    body: JSON.stringify({ symbol, ...updates }),
  })
export const getPortfolioSuggestions = () => fetchApi<any>('/portfolio/config/suggestions')

// Portfolio Config Update
export interface PortfolioConfigUpdateParams {
  // Trading Control
  execution_mode?: 'FULL_AUTO' | 'SEMI_AUTO' | 'PAPER'
  max_total_positions?: number
  max_daily_trades?: number
  total_risk_budget_pct?: number
  daily_loss_limit_pct?: number
  // Fine-tuning
  use_atr_stops?: boolean
  atr_stop_multiplier?: number
  atr_trailing_multiplier?: number
  risk_reward_ratio?: number
  // Schedule
  schedule?: {
    morning_analysis_hour?: number
    midday_review_hour?: number
    evening_reflect_hour?: number
    timezone?: string
  }
}

export const updatePortfolioConfig = (params: PortfolioConfigUpdateParams) =>
  fetchApi<{ success: boolean; updated_fields: string[]; message: string; error?: string }>(
    '/portfolio/config/update',
    {
      method: 'POST',
      body: JSON.stringify(params),
    }
  )

export const startPortfolioAutomation = () => fetchApi<any>('/portfolio/start', { method: 'POST' })
export const stopPortfolioAutomation = () => fetchApi<any>('/portfolio/stop', { method: 'POST' })
export const triggerDailyCycle = (cycleType: 'morning' | 'midday' | 'evening') =>
  fetchApi<any>(`/portfolio/trigger?cycle_type=${cycleType}`, { method: 'POST' })
export const diagnosePortfolioAutomation = () => fetchApi<{
  imports_ok: boolean
  config_ok: boolean
  scheduler_ok: boolean
  errors: string[]
  config_file?: string
  execution_mode?: string
  symbols?: string[]
  traceback?: string
}>('/portfolio/diagnose')

// Daily Cycle (Prediction Tracking)
export interface DailyCycleStatus {
  running: boolean
  pid: number | null
  last_run: string | null
  pending_predictions: number
  symbols: string[]
  run_at: number | null
  started_at: string | null
  // Persistent state fields
  enabled: boolean
  last_start: string | null
  last_stop: string | null
  stop_reason: string | null
}

export interface AutomationStatus {
  running: boolean
  pid?: number
  last_run?: string
  next_run?: string
  // Persistent state fields
  enabled: boolean
  last_start: string | null
  last_stop: string | null
  stop_reason: string | null
}

export interface DailyCycleStartRequest {
  symbols?: string[]
  use_market_watch?: boolean
  run_at?: number
  stagger_minutes?: number
}

export interface PendingPrediction {
  symbol: string
  signal: string
  expected_direction: string
  price_at_analysis: number
  analysis_timestamp: string
  evaluation_due: string
  filename: string
}

export const getDailyCycleStatus = () => fetchApi<DailyCycleStatus>('/daily-cycle/status')
export const saveSelectedSymbols = (symbols: string[]) =>
  fetchApi<{ success: boolean; symbols_saved: number }>('/daily-cycle/save-symbols', {
    method: 'POST',
    body: JSON.stringify({ symbols }),
  })
export const startDailyCycle = (request: DailyCycleStartRequest = {}) =>
  fetchApi<any>('/daily-cycle/start', {
    method: 'POST',
    body: JSON.stringify(request),
  })
export const stopDailyCycle = () => fetchApi<any>('/daily-cycle/stop', { method: 'POST' })
export const getPendingPredictions = (symbol?: string) => {
  const params = symbol ? `?symbol=${symbol}` : ''
  return fetchApi<{ predictions: PendingPrediction[] }>(`/daily-cycle/predictions${params}`)
}
export const getDailyCycleLogs = (lines?: number, file?: string) => {
  const params = new URLSearchParams()
  if (lines) params.set('lines', lines.toString())
  if (file) params.set('file', file)
  return fetchApi<{
    file?: string
    total_lines?: number
    returned_lines?: number
    logs: string[]
    available_files: string[]
    message?: string
    error?: string
  }>(`/daily-cycle/logs?${params}`)
}

// SMC Analysis
export const runSmcAnalysis = (
  symbol: string,
  timeframe?: string,
  options?: { fvgMinSize?: number; lookback?: number; debug?: boolean }
) => {
  const params = new URLSearchParams({ symbol, timeframe: timeframe || 'H1' })
  if (options?.fvgMinSize !== undefined) params.set('fvg_min_size', options.fvgMinSize.toString())
  if (options?.lookback !== undefined) params.set('lookback', options.lookback.toString())
  if (options?.debug) params.set('debug', 'true')
  return fetchApi<any>(`/smc/analysis?${params}`)
}

// Market Regime
export const getMarketRegime = (symbol: string, timeframe?: string) => {
  const params = new URLSearchParams({ timeframe: timeframe || 'H1' })
  return fetchApi<any>(`/regime/${symbol}?${params}`)
}

// Trade Execution
export interface MarketOrderParams {
  symbol: string
  direction: 'BUY' | 'SELL'
  volume: number
  stop_loss?: number
  take_profit?: number
  comment?: string
}

export interface LimitOrderParams {
  symbol: string
  direction: 'BUY' | 'SELL'
  volume: number
  entry_price: number
  stop_loss?: number
  take_profit?: number
  comment?: string
}

export interface PositionSizeParams {
  symbol: string
  entry_price: number
  stop_loss: number
  take_profit?: number  // For actual R:R calculation
  risk_amount?: number
  risk_percent?: number
}

export const placeMarketOrder = (params: MarketOrderParams) =>
  fetchApi<any>('/trade/market', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const placeLimitOrder = (params: LimitOrderParams) =>
  fetchApi<any>('/trade/limit', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const calculateTradeSize = (params: PositionSizeParams) =>
  fetchApi<any>('/trade/calculate-size', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const getSymbolInfo = (symbol: string) =>
  fetchApi<any>(`/trade/symbol-info/${symbol}`)

export const getMarketStatus = (symbol: string) =>
  fetchApi<{ symbol: string; open: boolean; reason: string; trade_mode: number; session: string }>(
    `/trade/market-status/${symbol}`
  )

export const getMarketStatusMulti = (symbols: string[]) =>
  fetchApi<{ symbols: Record<string, { open: boolean; reason: string }>; session: string }>(
    `/trade/market-status?symbols=${symbols.join(',')}`
  )

export const getSwingLevels = (symbol: string, direction: 'BUY' | 'SELL', timeframe: string = 'H1') =>
  fetchApi<any>(`/trade/swing-levels/${symbol}?direction=${direction}&timeframe=${timeframe}`)

export const getChartCandles = (symbol: string, timeframe: string = 'H1', bars: number = 100) =>
  fetchApi<any>(`/chart/candles/${symbol}?timeframe=${timeframe}&bars=${bars}`)

export const getMarketWatchSymbols = () =>
  fetchApi<any>('/trade/market-watch')

export const addToMarketWatch = (symbol: string) =>
  fetchApi<any>('/trade/market-watch/add', {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  })

export const removeFromMarketWatch = (symbol: string) =>
  fetchApi<any>('/trade/market-watch/remove', {
    method: 'POST',
    body: JSON.stringify({ symbol }),
  })

export const searchSymbols = (query: string) =>
  fetchApi<any>(`/trade/symbols/search?q=${encodeURIComponent(query)}`)

export interface SaveDecisionParams {
  symbol: string
  action: 'BUY' | 'SELL'
  entry_type: 'market' | 'limit'
  entry_price: number
  stop_loss?: number
  take_profit?: number
  volume: number
  mt5_ticket?: number
  rationale?: string
  risk_percent?: number
  analysis_context?: any
}

export const saveTradeDecision = (params: SaveDecisionParams) =>
  fetchApi<any>('/trade/save-decision', {
    method: 'POST',
    body: JSON.stringify(params),
  })

// ===== Quant Automation =====

export interface QuantAutomationConfig {
  instance_name: string
  pipeline: 'quant' | 'volume_profile' | 'smc_quant' | 'multi_agent'
  symbols: string[]
  timeframe: string
  analysis_interval_seconds: number
  position_check_interval_seconds: number
  auto_execute: boolean
  min_confidence: number
  max_positions_per_symbol: number
  max_total_positions: number
  enable_trailing_stop: boolean
  trailing_stop_atr_multiplier: number
  move_to_breakeven_pct: number
  max_risk_per_trade_pct: number
  default_lot_size: number
  daily_loss_limit_pct: number
  max_consecutive_losses: number
}

export interface QuantAutomationStatus {
  status: 'stopped' | 'running' | 'paused' | 'error'
  running: boolean
  error: string | null
  config: QuantAutomationConfig | null
  instance_name?: string
  positions?: {
    managed: number
    max_per_symbol: number
    max_total: number
  }
  last_analysis?: Record<string, string | null>
  last_position_check?: string | null
  recent_results?: Array<{
    timestamp: string
    symbol: string
    signal: string
    confidence: number
    executed: boolean
  }>
  guardrails?: {
    can_trade: boolean
    status_summary: string
  }
}

export interface QuantAnalysisResult {
  symbol: string
  pipeline: string
  signal: string
  confidence: number
  entry_price: number | null
  stop_loss: number | null
  take_profit: number | null
  rationale: string
  executed: boolean
  execution_ticket: number | null
  execution_error: string | null
  duration_seconds: number
}

export interface QuantAutomationHistory {
  instance_name: string
  analysis_results: Array<QuantAnalysisResult & { timestamp: string }>
  position_results: Array<{
    timestamp: string
    ticket: number
    symbol: string
    action: string
    old_sl: number | null
    new_sl: number | null
    old_tp: number | null
    new_tp: number | null
    close_reason: string | null
    pnl: number | null
  }>
}

export interface VpQuantAnalysisResult {
  status: string
  symbol: string
  signal: string
  confidence: number
  entry_price: number | null
  stop_loss: number | null
  take_profit: number | null
  justification: string
  invalidation: string
  risk_level: string | null
  risk_reward_ratio: number | null
  current_price: number
  volume_profile: {
    poc: number
    poc_volume_pct: number
    value_area_high: number
    value_area_low: number
    hvn_count: number
    lvn_count: number
  } | null
  analysis_mode: string
  prompt_sent?: string
}

export const getQuantAutomationStatus = (instanceName: string = 'quant') =>
  fetchApi<QuantAutomationStatus>(`/automation/quant/status?instance=${instanceName}`)

export const listAutomationInstances = () =>
  fetchApi<{ instances: Record<string, QuantAutomationStatus> }>('/automation/quant/instances')

export const startQuantAutomation = (config: Partial<QuantAutomationConfig>) =>
  fetchApi<{ status: string; instance_name: string; config: QuantAutomationConfig }>('/automation/quant/start', {
    method: 'POST',
    body: JSON.stringify(config),
  })

export const stopQuantAutomation = (instanceName: string = 'quant') =>
  fetchApi<{ status: string }>(`/automation/quant/stop?instance=${instanceName}`, {
    method: 'POST',
  })

export const pauseQuantAutomation = (instanceName: string = 'quant') =>
  fetchApi<{ status: string }>(`/automation/quant/pause?instance=${instanceName}`, {
    method: 'POST',
  })

export const resumeQuantAutomation = (instanceName: string = 'quant') =>
  fetchApi<{ status: string }>(`/automation/quant/resume?instance=${instanceName}`, {
    method: 'POST',
  })

export const updateQuantAutomationConfig = (updates: Partial<QuantAutomationConfig>, instanceName: string = 'quant') =>
  fetchApi<{ status: string; config: QuantAutomationConfig }>(`/automation/quant/config?instance=${instanceName}`, {
    method: 'POST',
    body: JSON.stringify(updates),
  })

export const deleteAutomationConfig = (instanceName: string) =>
  fetchApi<{ status: string }>(`/automation/quant/config/${instanceName}`, {
    method: 'DELETE',
  })

export const renameAutomationInstance = (oldName: string, newName: string) =>
  fetchApi<{ status: string; old_name: string; new_name: string }>(
    `/automation/quant/config/${oldName}/rename?new_name=${encodeURIComponent(newName)}`,
    { method: 'PUT' }
  )

export const testQuantAnalysis = (symbol: string, pipeline: string = 'quant') =>
  fetchApi<QuantAnalysisResult>(`/automation/quant/test-analysis/${symbol}?pipeline=${pipeline}`, {
    method: 'POST',
  })

export const getQuantAutomationHistory = (instanceName: string = 'quant') =>
  fetchApi<QuantAutomationHistory>(`/automation/quant/history?instance=${instanceName}`)

export const runVpQuantAnalysis = (symbol: string, timeframe: string = 'H1') =>
  fetchApi<VpQuantAnalysisResult>('/analysis/vp-quant', {
    method: 'POST',
    body: JSON.stringify({ symbol, timeframe }),
  })

export const runSmcQuantAnalysis = (symbol: string, timeframe: string = 'H1') =>
  fetchApi<QuantAnalysisResult>('/analysis/smc-quant', {
    method: 'POST',
    body: JSON.stringify({ symbol, timeframe }),
  })

// Performance & Evaluation
export interface PerformanceStats {
  total_closed: number
  active: number
  wins: number
  losses: number
  win_rate: number
  total_pnl: number
  avg_pnl: number
  avg_win: number
  avg_loss: number
  best_trade: { decision_id: string; symbol: string; action: string; pnl: number; pnl_percent: number; exit_reason: string; exit_date: string } | null
  worst_trade: { decision_id: string; symbol: string; action: string; pnl: number; pnl_percent: number; exit_reason: string; exit_date: string } | null
  by_symbol: Record<string, { trades: number; wins: number; pnl: number; win_rate: number }>
  by_exit_reason: Record<string, { count: number; pnl: number }>
  equity_curve: Array<{ date: string; pnl: number; symbol: string; trade_pnl: number }>
  streaks: { current_streak: number; max_win_streak: number; max_loss_streak: number }
  quality: {
    sl_placement: Record<string, number>
    tp_placement: Record<string, number>
  }
}

export const getPerformanceStats = (symbol?: string, days?: number) => {
  const params = new URLSearchParams()
  if (symbol) params.set('symbol', symbol)
  if (days) params.set('days', days.toString())
  const qs = params.toString()
  return fetchApi<PerformanceStats>(`/decisions/performance${qs ? `?${qs}` : ''}`)
}

export const reconcileDecisions = () =>
  fetchApi<{ reconciled_count: number; reconciled: Array<{ decision_id: string; symbol: string; pnl: number; exit_reason: string }> }>(
    '/decisions/reconcile', { method: 'POST' }
  )

// Health
export const healthCheck = () => fetchApi<any>('/health')
