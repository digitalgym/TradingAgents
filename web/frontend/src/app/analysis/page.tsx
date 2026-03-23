"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { runAnalysis, getAnalysisStatus, getMarketRegime, getCachedAnalysis, CachedAnalysis, getAgentCacheStatus, AgentCacheStatus, runSmcAnalysis, getChartCandles, getMarketWatchSymbols, runRuleBasedAnalysis, runQuantAnalysis, runSmcQuantAnalysis, runVpQuantAnalysis, checkLLMStatus, LLMStatus } from "@/lib/api"
import { PriceChart } from "@/components/PositionChart"
import { SmcAnalysisPanel } from "@/components/SmcAnalysisPanel"
import { TradeExecutionWizard } from "@/components/TradeExecutionWizard"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Play,
  Loader2,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  BarChart,
  ChevronDown,
  User,
  Brain,
  Newspaper,
  MessageSquare,
  Scale,
  Shield,
  Target,
  Clock,
  HelpCircle,
  Zap,
  Database,
  RefreshCw,
  Calculator,  // For rule-based mode
} from "lucide-react"
import ReactMarkdown from "react-markdown"

// Metric explanations for tooltips
const METRIC_EXPLANATIONS = {
  regime: "Market regime identifies the current trend direction and strength. Bullish trending means prices are consistently moving higher, bearish trending means lower, and ranging means prices are moving sideways.",
  volatility: "Volatility measures how much price is fluctuating. High volatility means larger price swings and higher risk. Low volatility means calmer markets. Normal is average historical volatility.",
  adx: "Average Directional Index (ADX) measures trend strength from 0-100. Below 20 = weak/no trend, 20-40 = developing trend, 40-60 = strong trend, above 60 = very strong trend.",
  atr_percentile: "ATR Percentile shows current volatility relative to historical levels. Above 80% means unusually high volatility, below 20% means unusually calm. 50% is average.",
}

// Fallback symbols if Market Watch fetch fails
const FALLBACK_SYMBOLS = [
  "XAUUSD",
  "XAGUSD",
  "XPTUSD",
  "COPPER-C",
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "AUDUSD",
  "USDCAD",
]

const TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]

// Agent definitions with icons and descriptions
const AGENTS = [
  { id: "market_analyst", name: "Market Analyst", icon: BarChart, description: "Technical analysis & price action" },
  { id: "social_analyst", name: "Social Sentiment", icon: MessageSquare, description: "Social media sentiment" },
  { id: "news_analyst", name: "News Analyst", icon: Newspaper, description: "Market news analysis" },
  { id: "bull_researcher", name: "Bull Researcher", icon: TrendingUp, description: "Bullish case arguments" },
  { id: "bear_researcher", name: "Bear Researcher", icon: TrendingUp, description: "Bearish case arguments", iconClass: "rotate-180" },
  { id: "research_manager", name: "Research Manager", icon: Brain, description: "Debate synthesis & judgment" },
  { id: "trader", name: "Trader Agent", icon: User, description: "Trade plan formulation" },
  { id: "risky_analyst", name: "Aggressive Risk", icon: Target, description: "High-risk scenarios" },
  { id: "safe_analyst", name: "Conservative Risk", icon: Shield, description: "Low-risk scenarios" },
  { id: "neutral_analyst", name: "Balanced Risk", icon: Scale, description: "Balanced approach" },
  { id: "risk_manager", name: "Risk Manager", icon: Shield, description: "Final risk assessment" },
]

interface AgentStatus {
  id: string
  status: "pending" | "running" | "completed"
  output?: string
}

// Helper to persist/retrieve running task
const TASK_STORAGE_KEY = "analysis_running_task"

interface StoredTask {
  taskId: string
  symbol: string
  timeframe: string
  startedAt: number
}

function saveRunningTask(task: StoredTask) {
  localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(task))
}

function getRunningTask(): StoredTask | null {
  try {
    const stored = localStorage.getItem(TASK_STORAGE_KEY)
    if (!stored) return null
    const task = JSON.parse(stored) as StoredTask
    // Expire after 10 minutes
    if (Date.now() - task.startedAt > 10 * 60 * 1000) {
      localStorage.removeItem(TASK_STORAGE_KEY)
      return null
    }
    return task
  } catch {
    return null
  }
}

function clearRunningTask() {
  localStorage.removeItem(TASK_STORAGE_KEY)
}

export default function AnalysisPage() {
  const [symbol, setSymbol] = useState("XAUUSD")
  const [timeframe, setTimeframe] = useState("H1")
  const [useSmc, setUseSmc] = useState(true)
  const [useRuleBased, setUseRuleBased] = useState(false)  // Rule-based mode (no LLM)
  const [useQuantAnalysis, setUseQuantAnalysis] = useState(false)  // Quant mode (SMC + indicators, single LLM)
  const [analysisMode, setAnalysisMode] = useState<string>("multi_agent")  // Pipeline selector
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState<string>("")
  const [currentStepTitle, setCurrentStepTitle] = useState<string>("")
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [regime, setRegime] = useState<any>(null)
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([])
  const [agentOutputs, setAgentOutputs] = useState<Record<string, { title: string; output: string }>>({})
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())
  const [tradingPlan, setTradingPlan] = useState<{
    trader_plan: string;
    risk_decision: string;
    investment_plan: string;
  } | null>(null)
  const [showTradeWizard, setShowTradeWizard] = useState(false)
  const [cachedAnalysis, setCachedAnalysis] = useState<CachedAnalysis | null>(null)
  const [loadingCache, setLoadingCache] = useState(false)
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [forceFresh, setForceFresh] = useState(false)
  const [agentCacheStatus, setAgentCacheStatus] = useState<AgentCacheStatus | null>(null)
  const [watchlistSymbols, setWatchlistSymbols] = useState<string[]>(FALLBACK_SYMBOLS)
  const [loadingWatchlist, setLoadingWatchlist] = useState(true)
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null)
  const [checkingLlm, setCheckingLlm] = useState(false)
  const [minConfidence, setMinConfidence] = useState(50)  // Minimum confidence threshold (0-100)

  // Check LLM status on mount
  useEffect(() => {
    const checkLlm = async () => {
      setCheckingLlm(true)
      const { data } = await checkLLMStatus()
      if (data) {
        setLlmStatus(data)
        // Auto-enable rule-based if LLM is unavailable
        if (data.recommendation === "rule-based" && data.status !== "available") {
          setUseRuleBased(true)
          setAnalysisMode("rule_based")
        }
      }
      setCheckingLlm(false)
    }
    checkLlm()
  }, [])

  // Fetch watchlist symbols from Market Watch on mount
  useEffect(() => {
    const fetchWatchlist = async () => {
      setLoadingWatchlist(true)
      const { data, error } = await getMarketWatchSymbols()
      if (data?.symbols && data.symbols.length > 0) {
        // API returns objects with {symbol, description, bid, ...}, extract symbol names
        const symbolNames = data.symbols.map((s: any) => typeof s === 'string' ? s : s.symbol)
        setWatchlistSymbols(symbolNames)
        // If current symbol isn't in watchlist, select the first one
        if (!symbolNames.includes(symbol)) {
          setSymbol(symbolNames[0])
        }
      }
      // If error or empty, keep using FALLBACK_SYMBOLS
      setLoadingWatchlist(false)
    }
    fetchWatchlist()
  }, [])

  // Chart/SMC data state
  const [smcData, setSmcData] = useState<{
    orderBlocks: any[]
    fairValueGaps: any[]
    liquidityZones: any[]
    atrValue?: number
    pdh?: number  // Previous Day High
    pdl?: number  // Previous Day Low
    // NEW: Extended SMC features
    equalLevels?: any[]
    breakerBlocks?: any[]
    oteZones?: any[]
    premiumDiscount?: any
    confluenceScore?: any
    // Structure breaks for chart labels
    structure?: any
    // NEW: Advanced SMC patterns
    liquidity_sweeps?: any[]
    inducements?: any[]
    rejection_blocks?: any[]
    turtle_soup?: any[]
    alerts?: any[]
  } | null>(null)
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)
  const [chartLoading, setChartLoading] = useState(false)

  // Check for cached analysis and agent cache status when symbol/timeframe changes
  useEffect(() => {
    const checkCache = async () => {
      setLoadingCache(true)
      // Fetch both full analysis cache and agent-level cache in parallel
      const [analysisResult, agentCacheResult] = await Promise.all([
        getCachedAnalysis(symbol, timeframe),
        getAgentCacheStatus(symbol),
      ])
      if (analysisResult.data) {
        setCachedAnalysis(analysisResult.data)
      } else {
        setCachedAnalysis(null)
      }
      if (agentCacheResult.data) {
        setAgentCacheStatus(agentCacheResult.data)
      } else {
        setAgentCacheStatus(null)
      }
      setLoadingCache(false)
    }
    checkCache()
  }, [symbol, timeframe])

  // Fetch SMC data and current price for the chart when symbol/timeframe changes
  useEffect(() => {
    const fetchChartData = async () => {
      setChartLoading(true)
      try {
        // Fetch SMC analysis and candles in parallel
        const [smcResult, candlesResult] = await Promise.all([
          runSmcAnalysis(symbol, timeframe, { lookback: 100 }),
          getChartCandles(symbol, timeframe, 50),
        ])

        if (smcResult.data) {
          setSmcData({
            orderBlocks: smcResult.data.order_blocks || [],
            fairValueGaps: smcResult.data.fair_value_gaps || [],
            liquidityZones: smcResult.data.liquidity_zones || [],
            atrValue: smcResult.data.atr_value,
            pdh: smcResult.data.pdh,  // Previous Day High
            pdl: smcResult.data.pdl,  // Previous Day Low
            // NEW: Extended SMC features
            equalLevels: smcResult.data.equal_levels || [],
            breakerBlocks: smcResult.data.breaker_blocks || [],
            oteZones: smcResult.data.ote_zones || [],
            premiumDiscount: smcResult.data.premium_discount,
            confluenceScore: smcResult.data.confluence_score,
            // Structure breaks for chart labels
            structure: smcResult.data.structure,
            // NEW: Advanced SMC patterns
            liquidity_sweeps: smcResult.data.liquidity_sweeps || [],
            inducements: smcResult.data.inducements || [],
            rejection_blocks: smcResult.data.rejection_blocks || [],
            turtle_soup: smcResult.data.turtle_soup || [],
            alerts: smcResult.data.alerts || [],
          })
        }

        // Get current price from the last candle
        if (candlesResult.data?.candles?.length > 0) {
          const lastCandle = candlesResult.data.candles[candlesResult.data.candles.length - 1]
          setCurrentPrice(lastCandle.close)
        }
      } catch (err) {
        console.error("Failed to fetch chart data:", err)
      }
      setChartLoading(false)
    }
    fetchChartData()
  }, [symbol, timeframe])

  // Poll for task status - extracted to be reusable for resume
  const pollTaskStatus = async (taskId: string, isResume: boolean = false) => {
    if (!isResume) {
      // Initialize UI for new analysis
      setAgentStatuses(AGENTS.map(a => ({
        id: a.id,
        status: a.id === "market_analyst" ? "running" : "pending"
      })))
      setCurrentStep("market_analyst")
      setCurrentStepTitle("Market Analyst")
    }

    let completed = false
    let attempts = 0
    const maxAttempts = 180 // 6 minutes max

    while (!completed && attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 2000))
      const statusRes = await getAnalysisStatus(taskId)
      const statusData = statusRes.data

      if (statusData?.status === "completed") {
        completed = true
        clearRunningTask()
        setActiveTaskId(null)
        setResult(statusData.decision)
        setProgress(100)
        setCurrentStep("complete")
        setCurrentStepTitle("Analysis Complete")
        setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "completed" })))
        if (statusData.agent_outputs) {
          setAgentOutputs(statusData.agent_outputs)
          const agentsWithOutput = Object.keys(statusData.agent_outputs)
          setExpandedAgents(new Set(agentsWithOutput.slice(0, 3)))
        }
        if (statusData.trading_plan) {
          setTradingPlan(statusData.trading_plan)
        }
      } else if (statusData?.status === "error") {
        clearRunningTask()
        setActiveTaskId(null)
        setError(statusData.error || "Analysis failed")
        completed = true
      } else if (statusData?.status === "running") {
        const serverProgress = statusData.progress || 10
        setProgress(serverProgress)

        if (statusData.current_step_title) {
          setCurrentStepTitle(statusData.current_step_title)
        }

        const completedAgents = statusData.steps_completed || []
        const inProgressAgents = statusData.in_progress_agents || []

        setAgentStatuses(AGENTS.map(a => {
          if (completedAgents.includes(a.id)) {
            return { id: a.id, status: "completed" as const }
          } else if (inProgressAgents.includes(a.id)) {
            return { id: a.id, status: "running" as const }
          } else if (statusData.agent_outputs && statusData.agent_outputs[a.id]) {
            return { id: a.id, status: "completed" as const }
          }
          return { id: a.id, status: "pending" as const }
        }))

        if (statusData.agent_outputs) {
          setAgentOutputs(prev => {
            const newOutputs = { ...prev, ...statusData.agent_outputs }
            const newlyCompleted = Object.keys(statusData.agent_outputs).filter(
              id => !prev[id]
            )
            if (newlyCompleted.length > 0) {
              setExpandedAgents(expandedPrev => {
                const newSet = new Set(expandedPrev)
                newlyCompleted.forEach(id => newSet.add(id))
                return newSet
              })
            }
            return newOutputs
          })
        }
      } else if (!statusData) {
        // Task not found - it may have been cleaned up or server restarted
        clearRunningTask()
        setActiveTaskId(null)
        if (isResume) {
          setError("Previous analysis task not found. Please start a new analysis.")
        }
        completed = true
      }
      attempts++
    }

    if (!completed) {
      clearRunningTask()
      setActiveTaskId(null)
      setError("Analysis timed out - the multi-agent system may be overloaded")
    }

    setRunning(false)
  }

  // Check for running task on page load and resume polling
  useEffect(() => {
    const storedTask = getRunningTask()
    if (storedTask) {
      // Resume polling for the stored task
      setSymbol(storedTask.symbol)
      setTimeframe(storedTask.timeframe)
      setRunning(true)
      setActiveTaskId(storedTask.taskId)
      setProgress(10)
      setCurrentStepTitle("Resuming analysis...")
      setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "pending" })))
      pollTaskStatus(storedTask.taskId, true)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Resume polling when tab becomes visible again
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible" && activeTaskId && running) {
        // Tab became visible and we have an active task - check status immediately
        getAnalysisStatus(activeTaskId).then(res => {
          if (res.data?.status === "completed") {
            // Task finished while tab was hidden - update UI
            setResult(res.data.decision)
            setProgress(100)
            setCurrentStep("complete")
            setCurrentStepTitle("Analysis Complete")
            setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "completed" })))
            if (res.data.agent_outputs) {
              setAgentOutputs(res.data.agent_outputs)
            }
            if (res.data.trading_plan) {
              setTradingPlan(res.data.trading_plan)
            }
            clearRunningTask()
            setActiveTaskId(null)
            setRunning(false)
          } else if (res.data?.status === "error") {
            setError(res.data.error || "Analysis failed")
            clearRunningTask()
            setActiveTaskId(null)
            setRunning(false)
          }
          // If still running, the poll loop will continue
        })
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange)
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange)
  }, [activeTaskId, running])

  // Load cached analysis result
  const handleUseCachedAnalysis = () => {
    if (!cachedAnalysis?.result) return
    const cached = cachedAnalysis.result
    setResult(cached.decision)
    setTradingPlan(cached.trading_plan)
    setAgentOutputs(cached.agent_outputs || {})
    setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "completed" })))
    // Auto-expand agents that have output
    const agentsWithOutput = Object.keys(cached.agent_outputs || {})
    setExpandedAgents(new Set(agentsWithOutput.slice(0, 3)))
  }

  const toggleAgentExpanded = (agentId: string) => {
    setExpandedAgents(prev => {
      const newSet = new Set(prev)
      if (newSet.has(agentId)) {
        newSet.delete(agentId)
      } else {
        newSet.add(agentId)
      }
      return newSet
    })
  }

  const handleRunAnalysis = async () => {
    setRunning(true)
    setProgress(5)
    setResult(null)
    setError(null)
    setCurrentStep("initializing")
    setCurrentStepTitle("Initializing...")
    setAgentOutputs({})
    setExpandedAgents(new Set())
    setTradingPlan(null)

    // === RULE-BASED MODE (No LLM) ===
    if (useRuleBased) {
      setCurrentStepTitle("Running rule-based SMC analysis...")
      setProgress(30)

      const { data, error: apiError } = await runRuleBasedAnalysis(symbol, timeframe)

      if (apiError) {
        setError(apiError)
        setRunning(false)
        return
      }

      if (data?.status === "error") {
        setError(data.error || "Rule-based analysis failed")
        setRunning(false)
        return
      }

      setProgress(100)
      setCurrentStep("complete")
      setCurrentStepTitle("Analysis Complete (Rule-Based)")

      // Set regime from response
      if (data?.regime) {
        setRegime({
          regime: data.regime.market_regime,
          volatility: data.regime.volatility,
          adx: data.regime.adx,
          atr_percentile: data.regime.atr ? (data.regime.atr / (data.current_price || 1)) * 10000 : null
        })
      }

      // Set result in same format as LLM analysis
      setResult(data?.decision || null)
      setRunning(false)
      return
    }

    // === QUANT ANALYSIS MODE (SMC + Indicators, Single LLM) ===
    if (useQuantAnalysis) {
      setCurrentStepTitle("Running quant SMC analysis...")
      setProgress(30)

      const { data, error: apiError } = await runQuantAnalysis(symbol, timeframe)

      if (apiError) {
        setError(apiError)
        setRunning(false)
        return
      }

      if (data?.status === "error") {
        setError(data.error || "Quant analysis failed")
        setRunning(false)
        return
      }

      setProgress(100)
      setCurrentStep("complete")
      setCurrentStepTitle("Analysis Complete (Quant)")

      // Set regime from response
      if (data?.regime) {
        setRegime({
          regime: data.regime.market_regime,
          volatility: data.regime.volatility,
          adx: data.regime.adx,
          atr_percentile: data.regime.atr ? (data.regime.atr / (data.current_price || 1)) * 10000 : null
        })
      }

      // Set result in same format as LLM analysis
      setResult(data?.decision || null)
      setRunning(false)
      return
    }

    // === SMC QUANT MODE (Deep SMC-focused, Single LLM) ===
    if (analysisMode === "smc_quant") {
      setCurrentStepTitle("Running deep SMC quant analysis...")
      setProgress(30)

      const { data, error: apiError } = await runSmcQuantAnalysis(symbol, timeframe)

      if (apiError) {
        setError(apiError)
        setRunning(false)
        return
      }

      if (data?.status === "error") {
        setError(data.error || "SMC quant analysis failed")
        setRunning(false)
        return
      }

      setProgress(100)
      setCurrentStep("complete")
      setCurrentStepTitle("Analysis Complete (SMC Quant)")

      // Set regime from response
      if (data?.regime) {
        setRegime({
          regime: data.regime.market_regime,
          volatility: data.regime.volatility,
          adx: data.regime.adx,
          atr_percentile: data.regime.atr ? (data.regime.atr / (data.current_price || 1)) * 10000 : null
        })
      }

      // Set result in same format as LLM analysis
      setResult(data?.decision || null)
      setRunning(false)
      return
    }

    // === VOLUME PROFILE QUANT MODE ===
    if (analysisMode === "volume_profile") {
      setCurrentStepTitle("Running Volume Profile quant analysis...")
      setProgress(30)

      const { data, error: apiError } = await runVpQuantAnalysis(symbol, timeframe)

      if (apiError) {
        setError(apiError)
        setRunning(false)
        return
      }

      if (data?.status === "error") {
        setError("Volume Profile analysis failed")
        setRunning(false)
        return
      }

      setProgress(100)
      setCurrentStep("complete")
      setCurrentStepTitle("Analysis Complete (Volume Profile)")

      // Map VP result to standard decision format
      if (data) {
        setResult({
          signal: data.signal,
          confidence: data.confidence,
          entry_price: data.entry_price,
          stop_loss: data.stop_loss,
          take_profit: data.take_profit,
          rationale: data.justification,
          risk_level: data.risk_level,
          risk_reward_ratio: data.risk_reward_ratio,
          volume_profile: data.volume_profile,
        })
      }
      setRunning(false)
      return
    }

    // === LLM-BASED MODE (with automatic fallback) ===
    // Initialize agent statuses - all start as pending
    setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "pending" })))

    // Fetch market regime
    const regimeRes = await getMarketRegime(symbol, timeframe)
    if (regimeRes.data) setRegime(regimeRes.data)
    setProgress(10)

    // Start analysis
    const { data, error: apiError } = await runAnalysis(symbol, timeframe, useSmc, forceFresh)

    // Check for credit exhaustion error - auto-fallback to rule-based
    if (apiError) {
      const isCreditsError = apiError.toLowerCase().includes("credit") ||
                            apiError.toLowerCase().includes("quota") ||
                            apiError.toLowerCase().includes("rate limit") ||
                            apiError.toLowerCase().includes("insufficient")

      if (isCreditsError) {
        // Auto-fallback to rule-based analysis
        setCurrentStepTitle("LLM credits exhausted - falling back to rule-based...")
        setProgress(30)

        const fallbackResult = await runRuleBasedAnalysis(symbol, timeframe)
        if (fallbackResult.data && fallbackResult.data.status !== "error") {
          setProgress(100)
          setCurrentStep("complete")
          setCurrentStepTitle("Analysis Complete (Fallback: Rule-Based)")
          setResult({
            ...fallbackResult.data.decision,
            rationale: `**⚠️ LLM Unavailable - Using Rule-Based Fallback**\n\n${fallbackResult.data.decision?.rationale || ""}`
          })
          setRunning(false)
          return
        }
      }

      setError(apiError)
      setRunning(false)
      return
    }

    const taskId = data?.task_id
    if (!taskId) {
      setError("Failed to start analysis")
      setRunning(false)
      return
    }

    // Save task to localStorage so it can survive tab switches/refreshes
    saveRunningTask({
      taskId,
      symbol,
      timeframe,
      startedAt: Date.now()
    })
    setActiveTaskId(taskId)

    // Start polling (this will update UI and clear task when done)
    pollTaskStatus(taskId, false)
  }

  const getAgentStatus = (agentId: string): "pending" | "running" | "completed" => {
    const status = agentStatuses.find(a => a.id === agentId)
    return status?.status || "pending"
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Run Analysis</h1>
        <p className="text-muted-foreground">
          Execute multi-agent analysis on any symbol
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Analysis Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Settings</CardTitle>
            <CardDescription>Configure and run your analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* LLM Status Indicator */}
            <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50 border">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">LLM Status</span>
              </div>
              <div className="flex items-center gap-2">
                {checkingLlm ? (
                  <Badge variant="outline" className="text-xs">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    Checking...
                  </Badge>
                ) : llmStatus ? (
                  <>
                    <Badge
                      variant={
                        llmStatus.status === "available" ? "secondary" :
                        llmStatus.status === "no_credits" ? "destructive" :
                        "outline"
                      }
                      className={`text-xs ${
                        llmStatus.status === "available" ? "bg-green-500/20 text-green-500 border-green-500/50" : ""
                      }`}
                    >
                      {llmStatus.status === "available" ? "✓ Ready" :
                       llmStatus.status === "no_credits" ? "No Credits" :
                       llmStatus.status === "not_configured" ? "Not Configured" :
                       llmStatus.status === "rate_limited" ? "Rate Limited" :
                       "Unavailable"}
                    </Badge>
                    {llmStatus.provider && llmStatus.status === "available" && (
                      <span className="text-xs text-muted-foreground">{llmStatus.provider}</span>
                    )}
                  </>
                ) : (
                  <Badge variant="outline" className="text-xs">Unknown</Badge>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={async () => {
                    setCheckingLlm(true)
                    const { data } = await checkLLMStatus()
                    if (data) setLlmStatus(data)
                    setCheckingLlm(false)
                  }}
                  disabled={checkingLlm}
                >
                  <RefreshCw className={`h-3 w-3 ${checkingLlm ? "animate-spin" : ""}`} />
                </Button>
              </div>
            </div>

            {/* Warning when LLM unavailable */}
            {llmStatus && llmStatus.status !== "available" && !useRuleBased && (
              <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-600 dark:text-yellow-400 text-sm">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <span>{llmStatus.message}</span>
              </div>
            )}
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {watchlistSymbols.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Timeframe</Label>
              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIMEFRAMES.map((tf) => (
                    <SelectItem key={tf} value={tf}>
                      {tf}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Analysis Pipeline</Label>
              <Select
                value={analysisMode}
                onValueChange={(mode) => {
                  setAnalysisMode(mode)
                  setUseRuleBased(mode === "rule_based")
                  setUseQuantAnalysis(mode === "smc_quant_basic")
                  setUseSmc(mode !== "rule_based")
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="multi_agent">
                    <span className="flex items-center gap-2">
                      <Brain className="h-3.5 w-3.5" />
                      Multi-Agent AI
                    </span>
                  </SelectItem>
                  <SelectItem value="smc_quant_basic">
                    <span className="flex items-center gap-2">
                      <BarChart className="h-3.5 w-3.5" />
                      SMC Quant Basic
                    </span>
                  </SelectItem>
                  <SelectItem value="smc_quant">
                    <span className="flex items-center gap-2">
                      <BarChart className="h-3.5 w-3.5" />
                      SMC Quant
                    </span>
                  </SelectItem>
                  <SelectItem value="volume_profile">
                    <span className="flex items-center gap-2">
                      <BarChart className="h-3.5 w-3.5" />
                      Volume Profile Quant
                    </span>
                  </SelectItem>
                  <SelectItem value="rule_based">
                    <span className="flex items-center gap-2">
                      <Calculator className="h-3.5 w-3.5" />
                      Rule-Based (No LLM)
                    </span>
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {analysisMode === "multi_agent" && "Full 11-agent pipeline: market, news, sentiment, risk, debate"}
                {analysisMode === "smc_quant_basic" && "SMC + indicators, single LLM call — fast & focused"}
                {analysisMode === "smc_quant" && "Deep SMC quant: Order Blocks, FVGs, BOS/CHoCH, liquidity — institutional focus"}
                {analysisMode === "volume_profile" && "Volume profile zones (POC, VAH, VAL) + single LLM call"}
                {analysisMode === "rule_based" && "Pure SMC rules, no LLM — instant & free"}
              </p>
            </div>

            <div className="flex items-center justify-between">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="flex items-center gap-1 cursor-help">
                      Force Fresh
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </Label>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">Bypass cached agent outputs (social, news, fundamentals) and re-run all agents with fresh data. Use this if market conditions have changed significantly.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div className="flex items-center gap-2">
                {agentCacheStatus && Object.keys(agentCacheStatus.agents).length > 0 && !forceFresh && (
                  <Badge variant="outline" className="text-xs">
                    {Object.keys(agentCacheStatus.agents).length} cached
                  </Badge>
                )}
                <Switch checked={forceFresh} onCheckedChange={setForceFresh} />
              </div>
            </div>

            {/* Confidence Threshold Slider */}
            <div className="space-y-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="flex items-center gap-1 cursor-help">
                      <Shield className="h-3.5 w-3.5 text-muted-foreground" />
                      Min Confidence
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </Label>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm font-medium mb-1">Confidence Threshold Filter</p>
                    <p className="text-sm">Only consider trades with confidence above this level. Trades below threshold will show a warning and Execute button will be disabled.</p>
                    <ul className="text-sm mt-1 space-y-0.5">
                      <li>• 70%+ = High confidence setups</li>
                      <li>• 50-69% = Moderate confidence</li>
                      <li>• Below 50% = Low confidence, use caution</li>
                    </ul>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Slider
                value={minConfidence}
                onValueChange={setMinConfidence}
                min={0}
                max={100}
                step={5}
              />
            </div>

            <Button
              className="w-full"
              onClick={handleRunAnalysis}
              disabled={running}
            >
              {running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : analysisMode === "rule_based" ? (
                <>
                  <Calculator className="mr-2 h-4 w-4" />
                  Run Rule-Based Analysis
                </>
              ) : analysisMode === "smc_quant_basic" ? (
                <>
                  <BarChart className="mr-2 h-4 w-4" />
                  Run SMC Quant Basic
                </>
              ) : analysisMode === "volume_profile" ? (
                <>
                  <BarChart className="mr-2 h-4 w-4" />
                  Run Volume Profile Analysis
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Multi-Agent Analysis
                </>
              )}
            </Button>

            {/* Cached Analysis Section */}
            {!running && cachedAnalysis?.cached && cachedAnalysis.result && (
              <div className="space-y-3 p-3 rounded-lg bg-muted/50 border">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium">Cached Analysis Available</span>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <div className="flex items-center justify-between">
                    <span>Last run:</span>
                    <span>
                      {cachedAnalysis.cached_at
                        ? new Date(cachedAnalysis.cached_at).toLocaleString()
                        : "Unknown"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Age:</span>
                    <Badge variant={cachedAnalysis.is_fresh ? "secondary" : "outline"} className="text-xs">
                      {cachedAnalysis.age_hours !== undefined
                        ? cachedAnalysis.age_hours < 1
                          ? `${Math.round(cachedAnalysis.age_hours * 60)} min ago`
                          : `${cachedAnalysis.age_hours.toFixed(1)} hrs ago`
                        : "Unknown"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Signal:</span>
                    <Badge
                      variant={
                        cachedAnalysis.result.decision?.signal === "BUY"
                          ? "buy"
                          : cachedAnalysis.result.decision?.signal === "SELL"
                          ? "sell"
                          : "secondary"
                      }
                    >
                      {cachedAnalysis.result.decision?.signal || "N/A"}
                    </Badge>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={handleUseCachedAnalysis}
                >
                  <Database className="mr-2 h-3 w-3" />
                  Use Cached Result
                </Button>
              </div>
            )}

            {!running && loadingCache && (
              <div className="flex items-center justify-center py-2 text-xs text-muted-foreground">
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                Checking for cached analysis...
              </div>
            )}

            {running && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{currentStepTitle}</span>
                  <span className="font-medium">{progress}%</span>
                </div>
                <Progress value={progress} />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Market Regime */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart className="h-5 w-5" />
              Market Regime
            </CardTitle>
            <CardDescription>Current market conditions for {symbol}</CardDescription>
          </CardHeader>
          <CardContent>
            {regime ? (
              <TooltipProvider>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground flex items-center gap-1 cursor-help">
                          Regime
                          <HelpCircle className="h-3 w-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{METRIC_EXPLANATIONS.regime}</p>
                      </TooltipContent>
                    </Tooltip>
                    <Badge
                      variant={
                        regime.regime?.includes("bullish")
                          ? "buy"
                          : regime.regime?.includes("bearish")
                          ? "sell"
                          : "secondary"
                      }
                    >
                      {regime.regime || "Unknown"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground flex items-center gap-1 cursor-help">
                          Volatility
                          <HelpCircle className="h-3 w-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{METRIC_EXPLANATIONS.volatility}</p>
                      </TooltipContent>
                    </Tooltip>
                    <Badge
                      variant={
                        regime.volatility === "high"
                          ? "destructive"
                          : regime.volatility === "low"
                          ? "secondary"
                          : "outline"
                      }
                    >
                      {regime.volatility || "Normal"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground flex items-center gap-1 cursor-help">
                          ADX
                          <HelpCircle className="h-3 w-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{METRIC_EXPLANATIONS.adx}</p>
                      </TooltipContent>
                    </Tooltip>
                    <span className="font-medium">{regime.adx?.toFixed(2) || "N/A"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground flex items-center gap-1 cursor-help">
                          ATR Percentile
                          <HelpCircle className="h-3 w-3" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-sm">{METRIC_EXPLANATIONS.atr_percentile}</p>
                      </TooltipContent>
                    </Tooltip>
                    <span className="font-medium">
                      {regime.atr_percentile ? `${regime.atr_percentile.toFixed(1)}%` : "N/A"}
                    </span>
                  </div>
                </div>
              </TooltipProvider>
            ) : (
              <div className="flex h-32 items-center justify-center text-muted-foreground">
                Run analysis to see market regime
              </div>
            )}
          </CardContent>
        </Card>

        {/* Status */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Status</CardTitle>
          </CardHeader>
          <CardContent>
            {error ? (
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            ) : result ? (
              <div className="flex items-center gap-2 text-green-500">
                <CheckCircle2 className="h-5 w-5" />
                <span>Analysis complete</span>
              </div>
            ) : running ? (
              <div className="flex items-center gap-2 text-primary">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Running multi-agent analysis...</span>
              </div>
            ) : (
              <div className="text-muted-foreground">Ready to analyze</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Price Chart with SMC Analysis */}
      {currentPrice && (
        <div className="grid gap-6 lg:grid-cols-[1fr,400px]">
          <PriceChart
            symbol={symbol}
            direction={result?.signal === "SELL" ? "SELL" : result?.signal === "BUY" ? "BUY" : regime?.regime?.includes("bearish") ? "SELL" : "BUY"}
            entryPrice={result?.entry_price || 0}
            currentPrice={currentPrice}
            stopLoss={result?.stop_loss}
            takeProfit={result?.take_profit}
            orderBlocks={smcData?.orderBlocks}
            fairValueGaps={smcData?.fairValueGaps}
            liquidityZones={smcData?.liquidityZones}
            breakerBlocks={smcData?.breakerBlocks}
            equalLevels={smcData?.equalLevels}
            oteZones={smcData?.oteZones}
            structureBreaks={[
              ...(smcData?.structure?.all_bos || smcData?.structure?.recent_bos || []),
              ...(smcData?.structure?.all_choc || smcData?.structure?.recent_choc || []),
            ]}
            premiumDiscount={smcData?.premiumDiscount}
            pdh={smcData?.pdh}
            pdl={smcData?.pdl}
            atrValue={smcData?.atrValue}
            digits={symbol.includes("JPY") ? 3 : symbol.includes("XAU") || symbol.includes("XAG") ? 2 : 5}
            tradeDecision={result ? {
              signal: result.signal,
              confidence: result.confidence,
              entry_price: result.entry_price,
              stop_loss: result.stop_loss,
              take_profit: result.take_profit,
              rationale: result.rationale,
              setup_type: result.setup_type,
            } : null}
          />
          <SmcAnalysisPanel
            symbol={symbol}
            currentPrice={currentPrice}
            orderBlocks={smcData?.orderBlocks || []}
            fairValueGaps={smcData?.fairValueGaps || []}
            liquidityZones={smcData?.liquidityZones || []}
            digits={symbol.includes("JPY") ? 3 : symbol.includes("XAU") || symbol.includes("XAG") ? 2 : 5}
            // NEW: Extended SMC features
            equalLevels={smcData?.equalLevels}
            breakerBlocks={smcData?.breakerBlocks}
            oteZones={smcData?.oteZones}
            premiumDiscount={smcData?.premiumDiscount}
            confluenceScore={smcData?.confluenceScore}
            // NEW: Advanced SMC patterns
            liquiditySweeps={smcData?.liquidity_sweeps}
            inducements={smcData?.inducements}
            rejectionBlocks={smcData?.rejection_blocks}
            turtleSoup={smcData?.turtle_soup}
            alerts={smcData?.alerts}
          />
        </div>
      )}

      {chartLoading && !currentPrice && (
        <Card>
          <CardContent className="flex items-center justify-center h-[200px]">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading chart data...</span>
          </CardContent>
        </Card>
      )}

      {/* Analysis Result - Show above Agent Progress */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Analysis Result
              </CardTitle>
              {cachedAnalysis?.cached && cachedAnalysis?.cached_at && !running && (
                <Badge variant="outline" className="gap-1 text-xs">
                  <Database className="h-3 w-3" />
                  Cached: {new Date(cachedAnalysis.cached_at).toLocaleString()}
                </Badge>
              )}
            </div>
            <CardDescription>
              {symbol} - {timeframe}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <span className="text-muted-foreground">Signal:</span>
                  <Badge
                    variant={
                      result.signal === "BUY"
                        ? "buy"
                        : result.signal === "SELL"
                        ? "sell"
                        : "hold"
                    }
                    className="text-lg px-4 py-1"
                  >
                    {result.signal || "HOLD"}
                  </Badge>
                </div>

                {/* Confidence threshold warning */}
                {result.confidence !== undefined && (result.confidence * 100) < minConfidence && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-600 dark:text-yellow-400 text-sm mb-4">
                    <AlertCircle className="h-4 w-4 flex-shrink-0" />
                    <span>
                      Confidence ({(result.confidence * 100).toFixed(0)}%) is below your threshold ({minConfidence}%). Consider waiting for a higher confidence setup.
                    </span>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Confidence</p>
                    <p className={`text-xl font-bold ${
                      result.confidence !== undefined && (result.confidence * 100) < minConfidence
                        ? "text-yellow-500"
                        : (result.confidence * 100) >= 70
                          ? "text-green-500"
                          : ""
                    }`}>
                      {result.confidence ? `${(result.confidence * 100).toFixed(0)}%` : "N/A"}
                      {result.confidence !== undefined && (result.confidence * 100) < minConfidence && (
                        <span className="text-xs ml-1 text-yellow-500">(below threshold)</span>
                      )}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Entry Price</p>
                    <p className="text-xl font-bold">{result.entry_price || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Stop Loss</p>
                    <p className="text-xl font-bold text-red-500">
                      {result.stop_loss || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Take Profit</p>
                    <p className="text-xl font-bold text-green-500">
                      {result.take_profit || "N/A"}
                    </p>
                  </div>
                </div>

                {result.setup_type && (
                  <div>
                    <p className="text-sm text-muted-foreground">Setup Type</p>
                    <Badge variant="outline">{result.setup_type}</Badge>
                  </div>
                )}
              </div>

              <div>
                <p className="text-sm text-muted-foreground mb-2">Rationale</p>
                <ScrollArea className="h-48 rounded-md border p-4">
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>
                      {result.rationale || result.full_response || "No rationale provided"}
                    </ReactMarkdown>
                  </div>
                </ScrollArea>
              </div>
            </div>

            {result.key_factors && result.key_factors.length > 0 && (
              <div className="mt-6">
                <p className="text-sm text-muted-foreground mb-2">Key Factors</p>
                <div className="flex flex-wrap gap-2">
                  {result.key_factors.map((factor: string, i: number) => (
                    <Badge key={i} variant="secondary">
                      {factor}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Execute Trade Button */}
            {result.signal && result.signal !== "HOLD" && (
              <div className="mt-6 pt-4 border-t">
                {result.confidence !== undefined && (result.confidence * 100) < minConfidence ? (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Button
                            size="lg"
                            className="w-full"
                            variant="outline"
                            disabled
                          >
                            <Zap className="h-5 w-5 mr-2 opacity-50" />
                            Execute {result.signal} Trade
                          </Button>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Confidence ({(result.confidence * 100).toFixed(0)}%) is below your threshold ({minConfidence}%)</p>
                        <p className="text-xs text-muted-foreground mt-1">Adjust the slider to lower the threshold or wait for a better setup</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ) : (
                  <Button
                    size="lg"
                    className="w-full"
                    variant={result.signal === "BUY" ? "default" : "destructive"}
                    onClick={() => setShowTradeWizard(true)}
                  >
                    <Zap className="h-5 w-5 mr-2" />
                    Execute {result.signal} Trade
                  </Button>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Full Trading Plan */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Multi-Stage Trading Plan
            </CardTitle>
            <CardDescription>
              Full trading plan from the AI agents - review and execute trades as appropriate
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {tradingPlan && (tradingPlan.trader_plan || tradingPlan.risk_decision) ? (
              <>
                {/* Trader's Investment Plan */}
                {tradingPlan.trader_plan && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                      <User className="h-4 w-4" />
                      Trader Agent Plan
                    </div>
                    <ScrollArea className="h-64 rounded-md border p-4 bg-muted/30">
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown>{tradingPlan.trader_plan}</ReactMarkdown>
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {/* Risk Manager's Final Decision */}
                {tradingPlan.risk_decision && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                      <Shield className="h-4 w-4" />
                      Risk Manager Decision
                    </div>
                    <ScrollArea className="h-64 rounded-md border p-4 bg-muted/30">
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown>{tradingPlan.risk_decision}</ReactMarkdown>
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-2 pt-4 border-t">
                  <Button
                    variant="outline"
                    onClick={() => {
                      // Copy full plan to clipboard
                      const fullPlan = `TRADING PLAN FOR ${symbol}\n\n` +
                        `TRADER AGENT PLAN:\n${tradingPlan.trader_plan || "N/A"}\n\n` +
                        `RISK MANAGER DECISION:\n${tradingPlan.risk_decision || "N/A"}`
                      navigator.clipboard.writeText(fullPlan)
                    }}
                  >
                    Copy Full Plan
                  </Button>
                  <div className="flex-1" />
                  {result?.signal && result.signal !== "HOLD" && (
                    result.confidence !== undefined && (result.confidence * 100) < minConfidence ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div>
                              <Button variant="outline" disabled>
                                <Zap className="h-4 w-4 mr-2 opacity-50" />
                                Execute Trade
                              </Button>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Below confidence threshold ({minConfidence}%)</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <Button
                        variant={result.signal === "BUY" ? "default" : "destructive"}
                        onClick={() => setShowTradeWizard(true)}
                      >
                        <Zap className="h-4 w-4 mr-2" />
                        Execute Trade
                      </Button>
                    )
                  )}
                </div>
              </>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p className="mb-2">Trading plan data not available.</p>
                <p className="text-xs">The full response is shown in the Rationale section above.</p>
                <p className="text-xs mt-2">Check the Risk Manager output in Agent Progress below for full details.</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Agent Progress Timeline */}
      {(running || result) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Agent Progress
            </CardTitle>
            <CardDescription>
              Real-time status of each AI agent in the analysis pipeline
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {AGENTS.map((agent, index) => {
                const status = getAgentStatus(agent.id)
                const Icon = agent.icon
                const output = agentOutputs[agent.id]
                const isExpanded = expandedAgents.has(agent.id)
                // Check if this agent has cached output (only for cacheable agents)
                const cacheableAgents = ["social_analyst", "news_analyst", "fundamentals_analyst"]
                const agentCache = cacheableAgents.includes(agent.id) && agentCacheStatus?.agents?.[agent.id]
                const isCached = agentCache && agentCache.cached && !agentCache.expired

                return (
                  <Collapsible
                    key={agent.id}
                    open={isExpanded && !!output}
                    onOpenChange={() => output && toggleAgentExpanded(agent.id)}
                  >
                    <div
                      className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
                        status === "running"
                          ? "bg-amber-500/10 border border-amber-500/30"
                          : status === "completed"
                          ? "bg-green-500/10 border border-green-500/30"
                          : "bg-muted/30 border border-transparent"
                      }`}
                    >
                      {/* Status Icon */}
                      <div className={`flex-shrink-0 ${status === "running" ? "animate-pulse" : ""}`}>
                        {status === "completed" ? (
                          <CheckCircle2 className="h-5 w-5 text-green-500" />
                        ) : status === "running" ? (
                          <Loader2 className="h-5 w-5 text-amber-500 animate-spin" />
                        ) : (
                          <Clock className="h-5 w-5 text-muted-foreground" />
                        )}
                      </div>

                      {/* Agent Icon */}
                      <div className={`p-2 rounded-full ${
                        status === "completed" ? "bg-green-500/20" :
                        status === "running" ? "bg-amber-500/20" : "bg-muted"
                      }`}>
                        <Icon className={`h-4 w-4 ${agent.iconClass || ""} ${
                          status === "completed" ? "text-green-500" :
                          status === "running" ? "text-amber-500" : "text-muted-foreground"
                        }`} />
                      </div>

                      {/* Agent Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`font-medium ${
                            status === "completed" ? "text-green-500" :
                            status === "running" ? "text-amber-500" : "text-muted-foreground"
                          }`}>
                            {agent.name}
                          </span>
                          {status === "running" && (
                            <Badge variant="outline" className="text-xs border-amber-500/50 text-amber-500">
                              Processing
                            </Badge>
                          )}
                          {/* Cache indicator for cacheable agents */}
                          {isCached && status !== "running" && (
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Badge variant="secondary" className="text-xs gap-1">
                                    <Database className="h-3 w-3" />
                                    {agentCache.age_hours !== undefined
                                      ? agentCache.age_hours < 1
                                        ? `${Math.round(agentCache.age_hours * 60)}m`
                                        : `${agentCache.age_hours.toFixed(1)}h`
                                      : "cached"}
                                  </Badge>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="text-xs">Cached output will be used (TTL: {agentCache.ttl_hours}h)</p>
                                  <p className="text-xs text-muted-foreground">Enable &quot;Force Fresh&quot; to re-run</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {agent.description}
                        </p>
                      </div>

                      {/* Expand Button */}
                      {output && (
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                          </Button>
                        </CollapsibleTrigger>
                      )}
                    </div>

                    {/* Agent Output */}
                    <CollapsibleContent>
                      {output && (
                        <div className="ml-12 mt-2 p-3 bg-muted/50 rounded-lg border">
                          <p className="text-xs font-medium text-muted-foreground mb-2">Output:</p>
                          <ScrollArea className="max-h-[400px]">
                            <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap break-words">
                              <ReactMarkdown>{output.output}</ReactMarkdown>
                            </div>
                          </ScrollArea>
                        </div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Trade Execution Wizard */}
      <TradeExecutionWizard
        open={showTradeWizard}
        onOpenChange={setShowTradeWizard}
        symbol={symbol}
        signal={result?.signal || "HOLD"}
        suggestedEntry={result?.entry_price}
        suggestedStopLoss={result?.stop_loss}
        suggestedTakeProfit={result?.take_profit}
        rationale={result?.rationale || result?.full_response}
        smcLevels={result?.smc_levels || result?.analysis_context?.smc_levels}
        analysisContext={result?.analysis_context}  // Full state for reflection/learning
        confidence={result?.confidence}
      />
    </div>
  )
}
