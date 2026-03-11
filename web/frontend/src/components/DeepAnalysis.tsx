"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
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
  RefreshCw,
  X,
  Check,
  Edit2,
  AlertTriangle,
} from "lucide-react"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import ReactMarkdown from "react-markdown"
import {
  startPositionDeepAnalysis,
  getPositionDeepAnalysisStatus,
  runAnalysis,
  getAnalysisStatus,
} from "@/lib/api"

// Agent definitions
const AGENTS = [
  { id: "market_analyst", name: "Market Analyst", icon: BarChart, description: "Technical analysis & price action" },
  { id: "social_analyst", name: "Social Sentiment", icon: MessageSquare, description: "Social media sentiment" },
  { id: "news_analyst", name: "News Analyst", icon: Newspaper, description: "Market news analysis" },
  { id: "bull_researcher", name: "Bull Researcher", icon: TrendingUp, description: "Bullish case arguments" },
  { id: "bear_researcher", name: "Bear Researcher", icon: TrendingUp, description: "Bearish case arguments", iconClass: "rotate-180" },
  { id: "research_manager", name: "Research Manager", icon: Brain, description: "Debate synthesis & judgment" },
  { id: "trader", name: "Trader Agent", icon: User, description: "Trade plan formulation" },
  { id: "risk_manager", name: "Risk Manager", icon: Shield, description: "Final risk assessment" },
  { id: "position_manager", name: "Position Manager", icon: Target, description: "Position-specific management decision", positionModeOnly: true },
]

// Pre-process markdown to ensure proper formatting
function normalizeMarkdown(text: string): string {
  if (!text) return ""

  let normalized = text
    // Normalize line endings
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    // Ensure blank line before headings (# ## ### #### etc.)
    .replace(/([^\n])\n(#{1,6} )/g, "$1\n\n$2")
    // Ensure blank line before list items after non-list content
    .replace(/([^\n-*\d])\n([-*] )/g, "$1\n\n$2")
    .replace(/([^\n-*\d])\n(\d+\. )/g, "$1\n\n$2")

  return normalized
}

// Markdown renderer using react-markdown
function FormattedMarkdown({ content }: { content: string }) {
  if (!content) return null

  const normalizedContent = normalizeMarkdown(content)

  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        components={{
          h1: ({ children }) => <h2 className="text-lg font-bold mt-4 mb-2">{children}</h2>,
          h2: ({ children }) => <h3 className="text-base font-semibold mt-3 mb-1">{children}</h3>,
          h3: ({ children }) => <h4 className="text-sm font-semibold mt-3 mb-1">{children}</h4>,
          h4: ({ children }) => <h5 className="text-sm font-semibold mt-2 mb-1 text-muted-foreground">{children}</h5>,
          h5: ({ children }) => <h6 className="text-xs font-semibold mt-2 mb-1 text-muted-foreground">{children}</h6>,
          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
          em: ({ children }) => <em>{children}</em>,
          code: ({ children }) => <code className="bg-muted px-1 py-0.5 rounded text-xs">{children}</code>,
          pre: ({ children }) => <pre className="bg-muted p-2 rounded text-xs my-2 overflow-x-auto">{children}</pre>,
          ul: ({ children }) => <ul className="list-disc ml-4 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal ml-4 space-y-1">{children}</ol>,
          li: ({ children }) => <li>{children}</li>,
          p: ({ children }) => <p className="mb-2">{children}</p>,
        }}
      >
        {normalizedContent}
      </ReactMarkdown>
    </div>
  )
}

interface AgentStatus {
  id: string
  status: "pending" | "running" | "completed"
  output?: string
}

export interface PositionReviewResult {
  recommendation: "HOLD" | "ADJUST" | "CLOSE"
  suggested_sl?: number
  suggested_tp?: number
  suggested_trailing_sl?: number
  trailing_sl_source?: string
  close_reason?: string
  bias?: string
  bias_aligns?: boolean
  structure_shift?: boolean
  sl_at_risk?: boolean
  sl_risk_reason?: string
  // Position Manager fields
  urgency?: "immediate" | "high" | "normal" | "low"
  risk_assessment?: string
  key_factors?: string[]
  pm_reasoning?: string
}

interface DeepAnalysisProps {
  /** Mode: "new" for new trade analysis, "position" for reviewing existing position */
  mode: "new" | "position"
  /** Symbol to analyze (required for "new" mode) */
  symbol?: string
  /** Position ticket (required for "position" mode) */
  ticket?: number
  /** Timeframe for analysis */
  timeframe?: string
  /** Position context (for position mode) */
  positionContext?: {
    direction: string
    entry_price: number
    current_price: number
    current_sl: number
    current_tp: number
    volume: number
    profit: number
    pnl_pct: number
  }
  /** Callback when analysis completes with recommendations */
  onComplete?: (result: {
    recommendation?: string
    suggested_sl?: number
    suggested_tp?: number
    trading_plan?: any
    decision?: any
    position_review?: PositionReviewResult
  }) => void
  /** Callback to apply SL/TP changes */
  onApplyChanges?: (sl?: number, tp?: number) => void
  /** Callback to close position */
  onClosePosition?: () => void
}

export function DeepAnalysis({
  mode,
  symbol,
  ticket,
  timeframe = "H1",
  positionContext,
  onComplete,
  onApplyChanges,
  onClosePosition,
}: DeepAnalysisProps) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStepTitle, setCurrentStepTitle] = useState("")
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([])
  const [agentOutputs, setAgentOutputs] = useState<Record<string, { title: string; output: string }>>({})
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())
  const [tradingPlan, setTradingPlan] = useState<any>(null)
  const [positionReview, setPositionReview] = useState<PositionReviewResult | null>(null)

  // Edit mode for adjustments
  const [editMode, setEditMode] = useState(false)
  const [editSl, setEditSl] = useState("")
  const [editTp, setEditTp] = useState("")

  // Confirmation dialogs
  const [showCloseConfirm, setShowCloseConfirm] = useState(false)
  const [showApplyConfirm, setShowApplyConfirm] = useState(false)

  // Ref to track current task ID for cancellation
  const currentTaskIdRef = useRef<string | null>(null)

  // Cleanup on unmount - stop any running polls
  useEffect(() => {
    return () => {
      currentTaskIdRef.current = null
    }
  }, [])

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

  const getAgentStatus = (agentId: string): "pending" | "running" | "completed" => {
    const status = agentStatuses.find(a => a.id === agentId)
    return status?.status || "pending"
  }

  const handleRunAnalysis = async () => {
    setRunning(true)
    setProgress(5)
    setResult(null)
    setError(null)
    setCurrentStepTitle("Initializing...")
    setAgentOutputs({})
    setExpandedAgents(new Set())
    setTradingPlan(null)
    setPositionReview(null)
    setEditMode(false)

    setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "pending" })))

    let taskId: string | null = null

    try {
      if (mode === "position" && ticket) {
        // Position review mode
        const { data, error: apiError } = await startPositionDeepAnalysis(ticket, timeframe, true)
        if (apiError) {
          setError(apiError)
          setRunning(false)
          return
        }
        taskId = data?.task_id
      } else if (mode === "new" && symbol) {
        // New analysis mode
        const { data, error: apiError } = await runAnalysis(symbol, timeframe, true)
        if (apiError) {
          setError(apiError)
          setRunning(false)
          return
        }
        taskId = data?.task_id
      } else {
        setError("Invalid analysis configuration")
        setRunning(false)
        return
      }

      if (!taskId) {
        setError("Failed to start analysis")
        setRunning(false)
        return
      }

      // Track current task ID for cancellation
      currentTaskIdRef.current = taskId

      // Start first agent as running
      setAgentStatuses(AGENTS.map(a => ({
        id: a.id,
        status: a.id === "market_analyst" ? "running" : "pending"
      })))
      setCurrentStepTitle("Market Analyst")

      // Poll for completion
      let completed = false
      let attempts = 0
      const maxAttempts = 180 // 6 minutes max
      const thisTaskId = taskId // Capture for closure

      while (!completed && attempts < maxAttempts) {
        await new Promise((r) => setTimeout(r, 2000))

        // Check if a newer analysis has started - if so, stop polling this one
        if (currentTaskIdRef.current !== thisTaskId) {
          console.log(`Polling cancelled for ${thisTaskId} - newer task started`)
          return
        }

        const statusFn = mode === "position" ? getPositionDeepAnalysisStatus : getAnalysisStatus
        const statusRes = await statusFn(thisTaskId)

        // Handle 404 or other errors - stop polling gracefully
        if (statusRes.error) {
          if (statusRes.error.includes("404") || statusRes.error.includes("not found")) {
            console.log(`Task ${thisTaskId} not found (404) - stopping poll`)
            // Only show error if this is still the current task
            if (currentTaskIdRef.current === thisTaskId) {
              setError("Analysis task not found - it may have expired")
            }
            completed = true
            break
          }
          // For other errors, log but continue polling (might be transient)
          console.warn(`Polling error for ${thisTaskId}:`, statusRes.error)
          attempts++
          continue
        }

        const statusData = statusRes.data

        if (statusData?.status === "completed") {
          completed = true
          setResult(statusData.decision)
          setProgress(100)
          setCurrentStepTitle("Analysis Complete")
          setAgentStatuses(AGENTS.map(a => ({ id: a.id, status: "completed" })))

          if (statusData.agent_outputs) {
            setAgentOutputs(statusData.agent_outputs)
            const agentsWithOutput = Object.keys(statusData.agent_outputs)
            setExpandedAgents(new Set(agentsWithOutput.slice(0, 2)))
          }

          if (statusData.trading_plan) {
            setTradingPlan(statusData.trading_plan)
          }

          if (statusData.position_review) {
            setPositionReview(statusData.position_review)
            // Pre-fill edit values
            if (statusData.position_review.suggested_sl) {
              setEditSl(statusData.position_review.suggested_sl.toString())
            }
            if (statusData.position_review.suggested_tp) {
              setEditTp(statusData.position_review.suggested_tp.toString())
            }
          }

          // Call onComplete callback
          if (onComplete) {
            onComplete({
              recommendation: statusData.position_review?.recommendation,
              suggested_sl: statusData.position_review?.suggested_sl,
              suggested_tp: statusData.position_review?.suggested_tp,
              trading_plan: statusData.trading_plan,
              decision: statusData.decision,
              position_review: statusData.position_review,
            })
          }
        } else if (statusData?.status === "error") {
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
            } else {
              if (statusData.agent_outputs && statusData.agent_outputs[a.id]) {
                return { id: a.id, status: "completed" as const }
              }
              return { id: a.id, status: "pending" as const }
            }
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
        }
        attempts++
      }

      if (!completed) {
        setError("Analysis timed out - the multi-agent system may be overloaded")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
    }

    setRunning(false)
  }

  const handleApplyChanges = () => {
    const sl = editSl ? parseFloat(editSl) : undefined
    const tp = editTp ? parseFloat(editTp) : undefined
    if (onApplyChanges) {
      onApplyChanges(sl, tp)
    }
    setShowApplyConfirm(false)
    setEditMode(false)
  }

  const handleClosePosition = () => {
    if (onClosePosition) {
      onClosePosition()
    }
    setShowCloseConfirm(false)
  }

  return (
    <div className="space-y-4">
      {/* Analysis Controls */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Deep Analysis
                <HelpTooltip content="Multi-agent AI analysis that synthesizes market data, news, sentiment, and SMC structure to provide comprehensive trading recommendations." />
              </CardTitle>
              <CardDescription>
                {mode === "position"
                  ? "Full multi-agent review of your position"
                  : `Multi-agent analysis for ${symbol}`}
              </CardDescription>
            </div>
            <Button
              onClick={handleRunAnalysis}
              disabled={running}
              variant={result ? "outline" : "default"}
            >
              {running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : result ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Re-analyze
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Deep Analysis
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        {running && (
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{currentStepTitle}</span>
                <span className="font-medium">{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        )}
      </Card>

      {/* Error State */}
      {error && (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardContent className="py-4">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Position Review Result (for position mode) */}
      {mode === "position" && positionReview && (
        <Card className={
          positionReview.recommendation === "CLOSE"
            ? "border-red-500/50 bg-red-500/5"
            : positionReview.recommendation === "ADJUST"
              ? "border-yellow-500/50 bg-yellow-500/5"
              : "border-green-500/50 bg-green-500/5"
        }>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {positionReview.recommendation === "CLOSE" ? (
                <AlertTriangle className="h-5 w-5 text-red-500" />
              ) : positionReview.recommendation === "ADJUST" ? (
                <Edit2 className="h-5 w-5 text-yellow-500" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              )}
              Position Recommendation
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Recommendation Badge with Urgency */}
            <div className="flex items-center gap-4 flex-wrap">
              <Badge
                variant={
                  positionReview.recommendation === "CLOSE"
                    ? "destructive"
                    : positionReview.recommendation === "ADJUST"
                      ? "outline"
                      : "default"
                }
                className="text-lg px-4 py-1"
              >
                {positionReview.recommendation}
              </Badge>
              {positionReview.urgency && positionReview.urgency !== "normal" && (
                <Badge
                  variant={positionReview.urgency === "immediate" ? "destructive" : positionReview.urgency === "high" ? "outline" : "secondary"}
                  className="text-xs"
                >
                  {positionReview.urgency.toUpperCase()} URGENCY
                </Badge>
              )}
              {positionReview.risk_assessment && (
                <Badge
                  variant={
                    positionReview.risk_assessment.toLowerCase().includes("critical") ? "destructive" :
                    positionReview.risk_assessment.toLowerCase().includes("high") ? "outline" :
                    "secondary"
                  }
                  className="text-xs"
                >
                  Risk: {positionReview.risk_assessment}
                </Badge>
              )}
            </div>

            {/* Close reason or key factors */}
            {(positionReview.close_reason || (positionReview.key_factors && positionReview.key_factors.length > 0)) && (
              <div className="text-sm">
                {positionReview.close_reason && (
                  <p className="text-muted-foreground">{positionReview.close_reason}</p>
                )}
                {positionReview.key_factors && positionReview.key_factors.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-muted-foreground mb-1">Key Factors:</p>
                    <ul className="list-disc list-inside text-muted-foreground">
                      {positionReview.key_factors.slice(0, 3).map((factor, idx) => (
                        <li key={idx}>{factor}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Market Structure Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Market Bias</p>
                <Badge variant={positionReview.bias === "bullish" ? "buy" : positionReview.bias === "bearish" ? "sell" : "secondary"}>
                  {positionReview.bias?.toUpperCase() || "NEUTRAL"}
                </Badge>
              </div>
              <div>
                <p className="text-muted-foreground">Bias Alignment</p>
                <span className={positionReview.bias_aligns ? "text-green-500" : "text-red-500"}>
                  {positionReview.bias_aligns ? "Aligned" : "Against Position"}
                </span>
              </div>
              <div>
                <p className="text-muted-foreground">Structure Shift</p>
                <span className={positionReview.structure_shift ? "text-red-500" : "text-green-500"}>
                  {positionReview.structure_shift ? "CHOCH Detected" : "No Shift"}
                </span>
              </div>
              {positionReview.sl_at_risk && (
                <div>
                  <p className="text-muted-foreground">SL Risk</p>
                  <span className="text-yellow-500">{positionReview.sl_risk_reason || "At Risk"}</span>
                </div>
              )}
            </div>

            {/* Suggested Changes (for ADJUST recommendation) */}
            {positionReview.recommendation === "ADJUST" && (
              <div className="border rounded-lg p-4 space-y-4">
                <h4 className="font-semibold flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Suggested Adjustments
                </h4>

                {!editMode ? (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {positionReview.suggested_sl && (
                        <div>
                          <p className="text-xs text-muted-foreground">Suggested SL</p>
                          <p className="text-lg font-mono font-bold text-red-500">
                            {positionReview.suggested_sl.toFixed(5)}
                          </p>
                        </div>
                      )}
                      {positionReview.suggested_tp && (
                        <div>
                          <p className="text-xs text-muted-foreground">Suggested TP</p>
                          <p className="text-lg font-mono font-bold text-green-500">
                            {positionReview.suggested_tp.toFixed(5)}
                          </p>
                        </div>
                      )}
                      {positionReview.suggested_trailing_sl && (
                        <div>
                          <p className="text-xs text-muted-foreground flex items-center gap-1">
                            Trailing SL
                            <HelpTooltip content={`Source: ${positionReview.trailing_sl_source || 'SMC structure'}`} iconClassName="h-3 w-3" />
                          </p>
                          <p className="text-lg font-mono font-bold text-yellow-500">
                            {positionReview.suggested_trailing_sl.toFixed(5)}
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="flex gap-2">
                      <Button onClick={() => setShowApplyConfirm(true)} className="flex-1">
                        <Check className="mr-2 h-4 w-4" />
                        Accept & Apply
                      </Button>
                      <Button variant="outline" onClick={() => setEditMode(true)}>
                        <Edit2 className="mr-2 h-4 w-4" />
                        Edit Values
                      </Button>
                      <Button variant="ghost" onClick={() => setPositionReview({ ...positionReview, recommendation: "HOLD" })}>
                        <X className="mr-2 h-4 w-4" />
                        Reject
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Stop Loss</p>
                        <Input
                          type="number"
                          step="any"
                          value={editSl}
                          onChange={(e) => setEditSl(e.target.value)}
                          placeholder="Enter SL"
                          className="font-mono"
                        />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Take Profit</p>
                        <Input
                          type="number"
                          step="any"
                          value={editTp}
                          onChange={(e) => setEditTp(e.target.value)}
                          placeholder="Enter TP"
                          className="font-mono"
                        />
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button onClick={() => setShowApplyConfirm(true)} className="flex-1">
                        <Check className="mr-2 h-4 w-4" />
                        Apply Changes
                      </Button>
                      <Button variant="outline" onClick={() => setEditMode(false)}>
                        Cancel
                      </Button>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Quick Actions for HOLD - accept suggested adjustments */}
            {positionReview.recommendation === "HOLD" && (positionReview.suggested_sl || positionReview.suggested_tp || positionReview.suggested_trailing_sl) && (
              <div className="border rounded-lg p-4 space-y-3">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Suggested Adjustments
                  <HelpTooltip content="Optional adjustments detected. You can apply these while continuing to hold." />
                </h4>
                <div className="flex flex-wrap gap-2">
                  {positionReview.suggested_sl && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setEditSl(positionReview.suggested_sl!.toString())
                        setShowApplyConfirm(true)
                      }}
                    >
                      <Check className="mr-2 h-3 w-3" />
                      Accept SL: {positionReview.suggested_sl.toFixed(5)}
                    </Button>
                  )}
                  {positionReview.suggested_tp && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setEditTp(positionReview.suggested_tp!.toString())
                        setShowApplyConfirm(true)
                      }}
                    >
                      <Check className="mr-2 h-3 w-3" />
                      Accept TP: {positionReview.suggested_tp.toFixed(5)}
                    </Button>
                  )}
                  {positionReview.suggested_trailing_sl && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setEditSl(positionReview.suggested_trailing_sl!.toString())
                        setShowApplyConfirm(true)
                      }}
                    >
                      <Check className="mr-2 h-3 w-3" />
                      Trail SL to: {positionReview.suggested_trailing_sl.toFixed(5)}
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* Close Action (for CLOSE recommendation) */}
            {positionReview.recommendation === "CLOSE" && (
              <div className="flex flex-wrap gap-2 pt-2">
                <Button variant="destructive" onClick={() => setShowCloseConfirm(true)}>
                  <X className="mr-2 h-4 w-4" />
                  Close Position Now
                </Button>
                {positionContext?.entry_price && (
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setEditTp(positionContext.entry_price.toString())
                      setShowApplyConfirm(true)
                    }}
                  >
                    <Target className="mr-2 h-4 w-4" />
                    Set TP to Breakeven
                  </Button>
                )}
                <Button variant="outline" onClick={() => setPositionReview({ ...positionReview, recommendation: "HOLD" })}>
                  Ignore & Hold
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Trading Plan */}
      {tradingPlan && (tradingPlan.trader_plan || tradingPlan.risk_decision) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              {mode === "position" ? "Position Review Plan" : "Trading Plan"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {tradingPlan.trader_plan && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                  <User className="h-4 w-4" />
                  Trader Agent Analysis
                </div>
                <ScrollArea className="h-48 rounded-md border p-4 bg-muted/30">
                  <div className="text-sm">
                    <FormattedMarkdown content={tradingPlan.trader_plan} />
                  </div>
                </ScrollArea>
              </div>
            )}

            {tradingPlan.risk_decision && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                  <Shield className="h-4 w-4" />
                  Risk Manager Decision
                </div>
                <ScrollArea className="h-48 rounded-md border p-4 bg-muted/30">
                  <div className="text-sm">
                    <FormattedMarkdown content={tradingPlan.risk_decision} />
                  </div>
                </ScrollArea>
              </div>
            )}

            {/* Position Manager Reasoning (only in position mode) */}
            {mode === "position" && positionReview?.pm_reasoning && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                  <Target className="h-4 w-4" />
                  Position Manager Analysis
                  <HelpTooltip content="Dedicated position management agent that synthesizes all analysis to make HOLD/ADJUST/CLOSE decisions specific to your existing position." />
                </div>
                <ScrollArea className="h-48 rounded-md border p-4 bg-muted/30">
                  <div className="text-sm">
                    <FormattedMarkdown content={positionReview.pm_reasoning} />
                  </div>
                </ScrollArea>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Agent Progress */}
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
              {AGENTS
                .filter((agent) => !agent.positionModeOnly || mode === "position")
                .map((agent) => {
                const status = getAgentStatus(agent.id)
                const Icon = agent.icon
                const output = agentOutputs[agent.id]
                const isExpanded = expandedAgents.has(agent.id)

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
                      <div className={`flex-shrink-0 ${status === "running" ? "animate-pulse" : ""}`}>
                        {status === "completed" ? (
                          <CheckCircle2 className="h-5 w-5 text-green-500" />
                        ) : status === "running" ? (
                          <Loader2 className="h-5 w-5 text-amber-500 animate-spin" />
                        ) : (
                          <Clock className="h-5 w-5 text-muted-foreground" />
                        )}
                      </div>

                      <div className={`p-2 rounded-full ${
                        status === "completed" ? "bg-green-500/20" :
                          status === "running" ? "bg-amber-500/20" : "bg-muted"
                      }`}>
                        <Icon className={`h-4 w-4 ${agent.iconClass || ""} ${
                          status === "completed" ? "text-green-500" :
                            status === "running" ? "text-amber-500" : "text-muted-foreground"
                        }`} />
                      </div>

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
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {agent.description}
                        </p>
                      </div>

                      {output && (
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                            <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                          </Button>
                        </CollapsibleTrigger>
                      )}
                    </div>

                    <CollapsibleContent>
                      {output && (
                        <div className="ml-12 mt-2 p-3 bg-muted/50 rounded-lg border">
                          <p className="text-xs font-medium text-muted-foreground mb-2">Output:</p>
                          <div className="text-sm">
                            <FormattedMarkdown content={output.output} />
                          </div>
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

      {/* Apply Confirmation Dialog */}
      <Dialog open={showApplyConfirm} onOpenChange={setShowApplyConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Changes</DialogTitle>
            <DialogDescription>
              Apply the following changes to your position?
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-4 py-4">
            {editSl && (
              <div>
                <p className="text-sm text-muted-foreground">New Stop Loss</p>
                <p className="text-lg font-mono font-bold text-red-500">{parseFloat(editSl).toFixed(5)}</p>
              </div>
            )}
            {editTp && (
              <div>
                <p className="text-sm text-muted-foreground">New Take Profit</p>
                <p className="text-lg font-mono font-bold text-green-500">{parseFloat(editTp).toFixed(5)}</p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowApplyConfirm(false)}>
              Cancel
            </Button>
            <Button onClick={handleApplyChanges}>
              Apply Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Close Confirmation Dialog */}
      <Dialog open={showCloseConfirm} onOpenChange={setShowCloseConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-500">
              <AlertTriangle className="h-5 w-5" />
              Close Position?
            </DialogTitle>
            <DialogDescription>
              The AI analysis recommends closing this position{positionReview?.close_reason ? `: ${positionReview.close_reason}` : ""}.
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCloseConfirm(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleClosePosition}>
              Close Position
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
