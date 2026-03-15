"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import {
  reviewPosition,
  getReviewStatus,
  modifyPosition,
  closePosition,
  runSmcAnalysis,
  getPositions,
  runAssumptionReview,
  type AssumptionReviewReport,
} from "@/lib/api"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { formatCurrency, getProfitColor } from "@/lib/utils"
import {
  ArrowLeft,
  RefreshCw,
  Loader2,
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  BarChart3,
  Activity,
  Layers,
  X,
  ShieldCheck,
  CheckCircle,
  XCircle,
  Info,
} from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { DeepAnalysis } from "@/components/DeepAnalysis"
import { PositionChart } from "@/components/PositionChart"

export default function PositionReviewPage() {
  const params = useParams()
  const router = useRouter()
  const ticket = parseInt(params.ticket as string)

  const [position, setPosition] = useState<any>(null)
  const [atrData, setAtrData] = useState<any>(null)
  const [smcData, setSmcData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [atrLoading, setAtrLoading] = useState(false)
  const [smcLoading, setSmcLoading] = useState(false)

  // SMC timeframe state
  const [smcTimeframe, setSmcTimeframe] = useState("H1")
  const [multiTimeframe, setMultiTimeframe] = useState(false)
  const [mtfData, setMtfData] = useState<Record<string, any>>({}) // keyed by timeframe
  const [smcSensitive, setSmcSensitive] = useState(true) // More sensitive FVG detection
  const [smcLookback, setSmcLookback] = useState(50) // Bars to analyze

  // SL/TP modification state
  const [newSl, setNewSl] = useState("")
  const [newTp, setNewTp] = useState("")
  const [modifying, setModifying] = useState(false)

  // Close position state
  const [closeDialogOpen, setCloseDialogOpen] = useState(false)
  const [closing, setClosing] = useState(false)

  // Assumption review state
  const [assumptionReview, setAssumptionReview] = useState<AssumptionReviewReport | null>(null)
  const [assumptionLoading, setAssumptionLoading] = useState(false)
  const [assumptionError, setAssumptionError] = useState<string | null>(null)

  const TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]

  const fetchPosition = async () => {
    const { data } = await getPositions()
    if (data?.positions) {
      const pos = data.positions.find((p: any) => p.ticket === ticket)
      if (pos) {
        setPosition(pos)
        setNewSl(pos.sl?.toString() || "")
        setNewTp(pos.tp?.toString() || "")
      }
    }
    setLoading(false)
  }

  const fetchAtr = async () => {
    if (!position) return
    setAtrLoading(true)
    const { data, error } = await reviewPosition(ticket, "atr")
    if (error) {
      console.error("Error starting ATR review:", error)
      setAtrLoading(false)
      return
    }

    const taskId = data?.task_id
    if (!taskId) {
      console.error("No task_id returned from review")
      setAtrLoading(false)
      return
    }

    // Poll for results
    let completed = false
    let attempts = 0
    const maxAttempts = 60 // 2 minutes max

    while (!completed && attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 1000))
      const statusRes = await getReviewStatus(taskId)
      const statusData = statusRes.data

      if (statusData?.status === "completed") {
        completed = true
        if (statusData.result) setAtrData(statusData.result)
      } else if (statusData?.status === "error") {
        console.error("ATR review error:", statusData.error)
        completed = true
      }
      attempts++
    }

    setAtrLoading(false)
  }

  const fetchSmc = async (timeframe?: string) => {
    if (!position) return
    const tf = timeframe || smcTimeframe
    setSmcLoading(true)
    // Use configurable sensitivity: 0.1 for sensitive (10% ATR), 0.3 for standard (30% ATR)
    const fvgMinSize = smcSensitive ? 0.1 : 0.3
    const { data } = await runSmcAnalysis(position.symbol, tf, {
      fvgMinSize,
      lookback: smcLookback,
      debug: true
    })
    if (data) {
      setSmcData(data)
      setMtfData(prev => ({ ...prev, [tf]: data }))
    }
    setSmcLoading(false)
  }

  const fetchMultiTimeframeSmc = async () => {
    if (!position) return
    setSmcLoading(true)
    const results: Record<string, any> = {}
    const fvgMinSize = smcSensitive ? 0.1 : 0.3

    // Fetch all timeframes in parallel
    const promises = TIMEFRAMES.map(async (tf) => {
      const { data } = await runSmcAnalysis(position.symbol, tf, {
        fvgMinSize,
        lookback: smcLookback,
        debug: true
      })
      if (data) results[tf] = data
    })

    await Promise.all(promises)
    setMtfData(results)
    // Set primary view to H1 or first available
    if (results[smcTimeframe]) {
      setSmcData(results[smcTimeframe])
    } else if (results["H1"]) {
      setSmcData(results["H1"])
    }
    setSmcLoading(false)
  }

  const handleTimeframeChange = (tf: string) => {
    setSmcTimeframe(tf)
    if (mtfData[tf]) {
      setSmcData(mtfData[tf])
    } else {
      fetchSmc(tf)
    }
  }

  const handleModify = async () => {
    setModifying(true)
    const sl = newSl ? parseFloat(newSl) : undefined
    const tp = newTp ? parseFloat(newTp) : undefined
    const { error } = await modifyPosition(ticket, sl, tp)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert("Position modified successfully")
      fetchPosition()
    }
    setModifying(false)
  }

  const handleClose = async () => {
    setClosing(true)
    const { error } = await closePosition(ticket)
    if (error) {
      alert(`Error closing position: ${error}`)
      setClosing(false)
    } else {
      setCloseDialogOpen(false)
      router.push("/positions")
    }
  }

  const applyTrailingSl = () => {
    if (atrData?.atr_analysis?.trailing_sl) {
      setNewSl(atrData.atr_analysis.trailing_sl.toString())
    }
  }

  const fetchAssumptionReview = async (useLlm: boolean = true) => {
    if (!position) return
    setAssumptionLoading(true)
    setAssumptionError(null)
    setAssumptionReview(null)

    // Try to find the automation instance that owns this position via the source field
    // We pass position.source (from the positions list) as the instance name
    const source = position.source || position.comment || ""

    if (!source) {
      setAssumptionError("No automation source found for this position. Assumption review requires a trade decision record.")
      setAssumptionLoading(false)
      return
    }

    const { data, error } = await runAssumptionReview(source, useLlm)
    if (error) {
      setAssumptionError(error)
      setAssumptionLoading(false)
      return
    }

    // Find the report for this specific ticket
    const report = data?.reports?.find((r) => r.ticket === ticket)
    if (report) {
      setAssumptionReview(report)
    } else if (data?.reports?.length === 0) {
      setAssumptionError("No matching trade decision found for this position. It may have been opened manually without a decision record.")
    } else {
      setAssumptionError("Position not found in assumption review results.")
    }
    setAssumptionLoading(false)
  }

  useEffect(() => {
    fetchPosition()
  }, [ticket])

  useEffect(() => {
    if (position) {
      // Auto-fetch ATR and SMC (both free/local). AI analysis requires manual trigger to save credits.
      fetchAtr()
      fetchSmc()
    }
  }, [position?.ticket])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!position) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" onClick={() => router.push("/positions")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Positions
        </Button>
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            Position #{ticket} not found
          </CardContent>
        </Card>
      </div>
    )
  }

  const pnlPercent = (() => {
    if (!position.profit || !position.volume || !position.price_open) return null
    const pct = (position.profit / (position.price_open * position.volume * 100)) * 100
    if (isNaN(pct) || !isFinite(pct)) return null
    return pct.toFixed(2)
  })()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => router.push("/positions")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              {position.symbol}
              <Badge variant={position.type === "BUY" || position.type === 0 ? "buy" : "sell"}>
                {position.type === "BUY" || position.type === 0 ? "BUY" : "SELL"}
              </Badge>
              <span className="text-muted-foreground font-normal text-lg flex items-center gap-1">
                #{ticket}
                <HelpTooltip content="MT5 position ticket number - unique identifier for this trade in MetaTrader 5." iconClassName="h-3 w-3" />
              </span>
            </h1>
            <p className="text-muted-foreground">Position Review & Analysis</p>
          </div>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            fetchAtr()
            fetchSmc()
          }}
          disabled={atrLoading || smcLoading}
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${atrLoading || smcLoading ? "animate-spin" : ""}`} />
          Refresh Data
        </Button>
      </div>

      {/* Warning: No SL/TP Set */}
      {(!position.sl || position.sl === 0 || !position.tp || position.tp === 0) && (
        <Card className="border-yellow-500/50 bg-yellow-500/5">
          <CardContent className="py-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-yellow-500 mb-1">Risk Warning: Position Not Protected</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  {!position.sl || position.sl === 0
                    ? !position.tp || position.tp === 0
                      ? "This position has NO stop loss and NO take profit set. Your capital is at unlimited risk."
                      : "This position has NO stop loss set. Your capital is at unlimited risk if the market moves against you."
                    : "This position has NO take profit set. Consider setting a target to lock in profits."}
                </p>
                <div className="flex flex-wrap gap-2">
                  {(!position.sl || position.sl === 0) && atrData?.atr_analysis?.suggested_sl && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-yellow-500/50 hover:bg-yellow-500/10"
                      onClick={() => setNewSl(atrData.atr_analysis.suggested_sl.toString())}
                    >
                      <Target className="h-3 w-3 mr-1" />
                      Set ATR-based SL ({atrData.atr_analysis.suggested_sl.toFixed(5)})
                    </Button>
                  )}
                  {(!position.tp || position.tp === 0) && atrData?.atr_analysis?.tp_2r && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-yellow-500/50 hover:bg-yellow-500/10"
                      onClick={() => setNewTp(atrData.atr_analysis.tp_2r.toString())}
                    >
                      <Target className="h-3 w-3 mr-1" />
                      Set 2:1 TP ({atrData.atr_analysis.tp_2r.toFixed(5)})
                    </Button>
                  )}
                  {((!position.sl || position.sl === 0) || (!position.tp || position.tp === 0)) && !atrData && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-yellow-500/50 hover:bg-yellow-500/10"
                      onClick={fetchAtr}
                      disabled={atrLoading}
                    >
                      {atrLoading ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Target className="h-3 w-3 mr-1" />}
                      Load ATR Suggestions
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Position Summary */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Position Summary
              <HelpTooltip content="Overview of your open position including entry price, current P/L, and risk levels (SL/TP). Edit SL/TP directly and click Apply to update MT5." />
            </CardTitle>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setCloseDialogOpen(true)}
            >
              <X className="mr-2 h-4 w-4" />
              Close Position
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                Entry Price
                <HelpTooltip content="The price at which this position was opened." iconClassName="h-3 w-3" />
              </p>
              <p className="text-xl font-bold font-mono">{position.price_open?.toFixed(5)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                Current Price
                <HelpTooltip content="The current market price for this symbol." iconClassName="h-3 w-3" />
              </p>
              <p className="text-xl font-bold font-mono">{position.current_price?.toFixed(5)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                P/L
                <HelpTooltip content="Profit/Loss - your unrealized gain or loss on this position in account currency." iconClassName="h-3 w-3" />
              </p>
              <p className={`text-xl font-bold ${getProfitColor(position.profit)}`}>
                {formatCurrency(position.profit)}
                {pnlPercent && <span className="text-sm ml-1">({pnlPercent}%)</span>}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                Volume
                <HelpTooltip content="Position size in lots. 1 lot = 100,000 units of the base currency." iconClassName="h-3 w-3" />
              </p>
              <p className="text-xl font-bold">{position.volume}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                Stop Loss
                <HelpTooltip content="Price level where the position will automatically close to limit losses. Edit and click Apply to update." iconClassName="h-3 w-3" />
              </p>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  step="any"
                  value={newSl}
                  onChange={(e) => setNewSl(e.target.value)}
                  placeholder="None"
                  className="font-mono h-9 text-lg font-bold text-red-500 w-32 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                {atrData?.atr_analysis?.trailing_sl && (
                  <Button variant="ghost" size="sm" onClick={applyTrailingSl} title="Apply ATR trailing SL" className="h-9 px-2">
                    <Target className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
            <div>
              <p className="text-sm text-muted-foreground flex items-center gap-1">
                Take Profit
                <HelpTooltip content="Price level where the position will automatically close to lock in profits. Edit and click Apply to update." iconClassName="h-3 w-3" />
              </p>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  step="any"
                  value={newTp}
                  onChange={(e) => setNewTp(e.target.value)}
                  placeholder="None"
                  className="font-mono h-9 text-lg font-bold text-green-500 w-32 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                {atrData?.atr_analysis?.tp_2r && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setNewTp(atrData.atr_analysis.tp_2r.toString())}
                    title="Apply ATR-based TP (2:1 R:R)"
                    className="h-9 px-2"
                  >
                    <Target className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
          </div>
          {/* Apply/Reset buttons - show when values changed */}
          {(newSl !== (position.sl?.toString() || "") || newTp !== (position.tp?.toString() || "")) && (
            <div className="flex items-center justify-end gap-3 mt-4 pt-4 border-t">
              <span className="text-sm text-muted-foreground mr-2">
                Changes pending
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setNewSl(position.sl?.toString() || "")
                  setNewTp(position.tp?.toString() || "")
                }}
              >
                Reset
              </Button>
              <Button size="sm" onClick={handleModify} disabled={modifying}>
                {modifying && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Apply Changes
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Price Chart with SMC Zones */}
      <PositionChart
        symbol={position.symbol}
        direction={position.type === "BUY" || position.type === 0 ? "BUY" : "SELL"}
        entryPrice={position.price_open}
        currentPrice={position.current_price}
        stopLoss={newSl ? parseFloat(newSl) : position.sl}
        takeProfit={newTp ? parseFloat(newTp) : position.tp}
        orderBlocks={smcData?.order_blocks}
        fairValueGaps={smcData?.fair_value_gaps}
        liquidityZones={smcData?.liquidity_zones?.map((lz: any) => ({
          price: lz.price || lz.level,
          type: lz.type === "sell-side" || lz.type?.includes("high") ? "sell-side" : "buy-side",
          strength: lz.strength ? lz.strength * 100 : undefined,
          touched: lz.swept,
        }))}
        atrValue={atrData?.atr_analysis?.atr}
        digits={5}
      />

      {/* Assumption Review - Manual trigger */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <ShieldCheck className="h-4 w-4" />
                Assumption Review
                <HelpTooltip content="Checks whether the original trade assumptions still hold. Compares entry conditions against current SMC structure: has the bias shifted? Is the SL still protected? Are new zones blocking TP? Optionally asks an LLM for a nuanced interpretation." />
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Check if original trade thesis still holds
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => fetchAssumptionReview(false)}
                disabled={assumptionLoading}
              >
                {assumptionLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <ShieldCheck className="mr-2 h-4 w-4" />
                )}
                Quick Check
              </Button>
              <Button
                size="sm"
                onClick={() => fetchAssumptionReview(true)}
                disabled={assumptionLoading}
              >
                {assumptionLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <ShieldCheck className="mr-2 h-4 w-4" />
                )}
                Review with AI
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {assumptionError && (
            <div className="flex items-start gap-2 text-sm text-yellow-500 mb-4">
              <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <span>{assumptionError}</span>
            </div>
          )}

          {assumptionLoading && (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground mr-2" />
              <span className="text-sm text-muted-foreground">Analyzing market structure against trade assumptions...</span>
            </div>
          )}

          {!assumptionLoading && !assumptionReview && !assumptionError && (
            <p className="text-sm text-muted-foreground py-4 text-center">
              Click &quot;Quick Check&quot; for a fast SMC structure review, or &quot;Review with AI&quot; for a detailed LLM assessment.
            </p>
          )}

          {assumptionReview && (
            <div className="space-y-4">
              {/* Overall recommendation */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Recommendation:</span>
                  <Badge
                    variant={
                      assumptionReview.recommended_action === "close" ? "destructive" :
                      assumptionReview.recommended_action === "hold" ? "secondary" :
                      "outline"
                    }
                    className={
                      assumptionReview.recommended_action === "hold" ? "border-green-500/50 text-green-500" :
                      assumptionReview.recommended_action.startsWith("adjust") ? "border-yellow-500/50 text-yellow-500" :
                      ""
                    }
                  >
                    {assumptionReview.recommended_action.toUpperCase().replace("_", " ")}
                  </Badge>
                </div>
                <span className={`text-sm font-medium ${getProfitColor(assumptionReview.pnl_pct)}`}>
                  P&L: {assumptionReview.pnl_pct >= 0 ? "+" : ""}{assumptionReview.pnl_pct.toFixed(2)}%
                </span>
              </div>

              {/* Findings */}
              {assumptionReview.findings.length > 0 && (
                <div className="space-y-2">
                  {assumptionReview.findings.map((finding, i) => (
                    <div
                      key={i}
                      className={`flex items-start gap-2 p-3 rounded-lg text-sm ${
                        finding.severity === "critical"
                          ? "bg-red-500/10 border border-red-500/20"
                          : finding.severity === "warning"
                          ? "bg-yellow-500/10 border border-yellow-500/20"
                          : "bg-blue-500/10 border border-blue-500/20"
                      }`}
                    >
                      {finding.severity === "critical" ? (
                        <XCircle className="h-4 w-4 text-red-500 flex-shrink-0 mt-0.5" />
                      ) : finding.severity === "warning" ? (
                        <AlertTriangle className="h-4 w-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                      ) : (
                        <Info className="h-4 w-4 text-blue-500 flex-shrink-0 mt-0.5" />
                      )}
                      <div className="flex-1">
                        <span>{finding.message}</span>
                        {finding.suggested_value && finding.suggested_action && (
                          <div className="mt-1">
                            <Badge variant="outline" className="text-xs">
                              Suggested: {finding.suggested_action === "adjust_sl" ? "SL" : finding.suggested_action === "adjust_tp" ? "TP" : finding.suggested_action} → {finding.suggested_value.toFixed(5)}
                            </Badge>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {assumptionReview.findings.length === 0 && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-green-500/10 border border-green-500/20 text-sm">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>All original assumptions still hold. No structural issues detected.</span>
                </div>
              )}

              {/* Suggested SL/TP actions */}
              {(assumptionReview.suggested_sl || assumptionReview.suggested_tp) && (
                <div className="flex flex-wrap gap-2 pt-2 border-t">
                  {assumptionReview.suggested_sl && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-yellow-500/50 hover:bg-yellow-500/10"
                      onClick={() => {
                        setNewSl(assumptionReview.suggested_sl!.toString())
                      }}
                    >
                      <Target className="h-3 w-3 mr-1" />
                      Apply SL: {assumptionReview.suggested_sl.toFixed(5)}
                    </Button>
                  )}
                  {assumptionReview.suggested_tp && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="border-blue-500/50 hover:bg-blue-500/10"
                      onClick={() => {
                        setNewTp(assumptionReview.suggested_tp!.toString())
                      }}
                    >
                      <Target className="h-3 w-3 mr-1" />
                      Apply TP: {assumptionReview.suggested_tp.toFixed(5)}
                    </Button>
                  )}
                  {assumptionReview.recommended_action === "close" && (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => setCloseDialogOpen(true)}
                    >
                      <X className="h-3 w-3 mr-1" />
                      Close Position
                    </Button>
                  )}
                </div>
              )}

              {/* LLM Assessment */}
              {assumptionReview.llm_assessment && (
                <div className="mt-3 p-4 rounded-lg bg-muted/50 border">
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-1">
                    AI Assessment
                    <HelpTooltip content="LLM interpretation of the rule-based findings. Provides nuanced analysis considering the original trade rationale, current SMC structure, and overall market context." iconClassName="h-3 w-3" />
                  </h4>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
                    {assumptionReview.llm_assessment}
                  </p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ATR Volatility - Auto-loaded */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <Target className="h-4 w-4" />
              ATR & Risk Metrics
              <HelpTooltip content="Volatility and risk metrics calculated from price data." />
            </CardTitle>
            {atrLoading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          </div>
        </CardHeader>
        <CardContent>
          {atrData?.atr_analysis ? (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-6">
                <div>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    ATR (14)
                    <HelpTooltip content="Average True Range - measures volatility over 14 candles. Higher = more volatile market." iconClassName="h-3 w-3" />
                  </p>
                  <p className="text-lg font-bold font-mono">{atrData.atr_analysis.atr?.toFixed(5)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    Risk:Reward
                    <HelpTooltip content="Current R:R ratio based on your SL/TP. Higher is better - 2:1 means potential profit is 2x potential loss." iconClassName="h-3 w-3" />
                  </p>
                  <p className="text-lg font-bold">{atrData.atr_analysis.risk_reward?.toFixed(2) || "—"}:1</p>
                </div>
              </div>
              {/* SL/TP Suggestions */}
              <div className="flex flex-wrap gap-3">
                {atrData.atr_analysis.trailing_sl ? (
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                    <div>
                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                        Trailing SL
                        <HelpTooltip content="Stop loss 1.5 ATR from current price. Locks in profit while allowing normal volatility." iconClassName="h-3 w-3" />
                      </p>
                      <p className="text-sm font-bold font-mono text-yellow-500">{atrData.atr_analysis.trailing_sl?.toFixed(5)}</p>
                    </div>
                    <Button variant="outline" size="sm" onClick={applyTrailingSl} className="h-7">
                      Apply
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/30 border border-muted opacity-60">
                    <div>
                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                        Trailing SL
                        <HelpTooltip content="Trailing stop is only available when position is in profit. Move SL to lock in gains as price moves in your favor." iconClassName="h-3 w-3" />
                      </p>
                      <p className="text-sm font-mono text-muted-foreground">—</p>
                    </div>
                    <Button variant="outline" size="sm" disabled className="h-7">
                      Apply
                    </Button>
                  </div>
                )}
                {atrData.atr_analysis.tp_2r && (
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-green-500/10 border border-green-500/30">
                    <div>
                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                        TP (2:1 R:R)
                        <HelpTooltip content="Take profit for 2:1 risk-reward ratio. Potential profit is 2x your risk." iconClassName="h-3 w-3" />
                      </p>
                      <p className="text-sm font-bold font-mono text-green-500">{atrData.atr_analysis.tp_2r?.toFixed(5)}</p>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => setNewTp(atrData.atr_analysis.tp_2r.toString())} className="h-7">
                      Apply
                    </Button>
                  </div>
                )}
                {atrData.atr_analysis.tp_3r && (
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-green-500/10 border border-green-500/30">
                    <div>
                      <p className="text-xs text-muted-foreground flex items-center gap-1">
                        TP (3:1 R:R)
                        <HelpTooltip content="Take profit for 3:1 risk-reward ratio. Potential profit is 3x your risk." iconClassName="h-3 w-3" />
                      </p>
                      <p className="text-sm font-bold font-mono text-green-500">{atrData.atr_analysis.tp_3r?.toFixed(5)}</p>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => setNewTp(atrData.atr_analysis.tp_3r.toString())} className="h-7">
                      Apply
                    </Button>
                  </div>
                )}
              </div>
            </div>
          ) : atrLoading ? (
            <p className="text-sm text-muted-foreground">Loading ATR data...</p>
          ) : (
            <p className="text-sm text-muted-foreground">ATR data unavailable</p>
          )}
        </CardContent>
      </Card>

      {/* Deep Analysis - Multi-Agent */}
      <DeepAnalysis
        mode="position"
        ticket={ticket}
        symbol={position.symbol}
        timeframe="H1"
        positionContext={{
          direction: position.type === "BUY" || position.type === 0 ? "BUY" : "SELL",
          entry_price: position.price_open,
          current_price: position.current_price,
          current_sl: position.sl || 0,
          current_tp: position.tp || 0,
          volume: position.volume,
          profit: position.profit,
          pnl_pct: position.price_open ? ((position.profit || 0) / (position.price_open * position.volume * 100)) * 100 : 0,
        }}
        onApplyChanges={(sl, tp) => {
          if (sl !== undefined) setNewSl(sl.toString())
          if (tp !== undefined) setNewTp(tp.toString())
        }}
        onClosePosition={() => setCloseDialogOpen(true)}
      />

      {/* Market Structure (SMC) */}
      <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Market Structure (SMC)
                  <HelpTooltip content="Smart Money Concepts (SMC) analysis identifies institutional trading zones like Order Blocks, Fair Value Gaps, and Liquidity areas." />
                </CardTitle>
                <CardDescription>Smart Money Concepts analysis</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Select value={smcTimeframe} onValueChange={handleTimeframeChange}>
                  <SelectTrigger className="w-20 h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TIMEFRAMES.map((tf) => (
                      <SelectItem key={tf} value={tf}>{tf}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => multiTimeframe ? fetchMultiTimeframeSmc() : fetchSmc()}
                  disabled={smcLoading}
                >
                  {smcLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-4 mt-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="mtf"
                  checked={multiTimeframe}
                  onCheckedChange={(checked) => {
                    setMultiTimeframe(checked === true)
                    if (checked && Object.keys(mtfData).length < 2) {
                      fetchMultiTimeframeSmc()
                    }
                  }}
                />
                <label htmlFor="mtf" className="text-sm text-muted-foreground cursor-pointer">
                  Multi-TF
                </label>
                <HelpTooltip content="Analyze multiple timeframes simultaneously to find confluence between H1, H4, D1 etc." />
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="sensitive"
                  checked={smcSensitive}
                  onCheckedChange={(checked) => setSmcSensitive(checked === true)}
                />
                <label htmlFor="sensitive" className="text-sm text-muted-foreground cursor-pointer">
                  Sensitive FVG
                </label>
                <HelpTooltip content="Detect smaller FVGs (10% ATR vs 30% ATR threshold). Enable to find more gaps that may be missed with standard settings." />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted-foreground">Bars:</label>
                <Select value={smcLookback.toString()} onValueChange={(v) => setSmcLookback(parseInt(v))}>
                  <SelectTrigger className="w-20 h-7">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="30">30</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                    <SelectItem value="200">200</SelectItem>
                  </SelectContent>
                </Select>
                <HelpTooltip content="Number of candles to analyze. More bars includes older zones but may add noise." />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {smcLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : smcData ? (
              <div className="space-y-4">
                {/* Multi-timeframe Bias Comparison */}
                {multiTimeframe && Object.keys(mtfData).length > 0 && (
                  <div className="p-3 border rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground mb-2 flex items-center gap-1">Bias by Timeframe <HelpTooltip content="SMC market bias across multiple timeframes. Confluence (same bias on multiple TFs) increases confidence in direction." iconClassName="h-3 w-3" /></p>
                    <div className="flex flex-wrap gap-2">
                      {TIMEFRAMES.map((tf) => {
                        const data = mtfData[tf]
                        if (!data) return null
                        return (
                          <div
                            key={tf}
                            className={`flex items-center gap-1 px-2 py-1 rounded cursor-pointer border ${
                              smcTimeframe === tf ? "border-primary" : "border-transparent"
                            }`}
                            onClick={() => handleTimeframeChange(tf)}
                          >
                            <span className="text-xs font-medium">{tf}:</span>
                            <Badge
                              variant={
                                data.bias === "bullish" ? "buy" :
                                data.bias === "bearish" ? "sell" : "secondary"
                              }
                              className="text-xs"
                            >
                              {data.bias === "bullish" && <TrendingUp className="h-2 w-2 mr-0.5" />}
                              {data.bias === "bearish" && <TrendingDown className="h-2 w-2 mr-0.5" />}
                              {data.bias?.charAt(0).toUpperCase() || "N"}
                            </Badge>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Market Bias */}
                <div className="flex items-center gap-3">
                  <span className="text-sm text-muted-foreground flex items-center gap-1">Market Bias ({smcTimeframe}): <HelpTooltip content="Overall market direction based on SMC analysis - Bullish (price likely to rise), Bearish (price likely to fall), or Neutral." iconClassName="h-3 w-3" /></span>
                  <Badge
                    variant={
                      smcData.bias === "bullish" ? "buy" :
                      smcData.bias === "bearish" ? "sell" : "secondary"
                    }
                    className="text-sm"
                  >
                    {smcData.bias === "bullish" && <TrendingUp className="mr-1 h-3 w-3" />}
                    {smcData.bias === "bearish" && <TrendingDown className="mr-1 h-3 w-3" />}
                    {smcData.bias?.toUpperCase() || "NEUTRAL"}
                  </Badge>
                </div>

                {/* Detection Summary with Tooltips */}
                {smcData.summary && (
                  <div className="grid grid-cols-2 gap-2 p-3 border rounded-lg bg-muted/20 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-muted-foreground">FVGs:</span>
                      <span className="font-medium">{smcData.summary.unmitigated_fvgs || 0}</span>
                      <span className="text-muted-foreground">/ {smcData.summary.total_fvgs || 0}</span>
                      <HelpTooltip
                        content="Fair Value Gaps: Price imbalances from rapid moves where price may return. Shows unmitigated (active) / total detected."
                        side="right"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-muted-foreground">OBs:</span>
                      <span className="font-medium">{smcData.summary.unmitigated_obs || 0}</span>
                      <span className="text-muted-foreground">/ {smcData.summary.total_obs || 0}</span>
                      <HelpTooltip
                        content="Order Blocks: The last candle before a strong move, marking institutional entry zones. Shows unmitigated / total."
                        side="right"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-green-500">Bull:</span>
                      <span>{smcData.summary.bullish_fvgs || 0} FVG, {smcData.summary.bullish_obs || 0} OB</span>
                      <HelpTooltip
                        content="Bullish zones act as support - price may bounce up from these areas."
                        side="right"
                      />
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-red-500">Bear:</span>
                      <span>{smcData.summary.bearish_fvgs || 0} FVG, {smcData.summary.bearish_obs || 0} OB</span>
                      <HelpTooltip
                        content="Bearish zones act as resistance - price may reject down from these areas."
                        side="right"
                      />
                    </div>
                    {smcData.debug && (
                      <div className="col-span-2 flex items-center gap-2 text-xs text-muted-foreground mt-1 pt-1 border-t">
                        <span>Lookback: {smcData.debug.lookback_bars} bars</span>
                        <span>|</span>
                        <span>FVG min: {(smcData.debug.fvg_min_size_atr * 100).toFixed(0)}% ATR</span>
                        <span>|</span>
                        <span>Mitigated: {smcData.debug.mitigated_fvgs || 0}</span>
                        <HelpTooltip
                          content="Debug info: Lookback is bars analyzed. FVG min is the minimum gap size as % of ATR. Mitigated are FVGs that price has already returned to (filled)."
                          iconClassName="h-3 w-3"
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* Key Levels */}
                {smcData.key_levels?.length > 0 && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <p className="text-sm text-muted-foreground">Key Levels</p>
                      <HelpTooltip content="Important price levels from SMC analysis where price may react. Green = support/bullish, Red = resistance/bearish." />
                    </div>
                    <div className="space-y-1">
                      {smcData.key_levels.slice(0, 8).map((level: any, i: number) => {
                        const isNearSl = position.sl && Math.abs(level.price - position.sl) / position.sl < 0.005
                        const isNearTp = position.tp && Math.abs(level.price - position.tp) / position.tp < 0.005
                        const isNearEntry = Math.abs(level.price - position.price_open) / position.price_open < 0.005
                        // Get proper type string, filtering out numeric "0" values
                        const typeStr = level.type && level.type !== "0" && level.type !== 0 ? String(level.type) : ""
                        // Use direction field from backend if available, otherwise infer from type string
                        const isBullish = level.direction === "bullish" ||
                                          typeStr.toLowerCase().includes("support") ||
                                          typeStr.toLowerCase().includes("bull") ||
                                          typeStr.toLowerCase().includes("demand")
                        // Use the type from backend if valid, otherwise generate label
                        const levelLabel = typeStr || (level.source === "FVG" ? (isBullish ? "Bullish FVG" : "Bearish FVG") :
                                          level.source === "OB" ? (isBullish ? "Demand Zone" : "Supply Zone") : "Level")
                        // Tooltip descriptions
                        const getTooltip = () => {
                          if (level.source === "FVG") return "Fair Value Gap - price imbalance zone that may act as support/resistance"
                          if (level.source === "OB") return "Order Block - institutional entry zone from last candle before strong move"
                          if (typeStr.includes("Support")) return "Support level - price may bounce up from here"
                          if (typeStr.includes("Resistance")) return "Resistance level - price may reject down from here"
                          return "Key price level identified by SMC analysis"
                        }
                        return (
                          <div
                            key={i}
                            className={`flex items-center justify-between p-2 rounded text-sm ${
                              isNearSl ? "bg-red-500/10 border border-red-500/30" :
                              isNearTp ? "bg-green-500/10 border border-green-500/30" :
                              isNearEntry ? "bg-blue-500/10 border border-blue-500/30" : "bg-muted/30"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <Badge
                                variant={isBullish ? "buy" : "sell"}
                                className="text-xs"
                              >
                                {level.source}
                              </Badge>
                              <span className="text-xs text-muted-foreground">{levelLabel}</span>
                              <HelpTooltip content={getTooltip()} iconClassName="h-3 w-3" />
                              {isNearSl && <span className="text-xs text-red-500">(near SL)</span>}
                              {isNearTp && <span className="text-xs text-green-500">(near TP)</span>}
                              {isNearEntry && <span className="text-xs text-blue-500">(near entry)</span>}
                            </div>
                            <span className="font-mono">{level.price?.toFixed(5)}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Order Blocks */}
                {smcData.order_blocks?.length > 0 && (
                  <div>
                    <p className="text-sm text-muted-foreground mb-2 flex items-center gap-1">Order Blocks <HelpTooltip content="The last candle before a strong impulsive move, marking institutional entry zones." iconClassName="h-3 w-3" /></p>
                    <div className="flex flex-wrap gap-2">
                      {smcData.order_blocks.slice(0, 6).map((ob: any, i: number) => {
                        const midPrice = ((ob.top || ob.high || 0) + (ob.bottom || ob.low || 0)) / 2
                        return (
                          <Badge key={i} variant={ob.type === "bullish" ? "buy" : "sell"} className="text-xs">
                            {ob.type} OB @ {midPrice.toFixed(2)}
                          </Badge>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Liquidity Zones Warning */}
                {smcData.liquidity_zones?.some((z: any) => {
                  const zonePrice = z.price || z.level
                  return position.sl && Math.abs(zonePrice - position.sl) / position.sl < 0.003
                }) && (
                  <div className="flex items-center gap-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    <div>
                      <p className="font-medium flex items-center gap-1">
                        Stop Hunt Risk
                        <HelpTooltip content="Your stop loss is placed near a liquidity zone where many other traders' stops are likely clustered. Smart money may push price to this level to trigger stops before reversing." iconClassName="h-3 w-3" />
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Your SL is near a liquidity zone - consider adjusting
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground py-4">Click refresh to load SMC data</p>
            )}
          </CardContent>
        </Card>

      {/* Key Levels Chart */}
      {smcData?.key_levels?.length > 0 && position?.current_price && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Key Levels Chart
              <HelpTooltip
                content={
                  <div className="space-y-2">
                    <p><strong>What this shows:</strong> SMC price levels visualized as horizontal bars. Longer bars = higher price levels.</p>
                    <p><strong>Green bars:</strong> Support/bullish zones where price may bounce up.</p>
                    <p><strong>Red bars:</strong> Resistance/bearish zones where price may reject down.</p>
                    <p><strong>Yellow dashed line:</strong> Current market price.</p>
                    <p><strong>Blue line:</strong> Your entry price.</p>
                    <p><strong>Red/Green lines:</strong> Your SL and TP levels.</p>
                    <p>Use this to see how close price is to key SMC levels and whether your SL/TP align with market structure.</p>
                  </div>
                }
              />
            </CardTitle>
            <CardDescription>
              Price levels relative to current price ({position.current_price?.toFixed(2)})
              {position.sl && <span className="text-red-500"> | SL: {position.sl.toFixed(2)}</span>}
              {position.tp && <span className="text-green-500"> | TP: {position.tp.toFixed(2)}</span>}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[...smcData.key_levels]
                    .filter((level: any) => level.price && !isNaN(level.price))
                    .sort((a: any, b: any) => b.price - a.price)
                    .map((level: any, idx: number) => {
                      const typeStr = level.type?.toString() || ""
                      const isBullish = typeStr.toLowerCase().includes("support") ||
                                       typeStr.toLowerCase().includes("bull") ||
                                       typeStr.toLowerCase().includes("demand")
                      // Create a meaningful label
                      const label = level.source
                        ? `${level.source} ${typeStr !== "0" && typeStr ? `(${typeStr})` : ""}`
                        : typeStr || `Level ${idx + 1}`
                      return {
                        name: label.trim(),
                        price: level.price,
                        source: level.source,
                        distance: level.price - position.current_price,
                        isSupport: isBullish,
                      }
                    })}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
                >
                  <XAxis
                    type="number"
                    domain={['auto', 'auto']}
                    tickFormatter={(value) => typeof value === 'number' ? value.toFixed(0) : ''}
                    stroke="#888"
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={110}
                    tick={{ fontSize: 11 }}
                    stroke="#888"
                  />
                  <ChartTooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number) => [value.toFixed(2), "Price"]}
                    labelFormatter={(label) => label}
                  />
                  {/* Current Price Line */}
                  {position.current_price && !isNaN(position.current_price) && (
                    <ReferenceLine
                      x={position.current_price}
                      stroke="#facc15"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      label={{
                        value: `Current: ${position.current_price?.toFixed(2)}`,
                        position: "top",
                        fill: "#facc15",
                        fontSize: 11,
                      }}
                    />
                  )}
                  {/* Entry Price Line */}
                  {position.price_open && !isNaN(position.price_open) && (
                    <ReferenceLine
                      x={position.price_open}
                      stroke="#3b82f6"
                      strokeWidth={2}
                      label={{
                        value: `Entry: ${position.price_open?.toFixed(2)}`,
                        position: "insideTopRight",
                        fill: "#3b82f6",
                        fontSize: 11,
                      }}
                    />
                  )}
                  {/* Stop Loss Line */}
                  {position.sl && !isNaN(position.sl) && (
                    <ReferenceLine
                      x={position.sl}
                      stroke="#ef4444"
                      strokeWidth={2}
                      label={{
                        value: `SL: ${position.sl?.toFixed(2)}`,
                        position: "insideBottomRight",
                        fill: "#ef4444",
                        fontSize: 11,
                      }}
                    />
                  )}
                  {/* Take Profit Line */}
                  {position.tp && !isNaN(position.tp) && (
                    <ReferenceLine
                      x={position.tp}
                      stroke="#22c55e"
                      strokeWidth={2}
                      label={{
                        value: `TP: ${position.tp?.toFixed(2)}`,
                        position: "insideTopRight",
                        fill: "#22c55e",
                        fontSize: 11,
                      }}
                    />
                  )}
                  <Bar dataKey="price" radius={[0, 4, 4, 0]}>
                    {[...smcData.key_levels]
                      .filter((level: any) => level.price && !isNaN(level.price))
                      .sort((a: any, b: any) => b.price - a.price)
                      .map((entry: any, index: number) => {
                        const typeStr = entry.type?.toString().toLowerCase() || ""
                        const isSupport =
                          typeStr.includes("support") ||
                          typeStr.includes("bull") ||
                          typeStr.includes("demand")
                        return (
                          <Cell
                            key={`cell-${index}`}
                            fill={isSupport ? "#22c55e" : "#ef4444"}
                            fillOpacity={0.8}
                          />
                        )
                      })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-6 mt-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-green-500" />
                <span className="text-muted-foreground">Support / Bullish</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-red-500" />
                <span className="text-muted-foreground">Resistance / Bearish</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-yellow-400 border-dashed" />
                <span className="text-muted-foreground">Current Price</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-blue-500" />
                <span className="text-muted-foreground">Entry</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* SMC Details: Order Blocks, FVGs, Liquidity Zones */}
      {smcData && (
        <div className="grid gap-6 md:grid-cols-3">
          {/* Order Blocks */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-blue-500" />
                Order Blocks
                <HelpTooltip content="The last candle before a strong impulsive move. These zones mark where institutional traders entered positions and may act as support/resistance when price returns." />
              </CardTitle>
              <CardDescription>
                {smcData.order_blocks?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[250px]">
                {smcData.order_blocks?.length > 0 ? (
                  <div className="space-y-3">
                    {smcData.order_blocks.map((ob: any, i: number) => (
                      <div
                        key={i}
                        className="rounded-lg border p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <Badge variant={ob.type === "bullish" ? "buy" : "sell"}>
                            {ob.type}
                          </Badge>
                          {ob.strength && (
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              Strength: {(ob.strength * 100).toFixed(0)}%
                              <HelpTooltip
                                content={ob.type === "bullish"
                                  ? "Bullish OB strength: 80-100% = Very strong - if price returns here, high probability of bouncing UP. 50-79% = Moderate - may push price up but less reliable. Below 50% = Weak - not strong enough to confidently expect upward reaction."
                                  : "Bearish OB strength: 80-100% = Very strong - if price returns here, high probability of being pushed DOWN. 50-79% = Moderate - may push price down but inconclusive. Below 50% = Weak - not strong enough to confidently drive price lower."
                                }
                                iconClassName="h-3 w-3"
                              />
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">High:</span>{" "}
                            <span className="font-mono">{(ob.high || ob.top)?.toFixed(5)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Low:</span>{" "}
                            <span className="font-mono">{(ob.low || ob.bottom)?.toFixed(5)}</span>
                          </div>
                        </div>
                        {ob.mitigated !== undefined && (
                          <Badge variant={ob.mitigated ? "secondary" : "outline"}>
                            {ob.mitigated ? "Mitigated" : "Active"}
                          </Badge>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">
                    No order blocks found
                  </p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Fair Value Gaps */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-purple-500" />
                Fair Value Gaps
                <HelpTooltip content="Price imbalances caused by rapid moves where one candle's range doesn't overlap the next. Price often returns to 'fill' these gaps before continuing." />
              </CardTitle>
              <CardDescription>
                {smcData.fair_value_gaps?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[250px]">
                {smcData.fair_value_gaps?.length > 0 ? (
                  <div className="space-y-3">
                    {smcData.fair_value_gaps.map((fvg: any, i: number) => (
                      <div
                        key={i}
                        className="rounded-lg border p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <Badge variant={fvg.type === "bullish" ? "buy" : "sell"}>
                            {fvg.type}
                          </Badge>
                          {fvg.filled_pct !== undefined && (
                            <span className="text-xs text-muted-foreground">
                              {(fvg.filled_pct * 100).toFixed(0)}% filled
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">High:</span>{" "}
                            <span className="font-mono">{(fvg.high || fvg.top)?.toFixed(5)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Low:</span>{" "}
                            <span className="font-mono">{(fvg.low || fvg.bottom)?.toFixed(5)}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">
                    No FVGs found
                  </p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Liquidity Zones */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-orange-500" />
                Liquidity Zones
                <HelpTooltip content="Areas where stop losses cluster (swing highs/lows). Smart money often pushes price to these levels to trigger stops before reversing - known as 'stop hunts'." />
              </CardTitle>
              <CardDescription>
                {smcData.liquidity_zones?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[250px]">
                {smcData.liquidity_zones?.length > 0 ? (
                  <div className="space-y-3">
                    {smcData.liquidity_zones.map((lz: any, i: number) => {
                      const zonePrice = lz.price || lz.level
                      const isNearSl = position.sl && Math.abs(zonePrice - position.sl) / position.sl < 0.003
                      return (
                        <div
                          key={i}
                          className={`rounded-lg border p-3 space-y-2 ${isNearSl ? "border-yellow-500/50 bg-yellow-500/5" : ""}`}
                        >
                          <div className="flex items-center justify-between">
                            <Badge variant="outline">{lz.type || "Liquidity"}</Badge>
                            {lz.strength && (
                              <span className="text-xs text-muted-foreground flex items-center gap-1">
                                Strength: {(lz.strength * 100).toFixed(0)}%
                                <HelpTooltip
                                  content="Liquidity strength: 80-100% = High concentration - many stop losses clustered here, smart money will likely push price to this level to trigger stops, then reverse. 50-79% = Moderate - some stops here but less attractive target. Below 50% = Weak - few stops, unlikely to be specifically targeted for a stop hunt."
                                  iconClassName="h-3 w-3"
                                />
                              </span>
                            )}
                          </div>
                          <div className="text-sm">
                            <span className="text-muted-foreground">Level:</span>{" "}
                            <span className="font-mono">{zonePrice?.toFixed(5)}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {lz.swept !== undefined && (
                              <Badge variant={lz.swept ? "secondary" : "outline"}>
                                {lz.swept ? "Swept" : "Untested"}
                              </Badge>
                            )}
                            {isNearSl && (
                              <Badge variant="destructive" className="text-xs">
                                <AlertTriangle className="h-3 w-3 mr-1" />
                                Near SL
                              </Badge>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">
                    No liquidity zones found
                  </p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Close Position Confirmation Dialog */}
      <Dialog open={closeDialogOpen} onOpenChange={setCloseDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Close Position</DialogTitle>
            <DialogDescription>
              Are you sure you want to close this position?
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Symbol</span>
              <span className="font-medium">{position?.symbol}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Direction</span>
              <Badge variant={position?.type === "BUY" || position?.type === 0 ? "buy" : "sell"}>
                {position?.type === "BUY" || position?.type === 0 ? "BUY" : "SELL"}
              </Badge>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Volume</span>
              <span className="font-medium">{position?.volume} lots</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Current P/L</span>
              <span className={`font-medium ${getProfitColor(position?.profit)}`}>
                {formatCurrency(position?.profit)}
              </span>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCloseDialogOpen(false)} disabled={closing}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleClose} disabled={closing}>
              {closing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Close Position
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
