"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { getDecisions, getDecision, getDecisionStats, closeDecision, getRetryInfo, getPerformanceStats, reconcileDecisions, PerformanceStats } from "@/lib/api"
import { formatCurrency, getProfitColor, formatDate } from "@/lib/utils"
import { RefreshCw, Eye, Clock, Target, TrendingUp, TrendingDown, BarChart3, CheckCircle, XCircle, Loader2, RotateCcw, ArrowUpDown } from "lucide-react"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { TradeExecutionWizard } from "@/components/TradeExecutionWizard"
import { AreaChart, Area, XAxis, YAxis, Tooltip as ChartTooltip, ResponsiveContainer } from "recharts"

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState({ status: "all", symbol: "" })
  const [selectedDecision, setSelectedDecision] = useState<any>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // Close decision state
  const [closingDecisionId, setClosingDecisionId] = useState<string | null>(null)
  const [closeExitPrice, setCloseExitPrice] = useState("")
  const [closeOutcome, setCloseOutcome] = useState("win")
  const [closeNotes, setCloseNotes] = useState("")
  const [closing, setClosing] = useState(false)
  const [showCloseDialog, setShowCloseDialog] = useState(false)

  // Performance
  const [perfStats, setPerfStats] = useState<PerformanceStats | null>(null)
  const [perfPeriod, setPerfPeriod] = useState<string>("all")
  const [reconciling, setReconciling] = useState(false)

  // Retry failed trade state
  const [retryWizardOpen, setRetryWizardOpen] = useState(false)
  const [retryData, setRetryData] = useState<any>(null)

  const fetchDecisions = async () => {
    setLoading(true)
    const status = filter.status === "all" ? undefined : filter.status
    const symbol = filter.symbol || undefined
    const days = perfPeriod === "all" ? undefined : parseInt(perfPeriod)
    const [decisionsRes, statsRes, perfRes] = await Promise.all([
      getDecisions(100, status, symbol),
      getDecisionStats(),
      getPerformanceStats(symbol || undefined, days),
    ])
    if (decisionsRes.data) setDecisions(decisionsRes.data.decisions || [])
    if (statsRes.data) setStats(statsRes.data)
    if (perfRes.data) setPerfStats(perfRes.data)
    setLoading(false)
  }

  const handleReconcile = async () => {
    setReconciling(true)
    const { data, error } = await reconcileDecisions()
    if (error) {
      alert(`Reconcile error: ${error}`)
    } else if (data) {
      if (data.reconciled_count > 0) {
        alert(`Reconciled ${data.reconciled_count} trade(s)`)
        fetchDecisions()
      } else {
        alert("No trades to reconcile — all decisions are up to date")
      }
    }
    setReconciling(false)
  }

  const fetchDecisionDetail = async (id: string) => {
    setDetailLoading(true)
    const { data } = await getDecision(id)
    if (data) setSelectedDecision(data)
    setDetailLoading(false)
  }

  const handleCloseDecision = async () => {
    if (!closingDecisionId || !closeExitPrice) return
    setClosing(true)
    const { data, error } = await closeDecision(
      closingDecisionId,
      parseFloat(closeExitPrice),
      closeOutcome,
      closeNotes || undefined
    )
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert(`Decision closed as ${closeOutcome}`)
      setShowCloseDialog(false)
      setClosingDecisionId(null)
      setCloseExitPrice("")
      setCloseOutcome("win")
      setCloseNotes("")
      fetchDecisions()
    }
    setClosing(false)
  }

  const openCloseDialog = (dec: any) => {
    setClosingDecisionId(dec.id)
    setCloseExitPrice("")
    setShowCloseDialog(true)
  }

  const handleRetry = async (dec: any) => {
    const { data, error } = await getRetryInfo(dec.id)
    if (error) {
      alert(`Error loading retry info: ${error}`)
      return
    }
    setRetryData(data)
    setRetryWizardOpen(true)
  }

  useEffect(() => {
    fetchDecisions()
  }, [filter, perfPeriod])

  const uniqueSymbols = [...new Set(decisions.map((d) => d.symbol))].sort()

  const getOutcomeVariant = (outcome: any) => {
    if (!outcome) return "secondary"
    if (outcome.status === "failed") return "destructive"
    if (outcome.status === "retried") return "secondary"
    if (outcome.status === "open" || outcome.status === "active") return "outline"
    if (outcome.was_correct) return "success"
    return "destructive"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Trade Decisions</h1>
          <p className="text-muted-foreground">View and analyze past trading decisions</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={perfPeriod} onValueChange={setPerfPeriod}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="7">Last 7 Days</SelectItem>
              <SelectItem value="30">Last 30 Days</SelectItem>
              <SelectItem value="90">Last 90 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={handleReconcile} disabled={reconciling}>
            {reconciling ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ArrowUpDown className="mr-2 h-4 w-4" />}
            Reconcile
          </Button>
          <HelpTooltip content="Sync with MT5: finds active decisions whose positions have closed (SL/TP hit or manual close) and records the outcome automatically." />
          <Button variant="outline" onClick={fetchDecisions} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      {stats && !stats.error && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total Decisions</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_decisions || 0}</div>
              <p className="text-xs text-muted-foreground">
                {stats.open_decisions || 0} open, {stats.closed_decisions || 0} closed
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${(stats.win_rate || 0) >= 50 ? "text-green-500" : "text-red-500"}`}>
                {(stats.win_rate || 0).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">
                <span className="text-green-500">{stats.wins || 0} wins</span> / <span className="text-red-500">{stats.losses || 0} losses</span>
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total P/L</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getProfitColor(stats.total_pnl || 0)}`}>
                {formatCurrency(stats.total_pnl || 0)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Avg P/L</CardTitle>
              <TrendingDown className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getProfitColor(stats.avg_pnl || 0)}`}>
                {formatCurrency(stats.avg_pnl || 0)}
              </div>
              <p className="text-xs text-muted-foreground">Per closed decision</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Top Symbol</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {stats.by_symbol && Object.keys(stats.by_symbol).length > 0 ? (
                <>
                  <div className="text-2xl font-bold">
                    {Object.entries(stats.by_symbol).sort((a: any, b: any) => b[1].total - a[1].total)[0]?.[0] || "—"}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {Object.entries(stats.by_symbol).sort((a: any, b: any) => b[1].total - a[1].total)[0]?.[1]?.total || 0} decisions
                  </p>
                </>
              ) : (
                <div className="text-muted-foreground">N/A</div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Performance Dashboard */}
      {perfStats && perfStats.total_closed > 0 && (
        <div className="space-y-4">
          {/* Equity Curve */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <CardTitle className="text-lg">Equity Curve</CardTitle>
                <HelpTooltip content="Cumulative P/L over time. Upward slope = consistent profitability. Steep drops = drawdowns to investigate. Each point is a closed trade." />
              </div>
              <CardDescription>
                {perfStats.total_closed} closed trades | Streak: {perfStats.streaks.current_streak > 0 ? `${perfStats.streaks.current_streak}W` : perfStats.streaks.current_streak < 0 ? `${Math.abs(perfStats.streaks.current_streak)}L` : "—"}
                {" | "}Best streak: {perfStats.streaks.max_win_streak}W | Worst: {perfStats.streaks.max_loss_streak}L
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={perfStats.equity_curve}>
                  <defs>
                    <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={perfStats.total_pnl >= 0 ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={perfStats.total_pnl >= 0 ? "#22c55e" : "#ef4444"} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <ChartTooltip
                    contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px" }}
                    formatter={(value: number, name: string) => [
                      `$${value.toFixed(2)}`,
                      name === "pnl" ? "Cumulative P/L" : "Trade P/L",
                    ]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Area type="monotone" dataKey="pnl" stroke={perfStats.total_pnl >= 0 ? "#22c55e" : "#ef4444"} fill="url(#pnlGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            {/* By Symbol */}
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <CardTitle className="text-lg">Performance by Symbol</CardTitle>
                  <HelpTooltip content="Breakdown per symbol. Compare win rates and P/L to see which instruments your strategy works best on." />
                </div>
              </CardHeader>
              <CardContent>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 font-medium">Symbol</th>
                      <th className="text-right py-2 font-medium">Trades</th>
                      <th className="text-right py-2 font-medium">Win Rate</th>
                      <th className="text-right py-2 font-medium">P/L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(perfStats.by_symbol)
                      .sort(([, a]: any, [, b]: any) => b.trades - a.trades)
                      .map(([sym, data]: [string, any]) => (
                        <tr key={sym} className="border-b last:border-0">
                          <td className="py-2 font-medium">{sym}</td>
                          <td className="py-2 text-right">{data.trades}</td>
                          <td className={`py-2 text-right ${data.win_rate >= 50 ? "text-green-500" : "text-red-500"}`}>
                            {data.win_rate.toFixed(0)}%
                          </td>
                          <td className={`py-2 text-right font-medium ${getProfitColor(data.pnl)}`}>
                            {formatCurrency(data.pnl)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>

            {/* Exit Reasons & Quality */}
            <div className="space-y-4">
              {/* Exit Reasons */}
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-lg">Exit Reasons</CardTitle>
                    <HelpTooltip content="How your trades ended. tp_hit = take profit reached (ideal). sl_hit = stop loss triggered. manual_close = closed manually or by reversal signal." />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(perfStats.by_exit_reason).map(([reason, data]: [string, any]) => {
                      const pct = (data.count / perfStats.total_closed) * 100
                      const label = reason === "tp_hit" ? "Take Profit" : reason === "sl_hit" ? "Stop Loss" : reason === "manual_close" ? "Manual Close" : reason === "reversal_signal" ? "Reversal Signal" : reason
                      const color = reason === "tp_hit" ? "bg-green-500" : reason === "sl_hit" ? "bg-red-500" : "bg-yellow-500"
                      return (
                        <div key={reason}>
                          <div className="flex justify-between text-sm mb-1">
                            <span>{label}</span>
                            <span className="text-muted-foreground">{data.count} ({pct.toFixed(0)}%) | {formatCurrency(data.pnl)}</span>
                          </div>
                          <div className="h-2 rounded-full bg-muted overflow-hidden">
                            <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Trade Quality */}
              {(Object.values(perfStats.quality.sl_placement).some(v => v > 0) || Object.values(perfStats.quality.tp_placement).some(v => v > 0)) && (
                <Card>
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-lg">Trade Quality</CardTitle>
                      <HelpTooltip content="Analysis of your SL and TP placement quality. 'Appropriate' is the target. Too many 'too tight' SLs = getting stopped out unnecessarily. Too many 'too ambitious' TPs = missing profit." />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Stop Loss Placement</p>
                      <div className="flex gap-2 text-xs">
                        {Object.entries(perfStats.quality.sl_placement).map(([label, count]) => (
                          <Badge key={label} variant={label === "appropriate" ? "success" : label === "too_tight" ? "destructive" : "warning"}>
                            {label.replace("_", " ")}: {count as number}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Take Profit Placement</p>
                      <div className="flex gap-2 text-xs">
                        {Object.entries(perfStats.quality.tp_placement).map(([label, count]) => (
                          <Badge key={label} variant={label === "appropriate" ? "success" : label === "too_ambitious" ? "destructive" : "warning"}>
                            {label.replace("_", " ")}: {count as number}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Best/Worst Trade */}
              {(perfStats.best_trade || perfStats.worst_trade) && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Notable Trades</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {perfStats.best_trade && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-green-500 font-medium">Best</span>
                        <span>{perfStats.best_trade.symbol} {perfStats.best_trade.action}</span>
                        <span className="text-green-500 font-medium">{formatCurrency(perfStats.best_trade.pnl)}</span>
                      </div>
                    )}
                    {perfStats.worst_trade && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-red-500 font-medium">Worst</span>
                        <span>{perfStats.worst_trade.symbol} {perfStats.worst_trade.action}</span>
                        <span className="text-red-500 font-medium">{formatCurrency(perfStats.worst_trade.pnl)}</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="py-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                placeholder="Filter by symbol..."
                value={filter.symbol}
                onChange={(e) => setFilter({ ...filter, symbol: e.target.value.toUpperCase() })}
                className="max-w-xs"
              />
            </div>
            <Select
              value={filter.status}
              onValueChange={(v) => setFilter({ ...filter, status: v })}
            >
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="open">Open</SelectItem>
                <SelectItem value="closed">Closed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Decisions Table */}
      <Card>
        <CardContent className="p-0">
          <ScrollArea className="h-[600px]">
            <table className="w-full">
              <thead className="sticky top-0 bg-card border-b">
                <tr className="text-left">
                  <th className="p-4 font-medium">Time</th>
                  <th className="p-4 font-medium">Symbol</th>
                  <th className="p-4 font-medium">Signal</th>
                  <th className="p-4 font-medium">Confidence</th>
                  <th className="p-4 font-medium">Entry</th>
                  <th className="p-4 font-medium">SL / TP</th>
                  <th className="p-4 font-medium">Status</th>
                  <th className="p-4 font-medium">P/L</th>
                  <th className="p-4 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {decisions.length > 0 ? (
                  decisions.map((dec) => (
                    <tr key={dec.id} className="border-b hover:bg-muted/50">
                      <td className="p-4 text-sm">
                        <div className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          {formatDate(dec.timestamp)}
                        </div>
                      </td>
                      <td className="p-4 font-medium">{dec.symbol}</td>
                      <td className="p-4">
                        <Badge
                          variant={
                            dec.signal === "BUY"
                              ? "buy"
                              : dec.signal === "SELL"
                              ? "sell"
                              : "hold"
                          }
                        >
                          {dec.signal}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-16 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full bg-primary"
                              style={{ width: `${(dec.confidence || 0) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm">
                            {((dec.confidence || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="p-4">{dec.entry_price || "—"}</td>
                      <td className="p-4">
                        <div className="text-sm">
                          <span className="text-red-500">{dec.stop_loss || "—"}</span>
                          {" / "}
                          <span className="text-green-500">{dec.take_profit || "—"}</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant={getOutcomeVariant(dec.outcome)}>
                          {dec.outcome?.status || "pending"}
                        </Badge>
                      </td>
                      <td className="p-4">
                        {dec.outcome?.status === "failed" ? (
                          <span className="text-xs text-red-500 max-w-[200px] truncate block" title={dec.execution_error}>
                            {dec.execution_error || "Execution failed"}
                          </span>
                        ) : dec.outcome?.pnl !== undefined ? (
                          <span className={`font-medium ${getProfitColor(dec.outcome.pnl)}`}>
                            {formatCurrency(dec.outcome.pnl)}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="p-4">
                        <div className="flex gap-1">
                          {dec.outcome?.status === "failed" && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleRetry(dec)}
                              className="text-yellow-500 border-yellow-500 hover:bg-yellow-500/10"
                            >
                              <RotateCcw className="h-4 w-4 mr-1" />
                              Retry
                            </Button>
                          )}
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => fetchDecisionDetail(dec.id)}
                                title="View Details"
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            </DialogTrigger>
                          <DialogContent className="max-w-2xl">
                            <DialogHeader>
                              <DialogTitle>Decision Details</DialogTitle>
                              <DialogDescription>
                                {selectedDecision?.symbol} - {formatDate(selectedDecision?.timestamp)}
                              </DialogDescription>
                            </DialogHeader>
                            {detailLoading ? (
                              <div className="p-8 text-center">Loading...</div>
                            ) : selectedDecision ? (
                              <ScrollArea className="max-h-[60vh]">
                                <div className="space-y-4 p-4">
                                  {/* Signal Info */}
                                  <div className="flex items-center gap-4">
                                    <Badge
                                      variant={
                                        selectedDecision.signal === "BUY"
                                          ? "buy"
                                          : selectedDecision.signal === "SELL"
                                          ? "sell"
                                          : "hold"
                                      }
                                      className="text-lg px-4 py-1"
                                    >
                                      {selectedDecision.signal}
                                    </Badge>
                                    <span className="text-lg">
                                      Confidence:{" "}
                                      {((selectedDecision.confidence || 0) * 100).toFixed(0)}%
                                    </span>
                                  </div>

                                  <Separator />

                                  {/* Price Levels */}
                                  <div className="grid grid-cols-3 gap-4">
                                    <div>
                                      <p className="text-sm text-muted-foreground">Entry</p>
                                      <p className="text-lg font-medium">
                                        {selectedDecision.entry_price}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-sm text-muted-foreground">Stop Loss</p>
                                      <p className="text-lg font-medium text-red-500">
                                        {selectedDecision.stop_loss}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-sm text-muted-foreground">Take Profit</p>
                                      <p className="text-lg font-medium text-green-500">
                                        {selectedDecision.take_profit}
                                      </p>
                                    </div>
                                  </div>

                                  <Separator />

                                  {/* Metadata */}
                                  <div className="grid grid-cols-2 gap-4">
                                    {selectedDecision.setup_type && (
                                      <div>
                                        <p className="text-sm text-muted-foreground">Setup Type</p>
                                        <Badge variant="outline">{selectedDecision.setup_type}</Badge>
                                      </div>
                                    )}
                                    {selectedDecision.higher_tf_bias && (
                                      <div>
                                        <p className="text-sm text-muted-foreground">HTF Bias</p>
                                        <Badge
                                          variant={
                                            selectedDecision.higher_tf_bias === "bullish"
                                              ? "buy"
                                              : selectedDecision.higher_tf_bias === "bearish"
                                              ? "sell"
                                              : "secondary"
                                          }
                                        >
                                          {selectedDecision.higher_tf_bias}
                                        </Badge>
                                      </div>
                                    )}
                                    {selectedDecision.confluence_score !== undefined && (
                                      <div>
                                        <p className="text-sm text-muted-foreground">
                                          Confluence Score
                                        </p>
                                        <p className="font-medium">
                                          {selectedDecision.confluence_score}/10
                                        </p>
                                      </div>
                                    )}
                                    {selectedDecision.volatility_regime && (
                                      <div>
                                        <p className="text-sm text-muted-foreground">
                                          Volatility Regime
                                        </p>
                                        <Badge variant="secondary">
                                          {selectedDecision.volatility_regime}
                                        </Badge>
                                      </div>
                                    )}
                                  </div>

                                  <Separator />

                                  {/* Execution Error */}
                                  {selectedDecision.execution_error && (
                                    <div className="rounded-md border border-red-500/50 bg-red-500/10 p-3">
                                      <p className="text-sm font-medium text-red-500 mb-1">Execution Failed</p>
                                      <p className="text-sm text-red-400">{selectedDecision.execution_error}</p>
                                    </div>
                                  )}

                                  {/* Rationale */}
                                  <div>
                                    <p className="text-sm text-muted-foreground mb-2">Rationale</p>
                                    <div className="rounded-md border p-3 bg-muted/50">
                                      <p className="text-sm whitespace-pre-wrap">
                                        {selectedDecision.rationale || "No rationale provided"}
                                      </p>
                                    </div>
                                  </div>

                                  {/* Key Factors */}
                                  {selectedDecision.key_factors?.length > 0 && (
                                    <div>
                                      <p className="text-sm text-muted-foreground mb-2">
                                        Key Factors
                                      </p>
                                      <div className="flex flex-wrap gap-2">
                                        {selectedDecision.key_factors.map(
                                          (factor: string, i: number) => (
                                            <Badge key={i} variant="secondary">
                                              {factor}
                                            </Badge>
                                          )
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Outcome */}
                                  {selectedDecision.outcome && (
                                    <>
                                      <Separator />
                                      <div>
                                        <p className="text-sm text-muted-foreground mb-2">
                                          Outcome
                                        </p>
                                        <div className="grid grid-cols-3 gap-4">
                                          <div>
                                            <p className="text-xs text-muted-foreground">Status</p>
                                            <Badge
                                              variant={getOutcomeVariant(selectedDecision.outcome)}
                                            >
                                              {selectedDecision.outcome.status}
                                            </Badge>
                                          </div>
                                          {selectedDecision.outcome.exit_price && (
                                            <div>
                                              <p className="text-xs text-muted-foreground">
                                                Exit Price
                                              </p>
                                              <p className="font-medium">
                                                {selectedDecision.outcome.exit_price}
                                              </p>
                                            </div>
                                          )}
                                          {selectedDecision.outcome.pnl !== undefined && (
                                            <div>
                                              <p className="text-xs text-muted-foreground">P/L</p>
                                              <p
                                                className={`font-medium ${getProfitColor(
                                                  selectedDecision.outcome.pnl
                                                )}`}
                                              >
                                                {formatCurrency(selectedDecision.outcome.pnl)}
                                              </p>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </div>
                              </ScrollArea>
                            ) : null}
                          </DialogContent>
                          </Dialog>
                          {/* Close Decision Button - only for open/active decisions */}
                          {(!dec.outcome || dec.outcome?.status === "open" || dec.outcome?.status === "active") && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => openCloseDialog(dec)}
                              title="Close Decision"
                            >
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} className="p-8 text-center text-muted-foreground">
                      No decisions found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Close Decision Dialog */}
      <Dialog open={showCloseDialog} onOpenChange={setShowCloseDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Close Decision</DialogTitle>
            <DialogDescription>
              Record the outcome of this trade decision for learning
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Exit Price</Label>
              <Input
                type="number"
                step="any"
                value={closeExitPrice}
                onChange={(e) => setCloseExitPrice(e.target.value)}
                placeholder="Enter exit price..."
              />
            </div>
            <div className="space-y-2">
              <Label>Outcome</Label>
              <Select value={closeOutcome} onValueChange={setCloseOutcome}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="win">
                    <span className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Win
                    </span>
                  </SelectItem>
                  <SelectItem value="loss">
                    <span className="flex items-center gap-2">
                      <XCircle className="h-4 w-4 text-red-500" />
                      Loss
                    </span>
                  </SelectItem>
                  <SelectItem value="breakeven">Breakeven</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Notes (optional)</Label>
              <Input
                value={closeNotes}
                onChange={(e) => setCloseNotes(e.target.value)}
                placeholder="Any notes about this trade..."
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCloseDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCloseDecision} disabled={closing || !closeExitPrice}>
              {closing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Close Decision
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Retry Trade Wizard */}
      {retryData && (
        <TradeExecutionWizard
          open={retryWizardOpen}
          onOpenChange={(open) => {
            setRetryWizardOpen(open)
            if (!open) {
              setRetryData(null)
              fetchDecisions()
            }
          }}
          symbol={retryData.symbol}
          signal={retryData.signal}
          suggestedStopLoss={retryData.stop_loss}
          suggestedTakeProfit={retryData.take_profit}
          rationale={retryData.rationale}
          currentPrice={retryData.current_price?.bid || retryData.current_price?.ask}
          failureReason={retryData.execution_error}
          failedDecisionId={retryData.decision_id}
        />
      )}
    </div>
  )
}
