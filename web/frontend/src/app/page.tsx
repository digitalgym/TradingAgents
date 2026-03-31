"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { getDashboard, getRiskMetrics, getPortfolioStatus, getSignals } from "@/lib/api"
import { formatCurrency, formatPercent, getProfitColor, formatDate } from "@/lib/utils"
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  BarChart3,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Target,
  Eye,
  RotateCcw,
} from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { TradeExecutionWizard } from "@/components/TradeExecutionWizard"
import Link from "next/link"

export default function Dashboard() {
  const [dashboard, setDashboard] = useState<any>(null)
  const [riskMetrics, setRiskMetrics] = useState<any>(null)
  const [portfolioStatus, setPortfolioStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  // All signals from automation
  const [allSignals, setAllSignals] = useState<Array<any & { pipeline: string, instance: string }>>([])
  const [signalFilter, setSignalFilter] = useState<string>("actionable")
  const [pipelineFilter, setPipelineFilter] = useState<string>("all")
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  const [detailResult, setDetailResult] = useState<any>(null)
  const [reentryOpen, setReentryOpen] = useState(false)
  const [reentrySignal, setReentrySignal] = useState<any>(null)

  const pipelineLabels: Record<string, string> = {
    smc_quant_basic: "SMC Quant Basic",
    smc_quant: "SMC Quant",
    smc_mtf: "SMC MTF",
    breakout_quant: "Breakout",
    range_quant: "Range",
    volume_profile: "Volume Profile",
    rule_based: "Rule-Based",
    multi_agent: "Multi-Agent",
  }

  const pipelineColors: Record<string, string> = {
    smc_quant_basic: "text-purple-500",
    smc_quant: "text-emerald-500",
    smc_mtf: "text-indigo-500",
    breakout_quant: "text-orange-500",
    range_quant: "text-teal-500",
    volume_profile: "text-blue-500",
    rule_based: "text-cyan-500",
    multi_agent: "text-amber-500",
  }

  const fetchData = async () => {
    setLoading(true)
    const [dashRes, riskRes, portRes, signalsRes] = await Promise.all([
      getDashboard(),
      getRiskMetrics(),
      getPortfolioStatus(),
      getSignals({ limit: 500 }),
    ])
    if (dashRes.data) setDashboard(dashRes.data)
    if (riskRes.data) setRiskMetrics(riskRes.data)
    if (portRes.data) setPortfolioStatus(portRes.data)

    // Signals come from the DB table — already sorted by created_at DESC
    if (signalsRes.data?.signals) {
      const signals = signalsRes.data.signals.map((s: any) => ({
        ...s,
        timestamp: s.created_at,
        instance: s.source,
      }))
      setAllSignals(signals)
    }

    setLoading(false)
  }

  useEffect(() => {
    fetchData()
  }, [])

  const account = dashboard?.account || {}
  const positions = dashboard?.positions || { count: 0, items: [], total_profit: 0 }
  const decisions = dashboard?.recent_decisions || []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">Overview of your trading system</p>
        </div>
        <Button variant="outline" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Account Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Balance</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(account.balance || 0)}</div>
            <p className="text-xs text-muted-foreground">
              Free Margin: {formatCurrency(account.free_margin || 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Equity</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(account.equity || 0)}</div>
            <p className="text-xs text-muted-foreground">
              Margin Used: {formatCurrency(account.margin || 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Open P/L</CardTitle>
            {(account.profit || 0) >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getProfitColor(account.profit || 0)}`}>
              {formatCurrency(account.profit || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              {positions.count} open position{positions.count !== 1 ? "s" : ""}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Automation</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Badge variant={portfolioStatus?.running ? "success" : "secondary"}>
                {portfolioStatus?.running ? "Running" : "Stopped"}
              </Badge>
            </div>
            {portfolioStatus?.last_run && (
              <p className="mt-1 text-xs text-muted-foreground">
                Last: {formatDate(portfolioStatus.last_run)}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Risk Metrics */}
      {riskMetrics && !riskMetrics.error && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {riskMetrics.sharpe_ratio?.toFixed(2) || "N/A"}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Sortino Ratio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {riskMetrics.sortino_ratio?.toFixed(2) || "N/A"}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold text-red-500">
                {riskMetrics.max_drawdown ? formatPercent(-riskMetrics.max_drawdown * 100, 1) : "N/A"}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold">
                {riskMetrics.win_rate ? `${(riskMetrics.win_rate * 100).toFixed(1)}%` : "N/A"}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {/* Open Positions */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Open Positions</CardTitle>
                <CardDescription>
                  {positions.count} position{positions.count !== 1 ? "s" : ""} •{" "}
                  <span className={getProfitColor(positions.total_profit)}>
                    {formatCurrency(positions.total_profit)}
                  </span>
                </CardDescription>
              </div>
              <Link href="/positions">
                <Button variant="outline" size="sm">
                  View All
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[250px]">
              {positions.items?.length > 0 ? (
                <div className="space-y-3">
                  {positions.items.map((pos: any) => (
                    <div
                      key={pos.ticket}
                      className="flex items-center justify-between rounded-lg border p-3"
                    >
                      <div className="flex items-center gap-3">
                        <Badge variant={pos.type === "BUY" ? "buy" : "sell"}>{pos.type}</Badge>
                        <div>
                          <p className="font-medium">{pos.symbol}</p>
                          <p className="text-xs text-muted-foreground">
                            {pos.volume} lots @ {pos.open_price}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`font-medium ${getProfitColor(pos.profit)}`}>
                          {formatCurrency(pos.profit)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Current: {pos.current_price}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No open positions
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Recent Decisions */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Recent Decisions</CardTitle>
                <CardDescription>Latest trade decisions from analysis</CardDescription>
              </div>
              <Link href="/decisions">
                <Button variant="outline" size="sm">
                  View All
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[250px]">
              {decisions.length > 0 ? (
                <div className="space-y-3">
                  {decisions.map((dec: any) => (
                    <div
                      key={dec.id}
                      className="flex items-center justify-between rounded-lg border p-3"
                    >
                      <div className="flex items-center gap-3">
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
                        <div>
                          <p className="font-medium">{dec.symbol}</p>
                          <p className="text-xs text-muted-foreground">
                            Confidence: {(dec.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {formatDate(dec.timestamp)}
                        </div>
                        <Badge variant="outline" className="mt-1">
                          {dec.outcome?.status || "pending"}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No recent decisions
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* All Automation Signals */}
      {allSignals.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Recent Signals
                </CardTitle>
                <CardDescription>{allSignals.length} total signals from all automations</CardDescription>
              </div>
              <div className="flex items-center gap-3">
                {/* Pipeline filter */}
                <div className="flex items-center gap-1">
                  <span className="text-xs text-muted-foreground mr-1">Pipeline:</span>
                  {["all", ...Object.keys(pipelineLabels).filter(p => allSignals.some(s => s.pipeline === p))].map((p) => (
                    <Button
                      key={p}
                      variant={pipelineFilter === p ? "default" : "ghost"}
                      size="sm"
                      className={`h-6 px-2 text-xs ${pipelineFilter !== p && p !== "all" ? (pipelineColors[p] || "") : ""}`}
                      onClick={() => setPipelineFilter(p)}
                    >
                      {p === "all" ? "All" : pipelineLabels[p] || p}
                    </Button>
                  ))}
                </div>
                {/* Signal filter */}
                <div className="flex items-center gap-1">
                  <span className="text-xs text-muted-foreground mr-1">Signal:</span>
                  {["actionable", "all", "BUY", "SELL", "HOLD"].map((filter) => (
                    <Button
                      key={filter}
                      variant={signalFilter === filter ? "default" : "ghost"}
                      size="sm"
                      className={`h-6 px-2 text-xs ${
                        signalFilter === filter
                          ? ""
                          : filter === "BUY"
                          ? "text-green-500 hover:text-green-400"
                          : filter === "SELL"
                          ? "text-red-500 hover:text-red-400"
                          : filter === "HOLD"
                          ? "text-gray-500 hover:text-gray-400"
                          : ""
                      }`}
                      onClick={() => setSignalFilter(filter)}
                    >
                      {filter === "actionable" ? "Actionable" : filter === "all" ? "All" : filter}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              <div className="space-y-2">
                {allSignals
                  .filter((r) => {
                    const sf = signalFilter === "all" ? true : signalFilter === "actionable" ? r.signal !== "HOLD" : r.signal === signalFilter
                    const pf = pipelineFilter === "all" ? true : r.pipeline === pipelineFilter
                    return sf && pf
                  })
                  .map((result, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2.5 rounded-lg bg-muted/50">
                      <div className="flex items-center gap-3">
                        {result.signal === "BUY" ? (
                          <TrendingUp className="h-4 w-4 text-green-500" />
                        ) : result.signal === "SELL" ? (
                          <TrendingDown className="h-4 w-4 text-red-500" />
                        ) : (
                          <Clock className="h-4 w-4 text-gray-500" />
                        )}
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-medium">{result.symbol}</p>
                            <Badge variant={result.signal === "BUY" ? "buy" : result.signal === "SELL" ? "sell" : "secondary"} className="text-xs">
                              {result.signal}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {(result.confidence * 100).toFixed(0)}%
                            </span>
                            {result.executed && result.decision_id && (
                              <Badge variant="success" className="text-[10px] h-4 px-1">Executed</Badge>
                            )}
                            {result.executed && !result.decision_id && (
                              <Badge variant="outline" className="text-[10px] h-4 px-1 border-yellow-500 text-yellow-500">Pending</Badge>
                            )}
                            {!result.executed && result.execution_error && (
                              <Badge variant="destructive" className="text-[10px] h-4 px-1">Failed</Badge>
                            )}
                          </div>
                          <p className={`text-xs ${pipelineColors[result.pipeline] || "text-muted-foreground"}`}>
                            {pipelineLabels[result.pipeline] || result.pipeline}
                            <span className="text-muted-foreground ml-1">• {result.instance}</span>
                          </p>
                        </div>
                      </div>
                      <div className="text-right flex items-center gap-2">
                        <div>
                          {result.entry_price && (
                            <p className="text-xs text-muted-foreground">
                              Entry: {result.entry_price?.toFixed(2)}
                              {result.stop_loss && <span className="text-red-500 ml-2">SL: {result.stop_loss?.toFixed(2)}</span>}
                              {result.take_profit && <span className="text-green-500 ml-2">TP: {result.take_profit?.toFixed(2)}</span>}
                            </p>
                          )}
                          <p className="text-xs text-muted-foreground">
                            {formatDate(result.timestamp)}
                          </p>
                        </div>
                        {result.signal !== "HOLD" && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                            title="Re-enter trade"
                            onClick={() => {
                              setReentrySignal(result)
                              setReentryOpen(true)
                            }}
                          >
                            <RotateCcw className="h-3.5 w-3.5" />
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                          onClick={() => {
                            setDetailResult(result)
                            setDetailModalOpen(true)
                          }}
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* Signal Detail Modal */}
      <Dialog open={detailModalOpen} onOpenChange={setDetailModalOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {detailResult?.symbol}
              <Badge variant={detailResult?.signal === "BUY" ? "buy" : detailResult?.signal === "SELL" ? "sell" : "secondary"}>
                {detailResult?.signal}
              </Badge>
              <span className="text-sm font-normal text-muted-foreground">
                {((detailResult?.confidence || 0) * 100).toFixed(0)}% confidence
              </span>
            </DialogTitle>
          </DialogHeader>
          {detailResult && (
            <div className="space-y-4">
              {/* Price levels */}
              {(detailResult.entry_price || detailResult.stop_loss || detailResult.take_profit) && (
                <div className="grid grid-cols-3 gap-3">
                  {detailResult.entry_price && (
                    <div className="text-center p-2 rounded bg-muted/50">
                      <p className="text-xs text-muted-foreground">Entry</p>
                      <p className="font-medium text-blue-500">{detailResult.entry_price?.toFixed(2)}</p>
                    </div>
                  )}
                  {detailResult.stop_loss && (
                    <div className="text-center p-2 rounded bg-muted/50">
                      <p className="text-xs text-muted-foreground">Stop Loss</p>
                      <p className="font-medium text-red-500">{detailResult.stop_loss?.toFixed(2)}</p>
                    </div>
                  )}
                  {detailResult.take_profit && (
                    <div className="text-center p-2 rounded bg-muted/50">
                      <p className="text-xs text-muted-foreground">Take Profit</p>
                      <p className="font-medium text-green-500">{detailResult.take_profit?.toFixed(2)}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Analysis */}
              {detailResult.rationale && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Analysis</p>
                  <ScrollArea className="h-[200px]">
                    <p className="text-sm whitespace-pre-wrap">{detailResult.rationale}</p>
                  </ScrollArea>
                </div>
              )}

              {/* Meta */}
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className={pipelineColors[detailResult.pipeline] || ""}>
                  {pipelineLabels[detailResult.pipeline] || detailResult.pipeline}
                </span>
                <span>{formatDate(detailResult.timestamp)}</span>
                {detailResult.duration_seconds && <span>{detailResult.duration_seconds.toFixed(0)}s</span>}
              </div>

              {/* Re-enter button */}
              {detailResult.signal !== "HOLD" && (
                <Button
                  className="w-full"
                  variant={detailResult.signal === "BUY" ? "default" : "destructive"}
                  onClick={() => {
                    setReentrySignal(detailResult)
                    setDetailModalOpen(false)
                    setReentryOpen(true)
                  }}
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Re-enter {detailResult.signal} Trade
                </Button>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Re-entry Trade Execution Wizard */}
      {reentrySignal && (
        <TradeExecutionWizard
          open={reentryOpen}
          onOpenChange={setReentryOpen}
          symbol={reentrySignal.symbol}
          signal={reentrySignal.signal}
          suggestedEntry={reentrySignal.entry_price}
          suggestedStopLoss={reentrySignal.stop_loss}
          suggestedTakeProfit={reentrySignal.take_profit}
          rationale={reentrySignal.rationale}
        />
      )}
    </div>
  )
}
