"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { getRiskMetrics, getRiskGuardrails, getCircuitBreaker, getBreachHistory, resetCircuitBreaker, calculatePositionSize } from "@/lib/api"
import { formatCurrency, formatPercent, formatDate } from "@/lib/utils"
import {
  RefreshCw,
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Calculator,
  Target,
  Zap,
  History,
  RotateCcw,
  Loader2,
} from "lucide-react"

export default function RiskPage() {
  const [metrics, setMetrics] = useState<any>(null)
  const [guardrails, setGuardrails] = useState<any>(null)
  const [circuitBreaker, setCircuitBreaker] = useState<any>(null)
  const [breachHistory, setBreachHistory] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [resetting, setResetting] = useState(false)

  // Position size calculator
  const [calcSymbol, setCalcSymbol] = useState("XAUUSD")
  const [calcEntry, setCalcEntry] = useState("")
  const [calcSl, setCalcSl] = useState("")
  const [calcRisk, setCalcRisk] = useState("1")
  const [calcResult, setCalcResult] = useState<any>(null)
  const [calcLoading, setCalcLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    const [metricsRes, guardrailsRes, circuitRes, breachRes] = await Promise.all([
      getRiskMetrics(),
      getRiskGuardrails(),
      getCircuitBreaker(),
      getBreachHistory(20),
    ])
    if (metricsRes.data) setMetrics(metricsRes.data)
    if (guardrailsRes.data) setGuardrails(guardrailsRes.data)
    if (circuitRes.data) setCircuitBreaker(circuitRes.data)
    if (breachRes.data) setBreachHistory(breachRes.data.breaches || [])
    setLoading(false)
  }

  const handleResetCircuitBreaker = async () => {
    setResetting(true)
    const { data, error } = await resetCircuitBreaker()
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert("Circuit breaker reset successfully")
      fetchData()
    }
    setResetting(false)
  }

  useEffect(() => {
    fetchData()
  }, [])

  const handleCalculateSize = async () => {
    if (!calcEntry || !calcSl) return
    setCalcLoading(true)
    const { data } = await calculatePositionSize(
      calcSymbol,
      parseFloat(calcEntry),
      parseFloat(calcSl),
      parseFloat(calcRisk)
    )
    if (data) setCalcResult(data)
    setCalcLoading(false)
  }

  const getMetricColor = (value: number, thresholds: { good: number; bad: number }) => {
    if (value >= thresholds.good) return "text-green-500"
    if (value <= thresholds.bad) return "text-red-500"
    return "text-yellow-500"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Risk Management</h1>
          <p className="text-muted-foreground">Monitor risk metrics and calculate position sizes</p>
        </div>
        <Button variant="outline" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Risk Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${getMetricColor(
                metrics?.sharpe_ratio || 0,
                { good: 1, bad: 0 }
              )}`}
            >
              {metrics?.sharpe_ratio?.toFixed(2) || "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Risk-adjusted returns</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Sortino Ratio</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${getMetricColor(
                metrics?.sortino_ratio || 0,
                { good: 1.5, bad: 0 }
              )}`}
            >
              {metrics?.sortino_ratio?.toFixed(2) || "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Downside risk adjusted</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            <TrendingDown className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {metrics?.max_drawdown
                ? formatPercent(-metrics.max_drawdown * 100, 1)
                : "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Largest peak-to-trough decline</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div
              className={`text-2xl font-bold ${getMetricColor(
                (metrics?.win_rate || 0) * 100,
                { good: 55, bad: 45 }
              )}`}
            >
              {metrics?.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Winning trade percentage</p>
          </CardContent>
        </Card>
      </div>

      {/* Additional Metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Calmar Ratio</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">
              {metrics?.calmar_ratio?.toFixed(2) || "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Return / Max Drawdown</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Value at Risk (95%)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-500">
              {metrics?.var_95 ? formatCurrency(metrics.var_95) : "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Max expected daily loss</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Profit Factor</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={`text-xl font-bold ${getMetricColor(
                metrics?.profit_factor || 0,
                { good: 1.5, bad: 1 }
              )}`}
            >
              {metrics?.profit_factor?.toFixed(2) || "N/A"}
            </div>
            <p className="text-xs text-muted-foreground">Gross profit / Gross loss</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Risk Guardrails */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Risk Guardrails
            </CardTitle>
            <CardDescription>Active risk limits and constraints</CardDescription>
          </CardHeader>
          <CardContent>
            {guardrails && !guardrails.error ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Daily Loss Used</span>
                    <span>
                      {guardrails.daily_loss_used?.toFixed(1) || 0}% /{" "}
                      {guardrails.daily_loss_limit || 5}%
                    </span>
                  </div>
                  <Progress
                    value={
                      ((guardrails.daily_loss_used || 0) /
                        (guardrails.daily_loss_limit || 5)) *
                      100
                    }
                    className={
                      (guardrails.daily_loss_used || 0) > (guardrails.daily_loss_limit || 5) * 0.8
                        ? "[&>div]:bg-red-500"
                        : ""
                    }
                  />
                </div>

                <Separator />

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Consecutive Losses</p>
                    <p className="text-lg font-medium">
                      {guardrails.consecutive_losses || 0} /{" "}
                      {guardrails.max_consecutive_losses || 3}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Cooldown Status</p>
                    <Badge variant={guardrails.in_cooldown ? "destructive" : "success"}>
                      {guardrails.in_cooldown ? "Active" : "Clear"}
                    </Badge>
                  </div>
                </div>

                {guardrails.blocked && (
                  <div className="flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 p-3 text-destructive">
                    <AlertTriangle className="h-5 w-5" />
                    <span>Trading is currently blocked by risk guardrails</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground">
                {guardrails?.error || "Failed to load guardrails"}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Position Size Calculator */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5" />
              Position Size Calculator
            </CardTitle>
            <CardDescription>Calculate optimal position size using Kelly criterion</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Symbol</Label>
                  <Input
                    value={calcSymbol}
                    onChange={(e) => setCalcSymbol(e.target.value.toUpperCase())}
                    placeholder="XAUUSD"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Risk %</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={calcRisk}
                    onChange={(e) => setCalcRisk(e.target.value)}
                    placeholder="1.0"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Entry Price</Label>
                  <Input
                    type="number"
                    step="0.00001"
                    value={calcEntry}
                    onChange={(e) => setCalcEntry(e.target.value)}
                    placeholder="2650.00"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Stop Loss</Label>
                  <Input
                    type="number"
                    step="0.00001"
                    value={calcSl}
                    onChange={(e) => setCalcSl(e.target.value)}
                    placeholder="2640.00"
                  />
                </div>
              </div>
              <Button className="w-full" onClick={handleCalculateSize} disabled={calcLoading}>
                Calculate
              </Button>

              {calcResult && !calcResult.error && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Recommended Volume</span>
                    <span className="text-xl font-bold">{calcResult.recommended_volume} lots</span>
                  </div>
                  <Separator />
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Risk Amount:</span>{" "}
                      {formatCurrency(calcResult.risk_amount || 0)}
                    </div>
                    <div>
                      <span className="text-muted-foreground">Kelly %:</span>{" "}
                      {calcResult.kelly_percentage?.toFixed(2) || "N/A"}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Circuit Breaker & Breach History */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Circuit Breaker Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Circuit Breaker
            </CardTitle>
            <CardDescription>Emergency trading halt mechanism</CardDescription>
          </CardHeader>
          <CardContent>
            {circuitBreaker && !circuitBreaker.error ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="flex items-center gap-3">
                    <div className={`w-4 h-4 rounded-full ${circuitBreaker.active ? "bg-red-500 animate-pulse" : "bg-green-500"}`} />
                    <div>
                      <p className="font-medium">
                        {circuitBreaker.active ? "ACTIVE - Trading Halted" : "Clear - Trading Allowed"}
                      </p>
                      {circuitBreaker.reason && (
                        <p className="text-sm text-muted-foreground">{circuitBreaker.reason}</p>
                      )}
                    </div>
                  </div>
                  <Badge variant={circuitBreaker.active ? "destructive" : "success"}>
                    {circuitBreaker.active ? "TRIPPED" : "OK"}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-muted-foreground">Daily Loss</p>
                    <p className="font-medium">
                      {circuitBreaker.daily_loss_used?.toFixed(1) || 0}% / {circuitBreaker.daily_loss_limit || 5}%
                    </p>
                    <Progress
                      value={(circuitBreaker.daily_loss_used || 0) / (circuitBreaker.daily_loss_limit || 5) * 100}
                      className="mt-2"
                    />
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-muted-foreground">Consecutive Losses</p>
                    <p className="font-medium">
                      {circuitBreaker.consecutive_losses || 0} / {circuitBreaker.max_consecutive_losses || 3}
                    </p>
                    <Progress
                      value={(circuitBreaker.consecutive_losses || 0) / (circuitBreaker.max_consecutive_losses || 3) * 100}
                      className="mt-2"
                    />
                  </div>
                </div>

                {circuitBreaker.in_cooldown && circuitBreaker.cooldown_until && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    <span className="text-sm">Cooldown until: {formatDate(circuitBreaker.cooldown_until)}</span>
                  </div>
                )}

                {circuitBreaker.active && (
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="outline" className="w-full">
                        <RotateCcw className="mr-2 h-4 w-4" />
                        Reset Circuit Breaker
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Reset Circuit Breaker?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will clear the circuit breaker and allow trading to resume.
                          Only do this if you understand why it was triggered and have addressed the issue.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleResetCircuitBreaker} disabled={resetting}>
                          {resetting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                          Reset
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground">
                {circuitBreaker?.error || "Failed to load circuit breaker status"}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Breach History */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Risk Breach History
            </CardTitle>
            <CardDescription>Recent risk limit violations</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px]">
              {breachHistory.length > 0 ? (
                <div className="space-y-2">
                  {breachHistory.map((breach: any, i: number) => (
                    <div key={i} className="flex items-start gap-3 p-3 rounded-lg border hover:bg-muted/50">
                      <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <p className="font-medium text-sm">{breach.type || breach.rule || "Risk Breach"}</p>
                          <Badge variant="destructive" className="text-xs">
                            {breach.severity || "Warning"}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {breach.message || breach.description || "Risk limit exceeded"}
                        </p>
                        {breach.timestamp && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {formatDate(breach.timestamp)}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-[250px] text-muted-foreground">
                  <Shield className="h-12 w-12 mb-3 opacity-50" />
                  <p>No risk breaches recorded</p>
                  <p className="text-sm">Your trading is within safe limits</p>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
