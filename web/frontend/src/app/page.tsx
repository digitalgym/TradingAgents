"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { getDashboard, getRiskMetrics, getPortfolioStatus } from "@/lib/api"
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
} from "lucide-react"
import Link from "next/link"

export default function Dashboard() {
  const [dashboard, setDashboard] = useState<any>(null)
  const [riskMetrics, setRiskMetrics] = useState<any>(null)
  const [portfolioStatus, setPortfolioStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    const [dashRes, riskRes, portRes] = await Promise.all([
      getDashboard(),
      getRiskMetrics(),
      getPortfolioStatus(),
    ])
    if (dashRes.data) setDashboard(dashRes.data)
    if (riskRes.data) setRiskMetrics(riskRes.data)
    if (portRes.data) setPortfolioStatus(portRes.data)
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
    </div>
  )
}
