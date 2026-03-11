"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
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
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { getLearningStatus, getPatterns, updatePatterns, findSimilarTrades } from "@/lib/api"
import { formatDate, formatCurrency } from "@/lib/utils"
import { RefreshCw, Brain, TrendingUp, Layers, Zap, Search, Loader2, History, Target } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"

const SYMBOLS = ["XAUUSD", "XAGUSD", "XPTUSD", "EURUSD", "GBPUSD", "USDJPY"]

export default function LearningPage() {
  const [status, setStatus] = useState<any>(null)
  const [patterns, setPatterns] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [updating, setUpdating] = useState(false)

  // Similar trades search
  const [searchSymbol, setSearchSymbol] = useState("XAUUSD")
  const [searchDirection, setSearchDirection] = useState("BUY")
  const [searchResults, setSearchResults] = useState<any>(null)
  const [searching, setSearching] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    const [statusRes, patternsRes] = await Promise.all([
      getLearningStatus(),
      getPatterns(undefined, 20),
    ])
    if (statusRes.data) setStatus(statusRes.data)
    if (patternsRes.data) setPatterns(patternsRes.data.patterns || [])
    setLoading(false)
  }

  const handleUpdatePatterns = async () => {
    setUpdating(true)
    const { data, error } = await updatePatterns()
    if (error) {
      alert(`Error: ${error}`)
    } else if (data) {
      alert(`Pattern analysis complete. Found ${data.patterns_found} patterns.`)
      fetchData()
    }
    setUpdating(false)
  }

  const handleSearchSimilar = async () => {
    setSearching(true)
    const { data, error } = await findSimilarTrades(searchSymbol, searchDirection)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      setSearchResults(data)
    }
    setSearching(false)
  }

  useEffect(() => {
    fetchData()
  }, [])

  const agentWeights = status?.agent_weights || {}
  const weightData = Object.entries(agentWeights).map(([agent, weight]) => ({
    agent: agent.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase()),
    weight: (weight as number) * 100,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Learning System</h1>
          <p className="text-muted-foreground">
            Monitor agent weights, patterns, and learning progress
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleUpdatePatterns} disabled={updating}>
            {updating ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Brain className="mr-2 h-4 w-4" />
            )}
            Update Patterns
          </Button>
          <Button variant="outline" onClick={fetchData} disabled={loading}>
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Updates</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status?.total_updates || 0}</div>
            <p className="text-xs text-muted-foreground">Learning iterations</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Last Update</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold">
              {status?.last_update ? formatDate(status.last_update) : "Never"}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Patterns Found</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{patterns.length}</div>
            <p className="text-xs text-muted-foreground">Identified trading patterns</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Agent Weights Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Agent Weights
            </CardTitle>
            <CardDescription>
              Current weight distribution across analysis agents
            </CardDescription>
          </CardHeader>
          <CardContent>
            {weightData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={weightData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis type="number" domain={[0, 100]} stroke="#888" />
                  <YAxis dataKey="agent" type="category" width={120} stroke="#888" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                    }}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, "Weight"]}
                  />
                  <Bar dataKey="weight" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No agent weights available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Agent Weights List */}
        <Card>
          <CardHeader>
            <CardTitle>Weight Details</CardTitle>
            <CardDescription>Individual agent performance weights</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px]">
              <div className="space-y-4">
                {Object.entries(agentWeights).map(([agent, weight]) => (
                  <div key={agent} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="capitalize">{agent.replace("_", " ")}</span>
                      <span className="font-medium">
                        {((weight as number) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={(weight as number) * 100} />
                  </div>
                ))}
                {Object.keys(agentWeights).length === 0 && (
                  <p className="text-center text-muted-foreground py-8">
                    No agent weights available
                  </p>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Patterns */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Identified Patterns
          </CardTitle>
          <CardDescription>
            Trading patterns discovered through analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            {patterns.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {patterns.map((pattern, i) => (
                  <div key={i} className="rounded-lg border p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline">{pattern.type || "Pattern"}</Badge>
                      {pattern.win_rate && (
                        <span
                          className={`text-sm font-medium ${
                            pattern.win_rate > 0.5 ? "text-green-500" : "text-red-500"
                          }`}
                        >
                          {(pattern.win_rate * 100).toFixed(0)}% win rate
                        </span>
                      )}
                    </div>
                    {pattern.symbol && (
                      <p className="text-sm">
                        <span className="text-muted-foreground">Symbol:</span>{" "}
                        <span className="font-medium">{pattern.symbol}</span>
                      </p>
                    )}
                    {pattern.description && (
                      <p className="text-sm text-muted-foreground">{pattern.description}</p>
                    )}
                    {pattern.occurrences && (
                      <p className="text-xs text-muted-foreground">
                        {pattern.occurrences} occurrences
                      </p>
                    )}
                    {pattern.conditions && (
                      <div className="flex flex-wrap gap-1">
                        {pattern.conditions.map((cond: string, j: number) => (
                          <Badge key={j} variant="secondary" className="text-xs">
                            {cond}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                No patterns identified yet. Run more analyses to discover patterns.
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Similar Trades Search */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Similar Trades Search
          </CardTitle>
          <CardDescription>
            Find historical trades similar to your current setup for comparison
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4 mb-4">
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Select value={searchSymbol} onValueChange={setSearchSymbol}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SYMBOLS.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Direction</Label>
              <Select value={searchDirection} onValueChange={setSearchDirection}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="BUY">BUY</SelectItem>
                  <SelectItem value="SELL">SELL</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button onClick={handleSearchSimilar} disabled={searching} className="w-full">
                {searching ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Search className="mr-2 h-4 w-4" />
                )}
                Search
              </Button>
            </div>
          </div>

          {searchResults && (
            <div className="space-y-4">
              {/* Stats Summary */}
              <div className="grid grid-cols-4 gap-4 p-4 bg-muted/50 rounded-lg">
                <div>
                  <p className="text-sm text-muted-foreground">Found</p>
                  <p className="text-xl font-bold">{searchResults.stats?.total_found || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className={`text-xl font-bold ${(searchResults.stats?.win_rate || 0) >= 50 ? "text-green-500" : "text-red-500"}`}>
                    {(searchResults.stats?.win_rate || 0).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Wins / Losses</p>
                  <p className="text-xl font-bold">
                    <span className="text-green-500">{searchResults.stats?.wins || 0}</span>
                    {" / "}
                    <span className="text-red-500">{searchResults.stats?.losses || 0}</span>
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg P/L</p>
                  <p className={`text-xl font-bold ${(searchResults.stats?.avg_pnl || 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                    {formatCurrency(searchResults.stats?.avg_pnl || 0)}
                  </p>
                </div>
              </div>

              {/* Results List */}
              <ScrollArea className="h-[300px]">
                {searchResults.similar_trades?.length > 0 ? (
                  <div className="space-y-2">
                    {searchResults.similar_trades.map((trade: any, i: number) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50">
                        <div className="flex items-center gap-3">
                          <Badge variant={trade.signal === "BUY" ? "buy" : "sell"}>
                            {trade.signal}
                          </Badge>
                          <div>
                            <p className="font-medium">{trade.symbol}</p>
                            <p className="text-xs text-muted-foreground">{formatDate(trade.timestamp)}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`font-medium ${trade.was_correct ? "text-green-500" : "text-red-500"}`}>
                            {trade.was_correct ? "WIN" : "LOSS"}
                          </p>
                          <p className={`text-sm ${(trade.pnl || 0) >= 0 ? "text-green-500" : "text-red-500"}`}>
                            {formatCurrency(trade.pnl || 0)}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                    No similar trades found
                  </div>
                )}
              </ScrollArea>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
