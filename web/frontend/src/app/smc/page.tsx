"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { runSmcAnalysis } from "@/lib/api"
import { Loader2, TrendingUp, Target, Layers, AlertTriangle, BarChart3 } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts"

const SYMBOLS = [
  "XAUUSD",
  "XAGUSD",
  "XPTUSD",
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "AUDUSD",
  "USDCAD",
]

const TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]

export default function SmcPage() {
  const [symbol, setSymbol] = useState("XAUUSD")
  const [timeframe, setTimeframe] = useState("H1")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    const { data, error: apiError } = await runSmcAnalysis(symbol, timeframe)
    if (apiError) {
      setError(apiError)
    } else if (data?.error) {
      setError(data.error)
    } else {
      setResult(data)
    }
    setLoading(false)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">SMC Analysis</h1>
        <p className="text-muted-foreground">
          Smart Money Concepts - Order Blocks, FVGs, and Liquidity Zones
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle>Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Select value={symbol} onValueChange={setSymbol}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SYMBOLS.map((s) => (
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
            <Button className="w-full" onClick={handleAnalyze} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Run SMC Analysis"
              )}
            </Button>
            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertTriangle className="h-4 w-4" />
                {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Bias */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Market Bias
            </CardTitle>
          </CardHeader>
          <CardContent>
            {result?.bias ? (
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Badge
                    variant={
                      result.bias === "bullish"
                        ? "buy"
                        : result.bias === "bearish"
                        ? "sell"
                        : "secondary"
                    }
                    className="text-lg px-4 py-2"
                  >
                    {result.bias?.toUpperCase()}
                  </Badge>
                  {result.summary && (
                    <div className="text-sm text-muted-foreground">
                      <span className="text-green-500">{result.summary.bullish_fvgs || 0} Bull FVGs</span>
                      {" / "}
                      <span className="text-red-500">{result.summary.bearish_fvgs || 0} Bear FVGs</span>
                    </div>
                  )}
                </div>
                {result.bias_factors && result.bias_factors.length > 0 && (
                  <div className="space-y-2 pt-2 border-t">
                    <p className="text-sm font-medium">Analysis factors:</p>
                    <ul className="space-y-1">
                      {result.bias_factors.map((factor: string, i: number) => (
                        <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                          <span className="text-primary mt-1">•</span>
                          <span>{factor}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-muted-foreground">Run analysis to see bias</p>
            )}
          </CardContent>
        </Card>

        {/* Key Levels */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Key Levels
            </CardTitle>
          </CardHeader>
          <CardContent>
            {result?.key_levels?.length > 0 ? (
              <div className="space-y-2">
                {result.key_levels.map((level: any, i: number) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <Badge
                        variant={level.type?.includes("Support") || level.type?.includes("Bull") || level.type?.includes("Demand") ? "buy" : "sell"}
                        className="text-xs"
                      >
                        {level.source}
                      </Badge>
                      {level.type}
                    </span>
                    <span className="font-mono">{level.price?.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground">No key levels identified</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Key Levels Chart */}
      {result?.key_levels?.length > 0 && result?.current_price && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Key Levels Chart
            </CardTitle>
            <CardDescription>
              Price levels relative to current price ({result.current_price?.toFixed(2)})
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[...result.key_levels]
                    .sort((a: any, b: any) => b.price - a.price)
                    .map((level: any) => ({
                      name: level.type,
                      price: level.price,
                      source: level.source,
                      distance: level.price - result.current_price,
                      isSupport: level.type?.includes("Support") || level.type?.includes("Bull") || level.type?.includes("Demand"),
                    }))}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                >
                  <XAxis
                    type="number"
                    domain={[
                      (dataMin: number) => Math.floor(dataMin - 10),
                      (dataMax: number) => Math.ceil(dataMax + 10),
                    ]}
                    tickFormatter={(value) => value.toFixed(0)}
                    stroke="#888"
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={90}
                    tick={{ fontSize: 12 }}
                    stroke="#888"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number) => [value.toFixed(2), "Price"]}
                    labelFormatter={(label) => label}
                  />
                  <ReferenceLine
                    x={result.current_price}
                    stroke="#facc15"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{
                      value: `Current: ${result.current_price.toFixed(2)}`,
                      position: "top",
                      fill: "#facc15",
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="price" radius={[0, 4, 4, 0]}>
                    {[...result.key_levels]
                      .sort((a: any, b: any) => b.price - a.price)
                      .map((entry: any, index: number) => {
                        const isSupport =
                          entry.type?.includes("Support") ||
                          entry.type?.includes("Bull") ||
                          entry.type?.includes("Demand")
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
            </div>
          </CardContent>
        </Card>
      )}

      {result && (
        <div className="grid gap-6 md:grid-cols-3">
          {/* Order Blocks */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-blue-500" />
                Order Blocks
              </CardTitle>
              <CardDescription>
                {result.order_blocks?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                {result.order_blocks?.length > 0 ? (
                  <div className="space-y-3">
                    {result.order_blocks.map((ob: any, i: number) => (
                      <div
                        key={i}
                        className="rounded-lg border p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <Badge variant={ob.type === "bullish" ? "buy" : "sell"}>
                            {ob.type}
                          </Badge>
                          {ob.strength && (
                            <span className="text-xs text-muted-foreground">
                              Strength: {ob.strength}
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">High:</span>{" "}
                            {ob.high?.toFixed(5) || ob.top?.toFixed(5)}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Low:</span>{" "}
                            {ob.low?.toFixed(5) || ob.bottom?.toFixed(5)}
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
              </CardTitle>
              <CardDescription>
                {result.fair_value_gaps?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                {result.fair_value_gaps?.length > 0 ? (
                  <div className="space-y-3">
                    {result.fair_value_gaps.map((fvg: any, i: number) => (
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
                            {fvg.high?.toFixed(5) || fvg.top?.toFixed(5)}
                          </div>
                          <div>
                            <span className="text-muted-foreground">Low:</span>{" "}
                            {fvg.low?.toFixed(5) || fvg.bottom?.toFixed(5)}
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
              </CardTitle>
              <CardDescription>
                {result.liquidity_zones?.length || 0} identified
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                {result.liquidity_zones?.length > 0 ? (
                  <div className="space-y-3">
                    {result.liquidity_zones.map((lz: any, i: number) => (
                      <div
                        key={i}
                        className="rounded-lg border p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{lz.type || "Liquidity"}</Badge>
                          {lz.strength && (
                            <span className="text-xs text-muted-foreground">
                              Strength: {lz.strength}
                            </span>
                          )}
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Level:</span>{" "}
                          {lz.price?.toFixed(5) || lz.level?.toFixed(5)}
                        </div>
                        {lz.swept !== undefined && (
                          <Badge variant={lz.swept ? "secondary" : "outline"}>
                            {lz.swept ? "Swept" : "Untested"}
                          </Badge>
                        )}
                      </div>
                    ))}
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
    </div>
  )
}
