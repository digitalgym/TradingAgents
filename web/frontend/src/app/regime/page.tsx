"use client"

import { useEffect, useState } from "react"
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
import { Checkbox } from "@/components/ui/checkbox"
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { getMarketRegime } from "@/lib/api"
import { RefreshCw, TrendingUp, TrendingDown, Minus, Activity, Loader2, BarChart3, HelpCircle } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"

// Helper component for labels with tooltips
function LabelWithHelp({ children, tooltip }: { children: React.ReactNode; tooltip: string }) {
  return (
    <TooltipProvider>
      <UITooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex items-center gap-1 cursor-help text-xs text-muted-foreground">
            {children}
            <HelpCircle className="h-3 w-3" />
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          <p className="text-sm">{tooltip}</p>
        </TooltipContent>
      </UITooltip>
    </TooltipProvider>
  )
}

const SYMBOLS = [
  "XAUUSD",
  "XAGUSD",
  "XPTUSD",
  "COPPER-C",
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "AUDUSD",
]

const TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]

interface RegimeData {
  symbol: string
  regime: string
  volatility: string
  adx: number
  atr_percentile: number
  trend_strength: number
  details?: any
}

export default function RegimePage() {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(["XAUUSD", "XAGUSD", "XPTUSD"])
  const [timeframe, setTimeframe] = useState("H1")
  const [regimes, setRegimes] = useState<RegimeData[]>([])
  const [loading, setLoading] = useState(false)

  const fetchRegimes = async () => {
    setLoading(true)
    const results: RegimeData[] = []

    for (const symbol of selectedSymbols) {
      const { data, error } = await getMarketRegime(symbol, timeframe)
      if (data && !data.error) {
        results.push({
          symbol,
          regime: data.regime || data.market_regime || "unknown",
          volatility: data.volatility || data.volatility_regime || "normal",
          adx: data.adx || data.trend_strength || 0,
          atr_percentile: data.atr_percentile || 50,
          trend_strength: data.trend_strength || data.adx || 0,
          details: data,
        })
      } else {
        results.push({
          symbol,
          regime: "error",
          volatility: "unknown",
          adx: 0,
          atr_percentile: 0,
          trend_strength: 0,
        })
      }
    }

    setRegimes(results)
    setLoading(false)
  }

  useEffect(() => {
    if (selectedSymbols.length > 0) {
      fetchRegimes()
    }
  }, [])

  const toggleSymbol = (symbol: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    )
  }

  const getRegimeBadge = (regime: string) => {
    const r = regime?.toLowerCase() || ""
    if (r.includes("bullish") || r.includes("uptrend")) {
      return { variant: "buy" as const, icon: TrendingUp, label: regime }
    }
    if (r.includes("bearish") || r.includes("downtrend")) {
      return { variant: "sell" as const, icon: TrendingDown, label: regime }
    }
    if (r.includes("rang") || r.includes("sideway") || r.includes("consolidat")) {
      return { variant: "secondary" as const, icon: Minus, label: regime }
    }
    return { variant: "outline" as const, icon: Activity, label: regime }
  }

  const getVolatilityColor = (vol: string) => {
    const v = vol?.toLowerCase() || ""
    if (v.includes("high") || v.includes("extreme")) return "text-red-500"
    if (v.includes("low") || v.includes("calm")) return "text-green-500"
    return "text-yellow-500"
  }

  const chartData = regimes.map((r) => ({
    symbol: r.symbol,
    adx: r.adx,
    atr: r.atr_percentile,
    regime: r.regime,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Market Regime</h1>
          <p className="text-muted-foreground">
            Analyze market conditions across multiple symbols
          </p>
        </div>
        <Button variant="outline" onClick={fetchRegimes} disabled={loading || selectedSymbols.length === 0}>
          {loading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-4 w-4" />
          )}
          Analyze
        </Button>
      </div>

      {/* Symbol Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Symbol Selection
            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger>
                  <HelpCircle className="h-4 w-4 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="text-sm">Select the trading instruments you want to analyze. Comparing multiple symbols helps identify which markets have the strongest trends and best trading conditions.</p>
                </TooltipContent>
              </UITooltip>
            </TooltipProvider>
          </CardTitle>
          <CardDescription>Select symbols to analyze</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4 mb-4">
            {SYMBOLS.map((symbol) => (
              <div key={symbol} className="flex items-center space-x-2">
                <Checkbox
                  id={symbol}
                  checked={selectedSymbols.includes(symbol)}
                  onCheckedChange={() => toggleSymbol(symbol)}
                />
                <label htmlFor={symbol} className="text-sm font-medium cursor-pointer">
                  {symbol}
                </label>
              </div>
            ))}
          </div>
          <div className="flex gap-4 items-end">
            <div className="space-y-2">
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <Label className="inline-flex items-center gap-1 cursor-help">
                      Timeframe
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </Label>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">The chart timeframe to analyze. Lower timeframes (M15, M30) show short-term trends, while higher timeframes (H4, D1) reveal the bigger picture and are more reliable for trend direction.</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIMEFRAMES.map((tf) => (
                    <SelectItem key={tf} value={tf}>{tf}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={fetchRegimes} disabled={loading || selectedSymbols.length === 0}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Analyze {selectedSymbols.length} Symbol{selectedSymbols.length !== 1 ? "s" : ""}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Regime Comparison Chart */}
      {regimes.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Trend Strength Comparison
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger>
                    <HelpCircle className="h-4 w-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">This chart compares the ADX (trend strength) across all selected symbols. Green bars indicate bullish trends, red bars indicate bearish trends, and yellow bars indicate ranging markets. Longer bars mean stronger trends.</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
            </CardTitle>
            <CardDescription>ADX values across symbols (higher = stronger trend)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis type="number" domain={[0, 100]} stroke="#888" />
                  <YAxis dataKey="symbol" type="category" width={80} stroke="#888" />
                  <ChartTooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number, name: string) => [value.toFixed(1), name === "adx" ? "ADX" : "ATR %"]}
                  />
                  <Bar dataKey="adx" name="ADX" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => {
                      const r = entry.regime?.toLowerCase() || ""
                      let color = "#6b7280" // gray
                      if (r.includes("bullish") || r.includes("uptrend")) color = "#22c55e"
                      else if (r.includes("bearish") || r.includes("downtrend")) color = "#ef4444"
                      else if (r.includes("rang")) color = "#eab308"
                      return <Cell key={`cell-${index}`} fill={color} />
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-6 mt-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-green-500" />
                <span className="text-muted-foreground">Bullish</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-red-500" />
                <span className="text-muted-foreground">Bearish</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-yellow-500" />
                <span className="text-muted-foreground">Ranging</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-gray-500" />
                <span className="text-muted-foreground">Unknown</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Regime Details Grid */}
      {regimes.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {regimes.map((r) => {
            const badge = getRegimeBadge(r.regime)
            const Icon = badge.icon
            return (
              <Card key={r.symbol}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle>{r.symbol}</CardTitle>
                    <Badge variant={badge.variant} className="flex items-center gap-1">
                      <Icon className="h-3 w-3" />
                      {badge.label}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <LabelWithHelp tooltip="Measures how much price is fluctuating. High volatility means larger price swings and higher risk/reward. Low volatility means calmer markets with smaller moves.">
                          Volatility
                        </LabelWithHelp>
                        <p className={`font-medium ${getVolatilityColor(r.volatility)}`}>
                          {r.volatility || "Unknown"}
                        </p>
                      </div>
                      <div>
                        <LabelWithHelp tooltip="Average Directional Index (ADX) measures trend strength from 0-100. It doesn't indicate direction, only how strong the current trend is. Higher values = stronger trend.">
                          ADX
                        </LabelWithHelp>
                        <p className="font-medium">{r.adx?.toFixed(1) || "N/A"}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <LabelWithHelp tooltip="Shows current ATR (Average True Range) relative to historical levels. Above 80% = unusually high volatility, below 20% = unusually calm. Helps gauge if volatility is extreme.">
                          ATR Percentile
                        </LabelWithHelp>
                        <p className="font-medium">{r.atr_percentile?.toFixed(0) || "N/A"}%</p>
                      </div>
                      <div>
                        <LabelWithHelp tooltip="Human-readable interpretation of ADX. Weak = avoid trend strategies, Developing = trend may be starting, Strong/Very Strong = ideal for trend-following trades.">
                          Trend Strength
                        </LabelWithHelp>
                        <p className="font-medium">
                          {r.adx < 20 ? "Weak" : r.adx < 40 ? "Developing" : r.adx < 60 ? "Strong" : "Very Strong"}
                        </p>
                      </div>
                    </div>
                    {r.details?.bias_factors && r.details.bias_factors.length > 0 && (
                      <div className="pt-2 border-t">
                        <LabelWithHelp tooltip="The main technical and structural factors driving the current market bias. These are the key reasons why the market is trending or ranging.">
                          Key Factors
                        </LabelWithHelp>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {r.details.bias_factors.slice(0, 3).map((factor: string, i: number) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {factor.length > 30 ? factor.slice(0, 30) + "..." : factor}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {/* Regime Legend */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Regime Interpretation Guide
            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger>
                  <HelpCircle className="h-4 w-4 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="text-sm">Use this guide to understand what each regime means for your trading. Match your trading strategy to the current market regime for better results.</p>
                </TooltipContent>
              </UITooltip>
            </TooltipProvider>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <h4 className="font-medium inline-flex items-center gap-1 cursor-help">
                      Trend Regimes
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </h4>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">Market regime describes the overall directional bias. Trading WITH the regime (buying in bullish, selling in bearish) typically has higher success rates than fighting against it.</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <Badge variant="buy">Bullish Trending</Badge>
                  <span className="text-muted-foreground">Strong upward momentum, favor long positions</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="sell">Bearish Trending</Badge>
                  <span className="text-muted-foreground">Strong downward momentum, favor short positions</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">Ranging</Badge>
                  <span className="text-muted-foreground">Sideways movement, range trading strategies</span>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <h4 className="font-medium inline-flex items-center gap-1 cursor-help">
                      ADX Interpretation
                      <HelpCircle className="h-3 w-3 text-muted-foreground" />
                    </h4>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">ADX (Average Directional Index) measures trend STRENGTH, not direction. A high ADX in a bullish market means strong uptrend. A high ADX in a bearish market means strong downtrend. Low ADX = choppy, ranging markets.</p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>0-20</span>
                  <span className="text-muted-foreground">Weak/No trend - avoid trend strategies</span>
                </div>
                <div className="flex justify-between">
                  <span>20-40</span>
                  <span className="text-muted-foreground">Developing trend - be cautious</span>
                </div>
                <div className="flex justify-between">
                  <span>40-60</span>
                  <span className="text-muted-foreground">Strong trend - ideal for trend trading</span>
                </div>
                <div className="flex justify-between">
                  <span>60+</span>
                  <span className="text-muted-foreground">Very strong trend - may be extended</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
