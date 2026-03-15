"use client"

import { useEffect, useState, useMemo, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Loader2, RefreshCw, TrendingUp, TrendingDown } from "lucide-react"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { getChartCandles, runSmcAnalysis } from "@/lib/api"

interface Candle {
  time: string
  open: number
  high: number
  low: number
  close: number
}

interface Zone {
  top: number
  bottom: number
  type: "bullish" | "bearish"
  source: "OB" | "FVG"
  mitigated?: boolean
}

interface LiquidityZone {
  price: number
  type: "buy-side" | "sell-side"
  strength?: number
  touched?: boolean
}

interface EqualLevel {
  type: "eqh" | "eql"  // Equal High or Equal Low
  price: number
  touches: number
  strength?: number
  swept?: boolean
}

interface BreakerBlock {
  type: "bullish" | "bearish"
  top: number
  bottom: number
  original_ob_type?: string
  break_price?: number
  strength?: number
  mitigated?: boolean
}

interface OTEZone {
  type: "bullish" | "bearish"
  fib_618: number
  fib_705: number
  fib_79: number
  swing_high?: number
  swing_low?: number
}

interface StructureBreak {
  type: "high" | "low"
  price: number
  break_type: "BOS" | "CHOC"
  break_index?: number
  timestamp?: string
}

interface PremiumDiscountZone {
  current_zone: "premium" | "discount" | "equilibrium"
  equilibrium: number
  range_high: number
  range_low: number
  position_percent: number
}

interface TradeDecision {
  signal: "BUY" | "SELL" | "HOLD"
  confidence?: number
  entry_price?: number | null
  stop_loss?: number | null
  take_profit?: number | null
  rationale?: string
  setup_type?: string
}

interface PriceChartProps {
  symbol: string
  direction?: "BUY" | "SELL"
  entryPrice?: number
  currentPrice: number
  stopLoss?: number
  takeProfit?: number
  orderBlocks?: any[]
  fairValueGaps?: any[]
  liquidityZones?: LiquidityZone[]
  breakerBlocks?: BreakerBlock[]
  equalLevels?: EqualLevel[]
  oteZones?: OTEZone[]
  structureBreaks?: StructureBreak[]
  premiumDiscount?: PremiumDiscountZone | null
  pdh?: number  // Previous Day High
  pdl?: number  // Previous Day Low
  atrValue?: number
  digits?: number
  tradeDecision?: TradeDecision | null
}

// Calculate RSI from close prices
function calculateRSI(closes: number[], period: number = 14): number[] {
  const rsi: number[] = []
  if (closes.length < period + 1) return closes.map(() => 50) // Not enough data

  // Calculate price changes
  const changes: number[] = []
  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1])
  }

  // Initial average gain/loss
  let avgGain = 0
  let avgLoss = 0
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i]
    else avgLoss += Math.abs(changes[i])
  }
  avgGain /= period
  avgLoss /= period

  // First RSI value (pad with nulls for earlier candles)
  for (let i = 0; i < period; i++) {
    rsi.push(50) // Not enough data yet
  }

  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
  rsi.push(100 - 100 / (1 + rs))

  // Subsequent values using smoothed averages
  for (let i = period; i < changes.length; i++) {
    const gain = changes[i] > 0 ? changes[i] : 0
    const loss = changes[i] < 0 ? Math.abs(changes[i]) : 0
    avgGain = (avgGain * (period - 1) + gain) / period
    avgLoss = (avgLoss * (period - 1) + loss) / period
    const currentRs = avgLoss === 0 ? 100 : avgGain / avgLoss
    rsi.push(100 - 100 / (1 + currentRs))
  }

  return rsi
}

export function PriceChart({
  symbol,
  direction = "BUY",
  entryPrice,
  currentPrice,
  stopLoss,
  takeProfit,
  orderBlocks = [],
  fairValueGaps = [],
  liquidityZones = [],
  breakerBlocks = [],
  equalLevels = [],
  oteZones = [],
  structureBreaks = [],
  premiumDiscount,
  pdh,
  pdl,
  atrValue,
  digits = 5,
  tradeDecision,
}: PriceChartProps) {
  const [candles, setCandles] = useState<Candle[]>([])
  const [loading, setLoading] = useState(true)
  const [timeframe, setTimeframe] = useState("H1")
  const [bars, setBars] = useState(50)
  const [chartWidth, setChartWidth] = useState(800)
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-fetched SMC data (used when props are empty)
  const [autoSmcData, setAutoSmcData] = useState<any>(null)

  const fetchCandles = async () => {
    setLoading(true)
    const { data } = await getChartCandles(symbol, timeframe, bars)
    if (data?.candles) {
      setCandles(data.candles)
    }
    setLoading(false)
  }

  // Auto-fetch SMC data when OB/FVG not provided as props
  useEffect(() => {
    const needsSmcFetch = orderBlocks.length === 0 && fairValueGaps.length === 0
    if (!needsSmcFetch || !symbol) return

    const fetchSmc = async () => {
      const { data } = await runSmcAnalysis(symbol, timeframe, { lookback: bars })
      if (data) {
        setAutoSmcData(data)
      }
    }
    fetchSmc()
  }, [symbol, timeframe, bars, orderBlocks.length, fairValueGaps.length])

  // Merge props with auto-fetched SMC data
  const effectiveOrderBlocks = orderBlocks.length > 0 ? orderBlocks : (autoSmcData?.order_blocks || [])
  const effectiveFairValueGaps = fairValueGaps.length > 0 ? fairValueGaps : (autoSmcData?.fair_value_gaps || [])

  useEffect(() => {
    fetchCandles()
  }, [symbol, timeframe, bars])

  // Calculate RSI values from candle closes
  const rsiValues = useMemo(() => {
    if (candles.length === 0) return []
    const closes = candles.map(c => c.close)
    return calculateRSI(closes)
  }, [candles])

  // Calculate price range for the chart
  const { minPrice, maxPrice, priceRange } = useMemo(() => {
    if (candles.length === 0) return { minPrice: 0, maxPrice: 0, priceRange: 0 }

    let min = Math.min(...candles.map(c => c.low))
    let max = Math.max(...candles.map(c => c.high))

    // Include position levels in range
    if (entryPrice) {
      min = Math.min(min, entryPrice)
      max = Math.max(max, entryPrice)
    }
    if (stopLoss) {
      min = Math.min(min, stopLoss)
      max = Math.max(max, stopLoss)
    }
    if (takeProfit) {
      min = Math.min(min, takeProfit)
      max = Math.max(max, takeProfit)
    }

    // Include liquidity zones in range
    liquidityZones.forEach(lz => {
      min = Math.min(min, lz.price)
      max = Math.max(max, lz.price)
    })

    // Include equal levels in range
    equalLevels.forEach(el => {
      min = Math.min(min, el.price)
      max = Math.max(max, el.price)
    })

    // Include breaker blocks in range
    breakerBlocks.forEach(bb => {
      if (!bb.mitigated) {
        min = Math.min(min, bb.bottom)
        max = Math.max(max, bb.top)
      }
    })

    // Include OTE zones in range
    oteZones.forEach(ote => {
      min = Math.min(min, ote.fib_79)
      max = Math.max(max, ote.fib_618)
    })

    // Include premium/discount zone in range
    if (premiumDiscount) {
      min = Math.min(min, premiumDiscount.range_low)
      max = Math.max(max, premiumDiscount.range_high)
    }

    // Include structure breaks in range
    structureBreaks.forEach(sb => {
      min = Math.min(min, sb.price)
      max = Math.max(max, sb.price)
    })

    // Add 5% padding
    const range = max - min
    const padding = range * 0.05
    return {
      minPrice: min - padding,
      maxPrice: max + padding,
      priceRange: range + (padding * 2),
    }
  }, [candles, entryPrice, stopLoss, takeProfit, liquidityZones, equalLevels, breakerBlocks, oteZones, premiumDiscount, structureBreaks])

  // Convert price to Y coordinate
  const priceToY = (price: number, height: number) => {
    if (priceRange === 0) return height / 2
    return height - ((price - minPrice) / priceRange) * height
  }

  // Prepare zones from OBs and FVGs (using effective data that includes auto-fetched)
  const zones: Zone[] = useMemo(() => {
    const result: Zone[] = []

    effectiveOrderBlocks?.forEach((ob: any) => {
      if (ob.mitigated) return // Skip mitigated
      result.push({
        top: ob.high || ob.top,
        bottom: ob.low || ob.bottom,
        type: ob.type,
        source: "OB",
        mitigated: ob.mitigated,
      })
    })

    effectiveFairValueGaps?.forEach((fvg: any) => {
      if (fvg.mitigated) return // Skip mitigated
      result.push({
        top: fvg.high || fvg.top,
        bottom: fvg.low || fvg.bottom,
        type: fvg.type,
        source: "FVG",
        mitigated: fvg.mitigated,
      })
    })

    return result
  }, [effectiveOrderBlocks, effectiveFairValueGaps])

  const chartHeight = 400
  const leftPadding = 60
  const rightPadding = 20
  const candleWidth = Math.max(4, (chartWidth - leftPadding - rightPadding) / candles.length - 2)

  // Resize observer for responsive width
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width
        if (width > 0) {
          setChartWidth(width)
        }
      }
    })

    resizeObserver.observe(container)
    // Set initial width
    setChartWidth(container.clientWidth || 800)

    return () => resizeObserver.disconnect()
  }, [])

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Price Chart
            <HelpTooltip content="Candlestick chart showing price action with your position levels (entry, SL, TP) and SMC zones (Order Blocks, FVGs) overlaid." />
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger className="w-20 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="M15">M15</SelectItem>
                <SelectItem value="M30">M30</SelectItem>
                <SelectItem value="H1">H1</SelectItem>
                <SelectItem value="H4">H4</SelectItem>
                <SelectItem value="D1">D1</SelectItem>
              </SelectContent>
            </Select>
            <Select value={bars.toString()} onValueChange={(v) => setBars(parseInt(v))}>
              <SelectTrigger className="w-20 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="30">30</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={fetchCandles} disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center h-[400px]">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : candles.length === 0 ? (
          <div className="flex items-center justify-center h-[400px] text-muted-foreground">
            No candle data available
          </div>
        ) : (
          <div ref={containerRef} className="relative w-full">
            <svg
              width={chartWidth}
              height={chartHeight}
              className="bg-background w-full"
            >
              {/* Premium/Discount Background Zone */}
              {premiumDiscount && premiumDiscount.range_high > 0 && (
                <g>
                  {/* Premium zone (above equilibrium) - orange tint */}
                  <rect
                    x={leftPadding}
                    y={priceToY(premiumDiscount.range_high, chartHeight)}
                    width={chartWidth - leftPadding - rightPadding}
                    height={Math.abs(priceToY(premiumDiscount.equilibrium, chartHeight) - priceToY(premiumDiscount.range_high, chartHeight))}
                    fill="rgba(251, 191, 36, 0.08)"
                  />
                  {/* Discount zone (below equilibrium) - blue tint */}
                  <rect
                    x={leftPadding}
                    y={priceToY(premiumDiscount.equilibrium, chartHeight)}
                    width={chartWidth - leftPadding - rightPadding}
                    height={Math.abs(priceToY(premiumDiscount.range_low, chartHeight) - priceToY(premiumDiscount.equilibrium, chartHeight))}
                    fill="rgba(59, 130, 246, 0.08)"
                  />
                  {/* Equilibrium line */}
                  <line
                    x1={leftPadding}
                    y1={priceToY(premiumDiscount.equilibrium, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(premiumDiscount.equilibrium, chartHeight)}
                    stroke="rgba(255, 255, 255, 0.3)"
                    strokeWidth={1}
                    strokeDasharray="6 4"
                  />
                  {/* Labels */}
                  <text
                    x={leftPadding + 5}
                    y={priceToY(premiumDiscount.range_high, chartHeight) + 14}
                    fill="#fbbf24"
                    fontSize={11}
                    fontWeight="bold"
                    opacity={0.6}
                  >
                    Premium
                  </text>
                  <text
                    x={leftPadding + 5}
                    y={priceToY(premiumDiscount.range_low, chartHeight) - 6}
                    fill="#3b82f6"
                    fontSize={11}
                    fontWeight="bold"
                    opacity={0.6}
                  >
                    Discount
                  </text>
                </g>
              )}

              {/* Background grid lines */}
              {[0.25, 0.5, 0.75].map((pct) => (
                <line
                  key={pct}
                  x1={leftPadding}
                  y1={chartHeight * pct}
                  x2={chartWidth - rightPadding}
                  y2={chartHeight * pct}
                  stroke="hsl(var(--border))"
                  strokeDasharray="4 4"
                  opacity={0.3}
                />
              ))}

              {/* SMC Zones (OB/FVG rectangles) */}
              {zones.map((zone, idx) => {
                const y1 = priceToY(zone.top, chartHeight)
                const y2 = priceToY(zone.bottom, chartHeight)
                const height = Math.abs(y2 - y1)
                const isBullish = zone.type === "bullish"
                const isOB = zone.source === "OB"

                return (
                  <g key={`zone-${idx}`}>
                    <rect
                      x={leftPadding}
                      y={Math.min(y1, y2)}
                      width={chartWidth - leftPadding - rightPadding}
                      height={height}
                      fill={isBullish ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)"}
                      stroke={isBullish ? "rgba(34, 197, 94, 0.4)" : "rgba(239, 68, 68, 0.4)"}
                      strokeWidth={isOB ? 2 : 1}
                      strokeDasharray={isOB ? "none" : "4 4"}
                    />
                    {/* Left label: zone type */}
                    <text
                      x={leftPadding + 5}
                      y={Math.min(y1, y2) + 12}
                      fill={isBullish ? "#22c55e" : "#ef4444"}
                      fontSize={10}
                      opacity={0.8}
                    >
                      {zone.source} {isBullish ? "Bull" : "Bear"}{zone.mitigated ? " (mitigated)" : ""}
                    </text>
                    {/* Right label: Un-Mitigated status for active zones */}
                    {!zone.mitigated && (
                      <text
                        x={chartWidth - rightPadding - 5}
                        y={Math.min(y1, y2) + 12}
                        fill={isBullish ? "#22c55e" : "#ef4444"}
                        fontSize={10}
                        fontWeight="bold"
                        textAnchor="end"
                        opacity={0.9}
                      >
                        Un-Mitigated {isBullish ? "Support" : "Resistance"}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* Breaker Blocks - Failed OBs that flipped polarity */}
              {breakerBlocks.filter(bb => !bb.mitigated).map((bb, idx) => {
                const y1 = priceToY(bb.top, chartHeight)
                const y2 = priceToY(bb.bottom, chartHeight)
                const height = Math.abs(y2 - y1)
                const isBullish = bb.type === "bullish"
                // Breakers use cyan/orange to distinguish from regular OBs
                const fillColor = isBullish ? "rgba(6, 182, 212, 0.2)" : "rgba(249, 115, 22, 0.2)"
                const strokeColor = isBullish ? "rgba(6, 182, 212, 0.6)" : "rgba(249, 115, 22, 0.6)"

                return (
                  <g key={`breaker-${idx}`}>
                    <rect
                      x={leftPadding}
                      y={Math.min(y1, y2)}
                      width={chartWidth - leftPadding - rightPadding}
                      height={height}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={2}
                      strokeDasharray="6 3"
                    />
                    <text
                      x={leftPadding + 5}
                      y={Math.min(y1, y2) + 12}
                      fill={isBullish ? "#06b6d4" : "#f97316"}
                      fontSize={10}
                      opacity={0.9}
                    >
                      BRK {isBullish ? "Bull" : "Bear"}{bb.strength ? ` ${Math.round(bb.strength)}%` : ""}
                    </text>
                  </g>
                )
              })}

              {/* OTE Zones - Optimal Trade Entry (Fib 0.618-0.79) */}
              {oteZones.map((ote, idx) => {
                const y618 = priceToY(ote.fib_618, chartHeight)
                const y79 = priceToY(ote.fib_79, chartHeight)
                const height = Math.abs(y79 - y618)
                const isBullish = ote.type === "bullish"
                // OTE zones use blue tones
                const fillColor = isBullish ? "rgba(59, 130, 246, 0.12)" : "rgba(239, 68, 68, 0.12)"
                const strokeColor = isBullish ? "rgba(59, 130, 246, 0.4)" : "rgba(239, 68, 68, 0.4)"

                return (
                  <g key={`ote-${idx}`}>
                    <rect
                      x={leftPadding}
                      y={Math.min(y618, y79)}
                      width={chartWidth - leftPadding - rightPadding}
                      height={height}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={1}
                    />
                    {/* 0.705 midline */}
                    <line
                      x1={leftPadding}
                      y1={priceToY(ote.fib_705, chartHeight)}
                      x2={chartWidth - rightPadding}
                      y2={priceToY(ote.fib_705, chartHeight)}
                      stroke={strokeColor}
                      strokeWidth={1}
                      strokeDasharray="4 4"
                    />
                    <text
                      x={chartWidth - rightPadding - 5}
                      y={Math.min(y618, y79) + 12}
                      fill={isBullish ? "#3b82f6" : "#ef4444"}
                      fontSize={9}
                      textAnchor="end"
                      opacity={0.8}
                    >
                      OTE {isBullish ? "Bull" : "Bear"}
                    </text>
                  </g>
                )
              })}

              {/* Equal Levels - Equal Highs/Lows (liquidity magnets) */}
              {equalLevels.map((el, idx) => {
                const y = priceToY(el.price, chartHeight)
                const isHigh = el.type === "eqh"
                const isSwept = el.swept
                // Equal highs = teal, Equal lows = amber
                const color = isSwept
                  ? "#6b7280"
                  : isHigh ? "#14b8a6" : "#f59e0b"
                const opacity = isSwept ? 0.5 : 0.9

                return (
                  <g key={`eq-${idx}`}>
                    <line
                      x1={leftPadding}
                      y1={y}
                      x2={chartWidth - rightPadding}
                      y2={y}
                      stroke={color}
                      strokeWidth={isSwept ? 1 : 2}
                      strokeDasharray="2 4"
                      opacity={opacity}
                    />
                    {/* = markers to show equal level */}
                    {!isSwept && [0.25, 0.5, 0.75].map((pct) => (
                      <text
                        key={pct}
                        x={leftPadding + (chartWidth - leftPadding - rightPadding) * pct}
                        y={y + 4}
                        fill={color}
                        fontSize={10}
                        textAnchor="middle"
                        opacity={opacity}
                      >
                        =
                      </text>
                    ))}
                    <text
                      x={chartWidth - rightPadding - 5}
                      y={y - 5}
                      fill={color}
                      fontSize={9}
                      textAnchor="end"
                      fontWeight={el.touches >= 3 ? "bold" : "normal"}
                      opacity={opacity}
                    >
                      {isSwept
                        ? `${isHigh ? "EQH" : "EQL"} x${el.touches} (swept)`
                        : el.touches >= 3
                          ? `Proven ${isHigh ? "Resistance" : "Support"}, Retests=${el.touches}`
                          : `${isHigh ? "EQH" : "EQL"} x${el.touches}`
                      }
                    </text>
                  </g>
                )
              })}

              {/* ATR bands if available */}
              {atrValue && currentPrice && (
                <>
                  <line
                    x1={leftPadding}
                    y1={priceToY(currentPrice + atrValue, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(currentPrice + atrValue, chartHeight)}
                    stroke="#f59e0b"
                    strokeDasharray="8 4"
                    opacity={0.4}
                  />
                  <line
                    x1={leftPadding}
                    y1={priceToY(currentPrice - atrValue, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(currentPrice - atrValue, chartHeight)}
                    stroke="#f59e0b"
                    strokeDasharray="8 4"
                    opacity={0.4}
                  />
                </>
              )}

              {/* Liquidity Zones */}
              {liquidityZones.map((lz, idx) => {
                const y = priceToY(lz.price, chartHeight)
                const isBuySide = lz.type === "buy-side"
                const isSwept = lz.touched
                // Unswept: bright colors (purple/pink), Swept: muted gray tones
                const color = isSwept
                  ? "#6b7280" // Gray for swept (liquidity taken)
                  : isBuySide ? "#8b5cf6" : "#ec4899" // Purple for BSL, pink for SSL
                const opacity = isSwept ? 0.5 : 0.9

                return (
                  <g key={`liq-${idx}`}>
                    {/* Liquidity line - solid for unswept, dashed for swept */}
                    <line
                      x1={leftPadding}
                      y1={y}
                      x2={chartWidth - rightPadding}
                      y2={y}
                      stroke={color}
                      strokeWidth={isSwept ? 1.5 : 2}
                      strokeDasharray={isSwept ? "6 4" : "3 3"}
                      opacity={opacity}
                    />
                    {/* $ markers to indicate stop loss clusters (only for unswept) */}
                    {!isSwept && [0.2, 0.4, 0.6, 0.8].map((pct) => (
                      <text
                        key={pct}
                        x={leftPadding + (chartWidth - leftPadding - rightPadding) * pct}
                        y={y + 4}
                        fill={color}
                        fontSize={10}
                        textAnchor="middle"
                        opacity={opacity}
                      >
                        $
                      </text>
                    ))}
                    {/* X markers for swept zones to indicate liquidity taken */}
                    {isSwept && [0.3, 0.5, 0.7].map((pct) => (
                      <text
                        key={pct}
                        x={leftPadding + (chartWidth - leftPadding - rightPadding) * pct}
                        y={y + 4}
                        fill={color}
                        fontSize={9}
                        textAnchor="middle"
                        opacity={opacity}
                      >
                        ✗
                      </text>
                    ))}
                    {/* Label with swept/unswept status */}
                    <text
                      x={chartWidth - rightPadding - 5}
                      y={y - 5}
                      fill={color}
                      fontSize={9}
                      textAnchor="end"
                      opacity={opacity}
                    >
                      {isBuySide ? "BSL" : "SSL"}{lz.strength ? ` ${Math.round(lz.strength)}%` : ""} {isSwept ? "(swept)" : ""}
                    </text>
                  </g>
                )
              })}

              {/* PDH - Previous Day High */}
              {pdh && (
                <g>
                  <line
                    x1={leftPadding}
                    y1={priceToY(pdh, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(pdh, chartHeight)}
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="8 4"
                    opacity={0.8}
                  />
                  <text
                    x={leftPadding + 5}
                    y={priceToY(pdh, chartHeight) - 5}
                    fill="#f59e0b"
                    fontSize={10}
                    fontWeight="bold"
                  >
                    PDH
                  </text>
                  <text
                    x={chartWidth - rightPadding - 5}
                    y={priceToY(pdh, chartHeight) - 5}
                    fill="#f59e0b"
                    fontSize={9}
                    textAnchor="end"
                  >
                    {pdh.toFixed(digits)}
                  </text>
                </g>
              )}

              {/* PDL - Previous Day Low */}
              {pdl && (
                <g>
                  <line
                    x1={leftPadding}
                    y1={priceToY(pdl, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(pdl, chartHeight)}
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="8 4"
                    opacity={0.8}
                  />
                  <text
                    x={leftPadding + 5}
                    y={priceToY(pdl, chartHeight) + 12}
                    fill="#f59e0b"
                    fontSize={10}
                    fontWeight="bold"
                  >
                    PDL
                  </text>
                  <text
                    x={chartWidth - rightPadding - 5}
                    y={priceToY(pdl, chartHeight) + 12}
                    fill="#f59e0b"
                    fontSize={9}
                    textAnchor="end"
                  >
                    {pdl.toFixed(digits)}
                  </text>
                </g>
              )}

              {/* Candlesticks */}
              {candles.map((candle, idx) => {
                const x = leftPadding + idx * (candleWidth + 2) + candleWidth / 2
                const isBullish = candle.close >= candle.open
                const bodyTop = priceToY(Math.max(candle.open, candle.close), chartHeight)
                const bodyBottom = priceToY(Math.min(candle.open, candle.close), chartHeight)
                const bodyHeight = Math.max(1, bodyBottom - bodyTop)
                const wickTop = priceToY(candle.high, chartHeight)
                const wickBottom = priceToY(candle.low, chartHeight)

                return (
                  <g key={idx}>
                    {/* Wick */}
                    <line
                      x1={x}
                      y1={wickTop}
                      x2={x}
                      y2={wickBottom}
                      stroke={isBullish ? "#22c55e" : "#ef4444"}
                      strokeWidth={1}
                    />
                    {/* Body */}
                    <rect
                      x={x - candleWidth / 2}
                      y={bodyTop}
                      width={candleWidth}
                      height={bodyHeight}
                      fill={isBullish ? "#22c55e" : "#ef4444"}
                      stroke={isBullish ? "#22c55e" : "#ef4444"}
                    />
                  </g>
                )
              })}

              {/* BOS/CHoCH Structure Labels */}
              {structureBreaks.map((sb, idx) => {
                const y = priceToY(sb.price, chartHeight)
                const isBOS = sb.break_type === "BOS"
                const isLow = sb.type === "low"
                // BOS = trend continuation (yellow), CHoCH = reversal (cyan)
                const color = isBOS ? "#facc15" : "#22d3ee"
                const label = isBOS ? "BOS" : "CHoCH"

                return (
                  <g key={`struct-${idx}`}>
                    {/* Short line at break level */}
                    <line
                      x1={chartWidth - rightPadding - 120}
                      y1={y}
                      x2={chartWidth - rightPadding - 5}
                      y2={y}
                      stroke={color}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      opacity={0.7}
                    />
                    {/* Label above or below the line */}
                    <text
                      x={chartWidth - rightPadding - 62}
                      y={isLow ? y + 13 : y - 5}
                      fill={color}
                      fontSize={10}
                      fontWeight="bold"
                      textAnchor="middle"
                      opacity={0.9}
                    >
                      {label}
                    </text>
                    {/* Small diamond marker */}
                    <polygon
                      points={`${chartWidth - rightPadding - 125},${y} ${chartWidth - rightPadding - 121},${y - 4} ${chartWidth - rightPadding - 117},${y} ${chartWidth - rightPadding - 121},${y + 4}`}
                      fill={color}
                      opacity={0.8}
                    />
                  </g>
                )
              })}

              {/* Trade Decision Overlay */}
              {(() => {
                const decEntry = tradeDecision?.entry_price || entryPrice
                const decSl = tradeDecision?.stop_loss || stopLoss
                const decTp = tradeDecision?.take_profit || takeProfit
                const decSignal = tradeDecision?.signal || direction
                const isBuy = decSignal === "BUY"
                const hasDecision = tradeDecision && tradeDecision.signal !== "HOLD"

                // Determine if entry is away from current price (retrace needed)
                const entryDiff = decEntry && currentPrice ? Math.abs(decEntry - currentPrice) / currentPrice : 0
                const needsRetrace = entryDiff > 0.001 // More than 0.1% away
                const retraceDirection = decEntry && currentPrice
                  ? (decEntry < currentPrice ? "down" : "up")
                  : null

                // Calculate R:R ratio
                const riskPips = decEntry && decSl ? Math.abs(decEntry - decSl) : 0
                const rewardPips = decEntry && decTp ? Math.abs(decTp - decEntry) : 0
                const rrRatio = riskPips > 0 ? (rewardPips / riskPips) : 0

                return (
                  <>
                    {/* Risk zone (entry to SL) - red shading */}
                    {hasDecision && decEntry && decSl && (
                      <rect
                        x={leftPadding}
                        y={Math.min(priceToY(decEntry, chartHeight), priceToY(decSl, chartHeight))}
                        width={chartWidth - leftPadding - rightPadding}
                        height={Math.abs(priceToY(decEntry, chartHeight) - priceToY(decSl, chartHeight))}
                        fill="rgba(239, 68, 68, 0.06)"
                      />
                    )}

                    {/* Reward zone (entry to TP) - green shading */}
                    {hasDecision && decEntry && decTp && (
                      <rect
                        x={leftPadding}
                        y={Math.min(priceToY(decEntry, chartHeight), priceToY(decTp, chartHeight))}
                        width={chartWidth - leftPadding - rightPadding}
                        height={Math.abs(priceToY(decEntry, chartHeight) - priceToY(decTp, chartHeight))}
                        fill="rgba(34, 197, 94, 0.06)"
                      />
                    )}

                    {/* Entry Price Line */}
                    {decEntry && decEntry > 0 && (
                      <g>
                        <line
                          x1={leftPadding}
                          y1={priceToY(decEntry, chartHeight)}
                          x2={chartWidth - rightPadding}
                          y2={priceToY(decEntry, chartHeight)}
                          stroke="#3b82f6"
                          strokeWidth={2}
                        />
                        {/* Entry badge */}
                        <rect
                          x={chartWidth - rightPadding - 70}
                          y={priceToY(decEntry, chartHeight) - 10}
                          width={70}
                          height={20}
                          fill="#3b82f6"
                          rx={3}
                        />
                        <text
                          x={chartWidth - rightPadding - 35}
                          y={priceToY(decEntry, chartHeight) + 4}
                          fill="white"
                          fontSize={11}
                          textAnchor="middle"
                        >
                          Entry
                        </text>
                        {/* Context annotation */}
                        {hasDecision && (
                          <>
                            {/* Arrow + instruction label */}
                            <rect
                              x={leftPadding + 5}
                              y={priceToY(decEntry, chartHeight) - (needsRetrace ? 28 : 24)}
                              width={needsRetrace ? 160 : 100}
                              height={18}
                              fill={isBuy ? "rgba(34, 197, 94, 0.9)" : "rgba(239, 68, 68, 0.9)"}
                              rx={3}
                            />
                            <text
                              x={leftPadding + 10}
                              y={priceToY(decEntry, chartHeight) - (needsRetrace ? 16 : 12)}
                              fill="white"
                              fontSize={10}
                              fontWeight="bold"
                            >
                              {needsRetrace
                                ? `${isBuy ? "BUY" : "SELL"} - Wait for retrace ${retraceDirection === "down" ? "↓" : "↑"} to ${decEntry.toFixed(digits > 3 ? 2 : digits)}`
                                : `${isBuy ? "BUY" : "SELL"} at market`
                              }
                            </text>
                            {/* Retrace arrow from current price to entry */}
                            {needsRetrace && currentPrice && (
                              <>
                                <line
                                  x1={chartWidth / 2}
                                  y1={priceToY(currentPrice, chartHeight)}
                                  x2={chartWidth / 2}
                                  y2={priceToY(decEntry, chartHeight)}
                                  stroke={isBuy ? "#22c55e" : "#ef4444"}
                                  strokeWidth={1.5}
                                  strokeDasharray="4 3"
                                  opacity={0.6}
                                />
                                {/* Arrowhead */}
                                <polygon
                                  points={`${chartWidth / 2 - 4},${priceToY(decEntry, chartHeight) + (retraceDirection === "down" ? -8 : 8)} ${chartWidth / 2 + 4},${priceToY(decEntry, chartHeight) + (retraceDirection === "down" ? -8 : 8)} ${chartWidth / 2},${priceToY(decEntry, chartHeight)}`}
                                  fill={isBuy ? "#22c55e" : "#ef4444"}
                                  opacity={0.6}
                                />
                              </>
                            )}
                          </>
                        )}
                      </g>
                    )}

                    {/* Stop Loss Line */}
                    {decSl && decSl > 0 && (
                      <g>
                        <line
                          x1={leftPadding}
                          y1={priceToY(decSl, chartHeight)}
                          x2={chartWidth - rightPadding}
                          y2={priceToY(decSl, chartHeight)}
                          stroke="#ef4444"
                          strokeWidth={2}
                          strokeDasharray="6 3"
                        />
                        <rect
                          x={chartWidth - rightPadding - 70}
                          y={priceToY(decSl, chartHeight) - 10}
                          width={70}
                          height={20}
                          fill="#ef4444"
                          rx={3}
                        />
                        <text
                          x={chartWidth - rightPadding - 35}
                          y={priceToY(decSl, chartHeight) + 4}
                          fill="white"
                          fontSize={11}
                          textAnchor="middle"
                        >
                          SL
                        </text>
                        {/* SL price + risk annotation */}
                        {hasDecision && (
                          <text
                            x={leftPadding + 5}
                            y={priceToY(decSl, chartHeight) + (isBuy ? 14 : -6)}
                            fill="#ef4444"
                            fontSize={9}
                            opacity={0.8}
                          >
                            Exit if wrong @ {decSl.toFixed(digits > 3 ? 2 : digits)}
                          </text>
                        )}
                      </g>
                    )}

                    {/* Take Profit Line */}
                    {decTp && decTp > 0 && (
                      <g>
                        <line
                          x1={leftPadding}
                          y1={priceToY(decTp, chartHeight)}
                          x2={chartWidth - rightPadding}
                          y2={priceToY(decTp, chartHeight)}
                          stroke="#22c55e"
                          strokeWidth={2}
                          strokeDasharray="6 3"
                        />
                        <rect
                          x={chartWidth - rightPadding - 70}
                          y={priceToY(decTp, chartHeight) - 10}
                          width={70}
                          height={20}
                          fill="#22c55e"
                          rx={3}
                        />
                        <text
                          x={chartWidth - rightPadding - 35}
                          y={priceToY(decTp, chartHeight) + 4}
                          fill="white"
                          fontSize={11}
                          textAnchor="middle"
                        >
                          TP
                        </text>
                        {/* TP price + target annotation */}
                        {hasDecision && (
                          <text
                            x={leftPadding + 5}
                            y={priceToY(decTp, chartHeight) + (isBuy ? -6 : 14)}
                            fill="#22c55e"
                            fontSize={9}
                            opacity={0.8}
                          >
                            Target @ {decTp.toFixed(digits > 3 ? 2 : digits)}
                          </text>
                        )}
                      </g>
                    )}

                    {/* R:R ratio badge between SL and TP */}
                    {hasDecision && rrRatio > 0 && decEntry && decSl && decTp && (
                      <g>
                        <rect
                          x={chartWidth - rightPadding - 55}
                          y={(priceToY(decEntry, chartHeight) + priceToY(isBuy ? decSl : decTp, chartHeight)) / 2 - 10}
                          width={55}
                          height={20}
                          fill="rgba(0,0,0,0.7)"
                          rx={3}
                        />
                        <text
                          x={chartWidth - rightPadding - 28}
                          y={(priceToY(decEntry, chartHeight) + priceToY(isBuy ? decSl : decTp, chartHeight)) / 2 + 4}
                          fill={rrRatio >= 2 ? "#22c55e" : rrRatio >= 1 ? "#facc15" : "#ef4444"}
                          fontSize={10}
                          textAnchor="middle"
                          fontWeight="bold"
                        >
                          R:R {rrRatio.toFixed(1)}
                        </text>
                      </g>
                    )}

                    {/* Trade decision info box */}
                    {hasDecision && tradeDecision.confidence !== undefined && (
                      <g>
                        <rect
                          x={leftPadding}
                          y={chartHeight - 24}
                          width={220}
                          height={22}
                          fill="rgba(0,0,0,0.75)"
                          rx={4}
                        />
                        <text
                          x={leftPadding + 6}
                          y={chartHeight - 9}
                          fill={isBuy ? "#22c55e" : "#ef4444"}
                          fontSize={10}
                          fontWeight="bold"
                        >
                          {tradeDecision.signal}
                        </text>
                        <text
                          x={leftPadding + 35}
                          y={chartHeight - 9}
                          fill="#d1d5db"
                          fontSize={10}
                        >
                          {tradeDecision.confidence !== undefined
                            ? `${(tradeDecision.confidence * 100).toFixed(0)}% conf`
                            : ""}
                          {tradeDecision.setup_type ? ` · ${tradeDecision.setup_type}` : ""}
                          {rrRatio > 0 ? ` · R:R ${rrRatio.toFixed(1)}` : ""}
                        </text>
                      </g>
                    )}
                  </>
                )
              })()}

              {/* Current Price Line */}
              {currentPrice && (
                <g>
                  <line
                    x1={leftPadding}
                    y1={priceToY(currentPrice, chartHeight)}
                    x2={chartWidth - rightPadding}
                    y2={priceToY(currentPrice, chartHeight)}
                    stroke="#facc15"
                    strokeWidth={2}
                  />
                  <rect
                    x={leftPadding}
                    y={priceToY(currentPrice, chartHeight) - 10}
                    width={80}
                    height={20}
                    fill="#facc15"
                    rx={3}
                  />
                  <text
                    x={leftPadding + 40}
                    y={priceToY(currentPrice, chartHeight) + 4}
                    fill="black"
                    fontSize={11}
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    {currentPrice.toFixed(digits > 3 ? 2 : digits)}
                  </text>
                </g>
              )}

              {/* Y-axis price labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
                const price = maxPrice - pct * priceRange
                return (
                  <text
                    key={pct}
                    x={5}
                    y={chartHeight * pct + 4}
                    fill="hsl(var(--muted-foreground))"
                    fontSize={10}
                  >
                    {price.toFixed(digits > 3 ? 2 : digits)}
                  </text>
                )
              })}
            </svg>

            {/* RSI Panel */}
            {rsiValues.length > 0 && (
              <div className="mt-1">
                <svg
                  width={chartWidth}
                  height={100}
                  className="bg-background w-full"
                >
                  {/* RSI Background */}
                  <rect
                    x={leftPadding}
                    y={0}
                    width={chartWidth - leftPadding - rightPadding}
                    height={100}
                    fill="hsl(var(--muted) / 0.15)"
                    stroke="hsl(var(--border))"
                    strokeWidth={0.5}
                  />

                  {/* Overbought line (70) */}
                  <line
                    x1={leftPadding}
                    y1={30}
                    x2={chartWidth - rightPadding}
                    y2={30}
                    stroke="#ef4444"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    opacity={0.5}
                  />
                  {/* Midline (50) */}
                  <line
                    x1={leftPadding}
                    y1={50}
                    x2={chartWidth - rightPadding}
                    y2={50}
                    stroke="hsl(var(--muted-foreground))"
                    strokeWidth={1}
                    strokeDasharray="2 4"
                    opacity={0.3}
                  />
                  {/* Oversold line (30) */}
                  <line
                    x1={leftPadding}
                    y1={70}
                    x2={chartWidth - rightPadding}
                    y2={70}
                    stroke="#22c55e"
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    opacity={0.5}
                  />

                  {/* Overbought zone fill */}
                  <rect
                    x={leftPadding}
                    y={0}
                    width={chartWidth - leftPadding - rightPadding}
                    height={30}
                    fill="rgba(239, 68, 68, 0.05)"
                  />
                  {/* Oversold zone fill */}
                  <rect
                    x={leftPadding}
                    y={70}
                    width={chartWidth - leftPadding - rightPadding}
                    height={30}
                    fill="rgba(34, 197, 94, 0.05)"
                  />

                  {/* RSI Line */}
                  <polyline
                    fill="none"
                    stroke="#a78bfa"
                    strokeWidth={1.5}
                    points={rsiValues.map((rsi, idx) => {
                      const x = leftPadding + idx * (candleWidth + 2) + candleWidth / 2
                      const y = 100 - rsi // RSI 0-100 maps to y 100-0
                      return `${x},${y}`
                    }).join(" ")}
                  />

                  {/* RSI dots at overbought/oversold */}
                  {rsiValues.map((rsi, idx) => {
                    if (rsi <= 30 || rsi >= 70) {
                      const x = leftPadding + idx * (candleWidth + 2) + candleWidth / 2
                      const y = 100 - rsi
                      return (
                        <circle
                          key={`rsi-dot-${idx}`}
                          cx={x}
                          cy={y}
                          r={2.5}
                          fill={rsi >= 70 ? "#ef4444" : "#22c55e"}
                          opacity={0.8}
                        />
                      )
                    }
                    return null
                  })}

                  {/* Y-axis labels */}
                  <text x={5} y={7} fill="hsl(var(--muted-foreground))" fontSize={9}>100</text>
                  <text x={5} y={33} fill="#ef4444" fontSize={9} opacity={0.7}>70</text>
                  <text x={5} y={53} fill="hsl(var(--muted-foreground))" fontSize={9}>50</text>
                  <text x={5} y={73} fill="#22c55e" fontSize={9} opacity={0.7}>30</text>
                  <text x={5} y={98} fill="hsl(var(--muted-foreground))" fontSize={9}>0</text>

                  {/* Current RSI value badge */}
                  {rsiValues.length > 0 && (() => {
                    const currentRsi = rsiValues[rsiValues.length - 1]
                    const y = 100 - currentRsi
                    const color = currentRsi >= 70 ? "#ef4444" : currentRsi <= 30 ? "#22c55e" : "#a78bfa"
                    return (
                      <g>
                        <rect
                          x={chartWidth - rightPadding - 48}
                          y={Math.max(2, Math.min(y - 8, 86))}
                          width={48}
                          height={16}
                          fill={color}
                          rx={3}
                        />
                        <text
                          x={chartWidth - rightPadding - 24}
                          y={Math.max(13, Math.min(y + 3, 97))}
                          fill="white"
                          fontSize={10}
                          textAnchor="middle"
                          fontWeight="bold"
                        >
                          RSI {currentRsi.toFixed(1)}
                        </text>
                      </g>
                    )
                  })()}

                  {/* Label */}
                  <text
                    x={leftPadding + 5}
                    y={12}
                    fill="hsl(var(--muted-foreground))"
                    fontSize={10}
                    fontWeight="bold"
                    opacity={0.6}
                  >
                    RSI (14)
                  </text>
                </svg>
              </div>
            )}

            {/* Legend */}
            <div className="flex flex-wrap justify-center gap-4 mt-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-4 h-3 bg-green-500 rounded-sm" />
                <span className="text-muted-foreground">Bullish Candle</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-4 h-3 bg-red-500 rounded-sm" />
                <span className="text-muted-foreground">Bearish Candle</span>
              </div>
              {entryPrice && entryPrice > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-blue-500" />
                  <span className="text-muted-foreground">Entry</span>
                </div>
              )}
              {stopLoss && stopLoss > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-red-500 border-dashed border-t-2 border-red-500" />
                  <span className="text-muted-foreground">SL</span>
                </div>
              )}
              {takeProfit && takeProfit > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-green-500 border-dashed border-t-2 border-green-500" />
                  <span className="text-muted-foreground">TP</span>
                </div>
              )}
              <div className="flex items-center gap-1">
                <div className="w-4 h-0.5 bg-yellow-400" />
                <span className="text-muted-foreground">Current</span>
              </div>
              {tradeDecision && tradeDecision.signal !== "HOLD" && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-3 rounded-sm" style={{ background: "linear-gradient(to bottom, rgba(34,197,94,0.15), rgba(239,68,68,0.15))" }} />
                  <span className="text-muted-foreground">Trade Plan</span>
                  <HelpTooltip content="The shaded zones show your trade plan: green = reward zone (entry to TP), red = risk zone (entry to SL). The R:R badge shows your risk-to-reward ratio. If entry is away from current price, a dashed arrow shows where to wait for price to retrace before entering." />
                </div>
              )}
              {(pdh || pdl) && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-amber-500" style={{ borderTop: "2px dashed #f59e0b" }} />
                  <span className="text-muted-foreground">PDH/PDL</span>
                  <HelpTooltip content="Previous Day High and Low - key institutional levels. Price often reacts at these levels as they represent the previous day's range extremes. Smart money uses these for entries and targets." />
                </div>
              )}
              {zones.some(z => z.source === "OB") && (
                <>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-green-500/20 border-2 border-green-500/40 rounded-sm" />
                    <span className="text-muted-foreground">OB Bull</span>
                    <HelpTooltip content="Bullish Order Block - the last candle before a strong move UP. If price returns here, expect buyers to defend and push price UP. Good area for buy entries." />
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-red-500/20 border-2 border-red-500/40 rounded-sm" />
                    <span className="text-muted-foreground">OB Bear</span>
                    <HelpTooltip content="Bearish Order Block - the last candle before a strong move DOWN. If price returns here, expect sellers to defend and push price DOWN. Good area for sell entries." />
                  </div>
                </>
              )}
              {zones.some(z => z.source === "FVG") && (
                <>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-green-500/15 border border-dashed border-green-500/40 rounded-sm" />
                    <span className="text-muted-foreground">FVG Bull</span>
                    <HelpTooltip content="Bullish Fair Value Gap - price imbalance from a rapid move UP. Price often returns to fill this gap before continuing higher. Expect bounce UP from this zone." />
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-red-500/15 border border-dashed border-red-500/40 rounded-sm" />
                    <span className="text-muted-foreground">FVG Bear</span>
                    <HelpTooltip content="Bearish Fair Value Gap - price imbalance from a rapid move DOWN. Price often returns to fill this gap before continuing lower. Expect rejection DOWN from this zone." />
                  </div>
                </>
              )}
              <div className="flex items-center gap-1">
                <div className="w-4 h-0.5 bg-purple-500" style={{ borderTop: "2px dashed #8b5cf6" }} />
                <span className="text-muted-foreground">BSL</span>
                <HelpTooltip content="Buy-side Liquidity (unswept) - buy stop orders from SHORT positions clustered ABOVE swing highs. Smart money may push price UP to sweep these stops before reversing DOWN. The % indicates strength: higher % = more significant swing, more attractive target for smart money." />
              </div>
              <div className="flex items-center gap-1">
                <div className="w-4 h-0.5 bg-pink-500" style={{ borderTop: "2px dashed #ec4899" }} />
                <span className="text-muted-foreground">SSL</span>
                <HelpTooltip content="Sell-side Liquidity (unswept) - sell stop orders from LONG positions clustered BELOW swing lows. Smart money may push price DOWN to sweep these stops before reversing UP. The % indicates strength: higher % = more significant swing, more attractive target for smart money." />
              </div>
              <div className="flex items-center gap-1">
                <div className="w-4 h-0.5 bg-gray-500" style={{ borderTop: "2px dashed #6b7280" }} />
                <span className="text-muted-foreground">Swept</span>
                <HelpTooltip content="Swept liquidity - this level has already been taken (price moved through it). The stops have been triggered and liquidity collected. Less likely to act as a magnet, but shows where smart money already hunted." />
              </div>
              {/* Breaker Blocks */}
              {breakerBlocks.length > 0 && (
                <>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-cyan-500/20 border-2 border-dashed border-cyan-500/60 rounded-sm" />
                    <span className="text-muted-foreground">BRK Bull</span>
                    <HelpTooltip content="Bullish Breaker Block - a failed bearish OB that was broken and flipped bullish. These act as strong support zones because they trapped sellers who are now underwater. Price often bounces UP from breakers." />
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-3 bg-orange-500/20 border-2 border-dashed border-orange-500/60 rounded-sm" />
                    <span className="text-muted-foreground">BRK Bear</span>
                    <HelpTooltip content="Bearish Breaker Block - a failed bullish OB that was broken and flipped bearish. These act as strong resistance zones because they trapped buyers who are now underwater. Price often rejects DOWN from breakers." />
                  </div>
                </>
              )}
              {/* OTE Zones */}
              {oteZones.length > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-3 bg-blue-500/15 border border-blue-500/40 rounded-sm" />
                  <span className="text-muted-foreground">OTE</span>
                  <HelpTooltip content="Optimal Trade Entry zone (Fibonacci 0.618-0.79 retracement). This is the 'sweet spot' for entries after a swing. Price often retraces to this zone before continuing in the original direction. The center line is the 0.705 level - the ideal entry point." />
                </div>
              )}
              {/* RSI */}
              {rsiValues.length > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-0.5 bg-purple-400" />
                  <span className="text-muted-foreground">RSI</span>
                  <HelpTooltip content="RSI (14) - Relative Strength Index measures momentum. Above 70 = overbought (red dots), price may pull back. Below 30 = oversold (green dots), price may bounce. Use with SMC zones for confluence - e.g. RSI oversold at a bullish OB = strong buy signal." />
                </div>
              )}
              {/* Equal Levels */}
              {equalLevels.length > 0 && (
                <>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-0.5 bg-teal-500" style={{ borderTop: "2px dashed #14b8a6" }} />
                    <span className="text-muted-foreground">EQH</span>
                    <HelpTooltip content="Equal Highs - multiple swing highs at the same price level. These are liquidity magnets because many traders place stops above them. Smart money often sweeps these levels before reversing. The 'x2' or 'x3' shows how many times the level was touched - more touches = stronger magnet." />
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-4 h-0.5 bg-amber-500" style={{ borderTop: "2px dashed #f59e0b" }} />
                    <span className="text-muted-foreground">EQL</span>
                    <HelpTooltip content="Equal Lows - multiple swing lows at the same price level. These are liquidity magnets because many traders place stops below them. Smart money often sweeps these levels before reversing. The 'x2' or 'x3' shows how many times the level was touched - more touches = stronger magnet." />
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Backward-compatible alias
export const PositionChart = PriceChart
