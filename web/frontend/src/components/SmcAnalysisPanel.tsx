"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { BookOpen, TrendingUp, TrendingDown, Target, Shield, AlertTriangle, Lightbulb, ChevronDown, GraduationCap, Crosshair, Copy, ShoppingCart, Loader2, CheckCircle2, Equal, RefreshCw, Ruler, Gauge, Percent, Bell, Zap, MousePointerClick, Waves, Fish } from "lucide-react"
import { saveTradeDecision } from "@/lib/api"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

// Reusable Learn More collapsible component
function LearnMore({ title, children }: { title: string; children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors mt-2">
        <GraduationCap className="h-3 w-3" />
        <span>{title}</span>
        <ChevronDown className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`} />
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 text-xs text-muted-foreground bg-muted/30 rounded-md p-3 space-y-2">
        {children}
      </CollapsibleContent>
    </Collapsible>
  )
}

// FVG Order Dialog - allows placing limit orders for FVG setups
interface FvgOrderDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  setup: FvgTradeSetup | null
  symbol: string
  digits: number
}

function FvgOrderDialog({ open, onOpenChange, setup, symbol, digits }: FvgOrderDialogProps) {
  const [volume, setVolume] = useState("0.01")
  const [entryPrice, setEntryPrice] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [calculatedRisk, setCalculatedRisk] = useState<{ pips: number; amount: number } | null>(null)

  // Reset state when dialog opens with new setup
  useEffect(() => {
    if (setup && open) {
      setEntryPrice(setup.entry.toFixed(digits))
      setError(null)
      setIsSuccess(false)
    }
  }, [setup, open, digits])

  // Calculate risk when values change
  const calculateRisk = async () => {
    if (!setup || !volume) return

    try {
      const entry = parseFloat(entryPrice || setup.entry.toString())
      const sl = setup.stopLoss
      const pips = Math.abs(entry - sl)

      // Use API to calculate monetary risk
      const response = await fetch(`${API_URL}/api/trade/calculate-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          risk_amount: 100, // dummy value to get pip value
          stop_loss_pips: pips * (symbol.includes("JPY") ? 100 : 10000),
        }),
      })

      if (response.ok) {
        const data = await response.json()
        const pipValue = data.pip_value || 10
        const riskAmount = pips * parseFloat(volume) * pipValue * (symbol.includes("JPY") ? 100 : 10000)
        setCalculatedRisk({ pips: pips * (symbol.includes("JPY") ? 100 : 10000), amount: riskAmount })
      }
    } catch {
      // Fallback calculation
      const entry = parseFloat(entryPrice || setup.entry.toString())
      const pips = Math.abs(entry - setup.stopLoss) * (symbol.includes("JPY") ? 100 : 10000)
      setCalculatedRisk({ pips, amount: pips * parseFloat(volume) * 10 })
    }
  }

  const handleSubmit = async () => {
    if (!setup) return

    setIsSubmitting(true)
    setError(null)

    try {
      const entryPriceValue = parseFloat(entryPrice || setup.entry.toString())
      const volumeValue = parseFloat(volume)

      const response = await fetch(`${API_URL}/api/trade/limit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          direction: setup.direction,
          volume: volumeValue,
          entry_price: entryPriceValue,
          stop_loss: setup.stopLoss,
          take_profit: setup.takeProfit,
          comment: `FVG ${setup.direction} @ ${setup.quality}`,
        }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || "Failed to place order")
      }

      const orderData = await response.json()

      // Save decision for tracking/learning
      // Note: FVG trades from SMC panel don't have full analysis context
      // but we can still track the trade setup for pattern learning
      await saveTradeDecision({
        symbol,
        action: setup.direction,
        entry_type: "limit",
        entry_price: entryPriceValue,
        stop_loss: setup.stopLoss,
        take_profit: setup.takeProfit,
        volume: volumeValue,
        mt5_ticket: orderData.order_ticket,
        rationale: `FVG ${setup.direction} trade at ${setup.quality} quality. ` +
          `Confluences: ${setup.confluences.join(", ")}. ` +
          `Entry zone: ${setup.entryZone.bottom.toFixed(digits)}-${setup.entryZone.top.toFixed(digits)}. ` +
          `R:R 1:${setup.riskRewardRatio.toFixed(1)}`,
        // FVG trades don't have full analysis context (no multi-agent analysis),
        // but the rationale captures the SMC setup details for learning
      })

      setIsSuccess(true)
      setTimeout(() => {
        onOpenChange(false)
        setIsSuccess(false)
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to place order")
    } finally {
      setIsSubmitting(false)
    }
  }

  if (!setup) return null

  const formatPrice = (price: number) => price.toFixed(digits)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ShoppingCart className="h-5 w-5" />
            Place FVG {setup.direction} Order
          </DialogTitle>
          <DialogDescription>
            {setup.direction === "BUY"
              ? "Buy limit order at bullish FVG support zone"
              : "Sell limit order at bearish FVG resistance zone"
            }
          </DialogDescription>
        </DialogHeader>

        {isSuccess ? (
          <div className="flex flex-col items-center justify-center py-8 text-green-500">
            <CheckCircle2 className="h-12 w-12 mb-2" />
            <p className="font-medium">Order Placed Successfully!</p>
          </div>
        ) : (
          <>
            <div className="grid gap-4 py-4">
              {/* Symbol & Direction */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Symbol</span>
                <span className="font-mono font-medium">{symbol}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Direction</span>
                <Badge variant={setup.direction === "BUY" ? "buy" : "sell"}>
                  {setup.direction}
                </Badge>
              </div>

              {/* Volume */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="volume" className="text-right">
                  Volume
                </Label>
                <Input
                  id="volume"
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={volume}
                  onChange={(e) => setVolume(e.target.value)}
                  onBlur={calculateRisk}
                  className="col-span-3 font-mono"
                />
              </div>

              {/* Entry Price */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="entry" className="text-right">
                  Entry
                </Label>
                <Input
                  id="entry"
                  type="number"
                  step={Math.pow(10, -digits)}
                  value={entryPrice || formatPrice(setup.entry)}
                  onChange={(e) => setEntryPrice(e.target.value)}
                  onBlur={calculateRisk}
                  className="col-span-3 font-mono"
                />
              </div>

              {/* Stop Loss (read-only) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right text-red-400">SL</Label>
                <div className="col-span-3 font-mono text-red-400 bg-muted/50 px-3 py-2 rounded-md">
                  {formatPrice(setup.stopLoss)}
                </div>
              </div>

              {/* Take Profit (read-only) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label className="text-right text-green-400">TP</Label>
                <div className="col-span-3 font-mono text-green-400 bg-muted/50 px-3 py-2 rounded-md">
                  {formatPrice(setup.takeProfit)}
                </div>
              </div>

              {/* Risk/Reward Info */}
              <div className="bg-muted/30 rounded-md p-3 space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">R:R Ratio</span>
                  <span className="font-mono">1:{setup.riskRewardRatio.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Quality</span>
                  <Badge variant="outline" className={
                    setup.quality === "A+" ? "border-yellow-500 text-yellow-500" :
                    setup.quality === "A" ? "border-green-500 text-green-500" : ""
                  }>
                    {setup.quality}
                  </Badge>
                </div>
                {calculatedRisk && (
                  <div className="flex justify-between text-yellow-500">
                    <span>Risk</span>
                    <span className="font-mono">~{calculatedRisk.pips.toFixed(1)} pips</span>
                  </div>
                )}
              </div>

              {/* Confluences */}
              <div className="flex flex-wrap gap-1">
                {setup.confluences.map((conf, i) => (
                  <Badge key={i} variant="secondary" className="text-xs">
                    {conf}
                  </Badge>
                ))}
              </div>

              {error && (
                <div className="text-sm text-red-500 bg-red-950/30 p-2 rounded-md">
                  {error}
                </div>
              )}
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button
                variant={setup.direction === "BUY" ? "default" : "destructive"}
                onClick={handleSubmit}
                disabled={isSubmitting}
                className={setup.direction === "BUY" ? "bg-green-600 hover:bg-green-700" : ""}
              >
                {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Place {setup.direction} Limit Order
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

// Helper to copy FVG levels to clipboard
function copyFvgLevels(setup: FvgTradeSetup, digits: number) {
  const text = `${setup.direction} ${setup.quality}
Entry: ${setup.entry.toFixed(digits)}
SL: ${setup.stopLoss.toFixed(digits)}
TP: ${setup.takeProfit.toFixed(digits)}
R:R 1:${setup.riskRewardRatio.toFixed(1)}`

  navigator.clipboard.writeText(text)
}

interface OrderBlock {
  type: string
  top: number
  bottom: number
  strength?: number
  mitigated?: boolean
}

interface FairValueGap {
  type: string
  top: number
  bottom: number
  mitigated?: boolean
}

interface LiquidityZone {
  type: "buy-side" | "sell-side"
  price: number
  strength?: number
  touched?: boolean
}

// NEW: Equal Highs/Lows (liquidity targets)
interface EqualLevel {
  type: "eqh" | "eql"  // Equal High or Equal Low
  price: number
  touches: number  // How many times this level was touched
  strength?: number
  swept?: boolean  // Whether liquidity has been taken
}

// NEW: Breaker Blocks (failed OBs that flip polarity)
interface BreakerBlock {
  type: "bullish" | "bearish"
  top: number
  bottom: number
  original_ob_type?: string
  break_price?: number
  strength?: number
  mitigated?: boolean
}

// NEW: Optimal Trade Entry zones (Fibonacci 62-79% retracement)
interface OTEZone {
  type: "bullish" | "bearish"
  fib_618: number
  fib_705: number
  fib_79: number
  swing_high?: number
  swing_low?: number
}

// NEW: Premium/Discount zone position
interface PremiumDiscount {
  current_zone: "premium" | "discount" | "equilibrium"
  equilibrium: number
  premium_start: number
  discount_end: number
  range_high?: number
  range_low?: number
  position_percent: number
}

// NEW: Confluence score for trade quality
interface ConfluenceScore {
  total_score: number
  bias_alignment: number
  zone_proximity: number
  structure_confirmation: number
  liquidity_target: number
  mtf_alignment: number
  session_factor: number
  recommendation: string
}

// NEW: Liquidity Sweep - price grabs stops then reverses
interface LiquiditySweep {
  type: "bullish" | "bearish"
  sweep_level: number
  sweep_low: number
  sweep_high: number
  close_price: number
  rejection_strength: number  // 0-1, how strong the rejection wick was
  atr_penetration: number  // How far beyond the level in ATR terms
  timestamp: string
  session?: string
}

// NEW: Inducement - false breakout to trap traders
interface Inducement {
  type: "bullish" | "bearish"
  inducement_level: number
  inducement_index: number
  break_index: number
  reversal_index: number
  target_liquidity?: number
  timestamp: string
  trapped_direction: string  // "longs" or "shorts"
}

// NEW: Rejection Block - strong rejection candle at key level
interface RejectionBlock {
  type: "bullish" | "bearish"
  rejection_price: number
  body_top: number
  body_bottom: number
  wick_size: number
  wick_atr_ratio: number  // Wick size relative to ATR
  timestamp: string
  session?: string
  held: boolean
  mitigated: boolean
}

// NEW: Turtle Soup - failed breakout reversal pattern
interface TurtleSoup {
  type: "bullish" | "bearish"
  level: number
  penetration: number  // How far price went beyond
  penetration_atr: number
  timestamp: string
  swing_index: number
}

// NEW: Pattern Alert - actionable alert from pattern detection
interface PatternAlert {
  priority: "HIGH" | "MEDIUM" | "LOW"
  pattern_type: string
  direction: "bullish" | "bearish"
  message: string
  price_level?: number
  timestamp: string
  action_suggestion?: string
}

interface SmcAnalysisPanelProps {
  symbol: string
  currentPrice: number
  orderBlocks: OrderBlock[]
  fairValueGaps: FairValueGap[]
  liquidityZones: LiquidityZone[]
  digits?: number
  // NEW: Extended SMC data
  equalLevels?: EqualLevel[]
  breakerBlocks?: BreakerBlock[]
  oteZones?: OTEZone[]
  premiumDiscount?: PremiumDiscount | null
  confluenceScore?: ConfluenceScore | null
  // NEW: Advanced SMC patterns
  liquiditySweeps?: LiquiditySweep[]
  inducements?: Inducement[]
  rejectionBlocks?: RejectionBlock[]
  turtleSoup?: TurtleSoup[]
  alerts?: PatternAlert[]
}

// FVG Trade Setup - detects when a quality FVG trading opportunity exists
// NOTE: Only "bounce" trades (with-trend) are offered. Counter-trend "fill" trades
// are low probability and often result in liquidity traps.
interface FvgTradeSetup {
  direction: "BUY" | "SELL"
  fvg: FairValueGap
  entry: number  // Entry at FVG zone
  entryZone: { top: number; bottom: number }  // The zone to watch for entry
  stopLoss: number  // Beyond the FVG
  takeProfit: number  // Liquidity target
  riskRewardRatio: number
  quality: "A+" | "A" | "B"  // Setup quality rating
  confluences: string[]  // What makes this setup quality
  invalidation: string  // When to abandon the setup
}

function detectFvgTradeSetups(
  currentPrice: number,
  bullishFVGs: FairValueGap[],
  bearishFVGs: FairValueGap[],
  bullishOBs: OrderBlock[],
  bearishOBs: OrderBlock[],
  nearestBSL: LiquidityZone | undefined,
  nearestSSL: LiquidityZone | undefined,
  overallBias: "bullish" | "bearish" | "neutral"
): FvgTradeSetup[] {
  const setups: FvgTradeSetup[] = []

  // =====================================================================
  // BULLISH FVGs (below current price)
  // Strategy: Wait for price to retrace INTO the FVG, then BUY
  // The FVG acts as SUPPORT - expect bounce UP after fill
  // DO NOT short into bullish FVGs - that's a liquidity trap!
  // =====================================================================
  for (const fvg of bullishFVGs) {
    const fvgSize = fvg.top - fvg.bottom
    const fvgMid = (fvg.top + fvg.bottom) / 2
    const distanceToFvg = currentPrice - fvg.top
    const distancePercent = (distanceToFvg / currentPrice) * 100

    // Skip if FVG is too far (>2%) or price already below it
    if (distancePercent > 2 || distancePercent < -0.5) continue

    // Confluences that increase probability
    const confluences: string[] = []

    // OB confluence - FVG overlaps with bullish OB
    const hasOBConfluence = bullishOBs.some(ob =>
      ob.top >= fvg.bottom && ob.bottom <= fvg.top
    )
    if (hasOBConfluence) confluences.push("OB confluence")

    // Aligned with overall bias
    if (overallBias === "bullish") confluences.push("Aligned with bullish bias")

    // Clear liquidity target above
    if (nearestBSL) confluences.push(`BSL target at ${nearestBSL.price.toFixed(2)}`)

    // Price is approaching (setup is imminent)
    if (distancePercent < 0.5 && distancePercent > 0) confluences.push("Price approaching")

    // Entry zone: the FVG itself (aggressive at top, conservative at 50%)
    const entryAggressive = fvg.top
    const entryConservative = fvgMid  // 50% / Consequent Encroachment

    // Stop loss below the FVG with buffer
    const stopLoss = fvg.bottom - (fvgSize * 0.2)

    // Take profit at nearest BSL or minimum 2:1 RR
    const minTP = entryAggressive + (entryAggressive - stopLoss) * 2
    const takeProfit = nearestBSL && nearestBSL.price > minTP ? nearestBSL.price : minTP

    const riskRewardRatio = (takeProfit - entryAggressive) / (entryAggressive - stopLoss)

    // Quality rating
    let quality: "A+" | "A" | "B" = "B"
    if (confluences.length >= 3) quality = "A+"
    else if (confluences.length >= 2) quality = "A"

    if (riskRewardRatio >= 1.5) {
      setups.push({
        direction: "BUY",
        fvg,
        entry: entryAggressive,
        entryZone: { top: fvg.top, bottom: fvgMid },
        stopLoss,
        takeProfit,
        riskRewardRatio,
        quality,
        confluences,
        invalidation: `Close below ${fvg.bottom.toFixed(5)} invalidates the FVG`
      })
    }
  }

  // =====================================================================
  // BEARISH FVGs (above current price)
  // Strategy: Wait for price to rally INTO the FVG, then SELL
  // The FVG acts as RESISTANCE - expect rejection DOWN after fill
  // DO NOT buy into bearish FVGs - that's a liquidity trap!
  // =====================================================================
  for (const fvg of bearishFVGs) {
    const fvgSize = fvg.top - fvg.bottom
    const fvgMid = (fvg.top + fvg.bottom) / 2
    const distanceToFvg = fvg.bottom - currentPrice
    const distancePercent = (distanceToFvg / currentPrice) * 100

    // Skip if FVG is too far or price already above it
    if (distancePercent > 2 || distancePercent < -0.5) continue

    // Confluences
    const confluences: string[] = []

    const hasOBConfluence = bearishOBs.some(ob =>
      ob.top >= fvg.bottom && ob.bottom <= fvg.top
    )
    if (hasOBConfluence) confluences.push("OB confluence")
    if (overallBias === "bearish") confluences.push("Aligned with bearish bias")
    if (nearestSSL) confluences.push(`SSL target at ${nearestSSL.price.toFixed(2)}`)
    if (distancePercent < 0.5 && distancePercent > 0) confluences.push("Price approaching")

    // Entry zone
    const entryAggressive = fvg.bottom
    const entryConservative = fvgMid

    // Stop loss above the FVG
    const stopLoss = fvg.top + (fvgSize * 0.2)

    // Take profit at nearest SSL or minimum 2:1 RR
    const minTP = entryAggressive - (stopLoss - entryAggressive) * 2
    const takeProfit = nearestSSL && nearestSSL.price < minTP ? nearestSSL.price : minTP

    const riskRewardRatio = (entryAggressive - takeProfit) / (stopLoss - entryAggressive)

    let quality: "A+" | "A" | "B" = "B"
    if (confluences.length >= 3) quality = "A+"
    else if (confluences.length >= 2) quality = "A"

    if (riskRewardRatio >= 1.5) {
      setups.push({
        direction: "SELL",
        fvg,
        entry: entryAggressive,
        entryZone: { top: fvgMid, bottom: fvg.bottom },
        stopLoss,
        takeProfit,
        riskRewardRatio,
        quality,
        confluences,
        invalidation: `Close above ${fvg.top.toFixed(5)} invalidates the FVG`
      })
    }
  }

  // Sort by quality
  setups.sort((a, b) => {
    const qualityOrder = { "A+": 0, "A": 1, "B": 2 }
    return qualityOrder[a.quality] - qualityOrder[b.quality]
  })

  return setups
}

export function SmcAnalysisPanel({
  symbol,
  currentPrice,
  orderBlocks = [],
  fairValueGaps = [],
  liquidityZones = [],
  digits = 5,
  // NEW: Extended SMC data
  equalLevels = [],
  breakerBlocks = [],
  oteZones = [],
  premiumDiscount = null,
  confluenceScore = null,
  // NEW: Advanced SMC patterns
  liquiditySweeps = [],
  inducements = [],
  rejectionBlocks = [],
  turtleSoup = [],
  alerts = [],
}: SmcAnalysisPanelProps) {
  const formatPrice = (price: number) => price.toFixed(digits)

  // FVG Order Dialog state
  const [orderDialogOpen, setOrderDialogOpen] = useState(false)
  const [selectedSetup, setSelectedSetup] = useState<FvgTradeSetup | null>(null)
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  const handlePlaceOrder = (setup: FvgTradeSetup) => {
    setSelectedSetup(setup)
    setOrderDialogOpen(true)
  }

  const handleCopyLevels = (setup: FvgTradeSetup, idx: number) => {
    copyFvgLevels(setup, digits)
    setCopiedIndex(idx)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  // Categorize order blocks
  const bullishOBs = orderBlocks.filter(ob => ob.type === "bullish" && !ob.mitigated)
  const bearishOBs = orderBlocks.filter(ob => ob.type === "bearish" && !ob.mitigated)

  // Categorize FVGs
  const bullishFVGs = fairValueGaps.filter(fvg => fvg.type === "bullish" && !fvg.mitigated)
  const bearishFVGs = fairValueGaps.filter(fvg => fvg.type === "bearish" && !fvg.mitigated)

  // Categorize liquidity zones
  const bslZones = liquidityZones.filter(lz => lz.type === "buy-side")
  const sslZones = liquidityZones.filter(lz => lz.type === "sell-side")
  const unsweptBSL = bslZones.filter(lz => !lz.touched)
  const unsweptSSL = sslZones.filter(lz => !lz.touched)

  // Find nearest levels
  const nearestBullishOB = bullishOBs
    .filter(ob => ob.top < currentPrice)
    .sort((a, b) => b.top - a.top)[0]
  const nearestBearishOB = bearishOBs
    .filter(ob => ob.bottom > currentPrice)
    .sort((a, b) => a.bottom - b.bottom)[0]

  const nearestBullishFVG = bullishFVGs
    .filter(fvg => fvg.top < currentPrice)
    .sort((a, b) => b.top - a.top)[0]
  const nearestBearishFVG = bearishFVGs
    .filter(fvg => fvg.bottom > currentPrice)
    .sort((a, b) => a.bottom - b.bottom)[0]

  const nearestBSL = unsweptBSL
    .filter(lz => lz.price > currentPrice)
    .sort((a, b) => a.price - b.price)[0]
  const nearestSSL = unsweptSSL
    .filter(lz => lz.price < currentPrice)
    .sort((a, b) => b.price - a.price)[0]

  // Determine overall bias based on SMC structure
  const supportLevels = bullishOBs.length + bullishFVGs.length
  const resistanceLevels = bearishOBs.length + bearishFVGs.length
  const liquidityAbove = unsweptBSL.length
  const liquidityBelow = unsweptSSL.length

  let overallBias: "bullish" | "bearish" | "neutral" = "neutral"
  let biasReason = ""

  if (supportLevels > resistanceLevels && liquidityAbove > liquidityBelow) {
    overallBias = "bullish"
    biasReason = "More support levels below and unswept liquidity above suggests upside potential"
  } else if (resistanceLevels > supportLevels && liquidityBelow > liquidityAbove) {
    overallBias = "bearish"
    biasReason = "More resistance levels above and unswept liquidity below suggests downside potential"
  } else if (liquidityAbove > liquidityBelow) {
    overallBias = "bullish"
    biasReason = "Unswept buy-side liquidity above may attract price upward"
  } else if (liquidityBelow > liquidityAbove) {
    overallBias = "bearish"
    biasReason = "Unswept sell-side liquidity below may attract price downward"
  }

  // Detect FVG trade setup opportunities (both bounce and fill types)
  const fvgSetups = detectFvgTradeSetups(
    currentPrice,
    bullishFVGs,
    bearishFVGs,
    bullishOBs,
    bearishOBs,
    nearestBSL,
    nearestSSL,
    overallBias
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          SMC Analysis
          <HelpTooltip content="Smart Money Concepts (SMC) analysis explains institutional trading patterns visible on the chart. This panel helps you understand what each level means and how price might react." />
        </CardTitle>
        <CardDescription>
          Educational breakdown of {symbol} price action
        </CardDescription>
      </CardHeader>
      <ScrollArea className="h-[500px]">
        <CardContent className="space-y-6 text-sm">
          {/* Overall Market Bias */}
          <div className="rounded-lg border p-4 space-y-2">
            <div className="flex items-center gap-2 font-medium">
              {overallBias === "bullish" ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : overallBias === "bearish" ? (
                <TrendingDown className="h-4 w-4 text-red-500" />
              ) : (
                <Target className="h-4 w-4 text-yellow-500" />
              )}
              <span>Market Bias: </span>
              <Badge variant={overallBias === "bullish" ? "buy" : overallBias === "bearish" ? "sell" : "secondary"}>
                {overallBias.toUpperCase()}
              </Badge>
            </div>
            <p className="text-muted-foreground">{biasReason || "Mixed signals - wait for clearer structure"}</p>
          </div>

          {/* Pattern Alerts - High priority actionable alerts */}
          {alerts.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Bell className="h-4 w-4 text-yellow-500" />
                Pattern Alerts
                <Badge variant="outline" className="text-yellow-500 border-yellow-500">
                  {alerts.length} active
                </Badge>
                <HelpTooltip content="Real-time alerts from SMC pattern detection. HIGH priority alerts indicate immediate trading opportunities. MEDIUM alerts are developing setups. LOW are informational." />
              </h4>

              {alerts.slice(0, 5).map((alert, idx) => (
                <div
                  key={idx}
                  className={`rounded-lg border-l-4 p-3 space-y-1 ${
                    alert.priority === "HIGH"
                      ? "border-l-red-500 bg-red-950/20"
                      : alert.priority === "MEDIUM"
                      ? "border-l-yellow-500 bg-yellow-950/20"
                      : "border-l-blue-500 bg-blue-950/20"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        alert.priority === "HIGH" ? "destructive" :
                        alert.priority === "MEDIUM" ? "secondary" : "outline"
                      }
                      className={alert.priority === "MEDIUM" ? "bg-yellow-600" : ""}
                    >
                      {alert.priority}
                    </Badge>
                    <span className={`font-medium ${
                      alert.direction === "bullish" ? "text-green-400" : "text-red-400"
                    }`}>
                      {alert.pattern_type}
                    </span>
                    {alert.price_level && (
                      <span className="text-muted-foreground text-sm font-mono">
                        @ {formatPrice(alert.price_level)}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">{alert.message}</p>
                  {alert.action_suggestion && (
                    <p className="text-sm font-medium text-foreground">
                      💡 {alert.action_suggestion}
                    </p>
                  )}
                </div>
              ))}

              {alerts.length > 5 && (
                <p className="text-xs text-muted-foreground text-center">
                  +{alerts.length - 5} more alerts
                </p>
              )}
            </div>
          )}

          {/* FVG Trade Setups - Shows when opportunities exist */}
          {fvgSetups.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Crosshair className="h-4 w-4" />
                FVG Trade Setups
                <Badge variant="outline">{fvgSetups.length} available</Badge>
                <HelpTooltip content="Fair Value Gap trade setups. 'Bounce' trades target the reaction after FVG fill. 'Fill' trades are quick scalps targeting the 50% retracement into the gap." />
              </h4>

              {fvgSetups.map((setup, idx) => (
                <div
                  key={idx}
                  className={`rounded-lg border-2 p-4 space-y-3 ${
                    setup.direction === "BUY"
                      ? "border-green-500/50 bg-green-950/20"
                      : "border-red-500/50 bg-red-950/20"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 font-medium">
                      <Badge variant={setup.direction === "BUY" ? "buy" : "sell"}>
                        {setup.direction}
                      </Badge>
                      <span className="text-sm">
                        {setup.direction === "BUY" ? "at Bullish FVG" : "at Bearish FVG"}
                      </span>
                      <HelpTooltip
                        content={
                          setup.direction === "BUY"
                            ? "Bullish FVG acts as SUPPORT. Wait for price to retrace into the gap, then enter long expecting bounce UP. The 50% level (CE) often provides the best reaction."
                            : "Bearish FVG acts as RESISTANCE. Wait for price to rally into the gap, then enter short expecting rejection DOWN. The 50% level (CE) often provides the best reaction."
                        }
                      />
                    </div>
                    <Badge
                      variant="outline"
                      className={
                        setup.quality === "A+"
                          ? "border-yellow-500 text-yellow-500"
                          : setup.quality === "A"
                          ? "border-green-500 text-green-500"
                          : ""
                      }
                    >
                      {setup.quality}
                    </Badge>
                  </div>

                  {/* Entry Zone */}
                  <div className="text-xs bg-muted/30 rounded p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-muted-foreground">Entry Zone:</span>
                      <span className={`font-mono ${setup.direction === "BUY" ? "text-green-400" : "text-red-400"}`}>
                        {formatPrice(setup.entryZone.bottom)} - {formatPrice(setup.entryZone.top)}
                      </span>
                    </div>
                    <div className="text-muted-foreground">
                      {setup.direction === "BUY"
                        ? "Aggressive entry at top, conservative at 50% (CE)"
                        : "Aggressive entry at bottom, conservative at 50% (CE)"
                      }
                    </div>
                  </div>

                  {/* Trade Parameters */}
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="space-y-1">
                      <div className="text-muted-foreground text-xs">Entry (Aggressive)</div>
                      <div className={`font-mono font-medium ${setup.direction === "BUY" ? "text-green-400" : "text-red-400"}`}>
                        {formatPrice(setup.entry)}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-muted-foreground text-xs">Stop Loss</div>
                      <div className="font-mono font-medium text-red-400">
                        {formatPrice(setup.stopLoss)}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-muted-foreground text-xs">Take Profit</div>
                      <div className="font-mono font-medium text-green-400">
                        {formatPrice(setup.takeProfit)}
                      </div>
                    </div>
                  </div>

                  {/* Risk/Reward */}
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground">R:R</span>
                    <Badge variant="outline" className="font-mono">
                      1:{setup.riskRewardRatio.toFixed(1)}
                    </Badge>
                  </div>

                  {/* Trade Action Buttons */}
                  <div className="flex gap-2 pt-2">
                    <Button
                      size="sm"
                      variant={setup.direction === "BUY" ? "default" : "destructive"}
                      className={setup.direction === "BUY" ? "bg-green-600 hover:bg-green-700 flex-1" : "flex-1"}
                      onClick={() => handlePlaceOrder(setup)}
                    >
                      <ShoppingCart className="mr-1 h-3 w-3" />
                      Place Limit Order
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleCopyLevels(setup, idx)}
                      className="w-24"
                    >
                      {copiedIndex === idx ? (
                        <>
                          <CheckCircle2 className="mr-1 h-3 w-3 text-green-500" />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy className="mr-1 h-3 w-3" />
                          Copy
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Confluences */}
                  <div className="flex flex-wrap gap-1">
                    {setup.confluences.map((conf, i) => (
                      <Badge key={i} variant="secondary" className="text-xs">
                        {conf}
                      </Badge>
                    ))}
                  </div>

                  {/* How to Trade */}
                  {idx === 0 ? (
                    <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t border-border/50">
                      <p>
                        <span className="font-medium text-foreground">How to trade:</span>
                        {setup.direction === "BUY" ? (
                          <> Wait for price to retrace INTO the FVG zone. Look for bullish confirmation (engulfing, pin bar, higher low) at the 50% level or below. Enter long with stop below the FVG.</>
                        ) : (
                          <> Wait for price to rally INTO the FVG zone. Look for bearish confirmation (shooting star, engulfing, lower high) at the 50% level or above. Enter short with stop above the FVG.</>
                        )}
                      </p>
                      <p className="text-yellow-500/80">
                        <span className="font-medium">Invalidation:</span> {setup.invalidation}
                      </p>
                    </div>
                  ) : (
                    <Collapsible>
                      <CollapsibleTrigger className="text-xs text-blue-400 hover:text-blue-300">
                        Show trade instructions...
                      </CollapsibleTrigger>
                      <CollapsibleContent className="text-xs text-muted-foreground space-y-1 pt-2">
                        <p>
                          {setup.direction === "BUY"
                            ? "Wait for price to fill the FVG, look for bullish confirmation, enter long."
                            : "Wait for price to fill the FVG, look for bearish confirmation, enter short."
                          }
                        </p>
                        <p className="text-yellow-500/80">{setup.invalidation}</p>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>
              ))}

              <LearnMore title="Learn about FVG trading">
                <p><strong>FVG Trading Strategy (ICT/SMC)</strong></p>
                <p>
                  FVGs are price imbalances created by strong momentum. The key is to trade WITH the momentum that created the gap, not against it.
                </p>

                <p className="pt-2"><strong>Bullish FVG (below price):</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Acts as <strong className="text-green-400">SUPPORT</strong> - expect price to bounce UP</li>
                  <li>Wait for price to retrace INTO the gap</li>
                  <li>Look for bullish confirmation at 50% (CE) level</li>
                  <li>Enter LONG, stop below FVG, target liquidity above</li>
                </ul>

                <p className="pt-2"><strong>Bearish FVG (above price):</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Acts as <strong className="text-red-400">RESISTANCE</strong> - expect price to reject DOWN</li>
                  <li>Wait for price to rally INTO the gap</li>
                  <li>Look for bearish confirmation at 50% (CE) level</li>
                  <li>Enter SHORT, stop above FVG, target liquidity below</li>
                </ul>

                <p className="pt-2"><strong className="text-yellow-500">Why NOT to trade counter-trend:</strong></p>
                <p>
                  <strong>DO NOT</strong> short into a bullish FVG or buy into a bearish FVG. This is a common trap:
                </p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>You&apos;re fighting the momentum that created the gap</li>
                  <li>The FVG acts as support/resistance in its original direction</li>
                  <li>Smart money uses counter-trend traders as <strong>liquidity</strong></li>
                  <li>Price wicks into FVG → grabs your stop → continues original direction</li>
                  <li>This is classic <strong>inducement</strong> - the pullback looks tempting but is a trap</li>
                </ul>

                <p className="pt-2"><strong>When counter-trend IS valid:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>FVG is fully mitigated (price closes through entire gap)</li>
                  <li>FVG flips to <strong>Inversion FVG (IFVG)</strong></li>
                  <li>Clear CHoCH/BOS on higher timeframe confirms reversal</li>
                  <li>Multiple reversal confluences present</li>
                </ul>

                <p className="pt-2 text-muted-foreground italic">
                  Rule: Trade in the direction of the impulse that created the FVG. Anything else is gambling without strong reversal confluence.
                </p>
              </LearnMore>
            </div>
          )}

          {/* Liquidity Analysis */}
          {(liquidityZones.length > 0) && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Target className="h-4 w-4" />
                Liquidity Targets
                <HelpTooltip content="Smart money hunts stop losses clustered at swing highs (BSL) and swing lows (SSL). Price often moves toward these pools before reversing." />
              </h4>

              {nearestBSL && (
                <div className="pl-4 border-l-2 border-purple-500 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-purple-400">Buy-side Liquidity @ {formatPrice(nearestBSL.price)}</span>
                    {nearestBSL.strength && <Badge variant="outline">{Math.round(nearestBSL.strength)}% strength</Badge>}
                  </div>
                  <p className="text-muted-foreground">
                    Short sellers have their stop losses (buy stops) clustered above this swing high.
                    {nearestBSL.price - currentPrice < currentPrice * 0.005
                      ? " Price is very close - a sweep may be imminent. Watch for rejection after the sweep for a potential short entry."
                      : ` This is ${(((nearestBSL.price - currentPrice) / currentPrice) * 100).toFixed(2)}% above current price. Smart money may push price up to grab these stops before reversing down.`}
                  </p>
                </div>
              )}

              {nearestSSL && (
                <div className="pl-4 border-l-2 border-pink-500 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-pink-400">Sell-side Liquidity @ {formatPrice(nearestSSL.price)}</span>
                    {nearestSSL.strength && <Badge variant="outline">{Math.round(nearestSSL.strength)}% strength</Badge>}
                  </div>
                  <p className="text-muted-foreground">
                    Long traders have their stop losses (sell stops) clustered below this swing low.
                    {currentPrice - nearestSSL.price < currentPrice * 0.005
                      ? " Price is very close - a sweep may be imminent. Watch for rejection after the sweep for a potential long entry."
                      : ` This is ${(((currentPrice - nearestSSL.price) / currentPrice) * 100).toFixed(2)}% below current price. Smart money may push price down to grab these stops before reversing up.`}
                  </p>
                </div>
              )}

              {bslZones.filter(lz => lz.touched).length > 0 && (
                <div className="pl-4 border-l-2 border-gray-500 space-y-1">
                  <span className="font-medium text-gray-400">Swept Liquidity</span>
                  <p className="text-muted-foreground">
                    {bslZones.filter(lz => lz.touched).length} BSL and {sslZones.filter(lz => lz.touched).length} SSL zones have been swept recently.
                    This means smart money has already collected those stops - look for reversals from these sweep points or new liquidity pools to form.
                  </p>
                </div>
              )}

              {unsweptBSL.length === 0 && unsweptSSL.length === 0 && (
                <p className="text-muted-foreground pl-4">
                  No significant unswept liquidity detected. Price may consolidate until new swing points form, creating fresh liquidity pools.
                </p>
              )}

              <LearnMore title="Learn more about Liquidity">
                <p><strong>What is Liquidity in SMC?</strong></p>
                <p>
                  Liquidity refers to clusters of stop loss orders that accumulate at predictable price levels. Smart money &quot;hunts&quot; these stops to fill their large orders before moving price in their intended direction.
                </p>

                <p className="pt-2"><strong>Types of Liquidity:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Buy-side Liquidity (BSL):</strong> Stop losses from SHORT positions, clustered ABOVE swing highs. When triggered, these become buy orders.</li>
                  <li><strong>Sell-side Liquidity (SSL):</strong> Stop losses from LONG positions, clustered BELOW swing lows. When triggered, these become sell orders.</li>
                </ul>

                <p className="pt-2"><strong>Why does Smart Money hunt liquidity?</strong></p>
                <p>
                  Large institutions need opposite orders to fill their positions. If they want to sell, they need buyers. By pushing price UP to trigger buy stops, they get the buy orders needed to fill their large sell orders at premium prices.
                </p>

                <p className="pt-2"><strong>Liquidity Sweep (Stop Hunt):</strong></p>
                <p>
                  A sweep occurs when price breaks through a swing high/low, triggers the stops, then quickly reverses. This is the &quot;trap&quot; - retail traders get stopped out right before price moves in their original direction.
                </p>

                <p className="pt-2"><strong>How to trade liquidity:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Identify where liquidity pools exist (obvious swing highs/lows)</li>
                  <li>Expect price to be drawn to these levels like a magnet</li>
                  <li>Watch for reversal signals AFTER the sweep occurs</li>
                  <li>Enter opposite to the sweep direction once confirmed</li>
                  <li>Avoid placing stops at obvious levels - they will be hunted</li>
                </ul>

                <p className="pt-2"><strong>Liquidity Strength:</strong></p>
                <p>
                  The more times a swing high/low has been tested without breaking, the more stops accumulate there. Equal highs/lows (double tops/bottoms) are especially attractive targets because they&apos;re obvious to all traders.
                </p>
              </LearnMore>
            </div>
          )}

          {/* Liquidity Sweeps - Stop hunt reversals */}
          {liquiditySweeps.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Zap className="h-4 w-4 text-yellow-500" />
                Liquidity Sweeps
                <Badge variant="outline" className="text-yellow-500 border-yellow-500">
                  {liquiditySweeps.length} detected
                </Badge>
                <HelpTooltip content="Liquidity sweeps occur when price breaks a swing high/low to grab stops, then quickly reverses. These are high-probability reversal setups - smart money uses them to enter opposite positions." />
              </h4>

              {liquiditySweeps.slice(0, 3).map((sweep, idx) => (
                <div
                  key={idx}
                  className={`pl-4 border-l-2 ${
                    sweep.type === "bullish" ? "border-green-500" : "border-red-500"
                  } space-y-1`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${sweep.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {sweep.type === "bullish" ? "Bullish Sweep" : "Bearish Sweep"} @ {formatPrice(sweep.sweep_level)}
                    </span>
                    {sweep.rejection_strength >= 0.7 && (
                      <Badge variant="secondary" className="text-green-500">Strong Rejection</Badge>
                    )}
                    {sweep.session && (
                      <Badge variant="outline">{sweep.session}</Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {sweep.type === "bullish" ? (
                      <>
                        Price swept below {formatPrice(sweep.sweep_level)} to {formatPrice(sweep.sweep_low)}, grabbing sell-side liquidity, then closed at {formatPrice(sweep.close_price)}.
                        <span className="text-green-400 font-medium"> Bullish reversal signal - look for longs.</span>
                      </>
                    ) : (
                      <>
                        Price swept above {formatPrice(sweep.sweep_level)} to {formatPrice(sweep.sweep_high)}, grabbing buy-side liquidity, then closed at {formatPrice(sweep.close_price)}.
                        <span className="text-red-400 font-medium"> Bearish reversal signal - look for shorts.</span>
                      </>
                    )}
                  </div>
                  {sweep.atr_penetration > 0 && (
                    <div className="text-xs text-muted-foreground">
                      Penetration: {(sweep.atr_penetration * 100).toFixed(0)}% of ATR | Rejection strength: {(sweep.rejection_strength * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              ))}

              <LearnMore title="Learn about Liquidity Sweeps">
                <p><strong>What is a Liquidity Sweep?</strong></p>
                <p>
                  A sweep occurs when price breaks through a swing high or low to trigger stop losses, then quickly reverses. This &quot;stop hunt&quot; allows smart money to fill their orders using retail trader&apos;s stops as liquidity.
                </p>
                <p className="pt-2"><strong>How to trade sweeps:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bullish Sweep:</strong> Price breaks below swing low, grabs stops, reverses UP - look for longs</li>
                  <li><strong>Bearish Sweep:</strong> Price breaks above swing high, grabs stops, reverses DOWN - look for shorts</li>
                  <li>Enter after the reversal candle closes</li>
                  <li>Stop loss beyond the sweep extreme</li>
                  <li>Target the opposite liquidity pool</li>
                </ul>
                <p className="pt-2 text-yellow-500">
                  <strong>Key:</strong> Don&apos;t chase the initial move. Wait for confirmation that smart money has reversed the market.
                </p>
              </LearnMore>
            </div>
          )}

          {/* Inducements - False breakouts that trap traders */}
          {inducements.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <MousePointerClick className="h-4 w-4 text-orange-500" />
                Inducements
                <Badge variant="outline" className="text-orange-500 border-orange-500">
                  {inducements.length} traps
                </Badge>
                <HelpTooltip content="Inducements are false breakouts designed to trap retail traders. Price makes a minor high/low to entice traders, then reverses sharply. Smart money uses these to build positions opposite to trapped traders." />
              </h4>

              {inducements.slice(0, 3).map((ind, idx) => (
                <div
                  key={idx}
                  className={`pl-4 border-l-2 ${
                    ind.type === "bullish" ? "border-green-500" : "border-red-500"
                  } border-dashed space-y-1`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${ind.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {ind.type === "bullish" ? "Bullish Inducement" : "Bearish Inducement"} @ {formatPrice(ind.inducement_level)}
                    </span>
                    <Badge variant="secondary" className="text-orange-400">
                      Trapped {ind.trapped_direction}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {ind.type === "bullish" ? (
                      <>
                        False breakdown at {formatPrice(ind.inducement_level)} trapped short sellers.
                        <span className="text-green-400 font-medium"> Price reversed bullish - shorts are now fuel for upside move.</span>
                      </>
                    ) : (
                      <>
                        False breakout at {formatPrice(ind.inducement_level)} trapped long buyers.
                        <span className="text-red-400 font-medium"> Price reversed bearish - longs are now fuel for downside move.</span>
                      </>
                    )}
                  </div>
                  {ind.target_liquidity && (
                    <div className="text-xs text-muted-foreground">
                      Target liquidity: {formatPrice(ind.target_liquidity)}
                    </div>
                  )}
                </div>
              ))}

              <LearnMore title="Learn about Inducements">
                <p><strong>What is an Inducement?</strong></p>
                <p>
                  An inducement is a deliberate false move designed to trap traders. Smart money creates a minor high or low that looks like a breakout, enticing retail traders to enter, then reverses sharply to trap them.
                </p>
                <p className="pt-2"><strong>Why inducements work:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Retail traders see &quot;obvious&quot; breakout and enter</li>
                  <li>Their stops become fuel for smart money&apos;s move</li>
                  <li>Trapped traders panic close, accelerating the reversal</li>
                </ul>
                <p className="pt-2"><strong>How to trade inducements:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Wait for the false move to complete</li>
                  <li>Enter when price confirms reversal (closes back in range)</li>
                  <li>Trade in the direction of the reversal</li>
                  <li>Trapped traders&apos; stop hunts become your target</li>
                </ul>
              </LearnMore>
            </div>
          )}

          {/* Rejection Blocks - Strong rejection candles */}
          {rejectionBlocks.filter(rb => !rb.mitigated).length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Waves className="h-4 w-4 text-cyan-500" />
                Rejection Blocks
                <Badge variant="outline" className="text-cyan-500 border-cyan-500">
                  {rejectionBlocks.filter(rb => !rb.mitigated).length} active
                </Badge>
                <HelpTooltip content="Rejection blocks form when price tests a level and is forcefully rejected, leaving a long wick. The body of this candle becomes a zone where price may react on retest." />
              </h4>

              {rejectionBlocks.filter(rb => !rb.mitigated).slice(0, 3).map((rb, idx) => (
                <div
                  key={idx}
                  className={`pl-4 border-l-2 ${
                    rb.type === "bullish" ? "border-green-500" : "border-red-500"
                  } space-y-1`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${rb.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {rb.type === "bullish" ? "Bullish Rejection" : "Bearish Rejection"} @ {formatPrice(rb.rejection_price)}
                    </span>
                    {rb.wick_atr_ratio >= 1.5 && (
                      <Badge variant="secondary" className="text-cyan-400">Strong ({(rb.wick_atr_ratio).toFixed(1)}x ATR)</Badge>
                    )}
                    {rb.session && <Badge variant="outline">{rb.session}</Badge>}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {rb.type === "bullish" ? (
                      <>
                        Strong bullish rejection with {formatPrice(rb.wick_size)} wick. Body zone: {formatPrice(rb.body_bottom)} - {formatPrice(rb.body_top)}.
                        <span className="text-green-400 font-medium"> If price returns to this zone, expect bounce UP.</span>
                      </>
                    ) : (
                      <>
                        Strong bearish rejection with {formatPrice(rb.wick_size)} wick. Body zone: {formatPrice(rb.body_bottom)} - {formatPrice(rb.body_top)}.
                        <span className="text-red-400 font-medium"> If price returns to this zone, expect rejection DOWN.</span>
                      </>
                    )}
                  </div>
                </div>
              ))}

              <LearnMore title="Learn about Rejection Blocks">
                <p><strong>What is a Rejection Block?</strong></p>
                <p>
                  A rejection block forms when price tests a level and is forcefully pushed back, creating a long wick (shadow). The body of this candle marks a zone where institutions defended their position.
                </p>
                <p className="pt-2"><strong>How to trade rejection blocks:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Mark the body (open-close range) of the rejection candle</li>
                  <li>Wait for price to return to this zone</li>
                  <li>Enter in the direction of the original rejection</li>
                  <li>Stop beyond the rejection wick extreme</li>
                </ul>
                <p className="pt-2"><strong>Strength factors:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Longer wick = stronger rejection</li>
                  <li>Wick &gt; 1.5x ATR indicates significant rejection</li>
                  <li>Rejection at key level (OB, FVG) adds confluence</li>
                </ul>
              </LearnMore>
            </div>
          )}

          {/* Turtle Soup - Failed breakout reversals */}
          {turtleSoup.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Fish className="h-4 w-4 text-teal-500" />
                Turtle Soup Patterns
                <Badge variant="outline" className="text-teal-500 border-teal-500">
                  {turtleSoup.length} setups
                </Badge>
                <HelpTooltip content="Turtle Soup is a classic ICT/Linda Raschke pattern where price makes a new high/low beyond a prior swing, then immediately reverses. Named after trading against 'turtle traders' who enter on breakouts." />
              </h4>

              {turtleSoup.slice(0, 3).map((ts, idx) => (
                <div
                  key={idx}
                  className={`pl-4 border-l-2 ${
                    ts.type === "bullish" ? "border-green-500" : "border-red-500"
                  } space-y-1`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${ts.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {ts.type === "bullish" ? "Bullish Turtle Soup" : "Bearish Turtle Soup"} @ {formatPrice(ts.level)}
                    </span>
                    {ts.penetration_atr >= 0.5 && (
                      <Badge variant="secondary">Deep penetration</Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {ts.type === "bullish" ? (
                      <>
                        Price broke below prior swing low at {formatPrice(ts.level)} by {formatPrice(ts.penetration)}, then reversed.
                        <span className="text-green-400 font-medium"> Failed breakdown - bullish reversal setup.</span>
                      </>
                    ) : (
                      <>
                        Price broke above prior swing high at {formatPrice(ts.level)} by {formatPrice(ts.penetration)}, then reversed.
                        <span className="text-red-400 font-medium"> Failed breakout - bearish reversal setup.</span>
                      </>
                    )}
                  </div>
                </div>
              ))}

              <LearnMore title="Learn about Turtle Soup">
                <p><strong>What is Turtle Soup?</strong></p>
                <p>
                  Turtle Soup is a reversal pattern that fades breakout traders. When price makes a new high above a prior swing high (or new low below swing low) but immediately fails, it traps breakout traders and reverses.
                </p>
                <p className="pt-2"><strong>Why it&apos;s called Turtle Soup:</strong></p>
                <p>
                  Named after &quot;eating&quot; the famous Turtle Traders who entered on 20-day breakouts. Smart money fades these obvious breakouts for quick reversals.
                </p>
                <p className="pt-2"><strong>How to trade Turtle Soup:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bullish:</strong> Price breaks below prior low, fails, closes back above - go LONG</li>
                  <li><strong>Bearish:</strong> Price breaks above prior high, fails, closes back below - go SHORT</li>
                  <li>Stop beyond the failed breakout extreme</li>
                  <li>Target at least 1:2 risk/reward</li>
                </ul>
                <p className="pt-2 text-yellow-500">
                  <strong>Key:</strong> The breakout must fail quickly (same or next candle). A gradual failure is not Turtle Soup.
                </p>
              </LearnMore>
            </div>
          )}

          {/* NEW: Premium/Discount Zone */}
          {premiumDiscount && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Percent className="h-4 w-4" />
                Premium/Discount Zone
                <HelpTooltip content="Shows where price is relative to the current range. Buying in discount (below 50%) and selling in premium (above 50%) improves risk/reward. Equilibrium (50%) is neutral territory." />
              </h4>

              <div className="pl-4 border-l-2 border-blue-500 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium">Current Zone:</span>
                  <Badge variant={
                    premiumDiscount.current_zone === "discount" ? "buy" :
                    premiumDiscount.current_zone === "premium" ? "sell" : "secondary"
                  }>
                    {premiumDiscount.current_zone.toUpperCase()}
                  </Badge>
                  <span className="text-muted-foreground text-sm">
                    ({premiumDiscount.position_percent.toFixed(0)}% of range)
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center p-2 rounded bg-red-950/30">
                    <div className="text-red-400 font-medium">Premium</div>
                    <div className="font-mono">{formatPrice(premiumDiscount.premium_start)}</div>
                  </div>
                  <div className="text-center p-2 rounded bg-muted/30">
                    <div className="text-yellow-400 font-medium">Equilibrium</div>
                    <div className="font-mono">{formatPrice(premiumDiscount.equilibrium)}</div>
                  </div>
                  <div className="text-center p-2 rounded bg-green-950/30">
                    <div className="text-green-400 font-medium">Discount</div>
                    <div className="font-mono">{formatPrice(premiumDiscount.discount_end)}</div>
                  </div>
                </div>

                <p className="text-muted-foreground text-sm">
                  {premiumDiscount.current_zone === "discount"
                    ? "Price is in discount zone - favorable for BUY entries. Smart money accumulates positions here."
                    : premiumDiscount.current_zone === "premium"
                    ? "Price is in premium zone - favorable for SELL entries. Smart money distributes positions here."
                    : "Price is at equilibrium - no clear advantage. Wait for price to move to premium/discount for better entries."}
                </p>
              </div>

              <LearnMore title="Learn about Premium/Discount">
                <p><strong>Premium/Discount Zone Concept</strong></p>
                <p>
                  The range between a swing high and swing low is divided into zones. The 50% level is called Equilibrium (EQ).
                </p>
                <ul className="list-disc list-inside pl-2 space-y-1 pt-2">
                  <li><strong>Premium (above 50%):</strong> Price is &quot;expensive&quot; relative to the range. Favorable for selling.</li>
                  <li><strong>Discount (below 50%):</strong> Price is &quot;cheap&quot; relative to the range. Favorable for buying.</li>
                  <li><strong>Equilibrium (50%):</strong> Neutral zone. Neither buyers nor sellers have advantage.</li>
                </ul>
                <p className="pt-2">
                  <strong>Trading application:</strong> Enter longs in discount, shorts in premium. This gives you a statistical edge because you&apos;re buying low and selling high relative to the current range.
                </p>
              </LearnMore>
            </div>
          )}

          {/* NEW: Equal Highs/Lows */}
          {equalLevels.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Equal className="h-4 w-4" />
                Equal Highs/Lows (EQH/EQL)
                <Badge variant="outline">{equalLevels.filter(el => !el.swept).length} unswept</Badge>
                <HelpTooltip content="Equal highs (EQH) and equal lows (EQL) are major liquidity targets. When price touches the same level multiple times without breaking, stop losses accumulate there. Smart money targets these levels for liquidity." />
              </h4>

              {equalLevels.filter(el => !el.swept).slice(0, 3).map((el, idx) => (
                <div key={idx} className={`pl-4 border-l-2 ${el.type === "eqh" ? "border-purple-500" : "border-pink-500"} space-y-1`}>
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${el.type === "eqh" ? "text-purple-400" : "text-pink-400"}`}>
                      {el.type === "eqh" ? "Equal High" : "Equal Low"} @ {formatPrice(el.price)}
                    </span>
                    <Badge variant="outline">{el.touches} touches</Badge>
                    {el.touches >= 3 && (
                      <Badge variant="secondary" className="text-yellow-500">High Priority</Badge>
                    )}
                  </div>
                  <p className="text-muted-foreground text-sm">
                    {el.type === "eqh"
                      ? `Shorts have stops clustered above ${formatPrice(el.price)}. ${el.touches}+ tests makes this a high-priority target. Expect price to sweep this level and potentially reverse DOWN.`
                      : `Longs have stops clustered below ${formatPrice(el.price)}. ${el.touches}+ tests makes this a high-priority target. Expect price to sweep this level and potentially reverse UP.`}
                  </p>
                </div>
              ))}

              <LearnMore title="Learn about Equal Highs/Lows">
                <p><strong>What are Equal Highs/Lows?</strong></p>
                <p>
                  Equal highs (EQH) and equal lows (EQL) form when price touches the same level multiple times without breaking through. This creates &quot;obvious&quot; support/resistance that retail traders see.
                </p>
                <p className="pt-2"><strong>Why are they important?</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Stop losses cluster just beyond these levels</li>
                  <li>The more touches, the more liquidity accumulates</li>
                  <li>Smart money actively targets these pools</li>
                  <li>Often results in &quot;stop hunts&quot; followed by reversals</li>
                </ul>
                <p className="pt-2"><strong>Trading strategy:</strong></p>
                <p>
                  Don&apos;t place stops at obvious equal highs/lows. Instead, wait for the sweep, then look for reversal entries in the opposite direction after liquidity is taken.
                </p>
              </LearnMore>
            </div>
          )}

          {/* NEW: Breaker Blocks */}
          {breakerBlocks.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <RefreshCw className="h-4 w-4" />
                Breaker Blocks
                <Badge variant="outline">{breakerBlocks.length}</Badge>
                <HelpTooltip content="Breaker blocks are failed order blocks that 'flip' polarity. When an OB gets broken, it becomes a breaker - previous support becomes resistance and vice versa. High probability reversal zones." />
              </h4>

              {breakerBlocks.filter(bb => !bb.mitigated).slice(0, 2).map((bb, idx) => (
                <div key={idx} className={`pl-4 border-l-2 ${bb.type === "bullish" ? "border-green-500" : "border-red-500"} border-dashed space-y-1`}>
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${bb.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {bb.type === "bullish" ? "Bullish" : "Bearish"} Breaker @ {formatPrice(bb.bottom)} - {formatPrice(bb.top)}
                    </span>
                    {bb.strength && <Badge variant="outline">{Math.round(bb.strength * 100)}%</Badge>}
                  </div>
                  <p className="text-muted-foreground text-sm">
                    {bb.type === "bullish"
                      ? `Former bearish OB that got broken. Now acts as SUPPORT. If price returns here, expect a bounce UP. The flip confirms bullish momentum.`
                      : `Former bullish OB that got broken. Now acts as RESISTANCE. If price returns here, expect rejection DOWN. The flip confirms bearish momentum.`}
                  </p>
                </div>
              ))}

              <LearnMore title="Learn about Breaker Blocks">
                <p><strong>What is a Breaker Block?</strong></p>
                <p>
                  A breaker block forms when an order block fails - price breaks through it instead of respecting it. The failed OB then &quot;flips&quot; its role.
                </p>
                <p className="pt-2"><strong>How breakers form:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bullish Breaker:</strong> Bearish OB gets broken to the upside → becomes support</li>
                  <li><strong>Bearish Breaker:</strong> Bullish OB gets broken to the downside → becomes resistance</li>
                </ul>
                <p className="pt-2"><strong>Why breakers are powerful:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>They represent a shift in market structure</li>
                  <li>Traders who were wrong get trapped → fuel the new move</li>
                  <li>Often align with BOS/CHOCH for confluence</li>
                </ul>
                <p className="pt-2"><strong>Trading strategy:</strong></p>
                <p>
                  Trade breakers like order blocks but in the NEW direction. A bullish breaker is a buy zone, a bearish breaker is a sell zone. Stop goes beyond the breaker.
                </p>
              </LearnMore>
            </div>
          )}

          {/* NEW: OTE Zones */}
          {oteZones.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Ruler className="h-4 w-4" />
                Optimal Trade Entry (OTE) Zones
                <Badge variant="outline">{oteZones.length}</Badge>
                <HelpTooltip content="OTE is the 62-79% Fibonacci retracement zone of an impulse move. This is the 'sweet spot' where smart money enters after a pullback - not too early, not too late." />
              </h4>

              {oteZones.slice(0, 2).map((ote, idx) => (
                <div key={idx} className={`pl-4 border-l-2 ${ote.type === "bullish" ? "border-green-500" : "border-red-500"} space-y-2`}>
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${ote.type === "bullish" ? "text-green-400" : "text-red-400"}`}>
                      {ote.type === "bullish" ? "Bullish" : "Bearish"} OTE Zone
                    </span>
                    <HelpTooltip content={
                      ote.type === "bullish"
                        ? "Look for BUY entries when price retraces into this zone. The 70.5% level is the optimal entry point."
                        : "Look for SELL entries when price retraces into this zone. The 70.5% level is the optimal entry point."
                    } />
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs bg-muted/30 p-2 rounded">
                    <div className="text-center">
                      <div className="text-muted-foreground">61.8%</div>
                      <div className="font-mono">{formatPrice(ote.fib_618)}</div>
                    </div>
                    <div className="text-center">
                      <div className={`font-medium ${ote.type === "bullish" ? "text-green-400" : "text-red-400"}`}>70.5% (Optimal)</div>
                      <div className="font-mono">{formatPrice(ote.fib_705)}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-muted-foreground">79%</div>
                      <div className="font-mono">{formatPrice(ote.fib_79)}</div>
                    </div>
                  </div>

                  <p className="text-muted-foreground text-sm">
                    {ote.type === "bullish"
                      ? `Entry zone for LONG trades during pullbacks. Place limit orders in this zone with stops below the swing low.`
                      : `Entry zone for SHORT trades during rallies. Place limit orders in this zone with stops above the swing high.`}
                  </p>
                </div>
              ))}

              <LearnMore title="Learn about OTE">
                <p><strong>What is OTE (Optimal Trade Entry)?</strong></p>
                <p>
                  OTE is an ICT concept based on Fibonacci retracements. It&apos;s the zone between 62% and 79% retracement of an impulse move - the &quot;sweet spot&quot; for entries.
                </p>
                <p className="pt-2"><strong>Key levels:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>61.8%:</strong> The golden ratio - start of OTE zone</li>
                  <li><strong>70.5%:</strong> The optimal level - highest probability</li>
                  <li><strong>79%:</strong> Deep retracement - last chance entry</li>
                </ul>
                <p className="pt-2"><strong>How to use OTE:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>After an impulse move, wait for a pullback</li>
                  <li>Draw Fib from swing low to swing high (bullish) or vice versa</li>
                  <li>Enter in the 62-79% zone, optimal at 70.5%</li>
                  <li>Stop loss goes beyond the swing point</li>
                </ul>
                <p className="pt-2"><strong>Why OTE works:</strong></p>
                <p>
                  Smart money doesn&apos;t chase price. They wait for retracements to get better entries. The 70.5% level gives excellent risk/reward while still confirming the trend is intact.
                </p>
              </LearnMore>
            </div>
          )}

          {/* NEW: Confluence Score */}
          {confluenceScore && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Gauge className="h-4 w-4" />
                Confluence Score
                <HelpTooltip content="A composite score measuring how many SMC factors align. Higher scores indicate stronger trade setups. Score of 7+ suggests high-probability opportunity." />
              </h4>

              <div className="pl-4 border-l-2 border-yellow-500 space-y-3">
                <div className="flex items-center gap-3">
                  <div className={`text-3xl font-bold ${
                    confluenceScore.total_score >= 7 ? "text-green-400" :
                    confluenceScore.total_score >= 5 ? "text-yellow-400" : "text-red-400"
                  }`}>
                    {confluenceScore.total_score.toFixed(1)}
                  </div>
                  <div>
                    <Badge variant={
                      confluenceScore.total_score >= 7 ? "buy" :
                      confluenceScore.total_score >= 5 ? "secondary" : "sell"
                    }>
                      {confluenceScore.total_score >= 7 ? "HIGH PROBABILITY" :
                       confluenceScore.total_score >= 5 ? "MODERATE" : "LOW PROBABILITY"}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      Recommendation: <span className="font-medium">{confluenceScore.recommendation}</span>
                    </p>
                  </div>
                </div>

                {/* Score breakdown */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {confluenceScore.bias_alignment > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>Bias Alignment</span>
                      <span className="font-mono text-green-400">+{confluenceScore.bias_alignment.toFixed(1)}</span>
                    </div>
                  )}
                  {confluenceScore.zone_proximity > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>Zone Proximity</span>
                      <span className="font-mono text-green-400">+{confluenceScore.zone_proximity.toFixed(1)}</span>
                    </div>
                  )}
                  {confluenceScore.structure_confirmation > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>Structure</span>
                      <span className="font-mono text-green-400">+{confluenceScore.structure_confirmation.toFixed(1)}</span>
                    </div>
                  )}
                  {confluenceScore.liquidity_target > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>Liquidity Target</span>
                      <span className="font-mono text-green-400">+{confluenceScore.liquidity_target.toFixed(1)}</span>
                    </div>
                  )}
                  {confluenceScore.mtf_alignment > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>MTF Alignment</span>
                      <span className="font-mono text-green-400">+{confluenceScore.mtf_alignment.toFixed(1)}</span>
                    </div>
                  )}
                  {confluenceScore.session_factor > 0 && (
                    <div className="flex justify-between bg-muted/30 p-2 rounded">
                      <span>Session Factor</span>
                      <span className="font-mono text-green-400">+{confluenceScore.session_factor.toFixed(1)}</span>
                    </div>
                  )}
                </div>

                <p className="text-muted-foreground text-sm">
                  {confluenceScore.total_score >= 7
                    ? "Multiple SMC factors align - this is a high-probability setup. Consider taking the trade with proper risk management."
                    : confluenceScore.total_score >= 5
                    ? "Some factors align but not all. Wait for additional confluence or reduce position size."
                    : "Few factors align - low probability setup. Better to wait for clearer opportunities."}
                </p>
              </div>

              <LearnMore title="Learn about Confluence">
                <p><strong>What is Confluence?</strong></p>
                <p>
                  Confluence means multiple trading factors pointing in the same direction. The more factors that align, the higher the probability of success.
                </p>
                <p className="pt-2"><strong>Confluence factors:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bias Alignment:</strong> Trade direction matches overall market bias</li>
                  <li><strong>Zone Proximity:</strong> Price is at a key SMC zone (OB, FVG)</li>
                  <li><strong>Structure:</strong> BOS/CHOCH confirms direction</li>
                  <li><strong>Liquidity Target:</strong> Clear target for take profit</li>
                  <li><strong>MTF Alignment:</strong> Higher timeframes agree</li>
                  <li><strong>Session:</strong> Trading during active sessions (London/NY)</li>
                </ul>
                <p className="pt-2"><strong>Score interpretation:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>7-10:</strong> High probability - green light to trade</li>
                  <li><strong>5-7:</strong> Moderate - trade with caution</li>
                  <li><strong>Below 5:</strong> Low probability - better to wait</li>
                </ul>
              </LearnMore>
            </div>
          )}

          {/* Order Block Analysis */}
          {(bullishOBs.length > 0 || bearishOBs.length > 0) && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Order Blocks (Institutional Zones)
                <HelpTooltip content="Order blocks are the last candle before a strong move - they mark where institutions entered. Price often returns to these zones and bounces." />
              </h4>

              {nearestBullishOB && (
                <div className="pl-4 border-l-2 border-green-500 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-green-400">
                      Bullish OB @ {formatPrice(nearestBullishOB.bottom)} - {formatPrice(nearestBullishOB.top)}
                    </span>
                    {nearestBullishOB.strength && <Badge variant="outline">{Math.round(nearestBullishOB.strength * 100)}%</Badge>}
                  </div>
                  <p className="text-muted-foreground">
                    This is the last down candle before a strong bullish move.
                    {currentPrice - nearestBullishOB.top < (nearestBullishOB.top - nearestBullishOB.bottom) * 2
                      ? " Price is near this zone - expect buyers to defend and push price UP if it enters. Good area for buy entries with stop below the OB."
                      : " If price drops to this zone, expect a bullish reaction as institutions defend their entries."}
                  </p>
                </div>
              )}

              {nearestBearishOB && (
                <div className="pl-4 border-l-2 border-red-500 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-red-400">
                      Bearish OB @ {formatPrice(nearestBearishOB.bottom)} - {formatPrice(nearestBearishOB.top)}
                    </span>
                    {nearestBearishOB.strength && <Badge variant="outline">{Math.round(nearestBearishOB.strength * 100)}%</Badge>}
                  </div>
                  <p className="text-muted-foreground">
                    This is the last up candle before a strong bearish move.
                    {nearestBearishOB.bottom - currentPrice < (nearestBearishOB.top - nearestBearishOB.bottom) * 2
                      ? " Price is near this zone - expect sellers to defend and push price DOWN if it enters. Good area for sell entries with stop above the OB."
                      : " If price rises to this zone, expect a bearish reaction as institutions defend their entries."}
                  </p>
                </div>
              )}

              {bullishOBs.length === 0 && bearishOBs.length === 0 && (
                <p className="text-muted-foreground pl-4">
                  No unmitigated order blocks detected. Previous institutional zones have been revisited (mitigated).
                </p>
              )}

              <LearnMore title="Learn more about Order Blocks">
                <p><strong>What is an Order Block?</strong></p>
                <p>
                  An Order Block is the last candle of opposite color before a strong impulsive move. It marks where institutional traders placed their orders before driving price in their intended direction.
                </p>

                <p className="pt-2"><strong>Why do Order Blocks work?</strong></p>
                <p>
                  Institutions can&apos;t execute their entire position at once due to size. They accumulate positions, push price, then often return to add to their position at the same price level. They also defend these zones to protect their entries.
                </p>

                <p className="pt-2"><strong>How to identify valid Order Blocks:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bullish OB:</strong> The last bearish (red) candle before a strong move UP</li>
                  <li><strong>Bearish OB:</strong> The last bullish (green) candle before a strong move DOWN</li>
                  <li>The move after must break structure (BOS) to validate the OB</li>
                  <li>Stronger moves after the OB = stronger OB</li>
                </ul>

                <p className="pt-2"><strong>How to trade Order Blocks:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Wait for price to return to the OB zone (retest)</li>
                  <li>For bullish OBs: Look for buy entries at or slightly below the OB top</li>
                  <li>For bearish OBs: Look for sell entries at or slightly above the OB bottom</li>
                  <li>Stop loss goes beyond the OB (below for bullish, above for bearish)</li>
                  <li>Target the next liquidity level or opposing OB</li>
                </ul>

                <p className="pt-2"><strong>Order Block strength factors:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Unmitigated OBs (never retested) are stronger</li>
                  <li>OBs at premium/discount zones are more reliable</li>
                  <li>OBs that caused BOS are validated</li>
                  <li>Multiple timeframe confluence increases probability</li>
                </ul>

                <p className="pt-2"><strong>Mitigation:</strong></p>
                <p>
                  When price returns to an OB and the zone is touched, it becomes &quot;mitigated.&quot; After mitigation, the OB loses its power - institutions have likely exited or adjusted their positions.
                </p>
              </LearnMore>
            </div>
          )}

          {/* Fair Value Gap Analysis */}
          {(bullishFVGs.length > 0 || bearishFVGs.length > 0) && (
            <div className="space-y-3">
              <h4 className="font-medium flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Fair Value Gaps (Imbalances)
                <HelpTooltip content="FVGs are price imbalances from rapid moves where price didn't trade fairly. Price often returns to 'fill' these gaps before continuing." />
              </h4>

              {nearestBullishFVG && (
                <div className="pl-4 border-l-2 border-green-500/50 border-dashed space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-green-400">
                      Bullish FVG @ {formatPrice(nearestBullishFVG.bottom)} - {formatPrice(nearestBullishFVG.top)}
                    </span>
                  </div>
                  <p className="text-muted-foreground">
                    A gap left by rapid upward movement. This is an area where price moved so fast that no proper trading occurred.
                    {currentPrice - nearestBullishFVG.top < (nearestBullishFVG.top - nearestBullishFVG.bottom) * 3
                      ? " Price is close to this imbalance. If it drops into this zone, expect a bounce UP as the gap gets 'filled' and balance is restored."
                      : " If price retraces to fill this gap, it often acts as support with price bouncing UP."}
                  </p>
                </div>
              )}

              {nearestBearishFVG && (
                <div className="pl-4 border-l-2 border-red-500/50 border-dashed space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-red-400">
                      Bearish FVG @ {formatPrice(nearestBearishFVG.bottom)} - {formatPrice(nearestBearishFVG.top)}
                    </span>
                  </div>
                  <p className="text-muted-foreground">
                    A gap left by rapid downward movement. Price moved so fast that no proper trading occurred in this zone.
                    {nearestBearishFVG.bottom - currentPrice < (nearestBearishFVG.top - nearestBearishFVG.bottom) * 3
                      ? " Price is close to this imbalance. If it rises into this zone, expect a rejection DOWN as the gap gets 'filled'."
                      : " If price retraces to fill this gap, it often acts as resistance with price being pushed DOWN."}
                  </p>
                </div>
              )}

              {bullishFVGs.length + bearishFVGs.length > 2 && (
                <p className="text-muted-foreground pl-4">
                  Multiple unfilled FVGs detected ({bullishFVGs.length} bullish, {bearishFVGs.length} bearish).
                  Price has moved rapidly and may consolidate to fill these imbalances before the next major move.
                </p>
              )}

              <LearnMore title="Learn more about Fair Value Gaps">
                <p><strong>What is a Fair Value Gap?</strong></p>
                <p>
                  An FVG is a 3-candle pattern where the middle candle moves so aggressively that a gap forms between the first and third candles. This gap represents a price imbalance where no actual trading occurred.
                </p>

                <p className="pt-2"><strong>Why do FVGs form?</strong></p>
                <p>
                  They occur during periods of high momentum when institutions move price rapidly. The market doesn&apos;t have time to establish fair value, creating an &quot;imbalance&quot; that often needs to be corrected.
                </p>

                <p className="pt-2"><strong>How to identify an FVG:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li><strong>Bullish FVG:</strong> The low of candle 3 is ABOVE the high of candle 1 (gap above)</li>
                  <li><strong>Bearish FVG:</strong> The high of candle 3 is BELOW the low of candle 1 (gap below)</li>
                </ul>

                <p className="pt-2"><strong>How to trade FVGs:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Wait for price to return to the FVG zone (a &quot;fill&quot; or &quot;rebalance&quot;)</li>
                  <li>For bullish FVGs: Look for buy entries when price dips into the zone, expecting a bounce UP</li>
                  <li>For bearish FVGs: Look for sell entries when price rallies into the zone, expecting rejection DOWN</li>
                  <li>Place stops beyond the FVG zone (below for bullish, above for bearish)</li>
                </ul>

                <p className="pt-2"><strong>FVG reliability factors:</strong></p>
                <ul className="list-disc list-inside pl-2 space-y-1">
                  <li>Higher timeframe FVGs are more significant</li>
                  <li>FVGs from strong impulse moves are more likely to hold</li>
                  <li>FVGs that align with order blocks provide confluence</li>
                  <li>Partially filled FVGs may still provide reaction</li>
                </ul>
              </LearnMore>
            </div>
          )}

          {/* Trading Implications & Contextual Tips */}
          <div className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-yellow-500" />
              Trading Implications
            </h4>
            <div className="pl-4 space-y-3 text-muted-foreground">
              {/* Main setup suggestions */}
              {overallBias === "bullish" && nearestBullishOB && (
                <p>
                  <span className="text-green-400 font-medium">Bullish Setup:</span> Look for long entries at the bullish OB ({formatPrice(nearestBullishOB.top)}) with stop loss below {formatPrice(nearestBullishOB.bottom)}.
                  {nearestBSL && ` Target the BSL at ${formatPrice(nearestBSL.price)} for take profit.`}
                </p>
              )}
              {overallBias === "bearish" && nearestBearishOB && (
                <p>
                  <span className="text-red-400 font-medium">Bearish Setup:</span> Look for short entries at the bearish OB ({formatPrice(nearestBearishOB.bottom)}) with stop loss above {formatPrice(nearestBearishOB.top)}.
                  {nearestSSL && ` Target the SSL at ${formatPrice(nearestSSL.price)} for take profit.`}
                </p>
              )}
              {nearestBSL && nearestSSL && (
                <p>
                  <span className="text-yellow-400 font-medium">Liquidity Hunt:</span> Price is between BSL ({formatPrice(nearestBSL.price)}) and SSL ({formatPrice(nearestSSL.price)}).
                  Watch which side gets swept first - smart money often reverses after grabbing liquidity.
                </p>
              )}

              {/* Contextual Tips - appear based on specific scenarios */}
              {(() => {
                const tips: React.ReactNode[] = []

                // Confluence detection - OB + FVG overlap
                if (nearestBullishOB && nearestBullishFVG) {
                  const overlap = nearestBullishOB.top >= nearestBullishFVG.bottom && nearestBullishOB.bottom <= nearestBullishFVG.top
                  if (overlap) {
                    tips.push(
                      <div key="confluence-bull" className="bg-green-950/30 border border-green-800/50 rounded-md p-2">
                        <span className="text-green-400 font-medium">💡 Confluence Zone:</span>
                        {" "}Bullish OB and FVG overlap around {formatPrice(nearestBullishOB.top)}. Multiple SMC concepts aligning increases the probability of a strong bounce from this area.
                      </div>
                    )
                  }
                }
                if (nearestBearishOB && nearestBearishFVG) {
                  const overlap = nearestBearishOB.top >= nearestBearishFVG.bottom && nearestBearishOB.bottom <= nearestBearishFVG.top
                  if (overlap) {
                    tips.push(
                      <div key="confluence-bear" className="bg-red-950/30 border border-red-800/50 rounded-md p-2">
                        <span className="text-red-400 font-medium">💡 Confluence Zone:</span>
                        {" "}Bearish OB and FVG overlap around {formatPrice(nearestBearishOB.bottom)}. Multiple SMC concepts aligning increases the probability of a strong rejection from this area.
                      </div>
                    )
                  }
                }

                // Price proximity alerts
                if (nearestBullishOB && currentPrice - nearestBullishOB.top < (nearestBullishOB.top - nearestBullishOB.bottom)) {
                  tips.push(
                    <div key="proximity-bull-ob" className="bg-yellow-950/30 border border-yellow-800/50 rounded-md p-2">
                      <span className="text-yellow-400 font-medium">⚠️ Price Alert:</span>
                      {" "}Price is approaching the bullish OB. If price enters this zone, watch for bullish candlestick patterns (engulfing, pin bar) for entry confirmation.
                    </div>
                  )
                }
                if (nearestBearishOB && nearestBearishOB.bottom - currentPrice < (nearestBearishOB.top - nearestBearishOB.bottom)) {
                  tips.push(
                    <div key="proximity-bear-ob" className="bg-yellow-950/30 border border-yellow-800/50 rounded-md p-2">
                      <span className="text-yellow-400 font-medium">⚠️ Price Alert:</span>
                      {" "}Price is approaching the bearish OB. If price enters this zone, watch for bearish candlestick patterns (shooting star, bearish engulfing) for entry confirmation.
                    </div>
                  )
                }

                // Liquidity sweep potential
                if (nearestBSL && (nearestBSL.price - currentPrice) / currentPrice < 0.003) {
                  tips.push(
                    <div key="bsl-close" className="bg-purple-950/30 border border-purple-800/50 rounded-md p-2">
                      <span className="text-purple-400 font-medium">🎯 Stop Hunt Setup:</span>
                      {" "}BSL at {formatPrice(nearestBSL.price)} is very close ({((nearestBSL.price - currentPrice) / currentPrice * 100).toFixed(2)}% away). If price sweeps above, watch for immediate reversal - this is a classic liquidity grab setup for shorts.
                    </div>
                  )
                }
                if (nearestSSL && (currentPrice - nearestSSL.price) / currentPrice < 0.003) {
                  tips.push(
                    <div key="ssl-close" className="bg-pink-950/30 border border-pink-800/50 rounded-md p-2">
                      <span className="text-pink-400 font-medium">🎯 Stop Hunt Setup:</span>
                      {" "}SSL at {formatPrice(nearestSSL.price)} is very close ({((currentPrice - nearestSSL.price) / currentPrice * 100).toFixed(2)}% away). If price sweeps below, watch for immediate reversal - this is a classic liquidity grab setup for longs.
                    </div>
                  )
                }

                // High strength zone alerts
                const strongBSL = unsweptBSL.find(lz => (lz.strength || 0) >= 80)
                const strongSSL = unsweptSSL.find(lz => (lz.strength || 0) >= 80)
                if (strongBSL) {
                  tips.push(
                    <div key="strong-bsl" className="bg-muted/50 rounded-md p-2">
                      <span className="font-medium">📊 Strong Liquidity:</span>
                      {" "}High-strength BSL ({Math.round(strongBSL.strength || 0)}%) at {formatPrice(strongBSL.price)} - this is a major target. Expect significant buying pressure if swept, potentially fueling a reversal.
                    </div>
                  )
                }
                if (strongSSL) {
                  tips.push(
                    <div key="strong-ssl" className="bg-muted/50 rounded-md p-2">
                      <span className="font-medium">📊 Strong Liquidity:</span>
                      {" "}High-strength SSL ({Math.round(strongSSL.strength || 0)}%) at {formatPrice(strongSSL.price)} - this is a major target. Expect significant selling pressure if swept, potentially fueling a reversal.
                    </div>
                  )
                }

                // Mitigated zones warning
                const mitigatedOBs = orderBlocks.filter(ob => ob.mitigated).length
                const mitigatedFVGs = fairValueGaps.filter(fvg => fvg.mitigated).length
                if (mitigatedOBs > 0 || mitigatedFVGs > 0) {
                  tips.push(
                    <div key="mitigated-warning" className="bg-muted/50 rounded-md p-2">
                      <span className="font-medium">ℹ️ Mitigated Zones:</span>
                      {" "}{mitigatedOBs > 0 && `${mitigatedOBs} order block${mitigatedOBs > 1 ? "s" : ""}`}
                      {mitigatedOBs > 0 && mitigatedFVGs > 0 && " and "}
                      {mitigatedFVGs > 0 && `${mitigatedFVGs} FVG${mitigatedFVGs > 1 ? "s" : ""}`}
                      {" "}already touched. These zones have reduced strength - price may pass through more easily on the next test.
                    </div>
                  )
                }

                return tips.length > 0 ? tips : null
              })()}

              {orderBlocks.length === 0 && fairValueGaps.length === 0 && liquidityZones.length === 0 && (
                <p>
                  No significant SMC levels detected on this timeframe. Consider checking a higher timeframe for institutional levels, or wait for new structure to form.
                </p>
              )}
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4 pt-4 border-t">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">{bullishOBs.length + bullishFVGs.length}</div>
              <div className="text-xs text-muted-foreground">Support Zones</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-500">{bearishOBs.length + bearishFVGs.length}</div>
              <div className="text-xs text-muted-foreground">Resistance Zones</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-500">{unsweptBSL.length + unsweptSSL.length}</div>
              <div className="text-xs text-muted-foreground">Liquidity Targets</div>
            </div>
          </div>

          {/* Extended Stats (NEW features) */}
          {(equalLevels.length > 0 || breakerBlocks.length > 0 || oteZones.length > 0) && (
            <div className="grid grid-cols-3 gap-4 pt-2">
              <div className="text-center">
                <div className="text-lg font-bold text-blue-500">{equalLevels.length}</div>
                <div className="text-xs text-muted-foreground">EQH/EQL</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-orange-500">{breakerBlocks.filter(bb => !bb.mitigated).length}</div>
                <div className="text-xs text-muted-foreground">Breakers</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-yellow-500">{oteZones.length}</div>
                <div className="text-xs text-muted-foreground">OTE Zones</div>
              </div>
            </div>
          )}

          {/* Advanced Pattern Stats */}
          {(liquiditySweeps.length > 0 || inducements.length > 0 || rejectionBlocks.length > 0 || turtleSoup.length > 0 || alerts.length > 0) && (
            <div className="grid grid-cols-5 gap-2 pt-2">
              {alerts.length > 0 && (
                <div className="text-center">
                  <div className="text-lg font-bold text-yellow-500">{alerts.length}</div>
                  <div className="text-xs text-muted-foreground">Alerts</div>
                </div>
              )}
              {liquiditySweeps.length > 0 && (
                <div className="text-center">
                  <div className="text-lg font-bold text-amber-500">{liquiditySweeps.length}</div>
                  <div className="text-xs text-muted-foreground">Sweeps</div>
                </div>
              )}
              {inducements.length > 0 && (
                <div className="text-center">
                  <div className="text-lg font-bold text-orange-500">{inducements.length}</div>
                  <div className="text-xs text-muted-foreground">Inducements</div>
                </div>
              )}
              {rejectionBlocks.filter(rb => !rb.mitigated).length > 0 && (
                <div className="text-center">
                  <div className="text-lg font-bold text-cyan-500">{rejectionBlocks.filter(rb => !rb.mitigated).length}</div>
                  <div className="text-xs text-muted-foreground">Rejections</div>
                </div>
              )}
              {turtleSoup.length > 0 && (
                <div className="text-center">
                  <div className="text-lg font-bold text-teal-500">{turtleSoup.length}</div>
                  <div className="text-xs text-muted-foreground">Turtle Soup</div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </ScrollArea>

      {/* FVG Order Dialog */}
      <FvgOrderDialog
        open={orderDialogOpen}
        onOpenChange={setOrderDialogOpen}
        setup={selectedSetup}
        symbol={symbol}
        digits={digits}
      />
    </Card>
  )
}
