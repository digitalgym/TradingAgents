"use client"

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Loader2,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  Calculator,
  Target,
  Shield,
  DollarSign,
} from "lucide-react"
import {
  placeMarketOrder,
  placeLimitOrder,
  calculateTradeSize,
  getSymbolInfo,
  getSwingLevels,
  saveTradeDecision,
  markDecisionRetried,
  getMarketStatus,
} from "@/lib/api"

interface SmcLevel {
  type: "order_block" | "fvg" | "liquidity" | "breaker" | "support" | "resistance" | "pdl" | "pdh"
  price: number
  direction: "bullish" | "bearish"
  strength?: number
  description?: string
}

interface TradeExecutionWizardProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  symbol: string
  signal: "BUY" | "SELL" | "HOLD"
  suggestedEntry?: number
  suggestedStopLoss?: number
  suggestedTakeProfit?: number
  rationale?: string
  smcLevels?: SmcLevel[]
  currentPrice?: number
  analysisContext?: any  // Full analysis state for reflection/learning
  confidence?: number  // Signal confidence 0-1
  failureReason?: string  // If retrying a failed trade, shows why it failed
  failedDecisionId?: string  // Decision ID of the failed trade being retried
}

export function TradeExecutionWizard({
  open,
  onOpenChange,
  symbol,
  signal,
  suggestedEntry,
  suggestedStopLoss,
  suggestedTakeProfit,
  rationale,
  smcLevels = [],
  currentPrice: propCurrentPrice,
  analysisContext,
  confidence,
  failureReason,
  failedDecisionId,
}: TradeExecutionWizardProps) {
  // Form state
  const [entryPrice, setEntryPrice] = useState<string>("")
  const [stopLoss, setStopLoss] = useState<string>("")
  const [takeProfit, setTakeProfit] = useState<string>("")
  const [pullbackEntry, setPullbackEntry] = useState<string>("")
  const [volume, setVolume] = useState<string>("0.01")
  const [riskPercent, setRiskPercent] = useState<string>("1.0")

  // Calculated values
  const [positionSize, setPositionSize] = useState<any>(null)
  const [symbolInfo, setSymbolInfo] = useState<any>(null)
  const [swingLevels, setSwingLevels] = useState<any[]>([])
  const [swingLevelsLoading, setSwingLevelsLoading] = useState(false)

  // UI state
  const [loading, setLoading] = useState(false)
  const [calculating, setCalculating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>("market")
  const [pullbackError, setPullbackError] = useState<string | null>(null)
  const [stopLossError, setStopLossError] = useState<string | null>(null)
  const [marketClosed, setMarketClosed] = useState<string | null>(null)

  // Get current price from props or symbol info
  const currentPrice = propCurrentPrice || (signal === "BUY" ? symbolInfo?.ask : symbolInfo?.bid) || parseFloat(entryPrice)

  // Filter SMC levels relevant for the trade direction
  const relevantSmcLevels = smcLevels.filter((level) => {
    if (signal === "BUY") {
      // For BUY, we want bullish levels below current price (support, PDL, bullish OB/FVG)
      const isBullishLevel = level.direction === "bullish" ||
                             level.type === "fvg" ||
                             level.type === "support" ||
                             level.type === "pdl"
      return level.price < currentPrice && isBullishLevel
    } else {
      // For SELL, we want bearish levels above current price (resistance, PDH, bearish OB/FVG)
      const isBearishLevel = level.direction === "bearish" ||
                             level.type === "fvg" ||
                             level.type === "resistance" ||
                             level.type === "pdh"
      return level.price > currentPrice && isBearishLevel
    }
  }).sort((a, b) => {
    // Sort by strength first (higher strength = more important), then by distance
    const strengthDiff = (b.strength || 0.5) - (a.strength || 0.5)
    if (Math.abs(strengthDiff) > 0.1) return strengthDiff
    // Then sort by distance from current price (nearest first)
    return Math.abs(a.price - currentPrice) - Math.abs(b.price - currentPrice)
  })

  // Get the best SMC level for pullback entry
  const suggestedSmcLevel = relevantSmcLevels[0]

  // Validate pullback entry price
  const validatePullbackEntry = (price: string): string | null => {
    const priceNum = parseFloat(price)
    if (!price || isNaN(priceNum)) return null

    if (!currentPrice) return null

    if (signal === "BUY" && priceNum >= currentPrice) {
      return `For a BUY order, pullback entry must be below current price (${currentPrice.toFixed(symbolInfo?.digits || 5)})`
    }
    if (signal === "SELL" && priceNum <= currentPrice) {
      return `For a SELL order, pullback entry must be above current price (${currentPrice.toFixed(symbolInfo?.digits || 5)})`
    }
    return null
  }

  // Validate stop loss against entry price (must be on correct side)
  const validateStopLoss = (sl: string, entry: number): string | null => {
    const slNum = parseFloat(sl)
    if (!sl || isNaN(slNum) || !entry) return null

    const digits = symbolInfo?.digits || 5

    if (signal === "BUY") {
      if (slNum >= entry) {
        return `For a BUY order, stop loss (${slNum.toFixed(digits)}) must be BELOW entry price (${entry.toFixed(digits)})`
      }
    } else if (signal === "SELL") {
      if (slNum <= entry) {
        return `For a SELL order, stop loss (${slNum.toFixed(digits)}) must be ABOVE entry price (${entry.toFixed(digits)})`
      }
    }
    return null
  }

  // Initialize form with suggested values
  useEffect(() => {
    if (open) {
      setEntryPrice(suggestedEntry?.toString() || "")
      setStopLoss(suggestedStopLoss?.toString() || "")
      setTakeProfit(suggestedTakeProfit?.toString() || "")
      setError(null)
      setSuccess(null)
      setPullbackError(null)
      setSwingLevels([])

      // Fetch symbol info and swing levels
      fetchSymbolInfo()
      if (signal !== "HOLD") {
        fetchSwingLevels()
      }
    }
  }, [open, suggestedEntry, suggestedStopLoss, suggestedTakeProfit])

  const fetchSwingLevels = async () => {
    if (!symbol || signal === "HOLD") return
    setSwingLevelsLoading(true)
    const { data, error } = await getSwingLevels(symbol, signal, "H1")
    if (data && data.levels) {
      setSwingLevels(data.levels)
    }
    setSwingLevelsLoading(false)
  }

  // Set pullback entry based on best SMC level when available
  useEffect(() => {
    if (open && suggestedSmcLevel && !pullbackEntry) {
      setPullbackEntry(suggestedSmcLevel.price.toString())
    }
  }, [open, suggestedSmcLevel])

  // Validate pullback entry when it changes
  useEffect(() => {
    if (pullbackEntry) {
      const validationError = validatePullbackEntry(pullbackEntry)
      setPullbackError(validationError)
    } else {
      setPullbackError(null)
    }
  }, [pullbackEntry, currentPrice, signal])

  // Validate stop loss against actual market price (not displayed entry which may be stale)
  useEffect(() => {
    if (!stopLoss || !symbolInfo) {
      setStopLossError(null)
      return
    }

    // Use actual market price for validation, not the potentially stale entry price
    const actualMarketPrice = signal === "BUY" ? symbolInfo.ask : symbolInfo.bid
    const entryToValidate = activeTab === "market"
      ? actualMarketPrice || parseFloat(entryPrice)
      : parseFloat(pullbackEntry)

    if (entryToValidate) {
      const slError = validateStopLoss(stopLoss, entryToValidate)
      setStopLossError(slError)
    } else {
      setStopLossError(null)
    }
  }, [stopLoss, symbolInfo, signal, activeTab, entryPrice, pullbackEntry])

  // Calculate position size when parameters change
  useEffect(() => {
    const entry = activeTab === "market" ? parseFloat(entryPrice) : parseFloat(pullbackEntry)
    const sl = parseFloat(stopLoss)
    const tp = parseFloat(takeProfit) || undefined
    const risk = parseFloat(riskPercent)

    if (entry && sl && risk && symbol) {
      calculatePositionSize(entry, sl, risk, tp)
    }
  }, [entryPrice, pullbackEntry, stopLoss, takeProfit, riskPercent, activeTab, symbol])

  const fetchSymbolInfo = async () => {
    // Check market status
    const { data: mktData } = await getMarketStatus(symbol)
    if (mktData && !mktData.open) {
      setMarketClosed(mktData.reason)
    } else {
      setMarketClosed(null)
    }

    const { data, error } = await getSymbolInfo(symbol)
    if (data) {
      setSymbolInfo(data)
      const marketPrice = signal === "BUY" ? data.ask : data.bid

      // For market orders, always use current market price since that's where we'll actually execute
      // The suggested entry is just informational - actual execution is at market
      if (marketPrice) {
        setEntryPrice(marketPrice.toString())
      }

      // Also validate SL and TP - clear if they're obviously wrong
      if (suggestedStopLoss && marketPrice) {
        const slDeviation = Math.abs(suggestedStopLoss - marketPrice) / marketPrice
        if (slDeviation > 0.5) {
          // SL is more than 50% away from market - likely hallucinated
          setStopLoss("")
        } else {
          // Check if SL is on the wrong side of entry
          const slOnWrongSide = signal === "BUY"
            ? suggestedStopLoss >= marketPrice
            : suggestedStopLoss <= marketPrice
          if (slOnWrongSide) {
            // SL is on wrong side of entry - clear it so user must set valid SL
            setStopLoss("")
          }
        }
      }
      if (suggestedTakeProfit && marketPrice) {
        const tpDeviation = Math.abs(suggestedTakeProfit - marketPrice) / marketPrice
        if (tpDeviation > 0.5) {
          // TP is more than 50% away from market - likely hallucinated
          setTakeProfit("")
        }
      }
    }
  }

  const calculatePositionSize = async (entry: number, sl: number, risk: number, tp?: number) => {
    setCalculating(true)
    const { data, error } = await calculateTradeSize({
      symbol,
      entry_price: entry,
      stop_loss: sl,
      take_profit: tp,  // Include TP for actual R:R calculation
      risk_percent: risk,
    })
    if (data) {
      setPositionSize(data)
      setVolume(data.position_size.toString())
    }
    setCalculating(false)
  }

  const handleExecuteMarket = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    const { data, error: apiError } = await placeMarketOrder({
      symbol,
      direction: signal as "BUY" | "SELL",
      volume: parseFloat(volume),
      stop_loss: parseFloat(stopLoss) || undefined,
      take_profit: parseFloat(takeProfit) || undefined,
      comment: "TradingAgents Analysis",
    })

    if (apiError) {
      setError(apiError)
    } else if (data) {
      // Save decision for learning/tracking (includes analysis context for reflection)
      const decisionResult = await saveTradeDecision({
        symbol,
        action: signal as "BUY" | "SELL",
        entry_type: "market",
        entry_price: data.price,
        stop_loss: parseFloat(stopLoss) || undefined,
        take_profit: parseFloat(takeProfit) || undefined,
        volume: parseFloat(volume),
        mt5_ticket: data.order_ticket,
        rationale: rationale,
        risk_percent: parseFloat(riskPercent) || undefined,
        confidence: confidence,
        analysis_context: analysisContext,  // Full state for reflection/learning
      })

      // Mark original failed decision as retried
      if (failedDecisionId) {
        await markDecisionRetried(failedDecisionId)
      }

      const savedMsg = decisionResult.data?.decision_id
        ? ` | Decision saved for learning`
        : ""
      setSuccess(`Order executed! Ticket: ${data.order_ticket}, Price: ${data.price}${savedMsg}`)
    }
    setLoading(false)
  }

  const handleExecuteLimit = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    const { data, error: apiError } = await placeLimitOrder({
      symbol,
      direction: signal as "BUY" | "SELL",
      volume: parseFloat(volume),
      entry_price: parseFloat(pullbackEntry),
      stop_loss: parseFloat(stopLoss) || undefined,
      take_profit: parseFloat(takeProfit) || undefined,
      comment: "TradingAgents Pullback",
    })

    if (apiError) {
      setError(apiError)
    } else if (data) {
      // Save decision for learning/tracking (includes analysis context for reflection)
      const decisionResult = await saveTradeDecision({
        symbol,
        action: signal as "BUY" | "SELL",
        entry_type: "limit",
        entry_price: parseFloat(pullbackEntry),
        stop_loss: parseFloat(stopLoss) || undefined,
        take_profit: parseFloat(takeProfit) || undefined,
        volume: parseFloat(volume),
        mt5_ticket: data.order_ticket,
        rationale: rationale,
        risk_percent: parseFloat(riskPercent) || undefined,
        confidence: confidence,
        analysis_context: analysisContext,  // Full state for reflection/learning
      })

      // Mark original failed decision as retried
      if (failedDecisionId) {
        await markDecisionRetried(failedDecisionId)
      }

      const savedMsg = decisionResult.data?.decision_id
        ? ` | Decision saved for learning`
        : ""
      setSuccess(`Pending order placed! Ticket: ${data.order_ticket}, Type: ${data.order_type}${savedMsg}`)
    }
    setLoading(false)
  }

  // Calculate risk/reward ratio
  const calculateRR = (entry: number, sl: number, tp: number) => {
    if (!entry || !sl || !tp) return null
    const risk = Math.abs(entry - sl)
    const reward = Math.abs(tp - entry)
    if (risk === 0) return null
    return (reward / risk).toFixed(2)
  }

  const currentEntry = activeTab === "market" ? parseFloat(entryPrice) : parseFloat(pullbackEntry)
  const rr = calculateRR(currentEntry, parseFloat(stopLoss), parseFloat(takeProfit))

  if (signal === "HOLD") {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>No Trade Signal</DialogTitle>
            <DialogDescription>
              The analysis recommends HOLD - no trade execution available.
            </DialogDescription>
          </DialogHeader>
          <div className="py-6 text-center">
            <Badge variant="secondary" className="text-lg px-4 py-2">HOLD</Badge>
            <p className="mt-4 text-sm text-muted-foreground">
              Wait for a clearer signal before entering a trade.
            </p>
          </div>
          <Button onClick={() => onOpenChange(false)}>Close</Button>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Execute Trade - {symbol}
            <Badge variant={signal === "BUY" ? "buy" : "sell"} className="ml-2">
              {signal}
            </Badge>
          </DialogTitle>
          <DialogDescription>
            {failureReason ? "Retry failed trade with adjusted parameters" : "Review and execute the trade based on AI analysis"}
          </DialogDescription>
        </DialogHeader>

        {marketClosed && (
          <Alert className="border-yellow-500/50 bg-yellow-500/10">
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
            <AlertDescription>
              <p className="font-medium text-yellow-500">Market closed for {symbol}</p>
              <p className="text-sm text-muted-foreground">{marketClosed}</p>
            </AlertDescription>
          </Alert>
        )}

        {failureReason && (
          <Alert className="border-red-500/50 bg-red-500/10">
            <AlertTriangle className="h-4 w-4 text-red-500" />
            <AlertDescription>
              <p className="font-medium text-red-500 mb-1">Previous execution failed</p>
              <p className="text-sm text-red-400">{failureReason}</p>
              <p className="text-sm text-muted-foreground mt-1">
                Entry has been updated to current market price. Review SL/TP and adjust if needed before retrying.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="border-green-500 bg-green-500/10">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            <AlertDescription className="text-green-500">{success}</AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="market">
              <TrendingUp className="h-4 w-4 mr-2" />
              Market Entry
            </TabsTrigger>
            <TabsTrigger value="pullback">
              <Target className="h-4 w-4 mr-2" />
              Pullback Entry
            </TabsTrigger>
          </TabsList>

          <TabsContent value="market" className="space-y-4 mt-4">
            {/* Pros and Cons */}
            <div className="grid grid-cols-2 gap-3 p-3 bg-muted/30 rounded-lg text-sm">
              <div>
                <p className="font-medium text-green-500 mb-1">Pros</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Guaranteed entry - won't miss the move</li>
                  <li>• Immediate position, immediate exposure</li>
                  <li>• Simple execution, no waiting</li>
                </ul>
              </div>
              <div>
                <p className="font-medium text-red-500 mb-1">Cons</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Higher entry cost (pay the spread)</li>
                  <li>• Worse risk/reward ratio</li>
                  <li>• May enter at poor price level</li>
                </ul>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Entry Price (Market)</Label>
                <Input
                  type="number"
                  step="any"
                  value={entryPrice}
                  onChange={(e) => setEntryPrice(e.target.value)}
                  placeholder="Current market price"
                />
                <p className="text-xs text-muted-foreground">
                  {symbolInfo?.ask && `Ask: ${symbolInfo.ask} | Bid: ${symbolInfo.bid}`}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Volume (Lots)</Label>
                <Input
                  type="number"
                  step="0.01"
                  min={symbolInfo?.volume_min || 0.01}
                  max={symbolInfo?.volume_max || 100}
                  value={volume}
                  onChange={(e) => setVolume(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Min: {symbolInfo?.volume_min || "0.01"} | Max: {symbolInfo?.volume_max || "100"}
                </p>
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-red-500" />
                  Stop Loss
                </Label>
                <Input
                  type="number"
                  step="any"
                  value={stopLoss}
                  onChange={(e) => setStopLoss(e.target.value)}
                  className={stopLossError ? "border-red-500" : "border-red-500/50"}
                />
                {stopLossError && (
                  <p className="text-xs text-red-500">{stopLossError}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-500" />
                  Take Profit
                </Label>
                <Input
                  type="number"
                  step="any"
                  value={takeProfit}
                  onChange={(e) => setTakeProfit(e.target.value)}
                  className="border-green-500/50"
                />
              </div>
            </div>

            <Button
              className="w-full"
              variant={signal === "BUY" ? "default" : "destructive"}
              onClick={handleExecuteMarket}
              disabled={loading || !entryPrice || !stopLoss || !volume || !!stopLossError || !!marketClosed}
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : signal === "BUY" ? (
                <TrendingUp className="h-4 w-4 mr-2" />
              ) : (
                <TrendingDown className="h-4 w-4 mr-2" />
              )}
              Execute Market {signal}
            </Button>
          </TabsContent>

          <TabsContent value="pullback" className="space-y-4 mt-4">
            {/* Pros and Cons */}
            <div className="grid grid-cols-2 gap-3 p-3 bg-muted/30 rounded-lg text-sm">
              <div>
                <p className="font-medium text-green-500 mb-1">Pros</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Better entry price = higher profits</li>
                  <li>• Improved risk/reward ratio</li>
                  <li>• Enter at SMC key levels (OB/FVG)</li>
                  <li>• Tighter stop loss possible</li>
                </ul>
              </div>
              <div>
                <p className="font-medium text-red-500 mb-1">Cons</p>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• May miss the trade entirely</li>
                  <li>• Price might not pull back</li>
                  <li>• Requires patience and discipline</li>
                  <li>• Order may expire unfilled</li>
                </ul>
              </div>
            </div>

            <Alert>
              <Target className="h-4 w-4" />
              <AlertDescription>
                A limit order will be placed at your specified entry price.
                {signal === "BUY"
                  ? " For BUY orders, the pullback price must be BELOW current price."
                  : " For SELL orders, the pullback price must be ABOVE current price."}
              </AlertDescription>
            </Alert>

            {/* SMC Level Suggestions */}
            {relevantSmcLevels.length > 0 && (
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">SMC Levels (click to use)</Label>
                <div className="flex flex-wrap gap-2">
                  {relevantSmcLevels.slice(0, 4).map((level, idx) => (
                    <Button
                      key={idx}
                      variant={pullbackEntry === level.price.toString() ? "default" : "outline"}
                      size="sm"
                      onClick={() => setPullbackEntry(level.price.toString())}
                      className="text-xs"
                    >
                      <span className="capitalize mr-1">
                        {level.type === "order_block" ? "OB" :
                         level.type === "fvg" ? "FVG" :
                         level.type === "liquidity" ? "LIQ" :
                         level.type === "support" ? "SUP" :
                         level.type === "resistance" ? "RES" :
                         level.type === "pdl" ? "PDL" :
                         level.type === "pdh" ? "PDH" : "BRK"}
                      </span>
                      {level.price.toFixed(symbolInfo?.digits || 5)}
                      {level.strength && (
                        <span className="ml-1 text-muted-foreground">({Math.round(level.strength * 100)}%)</span>
                      )}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {relevantSmcLevels.length === 0 && smcLevels.length > 0 && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  No SMC levels found {signal === "BUY" ? "below" : "above"} current price for a {signal} pullback entry.
                </AlertDescription>
              </Alert>
            )}

            {/* Swing Level Suggestions - fetched from market data */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground flex items-center gap-2">
                Swing {signal === "BUY" ? "Lows" : "Highs"} (click to use)
                {swingLevelsLoading && <Loader2 className="h-3 w-3 animate-spin" />}
              </Label>
              {swingLevels.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {swingLevels.map((level, idx) => (
                    <Button
                      key={idx}
                      variant={pullbackEntry === level.price.toString() ? "default" : "outline"}
                      size="sm"
                      onClick={() => setPullbackEntry(level.price.toString())}
                      className="text-xs"
                      title={level.label}
                    >
                      <span className="mr-1">
                        {level.type === "pdl" ? "PDL" :
                         level.type === "pdh" ? "PDH" :
                         level.type === "swing_low" ? "SL" : "SH"}
                      </span>
                      {level.price}
                      <span className="ml-1 text-muted-foreground">
                        ({level.distance_pct}%)
                      </span>
                    </Button>
                  ))}
                </div>
              ) : !swingLevelsLoading ? (
                <p className="text-xs text-muted-foreground">
                  No swing levels found {signal === "BUY" ? "below" : "above"} current price
                </p>
              ) : null}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Pullback Entry Price</Label>
                <Input
                  type="number"
                  step="any"
                  value={pullbackEntry}
                  onChange={(e) => setPullbackEntry(e.target.value)}
                  placeholder="Enter limit price"
                  className={pullbackError ? "border-red-500" : ""}
                />
                {pullbackError ? (
                  <p className="text-xs text-red-500">{pullbackError}</p>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    {currentPrice && `Current price: ${currentPrice.toFixed(symbolInfo?.digits || 5)}`}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label>Volume (Lots)</Label>
                <Input
                  type="number"
                  step="0.01"
                  min={symbolInfo?.volume_min || 0.01}
                  max={symbolInfo?.volume_max || 100}
                  value={volume}
                  onChange={(e) => setVolume(e.target.value)}
                />
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-red-500" />
                  Stop Loss
                </Label>
                <Input
                  type="number"
                  step="any"
                  value={stopLoss}
                  onChange={(e) => setStopLoss(e.target.value)}
                  className={stopLossError ? "border-red-500" : "border-red-500/50"}
                />
                {stopLossError && (
                  <p className="text-xs text-red-500">{stopLossError}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-500" />
                  Take Profit
                </Label>
                <Input
                  type="number"
                  step="any"
                  value={takeProfit}
                  onChange={(e) => setTakeProfit(e.target.value)}
                  className="border-green-500/50"
                />
              </div>
            </div>

            <Button
              className="w-full"
              variant="outline"
              onClick={handleExecuteLimit}
              disabled={loading || !pullbackEntry || !stopLoss || !volume || !!pullbackError || !!stopLossError || !!marketClosed}
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Target className="h-4 w-4 mr-2" />
              )}
              Place Limit Order
            </Button>
          </TabsContent>
        </Tabs>

        <Separator />

        {/* Risk Calculator */}
        <div className="space-y-4">
          <h4 className="font-medium flex items-center gap-2">
            <Calculator className="h-4 w-4" />
            Risk Calculator
          </h4>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label>Risk %</Label>
              <Input
                type="number"
                step="0.1"
                min="0.1"
                max="10"
                value={riskPercent}
                onChange={(e) => setRiskPercent(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Calculated Size</Label>
              <div className="text-lg font-semibold">
                {calculating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  `${volume} lots`
                )}
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Risk : Reward</Label>
              <div className="text-lg font-semibold">
                {rr ? (
                  <span className={parseFloat(rr) >= 2 ? "text-green-500" : parseFloat(rr) >= 1 ? "text-yellow-500" : "text-red-500"}>
                    1 : {rr}
                  </span>
                ) : (
                  "N/A"
                )}
              </div>
            </div>
          </div>

          {positionSize && (
            <div className="space-y-3">
              {/* R:R Warning */}
              {positionSize.rr_warning && (
                <div className="flex items-center gap-2 p-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-500">
                  <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                  <span>{positionSize.rr_warning}</span>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4 p-3 bg-muted/50 rounded-lg text-sm">
                <div>
                  <span className="text-muted-foreground">Risk Amount:</span>
                  <span className="ml-2 font-medium">${positionSize.risk_amount?.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Account Balance:</span>
                  <span className="ml-2 font-medium">${positionSize.account_balance?.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Potential Loss:</span>
                  <span className="ml-2 font-medium text-red-500">-${positionSize.potential_loss?.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">
                    {positionSize.actual_profit !== null ? "Potential Profit:" : "Potential Profit (2R):"}
                  </span>
                  <span className={`ml-2 font-medium ${positionSize.actual_profit !== null && positionSize.actual_rr < 1 ? "text-red-500" : "text-green-500"}`}>
                    +${(positionSize.actual_profit ?? positionSize.potential_profit_2r)?.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Trade Summary */}
        <div className="p-4 bg-muted/30 rounded-lg space-y-2">
          <h4 className="font-medium">Trade Summary</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Symbol:</span>
              <span className="font-medium">{symbol}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Direction:</span>
              <Badge variant={signal === "BUY" ? "buy" : "sell"}>{signal}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Entry:</span>
              <span className="font-medium">
                {activeTab === "market" ? entryPrice : pullbackEntry || "Not set"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Volume:</span>
              <span className="font-medium">{volume} lots</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Stop Loss:</span>
              <span className="font-medium text-red-500">{stopLoss || "Not set"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Take Profit:</span>
              <span className="font-medium text-green-500">{takeProfit || "Not set"}</span>
            </div>
          </div>
          <div className="pt-2 mt-2 border-t text-xs text-muted-foreground">
            <p>This trade will be saved for learning. The system tracks outcomes to improve future recommendations.</p>
          </div>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} className="flex-1">
            Cancel
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
