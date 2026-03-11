"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Settings, Server, Database, Bot, TrendingUp, Loader2, RefreshCw, X, Plus, Search, Sparkles, Shield, Clock, Check, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { getPortfolioConfig, getMarketWatchSymbols, addToMarketWatch, removeFromMarketWatch, searchSymbols, addPortfolioSymbol, removePortfolioSymbol, getPortfolioSuggestions, updatePortfolioConfig, PortfolioConfigUpdateParams } from "@/lib/api"

interface SymbolConfig {
  symbol: string
  max_positions: number
  risk_budget_pct: number
  correlation_group: string
  timeframes: string[]
  enabled: boolean
  min_confidence: number
}

interface PortfolioConfig {
  symbols?: SymbolConfig[]
  max_total_positions?: number
  max_daily_trades?: number
  execution_mode?: string
  total_risk_budget_pct?: number
  daily_loss_limit_pct?: number
  use_atr_stops?: boolean
  atr_stop_multiplier?: number
  atr_trailing_multiplier?: number
  risk_reward_ratio?: number
  schedule?: {
    morning_analysis_hour?: number
    midday_review_hour?: number
    evening_reflect_hour?: number
    timezone?: string
  }
  error?: string
}

interface MarketWatchSymbol {
  symbol: string
  description: string
  bid: number | null
  ask: number | null
  spread: number
  digits: number
  currency_base: string
  currency_profit: string
  contract_size: number
  volume_min: number
}

interface MarketWatchData {
  symbols?: MarketWatchSymbol[]
  count?: number
  error?: string
}

interface SearchResult {
  symbol: string
  description: string
  visible: boolean
}

export default function SettingsPage() {
  const [config, setConfig] = useState<PortfolioConfig | null>(null)
  const [marketWatch, setMarketWatch] = useState<MarketWatchData | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMarketWatch, setLoadingMarketWatch] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  // Portfolio config editing state
  const [portfolioSearchQuery, setPortfolioSearchQuery] = useState("")
  const [portfolioSearchResults, setPortfolioSearchResults] = useState<SearchResult[]>([])
  const [portfolioSearching, setPortfolioSearching] = useState(false)
  const [portfolioActionLoading, setPortfolioActionLoading] = useState<string | null>(null)
  const [portfolioActionError, setPortfolioActionError] = useState<string | null>(null)

  // AI Suggestions state
  interface Suggestion {
    symbol: string
    reason: string
    correlation_group: string
    priority: string
  }
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [suggestionsAnalysis, setSuggestionsAnalysis] = useState("")
  const [suggestionsRiskNotes, setSuggestionsRiskNotes] = useState("")
  const [loadingSuggestions, setLoadingSuggestions] = useState(false)
  const [suggestionsError, setSuggestionsError] = useState<string | null>(null)
  const [showSuggestions, setShowSuggestions] = useState(false)

  // Config editing state
  const [configEdits, setConfigEdits] = useState<PortfolioConfigUpdateParams>({})
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  const fetchConfig = async () => {
    setLoading(true)
    const { data } = await getPortfolioConfig()
    setConfig(data)
    setLoading(false)
  }

  const fetchMarketWatch = async () => {
    setLoadingMarketWatch(true)
    const { data, error } = await getMarketWatchSymbols()
    if (error) {
      setMarketWatch({ error })
    } else {
      setMarketWatch(data)
    }
    setLoadingMarketWatch(false)
  }

  const handleSearch = async (query: string) => {
    setSearchQuery(query)
    if (query.length < 2) {
      setSearchResults([])
      return
    }
    setSearching(true)
    const { data } = await searchSymbols(query)
    if (data?.symbols) {
      // Filter out symbols already in Market Watch
      const watchedSymbols = new Set(marketWatch?.symbols?.map(s => s.symbol) || [])
      setSearchResults(data.symbols.filter((s: SearchResult) => !watchedSymbols.has(s.symbol)))
    }
    setSearching(false)
  }

  const handleAddSymbol = async (symbol: string) => {
    setActionLoading(symbol)
    setActionError(null)
    const { data, error } = await addToMarketWatch(symbol)
    if (error) {
      setActionError(error)
    } else if (data?.success) {
      await fetchMarketWatch()
      setSearchQuery("")
      setSearchResults([])
    } else if (data?.error) {
      setActionError(data.error)
    }
    setActionLoading(null)
  }

  const handleRemoveSymbol = async (symbol: string) => {
    setActionLoading(symbol)
    setActionError(null)
    const { data, error } = await removeFromMarketWatch(symbol)
    if (error) {
      setActionError(error)
    } else if (data?.success) {
      await fetchMarketWatch()
    } else if (data?.error) {
      setActionError(data.error)
    }
    setActionLoading(null)
  }

  // Portfolio config handlers
  const handlePortfolioSearch = async (query: string) => {
    setPortfolioSearchQuery(query)
    if (query.length < 2) {
      setPortfolioSearchResults([])
      return
    }
    setPortfolioSearching(true)
    const { data } = await searchSymbols(query)
    if (data?.symbols) {
      // Filter out symbols already in portfolio config
      const configSymbols = new Set(config?.symbols?.map(s => s.symbol) || [])
      setPortfolioSearchResults(data.symbols.filter((s: SearchResult) => !configSymbols.has(s.symbol)))
    }
    setPortfolioSearching(false)
  }

  const handleAddPortfolioSymbol = async (symbol: string) => {
    setPortfolioActionLoading(symbol)
    setPortfolioActionError(null)
    const { data, error } = await addPortfolioSymbol(symbol)
    if (error) {
      setPortfolioActionError(error)
    } else if (data?.success) {
      await fetchConfig()
      setPortfolioSearchQuery("")
      setPortfolioSearchResults([])
    } else if (data?.error) {
      setPortfolioActionError(data.error)
    }
    setPortfolioActionLoading(null)
  }

  const handleRemovePortfolioSymbol = async (symbol: string) => {
    setPortfolioActionLoading(symbol)
    setPortfolioActionError(null)
    const { data, error } = await removePortfolioSymbol(symbol)
    if (error) {
      setPortfolioActionError(error)
    } else if (data?.success) {
      await fetchConfig()
    } else if (data?.error) {
      setPortfolioActionError(data.error)
    }
    setPortfolioActionLoading(null)
  }

  // Fetch AI suggestions for portfolio balance
  const fetchSuggestions = async () => {
    setLoadingSuggestions(true)
    setSuggestionsError(null)
    setShowSuggestions(true)
    const { data, error } = await getPortfolioSuggestions()
    if (error) {
      setSuggestionsError(error)
    } else if (data?.success) {
      setSuggestions(data.suggestions || [])
      setSuggestionsAnalysis(data.portfolio_analysis || "")
      setSuggestionsRiskNotes(data.risk_notes || "")
    } else if (data?.error) {
      setSuggestionsError(data.error)
    } else if (data?.raw_response) {
      setSuggestionsError("Could not parse suggestions. Raw response: " + data.raw_response.substring(0, 200))
    }
    setLoadingSuggestions(false)
  }

  const handleAddSuggestion = async (symbol: string) => {
    await handleAddPortfolioSymbol(symbol)
    // Remove from suggestions list
    setSuggestions(prev => prev.filter(s => s.symbol !== symbol))
  }

  // Config update handlers
  const updateField = <K extends keyof PortfolioConfigUpdateParams>(
    field: K,
    value: PortfolioConfigUpdateParams[K]
  ) => {
    setConfigEdits(prev => ({ ...prev, [field]: value }))
    setHasChanges(true)
    setSaveSuccess(false)
    setSaveError(null)
  }

  const handleSaveConfig = async () => {
    setSaving(true)
    setSaveError(null)
    setSaveSuccess(false)

    const { data, error } = await updatePortfolioConfig(configEdits)

    if (error) {
      setSaveError(error)
    } else if (data?.error) {
      setSaveError(data.error)
    } else if (data?.success) {
      setSaveSuccess(true)
      setConfigEdits({})
      setHasChanges(false)
      await fetchConfig()
    }

    setSaving(false)
  }

  useEffect(() => {
    fetchConfig()
    fetchMarketWatch()
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">System configuration and information</p>
      </div>

      {/* MT5 Market Watch Symbols */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              MT5 Market Watch
            </CardTitle>
            <Button variant="ghost" size="icon" onClick={fetchMarketWatch} disabled={loadingMarketWatch}>
              {loadingMarketWatch ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>
          <CardDescription>
            Symbols currently in your MT5 Market Watch ({marketWatch?.count || 0} symbols)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Search to add symbols */}
          <div className="mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search symbols to add (e.g., EUR, GOLD, BTC)..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-9"
              />
              {searching && (
                <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            {searchResults.length > 0 && (
              <div className="mt-2 border rounded-lg divide-y max-h-[200px] overflow-y-auto">
                {searchResults.map((result) => (
                  <div
                    key={result.symbol}
                    className="flex items-center justify-between p-2 hover:bg-muted/50"
                  >
                    <div>
                      <span className="font-medium">{result.symbol}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{result.description}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleAddSymbol(result.symbol)}
                      disabled={actionLoading === result.symbol}
                    >
                      {actionLoading === result.symbol ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Plus className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                ))}
              </div>
            )}
            {actionError && (
              <p className="mt-2 text-sm text-red-500">{actionError}</p>
            )}
          </div>

          <Separator className="my-4" />

          {loadingMarketWatch ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : marketWatch?.error ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>Could not load Market Watch: {marketWatch.error}</p>
              <p className="text-sm mt-2">Make sure MT5 is running and the backend is connected</p>
            </div>
          ) : marketWatch?.symbols && marketWatch.symbols.length > 0 ? (
            <div className="space-y-4">
              <div className="grid gap-2 max-h-[400px] overflow-y-auto">
                {marketWatch.symbols.map((sym) => (
                  <div
                    key={sym.symbol}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="font-medium">{sym.symbol}</div>
                      <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                        {sym.description}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <div className="text-right">
                        <span className="text-green-500">{sym.bid?.toFixed(sym.digits)}</span>
                        <span className="text-muted-foreground mx-1">/</span>
                        <span className="text-red-500">{sym.ask?.toFixed(sym.digits)}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {sym.spread} pts
                      </Badge>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-red-500"
                        onClick={() => handleRemoveSymbol(sym.symbol)}
                        disabled={actionLoading === sym.symbol}
                      >
                        {actionLoading === sym.symbol ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <X className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <p>No symbols in Market Watch</p>
              <p className="text-sm mt-2">Use the search above to add symbols</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Automation Settings Section */}
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold">Automation Settings</h2>
          <p className="text-sm text-muted-foreground">Configure automated trading behavior, risk limits, and schedule</p>
        </div>

        {/* Trading Controls Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Trading Controls
            </CardTitle>
            <CardDescription>
              Core automation settings that control trading behavior
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Execution Mode */}
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <label className="text-sm font-medium">Execution Mode</label>
                <HelpTooltip content="FULL_AUTO: System places trades automatically without confirmation. SEMI_AUTO: System generates signals but requires your approval before placing trades. PAPER: All trades are simulated - use this to test strategies without risking real money." />
              </div>
              <Select
                value={configEdits.execution_mode || config?.execution_mode?.toUpperCase() || "PAPER"}
                onValueChange={(value) => updateField('execution_mode', value as 'FULL_AUTO' | 'SEMI_AUTO' | 'PAPER')}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="FULL_AUTO">
                    <span className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-green-500" />
                      Full Auto
                    </span>
                  </SelectItem>
                  <SelectItem value="SEMI_AUTO">
                    <span className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-yellow-500" />
                      Semi Auto
                    </span>
                  </SelectItem>
                  <SelectItem value="PAPER">
                    <span className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-blue-500" />
                      Paper Trading
                    </span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Max Positions & Daily Trades */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Max Total Positions</label>
                  <HelpTooltip content="Maximum number of open positions allowed at any time across all symbols. Lower = safer but may miss opportunities. Higher = more exposure but increased risk. Recommended: 3-5 for beginners, up to 10 for experienced traders." />
                </div>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={configEdits.max_total_positions ?? config?.max_total_positions ?? 5}
                  onChange={(e) => updateField('max_total_positions', parseInt(e.target.value) || 1)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Max Daily Trades</label>
                  <HelpTooltip content="Maximum number of new trades allowed per day. Prevents overtrading which often leads to losses. After this limit is reached, no new positions will be opened until the next day. Recommended: 2-3 for swing trading, 5-10 for day trading." />
                </div>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={configEdits.max_daily_trades ?? config?.max_daily_trades ?? 3}
                  onChange={(e) => updateField('max_daily_trades', parseInt(e.target.value) || 1)}
                />
              </div>
            </div>

            {/* Risk Limits */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Total Risk Budget (%)</label>
                  <HelpTooltip content="Maximum percentage of your account equity that can be at risk across all open positions combined. Example: With $10,000 account and 6% budget = $600 max total risk. If you hit this limit, no new trades until existing positions are closed. Recommended: 5-10% for conservative, up to 15% for aggressive." />
                </div>
                <Input
                  type="number"
                  min={0.5}
                  max={20}
                  step={0.5}
                  value={configEdits.total_risk_budget_pct ?? config?.total_risk_budget_pct ?? 6.0}
                  onChange={(e) => updateField('total_risk_budget_pct', parseFloat(e.target.value) || 0.5)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Daily Loss Limit (%)</label>
                  <HelpTooltip content="If your daily losses reach this percentage of your account, trading is automatically halted for the day. This 'circuit breaker' protects you from catastrophic losses during bad market conditions or emotional trading. Recommended: 2-3% for conservative, up to 5% for aggressive." />
                </div>
                <Input
                  type="number"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={configEdits.daily_loss_limit_pct ?? config?.daily_loss_limit_pct ?? 3.0}
                  onChange={(e) => updateField('daily_loss_limit_pct', parseFloat(e.target.value) || 0.5)}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stop Loss Settings Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Stop Loss Settings
            </CardTitle>
            <CardDescription>
              Configure how stop losses are calculated and managed
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* ATR Stops Toggle */}
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-1">
                  <span className="text-sm font-medium">Use ATR-Based Stops</span>
                  <HelpTooltip content="ATR (Average True Range) measures market volatility. When enabled, stop losses are placed at a distance based on recent price movement. This means tighter stops in calm markets and wider stops in volatile markets - adapting to current conditions rather than using fixed pip distances." />
                </div>
                <p className="text-xs text-muted-foreground">
                  Dynamically adjust stop loss distance based on volatility
                </p>
              </div>
              <Switch
                checked={configEdits.use_atr_stops ?? config?.use_atr_stops ?? true}
                onCheckedChange={(checked) => updateField('use_atr_stops', checked)}
              />
            </div>

            {/* ATR Multipliers */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">ATR Stop Multiplier</label>
                  <HelpTooltip content="Multiplier for initial stop loss distance. SL = Entry +/- (ATR x Multiplier). Higher = wider stops, less likely to get stopped out early, but larger potential loss per trade. Lower = tighter stops, more likely to get stopped out, but smaller losses. Recommended: 1.5-2.5 for most strategies." />
                </div>
                <Input
                  type="number"
                  min={0.5}
                  max={5}
                  step={0.1}
                  value={configEdits.atr_stop_multiplier ?? config?.atr_stop_multiplier ?? 2.0}
                  onChange={(e) => updateField('atr_stop_multiplier', parseFloat(e.target.value) || 0.5)}
                  disabled={!(configEdits.use_atr_stops ?? config?.use_atr_stops ?? true)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">ATR Trailing Multiplier</label>
                  <HelpTooltip content="Multiplier for trailing stop distance once in profit. The trailing stop follows price at this distance. Lower = locks in profits faster but may exit too early. Higher = gives trades room to breathe but may give back more profit. Recommended: 1.0-2.0, typically less than initial stop multiplier." />
                </div>
                <Input
                  type="number"
                  min={0.5}
                  max={5}
                  step={0.1}
                  value={configEdits.atr_trailing_multiplier ?? config?.atr_trailing_multiplier ?? 1.5}
                  onChange={(e) => updateField('atr_trailing_multiplier', parseFloat(e.target.value) || 0.5)}
                  disabled={!(configEdits.use_atr_stops ?? config?.use_atr_stops ?? true)}
                />
              </div>
            </div>

            {/* Risk/Reward Ratio */}
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <label className="text-sm font-medium">Risk/Reward Ratio</label>
                <HelpTooltip content="Target ratio of potential profit to potential loss. A 2:1 ratio means targeting $2 profit for every $1 risked. Higher ratios = fewer winning trades needed to be profitable. Example: With 2:1 R:R, you only need to win 34% of trades to break even. Recommended: 1.5:1 to 3:1 depending on strategy." />
              </div>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  min={0.5}
                  max={10}
                  step={0.1}
                  value={configEdits.risk_reward_ratio ?? config?.risk_reward_ratio ?? 2.0}
                  onChange={(e) => updateField('risk_reward_ratio', parseFloat(e.target.value) || 0.5)}
                  className="w-24"
                />
                <span className="text-sm text-muted-foreground">: 1</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Current: Risk $1 to potentially gain ${(configEdits.risk_reward_ratio ?? config?.risk_reward_ratio ?? 2.0).toFixed(1)}
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Schedule Settings Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Schedule Settings
            </CardTitle>
            <CardDescription>
              Configure when automated analysis cycles run
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Morning Analysis</label>
                  <HelpTooltip content="When the morning analysis runs. This cycle analyzes all configured symbols, generates trading signals, and (in FULL_AUTO mode) places trades. Best run before major market sessions open. Example: 8:00 for London session, 13:00 for NY session." />
                </div>
                <Select
                  value={String(configEdits.schedule?.morning_analysis_hour ?? config?.schedule?.morning_analysis_hour ?? 8)}
                  onValueChange={(value) => updateField('schedule', {
                    ...(configEdits.schedule || {}),
                    morning_analysis_hour: parseInt(value)
                  })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 24 }, (_, i) => (
                      <SelectItem key={i} value={String(i)}>
                        {i.toString().padStart(2, '0')}:00
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Midday Review</label>
                  <HelpTooltip content="When the midday review runs. This cycle reviews all open positions, checks if stop losses should be adjusted, and evaluates market conditions. Useful for catching trend changes mid-session." />
                </div>
                <Select
                  value={String(configEdits.schedule?.midday_review_hour ?? config?.schedule?.midday_review_hour ?? 13)}
                  onValueChange={(value) => updateField('schedule', {
                    ...(configEdits.schedule || {}),
                    midday_review_hour: parseInt(value)
                  })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 24 }, (_, i) => (
                      <SelectItem key={i} value={String(i)}>
                        {i.toString().padStart(2, '0')}:00
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-1">
                  <label className="text-sm font-medium">Evening Reflect</label>
                  <HelpTooltip content="When the evening reflection runs. This cycle processes closed trades, extracts lessons learned, and updates the AI memory. Run after market sessions close when most trades are settled." />
                </div>
                <Select
                  value={String(configEdits.schedule?.evening_reflect_hour ?? config?.schedule?.evening_reflect_hour ?? 20)}
                  onValueChange={(value) => updateField('schedule', {
                    ...(configEdits.schedule || {}),
                    evening_reflect_hour: parseInt(value)
                  })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 24 }, (_, i) => (
                      <SelectItem key={i} value={String(i)}>
                        {i.toString().padStart(2, '0')}:00
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Timezone */}
            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <label className="text-sm font-medium">Timezone</label>
                <HelpTooltip content="Timezone for all scheduled times above. Make sure this matches your broker's server time or your local time, depending on your preference." />
              </div>
              <Input
                value={configEdits.schedule?.timezone ?? config?.schedule?.timezone ?? "UTC"}
                onChange={(e) => updateField('schedule', {
                  ...(configEdits.schedule || {}),
                  timezone: e.target.value
                })}
                placeholder="e.g., UTC, America/New_York, Europe/London"
                className="max-w-xs"
              />
            </div>
          </CardContent>
        </Card>

        {/* Save Section */}
        <div className="flex items-center justify-between p-4 border rounded-lg bg-muted/30">
          <div className="space-y-1">
            {saveError && (
              <Alert variant="destructive" className="py-2">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{saveError}</AlertDescription>
              </Alert>
            )}
            {saveSuccess && (
              <Alert className="py-2 border-green-500/50 text-green-600 dark:text-green-400">
                <Check className="h-4 w-4" />
                <AlertDescription>Settings saved successfully</AlertDescription>
              </Alert>
            )}
            {hasChanges && !saveError && !saveSuccess && (
              <p className="text-sm text-yellow-600 dark:text-yellow-400">You have unsaved changes</p>
            )}
            {!hasChanges && !saveError && !saveSuccess && (
              <p className="text-sm text-muted-foreground">No changes to save</p>
            )}
          </div>
          <Button
            onClick={handleSaveConfig}
            disabled={!hasChanges || saving}
          >
            {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Save Changes
          </Button>
        </div>
      </div>

      {/* Portfolio Trading Config */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Portfolio Trading Config
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={fetchSuggestions}
                disabled={loadingSuggestions}
                className="gap-2"
              >
                {loadingSuggestions ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                AI Suggestions
              </Button>
              <Button variant="ghost" size="icon" onClick={fetchConfig} disabled={loading}>
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          <CardDescription>Symbols configured for automated trading in portfolio_config.yaml ({config?.symbols?.length || 0} symbols)</CardDescription>
        </CardHeader>
        <CardContent>
          {/* AI Suggestions Panel */}
          {showSuggestions && (
            <div className="mb-4 p-4 rounded-lg border bg-muted/30">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-yellow-500" />
                  AI Portfolio Suggestions
                </h4>
                <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setShowSuggestions(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {loadingSuggestions ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  <span className="ml-2 text-sm text-muted-foreground">Analyzing portfolio...</span>
                </div>
              ) : suggestionsError ? (
                <p className="text-sm text-red-500">{suggestionsError}</p>
              ) : (
                <div className="space-y-3">
                  {suggestionsAnalysis && (
                    <p className="text-sm text-muted-foreground">{suggestionsAnalysis}</p>
                  )}
                  {suggestions.length > 0 ? (
                    <div className="space-y-2">
                      {suggestions.map((s) => (
                        <div key={s.symbol} className="flex items-center justify-between p-2 rounded border bg-card">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{s.symbol}</span>
                              <Badge variant={s.priority === "high" ? "default" : s.priority === "medium" ? "secondary" : "outline"} className="text-xs">
                                {s.priority}
                              </Badge>
                              <Badge variant="outline" className="text-xs">{s.correlation_group}</Badge>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">{s.reason}</p>
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleAddSuggestion(s.symbol)}
                            disabled={portfolioActionLoading === s.symbol}
                          >
                            {portfolioActionLoading === s.symbol ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Plus className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No suggestions available</p>
                  )}
                  {suggestionsRiskNotes && (
                    <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
                      <strong>Risk Note:</strong> {suggestionsRiskNotes}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Search to add symbols */}
          <div className="mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search symbols to add to portfolio..."
                value={portfolioSearchQuery}
                onChange={(e) => handlePortfolioSearch(e.target.value)}
                className="pl-9"
              />
              {portfolioSearching && (
                <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            {portfolioSearchResults.length > 0 && (
              <div className="mt-2 border rounded-lg divide-y max-h-[200px] overflow-y-auto">
                {portfolioSearchResults.map((result) => (
                  <div
                    key={result.symbol}
                    className="flex items-center justify-between p-2 hover:bg-muted/50"
                  >
                    <div>
                      <span className="font-medium">{result.symbol}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{result.description}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleAddPortfolioSymbol(result.symbol)}
                      disabled={portfolioActionLoading === result.symbol}
                    >
                      {portfolioActionLoading === result.symbol ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Plus className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                ))}
              </div>
            )}
            {portfolioActionError && (
              <p className="mt-2 text-sm text-red-500">{portfolioActionError}</p>
            )}
          </div>

          <Separator className="my-4" />

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : config?.error ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>Could not load config: {config.error}</p>
              <p className="text-sm mt-2">Make sure the backend is running</p>
            </div>
          ) : config?.symbols && config.symbols.length > 0 ? (
            <div className="space-y-4">
              <div className="grid gap-3">
                {config.symbols.map((sym) => (
                  <div
                    key={sym.symbol}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="font-medium">{sym.symbol}</div>
                      <Badge variant={sym.enabled ? "default" : "secondary"}>
                        {sym.enabled ? "Enabled" : "Disabled"}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {sym.correlation_group}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span>Risk: {sym.risk_budget_pct}%</span>
                      <span>Min Conf: {sym.min_confidence}</span>
                      <span className="text-xs">{sym.timeframes?.join(", ")}</span>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-red-500"
                        onClick={() => handleRemovePortfolioSymbol(sym.symbol)}
                        disabled={portfolioActionLoading === sym.symbol}
                      >
                        {portfolioActionLoading === sym.symbol ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <X className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
              <Separator />
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Max Total Positions</span>
                  <p className="font-medium">{config.max_total_positions || "N/A"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Max Daily Trades</span>
                  <p className="font-medium">{config.max_daily_trades || "N/A"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Execution Mode</span>
                  <p className="font-medium capitalize">{config.execution_mode?.replace("_", " ") || "N/A"}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <p>No symbols configured</p>
              <p className="text-sm mt-2">Use the search above to add symbols</p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* API Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              API Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Backend URL</span>
              <code className="rounded bg-muted px-2 py-1 text-sm">
                http://localhost:8000
              </code>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">WebSocket URL</span>
              <code className="rounded bg-muted px-2 py-1 text-sm">
                ws://localhost:8000/ws
              </code>
            </div>
            <Separator />
            <div className="text-sm text-muted-foreground">
              <p>To start the backend server, run:</p>
              <code className="mt-2 block rounded bg-muted p-3 text-xs">
                cd web/backend && uvicorn main:app --reload
              </code>
            </div>
          </CardContent>
        </Card>

        {/* System Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5" />
              System Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Frontend Version</span>
              <Badge variant="outline">1.0.0</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Framework</span>
              <span>Next.js 14 + shadcn/ui</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Backend</span>
              <span>FastAPI</span>
            </div>
          </CardContent>
        </Card>

        {/* Database Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Storage
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Memory Database</span>
              <span>ChromaDB (SQLite)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Location</span>
              <code className="rounded bg-muted px-2 py-1 text-sm">memory_db/</code>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Decisions Store</span>
              <code className="rounded bg-muted px-2 py-1 text-sm">
                examples/trade_decisions/
              </code>
            </div>
          </CardContent>
        </Card>

        {/* Quick Start */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Quick Start
            </CardTitle>
            <CardDescription>How to run the web UI</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">1. Start the backend:</p>
              <code className="block rounded bg-muted p-3 text-xs">
                cd web/backend<br />
                pip install -r requirements.txt<br />
                uvicorn main:app --reload --port 8000
              </code>
            </div>
            <div>
              <p className="text-sm font-medium mb-2">2. Start the frontend:</p>
              <code className="block rounded bg-muted p-3 text-xs">
                cd web/frontend<br />
                npm install<br />
                npm run dev
              </code>
            </div>
            <div>
              <p className="text-sm font-medium mb-2">3. Open in browser:</p>
              <code className="block rounded bg-muted p-3 text-xs">
                http://localhost:3000
              </code>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
