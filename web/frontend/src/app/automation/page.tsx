"use client"

import { useEffect, useState, useCallback, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  getPortfolioStatus,
  getPortfolioConfig,
  startPortfolioAutomation,
  stopPortfolioAutomation,
  triggerDailyCycle,
  getDailyCycleStatus,
  startDailyCycle,
  stopDailyCycle,
  getPendingPredictions,
  getMarketWatchSymbols,
  diagnosePortfolioAutomation,
  saveSelectedSymbols,
  DailyCycleStatus,
  PendingPrediction,
  // Quant Automation
  listAutomationInstances,
  startQuantAutomation,
  stopQuantAutomation,
  pauseQuantAutomation,
  resumeQuantAutomation,
  updateQuantAutomationConfig,
  deleteAutomationConfig,
  renameAutomationInstance,
  testQuantAnalysis,
  getQuantAutomationHistory,
  runVpQuantAnalysis,
  getMarketStatusMulti,
  QuantAutomationStatus,
  QuantAutomationConfig,
  QuantAutomationHistory,
} from "@/lib/api"
import { Checkbox } from "@/components/ui/checkbox"
import { formatDate } from "@/lib/utils"
import {
  RefreshCw,
  Play,
  Square,
  Bot,
  Sun,
  Clock,
  Moon,
  Loader2,
  Settings,
  AlertTriangle,
  Brain,
  TrendingUp,
  TrendingDown,
  Pause,
  Zap,
  Activity,
  Target,
  RotateCcw,
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp,
  Search,
  X,
  Pencil,
  Check,
  Eye,
} from "lucide-react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { TradeExecutionWizard } from "@/components/TradeExecutionWizard"

interface MarketWatchSymbol {
  symbol: string
  description: string
}

function SymbolMultiSelect({
  selected,
  onChange,
  available,
  disabled,
}: {
  selected: string[]
  onChange: (symbols: string[]) => void
  available: MarketWatchSymbol[]
  disabled?: boolean
}) {
  const [search, setSearch] = useState("")
  const filtered = available.filter(
    s =>
      s.symbol.toLowerCase().includes(search.toLowerCase()) ||
      s.description.toLowerCase().includes(search.toLowerCase())
  )

  const toggle = (symbol: string) => {
    if (disabled) return
    if (selected.includes(symbol)) {
      onChange(selected.filter(s => s !== symbol))
    } else {
      onChange([...selected, symbol])
    }
  }

  const remove = (symbol: string) => {
    if (disabled) return
    onChange(selected.filter(s => s !== symbol))
  }

  return (
    <div className="space-y-2">
      {/* Selected badges */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map(s => (
            <Badge key={s} variant="secondary" className="text-xs flex items-center gap-1">
              {s}
              {!disabled && (
                <X className="h-3 w-3 cursor-pointer hover:text-destructive" onClick={() => remove(s)} />
              )}
            </Badge>
          ))}
        </div>
      )}

      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search symbols..."
          className="pl-8 h-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          disabled={disabled}
        />
      </div>

      {/* Symbol list */}
      <ScrollArea className="h-[120px] rounded-md border p-2">
        {filtered.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-2">
            {search ? "No matching symbols" : "No symbols in Market Watch"}
          </p>
        ) : (
          <div className="space-y-1">
            {filtered.map(s => (
              <div
                key={s.symbol}
                className={`flex items-center gap-2 px-2 py-1 rounded-sm cursor-pointer hover:bg-muted/50 ${
                  selected.includes(s.symbol) ? "bg-muted" : ""
                } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
                onClick={() => toggle(s.symbol)}
              >
                <Checkbox
                  checked={selected.includes(s.symbol)}
                  disabled={disabled}
                  className="pointer-events-none"
                />
                <span className="text-sm font-medium">{s.symbol}</span>
                <span className="text-xs text-muted-foreground truncate">{s.description}</span>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

export default function AutomationPage() {
  const [status, setStatus] = useState<any>(null)
  const [config, setConfig] = useState<any>(null)
  const [dailyCycleStatus, setDailyCycleStatus] = useState<DailyCycleStatus | null>(null)
  const [pendingPredictions, setPendingPredictions] = useState<PendingPrediction[]>([])
  const [marketWatchSymbols, setMarketWatchSymbols] = useState<MarketWatchSymbol[]>([])
  // null = not initialized yet, Set = user has interacted or initial load done
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string> | null>(null)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [diagnostics, setDiagnostics] = useState<{
    imports_ok: boolean
    config_ok: boolean
    scheduler_ok: boolean
    errors: string[]
    traceback?: string
  } | null>(null)
  const [diagnosing, setDiagnosing] = useState(false)

  // Dynamic automation instances
  interface InstanceState {
    status: QuantAutomationStatus | null
    history: QuantAutomationHistory | null
    config: Partial<QuantAutomationConfig>
    actionLoading: string | null
    error: string | null
    expanded: boolean
    testSymbol: string
    testResult: any
  }
  const [instances, setInstances] = useState<Record<string, InstanceState>>({})
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [newInstanceName, setNewInstanceName] = useState('')
  const [newInstancePipeline, setNewInstancePipeline] = useState<string>('quant')
  const [newInstanceSymbols, setNewInstanceSymbols] = useState<string[]>(['XAUUSD'])
  const [renamingInstance, setRenamingInstance] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')

  // Retry failed trade state
  const [retryWizardOpen, setRetryWizardOpen] = useState(false)
  const [retryTradeData, setRetryTradeData] = useState<any>(null)
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  const [detailResult, setDetailResult] = useState<any>(null)

  // Market status
  const [marketStatus, setMarketStatus] = useState<Record<string, { open: boolean; reason: string }>>({})
  const [marketSession, setMarketSession] = useState<string>("")

  // Ref to track if we've done initial selection (survives re-renders, not HMR)
  const hasInitialized = useRef(false)

  // Polling function - updates data but NEVER touches selectedSymbols
  const fetchData = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true)
    const [statusRes, configRes, dailyCycleRes, predictionsRes, marketWatchRes, instancesRes] = await Promise.all([
      getPortfolioStatus(),
      getPortfolioConfig(),
      getDailyCycleStatus(),
      getPendingPredictions(),
      getMarketWatchSymbols(),
      listAutomationInstances(),
    ])
    if (statusRes.data) setStatus(statusRes.data)
    if (configRes.data) setConfig(configRes.data)
    if (dailyCycleRes.data) setDailyCycleStatus(dailyCycleRes.data)
    if (predictionsRes.data) setPendingPredictions(predictionsRes.data.predictions || [])
    if (marketWatchRes.data?.symbols) {
      setMarketWatchSymbols(marketWatchRes.data.symbols)
    }

    // Update dynamic instances from server
    if (instancesRes.data?.instances) {
      const serverInstances = instancesRes.data.instances as Record<string, QuantAutomationStatus>
      // Fetch history for all running instances
      const instanceNames = Object.keys(serverInstances)
      const historyResults = await Promise.all(
        instanceNames.map(name => getQuantAutomationHistory(name))
      )

      setInstances(prev => {
        const updated: Record<string, InstanceState> = {}
        instanceNames.forEach((name, idx) => {
          const serverStatus = serverInstances[name]
          const prevInstance = prev[name]
          updated[name] = {
            status: serverStatus,
            history: historyResults[idx]?.data || prevInstance?.history || null,
            config: serverStatus.config || prevInstance?.config || { instance_name: name },
            actionLoading: prevInstance?.actionLoading || null,
            error: prevInstance?.error || null,
            expanded: prevInstance?.expanded ?? false,
            testSymbol: prevInstance?.testSymbol || '',
            testResult: prevInstance?.testResult || null,
          }
        })
        return updated
      })

      // Fetch market status for all configured symbols across all instances
      const allSymbols = new Set<string>()
      Object.values(serverInstances).forEach(inst => {
        inst.config?.symbols?.forEach((s: string) => allSymbols.add(s))
      })
      if (allSymbols.size > 0) {
        const mktRes = await getMarketStatusMulti(Array.from(allSymbols))
        if (mktRes.data) {
          setMarketStatus(mktRes.data.symbols || {})
          setMarketSession(mktRes.data.session || "")
        }
      }
    }

    if (showLoading) setLoading(false)
  }, [])

  const toggleSymbol = (symbol: string) => {
    setSelectedSymbols(prev => {
      const current = prev ?? new Set<string>()
      const newSet = new Set(current)
      if (newSet.has(symbol)) {
        newSet.delete(symbol)
      } else {
        newSet.add(symbol)
      }
      return newSet
    })
  }

  const selectAllSymbols = () => {
    setSelectedSymbols(new Set(marketWatchSymbols.map(s => s.symbol)))
  }

  const deselectAllSymbols = () => {
    setSelectedSymbols(new Set())
  }

  // Initial load effect - runs once on mount
  useEffect(() => {
    let isMounted = true

    const initializeData = async () => {
      setLoading(true)
      const [statusRes, configRes, dailyCycleRes, predictionsRes, marketWatchRes, instancesRes] = await Promise.all([
        getPortfolioStatus(),
        getPortfolioConfig(),
        getDailyCycleStatus(),
        getPendingPredictions(),
        getMarketWatchSymbols(),
        listAutomationInstances(),
      ])

      if (!isMounted) return

      if (statusRes.data) setStatus(statusRes.data)
      if (configRes.data) setConfig(configRes.data)
      if (dailyCycleRes.data) setDailyCycleStatus(dailyCycleRes.data)
      if (predictionsRes.data) setPendingPredictions(predictionsRes.data.predictions || [])
      if (marketWatchRes.data?.symbols) {
        setMarketWatchSymbols(marketWatchRes.data.symbols)
        if (!hasInitialized.current) {
          hasInitialized.current = true
          const marketWatchSymbolNames = new Set(marketWatchRes.data.symbols.map((s: MarketWatchSymbol) => s.symbol))
          const persistedSymbols = dailyCycleRes.data?.symbols || []
          if (persistedSymbols.length > 0) {
            const validPersistedSymbols = persistedSymbols.filter((s: string) => marketWatchSymbolNames.has(s))
            setSelectedSymbols(new Set(validPersistedSymbols))
          } else {
            setSelectedSymbols(marketWatchSymbolNames)
          }
        }
      }

      // Initialize dynamic instances
      if (instancesRes.data?.instances) {
        const serverInstances = instancesRes.data.instances as Record<string, QuantAutomationStatus>
        const instanceNames = Object.keys(serverInstances)
        const historyResults = await Promise.all(
          instanceNames.map(name => getQuantAutomationHistory(name))
        )

        if (!isMounted) return

        const initial: Record<string, InstanceState> = {}
        instanceNames.forEach((name, idx) => {
          const serverStatus = serverInstances[name]
          initial[name] = {
            status: serverStatus,
            history: historyResults[idx]?.data || null,
            config: serverStatus.config || { instance_name: name },
            actionLoading: null,
            error: null,
            expanded: false,
            testSymbol: '',
            testResult: null,
          }
        })
        setInstances(initial)

        // Fetch market status
        const allSymbols = new Set<string>()
        Object.values(serverInstances).forEach(inst => {
          inst.config?.symbols?.forEach((s: string) => allSymbols.add(s))
        })
        if (allSymbols.size > 0) {
          const mktRes = await getMarketStatusMulti(Array.from(allSymbols))
          if (mktRes.data) {
            setMarketStatus(mktRes.data.symbols || {})
            setMarketSession(mktRes.data.session || "")
          }
        }
      }

      setLoading(false)
    }

    initializeData()

    const interval = setInterval(() => fetchData(false), 10000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, []) // Empty deps - only run on mount

  // Persist selected symbols whenever they change (after initial load)
  useEffect(() => {
    // Skip if not initialized yet or no symbols selected
    if (!hasInitialized.current || selectedSymbols === null) return

    // Debounce the save to avoid too many API calls
    const timeoutId = setTimeout(() => {
      saveSelectedSymbols(Array.from(selectedSymbols)).catch(err => {
        console.error("Failed to save selected symbols:", err)
      })
    }, 500)

    return () => clearTimeout(timeoutId)
  }, [selectedSymbols])

  const [startError, setStartError] = useState<string | null>(null)

  const handleStart = async () => {
    setActionLoading("start")
    setStartError(null)

    const result = await startPortfolioAutomation()
    if (result.data?.error) {
      setStartError(result.data.error)
      console.error("Automation start failed:", result.data.error)
      setActionLoading(null)
      return
    }

    // Poll until automation is confirmed running (or timeout after 30s)
    const maxAttempts = 15
    const pollInterval = 2000 // 2 seconds

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))
      const statusRes = await getPortfolioStatus()

      if (statusRes.data?.running) {
        // Automation confirmed running
        await fetchData()
        setActionLoading(null)
        return
      }

      // Check if it failed to start
      if (statusRes.data?.stop_reason && statusRes.data.stop_reason !== "user_stopped") {
        setStartError(`Automation failed: ${statusRes.data.stop_reason}`)
        await fetchData()
        setActionLoading(null)
        return
      }
    }

    // Timeout - still not confirmed running
    setStartError("Automation did not start within 30 seconds. Check logs for errors.")
    await fetchData()
    setActionLoading(null)
  }

  const handleStop = async () => {
    setActionLoading("stop")
    await stopPortfolioAutomation()

    // Poll until automation is confirmed stopped (or timeout after 10s)
    const maxAttempts = 5
    const pollInterval = 2000

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))
      const statusRes = await getPortfolioStatus()

      if (!statusRes.data?.running) {
        // Automation confirmed stopped
        await fetchData()
        setActionLoading(null)
        return
      }
    }

    // Timeout but continue anyway
    await fetchData()
    setActionLoading(null)
  }

  const handleDiagnose = async () => {
    setDiagnosing(true)
    const result = await diagnosePortfolioAutomation()
    if (result.data) {
      setDiagnostics(result.data)
    }
    setDiagnosing(false)
  }

  const handleTrigger = async (cycle: "morning" | "midday" | "evening") => {
    setActionLoading(cycle)
    await triggerDailyCycle(cycle)
    await fetchData()
    setActionLoading(null)
  }

  const handleDailyCycleStart = async () => {
    setActionLoading("dc-start")
    await startDailyCycle({
      symbols: Array.from(selectedSymbols ?? []),
      run_at: 9,
      stagger_minutes: 5,
    })

    // Poll until daily cycle is confirmed running (or timeout after 30s)
    const maxAttempts = 15
    const pollInterval = 2000

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))
      const statusRes = await getDailyCycleStatus()

      if (statusRes.data?.running) {
        await fetchData()
        setActionLoading(null)
        return
      }
    }

    // Timeout
    await fetchData()
    setActionLoading(null)
  }

  const handleDailyCycleStop = async () => {
    setActionLoading("dc-stop")
    await stopDailyCycle()

    // Poll until daily cycle is confirmed stopped (or timeout after 10s)
    const maxAttempts = 5
    const pollInterval = 2000

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))
      const statusRes = await getDailyCycleStatus()

      if (!statusRes.data?.running) {
        await fetchData()
        setActionLoading(null)
        return
      }
    }

    // Timeout
    await fetchData()
    setActionLoading(null)
  }

  // Dynamic instance helpers
  const updateInstance = (name: string, updates: Partial<InstanceState>) => {
    setInstances(prev => ({
      ...prev,
      [name]: { ...prev[name], ...updates },
    }))
  }

  // Instance action handlers
  const handleInstanceStart = async (name: string) => {
    const inst = instances[name]
    if (!inst) return
    updateInstance(name, { actionLoading: "start", error: null })
    const result = await startQuantAutomation({ ...inst.config, instance_name: name })
    if (result.data) {
      await fetchData()
    } else if (result.error) {
      updateInstance(name, { error: result.error })
    }
    updateInstance(name, { actionLoading: null })
  }

  const handleInstanceStop = async (name: string) => {
    updateInstance(name, { actionLoading: "stop" })
    await stopQuantAutomation(name)
    await fetchData()
    updateInstance(name, { actionLoading: null })
  }

  const handleInstancePause = async (name: string) => {
    updateInstance(name, { actionLoading: "pause" })
    await pauseQuantAutomation(name)
    await fetchData()
    updateInstance(name, { actionLoading: null })
  }

  const handleInstanceResume = async (name: string) => {
    updateInstance(name, { actionLoading: "resume" })
    await resumeQuantAutomation(name)
    await fetchData()
    updateInstance(name, { actionLoading: null })
  }

  const handleInstanceTest = async (name: string) => {
    const inst = instances[name]
    if (!inst?.testSymbol) return
    updateInstance(name, { actionLoading: "test" })
    const pipeline = inst.config.pipeline || 'quant'
    let testResult: any = null
    if (pipeline === 'volume_profile') {
      const result = await runVpQuantAnalysis(inst.testSymbol, 'H1')
      if (result.data) testResult = result.data
    } else {
      const result = await testQuantAnalysis(inst.testSymbol, pipeline)
      if (result.data) testResult = result.data
    }
    updateInstance(name, { actionLoading: null, testResult })
  }

  const handleInstanceConfigUpdate = (name: string, key: keyof QuantAutomationConfig, value: any) => {
    updateInstance(name, { config: { ...instances[name]?.config, [key]: value } })
    updateQuantAutomationConfig({ [key]: value }, name)
  }

  const handleInstanceDelete = async (name: string) => {
    const result = await deleteAutomationConfig(name)
    if (result.data) {
      setInstances(prev => {
        const next = { ...prev }
        delete next[name]
        return next
      })
    }
  }

  const handleInstanceRename = async (oldName: string) => {
    const newName = renameValue.trim()
    if (!newName || newName === oldName) {
      setRenamingInstance(null)
      return
    }
    if (instances[newName]) {
      alert(`Instance "${newName}" already exists.`)
      return
    }
    updateInstance(oldName, { actionLoading: 'rename' })
    const result = await renameAutomationInstance(oldName, newName)
    if (result.error) {
      updateInstance(oldName, { actionLoading: null, error: result.error })
    } else {
      setRenamingInstance(null)
      await fetchData()
    }
  }

  const handleAddInstance = async () => {
    if (!newInstanceSymbols.length) return
    const name = newInstanceName.trim() || `${newInstancePipeline}_${newInstanceSymbols[0] || 'default'}`.toLowerCase()

    const config: Partial<QuantAutomationConfig> = {
      instance_name: name,
      pipeline: newInstancePipeline as any,
      symbols: newInstanceSymbols,
      timeframe: 'H1',
      analysis_interval_seconds: 180,
      auto_execute: false,
      min_confidence: 0.65,
      max_positions_per_symbol: 1,
      max_total_positions: 3,
      enable_trailing_stop: true,
      default_lot_size: 0.01,
      max_risk_per_trade_pct: 1.0,
    }

    // Save config to backend (without starting)
    const result = await updateQuantAutomationConfig(config, name)
    if (result.error) {
      alert(`Failed to create instance: ${result.error}`)
      return
    }
    await fetchData()
    setAddDialogOpen(false)
    setNewInstanceName('')
    setNewInstancePipeline('quant')
    setNewInstanceSymbols(['XAUUSD'])
  }

  const pipelineLabels: Record<string, string> = {
    quant: "Quant Analyst",
    smc_quant: "SMC Quant",
    breakout_quant: "Breakout Quant",
    volume_profile: "Volume Profile",
    multi_agent: "Multi-Agent AI",
  }

  const pipelineColors: Record<string, string> = {
    quant: "text-purple-500",
    smc_quant: "text-emerald-500",
    breakout_quant: "text-orange-500",
    volume_profile: "text-blue-500",
    multi_agent: "text-amber-500",
  }

  const executionModeColors: Record<string, string> = {
    FULL_AUTO: "bg-green-500",
    SEMI_AUTO: "bg-yellow-500",
    PAPER: "bg-blue-500",
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Portfolio Automation</h1>
          <p className="text-muted-foreground">
            Manage automated trading cycles and configuration
          </p>
        </div>
        <Button variant="outline" onClick={() => fetchData(true)} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Status & Controls */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5" />
              Automation Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Status</span>
              <Badge variant={status?.running ? "success" : "secondary"} className="text-sm">
                {status?.running ? "Running" : "Stopped"}
              </Badge>
            </div>
            {status?.pid && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Process ID</span>
                <span className="font-mono">{status.pid}</span>
              </div>
            )}
            {status?.last_run && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Last Run</span>
                <span>{formatDate(status.last_run)}</span>
              </div>
            )}
            {status?.next_run && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Next Run</span>
                <span>{formatDate(status.next_run)}</span>
              </div>
            )}

            <Separator />

            <div className="flex gap-2">
              {status?.running ? (
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={handleStop}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === "stop" ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Square className="mr-2 h-4 w-4" />
                  )}
                  Stop Automation
                </Button>
              ) : (
                <Button
                  variant="success"
                  className="flex-1"
                  onClick={handleStart}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === "start" ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  Start Automation
                </Button>
              )}
            </div>

            {startError && (
              <div className="mt-3 p-3 rounded-md bg-destructive/10 border border-destructive/20">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium text-destructive">Failed to start automation</p>
                    <p className="text-muted-foreground mt-1 break-words">{startError}</p>
                  </div>
                </div>
              </div>
            )}

            {status?.stop_reason && status?.stop_reason !== "user_stopped" && !status?.running && (
              <div className="mt-3 p-3 rounded-md bg-destructive/10 border border-destructive/20">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium text-destructive">Automation stopped unexpectedly</p>
                    <p className="text-muted-foreground mt-1 break-words">{status.stop_reason}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Diagnostics section */}
            <div className="mt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDiagnose}
                disabled={diagnosing}
              >
                {diagnosing ? (
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                ) : (
                  <Settings className="mr-2 h-3 w-3" />
                )}
                Run Diagnostics
              </Button>

              {diagnostics && (
                <div className="mt-3 p-3 rounded-md bg-muted/50 text-sm space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant={diagnostics.imports_ok ? "success" : "destructive"}>
                      Imports {diagnostics.imports_ok ? "OK" : "FAIL"}
                    </Badge>
                    <Badge variant={diagnostics.config_ok ? "success" : "destructive"}>
                      Config {diagnostics.config_ok ? "OK" : "FAIL"}
                    </Badge>
                    <Badge variant={diagnostics.scheduler_ok ? "success" : "destructive"}>
                      Scheduler {diagnostics.scheduler_ok ? "OK" : "FAIL"}
                    </Badge>
                  </div>
                  {diagnostics.errors.length > 0 && (
                    <div className="text-destructive space-y-1">
                      {diagnostics.errors.map((err, i) => (
                        <p key={i} className="break-words">{err}</p>
                      ))}
                    </div>
                  )}
                  {diagnostics.traceback && (
                    <details className="text-xs">
                      <summary className="cursor-pointer text-muted-foreground">Show traceback</summary>
                      <pre className="mt-2 p-2 bg-muted rounded overflow-auto max-h-40 whitespace-pre-wrap">
                        {diagnostics.traceback}
                      </pre>
                    </details>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Manual Triggers */}
        <Card>
          <CardHeader>
            <CardTitle>Manual Triggers</CardTitle>
            <CardDescription>Manually trigger daily cycle phases</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => handleTrigger("morning")}
              disabled={actionLoading !== null}
            >
              {actionLoading === "morning" ? (
                <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              ) : (
                <Sun className="mr-3 h-5 w-5 text-yellow-500" />
              )}
              <div className="text-left">
                <p className="font-medium">Morning Analysis</p>
                <p className="text-xs text-muted-foreground">
                  Analyze symbols and execute trades
                </p>
              </div>
            </Button>

            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => handleTrigger("midday")}
              disabled={actionLoading !== null}
            >
              {actionLoading === "midday" ? (
                <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              ) : (
                <Clock className="mr-3 h-5 w-5 text-blue-500" />
              )}
              <div className="text-left">
                <p className="font-medium">Midday Review</p>
                <p className="text-xs text-muted-foreground">
                  Review positions and update stops
                </p>
              </div>
            </Button>

            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => handleTrigger("evening")}
              disabled={actionLoading !== null}
            >
              {actionLoading === "evening" ? (
                <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              ) : (
                <Moon className="mr-3 h-5 w-5 text-purple-500" />
              )}
              <div className="text-left">
                <p className="font-medium">Evening Reflect</p>
                <p className="text-xs text-muted-foreground">
                  Process closed trades and learn patterns
                </p>
              </div>
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Daily Cycle (Prediction Learning) */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Daily Learning Cycle
              <HelpTooltip
                content="Tracks predictions and evaluates them after 24 hours. Correct predictions strengthen memories with high confidence. Incorrect ones get analyzed for lessons. Over time this improves the AI's decision-making by learning what works and what doesn't in specific market conditions."
              />
            </CardTitle>
            <CardDescription>
              Continuous prediction tracking and memory refinement
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Status</span>
              <Badge variant={dailyCycleStatus?.running ? "success" : "secondary"} className="text-sm">
                {dailyCycleStatus?.running ? "Running" : "Stopped"}
              </Badge>
            </div>
            {dailyCycleStatus?.pid && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Process ID</span>
                <span className="font-mono">{dailyCycleStatus.pid}</span>
              </div>
            )}
            {dailyCycleStatus?.running && (
              <div className="space-y-1">
                <span className="text-muted-foreground text-sm">Tracking Symbols</span>
                {dailyCycleStatus.symbols && dailyCycleStatus.symbols.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {dailyCycleStatus.symbols.map((sym) => (
                      <Badge key={sym} variant="outline" className="text-xs">
                        {sym}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-yellow-500">
                    No symbols configured. Stop and restart with symbols selected.
                  </div>
                )}
              </div>
            )}
            {dailyCycleStatus?.run_at !== null && dailyCycleStatus?.run_at !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Daily Run Time</span>
                <span>{dailyCycleStatus.run_at}:00</span>
              </div>
            )}
            {dailyCycleStatus?.last_run && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Last Run</span>
                <span>{formatDate(dailyCycleStatus.last_run)}</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Pending Predictions</span>
              <Badge variant="outline">{dailyCycleStatus?.pending_predictions || 0}</Badge>
            </div>



            <Separator />

            {/* Symbol selection (only when stopped) */}
            {!dailyCycleStatus?.running && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Select Symbols</span>
                    <HelpTooltip
                      content="Select which symbols to track for predictions. Fewer symbols = lower LLM costs. Each symbol runs a full analysis cycle daily."
                      iconClassName="h-3 w-3"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs"
                      onClick={selectAllSymbols}
                    >
                      All
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs"
                      onClick={deselectAllSymbols}
                    >
                      None
                    </Button>
                  </div>
                </div>
                {marketWatchSymbols.length > 0 ? (
                  <div className="grid grid-cols-2 gap-2 max-h-[150px] overflow-y-auto pr-2">
                    {marketWatchSymbols.map((s) => (
                      <div key={s.symbol} className="flex items-center gap-2">
                        <Checkbox
                          id={`symbol-${s.symbol}`}
                          checked={selectedSymbols?.has(s.symbol) ?? false}
                          onCheckedChange={() => toggleSymbol(s.symbol)}
                        />
                        <label
                          htmlFor={`symbol-${s.symbol}`}
                          className="text-sm cursor-pointer truncate"
                          title={s.description}
                        >
                          {s.symbol}
                        </label>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-yellow-500">
                    No symbols in MT5 Market Watch. Add symbols in MT5 to track them.
                  </div>
                )}
                <div className="text-xs text-muted-foreground">
                  {selectedSymbols?.size ?? 0} of {marketWatchSymbols.length} symbol(s) selected
                </div>
              </div>
            )}

            <div className="flex gap-2">
              {dailyCycleStatus?.running ? (
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={handleDailyCycleStop}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === "dc-stop" ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Square className="mr-2 h-4 w-4" />
                  )}
                  Stop Learning
                </Button>
              ) : (
                <Button
                  variant="success"
                  className="flex-1"
                  onClick={handleDailyCycleStart}
                  disabled={actionLoading !== null || !selectedSymbols || selectedSymbols.size === 0}
                >
                  {actionLoading === "dc-start" ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  Start Learning
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Pending Predictions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Pending Predictions
              <HelpTooltip
                content="Predictions waiting to be evaluated. After 24 hours, the actual price movement is compared to the prediction. Correct predictions boost memory confidence, incorrect ones trigger lesson extraction for future improvement."
              />
            </CardTitle>
            <CardDescription>Awaiting 24-hour evaluation</CardDescription>
          </CardHeader>
          <CardContent>
            {pendingPredictions.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">
                No pending predictions. Start the daily cycle to begin tracking.
              </p>
            ) : (
              <ScrollArea className="h-[200px]">
                <div className="space-y-3">
                  {pendingPredictions.map((pred, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                    >
                      <div className="flex items-center gap-3">
                        {pred.signal === "BUY" ? (
                          <TrendingUp className="h-5 w-5 text-green-500" />
                        ) : pred.signal === "SELL" ? (
                          <TrendingDown className="h-5 w-5 text-red-500" />
                        ) : (
                          <Clock className="h-5 w-5 text-gray-500" />
                        )}
                        <div>
                          <p className="font-medium">{pred.symbol}</p>
                          <p className="text-xs text-muted-foreground">
                            {pred.signal} @ ${pred.price_at_analysis?.toFixed(2)}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge
                          variant={pred.expected_direction === "up" ? "buy" : pred.expected_direction === "down" ? "sell" : "secondary"}
                        >
                          {pred.expected_direction}
                        </Badge>
                        <p className="text-xs text-muted-foreground mt-1">
                          Due: {formatDate(pred.evaluation_due)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Configuration */}
      {config && !config.error && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Execution Mode</span>
                <Badge
                  className={executionModeColors[config.execution_mode] || "bg-gray-500"}
                >
                  {config.execution_mode}
                </Badge>
              </div>

              {config.schedule && (
                <>
                  <Separator />
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Schedule</p>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Morning</p>
                        <p>{config.schedule.morning_analysis_hour || 8}:00</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Midday</p>
                        <p>{config.schedule.midday_review_hour || 13}:00</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Evening</p>
                        <p>{config.schedule.evening_reflect_hour || 20}:00</p>
                      </div>
                    </div>
                    {config.schedule.timezone && (
                      <p className="text-xs text-muted-foreground">
                        Timezone: {config.schedule.timezone}
                      </p>
                    )}
                  </div>
                </>
              )}

              {config.risk_limits && (
                <>
                  <Separator />
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Risk Limits</p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Daily Loss Limit</p>
                        <p>{config.risk_limits.daily_loss_limit_pct}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Max Consecutive Losses</p>
                        <p>{config.risk_limits.max_consecutive_losses}</p>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Symbols */}
          <Card>
            <CardHeader>
              <CardTitle>Configured Symbols</CardTitle>
              <CardDescription>
                {config.symbols?.length || 0} symbols configured for automation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                {config.symbols?.length > 0 ? (
                  <div className="space-y-3">
                    {config.symbols.map((sym: any) => (
                      <div
                        key={sym.symbol}
                        className="rounded-lg border p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{sym.symbol}</span>
                          <Badge variant="outline">
                            Max {sym.max_positions} pos
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                          <div>Risk Budget: {sym.risk_budget_pct}%</div>
                          <div>Min Confidence: {(sym.min_confidence * 100).toFixed(0)}%</div>
                        </div>
                        {sym.timeframes && (
                          <div className="flex flex-wrap gap-1">
                            {sym.timeframes.map((tf: string) => (
                              <Badge key={tf} variant="secondary" className="text-xs">
                                {tf}
                              </Badge>
                            ))}
                          </div>
                        )}
                        {sym.correlation_group && (
                          <p className="text-xs text-muted-foreground">
                            Correlation Group: {sym.correlation_group}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-muted-foreground">
                    No symbols configured
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}

      {config?.error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 py-4 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            <span>Failed to load configuration: {config.error}</span>
          </CardContent>
        </Card>
      )}

      {/* Quant Automation Instances */}
      <Separator className="my-8" />
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Zap className="h-6 w-6 text-purple-500" />
              Quant Automation
            </h2>
            <p className="text-muted-foreground">
              Configure and run multiple automation instances with different pipelines and symbols
            </p>
          </div>
          <div className="flex items-center gap-2">
            {marketSession && (
              <Badge variant="outline" className="capitalize text-xs">
                {marketSession.replace(/_/g, " ")}
              </Badge>
            )}
            <Button onClick={() => setAddDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Instance
            </Button>
          </div>
        </div>

        {Object.keys(instances).length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Zap className="h-12 w-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">No automation instances configured.</p>
              <p className="text-sm text-muted-foreground mt-1">Click "Add Instance" to create one.</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {Object.entries(instances).map(([name, inst]) => {
              const instStatus = inst.status?.status || "stopped"
              const isRunning = instStatus === "running"
              const isPaused = instStatus === "paused"
              const isStopped = !isRunning && !isPaused
              const pipeline = inst.config.pipeline || "quant"

              return (
                <Card key={name}>
                  {/* Instance Header - Always visible */}
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <CardTitle className="flex items-center gap-2 text-lg">
                          <Zap className={`h-5 w-5 ${pipelineColors[pipeline] || "text-purple-500"}`} />
                          {renamingInstance === name ? (
                            <form
                              className="flex items-center gap-1"
                              onSubmit={(e) => { e.preventDefault(); handleInstanceRename(name) }}
                            >
                              <Input
                                value={renameValue}
                                onChange={(e) => setRenameValue(e.target.value)}
                                className="h-7 w-40 text-sm"
                                autoFocus
                                onBlur={() => handleInstanceRename(name)}
                                onKeyDown={(e) => { if (e.key === 'Escape') setRenamingInstance(null) }}
                              />
                              <Button type="submit" variant="ghost" size="sm" className="h-7 w-7 p-0">
                                <Check className="h-3.5 w-3.5" />
                              </Button>
                            </form>
                          ) : (
                            <span className="flex items-center gap-1 group">
                              {name}
                              {isStopped && (
                                <button
                                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                                  onClick={() => { setRenamingInstance(name); setRenameValue(name) }}
                                >
                                  <Pencil className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
                                </button>
                              )}
                            </span>
                          )}
                        </CardTitle>
                        <Badge variant="outline" className="capitalize text-xs">
                          {pipelineLabels[pipeline] || pipeline}
                        </Badge>
                        <Badge
                          variant={
                            isRunning ? "success" :
                            isPaused ? "warning" :
                            instStatus === "error" ? "destructive" :
                            "secondary"
                          }
                          className="text-xs"
                        >
                          {instStatus}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        {/* Symbol badges with market status */}
                        <div className="flex flex-wrap gap-1">
                          {inst.config.symbols?.map(s => (
                            <Badge key={s} variant="outline" className="text-xs flex items-center gap-1">
                              <span
                                className={`h-2 w-2 rounded-full ${
                                  marketStatus[s]?.open ? "bg-green-500" : "bg-red-500"
                                }`}
                                title={marketStatus[s]?.reason || "Unknown"}
                              />
                              {s}
                            </Badge>
                          ))}
                        </div>

                        {/* Action buttons */}
                        {isRunning ? (
                          <>
                            <Button variant="outline" size="sm" onClick={() => handleInstancePause(name)} disabled={inst.actionLoading !== null}>
                              {inst.actionLoading === "pause" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Pause className="h-4 w-4" />}
                            </Button>
                            <Button variant="destructive" size="sm" onClick={() => handleInstanceStop(name)} disabled={inst.actionLoading !== null}>
                              {inst.actionLoading === "stop" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                            </Button>
                          </>
                        ) : isPaused ? (
                          <>
                            <Button variant="success" size="sm" onClick={() => handleInstanceResume(name)} disabled={inst.actionLoading !== null}>
                              {inst.actionLoading === "resume" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                            </Button>
                            <Button variant="destructive" size="sm" onClick={() => handleInstanceStop(name)} disabled={inst.actionLoading !== null}>
                              {inst.actionLoading === "stop" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                            </Button>
                          </>
                        ) : (
                          <>
                            <Button variant="success" size="sm" onClick={() => handleInstanceStart(name)} disabled={inst.actionLoading !== null || !inst.config.symbols?.length}>
                              {inst.actionLoading === "start" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => handleInstanceDelete(name)} disabled={inst.actionLoading !== null}>
                              <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                            </Button>
                          </>
                        )}

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => updateInstance(name, { expanded: !inst.expanded })}
                        >
                          {inst.expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>

                    {/* Quick info row */}
                    {inst.status?.config && !inst.expanded && (
                      <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                        <span>Interval: {inst.config.analysis_interval_seconds}s</span>
                        <span>Auto Execute: {inst.config.auto_execute ? "On" : "Off"}</span>
                        {inst.status.positions && (
                          <span>Positions: {inst.status.positions.managed}/{inst.status.positions.max_total}</span>
                        )}
                        {inst.status.guardrails && !inst.status.guardrails.can_trade && (
                          <span className="text-yellow-500">Guardrails active</span>
                        )}
                      </div>
                    )}
                  </CardHeader>

                  {/* Expanded content */}
                  {inst.expanded && (
                    <CardContent className="space-y-6 pt-0">
                      {inst.error && (
                        <div className="rounded-md bg-red-500/10 border border-red-500/20 p-3">
                          <p className="text-sm text-red-500">{inst.error}</p>
                        </div>
                      )}

                      {inst.status?.error && (
                        <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
                          <p className="text-sm text-destructive">{inst.status.error}</p>
                        </div>
                      )}

                      {inst.status?.guardrails && !inst.status.guardrails.can_trade && (
                        <div className="p-3 rounded-md bg-yellow-500/10 border border-yellow-500/20">
                          <p className="text-sm text-yellow-600">{inst.status.guardrails.status_summary}</p>
                        </div>
                      )}

                      <div className="grid gap-6 lg:grid-cols-2">
                        {/* Configuration */}
                        <div className="space-y-4">
                          <h3 className="text-sm font-semibold flex items-center gap-2">
                            <Settings className="h-4 w-4" />
                            Configuration
                          </h3>

                          <div className="space-y-2">
                            <Label>Symbols</Label>
                            <SymbolMultiSelect
                              selected={inst.config.symbols || []}
                              onChange={(syms) => handleInstanceConfigUpdate(name, 'symbols', syms)}
                              available={marketWatchSymbols}
                              disabled={isRunning}
                            />
                          </div>

                          <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                              <Label className="text-xs">Analysis Interval (s)</Label>
                              <Input
                                type="number"
                                value={inst.config.analysis_interval_seconds}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'analysis_interval_seconds', parseInt(e.target.value) || 180)}
                                min={60} max={3600} disabled={isRunning}
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Min Confidence</Label>
                              <Input
                                type="number"
                                value={inst.config.min_confidence}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'min_confidence', parseFloat(e.target.value) || 0.65)}
                                min={0} max={1} step={0.05} disabled={isRunning}
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Default Lot Size</Label>
                              <Input
                                type="number"
                                value={inst.config.default_lot_size}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'default_lot_size', parseFloat(e.target.value) || 0.01)}
                                min={0.01} step={0.01} disabled={isRunning}
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Max Risk %</Label>
                              <Input
                                type="number"
                                value={inst.config.max_risk_per_trade_pct}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'max_risk_per_trade_pct', parseFloat(e.target.value) || 1)}
                                min={0.1} max={5} step={0.1} disabled={isRunning}
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Max Pos/Symbol</Label>
                              <Input
                                type="number"
                                value={inst.config.max_positions_per_symbol}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'max_positions_per_symbol', parseInt(e.target.value) || 1)}
                                min={1} max={5} disabled={isRunning}
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs">Max Total Pos</Label>
                              <Input
                                type="number"
                                value={inst.config.max_total_positions}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'max_total_positions', parseInt(e.target.value) || 3)}
                                min={1} max={10} disabled={isRunning}
                              />
                            </div>
                          </div>

                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Label htmlFor={`auto-exec-${name}`} className="text-xs">Auto Execute</Label>
                                <HelpTooltip content="When enabled, trades will be automatically placed when signals meet confidence threshold." />
                              </div>
                              <Switch
                                id={`auto-exec-${name}`}
                                checked={inst.config.auto_execute}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'auto_execute', v)}
                                disabled={isRunning}
                              />
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Label htmlFor={`trail-${name}`} className="text-xs">Trailing Stop</Label>
                                <HelpTooltip content="Automatically trail stop loss as price moves in your favor using ATR-based distance." />
                              </div>
                              <Switch
                                id={`trail-${name}`}
                                checked={inst.config.enable_trailing_stop}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'enable_trailing_stop', v)}
                                disabled={isRunning}
                              />
                            </div>
                          </div>

                          {/* Test Analysis */}
                          <Separator />
                          <div className="space-y-2">
                            <h4 className="text-xs font-semibold flex items-center gap-1">
                              <Brain className="h-3 w-3" /> Test Analysis
                            </h4>
                            <div className="flex gap-2">
                              <Input
                                placeholder="Symbol (e.g., XAUUSD)"
                                className="h-8 text-xs"
                                value={inst.testSymbol}
                                onChange={(e) => updateInstance(name, { testSymbol: e.target.value.toUpperCase() })}
                              />
                              <Button
                                size="sm"
                                className="h-8"
                                onClick={() => handleInstanceTest(name)}
                                disabled={!inst.testSymbol || inst.actionLoading === "test"}
                              >
                                {inst.actionLoading === "test" ? <Loader2 className="h-3 w-3 animate-spin" /> : "Test"}
                              </Button>
                            </div>
                            {inst.testResult && (
                              <div className="p-3 rounded-lg bg-muted/50 space-y-2 text-xs">
                                <div className="flex items-center justify-between">
                                  <span className="font-medium">{inst.testResult.symbol}</span>
                                  <Badge
                                    variant={inst.testResult.signal === "BUY" ? "buy" : inst.testResult.signal === "SELL" ? "sell" : "secondary"}
                                    className="text-xs"
                                  >
                                    {inst.testResult.signal}
                                  </Badge>
                                </div>
                                <div className="grid grid-cols-2 gap-1">
                                  <span className="text-muted-foreground">Confidence: <span className="font-medium text-foreground">{((inst.testResult.confidence || 0) * 100).toFixed(0)}%</span></span>
                                  {inst.testResult.entry_price && <span className="text-muted-foreground">Entry: <span className="font-medium text-foreground">{inst.testResult.entry_price?.toFixed(2)}</span></span>}
                                  {inst.testResult.stop_loss && <span className="text-muted-foreground">SL: <span className="font-medium text-red-500">{inst.testResult.stop_loss?.toFixed(2)}</span></span>}
                                  {inst.testResult.take_profit && <span className="text-muted-foreground">TP: <span className="font-medium text-green-500">{inst.testResult.take_profit?.toFixed(2)}</span></span>}
                                </div>
                                {(inst.testResult.rationale || inst.testResult.justification) && (
                                  <p className="text-muted-foreground">{(inst.testResult.rationale || inst.testResult.justification)?.slice(0, 150)}...</p>
                                )}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Recent Results */}
                        <div className="space-y-4">
                          <h3 className="text-sm font-semibold flex items-center gap-2">
                            <Target className="h-4 w-4" />
                            Recent Results
                          </h3>
                          {inst.history?.analysis_results && inst.history.analysis_results.length > 0 ? (
                            <ScrollArea className="h-[350px]">
                              <div className="space-y-2">
                                {inst.history.analysis_results.slice().reverse().map((result, idx) => (
                                  <div key={idx} className="flex items-center justify-between p-2.5 rounded-lg bg-muted/50">
                                    <div className="flex items-center gap-2">
                                      {result.signal === "BUY" ? (
                                        <TrendingUp className="h-4 w-4 text-green-500" />
                                      ) : result.signal === "SELL" ? (
                                        <TrendingDown className="h-4 w-4 text-red-500" />
                                      ) : (
                                        <Clock className="h-4 w-4 text-gray-500" />
                                      )}
                                      <div>
                                        <p className="text-sm font-medium">{result.symbol}</p>
                                        <p className="text-xs text-muted-foreground">
                                          {result.signal} ({(result.confidence * 100).toFixed(0)}%)
                                        </p>
                                      </div>
                                    </div>
                                    <div className="text-right flex flex-col items-end gap-1">
                                      <div className="flex items-center gap-1">
                                        {result.executed ? (
                                          <Badge variant="success" className="text-xs">Executed</Badge>
                                        ) : result.execution_error ? (
                                          <>
                                            <Badge variant="destructive" className="text-xs">Failed</Badge>
                                            <Button
                                              variant="ghost"
                                              size="sm"
                                              className="h-5 px-1 text-xs text-yellow-500 hover:text-yellow-400"
                                              onClick={() => {
                                                setRetryTradeData({
                                                  symbol: result.symbol,
                                                  signal: result.signal,
                                                  stop_loss: result.stop_loss,
                                                  take_profit: result.take_profit,
                                                  rationale: result.rationale,
                                                  execution_error: result.execution_error,
                                                })
                                                setRetryWizardOpen(true)
                                              }}
                                            >
                                              <RotateCcw className="h-3 w-3 mr-0.5" />
                                              Retry
                                            </Button>
                                          </>
                                        ) : (
                                          <Badge variant="secondary" className="text-xs">Signal</Badge>
                                        )}
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
                                          onClick={() => {
                                            setDetailResult(result)
                                            setDetailModalOpen(true)
                                          }}
                                        >
                                          <Eye className="h-3.5 w-3.5" />
                                        </Button>
                                      </div>
                                      <p className="text-xs text-muted-foreground">
                                        {formatDate(result.timestamp)}
                                      </p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </ScrollArea>
                          ) : (
                            <p className="text-sm text-muted-foreground py-8 text-center">
                              No results yet. Start the automation to see signals.
                            </p>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  )}
                </Card>
              )
            })}
          </div>
        )}
      </div>

      {/* Add Instance Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Automation Instance</DialogTitle>
            <DialogDescription>
              Create a new automation instance with a pipeline and symbol(s).
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Instance Name (optional)</Label>
              <Input
                value={newInstanceName}
                onChange={(e) => setNewInstanceName(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_'))}
                placeholder="Auto-generated from pipeline + symbol"
              />
              <p className="text-xs text-muted-foreground">Leave empty for auto-generated name</p>
            </div>
            <div className="space-y-2">
              <Label>Pipeline</Label>
              <Select value={newInstancePipeline} onValueChange={setNewInstancePipeline}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="quant">Quant Analyst</SelectItem>
                  <SelectItem value="smc_quant">SMC Quant</SelectItem>
                  <SelectItem value="breakout_quant">Breakout Quant</SelectItem>
                  <SelectItem value="volume_profile">Volume Profile</SelectItem>
                  <SelectItem value="multi_agent">Multi-Agent AI</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Symbols</Label>
              <SymbolMultiSelect
                selected={newInstanceSymbols}
                onChange={setNewInstanceSymbols}
                available={marketWatchSymbols}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleAddInstance} disabled={newInstanceSymbols.length === 0}>
              Create Instance
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Retry Trade Wizard */}
      {retryTradeData && (
        <TradeExecutionWizard
          open={retryWizardOpen}
          onOpenChange={(open) => {
            setRetryWizardOpen(open)
            if (!open) setRetryTradeData(null)
          }}
          symbol={retryTradeData.symbol}
          signal={retryTradeData.signal}
          suggestedStopLoss={retryTradeData.stop_loss}
          suggestedTakeProfit={retryTradeData.take_profit}
          rationale={retryTradeData.rationale}
          failureReason={retryTradeData.execution_error}
        />
      )}

      {/* Result Detail Modal */}
      <Dialog open={detailModalOpen} onOpenChange={setDetailModalOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {detailResult?.symbol}
              {detailResult?.signal === "BUY" ? (
                <Badge variant="buy">{detailResult.signal}</Badge>
              ) : detailResult?.signal === "SELL" ? (
                <Badge variant="sell">{detailResult.signal}</Badge>
              ) : (
                <Badge variant="secondary">{detailResult?.signal}</Badge>
              )}
              <span className="text-sm font-normal text-muted-foreground">
                {detailResult?.confidence != null && `${(detailResult.confidence * 100).toFixed(0)}% confidence`}
              </span>
            </DialogTitle>
          </DialogHeader>
          {detailResult && (
            <div className="space-y-4">
              {/* Price Levels */}
              {(detailResult.entry_price || detailResult.stop_loss || detailResult.take_profit) && (
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-lg bg-muted/50 p-2.5 text-center">
                    <p className="text-xs text-muted-foreground mb-0.5">Entry</p>
                    <p className="text-sm font-medium text-blue-500">
                      {detailResult.entry_price != null ? detailResult.entry_price.toFixed(detailResult.entry_price > 100 ? 2 : 5) : "—"}
                    </p>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-2.5 text-center">
                    <p className="text-xs text-muted-foreground mb-0.5">Stop Loss</p>
                    <p className="text-sm font-medium text-red-500">
                      {detailResult.stop_loss != null ? detailResult.stop_loss.toFixed(detailResult.stop_loss > 100 ? 2 : 5) : "—"}
                    </p>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-2.5 text-center">
                    <p className="text-xs text-muted-foreground mb-0.5">Take Profit</p>
                    <p className="text-sm font-medium text-green-500">
                      {detailResult.take_profit != null ? detailResult.take_profit.toFixed(detailResult.take_profit > 100 ? 2 : 5) : "—"}
                    </p>
                  </div>
                </div>
              )}

              {/* Justification & Invalidation */}
              {(() => {
                const raw = detailResult.rationale || ""
                const parts = raw.split(/\*\*Invalidation\*\*:\s*/i)
                const justification = parts[0]?.trim()
                const invalidation = parts[1]?.trim()
                return (
                  <>
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Justification</p>
                      <ScrollArea className="h-[120px]">
                        <p className="text-sm whitespace-pre-wrap">{justification || "No justification provided"}</p>
                      </ScrollArea>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Invalidation</p>
                      <p className="text-sm text-yellow-500 whitespace-pre-wrap">{invalidation || "—"}</p>
                    </div>
                  </>
                )
              })()}

              {/* Meta Info */}
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Pipeline</span>
                  <span className={pipelineColors[detailResult.pipeline] || ""}>
                    {pipelineLabels[detailResult.pipeline] || detailResult.pipeline}
                  </span>
                </div>
                {detailResult.duration_seconds != null && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Duration</span>
                    <span>{detailResult.duration_seconds.toFixed(1)}s</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span>
                    {detailResult.executed ? (
                      <Badge variant="success" className="text-xs">Executed{detailResult.execution_ticket ? ` #${detailResult.execution_ticket}` : ""}</Badge>
                    ) : detailResult.execution_error ? (
                      <Badge variant="destructive" className="text-xs">Failed</Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">Signal Only</Badge>
                    )}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Time</span>
                  <span>{formatDate(detailResult.timestamp)}</span>
                </div>
              </div>

              {/* Execution Error */}
              {detailResult.execution_error && (
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Execution Error</p>
                  <p className="text-sm text-red-500">{detailResult.execution_error}</p>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
