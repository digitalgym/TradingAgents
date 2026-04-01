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
  // Scanner
  runPairScan,
  getScannerStatus,
  PairScoreResult,
  ScanResultResponse,
  // Quant Automation
  listAutomationInstances,
  getQuantAutomationStatus,
  startQuantAutomation,
  stopQuantAutomation,
  startAllQuantAutomations,
  stopAllQuantAutomations,
  pauseQuantAutomation,
  resumeQuantAutomation,
  updateQuantAutomationConfig,
  deleteAutomationConfig,
  renameAutomationInstance,
  testQuantAnalysis,
  getQuantAutomationHistory,
  runVpQuantAnalysis,
  getMarketStatusMulti,
  getSymbolLimits,
  updateSymbolLimit,
  startTune,
  getTuneStatus,
  applyTuneResult,
  getTuneHistory,
  revertTune,
  TuneTaskState,
  TuneResultEntry,
  TuneHistoryRecord,
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
  Radar,
} from "lucide-react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { HelpTooltip } from "@/components/ui/help-tooltip"
import { TradeExecutionWizard } from "@/components/TradeExecutionWizard"
import { TradeManagementSection } from "@/components/automation/trade-management-section"
import { useAutomationStatus, AutomationStatusEvent } from "@/lib/websocket"

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
    tuneStatus: TuneTaskState | null
    tuneError: string | null
    tuneExpanded: boolean
    tuneHistory: TuneHistoryRecord[] | null
    tuneHistoryKey: string | null
    tuneHistoryOpen: boolean
  }
  const [instances, setInstances] = useState<Record<string, InstanceState>>({})
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [newInstanceName, setNewInstanceName] = useState('')
  const [newInstancePipeline, setNewInstancePipeline] = useState<string>('smc_quant_basic')
  const [newInstanceTimeframe, setNewInstanceTimeframe] = useState<string>('H1')
  const [newInstanceSymbols, setNewInstanceSymbols] = useState<string[]>(['XAUUSD'])
  const [renamingInstance, setRenamingInstance] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')

  // Recent results filter (per instance)
  const [resultSignalFilters, setResultSignalFilters] = useState<Record<string, string>>({}) // "all" | "actionable" | "BUY" | "SELL" | "HOLD"

  // Retry failed trade state
  const [retryWizardOpen, setRetryWizardOpen] = useState(false)
  const [retryTradeData, setRetryTradeData] = useState<any>(null)
  const [detailModalOpen, setDetailModalOpen] = useState(false)
  const [detailResult, setDetailResult] = useState<any>(null)

  // Market status
  const [marketStatus, setMarketStatus] = useState<Record<string, { open: boolean; reason: string }>>({})
  const [marketSession, setMarketSession] = useState<string>("")

  // Scanner state
  const [scanResult, setScanResult] = useState<ScanResultResponse | null>(null)
  const [scanning, setScanning] = useState(false)
  const [scanMinScore, setScanMinScore] = useState(40)
  const [scanTimeframe, setScanTimeframe] = useState("H4")

  // Global symbol position limits
  const [symbolLimits, setSymbolLimits] = useState<Record<string, { max_positions: number }>>({})
  const [symbolLimitsLoaded, setSymbolLimitsLoaded] = useState(false)

  // Ref to track if we've done initial selection (survives re-renders, not HMR)
  const hasInitialized = useRef(false)

  // WebSocket: cross-client status sync (updates status badge on other tabs)
  useAutomationStatus(useCallback((event: AutomationStatusEvent) => {
    const { instance, status: newStatus } = event
    const finalStatuses = ["running", "stopped", "paused", "error"] as const
    if (!(finalStatuses as readonly string[]).includes(newStatus)) return

    setInstances(prev => {
      const inst = prev[instance]
      if (!inst?.status) return prev
      // Only update status badge — don't touch actionLoading or config
      if (inst.status.status === newStatus) return prev
      return {
        ...prev,
        [instance]: {
          ...inst,
          status: {
            ...inst.status,
            status: newStatus as QuantAutomationStatus["status"],
            running: newStatus === "running",
          },
        },
      }
    })
  }, []))

  // Polling function - updates data but NEVER touches selectedSymbols
  const fetchData = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true)
    const [statusRes, configRes, dailyCycleRes, predictionsRes, marketWatchRes, instancesRes, symbolLimitsRes] = await Promise.all([
      getPortfolioStatus(),
      getPortfolioConfig(),
      getDailyCycleStatus(),
      getPendingPredictions(),
      getMarketWatchSymbols(),
      listAutomationInstances(),
      getSymbolLimits(),
    ])
    if (statusRes.data) setStatus(statusRes.data)
    if (configRes.data) setConfig(configRes.data)
    if (dailyCycleRes.data) setDailyCycleStatus(dailyCycleRes.data)
    if (predictionsRes.data) setPendingPredictions(predictionsRes.data.predictions || [])
    if (marketWatchRes.data?.symbols) {
      setMarketWatchSymbols(marketWatchRes.data.symbols)
    }
    if (symbolLimitsRes.data?.limits) {
      setSymbolLimits(symbolLimitsRes.data.limits)
      setSymbolLimitsLoaded(true)
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
            // Local config is source of truth for the form — only use server on first load
            config: prevInstance?.config || serverStatus.config || { instance_name: name },
            actionLoading: prevInstance?.actionLoading || null,
            error: prevInstance?.error || null,
            expanded: prevInstance?.expanded ?? false,
            testSymbol: prevInstance?.testSymbol || '',
            testResult: prevInstance?.testResult || null,
            tuneStatus: prevInstance?.tuneStatus || null,
            tuneError: prevInstance?.tuneError || null,
            tuneExpanded: prevInstance?.tuneExpanded ?? false,
            tuneHistory: prevInstance?.tuneHistory || null,
            tuneHistoryKey: prevInstance?.tuneHistoryKey || null,
            tuneHistoryOpen: prevInstance?.tuneHistoryOpen ?? false,
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
          const marketWatchSymbolNames = new Set<string>(marketWatchRes.data.symbols.map((s: MarketWatchSymbol) => s.symbol))
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
            tuneStatus: null,
            tuneError: null,
            tuneExpanded: false,
            tuneHistory: null,
            tuneHistoryKey: null,
            tuneHistoryOpen: false,
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

  // Poll instance status until it reaches expected state (or timeout)
  const pollUntilStatus = async (name: string, expected: string[], maxSeconds = 30) => {
    const interval = 2000
    const maxAttempts = Math.ceil((maxSeconds * 1000) / interval)
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise(r => setTimeout(r, interval))
      const res = await getQuantAutomationStatus(name)
      if (res.data && expected.includes(res.data.status)) {
        updateInstance(name, { status: res.data, actionLoading: null, error: null })
        return
      }
    }
    // Timeout — clear spinner, refresh
    await fetchData()
    updateInstance(name, { actionLoading: null })
  }

  // Instance action handlers
  const handleInstanceStart = async (name: string) => {
    const inst = instances[name]
    if (!inst) return
    updateInstance(name, { actionLoading: "start", error: null })
    const result = await startQuantAutomation({ ...inst.config, instance_name: name })
    if (result.error) {
      updateInstance(name, { error: result.error, actionLoading: null })
      return
    }
    await pollUntilStatus(name, ["running", "error"])
  }

  const handleInstanceStop = async (name: string) => {
    updateInstance(name, { actionLoading: "stop" })
    const result = await stopQuantAutomation(name)
    if (result.error) {
      updateInstance(name, { error: result.error, actionLoading: null })
      return
    }
    await pollUntilStatus(name, ["stopped"])
  }

  const handleInstancePause = async (name: string) => {
    updateInstance(name, { actionLoading: "pause" })
    const result = await pauseQuantAutomation(name)
    if (result.error) {
      updateInstance(name, { error: result.error, actionLoading: null })
      return
    }
    await pollUntilStatus(name, ["paused"])
  }

  const handleInstanceResume = async (name: string) => {
    updateInstance(name, { actionLoading: "resume" })
    const result = await resumeQuantAutomation(name)
    if (result.error) {
      updateInstance(name, { error: result.error, actionLoading: null })
      return
    }
    await pollUntilStatus(name, ["running"])
  }

  const handleStartAll = async () => {
    const stoppedNames = Object.entries(instances)
      .filter(([, inst]) => !inst.status?.running)
      .map(([name]) => name)
    if (stoppedNames.length === 0) return

    // Set all to loading — WebSocket will clear each when worker confirms
    stoppedNames.forEach(name => updateInstance(name, { actionLoading: "start", error: null }))

    const result = await startAllQuantAutomations()
    if (result.error) {
      stoppedNames.forEach(name => updateInstance(name, { actionLoading: null, error: result.error || "Start failed" }))
      return
    }
    // Check for per-instance errors
    if (result.data?.results) {
      for (const [name, res] of Object.entries(result.data.results)) {
        if (res.error) updateInstance(name, { error: res.error, actionLoading: null })
      }
    }
    // Poll all until they reach running (or timeout)
    await Promise.all(stoppedNames.map(name => pollUntilStatus(name, ["running", "error"])))
  }

  const handleStopAll = async () => {
    const runningNames = Object.entries(instances)
      .filter(([, inst]) => inst.status?.running)
      .map(([name]) => name)
    if (runningNames.length === 0) return

    runningNames.forEach(name => updateInstance(name, { actionLoading: "stop" }))

    const result = await stopAllQuantAutomations()
    if (result.error) {
      runningNames.forEach(name => updateInstance(name, { actionLoading: null, error: result.error || "Stop failed" }))
      return
    }
    // Poll all until they reach stopped (or timeout)
    await Promise.all(runningNames.map(name => pollUntilStatus(name, ["stopped"])))
  }

  const handleRunScan = async () => {
    setScanning(true)
    setScanResult(null)
    const result = await runPairScan(scanMinScore, 10, scanTimeframe)
    if (result.data) {
      setScanResult(result.data)
    }
    setScanning(false)
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

  const handleStartTune = async (name: string) => {
    updateInstance(name, { tuneStatus: null, tuneError: null, actionLoading: "tune" })
    const result = await startTune(name)
    if (result.error) {
      updateInstance(name, { tuneError: result.error, actionLoading: null })
      return
    }
    // Start polling for tune status
    updateInstance(name, { actionLoading: null })
    pollTuneStatus(name)
  }

  const pollTuneStatus = (name: string) => {
    const interval = setInterval(async () => {
      const result = await getTuneStatus(name)
      if (result.error) {
        clearInterval(interval)
        updateInstance(name, { tuneError: result.error })
        return
      }
      if (result.data) {
        updateInstance(name, { tuneStatus: result.data, tuneExpanded: true, tuneError: null })
        if (result.data.status !== "running") {
          clearInterval(interval)
        }
      }
    }, 1000)
  }

  const handleApplyTuneResult = async (name: string) => {
    updateInstance(name, { actionLoading: "apply_tune" })
    const result = await applyTuneResult(name)
    if (result.error) {
      updateInstance(name, { tuneError: result.error, actionLoading: null })
    } else {
      updateInstance(name, { actionLoading: null })
      await fetchData()
      loadTuneHistory(name)
    }
  }

  const loadTuneHistory = async (name: string) => {
    const result = await getTuneHistory(name)
    if (result.data) {
      updateInstance(name, { tuneHistory: result.data.records, tuneHistoryKey: result.data.key })
    }
  }

  const handleRevertTune = async (name: string, recordIndex: number) => {
    updateInstance(name, { actionLoading: "revert_tune" })
    const result = await revertTune(name, recordIndex)
    if (result.error) {
      updateInstance(name, { tuneError: result.error, actionLoading: null })
    } else {
      updateInstance(name, { actionLoading: null })
      await fetchData()
      loadTuneHistory(name)
    }
  }

  const handleInstanceConfigUpdate = (name: string, key: keyof QuantAutomationConfig, value: any) => {
    // Update local state immediately (local config is source of truth for the form)
    updateInstance(name, { config: { ...instances[name]?.config, [key]: value } })
    // Persist to server (fire-and-forget is fine — local state won't be overwritten by polling)
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

    const defaults = pipelineDefaults[newInstancePipeline] || pipelineDefaults.smc_quant_basic
    const config: Partial<QuantAutomationConfig> = {
      instance_name: name,
      pipeline: newInstancePipeline as any,
      symbols: newInstanceSymbols,
      timeframe: newInstanceTimeframe,
      analysis_interval_seconds: defaults.interval,
      auto_execute: false,
      min_confidence: defaults.confidence,
      max_positions_per_symbol: 1,
      enable_trailing_stop: true,
      trailing_stop_atr_multiplier: defaults.atrMultiplier,
      enable_breakeven_stop: true,
      move_to_breakeven_atr_mult: 1.5,
      enable_reversal_close: true,
      default_lot_size: 0.01,
      max_risk_per_trade_pct: 1.0,
      // Auto-enable scanner for SCANNER_AUTO pipeline
      ...(newInstancePipeline === 'scanner_auto' ? {
        enable_scanner: true,
        scanner_interval_seconds: 300,
        scanner_min_score: 40,
        scanner_max_candidates: 3,
        scanner_timeframe: 'H4',
      } : {}),
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
    setNewInstancePipeline('smc_quant_basic')
    setNewInstanceTimeframe('H1')
    setNewInstanceSymbols(['XAUUSD'])
  }

  const pipelineLabels: Record<string, string> = {
    rule_based: "SMC Rule-Based",
    smc_quant_basic: "SMC Quant",
    smc_quant: "SMC Quant Deep",
    smc_mtf: "SMC Multi-Timeframe",
    breakout_quant: "BB Squeeze Breakout",
    donchian_breakout: "Donchian Breakout",
    gold_trend_pullback: "Gold Trend-Pullback",
    range_quant: "Range Mean Reversion",
    volume_profile: "Volume Profile",
    multi_agent: "Multi-Agent AI",
    xgboost: "ML Auto-Select",
    xgboost_ensemble: "ML Ensemble Vote",
    scanner_auto: "Scanner Auto",
    gold_silver_pullback: "Gold/Silver Pullback",
    gold_silver_pullback_mtf: "Gold/Silver MTF",
  }

  const pipelineColors: Record<string, string> = {
    rule_based: "text-cyan-500",
    smc_quant_basic: "text-purple-500",
    smc_quant: "text-emerald-500",
    smc_mtf: "text-indigo-500",
    breakout_quant: "text-orange-500",
    donchian_breakout: "text-orange-400",
    gold_trend_pullback: "text-yellow-400",
    range_quant: "text-teal-500",
    volume_profile: "text-blue-500",
    multi_agent: "text-amber-500",
    xgboost: "text-rose-500",
    xgboost_ensemble: "text-pink-500",
    scanner_auto: "text-fuchsia-500",
    gold_silver_pullback: "text-yellow-500",
    gold_silver_pullback_mtf: "text-amber-500",
  }

  // Pipeline descriptions with backtest results from XAUUSD (Oct 2022 - Dec 2025, 827 D1 bars)
  // Backtest script: tests/backtest_all_pipelines.py
  const pipelineDescriptions: Record<string, { summary: string; details: string; recommendedTimeframes: string; recommendedInterval: string }> = {
    rule_based: {
      summary: "Pure rules-based SMC analysis with no LLM. Instant, zero API cost.",
      details: "Backtest (XAUUSD D1): 75% WR, Sharpe 1.49, PF 2.22 on 12 trades. BUY signals 87.5% WR vs SELL 50%. Uses OB proximity + structural bias. Best with hold=3 bars, OB proximity 0.8%. Strong but selective - fewer signals than other strategies.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "15-30 min (instant computation)",
    },
    smc_quant_basic: {
      summary: "LLM-powered SMC analysis with standard depth.",
      details: "Uses same underlying SMC signals as rule-based (75% WR on D1) but adds LLM reasoning for nuanced trade decisions. Good balance of analysis depth and API cost. LLM may improve or filter the 12 raw signals per ~800 bars.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "30-60 min",
    },
    smc_quant: {
      summary: "Deep SMC analysis with extended lookback and LLM confluence.",
      details: "Extended analysis window with deeper structural assessment. Same base signals as rule-based (75% WR D1, Sharpe 1.49) but LLM evaluates more context. Higher API cost for more thorough analysis.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "60-120 min",
    },
    smc_mtf: {
      summary: "Multi-timeframe OTE + Channel analysis. No LLM, instant & free.",
      details: "Combines Higher TF (D1/H4) bias with Lower TF (H4/H1) entries using OTE (Fibonacci retracement), regression channels, weekend gaps, and protected level flips. Alignment score 0-100 determines signal strength. Requires entry confirmation (weekend gap or protected flip) for high-confidence trades.",
      recommendedTimeframes: "D1 (HTF=D1, LTF=H4), H4 (HTF=H4, LTF=H1)",
      recommendedInterval: "15-30 min (instant computation)",
    },
    breakout_quant: {
      summary: "Consolidation breakout detection with BB squeeze.",
      details: "Backtest (XAUUSD D1): 60% WR, Sharpe 0.50, PF 1.47 on 25 trades. Best with lookback=15, hold=10 bars, squeeze>60. BUY avg +2.2% vs SELL avg -0.004%. On H4: 54.4% WR with 960 trades, PF 1.64 (more trades, lower WR).",
      recommendedTimeframes: "D1 (best WR), H4 (more signals)",
      recommendedInterval: "30-60 min",
    },
    range_quant: {
      summary: "Mean-reversion at range extremes with structural bias filter. Now has hard regime gate.",
      details: "LLM analysis only fires when market is confirmed ranging (MR score >45, ADX <20, trend strength <0.35). Blocked for volatile pairs (XAUUSD, XAGUSD, BTCUSD). BUY at discount with bullish bias ~70% WR. Includes BB bounce + ratio z-score mechanical fallbacks.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "60-120 min (ranges evolve slowly)",
    },
    donchian_breakout: {
      summary: "Donchian channel breakout with trend/squeeze regime gate + silver-lead confirmation for metals. No LLM.",
      details: "Enters on Donchian 20-period channel breakouts when market is trending (ADX >25) or in a BB squeeze that's expanding. SMA50/200 trend bias filter prevents counter-trend entries. For XAUUSD/XAGUSD, requires silver to confirm by breaking its own Donchian within 3 bars. XGBoost model learns which breakouts succeed. 3:1 R:R target with 2x ATR stop.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "15-60 min (instant inference)",
    },
    gold_trend_pullback: {
      summary: "BUY-only gold pullback strategy. 64% WR, Sharpe 1.18, PF 5.5. No LLM, instant.",
      details: "Backtested on 800 D1 bars (3.2 years): buys pullbacks in gold's uptrend when SMA50>SMA100, price dips ≥1x ATR from swing high, and a bullish reversal candle forms. Requires 2+ confluence (RSI oversold, BB lower touch, silver rising, Au/Ag ratio not extreme, near EMA50). SL=1.5x ATR, TP=3:1 R:R, trailing stop 2x ATR. ~18 trades/year. XAUUSD only — sells blocked.",
      recommendedTimeframes: "D1",
      recommendedInterval: "240 min (daily signals)",
    },
    volume_profile: {
      summary: "Value area reversion using volume distribution.",
      details: "Backtest (XAUUSD D1): 49.7% WR, negative Sharpe on pure signals. BUY outside VA: 57.5% WR. SELL outside VA: 47.6% WR. Volume profile works better as confluence filter with other strategies than standalone. LLM analysis adds significant value here.",
      recommendedTimeframes: "H4, D1",
      recommendedInterval: "30-60 min",
    },
    multi_agent: {
      summary: "Full multi-agent debate with multiple analyst perspectives.",
      details: "Runs multiple specialized analysts that debate and reach consensus. Highest analysis quality but also highest API cost and latency. Not backtested (requires LLM). Best for important decisions on higher timeframes.",
      recommendedTimeframes: "H4, D1",
      recommendedInterval: "120-240 min",
    },
    xgboost: {
      summary: "XGBoost ML strategy — auto-selects best model for the pair. Zero API cost, sub-100ms.",
      details: "Uses trained XGBoost models to predict price direction. Strategy selector picks the best model (trend following, mean reversion, breakout, SMC zones, or volume profile) based on backtest performance for this pair. No LLM calls — pure ML inference. Must train models first.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "15-60 min (instant inference)",
    },
    xgboost_ensemble: {
      summary: "XGBoost ensemble — multiple ML models vote. Zero API cost, sub-100ms.",
      details: "Runs all available trained XGBoost models and uses majority voting (min 2 agree at 60%+ probability). Higher conviction than single model but requires multiple trained models. No LLM calls.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "15-60 min (instant inference)",
    },
    scanner_auto: {
      summary: "Scans 17 pairs, detects regime, auto-routes each to the best pipeline. Fully adaptive.",
      details: "The scanner analyses momentum, trend strength, volatility and squeeze across the full watchlist. For each qualifying pair it classifies the regime (strong trend → rule_based, moderate trend → xgboost, ranging → range_quant, squeeze → breakout_quant, volatile trend → smc_mtf) and dispatches to the best pipeline automatically. As markets shift regime, the automation shifts with them. Scanner is always enabled in this mode.",
      recommendedTimeframes: "D1 (best), H4",
      recommendedInterval: "15-60 min (scanner + analysis)",
    },
    gold_silver_pullback: {
      summary: "Gold pullback strategy on D1 using silver momentum + Au/Ag ratio. Long-term, ~8 trades/year.",
      details: "D1-only: 75% WR, Sharpe 2.15, PF 6.40 on 28 trades (~3.5 years). Very selective — requires 5/7 confluence (trend, pullback depth, fib zone, reversal candle, silver confirmation, ratio filter, structure). Best for long-term validation alongside the MTF version.",
      recommendedTimeframes: "D1",
      recommendedInterval: "240 min (daily signals)",
    },
    gold_silver_pullback_mtf: {
      summary: "Gold pullback with D1 trend + H4 entries. ~12 trades/year, no LLM.",
      details: "Backtest: 55.8% WR, Sharpe 1.25, PF 3.70 on 43 trades. D1 provides trend filter (SMA50>100) + Au/Ag ratio. H4 provides entry: pullback 1.5x ATR + reversal candle closing above prior high + silver breakout/acceleration. 3:1 RR, wide 2.5x ATR stop. Trades every ~3 weeks. Requires XAUUSD + XAGUSD in MT5.",
      recommendedTimeframes: "D1+H4",
      recommendedInterval: "120 min (H4 candle period)",
    },
  }

  // Sensible defaults per pipeline. Timeframe from XAUUSD backtest (D1 wins across all strategies).
  // Confidence and ATR multiplier are conservative starting points — use the Tune button to
  // find optimal values for your specific symbol.
  const pipelineDefaults: Record<string, { timeframe: string; interval: number; confidence: number; atrMultiplier: number }> = {
    rule_based:     { timeframe: "D1", interval: 900,   confidence: 0.65, atrMultiplier: 1.5 },
    smc_quant_basic:{ timeframe: "D1", interval: 1800,  confidence: 0.65, atrMultiplier: 1.5 },
    smc_quant:      { timeframe: "D1", interval: 3600,  confidence: 0.70, atrMultiplier: 2.0 },
    smc_mtf:        { timeframe: "D1", interval: 900,   confidence: 0.60, atrMultiplier: 1.5 },
    breakout_quant: { timeframe: "D1", interval: 1800,  confidence: 0.65, atrMultiplier: 1.5 },
    range_quant:    { timeframe: "D1", interval: 3600,  confidence: 0.70, atrMultiplier: 2.5 },
    donchian_breakout: { timeframe: "D1", interval: 900, confidence: 0.60, atrMultiplier: 2.0 },
    gold_trend_pullback: { timeframe: "D1", interval: 14400, confidence: 0.40, atrMultiplier: 1.5 },
    volume_profile: { timeframe: "H4", interval: 1800,  confidence: 0.65, atrMultiplier: 2.0 },
    multi_agent:    { timeframe: "D1", interval: 7200,  confidence: 0.70, atrMultiplier: 2.0 },
    xgboost:        { timeframe: "D1", interval: 900,   confidence: 0.60, atrMultiplier: 1.5 },
    xgboost_ensemble:{ timeframe: "D1", interval: 900,  confidence: 0.60, atrMultiplier: 1.5 },
    scanner_auto:    { timeframe: "D1", interval: 900,  confidence: 0.60, atrMultiplier: 1.5 },
    gold_silver_pullback: { timeframe: "D1", interval: 14400, confidence: 0.60, atrMultiplier: 2.5 },
    gold_silver_pullback_mtf: { timeframe: "D1", interval: 7200, confidence: 0.60, atrMultiplier: 2.5 },
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

      {/* Trade Management Agent */}
      <Separator className="my-8" />
      <TradeManagementSection />

      {/* Pair Scanner */}
      <Separator className="my-8" />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Radar className="h-5 w-5 text-rose-500" />
            Pair Scanner
          </CardTitle>
          <CardDescription>
            Scan the full watchlist for high-momentum pairs. Use with XGBoost pipelines or manually pick symbols.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <label className="text-xs text-muted-foreground">Min Score</label>
              <HelpTooltip content="Minimum momentum score (0-100) to qualify. Score = ATR expansion (25) + ADX strength (25) + directional move (20) + structure break (15) + EMA alignment (10) + volume (5). 40 = moderate, 60 = strict." />
            </div>
            <Input
              type="number"
              className="h-8 w-20 text-xs"
              min={0}
              max={100}
              value={scanMinScore}
              onChange={(e) => setScanMinScore(parseInt(e.target.value) || 40)}
            />
            <div className="flex items-center gap-1">
              <label className="text-xs text-muted-foreground">Timeframe</label>
              <HelpTooltip content="Timeframe for momentum data. H4 balances signal and noise. D1 for swing. H1 for fast rotation." />
            </div>
            <Select value={scanTimeframe} onValueChange={setScanTimeframe}>
              <SelectTrigger className="h-8 w-20 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="H1">H1</SelectItem>
                <SelectItem value="H4">H4</SelectItem>
                <SelectItem value="D1">D1</SelectItem>
              </SelectContent>
            </Select>
            <Button size="sm" className="h-8" onClick={handleRunScan} disabled={scanning}>
              {scanning ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Radar className="mr-1 h-3 w-3" />}
              Scan
            </Button>
          </div>

          {scanResult && (
            <div className="space-y-3">
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>Scanned {scanResult.watchlist_size} pairs</span>
                <span>{scanResult.shortlist.length} qualified</span>
                <span>{scanResult.disqualified_count} filtered out</span>
              </div>

              {/* Shortlist */}
              {scanResult.shortlist.length > 0 && (
                <div className="space-y-1">
                  <h4 className="text-xs font-semibold text-green-500">Candidates</h4>
                  <div className="grid gap-1">
                    {scanResult.shortlist.map((pair) => (
                      <div key={pair.symbol} className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-muted/50 text-xs">
                        <span className="font-medium w-16">{pair.symbol}</span>
                        <Badge
                          variant={pair.direction === "LONG" ? "buy" : "sell"}
                          className="text-[10px] px-1.5 py-0"
                        >
                          {pair.direction}
                        </Badge>
                        <div className="flex items-center gap-1">
                          <span className="text-muted-foreground">Score:</span>
                          <span className={`font-bold ${pair.momentum_score >= 70 ? "text-green-500" : pair.momentum_score >= 50 ? "text-yellow-500" : "text-muted-foreground"}`}>
                            {pair.momentum_score}
                          </span>
                        </div>
                        {pair.regime && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-fuchsia-500 text-fuchsia-500">
                            {pair.regime.replace("_", " ")}
                          </Badge>
                        )}
                        {pair.recommended_pipeline && (
                          <Badge variant="outline" className={`text-[10px] px-1.5 py-0 ${pipelineColors[pair.recommended_pipeline] || ""}`}>
                            {pair.recommended_timeframe || ""} → {pipelineLabels[pair.recommended_pipeline] || pair.recommended_pipeline}
                          </Badge>
                        )}
                        <div className="flex items-center gap-2 ml-auto text-muted-foreground">
                          {pair.structure_break && <Badge variant="outline" className="text-[10px] px-1 py-0">Break</Badge>}
                          {pair.ema_alignment && <Badge variant="outline" className="text-[10px] px-1 py-0">EMA</Badge>}
                          {pair.volume_confirmation && <Badge variant="outline" className="text-[10px] px-1 py-0">Vol</Badge>}
                          <span>ADX {pair.adx_strength.toFixed(0)}</span>
                          <span>ATR×{pair.atr_expansion.toFixed(1)}</span>
                          <span>Move {pair.directional_move_pct > 0 ? "+" : ""}{pair.directional_move_pct.toFixed(2)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Disqualified (collapsible) */}
              {scanResult.disqualified && scanResult.disqualified.length > 0 && (
                <details className="text-xs">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                    {scanResult.disqualified.length} disqualified pairs
                  </summary>
                  <div className="grid gap-1 mt-1">
                    {scanResult.disqualified.map((pair) => (
                      <div key={pair.symbol} className="flex items-center gap-2 px-3 py-1 rounded-md text-muted-foreground">
                        <span className="w-16">{pair.symbol}</span>
                        <span className="text-[10px]">{pair.disqualify_reason}</span>
                        <span className="ml-auto">Score: {pair.momentum_score}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          )}
        </CardContent>
      </Card>

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
            {Object.values(instances).some(i => !i.status?.running) && (
              <Button variant="outline" size="sm" onClick={handleStartAll}>
                <Play className="mr-1 h-3 w-3" />
                Start All
              </Button>
            )}
            {Object.values(instances).some(i => i.status?.running) && (
              <Button variant="outline" size="sm" onClick={handleStopAll}>
                <Square className="mr-1 h-3 w-3" />
                Stop All
              </Button>
            )}
            <Button onClick={() => setAddDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Instance
            </Button>
          </div>
        </div>

        {/* Global Symbol Position Limits */}
        {symbolLimitsLoaded && Object.keys(symbolLimits).length > 0 && (
          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Global Position Limits
                <HelpTooltip content="Account-wide max positions per symbol across ALL automations. If 3 automations trade XAUUSD and the global max is 3, each can open 1. Auto-populated from all configured automation symbols." />
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-3 pt-0">
              <div className="flex flex-wrap gap-3">
                {Object.entries(symbolLimits).sort(([a], [b]) => a.localeCompare(b)).map(([sym, cfg]) => (
                  <div key={sym} className="flex items-center gap-2 rounded-md border px-3 py-1.5">
                    <span className="text-sm font-medium">{sym}</span>
                    <span className="text-xs text-muted-foreground">max</span>
                    <Input
                      type="number"
                      className="h-7 w-14 text-xs text-center"
                      value={cfg.max_positions}
                      min={1}
                      max={20}
                      onChange={async (e) => {
                        const val = parseInt(e.target.value)
                        if (val >= 1 && val <= 20) {
                          setSymbolLimits(prev => ({...prev, [sym]: { max_positions: val }}))
                          await updateSymbolLimit(sym, val)
                        }
                      }}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

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
                        {pipelineDescriptions[pipeline] && (
                          <HelpTooltip content={`${pipelineDescriptions[pipeline].summary} Best TF: ${pipelineDescriptions[pipeline].recommendedTimeframes}. Suggested interval: ${pipelineDescriptions[pipeline].recommendedInterval}.`} />
                        )}
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
                        <span>TF: {inst.config.timeframe || 'H1'}</span>
                        <span>Interval: {inst.config.analysis_interval_seconds}s</span>
                        <span>Auto Execute: {inst.config.auto_execute ? "On" : "Off"}</span>
                        {inst.status.positions && (
                          <span>Positions: {inst.status.positions.managed} (max {inst.status.positions.max_per_symbol}/sym)</span>
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
                              <Label className="text-xs">Timeframe</Label>
                              <Select
                                value={inst.config.timeframe || 'H1'}
                                onValueChange={(v) => handleInstanceConfigUpdate(name, 'timeframe', v)}
                                disabled={isRunning}
                              >
                                <SelectTrigger className="h-9">
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
                            </div>
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
                              <div className="flex items-center gap-1">
                                <Label className="text-xs">Max Pos/Symbol</Label>
                                <HelpTooltip content="How many positions THIS automation can open per symbol. The global limit (across all automations) is set in the Global Position Limits table above." />
                              </div>
                              <Input
                                type="number"
                                value={inst.config.max_positions_per_symbol}
                                onChange={(e) => handleInstanceConfigUpdate(name, 'max_positions_per_symbol', parseInt(e.target.value) || 1)}
                                min={1} max={5} disabled={isRunning}
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
                                <Label htmlFor={`delegate-${name}`} className="text-xs">Delegate to Trade Manager</Label>
                                <HelpTooltip content="Hand off position management (trailing stops, breakeven, reversal close) to the centralized Trade Management Agent. When enabled, this instance stops managing its own positions. Make sure the Trade Manager is running before enabling." />
                              </div>
                              <Switch
                                id={`delegate-${name}`}
                                checked={inst.config.delegate_position_management ?? false}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'delegate_position_management', v)}
                                disabled={isRunning}
                              />
                            </div>
                            {inst.config.delegate_position_management && (
                              <div className="p-2 rounded-md bg-blue-500/10 border border-blue-500/20 text-xs text-blue-500">
                                Position management delegated to Trade Manager. Trailing, breakeven, and reversal settings below are ignored.
                              </div>
                            )}
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Label htmlFor={`trail-${name}`} className="text-xs">Trailing Stop</Label>
                                <HelpTooltip content="Automatically trail stop loss as price moves in your favor using ATR-based distance." />
                              </div>
                              <Switch
                                id={`trail-${name}`}
                                checked={inst.config.enable_trailing_stop}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'enable_trailing_stop', v)}
                                disabled={isRunning || inst.config.delegate_position_management}
                              />
                            </div>
                            {inst.config.enable_trailing_stop && (
                              <div className="flex items-center justify-between pl-4">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor={`trail-mult-${name}`} className="text-xs">Trail Distance (ATR×)</Label>
                                  <HelpTooltip content="How far behind price the trailing stop follows, as a multiple of ATR. 1.5× = tight, locks profit fast but risks early exit. 2.5× = moderate, good balance. 3.0-4.0× = wide, gives room but gives back more profit. For XAUUSD, 2.5-3.0× is recommended." />
                                </div>
                                <Input
                                  id={`trail-mult-${name}`}
                                  type="number"
                                  step={0.5}
                                  min={0.5}
                                  max={10}
                                  className="h-7 w-20 text-xs"
                                  value={inst.config.trailing_stop_atr_multiplier ?? 1.5}
                                  onChange={(e) => handleInstanceConfigUpdate(name, 'trailing_stop_atr_multiplier', parseFloat(e.target.value) || 1.5)}
                                  disabled={isRunning}
                                />
                              </div>
                            )}
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Label htmlFor={`be-${name}`} className="text-xs">Breakeven Stop</Label>
                                <HelpTooltip content="Moves SL to entry price once profit exceeds a threshold (N × ATR). Protects from turning a winner into a loser, but can cause early exits if price retraces briefly. Disable this when testing to avoid interference." />
                              </div>
                              <Switch
                                id={`be-${name}`}
                                checked={inst.config.enable_breakeven_stop ?? true}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'enable_breakeven_stop', v)}
                                disabled={isRunning || inst.config.delegate_position_management}
                              />
                            </div>
                            {inst.config.enable_breakeven_stop && (
                              <div className="flex items-center justify-between pl-4">
                                <div className="flex items-center gap-2">
                                  <Label htmlFor={`be-mult-${name}`} className="text-xs">BE Threshold (ATR×)</Label>
                                  <HelpTooltip content="How much profit (as ATR multiple) before moving SL to breakeven. 1.0× = early, moves quickly but more false exits. 1.5× = standard. 2.0-3.0× = conservative, gives trade more room." />
                                </div>
                                <Input
                                  id={`be-mult-${name}`}
                                  type="number"
                                  step={0.5}
                                  min={0.5}
                                  max={10}
                                  className="h-7 w-20 text-xs"
                                  value={inst.config.move_to_breakeven_atr_mult ?? 1.5}
                                  onChange={(e) => handleInstanceConfigUpdate(name, 'move_to_breakeven_atr_mult', parseFloat(e.target.value) || 1.5)}
                                  disabled={isRunning}
                                />
                              </div>
                            )}
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Label htmlFor={`rev-${name}`} className="text-xs">Reversal Close</Label>
                                <HelpTooltip content="Runs analysis each position check cycle and closes the position if a strong reversal signal is detected (opposite direction with >70% confidence). Disable this to let trades run to SL/TP only." />
                              </div>
                              <Switch
                                id={`rev-${name}`}
                                checked={inst.config.enable_reversal_close ?? true}
                                onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'enable_reversal_close', v)}
                                disabled={isRunning || inst.config.delegate_position_management}
                              />
                            </div>
                          </div>

                          {/* Scanner Settings */}
                            <>
                              <Separator />
                              <div className="space-y-2">
                                <h4 className="text-xs font-semibold flex items-center gap-1">
                                  <Search className="h-3 w-3" /> Pair Scanner
                                  <HelpTooltip content="When enabled, the scanner automatically finds high-momentum pairs from the full watchlist (17 pairs) each cycle. Instead of analyzing only your configured symbols, it picks the top movers and runs your chosen pipeline on those. Disable to stick with your manually selected symbols." />
                                </h4>
                                <div className="flex items-center gap-2">
                                  <Switch
                                    checked={inst.config.enable_scanner ?? false}
                                    onCheckedChange={(v) => handleInstanceConfigUpdate(name, 'enable_scanner', v)}
                                    disabled={isRunning}
                                  />
                                  <span className="text-xs text-muted-foreground">
                                    {inst.config.enable_scanner ? "Scanner picks symbols dynamically" : "Using manual symbol list"}
                                  </span>
                                </div>
                                {inst.config.enable_scanner && (
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <div className="flex items-center gap-1">
                                        <label className="text-xs text-muted-foreground">Scan Interval (s)</label>
                                        <HelpTooltip content="How often the scanner re-scans the watchlist. 300s (5 min) is a good default — pairs don't change momentum that fast." />
                                      </div>
                                      <Input
                                        type="number"
                                        className="h-7 text-xs"
                                        value={inst.config.scanner_interval_seconds ?? 300}
                                        onChange={(e) => handleInstanceConfigUpdate(name, 'scanner_interval_seconds', parseInt(e.target.value) || 300)}
                                        disabled={isRunning}
                                      />
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-1">
                                        <label className="text-xs text-muted-foreground">Min Score</label>
                                        <HelpTooltip content="Minimum momentum score (0-100) to qualify. 40 = moderate filter (more signals). 60 = strict (fewer but stronger). Score combines: ATR expansion (25pts), ADX trend strength (25pts), directional move (20pts), structure break (15pts), EMA alignment (10pts), volume (5pts)." />
                                      </div>
                                      <Input
                                        type="number"
                                        className="h-7 text-xs"
                                        min={0}
                                        max={100}
                                        value={inst.config.scanner_min_score ?? 40}
                                        onChange={(e) => handleInstanceConfigUpdate(name, 'scanner_min_score', parseInt(e.target.value) || 40)}
                                        disabled={isRunning}
                                      />
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-1">
                                        <label className="text-xs text-muted-foreground">Max Candidates</label>
                                        <HelpTooltip content="Maximum pairs to trade from scanner results. 3 = focused (recommended). More candidates = more positions to manage but more diversification." />
                                      </div>
                                      <Input
                                        type="number"
                                        className="h-7 text-xs"
                                        min={1}
                                        max={10}
                                        value={inst.config.scanner_max_candidates ?? 3}
                                        onChange={(e) => handleInstanceConfigUpdate(name, 'scanner_max_candidates', parseInt(e.target.value) || 3)}
                                        disabled={isRunning}
                                      />
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-1">
                                        <label className="text-xs text-muted-foreground">Scanner TF</label>
                                        <HelpTooltip content="Timeframe for scanner momentum data. H4 is recommended — captures intraday momentum without noise. D1 for swing trading, H1 for faster rotation." />
                                      </div>
                                      <Select
                                        value={inst.config.scanner_timeframe ?? "H4"}
                                        onValueChange={(v) => handleInstanceConfigUpdate(name, 'scanner_timeframe', v)}
                                        disabled={isRunning}
                                      >
                                        <SelectTrigger className="h-7 text-xs">
                                          <SelectValue />
                                        </SelectTrigger>
                                        <SelectContent>
                                          <SelectItem value="H1">H1</SelectItem>
                                          <SelectItem value="H4">H4</SelectItem>
                                          <SelectItem value="D1">D1</SelectItem>
                                        </SelectContent>
                                      </Select>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </>

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

                          {/* Auto-Tuner */}
                          <Separator />
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <h4 className="text-xs font-semibold flex items-center gap-1">
                                <Zap className="h-3 w-3" /> Tune Parameters
                                <HelpTooltip content="Runs a backtest parameter sweep on historical data for this symbol/pipeline. Finds the optimal timeframe, confidence threshold, and other settings. Takes 30-60 seconds for most pipelines (SMC pipelines may take longer). Results are based on pure-signal backtesting (no LLM) - actual results may differ." />
                              </h4>
                              {inst.config.pipeline !== "multi_agent" && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-7 text-xs gap-1"
                                  onClick={() => handleStartTune(name)}
                                  disabled={inst.actionLoading === "tune" || inst.tuneStatus?.status === "running"}
                                >
                                  {inst.tuneStatus?.status === "running" ? (
                                    <><Loader2 className="h-3 w-3 animate-spin" /> Tuning...</>
                                  ) : (
                                    <><Zap className="h-3 w-3" /> Tune</>
                                  )}
                                </Button>
                              )}
                            </div>

                            {inst.config.pipeline === "multi_agent" && (
                              <p className="text-xs text-muted-foreground">Multi-agent pipeline requires LLM and cannot be auto-tuned.</p>
                            )}

                            {inst.tuneError && inst.tuneStatus?.status !== "error" && (
                              <div className="p-2 rounded bg-red-500/10 border border-red-500/20 text-xs text-red-400">
                                {inst.tuneError}
                              </div>
                            )}

                            {/* Step-based progress during tuning */}
                            {inst.tuneStatus?.status === "running" && inst.tuneStatus.progress && (
                              <div className="space-y-2 p-3 rounded-lg bg-muted/50">
                                {/* Steps list */}
                                {inst.tuneStatus.progress.steps && inst.tuneStatus.progress.steps.length > 0 && (
                                  <div className="space-y-1">
                                    {inst.tuneStatus.progress.steps.map((step, i) => (
                                      <div key={i} className="flex items-center gap-2 text-xs">
                                        {step.status === "done" ? (
                                          <Check className="h-3 w-3 text-green-500 flex-shrink-0" />
                                        ) : step.status === "running" ? (
                                          <Loader2 className="h-3 w-3 animate-spin text-primary flex-shrink-0" />
                                        ) : (
                                          <div className="h-3 w-3 rounded-full border border-muted-foreground/30 flex-shrink-0" />
                                        )}
                                        <span className={step.status === "done" ? "text-muted-foreground" : step.status === "running" ? "text-foreground font-medium" : "text-muted-foreground/50"}>
                                          {step.name}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                {/* Current phase detail + progress bar */}
                                <div className="space-y-1 pt-1 border-t border-border/50">
                                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                                    <span className="truncate">{inst.tuneStatus.progress.message}</span>
                                    {inst.tuneStatus.progress.total > 0 && (
                                      <span className="flex-shrink-0 ml-2">{inst.tuneStatus.progress.current}/{inst.tuneStatus.progress.total}</span>
                                    )}
                                  </div>
                                  <div className="w-full bg-muted rounded-full h-1.5">
                                    <div
                                      className={`h-1.5 rounded-full transition-all duration-300 ${
                                        inst.tuneStatus.progress.phase === "sweeping" || inst.tuneStatus.progress.phase === "smc_precompute"
                                          ? "bg-primary"
                                          : "bg-primary/60 animate-pulse"
                                      }`}
                                      style={{
                                        width: inst.tuneStatus.progress.total > 0
                                          ? `${Math.max(5, (inst.tuneStatus.progress.current / inst.tuneStatus.progress.total) * 100)}%`
                                          : "100%",
                                      }}
                                    />
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Tune results */}
                            {inst.tuneStatus?.status === "done" && inst.tuneStatus.result?.best && (
                              <div className="space-y-2">
                                <div className="p-3 rounded-lg bg-muted/50 space-y-2 text-xs">
                                  <div className="flex items-center justify-between">
                                    <span className="font-semibold">Best Configuration</span>
                                    <span className="text-muted-foreground">{inst.tuneStatus.result.duration_seconds}s</span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-2">
                                    <div>
                                      <span className="text-muted-foreground">Win Rate</span>
                                      <p className="font-medium text-green-500">{inst.tuneStatus.result.best.win_rate.toFixed(1)}%</p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">Sharpe</span>
                                      <p className="font-medium">{inst.tuneStatus.result.best.sharpe.toFixed(2)}</p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">Profit Factor</span>
                                      <p className="font-medium">{inst.tuneStatus.result.best.profit_factor.toFixed(2)}</p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">Trades</span>
                                      <p className="font-medium">{inst.tuneStatus.result.best.total_trades}</p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">BUY WR</span>
                                      <p className="font-medium text-green-500">{inst.tuneStatus.result.best.buy_win_rate.toFixed(0)}%
                                        <span className="text-muted-foreground ml-1">({inst.tuneStatus.result.best.buy_trades})</span>
                                      </p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">SELL WR</span>
                                      <p className="font-medium text-red-500">{inst.tuneStatus.result.best.sell_win_rate.toFixed(0)}%
                                        <span className="text-muted-foreground ml-1">({inst.tuneStatus.result.best.sell_trades})</span>
                                      </p>
                                    </div>
                                  </div>
                                  <div className="text-muted-foreground">
                                    <span className="font-medium text-foreground">Params:</span>{" "}
                                    TF={inst.tuneStatus.result.best.timeframe},{" "}
                                    {Object.entries(inst.tuneStatus.result.best.params).map(([k, v]) => `${k}=${v}`).join(", ")}
                                  </div>

                                  {/* Config changes to apply */}
                                  {inst.tuneStatus.result.config_updates && Object.keys(inst.tuneStatus.result.config_updates).length > 0 && (
                                    <div className="mt-2 p-2 rounded bg-blue-500/10 border border-blue-500/20">
                                      <span className="font-medium text-blue-400">Suggested config changes:</span>
                                      <div className="mt-1 space-y-0.5">
                                        {Object.entries(inst.tuneStatus.result.config_updates).map(([key, value]) => (
                                          <div key={key} className="flex items-center justify-between">
                                            <span className="text-muted-foreground">{key}</span>
                                            <span>
                                              <span className="text-muted-foreground line-through mr-2">
                                                {inst.config[key as keyof QuantAutomationConfig] !== undefined
                                                  ? String(inst.config[key as keyof QuantAutomationConfig])
                                                  : "—"}
                                              </span>
                                              <span className="text-blue-400 font-medium">{String(value)}</span>
                                            </span>
                                          </div>
                                        ))}
                                      </div>
                                      {!inst.tuneStatus.applied && (
                                        <Button
                                          size="sm"
                                          className="mt-2 h-7 text-xs w-full gap-1"
                                          onClick={() => handleApplyTuneResult(name)}
                                          disabled={inst.actionLoading === "apply_tune"}
                                        >
                                          {inst.actionLoading === "apply_tune" ? (
                                            <><Loader2 className="h-3 w-3 animate-spin" /> Applying...</>
                                          ) : (
                                            <><Check className="h-3 w-3" /> Apply Best Config</>
                                          )}
                                        </Button>
                                      )}
                                      {inst.tuneStatus.applied && (
                                        <p className="mt-1 text-green-400 text-center">Applied</p>
                                      )}
                                    </div>
                                  )}
                                </div>

                                {/* Top 5 collapsible */}
                                {inst.tuneStatus.result.top_5 && inst.tuneStatus.result.top_5.length > 1 && (
                                  <div>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-6 text-xs w-full gap-1"
                                      onClick={() => updateInstance(name, { tuneExpanded: !inst.tuneExpanded })}
                                    >
                                      {inst.tuneExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                                      Top {inst.tuneStatus.result.top_5.length} configs ({inst.tuneStatus.result.all_count} tested)
                                    </Button>
                                    {inst.tuneExpanded && (
                                      <div className="space-y-1 mt-1">
                                        {inst.tuneStatus.result.top_5.slice(1).map((r, i) => (
                                          <div key={i} className="flex items-center justify-between text-xs p-1.5 rounded bg-muted/30">
                                            <span className="text-muted-foreground">
                                              #{i + 2} {r.timeframe} {Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(", ")}
                                            </span>
                                            <span>
                                              WR {r.win_rate.toFixed(0)}% | S {r.sharpe.toFixed(2)} | PF {r.profit_factor.toFixed(1)} | {r.total_trades}t
                                            </span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Error state */}
                            {inst.tuneStatus?.status === "error" && (
                              <div className="p-2 rounded bg-red-500/10 border border-red-500/20 text-xs text-red-400">
                                Tune failed: {inst.tuneStatus.error || inst.tuneStatus.result?.error || "Unknown error"}
                              </div>
                            )}

                            {/* Tune History */}
                            {inst.config.pipeline !== "multi_agent" && (
                              <div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-6 text-xs w-full gap-1 text-muted-foreground"
                                  onClick={() => {
                                    if (!inst.tuneHistory) loadTuneHistory(name)
                                    updateInstance(name, { tuneHistoryOpen: !inst.tuneHistoryOpen })
                                  }}
                                >
                                  <Clock className="h-3 w-3" />
                                  {inst.tuneHistoryOpen ? "Hide" : "Show"} Tune History
                                  {inst.tuneHistoryKey && (
                                    <span className="text-muted-foreground/60 text-[10px]">({inst.tuneHistoryKey})</span>
                                  )}
                                  {!inst.tuneHistoryKey && inst.tuneHistory && inst.tuneHistory.length > 0 && (
                                    <span className="text-muted-foreground/60">({inst.tuneHistory.length})</span>
                                  )}
                                </Button>
                                {inst.tuneHistoryOpen && inst.tuneHistory && (
                                  <div className="space-y-1.5 mt-1">
                                    {inst.tuneHistory.length === 0 && (
                                      <p className="text-xs text-muted-foreground text-center py-2">No tune history yet</p>
                                    )}
                                    {inst.tuneHistory.slice().reverse().map((rec, idx) => {
                                      const realIdx = inst.tuneHistory!.length - 1 - idx
                                      return (
                                        <div key={idx} className="p-2 rounded bg-muted/30 text-xs space-y-1">
                                          <div className="flex items-center justify-between">
                                            <span className="text-muted-foreground">
                                              {new Date(rec.timestamp).toLocaleDateString()} {new Date(rec.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </span>
                                            <div className="flex items-center gap-1">
                                              {rec.applied && !rec.reverted && (
                                                <Badge variant="outline" className="text-[10px] h-4 px-1">Applied</Badge>
                                              )}
                                              {rec.reverted && (
                                                <Badge variant="secondary" className="text-[10px] h-4 px-1">Reverted</Badge>
                                              )}
                                              {rec.applied && rec.config_before_apply && !rec.reverted && (
                                                <Button
                                                  variant="ghost"
                                                  size="sm"
                                                  className="h-5 px-1 text-[10px] text-yellow-500 hover:text-yellow-400"
                                                  onClick={() => handleRevertTune(name, realIdx)}
                                                  disabled={inst.actionLoading === "revert_tune"}
                                                >
                                                  <RotateCcw className="h-2.5 w-2.5 mr-0.5" />
                                                  Revert
                                                </Button>
                                              )}
                                            </div>
                                          </div>
                                          {rec.best && (
                                            <div className="flex items-center gap-3 text-muted-foreground">
                                              <span>WR <span className="text-foreground">{rec.best.win_rate.toFixed(0)}%</span></span>
                                              <span>S <span className="text-foreground">{rec.best.sharpe.toFixed(2)}</span></span>
                                              <span>PF <span className="text-foreground">{rec.best.profit_factor.toFixed(1)}</span></span>
                                              <span>{rec.best.total_trades}t</span>
                                              <span className="ml-auto">{rec.duration_seconds}s</span>
                                            </div>
                                          )}
                                          {rec.config_updates && Object.keys(rec.config_updates).length > 0 && (
                                            <div className="text-muted-foreground">
                                              {Object.entries(rec.config_updates).map(([k, v]) => `${k}=${v}`).join(", ")}
                                            </div>
                                          )}
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                        {/* Recent Results */}
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-sm font-semibold flex items-center gap-2">
                              <Target className="h-4 w-4" />
                              Recent Results
                            </h3>
                            <div className="flex items-center gap-1">
                              {["actionable", "all", "BUY", "SELL", "HOLD"].map((filter) => (
                                <Button
                                  key={filter}
                                  variant={(resultSignalFilters[name] || "actionable") === filter ? "default" : "ghost"}
                                  size="sm"
                                  className={`h-6 px-2 text-xs ${
                                    (resultSignalFilters[name] || "actionable") === filter
                                      ? ""
                                      : filter === "BUY"
                                      ? "text-green-500 hover:text-green-400"
                                      : filter === "SELL"
                                      ? "text-red-500 hover:text-red-400"
                                      : filter === "HOLD"
                                      ? "text-gray-500 hover:text-gray-400"
                                      : ""
                                  }`}
                                  onClick={() => setResultSignalFilters(prev => ({ ...prev, [name]: filter }))}
                                >
                                  {filter === "actionable" ? "Actionable" : filter === "all" ? "All" : filter}
                                </Button>
                              ))}
                            </div>
                          </div>
                          {inst.history?.analysis_results && inst.history.analysis_results.length > 0 ? (
                            <ScrollArea className="h-[350px]">
                              <div className="space-y-2">
                                {inst.history.analysis_results.slice().reverse()
                                  .filter((result) => {
                                    const f = resultSignalFilters[name] || "actionable"
                                    return f === "all" ? true : f === "actionable" ? result.signal !== "HOLD" : result.signal === f
                                  })
                                  .map((result, idx) => (
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
                                        {result.executed && result.decision_id ? (
                                          <Badge variant="success" className="text-xs">Executed</Badge>
                                        ) : result.executed && !result.decision_id ? (
                                          <Badge variant="outline" className="text-xs border-yellow-500 text-yellow-500">Pending</Badge>
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
              <Select value={newInstancePipeline} onValueChange={(v) => {
                setNewInstancePipeline(v)
                const def = pipelineDefaults[v]
                if (def) setNewInstanceTimeframe(def.timeframe)
              }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* --- SMC-based (institutional levels) --- */}
                  <SelectItem value="rule_based">SMC Rule-Based (no LLM, instant)</SelectItem>
                  <SelectItem value="smc_quant_basic">SMC Quant (LLM)</SelectItem>
                  <SelectItem value="smc_quant">SMC Quant Deep (LLM, extended)</SelectItem>
                  <SelectItem value="smc_mtf">SMC Multi-Timeframe (no LLM)</SelectItem>
                  {/* --- Breakout strategies --- */}
                  <SelectItem value="breakout_quant">BB Squeeze Breakout (LLM)</SelectItem>
                  <SelectItem value="donchian_breakout">Donchian Breakout (no LLM, instant)</SelectItem>
                  {/* --- Mean reversion / ranging --- */}
                  <SelectItem value="range_quant">Range Mean Reversion (LLM)</SelectItem>
                  <SelectItem value="volume_profile">Volume Profile (LLM)</SelectItem>
                  {/* --- ML-driven --- */}
                  <SelectItem value="xgboost">ML Auto-Select (no LLM, instant)</SelectItem>
                  <SelectItem value="xgboost_ensemble">ML Ensemble Vote (no LLM, instant)</SelectItem>
                  {/* --- Metals-specific --- */}
                  <SelectItem value="gold_trend_pullback">Gold Trend-Pullback D1 (no LLM, best WR)</SelectItem>
                  <SelectItem value="gold_silver_pullback">Gold/Silver Pullback D1 (no LLM)</SelectItem>
                  <SelectItem value="gold_silver_pullback_mtf">Gold/Silver MTF D1+H4 (no LLM)</SelectItem>
                  {/* --- Multi-agent / auto --- */}
                  <SelectItem value="multi_agent">Multi-Agent AI (LLM, high cost)</SelectItem>
                  <SelectItem value="scanner_auto">Scanner Auto (scans all pairs)</SelectItem>
                </SelectContent>
              </Select>
              {pipelineDescriptions[newInstancePipeline] && (
                <div className="rounded-md border border-border bg-muted/50 p-3 space-y-1.5">
                  <p className="text-sm text-foreground">{pipelineDescriptions[newInstancePipeline].summary}</p>
                  <p className="text-xs text-muted-foreground">{pipelineDescriptions[newInstancePipeline].details}</p>
                  <div className="flex gap-4 pt-1">
                    <div className="text-xs">
                      <span className="text-muted-foreground">Best timeframes: </span>
                      <span className="text-foreground font-medium">{pipelineDescriptions[newInstancePipeline].recommendedTimeframes}</span>
                    </div>
                    <div className="text-xs">
                      <span className="text-muted-foreground">Interval: </span>
                      <span className="text-foreground font-medium">{pipelineDescriptions[newInstancePipeline].recommendedInterval}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="space-y-2">
              <Label>Timeframe</Label>
              <Select value={newInstanceTimeframe} onValueChange={setNewInstanceTimeframe}>
                <SelectTrigger>
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
                    {detailResult.executed && detailResult.decision_id ? (
                      <Badge variant="success" className="text-xs">Executed ({detailResult.decision_id})</Badge>
                    ) : detailResult.executed && !detailResult.decision_id ? (
                      <Badge variant="outline" className="text-xs border-yellow-500 text-yellow-500">Pending (unfilled)</Badge>
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
