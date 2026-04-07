"use client"

import { useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  getTradeManagerStatus,
  getTradeManagerConfig,
  updateTradeManagerConfig,
  startTradeManager,
  stopTradeManager,
  getTradeManagerActions,
  getTradeManagerPolicies,
  getTradeManagerAlerts,
  acknowledgeTradeManagerAlert,
  deleteTradeManagerPolicy,
  getPositionsAtr,
} from "@/lib/api"
import { formatDate } from "@/lib/utils"
import { HelpTooltip, LabelWithHelp } from "@/components/ui/help-tooltip"
import {
  Play,
  Square,
  Loader2,
  Settings2,
  Shield,
  Activity,
  AlertTriangle,
  Target,
  Clock,
  TrendingUp,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Trash2,
  Bell,
  Eye,
} from "lucide-react"

export function TradeManagementSection() {
  const [status, setStatus] = useState<any>(null)
  const [config, setConfig] = useState<any>(null)
  const [actions, setActions] = useState<any[]>([])
  const [policies, setPolicies] = useState<any[]>([])
  const [alerts, setAlerts] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)
  const [actionsExpanded, setActionsExpanded] = useState(false)
  const [alertsExpanded, setAlertsExpanded] = useState(false)
  const [atrBySymbol, setAtrBySymbol] = useState<Record<string, number | null>>({})

  // Local config edits
  const [localConfig, setLocalConfig] = useState<Record<string, any>>({})
  const [configDirty, setConfigDirty] = useState(false)

  const fetchData = useCallback(async () => {
    const [statusRes, configRes, actionsRes, policiesRes, alertsRes, atrRes] = await Promise.all([
      getTradeManagerStatus(),
      getTradeManagerConfig(),
      getTradeManagerActions(undefined, 20),
      getTradeManagerPolicies(),
      getTradeManagerAlerts(10, false),
      getPositionsAtr(),
    ])
    if (atrRes.data?.atr) setAtrBySymbol(atrRes.data.atr)
    if (statusRes.data) setStatus(statusRes.data)
    if (configRes.data?.config) {
      setConfig(configRes.data.config)
      if (!configDirty) setLocalConfig(configRes.data.config)
    }
    if (actionsRes.data?.actions) setActions(actionsRes.data.actions)
    if (policiesRes.data?.policies) setPolicies(policiesRes.data.policies)
    if (alertsRes.data?.alerts) setAlerts(alertsRes.data.alerts)
    setLoading(false)
  }, [configDirty])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [fetchData])

  const isRunning = status?.status === "running"
  const isStopped = !isRunning

  const handleStart = async () => {
    setActionLoading("start")
    const { error } = await startTradeManager(localConfig)
    if (error) alert(`Error: ${error}`)
    setTimeout(fetchData, 2000)
    setActionLoading(null)
  }

  const handleStop = async () => {
    setActionLoading("stop")
    const { error } = await stopTradeManager()
    if (error) alert(`Error: ${error}`)
    setTimeout(fetchData, 2000)
    setActionLoading(null)
  }

  const handleSaveConfig = async () => {
    const { error } = await updateTradeManagerConfig(localConfig)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      setConfigDirty(false)
      fetchData()
    }
  }

  const updateLocal = (key: string, value: any) => {
    setLocalConfig((prev) => ({ ...prev, [key]: value }))
    setConfigDirty(true)
  }

  const handleAcknowledgeAlert = async (alertId: number) => {
    await acknowledgeTradeManagerAlert(alertId)
    fetchData()
  }

  const handleDeletePolicy = async (ticket: number) => {
    await deleteTradeManagerPolicy(ticket)
    fetchData()
  }

  const unacknowledgedAlerts = alerts.filter((a: any) => !a.acknowledged)

  const actionTypeIcon = (type: string) => {
    switch (type) {
      case "trailing_stop": return <Target className="h-3.5 w-3.5 text-yellow-500" />
      case "breakeven": return <Shield className="h-3.5 w-3.5 text-green-500" />
      case "scalp_breakeven": return <Shield className="h-3.5 w-3.5 text-cyan-500" />
      case "partial_close": return <TrendingUp className="h-3.5 w-3.5 text-blue-500" />
      case "partial_tp": return <TrendingUp className="h-3.5 w-3.5 text-blue-500" />
      case "time_flag": return <Clock className="h-3.5 w-3.5 text-orange-500" />
      case "scalp_tp": return <CheckCircle className="h-3.5 w-3.5 text-cyan-500" />
      case "scalp_time_close": return <Clock className="h-3.5 w-3.5 text-cyan-500" />
      case "opposing_analysis": return <AlertTriangle className="h-3.5 w-3.5 text-purple-500" />
      case "close_hedge": return <XCircle className="h-3.5 w-3.5 text-purple-500" />
      case "assumption_review": return <Eye className="h-3.5 w-3.5 text-indigo-500" />
      default: return <Activity className="h-3.5 w-3.5 text-muted-foreground" />
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Settings2 className="h-6 w-6 text-blue-500" />
            Trade Management
          </h2>
          <p className="text-muted-foreground">
            Centralized position management — trailing stops, breakeven, partial TP, risk monitoring
          </p>
        </div>
        <div className="flex items-center gap-2">
          {unacknowledgedAlerts.length > 0 && (
            <Badge variant="destructive" className="gap-1">
              <Bell className="h-3 w-3" />
              {unacknowledgedAlerts.length} alert{unacknowledgedAlerts.length > 1 ? "s" : ""}
            </Badge>
          )}
          <Badge variant={isRunning ? "success" : "secondary"} className="gap-1">
            <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-green-400 animate-pulse" : "bg-gray-400"}`} />
            {isRunning ? "Running" : "Stopped"}
          </Badge>
        </div>
      </div>

      {/* Main Card */}
      <Card>
        <CardHeader className="cursor-pointer" onClick={() => setExpanded(!expanded)}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CardTitle className="text-lg">Trade Manager</CardTitle>
              {status?.managed_positions !== undefined && (
                <Badge variant="outline" className="text-xs">
                  {status.managed_positions} position{status.managed_positions !== 1 ? "s" : ""} managed
                </Badge>
              )}
              {status?.actions_today !== undefined && status.actions_today > 0 && (
                <Badge variant="outline" className="text-xs">
                  {status.actions_today} action{status.actions_today !== 1 ? "s" : ""} today
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              {isStopped ? (
                <Button size="sm" onClick={(e) => { e.stopPropagation(); handleStart() }} disabled={actionLoading !== null}>
                  {actionLoading === "start" ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Play className="mr-1 h-3 w-3" />}
                  Start
                </Button>
              ) : (
                <Button variant="destructive" size="sm" onClick={(e) => { e.stopPropagation(); handleStop() }} disabled={actionLoading !== null}>
                  {actionLoading === "stop" ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Square className="mr-1 h-3 w-3" />}
                  Stop
                </Button>
              )}
              {expanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
            </div>
          </div>
          {status?.last_action && (
            <CardDescription>Last action: {status.last_action}</CardDescription>
          )}
        </CardHeader>

        {expanded && (
          <CardContent className="space-y-6">
            {/* Configuration */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Settings2 className="h-4 w-4" />
                Configuration
              </h3>

              {/* Per-Symbol Settings */}
              {Object.keys(atrBySymbol).length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <Label className="text-xs font-medium">Per-Symbol Settings</Label>
                    <HelpTooltip content="Override trailing stop and breakeven multipliers per symbol. Leave blank to use the global defaults above. ATR(14) shown for reference so you can see what each multiplier means in price distance." />
                  </div>
                  <div className="rounded-lg border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-muted/30">
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">Symbol</th>
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">ATR(14)</th>
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">
                            <div className="flex items-center gap-1">
                              Trail Mult
                              <HelpTooltip content="Trailing stop distance = this value x ATR. The actual price distance is shown in the ATR column." iconClassName="h-3 w-3" />
                            </div>
                          </th>
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">Trail Distance</th>
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">
                            <div className="flex items-center gap-1">
                              BE Mult
                              <HelpTooltip content="Move SL to breakeven when profit reaches this many ATR." iconClassName="h-3 w-3" />
                            </div>
                          </th>
                          <th className="p-2 text-left text-xs font-medium text-muted-foreground">BE Threshold</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(atrBySymbol).map(([sym, atr]) => {
                          const symSettings = (localConfig.symbol_settings || {})[sym] || {}
                          const effectiveTrail = symSettings.trailing_stop_atr_multiplier ?? localConfig.trailing_stop_atr_multiplier ?? 1.5
                          const effectiveBe = symSettings.breakeven_atr_multiplier ?? localConfig.breakeven_atr_multiplier ?? 1.5

                          const updateSymSetting = (key: string, value: string) => {
                            const numVal = parseFloat(value)
                            const current = { ...(localConfig.symbol_settings || {}) }
                            if (!value || isNaN(numVal)) {
                              // Clear override
                              if (current[sym]) {
                                delete current[sym][key]
                                if (Object.keys(current[sym]).length === 0) delete current[sym]
                              }
                            } else {
                              current[sym] = { ...(current[sym] || {}), [key]: numVal }
                            }
                            updateLocal("symbol_settings", current)
                          }

                          return (
                            <tr key={sym} className="border-b last:border-0">
                              <td className="p-2 font-medium">{sym}</td>
                              <td className="p-2 text-muted-foreground">{atr ? atr.toFixed(2) : "N/A"}</td>
                              <td className="p-2">
                                <Input
                                  type="number"
                                  step="0.1"
                                  min="0.5"
                                  max="10"
                                  placeholder={String(localConfig.trailing_stop_atr_multiplier ?? 1.5)}
                                  value={symSettings.trailing_stop_atr_multiplier ?? ""}
                                  onChange={(e) => updateSymSetting("trailing_stop_atr_multiplier", e.target.value)}
                                  className="h-7 w-20 text-xs"
                                />
                              </td>
                              <td className="p-2 text-muted-foreground text-xs">
                                {atr ? (effectiveTrail * atr).toFixed(2) : "—"}
                              </td>
                              <td className="p-2">
                                <Input
                                  type="number"
                                  step="0.1"
                                  min="0.5"
                                  max="10"
                                  placeholder={String(localConfig.breakeven_atr_multiplier ?? 1.5)}
                                  value={symSettings.breakeven_atr_multiplier ?? ""}
                                  onChange={(e) => updateSymSetting("breakeven_atr_multiplier", e.target.value)}
                                  className="h-7 w-20 text-xs"
                                />
                              </td>
                              <td className="p-2 text-muted-foreground text-xs">
                                {atr ? (effectiveBe * atr).toFixed(2) : "—"}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {/* Management interval */}
                <div className="space-y-1">
                  <LabelWithHelp help="How often to check positions in seconds. 900 (15 min) is recommended for normal use. Lower values like 60 or 30 for tighter management." htmlFor="tma-interval">
                    Check Interval (s)
                  </LabelWithHelp>
                  <Input
                    id="tma-interval"
                    type="number"
                    step="1"
                    min="5"
                    max="3600"
                    value={localConfig.management_interval_seconds ?? 900}
                    onChange={(e) => updateLocal("management_interval_seconds", parseFloat(e.target.value))}
                  />
                </div>

                {/* Trailing ATR multiplier */}
                <div className="space-y-1">
                  <LabelWithHelp help="Default trailing stop distance as ATR multiplier. Lower = tighter (locks profit faster, risk of noise). Higher = wider (more room). XAUUSD: 2.0-4.0x recommended." htmlFor="tma-trail-mult">
                    Trail ATR Multiplier
                  </LabelWithHelp>
                  <Input
                    id="tma-trail-mult"
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="10"
                    value={localConfig.trailing_stop_atr_multiplier ?? 1.5}
                    onChange={(e) => updateLocal("trailing_stop_atr_multiplier", parseFloat(e.target.value))}
                  />
                </div>

                {/* Breakeven ATR multiplier */}
                <div className="space-y-1">
                  <LabelWithHelp help="Move SL to breakeven when profit reaches this many ATR. E.g., 1.5 means: when position is 1.5x ATR in profit, move SL to entry price." htmlFor="tma-be-mult">
                    Breakeven ATR Mult
                  </LabelWithHelp>
                  <Input
                    id="tma-be-mult"
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="10"
                    value={localConfig.breakeven_atr_multiplier ?? 1.5}
                    onChange={(e) => updateLocal("breakeven_atr_multiplier", parseFloat(e.target.value))}
                  />
                </div>

                {/* Max exposure */}
                <div className="space-y-1">
                  <LabelWithHelp help="Maximum total account exposure as % of balance. Alert fires when exceeded. E.g., 10% means all open position values combined can't exceed 10% of balance." htmlFor="tma-exposure">
                    Max Exposure %
                  </LabelWithHelp>
                  <Input
                    id="tma-exposure"
                    type="number"
                    step="1"
                    min="1"
                    max="100"
                    value={localConfig.max_account_exposure_pct ?? 10}
                    onChange={(e) => updateLocal("max_account_exposure_pct", parseFloat(e.target.value))}
                  />
                </div>

                {/* Max position hours */}
                <div className="space-y-1">
                  <LabelWithHelp help="Flag positions open longer than this many hours. Only applies if time limits are enabled." htmlFor="tma-max-hours">
                    Max Position Hours
                  </LabelWithHelp>
                  <Input
                    id="tma-max-hours"
                    type="number"
                    step="1"
                    min="1"
                    max="720"
                    value={localConfig.max_position_hours ?? 48}
                    onChange={(e) => updateLocal("max_position_hours", parseFloat(e.target.value))}
                  />
                </div>

                {/* Partial TP % */}
                <div className="space-y-1">
                  <LabelWithHelp help="When partial TP fires, close this % of the position. E.g., 50 means close half at the intermediate level." htmlFor="tma-partial-pct">
                    Partial Close %
                  </LabelWithHelp>
                  <Input
                    id="tma-partial-pct"
                    type="number"
                    step="5"
                    min="10"
                    max="90"
                    value={localConfig.partial_tp_percent ?? 50}
                    onChange={(e) => updateLocal("partial_tp_percent", parseFloat(e.target.value))}
                  />
                </div>
              </div>

              {/* Switches */}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-yellow-500" />
                    <Label className="text-sm">Trailing Stop</Label>
                    <HelpTooltip content="Automatically trail SL behind price using ATR distance. Locks in profit as price moves in your favor." />
                  </div>
                  <Switch
                    checked={localConfig.enable_trailing_stop ?? true}
                    onCheckedChange={(v) => updateLocal("enable_trailing_stop", v)}
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <Shield className="h-4 w-4 text-green-500" />
                    <Label className="text-sm">Breakeven</Label>
                    <HelpTooltip content="Move SL to entry price once position reaches a profit threshold. Eliminates risk on the trade." />
                  </div>
                  <Switch
                    checked={localConfig.enable_breakeven_stop ?? true}
                    onCheckedChange={(v) => updateLocal("enable_breakeven_stop", v)}
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                    <Label className="text-sm">Partial TP</Label>
                    <HelpTooltip content="Close a portion of the position at an intermediate profit level. Locks partial profit while letting the rest run to full TP." />
                  </div>
                  <Switch
                    checked={localConfig.enable_partial_tp ?? false}
                    onCheckedChange={(v) => updateLocal("enable_partial_tp", v)}
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-purple-500" />
                    <Label className="text-sm">Assumption Review</Label>
                    <HelpTooltip content="Each cycle, reviews open positions against current SMC structure. Checks if bias has shifted, if SL/TP zones are still valid, if CHOCH occurred against your position. Can optionally auto-adjust SL/TP." />
                  </div>
                  <Switch
                    checked={localConfig.enable_assumption_review ?? true}
                    onCheckedChange={(v) => updateLocal("enable_assumption_review", v)}
                  />
                </div>

                {(localConfig.enable_assumption_review ?? true) && (
                  <>
                    <div className="flex items-center justify-between rounded-lg border p-3 ml-4">
                      <div className="flex items-center gap-2">
                        <Label className="text-sm">Auto-Apply SL/TP</Label>
                        <HelpTooltip content="When assumption review recommends adjusting SL or TP, apply it automatically. Close recommendations still require manual confirmation. Turn this off to just log recommendations without acting." />
                      </div>
                      <Switch
                        checked={localConfig.assumption_review_auto_apply ?? false}
                        onCheckedChange={(v) => updateLocal("assumption_review_auto_apply", v)}
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border p-3 ml-4">
                      <div className="flex items-center gap-2">
                        <Label className="text-sm">Use LLM Assessment</Label>
                        <HelpTooltip content="Add an LLM-powered nuanced assessment on top of the rule-based checks. More insightful but uses API tokens. Disable to use rule-based checks only (free, faster)." />
                      </div>
                      <Switch
                        checked={localConfig.assumption_review_use_llm ?? false}
                        onCheckedChange={(v) => updateLocal("assumption_review_use_llm", v)}
                      />
                    </div>
                  </>
                )}

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-orange-500" />
                    <Label className="text-sm">Time Limits</Label>
                    <HelpTooltip content="Flag positions that have been open longer than the configured max hours. Useful for identifying stale trades." />
                  </div>
                  <Switch
                    checked={localConfig.enable_time_limit ?? false}
                    onCheckedChange={(v) => updateLocal("enable_time_limit", v)}
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-cyan-500" />
                    <Label className="text-sm">Scalp Mode</Label>
                    <HelpTooltip content="Volume Profile trades get scalp management: quick TP at 1.5x ATR, breakeven at 0.5x ATR, 6h max hold time. VP bounces are short-lived — this captures the initial move and exits before reversal." />
                  </div>
                  <Switch
                    checked={localConfig.enable_scalp_mode ?? true}
                    onCheckedChange={(v) => updateLocal("enable_scalp_mode", v)}
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-purple-500" />
                    <Label className="text-sm">Opposing Position Check</Label>
                    <HelpTooltip content="Detects opposing positions on the same symbol (BUY + SELL). Scores each on momentum, age, profit, and regime alignment. Recommends closing the weaker one to eliminate hedge drag." />
                  </div>
                  <Switch
                    checked={localConfig.enable_opposing_check ?? true}
                    onCheckedChange={(v) => updateLocal("enable_opposing_check", v)}
                  />
                </div>

                {(localConfig.enable_opposing_check ?? true) && (
                  <div className="flex items-center justify-between rounded-lg border p-3 ml-4">
                    <div className="flex items-center gap-2">
                      <Label className="text-sm">Auto-Close Weaker</Label>
                      <HelpTooltip content="When enabled, automatically closes the weaker opposing position instead of just logging a recommendation. The close rationale is captured in the decision record for learning." />
                    </div>
                    <Switch
                      checked={localConfig.auto_resolve_opposing ?? true}
                      onCheckedChange={(v) => updateLocal("auto_resolve_opposing", v)}
                    />
                  </div>
                )}

                <div className="flex items-center justify-between rounded-lg border p-3">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                    <Label className="text-sm">Exposure Monitor</Label>
                    <HelpTooltip content="Monitor total account exposure and alert when it exceeds the configured maximum. Also checks for correlated positions (e.g., multiple gold trades)." />
                  </div>
                  <Switch
                    checked={localConfig.max_account_exposure_pct !== undefined}
                    onCheckedChange={(v) => updateLocal("max_account_exposure_pct", v ? 10 : 0)}
                  />
                </div>
              </div>

              {configDirty && (
                <div className="flex justify-end">
                  <Button size="sm" onClick={handleSaveConfig}>
                    Save Configuration
                  </Button>
                </div>
              )}
            </div>

            <Separator />

            {/* Per-Position Policies */}
            {policies.length > 0 && (
              <>
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    <Shield className="h-4 w-4" />
                    Position Policies
                    <Badge variant="outline" className="text-xs">{policies.length}</Badge>
                  </h3>
                  <div className="space-y-2">
                    {policies.map((p: any) => (
                      <div key={p.ticket} className="flex items-center justify-between rounded-lg border p-3 text-sm">
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-muted-foreground">#{p.ticket}</span>
                          <span className="font-medium">{p.symbol}</span>
                          {p.policy?.frozen && <Badge variant="secondary" className="text-xs">Frozen</Badge>}
                          {p.policy?.trailing_atr_mult && (
                            <Badge variant="outline" className="text-xs">Trail {p.policy.trailing_atr_mult}x</Badge>
                          )}
                          {p.policy?.breakeven_enabled === false && (
                            <Badge variant="outline" className="text-xs">No BE</Badge>
                          )}
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeletePolicy(p.ticket)}
                          title="Remove policy (use defaults)"
                        >
                          <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
                <Separator />
              </>
            )}

            {/* Recent Actions */}
            <div className="space-y-3">
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setActionsExpanded(!actionsExpanded)}
              >
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Recent Actions
                  {actions.length > 0 && <Badge variant="outline" className="text-xs">{actions.length}</Badge>}
                </h3>
                {actionsExpanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
              </div>

              {actionsExpanded && (
                <ScrollArea className="h-[250px]">
                  {actions.length > 0 ? (
                    <div className="space-y-1.5">
                      {actions.map((a: any, i: number) => (
                        <div key={i} className="flex items-start gap-2 rounded-lg border p-2.5 text-xs">
                          {actionTypeIcon(a.action_type)}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-muted-foreground">#{a.ticket}</span>
                              <span className="font-medium">{a.symbol}</span>
                              <Badge variant={a.success ? "outline" : "destructive"} className="text-[10px] px-1 py-0">
                                {a.action_type.replace(/_/g, " ")}
                              </Badge>
                            </div>
                            <p className="text-muted-foreground mt-0.5 truncate">{a.reason}</p>
                            {a.old_value && a.new_value && (
                              <p className="text-muted-foreground">
                                {a.old_value.toFixed(2)} → {a.new_value.toFixed(2)}
                              </p>
                            )}
                          </div>
                          <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                            {formatDate(a.created_at)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground text-center py-4">No management actions yet</p>
                  )}
                </ScrollArea>
              )}
            </div>

            {/* Risk Alerts */}
            {alerts.length > 0 && (
              <>
                <Separator />
                <div className="space-y-3">
                  <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => setAlertsExpanded(!alertsExpanded)}
                  >
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <Bell className="h-4 w-4" />
                      Risk Alerts
                      {unacknowledgedAlerts.length > 0 && (
                        <Badge variant="destructive" className="text-xs">{unacknowledgedAlerts.length} new</Badge>
                      )}
                    </h3>
                    {alertsExpanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
                  </div>

                  {alertsExpanded && (
                    <div className="space-y-2">
                      {alerts.map((a: any) => (
                        <div
                          key={a.id}
                          className={`flex items-start gap-2 rounded-lg border p-3 text-sm ${
                            a.acknowledged ? "opacity-50" : ""
                          } ${
                            a.severity === "critical"
                              ? "border-red-500/30 bg-red-500/5"
                              : a.severity === "warning"
                              ? "border-yellow-500/30 bg-yellow-500/5"
                              : ""
                          }`}
                        >
                          <AlertTriangle className={`h-4 w-4 mt-0.5 shrink-0 ${
                            a.severity === "critical" ? "text-red-500" : a.severity === "warning" ? "text-yellow-500" : "text-blue-500"
                          }`} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs capitalize">{a.alert_type}</Badge>
                              <Badge variant={a.severity === "critical" ? "destructive" : "secondary"} className="text-xs">
                                {a.severity}
                              </Badge>
                            </div>
                            <p className="mt-1">{a.message}</p>
                            <p className="text-xs text-muted-foreground mt-1">{formatDate(a.created_at)}</p>
                          </div>
                          {!a.acknowledged && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleAcknowledgeAlert(a.id)}
                              title="Acknowledge"
                            >
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            </Button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  )
}
