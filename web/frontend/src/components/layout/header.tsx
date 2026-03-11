"use client"

import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getStatus } from "@/lib/api"
import { RefreshCw, Wifi, WifiOff, Brain, AlertTriangle } from "lucide-react"
import { formatCurrency } from "@/lib/utils"
import { HelpTooltip } from "@/components/ui/help-tooltip"

export function Header() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const fetchStatus = async () => {
    setLoading(true)
    const { data } = await getStatus()
    if (data) setStatus(data)
    setLoading(false)
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card px-6">
      <div className="flex items-center gap-4">
        <Badge variant={status?.mt5?.connected ? "success" : "destructive"} className="gap-1">
          {status?.mt5?.connected ? (
            <>
              <Wifi className="h-3 w-3" /> MT5 Connected
            </>
          ) : (
            <>
              <WifiOff className="h-3 w-3" /> MT5 Disconnected
            </>
          )}
        </Badge>
        {status?.mt5?.connected && (
          <>
            <Badge variant={status?.mt5?.auto_trading ? "success" : "warning"}>
              {status?.mt5?.auto_trading ? "AutoTrading ON" : "AutoTrading OFF"}
            </Badge>
            <div className="flex items-center gap-1">
              <Badge
                variant={
                  status?.automation?.running
                    ? "success"
                    : status?.automation?.enabled && !status?.automation?.running
                    ? "destructive"
                    : "secondary"
                }
                className="gap-1"
              >
                {status?.automation?.enabled && !status?.automation?.running && (
                  <AlertTriangle className="h-3 w-3" />
                )}
                {status?.automation?.running
                  ? "Automation Running"
                  : status?.automation?.enabled
                  ? "Automation Crashed"
                  : "Automation Stopped"}
              </Badge>
              {status?.automation?.enabled && !status?.automation?.running && (
                <HelpTooltip
                  content="Automation was enabled but the process stopped unexpectedly. It will auto-restart when the backend restarts, or you can manually restart it from the Automation page."
                  iconClassName="h-3 w-3"
                />
              )}
            </div>
          </>
        )}
        <div className="flex items-center gap-1">
          <Badge
            variant={
              status?.daily_cycle?.running
                ? "success"
                : status?.daily_cycle?.enabled && !status?.daily_cycle?.running
                ? "destructive"
                : "secondary"
            }
            className="gap-1"
          >
            <Brain className="h-3 w-3" />
            {status?.daily_cycle?.running
              ? "Learning Active"
              : status?.daily_cycle?.enabled
              ? "Learning Crashed"
              : "Learning Stopped"}
          </Badge>
          <HelpTooltip
            content={
              status?.daily_cycle?.enabled && !status?.daily_cycle?.running
                ? "Learning cycle was enabled but the process stopped unexpectedly. It will auto-restart when the backend restarts, or you can manually restart it from the Automation page."
                : "Daily Cycle tracks predictions and evaluates them after 24 hours. Correct predictions strengthen memories, incorrect ones get analyzed for improvement. Start it from the Automation page."
            }
            iconClassName="h-3 w-3"
          />
        </div>
      </div>

      <div className="flex items-center gap-6">
        {status?.mt5?.connected && (
          <div className="flex items-center gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Balance:</span>{" "}
              <span className="font-medium">{formatCurrency(status?.mt5?.balance || 0)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Equity:</span>{" "}
              <span className="font-medium">{formatCurrency(status?.mt5?.equity || 0)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">P/L:</span>{" "}
              <span
                className={`font-medium ${
                  (status?.mt5?.profit || 0) >= 0 ? "text-green-500" : "text-red-500"
                }`}
              >
                {formatCurrency(status?.mt5?.profit || 0)}
              </span>
            </div>
          </div>
        )}
        <Button variant="ghost" size="icon" onClick={fetchStatus} disabled={loading}>
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </div>
    </header>
  )
}
