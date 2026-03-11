"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  getPositions,
  getOrders,
  modifyPosition,
  closePosition,
  batchClosePositions,
  batchReviewPositions,
  getBatchReviewStatus,
  cancelOrder,
  batchCancelOrders,
  setPositionBreakeven,
  setPositionTrailing,
  disablePositionTrailing,
} from "@/lib/api"
import { formatCurrency, getProfitColor, formatDate } from "@/lib/utils"
import {
  RefreshCw,
  Edit,
  X,
  Loader2,
  Search,
  XCircle,
  CheckSquare,
  Square,
  Trash2,
  Eye,
  TrendingUp,
  TrendingDown,
  Shield,
  Target,
} from "lucide-react"
import { HelpTooltip } from "@/components/ui/help-tooltip"

export default function PositionsPage() {
  const router = useRouter()
  const [positions, setPositions] = useState<any[]>([])
  const [orders, setOrders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [modifyingTicket, setModifyingTicket] = useState<number | null>(null)
  const [closingTicket, setClosingTicket] = useState<number | null>(null)
  const [newSl, setNewSl] = useState("")
  const [newTp, setNewTp] = useState("")

  // Selection state
  const [selectedPositions, setSelectedPositions] = useState<Set<number>>(new Set())
  const [selectedOrders, setSelectedOrders] = useState<Set<number>>(new Set())

  // Batch review state
  const [batchReviewData, setBatchReviewData] = useState<any[]>([])
  const [showBatchReview, setShowBatchReview] = useState(false)
  const [batchReviewLoading, setBatchReviewLoading] = useState(false)

  // Batch action state
  const [batchClosing, setBatchClosing] = useState(false)
  const [batchCancelling, setBatchCancelling] = useState(false)

  // Quick action state
  const [quickActionTicket, setQuickActionTicket] = useState<number | null>(null)
  const [quickActionType, setQuickActionType] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    const [posRes, ordRes] = await Promise.all([getPositions(), getOrders()])
    if (posRes.data) setPositions(posRes.data.positions || [])
    if (ordRes.data) setOrders(ordRes.data.orders || [])
    setLoading(false)
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleModify = async (ticket: number) => {
    const sl = newSl ? parseFloat(newSl) : undefined
    const tp = newTp ? parseFloat(newTp) : undefined

    const { data, error } = await modifyPosition(ticket, sl, tp)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      setModifyingTicket(null)
      setNewSl("")
      setNewTp("")
      fetchData()
    }
  }

  const handleClose = async (ticket: number) => {
    setClosingTicket(ticket)
    const { data, error } = await closePosition(ticket)
    if (error) {
      alert(`Error: ${error}`)
    }
    setClosingTicket(null)
    fetchData()
  }

  const handleBreakeven = async (ticket: number) => {
    setQuickActionTicket(ticket)
    setQuickActionType("breakeven")
    const { data, error } = await setPositionBreakeven(ticket)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert(data.message)
    }
    setQuickActionTicket(null)
    setQuickActionType(null)
    fetchData()
  }

  const handleTrailingStop = async (ticket: number, isCurrentlyActive: boolean) => {
    setQuickActionTicket(ticket)
    setQuickActionType("trailing")

    if (isCurrentlyActive) {
      // Disable trailing
      const { data, error } = await disablePositionTrailing(ticket)
      if (error) {
        alert(`Error: ${error}`)
      } else {
        alert(data.message)
      }
    } else {
      // Enable trailing
      const { data, error } = await setPositionTrailing(ticket, 1.5)
      if (error) {
        alert(`Error: ${error}`)
      } else {
        alert(data.message)
      }
    }

    setQuickActionTicket(null)
    setQuickActionType(null)
    fetchData()
  }

  const handleBatchReview = async () => {
    setBatchReviewLoading(true)
    const tickets = selectedPositions.size > 0 ? Array.from(selectedPositions) : undefined
    const { data, error } = await batchReviewPositions(tickets)
    if (error) {
      alert(`Error: ${error}`)
      setBatchReviewLoading(false)
      return
    }

    const taskId = data?.task_id
    if (!taskId) {
      alert("Failed to start batch review")
      setBatchReviewLoading(false)
      return
    }

    // Poll for results
    let completed = false
    let attempts = 0
    const maxAttempts = 60 // 2 minutes max

    while (!completed && attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 2000))
      const statusRes = await getBatchReviewStatus(taskId)
      const statusData = statusRes.data

      if (statusData?.status === "completed") {
        completed = true
        setBatchReviewData(statusData.result?.positions || [])
        setShowBatchReview(true)
      } else if (statusData?.status === "error") {
        alert(`Error: ${statusData.error || "Batch review failed"}`)
        completed = true
      }
      attempts++
    }

    if (!completed) {
      alert("Batch review timed out")
    }
    setBatchReviewLoading(false)
  }

  const handleBatchClose = async () => {
    if (selectedPositions.size === 0) return
    setBatchClosing(true)
    const { data, error } = await batchClosePositions(Array.from(selectedPositions))
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert(`Closed ${data.closed} position(s). ${data.failed} failed.`)
      setSelectedPositions(new Set())
      fetchData()
    }
    setBatchClosing(false)
  }

  const handleCancelOrder = async (ticket: number) => {
    const { data, error } = await cancelOrder(ticket)
    if (error) {
      alert(`Error: ${error}`)
    }
    fetchData()
  }

  const handleBatchCancelOrders = async () => {
    if (selectedOrders.size === 0) return
    setBatchCancelling(true)
    const { data, error } = await batchCancelOrders(Array.from(selectedOrders))
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert(`Cancelled ${data.cancelled} order(s). ${data.failed} failed.`)
      setSelectedOrders(new Set())
      fetchData()
    }
    setBatchCancelling(false)
  }

  const togglePositionSelection = (ticket: number) => {
    const newSet = new Set(selectedPositions)
    if (newSet.has(ticket)) {
      newSet.delete(ticket)
    } else {
      newSet.add(ticket)
    }
    setSelectedPositions(newSet)
  }

  const toggleOrderSelection = (ticket: number) => {
    const newSet = new Set(selectedOrders)
    if (newSet.has(ticket)) {
      newSet.delete(ticket)
    } else {
      newSet.add(ticket)
    }
    setSelectedOrders(newSet)
  }

  const selectAllPositions = () => {
    if (selectedPositions.size === positions.length) {
      setSelectedPositions(new Set())
    } else {
      setSelectedPositions(new Set(positions.map((p) => p.ticket)))
    }
  }

  const selectAllOrders = () => {
    if (selectedOrders.size === orders.length) {
      setSelectedOrders(new Set())
    } else {
      setSelectedOrders(new Set(orders.map((o) => o.ticket)))
    }
  }

  const applyReviewSuggestion = async (ticket: number, newSl?: number, newTp?: number) => {
    if (!newSl && !newTp) return
    const { data, error } = await modifyPosition(ticket, newSl, newTp)
    if (error) {
      alert(`Error: ${error}`)
    } else {
      alert("Position updated successfully")
      setReviewingTicket(null)
      setReviewData(null)
      fetchData()
    }
  }

  const totalProfit = positions.reduce((sum, p) => sum + (p.profit || 0), 0)
  const totalSwap = positions.reduce((sum, p) => sum + (p.swap || 0), 0)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Positions & Orders</h1>
          <p className="text-muted-foreground">Manage your open positions and pending orders</p>
        </div>
        <Button variant="outline" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Open Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{positions.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Pending Orders</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{orders.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Profit</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getProfitColor(totalProfit)}`}>
              {formatCurrency(totalProfit)}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Swap</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getProfitColor(totalSwap)}`}>
              {formatCurrency(totalSwap)}
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="positions">
        <TabsList>
          <TabsTrigger value="positions">
            Open Positions ({positions.length})
          </TabsTrigger>
          <TabsTrigger value="orders">Pending Orders ({orders.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="positions" className="mt-4">
          {/* Batch Actions Toolbar */}
          <div className="flex items-center gap-2 mb-4 p-3 bg-muted/50 rounded-lg">
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllPositions}
              className="gap-2"
            >
              {selectedPositions.size === positions.length && positions.length > 0 ? (
                <CheckSquare className="h-4 w-4" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              {selectedPositions.size === positions.length && positions.length > 0
                ? "Deselect All"
                : "Select All"}
            </Button>

            {selectedPositions.size > 0 && (
              <span className="text-sm text-muted-foreground mx-2">
                {selectedPositions.size} selected
              </span>
            )}

            <div className="flex-1" />

            <Button
              variant="outline"
              size="sm"
              onClick={handleBatchReview}
              disabled={batchReviewLoading}
              className="gap-2"
            >
              {batchReviewLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              {selectedPositions.size > 0 ? `Review Selected (${selectedPositions.size})` : "Review All"}
            </Button>

            {selectedPositions.size > 0 && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm" className="gap-2">
                    <XCircle className="h-4 w-4" />
                    Close Selected ({selectedPositions.size})
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Close {selectedPositions.size} position(s)?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will close all selected positions at market price. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleBatchClose} disabled={batchClosing}>
                      {batchClosing ? "Closing..." : "Close Positions"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
          </div>

          <Card>
            <CardContent className="p-0">
              <ScrollArea className="h-[500px]">
                <table className="w-full">
                  <thead className="sticky top-0 bg-card border-b">
                    <tr className="text-left">
                      <th className="p-4 w-10"></th>
                      <th className="p-4 font-medium">Ticket</th>
                      <th className="p-4 font-medium">Symbol</th>
                      <th className="p-4 font-medium">Type</th>
                      <th className="p-4 font-medium">Volume</th>
                      <th className="p-4 font-medium">Open Price</th>
                      <th className="p-4 font-medium">Current</th>
                      <th className="p-4 font-medium">SL</th>
                      <th className="p-4 font-medium">TP</th>
                      <th className="p-4 font-medium">Profit</th>
                      <th className="p-4 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.length > 0 ? (
                      positions.map((pos) => (
                        <tr
                          key={pos.ticket}
                          className={`border-b hover:bg-muted/50 ${
                            selectedPositions.has(pos.ticket) ? "bg-muted/30" : ""
                          }`}
                        >
                          <td className="p-4">
                            <Checkbox
                              checked={selectedPositions.has(pos.ticket)}
                              onCheckedChange={() => togglePositionSelection(pos.ticket)}
                            />
                          </td>
                          <td className="p-4 font-mono text-sm">{pos.ticket}</td>
                          <td className="p-4 font-medium">{pos.symbol}</td>
                          <td className="p-4">
                            <Badge variant={pos.type === "BUY" ? "buy" : "sell"}>
                              {pos.type}
                            </Badge>
                          </td>
                          <td className="p-4">{pos.volume}</td>
                          <td className="p-4">{pos.open_price}</td>
                          <td className="p-4">{pos.current_price}</td>
                          <td className="p-4 text-red-500">{pos.sl || "—"}</td>
                          <td className="p-4 text-green-500">{pos.tp || "—"}</td>
                          <td className={`p-4 font-medium ${getProfitColor(pos.profit)}`}>
                            {formatCurrency(pos.profit)}
                          </td>
                          <td className="p-4">
                            <div className="flex gap-1">
                              {/* Review Button - navigates to dedicated review page */}
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => router.push(`/positions/${pos.ticket}`)}
                                title="Review Position"
                              >
                                <Eye className="h-4 w-4" />
                              </Button>

                              {/* Breakeven Button */}
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleBreakeven(pos.ticket)}
                                disabled={(quickActionTicket === pos.ticket && quickActionType === "breakeven") || pos.profit <= 0}
                                title={pos.profit > 0 ? "Set SL to entry (breakeven)" : "Position must be in profit for breakeven"}
                                className={pos.profit > 0 ? "text-green-600 hover:text-green-700 hover:bg-green-50" : "text-muted-foreground"}
                              >
                                {quickActionTicket === pos.ticket && quickActionType === "breakeven" ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Shield className="h-4 w-4" />
                                )}
                              </Button>

                              {/* Trailing Stop Button */}
                              <Button
                                variant={pos.trailing_active ? "default" : "outline"}
                                size="sm"
                                onClick={() => handleTrailingStop(pos.ticket, pos.trailing_active)}
                                disabled={quickActionTicket === pos.ticket && quickActionType === "trailing"}
                                title={pos.trailing_active
                                  ? `Trailing active (${pos.trailing_distance?.toFixed(2)}) - Click to disable`
                                  : "Enable trailing stop (1.5x ATR)"
                                }
                                className={pos.trailing_active
                                  ? "bg-yellow-500 hover:bg-yellow-600 text-white"
                                  : "text-yellow-600 hover:text-yellow-700 hover:bg-yellow-50"
                                }
                              >
                                {quickActionTicket === pos.ticket && quickActionType === "trailing" ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Target className="h-4 w-4" />
                                )}
                              </Button>

                              {/* Modify Button */}
                              <Dialog>
                                <DialogTrigger asChild>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                      setModifyingTicket(pos.ticket)
                                      setNewSl(pos.sl?.toString() || "")
                                      setNewTp(pos.tp?.toString() || "")
                                    }}
                                    title="Modify SL/TP"
                                  >
                                    <Edit className="h-4 w-4" />
                                  </Button>
                                </DialogTrigger>
                                <DialogContent>
                                  <DialogHeader>
                                    <DialogTitle>Modify Position</DialogTitle>
                                    <DialogDescription>
                                      Update SL/TP for {pos.symbol} #{pos.ticket}
                                    </DialogDescription>
                                  </DialogHeader>
                                  <div className="space-y-4 py-4">
                                    {/* Quick Actions */}
                                    <div className="space-y-2">
                                      <div className="flex items-center gap-1">
                                        <Label>Quick Actions</Label>
                                        <HelpTooltip content="Quick actions to protect profits. Breakeven sets SL to entry price (requires profit). Trailing sets SL based on current price minus ATR distance." />
                                      </div>
                                      <div className="flex gap-2">
                                        <Button
                                          variant="outline"
                                          size="sm"
                                          onClick={() => handleBreakeven(pos.ticket)}
                                          disabled={quickActionTicket === pos.ticket || pos.profit <= 0}
                                          className={`flex-1 ${pos.profit > 0 ? "text-green-600 border-green-200 hover:bg-green-50" : "text-muted-foreground"}`}
                                        >
                                          {quickActionTicket === pos.ticket && quickActionType === "breakeven" ? (
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                          ) : (
                                            <Shield className="mr-2 h-4 w-4" />
                                          )}
                                          Breakeven (SL → {pos.open_price})
                                        </Button>
                                        <Button
                                          variant={pos.trailing_active ? "default" : "outline"}
                                          size="sm"
                                          onClick={() => handleTrailingStop(pos.ticket, pos.trailing_active)}
                                          disabled={quickActionTicket === pos.ticket}
                                          className={`flex-1 ${pos.trailing_active
                                            ? "bg-yellow-500 hover:bg-yellow-600 text-white border-yellow-500"
                                            : "text-yellow-600 border-yellow-200 hover:bg-yellow-50"
                                          }`}
                                        >
                                          {quickActionTicket === pos.ticket && quickActionType === "trailing" ? (
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                          ) : (
                                            <Target className="mr-2 h-4 w-4" />
                                          )}
                                          {pos.trailing_active ? "Trailing Active" : "Trail (1.5x ATR)"}
                                        </Button>
                                      </div>
                                    </div>

                                    <div className="space-y-2">
                                      <Label>Stop Loss</Label>
                                      <Input
                                        type="number"
                                        step="0.00001"
                                        value={newSl}
                                        onChange={(e) => setNewSl(e.target.value)}
                                        placeholder="Enter new SL"
                                      />
                                    </div>
                                    <div className="space-y-2">
                                      <Label>Take Profit</Label>
                                      <Input
                                        type="number"
                                        step="0.00001"
                                        value={newTp}
                                        onChange={(e) => setNewTp(e.target.value)}
                                        placeholder="Enter new TP"
                                      />
                                    </div>
                                  </div>
                                  <DialogFooter>
                                    <Button onClick={() => handleModify(pos.ticket)}>
                                      Save Changes
                                    </Button>
                                  </DialogFooter>
                                </DialogContent>
                              </Dialog>

                              {/* Close Button */}
                              <Button
                                variant="destructive"
                                size="sm"
                                onClick={() => handleClose(pos.ticket)}
                                disabled={closingTicket === pos.ticket}
                                title="Close Position"
                              >
                                {closingTicket === pos.ticket ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <X className="h-4 w-4" />
                                )}
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={11} className="p-8 text-center text-muted-foreground">
                          No open positions
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="orders" className="mt-4">
          {/* Batch Actions Toolbar for Orders */}
          <div className="flex items-center gap-2 mb-4 p-3 bg-muted/50 rounded-lg">
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllOrders}
              className="gap-2"
            >
              {selectedOrders.size === orders.length && orders.length > 0 ? (
                <CheckSquare className="h-4 w-4" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              {selectedOrders.size === orders.length && orders.length > 0
                ? "Deselect All"
                : "Select All"}
            </Button>

            {selectedOrders.size > 0 && (
              <span className="text-sm text-muted-foreground mx-2">
                {selectedOrders.size} selected
              </span>
            )}

            <div className="flex-1" />

            {selectedOrders.size > 0 && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm" className="gap-2">
                    <Trash2 className="h-4 w-4" />
                    Cancel Selected ({selectedOrders.size})
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Cancel {selectedOrders.size} order(s)?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will cancel all selected pending orders. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Keep Orders</AlertDialogCancel>
                    <AlertDialogAction onClick={handleBatchCancelOrders} disabled={batchCancelling}>
                      {batchCancelling ? "Cancelling..." : "Cancel Orders"}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
          </div>

          <Card>
            <CardContent className="p-0">
              <ScrollArea className="h-[500px]">
                <table className="w-full">
                  <thead className="sticky top-0 bg-card border-b">
                    <tr className="text-left">
                      <th className="p-4 w-10"></th>
                      <th className="p-4 font-medium">Ticket</th>
                      <th className="p-4 font-medium">Symbol</th>
                      <th className="p-4 font-medium">Type</th>
                      <th className="p-4 font-medium">Volume</th>
                      <th className="p-4 font-medium">Price</th>
                      <th className="p-4 font-medium">SL</th>
                      <th className="p-4 font-medium">TP</th>
                      <th className="p-4 font-medium">Time</th>
                      <th className="p-4 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orders.length > 0 ? (
                      orders.map((order) => (
                        <tr
                          key={order.ticket}
                          className={`border-b hover:bg-muted/50 ${
                            selectedOrders.has(order.ticket) ? "bg-muted/30" : ""
                          }`}
                        >
                          <td className="p-4">
                            <Checkbox
                              checked={selectedOrders.has(order.ticket)}
                              onCheckedChange={() => toggleOrderSelection(order.ticket)}
                            />
                          </td>
                          <td className="p-4 font-mono text-sm">{order.ticket}</td>
                          <td className="p-4 font-medium">{order.symbol}</td>
                          <td className="p-4">
                            <Badge
                              variant={
                                order.type.includes("BUY") ? "buy" : "sell"
                              }
                            >
                              {order.type}
                            </Badge>
                          </td>
                          <td className="p-4">{order.volume}</td>
                          <td className="p-4">{order.price}</td>
                          <td className="p-4 text-red-500">{order.sl || "—"}</td>
                          <td className="p-4 text-green-500">{order.tp || "—"}</td>
                          <td className="p-4 text-sm text-muted-foreground">
                            {formatDate(order.time_setup)}
                          </td>
                          <td className="p-4">
                            <Button
                              variant="destructive"
                              size="sm"
                              onClick={() => handleCancelOrder(order.ticket)}
                              title="Cancel Order"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={10} className="p-8 text-center text-muted-foreground">
                          No pending orders
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Batch Review Dialog */}
      <Dialog open={showBatchReview} onOpenChange={setShowBatchReview}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Batch Position Review
            </DialogTitle>
            <DialogDescription>
              ATR-based analysis and recommendations for {batchReviewData.length} position(s)
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="h-[60vh]">
            <div className="space-y-4 pr-4">
              {batchReviewData.map((pos) => (
                <Card key={pos.ticket} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{pos.symbol}</span>
                        <Badge variant={pos.type === "BUY" ? "buy" : "sell"}>
                          {pos.type}
                        </Badge>
                        <span className="text-sm text-muted-foreground">#{pos.ticket}</span>
                      </div>
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Entry:</span>{" "}
                          <span className="font-mono">{pos.entry}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Current:</span>{" "}
                          <span className="font-mono">{pos.current_price}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">P/L:</span>{" "}
                          <span className={getProfitColor(pos.profit)}>
                            {pos.pnl_pct?.toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">ATR:</span>{" "}
                          <span className="font-mono">{pos.atr?.toFixed(5)}</span>
                        </div>
                      </div>
                    </div>
                    <Badge variant="outline">{pos.recommendation}</Badge>
                  </div>

                  {(pos.breakeven_sl || pos.trailing_sl) && (
                    <div className="mt-3 pt-3 border-t flex gap-4">
                      {pos.breakeven_sl && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => applyReviewSuggestion(pos.ticket, pos.breakeven_sl)}
                          className="gap-2"
                        >
                          <TrendingUp className="h-4 w-4 text-green-500" />
                          Breakeven: {pos.breakeven_sl}
                        </Button>
                      )}
                      {pos.trailing_sl && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => applyReviewSuggestion(pos.ticket, pos.trailing_sl)}
                          className="gap-2"
                        >
                          <TrendingDown className="h-4 w-4 text-yellow-500" />
                          Trail: {pos.trailing_sl}
                        </Button>
                      )}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  )
}
