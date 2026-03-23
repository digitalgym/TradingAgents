"use client"

import { useEffect, useRef, useState, useCallback } from "react"

interface WebSocketMessage {
  type: string
  [key: string]: any
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log("WebSocket connected")
        setIsConnected(true)
        reconnectCountRef.current = 0
        onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage
          setLastMessage(message)
          onMessage?.(message)
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e)
        }
      }

      ws.onclose = () => {
        console.log("WebSocket disconnected")
        setIsConnected(false)
        onDisconnect?.()

        // Attempt reconnection
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++
            console.log(`Reconnecting... (attempt ${reconnectCountRef.current})`)
            connect()
          }, reconnectInterval)
        }
      }

      ws.onerror = (error) => {
        console.error("WebSocket error:", error)
      }

      wsRef.current = ws
    } catch (error) {
      console.error("Failed to create WebSocket:", error)
    }
  }, [onConnect, onDisconnect, onMessage, reconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectCountRef.current = reconnectAttempts // Prevent reconnection
    wsRef.current?.close()
    wsRef.current = null
  }, [reconnectAttempts])

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  const subscribe = useCallback((channel: string) => {
    send({ type: "subscribe", channel })
  }, [send])

  const unsubscribe = useCallback((channel: string) => {
    send({ type: "unsubscribe", channel })
  }, [send])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  }
}

// Hook for subscribing to analysis updates
export function useAnalysisUpdates(onUpdate?: (data: any) => void) {
  const [analysisStatus, setAnalysisStatus] = useState<any>(null)

  const { isConnected } = useWebSocket({
    onMessage: (message) => {
      if (
        message.type === "analysis_started" ||
        message.type === "analysis_completed" ||
        message.type === "analysis_error"
      ) {
        setAnalysisStatus(message)
        onUpdate?.(message)
      }
    },
  })

  return { isConnected, analysisStatus }
}

// Hook for subscribing to position updates
export function usePositionUpdates(onUpdate?: (data: any) => void) {
  const [positions, setPositions] = useState<any[]>([])

  const { isConnected } = useWebSocket({
    onMessage: (message) => {
      if (message.type === "positions_update") {
        setPositions(message.positions || [])
        onUpdate?.(message)
      }
    },
  })

  return { isConnected, positions }
}

// Hook for subscribing to automation status updates (start/stop/pause/resume)
export interface AutomationStatusEvent {
  type: "automation_status"
  instance: string
  status: string // running, stopped, paused, pending_start, stopping, error
  error?: string
}

export function useAutomationStatus(onUpdate?: (event: AutomationStatusEvent) => void) {
  const onUpdateRef = useRef(onUpdate)
  onUpdateRef.current = onUpdate

  const { isConnected } = useWebSocket({
    onMessage: useCallback((message: WebSocketMessage) => {
      if (message.type === "automation_status") {
        onUpdateRef.current?.(message as AutomationStatusEvent)
      }
    }, []),
  })

  return { isConnected }
}
