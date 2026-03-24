import { useEffect } from 'react'
import { wsClient } from '@/services/websocket-client'
import type {
  AutoTraceResponse,
  GapDetectionResponse,
} from '@/types/api'

interface UseWebSocketOptions {
  onEdgeSelected?: (data: { edge_id: number; trace_result: unknown }) => void
  onAutoTraceResult?: (data: AutoTraceResponse) => void
  onGapDetection?: (data: GapDetectionResponse) => void
  onTopologyUpdate?: (data: { edges: unknown[]; loops: unknown[] }) => void
  onParseProgress?: (data: { stage: string; progress: number }) => void
  onDisconnect?: () => void
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onEdgeSelected,
    onAutoTraceResult,
    onGapDetection,
    onTopologyUpdate,
    onParseProgress,
    onDisconnect,
  } = options

  useEffect(() => {
    // 连接 WebSocket
    wsClient.connect()

    // 订阅事件
    if (onEdgeSelected) {
      wsClient.on('edge_selected', onEdgeSelected)
    }
    if (onAutoTraceResult) {
      wsClient.on('auto_trace_result', onAutoTraceResult)
    }
    if (onGapDetection) {
      wsClient.on('gap_detection', onGapDetection)
    }
    if (onTopologyUpdate) {
      wsClient.on('topology_update', onTopologyUpdate)
    }
    if (onParseProgress) {
      wsClient.on('parse_progress', onParseProgress)
    }
    if (onDisconnect) {
      wsClient.on('disconnected', onDisconnect)
    }

    // 清理
    return () => {
      if (onEdgeSelected) {
        wsClient.off('edge_selected', onEdgeSelected)
      }
      if (onAutoTraceResult) {
        wsClient.off('auto_trace_result', onAutoTraceResult)
      }
      if (onGapDetection) {
        wsClient.off('gap_detection', onGapDetection)
      }
      if (onTopologyUpdate) {
        wsClient.off('topology_update', onTopologyUpdate)
      }
      if (onParseProgress) {
        wsClient.off('parse_progress', onParseProgress)
      }
      if (onDisconnect) {
        wsClient.off('disconnected', onDisconnect)
      }
    }
  }, [
    onEdgeSelected,
    onAutoTraceResult,
    onGapDetection,
    onTopologyUpdate,
    onParseProgress,
    onDisconnect,
  ])

  // 暴露方法
  const selectEdge = (edgeId: number) => {
    wsClient.selectEdge(edgeId)
  }

  const detectGaps = (tolerance: number) => {
    wsClient.detectGaps(tolerance)
  }

  const disconnect = () => {
    wsClient.disconnect()
  }

  return {
    isConnected: wsClient.isConnected(),
    selectEdge,
    detectGaps,
    disconnect,
  }
}
