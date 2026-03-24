import { EventEmitter } from 'eventemitter3'
import type {
  AutoTraceResponse,
  GapDetectionResponse,
  Edge,
} from '@/types/api'

// WebSocket 事件类型
interface WebSocketEvents {
  connected: () => void
  disconnected: () => void
  error: (error: Error) => void

  // 服务器推送事件
  edge_selected: (data: { edge_id: number; trace_result: unknown }) => void
  auto_trace_result: (data: AutoTraceResponse) => void
  gap_detection: (data: GapDetectionResponse) => void
  topology_update: (data: { edges: Edge[]; loops: unknown[] }) => void
  parse_progress: (data: { stage: string; progress: number }) => void
  pong: (data: { latency_ms: number }) => void
}

export class WebSocketClient extends EventEmitter<WebSocketEvents> {
  private ws: WebSocket | null = null
  private url: string
  private reconnectDelay = 1000
  private maxReconnectDelay = 30000
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private pingInterval: ReturnType<typeof setInterval> | null = null

  constructor(url: string) {
    super()
    this.url = url
  }

  /**
   * 连接 WebSocket
   */
  connect() {
    if (
      this.ws?.readyState === WebSocket.CONNECTING ||
      this.ws?.readyState === WebSocket.OPEN
    ) {
      return
    }

    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectDelay = 1000
      this.emit('connected')
      this.startPing()
    }

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected', event.code, event.reason)
      this.emit('disconnected')
      this.stopPing()
      this.scheduleReconnect()
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error', error)
      this.emit('error', new Error('WebSocket 连接错误'))
    }

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        this.emit(message.type as keyof WebSocketEvents, message.payload)
      } catch (error) {
        console.error('Failed to parse WebSocket message', error)
      }
    }
  }

  /**
   * 发送消息
   */
  send<T>(type: string, payload: T) {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket 未连接')
    }

    this.ws.send(JSON.stringify({ type, payload }))
  }

  /**
   * 断开连接
   */
  disconnect() {
    this.stopPing()

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    this.ws?.close()
    this.ws = null
  }

  /**
   * 检查连接状态
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  /**
   * 发送选边消息
   */
  selectEdge(edgeId: number) {
    this.send('select_edge', { edge_id: edgeId })
  }

  /**
   * 发送缺口检测消息
   */
  detectGaps(tolerance: number) {
    this.send('detect_gaps', { tolerance })
  }

  /**
   * 发送心跳
   */
  ping() {
    if (this.isConnected()) {
      this.send('ping', {})
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) {
      return
    }

    this.reconnectTimer = setTimeout(() => {
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxReconnectDelay
      )
      console.log(`Reconnecting in ${this.reconnectDelay}ms`)
      this.connect()
    }, this.reconnectDelay)
  }

  private startPing() {
    this.pingInterval = setInterval(() => {
      this.ping()
    }, 30000) // 30 秒心跳
  }

  private stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval)
      this.pingInterval = null
    }
  }
}

// 创建单例
export const wsClient = new WebSocketClient(
  (import.meta as any).env.VITE_WS_URL || 'ws://localhost:3000/ws'
)
