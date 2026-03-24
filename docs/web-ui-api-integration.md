# Web UI API 集成规范

**版本**: v0.1.0
**创建日期**: 2026 年 3 月 21 日
**后端 API**: v1.0 (见 `API.md`)

---

## 一、概述

### 1.1 集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Frontend                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   React     │  │   TanStack  │  │  WebSocket  │         │
│  │ Components  │  │   Query     │  │  Client     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓                ↓                  ↓                │
│  ┌─────────────────────────────────────────────────┐       │
│  │            API Client Layer                      │       │
│  │  - HTTP Client (axios/fetch)                    │       │
│  │  - Request/Response Transformers                │       │
│  │  - Error Handling                               │       │
│  │  - Authentication                               │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│                      Rust Backend                            │
│                   (保持现有架构不变)                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

| 功能 | 技术 | 理由 |
|------|------|------|
| HTTP 客户端 | native fetch | 浏览器原生，无需额外依赖 |
| 数据缓存 | TanStack Query | 自动缓存、重试、乐观更新 |
| WebSocket | native WebSocket | 浏览器原生支持 |
| 类型生成 | openapi-typescript | 从 OpenAPI Schema 自动生成 |
| 表单验证 | Zod | Schema 验证，与 TS 集成 |

---

## 二、API 客户端封装

### 2.1 基础客户端

```typescript
// services/api-client.ts
import { z } from 'zod'

// 基础配置
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000'
const API_TIMEOUT = 30000 // 30 秒

// 通用响应 Schema
export const ApiResponseSchema = <T extends z.ZodType>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema,
    message: z.string().optional(),
    request_id: z.string().optional(),
  })

// 错误响应 Schema
export const ErrorResponseSchema = z.object({
  request_id: z.string(),
  status: z.literal('FAILURE'),
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.unknown()).optional(),
    retryable: z.boolean().optional(),
    suggestion: z.string().optional(),
  }),
  latency_ms: z.number(),
})

export type ApiError = z.infer<typeof ErrorResponseSchema>

class ApiClient {
  private baseUrl: string
  private defaultHeaders: HeadersInit
  
  constructor(baseUrl: string, defaultHeaders: HeadersInit = {}) {
    this.baseUrl = baseUrl
    this.defaultHeaders = defaultHeaders
  }
  
  /**
   * 通用请求方法
   */
  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    }
    
    // 添加超时控制
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT)
    config.signal = controller.signal
    
    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const parsedError = ErrorResponseSchema.safeParse(errorData)
        
        if (parsedError.success) {
          throw new ApiError(parsedError.data)
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
      }
      
      const data = await response.json()
      return data as T
    } finally {
      clearTimeout(timeoutId)
    }
  }
  
  /**
   * GET 请求
   */
  async get<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    const queryString = params 
      ? '?' + new URLSearchParams(params).toString()
      : ''
    return this.request<T>(endpoint + queryString, { method: 'GET' })
  }
  
  /**
   * POST 请求
   */
  async post<T, B = unknown>(endpoint: string, body?: B): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
  }
  
  /**
   * 文件上传
   */
  async upload<T>(endpoint: string, file: File, onProgress?: (progress: number) => void): Promise<T> {
    const formData = new FormData()
    formData.append('file', file)
    
    const xhr = new XMLHttpRequest()
    
    return new Promise((resolve, reject) => {
      xhr.open('POST', `${this.baseUrl}${endpoint}`, true)
      
      Object.entries(this.defaultHeaders).forEach(([key, value]) => {
        xhr.setRequestHeader(key, value)
      })
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = (event.loaded / event.total) * 100
          onProgress(progress)
        }
      }
      
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`))
        }
      }
      
      xhr.onerror = () => reject(new Error('Network error'))
      
      xhr.send(formData)
    })
  }
}

export const apiClient = new ApiClient(API_BASE_URL, {
  'Accept': 'application/json',
})
```

---

### 2.2 类型定义

```typescript
// types/api.ts
import { z } from 'zod'

// ========== 健康检查 ==========
export const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'unhealthy', 'degraded']),
  version: z.string(),
  api_version: z.string(),
})

export type HealthResponse = z.infer<typeof HealthResponseSchema>

// ========== 文件处理 ==========
export const ProcessRequestSchema = z.object({
  file: z.instanceof(File),
  profile: z.enum(['architectural', 'mechanical', 'scanned', 'quick']).optional(),
})

export const ProcessResponseSchema = z.object({
  job_id: z.string(),
  status: z.enum(['completed', 'partial', 'failed']),
  message: z.string(),
  result: z.object({
    scene_summary: z.object({
      outer_boundaries: z.number(),
      holes: z.number(),
      total_points: z.number(),
    }),
    validation_summary: z.object({
      error_count: z.number(),
      warning_count: z.number(),
      passed: z.boolean(),
    }),
    output_size: z.number(),
  }),
  edges: z.array(z.object({
    id: z.number(),
    start: z.tuple([z.number(), z.number()]),
    end: z.tuple([z.number(), z.number()]),
    layer: z.string().optional(),
    is_wall: z.boolean(),
  })).optional(),
})

export type ProcessRequest = z.infer<typeof ProcessRequestSchema>
export type ProcessResponse = z.infer<typeof ProcessResponseSchema>

// ========== 配置管理 ==========
export const ProfileSchema = z.object({
  name: z.string(),
  description: z.string(),
})

export const ProfileDetailSchema = z.object({
  name: z.string(),
  topology: z.object({
    snap_tolerance_mm: z.number(),
    min_line_length_mm: z.number(),
    merge_angle_tolerance_deg: z.number(),
    max_gap_bridge_length_mm: z.number(),
  }),
  validator: z.object({
    closure_tolerance_mm: z.number(),
    min_area_m2: z.number(),
    min_edge_length_mm: z.number(),
    min_angle_deg: z.number(),
  }),
  export: z.object({
    format: z.enum(['json', 'binary']),
    json_indent: z.number(),
    auto_validate: z.boolean(),
  }),
})

export type Profile = z.infer<typeof ProfileSchema>
export type ProfileDetail = z.infer<typeof ProfileDetailSchema>

// ========== 交互功能 ==========
export const AutoTraceRequestSchema = z.object({
  edge_id: z.number(),
})

export const AutoTraceResponseSchema = z.object({
  success: z.boolean(),
  loop_points: z.array(z.tuple([z.number(), z.number()])),
  message: z.string(),
})

export const LassoRequestSchema = z.object({
  polygon: z.array(z.tuple([z.number(), z.number()])),
})

export const LassoResponseSchema = z.object({
  selected_edges: z.array(z.number()),
  loops: z.array(z.array(z.tuple([z.number(), z.number()]))),
  connected_components: z.number(),
})

export const GapDetectionRequestSchema = z.object({
  tolerance: z.number(),
})

export const GapInfoSchema = z.object({
  id: z.number(),
  start: z.tuple([z.number(), z.number()]),
  end: z.tuple([z.number(), z.number()]),
  length: z.number(),
  gap_type: z.enum(['collinear', 'orthogonal', 'angled', 'small']),
})

export const GapDetectionResponseSchema = z.object({
  gaps: z.array(GapInfoSchema),
  total_count: z.number(),
})

export type AutoTraceRequest = z.infer<typeof AutoTraceRequestSchema>
export type AutoTraceResponse = z.infer<typeof AutoTraceResponseSchema>
export type LassoRequest = z.infer<typeof LassoRequestSchema>
export type LassoResponse = z.infer<typeof LassoResponseSchema>
export type GapDetectionRequest = z.infer<typeof GapDetectionRequestSchema>
export type GapDetectionResponse = z.infer<typeof GapDetectionResponseSchema>
export type GapInfo = z.infer<typeof GapInfoSchema>

// ========== 语义标注 ==========
export const BoundarySemanticSchema = z.enum([
  'hard_wall',
  'absorptive_wall',
  'opening',
  'window',
  'door',
  'custom',
])

export const SetSemanticRequestSchema = z.object({
  segment_id: z.number(),
  semantic: BoundarySemanticSchema,
})

export const SetSemanticResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
})

export type BoundarySemantic = z.infer<typeof BoundarySemanticSchema>
export type SetSemanticRequest = z.infer<typeof SetSemanticRequestSchema>
export type SetSemanticResponse = z.infer<typeof SetSemanticResponseSchema>
```

---

### 2.3 服务层封装

```typescript
// services/file-service.ts
import { apiClient } from './api-client'
import type { ProcessRequest, ProcessResponse, Profile, ProfileDetail } from '@/types/api'

export class FileService {
  /**
   * 处理文件
   */
  async processFile(
    file: File, 
    profile?: ProcessRequest['profile'],
    onProgress?: (progress: number) => void
  ): Promise<ProcessResponse> {
    const endpoint = profile ? `/process?profile=${profile}` : '/process'
    return apiClient.upload<ProcessResponse>(endpoint, file, onProgress)
  }
  
  /**
   * 获取所有预设配置
   */
  async listProfiles(): Promise<Profile[]> {
    return apiClient.get('/config/profiles')
  }
  
  /**
   * 获取配置详情
   */
  async getProfile(name: string): Promise<ProfileDetail> {
    return apiClient.get(`/config/profile/${name}`)
  }
  
  /**
   * 导出场景
   */
  async exportScene(
    format: 'json' | 'binary' = 'json'
  ): Promise<Blob> {
    const response = await fetch(`${apiClient.baseUrl}/export?format=${format}`)
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    return response.blob()
  }
  
  /**
   * 下载导出文件
   */
  async downloadExport(
    format: 'json' | 'binary' = 'json',
    filename?: string
  ): Promise<void> {
    const blob = await this.exportScene(format)
    const url = URL.createObjectURL(blob)
    
    const a = document.createElement('a')
    a.href = url
    a.download = filename ?? `scene.${format === 'json' ? 'json' : 'bin'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }
}

export const fileService = new FileService()
```

```typescript
// services/interaction-service.ts
import { apiClient } from './api-client'
import type {
  AutoTraceRequest,
  AutoTraceResponse,
  LassoRequest,
  LassoResponse,
  GapDetectionRequest,
  GapDetectionResponse,
  SetSemanticRequest,
  SetSemanticResponse,
} from '@/types/api'

export class InteractionService {
  /**
   * 自动追踪
   */
  async autoTrace(request: AutoTraceRequest): Promise<AutoTraceResponse> {
    return apiClient.post<AutoTraceResponse>('/interact/auto_trace', request)
  }
  
  /**
   * 圈选区域
   */
  async lasso(request: LassoRequest): Promise<LassoResponse> {
    return apiClient.post<LassoResponse>('/interact/lasso', request)
  }
  
  /**
   * 缺口检测
   */
  async detectGaps(request: GapDetectionRequest): Promise<GapDetectionResponse> {
    return apiClient.post<GapDetectionResponse>('/interact/detect_gaps', request)
  }
  
  /**
   * 设置边界语义
   */
  async setSemantic(request: SetSemanticRequest): Promise<SetSemanticResponse> {
    return apiClient.post<SetSemanticResponse>('/interact/set_boundary_semantic', request)
  }
  
  /**
   * 获取交互状态
   */
  async getState(): Promise<{
    total_edges: number
    selected_edges: number[]
    detected_gaps: Array<{
      id: number
      start: [number, number]
      end: [number, number]
      length: number
      gap_type: string
    }>
  }> {
    return apiClient.get('/interact/state')
  }
}

export const interactionService = new InteractionService()
```

---

## 三、TanStack Query 集成

### 3.1 Query Client 配置

```typescript
// lib/query-client.ts
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 重试策略
      retry: (failureCount, error) => {
        // 不重试 4xx 错误
        if (error instanceof Error && error.message.includes('4')) {
          return false
        }
        return failureCount < 3
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      
      // 缓存策略
      staleTime: 5 * 60 * 1000, // 5 分钟
      gcTime: 10 * 60 * 1000, // 10 分钟后 GC
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      
      // 超时
      networkMode: 'online',
    },
    mutations: {
      retry: 1,
      networkMode: 'always',
    },
  },
})
```

### 3.2 自定义 Hooks

```typescript
// hooks/use-file-upload.ts
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { fileService } from '@/services/file-service'
import { useToast } from '@/hooks/use-toast'

export function useFileUpload() {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  
  const mutation = useMutation({
    mutationFn: async ({ file, profile }: { file: File; profile?: string }) => {
      return fileService.processFile(file, profile as any, (progress) => {
        // 更新上传进度
        queryClient.setQueryData(['uploadProgress'], progress)
      })
    },
    
    onSuccess: (data) => {
      toast({
        title: '处理成功',
        description: `已加载 ${data.result.scene_summary.outer_boundaries} 个外边界`,
        variant: 'success',
      })
      
      // 使场景查询失效
      queryClient.invalidateQueries({ queryKey: ['scene'] })
    },
    
    onError: (error) => {
      toast({
        title: '处理失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    },
  })
  
  return {
    uploadFile: mutation.mutateAsync,
    isUploading: mutation.isPending,
    progress: queryClient.getQueryData<number>(['uploadProgress']) ?? 0,
  }
}
```

```typescript
// hooks/use-auto-trace.ts
import { useMutation } from '@tanstack/react-query'
import { interactionService } from '@/services/interaction-service'
import { useToast } from '@/hooks/use-toast'

export function useAutoTrace() {
  const { toast } = useToast()
  
  const mutation = useMutation({
    mutationFn: async (edgeId: number) => {
      return interactionService.autoTrace({ edge_id: edgeId })
    },
    
    onSuccess: (data) => {
      if (data.success) {
        toast({
          title: '追踪成功',
          description: '已找到闭合环',
          variant: 'success',
        })
      } else {
        toast({
          title: '追踪失败',
          description: data.message,
          variant: 'warning',
        })
      }
    },
    
    onError: (error) => {
      toast({
        title: '追踪失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    },
  })
  
  return {
    autoTrace: mutation.mutateAsync,
    isTracing: mutation.isPending,
  }
}
```

```typescript
// hooks/use-scene.ts
import { useQuery } from '@tanstack/react-query'
import { fileService } from '@/services/file-service'

export function useScene() {
  const query = useQuery({
    queryKey: ['scene'],
    queryFn: async () => {
      // 从后端获取场景数据
      const response = await fetch('/api/scene')
      if (!response.ok) throw new Error('Failed to fetch scene')
      return response.json()
    },
    enabled: false, // 手动触发
    staleTime: Infinity, // 数据不会过期
  })
  
  return {
    scene: query.data,
    isLoading: query.isLoading,
    refetch: query.refetch,
  }
}
```

---

## 四、WebSocket 客户端

### 3.1 WebSocket 服务

```typescript
// services/websocket-client.ts
import { EventEmitter } from 'eventemitter3'
import type {
  AutoTraceResponse,
  GapDetectionResponse,
} from '@/types/api'

// WebSocket 事件类型
interface WebSocketEvents {
  'connected': () => void
  'disconnected': () => void
  'error': (error: Error) => void
  
  // 服务器推送事件
  'edge_selected': (data: { edge_id: number; trace_result: any }) => void
  'auto_trace_result': (data: AutoTraceResponse) => void
  'gap_detection': (data: GapDetectionResponse) => void
  'topology_update': (data: { edges: any[]; loops: any[] }) => void
  'parse_progress': (data: { stage: string; progress: number }) => void
  'pong': (data: { latency_ms: number }) => void
}

export class WebSocketClient extends EventEmitter<WebSocketEvents> {
  private ws: WebSocket | null = null
  private url: string
  private reconnectDelay = 1000
  private maxReconnectDelay = 30000
  private reconnectTimer: NodeJS.Timeout | null = null
  private pingInterval: NodeJS.Timeout | null = null
  
  constructor(url: string) {
    super()
    this.url = url
  }
  
  /**
   * 连接 WebSocket
   */
  connect() {
    if (this.ws?.readyState === WebSocket.CONNECTING || this.ws?.readyState === WebSocket.OPEN) {
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
      this.emit('error', new Error('WebSocket connection error'))
    }
    
    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        this.emit(message.type, message.payload)
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
      throw new Error('WebSocket is not connected')
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
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay)
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
  import.meta.env.VITE_WS_URL || 'ws://localhost:3000/ws'
)
```

---

### 3.2 WebSocket Hook

```typescript
// hooks/use-websocket.ts
import { useEffect, useCallback } from 'react'
import { wsClient } from '@/services/websocket-client'
import type { AutoTraceResponse, GapDetectionResponse } from '@/types/api'

interface UseWebSocketOptions {
  onEdgeSelected?: (data: { edge_id: number; trace_result: any }) => void
  onAutoTraceResult?: (data: AutoTraceResponse) => void
  onGapDetection?: (data: GapDetectionResponse) => void
  onTopologyUpdate?: (data: { edges: any[]; loops: any[] }) => void
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
  }, [onEdgeSelected, onAutoTraceResult, onGapDetection, onTopologyUpdate, onParseProgress, onDisconnect])
  
  // 暴露方法
  const selectEdge = useCallback((edgeId: number) => {
    wsClient.selectEdge(edgeId)
  }, [])
  
  const detectGaps = useCallback((tolerance: number) => {
    wsClient.detectGaps(tolerance)
  }, [])
  
  const disconnect = useCallback(() => {
    wsClient.disconnect()
  }, [])
  
  return {
    isConnected: wsClient.isConnected(),
    selectEdge,
    detectGaps,
    disconnect,
  }
}
```

---

## 五、错误处理

### 5.1 错误分类

```typescript
// lib/errors.ts
export class ApiError extends Error {
  constructor(
    public readonly data: {
      request_id: string
      status: string
      error: {
        code: string
        message: string
        details?: Record<string, unknown>
        retryable?: boolean
        suggestion?: string
      }
      latency_ms: number
    }
  ) {
    super(data.error.message)
    this.name = 'ApiError'
  }
  
  get code(): string {
    return this.data.error.code
  }
  
  get retryable(): boolean {
    return this.data.error.retryable ?? false
  }
  
  get suggestion(): string | undefined {
    return this.data.error.suggestion
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class TimeoutError extends Error {
  constructor() {
    super('Request timeout')
    this.name = 'TimeoutError'
  }
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly field?: string
  ) {
    super(message)
    this.name = 'ValidationError'
  }
}
```

### 5.2 错误边界

```typescript
// components/error-boundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }
  
  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }
  
  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error', error, errorInfo)
  }
  
  private handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }
  
  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }
      
      return (
        <div className="flex items-center justify-center min-h-[400px]">
          <Alert variant="destructive" className="max-w-md">
            <AlertCircle className="h-5 w-5" />
            <AlertTitle>出错了</AlertTitle>
            <AlertDescription className="mt-2">
              {this.state.error?.message ?? '未知错误'}
            </AlertDescription>
            <Button onClick={this.handleRetry} className="mt-4">
              重试
            </Button>
          </Alert>
        </div>
      )
    }
    
    return this.props.children
  }
}
```

---

## 六、环境配置

### 6.1 环境变量

```bash
# .env.example
# API 配置
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000/ws

# 功能开关
VITE_ENABLE_GPU=false
VITE_ENABLE_WEBSOCKET=true

# 调试
VITE_DEBUG=true
VITE_LOG_LEVEL=debug
```

### 6.2 TypeScript 配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## 七、测试

### 7.1 API 测试

```typescript
// tests/unit/api-client.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { apiClient } from '@/services/api-client'

// Mock fetch
global.fetch = vi.fn()

describe('ApiClient', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
  
  it('应该成功获取健康状态', async () => {
    const mockData = {
      status: 'healthy',
      version: '1.0.0',
      api_version: 'v1',
    }
    
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    } as Response)
    
    const result = await apiClient.get('/health')
    
    expect(result).toEqual(mockData)
    expect(fetch).toHaveBeenCalledWith('http://localhost:3000/health', {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: expect.any(AbortSignal),
    })
  })
  
  it('应该处理 API 错误', async () => {
    const errorData = {
      request_id: 'req-123',
      status: 'FAILURE',
      error: {
        code: 'DXF_PARSE_ERROR',
        message: 'DXF 解析失败',
      },
      latency_ms: 45,
    }
    
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => errorData,
    } as Response)
    
    await expect(apiClient.get('/process')).rejects.toThrow('DXF 解析失败')
  })
})
```

### 7.2 Hook 测试

```typescript
// tests/unit/use-file-upload.test.tsx
import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useFileUpload } from '@/hooks/use-file-upload'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
})

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}

describe('useFileUpload', () => {
  it('应该成功上传文件', async () => {
    const file = new File(['test'], 'test.dxf', { type: 'application/dxf' })
    
    const { result } = renderHook(() => useFileUpload(), { wrapper })
    
    // Mock fileService
    vi.mock('@/services/file-service', () => ({
      fileService: {
        processFile: vi.fn().mockResolvedValue({
          result: {
            scene_summary: {
              outer_boundaries: 1,
              holes: 0,
              total_points: 100,
            },
          },
        }),
      },
    }))
    
    await result.current.uploadFile({ file })
    
    await waitFor(() => {
      expect(result.current.isUploading).toBe(false)
    })
  })
})
```

---

**创建者**: CAD 团队
**最后更新**: 2026 年 3 月 21 日
