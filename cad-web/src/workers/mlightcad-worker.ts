/**
 * mlightcad 几何提取 Web Worker
 *
 * 在后台线程执行几何数据提取，避免阻塞主线程
 * 支持进度报告和错误处理
 */

import { extractGeometryFromDocument } from '../lib/mlightcad-geometry-extractor'

// ========== 类型定义 ==========

/**
 * ✅ S016: 完善的 Worker 消息类型定义
 * 使用泛型确保类型安全
 */

// 请求消息类型
export interface ExtractRequest {
  type: 'extract'
  payload: ExtractPayload
}

export interface CancelRequest {
  type: 'cancel'
}

export type WorkerRequest = ExtractRequest | CancelRequest

export interface ExtractPayload {
  documentData: ArrayBuffer
  fileName: string
  // ✅ S011: 增量解析选项
  options?: {
    /** 文件是否大于 5MB，启用增量解析 */
    isLargeFile?: boolean
    /** 优先解析阶段：'model_space' (仅模型空间) | 'all' (完整解析) */
    priority?: 'model_space' | 'all'
  }
}

// 响应消息类型
export interface ProgressResponse {
  type: 'progress'
  payload: ProgressPayload
}

export interface SuccessResponse<T = unknown> {
  type: 'success'
  payload: T
}

export interface ErrorResponse {
  type: 'error'
  payload: ErrorPayload
}

// 标准响应负载
export interface ProgressPayload {
  stage: 'loading' | 'parsing' | 'extracting' | 'complete'
  progress: number
  // ✅ S011: 可选的进度消息
  message?: string
}

/**
 * ✅ S013: 使用结构化克隆，直接使用 Edge[] 和 HatchEntity[]
 * 不再需要 ArrayBuffer 转换
 */
export interface GeometryExtractResult {
  edges: import('@/types/api').Edge[]
  hatches: import('@/types/api').HatchEntity[]
  bounds: {
    minX: number
    minY: number
    maxX: number
    maxY: number
  }
}

export type SuccessPayload = GeometryExtractResult

export interface ErrorPayload {
  message: string
  stack?: string
}

// Worker 响应联合类型
export type WorkerResponse = ProgressResponse | SuccessResponse<SuccessPayload> | ErrorResponse

// ========== 全局状态 ==========

let currentDocument: any = null
let isCancelling = false

// ========== 消息处理 ==========

/**
 * 处理提取请求
 */
async function handleExtract(payload: ExtractPayload): Promise<void> {
  isCancelling = false

  try {
    // ✅ S011: 检测文件大小，决定是否启用增量解析
    const fileSizeMB = payload.documentData.byteLength / (1024 * 1024)
    const isLargeFile = payload.options?.isLargeFile || fileSizeMB > 5
    const priority = payload.options?.priority || 'all'

    console.log('[MlightCad Worker] Starting extraction:', {
      fileSizeMB: fileSizeMB.toFixed(2),
      isLargeFile,
      priority,
    })

    // ✅ S008: 尝试使用预加载的 WASM 模块
    let module: any

    // 检查是否有预加载的模块（通过 window.__MLIGHTCAD__ 传递到 Worker）
    if ((self as any).__MLIGHTCAD__) {
      module = (self as any).__MLIGHTCAD__
      console.log('[MlightCad Worker] Using preloaded WASM module')
    } else {
      // 动态导入 mlightcad（避免在 Worker 启动时加载）
      module = await import('@mlightcad/cad-simple-viewer')
      console.log('[MlightCad Worker] Dynamically loaded WASM module')
    }

    const { AcApDocument, AcEdOpenMode } = module

    // 创建文档
    currentDocument = new AcApDocument()

    // 发送进度更新
    postMessage({
      type: 'progress',
      payload: { stage: 'loading', progress: 10 },
    } satisfies ProgressResponse)

    // 打开文档
    const success = await currentDocument.openDocument(
      payload.fileName,
      payload.documentData,
      { mode: AcEdOpenMode.Read }
    )

    if (!success || !currentDocument.database) {
      throw new Error('Failed to open document')
    }

    // 发送进度更新
    postMessage({
      type: 'progress',
      payload: { stage: 'parsing', progress: 30 },
    } satisfies ProgressResponse)

    // 检查是否取消
    if (isCancelling) {
      throw new Error('Cancelled')
    }

    // ✅ S011: 增量解析 - 优先解析模型空间
    if (isLargeFile && priority === 'model_space') {
      console.log('[MlightCad Worker] Large file detected, extracting model space first...')
      
      // 发送阶段性进度更新
      postMessage({
        type: 'progress',
        payload: { stage: 'extracting', progress: 50, message: '正在解析模型空间...' },
      } satisfies ProgressResponse)

      // 仅提取模型空间的几何数据
      const geometryData = await extractGeometryFromDocument(currentDocument, {
        lodLevel: 'medium', // 中等 LOD 以加快提取
      })

      // 检查是否取消
      if (isCancelling) {
        throw new Error('Cancelled')
      }

      // 发送第一阶段结果
      postMessage({
        type: 'progress',
        payload: { stage: 'complete', progress: 80, message: '模型空间解析完成' },
      } satisfies ProgressResponse)

      // 发送模型空间数据
      const response: SuccessResponse<SuccessPayload> = {
        type: 'success',
        payload: geometryData,
      }
      ;(self as unknown as DedicatedWorkerGlobalScope).postMessage(response)

      // 清理
      await cleanup()
      return
    }

    // 小文件或者需要完整解析，直接完整解析
    console.log('[MlightCad Worker] Extracting all data...')

    // 发送进度更新
    postMessage({
      type: 'progress',
      payload: { stage: 'extracting', progress: 60, message: '正在解析完整数据...' },
    } satisfies ProgressResponse)

    // 提取完整数据
    const fullGeometryData = await extractGeometryFromDocument(currentDocument, {
      lodLevel: 'high', // 高 LOD 保证质量
    })

    // 检查是否取消
    if (isCancelling) {
      throw new Error('Cancelled')
    }

    // 发送完整数据
    postMessage({
      type: 'progress',
      payload: { stage: 'complete', progress: 100 },
    } satisfies ProgressResponse)

    const fullResponse: SuccessResponse<SuccessPayload> = {
      type: 'success',
      payload: fullGeometryData,
    }
    ;(self as unknown as DedicatedWorkerGlobalScope).postMessage(fullResponse)

    // 清理
    await cleanup()

  } catch (error) {
    if (error instanceof Error && error.message === 'Cancelled') {
      postMessage({
        type: 'error',
        payload: { message: 'Extraction cancelled' } satisfies ErrorPayload,
      } satisfies ErrorResponse)
    } else {
      postMessage({
        type: 'error',
        payload: {
          message: error instanceof Error ? error.message : 'Unknown error',
          stack: error instanceof Error ? error.stack : undefined,
        } satisfies ErrorPayload,
      } satisfies ErrorResponse)
    }

    await cleanup()
  }
}

/**
 * 处理取消请求
 */
function handleCancel(): void {
  isCancelling = true
  console.log('[MlightCad Worker] Cancel requested')
}

/**
 * 清理资源
 */
async function cleanup(): Promise<void> {
  if (currentDocument) {
    try {
      currentDocument.database = null
      if (typeof currentDocument.dispose === 'function') {
        await currentDocument.dispose()
      }
    } catch (error) {
      console.error('[MlightCad Worker] Error during cleanup:', error)
    } finally {
      currentDocument = null
    }
  }
  isCancelling = false
}

// ========== Worker 消息监听 ==========

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { type } = event.data

  switch (type) {
    case 'extract': {
      const payload = (event.data as ExtractRequest).payload
      if (payload) {
        await handleExtract(payload)
      }
      break
    }

    case 'cancel':
      handleCancel()
      break

    default:
      console.warn('[MlightCad Worker] Unknown message type:', type)
  }
}
