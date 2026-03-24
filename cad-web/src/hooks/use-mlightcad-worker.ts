/**
 * mlightcad Web Worker Hook
 *
 * 在后台线程执行几何数据提取，避免阻塞主线程
 */

import { useCallback, useRef, useEffect } from 'react'
import type { Edge, HatchEntity } from '@/types/api'
import type {
  WorkerResponse,
  ProgressPayload,
  SuccessPayload,
  ErrorResponse,
} from '@/workers/mlightcad-worker'

export interface WorkerExtractResult {
  edges: Edge[]
  hatches: HatchEntity[]
  bounds: { minX: number; minY: number; maxX: number; maxY: number }
}

export interface WorkerProgress {
  stage: 'loading' | 'parsing' | 'extracting' | 'complete'
  progress: number
  // ✅ S011: 可选的进度消息
  message?: string
}

export function useMlightCadWorker() {
  const workerRef = useRef<Worker | null>(null)
  const callbacksRef = useRef<{
    onSuccess?: (result: WorkerExtractResult) => void
    onError?: (error: Error) => void
    onProgress?: (progress: WorkerProgress) => void
  }>({})

  // 初始化 Worker
  useEffect(() => {
    // 创建 Worker 实例 - 使用 Vite 的 ?worker 导入
    const workerFactory = new Worker(
      new URL('../workers/mlightcad-worker.ts', import.meta.url),
      { type: 'module' }
    )
    
    workerRef.current = workerFactory

    // 监听 Worker 消息 - ✅ S016: 使用具体类型
    workerRef.current.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const { type, payload } = event.data

      switch (type) {
        case 'progress': {
          const progressResponse = payload as ProgressPayload
          callbacksRef.current.onProgress?.(progressResponse)
          break
        }

        case 'success': {
          // ✅ S013: 使用结构化克隆直接接收数据
          // 现代浏览器支持结构化克隆算法，无需手动序列化/反序列化
          const successPayload = payload as SuccessPayload

          callbacksRef.current.onSuccess?.({
            edges: successPayload.edges,
            hatches: successPayload.hatches,
            bounds: successPayload.bounds,
          })
          break
        }

        case 'error': {
          const errorPayload = payload as ErrorResponse['payload']
          callbacksRef.current.onError?.(new Error(errorPayload.message))
          break
        }
      }
    }

    // 清理
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate()
        workerRef.current = null
      }
    }
  }, [])

  // 提取几何数据
  const extract = useCallback(
    (
      file: File,
      options: {
        onSuccess: (result: WorkerExtractResult) => void
        onError?: (error: Error) => void
        onProgress?: (progress: WorkerProgress) => void
        // ✅ S011: 增量解析选项
        incrementalOptions?: {
          /** 是否启用增量解析（默认根据文件大小自动判断） */
          enabled?: boolean
          /** 优先解析阶段 */
          priority?: 'model_space' | 'all'
        }
      }
    ) => {
      if (!workerRef.current) {
        console.error('[useMlightCadWorker] Worker not initialized')
        options.onError?.(new Error('Worker not initialized'))
        return
      }

      // 保存回调
      callbacksRef.current = {
        onSuccess: options.onSuccess,
        onError: options.onError,
        onProgress: options.onProgress,
      }

      // ✅ S010: 使用异步 FileReader 读取文件（不阻塞主线程）
      // 注意：File 对象不能直接传输到 Worker，需要在主线程读取
      // 但异步 FileReader 已经能避免阻塞主线程
      const reader = new FileReader()

      // 监听读取进度（仅大文件有效）
      reader.onprogress = (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 20 // 文件读取占 20% 进度
          options.onProgress?.({ stage: 'loading', progress })
        }
      }

      reader.onload = (e) => {
        if (!e.target?.result) {
          options.onError?.(new Error('Failed to read file'))
          return
        }

        // 文件读取完成，发送进度更新
        options.onProgress?.({ stage: 'loading', progress: 20 })

        // ✅ S011: 构建增量解析选项
        const fileSizeMB = file.size / (1024 * 1024)
        const extractOptions = {
          isLargeFile: fileSizeMB > 5,
          priority: options.incrementalOptions?.priority || 'all' as const,
        }

        // 发送数据到 Worker 进行几何提取
        workerRef.current?.postMessage({
          type: 'extract',
          payload: {
            documentData: e.target.result as ArrayBuffer,
            fileName: file.name,
            options: extractOptions,
          },
        })
      }

      reader.onerror = () => {
        options.onError?.(new Error('Failed to read file'))
      }

      // ✅ 异步读取文件，不会阻塞主线程
      reader.readAsArrayBuffer(file)
    },
    []
  )

  // 取消提取
  const cancel = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'cancel' })
    }
  }, [])

  return { extract, cancel }
}
