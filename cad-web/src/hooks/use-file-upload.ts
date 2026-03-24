import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { fileService } from '@/services/file-service'
import { useCanvasStore, type HatchEntity } from '@/stores/canvas-store'
import { toast } from 'sonner'
import { getFromCache, setCache, fileHash } from '@/lib/dxf-cache'
import { useMlightCadWorker } from '@/hooks/use-mlightcad-worker'

interface UseFileUploadOptions {
  profile?: 'architectural' | 'mechanical' | 'scanned' | 'quick'
  useFrontendParse?: boolean
  onSuccess?: () => void
  onError?: (error: Error) => void
}

export function useFileUpload(options: UseFileUploadOptions = {}) {
  const {
    profile,
    useFrontendParse = true,
    onSuccess,
    onError
  } = options

  const queryClient = useQueryClient()
  const { setEdges, setHatches, setLoading, setUploadProgress, fitToContent, setParseMethod } = useCanvasStore()
  const [isUploading, setIsUploading] = useState(false)

  const { extract, cancel } = useMlightCadWorker()

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      setIsUploading(true)
      setParseMethod('frontend')

      // ✅ 计算文件哈希，检查缓存
      const hash = await fileHash(file)
      const cachedEntry = getFromCache(hash)

      // ✅ 如果缓存命中，直接返回缓存数据
      if (cachedEntry) {
        console.log('[useFileUpload] Cache hit:', hash)
        setParseMethod('frontend')
        return {
          parseMethod: 'frontend' as const,
          edges: cachedEntry.data.edges,
          hatches: cachedEntry.data.hatches,
          bounds: cachedEntry.data.bounds,
        }
      }

      console.log('[useFileUpload] Cache miss, parsing file:', hash)

      // ✅ 优先使用前端解析 (Worker + mlightcad)
      if (useFrontendParse) {
        return new Promise((resolve, reject) => {
          extract(file, {
            onSuccess: (result) => {
              console.log('[useFileUpload] Worker extraction successful:', {
                edgesCount: result.edges.length,
                hatchesCount: result.hatches.length,
                bounds: result.bounds,
              })

              // 如果成功提取数据，返回前端解析结果并缓存
              if (result.edges.length > 0 || result.hatches.length > 0) {
                console.log('[useFileUpload] Frontend parse successful')
                setParseMethod('frontend')

                // ✅ 缓存解析结果
                const cacheData = {
                  edges: result.edges,
                  hatches: result.hatches,
                  bounds: result.bounds,
                }
                setCache(hash, cacheData)

                resolve({
                  parseMethod: 'frontend' as const,
                  ...cacheData,
                })
              } else {
                // 如果提取的数据为空，降级到后端
                console.log('[useFileUpload] Frontend extraction returned no data, falling back to backend')
                setParseMethod('backend')
                // 继续后端解析
                fallbackToBackend(file, profile, resolve, reject)
              }
            },
            onError: (error) => {
              console.warn('[useFileUpload] Worker extraction failed, falling back to backend:', error)
              setParseMethod('backend')
              fallbackToBackend(file, profile, resolve, reject)
            },
            onProgress: (_progress) => {
              setUploadProgress(_progress.progress)
            },
          })
        })
      }

      // ❌ 后端解析 (降级方案)
      return fallbackToBackend(file, profile)
    },

    onSuccess: (data: any) => {
      const method = data.parseMethod === 'frontend' ? '前端' : '后端'
      const edgesCount = data.edges?.length || 0
      const hatchesCount = data.hatches?.length || 0
      toast.success(`${method}解析成功：${edgesCount} 条边，${hatchesCount} 个填充`)

      console.log('[useFileUpload] Received data:', {
        parseMethod: data.parseMethod,
        edgesCount,
        hatchesCount,
        firstHatch: data.hatches?.[0],
      })

      // 更新画布数据
      if (data.edges) {
        setEdges(
          data.edges.map((edge: any) => ({
            ...edge,
            semantic: edge.is_wall ? ('hard_wall' as const) : undefined,
          }))
        )
      }

      if (data.hatches) {
        console.log('[useFileUpload] Setting hatches:', data.hatches.length)
        setHatches(data.hatches as HatchEntity[])
      }

      // 使场景查询失效
      queryClient.invalidateQueries({ queryKey: ['scene'] })

      // 自动适配内容
      if (data.edges && data.edges.length > 0) {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            fitToContent(0.1)
            console.log('[useFileUpload] fitToContent called after double RAF')
          })
        })
      }

      onSuccess?.()
    },

    onError: (error: Error) => {
      toast.error(error.message || '处理失败')
      onError?.(error)
    },

    onSettled: () => {
      setIsUploading(false)
      setUploadProgress(0)
      setLoading(false)
    },
  })

  const uploadFile = useCallback(
    async (file: File) => {
      setLoading(true)
      return mutation.mutateAsync(file)
    },
    [mutation, setLoading]
  )

  return {
    uploadFile,
    isUploading,
    progress: useCanvasStore((state) => state.uploadProgress),
    parseMethod: useCanvasStore((state) => state.parseMethod),  // S012: 从 store 读取
    cancel, // 暴露取消方法
  }
}

/**
 * 降级到后端解析的辅助函数
 */
async function fallbackToBackend(
  file: File,
  profile: 'architectural' | 'mechanical' | 'scanned' | 'quick' | undefined,
  resolve?: (result: any) => void,
  reject?: (error: Error) => void
): Promise<any> {
  try {
    const backendData = await fileService.processFile(file, profile, (_progress) => {
      // 注意：这里不更新 uploadProgress，因为 Worker 已经在更新
    })

    const result = {
      ...backendData,
      parseMethod: 'backend' as const,
    }

    if (resolve) {
      resolve(result)
    }
    
    return result
  } catch (error) {
    if (reject) {
      reject(error as Error)
    }
    throw error
  }
}
