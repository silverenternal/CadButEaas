import { useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from 'react'
import type { AcTrView2d, AcApDocument } from '@mlightcad/cad-simple-viewer'
import { extractGeometryFromDocument } from '@/lib/mlightcad-geometry-extractor'

export interface CadViewerProps {
  file?: File | string | null
  initialCamera?: {
    zoom: number
    offsetX: number
    offsetY: number
  }
  options?: {
    showGrid?: boolean
    showAxes?: boolean
    backgroundColor?: string
    enablePan?: boolean
    enableZoom?: boolean
    enableRotate?: boolean
  }
  onLoaded?: (data: CadViewerData) => void
  onError?: (error: Error) => void
  onCameraChange?: (camera: CameraState) => void
  onEntityClick?: (entityId: string, entityType: string) => void
  className?: string
  'data-testid'?: string
}

export interface CadViewerData {
  entities: any[]
  layers: string[]
  bounds: {
    minX: number
    minY: number
    maxX: number
    maxY: number
  }
  // 新增：提取的几何数据（前端解析）
  extractedEdges?: any[]
  extractedHatches?: any[]
}

export interface CameraState {
  zoom: number
  offsetX: number
  offsetY: number
}

export interface CadViewerRef {
  fitToContent: (padding?: number) => void
  resetView: () => void
  getViewer: () => AcTrView2d | null
  getDocument: () => AcApDocument | null
}

export const CadViewer = forwardRef<CadViewerRef, CadViewerProps>(function CadViewer(
  {
    file,
    options = {},
    onLoaded,
    onError,
    className,
    'data-testid': dataTestId,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<AcTrView2d | null>(null)
  const documentRef = useRef<AcApDocument | null>(null)
  const errorRef = useRef<Error | null>(null)

  // 默认配置
  const defaultOptions = {
    showGrid: true,
    showAxes: true,
    backgroundColor: '#1a1a2e',
    enablePan: true,
    enableZoom: true,
    enableRotate: false,
  }

  const mergedOptions = { ...defaultOptions, ...options }

  // 初始化查看器
  useEffect(() => {
    if (!containerRef.current) return

    const initViewer = async () => {
      try {
        const { AcTrView2d: View2d } = await import('@mlightcad/cad-simple-viewer')
        
        viewerRef.current = new View2d({
          container: containerRef.current || undefined,
          background: parseInt(mergedOptions.backgroundColor?.replace('#', '0x') || '0x1a1a2e'),
        })

        // 强制重新渲染
        viewerRef.current.isDirty = true
      } catch (err) {
        errorRef.current = err as Error
        onError?.(err as Error)
      }
    }

    initViewer()

    // 清理函数
    return () => {
      if (viewerRef.current) {
        viewerRef.current = null
      }
    }
  }, [])

  // 加载文件
  useEffect(() => {
    if (!file || !viewerRef.current) return

    const loadFile = async () => {
      try {
        errorRef.current = null

        const { AcApDocument: ApDocument, AcEdOpenMode: OpenMode } = await import('@mlightcad/cad-simple-viewer')

        const doc = new ApDocument()

        if (typeof file === 'string') {
          const success = await doc.openUri(file, { mode: OpenMode.Read })
          if (!success) throw new Error('Failed to load file from URL')
        } else {
          const content = await file.arrayBuffer()
          const success = await doc.openDocument(file.name, content, { mode: OpenMode.Read })
          if (!success) throw new Error('Failed to parse file')
        }

        documentRef.current = doc

        // 获取数据库并加载到查看器
        if (doc.database && viewerRef.current) {
          // mlightcad 需要将 database 加载到 viewer 的场景中
          // 使用 loadDatabase 或类似方法
          const anyViewer = viewerRef.current as any
          
          // 尝试访问 internalScene
          if (anyViewer.internalScene && anyViewer.loadDatabase) {
            await anyViewer.loadDatabase(doc.database)
          }
          
          // 使用几何数据提取器提取数据
          try {
            // 类型转换为我们的类型定义
            const geometryData = await extractGeometryFromDocument(doc as any)
            
            const sceneData = {
              entities: [],
              layers: Array.from(new Set(geometryData.edges.map(e => e.layer).filter((l): l is string => Boolean(l)))),
              bounds: geometryData.bounds,
              // 新增：提取的几何数据
              extractedEdges: geometryData.edges,
              extractedHatches: geometryData.hatches,
            }

            onLoaded?.(sceneData)
          } catch (extractError) {
            console.error('[CadViewer] Geometry extraction error:', extractError)
            // 即使提取失败，也返回基本数据
            const sceneData = {
              entities: [],
              layers: [],
              bounds: {
                minX: 0,
                minY: 0,
                maxX: 100,
                maxY: 100,
              },
            }
            onLoaded?.(sceneData)
          }
        }
      } catch (e) {
        const err = e as Error
        errorRef.current = err
        onError?.(err)
      }
    }

    loadFile()
  }, [file])

  // 适配内容
  const fitToContent = useCallback((_padding = 0.1) => {
    if (viewerRef.current) {
      viewerRef.current.zoomToFitDrawing(1000)
    }
  }, [])

  // 重置视图
  const resetView = useCallback(() => {
    // 重置视图逻辑
  }, [])

  // 导出方法
  useImperativeHandle(ref, () => ({
    fitToContent,
    resetView,
    getViewer: () => viewerRef.current,
    getDocument: () => documentRef.current,
  }), [fitToContent, resetView])

  return (
    <div ref={containerRef} className={className || 'w-full h-full'} data-testid={dataTestId}>
      {errorRef.current && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-xl max-w-md">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              加载失败
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              {errorRef.current.message}
            </p>
            <button
              onClick={() => {
                errorRef.current = null
                window.location.reload()
              }}
              className="mt-4 px-4 py-2 bg-primary text-white rounded hover:bg-primary/90"
            >
              关闭
            </button>
          </div>
        </div>
      )}
    </div>
  )
})

export default CadViewer
