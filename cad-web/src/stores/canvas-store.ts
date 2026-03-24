import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type { Edge, CameraState, Point, GapInfo, HatchEntity, HatchBoundaryPath, HatchPattern } from '@/types/api'  // P0-4 修复：从 api.ts 导入统一类型

// P0-4 修复：移除本地 HatchEntity 等类型定义，使用 api.ts 中的类型
// 重新导出 HatchEntity 等类型，方便其他模块使用
export type { HatchEntity, HatchBoundaryPath, HatchPattern }

// LOD 级别定义
export enum LodLevel {
  Low = 'low',       // 缩放 < 0.3: 仅渲染墙边，简化样式
  Medium = 'medium', // 缩放 0.3-0.7: 渲染所有边，简化样式
  High = 'high',     // 缩放 > 0.7: 完整渲染
}

// 视口裁剪配置
export interface Viewport {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

interface CanvasState {
  // 画布数据
  edges: Edge[]
  hatches: HatchEntity[]  // P0-2 新增：HATCH 实体
  gaps: GapInfo[]
  traceResult: Point[] | null

  // 选择状态
  selectedEdgeId: number | null
  selectedEdgeIds: number[]
  selectedHatchIds: number[]  // P0-2 新增：HATCH 选择

  // 相机状态
  camera: CameraState

  // 画布尺寸（用于 fitToContent）
  dimensions: { width: number; height: number }

  // 工具状态
  activeTool: 'select' | 'trace' | 'lasso' | 'pan'

  // 加载状态
  isLoading: boolean
  uploadProgress: number
  parseMethod: 'frontend' | 'backend' | null  // S012: 解析方法状态移入 store

  // 视口（用于 LOD 裁剪）
  viewport: Viewport | null

  // 性能配置
  maxEdgesRender: number
  lodThreshold: number

  // Actions
  setEdges: (edges: Edge[]) => void
  setHatches: (hatches: HatchEntity[]) => void  // P0-2 新增
  setGaps: (gaps: GapInfo[]) => void
  setTraceResult: (points: Point[] | null) => void
  selectEdge: (edgeId: number | null, multi?: boolean) => void
  selectHatch: (hatchId: number | null, multi?: boolean) => void  // P0-2 新增
  clearSelection: () => void
  setCamera: (camera: Partial<CameraState>) => void
  resetCamera: () => void
  setTool: (tool: 'select' | 'trace' | 'lasso' | 'pan') => void
  setLoading: (loading: boolean) => void
  setUploadProgress: (progress: number) => void
  setParseMethod: (method: 'frontend' | 'backend' | null) => void  // S012: 新增
  clearCanvas: () => void
  setViewport: (viewport: Viewport | null) => void
  fitToContent: (padding?: number) => void
  setDimensions: (dimensions: { width: number; height: number }) => void  // P0-3 新增
}

const defaultCamera: CameraState = {
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
}

// 获取 LOD 级别
export const getLodLevel = (zoom: number): LodLevel => {
  if (zoom < 0.3) return LodLevel.Low
  if (zoom < 0.7) return LodLevel.Medium
  return LodLevel.High
}

// 计算边的边界框
const getEdgeBounds = (edge: Edge) => ({
  minX: Math.min(edge.start[0], edge.end[0]),
  minY: Math.min(edge.start[1], edge.end[1]),
  maxX: Math.max(edge.start[0], edge.end[0]),
  maxY: Math.max(edge.start[1], edge.end[1]),
})

// 检查边是否在视口内
const isEdgeInViewport = (edge: Edge, viewport: Viewport): boolean => {
  const bounds = getEdgeBounds(edge)
  return !(
    bounds.maxX < viewport.minX ||
    bounds.minX > viewport.maxX ||
    bounds.maxY < viewport.minY ||
    bounds.minY > viewport.maxY
  )
}

// 计算边的长度
const getEdgeLength = (edge: Edge): number => {
  const dx = edge.end[0] - edge.start[0]
  const dy = edge.end[1] - edge.start[1]
  return Math.sqrt(dx * dx + dy * dy)
}

export const useCanvasStore = create<CanvasState>()(
  subscribeWithSelector((set, get) => ({
    edges: [],
    hatches: [],  // P0-2 新增
    gaps: [],
    traceResult: null,
    selectedEdgeId: null,
    selectedEdgeIds: [],
    selectedHatchIds: [],  // P0-2 新增
    camera: defaultCamera,
    dimensions: { width: 800, height: 600 },  // P0-3 新增：默认尺寸
    activeTool: 'select',
    isLoading: false,
    uploadProgress: 0,
    parseMethod: null,  // S012: 初始化解析方法状态
    maxEdgesRender: (import.meta as any).env.VITE_MAX_EDGES_RENDER || 10000,
    lodThreshold: (import.meta as any).env.VITE_LOD_THRESHOLD || 0.5,
    viewport: null as Viewport | null,

    setEdges: (edges: Edge[]) => set({ edges }),

    setHatches: (hatches: HatchEntity[]) => set({ hatches }),  // P0-2 新增

    setGaps: (gaps: GapInfo[]) => set({ gaps }),

    setTraceResult: (points: Point[] | null) => set({ traceResult: points }),

    selectEdge: (edgeId: number | null, multi = false) => {
      if (edgeId === null) {
        set({ selectedEdgeId: null, selectedEdgeIds: [] })
        return
      }

      if (multi) {
        const currentIds = get().selectedEdgeIds
        const newIds = currentIds.includes(edgeId)
          ? currentIds.filter((id) => id !== edgeId)
          : [...currentIds, edgeId]
        set({ selectedEdgeId: edgeId, selectedEdgeIds: newIds })
      } else {
        set({ selectedEdgeId: edgeId, selectedEdgeIds: [edgeId] })
      }
    },

    selectHatch: (hatchId: number | null, multi = false) => {  // P0-2 新增
      if (hatchId === null) {
        set({ selectedHatchIds: [] })
        return
      }

      if (multi) {
        const currentIds = get().selectedHatchIds
        const newIds = currentIds.includes(hatchId)
          ? currentIds.filter((id) => id !== hatchId)
          : [...currentIds, hatchId]
        set({ selectedHatchIds: newIds })
      } else {
        set({ selectedHatchIds: [hatchId] })
      }
    },

    clearSelection: () => set({
      selectedEdgeId: null,
      selectedEdgeIds: [],
      selectedHatchIds: [],  // P0-2 新增
    }),

    setCamera: (camera: Partial<CameraState>) =>
      set((state) => ({
        camera: { ...state.camera, ...camera },
      })),

    resetCamera: () => set({ camera: defaultCamera }),

    setTool: (tool: 'select' | 'trace' | 'lasso' | 'pan') =>
      set({ activeTool: tool }),

    setLoading: (loading: boolean) => set({ isLoading: loading }),

    setUploadProgress: (progress: number) => set({ uploadProgress: progress }),

    setParseMethod: (method: 'frontend' | 'backend' | null) => set({ parseMethod: method }),  // S012: 新增

    setViewport: (viewport: Viewport | null) =>
      set({ viewport }),

    // P0-3 新增：自动适配内容
    fitToContent: (padding: number = 0.1) => {
      const state = get()
      const { edges, hatches, dimensions } = state

      if (edges.length === 0 && hatches.length === 0) return

      // P1-1 修复：综合 edges 和 hatches 计算边界框
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity

      // 计算 edges 边界
      edges.forEach((edge) => {
        minX = Math.min(minX, edge.start[0], edge.end[0])
        minY = Math.min(minY, edge.start[1], edge.end[1])
        maxX = Math.max(maxX, edge.start[0], edge.end[0])
        maxY = Math.max(maxY, edge.start[1], edge.end[1])
      })

      // P1-1 新增：计算 hatches 边界
      hatches.forEach((hatch) => {
        hatch.boundary_paths.forEach((path) => {
          if (path.type === 'polyline' && path.points) {
            path.points.forEach((point) => {
              minX = Math.min(minX, point[0])
              minY = Math.min(minY, point[1])
              maxX = Math.max(maxX, point[0])
              maxY = Math.max(maxY, point[1])
            })
          } else if (path.type === 'arc' && path.center && path.radius) {
            // 圆弧边界：考虑圆心和半径
            minX = Math.min(minX, path.center[0] - path.radius)
            minY = Math.min(minY, path.center[1] - path.radius)
            maxX = Math.max(maxX, path.center[0] + path.radius)
            maxY = Math.max(maxY, path.center[1] + path.radius)
          } else if (path.type === 'ellipse_arc' && path.center && path.major_axis) {
            // ✅ P2-NEW-37: 椭圆弧边界：考虑长轴和短轴
            const semiMajor = Math.sqrt(path.major_axis[0] ** 2 + path.major_axis[1] ** 2)
            const semiMinor = semiMajor * (path.minor_axis_ratio ?? 1.0)
            minX = Math.min(minX, path.center[0] - semiMajor)
            maxX = Math.max(maxX, path.center[0] + semiMajor)
            minY = Math.min(minY, path.center[1] - semiMinor)
            maxY = Math.max(maxY, path.center[1] + semiMinor)
          } else if (path.type === 'spline' && path.control_points) {
            // ✅ P2-NEW-37: 样条边界：使用控制点边界框近似
            path.control_points.forEach((point) => {
              minX = Math.min(minX, point[0])
              minY = Math.min(minY, point[1])
              maxX = Math.max(maxX, point[0])
              maxY = Math.max(maxY, point[1])
            })
          }
        })
      })

      // 避免除以零
      if (minX === Infinity || maxX === -Infinity || minY === Infinity || maxY === -Infinity) {
        return
      }

      const contentWidth = maxX - minX
      const contentHeight = maxY - minY

      // 避免除以零
      if (contentWidth === 0 || contentHeight === 0) return

      // 计算合适的缩放（保留 padding）
      const scaleX = dimensions.width / contentWidth
      const scaleY = dimensions.height / contentHeight
      const zoom = Math.min(scaleX, scaleY) * (1 - padding)

      // 计算偏移（居中内容）
      const offsetX = (dimensions.width - contentWidth * zoom) / 2 - minX * zoom
      const offsetY = (dimensions.height - contentHeight * zoom) / 2 - minY * zoom

      set({ camera: { zoom, offsetX, offsetY } })
    },

    clearCanvas: () =>
      set({
        edges: [],
        hatches: [],  // P0-2 新增
        gaps: [],
        traceResult: null,
        selectedEdgeId: null,
        selectedEdgeIds: [],
        selectedHatchIds: [],  // P0-2 新增
        camera: defaultCamera,
        viewport: null,
        // 注意：不清空 dimensions，保持画布尺寸
      }),

    setDimensions: (dimensions: { width: number; height: number }) =>
      set({ dimensions }),
  }))
)

// ========== 优化的选择器 ==========

// 获取选中的边
export const selectSelectedEdge = (state: CanvasState) =>
  state.edges.find((e) => e.id === state.selectedEdgeId)

// 检查是否有选择
export const selectHasSelection = (state: CanvasState) =>
  state.selectedEdgeId !== null

// LOD 优化的边过滤
export const selectEdgesWithLod = (
  state: CanvasState,
  viewport?: Viewport | null
): Edge[] => {
  const { camera, edges, maxEdgesRender } = state
  const lodLevel = getLodLevel(camera.zoom)

  let filtered = edges

  // 1. 视口裁剪
  if (viewport) {
    filtered = filtered.filter((edge) => isEdgeInViewport(edge, viewport))
  }

  // 2. LOD 过滤
  if (lodLevel === LodLevel.Low) {
    // 仅渲染墙边和选中的边
    const selectedEdge = edges.find((e) => e.id === state.selectedEdgeId)
    filtered = filtered.filter((e) => e.is_wall || e.id === state.selectedEdgeId)
    if (selectedEdge && !filtered.includes(selectedEdge)) {
      filtered.push(selectedEdge)
    }
  } else if (lodLevel === LodLevel.Medium) {
    // 渲染所有边，但跳过非常短的边
    const minLength = 100 / camera.zoom // 动态阈值
    filtered = filtered.filter((e) => getEdgeLength(e) > minLength)
  }

  // 3. 数量限制（防止过多边渲染）
  if (filtered.length > maxEdgesRender) {
    // 优先保留墙边和选中的边
    const wallEdges = filtered.filter((e) => e.is_wall)
    const selectedEdge = filtered.find((e) => e.id === state.selectedEdgeId)
    
    filtered = wallEdges.slice(0, maxEdgesRender - 1)
    if (selectedEdge && !filtered.includes(selectedEdge)) {
      filtered.push(selectedEdge)
    }
  }

  return filtered
}

// 根据样式分组边（用于批处理渲染）
export interface EdgeGroup {
  isWall: boolean
  isSelected: boolean
  hasSemantic: boolean
  semanticType?: string
}

export const selectEdgesByGroup = (
  state: CanvasState,
  edges: Edge[]
): Map<string, Edge[]> => {
  const groups = new Map<string, Edge[]>()
  const selectedEdgeIds = new Set(state.selectedEdgeIds)

  edges.forEach((edge) => {
    const key = `${edge.is_wall}-${selectedEdgeIds.has(edge.id)}-${edge.semantic || 'none'}`
    const group = groups.get(key) || []
    group.push(edge)
    groups.set(key, group)
  })

  return groups
}

// 获取当前 LOD 级别
export const selectCurrentLodLevel = (state: CanvasState): LodLevel =>
  getLodLevel(state.camera.zoom)

// 获取性能统计
export interface PerformanceStats {
  totalEdges: number
  visibleEdges: number
  lodLevel: LodLevel
  estimatedRenderTime: number // 毫秒
}

export const selectPerformanceStats = (
  state: CanvasState,
  viewport?: Viewport | null
): PerformanceStats => {
  const visibleEdges = selectEdgesWithLod(state, viewport)
  const lodLevel = getLodLevel(state.camera.zoom)
  
  // 估算渲染时间（基于经验公式）
  const baseTime = 0.001 // 每条边的基础渲染时间（ms）
  const lodMultiplier = lodLevel === LodLevel.Low ? 0.3 : lodLevel === LodLevel.Medium ? 0.6 : 1
  const estimatedRenderTime = visibleEdges.length * baseTime * lodMultiplier

  return {
    totalEdges: state.edges.length,
    visibleEdges: visibleEdges.length,
    lodLevel,
    estimatedRenderTime,
  }
}
