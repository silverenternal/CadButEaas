import { useEffect, useState, useRef, useMemo, useCallback } from 'react'
import { Stage, Layer, Line, Group } from 'react-konva'
import { EdgeLayer } from './edge-layer'
import { SelectionLayer } from './selection-layer'
import { InteractionLayer } from './interaction-layer'
import { HatchLayer } from './hatch-layer'
import { AxisOverlay } from './axis-overlay'
import { PerformanceMonitor } from '@/components/performance-monitor'
import { calculateLodLineWidth } from './edge-layer'
import { useCanvasStore, selectEdgesWithLod, selectPerformanceStats, type Viewport } from '@/stores/canvas-store'

// P1-3 新增：LOD 层级枚举
export enum LodLevel {
  LOD0 = 'lod0',  // 墙体层（最低优先级，始终渲染）
  LOD1 = 'lod1',  // 门窗层（zoom > 0.3 时渲染）
  LOD2 = 'lod2',  // 家具层（zoom > 0.7 时渲染）
  LOD3 = 'lod3',  // 标注层（zoom > 1.5 时渲染）
}

export function CanvasViewer() {
  const {
    edges,
    hatches,
    selectedEdgeId,
    selectedEdgeIds,
    selectedHatchIds,
    camera,
    gaps,
    traceResult,
    selectEdge,
    selectHatch,
    setCamera,
    setViewport,
    setDimensions: setStoreDimensions,
  } = useCanvasStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<any>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  // P1-1 新增：用于 requestAnimationFrame 优化视口更新
  const viewportUpdateRef = useRef<number | null>(null)
  // isDragging 用于 Stage 的 onDragStart/onDragEnd 事件处理
  const [_isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const dims = {
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        }
        setDimensions(dims)
        setStoreDimensions(dims)  // P0-3 新增：同步到 store
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  // 计算视口（用于 LOD 裁剪）
  // P1-1 修复：使用 requestAnimationFrame 优化更新时机
  const updateViewport = useCallback(() => {
    if (!stageRef.current) return

    const stage = stageRef.current
    const width = stage.width()
    const height = stage.height()

    // 计算视口在世界坐标中的范围
    const viewport: Viewport = {
      minX: -camera.offsetX / camera.zoom,
      minY: -camera.offsetY / camera.zoom,
      maxX: (width - camera.offsetX) / camera.zoom,
      maxY: (height - camera.offsetY) / camera.zoom,
    }

    setViewport(viewport)
  }, [camera.offsetX, camera.offsetY, camera.zoom, setViewport])

  // P1-1 新增：使用 requestAnimationFrame 调度视口更新
  const scheduleViewportUpdate = useCallback(() => {
    if (viewportUpdateRef.current) {
      cancelAnimationFrame(viewportUpdateRef.current)
    }
    viewportUpdateRef.current = requestAnimationFrame(() => {
      updateViewport()
      viewportUpdateRef.current = null
    })
  }, [updateViewport])

  // 相机变化时更新视口
  useEffect(() => {
    updateViewport()
  }, [camera, updateViewport])

  // ✅ P0-2 修复：使用 Konva 原生事件绑定滚轮缩放
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return

    const wheelHandler = (e: any) => {
      e.evt.preventDefault()

      // ✅ P0-1 调试：记录滚轮事件
      console.log('[CanvasViewer] Native wheel event', {
        deltaY: e.evt.deltaY,
        zoom: camera.zoom,
        pointer: stage.getPointerPosition(),
      })

      const scaleBy = 1.05
      const oldScale = camera.zoom
      const pointer = stage.getPointerPosition()
      if (!pointer) return

      const mousePointTo = {
        x: (pointer.x - stage.x()) / oldScale,
        y: (pointer.y - stage.y()) / oldScale,
      }

      const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy
      const clampedScale = Math.max(0.1, Math.min(10, newScale))

      setCamera({
        zoom: clampedScale,
        offsetX: pointer.x - mousePointTo.x * clampedScale,
        offsetY: pointer.y - mousePointTo.y * clampedScale,
      })

      // ✅ P1-1 新增：立即更新视口
      scheduleViewportUpdate()
    }

    // 使用原生事件绑定
    stage.on('wheel', wheelHandler)

    return () => {
      stage.off('wheel', wheelHandler)
    }
  }, [camera.zoom, setCamera, scheduleViewportUpdate])

  // P1-3 修复：使用 LOD 分层过滤
  const lodFilteredEdges = useMemo(() => {
    const viewport = useCanvasStore.getState().viewport
    
    return selectEdgesWithLod(useCanvasStore.getState(), viewport)
  }, [edges, camera.zoom, camera.offsetX, camera.offsetY, dimensions])

  // P1-3 新增：按 LOD 层级分组边
  const edgesByLod = useMemo(() => {
    const lod0Edges: typeof edges = []  // 墙体
    const lod1Edges: typeof edges = []  // 门窗
    const lod2Edges: typeof edges = []  // 家具
    const lod3Edges: typeof edges = []  // 标注

    lodFilteredEdges.forEach((edge) => {
      // 选中的边始终渲染在最高层级
      if (selectedEdgeIds.includes(edge.id)) {
        lod0Edges.push(edge)
        return
      }

      // 根据语义和 zoom 级别分配 LOD
      if (edge.is_wall) {
        // LOD0: 墙体始终渲染
        lod0Edges.push(edge)
      } else if (edge.semantic === 'opening' || edge.semantic === 'window' || edge.semantic === 'door') {
        // LOD1: 门窗（zoom > 0.3 时渲染）
        if (camera.zoom > 0.3) {
          lod1Edges.push(edge)
        }
      } else {
        // 使用 layer 字段判断其他类型（因为 BoundarySemantic 没有 furniture/dimension/text）
        const layerLower = edge.layer?.toLowerCase() || ''
        const isFurniture = layerLower.includes('furniture') || layerLower.includes('家具')
        const isDimension = layerLower.includes('dim') || layerLower.includes('标注')
        const isText = layerLower.includes('text') || layerLower.includes('文字')

        if (isFurniture) {
          // LOD2: 家具（zoom > 0.7 时渲染）
          if (camera.zoom > 0.7) {
            lod2Edges.push(edge)
          }
        } else if (isDimension || isText) {
          // LOD3: 标注（zoom > 1.5 时渲染）
          if (camera.zoom > 1.5) {
            lod3Edges.push(edge)
          }
        } else {
          // 其他边根据 zoom 级别渲染
          if (camera.zoom > 0.5) {
            lod2Edges.push(edge)
          }
        }
      }
    })

    return { lod0Edges, lod1Edges, lod2Edges, lod3Edges }
  }, [lodFilteredEdges, camera.zoom, selectedEdgeIds])

  // 性能统计（开发模式下显示）
  const performanceStats = useMemo(
    () => selectPerformanceStats(useCanvasStore.getState(), useCanvasStore.getState().viewport),
    [edges.length, camera.zoom, camera.offsetX, camera.offsetY, dimensions]
  )

  // 开发模式下打印性能统计
  useEffect(() => {
    if ((import.meta as any).env.DEV && (import.meta as any).env.VITE_ENABLE_DEVTOOLS) {
      console.log('[Canvas Performance LOD]', {
        total: edges.length,
        visible: lodFilteredEdges.length,
        lod0: edgesByLod.lod0Edges.length,
        lod1: edgesByLod.lod1Edges.length,
        lod2: edgesByLod.lod2Edges.length,
        lod3: edgesByLod.lod3Edges.length,
      })
    }
  }, [edges.length, lodFilteredEdges.length, edgesByLod, camera.zoom])

  const selectedEdge = edges.find((e) => e.id === selectedEdgeId)

  // P1-2 新增：计算边界框
  const boundingBox = useMemo(() => {
    if (edges.length === 0) return null

    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity

    edges.forEach((edge) => {
      minX = Math.min(minX, edge.start[0], edge.end[0])
      minY = Math.min(minY, edge.start[1], edge.end[1])
      maxX = Math.max(maxX, edge.start[0], edge.end[0])
      maxY = Math.max(maxY, edge.start[1], edge.end[1])
    })

    return { minX, minY, maxX, maxY }
  }, [edges])

  return (
    <div ref={containerRef} className="w-full h-full bg-canvas-background">
      {/*
        ✅ P0-1 修复：Stage 配置优化
        - listening={true}: 确保事件监听启用
        - draggable={false}: 禁用 Konva 默认拖拽，使用自定义相机控制
      */}
      {dimensions.width > 0 && dimensions.height > 0 ? (
        <Stage
          key={`stage-${dimensions.width}-${dimensions.height}`}
          ref={stageRef}
          width={dimensions.width}
          height={dimensions.height}
          onDragStart={() => setIsDragging(true)}
          onDragEnd={() => {
            setIsDragging(false)
            updateViewport()
          }}
          onDragMove={() => {
            if (stageRef.current) {
              const stage = stageRef.current
              setCamera({
                offsetX: stage.x(),
                offsetY: stage.y(),
              })

              // ✅ P1-1 新增：拖动过程中更新视口
              scheduleViewportUpdate()
            }
          }}
          // ✅ P0-1 新增：确保事件捕获
          listening={true}
          // ✅ P0-1 新增：禁用 Konva 默认缩放，使用自定义相机
          scale={{ x: 1, y: 1 }}
        >
          {/* P0-2 新增：HATCH 层（在边之前渲染，作为背景填充） */}
          {hatches.length > 0 && (
            <Layer>
              {/* ✅ P2-4 新增：HATCH 视口裁剪，提升大型图纸性能 */}
              <HatchLayer
                hatches={hatches}
                camera={camera}
                selectedHatchIds={selectedHatchIds}
                onHatchClick={(hatchId: number) => selectHatch(hatchId)}
                canvasWidth={dimensions.width}
                canvasHeight={dimensions.height}
                enableViewportCulling={true}  // 启用视口裁剪
              />
            </Layer>
          )}

          {/* P1-2 新增：边界框（在边之前渲染） */}
          {boundingBox && (
            <Layer>
              <Line
                points={[
                  boundingBox.minX, boundingBox.minY,
                  boundingBox.maxX, boundingBox.minY,
                  boundingBox.maxX, boundingBox.maxY,
                  boundingBox.minX, boundingBox.maxY,
                  boundingBox.minX, boundingBox.minY,
                ]}
                stroke="#ef4444"  // ✅ 红色调试框
                strokeWidth={2 / camera.zoom}  // ✅ 保持固定视觉宽度
                dash={[10, 5]}
                listening={false}
                perfectDrawEnabled={false}
              />
            </Layer>
          )}

          {/* ✅ P0-1 修复：使用 Layer + Group 包裹所有几何内容，应用相机变换 */}
          {/* Layer 必须在 Group 外部，Group 用于应用相机变换 */}
          <Layer>
            <Group
              x={camera.offsetX}
              y={camera.offsetY}
              scaleX={camera.zoom}
              scaleY={camera.zoom}
            >
              {/* LOD0: 墙体层（最低优先级，始终渲染） */}
              <EdgeLayer
                edges={edgesByLod.lod0Edges}
                camera={camera}
                selectedEdgeIds={selectedEdgeIds}
                onEdgeClick={(edgeId) => selectEdge(edgeId)}
              />

              {/* LOD1: 门窗层（zoom > 0.3 时渲染） */}
              {camera.zoom > 0.3 && edgesByLod.lod1Edges.length > 0 && (
                <EdgeLayer
                  edges={edgesByLod.lod1Edges}
                  camera={camera}
                  selectedEdgeIds={selectedEdgeIds}
                  onEdgeClick={(edgeId) => selectEdge(edgeId)}
                />
              )}

              {/* LOD2: 家具层（zoom > 0.7 时渲染） */}
              {camera.zoom > 0.7 && edgesByLod.lod2Edges.length > 0 && (
                <EdgeLayer
                  edges={edgesByLod.lod2Edges}
                  camera={camera}
                  selectedEdgeIds={selectedEdgeIds}
                  onEdgeClick={(edgeId) => selectEdge(edgeId)}
                />
              )}

              {/* LOD3: 标注层（zoom > 1.5 时渲染） */}
              {camera.zoom > 1.5 && edgesByLod.lod3Edges.length > 0 && (
                <EdgeLayer
                  edges={edgesByLod.lod3Edges}
                  camera={camera}
                  selectedEdgeIds={selectedEdgeIds}
                  onEdgeClick={(edgeId) => selectEdge(edgeId)}
                />
              )}
            </Group>
          </Layer>

          {/* 选择高亮图层 - 不受相机变换影响 */}
          <Layer>
            <SelectionLayer
              selectedEdge={selectedEdge || null}
              selectedEdgeIds={selectedEdgeIds}
              traceResult={traceResult}
            />
          </Layer>

          {/* 交互图层（缺口标记等）- 不受相机变换影响 */}
          <Layer>
            <InteractionLayer gaps={gaps} />
          </Layer>

          {/* P1-1 新增：坐标轴覆盖层（最顶层，不受相机变换影响） */}
          <Layer>
            <AxisOverlay
              camera={camera}
              showOrigin={true}
              canvasHeight={dimensions.height}  // ✅ P0-1 修复：传入画布高度
              canvasWidth={dimensions.width}    // ✅ P0-1 新增：传入画布宽度
            />
          </Layer>
        </Stage>
      ) : null}

      {/* 性能统计覆盖层（开发模式）+ P2-3 性能监控组件 */}
      {(import.meta as any).env.DEV && (import.meta as any).env.VITE_ENABLE_DEVTOOLS && (
        <>
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono pointer-events-none">
            <div>Edges: {performanceStats.visibleEdges}/{performanceStats.totalEdges}</div>
            <div>LOD: {performanceStats.lodLevel}</div>
            <div>LOD0: {edgesByLod.lod0Edges.length} | LOD1: {edgesByLod.lod1Edges.length} | LOD2: {edgesByLod.lod2Edges.length} | LOD3: {edgesByLod.lod3Edges.length}</div>
            <div>Est. Render: {performanceStats.estimatedRenderTime.toFixed(2)}ms</div>
            {/* ✅ P1-2 新增：线宽调试 */}
            <div className="mt-1 border-t border-gray-600 pt-1">
              <div>Wall Width: {calculateLodLineWidth(camera.zoom, undefined, true).toFixed(2)}px</div>
              <div>Door Width: {calculateLodLineWidth(camera.zoom, 'door', false).toFixed(2)}px</div>
              <div>Dim Width: {calculateLodLineWidth(camera.zoom, 'dimension', false).toFixed(2)}px</div>
            </div>
          </div>
          
          {/* P2-3 新增：性能监控组件（右上角） */}
          <PerformanceMonitor
            enabled
            refreshInterval={1000}
          />
        </>
      )}
    </div>
  )
}
