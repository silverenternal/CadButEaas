import { useRef, useMemo, useCallback, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Stats } from '@react-three/drei'
import type { OrthographicCamera } from 'three'
import type { Edge, HatchEntity } from '@/types/api'
import { EdgeGroup } from './three-edge-group'
import { HatchGroup } from './three-hatch-group'
import { PerformanceMonitor } from '@/components/performance-monitor'

/**
 * ⚠️ CRITICAL: 此组件绝对不能导入 @mlightcad/cad-simple-viewer
 * 
 * 渲染必须完全使用纯 Three.js 和传入的 edges/hatches 数据
 * 任何 mlightcad 的导入都会导致 workingDatabase 错误：
 * "Error: The current working database must be set before using it!"
 * 
 * 数据流：解析器 → 纯数据 (edges/hatches) → 渲染器 (此组件)
 * 
 * 修复参考：FRONTEND_RENDER_FIX.json
 */

/**
 * 自定义坐标轴辅助组件
 */
function CustomAxesHelper({ size = 100 }: { size?: number }) {
  return (
    <group>
      {/* X 轴 - 红色 */}
      <mesh position={[size / 2, 0, 0]}>
        <boxGeometry args={[size, 0.5, 0.5]} />
        <meshBasicMaterial color="#ff0000" />
      </mesh>
      {/* Y 轴 - 绿色 */}
      <mesh position={[0, size / 2, 0]}>
        <boxGeometry args={[0.5, size, 0.5]} />
        <meshBasicMaterial color="#00ff00" />
      </mesh>
      {/* Z 轴 - 蓝色 */}
      <mesh position={[0, 0, size / 2]}>
        <boxGeometry args={[0.5, 0.5, size]} />
        <meshBasicMaterial color="#0000ff" />
      </mesh>
    </group>
  )
}

interface ThreeViewerProps {
  edges: Edge[]
  hatches: HatchEntity[]
  selectedEdgeIds: number[]
  showGrid?: boolean
  showAxes?: boolean
  showStats?: boolean
  onEdgeClick?: (edgeId: number) => void
}

/**
 * Three.js DXF 查看器组件
 * 
 * 功能:
 * - 使用 Three.js WebGL 渲染 DXF 边和 HATCH 填充
 * - 支持轨道控制器（缩放/平移/旋转）
 * - 坐标轴和网格辅助
 * - 性能统计
 */
function ThreeViewer({
  edges,
  hatches,
  selectedEdgeIds,
  showGrid = true,
  showAxes = true,
  showStats = false,
  onEdgeClick,
}: ThreeViewerProps) {
  // ✅ S018: 性能监控状态
  const [performanceStats, setPerformanceStats] = useState({
    drawCalls: 0,
    triangleCount: 0,
    lineCount: edges.length,
    edgeCount: edges.length,
    hatchCount: hatches.length,
  })

  // ✅ S018: 使用 useFrame 收集渲染统计
  useFrame(() => {
    // 统计几何体数量
    setPerformanceStats(prev => ({
      ...prev,
      lineCount: edges.length,
      edgeCount: edges.length,
      hatchCount: hatches.length,
      // Draw Call 估算：每个材质组一次 draw call
      drawCalls: Math.ceil(edges.length / 1000) + Math.ceil(hatches.length / 100),
    }))
  })
  // 计算场景边界，用于自动适配相机
  const sceneBounds = useMemo(() => {
    if (edges.length === 0 && hatches.length === 0) {
      return { min: { x: -100, y: -100 }, max: { x: 100, y: 100 } }
    }

    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity

    // 计算边的边界
    edges.forEach((edge) => {
      minX = Math.min(minX, edge.start[0], edge.end[0])
      minY = Math.min(minY, edge.start[1], edge.end[1])
      maxX = Math.max(maxX, edge.start[0], edge.end[0])
      maxY = Math.max(maxY, edge.start[1], edge.end[1])
    })

    // 计算 HATCH 的边界
    hatches.forEach((hatch) => {
      hatch.boundary_paths.forEach((path) => {
        if (path.points) {
          path.points.forEach((point) => {
            minX = Math.min(minX, point[0])
            minY = Math.min(minY, point[1])
            maxX = Math.max(maxX, point[0])
            maxY = Math.max(maxY, point[1])
          })
        }
        if (path.control_points) {
          path.control_points.forEach((point) => {
            minX = Math.min(minX, point[0])
            minY = Math.min(minY, point[1])
            maxX = Math.max(maxX, point[0])
            maxY = Math.max(maxY, point[1])
          })
        }
        if (path.center) {
          minX = Math.min(minX, path.center[0])
          minY = Math.min(minY, path.center[1])
          maxX = Math.max(maxX, path.center[0])
          maxY = Math.max(maxY, path.center[1])
        }
        // 圆弧边界
        if (path.type === 'arc' && path.center && path.radius) {
          minX = Math.min(minX, path.center[0] - path.radius)
          minY = Math.min(minY, path.center[1] - path.radius)
          maxX = Math.max(maxX, path.center[0] + path.radius)
          maxY = Math.max(maxY, path.center[1] + path.radius)
        }
        // 椭圆弧边界
        if (path.type === 'ellipse_arc' && path.center && path.major_axis) {
          const semiMajor = Math.sqrt(path.major_axis[0] ** 2 + path.major_axis[1] ** 2)
          const semiMinor = semiMajor * (path.minor_axis_ratio ?? 1.0)
          minX = Math.min(minX, path.center[0] - semiMajor)
          maxX = Math.max(maxX, path.center[0] + semiMajor)
          minY = Math.min(minY, path.center[1] - semiMinor)
          maxY = Math.max(maxY, path.center[1] + semiMinor)
        }
      })
    })

    // 避免无效边界
    if (minX === Infinity || maxX === -Infinity || minY === Infinity || maxY === -Infinity) {
      return { min: { x: -100, y: -100 }, max: { x: 100, y: 100 } }
    }

    return { min: { x: minX, y: minY }, max: { x: maxX, y: maxY } }
  }, [edges, hatches])

  return (
    <div className="w-full h-full bg-[#1a1a2e] relative">
      {/* 空状态提示 */}
      {edges.length === 0 && hatches.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
          <div className="text-white/50 text-lg">暂无数据，请上传 DXF 文件</div>
        </div>
      )}
      <Canvas
        camera={{
          position: [0, 0, 500],
          near: 0.1,
          far: 100000,
          zoom: 1,
        }}
        orthographic
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          preserveDrawingBuffer: true,
        }}
        className="w-full h-full"
      >
        {/* 场景配置 */}
        <color attach="background" args={['#1a1a2e']} />

        {/* 性能统计（开发模式） */}
        {showStats && <Stats />}

        {/* ✅ S018: 性能监控面板 */}
        <PerformanceMonitor
          enabled
          refreshInterval={1000}
          customStats={performanceStats}
        />

        {/* 相机控制器 */}
        <OrbitControls
          enableRotate={false}  // 禁用旋转，保持 2D 视图
          enableZoom={true}
          enablePan={true}
          zoomSpeed={0.5}
          panSpeed={0.5}
          minZoom={0.01}
          maxZoom={100}
          makeDefault
        />

        {/* 自动适配场景 */}
        <SceneFitter bounds={sceneBounds} />

        {/* 坐标轴辅助 - 自定义组件 */}
        {showAxes && <CustomAxesHelper size={100} />}

        {/* 网格辅助 */}
        {showGrid && (
          <Grid
            position={[0, 0, -0.1]}
            args={[
              Math.max(sceneBounds.max.x - sceneBounds.min.x, sceneBounds.max.y - sceneBounds.min.y) * 2,
              Math.max(sceneBounds.max.x - sceneBounds.min.x, sceneBounds.max.y - sceneBounds.min.y) * 2,
            ]}
            cellColor={0x444444}
            sectionColor={0x666666}
            cellSize={10}
            sectionSize={100}
            fadeDistance={1000}
            fadeStrength={1}
          />
        )}

        {/* 环境光 */}
        <ambientLight intensity={1} />

        {/* 渲染 DXF 边 */}
        <EdgeGroup
          edges={edges}
          selectedEdgeIds={selectedEdgeIds}
          onEdgeClick={onEdgeClick}
        />

        {/* 渲染 HATCH 填充 */}
        <HatchGroup hatches={hatches} />
      </Canvas>
    </div>
  )
}

/**
 * 场景适配组件
 * 根据场景边界自动调整相机位置
 */
function SceneFitter({ bounds }: { bounds: { min: { x: number; y: number }; max: { x: number; y: number } } }) {
  const { camera, size } = useThree()
  const previousRef = useRef<{ width: number; height: number; bounds: typeof bounds } | null>(null)

  useFrame(() => {
    // 避免重复计算
    if (
      previousRef.current &&
      Math.abs(previousRef.current.width - size.width) < 1 &&
      Math.abs(previousRef.current.height - size.height) < 1 &&
      previousRef.current.bounds === bounds
    ) {
      return
    }

    previousRef.current = { width: size.width, height: size.height, bounds }

    // 计算场景中心和大小
    const centerX = (bounds.min.x + bounds.max.x) / 2
    const centerY = (bounds.min.y + bounds.max.y) / 2
    const sceneWidth = bounds.max.x - bounds.min.x
    const sceneHeight = bounds.max.y - bounds.min.y

    console.log('[SceneFitter] Adjusting camera:', {
      sceneWidth,
      sceneHeight,
      canvasWidth: size.width,
      canvasHeight: size.height,
    })

    // 避免除以零
    if (sceneWidth <= 0 || sceneHeight <= 0 || size.width <= 0 || size.height <= 0) {
      console.warn('[SceneFitter] Invalid bounds or size, skipping adjustment')
      return
    }

    // 计算合适的缩放级别（保留 10% padding）
    const padding = 0.9
    const scaleX = size.width / sceneWidth
    const scaleY = size.height / sceneHeight
    const zoom = Math.min(scaleX, scaleY) * padding

    // 更新正交相机
    const orthoCamera = camera as OrthographicCamera
    const aspect = size.width / size.height
    const frustumSize = Math.max(sceneWidth, sceneHeight) / 2 / zoom

    orthoCamera.left = -frustumSize * aspect
    orthoCamera.right = frustumSize * aspect
    orthoCamera.top = frustumSize
    orthoCamera.bottom = -frustumSize
    orthoCamera.zoom = zoom

    // 设置相机位置（看向场景中心）
    camera.position.set(centerX, centerY, 500)
    camera.lookAt(centerX, centerY, 0)

    camera.updateProjectionMatrix()
  })

  return null
}

/**
 * 主导出组件
 */
interface ThreeCanvasViewerProps {
  edges: Edge[]
  hatches: HatchEntity[]
  selectedEdgeIds: number[]
  onEdgeClick?: (edgeId: number) => void
  showGrid?: boolean
  showAxes?: boolean
  showStats?: boolean
}

export function ThreeCanvasViewer({
  edges,
  hatches,
  selectedEdgeIds,
  onEdgeClick,
  showGrid = true,
  showAxes = true,
  showStats = false,
}: ThreeCanvasViewerProps) {
  const handleEdgeClick = useCallback(
    (edgeId: number) => {
      console.log('[ThreeViewer] Edge clicked:', edgeId)
      onEdgeClick?.(edgeId)
    },
    [onEdgeClick]
  )

  return (
    <ThreeViewer
      edges={edges}
      hatches={hatches}
      selectedEdgeIds={selectedEdgeIds}
      showGrid={showGrid}
      showAxes={showAxes}
      showStats={showStats}
      onEdgeClick={handleEdgeClick}
    />
  )
}

export default ThreeCanvasViewer
