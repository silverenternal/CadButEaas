import { useMemo, useRef, useCallback, useEffect } from 'react'
import * as THREE from 'three'
import type { Edge } from '@/types/api'

interface EdgeGroupProps {
  edges: Edge[]
  selectedEdgeIds: number[]
  onEdgeClick?: (edgeId: number) => void
}

/**
 * Three.js 边渲染组
 * ✅ S003: 使用合并几何体减少 Draw Call
 */
export function EdgeGroup({ edges, selectedEdgeIds, onEdgeClick }: EdgeGroupProps) {
  const selectedEdgeIdSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds])
  const edgeIdsRef = useRef<Map<string, number[]>>(new Map())

  // 分组边：普通边和选中的边
  const { normalEdges, selectedEdges, wallEdges, semanticEdges } = useMemo(() => {
    const normal: Edge[] = []
    const selected: Edge[] = []
    const walls: Edge[] = []
    const semantics: Map<string, Edge[]> = new Map()

    edges.forEach((edge) => {
      if (selectedEdgeIdSet.has(edge.id)) {
        selected.push(edge)
      } else if (edge.is_wall) {
        walls.push(edge)
      } else if (edge.semantic) {
        const group = semantics.get(edge.semantic) || []
        group.push(edge)
        semantics.set(edge.semantic, group)
      } else {
        normal.push(edge)
      }
    })

    return {
      normalEdges: normal,
      selectedEdges: selected,
      wallEdges: walls,
      semanticEdges: Array.from(semantics.entries()).map(([type, edges]) => ({ type, edges })),
    }
  }, [edges, selectedEdgeIdSet])

  const handleEdgeClick = useCallback(
    (edgeId: number) => {
      console.log('[ThreeViewer] Edge clicked:', edgeId)
      onEdgeClick?.(edgeId)
    },
    [onEdgeClick]
  )

  // ✅ 清理几何体缓存
  useEffect(() => {
    return () => {
      edgeIdsRef.current.clear()
    }
  }, [])

  return (
    <group name="edges">
      {/* 墙体边 - 蓝色 */}
      {wallEdges.length > 0 && (
        <EdgeBatch
          edges={wallEdges}
          color="#60a5fa"
          strokeWidth={2}
          onEdgeClick={handleEdgeClick}
          edgeIdsRef={edgeIdsRef}
        />
      )}

      {/* 普通边 - 灰色 */}
      {normalEdges.length > 0 && (
        <EdgeBatch
          edges={normalEdges}
          color="#94a3b8"
          strokeWidth={1.5}
          onEdgeClick={handleEdgeClick}
          edgeIdsRef={edgeIdsRef}
        />
      )}

      {/* 选中的边 - 金色高亮 */}
      {selectedEdges.length > 0 && (
        <EdgeBatch
          edges={selectedEdges}
          color="#fbbf24"
          strokeWidth={3}
          onEdgeClick={handleEdgeClick}
          edgeIdsRef={edgeIdsRef}
        />
      )}

      {/* 语义边 - 不同颜色 */}
      {semanticEdges.map(({ type, edges }) => (
        <EdgeBatch
          key={type}
          edges={edges}
          color={getSemanticColor(type)}
          strokeWidth={2}
          onEdgeClick={handleEdgeClick}
          edgeIdsRef={edgeIdsRef}
        />
      ))}
    </group>
  )
}

/**
 * 批量渲染边
 * ✅ S003: 合并几何体减少 Draw Call
 */
interface EdgeBatchProps {
  edges: Edge[]
  color: string
  strokeWidth: number
  onEdgeClick?: (edgeId: number) => void
  edgeIdsRef: React.MutableRefObject<Map<string, number[]>>
}

function EdgeBatch({ edges, color, strokeWidth, onEdgeClick, edgeIdsRef }: EdgeBatchProps) {
  const lineRef = useRef<THREE.LineSegments>(null)

  // ✅ S003: 创建合并的几何体
  const geometry = useMemo(() => {
    const positions: number[] = []
    const ids: number[] = []

    edges.forEach((edge) => {
      // 添加起点坐标
      positions.push(edge.start[0], edge.start[1], 0)
      // 添加终点坐标
      positions.push(edge.end[0], edge.end[1], 0)
      // 记录边 ID（用于点击检测）
      ids.push(edge.id, edge.id)
    })

    // 创建合并后的几何体
    const mergedGeometry = new THREE.BufferGeometry()
    mergedGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3)
    )
    mergedGeometry.setAttribute(
      'edgeId',
      new THREE.Uint32BufferAttribute(ids, 1)
    )

    // 存储边 ID 映射到 ref
    const batchKey = `${color}-${edges.length}-${Date.now()}`
    edgeIdsRef.current.set(batchKey, ids)

    return mergedGeometry
  }, [edges, color])

  // ✅ S003: 清理几何体
  useEffect(() => {
    return () => {
      geometry.dispose()
    }
  }, [geometry])

  // 创建材质
  const material = useMemo(() => {
    return new THREE.LineBasicMaterial({
      color,
      linewidth: strokeWidth,
      transparent: true,
      opacity: 0.9,
    })
  }, [color, strokeWidth])

  const handleClick = useCallback(
    (e: any) => {
      e.stopPropagation()
      
      // 从 geometry 中获取 edgeId 属性
      const geometry = e.object.geometry as THREE.BufferGeometry
      const edgeIdAttribute = geometry.getAttribute('edgeId') as THREE.BufferAttribute
      
      if (edgeIdAttribute) {
        const index = Math.floor(e.pointIndex / 2)
        const edgeId = edgeIdAttribute.getX(index)
        console.log('[EdgeBatch] Edge clicked:', edgeId)
        onEdgeClick?.(edgeId)
      }
    },
    [onEdgeClick]
  )

  return (
    <lineSegments
      ref={lineRef}
      geometry={geometry}
      material={material}
      onClick={handleClick}
      onPointerOver={() => {
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'default'
      }}
    />
  )
}

/**
 * 获取语义颜色
 */
function getSemanticColor(semantic: string): string {
  const colors: Record<string, string> = {
    hard_wall: '#60a5fa',
    absorptive_wall: '#8b5cf6',
    opening: '#f97316',
    window: '#06b6d4',
    door: '#84cc16',
    custom: '#ec4899',
  }
  return colors[semantic] || '#94a3b8'
}
