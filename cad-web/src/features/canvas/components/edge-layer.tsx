import { useMemo, useCallback } from 'react'
import { Line, Arc as KonvaArc } from 'react-konva'
import type { Edge, CameraState } from '@/types/api'

interface EdgeLayerProps {
  edges: Edge[]
  camera: CameraState
  selectedEdgeIds?: number[]
  onEdgeClick?: (edgeId: number) => void
}

// P1-1 新增：LOD 线宽计算函数
// 根据 zoom 级别和边的语义类型动态调整线宽
// ✅ P1-2 修复：导出供 canvas-viewer.tsx 调试覆盖层使用
// ✅ P2-2 修复：添加最小线宽限制，防止 zoom > 5.0 时线宽过细
export function calculateLodLineWidth(
  zoom: number,
  semantic?: string,
  isWall?: boolean
): number {
  // 基础 LOD（基于 zoom）
  let baseWidth: number
  if (zoom < 0.3) {
    // 概览模式：较细的线宽
    baseWidth = 0.8 + zoom * 1.5
  } else if (zoom < 1.5) {
    // 标准模式：中等线宽
    baseWidth = 0.8 + (zoom - 0.3) * 1.2
  } else {
    // 放大模式：最大线宽
    baseWidth = 2.5
  }

  // 语义乘数（根据边的重要性调整线宽）
  let multiplier = 1.0
  if (isWall) {
    // 墙体：最重要的边，线宽最粗
    multiplier = 1.3
  } else if (semantic === 'opening' || semantic === 'window' || semantic === 'door') {
    // 门窗：次重要的边
    multiplier = 1.1
  } else if (semantic === 'dimension' || semantic === 'text') {
    // 标注和文字：辅助信息，线宽较细
    multiplier = 0.7
  } else if (semantic === 'furniture') {
    // 家具：次要信息
    multiplier = 0.9
  }

  // 计算最终线宽（除以 zoom 保持视觉一致性）
  // ✅ P2-2 修复：添加最小线宽限制，防止 zoom > 5.0 时线宽过细
  const computedWidth = (baseWidth * multiplier) / zoom
  const minWidth = 0.5  // 最小线宽，保证高 zoom 下仍然可见
  return Math.max(computedWidth, minWidth)
}

// 边的样式配置
const EDGE_STYLES = {
  wall: {
    default: { stroke: '#60a5fa', strokeWidth: 2 },
    hover: { stroke: '#93c5fd' },
  },
  nonWall: {
    default: { stroke: '#94a3b8', strokeWidth: 1.5 },
    hover: { stroke: '#cbd5e1' },
  },
  selected: { stroke: '#fbbf24', strokeWidth: 3 },
  semantic: {
    hard_wall: '#60a5fa',
    absorptive_wall: '#8b5cf6',
    opening: '#f97316',
    window: '#06b6d4',
    door: '#84cc16',
    custom: '#ec4899',
  },
} as const

/**
 * ✅ 新增：渲染单条弧线
 * Konva Arc 使用角度制（度数），需要转换弧度到角度
 */
function ArcEdge({
  edge,
  camera,
  style,
  onClick,
}: {
  edge: Edge
  camera: CameraState
  style: {
    stroke: string
    strokeWidth: number
    opacity?: number
  }
  onClick?: (edgeId: number) => void
}) {
  const handleClick = useCallback(() => {
    if (onClick) {
      onClick(edge.id)
    }
  }, [onClick, edge.id])

  if (!edge.arc) {
    return null
  }

  const { center, radius, start_angle, end_angle, ccw } = edge.arc

  // ✅ 将弧度转换为角度（Konva 使用角度制）
  const startAngleDeg = (start_angle * 180) / Math.PI
  const endAngleDeg = (end_angle * 180) / Math.PI

  // ✅ 计算弧线的角度（处理 ccw 和跨越 0 度的情况）
  let angleDiff = endAngleDeg - startAngleDeg
  if (ccw) {
    // 逆时针
    if (angleDiff >= 0) {
      angleDiff -= 360
    }
  } else {
    // 顺时针（默认）
    if (angleDiff < 0) {
      angleDiff += 360
    }
  }

  // ✅ 计算包围盒（用于定位）
  // Konva Arc 的 (x, y) 是圆心位置
  const x = center[0]
  const y = center[1]

  // ✅ 半径需要根据 zoom 调整以保持视觉一致性
  const strokeWidth = calculateLodLineWidth(camera.zoom, edge.semantic, edge.is_wall)

  return (
    <KonvaArc
      key={`arc-${edge.id}`}
      x={x}
      y={y}
      innerRadius={radius - strokeWidth / 2}
      outerRadius={radius + strokeWidth / 2}
      angle={startAngleDeg}
      clockwise={!ccw}
      // ✅ Konva Arc 的 angle 属性是弧线的总角度
      arc={Math.abs(angleDiff)}
      stroke={style.stroke}
      strokeWidth={strokeWidth}
      opacity={style.opacity ?? 1}
      lineCap="round"
      lineJoin="round"
      hitStrokeWidth={Math.max(10 / camera.zoom, 5)}
      attrMyEdgeId={edge.id}
      onClick={handleClick}
      onTap={handleClick}
    />
  )
}

/**
 * 批量渲染直线边（使用 Shape 缓存优化）
 */
function BatchedLines({
  edges,
  camera,
  style,
  onEdgeClick,
}: {
  edges: Edge[]
  camera: CameraState
  style: {
    stroke: string
    strokeWidth: number
    opacity?: number
  }
  onEdgeClick?: (edgeId: number) => void
}) {
  const handleClick = useCallback(
    (e: any) => {
      const edgeId = e.target.getAttr('edgeId')
      if (edgeId && onEdgeClick) {
        onEdgeClick(edgeId)
      }
    },
    [onEdgeClick]
  )

  // 使用 useMemo 缓存边元素（P1-1 修复：使用 LOD 线宽）
  const edgeElements = useMemo(() => {
    return edges.map((edge) => (
      <Line
        key={`line-${edge.id}`}
        points={[edge.start[0], edge.start[1], edge.end[0], edge.end[1]]}
        stroke={style.stroke}
        strokeWidth={calculateLodLineWidth(camera.zoom, edge.semantic, edge.is_wall)}
        opacity={style.opacity ?? 1}
        tension={0}
        lineCap="round"
        lineJoin="round"
        hitStrokeWidth={Math.max(10 / camera.zoom, 5)}
        attrMyEdgeId={edge.id}
        onClick={handleClick}
        onTap={handleClick}
      />
    ))
  }, [edges, camera.zoom, style.stroke, style.strokeWidth, style.opacity, handleClick])

  return <>{edgeElements}</>
}

/**
 * 批量渲染弧线边
 */
function BatchedArcs({
  edges,
  camera,
  style,
  onEdgeClick,
}: {
  edges: Edge[]
  camera: CameraState
  style: {
    stroke: string
    strokeWidth: number
    opacity?: number
  }
  onEdgeClick?: (edgeId: number) => void
}) {
  const handleClick = useCallback(
    (e: any) => {
      const edgeId = e.target.getAttr('edgeId')
      if (edgeId && onEdgeClick) {
        onEdgeClick(edgeId)
      }
    },
    [onEdgeClick]
  )

  const arcElements = useMemo(() => {
    return edges.map((edge) => (
      <ArcEdge
        key={`arc-${edge.id}`}
        edge={edge}
        camera={camera}
        style={style}
        onClick={handleClick}
      />
    ))
  }, [edges, camera, style, handleClick])

  return <>{arcElements}</>
}

export function EdgeLayer({
  edges,
  camera,
  selectedEdgeIds = [],
  onEdgeClick,
}: EdgeLayerProps) {
  const selectedEdgeIdSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds])

  // ✅ 分组边（分离直线和弧线）
  const groupedEdges = useMemo(() => {
    const groups = new Map<string, { lines: Edge[]; arcs: Edge[] }>()

    edges.forEach((edge) => {
      // 确定边的组别
      let groupKey: string

      if (selectedEdgeIdSet.has(edge.id)) {
        groupKey = 'selected'
      } else if (edge.semantic && edge.semantic !== 'hard_wall') {
        groupKey = `semantic-${edge.semantic}`
      } else if (edge.is_wall) {
        groupKey = 'wall'
      } else {
        groupKey = 'other'
      }

      const group = groups.get(groupKey) || { lines: [], arcs: [] }
      
      // ✅ 根据是否有 arc 字段分类
      if (edge.arc) {
        group.arcs.push(edge)
      } else {
        group.lines.push(edge)
      }
      
      groups.set(groupKey, group)
    })

    return groups
  }, [edges, selectedEdgeIdSet])

  // 渲染各组边
  return (
    <>
      {/* 普通墙边 */}
      {groupedEdges.get('wall') && (
        <>
          <BatchedLines
            edges={groupedEdges.get('wall')!.lines}
            camera={camera}
            style={EDGE_STYLES.wall.default}
            onEdgeClick={onEdgeClick}
          />
          <BatchedArcs
            edges={groupedEdges.get('wall')!.arcs}
            camera={camera}
            style={EDGE_STYLES.wall.default}
            onEdgeClick={onEdgeClick}
          />
        </>
      )}

      {/* 其他边 */}
      {groupedEdges.get('other') && (
        <>
          <BatchedLines
            edges={groupedEdges.get('other')!.lines}
            camera={camera}
            style={EDGE_STYLES.nonWall.default}
            onEdgeClick={onEdgeClick}
          />
          <BatchedArcs
            edges={groupedEdges.get('other')!.arcs}
            camera={camera}
            style={EDGE_STYLES.nonWall.default}
            onEdgeClick={onEdgeClick}
          />
        </>
      )}

      {/* 选中的边 */}
      {groupedEdges.get('selected') && (
        <>
          <BatchedLines
            edges={groupedEdges.get('selected')!.lines}
            camera={camera}
            style={EDGE_STYLES.selected}
            onEdgeClick={onEdgeClick}
          />
          <BatchedArcs
            edges={groupedEdges.get('selected')!.arcs}
            camera={camera}
            style={EDGE_STYLES.selected}
            onEdgeClick={onEdgeClick}
          />
        </>
      )}

      {/* 语义边 */}
      {Array.from(groupedEdges.entries())
        .filter(([key]) => key.startsWith('semantic-'))
        .map(([key, group]) => {
          const semanticType = key.replace('semantic-', '')
          const color = EDGE_STYLES.semantic[semanticType as keyof typeof EDGE_STYLES.semantic]
          return (
            <div key={key}>
              <BatchedLines
                edges={group.lines}
                camera={camera}
                style={{ stroke: color, strokeWidth: 2 }}
                onEdgeClick={onEdgeClick}
              />
              <BatchedArcs
                edges={group.arcs}
                camera={camera}
                style={{ stroke: color, strokeWidth: 2 }}
                onEdgeClick={onEdgeClick}
              />
            </div>
          )
        })}
    </>
  )
}
