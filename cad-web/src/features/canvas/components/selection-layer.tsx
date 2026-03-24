import { Line, Circle } from 'react-konva'
import type { Edge, Point } from '@/types/api'

interface SelectionLayerProps {
  selectedEdge: Edge | null
  selectedEdgeIds: number[]
  traceResult: Point[] | null
}

export function SelectionLayer({
  selectedEdge,
  selectedEdgeIds,
  traceResult,
}: SelectionLayerProps) {
  return (
    <>
      {/* 高亮选中的边 */}
      {selectedEdge && (
        <Line
          points={[
            selectedEdge.start[0],
            selectedEdge.start[1],
            selectedEdge.end[0],
            selectedEdge.end[1],
          ]}
          stroke="#fbbf24"
          strokeWidth={4}
          tension={0}
          lineCap="round"
          opacity={0.8}
        />
      )}

      {/* 高亮所有选中的边 */}
      {selectedEdgeIds.map(() => {
        // 这里需要从 edges 中查找，简化处理
        return null
      })}

      {/* 绘制追踪路径 */}
      {traceResult && traceResult.length > 0 && (
        <>
          <Line
            points={traceResult.flatMap((p) => p)}
            stroke="#22c55e"
            strokeWidth={2}
            tension={0}
            lineCap="round"
            lineJoin="round"
            closed={true}
            opacity={0.8}
          />
          {/* 绘制路径点 */}
          {traceResult.map((point, index) => (
            <Circle
              key={index}
              x={point[0]}
              y={point[1]}
              radius={4}
              fill="#22c55e"
            />
          ))}
        </>
      )}
    </>
  )
}
