import { Line, Text } from 'react-konva'
import type { GapInfo } from '@/types/api'

interface InteractionLayerProps {
  gaps?: GapInfo[]
}

export function InteractionLayer({ gaps = [] }: InteractionLayerProps) {
  return (
    <>
      {/* 绘制缺口标记 */}
      {gaps.map((gap) => (
        <g key={gap.id}>
          <Line
            points={[gap.start[0], gap.start[1], gap.end[0], gap.end[1]]}
            stroke="#ef4444"
            strokeWidth={2}
            dash={[5, 5]}
            tension={0}
          />
          <Text
            x={(gap.start[0] + gap.end[0]) / 2}
            y={(gap.start[1] + gap.end[1]) / 2 - 10}
            text={`${gap.length.toFixed(1)}mm`}
            fontSize={12}
            fill="#ef4444"
            align="center"
          />
        </g>
      ))}
    </>
  )
}
