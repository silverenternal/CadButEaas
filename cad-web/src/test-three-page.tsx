import { ThreeCanvasViewer } from './features/canvas/components/three-viewer'
import { useCanvasStore } from '@/stores/canvas-store'

/**
 * 临时测试页面 - 只使用 Three.js，不导入 react-konva
 */
export function TestThreePage() {
  const { edges, hatches, selectedEdgeIds, selectEdge } = useCanvasStore()

  return (
    <div className="w-full h-full">
      <ThreeCanvasViewer
        edges={edges}
        hatches={hatches}
        selectedEdgeIds={selectedEdgeIds}
        onEdgeClick={selectEdge}
        showGrid={true}
        showAxes={true}
        showStats={true}
      />
    </div>
  )
}
