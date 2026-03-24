import { useState } from 'react'
import { CanvasViewer } from './canvas-viewer'
import { ThreeCanvasViewer } from './three-viewer'
import { useCanvasStore } from '@/stores/canvas-store'
import { Button } from '@/components/ui/button'
import { Box, Layers, AlertCircle } from 'lucide-react'

/**
 * 统一 Canvas 查看器包装组件
 * 支持在 Konva 和 Three.js 渲染器之间切换
 */
export function UnifiedCanvasViewer() {
  const { edges, hatches, selectedEdgeIds, selectEdge } = useCanvasStore()

  // 渲染器选择：'konva' | 'three'
  // 临时默认使用 Three.js 测试 react-konva 问题
  const [renderer, setRenderer] = useState<'konva' | 'three'>('three')

  const handleToggleRenderer = () => {
    console.log('[UnifiedCanvasViewer] Toggling renderer from', renderer, 'to', renderer === 'konva' ? 'three' : 'konva')
    setRenderer((prev) => (prev === 'konva' ? 'three' : 'konva'))
  }

  return (
    <div className="relative w-full h-full">
      {/* 渲染器切换按钮 */}
      <div className="absolute top-4 right-4 z-40">
        <div className="acrylic-strong rounded-lg p-1 shadow-lg border border-white/20">
          <Button
            onClick={handleToggleRenderer}
            size="sm"
            variant="ghost"
            className="h-8 px-3 gap-2"
            title={`切换到 ${renderer === 'konva' ? 'Three.js' : 'Konva'} 渲染器`}
          >
            {renderer === 'konva' ? (
              <>
                <Box className="w-4 h-4" />
                <span className="text-xs">3D 视图</span>
              </>
            ) : (
              <>
                <Layers className="w-4 h-4" />
                <span className="text-xs">2D 视图</span>
              </>
            )}
          </Button>
        </div>
      </div>

      {/* 根据选择渲染不同的查看器 */}
      {renderer === 'konva' ? (
        <CanvasViewer />
      ) : (
        <ThreeCanvasViewer
          edges={edges}
          hatches={hatches}
          selectedEdgeIds={selectedEdgeIds}
          onEdgeClick={selectEdge}
          showGrid={true}
          showAxes={true}
          showStats={(import.meta as any).env.DEV}
        />
      )}

      {/* 渲染器信息提示（开发模式） */}
      {(import.meta as any).env.DEV && (
        <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono pointer-events-none z-30">
          <div>Renderer: {renderer.toUpperCase()}</div>
          <div>Edges: {edges.length}</div>
          <div>Hatches: {hatches.length}</div>
          {edges.length === 0 && hatches.length === 0 && (
            <div className="text-yellow-400 mt-1 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" />
              <span>无数据</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default UnifiedCanvasViewer
