import { useState, useEffect } from 'react'
import { Plus, Minus, Expand, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useCanvasStore } from '@/stores/canvas-store'
import { Button } from '@components/ui/button'

interface ZoomControlsProps {
  className?: string
}

export function ZoomControls({ className }: ZoomControlsProps) {
  const { camera, setCamera, fitToContent } = useCanvasStore()
  const [showReset, setShowReset] = useState(false)

  const zoom = camera.zoom

  const handleZoomIn = () => {
    setCamera({ zoom: Math.min(zoom * 1.2, 5) })
  }

  const handleZoomOut = () => {
    setCamera({ zoom: Math.max(zoom / 1.2, 0.1) })
  }

  const handleFit = () => {
    fitToContent(0.1)
    setShowReset(false)
  }

  const handleReset = () => {
    setCamera({ zoom: 1 })
    setShowReset(false)
  }

  // 当 zoom 偏离 1 较多时显示复位按钮
  useEffect(() => {
    setShowReset(Math.abs(zoom - 1) > 0.15)
  }, [zoom])

  return (
    <div className={cn('flex flex-col items-center gap-2', className)}>
      <div className="acrylic-strong rounded-2xl p-2 shadow-2xl border border-white/20 space-y-2">
        {/* 放大 */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleZoomIn}
          className="w-10 h-10 rounded-xl"
          title="放大"
        >
          <Plus className="w-5 h-5" />
        </Button>

        {/* 缩放百分比 */}
        <div className="text-center">
          <span className="text-sm font-bold tabular-nums">
            {Math.round(zoom * 100)}%
          </span>
        </div>

        {/* 缩小 */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleZoomOut}
          className="w-10 h-10 rounded-xl"
          title="缩小"
        >
          <Minus className="w-5 h-5" />
        </Button>

        {/* 分隔线 */}
        <div className="h-px bg-white/10 my-1" />

        {/* 适配/复位 */}
        {showReset ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="w-10 h-10 rounded-xl"
            title="复位视图"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleFit}
            className="w-10 h-10 rounded-xl"
            title="适配内容"
          >
            <Expand className="w-4 h-4" />
          </Button>
        )}
      </div>
    </div>
  )
}
