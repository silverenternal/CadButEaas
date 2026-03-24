import { useState } from 'react'
import { Grid3X3 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@components/ui/button'

interface GridOverlayProps {
  className?: string
  visible?: boolean
  onToggle?: (visible: boolean) => void
}

export function GridOverlay({ className, visible: externalVisible, onToggle }: GridOverlayProps) {
  const [internalVisible, setInternalVisible] = useState(false)
  const [gridSize, setGridSize] = useState(100)
  const [showSettings, setShowSettings] = useState(false)

  const visible = externalVisible ?? internalVisible
  const setVisible = onToggle ?? setInternalVisible

  return (
    <div className={cn('absolute bottom-4 right-4 z-40 flex flex-col items-end gap-2', className)}>
      {/* 网格设置面板 */}
      {showSettings && (
        <div className="acrylic-strong rounded-2xl p-4 shadow-2xl border border-white/20 mb-2 fade-in">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">显示网格</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setVisible(!visible)}
                className={cn(
                  'w-10 h-6 rounded-full transition-colors',
                  visible ? 'bg-primary' : 'bg-muted'
                )}
              >
                <div
                  className={cn(
                    'w-4 h-4 rounded-full bg-white transition-transform',
                    visible && 'translate-x-4'
                  )}
                />
              </Button>
            </div>

            {visible && (
              <div className="space-y-2">
                <label className="text-xs text-muted-foreground">
                  网格间距：{gridSize}px
                </label>
                <input
                  type="range"
                  min="20"
                  max="200"
                  step="10"
                  value={gridSize}
                  onChange={(e) => setGridSize(Number(e.target.value))}
                  className="w-full h-2 bg-muted rounded-full appearance-none cursor-pointer"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* 网格切换按钮 */}
      <div className="flex items-center gap-2">
        {showSettings && (
          <div className="acrylic-strong rounded-full px-3 py-1 text-xs font-medium shadow-lg border border-white/20">
            网格：{gridSize}px
          </div>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowSettings(!showSettings)}
          className={cn(
            'w-10 h-10 rounded-full shadow-lg border border-white/20 transition-all',
            visible && 'bg-primary/10 text-primary'
          )}
          title="网格设置"
        >
          <Grid3X3 className="w-5 h-5" />
        </Button>
      </div>

      {/* CSS 网格覆盖层 */}
      {visible && (
        <div
          className="pointer-events-none fixed inset-0 z-0"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(128,128,128,0.1) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(128,128,128,0.1) 1px, transparent 1px)
            `,
            backgroundSize: `${gridSize}px ${gridSize}px`,
          }}
        />
      )}
    </div>
  )
}
