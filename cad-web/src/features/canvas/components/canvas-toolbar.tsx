import { MousePointer, Hand, ScanLine, Square } from 'lucide-react'
import { useCanvasStore } from '@/stores/canvas-store'
import { cn } from '@/lib/utils'

const tools = [
  { id: 'select', icon: MousePointer, label: '选择' },
  { id: 'trace', icon: ScanLine, label: '追踪' },
  { id: 'lasso', icon: Square, label: '圈选' },
  { id: 'pan', icon: Hand, label: '平移' },
] as const

export function CanvasToolbar() {
  const { activeTool, setTool } = useCanvasStore()

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 fade-in">
      <div className="acrylic-strong rounded-2xl shadow-2xl">
        <div className="flex items-center gap-1.5 p-2">
          {tools.map((tool) => {
            const Icon = tool.icon
            const isActive = activeTool === tool.id
            return (
              <button
                key={tool.id}
                onClick={() => setTool(tool.id as any)}
                className={cn(
                  'group relative w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300',
                  'hover:scale-105 active:scale-95',
                  isActive
                    ? 'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg shadow-primary/30'
                    : 'text-muted-foreground hover:bg-accent/60 hover:text-accent-foreground hover:shadow-md'
                )}
                title={tool.label}
              >
                {/* 激活状态光晕 */}
                {isActive && (
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-white/0 via-white/20 to-white/0 opacity-50 blur-sm" />
                )}
                {/* 悬停光效 */}
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-white/0 via-white/10 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <Icon className="relative w-5 h-5 transition-transform duration-300 group-hover:scale-110 group-hover:-translate-y-0.5" />
              </button>
            )
          })}
        </div>
        {/* 底部装饰线 */}
        <div className="h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />
      </div>
    </div>
  )
}
