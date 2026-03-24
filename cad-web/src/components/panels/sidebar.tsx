import { Layers } from 'lucide-react'
import { cn } from '@/lib/utils'
import { LayerPanel } from '@components/panels/layer-panel'

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  return (
    <aside
      className={cn(
        'w-64 border-r bg-background flex flex-col overflow-hidden',
        className
      )}
    >
      {/* 侧边栏标题 */}
      <div className="h-10 border-b flex items-center px-4 gap-2">
        <Layers className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">图层</span>
      </div>

      {/* 图层面板 */}
      <LayerPanel />
    </aside>
  )
}
