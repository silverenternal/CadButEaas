import { cn } from '@/lib/utils'

interface ViewCubeProps {
  className?: string
}

/**
 * 视图方向指示器（简化版 2D）
 * 在 2D 视图中显示北向指示器
 */
export function ViewCube({ className }: ViewCubeProps) {
  return (
    <div className={cn('relative w-16 h-16', className)}>
      <div className="acrylic-strong rounded-full p-3 shadow-2xl border border-white/20">
        {/* 罗盘外圈 */}
        <div className="relative w-full h-full">
          {/* 北向指示器 */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2">
            <div className="w-0 h-0 border-l-4 border-r-4 border-b-[8px] border-l-transparent border-r-transparent border-b-primary" />
          </div>
          
          {/* 中心点 */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-primary" />
          </div>

          {/* 方向标记 */}
          <div className="absolute top-1/2 left-0 -translate-x-1/2 -translate-y-1/2 text-xs font-bold text-muted-foreground">
            W
          </div>
          <div className="absolute top-1/2 right-0 translate-x-1/2 -translate-y-1/2 text-xs font-bold text-muted-foreground">
            E
          </div>
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 text-xs font-bold text-muted-foreground">
            S
          </div>

          {/* 装饰性圆环 */}
          <div className="absolute inset-2 rounded-full border border-white/10" />
        </div>
      </div>
    </div>
  )
}
