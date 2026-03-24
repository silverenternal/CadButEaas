import { Outlet } from 'react-router-dom'
import { MainToolbar } from '@components/toolbar/main-toolbar'
import { Sidebar } from '@components/panels/sidebar'
import { RightPanel } from '@components/panels/right-panel'
import { useAppStore } from '@/stores/app-store'
import { cn } from '@/lib/utils'

export function AppLayout() {
  const { sidebarOpen, rightPanelOpen } = useAppStore()

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-background">
      {/* 顶部工具栏 */}
      <MainToolbar />

      {/* 主体内容 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 左侧边栏 */}
        <Sidebar className={cn(!sidebarOpen && 'hidden')} />

        {/* 中央画布区域 */}
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>

        {/* 右侧属性面板 */}
        <RightPanel className={cn(!rightPanelOpen && 'hidden')} />
      </div>

      {/* 底部状态栏 */}
      <StatusBar />
    </div>
  )
}

function StatusBar() {
  return (
    <footer className="h-6 border-t bg-muted/50 px-4 flex items-center justify-between text-xs text-muted-foreground">
      <div className="flex items-center gap-4">
        <span>就绪</span>
        <span>|</span>
        <span>v0.1.0</span>
      </div>
      <div className="flex items-center gap-4">
        <span>坐标：0.00, 0.00</span>
        <span>|</span>
        <span>缩放：100%</span>
      </div>
    </footer>
  )
}
