import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { AnimatePresence, motion } from 'framer-motion'
import { queryClient } from './lib/query-client'
import { Toaster } from '@components/ui/sonner'
import { TooltipProvider } from '@components/ui/tooltip'
import { ErrorBoundary } from '@components/error-boundary'  // P2-2 新增
import { pageVariants } from '@/lib/animations.tsx'
import { cn } from '@/lib/utils'
import { CanvasPage } from '@features/canvas/pages/canvas-page'
import { TestThreePage } from './test-three-page'  // 临时测试 Three.js
import { ExploreMlightCadPage } from './pages/explore-mlightcad-page'  // 测试 mlightcad API
import { ConfigPage } from '@features/config/pages/config-page'
import { AcousticPage } from '@features/acoustic/pages/acoustic-page'
import { SettingsPage } from '@features/file/pages/settings-page'
import { LayoutDashboard, Settings2, Waves, FileText, Cpu } from 'lucide-react'

const navigation = [
  { name: '画布', href: '/', icon: LayoutDashboard },
  { name: 'MLightCad 测试', href: '/mlightcad-test', icon: Cpu },  // 新增
  { name: '配置', href: '/config', icon: Settings2 },
  { name: '声学', href: '/acoustic', icon: Waves },
  { name: '设置', href: '/settings', icon: FileText },
]

function NavItem({ item }: { item: typeof navigation[number] }) {
  const location = useLocation()
  const Icon = item.icon
  const isActive = location.pathname === item.href

  return (
    <Link to={item.href}>
      <motion.div
        className={cn(
          'group relative w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300',
          'hover:scale-110 active:scale-95',
          isActive
            ? 'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground shadow-lg shadow-primary/30'
            : 'text-muted-foreground hover:bg-accent/60 hover:text-accent-foreground hover:shadow-md'
        )}
        title={item.name}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        layout
        layoutId={`nav-${item.name}`}
      >
        {/* 激活状态光晕 */}
        {isActive && (
          <motion.div
            className="absolute inset-0 rounded-xl bg-gradient-to-r from-white/0 via-white/20 to-white/0 opacity-50 blur-sm"
            layoutId={`nav-glow-${item.name}`}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          />
        )}
        <motion.div
          className="transition-transform duration-300 group-hover:scale-110"
          whileHover={{ scale: 1.1 }}
        >
          <Icon className="w-5 h-5" />
        </motion.div>
      </motion.div>
    </Link>
  )
}

function AnimatedRoute({ element }: { element: React.ReactNode }) {
  return (
    <motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageVariants}
      className="h-full w-full"
    >
      {element}
    </motion.div>
  )
}

function AppContent() {
  const location = useLocation()

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gradient-to-br from-background via-background to-muted/30 gradient-animate">
      {/* 装饰性背景光斑 */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary/5 blur-[120px]"
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-500/5 blur-[120px]"
          animate={{
            scale: [1.1, 1, 1.1],
            opacity: [0.5, 0.3, 0.5],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* 左侧导航栏 - 亚克力效果 */}
      <aside className="relative z-10 w-16 acrylic-strong flex flex-col items-center py-4 gap-2">
        <motion.div
          className="w-full h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent mb-2"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        />
        {navigation.map((item) => (
          <NavItem key={item.name} item={item} />
        ))}
        <motion.div
          className="mt-auto w-full h-px bg-gradient-to-r from-transparent via-muted-foreground/20 to-transparent"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        />
      </aside>

      {/* 主体内容 */}
      <main className="relative z-10 flex-1 overflow-hidden">
        <AnimatePresence mode="wait" initial={false}>
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={<AnimatedRoute element={<CanvasPage />} />}
            />
            <Route
              path="/canvas-test"
              element={<AnimatedRoute element={<TestThreePage />} />}
            />
            <Route
              path="/mlightcad-test"
              element={<AnimatedRoute element={<ExploreMlightCadPage />} />}
            />
            <Route
              path="/config"
              element={<AnimatedRoute element={<ConfigPage />} />}
            />
            <Route
              path="/acoustic"
              element={<AnimatedRoute element={<AcousticPage />} />}
            />
            <Route
              path="/settings"
              element={<AnimatedRoute element={<SettingsPage />} />}
            />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        {/* P2-2 新增：全局 Error Boundary */}
        <ErrorBoundary
          onError={(error, errorInfo) => {
            // P2-2 修复：错误上报（可集成监控系统）
            console.error('[App] 全局错误捕获:', error, errorInfo)
            // TODO: 集成 Sentry/LogRocket 等监控系统
            // reportError(error, errorInfo)
          }}
        >
          <BrowserRouter>
            <AppContent />
            <ReactQueryDevtools initialIsOpen={false} />
            <Toaster />
          </BrowserRouter>
        </ErrorBoundary>
      </TooltipProvider>
    </QueryClientProvider>
  )
}

export default App
