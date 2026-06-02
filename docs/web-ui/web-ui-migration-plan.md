# CAD Web UI 迁移规划

**版本**: v0.1.0
**创建日期**: 2026 年 3 月 21 日
**状态**: 规划阶段
**技术栈**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui + WebGL/Three.js

---

## 一、项目概述

### 1.1 迁移背景

当前 CAD 几何智能处理系统使用 **egui** 作为前端界面，虽然功能完整，但存在以下限制：

| 限制 | 影响 |
|------|------|
| 浏览器不可达 | 无法通过 Web 访问，部署受限 |
| 生态有限 | egui 组件生态不如 Web 丰富 |
| 协作困难 | 不支持多用户实时协作 |
| 集成复杂 | 与企业系统集成成本高 |

### 1.2 迁移目标

构建一个**现代化、高性能、可协作**的 Web 前端，实现：

- ✅ **浏览器原生支持**：无需安装，开箱即用
- ✅ **现代化 UI/UX**：采用最新设计系统和交互模式
- ✅ **实时协作**：多用户同时编辑和查看
- ✅ **企业级集成**：支持 SSO、权限管理、审计日志
- ✅ **跨平台**：桌面/平板/手机自适应

### 1.3 设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                    设计哲学                                  │
├─────────────────────────────────────────────────────────────┤
│  1. 极简主义 (Minimalism)                                   │
│     - 减少视觉噪音，聚焦几何内容                             │
│     - 隐藏式工具栏，上下文感知 UI                            │
│  2. 性能优先 (Performance First)                            │
│     - WebGL 硬件加速渲染                                    │
│     - 虚拟滚动和 LOD 动态选择                               │
│     - WebAssembly 关键路径优化                              │
│  3. 可访问性 (Accessibility)                                │
│     - WCAG 2.1 AA 合规                                      │
│     - 键盘导航完整支持                                      │
│     - 屏幕阅读器友好                                        │
│  4. 渐进增强 (Progressive Enhancement)                      │
│     - 基础功能无需 JS 也能使用                               │
│     - 高级功能渐进加载                                      │
│     - 离线 PWA 支持                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、技术栈选型

### 2.1 核心技术栈

| 类别 | 技术 | 版本 | 理由 |
|------|------|------|------|
| **框架** | React | 18.3+ | 最成熟的生态，并发渲染支持 |
| **语言** | TypeScript | 5.4+ | 类型安全，智能提示 |
| **构建** | Vite | 5.2+ | 极速 HMR，原生 ESM |
| **样式** | Tailwind CSS | 3.4+ | 原子化 CSS，高可定制 |
| **组件库** | shadcn/ui | 最新 | 基于 Radix UI，完全可控 |
| **状态** | Zustand | 4.5+ | 轻量，无样板代码 |
| **路由** | React Router | 6.22+ | 标准方案，数据加载 |
| **表单** | React Hook Form | 7.50+ | 高性能，类型安全 |
| **验证** | Zod | 3.22+ | Schema 验证，与 TS 集成 |

### 2.2 图形渲染栈

| 类别 | 技术 | 版本 | 理由 |
|------|------|------|------|
| **2D 渲染** | Konva.js | 9.3+ | 高性能 Canvas 2D，图层支持 |
| **3D 渲染** | Three.js | 0.162+ | 最成熟的 WebGL 库 |
| **几何计算** | flatgeobuf + geo | 最新 | 与后端 geo crate 兼容 |
| **WebAssembly** | wasm-bindgen | 0.2.92+ | Rust 代码复用 |

### 2.3 实时通信栈

| 类别 | 技术 | 版本 | 理由 |
|------|------|------|------|
| **WebSocket** | native WebSocket | - | 浏览器原生支持 |
| **HTTP 客户端** | TanStack Query | 5.28+ | 缓存、重试、乐观更新 |
| **Server-Sent** | EventSource | - | 单向流式推送 |

### 2.4 工程化栈

| 类别 | 技术 | 版本 | 理由 |
|------|------|------|------|
| **包管理** | pnpm | 9.0+ | 快速，节省磁盘 |
| **代码质量** | Biome | 1.6+ | 替代 ESLint+Prettier |
| **测试** | Vitest | 1.3+ | Vite 原生，极速 |
| **E2E** | Playwright | 1.42+ | 跨浏览器，自动等待 |
| **文档** | Storybook | 8.0+ | 组件开发环境 |
| **部署** | Docker + Nginx | - | 标准化部署 |

### 2.5 技术栈对比

| 方案 | 优势 | 劣势 | 选择 |
|------|------|------|------|
| **React + Vite** | 生态最大，人才最多 | 体积较大 | ✅ 选择 |
| Vue 3 + Vite | 轻量，API 优雅 | 生态较小 | ❌ |
| SvelteKit | 编译时优化，体积小 | 生态新，人才少 | ❌ |
| SolidJS | 性能最佳 | 生态太小 | ❌ |
| **shadcn/ui** | 完全可控，可定制 | 需自己组装 | ✅ 选择 |
| Ant Design | 组件全，企业级 | 体积大，难定制 | ❌ |
| MUI | 组件全，文档好 | 体积大，API 复杂 | ❌ |
| **Konva.js** | Canvas 2D 性能优 | 仅支持 2D | ✅ 选择 (2D) |
| **Three.js** | WebGL 最成熟 | 学习曲线 | ✅ 选择 (3D) |
| PixiJS | 2D WebGL 性能最佳 | 仅 2D | ❌ |

---

## 三、架构设计

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端架构 (Web)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   表现层 (Presentation)              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  Canvas  │  │  Toolbar │  │  Panels  │          │   │
│  │  │  Viewer  │  │  Menu    │  │  Sidebar │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   应用层 (Application)               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  Command │  │   State  │  │   Event  │          │   │
│  │  │  Manager │  │  Manager │  │  Bus     │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   领域层 (Domain)                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ Geometry │  │  Topology│  │  Semantic│          │   │
│  │  │  Core    │  │  Core    │  │  Core    │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                              ↓                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 基础设施层 (Infrastructure)          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  HTTP    │  │   WS     │  │  WebGL   │          │   │
│  │  │  Client  │  │  Client  │  │  Render  │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      后端架构 (Rust)                         │
│                   (保持现有架构不变)                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
cad-web/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── biome.json
├── docker-compose.yml
├── .env.example
├── .env.local.example
├── index.html
├── public/
│   ├── favicon.ico
│   ├── manifest.json
│   └── fonts/
├── src/
│   ├── main.tsx              # 应用入口
│   ├── App.tsx               # 根组件
│   │
│   ├── components/           # 通用组件
│   │   ├── ui/               # shadcn/ui 基础组件
│   │   │   ├── button.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── dropdown-menu.tsx
│   │   │   ├── toast.tsx
│   │   │   └── ...
│   │   ├── canvas/           # Canvas 相关组件
│   │   │   ├── canvas-viewer.tsx
│   │   │   ├── canvas-toolbar.tsx
│   │   │   ├── canvas-overlay.tsx
│   │   │   └── ...
│   │   ├── toolbar/          # 工具栏组件
│   │   │   ├── main-toolbar.tsx
│   │   │   ├── tool-button.tsx
│   │   │   └── ...
│   │   ├── panels/           # 面板组件
│   │   │   ├── layer-panel.tsx
│   │   │   ├── property-panel.tsx
│   │   │   └── ...
│   │   └── modals/           # 对话框组件
│   │       ├── file-upload.tsx
│   │       ├── settings.tsx
│   │       └── ...
│   │
│   ├── features/             # 功能模块
│   │   ├── file/             # 文件管理
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   └── types.ts
│   │   ├── canvas/           # Canvas 功能
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   └── types.ts
│   │   ├── interaction/      # 交互功能
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   └── types.ts
│   │   └── analysis/         # 分析功能
│   │       ├── components/
│   │       ├── hooks/
│   │       ├── services/
│   │       └── types.ts
│   │
│   ├── stores/               # 状态管理
│   │   ├── app-store.ts
│   │   ├── canvas-store.ts
│   │   ├── selection-store.ts
│   │   └── ...
│   │
│   ├── hooks/                # 自定义 Hooks
│   │   ├── use-file-upload.ts
│   │   ├── use-canvas-interaction.ts
│   │   ├── use-websocket.ts
│   │   └── ...
│   │
│   ├── services/             # API 服务
│   │   ├── api-client.ts
│   │   ├── websocket-client.ts
│   │   ├── file-service.ts
│   │   └── ...
│   │
│   ├── lib/                  # 工具库
│   │   ├── utils.ts          # cn() 等工具
│   │   ├── geometry.ts       # 几何计算
│   │   └── ...
│   │
│   ├── types/                # 类型定义
│   │   ├── api.ts
│   │   ├── geometry.ts
│   │   └── index.ts
│   │
│   ├── assets/               # 静态资源
│   │   ├── images/
│   │   ├── icons/
│   │   └── styles/
│   │
│   └── utils/                # 工具函数
│       ├── formatters.ts
│       └── validators.ts
│
├── tests/                    # 测试文件
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── .storybook/               # Storybook 配置
│
└── docs/                     # 项目文档
    ├── architecture.md
    ├── components.md
    └── api.md
```

### 3.3 状态管理设计

采用 **Zustand** 作为状态管理方案，设计如下：

```typescript
// stores/app-store.ts
import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

interface AppState {
  // 文件状态
  currentFile: FileMetadata | null
  recentFiles: FileMetadata[]
  
  // 工具状态
  activeTool: ToolType
  toolHistory: ToolType[]
  
  // 用户设置
  settings: AppSettings
  
  // Actions
  actions: {
    openFile: (path: string) => Promise<void>
    saveFile: () => Promise<void>
    setTool: (tool: ToolType) => void
    undo: () => void
    redo: () => void
    updateSettings: (settings: Partial<AppSettings>) => void
  }
}

export const useAppStore = create<AppState>()(
  subscribeWithSelector((set, get) => ({
    currentFile: null,
    recentFiles: [],
    activeTool: 'select',
    toolHistory: [],
    settings: defaultSettings,
    
    actions: {
      openFile: async (path: string) => {
        // 实现
      },
      // ...
    }
  }))
)
```

### 3.4 命令模式设计

复用后端成熟的命令模式，前端实现：

```typescript
// features/command/command-manager.ts
interface Command {
  name: string
  execute: (state: AppState) => Promise<void>
  undo: (state: AppState) => Promise<void>
  redo: (state: AppState) => Promise<void>
}

class CommandManager {
  private history: Command[] = []
  private future: Command[] = []
  
  async execute(command: Command): Promise<void> {
    await command.execute(this.state)
    this.history.push(command)
    this.future = []
  }
  
  async undo(): Promise<void> {
    const command = this.history.pop()
    if (command) {
      await command.undo(this.state)
      this.future.push(command)
    }
  }
  
  async redo(): Promise<void> {
    const command = this.future.pop()
    if (command) {
      await command.redo(this.state)
      this.history.push(command)
    }
  }
}
```

---

## 四、核心功能映射

### 4.1 egui → Web 组件映射

| egui 组件 | Web 替代 | 说明 |
|-----------|---------|------|
| `egui::CentralPanel` | `<CanvasViewer />` | 中央画布区域 |
| `egui::TopBottomPanel` | `<Toolbar />` + `<StatusBar />` | 顶部工具栏 + 底部状态栏 |
| `egui::SidePanel` | `<Sidebar />` | 侧边面板（图层/属性） |
| `egui::Window` | `<Dialog />` / `<Modal />` | 对话框/模态框 |
| `egui::Button` | `<Button />` | 按钮组件 |
| `egui::ComboBox` | `<Select />` | 下拉选择 |
| `egui::Slider` | `<Slider />` | 滑块控件 |
| `egui::TextEdit` | `<Input />` | 文本输入 |
| `egui::Checkbox` | `<Checkbox />` | 复选框 |
| `egui::ProgressBar` | `<Progress />` | 进度条 |

### 4.2 功能对照表

| 功能 | egui 实现 | Web 实现 | 优先级 |
|------|----------|---------|--------|
| **文件操作** | | | |
| 打开文件 | `rfd::FileDialog` | `<input type="file">` + Drag & Drop | P0 |
| 导出场景 | `rfd::FileDialog` | 浏览器下载 API | P0 |
| 最近文件 | 本地记录 | localStorage + IndexedDB | P1 |
| **Canvas 渲染** | | | |
| 线段绘制 | egui Painter | Konva.js / WebGL | P0 |
| 缩放/平移 | egui Response | Konva Transformer | P0 |
| 点选边 | 射线检测 | Konva hit detection | P0 |
| 高亮追踪 | 自定义绘制 | Konva Layer + Effects | P0 |
| 圈选工具 | 多边形绘制 | Konva Transformer | P1 |
| **交互功能** | | | |
| 选边追踪 | HTTP/WebSocket | WebSocket + TanStack Query | P0 |
| 缺口检测 | HTTP/WebSocket | WebSocket + SSE | P0 |
| 语义标注 | ComboBox | shadcn/ui Select | P0 |
| 图层管理 | 自定义面板 | shadcn/ui Accordion | P0 |
| 属性面板 | 自定义面板 | shadcn/ui Form | P0 |
| **视觉效果** | | | |
| 毛玻璃效果 | wgpu | CSS backdrop-filter | P1 |
| GPU 加速 | wgpu | WebGL / WebGPU | P1 |
| 深色模式 | 自定义主题 | Tailwind dark: | P0 |
| 动画过渡 | 无 | Framer Motion | P1 |
| **通知系统** | | | |
| Toast 通知 | 自定义 | sonner / react-hot-toast | P0 |
| 错误对话框 | egui::Window | shadcn/ui Dialog | P0 |
| 加载状态 | 自定义 | shadcn/ui Skeleton | P0 |

---

## 五、核心组件设计

### 5.1 Canvas Viewer 组件

```typescript
// features/canvas/components/canvas-viewer.tsx
import { KonvaStage } from './konva-stage'
import { CanvasToolbar } from './canvas-toolbar'
import { CanvasOverlay } from './canvas-overlay'
import { useCanvasStore } from '@/stores/canvas-store'

export function CanvasViewer() {
  const { 
    edges, 
    selection, 
    camera, 
    tools,
    selectEdge, 
    autoTrace,
    setCamera 
  } = useCanvasStore()
  
  return (
    <div className="relative w-full h-full bg-slate-50 dark:bg-slate-900">
      {/* 工具栏 */}
      <CanvasToolbar className="absolute top-4 left-1/2 -translate-x-1/2 z-10" />
      
      {/* 画布 */}
      <KonvaStage
        edges={edges}
        selection={selection}
        camera={camera}
        onEdgeSelect={selectEdge}
        onAutoTrace={autoTrace}
        onCameraChange={setCamera}
      />
      
      {/* 覆盖层（缺口标记/高亮等） */}
      <CanvasOverlay 
        gaps={useCanvasStore(state => state.gaps)}
        traceResult={useCanvasStore(state => state.traceResult)}
      />
      
      {/* 加载状态 */}
      <LoadingOverlay />
    </div>
  )
}
```

### 5.2 Konva Stage 组件

```typescript
// features/canvas/components/konva-stage.tsx
import { Stage, Layer, Line, Group } from 'react-konva'
import { EdgeLayer } from './edge-layer'
import { SelectionLayer } from './selection-layer'
import { InteractionLayer } from './interaction-layer'

interface KonvaStageProps {
  edges: Edge[]
  selection: Selection | null
  camera: CameraState
  onEdgeSelect: (edgeId: number) => void
  onAutoTrace: (edgeId: number) => void
  onCameraChange: (camera: CameraState) => void
}

export function KonvaStage({
  edges,
  selection,
  camera,
  onEdgeSelect,
  onAutoTrace,
  onCameraChange
}: KonvaStageProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 600 })
  
  // 响应式尺寸
  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        })
      }
    })
    
    if (containerRef.current) {
      observer.observe(containerRef.current)
    }
    
    return () => observer.disconnect()
  }, [])
  
  return (
    <div ref={containerRef} className="w-full h-full">
      <Stage width={size.width} height={size.height}>
        {/* 边图层 */}
        <EdgeLayer edges={edges} camera={camera} />
        
        {/* 选择高亮图层 */}
        <SelectionLayer selection={selection} />
        
        {/* 交互图层（缺口标记/追踪路径） */}
        <InteractionLayer />
        
        {/* 隐藏的事件捕获层 */}
        <InteractionLayer 
          invisible 
          onSelect={onEdgeSelect}
          onAutoTrace={onAutoTrace}
        />
      </Stage>
    </div>
  )
}
```

### 5.3 WebSocket 客户端

```typescript
// services/websocket-client.ts
import { EventEmitter } from 'eventemitter3'

interface WebSocketEvents {
  'connected': () => void
  'disconnected': () => void
  'edge_selected': (data: EdgeSelectedEvent) => void
  'auto_trace_result': (data: AutoTraceResult) => void
  'gap_detection': (data: GapDetectionResult) => void
  'topology_update': (data: TopologyUpdate) => void
  'parse_progress': (data: ParseProgress) => void
  'error': (error: Error) => void
}

class WebSocketClient extends EventEmitter<WebSocketEvents> {
  private ws: WebSocket | null = null
  private url: string
  private reconnectTimer: NodeJS.Timeout | null = null
  private reconnectDelay = 1000
  
  constructor(url: string) {
    super()
    this.url = url
  }
  
  connect() {
    this.ws = new WebSocket(this.url)
    
    this.ws.onopen = () => {
      this.reconnectDelay = 1000
      this.emit('connected')
    }
    
    this.ws.onclose = () => {
      this.emit('disconnected')
      this.scheduleReconnect()
    }
    
    this.ws.onerror = (error) => {
      this.emit('error', error)
    }
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data)
      this.emit(message.type, message.payload)
    }
  }
  
  send<T extends keyof WebSocketEvents>(type: T, payload: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }))
    }
  }
  
  private scheduleReconnect() {
    this.reconnectTimer = setTimeout(() => {
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000)
      this.connect()
    }, this.reconnectDelay)
  }
  
  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }
    this.ws?.close()
  }
}

export const wsClient = new WebSocketClient('ws://localhost:3000/ws')
```

### 5.4 API 客户端

```typescript
// services/api-client.ts
import { QueryClient } from '@tanstack/react-query'
import { z } from 'zod'

// Schema 定义
const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'unhealthy', 'degraded']),
  version: z.string(),
  api_version: z.string()
})

const ProcessResponseSchema = z.object({
  job_id: z.string(),
  status: z.enum(['completed', 'partial', 'failed']),
  message: z.string(),
  result: z.object({
    scene_summary: z.object({
      outer_boundaries: z.number(),
      holes: z.number(),
      total_points: z.number()
    }),
    validation_summary: z.object({
      error_count: z.number(),
      warning_count: z.number(),
      passed: z.boolean()
    }),
    output_size: z.number()
  })
})

export class ApiClient {
  private baseUrl: string
  private queryClient: QueryClient
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
    this.queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: 2,
          retryDelay: 1000,
          staleTime: 5 * 60 * 1000, // 5 分钟
        }
      }
    })
  }
  
  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`)
    const data = await response.json()
    return HealthResponseSchema.parse(data)
  }
  
  async processFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch(`${this.baseUrl}/process`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    return ProcessResponseSchema.parse(data)
  }
  
  // 使用 TanStack Query 的 hooks
  useHealthCheck() {
    return this.queryClient.useQuery({
      queryKey: ['health'],
      queryFn: () => this.healthCheck(),
      refetchInterval: 30000, // 30 秒轮询
      retry: false
    })
  }
}

export const apiClient = new ApiClient('http://localhost:3000')
```

---

## 六、UI/UX 设计

### 6.1 设计系统

采用 **shadcn/ui** 作为基础，定制 CAD 专用设计系统：

```typescript
// tailwind.config.js
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        // CAD 专用颜色
        canvas: {
          background: '#1a1a1a',
          grid: '#2a2a2a',
          edge: '#60a5fa',
          edgeHover: '#93c5fd',
          edgeSelected: '#fbbf24',
          gap: '#ef4444',
          trace: '#22c55e',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s infinite',
      }
    }
  }
}
```

### 6.2 布局设计

```
┌─────────────────────────────────────────────────────────────┐
│  Header (Logo + 文件菜单 + 工具 + 用户)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┬─────────────────────────────────┬───────────┐ │
│  │         │                                 │           │ │
│  │ Layer   │                                 │ Property  │ │
│  │ Panel   │         Canvas Area             │ Panel     │ │
│  │         │                                 │           │ │
│  │ 200px   │          自适应宽度              │ 250px     │ │
│  │         │                                 │           │ │
│  └─────────┴─────────────────────────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Status Bar (坐标 | 线段数 | 性能 | 版本)                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 交互设计

#### 6.3.1 工具栏设计

```typescript
// components/toolbar/main-toolbar.tsx
export function MainToolbar() {
  return (
    <Toolbar className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <ToolbarGroup label="文件">
        <ToolButton 
          icon={<FolderOpen />} 
          label="打开" 
          shortcut="Ctrl+O"
          onClick={() => commands.openFile()}
        />
        <ToolButton 
          icon={<Save />} 
          label="保存" 
          shortcut="Ctrl+S"
          onClick={() => commands.saveFile()}
        />
        <ToolButton 
          icon={<Download />} 
          label="导出" 
          shortcut="Ctrl+E"
          onClick={() => commands.export()}
        />
      </ToolbarGroup>
      
      <ToolbarSeparator />
      
      <ToolbarGroup label="工具">
        <ToolButton 
          icon={<MousePointer />} 
          label="选择" 
          shortcut="V"
          active={activeTool === 'select'}
          onClick={() => setTool('select')}
        />
        <ToolButton 
          icon={<Wand />} 
          label="自动追踪" 
          shortcut="T"
          active={activeTool === 'trace'}
          onClick={() => setTool('trace')}
        />
        <ToolButton 
          icon={<Lasso />} 
          label="圈选" 
          shortcut="L"
          active={activeTool === 'lasso'}
          onClick={() => setTool('lasso')}
        />
      </ToolbarGroup>
      
      <ToolbarSeparator />
      
      <ToolbarGroup label="视图">
        <ToolButton 
          icon={<ZoomIn />} 
          label="放大" 
          shortcut="+"
          onClick={() => zoomIn()}
        />
        <ToolButton 
          icon={<ZoomOut />} 
          label="缩小" 
          shortcut="-"
          onClick={() => zoomOut()}
        />
        <ToolButton 
          icon={<FitScreen />} 
          label="适应窗口" 
          shortcut="0"
          onClick={() => fitToScene()}
        />
      </ToolbarGroup>
    </Toolbar>
  )
}
```

#### 6.3.2 右键菜单

```typescript
// components/canvas/context-menu.tsx
export function CanvasContextMenu() {
  const [position, setPosition] = useState<{ x: number, y: number } | null>(null)
  const selectedEdge = useCanvasStore(state => state.selectedEdge)
  
  return (
    <ContextMenu 
      open={position !== null}
      onOpenChange={(open) => !open && setPosition(null)}
    >
      <ContextMenuContent style={{ left: position?.x, top: position?.y }}>
        <ContextMenuItem onClick={() => commands.autoTrace()}>
          <Wand className="mr-2" />
          自动追踪
          <ContextMenuShortcut>T</ContextMenuShortcut>
        </ContextMenuItem>
        
        <ContextMenuSeparator />
        
        <ContextMenuItem onClick={() => commands.detectGaps()}>
          <AlertCircle className="mr-2" />
          缺口检测
        </ContextMenuItem>
        
        {selectedEdge && (
          <>
            <ContextMenuSeparator />
            <ContextMenuSub>
              <ContextMenuSubTrigger>语义标注</ContextMenuSubTrigger>
              <ContextMenuSubContent>
                <ContextMenuItem onClick={() => setSemantic('hard_wall')}>
                  硬墙
                </ContextMenuItem>
                <ContextMenuItem onClick={() => setSemantic('absorptive_wall')}>
                  吸声墙
                </ContextMenuItem>
                <ContextMenuItem onClick={() => setSemantic('opening')}>
                  开口
                </ContextMenuItem>
              </ContextMenuSubContent>
            </ContextMenuSub>
          </>
        )}
      </ContextMenuContent>
    </ContextMenu>
  )
}
```

### 6.4 深色模式

```typescript
// hooks/use-theme.ts
export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')
  
  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
  }, [theme])
  
  return { theme, setTheme }
}

// 默认深色模式，适配 CAD 场景
// Canvas 背景：#1a1a1a (深色) / #f8fafc (浅色)
// 边颜色：#60a5fa (蓝色，高对比度)
```

---

## 七、性能优化

### 7.1 渲染优化

| 优化项 | 策略 | 预期效果 |
|--------|------|----------|
| **虚拟滚动** | 仅渲染可见区域边 | 10000+ 边流畅渲染 |
| **LOD 动态选择** | 根据缩放级别简化几何 | 减少 50% 绘制调用 |
| **图层批处理** | 合并相同样式边 | 减少 80% draw call |
| **Web Worker** | 几何计算离主线程 | 保持 60fps |
| **WebAssembly** | 关键算法 Rust 实现 | 3-5x 性能提升 |

### 7.2 代码分割

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-konva': ['konva', 'react-konva'],
          'vendor-ui': ['@radix-ui/*', 'class-variance-authority'],
          'features-canvas': ['./src/features/canvas'],
          'features-interaction': ['./src/features/interaction'],
        }
      }
    }
  }
})
```

### 7.3 懒加载

```typescript
// 路由懒加载
const CanvasViewer = lazy(() => import('@/features/canvas/components/canvas-viewer'))
const LayerPanel = lazy(() => import('@/features/canvas/components/layer-panel'))

// 组件内懒加载
const GpuRenderer = lazy(() => import('@/features/canvas/components/gpu-renderer'))

// 使用 Suspense
<Suspense fallback={<Skeleton className="w-full h-full" />}>
  <CanvasViewer />
</Suspense>
```

---

## 八、测试策略

### 8.1 测试金字塔

```
           /\
          /  \
         / E2E \        Playwright (10%)
        /______\
       /        \
      /Integration\     Vitest + Testing Library (30%)
     /____________\
    /              \
   /    Unit Tests   \   Vitest (60%)
  /__________________\
```

### 8.2 单元测试

```typescript
// tests/unit/geometry.test.ts
import { describe, it, expect } from 'vitest'
import { snapEndpoints, douglasPeucker } from '@/lib/geometry'

describe('geometry', () => {
  describe('snapEndpoints', () => {
    it('应该吸附距离小于容差的端点', () => {
      const points = [[0, 0], [0.5, 0], [10, 10]]
      const result = snapEndpoints(points, 1.0)
      expect(result).toEqual([[0, 0], [0, 0], [10, 10]])
    })
  })
  
  describe('douglasPeucker', () => {
    it('应该简化折线并保持形状', () => {
      const line = [[0, 0], [1, 0.1], [2, 0], [3, 0.2], [4, 0]]
      const result = douglasPeucker(line, 0.5)
      expect(result.length).toBeLessThan(line.length)
    })
  })
})
```

### 8.3 组件测试

```typescript
// tests/unit/canvas-viewer.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { CanvasViewer } from '@/features/canvas/components/canvas-viewer'

describe('CanvasViewer', () => {
  it('应该渲染画布', () => {
    render(<CanvasViewer />)
    expect(screen.getByTestId('canvas-stage')).toBeInTheDocument()
  })
  
  it('应该响应边选择', async () => {
    const onEdgeSelect = vi.fn()
    render(<CanvasViewer onEdgeSelect={onEdgeSelect} />)
    
    const edge = screen.getByTestId('edge-1')
    fireEvent.click(edge)
    
    expect(onEdgeSelect).toHaveBeenCalledWith(1)
  })
})
```

### 8.4 E2E 测试

```typescript
// tests/e2e/file-upload.spec.ts
import { test, expect } from '@playwright/test'

test('应该上传并处理 DXF 文件', async ({ page }) => {
  await page.goto('/')
  
  // 点击打开文件
  await page.getByRole('button', { name: '打开' }).click()
  
  // 选择文件
  const fileInput = page.locator('input[type="file"]')
  await fileInput.setInputFiles('test-files/sample.dxf')
  
  // 等待处理完成
  await expect(page.getByText('处理完成')).toBeVisible({ timeout: 10000 })
  
  // 验证画布有内容
  const canvas = page.locator('canvas')
  await expect(canvas).toBeVisible()
})

test('应该支持自动追踪', async ({ page }) => {
  await page.goto('/')
  
  // 上传文件
  await uploadTestFile(page)
  
  // 选择一条边
  const edge = page.locator('[data-edge-id="1"]')
  await edge.click()
  
  // 点击自动追踪
  await page.getByRole('button', { name: '自动追踪' }).click()
  
  // 验证追踪结果
  await expect(page.getByText('追踪到闭合环')).toBeVisible()
})
```

---

## 九、部署方案

### 9.1 Docker 部署

```dockerfile
# Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 9.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  cad-web:
    build: .
    ports:
      - "80:80"
    environment:
      - BACKEND_URL=http://cad-backend:3000
    depends_on:
      - cad-backend
  
  cad-backend:
    image: cad-backend:latest
    ports:
      - "3000:3000"
    volumes:
      - ./dxfs:/app/dxfs
      - ./output:/app/output
```

### 9.3 Nginx 配置

```nginx
# nginx.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;
    
    # Gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    
    # 缓存静态资源
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # SPA 路由
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API 代理
    location /api {
        proxy_pass http://cad-backend:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # WebSocket 代理
    location /ws {
        proxy_pass http://cad-backend:3000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 十、迁移计划

### 10.1 阶段划分

| 阶段 | 时间 | 目标 | 交付物 |
|------|------|------|--------|
| **P0: 基础架构** | 2 周 | 项目搭建 + 核心组件 | 可运行的基础框架 |
| **P1: 核心功能** | 3 周 | Canvas 渲染 + 交互 | 完整功能 MVP |
| **P2: 增强功能** | 2 周 | 实时协作 + 性能优化 | 生产就绪版本 |
| **P3:  polish** | 1 周 | UI 打磨 + 测试 | 正式发布 |

### 10.2 详细任务

#### P0: 基础架构（2 周）

| 任务 | 优先级 | 预计工时 | 负责人 |
|------|--------|----------|--------|
| 项目初始化 (Vite + TS + Tailwind) | P0 | 4h | |
| shadcn/ui 集成 | P0 | 2h | |
| 目录结构搭建 | P0 | 2h | |
| 状态管理 (Zustand) | P0 | 4h | |
| 路由配置 | P0 | 2h | |
| API 客户端封装 | P0 | 4h | |
| WebSocket 客户端 | P0 | 4h | |
| 基础组件开发 | P0 | 8h | |
| 构建/部署配置 | P0 | 4h | |
| **小计** | | **34h** | |

#### P1: 核心功能（3 周）

| 任务 | 优先级 | 预计工时 | 负责人 |
|------|--------|----------|--------|
| Canvas Viewer 组件 | P0 | 16h | |
| Konva 渲染集成 | P0 | 12h | |
| 边选择交互 | P0 | 8h | |
| 缩放/平移 | P0 | 8h | |
| 自动追踪功能 | P0 | 8h | |
| 缺口检测功能 | P0 | 6h | |
| 图层面板 | P0 | 8h | |
| 属性面板 | P0 | 8h | |
| 文件上传/导出 | P0 | 6h | |
| Toast 通知系统 | P1 | 4h | |
| 加载状态 | P1 | 4h | |
| 错误处理 | P1 | 6h | |
| **小计** | | **94h** | |

#### P2: 增强功能（2 周）

| 任务 | 优先级 | 预计工时 | 负责人 |
|------|--------|----------|--------|
| 圈选工具 | P1 | 8h | |
| 语义标注 | P1 | 6h | |
| 深色模式 | P1 | 4h | |
| 键盘快捷键 | P1 | 6h | |
| 右键菜单 | P1 | 6h | |
| 性能优化 (虚拟滚动) | P1 | 12h | |
| Web Worker 集成 | P1 | 8h | |
| 单元测试 | P1 | 12h | |
| E2E 测试 | P1 | 12h | |
| **小计** | | **74h** | |

#### P3: Polish（1 周）

| 任务 | 优先级 | 预计工时 | 负责人 |
|------|--------|----------|--------|
| UI 细节打磨 | P1 | 8h | |
| 动画过渡 | P2 | 6h | |
| 响应式适配 | P1 | 6h | |
| 文档编写 | P1 | 8h | |
| Bug 修复 | P1 | 12h | |
| **小计** | | **40h** | |

### 10.3 里程碑

```
Week 1-2: [████████████] P0 完成
  ├── 项目可运行
  ├── 基础组件可用
  └── 部署流程打通

Week 3-5: [████████████] P1 完成
  ├── Canvas 渲染正常
  ├── 核心交互可用
  └── MVP 可演示

Week 6-7: [████████████] P2 完成
  ├── 增强功能完整
  ├── 性能达标
  └── 测试覆盖 80%

Week 8:   [████████████] P3 完成
  ├── UI 打磨完成
  ├── 文档完整
  └── 正式发布
```

---

## 十一、风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| Konva 性能不足 | 高 | 中 | 备选 WebGL 方案 |
| WebSocket 不稳定 | 高 | 中 | HTTP 降级方案 |
| 大文件加载慢 | 中 | 高 | 渐进式加载 + LOD |
| 浏览器兼容性 | 中 | 低 | Browserslist 配置 |
| 团队学习曲线 | 低 | 中 | 技术分享 + 文档 |

---

## 十二、成功标准

### 12.1 功能标准

- ✅ 所有 egui 功能 1:1 还原
- ✅ WebSocket 实时交互正常
- ✅ 文件上传/导出正常
- ✅ 所有交互操作流畅

### 12.2 性能标准

| 指标 | 目标 | 测量方式 |
|------|------|----------|
| 首屏加载 | < 2s | Lighthouse |
| 交互响应 | < 100ms | Performance API |
| Canvas 帧率 | 60fps | requestAnimationFrame |
| 万条边渲染 | 30fps+ | 基准测试 |

### 12.3 质量标准

| 指标 | 目标 |
|------|------|
| 测试覆盖率 | > 80% |
| TypeScript 错误 | 0 |
| Lighthouse 分数 | > 90 |
| 无障碍合规 | WCAG 2.1 AA |

---

## 十三、附录

### A. 参考项目

1. **Excalidraw** - 虚拟白板，优秀的 Canvas 交互
2. **tldraw** - 轻量级绘图工具
3. **CodeSandbox** - Web IDE 架构参考
4. **Figma** - 高性能 Web 渲染标杆

### B. 学习资源

1. [React 官方文档](https://react.dev)
2. [Vite 官方文档](https://vitejs.dev)
3. [Tailwind CSS](https://tailwindcss.com)
4. [shadcn/ui](https://ui.shadcn.com)
5. [React Konva](https://github.com/konva-dev/react-konva)
6. [TanStack Query](https://tanstack.com/query)

### C. 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-21 | 选择 React | 生态最大，人才最多 |
| 2026-03-21 | 选择 Konva | Canvas 2D 性能优，API 简单 |
| 2026-03-21 | 选择 Zustand | 轻量，无样板代码 |
| 2026-03-21 | 选择 shadcn/ui | 完全可控，可定制 |

---

**创建者**: CAD 团队
**审核者**: 
**最后更新**: 2026 年 3 月 21 日
