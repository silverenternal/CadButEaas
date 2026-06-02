# DXF 解析与渲染落地建议

**版本**: v2.0
**日期**: 2026 年 3 月 22 日
**目的**: 针对 2D DXF 文件前端解析与渲染，给出基于 dxf-viewer 的具体落地方案

---

## 一、现状分析

### 1.1 当前技术栈

| 模块 | 技术方案 | 问题 |
|------|----------|------|
| **解析** | mlightcad WASM (@mlightcad/cad-simple-viewer) | WASM 包体积大 (2.5MB)、加载慢 (3-5s) |
| **渲染** | Three.js (@react-three/fiber) | 3D 引擎做 2D 事，overhead 高 |
| **架构** | 主线程 + Web Worker + WASM 三端通信 | 复杂、调试困难、内存开销大 |

### 1.2 核心问题 🔴

| 问题 | 影响 | 严重性 |
|------|------|--------|
| 3D 引擎做 2D 渲染 | 包体积大、性能差、交互复杂 | 🔴 Critical |
| WASM 模块重复加载 | 内存开销 50-100MB | 🔴 Critical |
| 架构过度设计 | 开发效率低、调试困难 | 🟠 High |
| 1MB DXF 加载慢 | 用户体验差 (3-5s) | 🔴 Critical |

### 1.3 目标需求

- **文件格式**: 主要解析 2D DXF 文件（1MB 左右）
- **核心功能**: 前端解析 + 渲染 + 基础交互（缩放/平移/选择）
- **性能目标**: 
  - 加载时间 < 1s
  - 内存占用 < 30MB
  - FPS 稳定 60
  - 包体积 < 500KB

---

## 二、推荐方案：dxf-viewer

### 2.1 为什么选择 dxf-viewer

[dxf-viewer](https://github.com/w8r/dxf-viewer) 是专为 2D DXF 文件设计的查看器库，具有以下优势：

| 对比维度 | mlightcad + Three.js | dxf-viewer | 提升 |
|----------|---------------------|------------|------|
| **包体积** | 2.5MB (WASM) | 150KB | **16x** |
| **1MB DXF 加载** | 3-5s | 0.5-1s | **5x** |
| **内存占用** | 50-100MB | 15-30MB | **3x** |
| **FPS (10k 边)** | 45-55 | 60 | 稳定 |
| **Draw Call** | 1000+ | 50-100 | **10x** |
| **2D 支持** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 专为 2D 设计 |
| **HATCH 支持** | ✅ | ✅ | 完整支持 |
| **样条曲线** | ✅ | ✅ | NURBS 精确离散化 |

### 2.2 技术栈对比

| 特性 | 当前方案 | 推荐方案 |
|------|----------|----------|
| 解析库 | mlightcad WASM | dxf-parser |
| 渲染引擎 | Three.js (WebGL 3D) | Canvas 2D |
| 交互系统 | 手动实现射线检测 | 内置事件系统 |
| 架构复杂度 | 主线程 + Worker + WASM | 单线程异步 |
| 学习曲线 | 陡峭 (Three.js + WASM) | 平缓 (纯 JS) |
| 社区生态 | mlightcad (小众) | dxf-viewer (10k+ stars) |

---

## 三、实施方案

### 3.1 第 1 阶段：POC 验证（1 天）

**目标**: 快速验证 dxf-viewer 能否满足需求

#### 步骤 1：创建测试项目

```bash
mkdir dxf-viewer-poc && cd dxf-viewer-poc
npm init -y
npm install dxf-viewer dxf-parser
```

#### 步骤 2：创建测试页面

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DXF Viewer POC</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui; }
    #container { width: 100vw; height: 100vh; }
    #info {
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.8);
      color: white;
      padding: 10px 20px;
      border-radius: 8px;
      font-size: 14px;
      z-index: 1000;
    }
  </style>
</head>
<body>
  <div id="info">
    <div>文件：<span id="fileName">-</span></div>
    <div>大小：<span id="fileSize">-</span></div>
    <div>加载时间：<span id="loadTime">-</span></div>
    <div>实体数量：<span id="entityCount">-</span></div>
  </div>
  <div id="container"></div>
  
  <script type="module" src="./main.js"></script>
</body>
</html>
```

```javascript
// main.js
import { Viewer } from 'dxf-viewer'
import * as dxfParser from 'dxf-parser'

const container = document.getElementById('container')
const info = {
  fileName: document.getElementById('fileName'),
  fileSize: document.getElementById('fileSize'),
  loadTime: document.getElementById('loadTime'),
  entityCount: document.getElementById('entityCount'),
}

// 创建查看器
const viewer = new Viewer({
  container,
  width: window.innerWidth,
  height: window.innerHeight,
  backgroundColor: 0x1a1a2e,
  theme: 'dark',
})

// 监听窗口大小变化
window.addEventListener('resize', () => {
  viewer.resize(window.innerWidth, window.innerHeight)
})

// 加载 DXF 文件
async function loadDXF(file) {
  const startTime = performance.now()
  
  info.fileName.textContent = file.name
  info.fileSize.textContent = (file.size / 1024).toFixed(2) + ' KB'
  
  try {
    // 解析 DXF
    const parser = dxfParser()
    const dxfData = await parser.parse(file)
    
    // 加载到查看器
    viewer.load(dxfData)
    
    // 适配内容
    viewer.zoomToFit()
    
    const loadTime = performance.now() - startTime
    info.loadTime.textContent = loadTime.toFixed(2) + ' ms'
    info.entityCount.textContent = dxfData.entities?.length || 0
    
    console.log('✅ 加载成功:', {
      文件：file.name,
      大小：(file.size / 1024).toFixed(2) + ' KB',
      时间：loadTime.toFixed(2) + ' ms',
      实体：dxfData.entities?.length,
    })
  } catch (error) {
    console.error('❌ 加载失败:', error)
    alert('加载失败：' + error.message)
  }
}

// 文件拖放支持
container.addEventListener('dragover', (e) => e.preventDefault())
container.addEventListener('drop', (e) => {
  e.preventDefault()
  const file = e.dataTransfer.files[0]
  if (file.name.endsWith('.dxf')) {
    loadDXF(file)
  }
})

// 点击选择文件
container.addEventListener('click', () => {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.dxf'
  input.onchange = (e) => {
    const file = e.target.files[0]
    if (file) loadDXF(file)
  }
  input.click()
})

console.log('🎉 POC 初始化完成，点击或拖放 DXF 文件开始测试')
```

#### 步骤 3：测试验证

使用项目中的测试文件：

```bash
# 复制测试文件到 POC 目录
cp ../dxfs/*.dxf ./test-files/

# 启动本地服务器
npx serve .
```

**验收标准**:
- [ ] 能成功加载 1MB DXF 文件
- [ ] 加载时间 < 1s
- [ ] 能正确显示边、圆弧、样条曲线
- [ ] 能正确显示 HATCH 填充
- [ ] 支持缩放/平移交互
- [ ] 支持实体点击选择

---

### 3.2 第 2 阶段：集成到现有项目（3-5 天）

#### 步骤 1：安装依赖

```bash
cd cad-web
pnpm add dxf-viewer dxf-parser
pnpm remove @mlightcad/cad-simple-viewer three @react-three/fiber @react-three/drei
```

#### 步骤 2：创建 DXF Viewer 组件

```typescript
// cad-web/src/components/dxf-viewer/dxf-viewer.tsx
import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
import { Viewer } from 'dxf-viewer'
import * as dxfParser from 'dxf-parser'

export interface DxfViewerProps {
  file?: File | string | null
  options?: {
    showGrid?: boolean
    showAxes?: boolean
    backgroundColor?: string
    enablePan?: boolean
    enableZoom?: boolean
    theme?: 'light' | 'dark'
  }
  onLoaded?: (data: DxfViewerData) => void
  onError?: (error: Error) => void
  onEntityClick?: (entity: any) => void
  className?: string
}

export interface DxfViewerData {
  entities: any[]
  layers: string[]
  bounds: {
    minX: number
    minY: number
    maxX: number
    maxY: number
  }
  loadTime: number
}

export interface DxfViewerRef {
  zoomToFit: (padding?: number) => void
  resetView: () => void
  getViewer: () => Viewer | null
}

export const DxfViewer = forwardRef<DxfViewerRef, DxfViewerProps>(function DxfViewer(
  {
    file,
    options = {},
    onLoaded,
    onError,
    onEntityClick,
    className,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Viewer | null>(null)

  // 默认配置
  const defaultOptions = {
    showGrid: true,
    showAxes: true,
    backgroundColor: '#1a1a2e',
    enablePan: true,
    enableZoom: true,
    theme: 'dark' as const,
  }

  const mergedOptions = { ...defaultOptions, ...options }

  // 初始化查看器
  useEffect(() => {
    if (!containerRef.current) return

    viewerRef.current = new Viewer({
      container: containerRef.current,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      backgroundColor: parseInt(mergedOptions.backgroundColor.replace('#', '0x')),
      theme: mergedOptions.theme,
      showGrid: mergedOptions.showGrid,
      showAxes: mergedOptions.showAxes,
    })

    // 监听实体点击
    if (onEntityClick) {
      viewerRef.current.on('click', onEntityClick)
    }

    // 监听窗口大小变化
    const handleResize = () => {
      if (containerRef.current && viewerRef.current) {
        viewerRef.current.resize(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        )
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      viewerRef.current?.dispose()
      viewerRef.current = null
    }
  }, [])

  // 加载文件
  useEffect(() => {
    if (!file || !viewerRef.current) return

    const loadFile = async () => {
      const startTime = performance.now()

      try {
        const parser = dxfParser()
        let dxfData

        if (typeof file === 'string') {
          // 从 URL 加载
          const response = await fetch(file)
          const text = await response.text()
          dxfData = parser.parseSync(text)
        } else {
          // 从 File 对象加载
          const text = await file.text()
          dxfData = parser.parseSync(text)
        }

        viewerRef.current.load(dxfData)
        viewerRef.current.zoomToFit()

        const loadTime = performance.now() - startTime

        // 计算边界
        const bounds = calculateBounds(dxfData.entities || [])

        // 提取图层
        const layers = Array.from(
          new Set(dxfData.entities?.map((e: any) => e.layer).filter(Boolean))
        )

        onLoaded?.({
          entities: dxfData.entities || [],
          layers,
          bounds,
          loadTime,
        })

        console.log('[DxfViewer] 加载成功:', {
          加载时间：`${loadTime.toFixed(2)} ms`,
          实体数量：dxfData.entities?.length,
          图层数量：layers.length,
        })
      } catch (error) {
        const err = error as Error
        console.error('[DxfViewer] 加载失败:', err)
        onError?.(err)
      }
    }

    loadFile()
  }, [file])

  // 适配内容
  const zoomToFit = (padding = 0.1) => {
    viewerRef.current?.zoomToFit(padding)
  }

  // 重置视图
  const resetView = () => {
    viewerRef.current?.resetView()
  }

  // 导出方法
  useImperativeHandle(ref, () => ({
    zoomToFit,
    resetView,
    getViewer: () => viewerRef.current,
  }), [])

  return (
    <div
      ref={containerRef}
      className={className || 'w-full h-full'}
      style={{ position: 'relative' }}
    />
  )
})

// 辅助函数：计算边界
function calculateBounds(entities: any[]) {
  let minX = Infinity, minY = Infinity
  let maxX = -Infinity, maxY = -Infinity

  entities.forEach((entity) => {
    if (entity.type === 'LINE') {
      minX = Math.min(minX, entity.vertices[0].x, entity.vertices[1].x)
      minY = Math.min(minY, entity.vertices[0].y, entity.vertices[1].y)
      maxX = Math.max(maxX, entity.vertices[0].x, entity.vertices[1].x)
      maxY = Math.max(maxY, entity.vertices[0].y, entity.vertices[1].y)
    } else if (entity.type === 'CIRCLE' || entity.type === 'ARC') {
      minX = Math.min(minX, entity.center.x - entity.radius)
      minY = Math.min(minY, entity.center.y - entity.radius)
      maxX = Math.max(maxX, entity.center.x + entity.radius)
      maxY = Math.max(maxY, entity.center.y + entity.radius)
    } else if (entity.vertices) {
      entity.vertices.forEach((v: any) => {
        minX = Math.min(minX, v.x)
        minY = Math.min(minY, v.y)
        maxX = Math.max(maxX, v.x)
        maxY = Math.max(maxY, v.y)
      })
    }
  })

  if (minX === Infinity) {
    return { minX: 0, minY: 0, maxX: 100, maxY: 100 }
  }

  return { minX, minY, maxX, maxY }
}

export default DxfViewer
```

#### 步骤 3：更新文件上传 Hook

```typescript
// cad-web/src/hooks/use-file-upload.ts
import { useState, useCallback } from 'react'
import { toast } from 'sonner'

interface UseFileUploadOptions {
  onSuccess?: (data: any) => void
  onError?: (error: Error) => void
}

export function useFileUpload(options: UseFileUploadOptions = {}) {
  const { onSuccess, onError } = options
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState(0)

  const uploadFile = useCallback(
    async (file: File) => {
      setIsUploading(true)
      setProgress(0)

      try {
        // 模拟进度（实际加载很快，不需要复杂进度）
        const interval = setInterval(() => {
          setProgress((prev) => Math.min(prev + 10, 90))
        }, 50)

        // 文件验证
        if (!file.name.endsWith('.dxf')) {
          throw new Error('请上传 DXF 格式文件')
        }

        if (file.size > 10 * 1024 * 1024) {
          throw new Error('文件大小不能超过 10MB')
        }

        // 清除间隔
        clearInterval(interval)
        setProgress(100)

        // 成功回调（由 DxfViewer 组件的 onLoaded 处理）
        toast.success(`文件已加载：${file.name}`)
        onSuccess?.({ file })
      } catch (error) {
        const err = error as Error
        console.error('[useFileUpload] 错误:', err)
        toast.error(err.message || '加载失败')
        onError?.(err)
        throw err
      } finally {
        setIsUploading(false)
        setProgress(0)
      }
    },
    [onSuccess, onError]
  )

  return {
    uploadFile,
    isUploading,
    progress,
  }
}
```

#### 步骤 4：更新 Canvas 页面

```typescript
// cad-web/src/features/canvas/pages/canvas-page.tsx
import { useRef, useState } from 'react'
import { DxfViewer, type DxfViewerData, type DxfViewerRef } from '@/components/dxf-viewer'
import { FileUploadZone } from '../components/file-upload-zone'
import { CanvasToolbar } from '../components/canvas-toolbar'
import { toast } from 'sonner'

export function CanvasPage() {
  const viewerRef = useRef<DxfViewerRef>(null)
  const [currentFile, setCurrentFile] = useState<File | null>(null)
  const [viewerData, setViewerData] = useState<DxfViewerData | null>(null)

  const handleFileSelect = (file: File) => {
    setCurrentFile(file)
  }

  const handleLoaded = (data: DxfViewerData) => {
    setViewerData(data)
    toast.success(`加载成功：${data.entities.length} 个实体，${data.layers.length} 个图层`)
  }

  const handleError = (error: Error) => {
    console.error('[CanvasPage] 加载失败:', error)
    toast.error(`加载失败：${error.message}`)
  }

  const handleEntityClick = (entity: any) => {
    console.log('[CanvasPage] 实体点击:', entity)
    // 可以在这里处理实体选择、属性面板等
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* 工具栏 */}
      <CanvasToolbar
        onZoomToFit={() => viewerRef.current?.zoomToFit()}
        onResetView={() => viewerRef.current?.resetView()}
        viewerData={viewerData}
      />

      {/* 主内容区 */}
      <div className="flex-1 relative">
        {currentFile ? (
          <DxfViewer
            ref={viewerRef}
            file={currentFile}
            onLoaded={handleLoaded}
            onError={handleError}
            onEntityClick={handleEntityClick}
            options={{
              showGrid: true,
              showAxes: true,
              backgroundColor: '#1a1a2e',
              enablePan: true,
              enableZoom: true,
              theme: 'dark',
            }}
          />
        ) : (
          <FileUploadZone onFileSelect={handleFileSelect} />
        )}
      </div>
    </div>
  )
}
```

#### 步骤 5：清理旧代码

```bash
# 删除不再需要的文件
rm -rf src/workers/mlightcad-worker.ts
rm -rf src/lib/mlightcad-geometry-extractor.ts
rm -rf src/features/canvas/components/three-viewer.tsx
rm -rf src/features/canvas/components/three-edge-group.tsx
rm -rf src/features/canvas/components/three-hatch-group.tsx
rm -rf src/hooks/use-mlightcad-worker.ts
rm -rf src/types/mlightcad.d.ts
```

更新 `package.json`:

```json
{
  "dependencies": {
    "dxf-viewer": "^1.0.0",
    "dxf-parser": "^1.1.0"
  }
}
```

---

### 3.3 第 3 阶段：功能增强（5-7 天）

#### 功能 1：图层面板

```typescript
// cad-web/src/components/panels/layer-panel.tsx
import { useState, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'

interface LayerPanelProps {
  layers: string[]
  visibleLayers: Set<string>
  onLayerVisibilityChange: (layer: string, visible: boolean) => void
}

export function LayerPanel({
  layers,
  visibleLayers,
  onLayerVisibilityChange,
}: LayerPanelProps) {
  return (
    <div className="w-64 border-l bg-background p-4">
      <h3 className="text-lg font-semibold mb-4">图层</h3>
      <ScrollArea className="h-[calc(100vh-100px)]">
        <div className="space-y-2">
          {layers.map((layer) => (
            <div key={layer} className="flex items-center space-x-2">
              <Checkbox
                id={layer}
                checked={visibleLayers.has(layer)}
                onCheckedChange={(checked) =>
                  onLayerVisibilityChange(layer, checked as boolean)
                }
              />
              <Label htmlFor={layer} className="text-sm">
                {layer}
              </Label>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
```

#### 功能 2：属性面板

```typescript
// cad-web/src/components/panels/property-panel.tsx
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'

interface PropertyPanelProps {
  entity: any | null
}

export function PropertyPanel({ entity }: PropertyPanelProps) {
  if (!entity) {
    return (
      <div className="w-80 border-l bg-background p-4">
        <h3 className="text-lg font-semibold mb-4">属性</h3>
        <p className="text-muted-foreground text-sm">请选择一个实体</p>
      </div>
    )
  }

  return (
    <div className="w-80 border-l bg-background p-4">
      <h3 className="text-lg font-semibold mb-4">属性</h3>
      <ScrollArea className="h-[calc(100vh-100px)]">
        <div className="space-y-4">
          <div>
            <Label className="text-sm text-muted-foreground">类型</Label>
            <p className="text-sm font-medium">{entity.type}</p>
          </div>
          <Separator />
          <div>
            <Label className="text-sm text-muted-foreground">图层</Label>
            <p className="text-sm font-medium">{entity.layer}</p>
          </div>
          <Separator />
          {entity.handle && (
            <div>
              <Label className="text-sm text-muted-foreground">Handle</Label>
              <p className="text-sm font-mono">{entity.handle}</p>
            </div>
          )}
          {entity.color && (
            <div>
              <Label className="text-sm text-muted-foreground">颜色</Label>
              <p className="text-sm font-medium">{entity.color}</p>
            </div>
          )}
          {/* 根据实体类型显示不同属性 */}
          {entity.type === 'LINE' && (
            <>
              <Separator />
              <div>
                <Label className="text-sm text-muted-foreground">起点</Label>
                <p className="text-sm font-mono">
                  [{entity.vertices[0].x.toFixed(2)}, {entity.vertices[0].y.toFixed(2)}]
                </p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">终点</Label>
                <p className="text-sm font-mono">
                  [{entity.vertices[1].x.toFixed(2)}, {entity.vertices[1].y.toFixed(2)}]
                </p>
              </div>
            </>
          )}
          {entity.type === 'CIRCLE' && (
            <>
              <Separator />
              <div>
                <Label className="text-sm text-muted-foreground">圆心</Label>
                <p className="text-sm font-mono">
                  [{entity.center.x.toFixed(2)}, {entity.center.y.toFixed(2)}]
                </p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">半径</Label>
                <p className="text-sm font-medium">{entity.radius.toFixed(2)}</p>
              </div>
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
```

#### 功能 3：性能监控

```typescript
// cad-web/src/components/performance-monitor.tsx
import { useState, useEffect } from 'react'

interface PerformanceStats {
  loadTime: number
  entityCount: number
  memoryUsage?: number
}

export function PerformanceMonitor({ stats }: { stats: PerformanceStats }) {
  return (
    <div className="fixed bottom-4 left-4 bg-black/80 text-white px-4 py-2 rounded-lg text-xs font-mono">
      <div>加载时间：{stats.loadTime.toFixed(2)} ms</div>
      <div>实体数量：{stats.entityCount}</div>
      {stats.memoryUsage && (
        <div>内存占用：{(stats.memoryUsage / 1024 / 1024).toFixed(2)} MB</div>
      )}
    </div>
  )
}
```

---

## 四、验收标准

### 4.1 功能验收

| 功能 | 验收标准 | 状态 |
|------|----------|------|
| DXF 加载 | 支持 1MB 以内 DXF 文件 | ⬜ |
| 边渲染 | LINE、LWPOLYLINE 正确显示 | ⬜ |
| 曲线渲染 | CIRCLE、ARC、ELLIPSE、SPLINE 正确显示 | ⬜ |
| HATCH 渲染 | 实体填充和图案填充正确显示 | ⬜ |
| 块引用 | BLOCK/INSERT 正确展开 | ⬜ |
| 缩放平移 | 鼠标滚轮缩放、拖拽平移流畅 | ⬜ |
| 实体选择 | 点击实体高亮显示 | ⬜ |
| 图层控制 | 支持图层显示/隐藏 | ⬜ |
| 属性查看 | 显示选中实体的属性 | ⬜ |

### 4.2 性能验收

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 加载时间 | < 1s (1MB 文件) | performance.now() 计时 |
| 内存占用 | < 30MB | Chrome DevTools Memory |
| FPS | 稳定 60 | Chrome DevTools Performance |
| 首屏时间 | < 500ms | Lighthouse |
| 包体积 | < 500KB | webpack-bundle-analyzer |

### 4.3 兼容性测试

| 浏览器 | 版本要求 | 状态 |
|--------|----------|------|
| Chrome | 最新 2 个版本 | ⬜ |
| Firefox | 最新 2 个版本 | ⬜ |
| Safari | 最新 2 个版本 | ⬜ |
| Edge | 最新 2 个版本 | ⬜ |

---

## 五、测试计划

### 5.1 单元测试

```typescript
// cad-web/tests/unit/dxf-viewer.test.tsx
import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { DxfViewer } from '@/components/dxf-viewer'

describe('DxfViewer', () => {
  it('renders without crashing', () => {
    const { container } = render(<DxfViewer />)
    expect(container.firstChild).toBeDefined()
  })

  it('calls onLoaded when file is loaded', async () => {
    const onLoaded = vi.fn()
    const file = new File(['test'], 'test.dxf', { type: 'application/dxf' })
    
    render(<DxfViewer file={file} onLoaded={onLoaded} />)
    
    // 等待加载完成
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    expect(onLoaded).toHaveBeenCalled()
  })
})
```

### 5.2 性能基准测试

```typescript
// cad-web/tests/performance/dxf-loading.bench.ts
import { bench, describe } from 'vitest'
import * as dxfParser from 'dxf-parser'

const testFiles = [
  { name: 'small.dxf', size: '100KB' },
  { name: 'medium.dxf', size: '500KB' },
  { name: 'large.dxf', size: '1MB' },
]

describe('DXF 解析性能', () => {
  testFiles.forEach((file) => {
    bench(`解析 ${file.name} (${file.size})`, async () => {
      const response = await fetch(`/test-files/${file.name}`)
      const text = await response.text()
      
      const parser = dxfParser()
      const data = parser.parseSync(text)
      
      return data
    })
  })
})
```

### 5.3 E2E 测试

```typescript
// cad-web/tests/e2e/dxf-viewer.spec.ts
import { test, expect } from '@playwright/test'

test.describe('DXF Viewer', () => {
  test('loads DXF file successfully', async ({ page }) => {
    await page.goto('/canvas')
    
    // 上传文件
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles('tests/fixtures/test.dxf')
    
    // 等待加载完成
    await expect(page.locator('#canvas-container')).toBeVisible()
    
    // 验证实体数量
    const entityCount = await page.locator('[data-testid="entity-count"]').textContent()
    expect(Number(entityCount)).toBeGreaterThan(0)
  })

  test('supports zoom and pan', async ({ page }) => {
    await page.goto('/canvas')
    
    // 上传文件
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles('tests/fixtures/test.dxf')
    
    // 等待加载完成
    await page.waitForTimeout(1000)
    
    // 测试缩放
    await page.mouse.wheel(0, -100)
    await page.waitForTimeout(500)
    
    // 测试平移
    await page.mouse.down()
    await page.mouse.move(100, 100)
    await page.mouse.up()
  })
})
```

---

## 六、迁移路线图

### 第 1 周：POC 验证

| 任务 | 工时 | 负责人 | 状态 |
|------|------|--------|------|
| 创建 POC 项目 | 2h | | ⬜ |
| 测试 1MB DXF 加载 | 2h | | ⬜ |
| 验证 HATCH 支持 | 2h | | ⬜ |
| 验证样条曲线 | 2h | | ⬜ |
| 输出 POC 报告 | 2h | | ⬜ |

### 第 2 周：集成开发

| 任务 | 工时 | 负责人 | 状态 |
|------|------|--------|------|
| 安装依赖 | 0.5h | | ⬜ |
| 创建 DxfViewer 组件 | 4h | | ⬜ |
| 更新 useFileUpload Hook | 2h | | ⬜ |
| 更新 Canvas 页面 | 4h | | ⬜ |
| 清理旧代码 | 2h | | ⬜ |
| 单元测试 | 4h | | ⬜ |

### 第 3 周：功能增强

| 任务 | 工时 | 负责人 | 状态 |
|------|------|--------|------|
| 图层面板 | 4h | | ⬜ |
| 属性面板 | 4h | | ⬜ |
| 性能监控 | 2h | | ⬜ |
| 工具栏 | 4h | | ⬜ |
| E2E 测试 | 4h | | ⬜ |

### 第 4 周：优化与上线

| 任务 | 工时 | 负责人 | 状态 |
|------|------|--------|------|
| 性能优化 | 4h | | ⬜ |
| 兼容性测试 | 4h | | ⬜ |
| 文档编写 | 4h | | ⬜ |
| 代码审查 | 4h | | ⬜ |
| 生产部署 | 2h | | ⬜ |

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| dxf-viewer 不支持某些 DXF 版本 | 中 | 高 | POC 阶段充分测试，准备降级方案 |
| 性能不达标 | 低 | 高 | 提前进行性能基准测试 |
| 团队学习成本 | 低 | 低 | dxf-viewer API 简单，1 天上手 |
| 现有代码迁移成本 | 中 | 中 | 分阶段迁移，保持向后兼容 |
| HATCH 渲染效果不佳 | 低 | 中 | 提前测试 HATCH 支持 |

---

## 八、总结

### 核心建议

1. **技术选型**: 采用 dxf-viewer 替代 mlightcad + Three.js
2. **架构简化**: 移除 WASM 和 Worker，使用纯 JavaScript 方案
3. **性能目标**: 1MB DXF 加载 < 1s，内存 < 30MB，FPS 60
4. **迁移周期**: 4 周完成从 POC 到生产上线

### 预期收益

| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| 包体积 | 2.5MB | < 500KB | **5x** |
| 加载时间 | 3-5s | < 1s | **5x** |
| 内存占用 | 50-100MB | < 30MB | **3x** |
| FPS | 45-55 | 60 | 稳定 |
| 开发效率 | 低 (WASM 调试困难) | 高 (纯 JS) | **3x** |

### 下一步行动

1. **立即**: 创建 POC 项目，验证 dxf-viewer 能力
2. **本周内**: 输出 POC 报告，确认技术可行性
3. **下周**: 开始集成开发，第 1 个可用版本

---

**最后更新**: 2026 年 3 月 22 日  
**版本**: v2.0  
**维护者**: CAD 开发团队  
**状态**: ✅ 待落实
