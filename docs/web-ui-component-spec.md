# Web UI 组件设计规范

**版本**: v0.1.0
**创建日期**: 2026 年 3 月 21 日
**基于**: shadcn/ui + Radix UI + Tailwind CSS

---

## 一、设计原则

### 1.1 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                    设计哲学                                  │
├─────────────────────────────────────────────────────────────┤
│  1. 一致性 (Consistency)                                    │
│     - 相同的交互模式                                        │
│     - 相同的视觉语言                                        │
│     - 相同的反馈机制                                        │
│                                                              │
│  2. 可用性 (Usability)                                      │
│     - 直观易懂                                              │
│     - 高效操作                                              │
│     - 容错性强                                              │
│                                                              │
│  3. 可访问性 (Accessibility)                                │
│     - 键盘导航完整支持                                      │
│     - 屏幕阅读器友好                                        │
│     - 对比度符合 WCAG 2.1 AA                                 │
│                                                              │
│  4. 性能 (Performance)                                      │
│     - 快速响应                                              │
│     - 流畅动画                                              │
│     - 按需加载                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 组件分类

```
组件
├── 基础组件 (UI Primitives)
│   ├── Button
│   ├── Input
│   ├── Select
│   ├── Dialog
│   └── ...
│
├── 业务组件 (Business Components)
│   ├── CanvasViewer
│   ├── LayerPanel
│   ├── PropertyPanel
│   └── ...
│
└── 复合组件 (Composite Components)
    ├── Toolbar
    ├── Sidebar
    └── Modal
```

---

## 二、基础组件规范

### 2.1 Button 组件

```typescript
// components/ui/button.tsx
import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  // 基础样式
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors ' +
  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ' +
  'disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground shadow hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90',
        outline: 'border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-8',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, loading, leftIcon, rightIcon, children, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={loading || props.disabled}
        {...props}
      >
        {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        {!loading && leftIcon && <span className="mr-2">{leftIcon}</span>}
        {children}
        {!loading && rightIcon && <span className="ml-2">{rightIcon}</span>}
      </Comp>
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }
```

**使用示例**:

```tsx
// 主要按钮
<Button onClick={handleOpen}>
  <FolderOpen className="h-4 w-4" />
  打开文件
</Button>

// 加载状态
<Button loading onClick={handleSave}>
  保存
</Button>

// 图标按钮
<Button size="icon" variant="ghost">
  <Settings className="h-4 w-4" />
</Button>

// 破坏性操作
<Button variant="destructive" onClick={handleDelete}>
  删除
</Button>
```

---

### 2.2 Dialog 组件

```typescript
// components/ui/dialog.tsx
import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

const Dialog = DialogPrimitive.Root
const DialogTrigger = DialogPrimitive.Trigger
const DialogPortal = DialogPrimitive.Portal
const DialogClose = DialogPrimitive.Close

const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      'fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out ' +
      'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className
    )}
    {...props}
  />
))
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
    showCloseButton?: boolean
  }
>(({ className, children, showCloseButton = true, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 ' +
        'border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in ' +
        'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 ' +
        'data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 ' +
        'data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 ' +
        'data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg',
        className
      )}
      {...props}
    >
      {children}
      {showCloseButton && (
        <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
          <X className="h-4 w-4" />
          <span className="sr-only">关闭</span>
        </DialogPrimitive.Close>
      )}
    </DialogPrimitive.Content>
  </DialogPortal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName

const DialogHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex flex-col space-y-1.5 text-center sm:text-left', className)} {...props} />
)
DialogHeader.displayName = 'DialogHeader'

const DialogFooter = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2', className)} {...props} />
)
DialogFooter.displayName = 'DialogFooter'

const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn('text-lg font-semibold leading-none tracking-tight', className)}
    {...props}
  />
))
DialogTitle.displayName = DialogPrimitive.Title.displayName

const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn('text-sm text-muted-foreground', className)}
    {...props}
  />
))
DialogDescription.displayName = DialogPrimitive.Description.displayName

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogClose,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
}
```

**使用示例**:

```tsx
// 文件上传对话框
export function FileUploadDialog() {
  const [open, setOpen] = React.useState(false)
  const { uploadFile } = useFileUpload()
  
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button leftIcon={<Upload />}>
          上传文件
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>上传 CAD 文件</DialogTitle>
          <DialogDescription>
            支持 DXF 和 PDF 格式文件
          </DialogDescription>
        </DialogHeader>
        
        <FileUploadDropzone 
          accept={{
            'application/dxf': ['.dxf'],
            'application/pdf': ['.pdf']
          }}
          onUpload={async (file) => {
            await uploadFile(file)
            setOpen(false)
          }}
        />
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            取消
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
```

---

### 2.3 Select 组件

```typescript
// components/ui/select.tsx
import * as React from 'react'
import * as SelectPrimitive from '@radix-ui/react-select'
import { Check, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'

const Select = SelectPrimitive.Root
const SelectGroup = SelectPrimitive.Group
const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      'flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border ' +
      'border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background ' +
      'placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring ' +
      'disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1',
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    className={cn('flex cursor-default items-center justify-center py-1', className)}
    {...props}
  >
    <ChevronUp className="h-4 w-4" />
  </SelectPrimitive.ScrollUpButton>
))
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    className={cn('flex cursor-default items-center justify-center py-1', className)}
    {...props}
  >
    <ChevronDown className="h-4 w-4" />
  </SelectPrimitive.ScrollDownButton>
))
SelectScrollDownButton.displayName = SelectPrimitive.ScrollDownButton.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = 'popper', ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        'relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover ' +
        'text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out ' +
        'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 ' +
        'data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 ' +
        'data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
        position === 'popper' &&
          'data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 ' +
          'data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1',
        className
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        className={cn(
          'p-1',
          position === 'popper' &&
            'h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]'
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn('px-2 py-1.5 text-sm font-semibold', className)}
    {...props}
  />
))
SelectLabel.displayName = SelectPrimitive.Label.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      'relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-2 pr-8 text-sm ' +
      'outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none ' +
      'data-[disabled]:opacity-50',
      className
    )}
    {...props}
  >
    <span className="absolute right-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn('-mx-1 my-1 h-px bg-muted', className)}
    {...props}
  />
))
SelectSeparator.displayName = SelectPrimitive.Separator.displayName

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
}
```

**使用示例**:

```tsx
// 语义标注选择器
export function SemanticSelect({ edgeId }: { edgeId: number }) {
  const { setSemantic } = useBoundaryAnnotation()
  
  return (
    <Select onValueChange={(value) => setSemantic(edgeId, value as BoundarySemantic)}>
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="选择语义" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>边界类型</SelectLabel>
          <SelectItem value="hard_wall">硬墙</SelectItem>
          <SelectItem value="absorptive_wall">吸声墙</SelectItem>
          <SelectItem value="opening">开口</SelectItem>
          <SelectItem value="window">窗户</SelectItem>
          <SelectItem value="door">门</SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}
```

---

## 三、业务组件规范

### 3.1 Canvas Viewer 组件

```typescript
// features/canvas/components/canvas-viewer.tsx
import React, { useCallback, useRef } from 'react'
import { Stage, Layer } from 'react-konva'
import { EdgeLayer } from './edge-layer'
import { SelectionLayer } from './selection-layer'
import { InteractionLayer } from './interaction-layer'
import { CanvasToolbar } from './canvas-toolbar'
import { CanvasOverlay } from './canvas-overlay'
import { useCanvasStore } from '@/stores/canvas-store'
import { cn } from '@/lib/utils'

export interface CanvasViewerProps {
  className?: string
}

export function CanvasViewer({ className }: CanvasViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = React.useState({ width: 800, height: 600 })
  
  const {
    edges,
    selection,
    camera,
    gaps,
    traceResult,
    selectEdge,
    autoTrace,
    setCamera,
  } = useCanvasStore()
  
  // 响应式尺寸
  React.useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        })
      }
    })
    
    if (containerRef.current) {
      observer.observe(containerRef.current)
    }
    
    return () => observer.disconnect()
  }, [])
  
  // 处理边选择
  const handleEdgeClick = useCallback((edgeId: number) => {
    selectEdge(edgeId)
  }, [selectEdge])
  
  // 处理自动追踪
  const handleEdgeDoubleClick = useCallback((edgeId: number) => {
    autoTrace(edgeId)
  }, [autoTrace])
  
  // 处理画布缩放
  const handleWheel = useCallback((e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    
    const scaleBy = 1.05
    const stage = e.target.getStage()
    if (!stage) return
    
    const oldScale = camera.zoom
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    }
    
    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy
    const clampedScale = Math.max(0.1, Math.min(10, newScale))
    
    setCamera({
      ...camera,
      zoom: clampedScale,
      offsetX: pointer.x - mousePointTo.x * clampedScale,
      offsetY: pointer.y - mousePointTo.y * clampedScale,
    })
  }, [camera, setCamera])
  
  return (
    <div 
      ref={containerRef} 
      className={cn('relative w-full h-full bg-canvas-background overflow-hidden', className)}
    >
      {/* 工具栏 */}
      <CanvasToolbar className="absolute top-4 left-1/2 -translate-x-1/2 z-10" />
      
      {/* Konva 画布 */}
      <Stage 
        width={size.width} 
        height={size.height}
        onWheel={handleWheel}
      >
        {/* 边图层 */}
        <EdgeLayer 
          edges={edges}
          camera={camera}
          onEdgeClick={handleEdgeClick}
          onEdgeDoubleClick={handleEdgeDoubleClick}
        />
        
        {/* 选择高亮图层 */}
        <SelectionLayer 
          selection={selection}
          traceResult={traceResult}
        />
        
        {/* 交互图层 */}
        <InteractionLayer />
      </Stage>
      
      {/* 覆盖层（缺口标记等） */}
      <CanvasOverlay gaps={gaps} />
      
      {/* 加载状态 */}
      <LoadingOverlay />
    </div>
  )
}
```

---

### 3.2 Layer Panel 组件

```typescript
// features/canvas/components/layer-panel.tsx
import React from 'react'
import { Eye, EyeOff, Lock, Unlock } from 'lucide-react'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'
import { useLayerStore } from '@/stores/layer-store'
import { cn } from '@/lib/utils'

export interface LayerPanelProps {
  className?: string
}

export function LayerPanel({ className }: LayerPanelProps) {
  const { layers, toggleVisibility, toggleLock, selectedLayer, setSelectedLayer } = useLayerStore()
  
  return (
    <div className={cn('w-64 h-full border-r bg-background overflow-auto', className)}>
      <div className="p-4 border-b">
        <h2 className="text-sm font-semibold">图层</h2>
      </div>
      
      <Accordion type="single" collapsible className="w-full">
        {layers.map((layer) => (
          <AccordionItem 
            key={layer.id} 
            value={layer.id}
            className={cn(
              'border-b-0',
              selectedLayer === layer.id && 'bg-accent'
            )}
          >
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={layer.visible}
                  onCheckedChange={() => toggleVisibility(layer.id)}
                  onClick={(e) => e.stopPropagation()}
                />
                <span className="text-sm">{layer.name}</span>
                <span className="text-xs text-muted-foreground">
                  ({layer.count})
                </span>
              </div>
            </AccordionTrigger>
            
            <AccordionContent className="px-4 pb-3">
              <div className="flex items-center gap-2">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8"
                  onClick={() => toggleLock(layer.id)}
                >
                  {layer.locked ? (
                    <Lock className="h-4 w-4" />
                  ) : (
                    <Unlock className="h-4 w-4" />
                  )}
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8"
                  onClick={() => setSelectedLayer(layer.id)}
                >
                  选择所有
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  )
}
```

---

### 3.3 Property Panel 组件

```typescript
// features/canvas/components/property-panel.tsx
import React from 'react'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { useSelectionStore } from '@/stores/selection-store'
import { cn } from '@/lib/utils'

export interface PropertyPanelProps {
  className?: string
}

export function PropertyPanel({ className }: PropertyPanelProps) {
  const { selectedEdge, updateEdgeSemantic, updateEdgeMaterial } = useSelectionStore()
  
  if (!selectedEdge) {
    return (
      <div className={cn('w-72 h-full border-l bg-background p-4', className)}>
        <div className="flex items-center justify-center h-full text-muted-foreground">
          <p className="text-sm">选择一条边以查看属性</p>
        </div>
      </div>
    )
  }
  
  return (
    <div className={cn('w-72 h-full border-l bg-background overflow-auto', className)}>
      <div className="p-4 border-b">
        <h2 className="text-sm font-semibold">属性</h2>
      </div>
      
      <div className="p-4 space-y-4">
        {/* 基本信息 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">基本信息</h3>
          <div className="space-y-2">
            <div>
              <Label htmlFor="edge-id">边 ID</Label>
              <Input id="edge-id" value={selectedEdge.id} disabled />
            </div>
            <div>
              <Label htmlFor="edge-length">长度</Label>
              <Input 
                id="edge-length" 
                value={`${selectedEdge.length.toFixed(2)} mm`} 
                disabled 
              />
            </div>
            <div>
              <Label htmlFor="edge-layer">图层</Label>
              <Input id="edge-layer" value={selectedEdge.layer} disabled />
            </div>
          </div>
        </section>
        
        <Separator />
        
        {/* 语义标注 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">语义标注</h3>
          <div className="space-y-2">
            <div>
              <Label htmlFor="semantic">类型</Label>
              <Select
                value={selectedEdge.semantic}
                onValueChange={(value) => updateEdgeSemantic(selectedEdge.id, value)}
              >
                <SelectTrigger id="semantic">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hard_wall">硬墙</SelectItem>
                  <SelectItem value="absorptive_wall">吸声墙</SelectItem>
                  <SelectItem value="opening">开口</SelectItem>
                  <SelectItem value="window">窗户</SelectItem>
                  <SelectItem value="door">门</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="material">材料</Label>
              <Select
                value={selectedEdge.material}
                onValueChange={(value) => updateEdgeMaterial(selectedEdge.id, value)}
              >
                <SelectTrigger id="material">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="concrete">混凝土</SelectItem>
                  <SelectItem value="brick">砖墙</SelectItem>
                  <SelectItem value="glass">玻璃</SelectItem>
                  <SelectItem value="wood">木材</SelectItem>
                  <SelectItem value="metal">金属</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>
        
        <Separator />
        
        {/* 几何信息 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">几何信息</h3>
          <div className="space-y-2">
            <div>
              <Label>起点</Label>
              <div className="text-sm font-mono">
                [{selectedEdge.start[0].toFixed(2)}, {selectedEdge.start[1].toFixed(2)}]
              </div>
            </div>
            <div>
              <Label>终点</Label>
              <div className="text-sm font-mono">
                [{selectedEdge.end[0].toFixed(2)}, {selectedEdge.end[1].toFixed(2)}]
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
```

---

## 四、工具栏规范

### 4.1 Toolbar 组件

```typescript
// components/toolbar/toolbar.tsx
import React from 'react'
import { cn } from '@/lib/utils'

export interface ToolbarProps {
  className?: string
  children: React.ReactNode
}

export function Toolbar({ className, children }: ToolbarProps) {
  return (
    <div 
      className={cn(
        'flex items-center gap-1 p-1 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b',
        className
      )}
    >
      {children}
    </div>
  )
}

export interface ToolbarGroupProps {
  className?: string
  label?: string
  children: React.ReactNode
}

export function ToolbarGroup({ className, label, children }: ToolbarGroupProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      {label && (
        <span className="text-xs text-muted-foreground px-2">{label}</span>
      )}
      {children}
    </div>
  )
}

export interface ToolbarSeparatorProps {
  className?: string
}

export function ToolbarSeparator({ className }: ToolbarSeparatorProps) {
  return (
    <div 
      className={cn('w-px h-6 bg-border mx-1', className)} 
    />
  )
}

export interface ToolButtonProps {
  icon: React.ReactNode
  label: string
  shortcut?: string
  active?: boolean
  disabled?: boolean
  onClick?: () => void
}

export function ToolButton({ 
  icon, 
  label, 
  shortcut, 
  active, 
  disabled, 
  onClick 
}: ToolButtonProps) {
  return (
    <button
      className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        active 
          ? 'bg-accent text-accent-foreground' 
          : 'hover:bg-accent/50'
      )}
      onClick={onClick}
      disabled={disabled}
      title={shortcut ? `${label} (${shortcut})` : label}
    >
      {icon}
      <span className="hidden lg:inline">{label}</span>
      {shortcut && (
        <kbd className="hidden lg:inline-flex h-5 items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
          {shortcut}
        </kbd>
      )}
    </button>
  )
}
```

---

## 五、通知系统规范

### 5.1 Toast 组件

```typescript
// components/ui/toast.tsx
import React from 'react'
import * as ToastPrimitives from '@radix-ui/react-toast'
import { cva, type VariantProps } from 'class-variance-authority'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

const ToastProvider = ToastPrimitives.Provider

const ToastViewport = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Viewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    ref={ref}
    className={cn(
      'fixed top-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]',
      className
    )}
    {...props}
  />
))
ToastViewport.displayName = ToastPrimitives.Viewport.displayName

const toastVariants = cva(
  'group pointer-events-auto relative flex w-full items-center justify-between space-x-2 overflow-hidden rounded-md border p-4 pr-6 shadow-lg transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full',
  {
    variants: {
      variant: {
        default: 'border bg-background text-foreground',
        destructive:
          'destructive group border-destructive bg-destructive text-destructive-foreground',
        success:
          'border-green-500 bg-green-500 text-white',
        info:
          'border-blue-500 bg-blue-500 text-white',
        warning:
          'border-yellow-500 bg-yellow-500 text-white',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Root> &
    VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => {
  return (
    <ToastPrimitives.Root
      ref={ref}
      className={cn(toastVariants({ variant }), className)}
      {...props}
    />
  )
})
Toast.displayName = ToastPrimitives.Root.displayName

const ToastAction = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Action>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Action>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Action
    ref={ref}
    className={cn(
      'inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-1 focus:ring-ring disabled:pointer-events-none disabled:opacity-50 group-[.destructive]:border-muted/40 group-[.destructive]:hover:border-destructive/30 group-[.destructive]:hover:bg-destructive group-[.destructive]:hover:text-destructive-foreground group-[.destructive]:focus:ring-destructive',
      className
    )}
    {...props}
  />
))
ToastAction.displayName = ToastPrimitives.Action.displayName

const ToastClose = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Close>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    ref={ref}
    className={cn(
      'absolute right-1 top-1 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-1 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50 group-[.destructive]:focus:ring-red-400 group-[.destructive]:focus:ring-offset-red-600',
      className
    )}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.Close>
))
ToastClose.displayName = ToastPrimitives.Close.displayName

const ToastTitle = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Title>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    ref={ref}
    className={cn('text-sm font-semibold [&+div]:text-xs', className)}
    {...props}
  />
))
ToastTitle.displayName = ToastPrimitives.Title.displayName

const ToastDescription = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Description>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    ref={ref}
    className={cn('text-sm opacity-90', className)}
    {...props}
  />
))
ToastDescription.displayName = ToastPrimitives.Description.displayName

type ToastProps = React.ComponentPropsWithoutRef<typeof Toast>

type ToastActionElement = React.ReactElement<typeof ToastAction>

export {
  type ToastProps,
  type ToastActionElement,
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
}
```

---

### 5.2 useToast Hook

```typescript
// hooks/use-toast.ts
import React from 'react'
import type { ToastActionElement, ToastProps } from '@/components/ui/toast'

const TOAST_LIMIT = 5
const TOAST_REMOVE_DELAY = 5000

type ToasterToast = ToastProps & {
  id: string
  title?: React.ReactNode
  description?: React.ReactNode
  action?: ToastActionElement
}

const actionTypes = {
  ADD_TOAST: 'ADD_TOAST',
  UPDATE_TOAST: 'UPDATE_TOAST',
  DISMISS_TOAST: 'DISMISS_TOAST',
  REMOVE_TOAST: 'REMOVE_TOAST',
} as const

let count = 0

function genId() {
  count = (count + 1) % Number.MAX_VALUE
  return count.toString()
}

type ActionType = typeof actionTypes

type Action =
  | {
      type: ActionType['ADD_TOAST']
      toast: ToasterToast
    }
  | {
      type: ActionType['UPDATE_TOAST']
      toast: Partial<ToasterToast>
    }
  | {
      type: ActionType['DISMISS_TOAST']
      toastId?: ToasterToast['id']
    }
  | {
      type: ActionType['REMOVE_TOAST']
      toastId?: ToasterToast['id']
    }

interface State {
  toasts: ToasterToast[]
}

const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>()

const addToRemoveQueue = (toastId: string) => {
  if (toastTimeouts.has(toastId)) {
    return
  }

  const timeout = setTimeout(() => {
    toastTimeouts.delete(toastId)
    dispatch({
      type: 'REMOVE_TOAST',
      toastId: toastId,
    })
  }, TOAST_REMOVE_DELAY)

  toastTimeouts.set(toastId, timeout)
}

export const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case 'ADD_TOAST':
      return {
        ...state,
        toasts: [action.toast, ...state.toasts].slice(0, TOAST_LIMIT),
      }

    case 'UPDATE_TOAST':
      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === action.toast.id ? { ...t, ...action.toast } : t
        ),
      }

    case 'DISMISS_TOAST': {
      const { toastId } = action

      if (toastId) {
        addToRemoveQueue(toastId)
      } else {
        state.toasts.forEach((toast) => {
          addToRemoveQueue(toast.id)
        })
      }

      return {
        ...state,
        toasts: state.toasts.map((t) =>
          t.id === toastId || toastId === undefined
            ? {
                ...t,
                open: false,
              }
            : t
        ),
      }
    }
    case 'REMOVE_TOAST':
      if (action.toastId === undefined) {
        return {
          ...state,
          toasts: [],
        }
      }
      return {
        ...state,
        toasts: state.toasts.filter((t) => t.id !== action.toastId),
      }
  }
}

const listeners: Array<(state: State) => void> = []

let memoryState: State = { toasts: [] }

function dispatch(action: Action) {
  memoryState = reducer(memoryState, action)
  listeners.forEach((listener) => {
    listener(memoryState)
  })
}

type Toast = Omit<ToasterToast, 'id'>

function toast({ ...props }: Toast) {
  const id = genId()

  const update = (props: ToasterToast) =>
    dispatch({
      type: 'UPDATE_TOAST',
      toast: { ...props, id },
    })
  const dismiss = () => dispatch({ type: 'DISMISS_TOAST', toastId: id })

  dispatch({
    type: 'ADD_TOAST',
    toast: {
      ...props,
      id,
      open: true,
      onOpenChange: (open) => {
        if (!open) dismiss()
      },
    },
  })

  return {
    id: id,
    dismiss,
    update,
  }
}

function useToast() {
  const [state, setState] = React.useState<State>(memoryState)

  React.useEffect(() => {
    listeners.push(setState)
    return () => {
      const index = listeners.indexOf(setState)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }, [state])

  return {
    ...state,
    toast,
    dismiss: (toastId?: string) => dispatch({ type: 'DISMISS_TOAST', toastId }),
  }
}

export { useToast, toast }
```

**使用示例**:

```tsx
// 在组件中使用
export function FileUploader() {
  const { toast } = useToast()
  
  const handleUpload = async (file: File) => {
    try {
      await uploadFile(file)
      toast({
        title: '上传成功',
        description: '文件已开始处理',
        variant: 'success',
      })
    } catch (error) {
      toast({
        title: '上传失败',
        description: error instanceof Error ? error.message : '未知错误',
        variant: 'destructive',
      })
    }
  }
  
  return <FileUploadDropzone onUpload={handleUpload} />
}
```

---

## 六、主题定制

### 6.1 CSS 变量定义

```css
/* assets/styles/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    /* 浅色主题 */
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    
    --primary: 221.2 83.2% 53.3%;
    --primary-foreground: 210 40% 98%;
    
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 221.2 83.2% 53.3%;
    
    /* CAD 专用颜色 */
    --canvas-background: 220 20% 10%;
    --canvas-grid: 220 20% 16%;
    --canvas-edge: 210 100% 60%;
    --canvas-edge-hover: 210 100% 70%;
    --canvas-edge-selected: 45 100% 50%;
    --canvas-gap: 0 100% 50%;
    --canvas-trace: 142 76% 36%;
    
    --radius: 0.5rem;
  }
  
  .dark {
    /* 深色主题 */
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
    
    --primary: 217.2 91.2% 59.8%;
    --primary-foreground: 222.2 47.4% 11.2%;
    
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 224.3 76.3% 48%;
    
    /* CAD 专用颜色 (深色模式) */
    --canvas-background: 220 20% 8%;
    --canvas-grid: 220 20% 14%;
    --canvas-edge: 210 100% 65%;
    --canvas-edge-hover: 210 100% 75%;
    --canvas-edge-selected: 45 100% 55%;
    --canvas-gap: 0 100% 55%;
    --canvas-trace: 142 76% 40%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

---

## 七、响应式设计

### 7.1 断点定义

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    screens: {
      'sm': '640px',  // 手机横屏
      'md': '768px',  // 平板
      'lg': '1024px', // 小屏笔记本
      'xl': '1280px', // 桌面
      '2xl': '1536px',// 大屏桌面
    }
  }
}
```

### 7.2 响应式布局

```tsx
// 侧边栏响应式
export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false)
  
  return (
    <>
      {/* 移动端抽屉 */}
      <Drawer open={isOpen} onOpenChange={setIsOpen} className="md:hidden">
        <DrawerContent>
          <LayerPanel />
        </DrawerContent>
      </Drawer>
      
      {/* 桌面端固定侧边栏 */}
      <div className="hidden md:block">
        <LayerPanel className="w-64" />
      </div>
      
      {/* 移动端切换按钮 */}
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        onClick={() => setIsOpen(true)}
      >
        <PanelLeft className="h-5 w-5" />
      </Button>
    </>
  )
}
```

---

## 八、可访问性规范

### 8.1 键盘导航

```tsx
// 所有交互组件必须支持键盘操作
export function ToolbarButton({ label, onClick }) {
  return (
    <button
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick()
        }
      }}
      // 焦点可见
      className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      // 无障碍标签
      aria-label={label}
      // 快捷键提示
      title={`${label} (快捷键 T)`}
    >
      {label}
    </button>
  )
}
```

### 8.2 屏幕阅读器

```tsx
// 为图标添加 sr-only 文本
export function IconButton({ icon, label }) {
  return (
    <button aria-label={label}>
      {icon}
      <span className="sr-only">{label}</span>
    </button>
  )
}

// 状态变化通知
export function LoadingState() {
  return (
    <div role="status" aria-live="polite">
      <Loader2 className="animate-spin" />
      <span className="sr-only">加载中...</span>
    </div>
  )
}
```

---

**创建者**: CAD 团队
**最后更新**: 2026 年 3 月 21 日
