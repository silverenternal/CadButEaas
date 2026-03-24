import { FolderOpen, Save, Download, Undo, Redo, FileJson, FileBox } from 'lucide-react'
import { Button } from '@components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@components/ui/tooltip'
import { ToolButton } from './tool-button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@components/ui/dropdown-menu'
import { useAppStore } from '@/stores/app-store'
import { useCanvasStore } from '@/stores/canvas-store'
import { useFileUpload } from '@/hooks/use-file-upload'
import { exportService } from '@/services/export-service'
import { toast } from 'sonner'

export function MainToolbar() {
  const { setTool, activeTool } = useCanvasStore()
  const { toggleSidebar, toggleRightPanel } = useAppStore()
  const { uploadFile } = useFileUpload()

  const handleOpenFile = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.dxf,.pdf'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) {
        try {
          await uploadFile(file)
          toast.success(`文件 "${file.name}" 处理成功`)
        } catch (error) {
          toast.error(error instanceof Error ? error.message : '处理失败')
        }
      }
    }
    input.click()
  }

  const handleExport = async (format: 'json' | 'bincode') => {
    try {
      const result = await exportService.exportAndDownload(format, format === 'json')
      if (result.success) {
        toast.success(`导出成功：${result.fileName}`)
      } else {
        toast.error(result.error || '导出失败')
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '导出失败')
    }
  }

  return (
    <header className="h-12 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center px-4 gap-2">
      {/* Logo */}
      <div className="flex items-center gap-2 mr-4">
        <div className="h-8 w-8 rounded bg-primary flex items-center justify-center">
          <span className="text-primary-foreground font-bold text-sm">CAD</span>
        </div>
        <span className="font-semibold text-sm hidden md:inline-block">
          几何智能处理系统
        </span>
      </div>

      {/* 文件操作组 */}
      <div className="flex items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" onClick={handleOpenFile}>
              <FolderOpen className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>打开文件</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon">
              <Save className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>保存（开发中）</TooltipContent>
        </Tooltip>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <Download className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuItem onClick={() => handleExport('json')}>
              <FileJson className="h-4 w-4 mr-2" />
              导出为 JSON
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => handleExport('bincode')}>
              <FileBox className="h-4 w-4 mr-2" />
              导出为 Bincode
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="h-6 w-px bg-border mx-2" />

      {/* 撤销重做 */}
      <div className="flex items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon">
              <Undo className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>撤销（开发中）</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon">
              <Redo className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>重做（开发中）</TooltipContent>
        </Tooltip>
      </div>

      <div className="h-6 w-px bg-border mx-2" />

      {/* 工具选择 */}
      <div className="flex items-center gap-1">
        <ToolButton
          icon="select"
          label="选择"
          shortcut="V"
          active={activeTool === 'select'}
          onClick={() => setTool('select')}
        />
        <ToolButton
          icon="trace"
          label="追踪"
          shortcut="T"
          active={activeTool === 'trace'}
          onClick={() => setTool('trace')}
        />
        <ToolButton
          icon="lasso"
          label="圈选"
          shortcut="L"
          active={activeTool === 'lasso'}
          onClick={() => setTool('lasso')}
        />
        <ToolButton
          icon="pan"
          label="平移"
          shortcut="H"
          active={activeTool === 'pan'}
          onClick={() => setTool('pan')}
        />
      </div>

      <div className="flex-1" />

      {/* 视图控制 */}
      <div className="flex items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleSidebar}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <rect width="18" height="18" x="3" y="3" rx="2" />
                <path d="M9 3v18" />
              </svg>
            </Button>
          </TooltipTrigger>
          <TooltipContent>切换侧边栏</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleRightPanel}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <rect width="18" height="18" x="3" y="3" rx="2" />
                <path d="M15 3v18" />
              </svg>
            </Button>
          </TooltipTrigger>
          <TooltipContent>切换属性面板</TooltipContent>
        </Tooltip>
      </div>
    </header>
  )
}
