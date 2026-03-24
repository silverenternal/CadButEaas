import { Image, Maximize2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@components/ui/button'

interface FilePreviewCardProps {
  file: File
  onConfirm: () => void
  onCancel: () => void
  className?: string
}

export function FilePreviewCard({
  file,
  onConfirm,
  onCancel,
  className,
}: FilePreviewCardProps) {
  const fileType = file.name.split('.').pop()?.toLowerCase() || 'unknown'
  const fileSize = (file.size / (1024 * 1024)).toFixed(2)

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'dxf':
        return '📐'
      case 'dwg':
        return '📏'
      case 'pdf':
        return '📄'
      default:
        return '📁'
    }
  }

  return (
    <div className={cn('w-full max-w-md mx-auto', className)}>
      <div className="acrylic-strong rounded-3xl p-6 space-y-4 shadow-2xl border border-white/20">
        {/* 文件预览头部 */}
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/10 to-purple-500/10 flex items-center justify-center text-3xl shadow-lg">
            {getFileIcon(fileType)}
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-lg truncate">{file.name}</h3>
            <p className="text-sm text-muted-foreground">
              {fileSize} MB · {fileType.toUpperCase()}
            </p>
          </div>
        </div>

        {/* 预览区域（如果有缩略图） */}
        <div className="relative aspect-video rounded-xl bg-muted/50 overflow-hidden border border-white/10">
          {/* TODO: 如果有缩略图，在这里显示 */}
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            <div className="text-center space-y-2">
              <Image className="w-12 h-12 mx-auto opacity-50" />
              <p className="text-xs">预览图生成中...</p>
            </div>
          </div>
        </div>

        {/* 文件信息 */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="p-3 rounded-xl bg-muted/50">
            <p className="text-xs text-muted-foreground">修改时间</p>
            <p className="font-medium">
              {new Date(file.lastModified).toLocaleString('zh-CN')}
            </p>
          </div>
          <div className="p-3 rounded-xl bg-muted/50">
            <p className="text-xs text-muted-foreground">文件大小</p>
            <p className="font-medium">{fileSize} MB</p>
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="flex gap-3 pt-2">
          <Button
            variant="outline"
            onClick={onCancel}
            className="flex-1"
            size="lg"
          >
            取消
          </Button>
          <Button
            onClick={onConfirm}
            className="flex-1"
            size="lg"
            shine
          >
            <Maximize2 className="w-4 h-4 mr-2" />
            开始解析
          </Button>
        </div>
      </div>
    </div>
  )
}
