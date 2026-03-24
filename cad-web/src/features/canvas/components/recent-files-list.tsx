import { useState, useEffect } from 'react'
import { Clock, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface RecentFile {
  id: string
  name: string
  size: number
  lastModified: number
  type: 'dxf' | 'dwg' | 'pdf'
  thumbnail?: string
}

interface RecentFilesListProps {
  onFileSelect: (file: RecentFile) => void
  onFileDelete?: (id: string) => void
  maxFiles?: number
  className?: string
}

export function RecentFilesList({
  onFileSelect,
  onFileDelete,
  maxFiles = 5,
  className,
}: RecentFilesListProps) {
  const [recentFiles, setRecentFiles] = useState<RecentFile[]>([])

  useEffect(() => {
    // 从 localStorage 加载最近文件
    const stored = localStorage.getItem('recent_files')
    if (stored) {
      try {
        const files = JSON.parse(stored)
        setRecentFiles(files.slice(0, maxFiles))
      } catch (e) {
        console.error('Failed to load recent files:', e)
      }
    }
  }, [maxFiles])


  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setRecentFiles((prev) => {
      const updated = prev.filter((f) => f.id !== id)
      localStorage.setItem('recent_files', JSON.stringify(updated))
      return updated
    })
    onFileDelete?.(id)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  const formatTime = (timestamp: number) => {
    const now = Date.now()
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return '刚刚'
    if (minutes < 60) return `${minutes} 分钟前`
    if (hours < 24) return `${hours} 小时前`
    if (days < 7) return `${days} 天前`
    return new Date(timestamp).toLocaleDateString('zh-CN')
  }

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

  if (recentFiles.length === 0) {
    return null
  }

  return (
    <div className={cn('w-full space-y-3', className)}>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Clock className="w-4 h-4" />
        <span>最近文件</span>
      </div>

      <div className="space-y-2">
        {recentFiles.map((file) => (
          <div
            key={file.id}
            onClick={() => onFileSelect(file)}
            className={cn(
              'group flex items-center gap-3 p-3 rounded-xl cursor-pointer',
              'hover:bg-primary/5 hover:border-primary/20',
              'border border-transparent transition-all duration-200',
              'acrylic'
            )}
          >
            {/* 文件图标 */}
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary/10 to-purple-500/10 flex items-center justify-center text-xl">
              {getFileIcon(file.type)}
            </div>

            {/* 文件信息 */}
            <div className="flex-1 min-w-0">
              <p className="font-medium text-sm truncate">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatFileSize(file.size)} · {formatTime(file.lastModified)}
              </p>
            </div>

            {/* 删除按钮 */}
            <button
              onClick={(e) => handleDelete(file.id, e)}
              className="opacity-0 group-hover:opacity-100 p-2 hover:bg-error/10 rounded-lg transition-all"
            >
              <Trash2 className="w-4 h-4 text-error" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

// 导出添加最近文件的工具函数
export function addRecentFile(file: File) {
  const stored = localStorage.getItem('recent_files')
  const files: RecentFile[] = stored ? JSON.parse(stored) : []
  
  const newFile: RecentFile = {
    id: `${file.name}-${file.lastModified}`,
    name: file.name,
    size: file.size,
    lastModified: file.lastModified,
    type: file.name.split('.').pop()?.toLowerCase() as 'dxf' | 'dwg' | 'pdf' || 'dxf',
  }

  const filtered = files.filter((f) => f.id !== newFile.id)
  const updated = [newFile, ...filtered].slice(0, 5)
  localStorage.setItem('recent_files', JSON.stringify(updated))
}
