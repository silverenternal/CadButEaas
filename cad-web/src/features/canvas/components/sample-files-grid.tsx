import { useState } from 'react'
import { Zap, Download } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@components/ui/button'

interface SampleFile {
  id: string
  name: string
  description: string
  size: string
  type: 'dxf' | 'dwg'
  category: 'architectural' | 'mechanical' | 'basic'
  difficulty: 'easy' | 'medium' | 'hard'
  url: string
}

interface SampleFilesGridProps {
  onFileSelect: (sample: SampleFile) => void
  className?: string
}

const SAMPLE_FILES: SampleFile[] = [
  {
    id: 'sample-1',
    name: '建筑平面图',
    description: '简单的住宅平面布局',
    size: '2.3 MB',
    type: 'dxf',
    category: 'architectural',
    difficulty: 'easy',
    url: '/samples/floor-plan.dxf',
  },
  {
    id: 'sample-2',
    name: '机械零件图',
    description: '标准机械零件三视图',
    size: '1.8 MB',
    type: 'dxf',
    category: 'mechanical',
    difficulty: 'medium',
    url: '/samples/mechanical-part.dxf',
  },
  {
    id: 'sample-3',
    name: '基础几何图形',
    description: '直线、圆、弧练习',
    size: '0.5 MB',
    type: 'dxf',
    category: 'basic',
    difficulty: 'easy',
    url: '/samples/basic-shapes.dxf',
  },
]

export function SampleFilesGrid({ onFileSelect, className }: SampleFilesGridProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const handleLoad = async (sample: SampleFile) => {
    setLoadingId(sample.id)
    try {
      // 实际项目中，这里应该从服务器加载示例文件
      // 或者使用内置的 base64 编码的示例文件
      const response = await fetch(sample.url)
      if (!response.ok) {
        throw new Error('Failed to load sample file')
      }
      onFileSelect(sample)
    } catch (error) {
      console.error('Failed to load sample:', error)
      // 如果示例文件不存在，显示提示
      alert('示例文件加载中，请稍后...（示例文件需要在服务器上提供）')
    } finally {
      setLoadingId(null)
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'text-success bg-success/10'
      case 'medium':
        return 'text-warning bg-warning/10'
      case 'hard':
        return 'text-error bg-error/10'
    }
  }

  const getDifficultyText = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return '简单'
      case 'medium':
        return '中等'
      case 'hard':
        return '复杂'
    }
  }

  return (
    <div className={cn('w-full space-y-3', className)}>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Zap className="w-4 h-4" />
        <span>示例文件</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {SAMPLE_FILES.map((sample) => (
          <div
            key={sample.id}
            className={cn(
              'group relative p-4 rounded-2xl border transition-all duration-300',
              'hover:border-primary/50 hover:shadow-lg hover:shadow-primary/10',
              'hover:-translate-y-1',
              'acrylic-strong'
            )}
          >
            {/* 装饰性背景 */}
            <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
              <div className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-br from-primary/5 to-purple-500/5 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>

            <div className="relative z-10 space-y-3">
              {/* 头部 */}
              <div className="flex items-start justify-between">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/10 to-purple-500/10 flex items-center justify-center text-xl">
                  {sample.category === 'architectural' && '🏠'}
                  {sample.category === 'mechanical' && '⚙️'}
                  {sample.category === 'basic' && '📐'}
                </div>
                <span
                  className={cn(
                    'px-2 py-0.5 text-xs font-medium rounded-full',
                    getDifficultyColor(sample.difficulty)
                  )}
                >
                  {getDifficultyText(sample.difficulty)}
                </span>
              </div>

              {/* 内容 */}
              <div className="space-y-1">
                <h4 className="font-semibold">{sample.name}</h4>
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {sample.description}
                </p>
              </div>

              {/* 底部信息 */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {sample.size} · {sample.type.toUpperCase()}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleLoad(sample)}
                  disabled={loadingId === sample.id}
                  className="h-8 px-3 text-xs"
                >
                  {loadingId === sample.id ? (
                    <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <>
                      <Download className="w-3 h-3 mr-1" />
                      加载
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 提示信息 */}
      <p className="text-xs text-muted-foreground text-center">
        💡 提示：示例文件用于快速体验功能，无需上传自己的文件
      </p>
    </div>
  )
}
