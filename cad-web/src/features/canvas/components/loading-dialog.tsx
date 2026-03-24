import { useState, useEffect } from 'react'
import { Loader2, CheckCircle, Clock, File } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LoadingStep {
  key: string
  label: string
  icon?: React.ReactNode
}

interface LoadingDialogProps {
  isOpen: boolean
  fileName: string
  fileSize: number
  currentStep: string
  progress: number
  steps?: LoadingStep[]
  estimatedTimeRemaining?: number // 秒
  onCancel?: () => void
  showLogs?: boolean
  logs?: string[]
  className?: string
}

const DEFAULT_STEPS: LoadingStep[] = [
  { key: 'upload', label: '上传文件中', icon: <File className="w-4 h-4" /> },
  { key: 'parse', label: '解析几何数据', icon: <Loader2 className="w-4 h-4" /> },
  { key: 'extract', label: '提取边和填充', icon: <Clock className="w-4 h-4" /> },
  { key: 'render', label: '渲染画布', icon: <CheckCircle className="w-4 h-4" /> },
]

export function LoadingDialog({
  isOpen,
  fileName,
  fileSize,
  currentStep,
  progress,
  steps = DEFAULT_STEPS,
  estimatedTimeRemaining,
  onCancel,
  showLogs = false,
  logs = [],
  className,
}: LoadingDialogProps) {
  const [expanded, setExpanded] = useState(false)
  const [startTime] = useState(Date.now())
  const [elapsedTime, setElapsedTime] = useState(0)

  useEffect(() => {
    if (isOpen) {
      const interval = setInterval(() => {
        setElapsedTime(Date.now() - startTime)
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [isOpen, startTime])

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.ceil(seconds)}秒`
    const mins = Math.floor(seconds / 60)
    const secs = Math.ceil(seconds % 60)
    return `${mins}分${secs}秒`
  }

  const getStepStatus = (stepKey: string) => {
    const stepIndex = steps.findIndex((s) => s.key === stepKey)
    const currentIndex = steps.findIndex((s) => s.key === currentStep)

    if (stepIndex < currentIndex) return 'completed'
    if (stepIndex === currentIndex) return 'current'
    return 'pending'
  }

  if (!isOpen) return null

  return (
    <div className={cn('fixed inset-0 z-[200] flex items-center justify-center p-4', className)}>
      {/* 背景遮罩 */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      {/* 对话框 */}
      <div className="relative w-full max-w-md">
        <div className="acrylic-strong rounded-3xl p-6 shadow-2xl border border-white/20 fade-in">
          {/* 头部 */}
          <div className="flex items-start gap-4 mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-2xl blur-xl" />
              <div className="relative w-12 h-12 bg-gradient-to-br from-primary to-primary/80 rounded-2xl flex items-center justify-center shadow-lg">
                <Loader2 className="w-6 h-6 text-primary-foreground animate-spin" />
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-bold text-lg">正在处理文件</h3>
              <p className="text-sm text-muted-foreground truncate">{fileName}</p>
              <p className="text-xs text-muted-foreground">{formatFileSize(fileSize)}</p>
            </div>
          </div>

          {/* 进度条 */}
          <div className="mb-6 space-y-2">
            <div className="relative h-3 bg-muted/50 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-purple-500 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
              {/* 进度条光效 */}
              <div className="absolute inset-y-0 left-0 w-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[shimmer_1s_infinite]" />
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-primary">{progress}%</span>
              {estimatedTimeRemaining && (
                <span className="text-muted-foreground">
                  预计剩余：{formatTime(estimatedTimeRemaining)}
                </span>
              )}
            </div>
          </div>

          {/* 步骤列表 */}
          <div className="space-y-2 mb-4">
            {steps.map((step) => {
              const status = getStepStatus(step.key)
              return (
                <div
                  key={step.key}
                  className={cn(
                    'flex items-center gap-3 p-2 rounded-lg transition-all',
                    status === 'current' && 'bg-primary/10',
                    status === 'completed' && 'text-success'
                  )}
                >
                  <div
                    className={cn(
                      'w-6 h-6 rounded-full flex items-center justify-center',
                      status === 'completed' && 'bg-success/10',
                      status === 'current' && 'bg-primary/10',
                      status === 'pending' && 'bg-muted/50'
                    )}
                  >
                    {status === 'completed' ? (
                      <CheckCircle className="w-3.5 h-3.5" />
                    ) : status === 'current' ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/30" />
                    )}
                  </div>
                  <span
                    className={cn(
                      'text-sm',
                      status === 'current' && 'font-medium',
                      status === 'pending' && 'text-muted-foreground'
                    )}
                  >
                    {step.label}
                  </span>
                </div>
              )
            })}
          </div>

          {/* 已用时间 */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
            <Clock className="w-3 h-3" />
            <span>已用时：{formatTime(elapsedTime / 1000)}</span>
          </div>

          {/* 日志展开区域 */}
          {showLogs && logs.length > 0 && (
            <div className="mb-4">
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-xs text-muted-foreground hover:text-primary transition-colors flex items-center gap-1"
              >
                {expanded ? '收起日志' : '查看详细日志'}
              </button>
              {expanded && (
                <div className="mt-2 p-3 bg-muted/50 rounded-xl max-h-32 overflow-y-auto font-mono text-xs">
                  {logs.map((log, i) => (
                    <div key={i} className="py-0.5">
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 取消按钮 */}
          {onCancel && (
            <Button
              variant="outline"
              onClick={onCancel}
              className="w-full"
              size="lg"
            >
              取消
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

// Button 组件的简单实现（如果项目中还没有）
function Button({
  variant = 'default',
  size = 'default',
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'sm' | 'default' | 'lg'
}) {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-xl transition-colors focus:outline-none focus:ring-2 focus:ring-primary/20 disabled:opacity-50 disabled:pointer-events-none'
  
  const variants = {
    default: 'bg-primary text-primary-foreground hover:bg-primary/90',
    outline: 'border border-white/20 bg-transparent hover:bg-white/10',
    ghost: 'hover:bg-white/10',
  }

  const sizes = {
    sm: 'h-8 px-3 text-sm',
    default: 'h-10 px-4',
    lg: 'h-12 px-6 text-lg',
  }

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className || ''}`}
      {...props}
    />
  )
}
