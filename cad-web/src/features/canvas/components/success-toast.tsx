import { useEffect } from 'react'
import { CheckCircle, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SuccessToastProps {
  isOpen: boolean
  message: string
  description?: string
  duration?: number
  onClose: () => void
  className?: string
}

export function SuccessToast({
  isOpen,
  message,
  description,
  duration = 3000,
  onClose,
  className,
}: SuccessToastProps) {
  useEffect(() => {
    if (isOpen && duration > 0) {
      const timer = setTimeout(onClose, duration)
      return () => clearTimeout(timer)
    }
  }, [isOpen, duration, onClose])

  if (!isOpen) return null

  return (
    <div className={cn('fixed top-4 right-4 z-[200] fade-in', className)}>
      <div className="acrylic-strong rounded-2xl p-4 shadow-2xl border border-white/20 flex items-center gap-3">
        {/* 成功图标 */}
        <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center flex-shrink-0">
          <CheckCircle className="w-5 h-5 text-success" />
        </div>

        {/* 文字内容 */}
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm">{message}</p>
          {description && (
            <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
          )}
        </div>

        {/* 关闭按钮 */}
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-muted rounded-lg transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
