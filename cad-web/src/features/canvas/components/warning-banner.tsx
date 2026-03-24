import { AlertTriangle, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@components/ui/button'

interface WarningBannerProps {
  isOpen: boolean
  title?: string
  message: string
  description?: string
  actionLabel?: string
  onAction?: () => void
  onClose: () => void
  className?: string
}

export function WarningBanner({
  isOpen,
  title,
  message,
  description,
  actionLabel,
  onAction,
  onClose,
  className,
}: WarningBannerProps) {
  if (!isOpen) return null

  return (
    <div className={cn('fixed top-4 left-1/2 -translate-x-1/2 z-[200] w-full max-w-lg fade-in', className)}>
      <div className="acrylic-strong rounded-2xl p-4 shadow-2xl border border-warning/20 bg-warning/10">
        <div className="flex items-start gap-3">
          {/* 警告图标 */}
          <div className="w-10 h-10 rounded-xl bg-warning/10 flex items-center justify-center flex-shrink-0">
            <AlertTriangle className="w-5 h-5 text-warning" />
          </div>

          {/* 文字内容 */}
          <div className="flex-1 min-w-0">
            <h4 className="font-semibold text-sm mb-1">
              {title || '请注意'}
            </h4>
            <p className="text-sm text-muted-foreground">{message}</p>
            {description && (
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            )}

            {/* 操作按钮 */}
            {actionLabel && onAction && (
              <div className="mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onAction}
                  className="h-8 text-xs border-warning/20 hover:bg-warning/20"
                >
                  {actionLabel}
                </Button>
              </div>
            )}
          </div>

          {/* 关闭按钮 */}
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-warning/20 rounded-lg transition-colors flex-shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
