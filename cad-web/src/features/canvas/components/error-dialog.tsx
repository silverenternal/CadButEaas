import { AlertCircle, X, AlertTriangle, Info, HelpCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@components/ui/button'

interface ErrorAction {
  label: string
  onClick: () => void
  variant?: 'default' | 'outline' | 'link'
}

interface ErrorDialogProps {
  isOpen: boolean
  title?: string
  message: string
  description?: string
  suggestions?: string[]
  actions?: ErrorAction[]
  onClose?: () => void
  type?: 'error' | 'warning' | 'info'
  className?: string
}

const ERROR_TYPES = {
  error: {
    icon: AlertCircle,
    color: 'text-error',
    bgColor: 'bg-error/10',
    borderColor: 'border-error/20',
    gradient: 'from-error/20',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/20',
    gradient: 'from-warning/20',
  },
  info: {
    icon: Info,
    color: 'text-info',
    bgColor: 'bg-info/10',
    borderColor: 'border-info/20',
    gradient: 'from-info/20',
  },
}

export function ErrorDialog({
  isOpen,
  title,
  message,
  description,
  suggestions,
  actions,
  onClose,
  type = 'error',
  className,
}: ErrorDialogProps) {
  const config = ERROR_TYPES[type]
  const Icon = config.icon

  if (!isOpen) return null

  return (
    <div className={cn('fixed inset-0 z-[200] flex items-center justify-center p-4', className)}>
      {/* 背景遮罩 */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      {/* 对话框 */}
      <div className="relative w-full max-w-md">
        <div className="acrylic-strong rounded-3xl p-6 shadow-2xl border border-white/20 fade-in">
          {/* 关闭按钮 */}
          {onClose && (
            <button
              onClick={onClose}
              className="absolute top-4 right-4 p-2 hover:bg-muted rounded-full transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}

          {/* 图标和标题 */}
          <div className="flex items-start gap-4 mb-4">
            <div className={cn(
              'w-12 h-12 rounded-2xl flex items-center justify-center',
              config.bgColor,
              config.color
            )}>
              <Icon className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-lg mb-1">
                {title || (type === 'error' ? '出错了' : type === 'warning' ? '请注意' : '提示信息')}
              </h3>
              <p className="text-muted-foreground text-sm">{message}</p>
            </div>
          </div>

          {/* 描述 */}
          {description && (
            <div className="mb-4 p-4 rounded-xl bg-muted/50 text-sm text-muted-foreground">
              {description}
            </div>
          )}

          {/* 可能原因 */}
          {suggestions && suggestions.length > 0 && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                <HelpCircle className="w-4 h-4" />
                建议操作
              </h4>
              <ul className="space-y-1.5">
                {suggestions.map((suggestion, i) => (
                  <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                    <span className="text-primary mt-0.5">•</span>
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 操作按钮 */}
          {actions && actions.length > 0 && (
            <div className="flex gap-3">
              {actions.map((action, i) => (
                <Button
                  key={i}
                  variant={action.variant || 'default'}
                  onClick={action.onClick}
                  className="flex-1"
                  size="lg"
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}

          {/* 技术支持 */}
          <div className="mt-4 pt-4 border-t border-white/10 text-center">
            <p className="text-xs text-muted-foreground">
              需要帮助？联系 <a href="mailto:support@example.com" className="text-primary hover:underline">support@example.com</a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
