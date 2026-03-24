import { MousePointer, Wand, Lasso, Hand } from 'lucide-react'
import { Button } from '@components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@components/ui/tooltip'
import { cn } from '@/lib/utils'

interface ToolButtonProps {
  icon: 'select' | 'trace' | 'lasso' | 'pan'
  label: string
  shortcut?: string
  active?: boolean
  onClick?: () => void
}

const iconMap = {
  select: MousePointer,
  trace: Wand,
  lasso: Lasso,
  pan: Hand,
}

export function ToolButton({
  icon,
  label,
  shortcut,
  active,
  onClick,
}: ToolButtonProps) {
  const Icon = iconMap[icon]

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={active ? 'secondary' : 'ghost'}
          size="icon"
          className={cn('relative', active && 'bg-secondary')}
          onClick={onClick}
        >
          <Icon className="h-4 w-4" />
          {active && (
            <span className="absolute -bottom-1 h-0.5 w-full bg-primary" />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <div className="flex items-center gap-2">
          <span>{label}</span>
          {shortcut && (
            <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded">
              {shortcut}
            </kbd>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
