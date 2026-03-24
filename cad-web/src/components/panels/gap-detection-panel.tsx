import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { interactionService } from '@/services/interaction-service'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { useCanvasStore } from '@/stores/canvas-store'
import { toast } from 'sonner'
import { AlertCircle, Wrench, AlertTriangle } from 'lucide-react'
import type { GapInfo } from '@/types/api'

interface GapItemProps {
  gap: GapInfo
  onBridge: (gapId: number) => void
  isBridging: boolean
}

function GapItem({ gap, onBridge, isBridging }: GapItemProps) {
  const gapTypeColors: Record<string, string> = {
    collinear: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    orthogonal: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    angled: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    small: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  }

  const gapTypeLabels: Record<string, string> = {
    collinear: '共线缺口',
    orthogonal: '正交缺口',
    angled: '斜角缺口',
    small: '小缺口',
  }

  return (
    <div className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className={`px-2 py-0.5 rounded text-xs font-medium ${gapTypeColors[gap.gap_type] || 'bg-gray-100 text-gray-800'}`}>
          {gapTypeLabels[gap.gap_type] || gap.gap_type}
        </div>
        <span className="text-xs text-muted-foreground">
          {gap.length.toFixed(2)} mm
        </span>
      </div>
      
      <div className="space-y-1 text-xs text-muted-foreground">
        <div>起点：[{gap.start[0].toFixed(2)}, {gap.start[1].toFixed(2)}]</div>
        <div>终点：[{gap.end[0].toFixed(2)}, {gap.end[1].toFixed(2)}]</div>
      </div>

      <Button
        size="sm"
        variant="outline"
        className="w-full mt-2"
        onClick={() => onBridge(gap.id)}
        disabled={isBridging}
      >
        <Wrench className="w-3 h-3 mr-1" />
        {isBridging ? '桥接中...' : '桥接缺口'}
      </Button>
    </div>
  )
}

export function GapDetectionPanel() {
  const [tolerance, setTolerance] = useState(2.0)
  const { gaps, setGaps } = useCanvasStore()

  // 缺口检测
  const detectMutation = useMutation({
    mutationFn: async (tol: number) => {
      return interactionService.detectGaps({ tolerance: tol })
    },
    onSuccess: (data) => {
      setGaps(data.gaps)
      toast.success(`检测到 ${data.total_count} 个缺口`)
    },
    onError: (error: Error) => {
      toast.error(error.message || '缺口检测失败')
    },
  })

  // 缺口桥接
  const bridgeMutation = useMutation({
    mutationFn: async (gapId: number) => {
      return interactionService.snapBridge({ gap_id: gapId })
    },
    onSuccess: () => {
      toast.success('缺口已桥接')
      // 重新检测缺口
      detectMutation.mutate(tolerance)
    },
    onError: (error: Error) => {
      toast.error(error.message || '缺口桥接失败')
    },
  })

  const handleDetect = () => {
    detectMutation.mutate(tolerance)
  }

  const handleBridge = (gapId: number) => {
    bridgeMutation.mutate(gapId)
  }

  return (
    <div className="flex flex-col h-full">
      {/* 标题 */}
      <div className="h-10 border-b flex items-center px-4 gap-2">
        <AlertCircle className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-medium">缺口检测</span>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* 检测设置 */}
          <section>
            <h3 className="text-xs font-medium text-muted-foreground mb-2">
              检测参数
            </h3>
            <div className="space-y-2">
              <div>
                <Label htmlFor="tolerance">吸附容差 (mm)</Label>
                <div className="flex gap-2">
                  <Input
                    id="tolerance"
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={tolerance}
                    onChange={(e) => setTolerance(parseFloat(e.target.value) || 0)}
                  />
                  <Button
                    size="sm"
                    onClick={handleDetect}
                    disabled={detectMutation.isPending}
                  >
                    {detectMutation.isPending ? '检测中...' : '检测'}
                  </Button>
                </div>
              </div>
            </div>
          </section>

          <Separator />

          {/* 缺口列表 */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-medium text-muted-foreground">
                检测到的缺口
              </h3>
              {gaps.length > 0 && (
                <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                  {gaps.length}
                </span>
              )}
            </div>

            {gaps.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">点击"检测"按钮查找缺口</p>
              </div>
            ) : (
              <div className="space-y-2">
                {gaps.map((gap) => (
                  <GapItem
                    key={gap.id}
                    gap={gap}
                    onBridge={handleBridge}
                    isBridging={bridgeMutation.isPending}
                  />
                ))}
              </div>
            )}
          </section>
        </div>
      </ScrollArea>
    </div>
  )
}
