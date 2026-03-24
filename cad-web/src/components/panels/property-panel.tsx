import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useCanvasStore, selectSelectedEdge } from '@/stores/canvas-store'
import type { BoundarySemantic } from '@/types/api'

export function PropertyPanel() {
  const selectedEdge = useCanvasStore(selectSelectedEdge)

  if (!selectedEdge) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground p-4">
        <div className="text-center space-y-2">
          <p className="text-sm">选择一条边以查看属性</p>
        </div>
      </div>
    )
  }

  const length = Math.sqrt(
    Math.pow(selectedEdge.end[0] - selectedEdge.start[0], 2) +
      Math.pow(selectedEdge.end[1] - selectedEdge.start[1], 2)
  )

  return (
    <ScrollArea className="flex-1">
      <div className="p-4 space-y-4">
        {/* 基本信息 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            基本信息
          </h3>
          <div className="space-y-2">
            <div>
              <Label htmlFor="edge-id">边 ID</Label>
              <Input id="edge-id" value={selectedEdge.id.toString()} disabled />
            </div>
            <div>
              <Label htmlFor="edge-length">长度</Label>
              <Input
                id="edge-length"
                value={`${length.toFixed(2)} mm`}
                disabled
              />
            </div>
            <div>
              <Label htmlFor="edge-layer">图层</Label>
              <Input
                id="edge-layer"
                value={selectedEdge.layer || '未分类'}
                disabled
              />
            </div>
          </div>
        </section>

        <Separator />

        {/* 语义标注 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            语义标注
          </h3>
          <div className="space-y-2">
            <div>
              <Label htmlFor="semantic">类型</Label>
              <Select
                value={selectedEdge.semantic || 'custom'}
                onValueChange={(value: BoundarySemantic) =>
                  console.log('Update semantic', selectedEdge.id, value)
                }
              >
                <SelectTrigger id="semantic">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hard_wall">硬墙</SelectItem>
                  <SelectItem value="absorptive_wall">吸声墙</SelectItem>
                  <SelectItem value="opening">开口</SelectItem>
                  <SelectItem value="window">窗户</SelectItem>
                  <SelectItem value="door">门</SelectItem>
                  <SelectItem value="custom">自定义</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        <Separator />

        {/* 坐标信息 */}
        <section>
          <h3 className="text-xs font-medium text-muted-foreground mb-2">
            坐标信息
          </h3>
          <div className="space-y-2">
            <div>
              <Label htmlFor="start-point">起点</Label>
              <Input
                id="start-point"
                value={`${selectedEdge.start[0].toFixed(2)}, ${selectedEdge.start[1].toFixed(2)}`}
                disabled
              />
            </div>
            <div>
              <Label htmlFor="end-point">终点</Label>
              <Input
                id="end-point"
                value={`${selectedEdge.end[0].toFixed(2)}, ${selectedEdge.end[1].toFixed(2)}`}
                disabled
              />
            </div>
          </div>
        </section>
      </div>
    </ScrollArea>
  )
}
