import { Lock, Unlock } from 'lucide-react'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useLayerStore } from '@/stores/layer-store'
import { cn } from '@/lib/utils'

export function LayerPanel() {
  const { layers, toggleVisibility, toggleLock, selectedLayer, setSelectedLayer } =
    useLayerStore()

  // 模拟图层数据
  const mockLayers = [
    { id: 'layer-1', name: '墙体', visible: true, locked: false, count: 128, color: '#60a5fa' },
    { id: 'layer-2', name: '门窗', visible: true, locked: false, count: 24, color: '#22c55e' },
    { id: 'layer-3', name: '标注', visible: true, locked: true, count: 56, color: '#fbbf24' },
    { id: 'layer-4', name: '家具', visible: false, locked: false, count: 42, color: '#f472b6' },
  ]

  // 初始化图层数据
  if (layers.length === 0) {
    useLayerStore.getState().setLayers(mockLayers)
  }

  const allLayers = layers.length > 0 ? layers : mockLayers

  return (
    <ScrollArea className="flex-1">
      <Accordion type="single" collapsible className="w-full">
        {allLayers.map((layer) => (
          <AccordionItem
            key={layer.id}
            value={layer.id}
            className={cn(
              'border-b border-border/50',
              selectedLayer === layer.id && 'bg-accent/50'
            )}
          >
            <AccordionTrigger className="px-4 py-3 hover:no-underline">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={layer.visible}
                  onCheckedChange={() => toggleVisibility(layer.id)}
                  onClick={(e) => e.stopPropagation()}
                />
                <div
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: layer.color }}
                />
                <span className="text-sm">{layer.name}</span>
                <span className="text-xs text-muted-foreground">
                  ({layer.count})
                </span>
              </div>
            </AccordionTrigger>

            <AccordionContent className="px-4 pb-3">
              <div className="flex items-center gap-2">
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7"
                  onClick={() => toggleLock(layer.id)}
                >
                  {layer.locked ? (
                    <Lock className="h-3.5 w-3.5" />
                  ) : (
                    <Unlock className="h-3.5 w-3.5" />
                  )}
                </Button>

                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={() => setSelectedLayer(layer.id)}
                >
                  选择所有
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </ScrollArea>
  )
}
