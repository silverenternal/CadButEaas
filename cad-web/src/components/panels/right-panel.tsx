import { Settings, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { PropertyPanel } from '@components/panels/property-panel'
import { GapDetectionPanel } from '@components/panels/gap-detection-panel'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@components/ui/tabs'
import { useState } from 'react'

interface RightPanelProps {
  className?: string
}

export function RightPanel({ className }: RightPanelProps) {
  const [activeTab, setActiveTab] = useState<'properties' | 'gaps'>('properties')

  return (
    <aside
      className={cn(
        'w-72 border-l bg-background flex flex-col overflow-hidden',
        className
      )}
    >
      {/* 右侧面板标题 */}
      <div className="h-10 border-b flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">属性</span>
        </div>
      </div>

      {/* 标签页切换 */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'properties' | 'gaps')} className="flex-1 flex flex-col">
        <TabsList className="w-full rounded-none border-b bg-transparent p-0 h-10">
          <TabsTrigger
            value="properties"
            className="rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary"
          >
            属性
          </TabsTrigger>
          <TabsTrigger
            value="gaps"
            className="rounded-none data-[state=active]:border-b-2 data-[state=active]:border-primary"
          >
            <AlertCircle className="w-4 h-4 mr-1" />
            缺口
          </TabsTrigger>
        </TabsList>

        <TabsContent value="properties" className="flex-1 m-0 overflow-hidden">
          <PropertyPanel />
        </TabsContent>

        <TabsContent value="gaps" className="flex-1 m-0 overflow-hidden">
          <GapDetectionPanel />
        </TabsContent>
      </Tabs>
    </aside>
  )
}
