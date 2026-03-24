import { Moon, Sun, Monitor } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useTheme } from '@/hooks/use-theme'
import { useAppStore } from '@/stores/app-store'

export function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { settings, updateSettings } = useAppStore()

  return (
    <ScrollArea className="h-full">
      <div className="max-w-2xl mx-auto p-8 space-y-8">
        <div>
          <h1 className="text-2xl font-bold">设置</h1>
          <p className="text-muted-foreground">
            管理应用程序偏好和配置
          </p>
        </div>

        <Separator />

        {/* 外观设置 */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">外观</h2>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>主题</Label>
                <p className="text-sm text-muted-foreground">
                  选择应用程序的颜色主题
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant={theme === 'light' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setTheme('light')}
                >
                  <Sun className="h-4 w-4" />
                </Button>
                <Button
                  variant={theme === 'dark' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setTheme('dark')}
                >
                  <Moon className="h-4 w-4" />
                </Button>
                <Button
                  variant={theme === 'system' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setTheme('system')}
                >
                  <Monitor className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="show-grid">显示网格</Label>
                <p className="text-sm text-muted-foreground">
                  在画布上显示参考网格
                </p>
              </div>
              <Switch
                id="show-grid"
                checked={settings.showGrid}
                onCheckedChange={(checked) =>
                  updateSettings({ showGrid: checked })
                }
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="snap-to-grid">吸附到网格</Label>
                <p className="text-sm text-muted-foreground">
                  移动时自动吸附到网格点
                </p>
              </div>
              <Switch
                id="snap-to-grid"
                checked={settings.snapToGrid}
                onCheckedChange={(checked) =>
                  updateSettings({ snapToGrid: checked })
                }
              />
            </div>
          </div>
        </section>

        <Separator />

        {/* 文件设置 */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">文件</h2>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="auto-save">自动保存</Label>
                <p className="text-sm text-muted-foreground">
                  定期自动保存工作进度
                </p>
              </div>
              <Switch
                id="auto-save"
                checked={settings.autoSave}
                onCheckedChange={(checked) =>
                  updateSettings({ autoSave: checked })
                }
              />
            </div>
          </div>
        </section>

        <Separator />

        {/* 关于 */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">关于</h2>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">版本</span>
              <span>0.1.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">构建日期</span>
              <span>2026 年 3 月 21 日</span>
            </div>
          </div>
        </section>
      </div>
    </ScrollArea>
  )
}
