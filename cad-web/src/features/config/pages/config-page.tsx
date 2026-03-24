import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/services/api-client'
import { toast } from 'sonner'
import { Settings, Check, Loader2, FileText, Cpu, Download } from 'lucide-react'

interface ConfigProfile {
  name: string
  description: string
}

export function ConfigPage() {
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null)

  // 获取预设配置列表
  const { data: profilesData, isLoading: loadingProfiles } = useQuery({
    queryKey: ['config-profiles'],
    queryFn: async () => {
      const response = await fetch(`${apiClient.getBaseUrl()}/config/profiles`)
      if (!response.ok) throw new Error('获取配置列表失败')
      return response.json()
    },
  })

  // 获取选中配置的详情
  const { data: configDetails, isLoading: loadingDetails } = useQuery({
    queryKey: ['config-profile', selectedProfile],
    queryFn: async () => {
      if (!selectedProfile) return null
      const response = await fetch(
        `${apiClient.getBaseUrl()}/config/profile/${selectedProfile}`
      )
      if (!response.ok) throw new Error('获取配置详情失败')
      return response.json()
    },
    enabled: !!selectedProfile,
  })

  const handleUseProfile = async (profileName: string) => {
    try {
      toast.success(`已选择 "${profileName}" 配置`)
      setSelectedProfile(profileName)
    } catch (error) {
      toast.error('设置配置失败')
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* 页面标题 - 亚克力效果 */}
      <header className="h-14 acrylic-light border-b flex items-center px-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg shadow-primary/30">
            <Settings className="w-4 h-4 text-primary-foreground" />
          </div>
          <h1 className="text-lg font-semibold">配置管理</h1>
        </div>
      </header>

      {/* 主体内容 */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full grid grid-cols-3 gap-6 p-6">
          {/* 左侧：配置列表 */}
          <div className="col-span-1 space-y-4">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-muted-foreground" />
              <h2 className="text-sm font-medium">预设配置</h2>
            </div>
            {loadingProfiles ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="space-y-2">
                {profilesData?.profiles?.map((profile: ConfigProfile) => (
                  <button
                    key={profile.name}
                    onClick={() => handleUseProfile(profile.name)}
                    className={`w-full p-4 rounded-xl border text-left transition-all duration-300 group ${
                      selectedProfile === profile.name
                        ? 'border-primary/50 bg-gradient-to-br from-primary/10 to-primary/5 shadow-md'
                        : 'acrylic-light hover:bg-accent/60 hover:shadow-md hover:-translate-y-0.5'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium">{profile.name}</h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          {profile.description}
                        </p>
                      </div>
                      {selectedProfile === profile.name && (
                        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg shadow-primary/30">
                          <Check className="w-4 h-4 text-primary-foreground" />
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* 右侧：配置详情 */}
          <div className="col-span-2">
            {selectedProfile ? (
              <div className="space-y-6 fade-in">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium">
                    配置详情 - <span className="text-primary">{selectedProfile}</span>
                  </h2>
                </div>
                {loadingDetails ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                  </div>
                ) : configDetails ? (
                  <div className="space-y-4">
                    {/* 拓扑配置 */}
                    <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500/20 to-blue-600/20 flex items-center justify-center">
                          <Settings className="w-4 h-4 text-blue-600" />
                        </div>
                        <h3 className="font-semibold">拓扑设置</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <ConfigItem label="吸附容差" value={`${configDetails.topology.snap_tolerance_mm} mm`} />
                        <ConfigItem label="最小线段长度" value={`${configDetails.topology.min_line_length_mm} mm`} />
                        <ConfigItem label="角度合并容差" value={`${configDetails.topology.merge_angle_tolerance_deg}°`} />
                        <ConfigItem label="最大缺口桥接" value={`${configDetails.topology.max_gap_bridge_length_mm} mm`} />
                      </div>
                    </div>

                    {/* 验证配置 */}
                    <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500/20 to-green-600/20 flex items-center justify-center">
                          <Check className="w-4 h-4 text-green-600" />
                        </div>
                        <h3 className="font-semibold">验证设置</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <ConfigItem label="闭合容差" value={`${configDetails.validator.closure_tolerance_mm} mm`} />
                        <ConfigItem label="最小面积" value={`${configDetails.validator.min_area_m2} m²`} />
                        <ConfigItem label="最小边长" value={`${configDetails.validator.min_edge_length_mm} mm`} />
                        <ConfigItem label="最小角度" value={`${configDetails.validator.min_angle_deg}°`} />
                      </div>
                    </div>

                    {/* 导出配置 */}
                    <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center">
                          <Download className="w-4 h-4 text-purple-600" />
                        </div>
                        <h3 className="font-semibold">导出设置</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <ConfigItem label="格式" value={configDetails.export.format} />
                        <ConfigItem label="JSON 缩进" value={configDetails.export.json_indent.toString()} />
                        <ConfigItem 
                          label="自动验证" 
                          value={configDetails.export.auto_validate ? '启用' : '禁用'} 
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground py-8">
                    无法加载配置详情
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center space-y-4 fade-in">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center">
                    <Settings className="w-8 h-8 text-muted-foreground/50" />
                  </div>
                  <p className="text-muted-foreground">选择一个配置查看详情</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg bg-background/50">
      <span className="text-sm text-muted-foreground">{label}</span>
      <p className="text-base font-semibold mt-1">{value}</p>
    </div>
  )
}
