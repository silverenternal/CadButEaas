import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/services/api-client'
import { Button } from '@components/ui/button'
import { toast } from 'sonner'
import { Waves, Play, Loader2, Activity, Volume2, Box, Gauge } from 'lucide-react'

interface AcousticAnalysisResult {
  result: {
    type: string
    surface_ids?: number[]
    total_area?: number
    material_distribution?: Array<{
      material_name: string
      area: number
      percentage: number
    }>
    equivalent_absorption_area?: Record<string, number>
    average_absorption_coefficient?: Record<string, number>
    volume?: number
    total_surface_area?: number
    formula?: string
    t60?: Record<string, number>
    edt?: Record<string, number>
  }
  computation_time: number
  metrics: {
    surface_count: number
    computation_time_ms: number
  }
}

export function AcousticPage() {
  const [analysisType, setAnalysisType] = useState<'selection' | 'room' | 'comparative'>('selection')
  const [boundary, setBoundary] = useState({
    minX: 0,
    minY: 0,
    maxX: 10,
    maxY: 10,
  })
  const [roomId, setRoomId] = useState(0)
  const [roomHeight, setRoomHeight] = useState(3.0)
  const [formula, setFormula] = useState<'SABINE' | 'EYRING' | 'AUTO'>('AUTO')

  // 执行声学分析
  const analyzeMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await fetch(`${apiClient.getBaseUrl()}/acoustic/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
      if (!response.ok) throw new Error('声学分析失败')
      return response.json()
    },
    onSuccess: (data: AcousticAnalysisResult) => {
      toast.success(`分析完成，耗时 ${data.metrics.computation_time_ms}ms`)
    },
    onError: (error: Error) => {
      toast.error(error.message)
    },
  })

  const handleAnalyzeSelection = () => {
    analyzeMutation.mutate({
      type: 'SELECTION_MATERIAL_STATS',
      boundary: {
        type: 'RECT',
        min: [boundary.minX, boundary.minY],
        max: [boundary.maxX, boundary.maxY],
      },
      mode: 'SMART',
    })
  }

  const handleAnalyzeRoom = () => {
    analyzeMutation.mutate({
      type: 'ROOM_REVERBERATION',
      room_id: roomId,
      formula,
      room_height: roomHeight,
    })
  }

  const result = analyzeMutation.data as AcousticAnalysisResult | undefined

  return (
    <div className="h-full flex flex-col">
      {/* 页面标题 - 亚克力效果 */}
      <header className="h-14 acrylic-light border-b flex items-center px-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
            <Waves className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-lg font-semibold">声学分析</h1>
        </div>
      </header>

      {/* 主体内容 */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full grid grid-cols-3 gap-6 p-6">
          {/* 左侧：分析设置 */}
          <div className="col-span-1 space-y-6">
            {/* 分析类型选择 */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-muted-foreground" />
                <h2 className="text-sm font-medium">分析类型</h2>
              </div>
              <div className="space-y-2">
                <button
                  onClick={() => setAnalysisType('selection')}
                  className={`w-full p-4 rounded-xl border text-left transition-all duration-300 group ${
                    analysisType === 'selection'
                      ? 'border-purple-500/50 bg-gradient-to-br from-purple-500/10 to-purple-500/5 shadow-md'
                      : 'acrylic-light hover:bg-accent/60 hover:shadow-md hover:-translate-y-0.5'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
                      analysisType === 'selection' 
                        ? 'bg-gradient-to-br from-purple-500 to-purple-600' 
                        : 'bg-muted'
                    }`}>
                      <Activity className={`w-5 h-5 ${
                        analysisType === 'selection' ? 'text-white' : 'text-muted-foreground'
                      }`} />
                    </div>
                    <span className="font-medium">选区材料统计</span>
                  </div>
                </button>
                <button
                  onClick={() => setAnalysisType('room')}
                  className={`w-full p-4 rounded-xl border text-left transition-all duration-300 group ${
                    analysisType === 'room'
                      ? 'border-purple-500/50 bg-gradient-to-br from-purple-500/10 to-purple-500/5 shadow-md'
                      : 'acrylic-light hover:bg-accent/60 hover:shadow-md hover:-translate-y-0.5'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
                      analysisType === 'room' 
                        ? 'bg-gradient-to-br from-purple-500 to-purple-600' 
                        : 'bg-muted'
                    }`}>
                      <Waves className={`w-5 h-5 ${
                        analysisType === 'room' ? 'text-white' : 'text-muted-foreground'
                      }`} />
                    </div>
                    <span className="font-medium">房间混响时间</span>
                  </div>
                </button>
              </div>
            </div>

            {/* 选区分析设置 */}
            {analysisType === 'selection' && (
              <div className="acrylic-light rounded-2xl p-5 border border-white/20 fade-in">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500/20 to-blue-600/20 flex items-center justify-center">
                    <Box className="w-4 h-4 text-blue-600" />
                  </div>
                  <h3 className="font-semibold">区域设置</h3>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <InputGroup label="最小 X" value={boundary.minX} onChange={(v) => setBoundary({ ...boundary, minX: v })} />
                  <InputGroup label="最小 Y" value={boundary.minY} onChange={(v) => setBoundary({ ...boundary, minY: v })} />
                  <InputGroup label="最大 X" value={boundary.maxX} onChange={(v) => setBoundary({ ...boundary, maxX: v })} />
                  <InputGroup label="最大 Y" value={boundary.maxY} onChange={(v) => setBoundary({ ...boundary, maxY: v })} />
                </div>
                <Button
                  onClick={handleAnalyzeSelection}
                  disabled={analyzeMutation.isPending}
                  className="w-full mt-4"
                  size="lg"
                  shine
                >
                  {analyzeMutation.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      开始分析
                    </>
                  )}
                </Button>
              </div>
            )}

            {/* 房间分析设置 */}
            {analysisType === 'room' && (
              <div className="acrylic-light rounded-2xl p-5 border border-white/20 fade-in">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500/20 to-green-600/20 flex items-center justify-center">
                    <Volume2 className="w-4 h-4 text-green-600" />
                  </div>
                  <h3 className="font-semibold">房间设置</h3>
                </div>
                <div className="space-y-4">
                  <InputGroup label="房间 ID" value={roomId} onChange={setRoomId} />
                  <InputGroup label="房间高度 (m)" value={roomHeight} onChange={setRoomHeight} step={0.1} />
                  <div>
                    <label className="text-sm text-muted-foreground">混响公式</label>
                    <select
                      value={formula}
                      onChange={(e) =>
                        setFormula(e.target.value as 'SABINE' | 'EYRING' | 'AUTO')
                      }
                      className="w-full mt-1 px-3 py-2.5 border rounded-lg bg-background/50 backdrop-blur text-sm transition-colors hover:border-primary/50"
                    >
                      <option value="SABINE">SABINE (低吸声)</option>
                      <option value="EYRING">EYRING (高吸声)</option>
                      <option value="AUTO">AUTO (自动)</option>
                    </select>
                  </div>
                </div>
                <Button
                  onClick={handleAnalyzeRoom}
                  disabled={analyzeMutation.isPending}
                  className="w-full mt-4"
                  size="lg"
                  shine
                >
                  {analyzeMutation.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      开始分析
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>

          {/* 右侧：分析结果 */}
          <div className="col-span-2">
            {result ? (
              <div className="space-y-4 fade-in">
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-muted-foreground" />
                  <h2 className="text-sm font-medium">分析结果</h2>
                </div>

                {/* 选区材料统计结果 */}
                {result.result.type === 'SELECTION_MATERIAL_STATS' && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <StatCard 
                        label="总面积" 
                        value={`${result.result.total_area?.toFixed(2)} m²`} 
                        gradient="from-blue-500 to-cyan-500"
                      />
                      <StatCard 
                        label="表面数量" 
                        value={result.metrics.surface_count.toString()} 
                        gradient="from-green-500 to-emerald-500"
                      />
                      <StatCard 
                        label="计算时间" 
                        value={`${result.metrics.computation_time_ms} ms`} 
                        gradient="from-purple-500 to-pink-500"
                      />
                    </div>

                    {/* 材料分布 */}
                    {result.result.material_distribution && (
                      <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                        <h3 className="font-semibold mb-4">材料分布</h3>
                        <div className="space-y-3">
                          {result.result.material_distribution.map((mat, i) => (
                            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-background/50">
                              <div>
                                <div className="font-medium">{mat.material_name}</div>
                                <div className="text-sm text-muted-foreground">
                                  {mat.area.toFixed(2)} m²
                                </div>
                              </div>
                              <div className="text-lg font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                                {mat.percentage.toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* 吸声系数 */}
                    {result.result.equivalent_absorption_area && (
                      <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                        <h3 className="font-semibold mb-4">等效吸声面积</h3>
                        <div className="grid grid-cols-3 gap-4">
                          {Object.entries(result.result.equivalent_absorption_area).map(
                            ([freq, value]) => (
                              <div key={freq} className="p-3 rounded-lg bg-background/50 text-center">
                                <div className="text-sm text-muted-foreground">{freq}</div>
                                <div className="text-lg font-bold">{value.toFixed(2)} m²</div>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 房间混响时间结果 */}
                {result.result.type === 'ROOM_REVERBERATION' && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <StatCard 
                        label="房间体积" 
                        value={`${result.result.volume?.toFixed(2)} m³`} 
                        gradient="from-blue-500 to-cyan-500"
                      />
                      <StatCard 
                        label="总表面积" 
                        value={`${result.result.total_surface_area?.toFixed(2)} m²`} 
                        gradient="from-green-500 to-emerald-500"
                      />
                      <StatCard 
                        label="公式" 
                        value={result.result.formula || 'N/A'} 
                        gradient="from-purple-500 to-pink-500"
                      />
                    </div>

                    {/* T60 混响时间 */}
                    {result.result.t60 && (
                      <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                        <h3 className="font-semibold mb-4">混响时间 T60 (秒)</h3>
                        <div className="grid grid-cols-6 gap-4">
                          {Object.entries(result.result.t60).map(([freq, value]) => (
                            <div key={freq} className="p-4 rounded-xl bg-gradient-to-br from-primary/10 to-purple-500/10 text-center">
                              <div className="text-sm text-muted-foreground mb-1">{freq}</div>
                              <div className="text-2xl font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">
                                {value.toFixed(2)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* EDT 早期衰减时间 */}
                    {result.result.edt && (
                      <div className="acrylic-light rounded-2xl p-5 border border-white/20">
                        <h3 className="font-semibold mb-4">早期衰减时间 EDT (秒)</h3>
                        <div className="grid grid-cols-6 gap-4">
                          {Object.entries(result.result.edt).map(([freq, value]) => (
                            <div key={freq} className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 text-center">
                              <div className="text-sm text-muted-foreground mb-1">{freq}</div>
                              <div className="text-2xl font-bold bg-gradient-to-r from-green-500 to-emerald-500 bg-clip-text text-transparent">
                                {value.toFixed(2)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center space-y-4 fade-in">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center">
                    <Waves className="w-8 h-8 text-purple-500/50" />
                  </div>
                  <p className="text-muted-foreground">选择分析类型并点击开始</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function InputGroup({ 
  label, 
  value, 
  onChange, 
  step = 1 
}: { 
  label: string
  value: number
  onChange: (v: number) => void
  step?: number
}) {
  return (
    <div>
      <label className="text-sm text-muted-foreground">{label}</label>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 px-3 py-2.5 border rounded-lg bg-background/50 backdrop-blur text-sm transition-all duration-300 hover:border-primary/50 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
      />
    </div>
  )
}

function StatCard({ 
  label, 
  value, 
  gradient 
}: { 
  label: string
  value: string | number
  gradient: string
}) {
  return (
    <div className="acrylic-strong rounded-2xl p-5 border border-white/20 hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
      <div className="text-sm text-muted-foreground mb-2">{label}</div>
      <div className={`text-2xl font-bold bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>
        {value}
      </div>
    </div>
  )
}
