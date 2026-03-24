import { apiClient } from './api-client'
import type {
  AutoTraceRequest,
  AutoTraceResponse,
  LassoRequest,
  LassoResponse,
  GapDetectionRequest,
  GapDetectionResponse,
  SetSemanticRequest,
  SetSemanticResponse,
  SnapBridgeRequest,
} from '@/types/api'

export class InteractionService {
  /**
   * 自动追踪
   */
  async autoTrace(request: AutoTraceRequest): Promise<AutoTraceResponse> {
    return apiClient.post<AutoTraceResponse>('/interact/auto_trace', request)
  }

  /**
   * 圈选区域
   */
  async lasso(request: LassoRequest): Promise<LassoResponse> {
    return apiClient.post<LassoResponse>('/interact/lasso', request)
  }

  /**
   * 缺口检测
   */
  async detectGaps(request: GapDetectionRequest): Promise<GapDetectionResponse> {
    return apiClient.post<GapDetectionResponse>('/interact/detect_gaps', request)
  }

  /**
   * 缺口桥接
   */
  async snapBridge(request: SnapBridgeRequest): Promise<void> {
    return apiClient.post<void>('/interact/snap_bridge', request)
  }

  /**
   * 设置边界语义
   */
  async setSemantic(request: SetSemanticRequest): Promise<SetSemanticResponse> {
    return apiClient.post<SetSemanticResponse>(
      '/interact/set_boundary_semantic',
      request
    )
  }

  /**
   * 获取交互状态
   */
  async getState(): Promise<{
    total_edges: number
    selected_edges: number[]
    detected_gaps: Array<{
      id: number
      start: [number, number]
      end: [number, number]
      length: number
      gap_type: string
    }>
  }> {
    return apiClient.get('/interact/state')
  }
}

export const interactionService = new InteractionService()
