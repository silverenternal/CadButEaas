import { apiClient } from './api-client'
import type {
  ProcessResponse,
  Profile,
  ProfileDetail,
} from '@/types/api'

export class FileService {
  /**
   * 处理文件
   */
  async processFile(
    file: File,
    profile?: 'architectural' | 'mechanical' | 'scanned' | 'quick',
    onProgress?: (progress: number) => void
  ): Promise<ProcessResponse> {
    const endpoint = profile ? `/process?profile=${profile}` : '/process'
    return apiClient.upload<ProcessResponse>(endpoint, file, onProgress)
  }

  /**
   * 获取所有预设配置
   */
  async listProfiles(): Promise<Profile[]> {
    return apiClient.get('/config/profiles')
  }

  /**
   * 获取配置详情
   */
  async getProfile(name: string): Promise<ProfileDetail> {
    return apiClient.get(`/config/profile/${name}`)
  }

  /**
   * 导出场景
   */
  async exportScene(format: 'json' | 'binary' = 'json'): Promise<Blob> {
    return apiClient.download('/export', { format })
  }

  /**
   * 下载导出文件
   */
  async downloadExport(
    format: 'json' | 'binary' = 'json',
    filename?: string
  ): Promise<void> {
    const blob = await this.exportScene(format)
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = filename ?? `scene.${format === 'json' ? 'json' : 'bin'}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }
}

export const fileService = new FileService()
