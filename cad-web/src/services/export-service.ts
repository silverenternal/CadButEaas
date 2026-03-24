import { apiClient } from './api-client'
import type {
  ExportRequest,
  ExportResponse,
} from '@/types/api'

export class ExportService {
  /**
   * 导出场景
   */
  async export(request: ExportRequest): Promise<ExportResponse> {
    return apiClient.post<ExportResponse>('/export', request)
  }

  /**
   * 下载文件
   */
  async download(fileName: string): Promise<Blob> {
    return apiClient.download(`/download/${fileName}`)
  }

  /**
   * 导出为 JSON 并下载
   */
  async exportAndDownload(format: 'json' | 'bincode' = 'json', pretty = true) {
    const result = await this.export({ format, pretty })
    
    if (result.success && result.download_url) {
      const blob = await this.download(result.file_name!)
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = result.file_name!
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
      return { success: true, fileName: result.file_name }
    }
    
    return { success: false, error: result.message }
  }
}

export const exportService = new ExportService()
