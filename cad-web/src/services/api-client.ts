import { ErrorResponseSchema } from '@/types/api'

// 基础配置
const API_BASE_URL = (import.meta as any).env.VITE_API_URL || 'http://localhost:3000'
const API_TIMEOUT = 30000 // 30 秒

export class ApiError extends Error {
  constructor(
    public readonly data: {
      request_id: string
      status: string
      error: {
        code: string
        message: string
        details?: Record<string, unknown>
        retryable?: boolean
        suggestion?: string
      }
      latency_ms: number
    }
  ) {
    super(data.error.message)
    this.name = 'ApiError'
  }

  get code(): string {
    return this.data.error.code
  }

  get retryable(): boolean {
    return this.data.error.retryable ?? false
  }

  get suggestion(): string | undefined {
    return this.data.error.suggestion
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class TimeoutError extends Error {
  constructor() {
    super('请求超时')
    this.name = 'TimeoutError'
  }
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly field?: string
  ) {
    super(message)
    this.name = 'ValidationError'
  }
}

class ApiClient {
  private baseUrl: string
  private defaultHeaders: HeadersInit

  constructor(baseUrl: string, defaultHeaders: HeadersInit = {}) {
    this.baseUrl = baseUrl
    this.defaultHeaders = defaultHeaders
  }

  /**
   * 获取基础 URL
   */
  getBaseUrl(): string {
    return this.baseUrl
  }

  /**
   * 通用请求方法（P2-1 修复：添加重试机制）
   */
  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    const MAX_RETRIES = 3
    const RETRY_DELAY = 1000 // 1 秒

    let lastError: Error | undefined

    // P2-1 新增：重试循环
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const config: RequestInit = {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
        },
      }

      // 添加超时控制
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT)
      config.signal = controller.signal

      try {
        const response = await fetch(url, config)

        if (!response.ok) {
          // P2-1 修复：5xx 错误可重试
          if (response.status >= 500 && attempt < MAX_RETRIES) {
            const delay = RETRY_DELAY * Math.pow(2, attempt) // 指数退避
            await new Promise(resolve => setTimeout(resolve, delay))
            continue
          }

          const errorData = await response.json().catch(() => ({}))
          const parsedError = ErrorResponseSchema.safeParse(errorData)

          if (parsedError.success) {
            throw new ApiError(parsedError.data)
          } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`)
          }
        }

        const data = await response.json()
        return data as T
      } catch (error) {
        clearTimeout(timeoutId)

        // P2-1 修复：网络错误和超时错误可重试
        if (error instanceof Error && error.name === 'AbortError') {
          lastError = new TimeoutError()
          if (attempt < MAX_RETRIES) {
            const delay = RETRY_DELAY * Math.pow(2, attempt)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue
          }
          throw lastError
        }

        if (error instanceof TypeError && error.message.includes('fetch')) {
          lastError = new NetworkError('网络连接失败')
          if (attempt < MAX_RETRIES) {
            const delay = RETRY_DELAY * Math.pow(2, attempt)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue
          }
          throw lastError
        }

        // 其他错误直接抛出
        throw error
      } finally {
        clearTimeout(timeoutId)
      }
    }

    // 理论上不会到这里，但为了类型安全
    throw lastError || new Error('请求失败')
  }

  /**
   * GET 请求
   */
  async get<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    const queryString = params
      ? '?' + new URLSearchParams(params).toString()
      : ''
    return this.request<T>(endpoint + queryString, { method: 'GET' })
  }

  /**
   * POST 请求
   */
  async post<T, B = unknown>(endpoint: string, body?: B): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
  }

  /**
   * 文件上传
   */
  async upload<T>(
    endpoint: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData()
    formData.append('file', file)

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.open('POST', `${this.baseUrl}${endpoint}`, true)

      Object.entries(this.defaultHeaders).forEach(([key, value]) => {
        if (key.toLowerCase() !== 'content-type') {
          xhr.setRequestHeader(key, value)
        }
      })

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = (event.loaded / event.total) * 100
          onProgress(Math.round(progress))
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText))
          } catch (error) {
            reject(new Error('响应解析失败'))
          }
        } else {
          // 尝试解析错误响应
          try {
            const errorData = JSON.parse(xhr.responseText)
            const message = errorData.message || errorData.error?.message || `HTTP ${xhr.status}: ${xhr.statusText}`
            reject(new NetworkError(message))
          } catch {
            reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`))
          }
        }
      }

      xhr.onerror = () => {
        // 尝试获取更详细的错误信息
        if (xhr.status === 0) {
          reject(new NetworkError('无法连接到服务器，请检查后端服务是否运行'))
        } else if (xhr.status === 413) {
          reject(new NetworkError('文件过大，请上传小于 50MB 的文件'))
        } else if (xhr.status === 400) {
          reject(new NetworkError('请求格式错误'))
        } else if (xhr.status === 500) {
          reject(new NetworkError('服务器内部错误'))
        } else {
          reject(new NetworkError(`网络错误 (HTTP ${xhr.status})`))
        }
      }

      xhr.send(formData)
    })
  }

  /**
   * 下载文件
   */
  async download(endpoint: string, params?: Record<string, string>): Promise<Blob> {
    const queryString = params
      ? '?' + new URLSearchParams(params).toString()
      : ''
    
    const response = await fetch(`${this.baseUrl}${endpoint}${queryString}`, {
      headers: this.defaultHeaders,
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    return response.blob()
  }
}

export const apiClient = new ApiClient(API_BASE_URL, {
  Accept: 'application/json',
})
