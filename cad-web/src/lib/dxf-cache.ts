/**
 * DXF 文件缓存工具
 * 使用 LRUCache 实现文件解析结果缓存，避免重复解析
 * ✅ S009: 添加内存限制和主动清理机制
 */

export interface CacheEntry {
  data: any
  timestamp: number
  fileHash: string
  estimatedSize: number // 估算的内存占用（字节）
}

const MAX_CACHE_SIZE = 100 // 最多缓存 100 个文件
const MAX_CACHE_AGE = 1000 * 60 * 60 * 24 // 24 小时过期
const MAX_CACHE_MEMORY_MB = 50 // ✅ S009: 最大内存占用 50MB
const CLEANUP_INTERVAL = 60000 // ✅ S009: 每分钟主动清理一次

// 简单的 LRU 缓存实现（不依赖外部库）
class SimpleLRUCache<K, V> {
  private cache: Map<K, V>
  private maxSize: number

  constructor(maxSize: number) {
    this.cache = new Map()
    this.maxSize = maxSize
  }

  get(key: K): V | undefined {
    const value = this.cache.get(key)
    if (value !== undefined) {
      // 访问后移到末尾（最新）
      this.cache.delete(key)
      this.cache.set(key, value)
    }
    return value
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key)
    } else if (this.cache.size >= this.maxSize) {
      // 删除最旧的条目
      const firstKey = this.cache.keys().next().value
      if (firstKey !== undefined) {
        this.cache.delete(firstKey)
      }
    }
    this.cache.set(key, value)
  }

  clear(): void {
    this.cache.clear()
  }

  delete(key: K): boolean {
    return this.cache.delete(key)
  }

  has(key: K): boolean {
    return this.cache.has(key)
  }

  get size(): number {
    return this.cache.size
  }

  // ✅ S009: 获取所有键值对用于内存计算
  entries(): IterableIterator<[K, V]> {
    return this.cache.entries()
  }

  // ✅ S009: 获取第一个键（最旧的）
  firstKey(): K | undefined {
    return this.cache.keys().next().value
  }
}

const cache = new SimpleLRUCache<string, CacheEntry>(MAX_CACHE_SIZE)

// ✅ S009: 主动清理定时器
let cleanupTimer: NodeJS.Timeout | null = null

/**
 * ✅ S009: 启动主动清理机制
 */
function startCleanupTimer(): void {
  if (cleanupTimer) return
  
  cleanupTimer = setInterval(() => {
    cleanupExpired()
    cleanupMemoryLimit()
  }, CLEANUP_INTERVAL)
}

/**
 * ✅ S009: 清理过期缓存
 */
function cleanupExpired(): void {
  const now = Date.now()
  const expiredKeys: string[] = []

  for (const [key, entry] of cache.entries()) {
    if (now - entry.timestamp > MAX_CACHE_AGE) {
      expiredKeys.push(key)
    }
  }

  expiredKeys.forEach(key => cache.delete(key))
  
  if (expiredKeys.length > 0) {
    console.log('[DXFCache] Cleaned up', expiredKeys.length, 'expired entries')
  }
}

/**
 * ✅ S009: 根据内存限制清理缓存
 */
function cleanupMemoryLimit(): void {
  let totalMemory = 0
  const entries: Array<{ key: string; entry: CacheEntry }> = []

  // 计算总内存
  for (const [key, entry] of cache.entries()) {
    totalMemory += entry.estimatedSize
    entries.push({ key, entry })
  }

  const maxMemoryBytes = MAX_CACHE_MEMORY_MB * 1024 * 1024
  
  if (totalMemory > maxMemoryBytes) {
    console.log('[DXFCache] Memory limit exceeded, cleaning up...')
    
    // 按时间排序，优先删除最旧的
    entries.sort((a, b) => a.entry.timestamp - b.entry.timestamp)
    
    let freedMemory = 0
    for (const { key, entry } of entries) {
      if (totalMemory - freedMemory <= maxMemoryBytes) break
      
      cache.delete(key)
      freedMemory += entry.estimatedSize
      console.log('[DXFCache] Evicted entry:', key)
    }
    
    console.log('[DXFCache] Freed', (freedMemory / 1024 / 1024).toFixed(2), 'MB')
  }
}

/**
 * 计算文件的 SHA-256 哈希值
 */
export async function fileHash(file: File): Promise<string> {
  const buffer = await file.arrayBuffer()
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * 从缓存获取数据
 */
export function getFromCache(hash: string): CacheEntry | undefined {
  const entry = cache.get(hash)

  if (entry) {
    // 检查是否过期
    const isExpired = Date.now() - entry.timestamp > MAX_CACHE_AGE
    if (isExpired) {
      cache.delete(hash)
      return undefined
    }
  }

  return entry
}

/**
 * 设置缓存
 * ✅ S009: 估算内存占用
 */
export function setCache(hash: string, data: any): void {
  // 估算内存占用（edges + hatches 的 JSON 大小）
  const estimatedSize = estimateCacheSize(data)
  
  cache.set(hash, {
    data,
    timestamp: Date.now(),
    fileHash: hash,
    estimatedSize,
  })
  
  // 检查是否超过内存限制
  cleanupMemoryLimit()
}

/**
 * ✅ S009: 估算缓存数据的内存占用
 */
function estimateCacheSize(data: { edges?: any[]; hatches?: any[] }): number {
  let size = 0
  
  if (data.edges) {
    // 每条边约 100 字节
    size += data.edges.length * 100
  }
  
  if (data.hatches) {
    // 每个填充约 500 字节
    size += data.hatches.length * 500
  }
  
  return size
}

/**
 * 清除缓存
 */
export function clearCache(): void {
  cache.clear()
  if (cleanupTimer) {
    clearInterval(cleanupTimer)
    cleanupTimer = null
  }
}

/**
 * 获取缓存大小
 */
export function getCacheSize(): number {
  return cache.size
}

/**
 * 从缓存中删除指定文件
 */
export function removeFromCache(hash: string): boolean {
  return cache.delete(hash)
}

/**
 * ✅ S009: 获取缓存统计信息
 */
export function getCacheStats(): {
  size: number
  estimatedMemoryMB: number
  maxMemoryMB: number
} {
  let totalMemory = 0
  
  for (const [, entry] of cache.entries()) {
    totalMemory += entry.estimatedSize
  }
  
  return {
    size: cache.size,
    estimatedMemoryMB: totalMemory / 1024 / 1024,
    maxMemoryMB: MAX_CACHE_MEMORY_MB,
  }
}

/**
 * ✅ S009: 初始化缓存系统（启动主动清理）
 */
export function initCache(): void {
  startCleanupTimer()
  console.log('[DXFCache] Initialized with max memory:', MAX_CACHE_MEMORY_MB, 'MB')
}

// ✅ S009: 模块加载时自动初始化
initCache()
