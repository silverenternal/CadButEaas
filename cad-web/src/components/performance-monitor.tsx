/**
 * 性能监控面板
 * ✅ S018: 实时显示 Draw Call、内存、FPS 等性能指标
 */

import { useState, useEffect, useRef } from 'react'
import { getCacheStats } from '@/lib/dxf-cache'

export interface PerformanceStats {
  // 渲染性能
  fps: number
  drawCalls: number
  triangleCount: number
  lineCount: number
  
  // 内存占用
  jsHeapSize: number
  jsHeapUsed: number
  cacheMemoryMB: number
  
  // 缓存统计
  cacheSize: number
  cacheHitRate: number
  
  // 几何数据
  edgeCount: number
  hatchCount: number
}

export interface PerformanceMonitorProps {
  /** 是否显示监控面板 */
  enabled?: boolean
  /** 刷新间隔 (ms) */
  refreshInterval?: number
  /** 自定义统计信息 */
  customStats?: Partial<PerformanceStats>
}

/**
 * 性能监控面板组件
 */
export function PerformanceMonitor({
  enabled = false,
  refreshInterval = 1000,
  customStats,
}: PerformanceMonitorProps) {
  const [stats, setStats] = useState<PerformanceStats>({
    fps: 0,
    drawCalls: 0,
    triangleCount: 0,
    lineCount: 0,
    jsHeapSize: 0,
    jsHeapUsed: 0,
    cacheMemoryMB: 0,
    cacheSize: 0,
    cacheHitRate: 0,
    edgeCount: 0,
    hatchCount: 0,
  })

  // 更新缓存统计
  useEffect(() => {
    if (!enabled) return

    const updateCacheStats = () => {
      const cacheStats = getCacheStats()
      setStats(prev => ({
        ...prev,
        cacheSize: cacheStats.size,
        cacheMemoryMB: cacheStats.estimatedMemoryMB,
      }))
    }

    updateCacheStats()
    const interval = setInterval(updateCacheStats, refreshInterval)
    return () => clearInterval(interval)
  }, [enabled, refreshInterval])

  // 更新性能统计 (FPS, Draw Calls 等)
  useEffect(() => {
    if (!enabled) return

    let frameCount = 0
    let lastTime = performance.now()
    let fps = 0

    const updatePerformanceStats = () => {
      frameCount++
      const currentTime = performance.now()
      const delta = currentTime - lastTime

      if (delta >= 1000) {
        fps = Math.round((frameCount * 1000) / delta)
        frameCount = 0
        lastTime = currentTime

        // 获取内存信息 (如果浏览器支持)
        let jsHeapSize = 0
        let jsHeapUsed = 0
        if (typeof performance !== 'undefined' && 'memory' in performance) {
          const memory = (performance as any).memory
          jsHeapSize = Math.round(memory.jsHeapSizeLimit / 1024 / 1024)
          jsHeapUsed = Math.round(memory.usedJSHeapSize / 1024 / 1024)
        }

        setStats(prev => ({
          ...prev,
          fps,
          jsHeapSize,
          jsHeapUsed,
          ...customStats,
        }))
      }

      if (enabled) {
        requestAnimationFrame(updatePerformanceStats)
      }
    }

    requestAnimationFrame(updatePerformanceStats)

    return () => {
      enabled = false
    }
  }, [enabled, customStats])

  if (!enabled) {
    return null
  }

  return (
    <div
      style={{
        position: 'fixed',
        top: 16,
        right: 16,
        zIndex: 9999,
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        borderRadius: 8,
        padding: 12,
        color: '#fff',
        fontFamily: 'monospace',
        fontSize: 11,
        lineHeight: 1.6,
        minWidth: 220,
        maxWidth: 280,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      {/* 标题 */}
      <div
        style={{
          fontSize: 12,
          fontWeight: 'bold',
          marginBottom: 8,
          paddingBottom: 6,
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
          color: '#60a5fa',
        }}
      >
        📊 性能监控
      </div>

      {/* 渲染性能 */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ color: '#9ca3af', fontSize: 10, marginBottom: 4 }}>渲染性能</div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>FPS</span>
          <span style={{ color: getFpsColor(stats.fps) }}>{stats.fps}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Draw Calls</span>
          <span style={{ color: getDrawCallColor(stats.drawCalls) }}>{stats.drawCalls}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Triangles</span>
          <span>{stats.triangleCount.toLocaleString()}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Lines</span>
          <span>{stats.lineCount.toLocaleString()}</span>
        </div>
      </div>

      {/* 内存占用 */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ color: '#9ca3af', fontSize: 10, marginBottom: 4 }}>内存占用</div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>JS Heap</span>
          <span>{stats.jsHeapUsed} / {stats.jsHeapSize || 'N/A'} MB</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Cache</span>
          <span style={{ color: getMemoryColor(stats.cacheMemoryMB) }}>{stats.cacheMemoryMB.toFixed(1)} MB</span>
        </div>
      </div>

      {/* 缓存统计 */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ color: '#9ca3af', fontSize: 10, marginBottom: 4 }}>缓存统计</div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>缓存数量</span>
          <span>{stats.cacheSize}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>命中率</span>
          <span>{stats.cacheHitRate.toFixed(0)}%</span>
        </div>
      </div>

      {/* 几何数据 */}
      <div>
        <div style={{ color: '#9ca3af', fontSize: 10, marginBottom: 4 }}>几何数据</div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Edges</span>
          <span>{stats.edgeCount.toLocaleString()}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Hatches</span>
          <span>{stats.hatchCount.toLocaleString()}</span>
        </div>
      </div>
    </div>
  )
}

/**
 * 根据 FPS 值返回颜色
 */
function getFpsColor(fps: number): string {
  if (fps >= 55) return '#22c55e' // 绿色 - 优秀
  if (fps >= 30) return '#eab308' // 黄色 - 良好
  return '#ef4444' // 红色 - 差
}

/**
 * 根据 Draw Call 数量返回颜色
 */
function getDrawCallColor(drawCalls: number): string {
  if (drawCalls <= 50) return '#22c55e' // 绿色 - 优秀
  if (drawCalls <= 200) return '#eab308' // 黄色 - 良好
  return '#ef4444' // 红色 - 差
}

/**
 * 根据内存占用返回颜色
 */
function getMemoryColor(memoryMB: number): string {
  if (memoryMB <= 30) return '#22c55e' // 绿色 - 优秀
  if (memoryMB <= 50) return '#eab308' // 黄色 - 良好
  return '#ef4444' // 红色 - 差
}

/**
 * 性能监控 Hook - 用于在组件外部更新统计
 */
export function usePerformanceMonitor() {
  const updateStatsRef = useRef<((stats: Partial<PerformanceStats>) => void) | null>(null)

  const updateStats = (stats: Partial<PerformanceStats>) => {
    if (updateStatsRef.current) {
      updateStatsRef.current(stats)
    }
  }

  return { updateStats }
}
