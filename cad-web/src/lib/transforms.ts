/**
 * 坐标转换工具
 * 
 * 提供世界坐标和屏幕坐标之间的转换功能
 * 
 * @module lib/transforms
 */

import type { CameraState, Point } from '@/types/api'

/**
 * 将世界坐标转换为屏幕坐标
 * 
 * 转换公式：screen = world * zoom + offset
 * 
 * @param point - 世界坐标点 [x, y]
 * @param camera - 相机状态（zoom, offsetX, offsetY）
 * @returns 屏幕坐标点 [x, y]
 * 
 * @example
 * ```typescript
 * const camera = { zoom: 2, offsetX: 100, offsetY: 100 }
 * const worldPoint: Point = [50, 50]
 * const screenPoint = worldToScreen(worldPoint, camera)
 * // screenPoint = [200, 200]  // (50 * 2) + 100
 * ```
 */
export function worldToScreen(point: Point, camera: CameraState): Point {
  return [
    point[0] * camera.zoom + camera.offsetX,
    point[1] * camera.zoom + camera.offsetY,
  ]
}

/**
 * 将屏幕坐标转换为世界坐标
 * 
 * 转换公式：world = (screen - offset) / zoom
 * 
 * @param point - 屏幕坐标点 [x, y]
 * @param camera - 相机状态（zoom, offsetX, offsetY）
 * @returns 世界坐标点 [x, y]
 * 
 * @example
 * ```typescript
 * const camera = { zoom: 2, offsetX: 100, offsetY: 100 }
 * const screenPoint: Point = [200, 200]
 * const worldPoint = screenToWorld(screenPoint, camera)
 * // worldPoint = [50, 50]  // (200 - 100) / 2
 * ```
 */
export function screenToWorld(point: Point, camera: CameraState): Point {
  return [
    (point[0] - camera.offsetX) / camera.zoom,
    (point[1] - camera.offsetY) / camera.zoom,
  ]
}

/**
 * 计算边界框
 * 
 * @param points - 点数组
 * @returns 边界框 { minX, minY, maxX, maxY }
 */
export function calculateBoundingBox(points: Point[]): {
  minX: number
  minY: number
  maxX: number
  maxY: number
} {
  if (points.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0 }
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  points.forEach((point) => {
    minX = Math.min(minX, point[0])
    minY = Math.min(minY, point[1])
    maxX = Math.max(maxX, point[0])
    maxY = Math.max(maxY, point[1])
  })

  return { minX, minY, maxX, maxY }
}

/**
 * 计算适合内容的相机参数
 * 
 * 自动计算缩放和偏移，使内容适配视口
 * 
 * @param boundingBox - 内容边界框
 * @param viewportWidth - 视口宽度（像素）
 * @param viewportHeight - 视口高度（像素）
 * @param padding - 内边距比例（默认 0.1，即 90% 填充）
 * @returns 相机参数 { zoom, offsetX, offsetY }
 */
export function fitToContent(
  boundingBox: { minX: number; minY: number; maxX: number; maxY: number },
  viewportWidth: number,
  viewportHeight: number,
  padding: number = 0.1
): { zoom: number; offsetX: number; offsetY: number } {
  const contentWidth = boundingBox.maxX - boundingBox.minX
  const contentHeight = boundingBox.maxY - boundingBox.minY

  // 避免除以零
  if (contentWidth === 0 || contentHeight === 0) {
    return { zoom: 1, offsetX: 0, offsetY: 0 }
  }

  // 计算缩放比例（保留 padding）
  const scaleX = viewportWidth / contentWidth
  const scaleY = viewportHeight / contentHeight
  const zoom = Math.min(scaleX, scaleY) * (1 - padding)

  // 计算偏移（居中内容）
  const offsetX = (viewportWidth - contentWidth * zoom) / 2 - boundingBox.minX * zoom
  const offsetY = (viewportHeight - contentHeight * zoom) / 2 - boundingBox.minY * zoom

  return { zoom, offsetX, offsetY }
}
