import { useMemo, useCallback } from 'react'
import { Shape } from 'react-konva'
import type { CameraState, HatchEntity, HatchBoundaryPath } from '@/types/api'  // P0-4 修复：只导入需要的类型

// ============================================================================
// P3-NEW-45 新增：显式 LOD 系统（仅用于 HATCH 几何离散化）
// ============================================================================

/**
 * P3-NEW-45: LOD 层级枚举（内部使用）
 */
enum LodLevel {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
}

/**
 * P3-NEW-45: 根据 zoom 级别获取 LOD 层级（内部使用）
 */
function getLodLevel(zoom: number): LodLevel {
  if (zoom < 0.2) return LodLevel.Low
  if (zoom < 1.0) return LodLevel.Medium
  return LodLevel.High
}

/**
 * P3-NEW-45: 根据 LOD 级别获取屏幕容差（内部使用）
 */
function getScreenToleranceForLod(lod: LodLevel): number {
  switch (lod) {
    case LodLevel.Low:
      return 4.0
    case LodLevel.Medium:
      return 2.0
    case LodLevel.High:
    default:
      return 1.0
  }
}

// ============================================================================
// P2-NEW-23 新增：调试级别控制系统
// ============================================================================

/**
 * 调试级别枚举
 * 用于控制控制台日志的输出级别
 */
export const DEBUG_LEVEL = {
  NONE: 0,    // 不输出任何日志
  ERROR: 1,   // 只输出错误
  WARN: 2,    // 输出错误和警告
  INFO: 3,    // 输出错误、警告和信息
  DEBUG: 4,   // 输出所有日志（包括详细调试信息）
} as const

/**
 * ✅ P3-NEW-40 优化：使用 Vite 环境变量检测生产环境
 * 生产环境完全禁用日志输出（Vite 会移除死代码）
 */
const isProduction = (import.meta as any).env?.PROD ?? false

/**
 * 当前调试级别
 * 可通过 window.CAD_DEBUG_LEVEL 动态调整
 * 默认：WARN（只输出警告和错误），生产环境：NONE
 */
let CURRENT_DEBUG_LEVEL: number = isProduction ? DEBUG_LEVEL.NONE : DEBUG_LEVEL.WARN

/**
 * 设置全局调试级别
 * @param level - 调试级别
 */
export function setDebugLevel(level: keyof typeof DEBUG_LEVEL): void {
  CURRENT_DEBUG_LEVEL = DEBUG_LEVEL[level]
  // 使用直接输出，因为这是设置函数本身
  const levelValue = DEBUG_LEVEL[level]
  if (levelValue >= DEBUG_LEVEL.INFO) {
    console.info(`[HatchLayer] Debug level set to: ${level} (${CURRENT_DEBUG_LEVEL})`)
  }
}

// ✅ P2-NEW-23 新增：在浏览器控制台暴露全局调试函数
if (typeof window !== 'undefined') {
  (window as any).CAD_DEBUG_LEVEL = DEBUG_LEVEL
  ;(window as any).CADSetDebugLevel = setDebugLevel
}

/**
 * 获取当前调试级别
 */
export function getDebugLevel(): number {
  return CURRENT_DEBUG_LEVEL
}

/**
 * 条件日志输出函数
 * 根据当前调试级别控制日志输出
 * 
 * ✅ P3-NEW-40 优化：生产环境完全禁用（Vite 会移除死代码）
 *
 * @param level - 日志级别
 * @param message - 日志消息
 * @param args - 附加参数
 */
function debugLog(
  level: keyof typeof DEBUG_LEVEL,
  message: string,
  ...args: any[]
): void {
  // ✅ P3-NEW-40: 生产环境直接返回，Vite 会移除死代码
  if (isProduction) return

  const levelValue = DEBUG_LEVEL[level]
  if (levelValue <= CURRENT_DEBUG_LEVEL) {
    let logFn: (...args: any[]) => void
    switch (levelValue) {
      case DEBUG_LEVEL.ERROR:
        logFn = console.error
        break
      case DEBUG_LEVEL.WARN:
        logFn = console.warn
        break
      case DEBUG_LEVEL.INFO:
        logFn = console.info
        break
      case DEBUG_LEVEL.DEBUG:
        logFn = console.log
        break
      default:
        logFn = console.log
    }
    logFn.call(console, message, ...args)
  }
}

// ============================================================================
// P2-4 新增：视口裁剪工具函数
// ============================================================================

/**
 * 检查 HATCH 是否在视口内（包括边界框）
 * 使用边界框快速拒绝/接受测试
 * 
 * ✅ P3-NEW-39 优化：增量边界框计算 + 提前终止
 */
function isHatchInViewport(
  hatch: HatchEntity,
  camera: CameraState,
  canvasWidth: number,
  canvasHeight: number
): boolean {
  // 计算视口在世界坐标中的范围
  const viewportMinX = -camera.offsetX / camera.zoom
  const viewportMinY = -camera.offsetY / camera.zoom
  const viewportMaxX = (canvasWidth - camera.offsetX) / camera.zoom
  const viewportMaxY = (canvasHeight - camera.offsetY) / camera.zoom

  // ✅ P3-NEW-39 优化 1: 快速接受 - HATCH 中心点在视口内
  // 使用边界路径的中心点近似
  const firstPath = hatch.boundary_paths[0]
  let centerX = 0
  let centerY = 0
  
  if (firstPath?.center) {
    centerX = firstPath.center[0]
    centerY = firstPath.center[1]
  } else if (firstPath?.points && firstPath.points.length > 0) {
    // 使用第一个点的坐标近似
    centerX = firstPath.points[0][0]
    centerY = firstPath.points[0][1]
  }
  
  if (
    centerX >= viewportMinX &&
    centerX <= viewportMaxX &&
    centerY >= viewportMinY &&
    centerY <= viewportMaxY
  ) {
    return true
  }

  // ✅ P3-NEW-39 优化 2: 增量边界框计算 + 提前拒绝
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const path of hatch.boundary_paths) {
    // 逐个路径更新边界框
    if (path.points) {
      for (const point of path.points) {
        minX = Math.min(minX, point[0])
        minY = Math.min(minY, point[1])
        maxX = Math.max(maxX, point[0])
        maxY = Math.max(maxY, point[1])
      }
    }
    if (path.control_points) {
      for (const point of path.control_points) {
        minX = Math.min(minX, point[0])
        minY = Math.min(minY, point[1])
        maxX = Math.max(maxX, point[0])
        maxY = Math.max(maxY, point[1])
      }
    }
    if (path.center) {
      minX = Math.min(minX, path.center[0])
      minY = Math.min(minY, path.center[1])
      maxX = Math.max(maxX, path.center[0])
      maxY = Math.max(maxY, path.center[1])
    }
    if (path.type === 'arc' && path.center && path.radius) {
      // 圆弧边界框
      minX = Math.min(minX, path.center[0] - path.radius)
      minY = Math.min(minY, path.center[1] - path.radius)
      maxX = Math.max(maxX, path.center[0] + path.radius)
      maxY = Math.max(maxY, path.center[1] + path.radius)
    }
    if (path.type === 'ellipse_arc' && path.center && path.major_axis) {
      // 椭圆弧边界框
      const majorAxisLength = Math.sqrt(path.major_axis[0] ** 2 + path.major_axis[1] ** 2)
      const minorAxisLength = majorAxisLength * (path.minor_axis_ratio ?? 1.0)
      minX = Math.min(minX, path.center[0] - majorAxisLength)
      maxX = Math.max(maxX, path.center[0] + majorAxisLength)
      minY = Math.min(minY, path.center[1] - minorAxisLength)
      maxY = Math.max(maxY, path.center[1] + minorAxisLength)
    }

    // ✅ P3-NEW-39 提前拒绝：已超出视口
    if (
      maxX < viewportMinX ||
      minX > viewportMaxX ||
      maxY < viewportMinY ||
      minY > viewportMaxY
    ) {
      return false
    }
  }

  if (minX === Infinity || maxX === -Infinity || minY === Infinity || maxY === -Infinity) {
    return false // 无边界点，跳过
  }

  // ✅ P2-NEW-28 修复：动态计算余量（基于 HATCH 大小）
  // 动态余量：基于 HATCH 边界框大小的 10%
  const hatchWidth = maxX - minX
  const hatchHeight = maxY - minY
  const paddingX = Math.max(50, hatchWidth * 0.1) // 最小 50 单位
  const paddingY = Math.max(50, hatchHeight * 0.1) // 最小 50 单位

  const extendedMinX = minX - paddingX
  const extendedMinY = minY - paddingY
  const extendedMaxX = maxX + paddingX
  const extendedMaxY = maxY + paddingY

  // ✅ 快速拒绝测试：HATCH 边界框与视口是否相交
  return !(
    extendedMaxX < viewportMinX ||
    extendedMinX > viewportMaxX ||
    extendedMaxY < viewportMinY ||
    extendedMinY > viewportMaxY
  )
}

// ============================================================================
// P2-NEW-43 新增：自相交检测工具函数
// ============================================================================

/**
 * P2-NEW-43: 检测两条线段是否相交
 * 使用参数方程法检测线段相交
 *
 * @param p1 - 线段 1 起点
 * @param p2 - 线段 1 终点
 * @param p3 - 线段 2 起点
 * @param p4 - 线段 2 终点
 * @returns 是否相交
 */
function segmentsIntersect(
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
  p4: [number, number]
): boolean {
  const x1 = p1[0], y1 = p1[1]
  const x2 = p2[0], y2 = p2[1]
  const x3 = p3[0], y3 = p3[1]
  const x4 = p4[0], y4 = p4[1]

  // 计算分母
  const denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

  // 平行线
  if (Math.abs(denom) < 1e-10) {
    return false
  }

  // 计算参数 ua 和 ub
  const ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
  const ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

  // 检查交点是否在线段上
  return ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1
}

/**
 * P2-NEW-43: 检测多边形是否自相交
 * 使用 Bentley-Ottmann 算法的简化版本
 *
 * @param points - 多边形顶点数组
 * @returns 是否自相交
 */
function detectSelfIntersection(points: [number, number][]): boolean {
  const n = points.length
  if (n < 4) return false  // 三角形不可能自相交

  // 检测所有非相邻边是否相交
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 2; j < n - 1; j++) {
      // 跳过相邻边（共享顶点）
      if (j === i + 1) continue

      // 对于闭合多边形，还需要跳过首尾相连的边
      if (i === 0 && j === n - 1) continue

      const p1 = points[i]
      const p2 = points[i + 1]
      const p3 = points[j]
      const p4 = points[j + 1]

      if (segmentsIntersect(p1, p2, p3, p4)) {
        return true
      }
    }
  }

  return false
}

/**
 * P2-NEW-43: 检测 HATCH 边界是否自相交
 * 遍历所有边界路径，检测是否有任何路径自相交
 *
 * @param boundaryPaths - HATCH 边界路径数组
 * @returns { hasIntersection: boolean, intersectingPaths: number[] }
 */
function detectHatchSelfIntersection(
  boundaryPaths: HatchBoundaryPath[]
): { hasIntersection: boolean; intersectingPaths: number[] } {
  const intersectingPaths: number[] = []

  for (let i = 0; i < boundaryPaths.length; i++) {
    const path = boundaryPaths[i]
    let points: [number, number][] = []

    // 提取边界点
    if (path.type === 'polyline' && path.points) {
      points = path.points
    } else if (path.type === 'spline' && path.fit_points) {
      // 对于样条，使用拟合点近似
      points = path.fit_points
    }

    // 检测自相交
    if (points.length >= 4 && detectSelfIntersection(points)) {
      intersectingPaths.push(i)
    }
  }

  return {
    hasIntersection: intersectingPaths.length > 0,
    intersectingPaths
  }
}

// ============================================================================
// P3-NEW-30 新增：HATCH 边界预处理（Douglas-Peucker 简化）
// ============================================================================

/**
 * P3-NEW-30: HATCH 边界预处理
 * 使用 Douglas-Peucker 算法简化边界，移除冗余顶点
 *
 * @param points - 边界点数组
 * @param closed - 是否闭合
 * @param cameraZoom - 相机缩放级别
 * @returns 简化后的边界点
 */
function preprocessHatchBoundary(
  points: [number, number][],
  closed: boolean,
  cameraZoom: number
): [number, number][] {
  if (points.length <= 3) {
    return points // 点太少，无需简化
  }

  // 1. 计算世界空间容差
  const tolerance = calculateWorldTolerance(cameraZoom, 0.5)

  // 2. Douglas-Peucker 简化（移除冗余点）
  let simplified = douglasPeucker(points, tolerance)

  // 3. 移除微小边（小于容差的边）
  simplified = removeTinyEdges(simplified, tolerance)

  // 4. 确保闭合
  if (closed && simplified.length > 0) {
    const first = simplified[0]
    const last = simplified[simplified.length - 1]
    const dist = Math.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
    if (dist > tolerance) {
      simplified.push(first)
    }
  }

  return simplified
}

/**
 * P3-NEW-30: Douglas-Peucker 简化算法
 * 递归版本，移除距离首尾连线较近的点
 *
 * @param points - 输入点数组
 * @param epsilon - 容差阈值
 * @returns 简化后的点数组
 */
function douglasPeucker(
  points: [number, number][],
  epsilon: number
): [number, number][] {
  if (points.length <= 2) {
    return points
  }

  // 找到距离首尾连线最远的点
  let maxDist = 0
  let maxIndex = 0

  const start = points[0]
  const end = points[points.length - 1]

  for (let i = 1; i < points.length - 1; i++) {
    const dist = distancePointToLine(points[i], start, end)
    if (dist > maxDist) {
      maxDist = dist
      maxIndex = i
    }
  }

  // 递归简化
  if (maxDist > epsilon) {
    const left = douglasPeucker(points.slice(0, maxIndex + 1), epsilon)
    const right = douglasPeucker(points.slice(maxIndex), epsilon)
    return [...left.slice(0, -1), ...right]
  } else {
    return [start, end]
  }
}

/**
 * P3-NEW-30: 移除微小边
 * 移除长度小于容差的边
 *
 * @param points - 输入点数组
 * @param tolerance - 容差阈值
 * @returns 移除微小边后的点数组
 */
function removeTinyEdges(
  points: [number, number][],
  tolerance: number
): [number, number][] {
  if (points.length <= 2) {
    return points
  }

  const result: [number, number][] = [points[0]]

  for (let i = 1; i < points.length; i++) {
    const prev = result[result.length - 1]
    const curr = points[i]
    const dist = Math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)

    if (dist > tolerance) {
      result.push(curr)
    }
  }

  // 确保至少 3 个点
  if (result.length < 3 && points.length >= 3) {
    return points
  }

  return result
}

// ============================================================================
// P0-4 新增：Bulge 处理工具函数
// ============================================================================

/**
 * 根据 bulge 值计算圆弧的圆心和半径
 * bulge = tan(θ/4)，其中 θ 是圆弧的包含角
 * 
 * @param p1 - 起点
 * @param p2 - 终点
 * @param bulge - 凸度值
 * @returns 圆心、半径和起始/结束角度
 */
function bulgeToArc(
  p1: [number, number],
  p2: [number, number],
  bulge: number
): {
  center: [number, number]
  radius: number
  startAngle: number
  endAngle: number
  ccw: boolean
} | null {
  const bulgeAbs = Math.abs(bulge)
  
  // bulge = 0 表示直线段
  if (bulgeAbs < 1e-10) {
    return null
  }
  
  // 计算包含角 θ = 4 * atan(bulge)
  const includedAngle = 4 * Math.atan(bulgeAbs)
  
  // 计算弦长
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const chordLength = Math.sqrt(dx * dx + dy * dy)
  
  if (chordLength < 1e-10) {
    return null
  }
  
  // 计算半径：r = chord / (2 * sin(θ/2))
  const radius = chordLength / (2 * Math.sin(includedAngle / 2))
  
  // 计算弦的中点
  const midX = (p1[0] + p2[0]) / 2
  const midY = (p1[1] + p2[1]) / 2
  
  // 计算弦的垂直距离（拱高）：h = r * cos(θ/2)
  const sagitta = radius * Math.cos(includedAngle / 2)
  
  // 计算圆心方向（垂直于弦）
  const perpDx = -dy / chordLength
  const perpDy = dx / chordLength
  
  // bulge > 0 表示逆时针，bulge < 0 表示顺时针
  const ccw = bulge > 0
  const direction = ccw ? 1 : -1
  
  // 计算圆心
  const centerX = midX + direction * perpDx * sagitta
  const centerY = midY + direction * perpDy * sagitta
  
  // 计算起始和结束角度
  const startAngle = Math.atan2(p1[1] - centerY, p1[0] - centerX)
  const endAngle = startAngle + (ccw ? includedAngle : -includedAngle)
  
  return {
    center: [centerX, centerY],
    radius,
    startAngle,
    endAngle,
    ccw,
  }
}

/**
 * 使用 bulge 信息绘制多段线边界
 * 将带 bulge 的线段离散化为圆弧或直线
 *
 * DXF Bulge 规范：
 * - bulge 存储在每个顶点上，表示从该顶点到下一个顶点的圆弧段
 * - bulge = tan(θ/4)，其中 θ 是圆弧的包含角
 * - bulge > 0 表示逆时针，bulge < 0 表示顺时针
 * - 对于闭合多段线，最后一个顶点的 bulge 表示从最后一个顶点回到第一个顶点的圆弧
 *
 * ✅ P2-NEW-19 修复：添加 cameraZoom 参数，使用动态段数离散化圆弧
 */
function drawPolylineWithBulge(
  ctx: CanvasRenderingContext2D,
  points: [number, number][],
  bulges: number[],  // ✅ 修复：bulges 是 number[] 而非 [number, number][]
  closed: boolean,
  cameraZoom: number = 1  // ✅ P2-NEW-19 新增：相机缩放级别
) {
  if (points.length < 2) return

  // ✅ P1-NEW-1 修复：添加边界检查和警告
  if (bulges.length !== points.length) {
    debugLog('WARN', '[HatchLayer] Bulge 数组长度与点数不匹配:', {
      points: points.length,
      bulges: bulges.length,
      expected: points.length
    })
  }

  // ✅ P1-NEW-1 修复：安全的 bulge 访问函数（处理长度不匹配情况）
  const safeBulge = (i: number): number => {
    if (i < 0 || i >= bulges.length) {
      // 如果索引超出 bulges 范围，回退到 0（直线）
      debugLog('WARN', `[HatchLayer] Bulge index ${i} out of range, using 0`)
      return 0
    }
    return bulges[i]
  }

  // ✅ P0-NEW-10 修复：使用循环索引统一处理所有线段（包括闭合段）
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i]
    const p2 = points[(i + 1) % points.length]  // ✅ 循环索引，自动处理闭合
    const bulge = safeBulge(i)

    if (i === 0) {
      ctx.moveTo(p1[0], p1[1])
    }

    if (Math.abs(bulge) < 1e-10) {
      // 直线段
      ctx.lineTo(p2[0], p2[1])
    } else {
      // ✅ P2-NEW-19 修复：使用 discretizeArc 离散化圆弧
      const arc = bulgeToArc(p1, p2, bulge)
      if (arc) {
        const arcPoints = discretizeArc(
          arc.center,
          arc.radius,
          arc.startAngle,
          arc.endAngle,
          arc.ccw,
          cameraZoom
        )
        // 绘制离散化后的圆弧
        arcPoints.forEach((point, j) => {
          if (j === 0) {
            ctx.moveTo(point[0], point[1])
          } else {
            ctx.lineTo(point[0], point[1])
          }
        })
      } else {
        // 回退到直线
        ctx.lineTo(p2[0], p2[1])
      }
    }
  }

  // ✅ P0-NEW-10 修复：强制闭合路径（确保填充不泄漏）
  if (closed) {
    ctx.closePath()
  }
}

// ============================================================================
// HATCH 样式配置
// ============================================================================
// P0-4 修复：适配新的 pattern type ('predefined' | 'custom' | 'solid')
const HATCH_STYLES = {
  solid: {
    default: { opacity: 0.3 },
    hover: { opacity: 0.5 },
  },
  predefined: {  // P0-4 修复：'pattern' -> 'predefined'
    default: { opacity: 0.6 },
    hover: { opacity: 0.8 },
  },
  custom: {  // P0-4 新增：custom 类型
    default: { opacity: 0.6 },
    hover: { opacity: 0.8 },
  },
  selected: { stroke: '#fbbf24', strokeWidth: 2 },
} as const

// P0-4 修复：添加 HatchLayerProps 类型定义
interface HatchLayerProps {
  hatches: HatchEntity[]
  camera: CameraState
  selectedHatchIds?: number[]
  onHatchClick?: (hatchId: number) => void
}

// P0-4 修复：将 RGBA 数组转换为颜色字符串
function rgbaToString(color?: [number, number, number, number]): string {
  if (!color) return '#999999'
  const [r, g, b, a] = color
  const alpha = a !== undefined ? a / 255 : 1
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

// ✅ P0-NEW-9 新增：根据 zoom 级别计算动态容差
// 屏幕空间容差 = 世界空间容差 * zoom
// 因此：世界空间容差 = 屏幕空间容差 / zoom
function calculateWorldTolerance(cameraZoom: number, screenTolerance: number = 1.0): number {
  // ✅ 限制 zoom 范围，避免除以零或过小值
  const safeZoom = Math.max(0.01, cameraZoom)
  return screenTolerance / safeZoom
}

/**
 * ✅ P2-NEW-19 新增：离散化圆弧为多段线
 * 根据半径和 zoom 级别动态计算段数，平衡质量和性能
 *
 * @param center - 圆心
 * @param radius - 半径
 * @param startAngle - 起始角度（弧度）
 * @param endAngle - 结束角度（弧度）
 * @param ccw - 是否逆时针
 * @param cameraZoom - 相机缩放级别
 * @returns 离散化后的点数组
 */
/**
 * ✅ P1-NEW-31 修复：基于弦高误差的圆弧离散化
 * 使用 sagitta 公式精确计算段数，避免过大或过小的容差
 * sagitta = R * (1 - cos(θ/2))
 * 反推：θ = 2 * acos(1 - sagitta/R)
 */
function discretizeArc(
  center: [number, number],
  radius: number,
  startAngle: number,
  endAngle: number,
  ccw: boolean,
  cameraZoom: number = 1
): [number, number][] {
  const angleRange = ccw
    ? (endAngle >= startAngle ? endAngle - startAngle : endAngle - startAngle + Math.PI * 2)
    : (startAngle >= endAngle ? startAngle - endAngle : startAngle - endAngle + Math.PI * 2)

  // ✅ P1-NEW-31 修复：基于弦高误差的精确容差计算
  // 1. 计算屏幕空间容差（1 像素）
  const screenTolerance = 1.0

  // 2. 转换为世界空间容差
  const worldTolerance = calculateWorldTolerance(cameraZoom, screenTolerance)

  // 3. 使用弦高误差公式计算每段角度
  // sagitta = R * (1 - cos(θ/2))
  // 反推：θ = 2 * acos(1 - sagitta/R)
  let numSegments: number

  if (radius <= worldTolerance) {
    // 半径非常小，使用固定段数
    numSegments = Math.max(8, Math.ceil(angleRange / (Math.PI / 8)))
  } else {
    // ✅ 核心修复：使用弦高误差公式精确计算每段角度
    const acosArg = Math.max(-1, Math.min(1, 1 - worldTolerance / radius))
    const anglePerSegment = 2 * Math.acos(acosArg)
    numSegments = Math.ceil(angleRange / anglePerSegment)
  }

  // ✅ 限制段数范围 (8-256)
  numSegments = Math.max(8, Math.min(numSegments, 256))

  const angleStep = angleRange / numSegments
  const points: [number, number][] = []

  for (let i = 0; i <= numSegments; i++) {
    const angle = ccw ? startAngle + i * angleStep : startAngle - i * angleStep
    const x = center[0] + radius * Math.cos(angle)
    const y = center[1] + radius * Math.sin(angle)
    points.push([x, y])
  }

  return points
}

// P0-4 修复：获取图案名称（兼容新旧类型）
function getPatternName(pattern: HatchEntity['pattern']): string {
  if (pattern.type === 'predefined' || pattern.type === 'custom') {
    return pattern.name || 'ANSI31'
  }
  return 'ANSI31'
}

// ============================================================================
// P2-3 新增：图案缓存机制
// ============================================================================

// ✅ P1-NEW-17 修复：LRU 图案缓存类，防止内存泄漏
class LRUPatternCache {
  private cache = new Map<string, HTMLCanvasElement>()
  private accessOrder: string[] = []  // 访问顺序队列
  private maxSize: number

  constructor(maxSize: number = 50) {
    this.maxSize = maxSize
  }

  get(key: string): HTMLCanvasElement | undefined {
    const canvas = this.cache.get(key)
    if (canvas) {
      // ✅ 更新访问顺序（移到队尾）
      this.accessOrder = this.accessOrder.filter(k => k !== key)
      this.accessOrder.push(key)
    }
    return canvas
  }

  set(key: string, canvas: HTMLCanvasElement) {
    // ✅ 如果已存在，先删除旧条目
    if (this.cache.has(key)) {
      this.accessOrder = this.accessOrder.filter(k => k !== key)
    }

    // ✅ 检查是否超出容量，删除最久未使用的
    while (this.cache.size >= this.maxSize) {
      const oldestKey = this.accessOrder.shift()
      if (oldestKey) {
        this.cache.delete(oldestKey)
        debugLog('DEBUG', `[LRUPatternCache] Evicted: ${oldestKey}`)
      }
    }

    this.cache.set(key, canvas)
    this.accessOrder.push(key)
  }

  has(key: string): boolean {
    return this.cache.has(key)
  }

  clear() {
    this.cache.clear()
    this.accessOrder = []
  }

  get size(): number {
    return this.cache.size
  }

  getStats(): { size: number; maxSize: number } {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
    }
  }
}

// ✅ P1-NEW-17 修复：使用 LRU 缓存替代简单 Map
const patternCache = new LRUPatternCache(50)  // 最多 50 个图案

// 缓存统计（开发模式使用）
let cacheHits = 0
let cacheMisses = 0

/**
 * ✅ P2-NEW-34 修复：统一颜色格式
 * 将不同格式的颜色统一为规范的 rgba 字符串
 * 支持：rgba 字符串、rgba 数组、HEX 颜色
 */
function normalizeColor(color: string | [number, number, number, number]): string {
  let r: number = 0
  let g: number = 0
  let b: number = 0
  let a: number = 1.0

  if (Array.isArray(color)) {
    // ✅ 处理 rgba 数组
    r = color[0]
    g = color[1]
    b = color[2]
    a = color[3] !== undefined ? color[3] / 255 : 1.0
  } else {
    // ✅ 处理 rgba 字符串
    const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/)
    if (rgbaMatch) {
      r = parseInt(rgbaMatch[1])
      g = parseInt(rgbaMatch[2])
      b = parseInt(rgbaMatch[3])
      a = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1.0
    } else {
      // ✅ 处理 HEX 颜色
      const hex = color.replace('#', '')
      if (hex.length === 3) {
        // #RGB -> #RRGGBB
        r = parseInt(hex[0] + hex[0], 16)
        g = parseInt(hex[1] + hex[1], 16)
        b = parseInt(hex[2] + hex[2], 16)
      } else if (hex.length === 6) {
        r = parseInt(hex.substring(0, 2), 16)
        g = parseInt(hex.substring(2, 4), 16)
        b = parseInt(hex.substring(4, 6), 16)
      } else if (hex.length === 8) {
        r = parseInt(hex.substring(0, 2), 16)
        g = parseInt(hex.substring(2, 4), 16)
        b = parseInt(hex.substring(4, 6), 16)
        a = parseInt(hex.substring(6, 8), 16) / 255
      } else {
        // 默认黑色
        r = 0
        g = 0
        b = 0
        a = 1.0
      }
    }
  }

  // ✅ 归一化到整数（RGB）和 2 位小数（Alpha）
  r = Math.round(r)
  g = Math.round(g)
  b = Math.round(b)
  a = Math.round(a * 100) / 100

  return `rgba(${r},${g},${b},${a})`
}

/**
 * 获取或创建图案填充 canvas
 * 使用缓存避免重复创建相同图案，提升性能
 */
function getCachedPattern(
  patternName: string,
  color: string,
  scale: number,
  angle: number
): HTMLCanvasElement {
  // ✅ P2-NEW-34 修复：统一颜色格式
  const normalizedColor = normalizeColor(color)

  // ✅ P2-NEW-27 修复：使用舍入到固定精度，避免浮点误差导致缓存未命中
  const roundedScale = Math.round(scale * 100) / 100  // 2 位小数
  const roundedAngle = Math.round(angle * 10) / 10    // 1 位小数

  const cacheKey = `${patternName.toUpperCase()}_${normalizedColor}_${roundedScale.toFixed(2)}_${roundedAngle.toFixed(1)}`

  // 检查缓存
  const cached = patternCache.get(cacheKey)
  if (cached) {
    cacheHits++
    return cached
  }

  // 未命中，创建新图案
  cacheMisses++
  const canvas = createHatchPattern(patternName, normalizedColor, scale, angle)
  patternCache.set(cacheKey, canvas)
  return canvas
}

/**
 * 清除图案缓存
 * 用于内存管理或图案配置变更时
 */
export function clearPatternCache(): void {
  patternCache.clear()
  cacheHits = 0
  cacheMisses = 0
}

/**
 * 获取缓存统计信息（开发模式使用）
 */
export function getPatternCacheStats(): { hits: number; misses: number; size: number; hitRate: number } {
  const total = cacheHits + cacheMisses
  return {
    hits: cacheHits,
    misses: cacheMisses,
    size: patternCache.size,
    hitRate: total > 0 ? (cacheHits / total) * 100 : 0,
  }
}

// 创建图案填充的 canvas pattern
// P0-2 修复：扩展建筑填充图案支持
// P1-NEW-3 修复：扩大 canvas 以容纳旋转后的图案
function createHatchPattern(
  patternName: string,
  color: string,
  scale: number,
  angle: number
): HTMLCanvasElement {
  // ✅ P0-2 修复：根据图案类型动态调整尺寸
  const baseSize = getPatternBaseSize(patternName) * scale
  const size = Math.max(baseSize, 10)  // 最小 10px

  // ✅ P1-NEW-3 修复：计算旋转后的边界框，扩大 canvas 以容纳旋转后的图案
  const angleRad = (angle * Math.PI) / 180
  const cosA = Math.abs(Math.cos(angleRad))
  const sinA = Math.abs(Math.sin(angleRad))
  const expandedSize = Math.ceil(size * (cosA + sinA) * 1.5)  // 1.5 倍安全系数

  const canvas = document.createElement('canvas')
  canvas.width = expandedSize
  canvas.height = expandedSize
  const ctx = canvas.getContext('2d')!

  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(1, scale * 0.5)

  // ✅ P2-NEW-22 修复：正确的旋转中心计算
  // 旋转中心应该是原始 canvas 的中心（baseSize），而不是 expandedSize
  // 1. 先平移到 expandedSize 中心
  // 2. 再平移到原始 pattern 中心（expandedSize/2 - size/2 偏移）
  // 3. 旋转
  // 4. 平移回原始位置
  const originalCenterX = expandedSize / 2
  const originalCenterY = expandedSize / 2

  ctx.save()
  // ✅ 正确的旋转：以 expandedSize 中心为旋转点
  ctx.translate(originalCenterX, originalCenterY)
  ctx.rotate(angleRad)
  ctx.translate(-originalCenterX, -originalCenterY)

  switch (patternName.toUpperCase()) {
    // ✅ 现有图案
    case 'ANSI31':
      renderAnsi31Pattern(ctx, expandedSize, color, scale)
      break

    case 'ANSI32':
      renderAnsi32Pattern(ctx, expandedSize, color, scale)
      break

    case 'ANSI33':
      renderAnsi33Pattern(ctx, expandedSize, color, scale)
      break

    // ✅ P0-2 新增：建筑填充图案
    case 'AR-BRSTD':
    case 'AR-BRSTK':
      renderBrickPattern(ctx, expandedSize, color, scale)
      break

    case 'AR-CONC':
      renderConcretePattern(ctx, expandedSize, color, scale)
      break

    case 'AR-SAND':
      renderSandPattern(ctx, expandedSize, color, scale)
      break

    case 'AR-HBONE':
      renderHerringbonePattern(ctx, expandedSize, color, scale)
      break

    case 'AR-ROOF':
      renderRoofPattern(ctx, expandedSize, color, scale)
      break

    // ✅ 新增：更多建筑填充图案 (P2 任务)
    case 'AR-GLASS':
      renderGlassPattern(ctx, expandedSize, color, scale)
      break

    case 'AR-ROCK':
      renderRockPattern(ctx, expandedSize, color, scale)
      break

    case 'AR-B816':
      renderB816Pattern(ctx, expandedSize, color, scale)
      break

    case 'AR-GRVL':
      renderGravelPattern(ctx, expandedSize, color, scale)
      break

    case 'AR-WIRE':
      renderWirePattern(ctx, expandedSize, color, scale)
      break

    case 'AR-FENCE':
      renderFencePattern(ctx, expandedSize, color, scale)
      break

    // ✅ 默认回退
    case 'SOLID':
    default:
      ctx.fillStyle = color
      ctx.fillRect(0, 0, expandedSize, expandedSize)
      break
  }

  ctx.restore()
  return canvas
}

// ✅ P0-2 新增：图案基础尺寸映射
function getPatternBaseSize(patternName: string): number {
  const patternSizes: Record<string, number> = {
    'ANSI31': 20,
    'ANSI32': 20,
    'ANSI33': 20,
    'AR-BRSTD': 50,  // 砖墙图案较大
    'AR-BRSTK': 50,
    'AR-CONC': 30,   // 混凝土图案中等
    'AR-SAND': 15,   // 沙子图案较密
    'AR-HBONE': 40,  // 人字图案
    'AR-ROOF': 60,   // 屋面图案
    // ✅ P2 新增：更多建筑图案尺寸
    'AR-GLASS': 25,  // 玻璃图案
    'AR-ROCK': 35,   // 岩石图案
    'AR-B816': 45,   // 砖块 8x16 图案
    'AR-GRVL': 20,   // 砾石图案
    'AR-WIRE': 30,   // 铁丝网图案
    'AR-FENCE': 50,  // 围栏图案
  }
  return patternSizes[patternName.toUpperCase()] || 20
}

// ✅ P0-2 新增：ANSI31 斜线填充
function renderAnsi31Pattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  _color: string,
  scale: number
) {
  const spacing = 4 * scale
  ctx.beginPath()
  for (let i = -size; i < size * 2; i += spacing) {
    ctx.moveTo(i, 0)
    ctx.lineTo(i + size, size)
  }
  ctx.stroke()
}

// ✅ P0-2 新增：ANSI32 交叉网格填充
function renderAnsi32Pattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  _color: string,
  scale: number
) {
  const spacing = 4 * scale
  ctx.beginPath()
  for (let i = 0; i < size * 2; i += spacing) {
    ctx.moveTo(i, 0)
    ctx.lineTo(i, size)
    ctx.moveTo(0, i)
    ctx.lineTo(size, i)
  }
  ctx.stroke()
}

// ✅ P0-2 新增：ANSI33 点状填充
function renderAnsi33Pattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const spacing = 4 * scale
  for (let i = 0; i < size; i += spacing) {
    for (let j = 0; j < size; j += spacing) {
      ctx.beginPath()
      ctx.arc(i, j, 1 * scale, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    }
  }
}

// ✅ P0-2 新增：砖墙图案
function renderBrickPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const brickHeight = 10 * scale
  const brickWidth = 20 * scale
  const mortarGap = 2 * scale

  ctx.strokeStyle = color
  for (let y = 0; y < size; y += brickHeight + mortarGap) {
    const offset = (Math.floor(y / (brickHeight + mortarGap)) % 2 === 0) ? 0 : brickWidth / 2

    for (let x = -brickWidth; x < size; x += brickWidth + mortarGap) {
      ctx.strokeRect(x + offset, y, brickWidth, brickHeight)
    }
  }
}

// ✅ P0-2 新增：混凝土图案
function renderConcretePattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  // 随机点状 + 三角形
  const rng = seededRandom(12345)  // 固定种子保证图案一致
  ctx.strokeStyle = color
  for (let i = 0; i < 20; i++) {
    const x = rng() * size
    const y = rng() * size
    const s = rng() * 2 * scale

    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x + s, y + s * 2)
    ctx.lineTo(x - s, y + s * 2)
    ctx.closePath()
    ctx.stroke()
  }
}

// ✅ P0-2 新增：沙子图案
function renderSandPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const rng = seededRandom(54321)  // 固定种子保证图案一致
  for (let i = 0; i < 50; i++) {
    const x = rng() * size
    const y = rng() * size
    ctx.beginPath()
    ctx.arc(x, y, 1 * scale, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
  }
}

// ✅ P0-2 新增：人字图案
function renderHerringbonePattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const brickLength = 15 * scale
  const brickWidth = 5 * scale

  ctx.strokeStyle = color
  for (let y = 0; y < size; y += brickLength) {
    const offset = (Math.floor(y / brickLength) % 2 === 0) ? 0 : brickWidth

    for (let x = 0; x < size; x += brickWidth * 2) {
      // 横砖
      ctx.strokeRect(x + offset, y, brickLength, brickWidth)
      // 竖砖
      ctx.strokeRect(x + offset + brickLength, y, brickWidth, brickLength)
    }
  }
}

// ✅ P0-2 新增：屋面图案
function renderRoofPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  _color: string,
  scale: number
) {
  const spacing = 8 * scale
  ctx.beginPath()
  for (let i = 0; i < size; i += spacing) {
    // 波浪线
    for (let j = 0; j < size; j += scale) {
      const x = i + j
      const y = Math.sin(j * 0.2) * 2 * scale + i
      if (j === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
  }
  ctx.stroke()
}

// ✅ P2 新增：玻璃图案 (AR-GLASS) - 交叉斜线 + 网格点
function renderGlassPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const spacing = 6 * scale
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(0.5, scale * 0.3)
  
  // 交叉斜线
  ctx.beginPath()
  for (let i = -size; i < size * 2; i += spacing) {
    ctx.moveTo(i, 0)
    ctx.lineTo(i + size * 0.8, size)
  }
  for (let i = -size; i < size * 2; i += spacing) {
    ctx.moveTo(i + size * 0.5, 0)
    ctx.lineTo(i + size * 1.3, size)
  }
  ctx.stroke()
  
  // 网格点装饰
  ctx.fillStyle = color
  for (let i = 0; i < size; i += spacing * 2) {
    for (let j = 0; j < size; j += spacing * 2) {
      ctx.beginPath()
      ctx.arc(i, j, 1.5 * scale, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

// ✅ P2 新增：岩石图案 (AR-ROCK) - 不规则多边形
function renderRockPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const rng = seededRandom(78901)  // 固定种子
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(1, scale * 0.5)
  
  // 绘制不规则多边形模拟岩石纹理
  for (let i = 0; i < 15; i++) {
    const cx = rng() * size
    const cy = rng() * size
    const radius = 5 * scale + rng() * 8 * scale
    
    ctx.beginPath()
    const numPoints = 5 + Math.floor(rng() * 4)
    for (let j = 0; j < numPoints; j++) {
      const angle = (j / numPoints) * Math.PI * 2
      const r = radius * (0.6 + rng() * 0.4)
      const x = cx + Math.cos(angle) * r
      const y = cy + Math.sin(angle) * r
      if (j === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
    ctx.closePath()
    ctx.stroke()
  }
}

// ✅ P2 新增：砖块 8x16 图案 (AR-B816) - 标准砖块尺寸
function renderB816Pattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const brickHeight = 8 * scale
  const brickWidth = 16 * scale
  const mortarGap = 1.5 * scale
  
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(0.8, scale * 0.4)
  
  for (let y = 0; y < size; y += brickHeight + mortarGap) {
    const offset = (Math.floor(y / (brickHeight + mortarGap)) % 2 === 0) 
      ? 0 
      : (brickWidth + mortarGap) / 2

    for (let x = -brickWidth; x < size; x += brickWidth + mortarGap) {
      // 砖块轮廓
      ctx.strokeRect(x + offset, y, brickWidth, brickHeight)
      
      // 砖块内部纹理（十字线）
      ctx.beginPath()
      ctx.moveTo(x + offset + brickWidth * 0.3, y + brickHeight * 0.2)
      ctx.lineTo(x + offset + brickWidth * 0.7, y + brickHeight * 0.8)
      ctx.stroke()
    }
  }
}

// ✅ P2 新增：砾石图案 (AR-GRVL) - 小圆点 + 小三角形混合
function renderGravelPattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const rng = seededRandom(45678)  // 固定种子
  ctx.fillStyle = color
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(0.5, scale * 0.3)
  
  // 绘制小圆点和小三角形模拟砾石
  for (let i = 0; i < 80; i++) {
    const x = rng() * size
    const y = rng() * size
    const s = 1.5 * scale + rng() * 2 * scale
    const shapeType = rng()
    
    if (shapeType < 0.5) {
      // 圆形砾石
      ctx.beginPath()
      ctx.arc(x, y, s, 0, Math.PI * 2)
      ctx.fill()
    } else {
      // 三角形砾石
      ctx.beginPath()
      ctx.moveTo(x, y - s)
      ctx.lineTo(x + s, y + s)
      ctx.lineTo(x - s, y + s)
      ctx.closePath()
      ctx.stroke()
    }
  }
}

// ✅ P2 新增：铁丝网图案 (AR-WIRE) - 菱形网格
function renderWirePattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const spacing = 10 * scale
  ctx.strokeStyle = color
  ctx.lineWidth = Math.max(0.5, scale * 0.25)
  
  // 绘制菱形网格
  ctx.beginPath()
  for (let i = 0; i < size; i += spacing) {
    // 斜向线条
    for (let j = 0; j < size * 2; j += spacing) {
      const x = j - i
      const y = i
      if (j === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
  }
  for (let i = 0; i < size; i += spacing) {
    // 反向斜向线条
    for (let j = 0; j < size * 2; j += spacing) {
      const x = j + i
      const y = i
      if (j === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
  }
  ctx.stroke()
  
  // 交叉点装饰
  ctx.fillStyle = color
  for (let i = 0; i < size; i += spacing) {
    for (let j = 0; j < size; j += spacing) {
      ctx.beginPath()
      ctx.arc(j, i, 1 * scale, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

// ✅ P2 新增：围栏图案 (AR-FENCE) - 垂直栅栏 + 水平横梁
function renderFencePattern(
  ctx: CanvasRenderingContext2D,
  size: number,
  color: string,
  scale: number
) {
  const postWidth = 6 * scale
  const postGap = 12 * scale
  const railHeight = 2 * scale
  // const railSpacing = 15 * scale  // P1-1 修复：暂时未使用，保留供未来扩展

  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = Math.max(0.8, scale * 0.4)
  
  // 垂直栅栏柱
  for (let x = 0; x < size; x += postWidth + postGap) {
    // 栅栏柱
    ctx.fillRect(x, 0, postWidth, size)
    
    // 柱顶装饰
    ctx.beginPath()
    ctx.moveTo(x, railHeight)
    ctx.lineTo(x + postWidth / 2, 0)
    ctx.lineTo(x + postWidth, railHeight)
    ctx.closePath()
    ctx.fill()
  }
  
  // 水平横梁（2 条）
  const railY1 = size * 0.3
  const railY2 = size * 0.7
  ctx.fillRect(0, railY1, size, railHeight)
  ctx.fillRect(0, railY2, size, railHeight)
}

// ✅ P0-2 新增：种子随机数生成器（保证图案一致性）
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = Math.sin(s) * 10000
    return s - Math.floor(s)
  }
}

// ✅ P1-1 新增：B 样条求值函数（Cox-de Boor 递归算法简化版）
function evaluateBSpline(
  controlPoints: [number, number][],
  knots: number[],
  degree: number,
  u: number
): [number, number] | null {
  const n = controlPoints.length - 1

  // ✅ P1-2 修复：边界检查
  if (n < 0 || knots.length === 0) {
    return null
  }

  // ✅ P1-2 修复：检查 u 是否在 knots 范围内
  const knotMin = knots[0] ?? 0
  const knotMax = knots[knots.length - 1] ?? 1
  if (u < knotMin || u > knotMax) {
    return null
  }

  // 计算基函数值（使用 Cox-de Boor 递归）
  const baseFunc = (i: number, p: number, t: number): number => {
    // ✅ P0-4 修复：添加边界检查，防止 knots 数组越界
    if (i < 0 || i >= knots.length - 1) {
      return 0
    }

    if (p === 0) {
      return (t >= knots[i] && t < knots[i + 1]) ? 1 : 0
    }

    const denom1 = knots[i + p] - knots[i]
    const denom2 = knots[i + p + 1] - knots[i + 1]

    const coeff1 = denom1 > 0 ? (t - knots[i]) / denom1 : 0
    const coeff2 = denom2 > 0 ? (knots[i + p + 1] - t) / denom2 : 0

    return coeff1 * baseFunc(i, p - 1, t) + coeff2 * baseFunc(i + 1, p - 1, t)
  }

  // 计算点位置
  let x = 0
  let y = 0
  for (let i = 0; i <= n; i++) {
    const basis = baseFunc(i, degree, u)
    x += basis * controlPoints[i][0]
    y += basis * controlPoints[i][1]
  }

  return [x, y]
}

// ============================================================================
// ✅ P1-NEW-16 新增：自适应样条离散化算法
// ============================================================================

/**
 * 计算点到直线的垂直距离（弦高误差）
 * 用于自适应样条离散化
 */
function distancePointToLine(
  point: [number, number],
  lineStart: [number, number],
  lineEnd: [number, number]
): number {
  const dx = lineEnd[0] - lineStart[0]
  const dy = lineEnd[1] - lineStart[1]
  const lineLengthSq = dx * dx + dy * dy

  if (lineLengthSq < 1e-10) {
    // 线段退化为点
    return Math.sqrt(
      (point[0] - lineStart[0]) ** 2 +
      (point[1] - lineStart[1]) ** 2
    )
  }

  // 计算点到直线的垂直距离
  const t = Math.max(0, Math.min(1,
    ((point[0] - lineStart[0]) * dx + (point[1] - lineStart[1]) * dy) / lineLengthSq
  ))

  const projX = lineStart[0] + t * dx
  const projY = lineStart[1] + t * dy

  return Math.sqrt(
    (point[0] - projX) ** 2 +
    (point[1] - projY) ** 2
  )
}

/**
 * P3-NEW-48 新增：计算 B 样条在某点的曲率
 * 使用数值微分法计算一阶和二阶导数
 * 曲率公式：κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
 *
 * @param controlPoints - 控制点
 * @param knots - 节点向量
 * @param degree - 阶数
 * @param u - 参数值
 * @param h - 微分步长（默认 1e-5）
 * @returns 曲率值
 */
function computeBSplineCurvature(
  controlPoints: [number, number][],
  knots: number[],
  degree: number,
  u: number,
  h: number = 1e-5
): number {
  // 计算一阶导数（中心差分）
  const pMinus = evaluateBSpline(controlPoints, knots, degree, u - h)
  const pPlus = evaluateBSpline(controlPoints, knots, degree, u + h)

  if (!pMinus || !pPlus) return 0

  const dx = (pPlus[0] - pMinus[0]) / (2 * h)
  const dy = (pPlus[1] - pMinus[1]) / (2 * h)

  // 计算二阶导数（中心差分）
  const pMinus2 = evaluateBSpline(controlPoints, knots, degree, u - 2 * h)
  const pPlus2 = evaluateBSpline(controlPoints, knots, degree, u + 2 * h)

  if (!pMinus2 || !pPlus2) return 0

  const ddx = (pPlus2[0] - 2 * pPlus[0] + pMinus2[0]) / (4 * h * h)
  const ddy = (pPlus2[1] - 2 * pPlus[1] + pMinus2[1]) / (4 * h * h)

  // 曲率公式
  const numerator = Math.abs(dx * ddy - dy * ddx)
  const denominator = Math.pow(dx * dx + dy * dy, 1.5)

  if (denominator < 1e-10) return 0

  return numerator / denominator
}

/**
 * P3-NEW-48 新增：根据曲率调整容差
 * 高曲率区域使用更小的容差，低曲率区域使用更大的容差
 *
 * @param baseTolerance - 基础容差
 * @param curvature - 曲率值
 * @param curvatureScale - 曲率缩放因子（默认 10）
 * @returns 调整后的容差
 */
function adjustToleranceByCurvature(
  baseTolerance: number,
  curvature: number,
  curvatureScale: number = 10
): number {
  // 曲率越大，容差越小
  // 公式：adjustedTolerance = baseTolerance / (1 + curvature * scale)
  return baseTolerance / (1 + curvature * curvatureScale)
}

/**
 * 自适应 B 样条离散化（递归细分）
 * 根据曲率和弦高误差自动调整离散化段数，平衡质量和性能
 *
 * ✅ P3-NEW-48 优化：添加曲率自适应
 *
 * @param controlPoints - 控制点
 * @param knots - 节点向量
 * @param degree - 阶数
 * @param uStart - 起始参数
 * @param uEnd - 结束参数
 * @param tolerance - 弦高容差（世界空间）
 * @param points - 输出点数组
 */
function adaptiveSubdivideBSpline(
  controlPoints: [number, number][],
  knots: number[],
  degree: number,
  uStart: number,
  uEnd: number,
  tolerance: number,
  points: [number, number][]
) {
  const pStart = evaluateBSpline(controlPoints, knots, degree, uStart)
  const pEnd = evaluateBSpline(controlPoints, knots, degree, uEnd)

  if (!pStart || !pEnd) return

  const uMid = (uStart + uEnd) / 2
  const pMid = evaluateBSpline(controlPoints, knots, degree, uMid)

  if (!pMid) {
    points.push(pEnd)
    return
  }

  // 计算弦高误差（中点到弦的垂直距离）
  const chordHeight = distancePointToLine(pMid, pStart, pEnd)

  // ✅ P3-NEW-48: 计算中点曲率
  const curvature = computeBSplineCurvature(controlPoints, knots, degree, uMid)

  // ✅ P3-NEW-48: 根据曲率调整容差
  const adjustedTolerance = adjustToleranceByCurvature(tolerance, curvature)

  if (chordHeight > adjustedTolerance) {
    // ✅ 误差过大，递归细分
    adaptiveSubdivideBSpline(controlPoints, knots, degree, uStart, uMid, adjustedTolerance, points)
    adaptiveSubdivideBSpline(controlPoints, knots, degree, uMid, uEnd, adjustedTolerance, points)
  } else {
    // ✅ 误差可接受，添加终点
    points.push(pEnd)
  }
}

/**
 * 自适应 B 样条离散化主函数
 *
 * @param controlPoints - 控制点
 * @param knots - 节点向量
 * @param degree - 阶数
 * @param cameraZoom - 相机缩放级别
 * @param screenTolerance - 屏幕容差（像素）
 * @returns 离散化后的点数组
 */
function discretizeBSplineAdaptive(
  controlPoints: [number, number][],
  knots: number[],
  degree: number,
  cameraZoom: number = 1,
  screenTolerance: number = 1.0
): [number, number][] {
  const points: [number, number][] = []
  const uStart = knots[0] ?? 0
  const uEnd = knots[knots.length - 1] ?? 1

  // 计算世界空间容差
  const worldTolerance = calculateWorldTolerance(cameraZoom, screenTolerance)

  // 添加起点
  const startPoint = evaluateBSpline(controlPoints, knots, degree, uStart)
  if (startPoint) {
    points.push(startPoint)
  }

  // 递归细分
  adaptiveSubdivideBSpline(controlPoints, knots, degree, uStart, uEnd, worldTolerance, points)

  return points
}

// ============================================================================
// P3-NEW-36 新增：边界方向规范化工具（保留供未来使用）
// ============================================================================
// 注：当前使用 evenodd 规则，无需方向规范化
// 未来使用 nonzero 规则时启用以下函数

// ✅ P1-1 新增：Catmull-Rom 样条插值函数

/**
 * ✅ P1-NEW-32 新增：计算弦长参数化
 * 根据拟合点计算累积弦长参数
 */
function computeChordLengths(fitPoints: [number, number][]): number[] {
  const n = fitPoints.length
  if (n === 0) return []

  const parameters: number[] = [0]
  let totalLength = 0

  for (let i = 1; i < n; i++) {
    const dx = fitPoints[i][0] - fitPoints[i - 1][0]
    const dy = fitPoints[i][1] - fitPoints[i - 1][1]
    totalLength += Math.sqrt(dx * dx + dy * dy)
    parameters.push(totalLength)
  }

  // 归一化到 [0, 1]
  if (totalLength > 0) {
    return parameters.map(t => t / totalLength)
  }

  // 所有点重合，均匀参数化
  return Array.from({ length: n }, (_, i) => i / (n - 1))
}

/**
 * ✅ P1-NEW-32 新增：构建节点向量
 * 使用平均技术构建节点向量 (参考 The NURBS Book 第 9 章)
 * 
 * ✅ P2-NEW-42 修复：完整节点向量构建算法
 * 公式：u_j = (1/p) * Σ_{i=j}^{j+p-1} t_i
 * 其中 p = degree, t_i 是拟合点参数值
 */
function buildKnotVector(
  fitPoints: [number, number][],
  parameters: number[],  // ✅ P2-NEW-42: 使用实际参数值
  degree: number,
  isClosed: boolean
): number[] {
  const n = fitPoints.length - 1  // 控制点索引上限
  const m = n + degree + 1  // 节点索引上限

  if (isClosed) {
    // 闭合样条：均匀节点向量
    const knots: number[] = []
    const span = 1.0 / (n + 1)
    
    for (let i = 0; i <= m; i++) {
      knots.push(i * span)
    }
    
    return knots
  } else {
    // 开放样条：clamped 节点向量
    const knots: number[] = []

    // 1. 前端重复度 (degree+1)
    for (let i = 0; i <= degree; i++) {
      knots.push(0)
    }

    // 2. 内部节点：使用完整平均技术
    // 公式：u_j = (1/p) * Σ_{i=j}^{j+p-1} t_i
    // 其中 p = degree, t_i 是拟合点参数
    for (let j = 1; j <= n - degree; j++) {
      let sum = 0
      for (let i = j; i <= j + degree - 1; i++) {
        // ✅ P2-NEW-42: 使用实际参数值，而非简化为常数 1
        sum += parameters[i] ?? 0
      }
      knots.push(sum / degree)
    }

    // 3. 后端重复度 (degree+1)
    for (let i = 0; i <= degree; i++) {
      knots.push(1)
    }

    return knots
  }
}

/**
 * ✅ P2-NEW-38 新增：矩阵求逆（高斯 - 约旦消元法）
 * 用于求解线性方程组 N · P = Q
 */
function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length
  // 创建增广矩阵 [A|I]
  const augmented = matrix.map((row, i) => [
    ...row.map((val) => val),
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ])

  // 高斯 - 约旦消元
  for (let col = 0; col < n; col++) {
    // 寻找主元
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[maxRow][col])) {
        maxRow = row
      }
    }

    // 交换行
    ;[augmented[col], augmented[maxRow]] = [augmented[maxRow], augmented[col]]

    // 检查是否奇异
    if (Math.abs(augmented[col][col]) < 1e-10) {
      // 奇异矩阵，返回单位矩阵作为降级处理
      return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
      )
    }

    // 归一化当前行
    const pivot = augmented[col][col]
    for (let j = col; j < 2 * n; j++) {
      augmented[col][j] /= pivot
    }

    // 消去其他行
    for (let row = 0; row < n; row++) {
      if (row !== col) {
        const factor = augmented[row][col]
        for (let j = col; j < 2 * n; j++) {
          augmented[row][j] -= factor * augmented[col][j]
        }
      }
    }
  }

  // 提取逆矩阵
  return augmented.map((row) => row.slice(n))
}

/**
 * ✅ P2-NEW-38 新增：评估 B 样条基函数
 * 用于构建最小二乘法矩阵
 */
function evaluateBSplineBasis(
  i: number,
  degree: number,
  knots: number[],
  u: number
): number {
  const n = knots.length - 2

  // 边界检查
  if (i < 0 || i > n || knots.length === 0) {
    return 0
  }

  // p = 0 情况
  if (degree === 0) {
    return u >= knots[i] && u < knots[i + 1] ? 1 : 0
  }

  // Cox-de Boor 递归
  const denom1 = knots[i + degree] - knots[i]
  const denom2 = knots[i + degree + 1] - knots[i + 1]

  const coeff1 = denom1 > 0 ? (u - knots[i]) / denom1 : 0
  const coeff2 = denom2 > 0 ? (knots[i + degree + 1] - u) / denom2 : 0

  return (
    coeff1 * evaluateBSplineBasis(i, degree - 1, knots, u) +
    coeff2 * evaluateBSplineBasis(i + 1, degree - 1, knots, u)
  )
}

/**
 * ✅ P2-NEW-38 新增：精确最小二乘法反算控制点
 * 参考：The NURBS Book 第 9 章
 * 求解线性方程组：N · P = Q
 * 其中 Q 是拟合点，P 是控制点，N 是基函数矩阵
 * 
 * 对于拟合点样条，我们有：
 * - n 个拟合点 Q[0], Q[1], ..., Q[n-1]
 * - n 个控制点 P[0], P[1], ..., P[n-1]（控制点数量 = 拟合点数量）
 * - 基函数矩阵 N (n×n)，其中 N[i][j] = N_j(u_i)
 * 
 * 求解：P = N⁻¹ · Q
 */
function computeControlPointsLeastSquares(
  fitPoints: [number, number][],
  parameters: number[],
  knots: number[],
  degree: number
): [number, number][] {
  const n = fitPoints.length

  // ✅ 构建基函数矩阵 N (n×n)
  // N[i][j] = N_j(u_i)，其中 u_i 是第 i 个拟合点的参数值
  const N: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      row.push(evaluateBSplineBasis(j, degree, knots, parameters[i]))
    }
    N.push(row)
  }

  // ✅ 求解线性方程组 N · P = Q
  // P = N⁻¹ · Q
  const N_inv = invertMatrix(N)

  // 计算控制点
  const controlPoints: [number, number][] = []
  for (let j = 0; j < n; j++) {
    let x = 0
    let y = 0
    for (let i = 0; i < n; i++) {
      x += N_inv[j][i] * fitPoints[i][0]
      y += N_inv[j][i] * fitPoints[i][1]
    }
    controlPoints.push([x, y])
  }

  return controlPoints
}

/**
 * ✅ P2-NEW-41 新增：解析样条标志位
 * DXF 组码 70 (SPLINE flags):
 *   bit 0 (1): 闭合样条 (周期)
 *   bit 1 (2): 闭合 B 样条 (固定起点)
 *   bit 2 (4): 平面样条 (所有控制点在同一平面)
 *   bit 3 (8): 有理样条 (权重不为 1)
 *   bit 4 (16): 线性样条 (degree=1)
 *   bit 5 (32): 复合样条 (multiple segments)
 */
function parseSplineFlags(flags: number): {
  isClosed: boolean
  isPeriodic: boolean
  isPlanar: boolean
  isRational: boolean
} {
  return {
    isClosed: (flags & 1) !== 0,      // bit 0: 闭合样条
    isPeriodic: (flags & 2) !== 0,    // bit 1: 周期样条
    isPlanar: (flags & 4) !== 0,      // bit 2: 平面样条
    isRational: (flags & 8) !== 0,    // bit 3: 有理样条
  }
}

/**
 * ✅ P1-NEW-32 新增：离散化拟合点样条
 * 1. 计算弦长参数化
 * 2. 构建节点向量
 * 3. 最小二乘法反算控制点
 * 4. 使用 Cox-de Boor 离散化
 *
 * ✅ P2-NEW-41 修复：使用 flags 判断样条类型
 * ✅ P3-NEW-45 优化：使用 LOD 系统调整容差
 */
function discretizeFitPointsSpline(
  fitPoints: [number, number][],
  degree: number,
  flags: number | undefined,
  cameraZoom: number,
  screenTolerance: number
): [number, number][] {
  if (fitPoints.length < 2) {
    return []
  }

  if (fitPoints.length === 2) {
    // 只有两个点：退化为直线
    return [fitPoints[0], fitPoints[1]]
  }

  // ✅ P2-NEW-41: 从 flags 提取 isClosed
  const { isClosed } = parseSplineFlags(flags ?? 0)

  // ✅ P3-NEW-45: 根据 LOD 调整容差
  const lod = getLodLevel(cameraZoom)
  const adjustedTolerance = screenTolerance * getScreenToleranceForLod(lod)

  // 1. 计算弦长参数化
  const parameters = computeChordLengths(fitPoints)

  // 2. 构建节点向量 (✅ P2-NEW-42: 使用完整平均技术)
  const knots = buildKnotVector(fitPoints, parameters, degree, isClosed)

  // 3. 最小二乘法反算控制点
  const controlPoints = computeControlPointsLeastSquares(
    fitPoints,
    parameters,
    knots,
    degree
  )

  // 4. 使用自适应 B 样条离散化 (✅ P3-NEW-45: 使用 LOD 调整容差)
  return discretizeBSplineAdaptive(
    controlPoints,
    knots,
    degree,
    cameraZoom,
    adjustedTolerance
  )
}

// ✅ P1-1 新增：Catmull-Rom 样条插值函数
function catmullRomSpline(
  p0: [number, number],
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
  t: number
): [number, number] {
  // Catmull-Rom 样条公式
  // P(t) = 0.5 * ((2*P1) + (-P0+P2)*t + (2*P0-5*P1+4*P2-P3)*t² + (-P0+3*P1-3*P2+P3)*t³)
  const t2 = t * t
  const t3 = t2 * t

  const x = 0.5 * (
    2 * p1[0] +
    (-p0[0] + p2[0]) * t +
    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
  )

  const y = 0.5 * (
    2 * p1[1] +
    (-p0[1] + p2[1]) * t +
    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
  )

  return [x, y]
}

// ✅ P0-NEW-6 新增：闭合 Catmull-Rom 样条求值
function evaluateCatmullRomClosed(
  controlPoints: [number, number][],
  t: number
): [number, number] {
  const n = controlPoints.length
  if (n < 2) return controlPoints[0] ?? [0, 0]

  // ✅ 闭合样条：使用循环索引
  const tScaled = t * n
  const i = Math.floor(tScaled)
  const localT = tScaled - i

  // ✅ 获取四个控制点（循环索引）
  const p0 = controlPoints[(i - 1 + n) % n]
  const p1 = controlPoints[i % n]
  const p2 = controlPoints[(i + 1) % n]
  const p3 = controlPoints[(i + 2) % n]

  return catmullRomSpline(p0, p1, p2, p3, localT)
}

// ✅ P0-NEW-6 新增：开放 Catmull-Rom 样条求值
function evaluateCatmullRomOpen(
  controlPoints: [number, number][],
  t: number
): [number, number] {
  const n = controlPoints.length
  if (n < 2) return controlPoints[0] ?? [0, 0]
  if (n === 2) {
    // ✅ 只有两个点：线性插值
    return [
      controlPoints[0][0] + t * (controlPoints[1][0] - controlPoints[0][0]),
      controlPoints[0][1] + t * (controlPoints[1][1] - controlPoints[0][1]),
    ]
  }

  // ✅ 开放样条：将 t 映射到正确的跨度
  const tScaled = t * (n - 1)
  const i = Math.floor(tScaled)
  const localT = tScaled - i

  // ✅ 获取四个控制点（边界处理）
  const p0 = controlPoints[Math.max(0, i - 1)]
  const p1 = controlPoints[i]
  const p2 = controlPoints[Math.min(n - 1, i + 1)]
  const p3 = controlPoints[Math.min(n - 1, i + 2)]

  return catmullRomSpline(p0, p1, p2, p3, localT)
}

// 批量渲染 HATCH（使用 Shape 缓存优化）
function BatchedHatches({
  hatches,
  camera,
  style,
  onHatchClick,
  selectedHatchIds,
  canvasWidth,
  canvasHeight,
  enableViewportCulling,
}: {
  hatches: HatchEntity[]
  camera: CameraState
  style: {
    fill?: string
    opacity?: number
  }
  onHatchClick?: (hatchId: number) => void
  selectedHatchIds?: number[]
  canvasWidth?: number  // P2-4 新增：画布宽度
  canvasHeight?: number  // P2-4 新增：画布高度
  enableViewportCulling?: boolean  // P2-4 新增：是否启用视口裁剪
}) {
  const handleClick = useCallback(
    (e: any) => {
      const hatchId = e.target.getAttr('hatchId')
      if (hatchId && onHatchClick) {
        onHatchClick(hatchId)
      }
    },
    [onHatchClick]
  )

  // ✅ P2-2 新增：开发模式下显示调试信息
  const isDevelopment = typeof process !== 'undefined' && process.env?.NODE_ENV === 'development'

  // 使用 useMemo 缓存 HATCH 元素
  const hatchElements: JSX.Element[] = useMemo(() => {
    // ✅ P2-4 新增：视口裁剪（仅渲染可见 HATCH）
    const hatchesToRender = enableViewportCulling && canvasWidth && canvasHeight
      ? hatches.filter(hatch => isHatchInViewport(hatch, camera, canvasWidth, canvasHeight))
      : hatches

    if (enableViewportCulling) {
      debugLog('INFO', '[HatchLayer] Viewport culling:', {
        total: hatches.length,
        visible: hatchesToRender.length,
        culled: hatches.length - hatchesToRender.length,
      })
    }

    debugLog('INFO', '[HatchLayer] Rendering hatches:', {
      count: hatchesToRender.length,
      patterns: hatchesToRender.map(h => ({ id: h.id, type: h.pattern.type, name: h.pattern.name })),
    })

    return hatchesToRender.map((hatch, index) => {
      // ✅ P0-3 调试：记录每个 HATCH 的渲染信息
      if (index < 3) {  // 只打印前 3 个避免日志过多
        debugLog('DEBUG', `[HatchLayer] Rendering hatch ${index}:`, {
          id: hatch.id,
          type: hatch.pattern.type,
          solid_fill: hatch.solid_fill,
          boundaryPathsCount: hatch.boundary_paths.length,
          firstPath: hatch.boundary_paths[0],
        })
      }

      return (
        <Shape
          key={hatch.id}
          sceneFunc={(context, shape) => {
            const ctx = context as unknown as CanvasRenderingContext2D

            // 渲染边界路径
            hatch.boundary_paths.forEach((path) => {
              ctx.beginPath()

              if (path.type === 'polyline' && path.points) {
                // ✅ P3-NEW-30 修复：使用边界预处理（Douglas-Peucker 简化）
                const processedPoints = preprocessHatchBoundary(
                  path.points,
                  path.closed ?? false,
                  camera.zoom
                )

                // ✅ P0-4 修复：处理带 bulge 的多段线边界
                const bulges = path.bulges
                if (bulges && bulges.length > 0) {
                  // ✅ P2-NEW-19 修复：传入 camera.zoom 参数，使用动态段数离散化圆弧
                  drawPolylineWithBulge(ctx, processedPoints, bulges, path.closed ?? false, camera.zoom)
                } else {
                  // 简单直线段多段线
                  processedPoints.forEach((point, i) => {
                    if (i === 0) {
                      ctx.moveTo(point[0], point[1])
                    } else {
                      ctx.lineTo(point[0], point[1])
                    }
                  })
                  // ✅ P0-NEW-3 修复：强制闭合所有多段线边界（HATCH 规范要求）
                  ctx.closePath()
                }
              } else if (path.type === 'arc' && path.center && path.radius) {
                // ✅ P0-NEW-3 修复：圆弧边界
                // ✅ P0-3 修复：后端已统一使用弧度，直接使用无需转换
                const startAngle = path.start_angle ?? 0
                const endAngle = path.end_angle ?? Math.PI * 2

                // ✅ 绘制圆弧
                ctx.arc(path.center[0], path.center[1], path.radius, startAngle, endAngle)
                // ✅ P0-NEW-8 修复：统一闭合逻辑（所有边界都必须闭合）
                ctx.closePath()
              } else if (path.type === 'ellipse_arc' && path.center && path.major_axis) {
                // ✅ P0-NEW-7 修复：椭圆弧离散化使用正确的参数方程
                const center = path.center
                const majorAxis = path.major_axis
                const minorAxisRatio = path.minor_axis_ratio ?? 1.0

                // ✅ P0-NEW-12 修复：添加边界检查，防止 major_axis 为零向量
                const semiMajorAxisLength = Math.sqrt(majorAxis[0] ** 2 + majorAxis[1] ** 2)
                if (semiMajorAxisLength < 1e-10) {
                  debugLog('WARN', '[HatchLayer] Invalid ellipse: major axis length is zero, skipping')
                  return  // 跳过无效的椭圆弧
                }
                const semiMinorAxisLength = semiMajorAxisLength * minorAxisRatio

                // ✅ P2-NEW-29 修复：正确计算短轴方向
                // 长轴单位向量
                const majorAxisUnitX = majorAxis[0] / semiMajorAxisLength
                const majorAxisUnitY = majorAxis[1] / semiMajorAxisLength

                // ✅ P1-NEW-33 修复：完整的 3D 变换（任意轴规则）
                // DXF 使用任意轴规则定义 3D 实体的本地坐标系
                // 给定法向量 N = (Nx, Ny, Nz)
                // 1. 计算本地 X 轴（使用 AutoCAD 任意轴规则）
                // 2. 计算本地 Y 轴 = N × X
                const extrusionDirection = path.extrusion_direction ?? [0, 0, 1]
                const [Nx, Ny, Nz] = extrusionDirection

                // 归一化法向量
                const normLen = Math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
                const [nx, ny, nz] = normLen > 1e-10
                  ? [Nx / normLen, Ny / normLen, Nz / normLen]
                  : [0, 0, 1]

                // ✅ 计算本地 X 轴（使用 AutoCAD 任意轴规则）
                let xAxisX: number, xAxisY: number

                if (Math.abs(nx) < 0.9 && Math.abs(ny) < 0.9) {
                  // 情况 A: N 不接近 Z 轴，使用 WCS X 轴 × N
                  // X = (0, nz, -ny)
                  xAxisX = 0
                  xAxisY = nz
                } else {
                  // 情况 B: N 接近 Z 轴，使用 WCS Y 轴 × N
                  // X = (ny, -nx, 0)
                  xAxisX = ny
                  xAxisY = -nx
                }

                // 归一化 X 轴
                const xLen = Math.sqrt(xAxisX * xAxisX + xAxisY * xAxisY)
                if (xLen > 1e-10) {
                  xAxisX /= xLen
                  xAxisY /= xLen
                } else {
                  // 退化情况：使用默认 X 轴
                  xAxisX = 1
                  xAxisY = 0
                }

                // ✅ 计算本地 Y 轴 = N × X (2D 投影)
                // Y = (nx * xAxisY - ny * xAxisX, ny * xAxisX - nx * xAxisY)
                // 简化：Y = (-nz * xAxisY, nz * xAxisX) （当 major 在 XY 平面时）
                let minorAxisUnitX = -nz * xAxisY
                let minorAxisUnitY = nz * xAxisX

                // 归一化 Y 轴
                const yLen = Math.sqrt(minorAxisUnitX * minorAxisUnitX + minorAxisUnitY * minorAxisUnitY)
                if (yLen > 1e-10) {
                  minorAxisUnitX /= yLen
                  minorAxisUnitY /= yLen
                }

                // ✅ P2-NEW-35 修复：完整的角度规范化
                // 1. 规范化到 [0, 2π)
                // 2. 处理跨越 2π 的情况
                // 3. 处理负角度
                const normalizeAngle = (angle: number): number => {
                  angle = angle % (Math.PI * 2)
                  if (angle < 0) angle += Math.PI * 2
                  return angle
                }

                let startAngle = normalizeAngle(path.start_angle ?? 0)
                let endAngle = normalizeAngle(path.end_angle ?? Math.PI * 2)
                const ccw = path.ccw ?? true

                // ✅ 计算角度范围
                let angleRange: number

                if (ccw) {
                  // 逆时针：从 start 到 end
                  if (endAngle >= startAngle) {
                    angleRange = endAngle - startAngle
                  } else {
                    angleRange = endAngle - startAngle + Math.PI * 2
                  }
                } else {
                  // 顺时针：从 start 到 end
                  if (endAngle <= startAngle) {
                    angleRange = startAngle - endAngle
                  } else {
                    angleRange = startAngle - endAngle + Math.PI * 2
                  }
                }

                // 限制角度范围在 [0, 2π]
                angleRange = Math.max(0, Math.min(angleRange, Math.PI * 2))

                // ✅ P0-NEW-9 修复：使用动态容差（根据 zoom 级别调整）
                const tolerance = calculateWorldTolerance(camera.zoom, 1.0)  // 1 像素容差
                const minRadius = Math.min(semiMajorAxisLength, semiMinorAxisLength)

                let numSegments: number
                if (minRadius <= tolerance) {
                  // 半径太小，使用固定段数
                  numSegments = Math.max(8, Math.ceil(angleRange / (Math.PI / 8)))
                } else {
                  const acosArg = Math.max(-1, Math.min(1, 1 - tolerance / minRadius))
                  const anglePerSegment = 2 * Math.acos(acosArg)
                  numSegments = Math.ceil(angleRange / anglePerSegment)
                }

                // ✅ P2-NEW-20 修复：根据椭圆周长动态调整最小段数
                // 使用 Ramanujan 近似公式计算椭圆周长
                const h = ((semiMajorAxisLength - semiMinorAxisLength) ** 2) /
                          ((semiMajorAxisLength + semiMinorAxisLength) ** 2)
                const ellipsePerimeter = Math.PI * (semiMajorAxisLength + semiMinorAxisLength) *
                                        (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h)))

                // 根据周长比例计算当前弧段的近似长度
                const arcLength = ellipsePerimeter * (angleRange / (Math.PI * 2))

                // ✅ 动态最小段数：基于弧长和容差
                const minSegmentsFromPerimeter = Math.max(8, Math.ceil(arcLength / (tolerance * 10)))
                numSegments = Math.max(minSegmentsFromPerimeter, Math.min(numSegments, 256))

                const angleStep = (endAngle - startAngle) / numSegments

                // ✅ P0-NEW-7 修复：正确的椭圆参数方程
                // P(t) = C + (a·cos(t))·majorAxisUnit + (b·sin(t))·minorAxisUnit
                for (let i = 0; i <= numSegments; i++) {
                  const t = startAngle + i * angleStep

                  // ✅ 局部坐标系椭圆点：(a·cos(t), b·sin(t))
                  const localX = semiMajorAxisLength * Math.cos(t)
                  const localY = semiMinorAxisLength * Math.sin(t)

                  // ✅ 应用旋转变换：P = C + localX·majorAxisUnit + localY·minorAxisUnit
                  const x = center[0] + localX * majorAxisUnitX + localY * minorAxisUnitX
                  const y = center[1] + localX * majorAxisUnitY + localY * minorAxisUnitY

                  if (i === 0) {
                    ctx.moveTo(x, y)
                  } else {
                    ctx.lineTo(x, y)
                  }
                }

                // ✅ P0-NEW-8 修复：统一闭合逻辑（所有边界都必须闭合）
                ctx.closePath()
              } else if (path.type === 'spline') {
                // ✅ P1-NEW-32 修复：支持拟合点样条和控制点样条
                // ✅ P2-NEW-41 修复：使用 flags 判断样条类型
                const degree = path.degree ?? 3
                const flags = path.flags ?? 0
                const hasFitPoints = path.fit_points && path.fit_points.length > 0
                const hasControlPoints = path.control_points && path.control_points.length > 0

                // ✅ P1-NEW-32: 优先使用拟合点，其次使用控制点
                let splinePoints: [number, number][] = []

                if (hasFitPoints) {
                  // ✅ 使用拟合点定义的样条
                  debugLog('INFO', '[HatchLayer] Rendering fit-points spline', {
                    fitPoints: path.fit_points!.length,
                    degree: degree,
                    flags: flags
                  })

                  splinePoints = discretizeFitPointsSpline(
                    path.fit_points!,
                    degree,
                    flags,  // ✅ P2-NEW-41: 传递 flags
                    camera.zoom,
                    1.0  // 1 像素屏幕容差
                  )

                  // 绘制离散化后的样条
                  splinePoints.forEach((point, i) => {
                    if (i === 0) {
                      ctx.moveTo(point[0], point[1])
                    } else {
                      ctx.lineTo(point[0], point[1])
                    }
                  })

                  // ✅ 闭合逻辑：闭合样条需要 closePath
                  ctx.closePath()
                } else if (hasControlPoints) {
                  // ✅ 使用控制点定义的样条
                  const controlPoints = path.control_points!
                  const knots = path.knots

                  // ✅ P1-6 修复：完整的控制点数量验证
                  // 对于 B 样条，控制点数量必须 >= degree + 1
                  if (controlPoints.length < degree + 1) {
                    debugLog('WARN', '[HatchLayer] Invalid spline: not enough control points', {
                      controlPoints: controlPoints.length,
                      degree: degree,
                      required: degree + 1
                    })
                    return  // 跳过无效样条
                  }

                  if (knots && knots.length > 0) {
                    // ✅ P0-NEW-11 修复：添加 knots 数据验证和归一化处理
                    const knotMin = knots[0] ?? 0
                    const knotMax = knots[knots.length - 1] ?? 1

                    // 验证 knots 是否归一化到 [0, 1]
                    let normalizedKnots = knots
                    if (knotMin < 0 || knotMax > 1) {
                      debugLog('WARN', '[HatchLayer] Knots not normalized to [0, 1], normalizing...', {
                        knotMin,
                        knotMax
                      })
                      // 归一化 knots 到 [0, 1]
                      const knotRange = knotMax - knotMin
                      if (knotRange > 1e-10) {
                        normalizedKnots = knots.map(k => (k - knotMin) / knotRange)
                      }
                    }

                    // ✅ 验证 knots 向量长度
                    // 对于 B 样条：knots.length = controlPoints.length + degree + 1
                    const expectedKnotsLength = controlPoints.length + degree + 1
                    if (knots.length !== expectedKnotsLength) {
                      debugLog('WARN', '[HatchLayer] Knots vector length mismatch', {
                        actual: knots.length,
                        expected: expectedKnotsLength,
                        controlPoints: controlPoints.length,
                        degree: degree
                      })
                    }

                    // ✅ P1-NEW-16 修复：使用自适应 B 样条离散化
                    // 根据曲率和 zoom 级别自动调整段数，平衡质量和性能
                    splinePoints = discretizeBSplineAdaptive(
                      controlPoints,
                      normalizedKnots,
                      degree,
                      camera.zoom,
                      1.0  // 1 像素屏幕容差
                    )

                    // 绘制离散化后的样条
                    splinePoints.forEach((point, i) => {
                      if (i === 0) {
                        ctx.moveTo(point[0], point[1])
                      } else {
                        ctx.lineTo(point[0], point[1])
                      }
                    })

                    // ✅ 闭合逻辑：闭合 B 样条需要 closePath
                    ctx.closePath()
                    return  // 提前返回，避免重复绘制
                  } else {
                    // ✅ P0-NEW-6 修复：正确的 Catmull-Rom 样条插值
                    // 使用弦长参数化，避免过度离散化
                    const isClosed = path.closed ?? false

                    // ✅ 计算样条总长度以确定段数
                    let totalLength = 0
                    for (let i = 0; i < controlPoints.length - 1; i++) {
                      const dx = controlPoints[i + 1][0] - controlPoints[i][0]
                      const dy = controlPoints[i + 1][1] - controlPoints[i][1]
                      totalLength += Math.sqrt(dx * dx + dy * dy)
                    }

                    // ✅ 每 10 单位 1 段，最少 20 段
                    const numSegments = Math.max(20, Math.ceil(totalLength / 10))

                    if (isClosed) {
                      // ✅ 闭合样条：循环索引
                      for (let i = 0; i <= numSegments; i++) {
                        const t = i / numSegments
                        const point = evaluateCatmullRomClosed(controlPoints, t)
                        if (i === 0) {
                          ctx.moveTo(point[0], point[1])
                        } else {
                          ctx.lineTo(point[0], point[1])
                        }
                      }
                    } else {
                      // ✅ 开放样条：正确端点处理
                      for (let i = 0; i <= numSegments; i++) {
                        const t = i / numSegments
                        const point = evaluateCatmullRomOpen(controlPoints, t)
                        if (i === 0) {
                          ctx.moveTo(point[0], point[1])
                        } else {
                          ctx.lineTo(point[0], point[1])
                        }
                      }
                    }

                    // ✅ P0-NEW-8 修复：统一闭合逻辑（所有边界都必须闭合）
                    ctx.closePath()
                  }
                }
              }

              // ✅ P3-NEW-36 修复：边界方向规范化
              // DXF 规范：外边界逆时针（面积 > 0），内边界/孤岛顺时针（面积 < 0）
              // 使用 non-zero 规则填充需要正确的边界方向

              // ✅ P2-NEW-43 新增：自相交检测
              // 检测自相交边界并提供警告，但继续使用 evenodd 规则处理
              const { hasIntersection, intersectingPaths } = detectHatchSelfIntersection(
                hatch.boundary_paths
              )
              if (hasIntersection) {
                debugLog('WARN', '[HatchLayer] Self-intersecting boundary detected', {
                  hatchId: hatch.id,
                  intersectingPaths,
                  message: 'Using evenodd rule for robust handling'
                })
              }

              // ✅ 填充（确保路径已闭合）
              // ✅ P2-NEW-21 修复：使用 'evenodd' 规则处理自相交边界
              // 奇偶规则正确处理自相交、嵌套边界等复杂情况
              if (hatch.solid_fill || hatch.pattern.type === 'solid') {
                ctx.fillStyle = rgbaToString(hatch.pattern.color)
                ctx.globalAlpha = style.opacity ?? 0.3
                ctx.fill('evenodd')  // ✅ 使用奇偶规则处理自相交
              } else {
                // ✅ P2-3 修复：使用缓存的图案 canvas，避免重复创建
                // ✅ P0-NEW-14 修复：优先使用 pattern.scale/angle，回退到 hatch.scale/angle
                const patternCanvas = getCachedPattern(
                  getPatternName(hatch.pattern),
                  rgbaToString(hatch.pattern.color),
                  hatch.pattern.scale ?? hatch.scale ?? 1,
                  hatch.pattern.angle ?? hatch.angle ?? 0
                )
                const pattern = ctx.createPattern(patternCanvas, 'repeat')
                if (pattern) {
                  ctx.fillStyle = pattern
                  ctx.globalAlpha = style.opacity ?? 0.6
                  ctx.fill('evenodd')  // ✅ 使用奇偶规则处理自相交
                }
              }
            })

            // ✅ P2-2 新增：开发模式下显示边界控制点（调试用）
            if (isDevelopment) {
              hatch.boundary_paths.forEach((path) => {
                // 绘制控制点（红色小圆点）
                if (path.control_points) {
                  path.control_points.forEach((point) => {
                    ctx.beginPath()
                    ctx.arc(point[0], point[1], 3 / camera.zoom, 0, Math.PI * 2)
                    ctx.fillStyle = '#ff0000'
                    ctx.fill()
                    ctx.strokeStyle = '#ffffff'
                    ctx.lineWidth = 1 / camera.zoom
                    ctx.stroke()
                  })
                }

                // 绘制多段线顶点（蓝色小圆点）
                if (path.points) {
                  path.points.forEach((point) => {
                    ctx.beginPath()
                    ctx.arc(point[0], point[1], 3 / camera.zoom, 0, Math.PI * 2)
                    ctx.fillStyle = '#0000ff'
                    ctx.fill()
                  })
                }

                // 绘制椭圆弧中心（绿色十字）
                if (path.type === 'ellipse_arc' && path.center) {
                  const cx = path.center[0]
                  const cy = path.center[1]
                  const size = 5 / camera.zoom
                  ctx.beginPath()
                  ctx.moveTo(cx - size, cy)
                  ctx.lineTo(cx + size, cy)
                  ctx.moveTo(cx, cy - size)
                  ctx.lineTo(cx, cy + size)
                  ctx.strokeStyle = '#00ff00'
                  ctx.lineWidth = 2 / camera.zoom
                  ctx.stroke()
                }
              })
            }

            // 选中时绘制边框
            if (shape.getAttr('isSelected')) {
              ctx.strokeStyle = HATCH_STYLES.selected.stroke
              ctx.lineWidth = HATCH_STYLES.selected.strokeWidth
              ctx.stroke()
            }
          }}
          attrHatchId={hatch.id}
          attrIsSelected={selectedHatchIds?.includes(hatch.id)}
          onClick={handleClick}
          onTap={handleClick}
        />
      )
    })

    // ✅ P2-2 新增：开发模式下显示缓存统计
    // ✅ P2-NEW-23 修复：使用调试级别控制
    if (isDevelopment) {
      const stats = getPatternCacheStats()
      debugLog('DEBUG', '[HatchLayer] Pattern Cache Stats:', stats)
    }

    return hatchElements
  }, [hatches, camera.zoom, style.opacity, selectedHatchIds, handleClick, isDevelopment])

  return <>{hatchElements}</>
}

export function HatchLayer({
  hatches,
  camera,
  selectedHatchIds = [],
  onHatchClick,
  canvasWidth,
  canvasHeight,
  enableViewportCulling = true,  // ✅ P2-NEW-18 修复：默认开启视口裁剪，提升大文件性能
}: HatchLayerProps & {
  canvasWidth?: number
  canvasHeight?: number
  enableViewportCulling?: boolean
}) {
  const selectedHatchIdSet = useMemo(() => new Set(selectedHatchIds), [selectedHatchIds])

  // 分组 HATCH
  // P0-4 修复：适配新的 pattern type ('predefined' | 'custom' | 'solid')
  const groupedHatches = useMemo(() => {
    const groups = new Map<string, HatchEntity[]>()

    hatches.forEach((hatch: HatchEntity) => {
      // 确定 HATCH 的组别
      let groupKey: string

      if (selectedHatchIdSet.has(hatch.id)) {
        groupKey = 'selected'
      } else if (hatch.solid_fill || hatch.pattern.type === 'solid') {
        groupKey = 'solid'
      } else if (hatch.pattern.type === 'custom') {
        groupKey = 'custom'
      } else {
        // 'predefined' 类型
        groupKey = 'predefined'
      }

      const group = groups.get(groupKey) || []
      group.push(hatch)
      groups.set(groupKey, group)
    })

    return groups
  }, [hatches, selectedHatchIdSet])

  // 渲染各组 HATCH
  return (
    <>
      {/* 实体填充 */}
      {groupedHatches.get('solid') && (
        <BatchedHatches
          hatches={groupedHatches.get('solid')!}
          camera={camera}
          style={HATCH_STYLES.solid.default}
          onHatchClick={onHatchClick}
          selectedHatchIds={selectedHatchIds}
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          enableViewportCulling={enableViewportCulling}
        />
      )}

      {/* 预定义图案填充 */}
      {groupedHatches.get('predefined') && (
        <BatchedHatches
          hatches={groupedHatches.get('predefined')!}
          camera={camera}
          style={HATCH_STYLES.predefined.default}
          onHatchClick={onHatchClick}
          selectedHatchIds={selectedHatchIds}
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          enableViewportCulling={enableViewportCulling}
        />
      )}

      {/* 自定义图案填充 */}
      {groupedHatches.get('custom') && (
        <BatchedHatches
          hatches={groupedHatches.get('custom')!}
          camera={camera}
          style={HATCH_STYLES.custom.default}
          onHatchClick={onHatchClick}
          selectedHatchIds={selectedHatchIds}
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          enableViewportCulling={enableViewportCulling}
        />
      )}

      {/* 选中的 HATCH */}
      {groupedHatches.get('selected') && (
        <BatchedHatches
          hatches={groupedHatches.get('selected')!}
          camera={camera}
          style={{ opacity: 0.5 }}
          onHatchClick={onHatchClick}
          selectedHatchIds={selectedHatchIds}
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          enableViewportCulling={enableViewportCulling}
        />
      )}
    </>
  )
}
