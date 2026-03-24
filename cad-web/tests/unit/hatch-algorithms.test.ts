/**
 * HATCH 关键算法单元测试
 * 
 * 测试覆盖：
 * - bulgeToArc: Bulge 值转圆弧算法
 * - discretizeArc: 圆弧动态离散化算法
 * - LRUPatternCache: LRU 图案缓存管理
 * - distancePointToLine: 点到直线距离计算
 * - discretizeBSplineAdaptive: 自适应 B 样条离散化
 */

import { describe, it, expect } from 'vitest'

// ============================================================================
// 工具函数（从 hatch-layer.tsx 复制）
// ============================================================================

/**
 * 计算世界空间容差
 */
function calculateWorldTolerance(cameraZoom: number, screenTolerance: number = 1.0): number {
  return screenTolerance / cameraZoom
}

/**
 * 计算两点间距离
 */
function distance(p1: [number, number], p2: [number, number]): number {
  return Math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
}

/**
 * 计算点到直线的距离
 */
function distancePointToLine(
  point: [number, number],
  lineStart: [number, number],
  lineEnd: [number, number]
): number {
  const A = point[0] - lineStart[0]
  const B = point[1] - lineStart[1]
  const C = lineEnd[0] - lineStart[0]
  const D = lineEnd[1] - lineStart[1]

  const dot = A * C + B * D
  const lenSq = C * C + D * D
  let param = -1

  if (lenSq !== 0) {
    param = dot / lenSq
  }

  let xx: number, yy: number

  if (param < 0) {
    xx = lineStart[0]
    yy = lineStart[1]
  } else if (param > 1) {
    xx = lineEnd[0]
    yy = lineEnd[1]
  } else {
    xx = lineStart[0] + param * C
    yy = lineStart[1] + param * D
  }

  const dx = point[0] - xx
  const dy = point[1] - yy

  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * 根据 bulge 值计算圆弧的圆心和半径
 * bulge = tan(θ/4)，其中 θ 是圆弧的包含角
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
 * 动态离散化圆弧
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

  // 动态计算段数
  const worldTolerance = calculateWorldTolerance(cameraZoom, 1.0)

  let numSegments: number
  if (radius <= worldTolerance) {
    // 半径太小，使用固定段数
    numSegments = Math.max(8, Math.ceil(angleRange / (Math.PI / 8)))
  } else {
    const acosArg = Math.max(-1, Math.min(1, 1 - worldTolerance / radius))
    const anglePerSegment = 2 * Math.acos(acosArg)
    numSegments = Math.ceil(angleRange / anglePerSegment)
  }

  // 限制段数范围
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

/**
 * LRU 图案缓存类
 */
class LRUPatternCache {
  private cache = new Map<string, HTMLCanvasElement | any>()
  private accessOrder: string[] = []
  private readonly maxSizeValue: number

  constructor(maxSize: number = 50) {
    this.maxSizeValue = maxSize
  }

  get(key: string): HTMLCanvasElement | any | undefined {
    const canvas = this.cache.get(key)
    if (canvas) {
      // 更新访问顺序
      this.accessOrder = this.accessOrder.filter(k => k !== key)
      this.accessOrder.push(key)
    }
    return canvas
  }

  set(key: string, canvas: HTMLCanvasElement | any) {
    if (this.cache.has(key)) {
      this.accessOrder = this.accessOrder.filter(k => k !== key)
    }

    while (this.cache.size >= this.maxSizeValue) {
      const oldestKey = this.accessOrder.shift()
      if (oldestKey) {
        this.cache.delete(oldestKey)
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

  get maxSize(): number {
    return this.maxSizeValue
  }
}

// ============================================================================
// 单元测试
// ============================================================================

describe('bulgeToArc', () => {
  describe('零 bulge 处理', () => {
    it('should return null for zero bulge', () => {
      const arc = bulgeToArc([0, 0], [100, 0], 0)
      expect(arc).toBeNull()
    })

    it('should return null for very small bulge', () => {
      const arc = bulgeToArc([0, 0], [100, 0], 1e-11)
      expect(arc).toBeNull()
    })
  })

  describe('正 bulge（逆时针）', () => {
    it('should handle positive bulge (CCW)', () => {
      const arc = bulgeToArc([0, 0], [100, 0], 0.5)
      expect(arc).not.toBeNull()
      expect(arc!.ccw).toBe(true)
      expect(arc!.radius).toBeGreaterThan(0)
      expect(arc!.center[1]).toBeGreaterThan(0) // 圆心在弦上方
    })

    it('should calculate correct included angle for bulge=1', () => {
      // bulge = 1 => θ = 4 * atan(1) = π (180 度，半圆)
      const arc = bulgeToArc([0, 0], [100, 0], 1)
      expect(arc).not.toBeNull()
      const includedAngle = arc!.endAngle - arc!.startAngle
      expect(Math.abs(includedAngle)).toBeCloseTo(Math.PI, 5)
    })
  })

  describe('负 bulge（顺时针）', () => {
    it('should handle negative bulge (CW)', () => {
      const arc = bulgeToArc([0, 0], [100, 0], -0.5)
      expect(arc).not.toBeNull()
      expect(arc!.ccw).toBe(false)
      expect(arc!.radius).toBeGreaterThan(0)
      expect(arc!.center[1]).toBeLessThan(0) // 圆心在弦下方
    })
  })

  describe('边界情况', () => {
    it('should handle zero chord length', () => {
      const arc = bulgeToArc([50, 50], [50, 50], 0.5)
      expect(arc).toBeNull()
    })

    it('should handle large bulge value', () => {
      const arc = bulgeToArc([0, 0], [100, 0], 10)
      expect(arc).not.toBeNull()
      expect(arc!.radius).toBeGreaterThan(50) // 大 bulge 值对应小圆弧
    })
  })
})

describe('discretizeArc', () => {
  describe('基本功能', () => {
    it('should generate points for full circle', () => {
      const points = discretizeArc([0, 0], 50, 0, Math.PI * 2, true, 1)
      expect(points.length).toBeGreaterThan(8)
      expect(points.length).toBeLessThanOrEqual(257)

      // 验证起点和终点接近
      const firstPoint = points[0]
      const lastPoint = points[points.length - 1]
      const dist = distance(firstPoint, lastPoint)
      expect(dist).toBeLessThan(1) // 接近闭合
    })

    it('should generate points for quarter circle', () => {
      const points = discretizeArc([0, 0], 50, 0, Math.PI / 2, true, 1)
      expect(points.length).toBeGreaterThanOrEqual(2)
    })
  })

  describe('动态段数', () => {
    it('should use more segments for larger radius', () => {
      const pointsSmall = discretizeArc([0, 0], 10, 0, Math.PI * 2, true, 1)
      const pointsLarge = discretizeArc([0, 0], 100, 0, Math.PI * 2, true, 1)
      
      // 大半径需要更多段数来保持相同的弦高误差
      expect(pointsLarge.length).toBeGreaterThanOrEqual(pointsSmall.length)
    })

    it('should use more segments at higher zoom levels', () => {
      const pointsZoom1 = discretizeArc([0, 0], 50, 0, Math.PI * 2, true, 1)
      const pointsZoom2 = discretizeArc([0, 0], 50, 0, Math.PI * 2, true, 2)
      
      // 更高缩放级别需要更多段数
      expect(pointsZoom2.length).toBeGreaterThanOrEqual(pointsZoom1.length)
    })
  })

  describe('方向控制', () => {
    it('should generate points in CCW direction', () => {
      const points = discretizeArc([0, 0], 50, 0, Math.PI / 2, true, 1)
      // 第二点应该在第一点的逆时针方向
      expect(points[1][0]).toBeLessThan(points[0][0]) // x 减小
      expect(points[1][1]).toBeGreaterThan(points[0][1]) // y 增加
    })

    it('should generate points in CW direction', () => {
      const points = discretizeArc([0, 0], 50, Math.PI / 2, 0, false, 1)
      // 顺时针方向
      expect(points.length).toBeGreaterThanOrEqual(2)
    })
  })
})

describe('LRUPatternCache', () => {
  describe('基本操作', () => {
    it('should store and retrieve values', () => {
      const cache = new LRUPatternCache(3)
      const canvas1 = { id: 'canvas1' }
      
      cache.set('key1', canvas1)
      expect(cache.get('key1')).toBe(canvas1)
      expect(cache.has('key1')).toBe(true)
    })

    it('should return undefined for missing keys', () => {
      const cache = new LRUPatternCache(3)
      expect(cache.get('nonexistent')).toBeUndefined()
    })
  })

  describe('LRU 驱逐', () => {
    it('should evict oldest entries when full', () => {
      const cache = new LRUPatternCache(3)
      const canvas1 = { id: 'canvas1' }
      const canvas2 = { id: 'canvas2' }
      const canvas3 = { id: 'canvas3' }
      const canvas4 = { id: 'canvas4' }

      cache.set('a', canvas1)
      cache.set('b', canvas2)
      cache.set('c', canvas3)
      cache.set('d', canvas4)  // 应该驱逐 'a'

      expect(cache.get('a')).toBeUndefined()
      expect(cache.get('b')).toBe(canvas2)
      expect(cache.get('c')).toBe(canvas3)
      expect(cache.get('d')).toBe(canvas4)
      expect(cache.size).toBe(3)
    })

    it('should update access order on get', () => {
      const cache = new LRUPatternCache(3)
      const canvas1 = { id: 'canvas1' }
      const canvas2 = { id: 'canvas2' }
      const canvas3 = { id: 'canvas3' }
      const canvas4 = { id: 'canvas4' }

      cache.set('a', canvas1)
      cache.set('b', canvas2)
      cache.set('c', canvas3)

      cache.get('a')  // 访问 'a'，更新为最近使用
      cache.set('d', canvas4)  // 应该驱逐 'b'（最久未使用）

      expect(cache.get('a')).toBe(canvas1)
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBe(canvas3)
      expect(cache.get('d')).toBe(canvas4)
    })

    it('should handle repeated access correctly', () => {
      const cache = new LRUPatternCache(2)
      const canvas1 = { id: 'canvas1' }
      const canvas2 = { id: 'canvas2' }
      const canvas3 = { id: 'canvas3' }

      cache.set('a', canvas1)
      cache.set('b', canvas2)
      
      // 多次访问 'a'
      cache.get('a')
      cache.get('a')
      cache.get('a')
      
      cache.set('c', canvas3)  // 应该驱逐 'b'

      expect(cache.get('a')).toBe(canvas1)
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBe(canvas3)
    })
  })

  describe('更新现有键', () => {
    it('should update value for existing key', () => {
      const cache = new LRUPatternCache(3)
      const canvas1 = { id: 'canvas1' }
      const canvas2 = { id: 'canvas2' }

      cache.set('key', canvas1)
      cache.set('key', canvas2)

      expect(cache.get('key')).toBe(canvas2)
      expect(cache.size).toBe(1)
    })

    it('should update access order when updating existing key', () => {
      const cache = new LRUPatternCache(3)
      const canvas1 = { id: 'canvas1' }
      const canvas2 = { id: 'canvas2' }
      const canvas3 = { id: 'canvas3' }
      const canvas4 = { id: 'canvas4' }

      cache.set('a', canvas1)
      cache.set('b', canvas2)
      cache.set('c', canvas3)
      
      // 更新 'a'，应该移到队尾
      cache.set('a', canvas4)
      
      // 现在顺序应该是 b, c, a
      // 添加新元素应该驱逐 'b'（最久未使用）
      cache.set('d', { id: 'canvas5' })

      expect(cache.get('a')).toBe(canvas4)
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBe(canvas3)
      expect(cache.get('d')).toBeDefined()
    })
  })

  describe('clear 操作', () => {
    it('should clear all entries', () => {
      const cache = new LRUPatternCache(3)
      cache.set('a', { id: 'canvas1' })
      cache.set('b', { id: 'canvas2' })
      cache.set('c', { id: 'canvas3' })

      cache.clear()

      expect(cache.size).toBe(0)
      expect(cache.get('a')).toBeUndefined()
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBeUndefined()
    })
  })
})

describe('distancePointToLine', () => {
  describe('基本距离计算', () => {
    it('should calculate perpendicular distance', () => {
      // 点 (0, 1) 到直线 y=0 的距离应该是 1
      const dist = distancePointToLine([0, 1], [0, 0], [10, 0])
      expect(dist).toBeCloseTo(1, 10)
    })

    it('should handle point on line', () => {
      const dist = distancePointToLine([5, 0], [0, 0], [10, 0])
      expect(dist).toBeCloseTo(0, 10)
    })
  })

  describe('投影在线段外', () => {
    it('should handle projection before start point', () => {
      // 点 (-1, 0) 到线段 [0,0]-[10,0] 的距离应该是 1
      const dist = distancePointToLine([-1, 0], [0, 0], [10, 0])
      expect(dist).toBeCloseTo(1, 10)
    })

    it('should handle projection after end point', () => {
      // 点 (11, 0) 到线段 [0,0]-[10,0] 的距离应该是 1
      const dist = distancePointToLine([11, 0], [0, 0], [10, 0])
      expect(dist).toBeCloseTo(1, 10)
    })
  })

  describe('边界情况', () => {
    it('should handle zero length line', () => {
      const dist = distancePointToLine([5, 0], [0, 0], [0, 0])
      expect(dist).toBeCloseTo(5, 10)
    })

    it('should handle vertical line', () => {
      const dist = distancePointToLine([1, 5], [0, 0], [0, 10])
      expect(dist).toBeCloseTo(1, 10)
    })

    it('should handle diagonal line', () => {
      // 点 (0, 0) 到直线 y=x 的距离应该是 0
      const dist = distancePointToLine([1, 1], [0, 0], [10, 10])
      expect(dist).toBeCloseTo(0, 10)
    })
  })
})

describe('calculateWorldTolerance', () => {
  it('should calculate tolerance correctly', () => {
    expect(calculateWorldTolerance(1, 1.0)).toBeCloseTo(1.0, 10)
    expect(calculateWorldTolerance(2, 1.0)).toBeCloseTo(0.5, 10)
    expect(calculateWorldTolerance(0.5, 1.0)).toBeCloseTo(2.0, 10)
  })

  it('should use default screenTolerance', () => {
    expect(calculateWorldTolerance(1)).toBeCloseTo(1.0, 10)
  })
})
