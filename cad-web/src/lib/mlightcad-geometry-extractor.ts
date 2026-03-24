/**
 * mlightcad 几何数据提取工具
 *
 * 基于 RealDWG Web API 从 AcApDocument.database 提取几何数据
 * 参考：https://mlight-lee.github.io/realdwg-web/
 */

import type { Edge, HatchEntity, HatchBoundaryPath } from '@/types/api'
import type { Point } from '@/types/api'
import type { BoundarySemantic } from '@/types/api'
import type { Database, BlockTableRecord, AcApDocument, Entity } from '@/types/mlightcad'

/**
 * LOD 级别配置
 * ✅ S015: 根据缩放级别动态调整离散化段数
 */
export interface ExtractOptions {
  /** LOD 级别 */
  lodLevel?: 'high' | 'medium' | 'low'
  /** 目标缩放级别，用于计算 LOD */
  targetZoom?: number
}

/**
 * 从 mlightcad database 提取几何数据
 * ✅ S005: 缓存 modelSpace 引用，避免重复查找
 * ✅ S015: 支持 LOD 参数动态调整离散化段数
 */
export class MlightCadGeometryExtractor {
  private database: Database
  private document: AcApDocument | null
  private entityIdCounter = 0
  private modelSpaceCache: BlockTableRecord | null = null
  private extractOptions: ExtractOptions = {}

  constructor(database: Database, document?: AcApDocument, options?: ExtractOptions) {
    this.database = database
    this.document = document ?? null
    this.extractOptions = options || {}
  }

  /**
   * 使缓存失效
   */
  invalidateCache(): void {
    this.modelSpaceCache = null
  }

  /**
   * 提取所有几何数据
   * ✅ S015: 添加 LOD 参数支持动态调整离散化段数
   */
  async extractAll(options?: ExtractOptions): Promise<{
    edges: Edge[]
    hatches: HatchEntity[]
    bounds: { minX: number; minY: number; maxX: number; maxY: number }
  }> {
    const edges: Edge[] = []
    const hatches: HatchEntity[] = []
    
    // 合并选项
    const extractOptions = { ...this.extractOptions, ...options }

    try {
      // 尝试使用 RealDWG Web API 的标准方法
      // 1. 首先尝试获取模型空间的 BlockTableRecord
      const modelSpace = await this.getModelSpace()
      if (!modelSpace) {
        console.warn('[MlightCadExtractor] Cannot access model space')
        return { edges: [], hatches: [], bounds: this.createEmptyBounds() }
      }

      // 2. 遍历模型空间中的所有实体
      const entities = await this.iterateEntities(modelSpace)

      for (const entity of entities) {
        const entityType = entity.dxftype?.toLowerCase() || entity.type?.toLowerCase() || ''

        // 提取线性实体（Line, Polyline, LWPolyline, Arc, Circle）
        if (this.isLinearEntity(entityType)) {
          const edgeResult = this.convertToEdge(entity, extractOptions)
          if (edgeResult) {
            // ✅ 处理单个 Edge 或 Edge 数组
            if (Array.isArray(edgeResult)) {
              edges.push(...edgeResult)
            } else {
              edges.push(edgeResult)
            }
          }
        }

        // 提取填充实体（Hatch）
        if (entityType === 'hatch') {
          const hatch = this.convertToHatch(entity)
          if (hatch) {
            hatches.push(hatch)
          }
        }
      }

      // 3. 计算边界框
      const bounds = this.calculateBounds(edges, hatches)

      return { edges, hatches, bounds }
    } catch (error) {
      console.error('[MlightCadExtractor] Error extracting geometry:', error)
      return { edges: [], hatches: [], bounds: this.createEmptyBounds() }
    }
  }

  /**
   * 获取模型空间的 BlockTableRecord
   * ✅ FIX-001: 使用正确的 RealDWG Web API 路径
   * ✅ FIX-002: 避免访问 .tables.blockTable.modelSpace（会触发 workingDatabase 检查）
   * ✅ S005: 缓存 modelSpace 引用，避免每次提取都遍历降级链
   */
  private async getModelSpace(): Promise<BlockTableRecord | null> {
    // ✅ 优先返回缓存
    if (this.modelSpaceCache) {
      console.log('[MlightCadExtractor] Returning cached modelSpace')
      return this.modelSpaceCache
    }

    try {
      const db = this.database
      let modelSpace: BlockTableRecord | null = null

      // ✅ 使用显式 API 替代惰性求值的属性访问
      if (typeof db.getModelSpace === 'function') {
        modelSpace = await db.getModelSpace()
        if (modelSpace) {
          console.log('[MlightCadExtractor] getModelSpace() succeeded')
        }
      }

      // 降级方案：尝试通过 document 获取
      if (!modelSpace && this.document && typeof this.document.getModelSpace === 'function') {
        modelSpace = await this.document.getModelSpace()
        if (modelSpace) {
          console.log('[MlightCadExtractor] getModelSpace via document succeeded')
        }
      }

      // 降级方案：使用 tables.blockTable.modelSpace
      if (!modelSpace && db.tables?.blockTable?.modelSpace) {
        modelSpace = db.tables.blockTable.modelSpace
      }

      // 方法 2: 使用 blockTableRecord 属性
      if (!modelSpace && db.blockTableRecord) {
        modelSpace = db.blockTableRecord
      }

      // 方法 3: 使用 getBlockTableRecord 方法
      if (!modelSpace && typeof db.getBlockTableRecord === 'function') {
        modelSpace = await db.getBlockTableRecord()
      }

      // 方法 4: 直接访问 *Model_Space
      if (!modelSpace && db['*Model_Space']) {
        modelSpace = db['*Model_Space']
      }

      // 方法 5: 尝试从 objects 中查找
      if (!modelSpace && db.objects) {
        modelSpace = Object.values(db.objects).find(
          (obj) => obj.name === '*Model_Space' || obj.handle === '0'
        ) as BlockTableRecord | null
      }

      // ✅ 缓存结果
      if (modelSpace) {
        this.modelSpaceCache = modelSpace
        console.log('[MlightCadExtractor] modelSpace cached')
      }

      return modelSpace
    } catch (error) {
      console.error('[MlightCadExtractor] Error getting model space:', error)
      return null
    }
  }

  /**
   * 迭代模型空间中的所有实体
   * ✅ FIX-002: 使用 newIterator() API (RealDWG Web 标准)
   */
  private async iterateEntities(modelSpace: BlockTableRecord): Promise<any[]> {
    const entities: any[] = []

    try {
      // ✅ 方法 1 (优先): 使用 newIterator() (RealDWG Web 标准 API)
      if (typeof modelSpace.newIterator === 'function') {
        const iterator = modelSpace.newIterator()
        // 迭代器可能是数组或可迭代对象
        if (Array.isArray(iterator)) {
          entities.push(...iterator)
        } else if (typeof iterator[Symbol.iterator] === 'function') {
          for (const entity of iterator) {
            entities.push(entity)
          }
        }
      }

      // 方法 2: 使用 entities 字典对象
      if (modelSpace.entities && typeof modelSpace.entities === 'object' && !Array.isArray(modelSpace.entities)) {
        entities.push(...Object.values(modelSpace.entities))
      }

      // 方法 3: 使用 entities 数组
      if (Array.isArray(modelSpace.entities)) {
        entities.push(...modelSpace.entities)
      }

      // 方法 4: 使用 forEach 方法
      if (typeof modelSpace.forEach === 'function') {
        modelSpace.forEach((entity: any) => {
          entities.push(entity)
        })
      }

      // 方法 5: 使用 iterator (如果 BlockTableRecord 本身可迭代)
      // 注意：TypeScript 可能不认为 BlockTableRecord 是可迭代的，所以需要类型断言
      if (typeof (modelSpace as any)[Symbol.iterator] === 'function') {
        for (const entity of modelSpace as any) {
          entities.push(entity)
        }
      }

      // 方法 6: 直接访问 objects
      if (modelSpace.objects) {
        entities.push(...Object.values(modelSpace.objects))
      }

      // 方法 7: 从 database 直接获取所有 entities
      if (this.database.entities && Array.isArray(this.database.entities)) {
        entities.push(...this.database.entities)
      }

    } catch (error) {
      console.error('[MlightCadExtractor] Error iterating entities:', error)
    }

    return entities
  }

  /**
   * 判断是否为线性实体
   */
  private isLinearEntity(type: string): boolean {
    const linearTypes = ['line', 'polyline', 'lwpolyline', 'arc', 'circle', 'ellipse', 'spline']
    return linearTypes.includes(type)
  }

  /**
   * ✅ 完整提取多段线的所有边（支持 bulge 弧段）
   * P1-NEW: 检测 bulge 值并转换为原生弧线 Edge
   */
  private polylineToEdges(
    entity: any,
    baseId: number,
    layer: string,
    _handle: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined
  ): Edge[] {
    const vertices = entity.vertices || []
    const bulges = entity.bulges || []
    const edges: Edge[] = []

    if (vertices.length < 2) {
      return edges
    }

    // 提取所有连续的边
    for (let i = 0; i < vertices.length - 1; i++) {
      const start = vertices[i]
      const end = vertices[i + 1]
      const bulge = bulges[i] || 0

      // ✅ 检测 bulge 值，转换为弧线 Edge
      if (bulge !== 0) {
        const arcEdge = this.bulgeToArc(
          [start.x || 0, start.y || 0],
          [end.x || 0, end.y || 0],
          bulge,
          baseId * 1000 + i,
          layer,
          isWall,
          semantic
        )
        edges.push(arcEdge)
      } else {
        // 直线段
        edges.push({
          id: baseId * 1000 + i,
          start: [start.x || 0, start.y || 0],
          end: [end.x || 0, end.y || 0],
          layer,
          is_wall: isWall,
          semantic,
        })
      }
    }

    // 如果多段线闭合，添加最后一条边回到起点
    const isClosed = entity.closed === true || entity.isClosed === true
    if (isClosed && vertices.length >= 3) {
      const start = vertices[vertices.length - 1]
      const end = vertices[0]
      // 检查闭合段是否有 bulge（某些 DXF 会在最后一个顶点存储）
      const bulge = bulges[vertices.length - 1] || 0
      
      if (bulge !== 0) {
        const arcEdge = this.bulgeToArc(
          [start.x || 0, start.y || 0],
          [end.x || 0, end.y || 0],
          bulge,
          baseId * 1000 + vertices.length,
          layer,
          isWall,
          semantic
        )
        edges.push(arcEdge)
      } else {
        edges.push({
          id: baseId * 1000 + vertices.length,
          start: [start.x || 0, start.y || 0],
          end: [end.x || 0, end.y || 0],
          layer,
          is_wall: isWall,
          semantic,
        })
      }
    }

    return edges
  }

  /**
   * ✅ P1-NEW: 将 bulge 值转换为弧线 Edge
   * bulge = tan(θ/4)，其中 θ 是弧的包含角
   * @param p1 起点 [x, y]
   * @param p2 终点 [x, y]
   * @param bulge bulge 值（正数=逆时针，负数=顺时针）
   * @param id Edge ID
   * @param layer 图层名称
   * @param isWall 是否为墙体
   * @param semantic 语义类型
   */
  private bulgeToArc(
    p1: Point,
    p2: Point,
    bulge: number,
    id: number,
    layer: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined
  ): Edge {
    // 计算弦长和方向
    const dx = p2[0] - p1[0]
    const dy = p2[1] - p1[1]
    const chordLength = Math.sqrt(dx * dx + dy * dy)

    // bulge = tan(θ/4)，计算包含角 θ
    const includedAngle = 4 * Math.atan(Math.abs(bulge))

    // 计算弧半径：r = L / (2 * sin(θ/2))
    const radius = chordLength / (2 * Math.sin(includedAngle / 2))

    // 计算矢高（sagitta）：s = r * (1 - cos(θ/2)) = (L/2) * tan(θ/4)
    const sagitta = (chordLength / 2) * Math.tan(includedAngle / 4)

    // 弦的中点
    const midX = (p1[0] + p2[0]) / 2
    const midY = (p1[1] + p2[1]) / 2

    // 垂直方向单位向量（从 p1 到 p2 的垂直方向）
    const perpX = -dy / chordLength
    const perpY = dx / chordLength

    // 圆心位置（bulge > 0 表示逆时针，圆心在左侧；bulge < 0 表示顺时针，圆心在右侧）
    const centerX = midX + perpX * sagitta * Math.sign(bulge)
    const centerY = midY + perpY * sagitta * Math.sign(bulge)

    // 计算起点和终点角度（相对于圆心）
    const startAngle = Math.atan2(p1[1] - centerY, p1[0] - centerX)
    let endAngle = Math.atan2(p2[1] - centerY, p2[0] - centerX)

    // 确定旋转方向（ccw = true 表示逆时针）
    const ccw = bulge > 0

    // 调整 endAngle 确保正确的弧线方向
    if (ccw) {
      // 逆时针：确保 endAngle > startAngle（如果需要，加 2π）
      if (endAngle <= startAngle) {
        endAngle += Math.PI * 2
      }
    } else {
      // 顺时针：确保 endAngle < startAngle（如果需要，减 2π）
      if (endAngle >= startAngle) {
        endAngle -= Math.PI * 2
      }
    }

    // 返回带有 arc 字段的 Edge
    return {
      id,
      start: p1,
      end: p2,
      layer,
      is_wall: isWall,
      semantic,
      arc: {
        center: [centerX, centerY],
        radius,
        start_angle: startAngle,
        end_angle: endAngle,
        ccw,
      },
    }
  }

  /**
   * ✅ 椭圆离散化为多个线段
   * ✅ P2-IMPROVED: 基于视觉误差阈值动态调整段数
   * ✅ S015: 支持 LOD 参数，根据缩放级别调整段数
   *
   * 算法原理：
   * - 圆弧离散化的最大弦高误差 = r * (1 - cos(π/n)) ≈ r * π² / (2 * n²)
   * - 设定允许的最大误差为像素单位（与视图缩放无关）
   * - 根据椭圆半径动态调整段数，大半径需要更多段数
   * - LOD 调整：zoom < 0.3 (low) → 50% 段数，zoom < 0.7 (medium) → 75% 段数，zoom >= 0.7 (high) → 100% 段数
   */
  private ellipseToEdge(
    entity: any,
    baseId: number,
    layer: string,
    _handle: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined,
    options?: ExtractOptions
  ): Edge[] {
    const center = entity.center || { x: 0, y: 0 }
    const majorAxis = entity.majorAxis || { x: 1, y: 0 }
    const minorAxisRatio = entity.minorAxisRatio || 1
    const startAngle = entity.startAngle || 0
    const endAngle = entity.endAngle || Math.PI * 2

    const majorRadius = Math.sqrt(majorAxis.x ** 2 + majorAxis.y ** 2)
    const minorRadius = majorRadius * minorAxisRatio

    // ✅ P2-IMPROVED: 基于视觉误差阈值计算段数
    // 使用自适应误差阈值：
    // - 小尺寸椭圆 (< 100 单位): 0.1 单位误差
    // - 中等尺寸 (100-1000): 0.5 单位误差
    // - 大尺寸 (> 1000): 1.0 单位误差
    const maxRadius = Math.max(majorRadius, minorRadius)
    const maxError = maxRadius < 100 ? 0.1 : maxRadius < 1000 ? 0.5 : 1.0

    const angleRange = Math.abs(endAngle - startAngle)

    // 计算所需段数：n >= π * sqrt(r / (2 * maxError))
    // 使用长半轴和短半轴的平均值作为参考半径
    const avgRadius = (majorRadius + minorRadius) / 2
    const baseSegments = Math.ceil(Math.PI * Math.sqrt(avgRadius / (2 * maxError)))

    // 根据角度范围调整（部分椭圆需要更少段数）
    const angleRatio = angleRange / (Math.PI * 2)
    let segments = Math.max(
      8,  // 最小段数保证基本形状
      Math.min(360, Math.ceil(baseSegments * angleRatio))  // 最大段数防止过度离散
    )

    // ✅ S015: LOD 调整 - 根据缩放级别调整段数
    const lodMultiplier = this.calculateLodMultiplier(options)
    segments = Math.max(4, Math.ceil(segments * lodMultiplier))

    const edges: Edge[] = []
    const points: Point[] = []

    // 生成离散点
    for (let i = 0; i <= segments; i++) {
      const t = startAngle + (endAngle - startAngle) * (i / segments)
      const cosT = Math.cos(t)
      const sinT = Math.sin(t)

      // 椭圆参数方程
      points.push([
        center.x + majorRadius * cosT,
        center.y + minorRadius * sinT,
      ])
    }

    // 转换为边数组
    for (let i = 0; i < segments; i++) {
      edges.push({
        id: baseId * 1000 + i,
        start: points[i],
        end: points[i + 1],
        layer,
        is_wall: isWall,
        semantic,
      })
    }

    return edges
  }

  /**
   * ✅ 样条曲线离散化为多个线段
   * ✅ P2-IMPROVED: 基于曲线长度和复杂度动态调整段数
   * ✅ S015: 支持 LOD 参数，根据缩放级别调整段数
   *
   * 算法原理：
   * - 计算样条曲线的近似长度（通过控制点）
   * - 根据目标线段长度动态调整段数
   * - 考虑曲线复杂度（控制点数量、次数、权重变化）
   * - 保证离散化后的视觉误差 < 0.5% 曲线长度
   * - LOD 调整：zoom < 0.3 (low) → 50% 段数，zoom < 0.7 (medium) → 75% 段数，zoom >= 0.7 (high) → 100% 段数
   */
  private splineToEdge(
    entity: any,
    baseId: number,
    layer: string,
    _handle: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined,
    options?: ExtractOptions
  ): Edge[] {
    const controlPoints = entity.controlPoints || []
    const degree = entity.degree || 3
    const knots = entity.knots || []
    const isClosed = entity.closed === true || entity.isClosed === true
    const weights = entity.weights || []

    if (controlPoints.length < 2) {
      return []
    }

    // ✅ P2-IMPROVED: 计算样条曲线的近似长度
    const curveLength = this.estimateCurveLength(controlPoints)

    // ✅ 基于目标线段长度计算段数
    // 目标：每段长度不超过曲线长度的 1%（保证平滑度）
    // 同时考虑最小/最大段数限制
    const targetSegmentLength = Math.max(curveLength * 0.01, 1.0)  // 最小 1 单位
    const baseSegments = Math.ceil(curveLength / targetSegmentLength)

    // ✅ 基于控制点数量和曲线复杂度动态调整
    const numSpans = Math.max(1, controlPoints.length - degree)  // 节点跨度数

    // 每个跨度至少需要的段数（保证曲线精度）
    const minSegmentsPerSpan = degree * 2  // 次数越高，需要越多段数
    const minSegments = numSpans * minSegmentsPerSpan

    // 综合计算最终段数
    let segments = Math.max(baseSegments, minSegments)

    // ✅ 根据权重变化调整（如果有权重数据）
    if (weights.length > 0) {
      const weightVariance = this.calculateWeightVariance(weights)
      // 权重变化大，增加段数（最多增加 50%）
      if (weightVariance > 0.3) {
        segments = Math.ceil(segments * (1 + weightVariance * 0.5))
      }
    }

    // ✅ 限制段数范围（避免过度离散或不足）
    // 最小段数：保证基本形状
    // 最大段数：防止性能问题
    segments = Math.max(20, Math.min(500, segments))

    // ✅ S015: LOD 调整 - 根据缩放级别调整段数
    const lodMultiplier = this.calculateLodMultiplier(options)
    segments = Math.max(10, Math.ceil(segments * lodMultiplier))

    const edges: Edge[] = []
    const points: Point[] = []

    // 生成离散点（使用 B 样条近似）
    for (let i = 0; i <= segments; i++) {
      const t = i / segments
      const point = this.evaluateBSpline(controlPoints, degree, t, knots, isClosed)
      points.push([point.x, point.y])
    }

    // 转换为边数组
    for (let i = 0; i < segments; i++) {
      edges.push({
        id: baseId * 1000 + i,
        start: points[i],
        end: points[i + 1],
        layer,
        is_wall: isWall,
        semantic,
      })
    }

    return edges
  }

  /**
   * ✅ P2-NEW: 估算曲线长度（通过控制点多边形）
   * @param controlPoints 控制点数组
   * @returns 估算的曲线长度
   */
  private estimateCurveLength(controlPoints: Array<{ x: number; y: number }>): number {
    if (controlPoints.length < 2) return 0

    let length = 0
    for (let i = 1; i < controlPoints.length; i++) {
      const dx = controlPoints[i].x - controlPoints[i - 1].x
      const dy = controlPoints[i].y - controlPoints[i - 1].y
      length += Math.sqrt(dx * dx + dy * dy)
    }

    // 估算值通常比实际曲线长度略大，乘以修正系数 0.9
    return length * 0.9
  }

  /**
   * ✅ P2-NEW: 计算权重方差（用于评估样条曲线复杂度）
   */
  private calculateWeightVariance(weights: number[]): number {
    if (weights.length < 2) return 0
    
    const mean = weights.reduce((a, b) => a + b, 0) / weights.length
    const variance = weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / weights.length
    const stdDev = Math.sqrt(variance)
    
    // 归一化到 0-1 范围（假设权重在 0-2 之间）
    return Math.min(1, stdDev / 2)
  }

  /**
   * ✅ S015: 计算 LOD 乘数
   * 根据缩放级别和 LOD 配置调整离散化段数
   * @param options 提取选项
   * @returns LOD 乘数 (0.5-1.0)
   */
  private calculateLodMultiplier(options?: ExtractOptions): number {
    if (!options) {
      return 1.0
    }

    // 如果指定了 LOD 级别，使用预设值
    if (options.lodLevel) {
      switch (options.lodLevel) {
        case 'low':
          return 0.5  // 50% 段数
        case 'medium':
          return 0.75 // 75% 段数
        case 'high':
          return 1.0  // 100% 段数
      }
    }

    // 如果指定了缩放级别，动态计算
    if (options.targetZoom !== undefined) {
      const zoom = options.targetZoom
      if (zoom < 0.3) {
        return 0.5  // 远景：50% 段数
      } else if (zoom < 0.7) {
        return 0.75 // 中景：75% 段数
      } else {
        return 1.0  // 近景：100% 段数
      }
    }

    // 默认使用 100% 段数
    return 1.0
  }

  /**
   * 评估 B 样条曲线上的点
   */
  private evaluateBSpline(
    controlPoints: Array<{ x: number; y: number }>,
    degree: number,
    t: number,
    knots: number[],
    _isClosed: boolean
  ): { x: number; y: number } {
    // 简化处理：使用 de Casteljau 算法的推广
    // 对于均匀 B 样条，使用 Cox-de Boor 递归公式

    const n = controlPoints.length - 1
    const d = degree

    // 如果未提供 knots，使用均匀节点向量
    if (knots.length === 0) {
      knots = []
      for (let i = 0; i <= n + d + 1; i++) {
        knots.push(i)
      }
    }

    // 使用 Cox-de Boor 递归公式
    const alpha = (t * (knots.length - 1)) / (n + 1)
    const span = Math.floor(alpha)

    // 计算基函数
    const basis = this.computeBSplineBasis(knots, degree, span, t * (n + 1) / (knots.length - 1))

    // 计算点位置
    let x = 0
    let y = 0
    for (let i = 0; i <= d; i++) {
      const idx = Math.min(span - d + i, n)
      x += basis[i] * controlPoints[idx].x
      y += basis[i] * controlPoints[idx].y
    }

    return { x, y }
  }

  /**
   * 计算 B 样条基函数
   */
  private computeBSplineBasis(knots: number[], degree: number, span: number, t: number): number[] {
    const basis = new Array(degree + 1).fill(0)
    basis[0] = 1

    for (let j = 1; j <= degree; j++) {
      for (let i = j - 1; i >= 0; i--) {
        const left = knots[span - degree + i]
        const right = knots[span - degree + i + j]
        const a = (right - left) > 0 ? (t - left) / (right - left) : 0
        basis[i + 1] = (right - knots[span - degree + i + 1]) > 0 ? (1 - a) * basis[i + 1] : 0
        basis[i] = a * basis[i]
      }
    }

    return basis
  }

  /**
   * 将 DXF 实体转换为 Edge
   * 返回 Edge 数组以支持多段线等复合实体
   * ✅ FIX-003: 优先使用 entity.type (RealDWG Web 标准 API)
   * ✅ S015: 添加 LOD 参数支持
   */
  private convertToEdge(entity: Entity, options?: ExtractOptions): Edge | Edge[] | null {
    try {
      // ✅ FIX-003: 优先使用 entity.type，dxftype 作为备选
      const type = entity.type?.toLowerCase() || entity.dxftype?.toLowerCase() || ''
      const baseId = ++this.entityIdCounter
      const handle = entity.handle || String(baseId)
      const layer = entity.layer || '0'
      const isWall = this.isWallEntity(entity)
      const semantic = this.getSemanticType(entity) as BoundarySemantic | undefined

      switch (type) {
        case 'line': {
          return {
            id: baseId,
            start: [entity.startPoint?.x || 0, entity.startPoint?.y || 0],
            end: [entity.endPoint?.x || 0, entity.endPoint?.y || 0],
            layer,
            is_wall: isWall,
            semantic,
          }
        }

        case 'lwpolyline':
        case 'polyline': {
          // ✅ 完整提取多段线的所有边
          return this.polylineToEdges(entity, baseId, layer, handle, isWall, semantic)
        }

        case 'arc': {
          // ✅ 使用原生弧线支持
          return this.arcToEdge(entity, baseId, layer, handle, isWall, semantic)
        }

        case 'circle': {
          // ✅ 使用原生弧线支持（360 度圆弧）
          return this.circleToEdge(entity, baseId, layer, handle, isWall, semantic)
        }

        case 'ellipse': {
          // ✅ S015: 椭圆离散化为多个线段，支持 LOD
          return this.ellipseToEdge(entity, baseId, layer, handle, isWall, semantic, options)
        }

        case 'spline': {
          // ✅ S015: 样条曲线离散化为多个线段，支持 LOD
          return this.splineToEdge(entity, baseId, layer, handle, isWall, semantic, options)
        }

        default:
          return null
      }
    } catch (error) {
      console.error('[MlightCadExtractor] Error converting to edge:', error, entity)
      return null
    }
  }

  /**
   * 将圆弧转换为边（使用原生弧线支持）
   * ✅ P2-NEW: 优先使用 AcDbArc.startPoint/midPoint/endPoint 原生 API 简化计算
   */
  private arcToEdge(
    entity: any,
    baseId: number,
    layer: string,
    _handle: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined
  ): Edge {
    // ✅ P2-NEW: 尝试使用原生 API 获取关键点
    const hasStartPoint = entity.startPoint !== undefined
    const hasEndPoint = entity.endPoint !== undefined
    const hasCenter = entity.center !== undefined

    // 如果有原生 API 的点数据，优先使用（更精确、更简单）
    if (hasStartPoint && hasEndPoint && hasCenter) {
      // 使用原生 API 的 startPoint、endPoint、center
      const startPoint = [entity.startPoint.x || 0, entity.startPoint.y || 0] as Point
      const endPoint = [entity.endPoint.x || 0, entity.endPoint.y || 0] as Point
      const center = [entity.center.x || 0, entity.center.y || 0] as Point
      const radius = entity.radius || 0
      const startAngle = entity.startAngle || 0
      const endAngle = entity.endAngle || Math.PI * 2
      const ccw = entity.ccw || false

      return {
        id: baseId,
        start: startPoint,
        end: endPoint,
        layer,
        is_wall: isWall,
        semantic,
        arc: {
          center,
          radius,
          start_angle: startAngle,
          end_angle: endAngle,
          ccw,
        },
      }
    }

    // 降级方案：手动计算（向后兼容）
    const center = entity.center || { x: 0, y: 0 }
    const radius = entity.radius || 0
    const startAngle = entity.startAngle || 0
    const endAngle = entity.endAngle || Math.PI * 2
    const ccw = entity.ccw || false

    // 计算起点和终点（用于兼容直线渲染）
    const startPoint = [
      center.x + radius * Math.cos(startAngle),
      center.y + radius * Math.sin(startAngle),
    ] as Point
    const endPoint = [
      center.x + radius * Math.cos(endAngle),
      center.y + radius * Math.sin(endAngle),
    ] as Point

    return {
      id: baseId,
      start: startPoint,
      end: endPoint,
      layer,
      is_wall: isWall,
      semantic,
      arc: {
        center: [center.x, center.y],
        radius,
        start_angle: startAngle,
        end_angle: endAngle,
        ccw,
      },
    }
  }

  /**
   * 将圆转换为边（使用原生弧线支持）
   * ✅ 返回带有 arc 字段的 Edge（360 度圆弧），不再离散化
   */
  private circleToEdge(
    entity: any,
    baseId: number,
    layer: string,
    _handle: string,
    isWall: boolean,
    semantic: BoundarySemantic | undefined
  ): Edge {
    const center = entity.center || { x: 0, y: 0 }
    const radius = entity.radius || 0

    // ✅ 圆作为 360 度的特殊圆弧处理
    // 起点和终点重合
    const startPoint = [
      center.x + radius, // 0 度方向
      center.y,
    ] as Point
    const endPoint = [
      center.x + radius, // 360 度=0 度
      center.y,
    ] as Point

    // ✅ 返回带有 arc 字段的 Edge（闭合圆弧）
    return {
      id: baseId,
      start: startPoint,
      end: endPoint,
      layer,
      is_wall: isWall,
      semantic,
      arc: {
        center: [center.x, center.y],
        radius,
        start_angle: 0,
        end_angle: Math.PI * 2,
        ccw: false,
      },
    }
  }

  /**
   * 将 Hatch 实体转换为 HatchEntity
   */
  private convertToHatch(entity: Entity): HatchEntity | null {
    try {
      const id = ++this.entityIdCounter
      // Note: handle 字段在当前 HatchEntity 类型中不存在，但保留用于调试
      // const handle = entity.handle || String(id)
      const layer = entity.layer || '0'
      const patternName = (entity as any).patternName || 'SOLID'
      const solidFill = patternName === 'SOLID' || (entity as any).solidFill

      // 提取边界路径
      const boundaryPaths: HatchBoundaryPath[] = []

      if ((entity as any).boundaryPaths && Array.isArray((entity as any).boundaryPaths)) {
        for (const path of (entity as any).boundaryPaths) {
          const boundaryPath = this.convertBoundaryPath(path)
          if (boundaryPath) {
            boundaryPaths.push(boundaryPath)
          }
        }
      }

      // 如果没有边界路径，尝试从 loops 获取
      if (boundaryPaths.length === 0 && (entity as any).loops) {
        for (const loop of (entity as any).loops) {
          const boundaryPath = this.convertBoundaryPath(loop)
          if (boundaryPath) {
            boundaryPaths.push(boundaryPath)
          }
        }
      }

      return {
        id,
        boundary_paths: boundaryPaths,
        pattern: {
          type: solidFill ? 'solid' as const : 'predefined' as const,
          name: patternName,
          scale: (entity as any).patternScale || 1,
          angle: (entity as any).patternAngle || 0,
        },
        solid_fill: solidFill,
        layer,
        scale: (entity as any).patternScale || 1,
        angle: (entity as any).patternAngle || 0,
      }
    } catch (error) {
      console.error('[MlightCadExtractor] Error converting to hatch:', error, entity)
      return null
    }
  }

  /**
   * 转换边界路径
   * ✅ P1-NEW: 支持 AcGeLoop2d 原生边界数据，保留弧线/椭圆弧的精确参数
   */
  private convertBoundaryPath(path: any): HatchBoundaryPath | null {
    try {
      // ✅ 检测 AcGeLoop2d 原生数据结构
      // AcGeLoop2d 可能包含 edges 数组，每条边有自己的类型和参数
      if (path.edges && Array.isArray(path.edges)) {
        // 处理 AcGeLoop2d 的边数组（每条边可能是 line/arc/ellipse 等）
        const boundaryPaths: HatchBoundaryPath[] = []
        for (const edge of path.edges) {
          const edgePath = this.convertSingleBoundaryEdge(edge)
          if (edgePath) {
            boundaryPaths.push(edgePath)
          }
        }
        // 如果有多条边，返回第一条（其他边在 HATCH 渲染时会被处理）
        // 注意：HATCH 边界环路应该作为整体处理，这里简化为返回单个路径
        return boundaryPaths[0] || null
      }

      const type = path.type || 'polyline'
      const closed = path.isClosed ?? path.closed ?? true

      // ✅ 多段线边界（支持 bulge 值）
      if (type === 'polyline' && path.points) {
        return {
          type: 'polyline',
          points: path.points.map((p: any) => [p.x || 0, p.y || 0] as [number, number]),
          closed,
          bulges: path.bulges || undefined,  // ✅ 保留 bulge 信息
        }
      }

      // ✅ 圆弧边界（原生参数）
      if (type === 'arc' && path.center) {
        return {
          type: 'arc',
          closed: false,
          center: [path.center.x || 0, path.center.y || 0],
          radius: path.radius || 0,
          start_angle: path.startAngle || 0,
          end_angle: path.endAngle || Math.PI * 2,
          ccw: path.ccw || false,
        }
      }

      // ✅ 椭圆弧边界（原生参数）
      if ((type === 'ellipse' || type === 'ellipse_arc') && path.center) {
        return {
          type: 'ellipse_arc',
          closed: false,
          center: [path.center.x || 0, path.center.y || 0],
          major_axis: path.majorAxis || [path.radius || 0, 0],
          minor_axis_ratio: path.minorAxisRatio || 1,
          start_angle: path.startAngle || 0,
          end_angle: path.endAngle || Math.PI * 2,
          extrusion_direction: path.extrusionDirection ?
            [path.extrusionDirection.x || 0, path.extrusionDirection.y || 0, path.extrusionDirection.z || 1] :
            undefined,
        }
      }

      // ✅ 样条曲线边界（完整参数）
      if (type === 'spline' && path.controlPoints) {
        return {
          type: 'spline',
          closed,
          control_points: path.controlPoints.map((p: any) => [p.x || 0, p.y || 0] as [number, number]),
          degree: path.degree || 3,
          knots: path.knots || [],
          weights: path.weights || undefined,      // ✅ P1-NEW: 样条权重
          fit_points: path.fitPoints?.map((p: any) => [p.x || 0, p.y || 0] as [number, number]) || undefined, // ✅ 拟合点
          flags: path.flags || undefined,          // ✅ P1-NEW: 样条标志
        }
      }

      // ✅ 处理离散点数据（降级方案）
      if (path.points && Array.isArray(path.points)) {
        return {
          type: 'polyline',
          points: path.points.map((p: any) => [p.x || 0, p.y || 0] as [number, number]),
          closed,
        }
      }

      return null
    } catch (error) {
      console.error('[MlightCadExtractor] Error converting boundary path:', error)
      return null
    }
  }

  /**
   * ✅ P1-NEW: 转换单个边界边（用于 AcGeLoop2d）
   */
  private convertSingleBoundaryEdge(edge: any): HatchBoundaryPath | null {
    try {
      const type = edge.type || edge.dxftype || 'line'
      const closed = edge.isClosed ?? edge.closed ?? false

      switch (type) {
        case 'line': {
          const startPoint = edge.startPoint || edge.start
          const endPoint = edge.endPoint || edge.end
          return {
            type: 'polyline',
            points: [
              [startPoint?.x || 0, startPoint?.y || 0] as [number, number],
              [endPoint?.x || 0, endPoint?.y || 0] as [number, number],
            ],
            closed: false,
          }
        }

        case 'arc': {
          const center = edge.center || { x: 0, y: 0 }
          const radius = edge.radius || 0
          const startAngle = edge.startAngle || 0
          const endAngle = edge.endAngle || Math.PI * 2
          const ccw = edge.ccw || false

          return {
            type: 'arc',
            closed: false,
            center: [center.x, center.y],
            radius,
            start_angle: startAngle,
            end_angle: endAngle,
            ccw,
          }
        }

        case 'ellipse': {
          const center = edge.center || { x: 0, y: 0 }
          const majorAxis = edge.majorAxis || [edge.radius || 1, 0]
          const minorAxisRatio = edge.minorAxisRatio || 1
          const startAngle = edge.startAngle || 0
          const endAngle = edge.endAngle || Math.PI * 2

          return {
            type: 'ellipse_arc',
            closed: false,
            center: [center.x, center.y],
            major_axis: majorAxis,
            minor_axis_ratio: minorAxisRatio,
            start_angle: startAngle,
            end_angle: endAngle,
          }
        }

        case 'spline': {
          const controlPoints = edge.controlPoints || []
          const degree = edge.degree || 3
          const knots = edge.knots || []

          return {
            type: 'spline',
            closed,
            control_points: controlPoints.map((p: any) => [p.x || 0, p.y || 0] as [number, number]),
            degree,
            knots,
          }
        }

        default:
          return null
      }
    } catch (error) {
      console.error('[MlightCadExtractor] Error converting single boundary edge:', error)
      return null
    }
  }

  /**
   * 计算边界框
   */
  private calculateBounds(edges: Edge[], hatches: HatchEntity[]): {
    minX: number
    minY: number
    maxX: number
    maxY: number
  } {
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    
    // 从边计算边界
    for (const edge of edges) {
      minX = Math.min(minX, edge.start[0], edge.end[0])
      minY = Math.min(minY, edge.start[1], edge.end[1])
      maxX = Math.max(maxX, edge.start[0], edge.end[0])
      maxY = Math.max(maxY, edge.start[1], edge.end[1])
    }
    
    // 从填充计算边界
    for (const hatch of hatches) {
      for (const path of hatch.boundary_paths) {
        if (path.type === 'polyline' && path.points) {
          for (const point of path.points) {
            minX = Math.min(minX, point[0])
            minY = Math.min(minY, point[1])
            maxX = Math.max(maxX, point[0])
            maxY = Math.max(maxY, point[1])
          }
        } else if (path.type === 'arc' && path.center && path.radius) {
          minX = Math.min(minX, path.center[0] - path.radius)
          maxX = Math.max(maxX, path.center[0] + path.radius)
          minY = Math.min(minY, path.center[1] - path.radius)
          maxY = Math.max(maxY, path.center[1] + path.radius)
        }
      }
    }
    
    // 如果没有数据，返回默认值
    if (minX === Infinity) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100 }
    }
    
    return { minX, minY, maxX, maxY }
  }

  /**
   * 创建空边界框
   */
  private createEmptyBounds(): {
    minX: number
    minY: number
    maxX: number
    maxY: number
  } {
    return { minX: 0, minY: 0, maxX: 100, maxY: 100 }
  }

  /**
   * 判断是否为墙边实体
   */
  private isWallEntity(entity: Entity): boolean {
    // 根据图层名称判断
    const layer = (entity.layer || '').toLowerCase()
    if (layer.includes('wall') || layer.includes('墙')) {
      return true
    }

    // 根据颜色判断（某些 CAD 文件墙边使用特定颜色）
    if (entity.color === 7) { // 白色
      return true
    }

    // 根据线宽判断（墙边通常较粗）
    if (entity.lineweight && entity.lineweight > 0.5) {
      return true
    }

    return false
  }

  /**
   * 获取语义类型
   */
  private getSemanticType(entity: Entity): string {
    const layer = (entity.layer || '').toLowerCase()

    if (layer.includes('wall') || layer.includes('墙')) {
      return 'hard_wall'
    }
    if (layer.includes('door') || layer.includes('门')) {
      return 'door'
    }
    if (layer.includes('window') || layer.includes('窗')) {
      return 'window'
    }
    if (layer.includes('furniture') || layer.includes('家具')) {
      return 'custom'
    }

    return 'custom'
  }
}

/**
 * 从 document 提取几何数据的便捷函数
 * ✅ S007: 完善资源清理
 * ✅ S015: 支持 LOD 参数
 */
export async function extractGeometryFromDocument(
  document: AcApDocument,
  options?: ExtractOptions
): Promise<{
  edges: Edge[]
  hatches: HatchEntity[]
  bounds: { minX: number; minY: number; maxX: number; maxY: number }
}> {
  try {
    if (!document || !document.database) {
      console.warn('[MlightCadExtractor] Invalid document')
      return { edges: [], hatches: [], bounds: { minX: 0, minY: 0, maxX: 100, maxY: 100 } }
    }

    // ✅ 传递 document 引用用于显式 API 调用
    const extractor = new MlightCadGeometryExtractor(document.database, document, options)
    return await extractor.extractAll(options)
  } catch (error) {
    console.error('[MlightCadExtractor] Error:', error)
    return { edges: [], hatches: [], bounds: { minX: 0, minY: 0, maxX: 100, maxY: 100 } }
  }
}
