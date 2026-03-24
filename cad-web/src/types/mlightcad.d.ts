/**
 * mlightcad 类型定义
 * 为 @mlightcad/cad-simple-viewer 提供 TypeScript 类型支持
 */

import type { Point } from '@/types/api'

// ========== 基础类型 ==========

export interface Point3D {
  x: number
  y: number
  z: number
}

export interface Vector3D {
  x: number
  y: number
  z: number
}

// ========== 数据库类型 ==========

export interface Database {
  getModelSpace(): Promise<BlockTableRecord | null>
  getBlockTableRecord(): Promise<BlockTableRecord | null>
  tables?: {
    blockTable?: {
      modelSpace?: BlockTableRecord
    }
  }
  blockTableRecord?: BlockTableRecord
  entities?: Entity[]
  objects?: Record<string, any>
  [key: string]: any
}

// ========== 文档类型 ==========

export interface AcApDocument {
  database: Database | null
  openDocument(name: string, data: ArrayBuffer, options: OpenDocumentOptions): Promise<boolean>
  getModelSpace?(): Promise<BlockTableRecord | null>
  dispose?(): Promise<void>
}

export interface OpenDocumentOptions {
  mode: AcEdOpenMode
}

export enum AcEdOpenMode {
  Read = 0,
  Write = 1,
}

// ========== 模型空间类型 ==========

export interface BlockTableRecord {
  name?: string
  handle?: string
  entities?: Entity[] | Record<string, Entity>
  objects?: Record<string, Entity>
  newIterator?(): EntityIterator | Entity[]
  forEach?(callback: (entity: Entity) => void): void
  [Symbol.iterator]?(): Iterator<Entity>
  [key: string]: any
}

export interface EntityIterator {
  [Symbol.iterator](): Iterator<Entity>
  next(): IteratorResult<Entity>
}

// ========== 实体类型 ==========

export interface Entity {
  handle?: string
  layer?: string
  type?: string
  dxftype?: string
  color?: number
  lineweight?: number
  [key: string]: any
}

// 线实体
export interface LineEntity extends Entity {
  type: 'line'
  startPoint: Point3D
  endPoint: Point3D
}

// 多段线实体
export interface PolylineEntity extends Entity {
  type: 'polyline' | 'lwpolyline'
  vertices: Vertex[]
  bulges?: number[]
  closed?: boolean
  isClosed?: boolean
}

export interface Vertex {
  x: number
  y: number
  z?: number
}

// 圆弧实体
export interface ArcEntity extends Entity {
  type: 'arc'
  center: Point3D
  radius: number
  startAngle: number
  endAngle: number
  ccw?: boolean
  startPoint?: Point3D
  endPoint?: Point3D
  midPoint?: Point3D
}

// 圆实体
export interface CircleEntity extends Entity {
  type: 'circle'
  center: Point3D
  radius: number
}

// 椭圆实体
export interface EllipseEntity extends Entity {
  type: 'ellipse'
  center: Point3D
  majorAxis: Vector3D
  minorAxisRatio: number
  startAngle: number
  endAngle: number
  extrusionDirection?: Vector3D
}

// 样条曲线实体
export interface SplineEntity extends Entity {
  type: 'spline'
  controlPoints: Point3D[]
  degree: number
  knots: number[]
  weights?: number[]
  fitPoints?: Point3D[]
  flags?: number
  closed?: boolean
  isClosed?: boolean
}

// 填充实体
export interface HatchEntity extends Entity {
  type: 'hatch'
  patternName: string
  solidFill?: boolean
  patternScale?: number
  patternAngle?: number
  boundaryPaths?: BoundaryPath[]
  loops?: Loop[]
}

export interface BoundaryPath {
  type: string
  points?: Point3D[]
  bulges?: number[]
  closed?: boolean
  isClosed?: boolean
  center?: Point3D
  radius?: number
  startAngle?: number
  endAngle?: number
  ccw?: boolean
  majorAxis?: Vector3D
  minorAxisRatio?: number
  controlPoints?: Point3D[]
  degree?: number
  knots?: number[]
  weights?: number[]
  fitPoints?: Point3D[]
  flags?: number
  edges?: BoundaryEdge[]
}

export interface BoundaryEdge {
  type: 'line' | 'arc' | 'ellipse' | 'spline'
  [key: string]: any
}

export interface Loop {
  type: string
  [key: string]: any
}

// ========== mlightcad 模块声明 ==========

declare module '@mlightcad/cad-simple-viewer' {
  export { AcApDocument }
  export { AcEdOpenMode }
  export type { Database, BlockTableRecord, Entity }
}
