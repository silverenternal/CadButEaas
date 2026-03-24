/**
 * @mlightcad/cad-simple-viewer 类型定义
 */

export interface MlightCadViewer {
  new (config: MlightCadConfig): MlightCadViewer
  
  loadFromFile(file: File): Promise<void>
  loadFromUrl(url: string): Promise<void>
  dispose(): void
  
  getSceneData(): MlightCadSceneData
  setCamera(camera: MlightCadCamera): void
  fitToContent(padding?: number): void
  resetCamera(): void
  
  on(event: 'cameraChange', callback: (camera: any) => void): void
  on(event: 'entityClick', callback: (entity: any) => void): void
  on(event: 'load', callback: (data: any) => void): void
  on(event: 'error', callback: (error: Error) => void): void
  
  off(event: string, callback: Function): void
}

export interface MlightCadConfig {
  container: HTMLElement
  options?: MlightCadOptions
}

export interface MlightCadOptions {
  showGrid?: boolean
  showAxes?: boolean
  backgroundColor?: string
  enablePan?: boolean
  enableZoom?: boolean
  enableRotate?: boolean
}

export interface MlightCadSceneData {
  entities: MlightCadEntity[]
  layers: string[]
  bounds: MlightCadBounds
}

export interface MlightCadEntity {
  handle: string
  type: string
  layer: string
  geometry: any
  properties: any
}

export interface MlightCadBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export interface MlightCadCamera {
  zoom: number
  offset: [number, number]
}

export interface MlightCadHatch {
  handle: string
  layer: string
  patternName: string
  solidFill: boolean
  patternScale: number
  patternAngle: number
  boundaryPaths: MlightCadBoundaryPath[]
}

export interface MlightCadBoundaryPath {
  type: number
  isClosed: boolean
  edges: MlightCadBoundaryEdge[]
}

export interface MlightCadBoundaryEdge {
  type: number
  start?: { x: number; y: number }
  end?: { x: number; y: number }
  center?: { x: number; y: number }
  radius?: number
  startAngle?: number
  endAngle?: number
  ccw?: boolean
}

export type MlightCadEventType = 
  | 'cameraChange'
  | 'entityClick'
  | 'load'
  | 'error'
  | 'progress'
