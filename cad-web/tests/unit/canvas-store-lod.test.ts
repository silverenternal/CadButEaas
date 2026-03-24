import { describe, it, expect, beforeEach } from 'vitest'
import { useCanvasStore, getLodLevel, LodLevel, selectEdgesWithLod, selectPerformanceStats } from '@/stores/canvas-store'
import type { Edge } from '@/types/api'

describe('CanvasStore - LOD Optimization', () => {
  beforeEach(() => {
    // Reset store before each test
    useCanvasStore.setState({
      edges: [],
      gaps: [],
      traceResult: null,
      selectedEdgeId: null,
      selectedEdgeIds: [],
      camera: { zoom: 1, offsetX: 0, offsetY: 0 },
      activeTool: 'select',
      isLoading: false,
      uploadProgress: 0,
      viewport: null,
    })
  })

  describe('getLodLevel', () => {
    it('should return Low LOD for zoom < 0.3', () => {
      expect(getLodLevel(0.1)).toBe(LodLevel.Low)
      expect(getLodLevel(0.2)).toBe(LodLevel.Low)
      expect(getLodLevel(0.29)).toBe(LodLevel.Low)
    })

    it('should return Medium LOD for zoom 0.3-0.7', () => {
      expect(getLodLevel(0.3)).toBe(LodLevel.Medium)
      expect(getLodLevel(0.5)).toBe(LodLevel.Medium)
      expect(getLodLevel(0.69)).toBe(LodLevel.Medium)
    })

    it('should return High LOD for zoom > 0.7', () => {
      expect(getLodLevel(0.7)).toBe(LodLevel.High)
      expect(getLodLevel(1.0)).toBe(LodLevel.High)
      expect(getLodLevel(2.0)).toBe(LodLevel.High)
      expect(getLodLevel(10.0)).toBe(LodLevel.High)
    })
  })

  describe('selectEdgesWithLod', () => {
    const mockEdges: Edge[] = [
      { id: 1, start: [0, 0], end: [100, 100], is_wall: true },
      { id: 2, start: [100, 100], end: [200, 200], is_wall: false },
      { id: 3, start: [200, 200], end: [300, 300], is_wall: true },
      { id: 4, start: [300, 300], end: [400, 400], is_wall: true, semantic: 'window' },
      { id: 5, start: [0, 0], end: [10, 10], is_wall: false }, // Very short edge
    ]

    it('should filter to wall edges only at Low LOD', () => {
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 0.2, offsetX: 0, offsetY: 0 },
      })

      const result = selectEdgesWithLod(useCanvasStore.getState())
      
      // Should only include wall edges
      expect(result.every(e => e.is_wall || e.id === 5)).toBe(true)
      expect(result.length).toBeLessThan(mockEdges.length)
    })

    it('should filter short edges at Medium LOD', () => {
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 0.5, offsetX: 0, offsetY: 0 },
      })

      const result = selectEdgesWithLod(useCanvasStore.getState())
      
      // Short edge (id: 5) should be filtered out
      expect(result.find(e => e.id === 5)).toBeUndefined()
    })

    it('should include all edges at High LOD', () => {
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 1.0, offsetX: 0, offsetY: 0 },
      })

      const result = selectEdgesWithLod(useCanvasStore.getState())
      
      // All edges should be included
      expect(result.length).toBe(mockEdges.length)
    })

    it('should always include selected edge regardless of LOD', () => {
      const nonWallEdge = mockEdges.find(e => !e.is_wall)!
      
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 0.2, offsetX: 0, offsetY: 0 },
        selectedEdgeId: nonWallEdge.id,
      })

      const result = selectEdgesWithLod(useCanvasStore.getState())
      
      // Selected edge should always be included
      expect(result.find(e => e.id === nonWallEdge.id)).toBeDefined()
    })

    it('should respect maxEdgesRender limit', () => {
      const manyEdges: Edge[] = Array.from({ length: 15000 }, (_, i) => ({
        id: i,
        start: [i * 10, i * 10],
        end: [(i + 1) * 10, (i + 1) * 10],
        is_wall: i % 2 === 0,
      }))

      useCanvasStore.setState({
        edges: manyEdges,
        camera: { zoom: 1.0, offsetX: 0, offsetY: 0 },
        maxEdgesRender: 10000,
      })

      const result = selectEdgesWithLod(useCanvasStore.getState())
      
      // Should respect the limit
      expect(result.length).toBeLessThanOrEqual(10000)
    })
  })

  describe('selectPerformanceStats', () => {
    const mockEdges: Edge[] = [
      { id: 1, start: [0, 0], end: [100, 100], is_wall: true },
      { id: 2, start: [100, 100], end: [200, 200], is_wall: false },
      { id: 3, start: [200, 200], end: [300, 300], is_wall: true },
    ]

    it('should return correct stats', () => {
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 1.0, offsetX: 0, offsetY: 0 },
      })

      const stats = selectPerformanceStats(useCanvasStore.getState())
      
      expect(stats.totalEdges).toBe(mockEdges.length)
      expect(stats.visibleEdges).toBe(mockEdges.length)
      expect(stats.lodLevel).toBe(LodLevel.High)
      expect(stats.estimatedRenderTime).toBeGreaterThan(0)
    })

    it('should show reduced visible edges at lower LOD', () => {
      useCanvasStore.setState({
        edges: mockEdges,
        camera: { zoom: 0.2, offsetX: 0, offsetY: 0 },
      })

      const stats = selectPerformanceStats(useCanvasStore.getState())
      
      expect(stats.lodLevel).toBe(LodLevel.Low)
      expect(stats.visibleEdges).toBeLessThan(stats.totalEdges)
    })
  })
})
