import { describe, it, expect, beforeEach } from 'vitest'
import { useCanvasStore } from '@/stores/canvas-store'
import type { Edge } from '@/types/api'

describe.skip('CanvasStore', () => {
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
    })
  })

  it('should set edges correctly', () => {
    const mockEdges: Edge[] = [
      {
        id: 1,
        start: [0, 0],
        end: [100, 100],
        is_wall: true,
      },
    ]

    useCanvasStore.getState().setEdges(mockEdges)

    expect(useCanvasStore.getState().edges).toEqual(mockEdges)
  })

  it('should select edge correctly', () => {
    const mockEdges: Edge[] = [
      { id: 1, start: [0, 0], end: [100, 100], is_wall: true },
      { id: 2, start: [100, 100], end: [200, 200], is_wall: false },
    ]

    useCanvasStore.getState().setEdges(mockEdges)
    useCanvasStore.getState().selectEdge(1)

    expect(useCanvasStore.getState().selectedEdgeId).toBe(1)
    expect(useCanvasStore.getState().selectedEdgeIds).toEqual([1])
  })

  it('should handle multi-selection', () => {
    useCanvasStore.getState().selectEdge(1)
    useCanvasStore.getState().selectEdge(2, true)

    expect(useCanvasStore.getState().selectedEdgeIds).toEqual([1, 2])
  })

  it('should update camera zoom', () => {
    useCanvasStore.getState().setCamera({ zoom: 2 })

    expect(useCanvasStore.getState().camera.zoom).toBe(2)
  })

  it('should clear selection', () => {
    useCanvasStore.getState().selectEdge(1)
    useCanvasStore.getState().clearSelection()

    expect(useCanvasStore.getState().selectedEdgeId).toBeNull()
    expect(useCanvasStore.getState().selectedEdgeIds).toEqual([])
  })

  it('should change active tool', () => {
    useCanvasStore.getState().setTool('trace')

    expect(useCanvasStore.getState().activeTool).toBe('trace')
  })
})
