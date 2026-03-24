import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type { Edge, BoundarySemantic } from '@/types/api'

interface SelectionState {
  selectedEdge: Edge | null
  hoverEdgeId: number | null

  actions: {
    setSelectedEdge: (edge: Edge | null) => void
    setHoverEdgeId: (edgeId: number | null) => void
    updateEdgeSemantic: (
      edgeId: number,
      semantic: BoundarySemantic
    ) => void
  }
}

export const useSelectionStore = create<SelectionState>()(
  subscribeWithSelector((set) => ({
    selectedEdge: null,
    hoverEdgeId: null,

    actions: {
      setSelectedEdge: (edge: Edge | null) => set({ selectedEdge: edge }),

      setHoverEdgeId: (edgeId: number | null) => set({ hoverEdgeId: edgeId }),

      updateEdgeSemantic: (edgeId: number, semantic: BoundarySemantic) =>
        set((state) => ({
          selectedEdge:
            state.selectedEdge?.id === edgeId
              ? { ...state.selectedEdge, semantic }
              : state.selectedEdge,
        })),
    },
  }))
)
