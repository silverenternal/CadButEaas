import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

interface Layer {
  id: string
  name: string
  visible: boolean
  locked: boolean
  count: number
  color: string
}

interface LayerState {
  layers: Layer[]
  selectedLayer: string | null

  setLayers: (layers: Layer[]) => void
  toggleVisibility: (layerId: string) => void
  toggleLock: (layerId: string) => void
  setSelectedLayer: (layerId: string | null) => void
  addLayer: (layer: Layer) => void
  removeLayer: (layerId: string) => void
}

export const useLayerStore = create<LayerState>()(
  subscribeWithSelector((set) => ({
    layers: [],
    selectedLayer: null,

    setLayers: (layers: Layer[]) => set({ layers }),

    toggleVisibility: (layerId: string) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.id === layerId
            ? { ...layer, visible: !layer.visible }
            : layer
        ),
      })),

    toggleLock: (layerId: string) =>
      set((state) => ({
        layers: state.layers.map((layer) =>
          layer.id === layerId
            ? { ...layer, locked: !layer.locked }
            : layer
        ),
      })),

    setSelectedLayer: (layerId: string | null) => set({ selectedLayer: layerId }),

    addLayer: (layer: Layer) =>
      set((state) => ({ layers: [...state.layers, layer] })),

    removeLayer: (layerId: string) =>
      set((state) => ({
        layers: state.layers.filter((layer) => layer.id !== layerId),
      })),
  }))
)
