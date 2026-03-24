import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

interface FileMetadata {
  name: string
  path: string
  size: number
  lastModified: number
}

type ToolType =
  | 'select'
  | 'trace'
  | 'lasso'
  | 'pan'
  | 'zoom'
  | 'measure'
  | 'annotate'

interface AppSettings {
  theme: 'light' | 'dark' | 'system'
  language: 'zh' | 'en'
  autoSave: boolean
  showGrid: boolean
  snapToGrid: boolean
  gridSize: number
}

interface AppState {
  // 文件状态
  currentFile: FileMetadata | null
  recentFiles: FileMetadata[]

  // 工具状态
  activeTool: ToolType
  toolHistory: ToolType[]

  // 用户设置
  settings: AppSettings

  // UI 状态
  sidebarOpen: boolean
  rightPanelOpen: boolean

  // Actions
  setCurrentFile: (file: FileMetadata | null) => void
  addRecentFile: (file: FileMetadata) => void
  setTool: (tool: ToolType) => void
  undo: () => void
  redo: () => void
  updateSettings: (settings: Partial<AppSettings>) => void
  toggleSidebar: () => void
  toggleRightPanel: () => void
}

const defaultSettings: AppSettings = {
  theme: 'dark',
  language: 'zh',
  autoSave: true,
  showGrid: true,
  snapToGrid: false,
  gridSize: 10,
}

export const useAppStore = create<AppState>()(
  subscribeWithSelector((set) => ({
    currentFile: null,
    recentFiles: [],
    activeTool: 'select',
    toolHistory: [],
    settings: defaultSettings,
    sidebarOpen: true,
    rightPanelOpen: true,

    setCurrentFile: (file: FileMetadata | null) => set({ currentFile: file }),

    addRecentFile: (file: FileMetadata) =>
      set((state) => {
        const filtered = state.recentFiles.filter((f) => f.path !== file.path)
        return { recentFiles: [file, ...filtered].slice(0, 10) }
      }),

    setTool: (tool: ToolType) =>
      set((state) => ({
        activeTool: tool,
        toolHistory: [...state.toolHistory, tool].slice(-10),
      })),

    undo: () => {
      // TODO: 实现撤销功能
      console.log('Undo')
    },

    redo: () => {
      // TODO: 实现重做功能
      console.log('Redo')
    },

    updateSettings: (settings: Partial<AppSettings>) =>
      set((state) => ({
        settings: { ...state.settings, ...settings },
      })),

    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

    toggleRightPanel: () =>
      set((state) => ({ rightPanelOpen: !state.rightPanelOpen })),
  }))
)
