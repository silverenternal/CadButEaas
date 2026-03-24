import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './assets/styles/index.css'

/**
 * ✅ S008: WASM 模块预加载
 * 在应用启动时预加载 mlightcad WASM 模块，避免首次上传时等待
 */
async function preloadWasmModule(): Promise<void> {
  try {
    console.log('[Main] Preloading mlightcad WASM module...')
    const module = await import('@mlightcad/cad-simple-viewer')
    
    // 将模块挂载到 window 上供 Worker 使用
    if (typeof window !== 'undefined') {
      (window as any).__MLIGHTCAD__ = module
      console.log('[Main] mlightcad WASM module preloaded successfully')
    }
  } catch (error) {
    console.error('[Main] Failed to preload mlightcad WASM module:', error)
  }
}

// 预加载 WASM 模块（不阻塞渲染）
preloadWasmModule()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
