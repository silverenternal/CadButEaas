/**
 * 测试页面：探索 @mlightcad/cad-simple-viewer 的 API
 * 用于研究如何从 AcApDocument.database 提取几何数据
 */
import { useState, useCallback } from 'react'
import { CadViewer } from '@/components/cad-viewer'
import type { CadViewerRef } from '@/components/cad-viewer'

export function ExploreMlightCadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [sceneData, setSceneData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [viewerRef, setViewerRef] = useState<CadViewerRef | null>(null)

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setSceneData(null)
      setError(null)
    }
  }, [])

  const handleLoaded = useCallback((data: any) => {
    console.log('[ExploreMlightCad] Loaded data:', data)
    setSceneData(data)
  }, [])

  const handleError = useCallback((err: Error) => {
    console.error('[ExploreMlightCad] Error:', err)
    setError(err.message)
  }, [])

  const handleRef = useCallback((ref: CadViewerRef | null) => {
    setViewerRef(ref)
  }, [])

  const handleExploreDatabase = useCallback(async () => {
    if (!viewerRef) return

    try {
      const viewer = viewerRef.getViewer()
      if (!viewer) {
        setError('Viewer not initialized')
        return
      }

      // 探索 viewer 的内部结构
      const anyViewer = viewer as any
      console.log('[ExploreMlightCad] Viewer instance:', viewer)
      console.log('[ExploreMlightCad] Viewer keys:', Object.keys(viewer))
      
      // 尝试访问 internalScene 和 internalCamera
      console.log('[ExploreMlightCad] internalScene:', anyViewer.internalScene)
      console.log('[ExploreMlightCad] internalCamera:', anyViewer.internalCamera)
      
      // 尝试访问 document
      console.log('[ExploreMlightCad] Viewer document:', anyViewer.document)
      
      // 如果有 document，探索 database
      if (anyViewer.document) {
        const doc = anyViewer.document
        console.log('[ExploreMlightCad] Document keys:', Object.keys(doc))
        console.log('[ExploreMlightCad] Document database:', doc.database)
        
        if (doc.database) {
          const db = doc.database
          console.log('[ExploreMlightCad] Database keys:', Object.keys(db))
          
          // 尝试获取 entities
          console.log('[ExploreMlightCad] Trying to get entities...')
          
          // 常见的 RealDWG API 方法
          const methods = [
            'getEntities',
            'entities',
            'getAllEntities',
            'extractEntities',
            'getSceneData',
            'getGeometry',
          ]
          
          for (const method of methods) {
            if (typeof db[method] === 'function') {
              console.log(`[ExploreMlightCad] Found method: ${method}`)
              try {
                const result = db[method]()
                console.log(`[ExploreMlightCad] ${method} result:`, result)
              } catch (e) {
                console.log(`[ExploreMlightCad] ${method} error:`, e)
              }
            }
          }
          
          // 尝试访问 blockTableRecord
          if (db.blockTableRecord) {
            console.log('[ExploreMlightCad] blockTableRecord:', db.blockTableRecord)
          }
        }
      }
      
      // 尝试从 scene 获取数据
      if (anyViewer.internalScene) {
        const scene = anyViewer.internalScene
        console.log('[ExploreMlightCad] Scene keys:', Object.keys(scene))
        console.log('[ExploreMlightCad] Scene children:', scene.children)
        
        if (scene.children) {
          scene.children.forEach((child: any, index: number) => {
            console.log(`[ExploreMlightCad] Child ${index}:`, {
              type: child.type,
              uuid: child.uuid,
              geometry: child.geometry,
              material: child.material,
            })
          })
        }
      }
      
      // 尝试使用 getSceneData 方法
      if (typeof anyViewer.getSceneData === 'function') {
        const data = anyViewer.getSceneData()
        console.log('[ExploreMlightCad] getSceneData():', data)
      }
      
    } catch (e) {
      const err = e as Error
      console.error('[ExploreMlightCad] Explore error:', err)
      setError(err.message)
    }
  }, [viewerRef])

  return (
    <div className="w-full h-full p-4 bg-gray-900 text-white">
      <h1 className="text-2xl font-bold mb-4">探索 mlightcad API</h1>
      
      <div className="mb-4">
        <input
          type="file"
          accept=".dxf,.dwg"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-400
            file:mr-4 file:py-2 file:px-4
            file:rounded file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-500 file:text-white
            hover:file:bg-blue-600"
        />
      </div>
      
      {file && (
        <div className="mb-4">
          <p className="text-sm text-gray-400">
            已选择文件：{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
          </p>
        </div>
      )}
      
      <div className="mb-4 flex gap-2">
        <button
          onClick={handleExploreDatabase}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-semibold"
        >
          探索 Database 结构
        </button>
      </div>
      
      {error && (
        <div className="mb-4 p-4 bg-red-900/50 border border-red-500 rounded">
          <p className="text-red-300">错误：{error}</p>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-4 h-[calc(100%-200px)]">
        <div className="border border-gray-700 rounded p-4 overflow-auto">
          <h2 className="text-lg font-semibold mb-2">CadViewer 组件</h2>
          {file ? (
            <CadViewer
              ref={handleRef}
              file={file}
              options={{
                backgroundColor: '#0f0f23',
                showGrid: true,
                showAxes: true,
              }}
              onLoaded={handleLoaded}
              onError={handleError}
              className="h-[400px] border border-gray-600 rounded"
            />
          ) : (
            <div className="h-[400px] flex items-center justify-center text-gray-500">
              请选择 DXF/DWG 文件
            </div>
          )}
        </div>
        
        <div className="border border-gray-700 rounded p-4 overflow-auto">
          <h2 className="text-lg font-semibold mb-2">场景数据</h2>
          <pre className="text-xs font-mono text-gray-300 overflow-auto">
            {sceneData 
              ? JSON.stringify(sceneData, null, 2) 
              : '加载中...'
            }
          </pre>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-blue-900/30 border border-blue-500 rounded">
        <h3 className="text-sm font-semibold mb-2">使用说明：</h3>
        <ol className="text-xs text-gray-300 list-decimal list-inside space-y-1">
          <li>选择一个 DXF 或 DWG 文件</li>
          <li>点击 "探索 Database 结构" 按钮</li>
          <li>查看浏览器控制台的详细输出</li>
          <li>根据输出结果调整几何数据提取逻辑</li>
        </ol>
      </div>
    </div>
  )
}

export default ExploreMlightCadPage
