import { useState, useCallback } from 'react'
import { UnifiedCanvasViewer } from '@features/canvas/components/unified-canvas-viewer'
import { CanvasToolbar } from '@features/canvas/components/canvas-toolbar'
import { CanvasErrorBoundary } from '@features/canvas/components/canvas-error-boundary'
import { ZoomControls } from '@features/canvas/components/zoom-controls'
import { GridOverlay } from '@features/canvas/components/grid-overlay'
import { ViewCube } from '@features/canvas/components/view-cube'
import { LoadingDialog } from '@features/canvas/components/loading-dialog'
import { ErrorDialog } from '@features/canvas/components/error-dialog'
import { SuccessToast } from '@features/canvas/components/success-toast'
import { WarningBanner } from '@features/canvas/components/warning-banner'
import { FileUploadZone } from '@features/canvas/components/file-upload-zone'
import { FilePreviewCard } from '@features/canvas/components/file-preview-card'
import { RecentFilesList, addRecentFile } from '@features/canvas/components/recent-files-list'
import { SampleFilesGrid } from '@features/canvas/components/sample-files-grid'
import { useCanvasStore } from '@/stores/canvas-store'
import { useFileUpload } from '@/hooks/use-file-upload'
import { cn } from '@/lib/utils'

export function CanvasPage() {
  const { edges, isLoading, uploadProgress } = useCanvasStore()
  const [showUploadZone, setShowUploadZone] = useState(false)
  const [previewFile, setPreviewFile] = useState<File | null>(null)
  const [showError, setShowError] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)
  const [showWarning, setShowWarning] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  const { uploadFile, cancel, progress, isUploading } = useFileUpload({
    useFrontendParse: true,
    onSuccess: () => {
      setShowSuccess(true)
      setShowUploadZone(false)
      setPreviewFile(null)
    },
    onError: (error: Error) => {
      setErrorMessage(error.message)
      setShowError(true)
    },
  })

  const handleFileSelect = useCallback((file: File) => {
    // 添加到最近文件
    addRecentFile(file)
    // 显示预览
    setPreviewFile(file)
  }, [])

  const handleConfirmUpload = useCallback(async () => {
    if (!previewFile) return
    try {
      await uploadFile(previewFile)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '文件处理失败')
      setShowError(true)
    }
  }, [previewFile, uploadFile])

  const handleCancelUpload = useCallback(() => {
    // 取消 Worker 中的解析任务
    cancel?.()
    setPreviewFile(null)
  }, [cancel])

  const handleSampleFileSelect = useCallback(async () => {
    // TODO: 实际项目中应该从服务器加载示例文件
    setShowWarning(true)
  }, [])

  const handleRecentFileSelect = useCallback(async () => {
    // TODO: 从存储中加载文件
    setShowWarning(true)
  }, [])

  return (
    <div className="w-full h-full relative">
      {/* 工具栏 */}
      <CanvasToolbar />

      {/* 右下角缩放控制 */}
      {edges.length > 0 && <ZoomControls className="absolute bottom-4 right-4" />}

      {/* 网格覆盖层 */}
      <GridOverlay />

      {/* 视图方向指示器 */}
      {edges.length > 0 && <ViewCube className="absolute bottom-4 left-4" />}

      <CanvasErrorBoundary>
        <UnifiedCanvasViewer />
      </CanvasErrorBoundary>

      {/* 空状态 - 文件上传区域 */}
      {edges.length === 0 && !isLoading && !previewFile && (
        <div className={cn(
          'absolute inset-0 z-[100] flex items-center justify-center p-4',
          showUploadZone ? 'fade-in' : 'fade-in'
        )}>
          <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
            <div className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-br from-primary/5 to-purple-500/5 blur-3xl" />
            <div className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-gradient-to-tr from-blue-500/5 to-cyan-500/5 blur-3xl" />
          </div>

          <div className="relative z-10 w-full max-w-4xl mx-auto space-y-8">
            {/* 主标题区域 */}
            <div className="text-center space-y-4">
              <h1 className="text-4xl font-bold">CAD 图纸查看器</h1>
              <p className="text-muted-foreground text-lg">
                支持 DXF、DWG 格式文件的在线查看和解析
              </p>
            </div>

            {/* 上传区域和示例文件并排显示 */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* 左侧：文件上传 */}
              <div className="space-y-4">
                <FileUploadZone
                  onFileSelect={handleFileSelect}
                  onCancel={handleCancelUpload}
                  progress={progress}
                  isUploading={isUploading}
                  maxSize={50}
                />
              </div>

              {/* 右侧：示例文件 */}
              <div className="space-y-4">
                <SampleFilesGrid
                  onFileSelect={handleSampleFileSelect}
                />
              </div>
            </div>

            {/* 最近文件 */}
            <RecentFilesList
              onFileSelect={handleRecentFileSelect}
              maxFiles={5}
            />
          </div>
        </div>
      )}

      {/* 文件预览卡片 */}
      {previewFile && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center p-4">
          {/* 背景遮罩 */}
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={handleCancelUpload}
          />
          
          <div className="relative z-10 w-full max-w-lg">
            <FilePreviewCard
              file={previewFile}
              onConfirm={handleConfirmUpload}
              onCancel={handleCancelUpload}
            />
          </div>
        </div>
      )}

      {/* 加载状态对话框 */}
      <LoadingDialog
        isOpen={isLoading}
        fileName={previewFile?.name || '文件'}
        fileSize={previewFile?.size || 0}
        currentStep={uploadProgress < 30 ? 'upload' : uploadProgress < 60 ? 'parse' : uploadProgress < 80 ? 'extract' : 'render'}
        progress={uploadProgress}
        estimatedTimeRemaining={isLoading ? Math.max(1, Math.round((100 - uploadProgress) / 10)) : undefined}
        showLogs={(import.meta as any).env.DEV}
        logs={[]}
      />

      {/* 错误对话框 */}
      <ErrorDialog
        isOpen={showError}
        onClose={() => setShowError(false)}
        message={errorMessage}
        type="error"
        suggestions={[
          '检查文件格式是否正确',
          '确认文件未损坏',
          '尝试重新导出 DXF/DWG 文件',
          '联系技术支持获取帮助',
        ]}
        actions={[
          {
            label: '重试',
            onClick: () => {
              setShowError(false)
              if (previewFile) {
                uploadFile(previewFile)
              }
            },
          },
          {
            label: '取消',
            onClick: () => setShowError(false),
            variant: 'outline',
          },
        ]}
      />

      {/* 成功提示 */}
      <SuccessToast
        isOpen={showSuccess}
        onClose={() => setShowSuccess(false)}
        message="文件解析成功"
        description="几何数据已加载到画布"
      />

      {/* 警告横幅 */}
      <WarningBanner
        isOpen={showWarning}
        onClose={() => setShowWarning(false)}
        message="示例文件功能开发中"
        description="请稍后从服务器加载示例文件，或上传您自己的 DXF/DWG 文件"
        actionLabel="上传文件"
        onAction={() => setShowWarning(false)}
      />
    </div>
  )
}
