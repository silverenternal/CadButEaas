import { useState, useCallback, useRef, useEffect } from 'react'
import { Upload, File, X, CheckCircle, AlertCircle, StopCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface FileUploadZoneProps {
  onFileSelect: (file: File) => void
  onCancel?: () => void
  accept?: string[]
  multiple?: boolean
  maxSize?: number // MB
  progress?: number // 0-100
  isUploading?: boolean
  className?: string
}

export function FileUploadZone({
  onFileSelect,
  onCancel,
  accept = ['.dxf', '.dwg', '.pdf'],
  multiple = false,
  maxSize = 50,
  progress = 0,
  isUploading = false,
  className,
}: FileUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const inputRef = useRef<HTMLInputElement>(null)

  // 同步外部上传状态
  useEffect(() => {
    if (isUploading) {
      setUploadStatus('uploading')
    } else if (uploadStatus === 'uploading' && !isUploading) {
      setUploadStatus('idle')
    }
  }, [isUploading, uploadStatus])

  const validateFile = useCallback((file: File): string | null => {
    // 检查文件类型
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!accept.includes(fileExt)) {
      return `不支持的文件格式 "${fileExt}"，支持 ${accept.join(', ')}`
    }

    // 检查文件大小
    const sizeInMB = file.size / (1024 * 1024)
    if (sizeInMB > maxSize) {
      return `文件大小 ${sizeInMB.toFixed(1)}MB 超过限制 ${maxSize}MB`
    }

    return null
  }, [accept, maxSize])

  const handleFile = useCallback((file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      setUploadStatus('error')
      return
    }

    setError(null)
    setSelectedFile(file)
    setUploadStatus('idle')
    onFileSelect(file)
  }, [validateFile, onFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleClick = useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      handleFile(files[0])
    }
    // 重置 input 以允许重复选择同一文件
    e.target.value = ''
  }, [handleFile])

  const handleCancel = useCallback(() => {
    setSelectedFile(null)
    setError(null)
    setUploadStatus('idle')
    if (inputRef.current) {
      inputRef.current.value = ''
    }
    // 调用外部取消回调
    onCancel?.()
  }, [onCancel])

  const handleRetry = useCallback(() => {
    if (selectedFile) {
      setUploadStatus('uploading')
      onFileSelect(selectedFile)
    }
  }, [selectedFile, onFileSelect])

  return (
    <div className={cn('w-full max-w-2xl mx-auto', className)}>
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(',')}
        multiple={multiple}
        onChange={handleInputChange}
        className="hidden"
      />

      {/* 拖拽上传区域 */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={cn(
          'relative border-2 border-dashed rounded-3xl p-12 transition-all duration-300 cursor-pointer',
          'hover:border-primary/50 hover:bg-primary/5',
          isDragOver && 'border-primary bg-primary/10 scale-[1.02]',
          'acrylic-strong'
        )}
      >
        {/* 装饰性背景 */}
        <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
          <div className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-to-br from-primary/5 to-purple-500/5 blur-3xl" />
          <div className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-gradient-to-tr from-blue-500/5 to-cyan-500/5 blur-3xl" />
        </div>

        <div className="relative z-10 flex flex-col items-center text-center space-y-6">
          {/* 图标 */}
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-full blur-xl" />
            <div className="relative w-20 h-20 bg-gradient-to-br from-primary to-primary/80 rounded-full flex items-center justify-center shadow-lg shadow-primary/30">
              <Upload className="w-10 h-10 text-primary-foreground" />
            </div>
          </div>

          {/* 文字说明 */}
          <div className="space-y-2">
            <h3 className="text-xl font-bold">
              {isDragOver ? '松开以上传文件' : '拖拽文件到此处'}
            </h3>
            <p className="text-muted-foreground text-sm">
              或 <span className="text-primary font-medium">点击选择文件</span>
            </p>
          </div>

          {/* 支持格式 */}
          <div className="flex flex-wrap justify-center gap-2">
            {accept.map((ext) => (
              <span
                key={ext}
                className="px-3 py-1 text-xs font-medium bg-muted/50 rounded-full text-muted-foreground"
              >
                {ext.toUpperCase()}
              </span>
            ))}
          </div>

          {/* 文件大小限制 */}
          <p className="text-xs text-muted-foreground">
            最大文件大小：{maxSize}MB
          </p>
        </div>
      </div>

      {/* 选中的文件 */}
      {selectedFile && (
        <div className="mt-4 acrylic-strong rounded-2xl p-4 space-y-3 fade-in">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className={cn(
                'w-10 h-10 rounded-xl flex items-center justify-center',
                uploadStatus === 'success' && 'bg-success/10',
                uploadStatus === 'error' && 'bg-error/10',
                uploadStatus === 'uploading' && 'bg-primary/10',
                uploadStatus === 'idle' && 'bg-muted'
              )}>
                {uploadStatus === 'success' && <CheckCircle className="w-5 h-5 text-success" />}
                {uploadStatus === 'error' && <AlertCircle className="w-5 h-5 text-error" />}
                {uploadStatus === 'uploading' && <Upload className="w-5 h-5 text-primary animate-pulse" />}
                {uploadStatus === 'idle' && <File className="w-5 h-5 text-muted-foreground" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{selectedFile.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {uploadStatus === 'error' ? (
                <>
                  <button
                    onClick={handleRetry}
                    className="px-3 py-1.5 text-sm font-medium text-primary hover:bg-primary/10 rounded-lg transition-colors"
                  >
                    重试
                  </button>
                  <button
                    onClick={handleCancel}
                    className="p-2 hover:bg-muted rounded-lg transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </>
              ) : uploadStatus === 'uploading' ? (
                <>
                  <div className="flex flex-col items-end gap-1">
                    <span className="text-xs text-muted-foreground">{Math.round(progress)}%</span>
                    <button
                      onClick={handleCancel}
                      className="px-3 py-1.5 text-sm font-medium text-error hover:bg-error/10 rounded-lg transition-colors flex items-center gap-1"
                    >
                      <StopCircle className="w-4 h-4" />
                      取消
                    </button>
                  </div>
                </>
              ) : (
                <button
                  onClick={handleCancel}
                  className="p-2 hover:bg-muted rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* 进度条 */}
          {uploadStatus === 'uploading' && (
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-primary/60 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          {/* 错误信息 */}
          {error && (
            <div className="flex items-start gap-2 text-sm text-error bg-error/10 p-3 rounded-xl">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <p>{error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
