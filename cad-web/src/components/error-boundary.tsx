// P2-2 新增：React Error Boundary 组件
import { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

/**
 * Error Boundary 组件
 * 
 * 用于捕获子组件树中的错误，防止整个应用崩溃
 * 提供友好的错误提示 UI 和错误恢复机制
 * 
 * ## 使用示例
 * 
 * ```tsx
 * // 基础用法
 * <ErrorBoundary>
 *   <CanvasViewer />
 * </ErrorBoundary>
 * 
 * // 自定义 fallback UI
 * <ErrorBoundary
 *   fallback={
 *     <div className="error-fallback">
 *       <h2>出现错误</h2>
 *       <p>请刷新页面重试</p>
 *     </div>
 *   }
 * >
 *   <CanvasViewer />
 * </ErrorBoundary>
 * 
 * // 错误上报
 * <ErrorBoundary
 *   onError={(error, errorInfo) => {
 *     // 上报到监控系统
 *     reportError(error, errorInfo)
 *   }}
 * >
 *   <CanvasViewer />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] 捕获到错误:', error, errorInfo)
    
    // 调用用户的错误回调
    this.props.onError?.(error, errorInfo)
    
    // 保存错误信息用于显示
    this.setState({ errorInfo })
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined })
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      // 使用自定义 fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      // 默认错误 UI
      return (
        <div className="error-boundary-fallback">
          <div className="error-content">
            <h2 className="error-title">出现错误</h2>
            
            {this.state.error && (
              <div className="error-details">
                <p className="error-message">{this.state.error.message}</p>
                {this.state.errorInfo && (
                  <details className="error-stack">
                    <summary>查看详细错误信息</summary>
                    <pre>{this.state.errorInfo.componentStack}</pre>
                  </details>
                )}
              </div>
            )}
            
            <div className="error-actions">
              <button 
                className="btn-retry"
                onClick={this.handleRetry}
              >
                重试
              </button>
              <button 
                className="btn-reload"
                onClick={this.handleReload}
              >
                刷新页面
              </button>
            </div>
          </div>
          
          <style>{`
            .error-boundary-fallback {
              display: flex;
              align-items: center;
              justify-content: center;
              min-height: 400px;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white;
              padding: 2rem;
              border-radius: 8px;
              margin: 1rem;
            }
            
            .error-content {
              text-align: center;
              max-width: 600px;
            }
            
            .error-title {
              font-size: 1.5rem;
              font-weight: 600;
              margin-bottom: 1rem;
            }
            
            .error-message {
              font-size: 0.875rem;
              opacity: 0.9;
              margin-bottom: 1rem;
              font-family: monospace;
              background: rgba(0, 0, 0, 0.2);
              padding: 0.5rem 1rem;
              border-radius: 4px;
            }
            
            .error-stack {
              text-align: left;
              margin-top: 1rem;
              font-size: 0.75rem;
              opacity: 0.8;
            }
            
            .error-stack pre {
              white-space: pre-wrap;
              word-break: break-word;
              max-height: 300px;
              overflow: auto;
              background: rgba(0, 0, 0, 0.3);
              padding: 1rem;
              border-radius: 4px;
            }
            
            .error-actions {
              display: flex;
              gap: 1rem;
              justify-content: center;
              margin-top: 1.5rem;
            }
            
            .btn-retry,
            .btn-reload {
              padding: 0.5rem 1.5rem;
              border: none;
              border-radius: 4px;
              font-size: 0.875rem;
              cursor: pointer;
              transition: all 0.2s;
            }
            
            .btn-retry {
              background: white;
              color: #667eea;
            }
            
            .btn-retry:hover {
              background: rgba(255, 255, 255, 0.9);
            }
            
            .btn-reload {
              background: rgba(255, 255, 255, 0.2);
              color: white;
              border: 1px solid rgba(255, 255, 255, 0.5);
            }
            
            .btn-reload:hover {
              background: rgba(255, 255, 255, 0.3);
            }
          `}</style>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * 错误边界高阶组件
 * 
 * @param WrappedComponent 需要错误保护的组件
 * @param fallback 可选的 fallback UI
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  fallback?: ReactNode
) {
  return function WithErrorBoundary(props: P) {
    return (
      <ErrorBoundary fallback={fallback}>
        <WrappedComponent {...props} />
      </ErrorBoundary>
    )
  }
}
