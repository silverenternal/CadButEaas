//! 加速器错误类型

use thiserror::Error;

/// 加速器错误
#[derive(Debug, Error)]
pub enum AcceleratorError {
    /// 加速器不可用
    #[error("加速器 {0} 不可用：{1}")]
    Unavailable(String, String),

    /// 初始化失败
    #[error("加速器初始化失败：{0}")]
    InitializationFailed(String),

    /// 操作不支持
    #[error("加速器 {accelerator} 不支持操作 {op}")]
    OperationNotSupported {
        accelerator: String,
        op: String,
    },

    /// 内存不足
    #[error("加速器内存不足：需要 {needed}MB，可用 {available}MB")]
    OutOfMemory { needed: u64, available: u64 },

    /// 执行错误
    #[error("执行错误：{0}")]
    ExecutionFailed(String),

    /// 数据格式错误
    #[error("数据格式错误：{0}")]
    InvalidDataFormat(String),

    /// 超时错误
    #[error("操作超时：{0}")]
    Timeout(String),

    /// 后端特定错误
    #[error("后端错误 [{backend}]: {message}")]
    BackendError {
        backend: String,
        message: String,
    },
}

impl AcceleratorError {
    /// 创建初始化失败错误
    pub fn init_failed(msg: impl Into<String>) -> Self {
        Self::InitializationFailed(msg.into())
    }

    /// 创建执行失败错误
    pub fn execution_failed(msg: impl Into<String>) -> Self {
        Self::ExecutionFailed(msg.into())
    }

    /// 创建后端错误
    pub fn backend_error(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
        }
    }
}

/// 加速器操作结果
pub type Result<T> = std::result::Result<T, AcceleratorError>;
