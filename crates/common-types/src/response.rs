//! 统一响应包装器
//!
//! 为所有服务提供统一的响应格式，支持：
//! - 请求追踪（request_id 关联）
//! - 结构化错误
//! - 性能指标（延迟）
//! - 分页支持

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::ErrorCode;
use crate::request::RequestId;

/// 响应状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResponseStatus {
    /// 成功
    Success,
    /// 部分成功
    PartialSuccess,
    /// 失败
    Failure,
}

/// 统一服务错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceError {
    /// 错误码
    pub code: ErrorCode,
    /// 错误消息
    pub message: String,
    /// 错误详情（结构化）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<ErrorDetails>,
    /// 是否可重试
    #[serde(default)]
    pub retryable: bool,
    /// 修复建议
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl ServiceError {
    /// 创建新错误
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
            retryable: false,
            suggestion: None,
        }
    }

    /// 设置详情
    pub fn with_details(mut self, details: ErrorDetails) -> Self {
        self.details = Some(details);
        self
    }

    /// 设置为可重试
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    /// 设置修复建议
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// 结构化错误详情
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorDetails {
    /// DXF 解析错误
    DxfParse {
        file: String,
        line: Option<usize>,
        raw_error: String,
    },
    /// PDF 解析错误
    PdfParse {
        file: String,
        page: Option<u32>,
        raw_error: String,
    },
    /// 拓扑错误
    Topology {
        stage: String,
        point_count: usize,
        segment_count: usize,
    },
    /// 验证错误
    Validation { issues: Vec<ValidationIssue> },
    /// 几何错误
    Geometry {
        issue_type: String,
        location: Option<GeoLocation>,
        severity: String,
    },
    /// 内部错误
    Internal {
        source: String,
        backtrace: Option<String>,
    },
}

/// 验证问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// 问题类型
    pub issue_type: String,
    /// 严重性
    pub severity: Severity,
    /// 描述
    pub message: String,
    /// 位置（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<GeoLocation>,
    /// 受影响的实体 ID
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_ids: Vec<String>,
}

/// 严重性级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// 几何位置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    /// X 坐标
    pub x: f64,
    /// Y 坐标
    pub y: f64,
    /// 可选的 Z 坐标
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
}

impl GeoLocation {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, z: None }
    }

    pub fn with_z(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z: Some(z) }
    }
}

/// 统一响应包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response<T> {
    /// 关联的请求 ID
    pub request_id: RequestId,
    /// 响应状态
    pub status: ResponseStatus,
    /// 响应负载（成功时）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<T>,
    /// 错误信息（失败时）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ServiceError>,
    /// 处理延迟（毫秒）
    pub latency_ms: u64,
    /// 额外元数据
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl<T> Response<T> {
    /// 创建成功响应
    pub fn success(request_id: RequestId, payload: T, latency_ms: u64) -> Self {
        Self {
            request_id,
            status: ResponseStatus::Success,
            payload: Some(payload),
            error: None,
            latency_ms,
            metadata: HashMap::new(),
        }
    }

    /// 创建失败响应
    pub fn failure(request_id: RequestId, error: ServiceError, latency_ms: u64) -> Self {
        Self {
            request_id,
            status: ResponseStatus::Failure,
            payload: None,
            error: Some(error),
            latency_ms,
            metadata: HashMap::new(),
        }
    }

    /// 创建部分成功响应
    pub fn partial_success(
        request_id: RequestId,
        payload: T,
        warnings: Vec<ServiceError>,
        latency_ms: u64,
    ) -> Self {
        let mut metadata = HashMap::new();
        if !warnings.is_empty() {
            metadata.insert(
                "warnings".to_string(),
                serde_json::to_string(&warnings).unwrap_or_default(),
            );
        }
        Self {
            request_id,
            status: ResponseStatus::PartialSuccess,
            payload: Some(payload),
            error: None,
            latency_ms,
            metadata,
        }
    }

    /// 检查是否成功
    pub fn is_success(&self) -> bool {
        matches!(
            self.status,
            ResponseStatus::Success | ResponseStatus::PartialSuccess
        )
    }

    /// 获取 payload 引用
    pub fn payload(&self) -> Option<&T> {
        self.payload.as_ref()
    }

    /// 获取错误引用
    pub fn error(&self) -> Option<&ServiceError> {
        self.error.as_ref()
    }
}

// ============================================================================
// 响应计时器
// ============================================================================

/// 响应计时器，用于自动计算延迟
pub struct ResponseTimer {
    start: Instant,
}

impl ResponseTimer {
    /// 启动计时器
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// 获取经过的毫秒数
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Default for ResponseTimer {
    fn default() -> Self {
        Self::start()
    }
}

// ============================================================================
// 预定义响应类型
// ============================================================================

/// 解析响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResponse {
    /// 解析后的几何数据（JSON 格式）
    pub geometry_json: String,
    /// 实体数量
    pub entity_count: usize,
    /// 警告信息
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// 拓扑响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyResponse {
    /// 拓扑数据（JSON 格式）
    pub topology_json: String,
    /// 节点数量
    pub node_count: usize,
    /// 边数量
    pub edge_count: usize,
    /// 环数量
    pub loop_count: usize,
}

/// 验证响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateResponse {
    /// 是否通过验证
    pub passed: bool,
    /// 验证问题列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<ValidationIssue>,
    /// 统计信息
    #[serde(default)]
    pub stats: ValidationStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    /// 检查的实体数量
    pub entities_checked: usize,
    /// 自相交数量
    pub self_intersections: usize,
    /// 重复边数量
    pub duplicate_edges: usize,
    /// 微小线段数量
    pub tiny_segments: usize,
    /// 超大坐标数量
    pub large_coordinates: usize,
}

/// 矢量化响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizeResponse {
    /// 矢量化后的几何数据（JSON 格式）
    pub geometry_json: String,
    /// 线段数量
    pub segment_count: usize,
    /// 处理时间（毫秒）
    pub processing_time_ms: u64,
}

/// 导出响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResponse {
    /// 输出文件路径
    pub output_path: String,
    /// 文件大小（字节）
    pub file_size_bytes: u64,
    /// 导出时间（毫秒）
    pub export_time_ms: u64,
}
