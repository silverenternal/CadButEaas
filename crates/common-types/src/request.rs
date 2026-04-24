//! 统一请求包装器
//!
//! 为所有服务提供统一的请求格式，支持：
//! - 请求追踪（request_id）
//! - 超时控制
//! - 元数据传递
//! - 本地/远程透明切换

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 请求 ID 类型
pub type RequestId = String;

/// 请求元数据
///
/// 支持分布式链路追踪（OpenTelemetry 兼容）：
/// - `trace_id`: 整个请求链路的唯一标识
/// - `span_id`: 当前服务调用的标识
/// - `parent_span_id`: 父服务调用的标识
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// 用户 ID（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// 会话 ID（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 追踪 ID（用于分布式追踪，整个请求链路的唯一标识）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Span ID（当前服务调用的标识）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    /// 父 Span ID（用于构建调用链）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// 认证令牌（用于服务间认证）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// 自定义键值对
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, String>,
}

impl RequestMetadata {
    /// 创建空元数据
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建带追踪 ID 的元数据
    pub fn with_trace_id(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: Some(trace_id.into()),
            ..Default::default()
        }
    }

    /// 创建带用户 ID 的元数据
    pub fn with_user_id(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            ..Default::default()
        }
    }

    /// 设置认证令牌
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// 添加额外元数据
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// 设置 Span ID
    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    /// 设置父 Span ID
    pub fn with_parent_span_id(mut self, parent_span_id: impl Into<String>) -> Self {
        self.parent_span_id = Some(parent_span_id.into());
        self
    }

    /// 创建完整的链路追踪元数据
    ///
    /// # 参数
    ///
    /// * `trace_id` - 整个请求链路的唯一标识
    /// * `span_id` - 当前服务调用的标识
    /// * `parent_span_id` - 父服务调用的标识（可选）
    pub fn with_trace(
        trace_id: impl Into<String>,
        span_id: impl Into<String>,
        parent_span_id: Option<String>,
    ) -> Self {
        Self {
            trace_id: Some(trace_id.into()),
            span_id: Some(span_id.into()),
            parent_span_id,
            ..Default::default()
        }
    }

    /// 为子服务创建新的 Span
    ///
    /// 当前 Span ID 变为父 Span ID，生成新的 Span ID
    pub fn child_span(&self, new_span_id: impl Into<String>) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: Some(new_span_id.into()),
            parent_span_id: self.span_id.clone(),
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            token: self.token.clone(),
            extra: self.extra.clone(),
        }
    }
}

/// 统一请求包装器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request<T> {
    /// 请求 ID（自动生成）
    pub id: RequestId,
    /// 请求负载
    pub payload: T,
    /// 请求元数据
    #[serde(default)]
    pub metadata: RequestMetadata,
    /// 超时时间（毫秒）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl<T> Request<T> {
    /// 创建新请求（自动生成 ID）
    pub fn new(payload: T) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            payload,
            metadata: RequestMetadata::default(),
            timeout_ms: None,
        }
    }

    /// 创建带超时的请求
    pub fn with_timeout(payload: T, timeout: Duration) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            payload,
            metadata: RequestMetadata::default(),
            timeout_ms: Some(timeout.as_millis() as u64),
        }
    }

    /// 设置元数据
    pub fn with_metadata(mut self, metadata: RequestMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// 设置超时（毫秒）
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// 获取超时 Duration
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout_ms.map(Duration::from_millis)
    }
}

impl<T> From<T> for Request<T> {
    fn from(payload: T) -> Self {
        Request::new(payload)
    }
}

// ============================================================================
// 预定义请求类型
// ============================================================================

/// 文件解析请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseFileRequest {
    /// 文件路径
    pub path: String,
    /// 是否自动修复
    #[serde(default)]
    pub auto_fix: bool,
    /// 图层过滤器
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_filter: Option<Vec<String>>,
}

impl ParseFileRequest {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            auto_fix: false,
            layer_filter: None,
        }
    }
}

/// 拓扑构建请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTopologyRequest {
    /// 输入几何数据（JSON 格式）
    pub geometry_json: String,
    /// 容差配置
    #[serde(default)]
    pub tolerance: ToleranceConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// 合并容差（米）
    #[serde(default = "default_merge_tolerance")]
    pub merge_tolerance: f64,
    /// 简化容差（米）
    #[serde(default = "default_simplify_tolerance")]
    pub simplify_tolerance: f64,
}

fn default_merge_tolerance() -> f64 {
    0.001 // 1mm
}

fn default_simplify_tolerance() -> f64 {
    0.0005 // 0.5mm
}

impl BuildTopologyRequest {
    pub fn new(geometry_json: impl Into<String>) -> Self {
        Self {
            geometry_json: geometry_json.into(),
            tolerance: ToleranceConfig::default(),
        }
    }
}

/// 几何验证请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateGeometryRequest {
    /// 几何数据（JSON 格式）
    pub geometry_json: String,
    /// 验证规则
    #[serde(default)]
    pub rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ValidationRule {
    #[default]
    SelfIntersection,
    DuplicateEdges,
    TinySegments,
    LargeCoordinates,
    OpenLoops,
}

impl ValidateGeometryRequest {
    pub fn new(geometry_json: impl Into<String>) -> Self {
        Self {
            geometry_json: geometry_json.into(),
            rules: vec![ValidationRule::default()],
        }
    }
}

/// PDF 矢量化请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizePdfRequest {
    /// PDF 文件路径
    pub path: String,
    /// 页码（从 1 开始，0 表示全部）
    #[serde(default = "default_page_number")]
    pub page: u32,
    /// 二值化阈值（0-255）
    #[serde(default = "default_threshold")]
    pub threshold: u8,
    /// 吸附容差（像素）
    #[serde(default = "default_snap_tolerance")]
    pub snap_tolerance_px: f64,
}

fn default_page_number() -> u32 {
    1
}

fn default_threshold() -> u8 {
    128
}

fn default_snap_tolerance() -> f64 {
    2.0
}

impl VectorizePdfRequest {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            page: 1,
            threshold: 128,
            snap_tolerance_px: 2.0,
        }
    }
}

/// 导出请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    /// 导出数据（JSON 格式）
    pub data_json: String,
    /// 导出格式
    pub format: ExportFormat,
    /// 输出路径
    pub output_path: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Dxf,
    Json,
    Svg,
    Pdf,
}

impl ExportRequest {
    pub fn new(
        data_json: impl Into<String>,
        format: ExportFormat,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            data_json: data_json.into(),
            format,
            output_path: output_path.into(),
        }
    }
}
