use common_types::SceneState;
use service_kit::ServiceMetricsData;
use std::path::Path;
use vectorize::{RasterStrategy, RasterVectorizationReport, SemanticCandidate};

/// 流水线请求
#[derive(Debug, Clone)]
pub struct PipelineRequest {
    /// 文件路径
    pub path: std::path::PathBuf,
    /// 输出路径（可选）
    pub output_path: Option<std::path::PathBuf>,
}

impl PipelineRequest {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            output_path: None,
        }
    }

    pub fn with_output_path(mut self, path: impl AsRef<Path>) -> Self {
        self.output_path = Some(path.as_ref().to_path_buf());
        self
    }
}

/// 处理进度
#[derive(Debug, Clone)]
pub struct ProcessProgress {
    pub stage: ProcessStage,
    pub percent: f32,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessStage {
    Parsing,
    Vectorizing,
    BuildingTopology,
    Validating,
    Exporting,
    Completed,
}

/// 处理结果
#[derive(Debug)]
pub struct ProcessResult {
    pub scene: SceneState,
    pub validation: validator::ValidationReport,
    pub output_bytes: Vec<u8>,
    pub text_annotations: Vec<common_types::TextAnnotation>,
    pub dimension_summary: common_types::DimensionSummary,
    pub raster_report: Option<RasterVectorizationReport>,
    pub semantic_candidates: Vec<SemanticCandidate>,
}

/// 光栅处理选项，供 API/CLI/测试共用。
#[derive(Debug, Clone, Default)]
pub struct RasterProcessingOptions {
    pub strategy: Option<RasterStrategy>,
    pub dpi_override: Option<(f64, f64)>,
    pub scale_calibration: Option<ScaleCalibration>,
    pub debug_artifacts: bool,
    pub semantic_mode: Option<String>,
    pub ocr_backend: Option<String>,
    pub max_retries: Option<usize>,
}

/// 用户提供的尺度校准。
#[derive(Debug, Clone)]
pub struct ScaleCalibration {
    pub known_distance_px: f64,
    pub known_distance_mm: f64,
    pub points_px: Option<([f64; 2], [f64; 2])>,
}

/// 流水线指标
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// 总处理请求数
    pub total_requests: u64,
    /// 成功请求数
    pub success_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 平均处理时间（毫秒）
    pub avg_processing_time_ms: f64,
    /// 各服务指标
    pub service_metrics: Vec<ServiceMetricsData>,
}

/// 流水线统计信息
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub services_initialized: bool,
}
