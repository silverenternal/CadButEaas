use interact::Edge;
use serde::{Deserialize, Serialize};
use vectorize::{RasterVectorizationReport, SemanticCandidate};

/// HATCH 实体（用于 API 响应）
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HatchEntity {
    pub id: usize,
    pub boundary_paths: Vec<HatchBoundaryPathResponse>,
    pub pattern: HatchPatternResponse,
    pub solid_fill: bool,
    pub layer: Option<String>,
    pub scale: f64,
    pub angle: f64,
}

/// HATCH 边界路径响应
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HatchBoundaryPathResponse {
    Polyline {
        points: Vec<[f64; 2]>,
        closed: bool,
        bulges: Option<Vec<f64>>,
    },
    Arc {
        center: [f64; 2],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,
    },
    EllipseArc {
        center: [f64; 2],
        major_axis: [f64; 2],
        minor_axis_ratio: f64,
        start_angle: f64,
        end_angle: f64,
        ccw: bool,
    },
    Spline {
        control_points: Vec<[f64; 2]>,
        knots: Vec<f64>,
        degree: u32,
    },
}

/// HATCH 图案响应
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HatchPatternResponse {
    Predefined {
        name: String,
        scale: f64,
        angle: f64,
    },
    Custom {
        pattern_def: HatchPatternDefinitionResponse,
        scale: f64,
        angle: f64,
    },
    Solid {
        color: [u8; 4],
    },
}

/// HATCH 图案定义响应
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HatchPatternDefinitionResponse {
    pub name: String,
    pub description: Option<String>,
    pub lines: Vec<HatchPatternLineResponse>,
}

/// HATCH 图案行响应
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HatchPatternLineResponse {
    pub start_point: [f64; 2],
    pub angle: f64,
    pub offset: [f64; 2],
    pub dash_pattern: Vec<f64>,
}

/// 健康检查响应
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub api_version: String,
}

/// 处理请求响应
#[derive(Serialize, Deserialize)]
pub struct ProcessResponse {
    #[serde(default = "default_process_schema_version")]
    pub schema_version: String,
    pub job_id: String,
    pub status: ProcessStatus,
    pub message: String,
    pub result: Option<ProcessResult>,
    pub errors: Vec<String>,
    #[serde(default)]
    pub edges: Option<Vec<Edge>>,
    #[serde(default)]
    pub hatches: Option<Vec<HatchEntity>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_report: Option<RasterVectorizationReport>,
    #[serde(default)]
    pub semantic_candidates: Vec<SemanticCandidate>,
}

pub(crate) fn default_process_schema_version() -> String {
    "process-response-1.1".to_string()
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStatus {
    Completed,
    Partial,
    Failed,
}

/// 处理结果详情
#[derive(Serialize, Deserialize)]
pub struct ProcessResult {
    pub scene_summary: SceneSummary,
    pub validation_summary: ValidationSummary,
    pub output_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SceneSummary {
    pub outer_boundaries: usize,
    pub holes: usize,
    pub total_points: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ValidationSummary {
    pub error_count: usize,
    pub warning_count: usize,
    pub passed: bool,
}

/// 缺口信息响应
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GapInfoResponse {
    pub id: usize,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
    pub gap_type: String,
}

/// 自动追踪结果响应
#[derive(Serialize, Deserialize)]
pub struct AutoTraceResponse {
    pub success: bool,
    pub loop_points: Option<Vec<[f64; 2]>>,
    pub message: String,
}

/// 圈选结果响应
#[derive(Serialize, Deserialize)]
pub struct LassoResponse {
    pub selected_edges: Vec<usize>,
    pub loops: Vec<Vec<[f64; 2]>>,
    pub connected_components: usize,
}

/// 缺口检测结果响应
#[derive(Serialize, Deserialize)]
pub struct GapDetectionResponse {
    pub gaps: Vec<GapInfoResponse>,
    pub total_count: usize,
}

/// 边信息响应
#[derive(Serialize, Deserialize)]
pub struct EdgeResponse {
    pub id: usize,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
}

/// 交互状态响应
#[derive(Serialize, Deserialize)]
pub struct InteractionStateResponse {
    pub total_edges: usize,
    pub selected_edges: Vec<usize>,
    pub detected_gaps: Vec<GapInfoResponse>,
}

/// 边选择请求
#[derive(Deserialize)]
pub struct SelectEdgeRequest {
    pub edge_id: usize,
}

/// 圈选请求
#[derive(Deserialize)]
pub struct LassoRequest {
    pub polygon: Vec<[f64; 2]>,
}

/// 缺口桥接请求
#[derive(Deserialize)]
pub struct SnapBridgeRequest {
    pub gap_id: usize,
}

/// 边界语义设置请求
#[derive(Deserialize)]
pub struct BoundarySemanticRequest {
    pub segment_id: usize,
    pub semantic: String,
}

/// 缺口检测请求
#[derive(Deserialize)]
pub struct GapDetectionRequest {
    pub tolerance: f64,
}

/// 导出请求
#[derive(Deserialize)]
pub struct ExportRequest {
    pub format: String,
    pub pretty: Option<bool>,
}

/// 导出响应
#[derive(Serialize, Deserialize)]
pub struct ExportResponse {
    pub success: bool,
    pub message: String,
    pub download_url: Option<String>,
    pub file_name: Option<String>,
    pub file_size: usize,
}

/// 预设配置列表响应
#[derive(Serialize, Deserialize)]
pub struct ProfileListResponse {
    pub profiles: Vec<ProfileInfo>,
}

/// 预设配置信息
#[derive(Serialize, Deserialize)]
pub struct ProfileInfo {
    pub name: String,
    pub description: String,
}

/// 预设配置详情响应
#[derive(Serialize, Deserialize)]
pub struct ProfileDetailResponse {
    pub name: String,
    pub description: String,
    pub topology: TopologyConfig,
    pub validator: ValidatorConfig,
    pub export: ExportConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectorize: Option<VectorizeProfileConfig>,
}

#[derive(Serialize, Deserialize)]
pub struct TopologyConfig {
    pub snap_tolerance_mm: f64,
    pub min_line_length_mm: f64,
    pub merge_angle_tolerance_deg: f64,
    pub max_gap_bridge_length_mm: f64,
    #[serde(default = "default_topology_algorithm")]
    pub algorithm: String,
    #[serde(default)]
    pub skip_intersection_check: bool,
    #[serde(default)]
    pub enable_parallel: bool,
    #[serde(default = "default_parallel_threshold")]
    pub parallel_threshold: usize,
}

fn default_topology_algorithm() -> String {
    "dfs".to_string()
}

fn default_parallel_threshold() -> usize {
    1000
}

#[derive(Serialize, Deserialize)]
pub struct ValidatorConfig {
    pub closure_tolerance_mm: f64,
    pub min_area_m2: f64,
    pub min_edge_length_mm: f64,
    pub min_angle_deg: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ExportConfig {
    pub format: String,
    pub json_indent: u8,
    pub auto_validate: bool,
}

#[derive(Serialize, Deserialize)]
pub struct VectorizeProfileConfig {
    pub adaptive_threshold: bool,
    pub skeletonize: bool,
    pub text_separation: bool,
    pub quality_assessment: bool,
    pub denoise: bool,
    pub enhance_contrast: bool,
}

/// WebSocket 消息类型
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum WsMessage {
    #[serde(rename = "connected")]
    Connected { session_id: String },
    #[serde(rename = "edge_selected")]
    EdgeSelected { edge_id: usize },
    #[serde(rename = "trace_result")]
    TraceResult {
        edges: Vec<usize>,
        loop_closed: bool,
    },
    #[serde(rename = "gaps_detected")]
    GapsDetected { gaps: Vec<GapInfoResponse> },
    #[serde(rename = "topology_ready")]
    TopologyReady { edge_count: usize },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
}
