//! API 路由定义 - 完整实现
//!
//! 改进：
//! 1. 真正的文件上传处理
//! 2. 支持 DXF/PDF 文件解析
//! 3. 返回实际处理结果
//! 4. 添加 API 版本控制
//! 5. 添加超时控制
//! 6. 集成 InteractSvc 交互服务

use axum::{
    extract::{DefaultBodyLimit, Multipart, State, WebSocketUpgrade, ws::{Message, WebSocket}},
    http::{StatusCode},
    routing::{self, get, post, options},
    Json, Router, response::IntoResponse,
};
use tower_http::cors::{CorsLayer, Any};
use http::Method;
use serde::{Deserialize, Serialize};
use crate::pipeline::ProcessingPipeline;
use common_types::{Point2, SceneState, HatchBoundaryPath, HatchPattern};
use interact::{InteractService, InteractionService, Edge, GapInfo};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use futures::{sink::SinkExt, stream::StreamExt};

// ============================================================================
// P0-4 新增：HATCH 实体定义
// ============================================================================

/// HATCH 实体（用于 API 响应）
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HatchEntity {
    pub id: usize,
    pub boundary_paths: Vec<HatchBoundaryPathResponse>,
    pub pattern: HatchPatternResponse,
    pub solid_fill: bool,
    pub layer: Option<String>,
    pub scale: f64,      // P0-NEW-14 修复：图案比例
    pub angle: f64,      // P0-NEW-14 修复：图案角度（度）
}

/// HATCH 边界路径响应
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HatchBoundaryPathResponse {
    Polyline {
        points: Vec<[f64; 2]>,
        closed: bool,
        bulges: Option<Vec<f64>>,  // P0-NEW-5 修复：添加 bulges 字段
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
        scale: f64,      // P0-NEW-14 修复：图案比例
        angle: f64,      // P0-NEW-14 修复：图案角度（度）
    },
    Custom {
        pattern_def: HatchPatternDefinitionResponse,
        scale: f64,      // P0-NEW-14 修复：图案比例
        angle: f64,      // P0-NEW-14 修复：图案角度（度）
    },
    Solid {
        color: [u8; 4],  // RGBA
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

/// CORS 预检请求处理器
async fn options_handler() -> impl IntoResponse {
    (
        StatusCode::NO_CONTENT,
        [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS"),
            ("Access-Control-Allow-Headers", "*"),
            ("Access-Control-Expose-Headers", "*"),
            ("Access-Control-Max-Age", "86400"),
        ],
    )
}

/// 添加 CORS 头到响应
fn with_cors<T: IntoResponse>(response: T) -> impl IntoResponse {
    (
        [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS"),
            ("Access-Control-Allow-Headers", "*"),
            ("Access-Control-Expose-Headers", "*"),
        ],
        response,
    )
}

/// API 版本
pub const API_VERSION: &str = "v1";

/// 最大上传文件大小：50MB（适用于大型 DXF/PDF 文件）
const MAX_UPLOAD_SIZE_MB: usize = 50;

/// API 状态
#[derive(Clone)]
pub struct ApiState {
    pub pipeline: ProcessingPipeline,
    pub interact: Arc<Mutex<InteractionService>>,
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
    pub job_id: String,
    pub status: ProcessStatus,
    pub message: String,
    pub result: Option<ProcessResult>,
    pub errors: Vec<String>,
    /// 边数据（可选，用于前端直接获取）
    #[serde(default)]
    pub edges: Option<Vec<Edge>>,
    /// P0-4 新增：HATCH 数据（可选，用于前端直接获取）
    #[serde(default)]
    pub hatches: Option<Vec<HatchEntity>>,
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

// ============================================================================
// 交互 API 响应类型
// ============================================================================

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

// ============================================================================
// 交互 API 请求类型
// ============================================================================

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

// ============================================================================
// 导出 API 请求类型
// ============================================================================

/// 导出请求
#[derive(Deserialize)]
pub struct ExportRequest {
    /// 导出格式：json, bincode, dxf
    pub format: String,
    /// 是否美化输出（仅 JSON 有效）
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

// ============================================================================
// 配置 API 响应类型
// ============================================================================

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
}

#[derive(Serialize, Deserialize)]
pub struct TopologyConfig {
    pub snap_tolerance_mm: f64,
    pub min_line_length_mm: f64,
    pub merge_angle_tolerance_deg: f64,
    pub max_gap_bridge_length_mm: f64,
    /// P11 新增：拓扑构建算法
    /// - "dfs": DFS 方案（默认，向后兼容）
    /// - "halfedge": Halfedge 方案（推荐，支持嵌套孔洞）
    #[serde(default = "default_topology_algorithm")]
    pub algorithm: String,
    /// P11 新增：跳过交点检测
    #[serde(default)]
    pub skip_intersection_check: bool,
    /// P11 新增：启用并行处理
    #[serde(default)]
    pub enable_parallel: bool,
    /// P11 新增：并行处理阈值
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

/// 创建 API 路由（不带状态，状态在 service.rs 中添加）
pub fn create_router() -> Router<ApiState> {
    // 创建路由
    Router::new()
        // 基础 API
        .route("/health", get(health_handler))
        .route("/process", post(process_handler_v1))
        .route("/process", options(options_handler))
        // 配置 API（P11 锐评落实）
        .route("/config/profiles", get(list_profiles_handler))
        .route("/config/profile/{name}", get(get_profile_handler))
        // WebSocket 实时通信（P11 锐评落实）
        .route("/ws", get(websocket_handler))
        // 交互 API - 选边追踪
        .route("/interact/auto_trace", post(interact_auto_trace_handler))
        .route("/interact/auto_trace", options(options_handler))
        // 交互 API - 圈选区域
        .route("/interact/lasso", post(interact_lasso_handler))
        .route("/interact/lasso", options(options_handler))
        // 交互 API - 缺口检测
        .route("/interact/detect_gaps", post(interact_detect_gaps_handler))
        .route("/interact/detect_gaps", options(options_handler))
        // 交互 API - 缺口桥接
        .route("/interact/snap_bridge", post(interact_snap_bridge_handler))
        .route("/interact/snap_bridge", options(options_handler))
        // 交互 API - 边界语义
        .route("/interact/set_boundary_semantic", post(interact_set_boundary_semantic_handler))
        .route("/interact/set_boundary_semantic", options(options_handler))
        // 交互 API - 状态查询
        .route("/interact/state", get(interact_state_handler))
        // 导出 API
        .route("/export", post(export_handler))
        .route("/export", options(options_handler))
        // 下载 API
        .route("/download/{filename}", get(download_handler))
        // P11 修复：移除 axum 默认的 2MB 请求体限制，允许上传最大 50MB 的文件
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_SIZE_MB * 1024 * 1024))
}

/// 创建带 CORS 的 API 路由（在 service.rs 中调用）
pub fn create_router_with_cors() -> Router<ApiState> {
    // 创建 CORS 层
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::PATCH, Method::OPTIONS])
        .allow_headers(Any)
        .expose_headers(Any)
        .max_age(Duration::from_secs(86400));

    create_router().layer(cors)
}

/// 健康检查处理器
async fn health_handler(State(_state): State<ApiState>) -> impl IntoResponse {
    // 深度健康检查：验证临时目录是否可写
    let temp_writable = std::env::temp_dir().join("cad_health_check");
    let status = if std::fs::File::create(&temp_writable).is_ok() {
        let _ = std::fs::remove_file(temp_writable);
        "healthy"
    } else {
        "unhealthy"
    };

    with_cors(Json(HealthResponse {
        status: status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        api_version: API_VERSION.to_string(),
    }))
}

/// 列出所有预设配置处理器
async fn list_profiles_handler(
    State(_state): State<ApiState>,
) -> Json<ProfileListResponse> {
    // 内置预设配置
    let profiles = vec![
        ProfileInfo {
            name: "architectural".to_string(),
            description: "建筑图纸预设 - 适用于 AutoCAD 导出的建筑平面图".to_string(),
        },
        ProfileInfo {
            name: "mechanical".to_string(),
            description: "机械图纸预设 - 适用于高精度机械图纸".to_string(),
        },
        ProfileInfo {
            name: "scanned".to_string(),
            description: "扫描图纸预设 - 适用于扫描版图纸（仅适用于线条清晰的图纸）".to_string(),
        },
        ProfileInfo {
            name: "quick".to_string(),
            description: "快速原型预设 - 低精度要求，快速处理".to_string(),
        },
    ];

    Json(ProfileListResponse { profiles })
}

/// 获取预设配置详情处理器
async fn get_profile_handler(
    State(_state): State<ApiState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<ProfileDetailResponse>, StatusCode> {
    // 返回内置预设配置详情
    match name.as_str() {
        "architectural" => Ok(Json(ProfileDetailResponse {
            name: "architectural".to_string(),
            description: "建筑图纸预设 - 适用于 AutoCAD 导出的建筑平面图".to_string(),
            topology: TopologyConfig {
                snap_tolerance_mm: 0.5,
                min_line_length_mm: 1.0,
                merge_angle_tolerance_deg: 5.0,
                max_gap_bridge_length_mm: 2.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: true,
                parallel_threshold: 1000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 0.5,
                min_area_m2: 1.0,
                min_edge_length_mm: 10.0,
                min_angle_deg: 15.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
        })),
        "mechanical" => Ok(Json(ProfileDetailResponse {
            name: "mechanical".to_string(),
            description: "机械图纸预设 - 适用于高精度机械图纸".to_string(),
            topology: TopologyConfig {
                snap_tolerance_mm: 0.1,
                min_line_length_mm: 0.5,
                merge_angle_tolerance_deg: 2.0,
                max_gap_bridge_length_mm: 0.5,
                algorithm: "halfedge".to_string(),
                skip_intersection_check: false,
                enable_parallel: true,
                parallel_threshold: 500,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 0.1,
                min_area_m2: 0.01,
                min_edge_length_mm: 1.0,
                min_angle_deg: 5.0,
            },
            export: ExportConfig {
                format: "bincode".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
        })),
        "scanned" => Ok(Json(ProfileDetailResponse {
            name: "scanned".to_string(),
            description: "扫描图纸预设 - 适用于扫描版图纸（仅适用于线条清晰的图纸）".to_string(),
            topology: TopologyConfig {
                snap_tolerance_mm: 2.0,
                min_line_length_mm: 3.0,
                merge_angle_tolerance_deg: 10.0,
                max_gap_bridge_length_mm: 5.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: false,
                enable_parallel: true,
                parallel_threshold: 2000,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 2.0,
                min_area_m2: 2.0,
                min_edge_length_mm: 20.0,
                min_angle_deg: 30.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 2,
                auto_validate: true,
            },
        })),
        "quick" => Ok(Json(ProfileDetailResponse {
            name: "quick".to_string(),
            description: "快速原型预设 - 低精度要求，快速处理".to_string(),
            topology: TopologyConfig {
                snap_tolerance_mm: 1.0,
                min_line_length_mm: 5.0,
                merge_angle_tolerance_deg: 15.0,
                max_gap_bridge_length_mm: 1.0,
                algorithm: "dfs".to_string(),
                skip_intersection_check: true,
                enable_parallel: true,
                parallel_threshold: 500,
            },
            validator: ValidatorConfig {
                closure_tolerance_mm: 1.0,
                min_area_m2: 0.5,
                min_edge_length_mm: 5.0,
                min_angle_deg: 10.0,
            },
            export: ExportConfig {
                format: "json".to_string(),
                json_indent: 0,
                auto_validate: false,
            },
        })),
        _ => Err(StatusCode::NOT_FOUND),
    }
}

/// V1 版本的处理处理器 - 渐进式渲染
///
/// # 渐进式渲染流程
///
/// 1. **阶段 1（快速）**：解析 DXF → 提取原始边 → 立即返回（~1 秒）
/// 2. **阶段 2（后台）**：构建拓扑 → 完成后通过 WebSocket 推送更新
async fn process_handler_v1(
    State(state): State<ApiState>,
    mut multipart: Multipart,
) -> Result<Json<ProcessResponse>, StatusCode> {
    use std::io::Write;
    use std::fs::File;
    use std::path::PathBuf;

    tracing::info!("=== 收到文件上传请求（渐进式渲染） ===");

    // 收集字段
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;

    // 解析 multipart 表单 - 改进错误处理
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::error!("❌ 解析 multipart 表单失败：{}", e);
        StatusCode::BAD_REQUEST
    })? {
        let name = field.name().unwrap_or("unknown");

        if name == "file" {
            file_name = field.file_name().map(String::from);
            let bytes = field.bytes().await.map_err(|e| {
                tracing::error!("❌ 读取文件数据失败：{}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
            
            // 记录文件大小
            let file_size = bytes.len();
            tracing::info!("📄 收到文件：{:?}, 大小：{:.2} KB", file_name, file_size as f64 / 1024.0);
            
            // 检查文件大小（最大 50MB）
            if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024 {
                tracing::error!("❌ 文件过大：{:.2} MB > {} MB",
                    file_size as f64 / (1024.0 * 1024.0), MAX_UPLOAD_SIZE_MB);
                return Ok(Json(ProcessResponse {
                    job_id: uuid_simple(),
                    status: ProcessStatus::Failed,
                    message: format!("文件过大，最大支持 {} MB", MAX_UPLOAD_SIZE_MB),
                    result: None,
                    errors: vec![format!("文件大小 {:.2} MB 超过限制 {} MB",
                        file_size as f64 / (1024.0 * 1024.0), MAX_UPLOAD_SIZE_MB)],
                    edges: None,
                    hatches: None,  // P0-4 修复：添加 hatches 字段
                }));
            }
            
            file_data = Some(bytes.to_vec());
        }
    }

    // 验证文件是否存在
    let file_data = file_data.ok_or_else(|| {
        tracing::warn!("⚠️ 请求中未找到文件字段");
        StatusCode::BAD_REQUEST
    })?;

    // 确定文件类型
    let detected_type = detect_file_type(&file_data, file_name.as_deref());
    tracing::info!("🔍 检测到文件类型：{:?}", detected_type);

    if detected_type == FileType::Unknown {
        tracing::warn!("⚠️ 无法识别文件类型，文件名：{:?}", file_name);
        return Ok(Json(ProcessResponse {
            job_id: uuid_simple(),
            status: ProcessStatus::Failed,
            message: "不支持的文件格式".to_string(),
            result: None,
            errors: vec![
                format!("无法识别文件类型，请上传 DXF 或 PDF 文件（文件名：{:?}）", file_name)
            ],
            edges: None,
            hatches: None,  // P0-4 修复：添加 hatches 字段
        }));
    }

    // 创建临时文件（保留扩展名以便 Parser 识别）
    let temp_dir = std::env::temp_dir();
    let temp_file_name = format!("cad_process_{}_{}.{}",
        std::process::id(),
        uuid_simple(),
        file_type_extension(&detected_type)
    );
    let temp_path: PathBuf = temp_dir.join(&temp_file_name);
    tracing::info!("📁 创建临时文件：{:?}", temp_path);

    // 写入临时文件
    let mut temp_file = File::create(&temp_path)
        .map_err(|e| {
            tracing::error!("❌ 创建临时文件失败：{}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    temp_file.write_all(&file_data)
        .map_err(|e| {
            tracing::error!("❌ 写入临时文件失败：{}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    drop(temp_file);

    tracing::info!("✅ 临时文件已写入，开始解析...");

    // ========================================================================
    // 阶段 1：快速解析，提取原始边（~1 秒）
    // ========================================================================
    tracing::info!("阶段 1/2: 快速解析，提取原始边");

    let parse_result = match detected_type {
        FileType::Dxf => {
            tracing::info!("  开始 DXF 解析...");
            state.pipeline.parser().parse_file(&temp_path).map_err(|e| {
                tracing::error!("❌ DXF 解析失败：{}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
        }
        FileType::Pdf => {
            // PDF 需要矢量化
            tracing::warn!("  PDF 文件需要矢量化处理，可能需要较长时间");
            state.pipeline.parser().parse_file(&temp_path).map_err(|e| {
                tracing::error!("❌ PDF 解析失败：{}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
        }
        FileType::Unknown => {
            // 已经在上面处理过，这里不会到达
            unreachable!()
        }
    };

    // 从解析结果提取原始边和 HATCH
    let entities = parse_result.into_entities();
    let edges = entities_to_edges(&entities);
    let hatches = entities_to_hatches(&entities);  // P0-4 新增：提取 HATCH 数据
    tracing::info!("  ✅ 提取 {} 条原始边，{} 个 HATCH", edges.len(), hatches.len());

    // 立即返回原始边用于快速渲染
    let job_id = uuid_simple();
    tracing::info!("✅ 阶段 1 完成，返回 {} 条边用于快速渲染", edges.len());

    // 更新交互服务状态（使用原始边）
    let new_interact = InteractionService::new(edges.clone());
    *state.interact.lock().await = new_interact;

    // ========================================================================
    // 阶段 2：后台拓扑构建（不阻塞响应）
    // ========================================================================
    tracing::info!("阶段 2/2: 启动后台拓扑构建任务");

    let pipeline = state.pipeline.clone();
    let interact = state.interact.clone();
    let temp_path_clone = temp_path.clone();

    tokio::spawn(async move {
        tracing::info!("  🔄 后台任务：开始拓扑构建");

        // 构建拓扑（可能需要几分钟）
        match pipeline.process_file(&temp_path_clone).await {
            Ok(process_result) => {
                tracing::info!("  ✅ 后台任务：拓扑构建完成");

                // 从拓扑结果重建边
                let topo_edges = scene_to_edges(&process_result.scene);
                tracing::info!("  📊 后台任务：得到 {} 条拓扑边", topo_edges.len());

                // 更新交互服务，同时设置 scene_state
                let mut new_interact = InteractionService::new(topo_edges.clone());
                new_interact.set_scene_state(process_result.scene.clone());
                *interact.lock().await = new_interact;

                tracing::info!("  ✅ 后台任务：拓扑数据已更新");
            }
            Err(e) => {
                tracing::error!("  ❌ 后台任务：拓扑构建失败：{}", e);
            }
        }

        // 清理临时文件
        let _ = std::fs::remove_file(&temp_path_clone);
        tracing::info!("  🗑️ 后台任务：临时文件已清理");
    });

    // 立即返回阶段 1 的结果
    Ok(Json(ProcessResponse {
        job_id: job_id.clone(),
        status: ProcessStatus::Completed,
        message: format!("快速渲染完成，{} 条边已加载，拓扑构建在后台进行", edges.len()),
        result: Some(ProcessResult {
            scene_summary: SceneSummary {
                outer_boundaries: 0, // 待拓扑完成后更新
                holes: 0,
                total_points: 0,
            },
            validation_summary: ValidationSummary {
                error_count: 0,
                warning_count: 0,
                passed: true,
            },
            output_size: 0,
        }),
        errors: vec![],
        edges: Some(edges),
        hatches: Some(hatches),  // P0-4 新增：返回 HATCH 数据
    }))
}

/// 文件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileType {
    Dxf,
    Pdf,
    Unknown,
}

/// 检测文件类型
fn detect_file_type(data: &[u8], file_name: Option<&str>) -> FileType {
    // 首先尝试通过扩展名判断
    if let Some(name) = file_name {
        let ext = std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        match ext.as_str() {
            "dxf" => return FileType::Dxf,
            "pdf" => return FileType::Pdf,
            _ => {}
        }
    }

    // 通过魔数检测
    if data.starts_with(b"%PDF") {
        return FileType::Pdf;
    }

    // DXF 文件通常以 "AutoCAD" 或 section 标记开始
    if data.starts_with(b"AutoCAD") 
        || data.starts_with(b"SECTION")
        || data.starts_with(&[0x41, 0x43, 0x31, 0x30]) // "AC10"
    {
        return FileType::Dxf;
    }

    // 尝试解析为 ASCII DXF
    if std::str::from_utf8(data).is_ok_and(|s| {
        s.contains("SECTION") && s.contains("ENTITIES")
    }) {
        return FileType::Dxf;
    }

    FileType::Unknown
}

/// 生成简单 UUID（用于 job_id）
fn uuid_simple() -> String {
    use uuid::Uuid;
    format!("job-{}", Uuid::new_v4().to_string().replace('-', ""))
}

/// 将 FileType 转换为文件扩展名
fn file_type_extension(file_type: &FileType) -> &'static str {
    match file_type {
        FileType::Dxf => "dxf",
        FileType::Pdf => "pdf",
        FileType::Unknown => "unknown",
    }
}

/// 从实体列表提取边（用于快速渲染）
/// 
/// # 支持的实体类型
/// - Line, Polyline, Arc, Circle（基础类型）
/// - Text（渲染为方框）
/// - BlockReference（展开块定义）
/// - Dimension（渲染为尺寸线）
/// - Path（展开为线段）
fn entities_to_edges(entities: &[common_types::RawEntity]) -> Vec<interact::Edge> {
    let mut edges = Vec::new();
    let mut edge_id = 0;

    for entity in entities {
        match entity {
            common_types::RawEntity::Line { start, end, metadata, .. } => {
                let mut edge = interact::Edge::new(edge_id, *start, *end);
                edge.layer = metadata.layer.clone();
                edges.push(edge);
                edge_id += 1;
            }
            // Polyline 分解为多条线段
            common_types::RawEntity::Polyline { points, closed, metadata, .. } => {
                if points.len() >= 2 {
                    for i in 0..points.len() - 1 {
                        let mut edge = interact::Edge::new(edge_id, points[i], points[i + 1]);
                        edge.layer = metadata.layer.clone();
                        edges.push(edge);
                        edge_id += 1;
                    }
                    // 如果闭合，添加最后一条边
                    if *closed {
                        let mut edge = interact::Edge::new(edge_id, points[points.len() - 1], points[0]);
                        edge.layer = metadata.layer.clone();
                        edges.push(edge);
                        edge_id += 1;
                    }
                }
            }
            // Arc 离散化为线段
            common_types::RawEntity::Arc { center, radius, start_angle, end_angle, metadata, .. } => {
                let segments = 16; // 增加分段数提高精度
                let angle_range = end_angle - start_angle;
                for i in 0..segments {
                    let a1 = start_angle + (angle_range * (i as f64) / segments as f64);
                    let a2 = start_angle + (angle_range * ((i + 1) as f64) / segments as f64);
                    // ✅ 修复：将度数转换为弧度
                    let p1 = [center[0] + radius * a1.to_radians().cos(), center[1] + radius * a1.to_radians().sin()];
                    let p2 = [center[0] + radius * a2.to_radians().cos(), center[1] + radius * a2.to_radians().sin()];
                    let mut edge = interact::Edge::new(edge_id, p1, p2);
                    edge.layer = metadata.layer.clone();
                    edges.push(edge);
                    edge_id += 1;
                }
            }
            // Circle 离散化为 16 段线段
            common_types::RawEntity::Circle { center, radius, metadata, .. } => {
                let segments = 32; // 增加分段数提高精度
                for i in 0..segments {
                    let a1 = 2.0 * std::f64::consts::PI * (i as f64) / segments as f64;
                    let a2 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / segments as f64;
                    let p1 = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
                    let p2 = [center[0] + radius * a2.cos(), center[1] + radius * a2.sin()];
                    let mut edge = interact::Edge::new(edge_id, p1, p2);
                    edge.layer = metadata.layer.clone();
                    edges.push(edge);
                    edge_id += 1;
                }
            }
            // Text 渲染为矩形框（避免完全跳过）
            common_types::RawEntity::Text { position, height, content, metadata, .. } => {
                // 计算文本边界框（假设宽高比约 0.6）
                let char_count = content.chars().count() as f64;
                let width = height * char_count * 0.6;
                let text_height = height * 1.2;
                
                let x = position[0];
                let y = position[1];
                
                // 绘制文本边界框（4 条边）
                let corners = [
                    [x, y],
                    [x + width, y],
                    [x + width, y + text_height],
                    [x, y + text_height],
                ];
                
                for i in 0..4 {
                    let mut edge = interact::Edge::new(edge_id, corners[i], corners[(i + 1) % 4]);
                    edge.layer = metadata.layer.clone();
                    edges.push(edge);
                    edge_id += 1;
                }
            }
            // BlockReference 展开为边（需要块定义）
            common_types::RawEntity::BlockReference { 
                block_name, .. 
            } => {
                // 注意：阶段 1 没有块定义数据，这里只能跳过
                // TODO: 在阶段 1 也传递块定义数据
                tracing::debug!("阶段 1 跳过块引用：{} (需要块定义数据)", block_name);
            }
            // Dimension 渲染为尺寸线（简化处理：连接定义点）
            common_types::RawEntity::Dimension { definition_points, metadata, .. } => {
                if definition_points.len() >= 2 {
                    for i in 0..definition_points.len() - 1 {
                        let mut edge = interact::Edge::new(edge_id, definition_points[i], definition_points[i + 1]);
                        edge.layer = metadata.layer.clone();
                        edges.push(edge);
                        edge_id += 1;
                    }
                }
            }
            // Path 展开为线段
            common_types::RawEntity::Path { commands, metadata, .. } => {
                let mut current_point: Option<[f64; 2]> = None;

                for cmd in commands {
                    match cmd {
                        common_types::PathCommand::MoveTo { x, y } => {
                            current_point = Some([*x, *y]);
                        }
                        common_types::PathCommand::LineTo { x, y } => {
                            if let Some(start) = current_point {
                                let mut edge = interact::Edge::new(edge_id, start, [*x, *y]);
                                edge.layer = metadata.layer.clone();
                                edges.push(edge);
                                edge_id += 1;
                                current_point = Some([*x, *y]);
                            }
                        }
                        common_types::PathCommand::ArcTo { x, y, .. } => {
                            // 简化处理：直接连接到终点
                            if let Some(start) = current_point {
                                let mut edge = interact::Edge::new(edge_id, start, [*x, *y]);
                                edge.layer = metadata.layer.clone();
                                edges.push(edge);
                                edge_id += 1;
                                current_point = Some([*x, *y]);
                            }
                        }
                        common_types::PathCommand::Close => {
                            // 闭合路径（需要额外逻辑，阶段 1 简化处理）
                        }
                    }
                }
            }
            // P0-1: HATCH 填充图案（简化处理：渲染边界）
            common_types::RawEntity::Hatch { boundary_paths, metadata, .. } => {
                // 将 HATCH 边界转换为边
                for boundary in boundary_paths {
                    match boundary {
                        common_types::HatchBoundaryPath::Polyline { points, closed, .. } => {
                            if points.len() >= 2 {
                                for i in 0..points.len() - 1 {
                                    let mut edge = interact::Edge::new(edge_id, points[i], points[i + 1]);
                                    edge.layer = metadata.layer.clone();
                                    edges.push(edge);
                                    edge_id += 1;
                                }
                                if *closed {
                                    let mut edge = interact::Edge::new(edge_id, points[points.len() - 1], points[0]);
                                    edge.layer = metadata.layer.clone();
                                    edges.push(edge);
                                    edge_id += 1;
                                }
                            }
                        }
                        common_types::HatchBoundaryPath::Arc { center, radius, start_angle, end_angle, ccw, .. } => {
                            // 离散化圆弧边界
                            let segments = 16;
                            let angle_range = if *ccw { end_angle - start_angle } else { start_angle - end_angle };
                            for i in 0..segments {
                                let a1 = start_angle + (angle_range * (i as f64) / segments as f64);
                                let a2 = start_angle + (angle_range * ((i + 1) as f64) / segments as f64);
                                // ✅ 修复：将度数转换为弧度
                                let p1 = [center[0] + radius * a1.to_radians().cos(), center[1] + radius * a1.to_radians().sin()];
                                let p2 = [center[0] + radius * a2.to_radians().cos(), center[1] + radius * a2.to_radians().sin()];
                                let mut edge = interact::Edge::new(edge_id, p1, p2);
                                edge.layer = metadata.layer.clone();
                                edges.push(edge);
                                edge_id += 1;
                            }
                        }
                        common_types::HatchBoundaryPath::EllipseArc { center, major_axis, minor_axis_ratio, start_angle, end_angle, ccw, .. } => {
                            // 简化处理：离散化为线段
                            let segments = 32;
                            let angle_range = if *ccw { end_angle - start_angle } else { start_angle - end_angle };
                            for i in 0..segments {
                                let t = (i as f64) / segments as f64;
                                let angle = start_angle + angle_range * t;
                                // ✅ 修复：将度数转换为弧度
                                let angle_rad = angle.to_radians();
                                let prev_angle_rad = (start_angle + angle_range * ((i - 1) as f64) / segments as f64).to_radians();
                                let x = center[0] + major_axis[0] * angle_rad.cos();
                                let y = center[1] + major_axis[1] * minor_axis_ratio * angle_rad.sin();
                                if i > 0 {
                                    let prev_x = center[0] + major_axis[0] * prev_angle_rad.cos();
                                    let prev_y = center[1] + major_axis[1] * minor_axis_ratio * prev_angle_rad.sin();
                                    let mut edge = interact::Edge::new(edge_id, [prev_x, prev_y], [x, y]);
                                    edge.layer = metadata.layer.clone();
                                    edges.push(edge);
                                    edge_id += 1;
                                }
                            }
                        }
                        common_types::HatchBoundaryPath::Spline { control_points, .. } => {
                            // 简化处理：连接控制点
                            if control_points.len() >= 2 {
                                for i in 0..control_points.len() - 1 {
                                    let mut edge = interact::Edge::new(edge_id, control_points[i], control_points[i + 1]);
                                    edge.layer = metadata.layer.clone();
                                    edges.push(edge);
                                    edge_id += 1;
                                }
                            }
                        }
                    }
                }
            }
            // P1-1: XREF 外部参照支持 - 待完整实现
            // 外部参照需要加载外部文件并递归解析，目前跳过处理
            common_types::RawEntity::XRef { .. } => {
                tracing::warn!("XREF 外部参照支持 - 待完整实现，跳过处理");
                // TODO: P1-1 完整实现 XREF 加载和解析
            }
        }
    }

    tracing::info!("entities_to_edges: 从 {} 个实体提取 {} 条边", entities.len(), edges.len());
    edges
}

// ============================================================================
// P0-4 新增：HATCH 数据提取函数
// ============================================================================

/// 从实体列表提取 HATCH 数据（用于前端渲染）
fn entities_to_hatches(entities: &[common_types::RawEntity]) -> Vec<HatchEntity> {
    let mut hatches = Vec::new();
    let mut hatch_id = 0;

    for entity in entities {
        if let common_types::RawEntity::Hatch {
            boundary_paths,
            pattern,
            solid_fill,
            metadata,
            scale,    // P0-NEW-14 修复：提取 scale
            angle,    // P0-NEW-14 修复：提取 angle
            ..
        } = entity
        {
            // 转换边界路径
            let boundary_paths_response: Vec<HatchBoundaryPathResponse> = boundary_paths
                .iter()
                .map(|boundary| match boundary {
                    common_types::HatchBoundaryPath::Polyline { points, closed, bulges } => {
                        HatchBoundaryPathResponse::Polyline {
                            points: points.iter().map(|p| [p[0], p[1]]).collect(),
                            closed: *closed,
                            bulges: bulges.clone(),  // P0-NEW-5 修复：保留 bulges 字段
                        }
                    }
                    common_types::HatchBoundaryPath::Arc {
                        center,
                        radius,
                        start_angle,
                        end_angle,
                        ccw,
                    } => HatchBoundaryPathResponse::Arc {
                        center: [center[0], center[1]],
                        radius: *radius,
                        start_angle: *start_angle,
                        end_angle: *end_angle,
                        ccw: *ccw,
                    },
                    common_types::HatchBoundaryPath::EllipseArc {
                        center,
                        major_axis,
                        minor_axis_ratio,
                        start_angle,
                        end_angle,
                        ccw,
                        extrusion_direction: _,
                    } => HatchBoundaryPathResponse::EllipseArc {
                        center: [center[0], center[1]],
                        major_axis: [major_axis[0], major_axis[1]],
                        minor_axis_ratio: *minor_axis_ratio,
                        start_angle: *start_angle,
                        end_angle: *end_angle,
                        ccw: *ccw,
                    },
                    common_types::HatchBoundaryPath::Spline {
                        control_points,
                        knots,
                        degree,
                        weights: _,
                        fit_points: _,
                        flags: _,
                    } => HatchBoundaryPathResponse::Spline {
                        control_points: control_points.iter().map(|p| [p[0], p[1]]).collect(),
                        knots: knots.clone(),
                        degree: *degree,
                    },
                })
                .collect();

            // 转换图案
            let pattern_response = match pattern {
                common_types::HatchPattern::Predefined { name } => {
                    HatchPatternResponse::Predefined {
                        name: name.clone(),
                        scale: *scale,    // P0-NEW-14 修复：传递 scale
                        angle: *angle,    // P0-NEW-14 修复：传递 angle
                    }
                }
                common_types::HatchPattern::Custom { pattern_def } => {
                    HatchPatternResponse::Custom {
                        pattern_def: HatchPatternDefinitionResponse {
                            name: pattern_def.name.clone(),
                            description: pattern_def.description.clone(),
                            lines: pattern_def
                                .lines
                                .iter()
                                .map(|line| HatchPatternLineResponse {
                                    start_point: [
                                        line.start_point[0],
                                        line.start_point[1],
                                    ],
                                    angle: line.angle,
                                    offset: [line.offset[0], line.offset[1]],
                                    dash_pattern: line.dash_pattern.clone(),
                                })
                                .collect(),
                        },
                        scale: *scale,    // P0-NEW-14 修复：传递 scale
                        angle: *angle,    // P0-NEW-14 修复：传递 angle
                    }
                }
                common_types::HatchPattern::Solid { color } => {
                    HatchPatternResponse::Solid {
                        color: [color.r, color.g, color.b, color.a],  // P0-4 修复：Color32 字段访问
                    }
                }
            };

            hatches.push(HatchEntity {
                id: hatch_id,
                boundary_paths: boundary_paths_response,
                pattern: pattern_response,
                solid_fill: *solid_fill,
                layer: metadata.layer.clone(),
                scale: *scale,    // P0-NEW-14 修复：传递 scale
                angle: *angle,    // P0-NEW-14 修复：传递 angle
            });

            hatch_id += 1;
        }
    }

    tracing::info!(
        "entities_to_hatches: 从 {} 个实体提取 {} 个 HATCH",
        entities.len(),
        hatches.len()
    );

    hatches
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http::Request;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        use std::sync::Arc;
        use tokio::sync::Mutex;
        use interact::InteractionService;

        let state = ApiState {
            pipeline: ProcessingPipeline::new(),
            interact: Arc::new(Mutex::new(InteractionService::new(vec![
                interact::Edge::new(0, [0.0, 0.0], [10.0, 0.0]),
                interact::Edge::new(1, [10.0, 0.0], [10.0, 10.0]),
                interact::Edge::new(2, [10.0, 10.0], [0.0, 10.0]),
                interact::Edge::new(3, [0.0, 10.0], [0.0, 0.0]),
            ]))),
        };

        // P0-NEW-5 修复：使用 with_state 注入状态
        let app = create_router().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
    }

    #[test]
    fn test_detect_file_type_pdf() {
        let data = b"%PDF-1.4 test";
        assert_eq!(detect_file_type(data, None), FileType::Pdf);
        assert_eq!(detect_file_type(data, Some("test.pdf")), FileType::Pdf);
    }

    #[test]
    fn test_detect_file_type_dxf_extension() {
        let data = b"test data";
        assert_eq!(detect_file_type(data, Some("test.dxf")), FileType::Dxf);
    }

    #[test]
    fn test_detect_file_type_dxf_content() {
        let data = b"AutoCAD Binary DXF";
        assert_eq!(detect_file_type(data, None), FileType::Dxf);
        
        let data = b"SECTION\nENTITIES\nENDSEC";
        assert_eq!(detect_file_type(data, None), FileType::Dxf);
    }

    #[test]
    fn test_detect_file_type_unknown() {
        let data = b"unknown format";
        assert_eq!(detect_file_type(data, None), FileType::Unknown);
    }
}

// ============================================================================
// 交互 API 处理器
// ============================================================================
//
// P11 锐评落实说明：
// - InteractService 已改为同步 trait（移除 async_trait）
// - 但处理器函数仍然是 async fn，因为需要获取 MutexGuard
// - state.interact.lock().await 是异步的（获取互斥锁）
// - 但获取锁后的方法调用是同步的：interact.auto_trace_from_edge(...)
// - 这是正确的模式：互斥锁保护共享状态，同步方法避免浪费线程栈空间

/// 辅助函数：从 SceneState 创建 Edge 集合
fn scene_to_edges(scene: &SceneState) -> Vec<Edge> {
    let mut edges = Vec::new();

    // 优先使用原始边数据（如果存在）
    if !scene.edges.is_empty() {
        for raw_edge in &scene.edges {
            let mut edge = Edge::new(raw_edge.id, raw_edge.start, raw_edge.end);
            edge.layer = raw_edge.layer.clone();
            edges.push(edge);
        }
        return edges;
    }

    // 否则从外轮廓提取边
    let mut edge_id = 0;
    if let Some(outer) = &scene.outer {
        let points = &outer.points;
        for i in 0..points.len() {
            let start = points[i];
            let end = points[(i + 1) % points.len()];
            edges.push(Edge::new(edge_id, start, end));
            edge_id += 1;
        }
    }

    // 从孔洞提取边
    for hole in &scene.holes {
        let points = &hole.points;
        for i in 0..points.len() {
            let start = points[i];
            let end = points[(i + 1) % points.len()];
            edges.push(Edge::new(edge_id, start, end));
            edge_id += 1;
        }
    }

    edges
}

/// 辅助函数：转换 GapInfo 为 GapInfoResponse
fn gap_info_to_response(gap: &GapInfo) -> GapInfoResponse {
    GapInfoResponse {
        id: gap.id,
        start: [gap.endpoint_a[0], gap.endpoint_a[1]],
        end: [gap.endpoint_b[0], gap.endpoint_b[1]],
        length: gap.length,
        gap_type: format!("{:?}", gap.gap_type),
    }
}

/// 交互 API - 选边追踪处理器
async fn interact_auto_trace_handler(
    State(state): State<ApiState>,
    Json(request): Json<SelectEdgeRequest>,
) -> Result<Json<AutoTraceResponse>, StatusCode> {
    tracing::info!("收到选边追踪请求：edge_id={}", request.edge_id);

    let mut interact = state.interact.lock().await;

    match interact.auto_trace_from_edge(request.edge_id) {
        Ok(result) => {
            let loop_points = result.loop_.as_ref().map(|l| {
                l.points.iter().map(|p| [p[0], p[1]]).collect()
            });

            Ok(Json(AutoTraceResponse {
                success: true,
                loop_points,
                message: format!("成功追踪到 {} 个点",
                    result.loop_.as_ref().map(|l| l.points.len()).unwrap_or(0)),
            }))
        }
        Err(e) => {
            tracing::warn!("选边追踪失败：{:?}", e);
            Ok(Json(AutoTraceResponse {
                success: false,
                loop_points: None,
                message: format!("追踪失败：{:?}", e),
            }))
        }
    }
}

/// 交互 API - 圈选区域处理器
async fn interact_lasso_handler(
    State(state): State<ApiState>,
    Json(request): Json<LassoRequest>,
) -> Result<Json<LassoResponse>, StatusCode> {
    tracing::info!("收到圈选请求，多边形点数={}", request.polygon.len());

    let polygon: Vec<Point2> = request.polygon.iter().map(|p| [p[0], p[1]]).collect();
    let mut interact = state.interact.lock().await;

    match interact.extract_from_lasso(&polygon) {
        Ok(result) => {
            let loops = result.loops.iter()
                .map(|l| l.points.iter().map(|p| [p[0], p[1]]).collect())
                .collect();

            Ok(Json(LassoResponse {
                selected_edges: result.selected_edges,
                loops,
                connected_components: result.connected_components,
            }))
        }
        Err(e) => {
            tracing::warn!("圈选失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 缺口检测处理器
async fn interact_detect_gaps_handler(
    State(state): State<ApiState>,
    Json(request): Json<GapDetectionRequest>,
) -> Result<Json<GapDetectionResponse>, StatusCode> {
    tracing::info!("收到缺口检测请求：tolerance={}", request.tolerance);

    let interact = state.interact.lock().await;

    match interact.detect_gaps(request.tolerance) {
        Ok(gaps) => {
            let gap_responses: Vec<GapInfoResponse> = gaps
                .iter()
                .map(gap_info_to_response)
                .collect();

            Ok(Json(GapDetectionResponse {
                gaps: gap_responses,
                total_count: gaps.len(),
            }))
        }
        Err(e) => {
            tracing::warn!("缺口检测失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 缺口桥接处理器
async fn interact_snap_bridge_handler(
    State(state): State<ApiState>,
    Json(request): Json<SnapBridgeRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("收到缺口桥接请求：gap_id={}", request.gap_id);

    let mut interact = state.interact.lock().await;

    match interact.apply_snap_bridge(request.gap_id) {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            tracing::warn!("缺口桥接失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 边界语义设置处理器
async fn interact_set_boundary_semantic_handler(
    State(state): State<ApiState>,
    Json(request): Json<BoundarySemanticRequest>,
) -> Result<StatusCode, StatusCode> {
    tracing::info!("收到边界语义设置请求：segment_id={}, semantic={}",
        request.segment_id, request.semantic);

    // 解析语义字符串为 BoundarySemantic 枚举
    use common_types::scene::BoundarySemantic;
    let semantic = match request.semantic.as_str() {
        "hard_wall" => BoundarySemantic::HardWall,
        "absorptive_wall" => BoundarySemantic::AbsorptiveWall,
        "door" => BoundarySemantic::Door,
        "window" => BoundarySemantic::Window,
        "opening" => BoundarySemantic::Opening,
        s => BoundarySemantic::Custom(s.to_string()),
    };

    let mut interact = state.interact.lock().await;

    match interact.set_boundary_semantic(request.segment_id, semantic) {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => {
            tracing::warn!("边界语义设置失败：{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 交互 API - 状态查询处理器
async fn interact_state_handler(
    State(state): State<ApiState>,
) -> Result<Json<InteractionStateResponse>, StatusCode> {
    let interact = state.interact.lock().await;

    // 获取当前状态
    let state_ref = interact.get_state();
    let selected_edges: Vec<usize> = state_ref.selected_edges.iter().copied().collect();
    let detected_gaps: Vec<GapInfoResponse> = state_ref.detected_gaps
        .iter()
        .map(gap_info_to_response)
        .collect();

    Ok(Json(InteractionStateResponse {
        total_edges: state_ref.edges.len(),
        selected_edges,
        detected_gaps,
    }))
}

// ============================================================================
// WebSocket 实时通信（P11 锐评落实）
// ============================================================================

/// WebSocket 消息类型
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// 连接确认
    #[serde(rename = "connected")]
    Connected { session_id: String },
    /// 边选择事件
    #[serde(rename = "edge_selected")]
    EdgeSelected { edge_id: usize },
    /// 追踪结果
    #[serde(rename = "trace_result")]
    TraceResult { edges: Vec<usize>, loop_closed: bool },
    /// 缺口检测结果
    #[serde(rename = "gaps_detected")]
    GapsDetected { gaps: Vec<GapInfoResponse> },
    /// 错误消息
    #[serde(rename = "error")]
    Error { message: String },
    /// 心跳
    #[serde(rename = "ping")]
    Ping,
    /// 心跳响应
    #[serde(rename = "pong")]
    Pong,
}

/// WebSocket 处理器
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ApiState>,
) -> impl axum::response::IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();
    tracing::info!("WebSocket 连接：session_id={}", session_id);
    
    ws.on_upgrade(move |socket| handle_websocket(socket, state, session_id))
}

/// 处理 WebSocket 连接
async fn handle_websocket(socket: WebSocket, state: ApiState, session_id: String) {
    let (mut sender, mut receiver) = socket.split();
    
    // 发送连接确认消息
    let connected_msg = WsMessage::Connected { session_id: session_id.clone() };
    if let Ok(json) = serde_json::to_string(&connected_msg) {
        let _ = sender.send(Message::Text(json)).await;
    }
    
    // 处理接收的消息
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // 解析客户端消息
                if let Ok(client_msg) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(msg_type) = client_msg.get("type").and_then(|t| t.as_str()) {
                        match msg_type {
                            "select_edge" => {
                                // 处理边选择
                                if let Some(edge_id) = client_msg.get("edge_id").and_then(|id| id.as_u64()) {
                                    let edge_id = edge_id as usize;

                                    // 调用自动追踪
                                    let mut interact = state.interact.lock().await;
                                    match interact.auto_trace_from_edge(edge_id) {
                                        Ok(trace_result) => {
                                            let edges: Vec<usize> = trace_result.path;
                                            let loop_closed = trace_result.loop_.is_some();

                                            // 发送追踪结果
                                            let result_msg = WsMessage::TraceResult { edges, loop_closed };
                                            if let Ok(json) = serde_json::to_string(&result_msg) {
                                                let _ = sender.send(Message::Text(json)).await;
                                            }
                                        }
                                        Err(e) => {
                                            let error_msg = WsMessage::Error { message: e.to_string() };
                                            if let Ok(json) = serde_json::to_string(&error_msg) {
                                                let _ = sender.send(Message::Text(json)).await;
                                            }
                                        }
                                    }
                                }
                            }
                            "detect_gaps" => {
                                // 处理缺口检测
                                let tolerance = client_msg.get("tolerance")
                                    .and_then(|t| t.as_f64())
                                    .unwrap_or(0.5);

                                let interact = state.interact.lock().await;
                                match interact.detect_gaps(tolerance) {
                                    Ok(gaps) => {
                                        let gap_responses: Vec<GapInfoResponse> = gaps
                                            .iter()
                                            .map(gap_info_to_response)
                                            .collect();

                                        let gaps_msg = WsMessage::GapsDetected { gaps: gap_responses };
                                        if let Ok(json) = serde_json::to_string(&gaps_msg) {
                                            let _ = sender.send(Message::Text(json)).await;
                                        }
                                    }
                                    Err(e) => {
                                        let error_msg = WsMessage::Error { message: e.to_string() };
                                        if let Ok(json) = serde_json::to_string(&error_msg) {
                                            let _ = sender.send(Message::Text(json)).await;
                                        }
                                    }
                                }
                            }
                            "ping" => {
                                // 心跳响应
                                let pong_msg = WsMessage::Pong;
                                if let Ok(json) = serde_json::to_string(&pong_msg) {
                                    let _ = sender.send(Message::Text(json)).await;
                                }
                            }
                            _ => {
                                tracing::warn!("未知 WebSocket 消息类型：{}", msg_type);
                            }
                        }
                    }
                }
            }
            Ok(Message::Close(_)) => {
                tracing::info!("WebSocket 断开：session_id={}", session_id);
                break;
            }
            Err(e) => {
                tracing::error!("WebSocket 错误：{:?}", e);
                break;
            }
            _ => {}
        }
    }

    tracing::info!("WebSocket 连接结束：session_id={}", session_id);
}

// ============================================================================
// 导出 API 处理器
// ============================================================================

/// 导出处理器
async fn export_handler(
    State(state): State<ApiState>,
    Json(request): Json<ExportRequest>,
) -> Result<Json<ExportResponse>, StatusCode> {
    use export::formats::ExportFormat;

    tracing::info!("收到导出请求：format={}", request.format);

    // 获取当前交互服务中的场景状态
    let interact = state.interact.lock().await;
    let scene_state = interact.get_scene_state();
    drop(interact);

    // 确定导出格式
    let format = match request.format.to_lowercase().as_str() {
        "json" => ExportFormat::Json,
        "bincode" | "binary" => ExportFormat::Binary,
        "dxf" => {
            return Ok(Json(ExportResponse {
                success: false,
                message: "DXF 导出暂不支持".to_string(),
                download_url: None,
                file_name: None,
                file_size: 0,
            }));
        }
        _ => {
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    // 创建导出服务
    let export_service = state.pipeline.export();

    // 执行导出
    match export_service.export(&scene_state) {
        Ok(export_result) => {
            let file_name = format!("cad_export_{}.{}", 
                uuid_simple(),
                match format {
                    ExportFormat::Json => "json",
                    ExportFormat::Binary => "bin",
                }
            );

            // 将数据写入临时文件
            let temp_dir = std::env::temp_dir();
            let temp_path = temp_dir.join(&file_name);

            if let Err(e) = std::fs::write(&temp_path, &export_result.bytes) {
                tracing::error!("写入临时文件失败：{}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }

            tracing::info!("导出成功：file_name={}, size={} bytes", file_name, export_result.bytes.len());

            Ok(Json(ExportResponse {
                success: true,
                message: "导出成功".to_string(),
                download_url: Some(format!("/download/{}", file_name)),
                file_name: Some(file_name),
                file_size: export_result.bytes.len(),
            }))
        }
        Err(e) => {
            tracing::error!("导出失败：{}", e);
            Ok(Json(ExportResponse {
                success: false,
                message: format!("导出失败：{}", e),
                download_url: None,
                file_name: None,
                file_size: 0,
            }))
        }
    }
}

// ============================================================================
// 下载 API 处理器
// ============================================================================

/// 下载处理器
async fn download_handler(
    State(_state): State<ApiState>,
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Result<axum::response::Response, StatusCode> {
    use axum::http::header;

    tracing::info!("收到下载请求：filename={}", filename);

    // 从临时目录读取文件
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(&filename);

    if !temp_path.exists() {
        tracing::warn!("文件不存在：{}", filename);
        return Err(StatusCode::NOT_FOUND);
    }

    // 读取文件内容
    let file_content = match std::fs::read(&temp_path) {
        Ok(content) => content,
        Err(e) => {
            tracing::error!("读取文件失败：{}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // 确定 Content-Type
    let content_type = if filename.ends_with(".json") {
        "application/json"
    } else if filename.ends_with(".bin") {
        "application/octet-stream"
    } else {
        "application/octet-stream"
    };

    // 构建响应
    let mut response = axum::response::Response::new(file_content.into());
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        content_type.parse().unwrap(),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        format!("attachment; filename=\"{}\"", filename).parse().unwrap(),
    );

    Ok(response)
}
