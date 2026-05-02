//! API 路由定义 - 完整实现
//!
//! 改进：
//! 1. 真正的文件上传处理
//! 2. 支持 DXF/PDF 文件解析
//! 3. 返回实际处理结果
//! 4. 添加 API 版本控制
//! 5. 添加超时控制
//! 6. 集成 InteractSvc 交互服务

mod dto;
mod edges;
mod export;
mod hatch;
mod interaction;
mod upload;
mod websocket;

pub use dto::*;

use edges::{entities_to_edges, gap_info_to_response, scene_to_edges};
use export::{download_handler, export_handler};
use hatch::entities_to_hatches;
use interaction::{
    interact_auto_trace_handler, interact_detect_gaps_handler, interact_lasso_handler,
    interact_set_boundary_semantic_handler, interact_snap_bridge_handler, interact_state_handler,
};
use upload::{
    detect_file_type, file_type_extension, parse_bool, parse_pair, uuid_simple, FileType,
};
use websocket::websocket_handler;

use crate::pipeline::{ProcessingPipeline, RasterProcessingOptions, ScaleCalibration};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, options, post},
    Json, Router,
};
use http::Method;
use interact::{InteractService, InteractionService};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use vectorize::{RasterStrategy, RasterVectorizationReport};

/// CORS 预检请求处理器
async fn options_handler() -> impl IntoResponse {
    (
        StatusCode::NO_CONTENT,
        [
            ("Access-Control-Allow-Origin", "*"),
            (
                "Access-Control-Allow-Methods",
                "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            ),
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
            (
                "Access-Control-Allow-Methods",
                "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            ),
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

/// 创建 API 路由（不带状态，状态在 service.rs 中添加）
pub fn create_router() -> Router<ApiState> {
    // 创建路由
    Router::new()
        // 基础 API
        .route("/health", get(health_handler))
        .route("/process", post(process_handler_v1))
        .route("/process", options(options_handler))
        .route("/process/raster", post(process_raster_handler))
        .route("/process/raster", options(options_handler))
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
        .route(
            "/interact/set_boundary_semantic",
            post(interact_set_boundary_semantic_handler),
        )
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
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::PATCH,
            Method::OPTIONS,
        ])
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
async fn list_profiles_handler(State(_state): State<ApiState>) -> Json<ProfileListResponse> {
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
        ProfileInfo {
            name: "photo_sketch".to_string(),
            description: "照片/手绘预设 - 适用于光栅图片矢量化，强预处理".to_string(),
        },
        ProfileInfo {
            name: "raster_clean".to_string(),
            description: "干净光栅线稿预设".to_string(),
        },
        ProfileInfo {
            name: "raster_scan".to_string(),
            description: "扫描光栅图纸预设".to_string(),
        },
        ProfileInfo {
            name: "raster_photo".to_string(),
            description: "拍照光栅图纸预设".to_string(),
        },
        ProfileInfo {
            name: "raster_sketch".to_string(),
            description: "手绘草图光栅预设".to_string(),
        },
        ProfileInfo {
            name: "raster_semantic".to_string(),
            description: "光栅语义解析预设".to_string(),
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
            vectorize: None,
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
            vectorize: None,
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
            vectorize: None,
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
            vectorize: None,
        })),
        "photo_sketch" => Ok(Json(ProfileDetailResponse {
            name: "photo_sketch".to_string(),
            description: "照片/手绘预设 - 适用于光栅图片矢量化，强预处理".to_string(),
            topology: TopologyConfig {
                snap_tolerance_mm: 2.0,
                min_line_length_mm: 3.0,
                merge_angle_tolerance_deg: 10.0,
                max_gap_bridge_length_mm: 5.0,
                algorithm: "halfedge".to_string(),
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
            vectorize: Some(VectorizeProfileConfig {
                adaptive_threshold: true,
                skeletonize: true,
                text_separation: true,
                quality_assessment: true,
                denoise: true,
                enhance_contrast: true,
            }),
        })),
        name if name.starts_with("raster_") => {
            let config = config::CadConfig::from_profile_file(name)
                .or_else(|_| config::CadConfig::from_profile(name))
                .map_err(|_| StatusCode::NOT_FOUND)?;
            Ok(Json(profile_detail_from_config(name, config)))
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}

fn profile_detail_from_config(name: &str, config: config::CadConfig) -> ProfileDetailResponse {
    ProfileDetailResponse {
        name: name.to_string(),
        description: format!("{} 光栅预设", name),
        topology: TopologyConfig {
            snap_tolerance_mm: config.topology.snap_tolerance_mm,
            min_line_length_mm: config.topology.min_line_length_mm,
            merge_angle_tolerance_deg: config.topology.merge_angle_tolerance_deg,
            max_gap_bridge_length_mm: config.topology.max_gap_bridge_length_mm,
            algorithm: config.topology.algorithm,
            skip_intersection_check: config.topology.skip_intersection_check,
            enable_parallel: config.topology.enable_parallel,
            parallel_threshold: config.topology.parallel_threshold,
        },
        validator: ValidatorConfig {
            closure_tolerance_mm: config.validator.closure_tolerance_mm,
            min_area_m2: config.validator.min_area_m2,
            min_edge_length_mm: config.validator.min_edge_length_mm,
            min_angle_deg: config.validator.min_angle_deg,
        },
        export: ExportConfig {
            format: config.export.format,
            json_indent: config.export.json_indent as u8,
            auto_validate: config.export.auto_validate,
        },
        vectorize: Some(VectorizeProfileConfig {
            adaptive_threshold: true,
            skeletonize: true,
            text_separation: true,
            quality_assessment: true,
            denoise: true,
            enhance_contrast: true,
        }),
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
    use std::fs::File;
    use std::io::Write;
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
            tracing::info!(
                "📄 收到文件：{:?}, 大小：{:.2} KB",
                file_name,
                file_size as f64 / 1024.0
            );

            // 检查文件大小（最大 50MB）
            if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024 {
                tracing::error!(
                    "❌ 文件过大：{:.2} MB > {} MB",
                    file_size as f64 / (1024.0 * 1024.0),
                    MAX_UPLOAD_SIZE_MB
                );
                return Ok(Json(ProcessResponse {
                    schema_version: default_process_schema_version(),
                    job_id: uuid_simple(),
                    status: ProcessStatus::Failed,
                    message: format!("文件过大，最大支持 {} MB", MAX_UPLOAD_SIZE_MB),
                    result: None,
                    errors: vec![format!(
                        "文件大小 {:.2} MB 超过限制 {} MB",
                        file_size as f64 / (1024.0 * 1024.0),
                        MAX_UPLOAD_SIZE_MB
                    )],
                    edges: None,
                    hatches: None, // P0-4 修复：添加 hatches 字段
                    raster_report: None,
                    semantic_candidates: Vec::new(),
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
            schema_version: default_process_schema_version(),
            job_id: uuid_simple(),
            status: ProcessStatus::Failed,
            message: "不支持的文件格式".to_string(),
            result: None,
            errors: vec![format!(
                "无法识别文件类型，请上传 DXF、PDF 或 PNG/JPG/BMP/TIFF/WebP 文件（文件名：{:?}）",
                file_name
            )],
            edges: None,
            hatches: None, // P0-4 修复：添加 hatches 字段
            raster_report: None,
            semantic_candidates: Vec::new(),
        }));
    }

    // 创建临时文件（保留扩展名以便 Parser 识别）
    let temp_dir = std::env::temp_dir();
    let temp_file_name = format!(
        "cad_process_{}_{}.{}",
        std::process::id(),
        uuid_simple(),
        file_type_extension(&detected_type)
    );
    let temp_path: PathBuf = temp_dir.join(&temp_file_name);
    tracing::info!("📁 创建临时文件：{:?}", temp_path);

    // 写入临时文件
    let mut temp_file = File::create(&temp_path).map_err(|e| {
        tracing::error!("❌ 创建临时文件失败：{}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    temp_file.write_all(&file_data).map_err(|e| {
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
            state
                .pipeline
                .parser()
                .parse_file(&temp_path)
                .map_err(|e| {
                    tracing::error!("❌ DXF 解析失败：{}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?
        }
        FileType::Pdf => {
            // PDF 需要矢量化
            tracing::warn!("  PDF 文件需要矢量化处理，可能需要较长时间");
            state
                .pipeline
                .parser()
                .parse_file(&temp_path)
                .map_err(|e| {
                    tracing::error!("❌ PDF 解析失败：{}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?
        }
        FileType::Png | FileType::Jpeg | FileType::Bmp | FileType::Tiff | FileType::WebP => {
            // 光栅图片直接走矢量化管线，不走 Parser
            tracing::info!("  光栅图片矢量化处理...");
            let response =
                process_raster_temp_file(&state, &temp_path, RasterProcessingOptions::default())
                    .await;
            let _ = std::fs::remove_file(&temp_path);
            return Ok(Json(response));
        }
        FileType::Unknown => {
            // 已经在上面处理过，这里不会到达
            unreachable!()
        }
    };

    // 从解析结果提取原始边和 HATCH
    let entities = parse_result.into_entities();
    let edges = entities_to_edges(&entities);
    let hatches = entities_to_hatches(&entities); // P0-4 新增：提取 HATCH 数据
    tracing::info!(
        "  ✅ 提取 {} 条原始边，{} 个 HATCH",
        edges.len(),
        hatches.len()
    );

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

                // 更新交互服务，同时设置 scene_state 和 topology_ready
                let mut new_interact = InteractionService::new(topo_edges.clone());
                new_interact.set_scene_state(process_result.scene.clone());
                new_interact.get_state_mut().topology_ready = true;
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
        schema_version: default_process_schema_version(),
        job_id: job_id.clone(),
        status: ProcessStatus::Completed,
        message: format!(
            "快速渲染完成，{} 条边已加载，拓扑构建在后台进行",
            edges.len()
        ),
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
        hatches: Some(hatches), // P0-4 新增：返回 HATCH 数据
        raster_report: None,
        semantic_candidates: Vec::new(),
    }))
}

async fn process_raster_handler(
    State(state): State<ApiState>,
    mut multipart: Multipart,
) -> Result<Json<ProcessResponse>, StatusCode> {
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut options = RasterProcessingOptions::default();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::error!("❌ 解析 multipart 表单失败：{}", e);
        StatusCode::BAD_REQUEST
    })? {
        let name = field.name().unwrap_or("unknown").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(String::from);
                let bytes = field.bytes().await.map_err(|e| {
                    tracing::error!("❌ 读取文件数据失败：{}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

                if bytes.len() > MAX_UPLOAD_SIZE_MB * 1024 * 1024 {
                    return Ok(Json(ProcessResponse {
                        schema_version: default_process_schema_version(),
                        job_id: uuid_simple(),
                        status: ProcessStatus::Failed,
                        message: format!("文件过大，最大支持 {} MB", MAX_UPLOAD_SIZE_MB),
                        result: None,
                        errors: vec![format!(
                            "文件大小 {:.2} MB 超过限制 {} MB",
                            bytes.len() as f64 / (1024.0 * 1024.0),
                            MAX_UPLOAD_SIZE_MB
                        )],
                        edges: None,
                        hatches: None,
                        raster_report: None,
                        semantic_candidates: Vec::new(),
                    }));
                }

                file_data = Some(bytes.to_vec());
            }
            "strategy" | "raster_strategy" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                options.strategy = value.parse::<RasterStrategy>().ok();
            }
            "dpi_override" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                options.dpi_override = parse_pair(&value);
            }
            "dpi_x" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                if let Ok(dpi_x) = value.parse::<f64>() {
                    let (_, dpi_y) = options.dpi_override.unwrap_or((dpi_x, dpi_x));
                    options.dpi_override = Some((dpi_x, dpi_y));
                }
            }
            "dpi_y" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                if let Ok(dpi_y) = value.parse::<f64>() {
                    let (dpi_x, _) = options.dpi_override.unwrap_or((dpi_y, dpi_y));
                    options.dpi_override = Some((dpi_x, dpi_y));
                }
            }
            "known_distance_px" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                if let Ok(px) = value.parse::<f64>() {
                    let known_mm = options
                        .scale_calibration
                        .as_ref()
                        .map(|c| c.known_distance_mm)
                        .unwrap_or(0.0);
                    options.scale_calibration = Some(ScaleCalibration {
                        known_distance_px: px,
                        known_distance_mm: known_mm,
                        points_px: None,
                    });
                }
            }
            "known_distance_mm" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                if let Ok(mm) = value.parse::<f64>() {
                    let known_px = options
                        .scale_calibration
                        .as_ref()
                        .map(|c| c.known_distance_px)
                        .unwrap_or(0.0);
                    options.scale_calibration = Some(ScaleCalibration {
                        known_distance_px: known_px,
                        known_distance_mm: mm,
                        points_px: None,
                    });
                }
            }
            "debug_artifacts" | "debug" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                options.debug_artifacts = parse_bool(&value);
            }
            "semantic_mode" => {
                options.semantic_mode =
                    Some(field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            "ocr_backend" => {
                options.ocr_backend =
                    Some(field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?);
            }
            "max_retries" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                options.max_retries = value.parse::<usize>().ok();
            }
            _ => {}
        }
    }

    let file_data = file_data.ok_or_else(|| {
        tracing::warn!("⚠️ /process/raster 请求中未找到文件字段");
        StatusCode::BAD_REQUEST
    })?;

    let detected_type = detect_file_type(&file_data, file_name.as_deref());
    if !detected_type.is_raster() {
        return Ok(Json(ProcessResponse {
            schema_version: default_process_schema_version(),
            job_id: uuid_simple(),
            status: ProcessStatus::Failed,
            message: "不支持的光栅图片格式".to_string(),
            result: None,
            errors: vec![format!(
                "/process/raster 仅支持 PNG/JPG/BMP/TIFF/WebP，检测到：{:?}",
                detected_type
            )],
            edges: None,
            hatches: None,
            raster_report: None,
            semantic_candidates: Vec::new(),
        }));
    }

    let temp_file_name = format!(
        "cad_raster_{}_{}.{}",
        std::process::id(),
        uuid_simple(),
        file_type_extension(&detected_type)
    );
    let temp_path: PathBuf = std::env::temp_dir().join(temp_file_name);

    let mut temp_file = File::create(&temp_path).map_err(|e| {
        tracing::error!("❌ 创建临时文件失败：{}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    temp_file.write_all(&file_data).map_err(|e| {
        tracing::error!("❌ 写入临时文件失败：{}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    drop(temp_file);

    let response = process_raster_temp_file(&state, &temp_path, options).await;
    let _ = std::fs::remove_file(&temp_path);

    Ok(Json(response))
}

async fn process_raster_temp_file(
    state: &ApiState,
    temp_path: &std::path::Path,
    options: RasterProcessingOptions,
) -> ProcessResponse {
    match state
        .pipeline
        .process_raster_file_with_options(temp_path, options)
        .await
    {
        Ok(process_result) => {
            let topo_edges = scene_to_edges(&process_result.scene);
            let mut new_interact = InteractionService::new(topo_edges.clone());
            new_interact.set_scene_state(process_result.scene.clone());
            new_interact.get_state_mut().topology_ready = true;
            *state.interact.lock().await = new_interact;

            ProcessResponse {
                schema_version: default_process_schema_version(),
                job_id: uuid_simple(),
                status: ProcessStatus::Completed,
                message: format!("光栅矢量化完成，提取 {} 条边", topo_edges.len()),
                result: Some(ProcessResult {
                    scene_summary: SceneSummary {
                        outer_boundaries: process_result.scene.outer.as_ref().map_or(0, |_| 1),
                        holes: process_result.scene.holes.len(),
                        total_points: process_result
                            .scene
                            .outer
                            .as_ref()
                            .map_or(0, |o| o.points.len())
                            + process_result
                                .scene
                                .holes
                                .iter()
                                .map(|h| h.points.len())
                                .sum::<usize>(),
                    },
                    validation_summary: ValidationSummary {
                        error_count: process_result.validation.summary.error_count,
                        warning_count: process_result.validation.summary.warning_count,
                        passed: process_result.validation.passed,
                    },
                    output_size: process_result.output_bytes.len(),
                }),
                errors: vec![],
                edges: Some(topo_edges),
                hatches: None,
                raster_report: process_result.raster_report,
                semantic_candidates: process_result.semantic_candidates,
            }
        }
        Err(e) => {
            tracing::error!("❌ 光栅矢量化失败：{}", e);
            ProcessResponse {
                schema_version: default_process_schema_version(),
                job_id: uuid_simple(),
                status: ProcessStatus::Failed,
                message: format!("矢量化失败：{}", e),
                result: None,
                errors: vec![e.to_string()],
                edges: None,
                hatches: None,
                raster_report: Some(RasterVectorizationReport::failed(
                    0,
                    0,
                    "raster_pipeline_failed",
                    e.to_string(),
                )),
                semantic_candidates: Vec::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use http::Request;
    use image::DynamicImage;
    use std::io::Cursor;
    use tower::util::ServiceExt;
    use vectorize::test_data::{generate_test_image, DrawingType, QualityConfig};

    fn make_test_png() -> Vec<u8> {
        let img = generate_test_image(
            DrawingType::Architectural,
            &QualityConfig::default(),
            256,
            256,
        );
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageLuma8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    fn multipart_png_body(boundary: &str, png: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"square.png\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
        body.extend_from_slice(png);
        body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());
        body
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        use interact::InteractionService;
        use std::sync::Arc;
        use tokio::sync::Mutex;

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

    #[test]
    fn test_detect_file_type_png_magic() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A];
        assert_eq!(detect_file_type(&data, None), FileType::Png);
    }

    #[test]
    fn test_detect_file_type_jpeg_magic() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_file_type(&data, None), FileType::Jpeg);
    }

    #[test]
    fn test_detect_file_type_bmp_magic() {
        let data = [0x42, 0x4D, 0x00, 0x00];
        assert_eq!(detect_file_type(&data, None), FileType::Bmp);
    }

    #[test]
    fn test_detect_file_type_tiff_magic() {
        let data_le = [0x49, 0x49, 0x2A, 0x00];
        let data_be = [0x4D, 0x4D, 0x00, 0x2A];
        assert_eq!(detect_file_type(&data_le, None), FileType::Tiff);
        assert_eq!(detect_file_type(&data_be, None), FileType::Tiff);
    }

    #[test]
    fn test_detect_file_type_raster_extension() {
        let data = b"not_an_image_header";
        assert_eq!(detect_file_type(data, Some("test.png")), FileType::Png);
        assert_eq!(detect_file_type(data, Some("photo.jpg")), FileType::Jpeg);
        assert_eq!(detect_file_type(data, Some("scan.bmp")), FileType::Bmp);
        assert_eq!(detect_file_type(data, Some("image.tiff")), FileType::Tiff);
    }

    #[test]
    fn test_detect_file_type_webp_magic() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"WEBP");
        assert_eq!(detect_file_type(&data, None), FileType::WebP);
    }

    #[tokio::test]
    async fn test_process_raster_endpoint_upload_png() {
        use interact::InteractionService;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let state = ApiState {
            pipeline: ProcessingPipeline::new(),
            interact: Arc::new(Mutex::new(InteractionService::new(vec![]))),
        };
        let app = create_router().with_state(state);
        let boundary = "CADBOUNDARY";
        let body = multipart_png_body(boundary, &make_test_png());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/process/raster")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={}", boundary),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: ProcessResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload.status, ProcessStatus::Completed);
        assert!(payload.result.is_some());
        assert!(payload
            .edges
            .as_ref()
            .is_some_and(|edges| !edges.is_empty()));
    }
}
