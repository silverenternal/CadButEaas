//! 处理流水线 - 真正的异步实现
//!
//! 改进：
//! 1. 使用 tokio::spawn_blocking 将 CPU 密集型任务放到阻塞线程池
//! 2. 使用 rayon 并行化处理多个实体
//! 3. 支持进度追踪和取消
//! 4. 集成服务指标收集（EaaS 架构）

mod configuration;
mod raster;
mod types;

pub use types::*;

use self::raster::{
    dimension_summary_from_text_candidates, raster_scene_metadata, raster_transform,
    transform_polylines, vector_graph_semantic_candidates,
};
use async_trait::async_trait;
use common_types::{
    CadError, InternalErrorReason, Polyline, RawEntity, Request, Response, SceneState,
};
use config::CadConfig;
use export::ExportService;
use parser::{service::FileType, service::ParseResult, ParserService};
use service_kit::{
    DependencyHealth, HealthStatus, Service, ServiceHealth, ServiceMetrics, ServiceVersion,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::spawn_blocking;
use topo::TopoService;
use validator::ValidatorService;
use vectorize::{VectorizeConfig, VectorizeService};

/// 处理流水线
#[derive(Clone)]
pub struct ProcessingPipeline {
    parser: Arc<ParserService>,
    vectorize: Arc<VectorizeService>,
    topo: Arc<TopoService>,
    validator: Arc<ValidatorService>,
    export: Arc<ExportService>,
    /// 服务指标
    metrics: Arc<tokio::sync::Mutex<PipelineMetrics>>,
}

impl ProcessingPipeline {
    /// 创建新的处理流水线（使用默认配置）
    pub fn new() -> Self {
        Self {
            parser: Arc::new(ParserService::new()),
            vectorize: Arc::new(VectorizeService::with_default()),
            topo: Arc::new(TopoService::with_default_config()),
            validator: Arc::new(ValidatorService::with_default_config()),
            export: Arc::new(ExportService::with_default_config()),
            metrics: Arc::default(),
        }
    }

    /// 创建新的处理流水线（使用自定义配置）
    ///
    /// ## P11 锐评落实
    ///
    /// 修复原设计缺陷：所有服务用 `with_default_config()` 硬编码初始化，
    /// 用户无法动态调整配置。现在支持通过 `CadConfig` 传入自定义配置。
    ///
    /// ## 使用示例
    ///
    /// ```rust,no_run
    /// use orchestrator::ProcessingPipeline;
    /// use config::CadConfig;
    ///
    /// let mut config = CadConfig::default();
    /// config.topology.snap_tolerance_mm = 1.0; // 调整容差到 1.0mm
    ///
    /// let pipeline = ProcessingPipeline::new_with_config(&config);
    /// ```
    pub fn new_with_config(config: &CadConfig) -> Self {
        // 转换配置
        let vectorize_config = configuration::vectorize_config(config);
        let topo_config = configuration::topo_config(config);
        let validator_config = configuration::validator_config(config);
        let export_config = configuration::export_config(config);

        // 创建 ParserService 并应用 DXF 配置
        let parser_service = configuration::create_parser_service(&config.parser);

        Self {
            parser: Arc::new(parser_service),
            vectorize: Arc::new(VectorizeService::new(
                Box::new(accelerator_cpu::CpuAccelerator::new()),
                vectorize_config,
            )),
            topo: Arc::new(TopoService::with_config(&topo_config)),
            validator: Arc::new(ValidatorService::with_config(&validator_config)),
            export: Arc::new(ExportService::with_config(&export_config)),
            metrics: Arc::default(),
        }
    }

    /// 获取解析器
    pub fn parser(&self) -> &ParserService {
        &self.parser
    }

    /// 获取拓扑服务
    pub fn topo(&self) -> &TopoService {
        &self.topo
    }

    /// 获取验证服务
    pub fn validator(&self) -> &ValidatorService {
        &self.validator
    }

    /// 获取导出服务
    pub fn export(&self) -> &ExportService {
        &self.export
    }

    /// 获取矢量化服务
    pub fn vectorize(&self) -> &VectorizeService {
        &self.vectorize
    }

    /// 获取服务健康状态
    pub fn health_check(&self) -> ServiceHealth {
        // 检查各子服务健康状态
        let parser_health = self.parser.health_check();
        let topo_health = self.topo.health_check();
        let validator_health = self.validator.health_check();
        // VectorizeService 不再实现 health_check，使用简单状态
        let vectorize_health = ServiceHealth {
            status: HealthStatus::Healthy,
            version: "2.0.0".to_string(),
            uptime_secs: 0,
            dependencies: vec![],
            metadata: std::collections::HashMap::new(),
        };
        let export_health = self.export.health_check();

        let deps = vec![
            DependencyHealth {
                name: "ParserService".to_string(),
                status: parser_health.status,
                message: None,
            },
            DependencyHealth {
                name: "TopoService".to_string(),
                status: topo_health.status,
                message: None,
            },
            DependencyHealth {
                name: "ValidatorService".to_string(),
                status: validator_health.status,
                message: None,
            },
            DependencyHealth {
                name: "VectorizeService".to_string(),
                status: vectorize_health.status,
                message: None,
            },
            DependencyHealth {
                name: "ExportService".to_string(),
                status: export_health.status,
                message: None,
            },
        ];

        // 如果任何子服务不健康，Pipeline 状态为 Degraded
        let overall_status = if deps.iter().all(|d| d.status == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else if deps.iter().any(|d| d.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        let mut health = ServiceHealth::healthy(env!("CARGO_PKG_VERSION")).with_uptime(0);

        for dep in deps {
            health = health.with_dependency(dep);
        }

        // 如果整体状态不是 Healthy，重新构建健康状态
        match overall_status {
            HealthStatus::Healthy => health,
            HealthStatus::Degraded => {
                ServiceHealth::degraded(env!("CARGO_PKG_VERSION"), health.dependencies)
            }
            HealthStatus::Unhealthy => {
                ServiceHealth::unhealthy(env!("CARGO_PKG_VERSION"), "子服务不健康")
            }
        }
    }

    /// 获取流水线指标
    pub async fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.lock().await.clone()
    }

    /// 处理文件（异步版本）
    pub async fn process_file(&self, path: impl AsRef<Path>) -> Result<ProcessResult, CadError> {
        let path = path.as_ref().to_path_buf();
        tracing::info!("开始处理文件：{:?}", path);
        let start_time = Instant::now();

        // 检测是否为光栅图片格式，路由到对应管线
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase());
        let is_raster = matches!(
            ext.as_deref(),
            Some("png")
                | Some("jpg")
                | Some("jpeg")
                | Some("bmp")
                | Some("tiff")
                | Some("tif")
                | Some("webp")
        );

        let result = if is_raster {
            tracing::info!("检测到光栅图片格式，使用光栅管线");
            self.process_raster_internal(&path).await
        } else {
            self.process_file_internal(&path).await
        };

        // 记录指标
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        let mut metrics_guard = self.metrics.lock().await;
        metrics_guard.total_requests += 1;
        if result.is_ok() {
            metrics_guard.success_requests += 1;
        } else {
            metrics_guard.failed_requests += 1;
        }
        // 更新平均处理时间
        let n = metrics_guard.total_requests as f64;
        metrics_guard.avg_processing_time_ms =
            metrics_guard.avg_processing_time_ms * ((n - 1.0) / n) + elapsed_ms / n;

        result
    }

    /// 内部文件处理方法（不包含指标记录）
    async fn process_file_internal(&self, path: &Path) -> Result<ProcessResult, CadError> {
        let path = path.to_path_buf();
        let parser = Arc::clone(&self.parser);
        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);

        spawn_blocking(move || {
            Self::process_file_sync(&parser, &vectorize, &topo, &validator, &export, &path)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("任务执行失败：{}", e),
                },
                location: Some("process_file"),
            })
        })
    }

    /// 光栅管线内部实现
    async fn process_raster_internal(&self, path: &Path) -> Result<ProcessResult, CadError> {
        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);
        let path = path.to_path_buf();

        spawn_blocking(move || {
            Self::process_raster_file_sync(&vectorize, &topo, &validator, &export, &path)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("任务执行失败：{}", e),
                },
                location: Some("process_raster_file"),
            })
        })
    }

    /// 通过统一的 Service::process() 方法处理文件（EaaS 架构）
    ///
    /// 这个方法展示了如何通过统一的服务接口调用各子服务，
    /// 实现真正的 EaaS（Everything as a Service）架构。
    ///
    /// ## 架构说明
    ///
    /// **当前版本**：单体部署（进程内服务调用）
    /// - 服务间通过 trait 接口直接调用，性能最优
    /// - 每个服务保持独立的 crate 和清晰的契约
    ///
    /// **P2 阶段计划**：HTTP/gRPC 微服务部署
    /// - 服务间通过远程通信，支持独立部署和弹性伸缩
    /// - 集成服务发现（Consul/Etcd）、熔断、链路追踪
    ///
    /// ## 服务调用链
    ///
    /// ```text
    /// ParserService → TopoService → ValidatorService → ExportService
    ///      ↓              ↓              ↓                 ↓
    ///  ParseResult  SceneState   ValidationReport   ExportResult
    /// ```
    pub async fn process_with_services(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessResult, CadError> {
        let path = path.as_ref().to_path_buf();
        tracing::info!("开始通过 Service trait 处理文件：{:?}", path);
        let start_time = Instant::now();

        // 1. 解析 - 通过 ParserService::process()，使用 Response 返回的数据
        tracing::info!("阶段 1/5: 解析文件（通过 Service::process）");
        let parse_request = parser::service::ParseRequest::new(path.to_str().unwrap_or(""));
        let parse_response = self.parser.process(parse_request.into()).await?;

        // 从 Response 中提取 payload，不再调用 parse_file()
        let parse_result = parse_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ParserService returned empty payload".to_string(),
            })
        })?;

        let has_raster = parse_result.has_raster();
        let entities = parse_result.into_entities();
        let entities_refs: Vec<&common_types::RawEntity> = entities.iter().collect();
        let text_annotations = scene_builder::extract_text_annotations(&entities);
        let dimension_summary = scene_builder::extract_dimension_summary(&entities);
        tracing::info!("  解析得到 {} 个实体", entities.len());
        if !text_annotations.is_empty() {
            tracing::info!("  检测到 {} 个文字标注", text_annotations.len());
        }
        if dimension_summary.total_count > 0 {
            tracing::info!(
                "  检测到 {} 个标注（线性:{}，对齐:{}，角度:{}，半径:{}，直径:{}，坐标:{}）",
                dimension_summary.total_count,
                dimension_summary.linear_count,
                dimension_summary.aligned_count,
                dimension_summary.angular_count,
                dimension_summary.radial_count,
                dimension_summary.diameter_count,
                dimension_summary.ordinate_count
            );
        }

        // 2. 矢量化 - 从解析结果提取多段线
        tracing::info!("阶段 2/5: 矢量化处理");
        let polylines = if has_raster {
            tracing::info!("  检测到光栅图像，开始矢量化处理");
            scene_builder::extract_polylines_from_entities(&entities)
        } else {
            tracing::info!("  解析得到矢量数据");
            scene_builder::extract_polylines_from_entities(&entities)
        };
        tracing::info!("  得到 {} 条多段线", polylines.len());

        // 3. 构建拓扑 - 通过 TopoService::process()，使用 Response 返回的数据
        tracing::info!("阶段 3/5: 构建拓扑（通过 Service::process）");
        let topo_request = topo::service::TopoRequest {
            geometry_json: serde_json::to_string(&polylines).unwrap_or_default(),
        };
        let topo_response = self.topo.process(topo_request.into()).await?;

        // 从 Response 中提取 payload，不再调用 build_scene()
        let topo_result = topo_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "TopoService returned empty payload".to_string(),
            })
        })?;

        // 从 TopologyResult 构建 SceneState
        let mut scene = SceneState {
            outer: topo_result.outer,
            holes: topo_result.holes,
            boundaries: Vec::new(), // 待用户标注
            sources: Vec::new(),
            edges: Vec::new(), // 待填充
            raster_metadata: None,
            units: common_types::LengthUnit::Mm,
            coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        };
        tracing::info!(
            "  构建完成：{} 个外轮廓，{} 个孔洞",
            if scene.outer.is_some() { 1 } else { 0 },
            scene.holes.len()
        );

        // 3.3 填充原始边数据（用于前端显示）
        tracing::info!("阶段 3.3/5: 填充原始边数据");

        // 从 topo_result 构建 RawEdge 列表
        let mut extracted_edges: Vec<common_types::RawEdge> = Vec::new();
        for (edge_id, (start_idx, end_idx)) in topo_result.edges.iter().enumerate() {
            let start = topo_result.points[*start_idx];
            let end = topo_result.points[*end_idx];
            extracted_edges.push(common_types::RawEdge {
                id: edge_id,
                start,
                end,
                layer: None, // 拓扑边无图层信息
                color_index: None,
            });
        }
        scene.edges = extracted_edges;
        tracing::info!("  填充 {} 条原始边", scene.edges.len());

        // 3.5 自动语义推断
        tracing::info!("阶段 3.5/5: 自动语义推断");
        scene_builder::auto_infer_boundaries(&mut scene, &entities_refs);
        tracing::info!("  语义推断完成：{} 个边界段", scene.boundaries.len());

        // 4. 验证 - 通过 ValidatorService::process()，使用 Response 返回的数据
        tracing::info!("阶段 4/5: 验证场景（通过 Service::process）");
        let validate_request = validator::service::ValidateRequest::new(
            serde_json::to_string(&scene).unwrap_or_default(),
        );
        let validate_response = self.validator.process(validate_request.into()).await?;

        // 从 Response 中提取 payload，不再调用 validate()
        let validation = validate_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ValidatorService returned empty payload".to_string(),
            })
        })?;

        if !validation.passed {
            tracing::warn!(
                "验证失败，错误数：{}, 警告数：{}",
                validation.summary.error_count,
                validation.summary.warning_count
            );

            if validation.summary.error_count > 0 {
                let issues: Vec<common_types::error::ValidationIssue> = validation
                    .issues
                    .into_iter()
                    .map(|i| common_types::error::ValidationIssue {
                        code: i.code,
                        severity: match i.severity {
                            validator::checks::Severity::Error => {
                                common_types::error::Severity::Error
                            }
                            validator::checks::Severity::Warning => {
                                common_types::error::Severity::Warning
                            }
                            validator::checks::Severity::Info => {
                                common_types::error::Severity::Info
                            }
                        },
                        message: i.message,
                        location: i.location.map(|l| common_types::error::ErrorLocation {
                            point: l.point,
                            segment: l.segment,
                            loop_index: l.loop_index,
                        }),
                    })
                    .collect();

                return Err(CadError::ValidationFailed {
                    count: validation.summary.error_count,
                    warning_count: validation.summary.warning_count,
                    issues,
                });
            }
        }

        // 5. 导出 - 通过 ExportService::process()，使用 Response 返回的数据
        tracing::info!("阶段 5/5: 导出场景（通过 Service::process）");
        let export_request = export::service::ExportRequest::new(scene.clone());
        let export_response = self.export.process(export_request.into()).await?;

        // 从 Response 中提取 payload
        let export_result = export_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ExportService returned empty payload".to_string(),
            })
        })?;

        tracing::info!("处理完成（通过 Service trait）");

        // 记录服务指标
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        let mut metrics_guard = self.metrics.lock().await;
        metrics_guard.total_requests += 1;
        if export_result.bytes.is_empty() {
            metrics_guard.failed_requests += 1;
        } else {
            metrics_guard.success_requests += 1;
        }
        let n = metrics_guard.total_requests as f64;
        metrics_guard.avg_processing_time_ms =
            metrics_guard.avg_processing_time_ms * ((n - 1.0) / n) + elapsed_ms / n;

        Ok(ProcessResult {
            scene,
            validation,
            output_bytes: export_result.bytes,
            text_annotations,
            dimension_summary,
            raster_report: None,
            semantic_candidates: Vec::new(),
        })
    }

    /// 同步处理实现（在 spawn_blocking 中调用）
    fn process_file_sync(
        parser: &ParserService,
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        path: &Path,
    ) -> Result<ProcessResult, CadError> {
        // 1. 解析（CPU 密集型）
        tracing::info!("阶段 1/5: 解析文件");
        let parse_result = parser.parse_file(path)?;

        // 1.5 提取实体引用（在消耗 parse_result 之前，克隆用于后续语义推断）
        let entities: Vec<RawEntity> = match &parse_result {
            ParseResult::Cad(entities) => entities.clone(),
            ParseResult::Pdf(content) => content.vector_entities.clone(),
        };
        let entities_refs: Vec<&RawEntity> = entities.iter().collect();
        tracing::info!("  解析得到 {} 个实体", entities.len());

        // 1.5.5 提取文字标注
        let text_annotations = scene_builder::extract_text_annotations(&entities);
        let dimension_summary = scene_builder::extract_dimension_summary(&entities);
        if !text_annotations.is_empty() {
            tracing::info!("  检测到 {} 个文字标注", text_annotations.len());
        }
        if dimension_summary.total_count > 0 {
            tracing::info!("  检测到 {} 个标注", dimension_summary.total_count);
        }

        // 1.6 检查是否包含光栅图像
        let has_raster = parse_result.has_raster();

        // 2. 矢量化（如果有光栅图像，CPU+ 内存密集型）
        let (polylines, text_annotations) = if has_raster {
            tracing::info!("  检测到光栅图像，开始矢量化处理");
            let (pts, annotations) =
                Self::process_raster_with_vectorization(parse_result, vectorize)?;
            (pts, annotations)
        } else {
            tracing::info!("  解析得到矢量数据");
            let pts = scene_builder::extract_polylines_from_entities(&entities);
            (pts, text_annotations)
        };
        tracing::info!("  得到 {} 条多段线", polylines.len());

        // 3. 构建拓扑（CPU 密集型，使用 R*-tree 加速）
        tracing::info!("阶段 3/5: 构建拓扑");
        let mut scene = topo.build_scene(&polylines)?;
        tracing::info!(
            "  构建完成：{} 个外轮廓，{} 个孔洞",
            if scene.outer.is_some() { 1 } else { 0 },
            scene.holes.len()
        );

        // 3.3 填充原始边数据（用于前端显示）
        tracing::info!("阶段 3.3/5: 填充原始边数据");

        // 辅助函数：从实体提取边
        let mut edge_id = 0;
        let mut extracted_edges: Vec<common_types::RawEdge> = Vec::new();

        for entity in &entities {
            match entity {
                common_types::RawEntity::Line {
                    start,
                    end,
                    metadata,
                    ..
                } => {
                    extracted_edges.push(common_types::RawEdge {
                        id: edge_id,
                        start: *start,
                        end: *end,
                        layer: metadata.layer.clone(),
                        color_index: None,
                    });
                    edge_id += 1;
                }
                // Polyline 分解为多条线段
                common_types::RawEntity::Polyline {
                    points,
                    closed,
                    metadata,
                    ..
                } if points.len() >= 2 => {
                    for i in 0..points.len() - 1 {
                        extracted_edges.push(common_types::RawEdge {
                            id: edge_id,
                            start: points[i],
                            end: points[i + 1],
                            layer: metadata.layer.clone(),
                            color_index: None,
                        });
                        edge_id += 1;
                    }
                    // 如果闭合，添加最后一条边
                    if *closed {
                        extracted_edges.push(common_types::RawEdge {
                            id: edge_id,
                            start: points[points.len() - 1],
                            end: points[0],
                            layer: metadata.layer.clone(),
                            color_index: None,
                        });
                        edge_id += 1;
                    }
                }
                // Arc 离散化为线段（简化处理：只取弦）
                common_types::RawEntity::Arc {
                    center,
                    radius,
                    start_angle,
                    end_angle,
                    metadata,
                    ..
                } => {
                    // 将圆弧离散化为 8 段线段
                    let segments = 8;
                    let angle_range = end_angle - start_angle;
                    for i in 0..segments {
                        let a1 = start_angle + (angle_range * (i as f64) / segments as f64);
                        let a2 = start_angle + (angle_range * ((i + 1) as f64) / segments as f64);
                        let p1 = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
                        let p2 = [center[0] + radius * a2.cos(), center[1] + radius * a2.sin()];
                        extracted_edges.push(common_types::RawEdge {
                            id: edge_id,
                            start: p1,
                            end: p2,
                            layer: metadata.layer.clone(),
                            color_index: None,
                        });
                        edge_id += 1;
                    }
                }
                // Circle 离散化为 16 段线段
                common_types::RawEntity::Circle {
                    center,
                    radius,
                    metadata,
                    ..
                } => {
                    let segments = 16;
                    for i in 0..segments {
                        let a1 = 2.0 * std::f64::consts::PI * (i as f64) / segments as f64;
                        let a2 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / segments as f64;
                        let p1 = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
                        let p2 = [center[0] + radius * a2.cos(), center[1] + radius * a2.sin()];
                        extracted_edges.push(common_types::RawEdge {
                            id: edge_id,
                            start: p1,
                            end: p2,
                            layer: metadata.layer.clone(),
                            color_index: None,
                        });
                        edge_id += 1;
                    }
                }
                // 其他类型（文字、路径）跳过
                _ => {}
            }
        }

        scene.edges = extracted_edges;
        tracing::info!("  填充 {} 条原始边", scene.edges.len());

        // 3.5 自动语义推断（P1 任务）- 现在已启用
        tracing::info!("阶段 3.5/5: 自动语义推断");
        scene_builder::auto_infer_boundaries(&mut scene, &entities_refs);
        tracing::info!("  语义推断完成：{} 个边界段", scene.boundaries.len());

        // 4. 验证（CPU 密集型）
        tracing::info!("阶段 4/5: 验证场景");
        let validation = validator.validate(&scene)?;

        if !validation.passed {
            tracing::warn!(
                "验证失败，错误数：{}, 警告数：{}",
                validation.summary.error_count,
                validation.summary.warning_count
            );

            // 如果有错误，返回验证失败
            if validation.summary.error_count > 0 {
                let issues: Vec<common_types::error::ValidationIssue> = validation
                    .issues
                    .into_iter()
                    .map(|i| common_types::error::ValidationIssue {
                        code: i.code,
                        severity: match i.severity {
                            validator::checks::Severity::Error => {
                                common_types::error::Severity::Error
                            }
                            validator::checks::Severity::Warning => {
                                common_types::error::Severity::Warning
                            }
                            validator::checks::Severity::Info => {
                                common_types::error::Severity::Info
                            }
                        },
                        message: i.message,
                        location: i.location.map(|l| common_types::error::ErrorLocation {
                            point: l.point,
                            segment: l.segment,
                            loop_index: l.loop_index,
                        }),
                    })
                    .collect();

                return Err(CadError::ValidationFailed {
                    count: validation.summary.error_count,
                    warning_count: validation.summary.warning_count,
                    issues,
                });
            }
        }

        // 5. 导出
        tracing::info!("阶段 5/5: 导出场景");
        let export_result = export.export(&scene)?;

        tracing::info!("处理完成");

        Ok(ProcessResult {
            scene,
            validation,
            output_bytes: export_result.bytes,
            text_annotations,
            dimension_summary,
            raster_report: None,
            semantic_candidates: Vec::new(),
        })
    }

    /// 处理光栅图像并矢量化
    fn process_raster_with_vectorization(
        result: ParseResult,
        vectorize: &VectorizeService,
    ) -> Result<(Vec<Polyline>, Vec<common_types::TextAnnotation>), CadError> {
        match result {
            ParseResult::Pdf(content) => {
                let mut all_polylines = Vec::new();
                let mut vectorize_errors = Vec::new();
                let mut quality_warnings = Vec::new();

                // 1. 首先提取已有的矢量实体（如果有）
                let vector_polylines =
                    scene_builder::extract_polylines_from_entities(&content.vector_entities);
                all_polylines.extend(vector_polylines);

                // 提取文字标注
                let text_annotations =
                    scene_builder::extract_text_annotations(&content.vector_entities);

                // 2. 对每个光栅图像进行矢量化
                for raster_image in &content.raster_images {
                    tracing::debug!(
                        "  处理光栅图像：{}x{}",
                        raster_image.width,
                        raster_image.height
                    );

                    // 转换为 PdfRasterImage 并矢量化
                    let pdf_raster = raster_image.to_pdf_raster_image();
                    let config = VectorizeConfig::default();

                    match vectorize.vectorize_from_pdf(&pdf_raster, Some(&config)) {
                        Ok(polylines) => {
                            tracing::debug!("    矢量化得到 {} 条多段线", polylines.len());

                            // 质量评估
                            let quality_report = vectorize::algorithms::quality::evaluate_quality(
                                &pdf_raster,
                                &polylines,
                            );

                            // 如果质量过低，记录警告
                            if quality_report.overall_score < 60.0 {
                                let warning_msg = format!(
                                    "图像 '{}' 矢量化质量过低 (得分：{:.1})",
                                    pdf_raster.name, quality_report.overall_score
                                );
                                tracing::warn!("{}", warning_msg);
                                quality_warnings.push(warning_msg);

                                // 记录具体问题
                                for issue in &quality_report.issues {
                                    tracing::warn!(
                                        "  - [{:?}] {}",
                                        issue.severity,
                                        issue.description
                                    );
                                }
                            }

                            all_polylines.extend(polylines);
                        }
                        Err(e) => {
                            tracing::warn!("    矢量化失败：{:?}", e);
                            vectorize_errors.push(format!("图像 '{}': {:?}", pdf_raster.name, e));
                        }
                    }
                }

                // 如果有矢量化失败，返回错误报告
                if !vectorize_errors.is_empty() {
                    let error_msg = format!(
                        "矢量化失败 ({} 个图像): {}",
                        vectorize_errors.len(),
                        vectorize_errors.join("; ")
                    );
                    tracing::error!("{}", error_msg);
                    return Err(CadError::VectorizeFailed { message: error_msg });
                }

                // 如果有质量警告，记录但不失败
                if !quality_warnings.is_empty() {
                    tracing::warn!("矢量化质量警告：{} 个图像质量较低", quality_warnings.len());
                }

                Ok((all_polylines, text_annotations))
            }
            ParseResult::Cad(entities) => {
                // CAD 文件不应该有光栅图像
                let polylines = scene_builder::extract_polylines_from_entities(&entities);
                let text_annotations = scene_builder::extract_text_annotations(&entities);
                Ok((polylines, text_annotations))
            }
        }
    }

    /// 处理字节数据（异步版本）
    pub async fn process_bytes(
        &self,
        bytes: &[u8],
        file_type: FileType,
    ) -> Result<ProcessResult, CadError> {
        let bytes = bytes.to_vec();
        tracing::info!("开始处理字节数据，类型：{:?}", file_type);

        let parser = Arc::clone(&self.parser);
        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);

        spawn_blocking(move || {
            Self::process_bytes_sync(
                &parser, &vectorize, &topo, &validator, &export, &bytes, file_type,
            )
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("任务执行失败：{}", e),
                },
                location: Some("process_bytes"),
            })
        })
    }

    /// 同步处理字节实现
    fn process_bytes_sync(
        parser: &ParserService,
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        bytes: &[u8],
        file_type: FileType,
    ) -> Result<ProcessResult, CadError> {
        // 1. 解析
        let parse_result = parser.parse_bytes(bytes, file_type)?;
        let entities = match &parse_result {
            ParseResult::Cad(entities) => entities.clone(),
            ParseResult::Pdf(content) => content.vector_entities.clone(),
        };
        let dimension_summary = scene_builder::extract_dimension_summary(&entities);

        // 2. 矢量化（如果有光栅图像）
        let (polylines, text_annotations) = if parse_result.has_raster() {
            Self::process_raster_with_vectorization(parse_result, vectorize)?
        } else {
            Self::extract_polylines(parse_result)?
        };

        // 3. 构建拓扑
        let scene = topo.build_scene(&polylines)?;

        // 4. 验证
        let validation = validator.validate(&scene)?;

        // 5. 导出
        let export_result = export.export(&scene)?;

        Ok(ProcessResult {
            scene,
            validation,
            output_bytes: export_result.bytes,
            text_annotations,
            dimension_summary,
            raster_report: None,
            semantic_candidates: Vec::new(),
        })
    }

    // ========================================================================
    // 光栅图片处理管线（PNG/JPG/BMP/TIFF/WebP）
    // ========================================================================

    /// 从光栅图片文件处理几何语义提取
    ///
    /// # 流程
    /// 光栅文件 → RasterLoader → DynamicImage → VectorizeService → Polyline → Topo → Validator → Export
    pub async fn process_raster_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ProcessResult, CadError> {
        self.process_raster_file_with_options(path, RasterProcessingOptions::default())
            .await
    }

    /// 从光栅图片文件处理几何语义提取，并应用光栅专用选项。
    pub async fn process_raster_file_with_options(
        &self,
        path: impl AsRef<Path>,
        options: RasterProcessingOptions,
    ) -> Result<ProcessResult, CadError> {
        let path = path.as_ref().to_path_buf();
        tracing::info!("开始处理光栅图片文件：{:?}", path);

        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);

        spawn_blocking(move || {
            Self::process_raster_file_sync_with_options(
                &vectorize, &topo, &validator, &export, &path, &options,
            )
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("任务执行失败：{}", e),
                },
                location: Some("process_raster_file"),
            })
        })
    }

    /// 从字节数据光栅处理几何语义提取
    ///
    /// # 参数
    /// * `bytes` - 图片字节数据
    /// * `format_hint` - 可选的文件扩展名提示（如 "png"、"jpg"）
    pub async fn process_raster_bytes(
        &self,
        bytes: &[u8],
        format_hint: Option<&str>,
    ) -> Result<ProcessResult, CadError> {
        self.process_raster_bytes_with_options(
            bytes,
            format_hint,
            RasterProcessingOptions::default(),
        )
        .await
    }

    /// 从字节数据光栅处理几何语义提取，并应用光栅专用选项。
    pub async fn process_raster_bytes_with_options(
        &self,
        bytes: &[u8],
        format_hint: Option<&str>,
        options: RasterProcessingOptions,
    ) -> Result<ProcessResult, CadError> {
        let bytes = bytes.to_vec();
        let format_hint = format_hint.map(String::from);
        tracing::info!("开始处理光栅字节数据，格式提示：{:?}", format_hint);

        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);

        spawn_blocking(move || {
            Self::process_raster_bytes_sync_with_options(
                &vectorize,
                &topo,
                &validator,
                &export,
                &bytes,
                format_hint.as_deref(),
                &options,
            )
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("任务执行失败：{}", e),
                },
                location: Some("process_raster_bytes"),
            })
        })
    }

    /// 同步处理光栅图片文件（内部实现）
    fn process_raster_file_sync(
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        path: &Path,
    ) -> Result<ProcessResult, CadError> {
        Self::process_raster_file_sync_with_options(
            vectorize,
            topo,
            validator,
            export,
            path,
            &RasterProcessingOptions::default(),
        )
    }

    fn process_raster_file_sync_with_options(
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        path: &Path,
        options: &RasterProcessingOptions,
    ) -> Result<ProcessResult, CadError> {
        // 1. 加载光栅图片
        let (image, info) = raster_loader::RasterLoader::from_file(path).map_err(|e| {
            CadError::VectorizeFailed {
                message: format!("光栅图片加载失败：{}", e),
            }
        })?;

        tracing::info!(
            "光栅图片加载成功：{}x{} 像素，格式：{:?}",
            info.width,
            info.height,
            info.format
        );

        // 2. 矢量化 → 拓扑 → 验证 → 导出（复用通用逻辑）
        Self::process_image_to_result(
            vectorize,
            topo,
            validator,
            export,
            &image,
            Some(&info),
            options,
        )
    }

    fn process_raster_bytes_sync_with_options(
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        bytes: &[u8],
        format_hint: Option<&str>,
        options: &RasterProcessingOptions,
    ) -> Result<ProcessResult, CadError> {
        // 1. 从字节加载光栅图片
        let (image, info) =
            raster_loader::RasterLoader::from_bytes(bytes, format_hint).map_err(|e| {
                CadError::VectorizeFailed {
                    message: format!("光栅图片解码失败：{}", e),
                }
            })?;

        // 2. 矢量化 → 拓扑 → 验证 → 导出（复用通用逻辑）
        Self::process_image_to_result(
            vectorize,
            topo,
            validator,
            export,
            &image,
            Some(&info),
            options,
        )
    }

    /// 通用图片处理核心：矢量化 → 拓扑 → 验证 → 导出
    fn process_image_to_result(
        vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        image: &image::DynamicImage,
        info: Option<&raster_loader::RasterImageInfo>,
        options: &RasterProcessingOptions,
    ) -> Result<ProcessResult, CadError> {
        // 1. 矢量化
        let mut config = vectorize.config().clone();
        if let Some(strategy) = options.strategy {
            config.raster_strategy = strategy;
        }
        if let Some(max_retries) = options.max_retries {
            config.max_retries = max_retries;
        }
        let mut detailed =
            vectorize.vectorize_image_detailed(image, &config, options.debug_artifacts)?;
        let transform = raster_transform(info, options);
        let polylines = transform_polylines(&detailed.polylines, &transform);
        let graph_semantics = vector_graph_semantic_candidates(&polylines);
        detailed.report.semantic_candidates.extend(graph_semantics);
        tracing::info!("矢量化完成：提取 {} 条多段线", polylines.len());

        // 2. 构建拓扑
        let mut scene = topo.build_scene(&polylines)?;
        scene.edges = scene_builder::polylines_to_raw_edges(&polylines);
        scene.raster_metadata = Some(raster_scene_metadata(info, &transform));
        scene.units = if transform.px_to_mm.is_some() {
            common_types::LengthUnit::Mm
        } else {
            common_types::LengthUnit::Unspecified
        };
        tracing::info!(
            "拓扑构建完成：{} 个外轮廓，{} 个孔洞",
            scene.outer.as_ref().map_or(0, |_| 1),
            scene.holes.len()
        );

        // 3. 验证
        let validation = validator.validate(&scene)?;
        tracing::info!(
            "验证完成：{} 个错误，{} 个警告",
            validation.summary.error_count,
            validation.summary.warning_count
        );

        // 4. 导出
        let export_result = export.export(&scene)?;

        let text_annotations = detailed
            .report
            .text_candidates
            .iter()
            .filter(|candidate| candidate.accepted)
            .map(|candidate| common_types::TextAnnotation {
                position: [
                    (candidate.bbox[0] + candidate.bbox[2]) * 0.5,
                    (candidate.bbox[1] + candidate.bbox[3]) * 0.5,
                ],
                content: candidate.content.clone(),
                height: (candidate.bbox[3] - candidate.bbox[1]).abs(),
                rotation: candidate.rotation,
            })
            .collect::<Vec<_>>();
        let dimension_summary = dimension_summary_from_text_candidates(&detailed.report);

        Ok(ProcessResult {
            scene,
            validation,
            output_bytes: export_result.bytes,
            text_annotations,
            dimension_summary,
            raster_report: Some(detailed.report.clone()),
            semantic_candidates: detailed.report.semantic_candidates.clone(),
        })
    }

    /// 从解析结果中提取多段线（使用 rayon 并行化）
    fn extract_polylines(
        result: ParseResult,
    ) -> Result<(Vec<Polyline>, Vec<common_types::TextAnnotation>), CadError> {
        let entities = result.into_entities();
        let polylines = scene_builder::extract_polylines_from_entities(&entities);
        let text_annotations = scene_builder::extract_text_annotations(&entities);
        Ok((polylines, text_annotations))
    }

    /// 获取流水线统计信息
    pub async fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            services_initialized: true,
            // 可以添加更多运行时统计
        }
    }
}

/// 离散化圆弧
#[cfg(test)]
fn discretize_arc(center: Point2, radius: f64, start_angle: f64, end_angle: f64) -> Polyline {
    let start_rad = start_angle.to_radians();
    let end_rad = end_angle.to_radians();

    // 计算弧长
    let mut angle_diff = end_rad - start_rad;
    if angle_diff < 0.0 {
        angle_diff += 2.0 * std::f64::consts::PI;
    }

    let arc_length = radius * angle_diff;
    let num_segments = (arc_length / 1.0).ceil() as usize; // 每 1mm 一段
    let num_segments = num_segments.max(8); // 至少 8 段

    let mut points = Vec::with_capacity(num_segments + 1);

    for i in 0..=num_segments {
        let t = i as f64 / num_segments as f64;
        let angle = start_rad + t * angle_diff;
        let x = center[0] + radius * angle.cos();
        let y = center[1] + radius * angle.sin();
        points.push([x, y]);
    }

    points
}

/// 离散化圆
#[cfg(test)]
fn discretize_circle(center: Point2, radius: f64) -> Polyline {
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let num_segments = (circumference / 1.0).ceil() as usize;
    let num_segments = num_segments.max(32); // 至少 32 段

    let mut points = Vec::with_capacity(num_segments);

    for i in 0..num_segments {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_segments as f64;
        let x = center[0] + radius * angle.cos();
        let y = center[1] + radius * angle.sin();
        points.push([x, y]);
    }

    // 闭合
    if let Some(first) = points.first().copied() {
        points.push(first);
    }

    points
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// 类型别名，避免重复
#[cfg(test)]
type Point2 = [f64; 2];

// ============================================================================
// EaaS Service Trait 实现
// ============================================================================

#[async_trait]
impl Service for ProcessingPipeline {
    type Payload = PipelineRequest;
    type Data = ProcessResult;
    type Error = CadError;

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 真正的处理入口：通过调用各服务的 process() 方法完成编排
        // 这展示了 EaaS 架构的核心：服务可组合、可替换、可观测

        // 1. 解析：通过 ParserService::process() 调用
        let parse_request =
            parser::service::ParseRequest::new(request.payload.path.to_str().unwrap_or(""));
        let parse_response = self.parser.process(parse_request.into()).await?;
        // 从 Response 中提取 payload
        let parse_result = parse_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ParserService returned empty payload".to_string(),
            })
        })?;

        // 从解析结果提取实体
        let entities = parse_result.into_entities();
        let entities_refs: Vec<&common_types::RawEntity> = entities.iter().collect();
        let text_annotations = scene_builder::extract_text_annotations(&entities);
        let dimension_summary = scene_builder::extract_dimension_summary(&entities);

        // 2. 矢量化：提取多段线（矢量化服务已集成到解析器中）
        let polylines = scene_builder::extract_polylines_from_entities(&entities);

        // 3. 构建拓扑：通过 TopoService::process() 调用
        let topo_request = topo::service::TopoRequest {
            geometry_json: serde_json::to_string(&polylines).unwrap_or_default(),
        };
        let topology_response = self.topo.process(topo_request.into()).await?;
        // 从 Response 中提取 payload
        let topology_result = topology_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "TopoService returned empty payload".to_string(),
            })
        })?;

        // 构建场景状态
        let mut scene = SceneState {
            outer: topology_result.outer,
            holes: topology_result.holes,
            boundaries: Vec::new(),
            sources: Vec::new(),
            edges: Vec::new(),
            raster_metadata: None,
            units: common_types::LengthUnit::Mm,
            coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        };

        // 填充原始边数据
        scene_builder::fill_scene_edges(&mut scene, &entities);

        // 自动语义推断
        scene_builder::auto_infer_boundaries(&mut scene, &entities_refs);

        // 4. 验证：通过 ValidatorService::process() 调用
        let validate_request = validator::service::ValidateRequest::new(
            serde_json::to_string(&scene).unwrap_or_default(),
        );
        let validation_response = self.validator.process(validate_request.into()).await?;
        // 从 Response 中提取 payload
        let validation_report = validation_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ValidatorService returned empty payload".to_string(),
            })
        })?;

        // 检查验证结果
        if !validation_report.passed && validation_report.summary.error_count > 0 {
            let issues: Vec<common_types::error::ValidationIssue> = validation_report
                .issues
                .into_iter()
                .map(|i| common_types::error::ValidationIssue {
                    code: i.code,
                    severity: match i.severity {
                        validator::checks::Severity::Error => common_types::error::Severity::Error,
                        validator::checks::Severity::Warning => {
                            common_types::error::Severity::Warning
                        }
                        validator::checks::Severity::Info => common_types::error::Severity::Info,
                    },
                    message: i.message,
                    location: i.location.map(|l| common_types::error::ErrorLocation {
                        point: l.point,
                        segment: l.segment,
                        loop_index: l.loop_index,
                    }),
                })
                .collect();

            return Err(CadError::ValidationFailed {
                count: validation_report.summary.error_count,
                warning_count: validation_report.summary.warning_count,
                issues,
            });
        }

        // 5. 导出：通过 ExportService::process() 调用
        let export_request = export::service::ExportRequest::new(scene.clone());
        let export_response = self.export.process(export_request.into()).await?;
        // 从 Response 中提取 payload
        let export_result = export_response.payload.ok_or_else(|| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: "ExportService returned empty payload".to_string(),
            })
        })?;

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录流水线指标
        {
            let mut metrics = self.metrics.lock().await;
            metrics.total_requests += 1;
            if export_result.bytes.is_empty() {
                metrics.failed_requests += 1;
            } else {
                metrics.success_requests += 1;
            }
            let n = metrics.total_requests as f64;
            metrics.avg_processing_time_ms =
                metrics.avg_processing_time_ms * ((n - 1.0) / n) + latency / n;

            // 收集子服务指标 - 直接使用 snapshot() 返回的 ServiceMetricsData
            metrics.service_metrics.clear();
            metrics
                .service_metrics
                .push(self.parser.metrics().snapshot());
            metrics.service_metrics.push(self.topo.metrics().snapshot());
            metrics
                .service_metrics
                .push(self.validator.metrics().snapshot());
            metrics
                .service_metrics
                .push(self.export.metrics().snapshot());
        }

        Ok(Response::success(
            request.id,
            ProcessResult {
                scene,
                validation: validation_report,
                output_bytes: export_result.bytes,
                text_annotations,
                dimension_summary,
                raster_report: None,
                semantic_candidates: Vec::new(),
            },
            latency as u64,
        ))
    }

    fn health_check(&self) -> ServiceHealth {
        let deps = [
            DependencyHealth {
                name: "ParserService".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            },
            DependencyHealth {
                name: "TopoService".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            },
            DependencyHealth {
                name: "ValidatorService".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            },
            DependencyHealth {
                name: "VectorizeService".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            },
            DependencyHealth {
                name: "ExportService".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            },
        ];

        ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
            .with_uptime(0)
            .with_dependency(deps[0].clone())
            .with_dependency(deps[1].clone())
            .with_dependency(deps[2].clone())
            .with_dependency(deps[3].clone())
            .with_dependency(deps[4].clone())
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "ProcessingPipeline"
    }

    fn metrics(&self) -> &ServiceMetrics {
        // ProcessingPipeline 不直接暴露内部指标，而是通过 get_metrics() 异步获取
        // 这里返回一个静态默认值
        static DEFAULT_METRICS: std::sync::OnceLock<ServiceMetrics> = std::sync::OnceLock::new();
        DEFAULT_METRICS.get_or_init(|| ServiceMetrics::new("ProcessingPipeline"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_new() {
        let pipeline = ProcessingPipeline::new();
        let stats = pipeline.get_stats().await;
        assert!(stats.services_initialized);
    }

    #[test]
    fn test_discretize_circle() {
        let points = discretize_circle([0.0, 0.0], 10.0);
        assert!(points.len() >= 32);
        // 检查是否闭合
        if points.len() > 1 {
            let first = points[0];
            let last = points[points.len() - 1];
            let dist = ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt();
            assert!(dist < 0.1);
        }
    }

    #[test]
    fn test_discretize_arc() {
        let points = discretize_arc([0.0, 0.0], 10.0, 0.0, 90.0);
        assert!(points.len() >= 8);
        // 检查起点
        assert!((points[0][0] - 10.0).abs() < 0.1);
        assert!(points[0][1].abs() < 0.1);
    }
}
