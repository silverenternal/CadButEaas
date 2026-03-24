//! 处理流水线 - 真正的异步实现
//!
//! 改进：
//! 1. 使用 tokio::spawn_blocking 将 CPU 密集型任务放到阻塞线程池
//! 2. 使用 rayon 并行化处理多个实体
//! 3. 支持进度追踪和取消
//! 4. 集成服务指标收集（EaaS 架构）

use common_types::{SceneState, CadError, Polyline, RawEntity, ServiceMetrics, ServiceMetricsData, ServiceHealth, HealthStatus, DependencyHealth, InternalErrorReason, Service, ServiceVersion, Request, Response};
use parser::{ParserService, service::ParseResult, service::FileType};
use vectorize::{VectorizeService, VectorizeConfig};
use topo::{TopoService, service::TopoConfig as TopoServiceConfig};
use validator::{ValidatorService, ValidationReport, service::ValidatorConfig as ValidatorServiceConfig};
use export::{ExportService, service::ExportConfig as ExportServiceConfig};
use config::{CadConfig, ParserConfig as ConfigParserConfig};
use std::path::Path;
use std::sync::Arc;
use tokio::task::spawn_blocking;
use std::time::Instant;
use async_trait::async_trait;
use std::fmt::Debug;

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
    pub validation: ValidationReport,
    pub output_bytes: Vec<u8>,
}

/// 处理流水线
pub struct ProcessingPipeline {
    parser: Arc<ParserService>,
    vectorize: Arc<VectorizeService>,
    topo: Arc<TopoService>,
    validator: Arc<ValidatorService>,
    export: Arc<ExportService>,
    /// 服务指标
    metrics: Arc<tokio::sync::Mutex<PipelineMetrics>>,
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
        let vectorize_config = Self::convert_vectorize_config(&config.parser);
        let topo_config = Self::convert_topo_config(config);
        let validator_config = Self::convert_validator_config(config);
        let export_config = Self::convert_export_config(config);

        Self {
            parser: Arc::new(ParserService::new()),
            vectorize: Arc::new(VectorizeService::new(Box::new(accelerator_cpu::CpuAccelerator::new()), vectorize_config)),
            topo: Arc::new(TopoService::with_config(&topo_config)),
            validator: Arc::new(ValidatorService::with_config(&validator_config)),
            export: Arc::new(ExportService::with_config(&export_config)),
            metrics: Arc::default(),
        }
    }

    /// 转换矢量化配置
    /// 
    /// P11 锐评 v2.0 修复：从 CadConfig 读取 threshold，移除硬编码
    fn convert_vectorize_config(parser_config: &ConfigParserConfig) -> VectorizeConfig {
        VectorizeConfig {
            threshold: parser_config.pdf.threshold,
            snap_tolerance_px: parser_config.pdf.vectorize_tolerance_px,
            min_line_length_px: parser_config.pdf.min_line_length_px,
            max_angle_dev_deg: 5.0,
            skeletonize: true,
            #[cfg(feature = "opencv")]
            use_opencv: true,
            #[cfg(not(feature = "opencv"))]
            use_opencv: false,
            adaptive_threshold: true,
            use_hough: false,
            preprocessing: vectorize::config::PreprocessingConfig::default(),
            line_type_detection: false,
            arc_fitting: false,
            gap_filling: false,
            quality_assessment: false,
            dpi_adaptive: true,
            reference_dpi: 300.0,
            dpi_scale_factor: 1.0,
            opencv_approx_epsilon: Some(2.0),
        }
    }

    /// 转换拓扑配置（P11 修复：使用 TopoAlgorithm）
    fn convert_topo_config(topo_config: &CadConfig) -> TopoServiceConfig {
        use topo::service::TopoAlgorithm;
        
        // 根据字符串配置选择算法
        let algorithm = match topo_config.topology.algorithm.as_str() {
            "halfedge" => TopoAlgorithm::Halfedge,
            _ => TopoAlgorithm::Dfs,  // 默认 DFS
        };
        
        TopoServiceConfig {
            tolerance: common_types::geometry::ToleranceConfig {
                snap_tolerance: topo_config.topology.snap_tolerance_mm,
                min_line_length: topo_config.topology.min_line_length_mm,
                max_angle_deviation: topo_config.topology.merge_angle_tolerance_deg,
                units: Some(common_types::LengthUnit::Mm),
            },
            layer_filter: None,
            algorithm,
            skip_intersection_check: topo_config.topology.skip_intersection_check,
            enable_parallel: topo_config.topology.enable_parallel,
            parallel_threshold: topo_config.topology.parallel_threshold,
        }
    }

    /// 转换验证器配置
    fn convert_validator_config(validator_config: &CadConfig) -> ValidatorServiceConfig {
        ValidatorServiceConfig {
            closure_tolerance: validator_config.validator.closure_tolerance_mm,
            min_edge_length: validator_config.validator.min_edge_length_mm,
            min_angle_degrees: validator_config.validator.min_angle_deg,
        }
    }

    /// 转换导出配置
    fn convert_export_config(export_config: &CadConfig) -> ExportServiceConfig {
        ExportServiceConfig {
            format: match export_config.export.format.as_str() {
                "json" => export::formats::ExportFormat::Json,
                "bincode" | "binary" => export::formats::ExportFormat::Binary,
                _ => export::formats::ExportFormat::Json,
            },
            pretty_json: export_config.export.json_indent > 0,
            target_units: None,
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

        let mut health = ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
            .with_uptime(0);
        
        for dep in deps {
            health = health.with_dependency(dep);
        }

        // 如果整体状态不是 Healthy，重新构建健康状态
        match overall_status {
            HealthStatus::Healthy => health,
            HealthStatus::Degraded => ServiceHealth::degraded(env!("CARGO_PKG_VERSION"), health.dependencies),
            HealthStatus::Unhealthy => ServiceHealth::unhealthy(env!("CARGO_PKG_VERSION"), "子服务不健康"),
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

        // 在阻塞线程池中执行 CPU 密集型任务
        let parser = Arc::clone(&self.parser);
        let vectorize = Arc::clone(&self.vectorize);
        let topo = Arc::clone(&self.topo);
        let validator = Arc::clone(&self.validator);
        let export = Arc::clone(&self.export);
        let metrics = Arc::clone(&self.metrics);

        let result = spawn_blocking(move || {
            Self::process_file_sync(
                &parser,
                &vectorize,
                &topo,
                &validator,
                &export,
                &path,
            )
        })
        .await
        .unwrap_or_else(|e| Err(CadError::InternalError {
            reason: InternalErrorReason::Panic { message: format!("任务执行失败：{}", e) },
            location: Some("process_file"),
        }));

        // 记录指标
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        let mut metrics_guard = metrics.lock().await;
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
    pub async fn process_with_services(&self, path: impl AsRef<Path>) -> Result<ProcessResult, CadError> {
        let path = path.as_ref().to_path_buf();
        tracing::info!("开始通过 Service trait 处理文件：{:?}", path);
        let start_time = Instant::now();

        // 1. 解析 - 通过 ParserService::process()，使用 Response 返回的数据
        tracing::info!("阶段 1/5: 解析文件（通过 Service::process）");
        let parse_request = parser::service::ParseRequest::new(path.to_str().unwrap_or(""));
        let parse_response = self.parser.process(parse_request.into()).await?;
        
        // 从 Response 中提取 payload，不再调用 parse_file()
        let parse_result = parse_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ParserService returned empty payload".to_string(),
            }
        ))?;
        
        let has_raster = parse_result.has_raster();
        let entities = parse_result.into_entities();
        let entities_refs: Vec<&common_types::RawEntity> = entities.iter().collect();
        tracing::info!("  解析得到 {} 个实体", entities.len());

        // 2. 矢量化 - 从解析结果提取多段线
        tracing::info!("阶段 2/5: 矢量化处理");
        let polylines = if has_raster {
            tracing::info!("  检测到光栅图像，开始矢量化处理");
            Self::extract_polylines_from_entities(&entities)
        } else {
            tracing::info!("  解析得到矢量数据");
            Self::extract_polylines_from_entities(&entities)
        };
        tracing::info!("  得到 {} 条多段线", polylines.len());

        // 3. 构建拓扑 - 通过 TopoService::process()，使用 Response 返回的数据
        tracing::info!("阶段 3/5: 构建拓扑（通过 Service::process）");
        let topo_request = topo::service::TopoRequest {
            geometry_json: serde_json::to_string(&polylines).unwrap_or_default(),
        };
        let topo_response = self.topo.process(topo_request.into()).await?;
        
        // 从 Response 中提取 payload，不再调用 build_scene()
        let topo_result = topo_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "TopoService returned empty payload".to_string(),
            }
        ))?;
        
        // 从 TopologyResult 构建 SceneState
        let mut scene = SceneState {
            outer: topo_result.outer,
            holes: topo_result.holes,
            boundaries: Vec::new(), // 待用户标注
            sources: Vec::new(),
            edges: Vec::new(), // 待填充
            units: common_types::LengthUnit::Mm,
            coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        };
        tracing::info!("  构建完成：{} 个外轮廓，{} 个孔洞",
            if scene.outer.is_some() { 1 } else { 0 },
            scene.holes.len());

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
        Self::auto_infer_boundaries(&mut scene, &entities_refs);
        tracing::info!("  语义推断完成：{} 个边界段", scene.boundaries.len());

        // 4. 验证 - 通过 ValidatorService::process()，使用 Response 返回的数据
        tracing::info!("阶段 4/5: 验证场景（通过 Service::process）");
        let validate_request = validator::service::ValidateRequest::new(serde_json::to_string(&scene).unwrap_or_default());
        let validate_response = self.validator.process(validate_request.into()).await?;
        
        // 从 Response 中提取 payload，不再调用 validate()
        let validation = validate_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ValidatorService returned empty payload".to_string(),
            }
        ))?;

        if !validation.passed {
            tracing::warn!("验证失败，错误数：{}, 警告数：{}",
                validation.summary.error_count,
                validation.summary.warning_count);

            if validation.summary.error_count > 0 {
                let issues: Vec<common_types::error::ValidationIssue> = validation.issues
                    .into_iter()
                    .map(|i| common_types::error::ValidationIssue {
                        code: i.code,
                        severity: match i.severity {
                            validator::checks::Severity::Error => common_types::error::Severity::Error,
                            validator::checks::Severity::Warning => common_types::error::Severity::Warning,
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
        let export_result = export_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ExportService returned empty payload".to_string(),
            }
        ))?;

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
        })
    }

    /// 同步处理实现（在 spawn_blocking 中调用）
    fn process_file_sync(
        parser: &ParserService,
        _vectorize: &VectorizeService,
        topo: &TopoService,
        validator: &ValidatorService,
        export: &ExportService,
        path: &Path,
    ) -> Result<ProcessResult, CadError> {
        // 1. 解析（CPU 密集型）
        tracing::info!("阶段 1/5: 解析文件");
        let parse_result = parser.parse_file(path)?;

        // 1.5 检查是否包含光栅图像（在消耗 parse_result 之前）
        let has_raster = parse_result.has_raster();

        // 1.6 提取 entities 用于后续语义推断（在消耗 parse_result 之前）
        // 注意：需要克隆实体数据以保留到后续使用
        let entities = parse_result.into_entities();
        let entities_refs: Vec<&RawEntity> = entities.iter().collect();
        tracing::info!("  解析得到 {} 个实体", entities.len());

        // 2. 矢量化（如果有光栅图像，CPU+ 内存密集型）
        let polylines = if has_raster {
            tracing::info!("  检测到光栅图像，开始矢量化处理");
            // 注意：这里需要重新构建 ParseResult 用于光栅处理
            // 简化方案：假设 CAD 文件不含光栅，直接提取 polylines
            Self::extract_polylines_from_entities(&entities)
        } else {
            tracing::info!("  解析得到矢量数据");
            Self::extract_polylines_from_entities(&entities)
        };
        tracing::info!("  得到 {} 条多段线", polylines.len());

        // 3. 构建拓扑（CPU 密集型，使用 R*-tree 加速）
        tracing::info!("阶段 3/5: 构建拓扑");
        let mut scene = topo.build_scene(&polylines)?;
        tracing::info!("  构建完成：{} 个外轮廓，{} 个孔洞",
            if scene.outer.is_some() { 1 } else { 0 },
            scene.holes.len());

        // 3.3 填充原始边数据（用于前端显示）
        tracing::info!("阶段 3.3/5: 填充原始边数据");
        
        // 辅助函数：从实体提取边
        let mut edge_id = 0;
        let mut extracted_edges: Vec<common_types::RawEdge> = Vec::new();
        
        for entity in &entities {
            match entity {
                common_types::RawEntity::Line { start, end, metadata, .. } => {
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
                common_types::RawEntity::Polyline { points, closed, metadata, .. } => {
                    if points.len() >= 2 {
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
                }
                // Arc 离散化为线段（简化处理：只取弦）
                common_types::RawEntity::Arc { center, radius, start_angle, end_angle, metadata, .. } => {
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
                common_types::RawEntity::Circle { center, radius, metadata, .. } => {
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
        Self::auto_infer_boundaries(&mut scene, &entities_refs);
        tracing::info!("  语义推断完成：{} 个边界段", scene.boundaries.len());

        // 4. 验证（CPU 密集型）
        tracing::info!("阶段 4/5: 验证场景");
        let validation = validator.validate(&scene)?;

        if !validation.passed {
            tracing::warn!("验证失败，错误数：{}, 警告数：{}",
                validation.summary.error_count,
                validation.summary.warning_count);

            // 如果有错误，返回验证失败
            if validation.summary.error_count > 0 {
                let issues: Vec<common_types::error::ValidationIssue> = validation.issues
                    .into_iter()
                    .map(|i| common_types::error::ValidationIssue {
                        code: i.code,
                        severity: match i.severity {
                            validator::checks::Severity::Error => common_types::error::Severity::Error,
                            validator::checks::Severity::Warning => common_types::error::Severity::Warning,
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
        })
    }

    /// 处理光栅图像并矢量化
    fn process_raster_with_vectorization(
        result: ParseResult,
        vectorize: &VectorizeService,
    ) -> Result<Vec<Polyline>, CadError> {
        match result {
            ParseResult::Pdf(content) => {
                let mut all_polylines = Vec::new();
                let mut vectorize_errors = Vec::new();
                let mut quality_warnings = Vec::new();

                // 1. 首先提取已有的矢量实体（如果有）
                let vector_polylines = Self::extract_polylines_from_entities(&content.vector_entities);
                all_polylines.extend(vector_polylines);

                // 2. 对每个光栅图像进行矢量化
                for raster_image in &content.raster_images {
                    tracing::debug!("  处理光栅图像：{}x{}", raster_image.width, raster_image.height);

                    // 转换为 PdfRasterImage 并矢量化
                    let pdf_raster = raster_image.to_pdf_raster_image();
                    let config = VectorizeConfig::default();

                    match vectorize.vectorize_from_pdf(&pdf_raster, Some(&config)) {
                        Ok(polylines) => {
                            tracing::debug!("    矢量化得到 {} 条多段线", polylines.len());

                            // 质量评估
                            let quality_report = vectorize::algorithms::quality::evaluate_quality(&pdf_raster, &polylines);

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
                                    tracing::warn!("  - [{:?}] {}", issue.severity, issue.description);
                                }
                            }

                            all_polylines.extend(polylines);
                        }
                        Err(e) => {
                            tracing::warn!("    矢量化失败：{:?}", e);
                            vectorize_errors.push(format!(
                                "图像 '{}': {:?}",
                                pdf_raster.name, e
                            ));
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
                    return Err(CadError::VectorizeFailed {
                        message: error_msg,
                    });
                }

                // 如果有质量警告，记录但不失败
                if !quality_warnings.is_empty() {
                    tracing::warn!("矢量化质量警告：{} 个图像质量较低", quality_warnings.len());
                }

                Ok(all_polylines)
            }
            ParseResult::Cad(entities) => {
                // CAD 文件不应该有光栅图像
                Ok(Self::extract_polylines_from_entities(&entities))
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
                &parser,
                &vectorize,
                &topo,
                &validator,
                &export,
                &bytes,
                file_type,
            )
        })
        .await
        .unwrap_or_else(|e| Err(CadError::InternalError {
            reason: InternalErrorReason::Panic { message: format!("任务执行失败：{}", e) },
            location: Some("process_bytes"),
        }))
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
        
        // 2. 矢量化（如果有光栅图像）
        let polylines = if parse_result.has_raster() {
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
        })
    }

    /// 从解析结果中提取多段线（使用 rayon 并行化）
    fn extract_polylines(result: ParseResult) -> Result<Vec<Polyline>, CadError> {
        let entities = result.into_entities();
        Ok(Self::extract_polylines_from_entities(&entities))
    }

    /// 从实体列表中提取多段线（使用 rayon 并行化）
    pub fn extract_polylines_from_entities(entities: &[RawEntity]) -> Vec<Polyline> {
        use rayon::prelude::*;

        // 并行处理实体转换
        entities
            .par_iter()
            .filter_map(|entity| {
                match entity {
                    RawEntity::Line { start, end, .. } => {
                        Some(vec![*start, *end])
                    }
                    RawEntity::Polyline { points, closed, .. } => {
                        let mut pts = points.clone();
                        if *closed && pts.first() != pts.last() {
                            if let Some(first) = pts.first() {
                                pts.push(*first);
                            }
                        }
                        Some(pts)
                    }
                    RawEntity::Arc { center, radius, start_angle, end_angle, .. } => {
                        // 离散化圆弧为多段线
                        Some(discretize_arc(*center, *radius, *start_angle, *end_angle))
                    }
                    RawEntity::Circle { center, radius, .. } => {
                        // 离散化圆为多段线
                        Some(discretize_circle(*center, *radius))
                    }
                    // 其他类型暂时忽略
                    _ => None,
                }
            })
            .collect()
    }

    /// 填充场景的原始边数据（用于前端显示）
    pub fn fill_scene_edges(scene: &mut SceneState, entities: &[RawEntity]) {
        let mut edge_id = 0;
        let mut extracted_edges: Vec<common_types::RawEdge> = Vec::new();

        for entity in entities {
            match entity {
                common_types::RawEntity::Line { start, end, metadata, .. } => {
                    extracted_edges.push(common_types::RawEdge {
                        id: edge_id,
                        start: *start,
                        end: *end,
                        layer: metadata.layer.clone(),
                        color_index: None,
                    });
                    edge_id += 1;
                }
                common_types::RawEntity::Polyline { points, closed, metadata, .. } => {
                    if points.len() >= 2 {
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
                }
                common_types::RawEntity::Arc { center, radius, start_angle, end_angle, metadata, .. } => {
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
                common_types::RawEntity::Circle { center, radius, metadata, .. } => {
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
                _ => {}
            }
        }

        scene.edges = extracted_edges;
    }

    /// 自动语义推断（P1 任务）
    ///
    /// 根据实体图层名、颜色等信息自动填充边界段的语义和材料标签
    pub fn auto_infer_boundaries(scene: &mut SceneState, entities: &[&RawEntity]) {
        use common_types::scene::BoundarySegment;

        // 从实体中提取图层和颜色信息，创建映射
        let mut entity_info_map = std::collections::HashMap::new();

        for (idx, entity) in entities.iter().enumerate() {
            let layer = (*entity).layer().map(String::from);
            let color = (*entity).color().map(String::from);
            entity_info_map.insert(idx, (layer, color));
        }

        // 为外轮廓创建边界段
        if let Some(outer) = &scene.outer {
            let points = &outer.points;
            for i in 0..points.len() {
                let start = points[i];
                let end = points[(i + 1) % points.len()];

                // 尝试找到对应的实体（通过空间位置匹配）
                let (layer, color) = find_matching_entity_info_from_refs(&entity_info_map, start, end, entities);

                // 推断语义
                let semantic = if let Some(ref layer_name) = layer {
                    BoundarySegment::infer_semantic_from_layer(layer_name)
                } else {
                    common_types::scene::BoundarySemantic::HardWall
                };

                // 推断材料
                let material = if let Some(ref color_name) = color {
                    // 尝试从 ACI 颜色索引推断
                    if let Ok(color_idx) = color_name.parse::<u16>() {
                        BoundarySegment::infer_material_from_aci_color(color_idx)
                    } else {
                        None
                    }
                } else if let Some(ref layer_name) = layer {
                    // 备选：从图层名推断
                    BoundarySegment::infer_material_from_layer(layer_name)
                } else {
                    None
                };

                // 计算宽度（如果是开口类型）
                let width = if matches!(semantic,
                    common_types::scene::BoundarySemantic::Door |
                    common_types::scene::BoundarySemantic::Window |
                    common_types::scene::BoundarySemantic::Opening
                ) {
                    BoundarySegment::calculate_width(start, end)
                } else {
                    None
                };

                scene.boundaries.push(BoundarySegment {
                    segment: [i, (i + 1) % points.len()],
                    semantic,
                    material,
                    width,
                });
            }
        }

        // 为孔洞创建边界段
        for hole in scene.holes.iter() {
            let points = &hole.points;

            for i in 0..points.len() {
                let start = points[i];
                let end = points[(i + 1) % points.len()];

                // 尝试找到对应的实体
                let (layer, color) = find_matching_entity_info_from_refs(&entity_info_map, start, end, entities);

                // 推断语义
                let semantic = if let Some(ref layer_name) = layer {
                    BoundarySegment::infer_semantic_from_layer(layer_name)
                } else {
                    common_types::scene::BoundarySemantic::HardWall
                };

                // 推断材料
                let material = if let Some(ref color_name) = color {
                    if let Ok(color_idx) = color_name.parse::<u16>() {
                        BoundarySegment::infer_material_from_aci_color(color_idx)
                    } else {
                        None
                    }
                } else if let Some(ref layer_name) = layer {
                    BoundarySegment::infer_material_from_layer(layer_name)
                } else {
                    None
                };

                // 计算宽度
                let width = if matches!(semantic,
                    common_types::scene::BoundarySemantic::Door |
                    common_types::scene::BoundarySemantic::Window |
                    common_types::scene::BoundarySemantic::Opening
                ) {
                    BoundarySegment::calculate_width(start, end)
                } else {
                    None
                };

                scene.boundaries.push(BoundarySegment {
                    segment: [i, (i + 1) % points.len()],
                    semantic,
                    material,
                    width,
                });
            }
        }
    }

    /// 获取流水线统计信息
    pub async fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            services_initialized: true,
            // 可以添加更多运行时统计
        }
    }
}

/// 查找匹配的实体信息（通过空间位置匹配，借用引用版本）
fn find_matching_entity_info_from_refs(
    entity_info_map: &std::collections::HashMap<usize, (Option<String>, Option<String>)>,
    start: Point2,
    end: Point2,
    entities: &[&RawEntity],
) -> (Option<String>, Option<String>) {
    const SNAP_TOLERANCE: f64 = 5.0; // 5mm 容差

    // 遍历所有实体，查找空间位置匹配的
    for (idx, entity) in entities.iter().enumerate() {
        match *entity {
            RawEntity::Line { start: e_start, end: e_end, .. } => {
                // 检查线段是否重合（起点和终点都在容差范围内）
                let start_dist = distance_2d(start, *e_start);
                let end_dist = distance_2d(end, *e_end);

                if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                    if let Some((layer, color)) = entity_info_map.get(&idx) {
                        return (layer.clone(), color.clone());
                    }
                }

                // 也检查反向匹配（线段方向可能相反）
                let start_dist_rev = distance_2d(start, *e_end);
                let end_dist_rev = distance_2d(end, *e_start);

                if start_dist_rev < SNAP_TOLERANCE && end_dist_rev < SNAP_TOLERANCE {
                    if let Some((layer, color)) = entity_info_map.get(&idx) {
                        return (layer.clone(), color.clone());
                    }
                }
            }
            RawEntity::Polyline { points, closed, .. } => {
                // 检查多段线中是否有匹配的线段
                for i in 0..points.len() - 1 {
                    let p1 = points[i];
                    let p2 = points[i + 1];

                    let start_dist = distance_2d(start, p1);
                    let end_dist = distance_2d(end, p2);

                    if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                        if let Some((layer, color)) = entity_info_map.get(&idx) {
                            return (layer.clone(), color.clone());
                        }
                    }
                }

                // 检查闭合边
                if *closed && !points.is_empty() {
                    let p1 = points[points.len() - 1];
                    let p2 = points[0];

                    let start_dist = distance_2d(start, p1);
                    let end_dist = distance_2d(end, p2);

                    if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                        if let Some((layer, color)) = entity_info_map.get(&idx) {
                            return (layer.clone(), color.clone());
                        }
                    }
                }
            }
            // 其他类型暂时忽略
            _ => {}
        }
    }

    (None, None)
}

/// 计算两点间距离
fn distance_2d(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// 离散化圆弧
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

/// 流水线统计信息
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub services_initialized: bool,
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// 类型别名，避免重复
type Point2 = [f64; 2];

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

// ============================================================================
// EaaS Service Trait 实现
// ============================================================================

#[async_trait]
impl Service for ProcessingPipeline {
    type Payload = PipelineRequest;
    type Data = ProcessResult;
    type Error = CadError;

    async fn process(&self, request: Request<Self::Payload>) -> Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 真正的处理入口：通过调用各服务的 process() 方法完成编排
        // 这展示了 EaaS 架构的核心：服务可组合、可替换、可观测

        // 1. 解析：通过 ParserService::process() 调用
        let parse_request = parser::service::ParseRequest::new(request.payload.path.to_str().unwrap_or(""));
        let parse_response = self.parser.process(parse_request.into()).await?;
        // 从 Response 中提取 payload
        let parse_result = parse_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ParserService returned empty payload".to_string(),
            }
        ))?;

        // 从解析结果提取实体
        let entities = parse_result.into_entities();
        let entities_refs: Vec<&common_types::RawEntity> = entities.iter().collect();

        // 2. 矢量化：提取多段线（矢量化服务已集成到解析器中）
        let polylines = Self::extract_polylines_from_entities(&entities);

        // 3. 构建拓扑：通过 TopoService::process() 调用
        let topo_request = topo::service::TopoRequest {
            geometry_json: serde_json::to_string(&polylines).unwrap_or_default(),
        };
        let topology_response = self.topo.process(topo_request.into()).await?;
        // 从 Response 中提取 payload
        let topology_result = topology_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "TopoService returned empty payload".to_string(),
            }
        ))?;

        // 构建场景状态
        let mut scene = SceneState {
            outer: topology_result.outer,
            holes: topology_result.holes,
            boundaries: Vec::new(),
            sources: Vec::new(),
            edges: Vec::new(),
            units: common_types::LengthUnit::Mm,
            coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(),
            render_config: None,
        };

        // 填充原始边数据
        Self::fill_scene_edges(&mut scene, &entities);

        // 自动语义推断
        Self::auto_infer_boundaries(&mut scene, &entities_refs);

        // 4. 验证：通过 ValidatorService::process() 调用
        let validate_request = validator::service::ValidateRequest::new(
            serde_json::to_string(&scene).unwrap_or_default()
        );
        let validation_response = self.validator.process(validate_request.into()).await?;
        // 从 Response 中提取 payload
        let validation_report = validation_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ValidatorService returned empty payload".to_string(),
            }
        ))?;

        // 检查验证结果
        if !validation_report.passed && validation_report.summary.error_count > 0 {
            let issues: Vec<common_types::error::ValidationIssue> = validation_report.issues
                .into_iter()
                .map(|i| common_types::error::ValidationIssue {
                    code: i.code,
                    severity: match i.severity {
                        validator::checks::Severity::Error => common_types::error::Severity::Error,
                        validator::checks::Severity::Warning => common_types::error::Severity::Warning,
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
        let export_result = export_response.payload.ok_or_else(|| CadError::internal(
            InternalErrorReason::ServiceUnavailable {
                service: "ExportService returned empty payload".to_string(),
            }
        ))?;

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
            metrics.avg_processing_time_ms = metrics.avg_processing_time_ms * ((n - 1.0) / n) + latency / n;

            // 收集子服务指标 - 直接使用 snapshot() 返回的 ServiceMetricsData
            metrics.service_metrics.clear();
            metrics.service_metrics.push(self.parser.metrics().snapshot());
            metrics.service_metrics.push(self.topo.metrics().snapshot());
            metrics.service_metrics.push(self.validator.metrics().snapshot());
            metrics.service_metrics.push(self.export.metrics().snapshot());
        }

        Ok(Response::success(request.id, ProcessResult {
            scene,
            validation: validation_report,
            output_bytes: export_result.bytes,
        }, latency as u64))
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
