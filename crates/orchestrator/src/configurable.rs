//! 可配置服务编排（P11 建议落地版）
//!
//! 支持：
//! - 动态服务编排（流程可配置，真正执行配置）
//! - 根据文件类型走不同流程
//! - 并行执行独立步骤
//! - 质量预检插件

use crate::pipeline::{ProcessResult, ProcessingPipeline};
use common_types::{CadError, Polyline, RawEntity, SceneState};
use once_cell::sync::Lazy;
use parser::ParserService;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tokio::task::spawn_blocking;
use validator::ValidationReport;

/// 服务阶段枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    /// 解析
    Parse,
    /// 矢量化
    Vectorize,
    /// 拓扑构建
    BuildTopology,
    /// 验证
    Validate,
    /// 导出
    Export,
    /// 质量预检（可选）
    QualityCheck,
    /// 声学分析（可选）
    AcousticAnalysis,
}

/// 阶段配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// 阶段名称
    pub stage: PipelineStage,
    /// 是否启用
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// 是否可跳过（非关键阶段）
    #[serde(default)]
    pub optional: bool,
    /// 超时时间（毫秒）
    pub timeout_ms: Option<u64>,
}

fn default_true() -> bool {
    true
}

/// 流程配置文件格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 流程名称
    pub name: String,
    /// 流程描述
    pub description: Option<String>,
    /// 阶段列表（按执行顺序）
    pub stages: Vec<StageConfig>,
    /// 是否并行执行独立阶段
    #[serde(default)]
    pub parallel: bool,
    /// 文件类型过滤器（可选）
    pub file_types: Option<Vec<String>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: Some("默认处理流程".to_string()),
            stages: vec![
                StageConfig {
                    stage: PipelineStage::Parse,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Vectorize,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::BuildTopology,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Validate,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Export,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
            ],
            parallel: false,
            file_types: None,
        }
    }
}

/// 可配置服务编排
///
/// 支持动态配置服务执行流程
pub struct ConfigurablePipeline {
    /// 基础流水线
    base_pipeline: Arc<ProcessingPipeline>,
    /// 流程配置
    config: PipelineConfig,
    /// 解析器服务（用于阶段执行）
    parser: Arc<ParserService>,
}

/// 阶段执行上下文（RwLock 共享版本，真正的写时复制）
///
/// # 设计说明（P11 锐评 v3.0 回应）
///
/// 使用 `Arc<RwLock<Arc<[T]>>>` 而非 `OnceLock<Arc<[T]>>` 的理由：
///
/// 1. **灵活性**：支持阶段重试/回滚（未来扩展）
/// 2. **调试友好**：可以在执行过程中更新中间状态用于调试
/// 3. **性能可接受**：每个阶段只写入一次，读多写少场景下 RwLock 开销很小
/// 4. **语义清晰**：RwLock 明确表示"可变状态"，OnceLock 表示"一次初始化"
///
/// ## 性能分析
///
/// 在并行执行时：
/// - 读时零拷贝：`Arc::clone` 只增加引用计数（O(1)）
/// - 写时原子替换：直接替换 Arc 指针（O(1)）
/// - 读写分离，支持多读者并发访问
/// - 内存高效：多个阶段可共享同一份数据
///
/// ## 如果未来需要优化
///
/// 如果性能分析显示锁开销是瓶颈，可以考虑：
/// - 使用 `OnceLock<Arc<[T]>>`（只初始化一次）
/// - 使用 `dashmap` 或 `sharded-slab` 等无锁数据结构
/// - 使用通道（channel）传递所有权而非共享引用
#[derive(Default)]
struct StageContext {
    /// 解析后的实体 - Arc<[T]> 支持零拷贝克隆
    entities: Arc<RwLock<Arc<[RawEntity]>>>,
    /// 矢量化后的多段线 - Arc<[T]> 支持零拷贝克隆
    polylines: Arc<RwLock<Arc<[Polyline]>>>,
    /// 拓扑场景
    scene: Arc<RwLock<Option<SceneState>>>,
    /// 验证报告
    validation: Arc<RwLock<Option<ValidationReport>>>,
    /// 导出字节
    output_bytes: Arc<RwLock<Vec<u8>>>,
    /// 文字标注
    text_annotations: Arc<RwLock<Vec<common_types::TextAnnotation>>>,
    /// 标注尺寸统计
    dimension_summary: Arc<RwLock<common_types::DimensionSummary>>,
}

/// 阶段诊断信息（在执行前采集，用于超时错误诊断）
#[derive(Clone)]
struct StageDiagnostic {
    stage: PipelineStage,
    entities_count: usize,
    polylines_count: usize,
    scene_status: &'static str,
    start_time: std::time::Instant,
}

// ========================================================================
// Prometheus 指标定义（P11 锐评建议 3：诊断信息可观测性）
// ========================================================================

/// 阶段超时计数器
/// 标签：stage（阶段名称）
static STAGE_TIMEOUT_COUNTER: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("stage_timeout_total", "阶段执行超时总次数", &["stage"])
        .expect("阶段超时指标注册失败")
});

/// 阶段执行耗时直方图
/// 标签：stage（阶段名称）
static STAGE_DURATION_HISTOGRAM: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "stage_duration_seconds",
        "阶段执行耗时（秒）",
        &["stage"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
    )
    .expect("阶段耗时指标注册失败")
});

/// 阶段取消计数器
/// 标签：stage（阶段名称）
static STAGE_CANCEL_COUNTER: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("stage_cancel_total", "阶段被取消总次数", &["stage"])
        .expect("阶段取消指标注册失败")
});

/// 阶段完成计数器
/// 标签：stage（阶段名称）
static STAGE_COMPLETED_COUNTER: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("stage_completed_total", "阶段完成总次数", &["stage"])
        .expect("阶段完成指标注册失败")
});

/// 超时或取消结果（精简版）
///
/// Timeout 和 Cancelled 都是"未完成"状态，用额外字段区分原因
#[derive(Debug, Clone, PartialEq, Eq)]
enum TimeoutOrCancel {
    /// 未完成：超时
    Timeout,
    /// 未完成：被取消（其他任务失败）
    Cancelled,
    /// 已完成
    Completed,
}

/// 并行任务结果枚举（简化版，所有任务都返回 Result<(), CadError>）
enum TaskResult {
    Parse(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    Vectorize(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    BuildTopology(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    Validate(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    Export(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    QualityCheck(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
    AcousticAnalysis(tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>>),
}

/// 任务元组类型：(阶段配置，任务结果，诊断信息)
type TaskTuple = (StageConfig, TaskResult, StageDiagnostic);

impl ConfigurablePipeline {
    /// 创建新的可配置流水线
    pub fn new(base_pipeline: ProcessingPipeline, config: PipelineConfig) -> Self {
        Self {
            base_pipeline: Arc::new(base_pipeline),
            config,
            parser: Arc::new(ParserService::new()),
        }
    }

    /// 从配置创建
    pub fn from_config(config: PipelineConfig) -> Result<Self, CadError> {
        let base_pipeline = ProcessingPipeline::new();
        Ok(Self::new(base_pipeline, config))
    }

    /// 处理文件（真正执行配置的阶段）
    ///
    /// # 执行流程
    ///
    /// 1. 检查文件类型过滤器
    /// 2. 按配置顺序执行各阶段
    /// 3. 支持并行执行无依赖阶段
    /// 4. 支持阶段超时和可选跳过
    pub async fn process_file(&self, path: impl AsRef<Path>) -> Result<ProcessResult, CadError> {
        let path = path.as_ref().to_path_buf();

        // 1. 检查文件类型过滤器
        if let Some(file_types) = &self.config.file_types {
            let file_ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if !file_types.iter().any(|ft| ft.to_lowercase() == file_ext) {
                return Err(CadError::pdf_parse(
                    path.clone(),
                    common_types::PdfParseReason::ExtractError(format!(
                        "文件类型 '{}' 不在允许列表中：{:?}",
                        file_ext, file_types
                    )),
                ));
            }
        }

        // 2. 初始化阶段上下文
        let ctx = StageContext::default();

        // 3. 按配置顺序执行阶段
        let stages_to_execute: Vec<&StageConfig> =
            self.config.stages.iter().filter(|s| s.enabled).collect();

        if stages_to_execute.is_empty() {
            return Err(CadError::internal(
                common_types::InternalErrorReason::ServiceUnavailable {
                    service: "没有启用的阶段可执行".to_string(),
                },
            ));
        }

        // 4. 执行阶段（支持并行）
        if self.config.parallel {
            self.execute_stages_parallel(&path, &ctx, &stages_to_execute)
                .await?;
        } else {
            self.execute_stages_sequential(&path, &ctx, &stages_to_execute)
                .await?;
        }

        // 5. 构建最终结果（从 RwLock 中提取数据）
        let scene_guard = ctx.scene.read().await;
        let validation_guard = ctx.validation.read().await;
        let output_bytes_guard = ctx.output_bytes.read().await;
        let text_annotations_guard = ctx.text_annotations.read().await;
        let dimension_summary_guard = ctx.dimension_summary.read().await;

        Ok(ProcessResult {
            scene: scene_guard.clone().unwrap_or_default(),
            validation: validation_guard.clone().unwrap_or_default(),
            output_bytes: (*output_bytes_guard).clone(),
            text_annotations: (*text_annotations_guard).clone(),
            dimension_summary: (*dimension_summary_guard).clone(),
            raster_report: None,
            semantic_candidates: Vec::new(),
        })
    }

    /// 串行执行阶段
    async fn execute_stages_sequential(
        &self,
        path: &Path,
        ctx: &StageContext,
        stages: &[&StageConfig],
    ) -> Result<(), CadError> {
        for stage in stages {
            tracing::info!("执行阶段：{:?} (optional={})", stage.stage, stage.optional);

            let result = match &stage.stage {
                PipelineStage::Parse => self.execute_parse(path, ctx).await,
                PipelineStage::Vectorize => self.execute_vectorize(ctx).await,
                PipelineStage::BuildTopology => self.execute_build_topology(ctx).await,
                PipelineStage::Validate => self.execute_validate(ctx).await,
                PipelineStage::Export => self.execute_export(ctx).await,
                PipelineStage::QualityCheck => self.execute_quality_check(ctx).await,
                PipelineStage::AcousticAnalysis => self.execute_acoustic_analysis(ctx).await,
            };

            // 处理执行结果
            match result {
                Ok(()) => {
                    tracing::debug!("阶段 {:?} 执行成功", stage.stage);
                }
                Err(e) => {
                    if stage.optional {
                        tracing::warn!("阶段 {:?} 执行失败但被跳过：{:?}", stage.stage, e);
                    } else {
                        tracing::error!("阶段 {:?} 执行失败：{:?}", stage.stage, e);
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// 并行执行阶段（无依赖阶段可并行）
    ///
    /// # 依赖关系
    ///
    /// - Parse → Vectorize → BuildTopology → Validate → Export
    /// - QualityCheck 需要 polylines 数据，所以需要 Parse + Vectorize 完成
    /// - AcousticAnalysis 可能需要验证结果，所以需要 BuildTopology + Validate 完成
    ///
    /// # P11 锐评修复（v7.0）
    ///
    /// - 缩短锁持有时间：快速克隆引用，锁外处理数据
    /// - 超时错误信息增强：添加诊断数据（实体数、多段线数等）
    /// - 快速失败机制：一个任务失败，立即取消其他任务
    async fn execute_stages_parallel(
        &self,
        path: &Path,
        ctx: &StageContext,
        stages: &[&StageConfig],
    ) -> Result<(), CadError> {
        // 简化的并行策略：将无依赖的阶段分组并行执行
        // 实际生产环境可以使用更复杂的依赖图调度

        let mut pending_stages: Vec<&StageConfig> = stages.to_vec();
        let mut completed_stages: Vec<PipelineStage> = Vec::new();

        while !pending_stages.is_empty() {
            // 找出当前可执行的阶段（依赖已满足）
            let ready_stages: Vec<&StageConfig> = pending_stages
                .iter()
                .copied()
                .filter(|s| self.stage_dependencies_met(&s.stage, &completed_stages))
                .collect();

            if ready_stages.is_empty() {
                return Err(CadError::internal(
                    common_types::InternalErrorReason::InvariantViolated {
                        invariant: "并行执行调度失败：没有可执行的阶段但仍有待处理阶段".to_string(),
                    },
                ));
            }

            // P11 锐评修复 2: 先订阅，再执行（避免错过取消信号）
            // 创建取消通道（容量为就绪阶段数量，避免 SendError）
            let (cancel_tx, _) = broadcast::channel::<()>(ready_stages.len().max(1));

            // 先创建所有接收者，再执行任务
            let mut cancel_rxs: Vec<broadcast::Receiver<()>> =
                Vec::with_capacity(ready_stages.len());
            for _ in &ready_stages {
                cancel_rxs.push(cancel_tx.subscribe());
            }

            // 并行执行所有就绪阶段
            let mut tasks: Vec<TaskTuple> = Vec::new();
            for (stage_idx, stage) in ready_stages.into_iter().enumerate() {
                let stage_clone = stage.clone();
                let path = path.to_path_buf();
                let timeout_ms = stage.timeout_ms.unwrap_or(300_000); // 默认 5 分钟
                let cancel_rx = cancel_rxs.swap_remove(stage_idx);

                // P11 锐评修复 4: 在执行前采集诊断信息
                let diagnostic = StageDiagnostic {
                    stage: stage_clone.stage.clone(),
                    entities_count: ctx.entities.read().await.len(),
                    polylines_count: ctx.polylines.read().await.len(),
                    scene_status: if ctx.scene.read().await.is_some() {
                        "已构建"
                    } else {
                        "未构建"
                    },
                    start_time: std::time::Instant::now(),
                };

                // 为每个阶段创建独立的任务
                match &stage.stage {
                    PipelineStage::Parse => {
                        let parser = Arc::clone(&self.parser);
                        let ctx_entities = Arc::clone(&ctx.entities);
                        // P11 锐评修复 3: 使用 spawn_blocking 执行 CPU 密集型任务
                        let task = tokio::task::spawn_blocking(move || {
                            let parse_result = parser.parse_file(&path)?;
                            let entities = parse_result.into_entities();
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            Ok::<_, CadError>(entities)
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel_parse(
                            task,
                            ctx_entities,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::Parse(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::Vectorize => {
                        let ctx_entities = Arc::clone(&ctx.entities);
                        let ctx_polylines = Arc::clone(&ctx.polylines);
                        // P11 锐评修复 3: 使用 spawn_blocking 执行 CPU 密集型任务
                        let task = tokio::task::spawn_blocking(move || {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let entities = futures::executor::block_on(ctx_entities.read());
                            let polylines =
                                ProcessingPipeline::extract_polylines_from_entities(&entities);
                            Ok::<_, CadError>(polylines)
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel_vectorize(
                            task,
                            ctx_polylines,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::Vectorize(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::BuildTopology => {
                        let pipeline = Arc::clone(&self.base_pipeline);
                        let ctx_polylines = Arc::clone(&ctx.polylines);
                        let ctx_entities = Arc::clone(&ctx.entities);
                        let ctx_scene = Arc::clone(&ctx.scene);
                        // P11 锐评修复 3: 使用 spawn_blocking 执行 CPU 密集型任务
                        let task = tokio::task::spawn_blocking(move || {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let polylines = futures::executor::block_on(ctx_polylines.read());
                            let entities = futures::executor::block_on(ctx_entities.read());
                            let mut scene = pipeline.topo().build_scene(&polylines)?;
                            ProcessingPipeline::fill_scene_edges(&mut scene, &entities);
                            let entities_refs: Vec<&RawEntity> = entities.iter().collect();
                            ProcessingPipeline::auto_infer_boundaries(&mut scene, &entities_refs);
                            Ok::<_, CadError>(scene)
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel_build_topology(
                            task,
                            ctx_scene,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::BuildTopology(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::Validate => {
                        let pipeline = Arc::clone(&self.base_pipeline);
                        let ctx_scene = Arc::clone(&ctx.scene);
                        let ctx_validation = Arc::clone(&ctx.validation);
                        // P11 锐评修复 3: 使用 spawn_blocking 执行 CPU 密集型任务
                        let task = tokio::task::spawn_blocking(move || {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let scene_opt = futures::executor::block_on(ctx_scene.read());
                            let scene = scene_opt
                                .as_ref()
                                .ok_or_else(|| {
                                    CadError::internal(
                                        common_types::InternalErrorReason::InvariantViolated {
                                            invariant: "验证阶段需要场景已构建".to_string(),
                                        },
                                    )
                                })?
                                .clone();
                            let validation = pipeline.validator().validate(&scene)?;
                            Ok::<_, CadError>(validation)
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel_validate(
                            task,
                            ctx_validation,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::Validate(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::Export => {
                        let pipeline = Arc::clone(&self.base_pipeline);
                        let ctx_scene = Arc::clone(&ctx.scene);
                        let ctx_output = Arc::clone(&ctx.output_bytes);
                        // P11 锐评修复 3: 使用 spawn_blocking 执行 CPU 密集型任务
                        let task = tokio::task::spawn_blocking(move || {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let scene_opt = futures::executor::block_on(ctx_scene.read());
                            let scene = scene_opt
                                .as_ref()
                                .ok_or_else(|| {
                                    CadError::internal(
                                        common_types::InternalErrorReason::InvariantViolated {
                                            invariant: "导出阶段需要场景已构建".to_string(),
                                        },
                                    )
                                })?
                                .clone();
                            let export_result = pipeline.export().export(&scene)?;
                            Ok::<_, CadError>(export_result.bytes)
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel_export(
                            task,
                            ctx_output,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::Export(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::QualityCheck => {
                        let ctx_entities = Arc::clone(&ctx.entities);
                        let ctx_polylines = Arc::clone(&ctx.polylines);
                        let task = tokio::spawn(async move {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let entities: Arc<[RawEntity]> =
                                Arc::clone(&*ctx_entities.read().await);
                            let polylines: Arc<[Polyline]> =
                                Arc::clone(&*ctx_polylines.read().await);
                            // 在锁外处理数据（不持有锁）
                            if entities.is_empty() {
                                return Err(CadError::internal(
                                    common_types::InternalErrorReason::InvariantViolated {
                                        invariant: "质量预检失败：实体数量为 0".to_string(),
                                    },
                                ));
                            }
                            tracing::info!(
                                "质量预检：{} 个实体，{} 条多段线",
                                entities.len(),
                                polylines.len()
                            );
                            Ok::<_, CadError>(())
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel(
                            task,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::QualityCheck(task_with_timeout),
                            diagnostic,
                        ));
                    }
                    PipelineStage::AcousticAnalysis => {
                        let ctx_scene = Arc::clone(&ctx.scene);
                        let task = tokio::spawn(async move {
                            // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
                            let scene_guard = ctx_scene.read().await;
                            if scene_guard.is_none() {
                                return Err(CadError::internal(
                                    common_types::InternalErrorReason::InvariantViolated {
                                        invariant: "声学分析需要场景已构建".to_string(),
                                    },
                                ));
                            }
                            drop(scene_guard);
                            tracing::info!("声学分析阶段完成（占位符）");
                            Ok::<_, CadError>(())
                        });
                        // 添加超时保护和取消机制
                        let task_with_timeout = Self::with_timeout_and_cancel(
                            task,
                            Duration::from_millis(timeout_ms),
                            cancel_rx,
                            diagnostic.clone(),
                        );
                        tasks.push((
                            stage_clone,
                            TaskResult::AcousticAnalysis(task_with_timeout),
                            diagnostic,
                        ));
                    }
                }
            }

            // 收集结果并检查错误（处理超时和取消）
            let mut first_error: Option<CadError> = None;
            for (stage, task_result, diagnostic) in tasks {
                let result: Result<(), CadError> = match task_result {
                    TaskResult::Parse(task)
                    | TaskResult::Vectorize(task)
                    | TaskResult::BuildTopology(task)
                    | TaskResult::Validate(task)
                    | TaskResult::Export(task)
                    | TaskResult::QualityCheck(task)
                    | TaskResult::AcousticAnalysis(task) => {
                        match task.await {
                            Ok(cancel_or_err) => match cancel_or_err {
                                Ok(TimeoutOrCancel::Timeout) => {
                                    // P11 锐评修复 4: 使用执行前采集的诊断信息
                                    // P11 锐评建议 3: 记录 Prometheus 指标
                                    let stage_name = format!("{:?}", stage.stage);
                                    let elapsed = diagnostic.start_time.elapsed();
                                    STAGE_TIMEOUT_COUNTER
                                        .with_label_values(&[&stage_name])
                                        .inc();
                                    STAGE_DURATION_HISTOGRAM
                                        .with_label_values(&[&stage_name])
                                        .observe(elapsed.as_secs_f64());

                                    Err(CadError::internal(common_types::InternalErrorReason::ServiceUnavailable {
                                        service: format!(
                                            "阶段 {:?} 执行超时（{}ms）- 诊断：实体={}, 多段线={}, 场景={}, 耗时={:?}, 建议：增加超时或优化数据",
                                            stage.stage, stage.timeout_ms.unwrap_or(300_000),
                                            diagnostic.entities_count, diagnostic.polylines_count, diagnostic.scene_status,
                                            elapsed
                                        ),
                                    }))
                                }
                                Ok(TimeoutOrCancel::Cancelled) => {
                                    // P11 锐评建议 3: 记录 Prometheus 指标
                                    let stage_name = format!("{:?}", stage.stage);
                                    STAGE_CANCEL_COUNTER.with_label_values(&[&stage_name]).inc();

                                    Err(CadError::internal(
                                        common_types::InternalErrorReason::ServiceUnavailable {
                                            service: format!(
                                                "阶段 {:?} 被取消（其他任务失败）",
                                                stage.stage
                                            ),
                                        },
                                    ))
                                }
                                Ok(TimeoutOrCancel::Completed) => {
                                    // P11 锐评建议 3: 记录 Prometheus 指标
                                    let stage_name = format!("{:?}", stage.stage);
                                    let elapsed = diagnostic.start_time.elapsed();
                                    STAGE_COMPLETED_COUNTER
                                        .with_label_values(&[&stage_name])
                                        .inc();
                                    STAGE_DURATION_HISTOGRAM
                                        .with_label_values(&[&stage_name])
                                        .observe(elapsed.as_secs_f64());
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            },
                            Err(join_err) => Err(CadError::internal(
                                common_types::InternalErrorReason::Panic {
                                    message: format!(
                                        "阶段 {:?} 任务执行失败：{}",
                                        stage.stage, join_err
                                    ),
                                },
                            )),
                        }
                    }
                };

                match result {
                    Ok(()) => {
                        completed_stages.push(stage.stage.clone());
                    }
                    Err(e) => {
                        if stage.optional {
                            tracing::warn!("阶段 {:?} 执行失败但被跳过：{:?}", stage.stage, e);
                            completed_stages.push(stage.stage.clone());
                        } else {
                            // P11 锐评修复 2: 处理 SendError
                            match cancel_tx.send(()) {
                                Ok(_) => tracing::info!("已发送取消信号"),
                                Err(broadcast::error::SendError(_)) => {
                                    tracing::warn!("取消信号发送失败：没有接收者");
                                }
                            }
                            first_error = Some(e);
                            break;
                        }
                    }
                }
            }

            // 如果有错误，返回第一个错误
            if let Some(e) = first_error {
                return Err(e);
            }

            // 从待处理列表中移除已执行的阶段
            pending_stages.retain(|s| !completed_stages.contains(&s.stage));
        }

        Ok(())
    }

    // ========================================================================
    // 超时和取消包装器（P11 锐评修复版）
    // ========================================================================

    /// 通用超时和取消包装器（用于简单任务）
    ///
    /// P11 锐评修复：
    /// - 使用 StageDiagnostic 传递执行前采集的诊断信息
    /// - 正确处理取消信号
    fn with_timeout_and_cancel(
        task: tokio::task::JoinHandle<Result<(), CadError>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(())) => Ok(TimeoutOrCancel::Completed),
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// Parse 阶段专用超时包装器（返回 entities）
    fn with_timeout_and_cancel_parse(
        task: tokio::task::JoinHandle<Result<Vec<RawEntity>, CadError>>,
        ctx_entities: Arc<RwLock<Arc<[RawEntity]>>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(entities)) => {
                            // P11 锐评修复 1: 零拷贝 - 直接转换为 Arc<[T]>
                            *ctx_entities.write().await = entities.into();
                            Ok(TimeoutOrCancel::Completed)
                        }
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// Vectorize 阶段专用超时包装器（返回 polylines）
    fn with_timeout_and_cancel_vectorize(
        task: tokio::task::JoinHandle<Result<Vec<Polyline>, CadError>>,
        ctx_polylines: Arc<RwLock<Arc<[Polyline]>>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(polylines)) => {
                            // P11 锐评修复 1: 零拷贝 - 直接转换为 Arc<[T]>
                            *ctx_polylines.write().await = polylines.into();
                            Ok(TimeoutOrCancel::Completed)
                        }
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// BuildTopology 阶段专用超时包装器（返回 scene）
    fn with_timeout_and_cancel_build_topology(
        task: tokio::task::JoinHandle<Result<SceneState, CadError>>,
        ctx_scene: Arc<RwLock<Option<SceneState>>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(scene)) => {
                            *ctx_scene.write().await = Some(scene);
                            Ok(TimeoutOrCancel::Completed)
                        }
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// Validate 阶段专用超时包装器（返回 validation）
    fn with_timeout_and_cancel_validate(
        task: tokio::task::JoinHandle<Result<ValidationReport, CadError>>,
        ctx_validation: Arc<RwLock<Option<ValidationReport>>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(validation)) => {
                            *ctx_validation.write().await = Some(validation);
                            Ok(TimeoutOrCancel::Completed)
                        }
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// Export 阶段专用超时包装器（返回 output_bytes）
    fn with_timeout_and_cancel_export(
        task: tokio::task::JoinHandle<Result<Vec<u8>, CadError>>,
        ctx_output: Arc<RwLock<Vec<u8>>>,
        timeout: Duration,
        mut cancel_rx: broadcast::Receiver<()>,
        diagnostic: StageDiagnostic,
    ) -> tokio::task::JoinHandle<Result<TimeoutOrCancel, CadError>> {
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = cancel_rx.recv() => {
                    Ok(TimeoutOrCancel::Cancelled)
                }
                _ = tokio::time::sleep(timeout) => {
                    Ok(TimeoutOrCancel::Timeout)
                }
                result = task => {
                    match result {
                        Ok(Ok(bytes)) => {
                            *ctx_output.write().await = bytes;
                            Ok(TimeoutOrCancel::Completed)
                        }
                        Ok(Err(e)) => Err(e),
                        Err(join_err) => Err(CadError::internal(common_types::InternalErrorReason::Panic {
                            message: format!("阶段 {:?} 任务执行失败：{}", diagnostic.stage, join_err),
                        })),
                    }
                }
            }
        })
    }

    /// 检查阶段依赖是否已满足
    ///
    /// # 依赖关系
    ///
    /// - Parse → Vectorize → BuildTopology → Validate → Export
    /// - QualityCheck 需要 polylines 数据，所以需要 Parse + Vectorize 完成
    /// - AcousticAnalysis 可能需要验证结果，所以需要 BuildTopology + Validate 完成
    fn stage_dependencies_met(&self, stage: &PipelineStage, completed: &[PipelineStage]) -> bool {
        match stage {
            PipelineStage::Parse => true, // 无依赖
            PipelineStage::Vectorize => completed.contains(&PipelineStage::Parse),
            PipelineStage::BuildTopology => completed.contains(&PipelineStage::Vectorize),
            PipelineStage::Validate => completed.contains(&PipelineStage::BuildTopology),
            PipelineStage::Export => completed.contains(&PipelineStage::Validate),
            // QualityCheck 需要 polylines 数据，所以需要 Parse + Vectorize 完成
            PipelineStage::QualityCheck => {
                completed.contains(&PipelineStage::Parse)
                    && completed.contains(&PipelineStage::Vectorize)
            }
            // AcousticAnalysis 可能需要验证结果，所以需要 BuildTopology + Validate 完成
            PipelineStage::AcousticAnalysis => {
                completed.contains(&PipelineStage::BuildTopology)
                    && completed.contains(&PipelineStage::Validate)
            }
        }
    }

    // ========================================================================
    // 阶段执行方法
    // ========================================================================

    /// 执行解析阶段
    async fn execute_parse(&self, path: &Path, ctx: &StageContext) -> Result<(), CadError> {
        let parser = Arc::clone(&self.parser);
        let path = path.to_path_buf();
        let ctx_entities = Arc::clone(&ctx.entities);

        let result = spawn_blocking(move || {
            let parse_result = parser.parse_file(&path)?;
            Ok::<_, CadError>(parse_result)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::internal(
                common_types::InternalErrorReason::Panic {
                    message: format!("解析任务执行失败：{}", e),
                },
            ))
        })?;

        // P11 锐评修复 1: 零拷贝 - 直接转换为 Arc<[T]>
        let entities_arc: Arc<[RawEntity]> = result.into_entities().into();
        *ctx_entities.write().await = entities_arc.clone();
        let entities_len = ctx_entities.read().await.len();

        // 提取文字标注和标注尺寸统计
        let text_annotations = ProcessingPipeline::extract_text_annotations(&entities_arc);
        *ctx.text_annotations.write().await = text_annotations;
        let dimension_summary = ProcessingPipeline::extract_dimension_summary(&entities_arc);
        *ctx.dimension_summary.write().await = dimension_summary;

        tracing::info!("解析阶段完成：得到 {} 个实体", entities_len);
        Ok(())
    }

    /// 执行矢量化阶段
    async fn execute_vectorize(&self, ctx: &StageContext) -> Result<(), CadError> {
        // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
        let entities: Arc<[RawEntity]> = Arc::clone(&*ctx.entities.read().await);
        let polylines = ProcessingPipeline::extract_polylines_from_entities(&entities);
        drop(entities); // 释放读锁
                        // P11 锐评修复 1: 零拷贝 - 直接转换为 Arc<[T]>
        *ctx.polylines.write().await = polylines.into();
        let polylines_len = ctx.polylines.read().await.len();
        tracing::info!("矢量化阶段完成：得到 {} 条多段线", polylines_len);
        Ok(())
    }

    /// 执行拓扑构建阶段
    async fn execute_build_topology(&self, ctx: &StageContext) -> Result<(), CadError> {
        let topo = self.base_pipeline.topo().clone();
        // P11 锐评修复 1: 零拷贝 - Arc::clone 只增加引用计数
        let polylines: Arc<[Polyline]> = Arc::clone(&*ctx.polylines.read().await);
        let entities: Arc<[RawEntity]> = Arc::clone(&*ctx.entities.read().await);
        drop(polylines);
        drop(entities);

        let ctx_polylines = Arc::clone(&ctx.polylines);
        let ctx_entities = Arc::clone(&ctx.entities);
        let ctx_scene = Arc::clone(&ctx.scene);

        let result = spawn_blocking(move || {
            let polylines = futures::executor::block_on(ctx_polylines.read());
            let entities = futures::executor::block_on(ctx_entities.read());
            let mut scene = topo.build_scene(&polylines)?;
            // 填充边数据
            ProcessingPipeline::fill_scene_edges(&mut scene, &entities);
            // 语义推断
            let entities_refs: Vec<&RawEntity> = entities.iter().collect();
            ProcessingPipeline::auto_infer_boundaries(&mut scene, &entities_refs);
            Ok::<_, CadError>(scene)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::internal(
                common_types::InternalErrorReason::Panic {
                    message: format!("拓扑构建任务执行失败：{}", e),
                },
            ))
        })?;

        *ctx_scene.write().await = Some(result);
        tracing::info!("拓扑构建阶段完成");
        Ok(())
    }

    /// 执行验证阶段
    async fn execute_validate(&self, ctx: &StageContext) -> Result<(), CadError> {
        let validator = self.base_pipeline.validator().clone();
        let scene = {
            let scene_guard = ctx.scene.read().await;
            scene_guard
                .as_ref()
                .ok_or_else(|| {
                    CadError::internal(common_types::InternalErrorReason::InvariantViolated {
                        invariant: "验证阶段需要场景已构建".to_string(),
                    })
                })?
                .clone()
        };

        let result = spawn_blocking(move || {
            let validation = validator.validate(&scene)?;
            Ok::<_, CadError>(validation)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::internal(
                common_types::InternalErrorReason::Panic {
                    message: format!("验证任务执行失败：{}", e),
                },
            ))
        })?;

        if !result.passed && result.summary.error_count > 0 {
            let issues: Vec<common_types::ValidationIssue> = result
                .issues
                .into_iter()
                .map(|i| common_types::ValidationIssue {
                    code: i.code,
                    severity: match i.severity {
                        validator::checks::Severity::Error => common_types::Severity::Error,
                        validator::checks::Severity::Warning => common_types::Severity::Warning,
                        validator::checks::Severity::Info => common_types::Severity::Info,
                    },
                    message: i.message,
                    location: i.location.map(|l| common_types::ErrorLocation {
                        point: l.point,
                        segment: l.segment,
                        loop_index: l.loop_index,
                    }),
                })
                .collect();

            return Err(CadError::ValidationFailed {
                count: result.summary.error_count,
                warning_count: result.summary.warning_count,
                issues,
            });
        }

        *ctx.validation.write().await = Some(result);
        tracing::info!("验证阶段完成");
        Ok(())
    }

    /// 执行导出阶段
    async fn execute_export(&self, ctx: &StageContext) -> Result<(), CadError> {
        let export = self.base_pipeline.export().clone();
        let scene = {
            let scene_guard = ctx.scene.read().await;
            scene_guard
                .as_ref()
                .ok_or_else(|| {
                    CadError::internal(common_types::InternalErrorReason::InvariantViolated {
                        invariant: "导出阶段需要场景已构建".to_string(),
                    })
                })?
                .clone()
        };

        let result = spawn_blocking(move || {
            let export_result = export.export(&scene)?;
            Ok::<_, CadError>(export_result.bytes)
        })
        .await
        .unwrap_or_else(|e| {
            Err(CadError::internal(
                common_types::InternalErrorReason::Panic {
                    message: format!("导出任务执行失败：{}", e),
                },
            ))
        })?;

        *ctx.output_bytes.write().await = result;
        tracing::info!("导出阶段完成");
        Ok(())
    }

    /// 执行质量预检阶段
    async fn execute_quality_check(&self, ctx: &StageContext) -> Result<(), CadError> {
        // 质量预检：检查实体数量、多段线质量等
        let entities_len = ctx.entities.read().await.len();
        let polylines_len = ctx.polylines.read().await.len();
        tracing::info!(
            "质量预检：{} 个实体，{} 条多段线",
            entities_len,
            polylines_len
        );

        // 简单检查：实体数量不能为 0
        if entities_len == 0 {
            return Err(CadError::internal(
                common_types::InternalErrorReason::InvariantViolated {
                    invariant: "质量预检失败：实体数量为 0".to_string(),
                },
            ));
        }

        Ok(())
    }

    /// 执行声学分析阶段
    async fn execute_acoustic_analysis(&self, ctx: &StageContext) -> Result<(), CadError> {
        // 声学分析需要场景已构建
        let scene_guard = ctx.scene.read().await;
        if scene_guard.is_none() {
            return Err(CadError::internal(
                common_types::InternalErrorReason::InvariantViolated {
                    invariant: "声学分析需要场景已构建".to_string(),
                },
            ));
        }

        // 占位符：实际声学分析逻辑
        tracing::info!("声学分析阶段完成（占位符）");
        Ok(())
    }

    /// 获取流程配置
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// 获取基础流水线
    pub fn base_pipeline(&self) -> &ProcessingPipeline {
        &self.base_pipeline
    }
}

/// 预定义流程配置
impl PipelineConfig {
    /// 快速原型流程（跳过验证）
    pub fn quick_prototype() -> Self {
        Self {
            name: "quick_prototype".to_string(),
            description: Some("快速原型流程，跳过验证".to_string()),
            stages: vec![
                StageConfig {
                    stage: PipelineStage::Parse,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Vectorize,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::BuildTopology,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                // 跳过验证
                StageConfig {
                    stage: PipelineStage::Validate,
                    enabled: false,
                    optional: true,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Export,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
            ],
            parallel: false,
            file_types: None,
        }
    }

    /// 严格验证流程（带质量预检）
    pub fn strict_validation() -> Self {
        Self {
            name: "strict_validation".to_string(),
            description: Some("严格验证流程，带质量预检".to_string()),
            stages: vec![
                StageConfig {
                    stage: PipelineStage::Parse,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::QualityCheck,
                    enabled: true,
                    optional: true,
                    timeout_ms: Some(5000), // 5 秒超时
                },
                StageConfig {
                    stage: PipelineStage::Vectorize,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::BuildTopology,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Validate,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Export,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
            ],
            parallel: false,
            file_types: None,
        }
    }

    /// PDF 专用流程（需要矢量化）
    pub fn pdf_workflow() -> Self {
        Self {
            name: "pdf_workflow".to_string(),
            description: Some("PDF 专用流程".to_string()),
            stages: vec![
                StageConfig {
                    stage: PipelineStage::Parse,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Vectorize,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::BuildTopology,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Validate,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Export,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
            ],
            parallel: false,
            file_types: Some(vec!["pdf".to_string()]),
        }
    }

    /// DXF 专用流程（可能跳过矢量化）
    pub fn dxf_workflow() -> Self {
        Self {
            name: "dxf_workflow".to_string(),
            description: Some("DXF 专用流程".to_string()),
            stages: vec![
                StageConfig {
                    stage: PipelineStage::Parse,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                // DXF 已经有矢量数据，可能不需要矢量化
                StageConfig {
                    stage: PipelineStage::Vectorize,
                    enabled: false,
                    optional: true,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::BuildTopology,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Validate,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
                StageConfig {
                    stage: PipelineStage::Export,
                    enabled: true,
                    optional: false,
                    timeout_ms: None,
                },
            ],
            parallel: false,
            file_types: Some(vec!["dxf".to_string()]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.stages.len(), 5);
    }

    #[test]
    fn test_quick_prototype_config() {
        let config = PipelineConfig::quick_prototype();
        assert_eq!(config.name, "quick_prototype");
        // 验证阶段应该被禁用
        let validate_stage = config
            .stages
            .iter()
            .find(|s| s.stage == PipelineStage::Validate)
            .unwrap();
        assert!(!validate_stage.enabled);
    }

    #[test]
    fn test_strict_validation_config() {
        let config = PipelineConfig::strict_validation();
        assert_eq!(config.name, "strict_validation");
        // 应该包含质量预检阶段
        let qc_stage = config
            .stages
            .iter()
            .find(|s| s.stage == PipelineStage::QualityCheck)
            .unwrap();
        assert!(qc_stage.enabled);
        assert!(qc_stage.optional);
    }

    #[test]
    fn test_pdf_workflow() {
        let config = PipelineConfig::pdf_workflow();
        assert_eq!(config.name, "pdf_workflow");
        assert_eq!(config.file_types, Some(vec!["pdf".to_string()]));
    }

    #[test]
    fn test_dxf_workflow() {
        let config = PipelineConfig::dxf_workflow();
        assert_eq!(config.name, "dxf_workflow");
        assert_eq!(config.file_types, Some(vec!["dxf".to_string()]));
        // 矢量化阶段应该被禁用
        let vec_stage = config
            .stages
            .iter()
            .find(|s| s.stage == PipelineStage::Vectorize)
            .unwrap();
        assert!(!vec_stage.enabled);
    }

    #[test]
    fn test_stage_dependencies() {
        let pipeline = ConfigurablePipeline::from_config(PipelineConfig::default()).unwrap();

        // Parse 无依赖
        assert!(pipeline.stage_dependencies_met(&PipelineStage::Parse, &[]));

        // Vectorize 依赖 Parse
        assert!(!pipeline.stage_dependencies_met(&PipelineStage::Vectorize, &[]));
        assert!(pipeline.stage_dependencies_met(&PipelineStage::Vectorize, &[PipelineStage::Parse]));

        // BuildTopology 依赖 Vectorize
        assert!(!pipeline
            .stage_dependencies_met(&PipelineStage::BuildTopology, &[PipelineStage::Parse]));
        assert!(pipeline.stage_dependencies_met(
            &PipelineStage::BuildTopology,
            &[PipelineStage::Parse, PipelineStage::Vectorize]
        ));
    }
}
