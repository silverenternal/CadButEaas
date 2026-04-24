//! 流程编排服务
//!
//! 协调各服务调用顺序，暴露 API 接口
//!
//! # 声学分析服务集成
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use orchestrator::OrchestratorService;
//! use acoustic::{AcousticInput, AcousticRequest, SelectionBoundary, SelectionMode};
//! use common_types::scene::SceneState;
//!
//! let orchestrator = OrchestratorService::default();
//!
//! // 执行声学分析
//! let input = AcousticInput {
//!     scene: SceneState::default(),
//!     request: AcousticRequest::SelectionMaterialStats {
//!         boundary: SelectionBoundary::rect([0.0, 0.0], [10.0, 10.0]),
//!         mode: SelectionMode::Smart,
//!     },
//! };
//!
//! let output = orchestrator.calculate_acoustic(input).await?;
//! println!("计算完成，耗时：{:.2}ms", output.metrics.computation_time_ms);
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod configurable;
pub mod pipeline;
pub mod service;

pub use configurable::{ConfigurablePipeline, PipelineConfig, PipelineStage, StageConfig};
pub use pipeline::ProcessingPipeline;
pub use service::OrchestratorService;

// 重新导出声学分析相关类型 (deprecated - 声学功能已停止开发)
#[allow(deprecated)]
pub use acoustic::{
    AcousticError, AcousticInput, AcousticMetrics, AcousticOutput, AcousticRequest, AcousticResult,
    AcousticService, AcousticServiceConfig, Frequency, ReverberationFormula, SelectionBoundary,
    SelectionMode,
};
