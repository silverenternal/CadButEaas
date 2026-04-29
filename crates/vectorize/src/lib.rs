//! 图像矢量化服务
//!
//! 将光栅图像转换为结构化线段集合
//!
//! # 架构设计 (P11 锐评落实)
//!
//! 使用可拔插的 Accelerator 抽象层，支持多种后端：
//! - CPU (纯 Rust 实现，fallback)
//! - wgpu (GPU 计算着色器)
//! - CUDA (NVIDIA GPU，未来实现)
//! - OpenCL (跨平台 GPU，未来实现)
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use vectorize::{VectorizeService, VectorizeConfig};
//! use accelerator_cpu::CpuAccelerator;
//!
//! // 使用 CPU 加速器
//! let accelerator = Box::new(CpuAccelerator::new());
//! let service = VectorizeService::new(accelerator, VectorizeConfig::default());
//! ```

pub mod algorithms;
pub mod benchmark_data;
pub mod config;
pub mod pipeline;
pub mod quality;
pub mod service;
pub mod test_data;

pub use algorithms::*;
pub use config::{RasterStrategy, VectorizeConfig};
pub use pipeline::{
    detect_raster_kind, strategy_for_kind, DebugArtifact, FitResult, FitStage, FitStageConfig,
    ImageWithContext, PreprocessStage, PreprocessStageConfig, PrimitiveCandidate, RasterKind,
    RasterVectorizationOutput, RasterVectorizationReport, RefineStage, RefineStageConfig,
    SemanticCandidate, SkeletonResult, SkeletonizeStage, SkeletonizeStageConfig, Stage, StageError,
    StageResult, StageStats, SymbolCandidate, TextCandidate, ThresholdStage, ThresholdStageConfig,
    TraceResult, TraceStage, TraceStageConfig, VectorizationAttemptReport, VectorizationStrategy,
    VectorizePipeline, VectorizePipelineBuilder,
};
pub use service::VectorizeService;

#[cfg(feature = "registry")]
pub use accelerator_registry;
