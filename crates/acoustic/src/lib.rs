//! 声学分析服务
//!
//! # 功能范围
//!
//! ## P0 核心功能（本期实现）
//! - ✅ 选区材料统计：框选区域 → 显示表面积、材料分布
//! - ✅ 选区等效吸声面积：按频率计算 Σ(S × α)
//! - ✅ 房间级 T60 计算：选择完整房间 → 计算混响时间
//!
//! ## P1 提升功能（本期实现）
//! - ✅ 多区域对比分析：对比 2+ 区域的材料配置
//! - ✅ 频率响应曲线：绘制吸声系数 - 频率曲线
//!
//! ## 明确不做
//! - ❌ 选区 T60 计算（违反物理定义）
//! - ❌ 选区 C50/C80（需要声源和接收点）
//! - ❌ 声线追踪（超出当前范围）
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use acoustic::{AcousticService, AcousticServiceConfig};
//! use common_types::acoustic::{AcousticInput, AcousticRequest, SelectionBoundary, SelectionMode};
//! use common_types::scene::SceneState;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // 创建服务
//! let service = AcousticService::new(AcousticServiceConfig::default());
//!
//! // 准备输入
//! let input = AcousticInput {
//!     scene: SceneState::default(),
//!     request: AcousticRequest::SelectionMaterialStats {
//!         boundary: SelectionBoundary::rect([0.0, 0.0], [10.0, 10.0]),
//!         mode: SelectionMode::Smart,
//!     },
//! };
//!
//! // 执行分析（使用 process_sync 方法）
//! let output = service.process_sync(input)?;
//! println!("计算完成，耗时：{:.2}ms", output.metrics.computation_time_ms);
//! # Ok(())
//! # }
//! ```
//!
//! # 架构设计
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   AcousticService                       │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────────────┐  ┌──────────────────────────────┐ │
//! │  │ SelectionCalc    │  │ ReverberationCalculator      │ │
//! │  │ - 选区识别        │  │ - Sabine/Eyring 公式          │ │
//! │  │ - 材料统计        │  │ - 房间体积估算               │ │
//! │  │ - 等效吸声面积    │  │ - T60/EDT 计算                │ │
//! │  └──────────────────┘  └──────────────────────────────┘ │
//! │  ┌──────────────────────────────────────────────────┐   │
//! │  │ ComparativeAnalyzer                              │   │
//! │  │ - 多区域对比                                      │   │
//! │  │ - 差异分析                                        │   │
//! │  └──────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```

pub mod service;
pub mod selection;
pub mod reverberation;
pub mod comparative;
pub mod material_db;

// 重新导出常用类型
pub use service::{AcousticService, AcousticServiceConfig};
pub use selection::SelectionCalculator;
pub use reverberation::ReverberationCalculator;
pub use comparative::ComparativeAnalyzer;

// 重新导出 common-types 中的声学类型
pub use common_types::acoustic::{
    AcousticInput,
    AcousticOutput,
    AcousticRequest,
    AcousticResult,
    AcousticMetrics,
    SelectionBoundary,
    SelectionMode,
    NamedSelection,
    ComparisonMetric,
    ReverberationFormula,
    Frequency,
    SelectionMaterialStatsResult,
    MaterialDistribution,
    ReverberationResult,
    ComparativeAnalysisResult,
    RegionStats,
    AcousticError,
};

/// Crate 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
