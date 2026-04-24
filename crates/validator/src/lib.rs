//! 几何验证服务
//!
//! # 概述
//!
//! 对场景状态执行声学仿真就绪性检查，确保几何数据符合仿真要求。
//!
//! # 验证项目
//!
//! ## 1. 闭合性检查（Closure Check）
//! 验证外边界和孔洞的首尾点距离是否小于容差（默认 0.5mm）。
//!
//! ## 2. 自相交检查（Self-Intersection Check）
//! 检测多边形是否存在自相交（蝴蝶结多边形）。
//!
//! ## 3. 洞关系检查（Hole Containment Check）
//! 验证所有孔洞是否位于外边界内部，且孔洞之间不互相穿越。
//!
//! ## 4. 微特征检查（Micro-Feature Check）
//! - 短边检测：边长 < `min_edge_length`（默认 10mm）
//! - 尖角检测：角度 < `min_angle_degrees`（默认 15°）
//! - 重复点检测：连续相同坐标点
//!
//! ## 5. 单位检查（Unit Check）
//! 验证场景单位是否明确指定，未指定时发出警告。
//!
//! # 输入输出
//!
//! ## 输入
//! - `SceneState`: 场景状态（外边界、孔洞、边界段等）
//! - `ValidationConfig`: 验证配置（容差、阈值等）
//!
//! ## 输出
//! - `ValidationReport`: 包含验证结果、问题列表、修复建议
//!
//! # 错误代码
//!
//! ## 错误（Error）- 必须修复
//!
//! | 代码 | 说明 | 严重性 |
//! |------|------|--------|
//! | `E001` | 环未闭合 | 阻断仿真 |
//! | `E002` | 自相交 | 阻断仿真 |
//! | `E003` | 孔洞在外边界外 | 阻断仿真 |
//! | `E004` | 孔洞互相穿越 | 阻断仿真 |
//!
//! ## 警告（Warning）- 建议修复
//!
//! | 代码 | 说明 | 影响 |
//! |------|------|------|
//! | `W001` | 短边 | 可能导致数值不稳定 |
//! | `W002` | 尖角 | 可能导致网格划分问题 |
//! | `W003` | 未指定单位 | 比例可能错误 |
//! | `W004` | 重复点 | 冗余数据 |
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use validator::{ValidatorService, ValidatorConfig};
//! use common_types::scene::SceneState;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let service = ValidatorService::new(ValidatorConfig::default());
//! let scene: SceneState = SceneState::default();
//!
//! let report = service.validate(&scene)?;
//!
//! if !report.passed {
//!     println!("验证失败：{} 个错误，{} 个警告", report.summary.error_count, report.summary.warning_count);
//!     for issue in &report.issues {
//!         println!("  - [{}] {}", issue.code, issue.message);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # 修复建议
//!
//! 验证服务会针对每个问题提供修复建议：
//!
//! - **闭合性问题**: 自动桥接缺口或提示用户确认
//! - **自相交**: 建议重新数字化或简化几何
//! - **短边/尖角**: 自动合并或提示用户确认

pub mod checks;
pub mod service;

pub use checks::*;
pub use service::{ValidatorConfig, ValidatorService};
