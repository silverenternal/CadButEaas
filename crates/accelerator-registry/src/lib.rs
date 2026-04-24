//! 加速器注册表
//!
//! 提供运行时加速器发现、注册和调度功能
//!
//! # 架构
//!
//! ```text
//! AcceleratorRegistry
//! ├── accelerators: Vec<Box<dyn Accelerator>>
//! ├── strategy: SchedulingStrategy
//! └── preferences: AcceleratorPreferences
//! ```

mod preferences;
mod registry;
mod strategy;

pub use preferences::*;
pub use registry::*;
pub use strategy::*;
