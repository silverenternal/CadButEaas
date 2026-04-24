//! 加速器 API - 统一抽象层
//!
//! 提供可拔插的加速器架构，支持多种后端：
//! - CPU (纯 Rust 实现，fallback)
//! - wgpu (GPU 计算着色器)
//! - CUDA (NVIDIA GPU)
//! - OpenCL (跨平台 GPU)
//!
//! # 架构设计
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Acceleration Framework                    │
//! │                                                              │
//! │  ┌────────────────────────────────────────────────────────┐ │
//! │  │              AcceleratorRegistry (运行时注册表)          │ │
//! │  │  - 发现可用加速器 (CUDA/OpenCL/wgpu/CPU)                │ │
//! │  │  - 能力查询 (内存带宽/计算单元/特性支持)                 │ │
//! │  │  - 优先级调度 (性能评分/用户偏好/功耗限制)               │ │
//! │  └────────────────────────────────────────────────────────┘ │
//! │                              ↓                               │
//! │  ┌────────────────────────────────────────────────────────┐ │
//! │  │              Accelerator Trait (统一接口)               │ │
//! │  │  - edge_detect()                                       │ │
//! │  │  - contour_extract()                                   │ │
//! │  │  - arc_fit()                                           │ │
//! │  │  - rtree_build()                                       │ │
//! │  └────────────────────────────────────────────────────────┘ │
//! │                              ↓                               │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
//! │  │  CUDA    │  │ OpenCL   │  │  wgpu    │  │   CPU    │   │
//! │  │ Backend  │  │ Backend  │  │ Backend  │  │ Fallback │   │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod error;
mod trait_def;
mod types;

pub use error::{AcceleratorError, Result as AcceleratorResult};
pub use trait_def::{Accelerator, AcceleratorRef, OptionAcceleratorRef};
pub use types::*;
