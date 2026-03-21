//! wgpu 加速器后端
//!
//! 使用 WebGPU 计算着色器加速几何处理任务
//!
//! # 支持的加速操作
//!
//! - 边缘检测（Sobel/Canny 计算着色器）
//! - 轮廓提取（并行轮廓追踪）
//! - 端点吸附（GPU 并行 R*-tree 构建）
//! - 圆弧拟合（GPU 最小二乘拟合）

mod wgpu_accelerator;
mod edge_detect;
mod context;

pub use wgpu_accelerator::WgpuAccelerator;
pub use context::WgpuContext;
