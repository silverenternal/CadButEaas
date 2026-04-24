//! CPU 加速器后端 - 纯 Rust 实现
//!
//! 提供所有加速功能的 CPU 实现，作为 GPU 后端的 fallback

mod arc_fit;
mod contour_extract;
mod cpu_accelerator;
mod edge_detect;
mod snap;

pub use cpu_accelerator::CpuAccelerator;

// 重新导出子模块，供其他 crate 使用
pub use arc_fit::fit_arc_cpu;
pub use contour_extract::extract_contours_cpu;
pub use edge_detect::detect_edges_cpu;
pub use snap::snap_endpoints_cpu;
