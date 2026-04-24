//! 渲染系统
//!
//! P11 锐评落实：将渲染逻辑从 CanvasWidget 中解耦，实现统一的渲染架构

#[cfg(feature = "gpu")]
mod glass_effect;
#[cfg(feature = "gpu")]
mod gpu_renderer_wrapper;
#[cfg(feature = "gpu")]
mod gpu_tier;
mod render_queue;
mod renderer;

#[cfg(feature = "gpu")]
pub use glass_effect::GlassEffectRenderer;
#[cfg(feature = "gpu")]
pub use gpu_renderer_wrapper::GpuRendererWrapper;
#[cfg(feature = "gpu")]
pub use gpu_tier::{detect_gpu_tier, GpuInfo, GpuTier, GpuTierConfig};
pub use render_queue::{LayerId, MaterialId, RenderQueue};
pub use renderer::{CpuRenderer, RenderContext, Renderer};
