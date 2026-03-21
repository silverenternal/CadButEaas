//! 应用状态分层系统
//!
//! P11 锐评落实：将 CadApp 的 28 个字段按职责分离为：
//! - SceneState: 业务状态（场景数据）
//! - UIState: UI 状态（交互状态）
//! - RenderState: 渲染状态（相机、LOD）
//! - LoadingState: 异步加载状态

mod scene;
mod ui;
mod render;
mod loading;
mod app;

pub use scene::SceneState;
pub use ui::{UIState, AutoTraceResult, ToastNotification, ToastType};
pub use render::{RenderState, Camera2D};
pub use loading::LoadingState;
pub use app::AppState;
