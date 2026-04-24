//! 应用状态分层系统
//!
//! P11 锐评落实：将 CadApp 的 28 个字段按职责分离为：
//! - SceneState: 业务状态（场景数据）
//! - UIState: UI 状态（交互状态）
//! - RenderState: 渲染状态（相机、LOD）
//! - LoadingState: 异步加载状态

mod app;
mod loading;
mod render;
mod scene;
mod ui;

pub use app::AppState;
pub use loading::{GapMarkerData, LoadingState};
pub use render::{Camera2D, RenderState};
pub use scene::SceneState;
pub use ui::{AutoTraceResult, ToastNotification, ToastType, UIState};
