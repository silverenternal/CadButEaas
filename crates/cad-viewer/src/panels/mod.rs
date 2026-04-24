//! UI 面板模块
//!
//! P11 锐评落实：所有面板组件实现 Component trait，可独立测试和复用

mod bottom_panel;
mod layer_panel;
mod left_panel;
#[allow(dead_code)] // 预留用于未来功能
mod right_panel;
mod toolbar;
mod visual_settings_panel;

pub use bottom_panel::BottomPanel;
pub use layer_panel::LayerPanel;
pub use left_panel::LeftPanel;
#[allow(unused_imports)] // 预留用于未来功能
pub use right_panel::RightPanel;
pub use toolbar::Toolbar;
