//! 组件系统
//!
//! P11 锐评落实：将 UI 组件从 CadApp 中解耦，实现独立可测试的组件架构

mod commands;
mod component;
mod event_collector;
mod registry;

pub use commands::{
    AutoTraceCommand, ClearSelectionCommand, Command, CommandManager, DetectGapsCommand,
    ExportSceneCommand, OpenFileCommand, RedoCommand, SelectEdge, SetLayerFilter,
    ToggleLassoToolCommand, ToggleLayerVisibility, UndoCommand,
};
pub use component::{Component, ComponentContext, EventResponse, UiEvent};
pub use event_collector::EventCollector;
pub use registry::ComponentRegistry;
