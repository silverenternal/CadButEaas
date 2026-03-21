//! 组件系统
//!
//! P11 锐评落实：将 UI 组件从 CadApp 中解耦，实现独立可测试的组件架构

mod component;
mod commands;
mod event_collector;
mod registry;

pub use component::{Component, EventResponse, UiEvent, ComponentContext};
pub use commands::{
    Command, CommandManager,
    ToggleLayerVisibility, SetLayerFilter, SelectEdge,
    OpenFileCommand, ExportSceneCommand, AutoTraceCommand, DetectGapsCommand,
    UndoCommand, RedoCommand, ClearSelectionCommand, ToggleLassoToolCommand,
};
pub use event_collector::EventCollector;
pub use registry::ComponentRegistry;
