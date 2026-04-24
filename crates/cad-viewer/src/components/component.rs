//! 组件 trait 和事件系统
//!
//! P11 锐评落实：
//! 1. 引入 ComponentContext，让组件通过命令队列产生命令
//! 2. 组件不再直接修改 pending_action，而是通过命令系统

use crate::state::AppState;
use crate::theme::MacOsTheme;
use eframe::egui;
use std::sync::Arc;

/// 事件响应
pub struct EventResponse {
    /// 事件是否已消耗（不再传递给其他组件）
    pub consumed: bool,
    /// 产生的命令（用于撤销/重做）
    pub commands: Vec<Box<dyn crate::components::Command>>,
}

impl EventResponse {
    /// 创建已消耗的事件响应
    pub fn consumed() -> Self {
        Self {
            consumed: true,
            commands: Vec::new(),
        }
    }

    /// 创建未消耗的事件响应
    pub fn ignored() -> Self {
        Self {
            consumed: false,
            commands: Vec::new(),
        }
    }

    /// 创建带有命令的事件响应
    pub fn with_command(cmd: impl crate::components::Command + 'static) -> Self {
        Self {
            consumed: true,
            commands: vec![Box::new(cmd)],
        }
    }

    /// 创建带有多个命令的事件响应（用于批量操作）
    #[allow(dead_code)]
    pub fn with_commands(cmds: Vec<Box<dyn crate::components::Command>>) -> Self {
        Self {
            consumed: true,
            commands: cmds,
        }
    }
}

impl Default for EventResponse {
    fn default() -> Self {
        Self::ignored()
    }
}

/// UI 事件
pub enum UiEvent {
    /// 鼠标点击（用于交互处理）
    #[allow(dead_code)]
    MouseClick {
        position: egui::Pos2,
        button: egui::PointerButton,
        modifiers: egui::Modifiers,
    },
    /// 鼠标移动（用于交互处理）
    #[allow(dead_code)]
    MouseMove {
        position: egui::Pos2,
        #[allow(dead_code)]
        delta: egui::Vec2,
    },
    /// 鼠标滚动
    #[allow(dead_code)]
    MouseScroll { delta: egui::Vec2 },
    /// 键盘按键
    #[allow(dead_code)]
    KeyPress {
        key: egui::Key,
        modifiers: egui::Modifiers,
    },
    /// 拖放
    #[allow(dead_code)]
    DragDrop { source: String, target: egui::Pos2 },
    /// 自定义事件
    #[allow(dead_code)]
    Custom { name: String, data: Option<String> },
}

/// 组件上下文 - 渲染时传递给组件，用于产生命令
///
/// P11 锐评落实：组件不再直接修改 pending_action，而是通过命令队列
/// P1 改进：添加 command_manager 引用，用于 UI 显示撤销/重做状态
/// P11 改进：添加 theme 引用，用于统一主题样式
/// P11 新增：添加玻璃渲染器可变引用，用于访问 GPU 效果
#[cfg(feature = "gpu")]
use crate::render::GlassEffectRenderer;

pub struct ComponentContext<'a> {
    pub state: &'a mut AppState,
    pub commands: Vec<Box<dyn crate::components::Command>>,
    pub command_manager: &'a crate::components::CommandManager,
    pub theme: Arc<MacOsTheme>,
    /// P11 新增：玻璃渲染器可变引用（用于 GPU 特性）
    #[cfg(feature = "gpu")]
    pub glass_renderer: Option<*mut GlassEffectRenderer>,
}

impl<'a> ComponentContext<'a> {
    #[cfg(not(feature = "gpu"))]
    pub fn new(
        state: &'a mut AppState,
        command_manager: &'a crate::components::CommandManager,
        theme: Arc<MacOsTheme>,
    ) -> Self {
        Self {
            state,
            commands: Vec::new(),
            command_manager,
            theme,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn new(
        state: &'a mut AppState,
        command_manager: &'a crate::components::CommandManager,
        theme: Arc<MacOsTheme>,
        glass_renderer: Option<&'a mut GlassEffectRenderer>,
    ) -> Self {
        Self {
            state,
            commands: Vec::new(),
            command_manager,
            theme,
            glass_renderer: glass_renderer.map(|r| r as *mut GlassEffectRenderer),
        }
    }

    /// 添加命令到队列
    pub fn push_command(&mut self, cmd: impl crate::components::Command + 'static) {
        self.commands.push(Box::new(cmd));
    }

    /// P11 新增：获取玻璃渲染器可变引用
    #[cfg(feature = "gpu")]
    pub fn glass_renderer_mut(&mut self) -> Option<&mut GlassEffectRenderer> {
        self.glass_renderer.map(|r| unsafe { &mut *r })
    }
}

/// 组件 trait（所有 UI 组件的基接口）
///
/// P11 锐评落实：render 方法现在接收 ComponentContext，用于产生命令
pub trait Component: Send + Sync {
    /// 组件名称（用于调试和日志）
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// 渲染组件
    ///
    /// 使用 ComponentContext 接收命令队列，而不是直接修改 pending_action
    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext);

    /// 处理输入事件
    fn handle_event(&mut self, event: &UiEvent, state: &mut AppState) -> EventResponse {
        let _ = (event, state);
        EventResponse::ignored()
    }

    /// 更新组件状态（每帧调用）（用于动画和定时任务）
    #[allow(dead_code)]
    fn update(&mut self, delta_time: f32, state: &AppState) {
        let _ = (delta_time, state);
    }
}
