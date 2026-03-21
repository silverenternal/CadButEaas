//! 组件注册表 - 管理组件生命周期和事件分发
//!
//! P11 锐评落实：让 CadApp 成为真正的协调器，而不是「什么都做」

use crate::components::{Component, UiEvent, ComponentContext};
use crate::state::AppState;
use crate::theme::MacOsTheme;
use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;

/// 组件注册表
///
/// 管理所有 UI 组件的生命周期，提供统一的事件分发和渲染接口
pub struct ComponentRegistry {
    /// 组件存储（按名称索引）
    components: HashMap<String, Box<dyn Component>>,
    /// 组件渲染顺序
    render_order: Vec<String>,
    /// 是否启用事件分发（用于暂停事件处理）
    #[allow(dead_code)]
    event_dispatch_enabled: bool,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            render_order: Vec::new(),
            event_dispatch_enabled: true,
        }
    }

    /// 注册组件
    pub fn register(&mut self, name: String, component: Box<dyn Component>) {
        if !self.components.contains_key(&name) {
            self.render_order.push(name.clone());
        }
        self.components.insert(name, component);
    }

    /// 注销组件（用于动态卸载）
    #[allow(dead_code)]
    pub fn unregister(&mut self, name: &str) -> Option<Box<dyn Component>> {
        self.render_order.retain(|n| n != name);
        self.components.remove(name)
    }

    /// 获取组件引用（用于查询组件状态）
    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<&dyn Component> {
        self.components.get(name).map(|b| b.as_ref())
    }

    /// 获取组件可变引用（用于外部访问组件）
    #[allow(dead_code)]
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn Component + '_)> {
        if self.components.contains_key(name) {
            Some(self.components.get_mut(name).unwrap().as_mut())
        } else {
            None
        }
    }

    /// 渲染所有组件
    ///
    /// P11 锐评落实：使用 ComponentContext 收集命令，而不是直接修改 pending_action
    /// 返回所有组件产生的命令列表
    #[cfg(not(feature = "gpu"))]
    pub fn render(
        &mut self,
        ctx: &egui::Context,
        state: &mut AppState,
        command_manager: &crate::components::CommandManager,
        theme: Arc<MacOsTheme>,
    ) -> Vec<Box<dyn crate::components::Command>> {
        let mut all_commands = Vec::new();

        for name in &self.render_order {
            if let Some(component) = self.components.get_mut(name) {
                let mut comp_ctx = ComponentContext::new(state, command_manager, theme.clone());
                component.render(ctx, &mut comp_ctx);
                all_commands.extend(comp_ctx.commands);
            }
        }

        all_commands
    }
    
    /// P11 新增：渲染所有组件（GPU 特性版本，传递 glass_renderer 引用）
    #[cfg(feature = "gpu")]
    pub fn render(
        &mut self,
        ctx: &egui::Context,
        state: &mut AppState,
        command_manager: &crate::components::CommandManager,
        theme: Arc<MacOsTheme>,
        mut glass_renderer: Option<&mut crate::render::GlassEffectRenderer>,
    ) -> Vec<Box<dyn crate::components::Command>> {
        let mut all_commands = Vec::new();

        for name in &self.render_order {
            if let Some(component) = self.components.get_mut(name) {
                // 使用 take() 来移动 glass_renderer，然后在下次迭代前恢复
                let glass_ref = glass_renderer.take();
                let mut comp_ctx = ComponentContext::new(state, command_manager, theme.clone(), glass_ref);
                component.render(ctx, &mut comp_ctx);
                all_commands.extend(comp_ctx.commands);
                // 恢复 glass_renderer 引用
                glass_renderer = comp_ctx.glass_renderer.map(|r| unsafe { &mut *r });
            }
        }

        all_commands
    }

    /// 分发事件到所有组件
    ///
    /// 返回已消耗的事件数量（用于调试）
    #[allow(dead_code)]
    pub fn dispatch_event(&mut self, event: &UiEvent, state: &mut AppState) -> usize {
        if !self.event_dispatch_enabled {
            return 0;
        }

        let mut consumed_count = 0;

        for name in &self.render_order {
            if let Some(component) = self.components.get_mut(name) {
                let response = component.handle_event(event, state);
                if response.consumed {
                    consumed_count += 1;
                }
            }
        }

        consumed_count
    }

    /// 分发事件并收集命令
    ///
    /// 返回所有组件产生的命令列表
    pub fn dispatch_event_with_commands(
        &mut self,
        event: &UiEvent,
        state: &mut AppState,
    ) -> Vec<Box<dyn crate::components::Command>> {
        let mut all_commands = Vec::new();

        for name in &self.render_order {
            if let Some(component) = self.components.get_mut(name) {
                let response = component.handle_event(event, state);
                all_commands.extend(response.commands);
            }
        }

        all_commands
    }

    /// 更新所有组件（每帧调用）（用于动画和定时任务）
    #[allow(dead_code)]
    pub fn update(&mut self, delta_time: f32, state: &AppState) {
        for component in self.components.values_mut() {
            component.update(delta_time, state);
        }
    }

    /// 启用/禁用事件分发（用于暂停事件处理）
    #[allow(dead_code)]
    pub fn set_event_dispatch(&mut self, enabled: bool) {
        self.event_dispatch_enabled = enabled;
    }

    /// 获取组件数量（用于调试）
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// 检查是否为空（用于调试）
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// 获取所有组件名称（用于调试和日志）
    #[allow(dead_code)]
    pub fn component_names(&self) -> Vec<&str> {
        self.render_order.iter().map(|s| s.as_str()).collect()
    }
}
