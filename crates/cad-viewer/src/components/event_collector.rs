//! 事件收集器 - 从 egui 输入收集统一事件
//!
//! P11 锐评落实：让事件系统真正落地，而不是「纸上谈兵」

use crate::components::UiEvent;
use eframe::egui;
use egui::{Context, Response, Vec2};

/// 事件收集器
pub struct EventCollector {
    /// 收集的事件列表（pub(crate) 允许 canvas 模块访问）
    pub(crate) events: Vec<UiEvent>,
}

impl Default for EventCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl EventCollector {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// 从 egui 输入收集事件（用于 Canvas 交互）
    #[allow(dead_code)]
    pub fn collect_from_egui(&mut self, ctx: &Context, response: &Response) {
        // 鼠标点击
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let modifiers = ctx.input(|i| i.modifiers);
                self.events.push(UiEvent::MouseClick {
                    position: pos,
                    button: egui::PointerButton::Primary,
                    modifiers,
                });
            }
        }

        if response.clicked_by(egui::PointerButton::Secondary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let modifiers = ctx.input(|i| i.modifiers);
                self.events.push(UiEvent::MouseClick {
                    position: pos,
                    button: egui::PointerButton::Secondary,
                    modifiers,
                });
            }
        }

        // 鼠标移动
        if response.hovered() {
            if let Some(pos) = response.interact_pointer_pos() {
                let delta = ctx.input(|i| i.pointer.delta());
                if delta != Vec2::ZERO {
                    self.events.push(UiEvent::MouseMove {
                        position: pos,
                        delta,
                    });
                }
            }
        }

        // 鼠标滚动
        let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
        if scroll_delta != Vec2::ZERO {
            self.events.push(UiEvent::MouseScroll {
                delta: scroll_delta,
            });
        }

        // 键盘按键
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    modifiers,
                    pressed: true,
                    ..
                } = event
                {
                    self.events.push(UiEvent::KeyPress {
                        key: *key,
                        modifiers: *modifiers,
                    });
                }
            }
        });

        // 拖放事件
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.events.push(UiEvent::DragDrop {
                    source: "canvas".to_string(),
                    target: pos,
                });
            }
        }
    }

    /// 从全局输入收集事件（不依赖特定 response）
    pub fn collect_global(&mut self, ctx: &Context) {
        // 键盘按键（全局）
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Key {
                    key,
                    modifiers,
                    pressed: true,
                    ..
                } = event
                {
                    self.events.push(UiEvent::KeyPress {
                        key: *key,
                        modifiers: *modifiers,
                    });
                }
            }
        });

        // 鼠标滚动（全局）
        let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
        if scroll_delta != Vec2::ZERO {
            self.events.push(UiEvent::MouseScroll {
                delta: scroll_delta,
            });
        }
    }

    /// 添加自定义事件（用于扩展事件类型）
    #[allow(dead_code)]
    pub fn add_custom(&mut self, name: String, data: Option<String>) {
        self.events.push(UiEvent::Custom { name, data });
    }

    /// 获取并清空收集的事件
    pub fn drain(&mut self) -> Vec<UiEvent> {
        std::mem::take(&mut self.events)
    }

    /// 获取事件引用（不消耗）（用于调试和日志）
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &UiEvent> {
        self.events.iter()
    }

    /// 清空所有事件（用于重置）
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.events.clear()
    }
}
