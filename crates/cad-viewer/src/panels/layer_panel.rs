//! 图层管理面板（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 实现 Component trait，不再直接依赖 CadApp
//! 2. 使用 ComponentContext 产生命令，而不是直接修改 pending_action
//! 3. 只依赖 AppState，实现组件独立性
//! 4. 应用 macOS 主题样式

use crate::components::{Component, ComponentContext, EventResponse, UiEvent};
use crate::components::{SetLayerFilter, ToggleLayerVisibility};
use crate::state::AppState;
use eframe::egui;
use egui::{Frame, Margin, Rounding};

const FILTER_MODES: &[&str] = &["All", "Walls", "Openings", "Architectural", "Furniture"];

/// 图层管理面板组件
pub struct LayerPanel {
    /// 面板是否展开
    expanded: bool,
}

impl Default for LayerPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerPanel {
    pub fn new() -> Self {
        Self { expanded: true }
    }
}

impl Component for LayerPanel {
    fn name(&self) -> &str {
        "图层管理面板"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        egui::SidePanel::right("layer_panel")
            .min_width(200.0)
            .max_width(280.0)
            .frame(Frame {
                fill: theme.sidebar_bg,
                rounding: Rounding::ZERO,
                inner_margin: Margin::same(12.0),
                // P11 新增：左侧边框
                stroke: egui::Stroke::new(1.0, theme.border),
                ..Default::default()
            })
            .show(ctx, |ui| {
                // 标题使用强调色
                ui.heading(
                    egui::RichText::new("🗂️ 图层管理")
                        .strong()
                        .color(theme.accent),
                );
                ui.separator();

                // P11 增强：按钮悬停效果
                ui.style_mut().visuals.widgets.hovered.bg_fill = theme.accent.linear_multiply(0.1);
                ui.style_mut().visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.0, theme.accent);

                // 统计信息
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("总边数:").color(theme.text_secondary));
                    ui.label(format!("{}", comp_ctx.state.scene.stats.total_edges));
                    ui.label(egui::RichText::new("| 可见:").color(theme.text_secondary));
                    ui.label(format!("{}", comp_ctx.state.scene.stats.visible_edges));
                });

                if comp_ctx.state.scene.stats.total_edges > 0 {
                    let visible_ratio = comp_ctx.state.scene.stats.visible_edges as f64
                        / comp_ctx.state.scene.stats.total_edges as f64
                        * 100.0;
                    ui.label(
                        egui::RichText::new(format!("可见比例：{:.1}%", visible_ratio))
                            .color(theme.text_secondary),
                    );
                }

                ui.separator();

                // 过滤模式选择
                ui.label(
                    egui::RichText::new("过滤模式:")
                        .color(theme.text_secondary)
                        .size(12.0),
                );
                egui::ComboBox::from_id_salt("layer_filter_mode")
                    .selected_text(comp_ctx.state.scene.layers.filter_mode.clone())
                    .show_ui(ui, |ui| {
                        for mode in FILTER_MODES {
                            let display_name = match *mode {
                                "All" => "全部图层",
                                "Walls" => "仅墙体",
                                "Openings" => "仅门窗",
                                "Architectural" => "建筑图层",
                                "Furniture" => "仅家具",
                                _ => mode,
                            };
                            ui.style_mut().visuals.widgets.hovered.bg_fill =
                                theme.accent.linear_multiply(0.1);
                            if ui
                                .selectable_label(
                                    comp_ctx.state.scene.layers.filter_mode == *mode,
                                    display_name,
                                )
                                .clicked()
                            {
                                // P11 锐评落实：通过命令队列产生命令，而不是直接修改 pending_action
                                comp_ctx.push_command(SetLayerFilter::new(mode.to_string()));
                                comp_ctx.state.add_log(&format!("图层过滤模式：{}", mode));
                            }
                        }
                    });

                ui.separator();

                // 图层列表
                ui.label(
                    egui::RichText::new("图层列表:")
                        .color(theme.text_secondary)
                        .size(12.0),
                );
                ui.separator();

                // 使用 ScrollArea 支持长列表
                egui::ScrollArea::vertical()
                    .max_height(ui.available_height() - 100.0)
                    .show(ui, |ui| {
                        let layers = comp_ctx.state.scene.get_unique_layers();

                        if layers.is_empty() {
                            ui.label(egui::RichText::new("无图层数据").color(theme.text_secondary));
                            return;
                        }

                        for layer in layers {
                            let visible = comp_ctx.state.scene.is_layer_visible(&layer);

                            ui.horizontal(|ui| {
                                // 可见性切换按钮
                                let icon = if visible { "👁️" } else { "🚫" };
                                let button_text = format!("{} {}", icon, &layer);

                                if ui.button(button_text).clicked() {
                                    // P11 锐评落实：通过命令队列产生命令
                                    comp_ctx
                                        .push_command(ToggleLayerVisibility::new(layer.clone()));
                                    comp_ctx.state.add_log(&format!(
                                        "图层 '{}' 可见性：{}",
                                        &layer,
                                        if !comp_ctx.state.scene.is_layer_visible(&layer) {
                                            "开启"
                                        } else {
                                            "关闭"
                                        }
                                    ));
                                }

                                // 显示该图层的边数
                                let edge_count = comp_ctx
                                    .state
                                    .scene
                                    .edges
                                    .iter()
                                    .filter(|e| e.layer.as_deref() == Some(&layer))
                                    .count();
                                ui.label(
                                    egui::RichText::new(format!("({})", edge_count))
                                        .color(theme.text_secondary),
                                );
                            });
                        }
                    });

                // 底部操作按钮
                ui.separator();
                ui.horizontal(|ui| {
                    ui.style_mut().visuals.widgets.hovered.bg_fill =
                        theme.accent.linear_multiply(0.1);
                    if ui.button("全部显示").clicked() {
                        comp_ctx.push_command(SetLayerFilter::new("All".to_string()));
                        comp_ctx.state.add_log("图层可见性已重置");
                    }
                    if ui.button("重置图层").clicked() {
                        for edge in &mut comp_ctx.state.scene.edges {
                            edge.visible = None;
                        }
                        comp_ctx.state.scene.layers.visibility.clear();
                        comp_ctx.state.scene.update_visibility_stats();
                        comp_ctx.state.add_log("图层可见性已重置");
                    }
                });
            });
    }

    fn handle_event(&mut self, event: &UiEvent, _state: &mut AppState) -> EventResponse {
        match event {
            // 处理图层相关的键盘快捷键
            UiEvent::KeyPress { key, modifiers } => {
                // L: 切换图层面板
                if *key == egui::Key::L && modifiers.ctrl {
                    self.expanded = !self.expanded;
                    return EventResponse::consumed();
                }
                EventResponse::ignored()
            }
            _ => EventResponse::ignored(),
        }
    }
}
