//! 底部日志面板（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 实现 Component trait，不再直接依赖 CadApp
//! 2. 使用 ComponentContext 产生命令，而不是直接修改 pending_action
//! 3. 只依赖 AppState，实现组件独立性
//! 4. 应用 macOS 主题样式

use crate::components::{Component, ComponentContext, EventResponse, UiEvent};
use crate::state::AppState;
use eframe::egui;
use egui::{Frame, Margin, Rounding};

/// 底部日志面板组件
pub struct BottomPanel {
    /// 是否自动滚动到底部
    auto_scroll: bool,
}

impl Default for BottomPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl BottomPanel {
    pub fn new() -> Self {
        Self { auto_scroll: true }
    }
}

impl Component for BottomPanel {
    fn name(&self) -> &str {
        "底部日志面板"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        egui::TopBottomPanel::bottom("log_panel")
            .min_height(150.0)
            .max_height(300.0)
            .default_height(200.0)
            .frame(Frame {
                fill: theme.toolbar_bg,
                rounding: Rounding::ZERO,
                shadow: theme.shadow.elevation_1,
                inner_margin: Margin::same(8.0),
                // P11 新增：顶部边框
                stroke: egui::Stroke::new(1.0, theme.border),
                ..Default::default()
            })
            .show(ctx, |ui| {
                // 标题使用强调色
                ui.heading(
                    egui::RichText::new("📜 日志控制台")
                        .strong()
                        .color(theme.accent),
                );
                ui.separator();

                // P11 增强：按钮悬停效果
                ui.style_mut().visuals.widgets.hovered.bg_fill = theme.accent.linear_multiply(0.15);
                ui.style_mut().visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.0, theme.accent);

                // 质量评分显示
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("解析质量评分:").color(theme.text_secondary));

                    // 根据评分显示不同颜色
                    let quality_color = if comp_ctx.state.scene.stats.total_edges > 0
                        && comp_ctx.state.scene.stats.visible_edges > 0
                    {
                        let ratio = comp_ctx.state.scene.stats.visible_edges as f64
                            / comp_ctx.state.scene.stats.total_edges as f64;
                        if ratio >= 0.9 {
                            egui::Color32::GREEN
                        } else if ratio >= 0.7 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::RED
                        }
                    } else {
                        egui::Color32::GRAY
                    };

                    ui.label(
                        egui::RichText::new(format!(
                            "{:.1}% (可见：{}/{})",
                            if comp_ctx.state.scene.stats.total_edges > 0 {
                                comp_ctx.state.scene.stats.visible_edges as f64
                                    / comp_ctx.state.scene.stats.total_edges as f64
                                    * 100.0
                            } else {
                                0.0
                            },
                            comp_ctx.state.scene.stats.visible_edges,
                            comp_ctx.state.scene.stats.total_edges
                        ))
                        .color(quality_color),
                    );
                });

                ui.separator();

                // 日志消息
                egui::ScrollArea::vertical()
                    .stick_to_bottom(self.auto_scroll)
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            for msg in &comp_ctx.state.log_messages {
                                ui.label(egui::RichText::new(msg).color(theme.text));
                            }
                        });
                    });

                // 控制选项
                ui.horizontal(|ui| {
                    ui.style_mut().visuals.widgets.hovered.bg_fill =
                        theme.accent.linear_multiply(0.1);
                    if ui.checkbox(&mut self.auto_scroll, "自动滚动").changed() {
                        comp_ctx.state.add_log(if self.auto_scroll {
                            "已启用日志自动滚动"
                        } else {
                            "已禁用日志自动滚动"
                        });
                    }
                    if ui.button("清空日志").clicked() {
                        comp_ctx.state.log_messages.clear();
                    }
                });
            });
    }

    fn handle_event(&mut self, _event: &UiEvent, _state: &mut AppState) -> EventResponse {
        EventResponse::ignored()
    }
}
