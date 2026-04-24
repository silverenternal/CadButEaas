//! 右侧属性面板（P11 落实版）
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

/// 右侧属性面板组件
pub struct RightPanel {
    /// 展开状态
    expanded: bool,
}

impl Default for RightPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl RightPanel {
    pub fn new() -> Self {
        Self { expanded: true }
    }
}

impl Component for RightPanel {
    fn name(&self) -> &str {
        "属性编辑面板"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        egui::SidePanel::right("properties")
            .min_width(250.0)
            .max_width(350.0)
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
                    egui::RichText::new("📝 属性编辑")
                        .strong()
                        .color(theme.accent),
                );
                ui.separator();

                // P11 增强：按钮悬停效果
                ui.style_mut().visuals.widgets.hovered.bg_fill = theme.accent.linear_multiply(0.1);
                ui.style_mut().visuals.widgets.hovered.fg_stroke =
                    egui::Stroke::new(1.0, theme.accent);

                // 选中的边信息
                ui.heading(
                    egui::RichText::new("选中边信息")
                        .color(theme.text_secondary)
                        .size(13.0),
                );
                ui.separator();

                if let Some(edge_id) = comp_ctx.state.ui.selected_edge() {
                    ui.label(egui::RichText::new(format!("边 ID: {}", edge_id)).color(theme.text));

                    // 查找边的详细信息
                    if let Some(edge) = comp_ctx.state.scene.edges.iter().find(|e| e.id == edge_id)
                    {
                        ui.label(
                            egui::RichText::new(format!(
                                "起点：[{:.2}, {:.2}]",
                                edge.start[0], edge.start[1]
                            ))
                            .color(theme.text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "终点：[{:.2}, {:.2}]",
                                edge.end[0], edge.end[1]
                            ))
                            .color(theme.text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("长度：{:.2}", edge.length()))
                                .color(theme.text_secondary),
                        );
                        ui.label(
                            egui::RichText::new(format!("图层：{:?}", edge.layer))
                                .color(theme.text_secondary),
                        );
                    }

                    ui.separator();
                } else {
                    ui.label(egui::RichText::new("未选择边").color(theme.text_secondary));
                    ui.label(
                        egui::RichText::new("提示：点击画布中的线段进行选择")
                            .color(theme.text_secondary)
                            .size(11.0),
                    );
                }

                ui.separator();

                // 追踪结果
                ui.heading(
                    egui::RichText::new("🔍 追踪结果")
                        .color(theme.text_secondary)
                        .size(13.0),
                );
                ui.separator();

                if let Some(trace) = &comp_ctx.state.ui.auto_trace_result {
                    ui.label(
                        egui::RichText::new(format!("追踪边数：{}", trace.edges.len()))
                            .color(theme.text),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "环闭合：{}",
                            if trace.loop_closed { "是" } else { "否" }
                        ))
                        .color(theme.text),
                    );
                    ui.label(
                        egui::RichText::new(format!("多边形点数：{}", trace.polygon.len()))
                            .color(theme.text),
                    );
                } else {
                    ui.label(egui::RichText::new("暂无追踪结果").color(theme.text_secondary));
                    ui.label(
                        egui::RichText::new("提示：选择边后点击自动追踪按钮")
                            .color(theme.text_secondary)
                            .size(11.0),
                    );
                }

                ui.separator();

                // 视图控制
                ui.separator();
                ui.heading(
                    egui::RichText::new("👁️ 视图控制")
                        .color(theme.text_secondary)
                        .size(13.0),
                );
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("缩放:").color(theme.text_secondary));
                    ui.style_mut().visuals.widgets.hovered.bg_fill =
                        theme.accent.linear_multiply(0.1);
                    if ui.button("➖").clicked() {
                        comp_ctx.state.render.camera.zoom =
                            (comp_ctx.state.render.camera.zoom - 0.1).max(0.1);
                    }
                    ui.label(format!("{:.0}%", comp_ctx.state.render.camera.zoom * 100.0));
                    if ui.button("➕").clicked() {
                        comp_ctx.state.render.camera.zoom =
                            (comp_ctx.state.render.camera.zoom + 0.1).min(10.0);
                    }
                    if ui.button("🔄 重置").clicked() {
                        comp_ctx.state.render.camera.zoom = 1.0;
                        comp_ctx.state.render.camera.pan = egui::Vec2::ZERO;
                    }
                });
            });
    }

    fn handle_event(&mut self, _event: &UiEvent, _state: &mut AppState) -> EventResponse {
        // 属性面板目前不处理特定事件
        EventResponse::ignored()
    }
}
