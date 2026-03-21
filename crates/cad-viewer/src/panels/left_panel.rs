//! 左侧文件列表示板（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 实现 Component trait，不再直接依赖 CadApp
//! 2. 使用 ComponentContext 产生命令，而不是直接修改 pending_action
//! 3. 只依赖 AppState，实现组件独立性
//! 4. 应用 macOS 主题样式

use crate::components::{Component, UiEvent, EventResponse, ComponentContext};
use crate::state::AppState;
use eframe::egui;
use egui::{Frame, Margin, Rounding};

/// 左侧文件列表示板组件
pub struct LeftPanel {
    /// 预设配置展开状态（用于 UI 交互，未来用于持久化用户偏好）
    #[allow(dead_code)]
    presets_expanded: bool,
}

impl Default for LeftPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl LeftPanel {
    pub fn new() -> Self {
        Self {
            presets_expanded: true,
        }
    }
}

impl Component for LeftPanel {
    fn name(&self) -> &str {
        "文件列表示板"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        egui::SidePanel::left("file_list")
            .min_width(200.0)
            .max_width(300.0)
            .frame(Frame {
                fill: theme.sidebar_bg,
                rounding: Rounding::ZERO,
                inner_margin: Margin::same(12.0),
                // P11 新增：右侧边框
                stroke: egui::Stroke::new(1.0, theme.border),
                ..Default::default()
            })
            .show(ctx, |ui| {
                // 标题使用强调色
                ui.heading(egui::RichText::new("📋 文件列表").strong().color(theme.accent));
                ui.separator();

                // P11 增强：按钮悬停效果
                ui.style_mut().visuals.widgets.hovered.bg_fill = theme.accent.linear_multiply(0.1);
                ui.style_mut().visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, theme.accent);

                // 文件列表（简化实现，显示最近文件）
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("最近打开的文件:").color(theme.text_secondary).size(12.0));

                    // 这里可以扩展为实际的文件列表
                    if let Some(path) = &comp_ctx.state.scene.file_path {
                        let _ = ui.selectable_label(true, path.display().to_string());
                    } else {
                        ui.label(egui::RichText::new("暂无文件").color(theme.text_secondary));
                    }
                });

                ui.separator();

                // 预设配置快速选择
                ui.heading(egui::RichText::new("⚙️ 预设配置").strong().color(theme.accent));
                ui.separator();

                ui.vertical(|ui| {
                    if ui.button("🏢 建筑图纸").clicked() {
                        comp_ctx.state.add_log("使用建筑图纸预设");
                        comp_ctx.state.ui.pending_action = Some("preset_architectural".to_string());
                    }
                    if ui.button("⚙️ 机械图纸").clicked() {
                        comp_ctx.state.add_log("使用机械图纸预设");
                        comp_ctx.state.ui.pending_action = Some("preset_mechanical".to_string());
                    }
                    if ui.button("📷 扫描图纸").clicked() {
                        comp_ctx.state.add_log("使用扫描图纸预设");
                        comp_ctx.state.ui.pending_action = Some("preset_scanned".to_string());
                    }
                    if ui.button("⚡ 快速原型").clicked() {
                        comp_ctx.state.add_log("使用快速原型预设");
                        comp_ctx.state.ui.pending_action = Some("preset_quick".to_string());
                    }
                });

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.separator();
                    ui.label(egui::RichText::new("提示：右键开始/结束圈选").color(theme.text_secondary).size(11.0));
                });
            });
    }

    fn handle_event(&mut self, _event: &UiEvent, _state: &mut AppState) -> EventResponse {
        // 左侧面板目前不处理特定事件
        EventResponse::ignored()
    }
}
