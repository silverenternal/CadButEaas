//! 顶部工具栏面板（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 实现 Component trait，不再直接依赖 CadApp
//! 2. 使用 ComponentContext 产生命令，而不是直接修改 pending_action
//! 3. 只依赖 AppState，实现组件独立性
//! 4. 应用 macOS 主题样式

use crate::components::{
    AutoTraceCommand, ClearSelectionCommand, DetectGapsCommand, ExportSceneCommand,
    OpenFileCommand, RedoCommand, ToggleLassoToolCommand, UndoCommand,
};
use crate::components::{Component, ComponentContext, EventResponse, UiEvent};
use crate::state::AppState;
use eframe::egui;
use egui::{Frame, Margin, Rounding};

/// 顶部工具栏组件
pub struct Toolbar {
    /// 是否正在圈选
    is_lassoing: bool,
    /// 视觉效果设置窗口是否打开
    #[cfg(feature = "gpu")]
    show_visual_settings: bool,
}

impl Default for Toolbar {
    fn default() -> Self {
        Self::new()
    }
}

impl Toolbar {
    pub fn new() -> Self {
        Self {
            is_lassoing: false,
            #[cfg(feature = "gpu")]
            show_visual_settings: false,
        }
    }
}

impl Component for Toolbar {
    fn name(&self) -> &str {
        "顶部工具栏"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        egui::TopBottomPanel::top("toolbar")
            .frame(Frame {
                fill: theme.toolbar_bg,
                rounding: Rounding::ZERO,
                shadow: theme.shadow.elevation_1,
                inner_margin: Margin::same(8.0),
                // P11 新增：底部边框
                stroke: egui::Stroke::new(1.0, theme.border),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // 标题使用强调色
                    ui.label(
                        egui::RichText::new("🛠️ CAD Viewer")
                            .strong()
                            .color(theme.accent)
                            .size(15.0),
                    );
                    ui.separator();

                    // P11 增强：按钮悬停效果
                    let hovered_bg = theme.accent.linear_multiply(0.15);
                    ui.style_mut().visuals.widgets.hovered.bg_fill = hovered_bg;
                    ui.style_mut().visuals.widgets.hovered.fg_stroke =
                        egui::Stroke::new(1.0, theme.accent);

                    // 文件操作
                    if ui
                        .button("📁 打开文件")
                        .on_hover_text("打开 DXF/PDF 文件 (Ctrl+O)")
                        .clicked()
                    {
                        // P11 锐评落实：使用 push_command()，不再直接修改 pending_action
                        comp_ctx.push_command(OpenFileCommand);
                    }

                    if ui
                        .button("💾 导出场景")
                        .on_hover_text("导出场景 (Ctrl+S)")
                        .clicked()
                    {
                        comp_ctx.push_command(ExportSceneCommand);
                    }

                    ui.separator();

                    // 交互工具
                    if ui
                        .button("🎯 自动追踪")
                        .on_hover_text("自动追踪闭合轮廓 (T)")
                        .clicked()
                    {
                        comp_ctx.push_command(AutoTraceCommand);
                    }

                    if ui
                        .button("⭕ 缺口检测")
                        .on_hover_text("检测并标注缺口 (G)")
                        .clicked()
                    {
                        comp_ctx.push_command(DetectGapsCommand);
                    }

                    ui.separator();

                    // 圈选工具
                    let lasso_btn = if self.is_lassoing {
                        ui.button("🔴 结束圈选 (右键)")
                    } else {
                        ui.button("⭕ 圈选工具 (右键)")
                    };

                    if lasso_btn.clicked() {
                        self.is_lassoing = !self.is_lassoing;
                        comp_ctx.push_command(ToggleLassoToolCommand::new(self.is_lassoing));
                    }

                    ui.separator();

                    // 清除选择
                    if ui
                        .button("❌ 清除选择")
                        .on_hover_text("清除所有选择 (Esc)")
                        .clicked()
                    {
                        comp_ctx.push_command(ClearSelectionCommand);
                    }

                    ui.separator();

                    // P1 改进：添加撤销/重做按钮
                    let can_undo = comp_ctx.command_manager.undo_depth() > 0;
                    let can_redo = comp_ctx.command_manager.redo_depth() > 0;

                    // 撤销按钮
                    if ui
                        .add_enabled(can_undo, egui::Button::new("↩ 撤销"))
                        .on_hover_text(format!(
                            "撤销 (Ctrl+Z)\n剩余 {} 步",
                            comp_ctx.command_manager.undo_depth()
                        ))
                        .clicked()
                    {
                        comp_ctx.push_command(UndoCommand);
                    }

                    // 重做按钮
                    if ui
                        .add_enabled(can_redo, egui::Button::new("↪ 重做"))
                        .on_hover_text(format!(
                            "重做 (Ctrl+Y)\n剩余 {} 步",
                            comp_ctx.command_manager.redo_depth()
                        ))
                        .clicked()
                    {
                        comp_ctx.push_command(RedoCommand);
                    }

                    ui.separator();

                    // P11 新增：视觉效果开关
                    #[cfg(feature = "gpu")]
                    {
                        let effects_enabled = comp_ctx.state.ui.visual_settings.enable_effects;

                        let btn_text = if effects_enabled {
                            "🎨 视觉效果 ✓"
                        } else {
                            "🎨 视觉效果"
                        };
                        if ui
                            .button(btn_text)
                            .on_hover_text("切换高级视觉效果（毛玻璃/阴影/MSAA）")
                            .clicked()
                        {
                            // 切换效果开关
                            let new_state = !effects_enabled;
                            comp_ctx.state.ui.visual_settings.enable_effects = new_state;

                            // 应用配置到毛玻璃渲染器
                            let config = comp_ctx.state.ui.visual_settings.get_effective_config();
                            if let Some(glass_renderer) = comp_ctx.glass_renderer_mut() {
                                glass_renderer.set_enabled(new_state && config.glass_effect);
                                glass_renderer.set_blur_radius(config.glass_blur_radius);
                            }

                            comp_ctx.state.add_log(if new_state {
                                "已启用高级视觉效果"
                            } else {
                                "已关闭高级视觉效果"
                            });
                        }

                        // 显示当前 GPU 等级
                        let gpu_tier = comp_ctx.state.ui.visual_settings.gpu_tier;
                        ui.label(
                            egui::RichText::new(format!("GPU: {}", gpu_tier))
                                .color(theme.text_secondary)
                                .size(11.0),
                        );
                    }

                    ui.separator();

                    // 显示当前文件
                    if let Some(path) = &comp_ctx.state.scene.file_path {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("📄 {}", path.display()))
                                .color(theme.text_secondary),
                        );
                    }

                    // P11 新增：显示 WebSocket 连接状态
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let ws_connected = comp_ctx.state.ui.websocket_connected;
                        let ws_status = if ws_connected {
                            "🟢 WebSocket 已连接"
                        } else {
                            "⚪ WebSocket 未连接"
                        };
                        let ws_color = if ws_connected {
                            egui::Color32::from_rgb(100, 200, 100)
                        } else {
                            egui::Color32::from_rgb(150, 150, 150)
                        };
                        ui.label(egui::RichText::new(ws_status).color(ws_color).size(11.0))
                            .on_hover_text(if ws_connected {
                                "WebSocket 实时交互已启用"
                            } else {
                                "WebSocket 未连接，使用 HTTP 轮询模式"
                            });
                    });
                });
            });
    }

    fn handle_event(&mut self, event: &UiEvent, _state: &mut AppState) -> EventResponse {
        match event {
            // 处理键盘快捷键
            UiEvent::KeyPress { key, modifiers } => {
                // Ctrl+O: 打开文件
                if modifiers.ctrl && *key == egui::Key::O {
                    return EventResponse::with_command(OpenFileCommand);
                }
                // Ctrl+S: 导出场景
                if modifiers.ctrl && *key == egui::Key::S {
                    return EventResponse::with_command(ExportSceneCommand);
                }
                // Ctrl+Z: 撤销
                if modifiers.ctrl && *key == egui::Key::Z {
                    return EventResponse::with_command(UndoCommand);
                }
                // Ctrl+Y: 重做
                if modifiers.ctrl && *key == egui::Key::Y {
                    return EventResponse::with_command(RedoCommand);
                }
                // T: 自动追踪
                if *key == egui::Key::T && !modifiers.ctrl {
                    return EventResponse::with_command(AutoTraceCommand);
                }
                // G: 缺口检测
                if *key == egui::Key::G && !modifiers.ctrl {
                    return EventResponse::with_command(DetectGapsCommand);
                }
                // Esc: 清除选择
                if *key == egui::Key::Escape {
                    return EventResponse::with_command(ClearSelectionCommand);
                }

                EventResponse::ignored()
            }
            _ => EventResponse::ignored(),
        }
    }
}
