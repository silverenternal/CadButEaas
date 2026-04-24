//! Canvas 渲染组件（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 使用 Renderer trait 进行渲染，而不是直接写渲染逻辑
//! 2. CanvasWidget 只负责协调和交互处理
//! 3. 渲染逻辑委托给 Renderer trait 实现
//! 4. UI 叠加层委托给 OverlayRenderer

mod overlay_renderer;

use crate::app::CadApp;
use crate::components::SelectEdge;
use crate::render::{CpuRenderer, RenderContext, Renderer};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Response, Sense};
use interact::EdgeId;

#[cfg(feature = "gpu")]
use crate::gpu_renderer_enhanced::RenderEntity;

/// Canvas 小部件
pub struct CanvasWidget<'a> {
    app: &'a mut CadApp,
    /// 使用 Renderer trait object
    renderer: Box<dyn Renderer>,
    /// 叠加层渲染器（P11 改进：有状态的组件）
    overlay: overlay_renderer::OverlayRenderer,
}

impl<'a> CanvasWidget<'a> {
    pub fn new(app: &'a mut CadApp) -> Self {
        Self {
            app,
            renderer: Box::new(CpuRenderer::new()),
            overlay: overlay_renderer::OverlayRenderer::new(),
        }
    }

    /// 获取相机引用（简化访问）
    #[inline]
    fn camera(&self) -> &crate::state::Camera2D {
        &self.app.state.render.camera
    }

    /// 获取场景引用（简化访问）
    #[inline]
    fn scene(&self) -> &crate::state::SceneState {
        &self.app.state.scene
    }

    /// 获取 UI 状态引用（简化访问）
    #[inline]
    fn ui(&self) -> &crate::state::UIState {
        &self.app.state.ui
    }

    /// 将屏幕坐标转换为世界坐标
    fn screen_to_world(&self, screen: Pos2, rect: Rect) -> [f64; 2] {
        self.camera().screen_to_world(screen, rect)
    }

    // ========================================================================
    // 使用 Renderer trait 渲染场景
    // ========================================================================

    /// 使用 Renderer trait 渲染所有边
    fn render_with_renderer(&mut self, rect: Rect, painter: &egui::Painter) {
        let mut ctx = RenderContext { painter, rect };

        // 分离借用以避免冲突
        {
            let scene = &self.app.state.scene;
            let camera = &self.app.state.render.camera;

            // 使用 Renderer trait 渲染
            self.renderer.begin_frame();
            self.renderer.render_scene(&mut ctx, scene, camera);
        }
        {
            let ui = &self.app.state.ui;
            let scene = &self.app.state.scene;
            let camera = &self.app.state.render.camera;

            self.renderer.render_ui(&mut ctx, ui, scene, camera);
        }
        self.renderer.end_frame();
    }

    /// 处理鼠标交互（P11 锐评落实：通过 EventCollector 产生事件）
    fn handle_interaction(&mut self, response: &Response, rect: Rect) {
        // 滚轮缩放 - 使用 raw_scroll_delta 获取滚动事件
        let scroll_delta = response.ctx.input(|i| i.raw_scroll_delta);
        if scroll_delta.y != 0.0 {
            let zoom_factor = (scroll_delta.y / 100.0).exp();

            // P11 锐评落实：以鼠标指针为中心缩放
            if let Some(mouse_pos) = response.hover_pos() {
                // 1. 获取缩放前鼠标位置对应的世界坐标
                let world_pos_before = self.screen_to_world(mouse_pos, rect);

                // 2. 应用缩放
                let new_zoom = (self.camera().zoom * zoom_factor).clamp(0.1, 10.0);

                // 3. 计算缩放后鼠标位置对应的世界坐标
                self.app.state.render.camera.zoom = new_zoom;
                let world_pos_after = self.screen_to_world(mouse_pos, rect);

                // 4. 调整 pan，保持鼠标下的世界坐标不变
                self.app.state.render.camera.pan.x +=
                    ((world_pos_before[0] - world_pos_after[0]) * new_zoom as f64) as f32;
                self.app.state.render.camera.pan.y +=
                    ((world_pos_before[1] - world_pos_after[1]) * new_zoom as f64) as f32;
            } else {
                // 没有鼠标位置时，使用原点在中心缩放
                self.app.state.render.camera.zoom =
                    (self.camera().zoom * zoom_factor).clamp(0.1, 10.0);
            }
        }

        // 左键拖动平移（传统 CAD 行为）
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.app.state.render.camera.pan.x += delta.x;
            self.app.state.render.camera.pan.y += delta.y;
        }

        // 中键平移（备用）
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            self.app.state.render.camera.pan.x += delta.x;
            self.app.state.render.camera.pan.y += delta.y;
        }

        // P0 改进：添加悬停高亮（Hover Highlight）
        if let Some(pos) = response.hover_pos() {
            let world_pos = self.screen_to_world(pos, rect);
            let tolerance = 5.0 / self.camera().zoom as f64;

            let mut hover_edge: Option<EdgeId> = None;
            let mut tooltip_text: Option<String> = None;

            for edge in &self.scene().edges {
                let dist = Self::point_to_segment_distance(world_pos, edge.start, edge.end);
                if dist < tolerance {
                    hover_edge = Some(edge.id);

                    // P11 改进：构建悬停 Tooltip 文本
                    let length = ((edge.end[0] - edge.start[0]).powi(2)
                        + (edge.end[1] - edge.start[1]).powi(2))
                    .sqrt();

                    let mut text = format!("边 #{}\n长度：{:.2}", edge.id, length);
                    if let Some(layer) = &edge.layer {
                        text.push_str(&format!("\n图层：{}", layer));
                    }
                    if edge.is_wall {
                        text.push_str("\n类型：墙体");
                    }
                    if let Some(line_style) = &edge.line_style {
                        text.push_str(&format!("\n线型：{:?}", line_style));
                    }

                    tooltip_text = Some(text);
                    break; // 找到第一个就停止
                }
            }

            // 存储悬停状态（用于渲染）
            self.app.state.ui.hovered_edge = hover_edge;
            // P11 改进：存储悬停 Tooltip 文本
            self.app.state.ui.hovered_tooltip = tooltip_text;
        } else {
            // 鼠标不在画布上时，清除悬停
            self.app.state.ui.hovered_edge = None;
            self.app.state.ui.hovered_tooltip = None;
        }

        // 左键点击选择边（优化版：视口内选择 + 提前退出）
        // P11 锐评落实：清理双轨制，使用直接命令方案（不再使用 EventCollector）
        if response.clicked_by(egui::PointerButton::Primary) && !response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let world_pos = self.screen_to_world(pos, rect);

                // 转换容差到世界坐标
                let tolerance = 5.0 / self.camera().zoom as f64;

                let mut nearest_edge: Option<(EdgeId, f64)> = None;

                for edge in &self.scene().edges {
                    let dist = Self::point_to_segment_distance(world_pos, edge.start, edge.end);

                    if dist < tolerance
                        && (nearest_edge.is_none() || dist < nearest_edge.unwrap().1)
                    {
                        nearest_edge = Some((edge.id, dist));

                        // 提前退出：如果距离非常近（< 1 像素），直接返回
                        if dist < tolerance * 0.2 {
                            break;
                        }
                    }
                }

                // P11 锐评落实：直接执行命令，不再使用 EventCollector 双轨制
                if let Some((id, _)) = nearest_edge {
                    let modifiers = response.ctx.input(|i| i.modifiers);
                    let append = modifiers.shift || modifiers.ctrl;

                    let cmd = SelectEdge::new(id, append);
                    self.app
                        .command_manager
                        .execute(Box::new(cmd), &mut self.app.state);
                    self.app.add_log(&format!("选中边 {}", id));
                }
            }
        }

        // 右键开始/结束圈选
        if response.clicked_by(egui::PointerButton::Secondary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let world_pos = self.screen_to_world(pos, rect);

                if self.ui().is_lassoing {
                    // 结束圈选
                    self.app.state.ui.is_lassoing = false;
                    self.app
                        .add_log(&format!("圈选完成，{} 个点", self.ui().lasso_points.len()));
                } else {
                    // 开始圈选
                    self.app.state.ui.is_lassoing = true;
                    self.app.state.ui.lasso_points = vec![world_pos];
                    self.app.add_log("开始圈选...");
                }
            }
        }

        // 圈选过程中添加点
        if self.ui().is_lassoing {
            if let Some(pos) = response.hover_pos() {
                let world_pos = self.screen_to_world(pos, rect);

                // 限制点的密度
                if let Some(last) = self.ui().lasso_points.last() {
                    let dist = ((world_pos[0] - last[0]).powi(2)
                        + (world_pos[1] - last[1]).powi(2))
                    .sqrt();
                    if dist > 1.0 {
                        self.app.state.ui.lasso_points.push(world_pos);
                    }
                } else {
                    self.app.state.ui.lasso_points.push(world_pos);
                }
            }
        }
    }

    /// 计算点到线段的最近距离
    fn point_to_segment_distance(point: [f64; 2], seg_start: [f64; 2], seg_end: [f64; 2]) -> f64 {
        let dx = seg_end[0] - seg_start[0];
        let dy = seg_end[1] - seg_start[1];

        if dx.abs() < 1e-10 && dy.abs() < 1e-10 {
            // 线段退化为点
            return ((point[0] - seg_start[0]).powi(2) + (point[1] - seg_start[1]).powi(2)).sqrt();
        }

        let t =
            ((point[0] - seg_start[0]) * dx + (point[1] - seg_start[1]) * dy) / (dx * dx + dy * dy);

        let t = t.clamp(0.0, 1.0);

        let nearest_x = seg_start[0] + t * dx;
        let nearest_y = seg_start[1] + t * dy;

        ((point[0] - nearest_x).powi(2) + (point[1] - nearest_y).powi(2)).sqrt()
    }

    // ========================================================================
    // P11 改进：Toast 交互功能（关闭按钮）
    // ========================================================================

    /// 渲染 Toast 交互元素（关闭按钮）
    fn render_toast_interactions(&mut self, ui: &mut egui::Ui, rect: Rect) {
        use egui::FontId;

        let _now = std::time::Instant::now();
        let mut y_offset = 10.0f32;
        let mut toasts_to_dismiss: Vec<usize> = Vec::new();

        // 获取活跃的 Toast
        let active_toasts = self.overlay.get_active_toasts(&self.app.state.ui);

        for (idx, toast) in active_toasts {
            if !toast.dismissible {
                y_offset += 55.0;
                continue;
            }

            // Toast 位置
            let toast_rect = Rect::from_min_size(
                Pos2::new(rect.max.x - 260.0, rect.min.y + y_offset),
                egui::vec2(250.0, 45.0),
            );

            // 关闭按钮位置
            let close_button_rect = Rect::from_min_size(
                Pos2::new(toast_rect.max.x - 25.0, toast_rect.min.y + 12.0),
                egui::vec2(20.0, 20.0),
            );

            // 在 Toast 区域上放置一个不可见的交互区域
            let close_response = ui
                .allocate_rect(close_button_rect, Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);

            // 绘制关闭按钮文本（× 号）
            let painter = ui.painter();
            painter.text(
                close_button_rect.center(),
                egui::Align2::CENTER_CENTER,
                "×",
                FontId::proportional(16.0),
                Color32::from_rgba_unmultiplied(200, 200, 200, 255),
            );

            if close_response.clicked() {
                toasts_to_dismiss.push(idx);
            }

            y_offset += 55.0;
        }

        // 处理关闭（倒序删除，避免索引失效）
        for idx in toasts_to_dismiss.into_iter().rev() {
            self.app.state.ui.dismiss_toast(idx);
        }
    }
}

impl<'a> egui::Widget for CanvasWidget<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        let painter = ui.painter();

        // 绘制背景 - P11 修复：使用更深的背景色提升对比度
        painter.rect_filled(
            rect,
            egui::Rounding::ZERO,
            Color32::from_rgb(15, 15, 15), // 从 (30,30,30) 改为 (15,15,15)
        );

        // P0-2: 使用 Renderer trait 渲染
        self.render_with_renderer(rect, painter);

        // P11 改进：UI 叠加层委托给有状态的 OverlayRenderer
        self.overlay
            .render(painter, rect, &self.app.state, &self.app.gap_markers);

        // 处理交互
        self.handle_interaction(&response, rect);

        // 显示缩放信息 - P11 修复：使用更精确的格式显示小缩放值
        painter.text(
            rect.min + egui::vec2(10.0, 20.0),
            egui::Align2::LEFT_TOP,
            format!("缩放：{:.6}%", (self.camera().zoom * 100.0) as f64),
            egui::FontId::monospace(12.0),
            Color32::from_rgb(200, 200, 200), // 使用更亮的灰色
        );

        // P11 改进：绘制可交互的 Toast 关闭按钮
        self.render_toast_interactions(ui, rect);

        // P11 改进：显示悬停 Tooltip
        if let Some(ref tooltip) = self.app.state.ui.hovered_tooltip {
            if let Some(mouse_pos) = response.hover_pos() {
                // 使用 egui::Area 显示悬停提示
                use egui::{Area, Frame, Id, Margin, Order};

                let tooltip_id = Id::new("hover_tooltip");
                Area::new(tooltip_id)
                    .order(Order::Foreground)
                    .fixed_pos(mouse_pos + egui::vec2(15.0, 15.0))
                    .show(ui.ctx(), |ui| {
                        Frame::popup(ui.style())
                            .inner_margin(Margin::same(8.0))
                            .show(ui, |ui| {
                                ui.label(tooltip);
                            });
                    });
            }
        }

        response
    }
}
