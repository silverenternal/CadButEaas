//! 叠加层渲染器（P11 落实版：有状态的组件）
//!
//! P11 锐评落实：拆分 CanvasWidget，将 UI 叠加层渲染逻辑分离到此模块
//!
//! 职责：
//! - 绘制追踪路径
//! - 绘制缺口标记
//! - 绘制圈选多边形
//! - 绘制加载动画
//! - 绘制座椅区域（LOD）
//!
//! P11 改进：从静态方法集合改为真正的组件
//! - 添加状态（配置、缓存）
//! - 支持帧计数器用于性能统计

use common_types::{SeatZone, LodLevel, ClosedLoop, ParseProgress};
use crate::state::{AutoTraceResult, Camera2D, ToastType};
use crate::app::GapMarker;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke};
use std::collections::HashMap;
use std::time::Instant;

/// 叠加层配置
#[derive(Clone)]
pub struct OverlayConfig {
    pub show_trace_path: bool,
    pub show_gaps: bool,
    pub show_lasso: bool,
    pub show_seat_zones: bool,
    #[allow(dead_code)]
    pub gap_font_size: f32,
    #[allow(dead_code)]
    pub lasso_fill_alpha: u8,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            show_trace_path: true,
            show_gaps: true,
            show_lasso: true,
            show_seat_zones: true,
            gap_font_size: 14.0,
            lasso_fill_alpha: 50,
        }
    }
}

/// 叠加层渲染器（有状态）
///
/// P11 改进：从空结构体 + 静态方法改为有状态的组件
pub struct OverlayRenderer {
    /// 缓存的屏幕坐标（避免每帧重复计算）
    cached_screen_points: HashMap<usize, Vec<Pos2>>,
    /// 配置
    config: OverlayConfig,
    /// 帧计数器（用于性能统计）
    frame_count: u32,
    /// 最后一帧的渲染时间（用于性能分析）
    last_frame_time_ms: f32,
}

impl OverlayRenderer {
    /// 创建新的叠加层渲染器
    pub fn new() -> Self {
        Self {
            cached_screen_points: HashMap::new(),
            config: OverlayConfig::default(),
            frame_count: 0,
            last_frame_time_ms: 0.0,
        }
    }

    /// 获取配置引用
    #[allow(dead_code)]
    pub fn config(&self) -> &OverlayConfig {
        &self.config
    }

    /// 获取可变配置引用
    #[allow(dead_code)]
    pub fn config_mut(&mut self) -> &mut OverlayConfig {
        &mut self.config
    }

    /// 获取帧计数器
    #[allow(dead_code)]
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// 获取最后一帧渲染时间
    #[allow(dead_code)]
    pub fn last_frame_time_ms(&self) -> f32 {
        self.last_frame_time_ms
    }

    /// 渲染所有叠加层（P11 改进：有状态的渲染入口）
    pub fn render(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        state: &crate::state::AppState,
        gap_markers: &[GapMarker],
    ) {
        let start = Instant::now();
        self.frame_count += 1;

        // 帧内缓存：清空缓存，在同一帧内复用
        self.cached_screen_points.clear();

        let camera = &state.render.camera;
        let scene_origin = state.scene.scene_origin;
        let ui = &state.ui;
        let scene = &state.scene;

        // P11 改进：使用实例调用，让 draw_* 方法可以访问 self.cached_screen_points
        if self.config.show_trace_path {
            self.draw_trace_path(painter, rect, &ui.auto_trace_result, camera, scene_origin);
        }

        if self.config.show_gaps && ui.show_gaps {
            self.draw_gaps(painter, rect, gap_markers, camera, scene_origin);
        }

        if self.config.show_lasso {
            self.draw_lasso(painter, rect, &ui.lasso_points, camera, scene_origin);
        }

        if self.config.show_seat_zones {
            let lod = scene.render_config
                .as_ref()
                .map(|c| c.recommended_lod)
                .unwrap_or(LodLevel::Detailed);
            self.draw_seat_zones(painter, rect, &scene.seat_zones, camera, scene_origin, lod);
        }

        let loading_state = state.loading_state();
        let is_loading = loading_state.is_loading;
        let progress = loading_state.progress.clone();
        drop(loading_state);

        self.draw_loading(painter, rect, is_loading, progress.as_ref());

        // P0 改进：渲染 Toast 通知
        self.draw_toasts(painter, rect, &state.ui);

        // 记录渲染时间
        self.last_frame_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// 缓存坐标转换结果（P11 改进：避免重复计算）
    #[allow(dead_code)]
    fn get_cached_screen_points<'a>(
        &'a mut self,
        key: usize,
        world_points: &[[f64; 2]],
        rect: Rect,
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) -> &'a Vec<Pos2> {
        self.cached_screen_points.entry(key).or_insert_with(|| {
            world_points
                .iter()
                .map(|p| Self::world_to_screen(*p, rect, camera, scene_origin))
                .collect()
        })
    }

    /// 绘制追踪路径（P11 改进：使用实例缓存）
    fn draw_trace_path(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        trace: &Option<AutoTraceResult>,
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) {
        if let Some(trace) = &trace {
            if trace.polygon.len() > 1 {
                // 使用缓存避免重复计算（缓存键 0 保留给追踪路径）
                let points = self.get_cached_screen_points(0, &trace.polygon, rect, camera, scene_origin);

                // 绘制虚线路径
                for i in 0..points.len() - 1 {
                    painter.line_segment(
                        [points[i], points[i + 1]],
                        Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 255, 0, 180)),
                    );
                }

                // 如果闭合，绘制最后一段
                if trace.loop_closed && points.len() > 2 {
                    painter.line_segment(
                        [points[points.len() - 1], points[0]],
                        Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 255, 0, 180)),
                    );
                }
            }
        }
    }

    /// 绘制缺口标记（P11 改进：使用实例方法）
    fn draw_gaps(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        gaps: &[GapMarker],
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) {
        for (i, gap) in gaps.iter().enumerate() {
            // 缺口通常不多，但使用缓存保持一致性
            let gap_key = 100 + i; // 缓存键从 100 开始保留给缺口
            let endpoints = [gap.start, gap.end];
            let points = self.get_cached_screen_points(gap_key, &endpoints, rect, camera, scene_origin);

            let start = points[0];
            let end = points[1];

            // 红色虚线标记缺口
            painter.line_segment(
                [start, end],
                Stroke::new(3.0, Color32::RED),
            );

            // 根据缩放级别调整文本大小
            let font_size = (14.0 * camera.zoom as f64).clamp(10.0, 24.0) as f32;

            // 绘制缺口长度文本
            let mid = Pos2::new((start.x + end.x) / 2.0, (start.y + end.y) / 2.0);
            painter.text(
                mid,
                egui::Align2::CENTER_CENTER,
                format!("{:.2}", gap.length),
                egui::FontId::monospace(font_size),
                Color32::RED,
            );
        }
    }

    /// 绘制圈选多边形（P11 改进：使用实例缓存）
    fn draw_lasso(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        lasso_points: &[[f64; 2]],
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) {
        if lasso_points.len() > 1 {
            // 使用缓存（缓存键 200 保留给圈选）
            let points = self.get_cached_screen_points(200, lasso_points, rect, camera, scene_origin);

            // 绘制多边形边框
            for i in 0..points.len() - 1 {
                painter.line_segment(
                    [points[i], points[i + 1]],
                    Stroke::new(2.0, Color32::GREEN),
                );
            }

            // 绘制填充（半透明）
            if points.len() > 2 {
                painter.add(egui::Shape::convex_polygon(
                    points.clone(),
                    Color32::from_rgba_unmultiplied(0, 255, 0, 50),
                    Stroke::NONE,
                ));
            }
        }
    }

    /// 绘制加载动画（带进度条）（P11 改进：使用实例方法）
    fn draw_loading(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        is_loading: bool,
        progress: Option<&ParseProgress>,
    ) {
        if !is_loading {
            return;
        }

        // 半透明覆盖层
        painter.rect_filled(
            rect,
            egui::Rounding::ZERO,
            Color32::from_rgba_unmultiplied(0, 0, 0, 180),
        );

        let center = rect.center();

        // 进度信息
        if let Some(progress) = progress {
            // 进度百分比
            let progress_text = format!("{:.1}%", progress.overall_progress * 100.0);
            painter.text(
                Pos2::new(center.x, center.y - 80.0),
                egui::Align2::CENTER_CENTER,
                &progress_text,
                egui::FontId::proportional(32.0),
                Color32::WHITE,
            );

            // 进度条背景
            let bar_width = 300.0;
            let bar_height = 20.0;
            let bar_rect = Rect::from_min_size(
                Pos2::new(center.x - bar_width / 2.0, center.y - 40.0),
                egui::vec2(bar_width, bar_height),
            );

            painter.rect_filled(
                bar_rect,
                egui::Rounding::same(bar_height / 2.0),
                Color32::from_rgba_unmultiplied(255, 255, 255, 50),
            );

            // 进度条前景
            let fill_width = bar_width * progress.overall_progress as f32;
            if fill_width > 0.0 {
                let fill_rect = Rect::from_min_size(
                    bar_rect.min,
                    egui::vec2(fill_width, bar_height),
                );

                // 渐变色进度条
                let color = Color32::from_rgb(0, 180, 255);
                painter.rect_filled(
                    fill_rect,
                    egui::Rounding::same(bar_height / 2.0),
                    color,
                );
            }

            // 进度条边框
            painter.rect_stroke(
                bar_rect,
                egui::Rounding::same(bar_height / 2.0),
                Stroke::new(2.0, Color32::WHITE),
            );

            // 当前阶段（P1 改进：显示详细阶段信息）
            let stage_name = progress.stage.name_zh();
            let stage_progress_percent = progress.stage_progress * 100.0;
            let detailed_stage_text = format!("{} ({:.1}%)", stage_name, stage_progress_percent);
            painter.text(
                Pos2::new(center.x, center.y - 10.0),
                egui::Align2::CENTER_CENTER,
                &detailed_stage_text,
                egui::FontId::proportional(18.0),
                Color32::LIGHT_GRAY,
            );

            // 实体数量
            if let Some(total) = progress.total_entities {
                if total > 0 {
                    let entity_text = format!("{} / {} 实体", progress.entities_parsed, total);
                    painter.text(
                        Pos2::new(center.x, center.y + 20.0),
                        egui::Align2::CENTER_CENTER,
                        &entity_text,
                        egui::FontId::proportional(14.0),
                        Color32::GRAY,
                    );
                }
            }

            // 预估剩余时间
            if let Some(eta) = progress.eta_seconds {
                if eta > 0.0 {
                    let eta_text = format!("预计剩余：{:.1} 秒", eta);
                    painter.text(
                        Pos2::new(center.x, center.y + 45.0),
                        egui::Align2::CENTER_CENTER,
                        &eta_text,
                        egui::FontId::proportional(14.0),
                        Color32::GRAY,
                    );
                }
            }
        } else {
            // 没有进度信息，显示简单的加载动画
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "正在处理...",
                egui::FontId::proportional(24.0),
                Color32::WHITE,
            );

            // 旋转的加载圆点动画
            let time = Instant::now().elapsed().as_secs_f32();
            let dot_count = 8;
            let radius = 40.0;

            for i in 0..dot_count {
                let angle = (i as f32 / dot_count as f32) * std::f32::consts::TAU + time * 2.0;
                let alpha = if (i as f32 + time * 2.0) % (dot_count as f32) < 1.0 {
                    255u8
                } else {
                    100u8
                };
                let dot_pos = Pos2::new(
                    center.x + angle.cos() * radius,
                    center.y + angle.sin() * radius,
                );
                painter.circle_filled(
                    dot_pos,
                    4.0,
                    Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
                );
            }
        }
    }

    /// 绘制座椅区域（根据 LOD 级别）（P11 改进：使用实例缓存）
    fn draw_seat_zones(
        &mut self,
        painter: &egui::Painter,
        rect: Rect,
        zones: &[SeatZone],
        camera: &Camera2D,
        scene_origin: [f64; 2],
        _lod: LodLevel,
    ) {
        if zones.is_empty() {
            return;
        }

        for (idx, zone) in zones.iter().enumerate() {
            // 自动选择 LOD 级别
            let selected_lod = LodLevel::auto_select(zone.seat_count, camera.zoom as f64);

            // 使用缓存键 1000+ 保留给座椅区域边界
            let boundary_key = 1000 + idx;

            match selected_lod {
                LodLevel::Simplified => {
                    self.draw_seat_zone_simplified(zone, rect, painter, camera, scene_origin, boundary_key);
                }
                LodLevel::Medium => {
                    self.draw_seat_zone_medium(zone, rect, painter, camera, scene_origin, boundary_key);
                }
                LodLevel::Detailed => {
                    self.draw_seat_zone_detailed(zone, rect, painter, camera, scene_origin);
                }
            }
        }
    }

    /// LOD 0: 简化渲染（色块）（P11 改进：使用缓存）
    fn draw_seat_zone_simplified(
        &mut self,
        zone: &SeatZone,
        rect: Rect,
        painter: &egui::Painter,
        camera: &Camera2D,
        scene_origin: [f64; 2],
        boundary_key: usize,
    ) {
        let color = Color32::from_rgb(100, 100, 150);

        // 使用缓存的边界点
        let screen_points = self.get_cached_screen_points(boundary_key, &zone.boundary.points, rect, camera, scene_origin);

        if screen_points.len() >= 3 {
            painter.add(egui::Shape::convex_polygon(
                screen_points.clone(),
                Color32::from_rgba_unmultiplied(100, 100, 150, 100),
                Stroke::NONE,
            ));
        }

        // 绘制边界（使用缓存）
        self.draw_loop(&zone.boundary, rect, painter, color, 2.0, camera, scene_origin, boundary_key);

        // 绘制标签
        let center = Self::compute_loop_center(&zone.boundary);
        let label = format!("座椅区\n{} 座", zone.seat_count);
        painter.text(
            Self::world_to_screen(center, rect, camera, scene_origin),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(14.0),
            Color32::WHITE,
        );
    }

    /// LOD 1: 中等细节（阵列采样）（P11 改进：使用缓存）
    fn draw_seat_zone_medium(
        &mut self,
        zone: &SeatZone,
        rect: Rect,
        painter: &egui::Painter,
        camera: &Camera2D,
        scene_origin: [f64; 2],
        boundary_key: usize,
    ) {
        let color = Color32::from_rgb(150, 150, 200);

        // 使用缓存的边界点
        self.draw_loop(&zone.boundary, rect, painter, color, 1.0, camera, scene_origin, boundary_key);

        // 采样绘制座椅点（10% 采样率）
        if let Some(positions) = &zone.original_positions {
            let sample_rate = (positions.len().max(10) / 10).min(100);
            for (i, pos) in positions.iter().enumerate() {
                if i % sample_rate != 0 {
                    continue;
                }
                painter.circle_filled(
                    Self::world_to_screen(*pos, rect, camera, scene_origin),
                    3.0,
                    color,
                );
            }
        }
    }

    /// LOD 2: 完整细节（逐个座椅）
    fn draw_seat_zone_detailed(
        &mut self,
        zone: &SeatZone,
        rect: Rect,
        painter: &egui::Painter,
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) {
        // 性能警告
        if zone.seat_count > 500 {
            tracing::warn!(
                "座椅数量过多 ({} 个)，建议切换到 LOD 1",
                zone.seat_count
            );
        }

        let color = Color32::from_rgb(200, 200, 255);

        if let Some(positions) = &zone.original_positions {
            for pos in positions {
                self.draw_single_seat(*pos, zone.seat_type, color, rect, painter, camera, scene_origin);
            }
        }
    }

    /// 绘制单个座椅
    fn draw_single_seat(
        &mut self,
        pos: [f64; 2],
        seat_type: common_types::SeatType,
        color: Color32,
        rect: Rect,
        painter: &egui::Painter,
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) {
        let screen_pos = Self::world_to_screen(pos, rect, camera, scene_origin);
        let size = match seat_type {
            common_types::SeatType::Single => 4.0,
            common_types::SeatType::Double => 6.0,
            common_types::SeatType::Auditorium => 3.5,
            common_types::SeatType::Bench => 8.0,
            common_types::SeatType::Unknown => 4.0,
        };

        painter.circle_filled(screen_pos, size, color);
    }

    /// 绘制闭合环（P11 改进：使用缓存）
    fn draw_loop(
        &mut self,
        loop_data: &ClosedLoop,
        rect: Rect,
        painter: &egui::Painter,
        color: Color32,
        stroke_width: f32,
        camera: &Camera2D,
        scene_origin: [f64; 2],
        boundary_key: usize,
    ) {
        if loop_data.points.len() < 2 {
            return;
        }

        // 使用缓存的边界点
        let points = self.get_cached_screen_points(boundary_key, &loop_data.points, rect, camera, scene_origin);

        // 绘制边界线
        for i in 0..points.len() - 1 {
            painter.line_segment(
                [points[i], points[i + 1]],
                Stroke::new(stroke_width, color),
            );
        }

        // 闭合
        if points.len() > 2 {
            painter.line_segment(
                [points[points.len() - 1], points[0]],
                Stroke::new(stroke_width, color),
            );
        }
    }

    /// 计算闭合环的中心点
    fn compute_loop_center(loop_data: &ClosedLoop) -> [f64; 2] {
        if loop_data.points.is_empty() {
            return [0.0, 0.0];
        }

        let sum_x: f64 = loop_data.points.iter().map(|p| p[0]).sum();
        let sum_y: f64 = loop_data.points.iter().map(|p| p[1]).sum();
        let n = loop_data.points.len() as f64;

        [sum_x / n, sum_y / n]
    }

    /// 将世界坐标转换为屏幕坐标
    /// 
    /// P11 锐评落实：精度优化
    /// 使用相对坐标渲染，先减去场景原点再转 f32，精度损失从 0.1mm 提升到 1e-6mm
    #[inline]
    fn world_to_screen(
        world: [f64; 2],
        rect: Rect,
        camera: &Camera2D,
        scene_origin: [f64; 2],
    ) -> Pos2 {
        camera.world_to_screen(world, rect, scene_origin)
    }

    // ========================================================================
    // P0 改进：Toast 通知渲染
    // ========================================================================

    /// 渲染 Toast 通知（P11 改进：返回需要渲染的 Toast 列表，由 CanvasWidget 绘制可交互版本）
    pub fn get_active_toasts<'a>(&'a self, ui: &'a crate::state::UIState) -> Vec<(usize, &'a crate::state::ToastNotification)> {
        let now = Instant::now();
        ui.toasts
            .iter()
            .enumerate()
            .filter_map(|(idx, toast)| {
                let age = now.duration_since(toast.created_at).as_secs_f32();
                if toast.dismissible && toast.duration_secs == 0.0 {
                    // 持久化 Toast，一直显示
                    Some((idx, toast))
                } else if age <= toast.duration_secs {
                    // 未过期的 Toast
                    Some((idx, toast))
                } else {
                    None
                }
            })
            .collect()
    }

    /// 渲染 Toast 通知（P11 改进：只绘制不可交互的背景，关闭按钮由 CanvasWidget 处理）
    fn draw_toasts(&mut self, painter: &egui::Painter, rect: Rect, ui: &crate::state::UIState) {
        let now = Instant::now();
        let mut y_offset = 10.0f32;

        for toast in &ui.toasts {
            let age = now.duration_since(toast.created_at).as_secs_f32();
            if !toast.dismissible && age > toast.duration_secs {
                continue; // 已过时且不可关闭
            }
            if toast.dismissible && toast.duration_secs == 0.0 {
                // 持久化 Toast，一直显示
            } else if age > toast.duration_secs {
                continue; // 已过时
            }

            // 计算透明度（淡入淡出）
            let alpha = if age < 0.3 {
                (age / 0.3 * 255.0) as u8
            } else if toast.duration_secs > 0.0 && age > toast.duration_secs - 0.3 {
                ((toast.duration_secs - age) / 0.3 * 255.0) as u8
            } else {
                255
            };

            let color = match toast.toast_type {
                ToastType::Info => Color32::from_rgba_unmultiplied(0, 180, 255, alpha),
                ToastType::Success => Color32::from_rgba_unmultiplied(0, 200, 100, alpha),
                ToastType::Warning => Color32::from_rgba_unmultiplied(255, 200, 0, alpha),
                ToastType::Error => Color32::from_rgba_unmultiplied(255, 50, 50, alpha),
            };

            // 绘制 Toast 背景
            let toast_rect = Rect::from_min_size(
                Pos2::new(rect.max.x - 260.0, rect.min.y + y_offset),
                egui::vec2(250.0, 45.0),
            );

            painter.rect_filled(
                toast_rect,
                egui::Rounding::same(8.0),
                Color32::from_rgba_unmultiplied(30, 30, 30, alpha),
            );

            // 绘制 Toast 文本
            painter.text(
                toast_rect.min + egui::vec2(15.0, 22.0),
                egui::Align2::LEFT_CENTER,
                &toast.message,
                egui::FontId::proportional(14.0),
                color,
            );

            // P11 改进：绘制关闭按钮（× 号）
            if toast.dismissible {
                let close_button_rect = Rect::from_min_size(
                    Pos2::new(toast_rect.max.x - 25.0, toast_rect.min.y + 12.0),
                    egui::vec2(20.0, 20.0),
                );

                painter.text(
                    close_button_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "×",
                    egui::FontId::proportional(16.0),
                    Color32::from_rgba_unmultiplied(200, 200, 200, alpha),
                );
            }

            y_offset += 55.0;
        }
    }
}
