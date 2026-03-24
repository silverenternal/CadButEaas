//! 渲染器 trait 和实现（P11 落实版）
//!
//! P11 锐评落实：
//! 1. 让 Renderer trait 真正被 CanvasWidget 使用
//! 2. 让 RenderQueue 在 CpuRenderer 中实际工作
//! 3. 统一 CPU/GPU 渲染接口

use crate::state::{SceneState, UIState, Camera2D};
use crate::render::RenderQueue;
use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Painter};

/// 渲染上下文
pub struct RenderContext<'a> {
    pub painter: &'a Painter,
    pub rect: Rect,
}

/// 渲染器 trait（统一管理 CPU/GPU 渲染）
pub trait Renderer: Send + Sync {
    /// 渲染器名称（用于调试和日志）
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// 开始帧
    fn begin_frame(&mut self);

    /// 渲染场景
    fn render_scene(&mut self, ctx: &mut RenderContext, scene: &SceneState, camera: &Camera2D);

    /// 渲染 UI 叠加层
    fn render_ui(&mut self, ctx: &mut RenderContext, ui: &UIState, scene: &SceneState, camera: &Camera2D);

    /// 结束帧
    fn end_frame(&mut self);

    /// 窗口大小改变（用于 GPU 资源重建）
    #[allow(dead_code)]
    fn resize(&mut self, width: u32, height: u32);
}

/// CPU 渲染器（egui 原生）- P11 落实版：使用 RenderQueue
pub struct CpuRenderer {
    /// 渲染队列（批量合并）
    render_queue: RenderQueue,
}

impl Default for CpuRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuRenderer {
    pub fn new() -> Self {
        Self {
            render_queue: RenderQueue::new(),
        }
    }

    /// 根据图层获取颜色
    fn get_layer_color(layer: Option<&str>) -> Color32 {
        let layer_upper = layer.unwrap_or("").to_uppercase();

        // 墙体图层 - 使用更亮的红色
        if layer_upper == "WALL" || layer_upper == "墙体" || layer_upper == "A-WALL"
            || layer_upper.starts_with("A-WALL-") || layer_upper.starts_with("WALL-")
            || layer_upper.contains("WALL") || layer_upper.contains("墙体") {
            return Color32::from_rgb(255, 100, 100);
        }

        // 门窗图层 - 使用更亮的黄色
        if layer_upper == "DOOR" || layer_upper == "门" || layer_upper == "A-DOOR"
            || layer_upper.starts_with("DOOR-") || layer_upper.starts_with("A-DOOR-")
            || layer_upper.contains("门") {
            return Color32::from_rgb(255, 255, 50);
        }
        if layer_upper == "WINDOW" || layer_upper == "窗" || layer_upper == "A-WINDOW"
            || layer_upper.starts_with("WINDOW-") || layer_upper.starts_with("A-WINDOW-")
            || layer_upper.contains("窗") {
            return Color32::from_rgb(50, 255, 150);
        }

        // 家具图层 - 使用更亮的蓝色
        if layer_upper == "FURNITURE" || layer_upper == "家具" || layer_upper == "A-FURN"
            || layer_upper.starts_with("FURN-") || layer_upper.starts_with("A-FURN-")
            || layer_upper == "A-FURNITURE" || layer_upper.contains("家具") {
            return Color32::from_rgb(150, 200, 255);
        }

        // 标注图层 - 使用更亮的黄色
        if layer_upper == "DIMENSION" || layer_upper == "标注" || layer_upper == "A-DIMS"
            || layer_upper.starts_with("DIMS-") || layer_upper == "A-ANNO-DIMS"
            || layer_upper.contains("标注") || layer_upper.contains("DIM") {
            return Color32::from_rgb(255, 255, 100);
        }

        // 默认颜色 - 使用白色而非灰色，提升对比度
        Color32::from_rgb(255, 255, 255)
    }

    /// Cohen-Sutherland 线段裁剪算法
    fn line_in_viewport(
        start: [f64; 2],
        end: [f64; 2],
        viewport_min: [f64; 2],
        viewport_max: [f64; 2],
    ) -> bool {
        let code1 = Self::compute_out_code(start, viewport_min, viewport_max);
        let code2 = Self::compute_out_code(end, viewport_min, viewport_max);

        if (code1 & code2) != 0 {
            return false;
        }

        true
    }

    fn compute_out_code(point: [f64; 2], viewport_min: [f64; 2], viewport_max: [f64; 2]) -> u8 {
        let mut code = 0u8;

        if point[0] < viewport_min[0] {
            code |= 0b0001;
        } else if point[0] > viewport_max[0] {
            code |= 0b0010;
        }

        if point[1] < viewport_min[1] {
            code |= 0b0100;
        } else if point[1] > viewport_max[1] {
            code |= 0b1000;
        }

        code
    }

    /// 构建渲染队列（P0-3 修复：LOD 动态线宽）
    fn build_render_queue(&mut self, scene: &SceneState, camera: &Camera2D, rect: Rect) {
        self.render_queue.clear();

        let (viewport_min, viewport_max) = camera.get_viewport_bounds(rect);

        // 添加边距，避免边缘裁剪
        let margin = 100.0 / camera.zoom as f64;
        let viewport_min = [viewport_min[0] - margin, viewport_min[1] - margin];
        let viewport_max = [viewport_max[0] + margin, viewport_max[1] + margin];

        let mut visible_count = 0;
        let mut clipped_count = 0;

        // P0-3 修复：根据缩放级别动态调整基础线宽
        // 缩放级别越低（zoom < 1），线宽越细，避免画面混乱
        // 缩放级别越高（zoom > 1），线宽越粗，保持清晰度
        let base_line_width = Self::calculate_lod_line_width(camera.zoom);

        for edge in &scene.edges {
            // 检查图层可见性
            if let Some(visible) = edge.visible {
                if !visible {
                    continue;
                }
            } else if let Some(layer) = &edge.layer {
                if !scene.layers.is_visible(layer) {
                    continue;
                }
            }

            // 检查视口可见性
            if !Self::line_in_viewport(edge.start, edge.end, viewport_min, viewport_max) {
                clipped_count += 1;
                continue;
            }
            visible_count += 1;

            // 转换坐标
            let start = camera.world_to_screen(edge.start, rect, scene.scene_origin);
            let end = camera.world_to_screen(edge.end, rect, scene.scene_origin);

            // 获取材质 - P0-3 修复：LOD 动态线宽 + 图层颜色
            let color = Self::get_layer_color(edge.layer.as_deref());
            
            // P0-3 修复：根据图层语义调整线宽
            // 墙体图层使用更粗的线，家具图层使用较细的线
            let layer_width_multiplier = Self::get_layer_width_multiplier(edge.layer.as_deref());
            let line_width = base_line_width * layer_width_multiplier;
            
            let material = crate::render::MaterialId {
                color,
                line_width,
            };

            let layer = crate::render::LayerId {
                name: edge.layer.clone().unwrap_or_default(),
            };

            // 添加到队列
            self.render_queue.add_line(start, end, material, layer);
        }

        // P11 调试：打印渲染统计
        eprintln!("[DEBUG] render: visible={}, clipped={}, total={}, zoom={:.2}", 
                  visible_count, clipped_count, scene.edges.len(), camera.zoom);
    }

    /// P0-3 修复：根据缩放级别计算 LOD 线宽
    ///
    /// ## LOD 策略（P11 修复版）
    /// - zoom < 0.3: 0.8-1.25px（概览模式，有线宽下限）
    /// - zoom 0.3-1.5: 1.25-2.45px（正常模式，线性过渡）
    /// - zoom > 1.5: 2.5px（放大模式，保持恒定避免过粗）
    fn calculate_lod_line_width(zoom: f32) -> f32 {
        const MIN_WIDTH: f32 = 0.8;   // P11 修复：提升最小线宽（原 0.5）
        const MAX_WIDTH: f32 = 2.5;   // P11 修复：降低最大线宽（原 3.0）

        // P11 修复：使用分段函数，避免极端值
        if zoom < 0.3 {
            // 缩小级别：线宽随 zoom 降低而变细，但有下限
            MIN_WIDTH + zoom * 1.5
        } else if zoom < 1.5 {
            // 标准级别：线性过渡
            MIN_WIDTH + (zoom - 0.3) * 1.2
        } else {
            // 放大级别：保持恒定，避免过粗
            MAX_WIDTH
        }
        .clamp(MIN_WIDTH, MAX_WIDTH)
    }

    /// P0-3 修复：根据图层语义获取线宽乘数
    ///
    /// ## 图层线宽优先级（P11 修复版）
    /// - 墙体：1.3x（原 1.5x，降低避免过粗）
    /// - 门窗：1.1x（原 1.2x，降低避免过粗）
    /// - 家具：0.9x（新增，略低于默认）
    /// - 标注：0.7x（原 0.8x，降低避免喧宾夺主）
    /// - 其他：1.0x（默认）
    fn get_layer_width_multiplier(layer: Option<&str>) -> f32 {
        let layer_upper = layer.unwrap_or("").to_uppercase();

        // 墙体图层 - 1.3x（P11 修复：降低 from 1.5x）
        if layer_upper.contains("WALL") || layer_upper.contains("墙体")
            || layer_upper.contains("结构") || layer_upper.contains("STRUCT") {
            return 1.3;
        }

        // 门窗图层 - 1.1x（P11 修复：降低 from 1.2x）
        if layer_upper.contains("DOOR") || layer_upper.contains("门")
            || layer_upper.contains("WINDOW") || layer_upper.contains("窗") {
            return 1.1;
        }

        // 标注图层 - 0.7x（P11 修复：降低 from 0.8x）
        if layer_upper.contains("DIM") || layer_upper.contains("标注")
            || layer_upper.contains("TEXT") || layer_upper.contains("注释") {
            return 0.7;
        }

        // 家具图层 - 0.9x（P11 新增）
        if layer_upper.contains("FURNITURE") || layer_upper.contains("家具") {
            return 0.9;
        }

        // 其他 - 默认
        1.0
    }
}

impl Renderer for CpuRenderer {
    fn name(&self) -> &str {
        "CPU Renderer"
    }

    fn begin_frame(&mut self) {
        self.render_queue.clear();
    }

    fn render_scene(&mut self, ctx: &mut RenderContext, scene: &SceneState, camera: &Camera2D) {
        // 构建渲染队列
        self.build_render_queue(scene, camera, ctx.rect);

        // 渲染队列
        self.render_queue.render(ctx.painter);
    }

    fn render_ui(&mut self, ctx: &mut RenderContext, ui: &UIState, scene: &SceneState, camera: &Camera2D) {
        // P0 改进：渲染悬停高亮（Hover Highlight）
        if let Some(hovered_id) = ui.hovered_edge {
            if let Some(hovered_edge) = scene.edges.iter().find(|e| e.id == hovered_id) {
                let start = camera.world_to_screen(hovered_edge.start, ctx.rect, scene.scene_origin);
                let end = camera.world_to_screen(hovered_edge.end, ctx.rect, scene.scene_origin);
                // 黄色高亮（比选中稍弱，使用半透明效果）
                ctx.painter.line_segment(
                    [start, end],
                    Stroke::new(3.0, Color32::from_rgba_unmultiplied(255, 255, 100, 200)),
                );
            }
        }

        // 渲染选择高亮
        for edge_id in &ui.selected_edges {
            // 在 scene 中查找对应的边并高亮
            if let Some(edge) = scene.edges.iter().find(|e| e.id == *edge_id) {
                let start = camera.world_to_screen(edge.start, ctx.rect, scene.scene_origin);
                let end = camera.world_to_screen(edge.end, ctx.rect, scene.scene_origin);
                ctx.painter.line_segment([start, end], Stroke::new(4.0, Color32::YELLOW));
            }
        }

        // 渲染圈选多边形
        if ui.lasso_points.len() > 1 {
            let points: Vec<Pos2> = ui.lasso_points.iter()
                .map(|p| {
                    let screen = camera.world_to_screen(*p, ctx.rect, scene.scene_origin);
                    screen
                })
                .collect();

            for i in 0..points.len() - 1 {
                ctx.painter.line_segment(
                    [points[i], points[i + 1]],
                    Stroke::new(2.0, Color32::GREEN),
                );
            }

            // 闭合
            if points.len() > 2 {
                ctx.painter.line_segment(
                    [points[points.len() - 1], points[0]],
                    Stroke::new(2.0, Color32::GREEN),
                );
            }
        }
    }

    fn end_frame(&mut self) {
        // 清理临时数据
    }

    fn resize(&mut self, _width: u32, _height: u32) {
        // 处理窗口大小改变
    }
}
