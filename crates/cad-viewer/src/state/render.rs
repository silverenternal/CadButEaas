//! 渲染状态（视图状态）
//!
//! 包含所有渲染相关的状态：相机、LOD、渲染配置等

use common_types::{LodLevel, RenderConfig};
use egui::Vec2;

/// 渲染状态（视图状态）
pub struct RenderState {
    /// 相机
    pub camera: Camera2D,
    /// LOD 级别（用于性能优化，未来用于自动切换渲染精度）
    #[allow(dead_code)]
    pub lod_level: LodLevel,
    /// 渲染配置（用于性能优化，未来用于自动切换渲染精度）
    #[allow(dead_code)]
    pub render_config: Option<RenderConfig>,
    /// 渲染统计（用于性能分析）
    #[allow(dead_code)]
    pub stats: RenderStats,
}

/// 2D 相机（封装缩放/平移）
pub struct Camera2D {
    /// 缩放级别
    pub zoom: f32,
    /// 平移偏移
    pub pan: Vec2,
    /// 视口矩形（屏幕坐标）（用于边界检测，未来用于视口裁剪）
    #[allow(dead_code)]
    pub viewport: Option<egui::Rect>,
}

/// 渲染统计
pub struct RenderStats {
    /// 渲染的边数（用于性能分析）
    #[allow(dead_code)]
    pub edges_rendered: usize,
    /// 渲染的座椅数（用于性能分析）
    #[allow(dead_code)]
    pub seats_rendered: usize,
    /// 帧时间（ms）（用于性能分析）
    #[allow(dead_code)]
    pub frame_time_ms: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
            viewport: None,
        }
    }
}

impl Default for RenderState {
    fn default() -> Self {
        Self {
            camera: Camera2D::default(),
            lod_level: LodLevel::Detailed,
            render_config: None,
            stats: RenderStats::default(),
        }
    }
}

impl Default for RenderStats {
    fn default() -> Self {
        Self {
            edges_rendered: 0,
            seats_rendered: 0,
            frame_time_ms: 0.0,
        }
    }
}

impl Camera2D {
    /// 创建新的相机
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置缩放级别
    #[allow(dead_code)]
    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.clamp(0.01, 100.0);
    }

    /// 缩放（相对当前值）
    #[allow(dead_code)]
    pub fn zoom_by(&mut self, factor: f32) {
        self.set_zoom(self.zoom * factor);
    }

    /// 设置平移
    #[allow(dead_code)]
    pub fn set_pan(&mut self, pan: Vec2) {
        self.pan = pan;
    }

    /// 平移（相对当前值）
    #[allow(dead_code)]
    pub fn pan_by(&mut self, delta: Vec2) {
        self.pan += delta;
    }

    /// 设置视口（用于边界检测）
    #[allow(dead_code)]
    pub fn set_viewport(&mut self, rect: egui::Rect) {
        self.viewport = Some(rect);
    }

    /// 获取视口边界（世界坐标）
    pub fn get_viewport_bounds(&self, rect: egui::Rect) -> ([f64; 2], [f64; 2]) {
        // 转换屏幕四角到世界坐标
        let top_left = self.screen_to_world(rect.min, rect);
        let bottom_right = self.screen_to_world(rect.max, rect);

        // 由于 Y 轴翻转，需要重新排序得到正确的 min/max
        let min = [
            top_left[0].min(bottom_right[0]),
            top_left[1].min(bottom_right[1]),
        ];
        let max = [
            top_left[0].max(bottom_right[0]),
            top_left[1].max(bottom_right[1]),
        ];
        (min, max)
    }

    /// 将世界坐标转换为屏幕坐标
    pub fn world_to_screen(&self, world: [f64; 2], rect: egui::Rect, scene_origin: [f64; 2]) -> egui::Pos2 {
        // 使用相对坐标渲染，提升大坐标场景精度
        let relative_world = [world[0] - scene_origin[0], world[1] - scene_origin[1]];
        let zoom = self.zoom as f64;
        let pan = self.pan;
        let center = rect.center();

        // 正确的变换顺序：先缩放，再平移，最后偏移到屏幕中心
        egui::Pos2::new(
            ((relative_world[0] * zoom) as f32 + pan.x) + center.x,
            ((-relative_world[1] * zoom) as f32 + pan.y) + center.y, // Y 轴翻转
        )
    }

    /// 将屏幕坐标转换为世界坐标
    pub fn screen_to_world(&self, screen: egui::Pos2, rect: egui::Rect) -> [f64; 2] {
        let zoom = self.zoom as f64;
        let pan = self.pan;
        let center = rect.center();

        // 防止除以零
        if zoom.abs() < 1e-10 {
            return [0.0, 0.0];
        }

        // 逆变换：先减去屏幕中心偏移，再减去平移，最后除以缩放
        [
            ((screen.x - center.x - pan.x) as f64 / zoom),
            ((center.y - screen.y + pan.y) as f64 / zoom), // Y 轴翻转
        ]
    }

    /// 自动适配场景边界
    pub fn fit_to_scene(&mut self, min: [f64; 2], max: [f64; 2], view_width: f32, view_height: f32) {
        let scene_width = max[0] - min[0];
        let scene_height = max[1] - min[1];

        // 计算适配缩放
        let zoom_x = view_width as f64 / scene_width;
        let zoom_y = view_height as f64 / scene_height;
        let zoom = zoom_x.min(zoom_y);

        // P11 调试：打印计算过程
        eprintln!("[DEBUG] fit_to_scene: scene_width={}, scene_height={}, zoom_x={}, zoom_y={}, zoom={}",
            scene_width, scene_height, zoom_x, zoom_y, zoom);

        // P11 修复：移除过小的 clamp 下限，支持大坐标场景（如 UTM 建筑坐标）
        // 原 clamp(0.1, 10.0) 会导致大坐标场景无法正确适配
        // 新策略：仅限制最大缩放（避免过小场景过度放大），不限制最小缩放
        let zoom = if zoom > 10.0 {
            // 超小场景，限制最大缩放
            10.0_f32
        } else {
            // 直接使用计算出的缩放值（可以非常小，如 1e-4 级别）
            zoom as f32
        };

        // P11 调试：打印最终缩放值
        eprintln!("[DEBUG] fit_to_scene: final zoom={}", zoom);

        self.zoom = zoom;

        // 计算场景中心
        let scene_center_x = (min[0] + max[0]) / 2.0;
        let scene_center_y = (min[1] + max[1]) / 2.0;

        // 设置平移使场景居中
        self.pan = Vec2::new(
            -scene_center_x as f32 * self.zoom,
            -scene_center_y as f32 * self.zoom,
        );
    }
}

impl RenderState {
    /// 创建新的渲染状态
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// 更新渲染统计（用于性能分析）
    #[allow(dead_code)]
    pub fn update_stats(&mut self, edges: usize, seats: usize) {
        self.stats.edges_rendered = edges;
        self.stats.seats_rendered = seats;
    }
}
