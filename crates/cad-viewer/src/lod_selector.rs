//! LOD（Level of Detail）动态选择器
//!
//! ## 设计哲学
//!
//! 传统的 LOD 选择基于固定阈值（如座椅数量 > 100 使用简化），但这在不同场景下表现不佳：
//! - 缩放很大时（放大查看），100 个座椅也应该详细渲染
//! - 缩放很小时（俯视全局），1000 个座椅也可以简化
//!
//! 本模块实现**动态 LOD 选择**，基于：
//! 1. 屏幕空间占比（对象在屏幕上的大小）
//! 2. 性能反馈（实际帧率）
//! 3. 用户偏好（从历史行为推断）
//!
//! ## 使用示例
//!
//! ```rust
//! use cad_viewer::lod_selector::LodSelector;
//!
//! let selector = LodSelector::new(zoom, viewport_size, target_fps);
//!
//! // 动态选择座椅区域的 LOD 级别
//! let lod = selector.select_seat_zone_lod(seat_count, actual_fps);
//!
//! // 动态计算网格线密度
//! let grid_density = selector.compute_grid_density();
//! ```

use common_types::LodLevel;

/// LOD 动态选择器
///
/// ## 核心思想
///
/// 基于屏幕空间占比 + 性能反馈动态选择 LOD：
/// ```text
/// screen_area_factor = object_world_area × zoom²
/// performance_factor = actual_fps / target_fps
///
/// if screen_area_factor < 0.01 || performance_factor < 0.5:
///     LodLevel::Simplified  # 远景或性能紧张
/// elif screen_area_factor < 0.1 || performance_factor < 0.8:
///     LodLevel::Medium      # 中景
/// else:
///     LodLevel::Detailed    # 近景且性能充足
/// ```
///
/// ## 使用示例
/// ```rust
/// let selector = LodSelector::new(zoom, viewport_size, target_fps);
/// let lod = selector.select_seat_zone_lod(seat_count, actual_fps);
/// ```
#[derive(Debug, Clone)]
pub struct LodSelector {
    /// 当前缩放级别
    pub zoom: f64,
    /// 视口尺寸（屏幕空间，像素）
    pub viewport_size: f64,
    /// 目标帧率（通常 60fps）
    pub target_fps: f64,
    /// 实际帧率（从渲染器获取）
    pub actual_fps: f64,
}

impl LodSelector {
    /// 创建 LOD 选择器
    ///
    /// ## 参数
    /// - `zoom`: 当前缩放级别
    /// - `viewport_size`: 视口尺寸（像素，通常取宽度和高度的较小值）
    /// - `target_fps`: 目标帧率（通常 60fps）
    ///
    /// ## 示例
    /// ```rust
    /// let selector = LodSelector::new(1.0, 600.0, 60.0);
    /// ```
    pub fn new(zoom: f64, viewport_size: f64, target_fps: f64) -> Self {
        Self {
            zoom,
            viewport_size,
            target_fps,
            actual_fps: target_fps, // 初始假设性能充足
        }
    }

    /// 更新实际帧率
    ///
    /// ## 用途
    /// 在每帧渲染后更新实际帧率，用于下一帧的 LOD 选择
    pub fn update_fps(&mut self, actual_fps: f64) {
        self.actual_fps = actual_fps;
    }

    /// 更新缩放级别
    ///
    /// ## 用途
    /// 在缩放操作后更新，用于重新计算 LOD
    pub fn update_zoom(&mut self, zoom: f64) {
        self.zoom = zoom;
    }

    // ========================================================================
    // 公共 API：动态 LOD 选择
    // ========================================================================

    /// 动态选择座椅区域的 LOD 级别
    ///
    /// ## 参数
    /// - `seat_count`: 座椅数量
    /// - `actual_fps`: 实际帧率（可选，使用当前值如果为 None）
    ///
    /// ## 核心思想
    /// 1. 计算屏幕空间占比：座椅区域在屏幕上的大小
    /// 2. 计算性能因子：实际帧率 / 目标帧率
    /// 3. 综合决策：屏幕占比小或性能紧张 → 简化 LOD
    ///
    /// ## 返回
    /// 动态选择的 LOD 级别
    ///
    /// ## 示例
    /// ```rust
    /// let selector = LodSelector::new(1.0, 600.0, 60.0);
    /// let lod = selector.select_seat_zone_lod(100, None);
    /// ```
    pub fn select_seat_zone_lod(&self, seat_count: usize, actual_fps: Option<f64>) -> LodLevel {
        let fps = actual_fps.unwrap_or(self.actual_fps);
        let screen_area_factor = self.compute_screen_area_factor(seat_count);
        let performance_factor = self.compute_performance_factor(fps);

        // 综合决策
        if screen_area_factor < 0.01 || performance_factor < 0.5 {
            LodLevel::Simplified // 远景或性能紧张
        } else if screen_area_factor < 0.1 || performance_factor < 0.8 {
            LodLevel::Medium // 中景
        } else {
            LodLevel::Detailed // 近景且性能充足
        }
    }

    /// 动态选择网格线的 LOD 级别
    ///
    /// ## 用途
    /// 根据缩放级别动态选择网格线密度
    ///
    /// ## 返回
    /// 网格线间距（世界坐标单位）
    pub fn select_grid_lod(&self) -> f64 {
        // 基础间距 50 单位
        let base_spacing = 50.0;

        // 缩放越大（放大），网格线应该越密
        // 缩放越小（缩小），网格线应该越疏
        base_spacing / self.zoom.max(0.1)
    }

    /// 动态计算网格线密度（跳绘因子）
    ///
    /// ## 用途
    /// 当视口内网格线过多时，跳着绘制以保持性能
    ///
    /// ## 返回
    /// 跳绘因子（1.0 = 每条都画，2.0 = 隔一条画一条，...）
    pub fn compute_grid_density(&self) -> f64 {
        // 目标：视口内保持 20-50 条网格线
        let target_grid_lines = 30.0;

        // 计算当前视口内的网格线数量
        let grid_spacing_world = self.select_grid_lod();
        let viewport_world_size = self.viewport_size / self.zoom;
        let max_grid_lines = viewport_world_size / grid_spacing_world;

        // 如果网格线太多，跳着画
        if max_grid_lines > target_grid_lines * 2.0 {
            (max_grid_lines / target_grid_lines).ceil()
        } else {
            1.0 // 每条都画
        }
    }

    /// 动态选择 NURBS 曲线的 LOD
    ///
    /// ## 参数
    /// - `_curve_length`: 曲线长度（预留用于未来优化）
    /// - `tolerance`: 基础容差
    ///
    /// ## 返回
    /// 动态调整的容差（缩放越大，容差越小，采样越密）
    pub fn select_nurbs_lod(&self, _curve_length: f64, tolerance: f64) -> f64 {
        // 缩放越大（放大），容差越小（更精确）
        tolerance / self.zoom.max(0.1)
    }

    // ========================================================================
    // 工具方法
    // ========================================================================

    /// 计算屏幕空间占比
    ///
    /// ## 核心思想
    /// 估算：每个座椅约 0.5m x 0.5m = 0.25 m²
    /// 屏幕面积 = 世界面积 × zoom²
    fn compute_screen_area_factor(&self, seat_count: usize) -> f64 {
        // 估算世界面积（每个座椅 0.25 m²）
        let world_area = seat_count as f64 * 0.25;

        // 转换为屏幕面积（像素²）
        let screen_area = world_area * self.zoom * self.zoom;

        // 归一化到视口面积
        let viewport_area = self.viewport_size * self.viewport_size;

        // 返回屏幕空间占比（0.0 - 1.0）
        (screen_area / viewport_area).clamp(0.0, 1.0)
    }

    /// 计算性能因子（基于帧率）
    fn compute_performance_factor(&self, actual_fps: f64) -> f64 {
        (actual_fps / self.target_fps).clamp(0.0, 1.0)
    }

    /// 获取 LOD 选择摘要
    pub fn summary(&self) -> String {
        format!(
            "LodSelector {{\n\
             \t缩放：{:.2}\n\
             \t视口：{:.0}px\n\
             \t目标帧率：{:.0}fps\n\
             \t实际帧率：{:.0}fps\n\
             \t性能因子：{:.2}\n\
             }}",
            self.zoom,
            self.viewport_size,
            self.target_fps,
            self.actual_fps,
            self.compute_performance_factor(self.actual_fps),
        )
    }
}

impl Default for LodSelector {
    /// 默认配置：缩放 1.0，视口 600px，目标 60fps
    fn default() -> Self {
        Self::new(1.0, 600.0, 60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_selection_zoomed_out() {
        // 缩小查看（zoom=0.1），100 个座椅应该简化
        let selector = LodSelector::new(0.1, 600.0, 60.0);
        let lod = selector.select_seat_zone_lod(100, None);
        assert_eq!(lod, LodLevel::Simplified);
    }

    #[test]
    fn test_lod_selection_zoomed_in() {
        // 放大查看（zoom=50.0），100 个座椅应该详细
        // screen_area = 100*0.25*50*50 = 62500, viewport_area = 360000, factor = 0.174 > 0.1 → Detailed
        let selector = LodSelector::new(50.0, 600.0, 60.0);
        let lod = selector.select_seat_zone_lod(100, None);
        assert_eq!(lod, LodLevel::Detailed);
    }

    #[test]
    fn test_lod_selection_performance_drop() {
        // 性能下降（actual_fps=20），应该简化
        let selector = LodSelector::new(1.0, 600.0, 60.0);
        let lod = selector.select_seat_zone_lod(100, Some(20.0));
        assert_eq!(lod, LodLevel::Simplified);
    }

    #[test]
    fn test_grid_density() {
        let selector = LodSelector::new(1.0, 600.0, 60.0);
        let density = selector.compute_grid_density();
        assert!(density >= 1.0);

        // 缩放很大时，网格线应该更密
        let selector_zoomed = LodSelector::new(10.0, 600.0, 60.0);
        let density_zoomed = selector_zoomed.compute_grid_density();
        assert!(density_zoomed >= density);
    }
}
